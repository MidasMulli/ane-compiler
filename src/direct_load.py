"""
direct_load.py — Direct .hwx loading via IOKit sel=3 ProgramCreate

Loads .hwx files directly into the ANE kext, bypassing aned compilation.
Returns program handles that can be used with _ANEClient.doEvaluateDirectWithModel.

Requirements:
- SIP off (for kext access)
- Entitlements: com.apple.ane.iokit-user-access + com.apple.ane.allow-dataChaining-access
- ANE must be warmed (at least one _ANEClient eval before first direct load)
- .hwx files must be in /Library/Caches/com.apple.aned/ (trusted path)

Protocol discovered by reverse engineering the H16ANE kext:
- sel=0: DeviceOpen (104/104)
- sel=3: ProgramCreate (32B wrapper → 3464B args, NULL output)
  - Wrapper: [args_ptr:8][args_size:8][output_ptr:8][output_size:8]
  - Args struct: [hwx_vm_ptr:8][hwx_size:8]...[flags_0x64:4]...[pid_0x180:4]...[path_0x984:1024]
  - Output buffer: 706360 bytes (program metadata)
  - Returns programHandle at output[0:8]
- sel=4: ProgramPrepare (56/56)
- sel=5: ProgramUnprepare (56/NULL)

Entitlement logic:
- com.apple.ane.allow-dataChaining-access = RESTRICTION on sel=3 gated path
  BUT required for sel=4/sel=2 dispatch table access
- With dataChaining + ANE warmed → sel=3 works via gated path
"""

import ctypes
import ctypes.util
import os
import struct
import shutil

# IOKit framework
_iokit = ctypes.CDLL(ctypes.util.find_library("IOKit"))
_iokit.IOServiceGetMatchingService.restype = ctypes.c_uint32
_iokit.IOServiceGetMatchingService.argtypes = [ctypes.c_uint32, ctypes.c_void_p]
_iokit.IOServiceMatching.restype = ctypes.c_void_p
_iokit.IOServiceMatching.argtypes = [ctypes.c_char_p]
_iokit.IOServiceOpen.restype = ctypes.c_int32
_iokit.IOServiceOpen.argtypes = [ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint32)]
_iokit.IOConnectCallStructMethod.restype = ctypes.c_int32
_iokit.IOConnectCallStructMethod.argtypes = [
    ctypes.c_uint32, ctypes.c_uint32,
    ctypes.c_void_p, ctypes.c_size_t,
    ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)
]
_iokit.IOServiceClose.restype = ctypes.c_int32
_iokit.IOServiceClose.argtypes = [ctypes.c_uint32]
_iokit.IOObjectRelease.restype = ctypes.c_int32
_iokit.IOObjectRelease.argtypes = [ctypes.c_uint32]

_libc = ctypes.CDLL(ctypes.util.find_library("c"))
_mach_task_self = ctypes.c_uint32.in_dll(_libc, "mach_task_self_")

# Constants
ANED_CACHE = "/Library/Caches/com.apple.aned/25E246/ModelAssetsCache"
ARGS_SIZE = 3464
OUTPUT_SIZE = 706360
kIOMainPortDefault = 0


class DirectLoader:
    """Direct .hwx loader via IOKit H11ANEIn kext."""

    def __init__(self):
        self._conn = None
        self._open()

    def _open(self):
        """Open IOKit connection and handshake."""
        svc = _iokit.IOServiceGetMatchingService(
            kIOMainPortDefault, _iokit.IOServiceMatching(b"H11ANEIn")
        )
        if not svc:
            raise RuntimeError("H11ANEIn service not found")

        conn = ctypes.c_uint32(0)
        kr = _iokit.IOServiceOpen(svc, _mach_task_self, 2, ctypes.byref(conn))
        _iokit.IOObjectRelease(svc)
        if kr != 0:
            raise RuntimeError(f"IOServiceOpen failed: 0x{kr:x}")

        # Handshake (sel=0)
        hs = (ctypes.c_uint8 * 104)()
        osz = ctypes.c_size_t(104)
        kr = _iokit.IOConnectCallStructMethod(conn, 0, hs, 104, hs, ctypes.byref(osz))
        if kr != 0:
            raise RuntimeError(f"Handshake failed: 0x{kr:x}")

        self._conn = conn.value

    def load_hwx(self, hwx_path: str) -> int:
        """
        Load a .hwx file directly into the ANE kext via sel=3 ProgramCreate.

        Args:
            hwx_path: Path to .hwx file. Must be under /Library/Caches/com.apple.aned/

        Returns:
            Program handle (uint64) for use with _ANEClient.doEvaluateDirectWithModel
        """
        if not os.path.exists(hwx_path):
            raise FileNotFoundError(f"HWX not found: {hwx_path}")

        hwx_size = os.path.getsize(hwx_path)

        # Read .hwx into VM-allocated memory
        with open(hwx_path, "rb") as f:
            hwx_data = f.read()

        # Allocate buffers using ctypes
        hwx_buf = (ctypes.c_uint8 * hwx_size)(*hwx_data)
        output_buf = (ctypes.c_uint8 * OUTPUT_SIZE)()

        # Build 3464-byte args struct
        args = (ctypes.c_uint8 * ARGS_SIZE)()
        struct.pack_into("<Q", args, 0x00, ctypes.addressof(hwx_buf))  # hwx pointer
        struct.pack_into("<Q", args, 0x08, hwx_size)  # hwx size
        struct.pack_into("<I", args, 0x64, 33)  # flags
        struct.pack_into("<I", args, 0x180, os.getpid())  # PID
        # Path string at 0x984
        path_bytes = hwx_path.encode("utf-8")[:1023]
        for i, b in enumerate(path_bytes):
            args[0x984 + i] = b

        # Build 32-byte wrapper
        wrapper = (ctypes.c_uint8 * 32)()
        struct.pack_into("<Q", wrapper, 0, ctypes.addressof(args))
        struct.pack_into("<Q", wrapper, 8, ARGS_SIZE)
        struct.pack_into("<Q", wrapper, 16, ctypes.addressof(output_buf))
        struct.pack_into("<Q", wrapper, 24, OUTPUT_SIZE)

        # sel=3 ProgramCreate (32B input, NULL output)
        kr = _iokit.IOConnectCallStructMethod(
            self._conn, 3, wrapper, 32, None, None
        )
        if kr != 0:
            raise RuntimeError(f"ProgramCreate failed: 0x{kr:x}")

        # Extract handle from output buffer
        handle = struct.unpack_from("<Q", bytes(output_buf), 0)[0]
        return handle

    def close(self):
        if self._conn is not None:
            _iokit.IOServiceClose(self._conn)
            self._conn = None

    def __del__(self):
        self.close()


def stage_hwx(src_path: str, model_name: str = "direct") -> str:
    """
    Copy a .hwx file to the aned cache directory (trusted path required by kext).

    Args:
        src_path: Source .hwx path
        model_name: Name for the cache subdirectory

    Returns:
        Path in the aned cache directory
    """
    import hashlib
    # Create deterministic hash from source path
    h = hashlib.sha256(src_path.encode()).hexdigest()[:40]
    cache_dir = os.path.join(ANED_CACHE, f"direct-{model_name}", h[:40], h[20:])
    os.makedirs(cache_dir, exist_ok=True)
    dst = os.path.join(cache_dir, "model.hwx")
    shutil.copy2(src_path, dst)
    return dst


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python direct_load.py <hwx_path>")
        sys.exit(1)

    hwx = sys.argv[1]

    # Stage to trusted path if needed
    if not hwx.startswith(ANED_CACHE):
        print(f"Staging {hwx} to aned cache...")
        hwx = stage_hwx(hwx)
        print(f"Staged at: {hwx}")

    loader = DirectLoader()
    handle = loader.load_hwx(hwx)
    print(f"Program handle: 0x{handle:x}")
    loader.close()
