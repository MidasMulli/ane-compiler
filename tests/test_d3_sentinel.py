#!/usr/bin/env python3
"""
Test D3: Sentinel weight passthrough test.

Does aned pass weight data through to .hwx __KERN_0 unchanged, or transform it?

Method:
  1. Generate 64x64 conv .mlmodelc with sentinel FP32 weights
  2. Compile via ane_eval_binary (triggers aned)
  3. Find compiled .hwx in aned cache
  4. Search __KERN_0 for sentinel values (as FP16)
"""

import os
import sys
import struct
import subprocess
import glob
import time
import shutil
import numpy as np

sys.path.insert(0, 'src')
from compiler import gen_conv_mlmodelc

# Sentinel values
SENTINELS = {
    'W[0,0]': 42.0,
    'W[1,0]': -42.0,
    'W[0,1]': 99.0,
    'W[2,2]': 123.456,
    'W[3,0]': -0.5,
}

def fp16_bytes(val):
    """Return 2-byte FP16 encoding of a float."""
    return struct.pack('<e', np.float16(val))

def find_all_occurrences(data, pattern):
    """Find all offsets of pattern in data."""
    offsets = []
    start = 0
    while True:
        idx = data.find(pattern, start)
        if idx == -1:
            break
        offsets.append(idx)
        start = idx + 1
    return offsets

def main():
    print("=" * 70)
    print("TEST D3: Sentinel Weight Passthrough")
    print("=" * 70)

    # 1. Build weights with sentinel values
    in_ch, out_ch = 64, 64
    W = np.zeros((out_ch, in_ch), dtype=np.float32)

    # Place sentinels
    W[0, 0] = 42.0
    W[1, 0] = -42.0
    W[0, 1] = 99.0
    W[2, 2] = 123.456
    W[3, 0] = -0.5

    model_dir = '/tmp/d3_sentinel.mlmodelc'
    if os.path.exists(model_dir):
        shutil.rmtree(model_dir)

    gen_conv_mlmodelc(model_dir, W, in_ch, out_ch, name='sentinel_conv')
    print(f"[1] Generated .mlmodelc at {model_dir}")

    # Show what we wrote to weights file
    wpath = os.path.join(model_dir, 'model.espresso.weights')
    with open(wpath, 'rb') as f:
        wdata = f.read()
    print(f"    Weights file: {len(wdata)} bytes")
    # FP32 weights start at 0x40 (v2 format)
    print(f"    First 8 FP32 values at 0x40:")
    for i in range(8):
        off = 0x40 + i * 4
        val = struct.unpack('<f', wdata[off:off+4])[0]
        print(f"      [{i}] = {val}")

    # 2. Compile via ane_eval_binary
    print(f"\n[2] Compiling via ane_eval_binary...")
    # Kill aned to clear any stale state
    subprocess.run(['killall', 'aned'], capture_output=True)
    time.sleep(1)

    result = subprocess.run(
        ['./tests/ane_eval_binary', model_dir, '64', '64'],
        capture_output=True, text=True, timeout=30
    )
    print(f"    stdout: {result.stdout[:500]}")
    if result.stderr:
        print(f"    stderr: {result.stderr[:500]}")

    # 3. Find .hwx in aned cache
    print(f"\n[3] Searching for .hwx in aned cache...")
    cache_base = '/Library/Caches/com.apple.aned'
    hwx_candidates = []

    for root, dirs, files in os.walk(cache_base):
        for f in files:
            if f.endswith('.hwx'):
                fpath = os.path.join(root, f)
                mtime = os.path.getmtime(fpath)
                # Only recent files (last 60 seconds)
                if time.time() - mtime < 60:
                    hwx_candidates.append((fpath, mtime))

    if not hwx_candidates:
        # Also check model.src directories
        for root, dirs, files in os.walk(cache_base):
            for f in files:
                fpath = os.path.join(root, f)
                mtime = os.path.getmtime(fpath)
                if time.time() - mtime < 120:
                    hwx_candidates.append((fpath, mtime))

    # Sort by modification time, newest first
    hwx_candidates.sort(key=lambda x: -x[1])

    if not hwx_candidates:
        print("    ERROR: No recent .hwx found in aned cache!")
        # List all recent files
        print("    Recent files in cache:")
        for root, dirs, files in os.walk(cache_base):
            for f in files:
                fpath = os.path.join(root, f)
                mtime = os.path.getmtime(fpath)
                if time.time() - mtime < 120:
                    print(f"      {fpath} ({time.time()-mtime:.0f}s ago)")
        return

    hwx_path = hwx_candidates[0][0]
    print(f"    Found: {hwx_path}")

    # 4. Read and parse .hwx
    with open(hwx_path, 'rb') as f:
        hwx = f.read()
    print(f"    .hwx size: {len(hwx)} bytes")

    # Find __KERN_0 section (typically at 0xC000)
    # Search for BEEFFACE header first
    magic = struct.unpack('<I', hwx[0:4])[0]
    print(f"    Magic: 0x{magic:08X}")

    # Parse Mach-O load commands to find __KERN_0
    # After BEEFFACE header: cputype(4), cpusubtype(4), filetype(4), ncmds(4), sizeofcmds(4), flags(4), reserved(4)
    ncmds = struct.unpack('<I', hwx[16:20])[0]
    sizeofcmds = struct.unpack('<I', hwx[20:24])[0]
    print(f"    ncmds: {ncmds}, sizeofcmds: {sizeofcmds}")

    # Walk load commands
    offset = 32  # after mach_header_64
    kern0_offset = None
    kern0_size = None

    for i in range(ncmds):
        if offset + 8 > len(hwx):
            break
        cmd = struct.unpack('<I', hwx[offset:offset+4])[0]
        cmdsize = struct.unpack('<I', hwx[offset+4:offset+8])[0]

        if cmd == 0x19:  # LC_SEGMENT_64
            segname = hwx[offset+8:offset+24].split(b'\x00')[0].decode('ascii', errors='replace')
            vmaddr = struct.unpack('<Q', hwx[offset+24:offset+32])[0]
            vmsize = struct.unpack('<Q', hwx[offset+32:offset+40])[0]
            fileoff = struct.unpack('<Q', hwx[offset+40:offset+48])[0]
            filesize = struct.unpack('<Q', hwx[offset+48:offset+56])[0]
            nsects = struct.unpack('<I', hwx[offset+64:offset+68])[0]
            print(f"    Segment {segname}: fileoff=0x{fileoff:X}, filesize=0x{filesize:X}, nsects={nsects}")

            # Parse sections within segment
            sect_off = offset + 72  # after segment_command_64 header
            for s in range(nsects):
                sectname = hwx[sect_off:sect_off+16].split(b'\x00')[0].decode('ascii', errors='replace')
                sect_segname = hwx[sect_off+16:sect_off+32].split(b'\x00')[0].decode('ascii', errors='replace')
                sect_addr = struct.unpack('<Q', hwx[sect_off+32:sect_off+40])[0]
                sect_size = struct.unpack('<Q', hwx[sect_off+40:sect_off+48])[0]
                sect_offset = struct.unpack('<I', hwx[sect_off+48:sect_off+52])[0]
                print(f"      Section {sectname} ({sect_segname}): offset=0x{sect_offset:X}, size=0x{sect_size:X}")

                if sectname == '__data' and sect_segname == '__KERN_0':
                    kern0_offset = sect_offset
                    kern0_size = sect_size

                sect_off += 80  # section_64 size

        offset += cmdsize

    if kern0_offset is None:
        print("\n    ERROR: __KERN_0 section not found!")
        # Fallback: try 0xC000
        kern0_offset = 0xC000
        kern0_size = len(hwx) - kern0_offset
        print(f"    Fallback: searching from 0xC000 to end ({kern0_size} bytes)")

    # 5. Search for sentinel values in __KERN_0
    print(f"\n[5] Searching for sentinel values in __KERN_0 (offset=0x{kern0_offset:X}, size=0x{kern0_size:X})...")
    kern0_data = hwx[kern0_offset:kern0_offset + kern0_size]

    print(f"\n{'='*70}")
    print("MEASUREMENT BLOCK — Test D3: Sentinel Weight Passthrough")
    print(f"{'='*70}")
    print(f"Model: inner_product 64x64")
    print(f"Input weights: FP32 in espresso.weights (v2 format, data at 0x40)")
    print(f".hwx total size: {len(hwx)} bytes")
    print(f"__KERN_0 offset: 0x{kern0_offset:X}")
    print(f"__KERN_0 size: 0x{kern0_size:X} ({kern0_size} bytes)")

    found_any = False
    for name, val in SENTINELS.items():
        fp16_val = np.float16(val)
        fp16_b = struct.pack('<e', fp16_val)
        fp32_b = struct.pack('<f', val)

        # Search __KERN_0 for FP16
        fp16_hits = find_all_occurrences(kern0_data, fp16_b)
        # Search whole file for FP16
        fp16_whole = find_all_occurrences(hwx, fp16_b)
        # Search __KERN_0 for FP32
        fp32_hits = find_all_occurrences(kern0_data, fp32_b)

        print(f"\n  Sentinel {name} = {val}")
        print(f"    FP16: 0x{fp16_b.hex().upper()}  FP32: 0x{fp32_b.hex().upper()}")

        if fp16_hits:
            found_any = True
            print(f"    __KERN_0 FP16 hits: {len(fp16_hits)} at offsets {[f'0x{o:X}' for o in fp16_hits[:10]]}")
            # Show relative to kern0 start
            for off in fp16_hits[:5]:
                abs_off = kern0_offset + off
                print(f"      file offset 0x{abs_off:X}, kern0 offset 0x{off:X}")
        else:
            print(f"    __KERN_0 FP16 hits: NONE")

        if fp32_hits:
            found_any = True
            print(f"    __KERN_0 FP32 hits: {len(fp32_hits)} at offsets {[f'0x{o:X}' for o in fp32_hits[:10]]}")

        if fp16_whole and not fp16_hits:
            print(f"    Whole-file FP16 hits: {len(fp16_whole)} at {[f'0x{o:X}' for o in fp16_whole[:10]]}")

    # 6. Dump first 256 bytes of __KERN_0 for inspection
    print(f"\n  First 256 bytes of __KERN_0:")
    for row in range(16):
        off = row * 16
        hex_vals = ' '.join(f'{kern0_data[off+i]:02X}' for i in range(min(16, len(kern0_data)-off)))
        print(f"    0x{off:04X}: {hex_vals}")

    # 7. Also dump the input FP32 weights layout for comparison
    print(f"\n  Input weights (FP32 at espresso.weights 0x40), first row (64 values):")
    for col in range(8):
        off = 0x40 + col * 4
        val = struct.unpack('<f', wdata[off:off+4])[0]
        print(f"    W[0,{col}] = {val} (FP32 0x{wdata[off:off+4].hex().upper()})")

    # 8. Try to find weight ordering pattern
    print(f"\n  Searching for tile replication pattern...")
    # In known weight packing: 16 tiles, each with same data
    # Tile size for 64x64 = 64*64*2/16 = 512 bytes per tile
    tile_size = in_ch * out_ch * 2 // 16  # FP16
    print(f"    Expected tile size (if uniform 16-tile replication): {tile_size} bytes")
    print(f"    __KERN_0 / 16 = {kern0_size / 16:.1f} bytes")

    # Check if __KERN_0 has 16 identical tiles
    if kern0_size >= tile_size * 16:
        tile0 = kern0_data[0:tile_size]
        identical_tiles = 0
        for t in range(16):
            tile_t = kern0_data[t*tile_size:(t+1)*tile_size]
            if tile_t == tile0:
                identical_tiles += 1
        print(f"    Tiles identical to tile 0: {identical_tiles}/16")

    # 9. Conclusion
    print(f"\n{'='*70}")
    if found_any:
        print("CONCLUSION: Sentinel values FOUND in __KERN_0.")
        print("aned passes weight data through (converted FP32→FP16, possibly rearranged).")
    else:
        print("CONCLUSION: Sentinel values NOT FOUND in __KERN_0.")
        print("aned TRANSFORMS weight data (encryption, compression, or format change).")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
