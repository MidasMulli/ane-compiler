#!/usr/bin/env python3
"""
Capture conv .hwx at real dimensions via _ANEInMemoryModel.

This bypasses CoreML's graph partitioner — compiles MIL directly to ANE.
Uses the same path as true_5_0_kill.py (proven SIP-on).

We compile individual conv1x1 ops at multiple dimensions to understand
how __KERN_0 tile layout scales with weight matrix size.

Run with: ~/.mlx-env/bin/python3 capture_conv_real_dims.py
"""
import objc, os, plistlib, ctypes, shutil, struct, glob, time
import numpy as np
from Foundation import *

objc.loadBundle('AppleNeuralEngine', globals(),
    bundle_path='/System/Library/PrivateFrameworks/AppleNeuralEngine.framework')

ANEInMemoryModel = objc.lookUpClass('_ANEInMemoryModel')
ANEInMemoryModelDescriptor = objc.lookUpClass('_ANEInMemoryModelDescriptor')

OUT = os.path.join(os.path.dirname(__file__), '..', 'templates')
os.makedirs(OUT, exist_ok=True)


def build_conv_mil(in_ch, out_ch, weight_seed=42):
    """Build MIL text for a conv1x1 (= linear projection)."""
    np.random.seed(weight_seed)
    w = np.random.randn(out_ch, in_ch).astype(np.float16) * 0.1

    # MIL format: conv with weight blob
    w_flat = ', '.join(f'{v:.6f}' for v in w.flatten().astype(np.float32))

    mil = f"""
program {{
  func main(input: tensor<fp16, [1, {in_ch}, 1, 1]>) -> tensor<fp16, [1, {out_ch}, 1, 1]> {{
    let weight = const(type = tensor<fp16, [{out_ch}, {in_ch}, 1, 1]>, val = [{w_flat}]);
    let result = conv(x = input, weight = weight, strides = [1, 1], pad_type = "valid", dilations = [1, 1], groups = 1);
    return result;
  }}
}}
"""
    return mil.encode('utf-8'), w


def build_conv_silu_mil(in_ch, out_ch, weight_seed=42):
    """Build MIL text for conv1x1 + SiLU."""
    np.random.seed(weight_seed)
    w = np.random.randn(out_ch, in_ch).astype(np.float16) * 0.1
    w_flat = ', '.join(f'{v:.6f}' for v in w.flatten().astype(np.float32))

    mil = f"""
program {{
  func main(input: tensor<fp16, [1, {in_ch}, 1, 1]>) -> tensor<fp16, [1, {out_ch}, 1, 1]> {{
    let weight = const(type = tensor<fp16, [{out_ch}, {in_ch}, 1, 1]>, val = [{w_flat}]);
    let conv_out = conv(x = input, weight = weight, strides = [1, 1], pad_type = "valid", dilations = [1, 1], groups = 1);
    let sig = sigmoid(x = conv_out);
    let result = mul(x = conv_out, y = sig);
    return result;
  }}
}}
"""
    return mil.encode('utf-8'), w


def compile_and_capture(mil_bytes, label):
    """Compile MIL to ANE, capture .hwx, return (hwx_path, model) or None."""
    ns_net = NSData.dataWithBytes_length_(mil_bytes, len(mil_bytes))
    opts = plistlib.dumps({'h17g': {'EnableLowEffortCPAllocation': True}}, fmt=plistlib.FMT_BINARY)
    ns_opts = NSData.dataWithBytes_length_(opts, len(opts))
    desc = ANEInMemoryModelDescriptor.alloc().initWithNetworkText_weights_optionsPlist_isMILModel_(
        ns_net, NSDictionary.dictionary(), ns_opts, True)
    model = ANEInMemoryModel.alloc().initWithDesctiptor_(desc)
    model.purgeCompiledModel()
    model.saveModelFiles()
    lmp = str(model.localModelPath())
    shutil.copy2(os.path.join(lmp, 'net.plist'), os.path.join(lmp, 'model.mil'))

    ok = model.compileWithQoS_options_error_(0, NSDictionary.dictionary(), None)
    if not ok:
        print(f"  {label}: COMPILE FAILED")
        return None

    # Search for .hwx in the local model path and ANE cache
    hwx_path = None

    # Check local model path
    for root, dirs, files in os.walk(lmp):
        for f in files:
            fp = os.path.join(root, f)
            if os.path.getsize(fp) > 32:
                try:
                    with open(fp, 'rb') as fh:
                        magic = struct.unpack('<I', fh.read(4))[0]
                        if magic == 0xBEEFFACE:
                            hwx_path = fp
                except:
                    pass

    # Check ANE cache (within last 5 seconds)
    if not hwx_path:
        cache_base = "/Library/Caches/com.apple.aned"
        if os.path.exists(cache_base):
            now = time.time()
            for root, dirs, files in os.walk(cache_base):
                for f in files:
                    if f.endswith('.hwx'):
                        fp = os.path.join(root, f)
                        if now - os.path.getmtime(fp) < 5:
                            hwx_path = fp

    # Also check InMemoryModelCache
    if not hwx_path:
        pattern = os.path.expanduser("/Library/Caches/com.apple.aned/*/InMemoryModelCache/**/model.hwx")
        for p in glob.glob(pattern, recursive=True):
            if time.time() - os.path.getmtime(p) < 10:
                hwx_path = p

    if hwx_path:
        dst = os.path.join(OUT, f"{label}.hwx")
        shutil.copy2(hwx_path, dst)
        sz = os.path.getsize(dst)
        data = open(dst, 'rb').read()
        ncmds = struct.unpack_from('<I', data, 0x10)[0]
        print(f"  {label}: CAPTURED {sz:,}B ncmds={ncmds} → {dst}")
        return dst, model
    else:
        # List what files ARE in local model path
        print(f"  {label}: NO .hwx found. Local path contents:")
        for root, dirs, files in os.walk(lmp):
            for f in files:
                fp = os.path.join(root, f)
                print(f"    {os.path.relpath(fp, lmp)} ({os.path.getsize(fp):,}B)")
        return None

    return None


def analyze_hwx(hwx_path, label, expected_weights=None):
    """Analyze tile layout of captured .hwx."""
    data = open(hwx_path, 'rb').read()
    sz = len(data)
    ncmds = struct.unpack_from('<I', data, 0x10)[0]

    # Parse sections
    offset = 32
    text_off = text_sz = kern0_off = kern0_sz = 0
    for _ in range(min(ncmds, 50)):
        if offset + 8 > len(data): break
        cmd = struct.unpack_from('<I', data, offset)[0]
        cmdsize = struct.unpack_from('<I', data, offset + 4)[0]
        if cmdsize == 0: break
        if cmd == 0x19:
            segname = data[offset+8:offset+24].split(b'\x00')[0].decode('ascii')
            nsects_off = offset + 64
            if nsects_off + 4 <= len(data):
                nsects = struct.unpack_from('<I', data, nsects_off)[0]
                for s in range(min(nsects, 10)):
                    s_off = offset + 72 + s * 80
                    if s_off + 80 > len(data): break
                    sectname = data[s_off:s_off+16].split(b'\x00')[0].decode('ascii')
                    size = struct.unpack_from('<Q', data, s_off + 40)[0]
                    foff = struct.unpack_from('<I', data, s_off + 48)[0]
                    if segname == '__TEXT' and sectname == '__text':
                        text_off, text_sz = foff, size
                    elif segname == '__KERN_0':
                        kern0_off, kern0_sz = foff, size
        offset += cmdsize

    print(f"\n{'='*60}")
    print(f"  {label}: {sz:,}B, ncmds={ncmds}")
    print(f"  __text: off=0x{text_off:X} sz={text_sz}B ({text_sz//4}w)")
    print(f"  __kern0: off=0x{kern0_off:X} sz={kern0_sz:,}B")

    if kern0_off == 0 or kern0_sz == 0:
        print("  NO __KERN_0 — cannot analyze tile layout")
        return

    kern = data[kern0_off:kern0_off + kern0_sz]

    # Find tile stride by looking for repeating patterns
    # Check first 64 bytes against various strides
    ref = kern[:64]
    tile_stride = 0
    for stride in range(64, min(kern0_sz, 65536), 64):
        if stride + 64 > kern0_sz:
            break
        candidate = kern[stride:stride + 64]
        if ref == candidate:
            tile_stride = stride
            break

    if tile_stride:
        num_tiles = kern0_sz // tile_stride
        remainder = kern0_sz % tile_stride
        print(f"  Tile stride: {tile_stride:,}B ({tile_stride//2} FP16)")
        print(f"  Tiles: {num_tiles} (remainder={remainder})")

        # Verify all tiles are identical (or mostly)
        t0 = kern[:tile_stride]
        diffs_per_tile = []
        for t in range(1, min(num_tiles, 16)):
            ti = kern[t * tile_stride:(t + 1) * tile_stride]
            if len(ti) < tile_stride:
                break
            d = sum(1 for a, b in zip(t0, ti) if a != b)
            diffs_per_tile.append(d)
        if diffs_per_tile:
            print(f"  Cross-tile diffs: min={min(diffs_per_tile)} max={max(diffs_per_tile)} "
                  f"avg={sum(diffs_per_tile)/len(diffs_per_tile):.1f}")

        # Analyze tile content
        fp16 = np.frombuffer(t0, dtype=np.float16)
        nz = np.count_nonzero(fp16)
        has_inf = np.any(np.isinf(fp16))
        print(f"  Tile 0: {len(fp16)} FP16 values, {nz} nonzero, has_inf={has_inf}")

        # If we have expected weights, try to find them in the tile
        if expected_weights is not None:
            w_flat = expected_weights.astype(np.float16).flatten()
            w_bytes = w_flat.tobytes()
            # Search for weight data in tile
            found_at = -1
            for search_off in range(0, min(tile_stride, 4096), 2):
                candidate = kern[search_off:search_off + len(w_bytes)]
                if len(candidate) == len(w_bytes):
                    cand_fp16 = np.frombuffer(candidate, dtype=np.float16)
                    if np.allclose(cand_fp16, w_flat, rtol=0, atol=1e-4):
                        found_at = search_off
                        break
            if found_at >= 0:
                print(f"  Weights found at tile offset {found_at} ({found_at//2} FP16)")
            else:
                # Search across entire kern0
                for search_off in range(0, kern0_sz - len(w_bytes), 2):
                    candidate = kern[search_off:search_off + len(w_bytes)]
                    cand_fp16 = np.frombuffer(candidate, dtype=np.float16)
                    if np.allclose(cand_fp16, w_flat[:16], rtol=0, atol=1e-4):
                        print(f"  Partial weight match at kern0 offset {search_off}")
                        break
                else:
                    print(f"  WARNING: Weights NOT found in __KERN_0")
                    # Show first few values for debugging
                    print(f"  Expected first 8: {w_flat[:8]}")
                    print(f"  Kern0 first 8 FP16: {np.frombuffer(kern[:16], dtype=np.float16)}")
    else:
        print(f"  Could not determine tile stride (no 64-byte repeat found)")
        # Dump first 128 bytes
        fp16 = np.frombuffer(kern[:128], dtype=np.float16)
        print(f"  First 64 FP16: {fp16}")


def main():
    print("="*60)
    print("  Reference .hwx capture at real dimensions")
    print("="*60)

    configs = [
        # (in_ch, out_ch, label, builder)
        (8, 8, "conv_8x8", build_conv_mil),
        (64, 64, "conv_64x64", build_conv_mil),
        (64, 128, "conv_64x128", build_conv_mil),
        (256, 512, "conv_256x512", build_conv_mil),
        (512, 1024, "conv_512x1024", build_conv_mil),
        (8, 8, "conv_silu_8x8", build_conv_silu_mil),
        (64, 64, "conv_silu_64x64", build_conv_silu_mil),
        (256, 512, "conv_silu_256x512", build_conv_silu_mil),
    ]

    for in_ch, out_ch, label, builder in configs:
        print(f"\n--- {label} ({in_ch}→{out_ch}, {in_ch*out_ch*2:,}B weights) ---")
        try:
            mil_bytes, weights = builder(in_ch, out_ch)
            result = compile_and_capture(mil_bytes, label)
            if result:
                hwx_path, model = result
                analyze_hwx(hwx_path, label, expected_weights=weights)
                model.unloadWithQoS_error_(0, None)
        except Exception as e:
            print(f"  EXCEPTION: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*60}")
    print("  Captured templates:")
    print(f"{'='*60}")
    for f in sorted(os.listdir(OUT)):
        if f.endswith('.hwx'):
            fp = os.path.join(OUT, f)
            print(f"  {f}: {os.path.getsize(fp):,}B")


if __name__ == '__main__':
    main()
