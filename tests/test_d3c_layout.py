#!/usr/bin/env python3
"""
Test D3c: Detailed weight layout analysis.

Now that we know sentinels appear in __KERN_0 as FP16, analyze the exact
layout: input position vs output position, tile replication, ordering.
"""

import os
import sys
import struct
import subprocess
import time
import shutil
import numpy as np

sys.path.insert(0, 'src')
from compiler import gen_conv_mlmodelc

def find_all(data, pattern):
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
    # Use dim=64 — small enough to trace every value
    dim = 64

    # Create a weight matrix where every value is unique and recognizable
    W = np.zeros((dim, dim), dtype=np.float32)

    # Set specific sentinel values at known positions
    # Use values that are unique in FP16 representation
    test_values = []
    for row in range(min(8, dim)):
        for col in range(min(8, dim)):
            # Use values like 10.0 + row + col*0.1 — all unique in FP16
            val = 10.0 + row + col * 0.125
            W[row, col] = val
            test_values.append((row, col, val))

    model_dir = '/tmp/d3c_layout.mlmodelc'
    if os.path.exists(model_dir):
        shutil.rmtree(model_dir)
    gen_conv_mlmodelc(model_dir, W, dim, dim, name='layout_conv')

    # Compile
    subprocess.run(['killall', 'aned'], capture_output=True)
    time.sleep(0.5)
    subprocess.run(
        ['./tests/ane_eval_binary', model_dir, str(dim), str(dim)],
        capture_output=True, text=True, timeout=30
    )

    # Find newest .hwx
    cache_base = '/Library/Caches/com.apple.aned'
    hwx_files = []
    for root, dirs, files in os.walk(cache_base):
        for f in files:
            if f.endswith('.hwx'):
                fpath = os.path.join(root, f)
                mtime = os.path.getmtime(fpath)
                if time.time() - mtime < 30:
                    hwx_files.append((fpath, mtime))
    hwx_files.sort(key=lambda x: -x[1])
    if not hwx_files:
        print("ERROR: No .hwx found!")
        return

    with open(hwx_files[0][0], 'rb') as f:
        hwx = f.read()

    # Find __KERN_0
    ncmds = struct.unpack('<I', hwx[16:20])[0]
    offset = 32
    kern0_off = None
    kern0_sz = None
    for i in range(ncmds):
        if offset + 8 > len(hwx):
            break
        cmd = struct.unpack('<I', hwx[offset:offset+4])[0]
        cmdsize = struct.unpack('<I', hwx[offset+4:offset+8])[0]
        if cmd == 0x19:
            segname = hwx[offset+8:offset+24].split(b'\x00')[0].decode('ascii', errors='replace')
            nsects = struct.unpack('<I', hwx[offset+64:offset+68])[0]
            sect_off = offset + 72
            for s in range(nsects):
                sn = hwx[sect_off:sect_off+16].split(b'\x00')[0].decode('ascii', errors='replace')
                ss = hwx[sect_off+16:sect_off+32].split(b'\x00')[0].decode('ascii', errors='replace')
                so = struct.unpack('<I', hwx[sect_off+48:sect_off+52])[0]
                sz = struct.unpack('<Q', hwx[sect_off+40:sect_off+48])[0]
                if 'kern' in sn.lower() or 'KERN' in ss:
                    kern0_off = so
                    kern0_sz = sz
                sect_off += 80
        offset += cmdsize

    if kern0_off is None:
        print("No __KERN_0 found!")
        return

    kern = hwx[kern0_off:kern0_off + kern0_sz]

    print(f"{'='*70}")
    print(f"MEASUREMENT BLOCK — Test D3: Weight Layout Analysis")
    print(f"{'='*70}")
    print(f"Model: inner_product {dim}x{dim}")
    print(f"__KERN_0: offset=0x{kern0_off:X}, size=0x{kern0_sz:X} ({kern0_sz} bytes)")
    print(f"Expected raw FP16 weight size: {dim*dim*2} bytes")
    print(f"Ratio __KERN_0 / raw: {kern0_sz / (dim*dim*2):.1f}x")

    # Search for each sentinel and record its position
    print(f"\nSentinel positions (input row,col → __KERN_0 offset):")
    print(f"{'Input':>12s} {'Value':>10s} {'FP16':>8s} {'Kern0 off':>12s} {'Kern0/raw':>10s}")
    print(f"{'-'*12} {'-'*10} {'-'*8} {'-'*12} {'-'*10}")

    for row, col, val in test_values:
        fp16_b = struct.pack('<e', np.float16(val))
        hits = find_all(kern, fp16_b)

        # Input position in FP32 row-major: row*dim + col (element index)
        # In FP16 linear: (row*dim + col) * 2
        raw_fp16_off = (row * dim + col) * 2

        if hits:
            for h in hits[:3]:
                ratio = h / raw_fp16_off if raw_fp16_off > 0 else float('inf')
                print(f"  W[{row},{col}] {val:10.3f} 0x{fp16_b.hex()} 0x{h:08X} {ratio:10.2f}")
        else:
            print(f"  W[{row},{col}] {val:10.3f} 0x{fp16_b.hex()} NOT FOUND")

    # Check tile replication
    print(f"\nTile replication analysis:")
    # WeightPacker uses 16 tiles. For 64x64:
    # tile_size = ceil(dim/16) * dim * 2 = 4 * 64 * 2 = 512 bytes
    tile_size_computed = ((dim + 15) // 16) * dim * 2
    print(f"  Computed tile size: {tile_size_computed} bytes")
    print(f"  16 tiles would be: {tile_size_computed * 16} bytes")

    # Check if first N bytes repeat every tile_size_computed
    if kern0_sz >= tile_size_computed * 2:
        tile0 = kern[0:tile_size_computed]
        matches = 0
        for t in range(1, min(16, kern0_sz // tile_size_computed)):
            tile_t = kern[t * tile_size_computed:(t + 1) * tile_size_computed]
            if tile_t == tile0:
                matches += 1
        print(f"  Tiles matching tile 0: {matches}")

    # Dump entire __KERN_0 as FP16 values (only non-zero)
    print(f"\nNon-zero FP16 values in __KERN_0:")
    for i in range(0, min(kern0_sz, 8192), 2):
        val = struct.unpack('<e', kern[i:i+2])[0]
        if val != 0.0:
            print(f"  offset 0x{i:04X}: {val:.4f}")

    # Dump first 256 bytes hex for visual
    print(f"\n__KERN_0 hex dump (first 256 bytes):")
    for row in range(16):
        off = row * 16
        hex_vals = ' '.join(f'{kern[off+j]:02X}' for j in range(16))
        # Also decode as FP16
        fp16_vals = []
        for j in range(0, 16, 2):
            v = struct.unpack('<e', kern[off+j:off+j+2])[0]
            if v != 0.0:
                fp16_vals.append(f'{v:.2f}')
        fp16_str = ' '.join(fp16_vals) if fp16_vals else ''
        print(f"  0x{off:04X}: {hex_vals}  {fp16_str}")

    # Compare input layout vs output layout
    print(f"\nInput weight layout (FP32, row-major W[out_ch, in_ch]):")
    print(f"  W[0,0..7] = {W[0,:8]}")
    print(f"  W[1,0..7] = {W[1,:8]}")
    print(f"  W[2,0..7] = {W[2,:8]}")
    print(f"  W[3,0..7] = {W[3,:8]}")

    print(f"\nOutput __KERN_0 layout (FP16, first 64 values):")
    for i in range(0, min(128, kern0_sz), 2):
        val = struct.unpack('<e', kern[i:i+2])[0]
        if val != 0.0:
            # Identify which input position this corresponds to
            # Search through test_values
            for row, col, tv in test_values:
                if abs(np.float16(tv) - val) < 0.001:
                    print(f"  kern[{i//2}] (0x{i:04X}) = {val:.4f} ← W[{row},{col}]")
                    break
            else:
                print(f"  kern[{i//2}] (0x{i:04X}) = {val:.4f} ← ???")


if __name__ == '__main__':
    main()
