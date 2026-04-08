#!/usr/bin/env python3
"""
Test D4: Multi-layer espresso model without coremltools.

Build a 3-layer espresso model (inner_product → activation → inner_product)
using generate_mlmodelc. Does aned compile it to a multi-pass .hwx?
"""

import os
import sys
import struct
import subprocess
import time
import shutil
import numpy as np

sys.path.insert(0, 'src')
from compiler import (
    gen_fused_ffn_mlmodelc,
    generate_mlmodelc,
    _write_espresso_net,
    _write_espresso_shape,
    _write_espresso_weights_multi,
    _write_metadata,
    _write_coremldata,
)

def parse_hwx_segments(hwx):
    """Parse all Mach-O segments and sections."""
    ncmds = struct.unpack('<I', hwx[16:20])[0]
    offset = 32
    segments = []
    for i in range(ncmds):
        if offset + 8 > len(hwx):
            break
        cmd = struct.unpack('<I', hwx[offset:offset+4])[0]
        cmdsize = struct.unpack('<I', hwx[offset+4:offset+8])[0]
        if cmd == 0x19:  # LC_SEGMENT_64
            segname = hwx[offset+8:offset+24].split(b'\x00')[0].decode('ascii', errors='replace')
            fileoff = struct.unpack('<Q', hwx[offset+40:offset+48])[0]
            filesize = struct.unpack('<Q', hwx[offset+48:offset+56])[0]
            nsects = struct.unpack('<I', hwx[offset+64:offset+68])[0]
            sections = []
            sect_off = offset + 72
            for s in range(nsects):
                sn = hwx[sect_off:sect_off+16].split(b'\x00')[0].decode('ascii', errors='replace')
                ss = hwx[sect_off+16:sect_off+32].split(b'\x00')[0].decode('ascii', errors='replace')
                so = struct.unpack('<I', hwx[sect_off+48:sect_off+52])[0]
                sz = struct.unpack('<Q', hwx[sect_off+40:sect_off+48])[0]
                sections.append((sn, ss, so, sz))
                sect_off += 80
            segments.append((segname, fileoff, filesize, sections))
        elif cmd == 0x04:  # LC_THREAD
            print(f"  LC_THREAD at load command offset 0x{offset:X}, size {cmdsize}")
        offset += cmdsize
    return segments

def count_passes_in_text(hwx, text_offset, text_size):
    """Count pipeline passes by looking for pass boundary markers."""
    PASS_BOUNDARY = 0x00FFF800
    PASS_BOUNDARY_ALT = 0x00FFF860
    PASS_BOUNDARY_ALT2 = 0x00FFF868
    text = hwx[text_offset:text_offset + text_size]
    passes = 0
    for i in range(0, len(text) - 3, 4):
        word = struct.unpack('<I', text[i:i+4])[0]
        if word in (PASS_BOUNDARY, PASS_BOUNDARY_ALT, PASS_BOUNDARY_ALT2):
            passes += 1
    return max(1, passes)  # at least 1 pass


def test_fused_ffn():
    """Test D4a: gen_fused_ffn_mlmodelc (known working)."""
    print(f"\n{'='*70}")
    print(f"TEST D4a: Fused FFN (inner_product → GELU → inner_product)")
    print(f"{'='*70}")

    in_ch, hidden_ch, out_ch = 64, 128, 64
    W_up = np.random.randn(hidden_ch, in_ch).astype(np.float32) * 0.01
    bias_up = np.random.randn(hidden_ch).astype(np.float32) * 0.01
    W_down = np.random.randn(out_ch, hidden_ch).astype(np.float32) * 0.01
    bias_down = np.random.randn(out_ch).astype(np.float32) * 0.01

    model_dir = '/tmp/d4a_fused_ffn.mlmodelc'
    if os.path.exists(model_dir):
        shutil.rmtree(model_dir)
    gen_fused_ffn_mlmodelc(model_dir, W_up, bias_up, W_down, bias_down,
                           in_ch, hidden_ch, out_ch)
    print(f"  Generated .mlmodelc")

    # Compile
    subprocess.run(['killall', 'aned'], capture_output=True)
    time.sleep(0.5)
    result = subprocess.run(
        ['./tests/ane_eval_binary', model_dir, str(in_ch), str(out_ch)],
        capture_output=True, text=True, timeout=30
    )
    print(f"  Compile: {'OK' if result.returncode == 0 else 'FAIL'}")
    if result.stderr:
        print(f"  stderr: {result.stderr[:200]}")

    return analyze_hwx("D4a")


def test_manual_3layer():
    """Test D4b: Manually build 3-layer model with generate_mlmodelc."""
    print(f"\n{'='*70}")
    print(f"TEST D4b: Manual 3-layer (inner_product → ReLU → inner_product)")
    print(f"{'='*70}")

    in_ch, hidden_ch, out_ch = 64, 128, 64

    W_up = np.random.randn(hidden_ch, in_ch).astype(np.float32) * 0.01
    bias_up = np.random.randn(hidden_ch).astype(np.float32) * 0.01
    W_down = np.random.randn(out_ch, hidden_ch).astype(np.float32) * 0.01
    bias_down = np.random.randn(out_ch).astype(np.float32) * 0.01

    model_dir = '/tmp/d4b_manual_3layer.mlmodelc'
    if os.path.exists(model_dir):
        shutil.rmtree(model_dir)

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(os.path.join(model_dir, 'model'), exist_ok=True)
    os.makedirs(os.path.join(model_dir, 'analytics'), exist_ok=True)
    os.makedirs(os.path.join(model_dir, 'neural_network_optionals'), exist_ok=True)

    # 3-layer graph: inner_product → activation(ReLU) → inner_product
    layers = [
        {
            "type": "inner_product",
            "name": "fc_up",
            "debug_info": "fc_up",
            "bottom": "input",
            "top": "fc_up_out",
            "nB": int(in_ch),
            "nC": int(hidden_ch),
            "has_biases": 1,
            "blob_weights": 3,
            "blob_biases": 1,
            "has_relu": 0,
            "has_tanh": 0,
            "has_prelu": 0,
            "weights": {},
        },
        {
            "type": "activation",
            "name": "relu",
            "bottom": "fc_up_out",
            "top": "relu_out",
            "mode": 1,  # ReLU
            "weights": {},
        },
        {
            "type": "inner_product",
            "name": "fc_down",
            "debug_info": "fc_down",
            "bottom": "relu_out",
            "top": "output",
            "nB": int(hidden_ch),
            "nC": int(out_ch),
            "has_biases": 1,
            "blob_weights": 7,
            "blob_biases": 5,
            "has_relu": 0,
            "has_tanh": 0,
            "has_prelu": 0,
            "weights": {},
            "attributes": {"is_output": 1},
        },
    ]

    shapes = {
        "input": (1, 1, 1, in_ch),
        "fc_up_out": (1, 1, 1, hidden_ch),
        "relu_out": (1, 1, 1, hidden_ch),
        "output": (1, 1, 1, out_ch),
    }

    # Write espresso.net
    _write_espresso_net(os.path.join(model_dir, 'model.espresso.net'), layers)

    # Write espresso.shape
    _write_espresso_shape(os.path.join(model_dir, 'model.espresso.shape'), shapes)

    # Write multi-layer weights (version 8)
    _write_espresso_weights_multi(
        os.path.join(model_dir, 'model.espresso.weights'),
        [(W_up, bias_up), (W_down, bias_down)]
    )

    # Metadata
    _write_metadata(
        os.path.join(model_dir, 'metadata.json'),
        inputs=[("input", [int(in_ch), 1, 1])],
        outputs=[("output", [int(out_ch), 1, 1])],
    )
    _write_coremldata(os.path.join(model_dir, 'coremldata.bin'))
    for sub in ['model', 'analytics', 'neural_network_optionals']:
        with open(os.path.join(model_dir, sub, 'coremldata.bin'), 'wb') as f:
            f.write(b'\x00' * 64)

    print(f"  Generated .mlmodelc with 3 layers")

    # Compile
    subprocess.run(['killall', 'aned'], capture_output=True)
    time.sleep(0.5)
    result = subprocess.run(
        ['./tests/ane_eval_binary', model_dir, str(in_ch), str(out_ch)],
        capture_output=True, text=True, timeout=30
    )
    print(f"  Compile: {'OK' if result.returncode == 0 else 'FAIL'}")
    if result.stderr:
        print(f"  stderr: {result.stderr[:200]}")

    return analyze_hwx("D4b")


def test_5layer():
    """Test D4c: 5-layer model (ip → relu → ip → gelu → ip)."""
    print(f"\n{'='*70}")
    print(f"TEST D4c: 5-layer (ip → relu → ip → GELU → ip)")
    print(f"{'='*70}")

    d1, d2, d3, d4 = 64, 128, 96, 64

    W1 = np.random.randn(d2, d1).astype(np.float32) * 0.01
    b1 = np.random.randn(d2).astype(np.float32) * 0.01
    W2 = np.random.randn(d3, d2).astype(np.float32) * 0.01
    b2 = np.random.randn(d3).astype(np.float32) * 0.01
    W3 = np.random.randn(d4, d3).astype(np.float32) * 0.01
    b3 = np.random.randn(d4).astype(np.float32) * 0.01

    model_dir = '/tmp/d4c_5layer.mlmodelc'
    if os.path.exists(model_dir):
        shutil.rmtree(model_dir)

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(os.path.join(model_dir, 'model'), exist_ok=True)
    os.makedirs(os.path.join(model_dir, 'analytics'), exist_ok=True)
    os.makedirs(os.path.join(model_dir, 'neural_network_optionals'), exist_ok=True)

    layers = [
        {
            "type": "inner_product",
            "name": "fc1",
            "debug_info": "fc1",
            "bottom": "input",
            "top": "fc1_out",
            "nB": d1,
            "nC": d2,
            "has_biases": 1,
            "blob_weights": 3,
            "blob_biases": 1,
            "has_relu": 0, "has_tanh": 0, "has_prelu": 0,
            "weights": {},
        },
        {
            "type": "activation",
            "name": "relu1",
            "bottom": "fc1_out",
            "top": "relu1_out",
            "mode": 1,
            "weights": {},
        },
        {
            "type": "inner_product",
            "name": "fc2",
            "debug_info": "fc2",
            "bottom": "relu1_out",
            "top": "fc2_out",
            "nB": d2,
            "nC": d3,
            "has_biases": 1,
            "blob_weights": 7,
            "blob_biases": 5,
            "has_relu": 0, "has_tanh": 0, "has_prelu": 0,
            "weights": {},
        },
        {
            "type": "activation",
            "name": "gelu1",
            "bottom": "fc2_out",
            "top": "gelu1_out",
            "mode": 19,
            "weights": {},
        },
        {
            "type": "inner_product",
            "name": "fc3",
            "debug_info": "fc3",
            "bottom": "gelu1_out",
            "top": "output",
            "nB": d3,
            "nC": d4,
            "has_biases": 1,
            "blob_weights": 11,
            "blob_biases": 9,
            "has_relu": 0, "has_tanh": 0, "has_prelu": 0,
            "weights": {},
            "attributes": {"is_output": 1},
        },
    ]

    shapes = {
        "input": (1, 1, 1, d1),
        "fc1_out": (1, 1, 1, d2),
        "relu1_out": (1, 1, 1, d2),
        "fc2_out": (1, 1, 1, d3),
        "gelu1_out": (1, 1, 1, d3),
        "output": (1, 1, 1, d4),
    }

    _write_espresso_net(os.path.join(model_dir, 'model.espresso.net'), layers)
    _write_espresso_shape(os.path.join(model_dir, 'model.espresso.shape'), shapes)

    # Need v8 multi-blob weights for 3 weight layers
    # Blob IDs: fc1=3, fc2=7, fc3=11; biases: fc1=1, fc2=5, fc3=9
    # _write_espresso_weights_multi handles 2 layers; extend for 3
    # Actually let's write it manually for 3 layers

    # For simplicity, use the _write_espresso_weights_multi which handles N layers
    _write_espresso_weights_multi(
        os.path.join(model_dir, 'model.espresso.weights'),
        [(W1, b1), (W2, b2), (W3, b3)]
    )

    _write_metadata(
        os.path.join(model_dir, 'metadata.json'),
        inputs=[("input", [d1, 1, 1])],
        outputs=[("output", [d4, 1, 1])],
    )
    _write_coremldata(os.path.join(model_dir, 'coremldata.bin'))
    for sub in ['model', 'analytics', 'neural_network_optionals']:
        with open(os.path.join(model_dir, sub, 'coremldata.bin'), 'wb') as f:
            f.write(b'\x00' * 64)

    print(f"  Generated .mlmodelc with 5 layers")

    subprocess.run(['killall', 'aned'], capture_output=True)
    time.sleep(0.5)
    result = subprocess.run(
        ['./tests/ane_eval_binary', model_dir, str(d1), str(d4)],
        capture_output=True, text=True, timeout=30
    )
    print(f"  Compile: {'OK' if result.returncode == 0 else 'FAIL'}")
    if result.stderr:
        print(f"  stderr: {result.stderr[:200]}")

    return analyze_hwx("D4c")


def analyze_hwx(label):
    """Find newest .hwx and analyze its structure."""
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
        print(f"  ERROR: No recent .hwx found!")
        return None

    hwx_path = hwx_files[0][0]
    with open(hwx_path, 'rb') as f:
        hwx = f.read()

    ncmds = struct.unpack('<I', hwx[16:20])[0]
    sizeofcmds = struct.unpack('<I', hwx[20:24])[0]

    print(f"\n  MEASUREMENT BLOCK — {label}")
    print(f"  {'='*60}")
    print(f"  .hwx size: {len(hwx)} bytes")
    print(f"  ncmds: {ncmds}")

    segments = parse_hwx_segments(hwx)
    text_offset = None
    text_size = None
    kern_segments = []

    for seg in segments:
        segname, fileoff, filesize, sections = seg
        for sn, ss, so, sz in sections:
            print(f"  {sn:12s} ({ss:12s}): offset=0x{so:05X} size=0x{sz:05X} ({sz} bytes)")
            if sn == '__text':
                text_offset = so
                text_size = sz
            if 'kern' in sn.lower():
                kern_segments.append((sn, ss, so, sz))

    # Count passes
    if text_offset is not None:
        passes = count_passes_in_text(hwx, text_offset, text_size)
        print(f"  __text passes: {passes}")
        print(f"  __text size: {text_size} bytes ({text_size // 4} words)")

        # Dump __text as words for inspection
        text = hwx[text_offset:text_offset + text_size]
        print(f"  __text words:")
        for i in range(0, min(len(text), 400), 4):
            w = struct.unpack('<I', text[i:i+4])[0]
            marker = ""
            if w == 0x00FFF868:
                marker = " ← PASS_BOUNDARY_ALT2"
            elif w == 0x00FFF800:
                marker = " ← PASS_BOUNDARY"
            elif w == 0x00FFF860:
                marker = " ← PASS_BOUNDARY_ALT"
            elif w == 0x22001440:
                marker = " ← PROGRAM_TERM1"
            elif w == 0x01040021:
                marker = " ← PROGRAM_TERM2"
            elif w & 0xFF000000 == 0x93000000:
                marker = f" ← conv opcode"
            elif w & 0xFF000000 == 0xB2000000:
                marker = f" ← PWL opcode"
            if marker or i < 40:
                print(f"    W[{i//4:3d}] (0x{text_offset+i:05X}): 0x{w:08X}{marker}")

    # KERN segments
    if kern_segments:
        total_kern = sum(sz for _, _, _, sz in kern_segments)
        print(f"  Total kernel data: {total_kern} bytes across {len(kern_segments)} segments")
    else:
        print(f"  No __KERN segments found")

    return hwx


def main():
    test_fused_ffn()
    time.sleep(1)
    test_manual_3layer()
    time.sleep(1)
    test_5layer()

    # Final summary
    print(f"\n{'='*70}")
    print(f"TEST D4 SUMMARY")
    print(f"{'='*70}")

if __name__ == '__main__':
    main()
