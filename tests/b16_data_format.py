#!/usr/bin/env python3
"""B16: Data format probing — FP16 vs INT8 weight format in .hwx"""

import sys, os, subprocess, time, struct, shutil, json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.compiler import gen_conv_mlmodelc

CAPTURE_TOOL = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'capture_and_eval')


def ct_save_and_compile(ct_model, base_name):
    """Save a coremltools model and compile to .mlmodelc"""
    mlmodel_path = f'/tmp/{base_name}.mlmodel'
    mlmodelc_path = f'/tmp/{base_name}.mlmodelc'

    ct_model.save(mlmodel_path)

    if os.path.exists(mlmodelc_path):
        shutil.rmtree(mlmodelc_path)

    result = subprocess.run(['xcrun', 'coremlcompiler', 'compile', mlmodel_path, '/tmp/'],
                           capture_output=True, text=True)
    if not os.path.exists(mlmodelc_path):
        # coremlcompiler may name it differently
        print(f"  WARNING: {mlmodelc_path} not found after compile")
        print(f"  stdout: {result.stdout[:200]}")
        print(f"  stderr: {result.stderr[:200]}")
        # Try looking for it
        for f in os.listdir('/tmp/'):
            if f.startswith(base_name) and f.endswith('.mlmodelc'):
                mlmodelc_path = f'/tmp/{f}'
                break

    return mlmodelc_path


def generate_models():
    """Generate FP16 and INT8 conv models."""
    import coremltools as ct
    from coremltools.models.neural_network import NeuralNetworkBuilder

    np.random.seed(42)
    in_ch, out_ch = 64, 64
    W = np.random.randn(out_ch, in_ch).astype(np.float32)

    paths = {}

    # Model 1: FP16 via coremltools
    print("=== Generating FP16 model via coremltools ===")
    builder = NeuralNetworkBuilder(
        input_features=[('input', ct.models.datatypes.Array(in_ch))],
        output_features=[('output', ct.models.datatypes.Array(out_ch))])
    builder.add_inner_product('fc', W, None, in_ch, out_ch,
                              has_bias=False,
                              input_name='input', output_name='output')
    model_fp16 = ct.models.MLModel(builder.spec)
    paths['fp16_ct'] = ct_save_and_compile(model_fp16, 'b16_fp16')
    print(f"  Compiled: {paths['fp16_ct']}")

    # Model 2: INT8 via coremltools quantization
    print("\n=== Generating INT8 model via coremltools ===")
    try:
        from coremltools.models.neural_network.quantization_utils import quantize_weights
        model_int8 = quantize_weights(model_fp16, nbits=8)
        paths['int8_ct'] = ct_save_and_compile(model_int8, 'b16_int8')
        print(f"  Compiled: {paths['int8_ct']}")
    except Exception as e:
        print(f"  INT8 quantization failed: {e}")

    # Model 3: INT4 via coremltools
    print("\n=== Generating INT4 model via coremltools ===")
    try:
        from coremltools.models.neural_network.quantization_utils import quantize_weights
        model_int4 = quantize_weights(model_fp16, nbits=4)
        paths['int4_ct'] = ct_save_and_compile(model_int4, 'b16_int4')
        print(f"  Compiled: {paths['int4_ct']}")
    except Exception as e:
        print(f"  INT4 quantization failed: {e}")

    # Model 4: FP16 via our espresso generator
    print("\n=== Generating FP16 model via espresso generator ===")
    espresso_path = '/tmp/b16_espresso.mlmodelc'
    gen_conv_mlmodelc(espresso_path, W, in_ch, out_ch)
    paths['fp16_esp'] = espresso_path
    print(f"  Generated: {espresso_path}")

    return paths


def compile_and_capture(model_path, hwx_output):
    """Compile .mlmodelc via aned and capture .hwx"""
    if not os.path.exists(CAPTURE_TOOL):
        print(f"  WARNING: {CAPTURE_TOOL} not found")
        return False
    result = subprocess.run([CAPTURE_TOOL, model_path, hwx_output],
                           capture_output=True, text=True, timeout=30)
    stderr = result.stderr.strip()
    if len(stderr) > 300:
        stderr = stderr[:300] + "..."
    print(f"  capture: {stderr}")
    return os.path.exists(hwx_output) and os.path.getsize(hwx_output) > 0


def parse_hwx(path):
    """Parse .hwx Mach-O-like binary."""
    with open(path, 'rb') as f:
        data = f.read()

    info = {
        'size': len(data),
        'magic': struct.unpack_from('<I', data, 0)[0],
        'ncmds': struct.unpack_from('<I', data, 0x10)[0],
        'sections': [],
        'text_data': None,
        'kern_data': None,
        'load_cmds': [],
    }

    offset = 32
    for i in range(info['ncmds']):
        if offset + 8 > len(data):
            break
        cmd, cmdsize = struct.unpack_from('<II', data, offset)
        info['load_cmds'].append({'cmd': cmd, 'cmdsize': cmdsize, 'offset': offset})

        if cmd == 0x19:  # LC_SEGMENT_64
            segname = data[offset+8:offset+24].split(b'\x00')[0].decode('ascii', errors='replace')
            nsects = struct.unpack_from('<I', data, offset+64)[0]

            for s in range(nsects):
                s_off = offset + 72 + s * 80
                sectname = data[s_off:s_off+16].split(b'\x00')[0].decode('ascii', errors='replace')
                size = struct.unpack_from('<Q', data, s_off+40)[0]
                foff = struct.unpack_from('<I', data, s_off+48)[0]

                sec_data = data[foff:foff+size] if foff + size <= len(data) else b''
                info['sections'].append({
                    'segment': segname, 'section': sectname,
                    'offset': foff, 'size': size, 'data': sec_data
                })

                if segname == '__TEXT' and sectname == '__text':
                    info['text_data'] = sec_data
                elif segname.startswith('__KERN'):
                    info['kern_data'] = sec_data

        offset += cmdsize

    return info


def diff_hwx(info1, info2, label1, label2):
    """Diff two parsed .hwx files."""
    print(f"\n{'='*60}")
    print(f"Diff: {label1} vs {label2}")
    print(f"{'='*60}")
    print(f"  Total size: {info1['size']} vs {info2['size']} ({info2['size'] - info1['size']:+d})")
    print(f"  ncmds: {info1['ncmds']} vs {info2['ncmds']}")
    print(f"  Load cmd types: {[hex(lc['cmd']) for lc in info1['load_cmds']]} vs {[hex(lc['cmd']) for lc in info2['load_cmds']]}")

    # Section comparison
    secs1 = {(s['segment'], s['section']): s for s in info1['sections']}
    secs2 = {(s['segment'], s['section']): s for s in info2['sections']}

    all_keys = sorted(set(list(secs1.keys()) + list(secs2.keys())))
    for key in all_keys:
        s1 = secs1.get(key)
        s2 = secs2.get(key)
        if s1 and s2:
            if s1['size'] == s2['size']:
                byte_diffs = sum(1 for a, b in zip(s1['data'], s2['data']) if a != b)
                match = f"SAME size={s1['size']}, {byte_diffs} byte diffs"
            else:
                match = f"DIFF size: {s1['size']} vs {s2['size']}"
            print(f"  {key[0]}.{key[1]:16s}: {match}")
        elif s1:
            print(f"  {key[0]}.{key[1]:16s}: only in {label1} (size={s1['size']})")
        else:
            print(f"  {key[0]}.{key[1]:16s}: only in {label2} (size={s2['size']})")

    # __text word-level diff
    if info1['text_data'] and info2['text_data']:
        t1, t2 = info1['text_data'], info2['text_data']
        if len(t1) == len(t2):
            nwords = len(t1) // 4
            word_diffs = []
            for w in range(nwords):
                v1 = struct.unpack_from('<I', t1, w*4)[0]
                v2 = struct.unpack_from('<I', t2, w*4)[0]
                if v1 != v2:
                    word_diffs.append((w, v1, v2))
            print(f"\n  __text word diffs ({len(word_diffs)}/{nwords}):")
            for w, v1, v2 in word_diffs[:30]:
                print(f"    W[{w:3d}]: 0x{v1:08x} -> 0x{v2:08x}")
        else:
            print(f"\n  __text: DIFFERENT LENGTHS ({len(t1)} vs {len(t2)})")

    # __KERN_0 summary
    if info1['kern_data'] and info2['kern_data']:
        k1, k2 = info1['kern_data'], info2['kern_data']
        print(f"\n  __KERN_0: {len(k1)} vs {len(k2)} bytes")
        if len(k1) == len(k2):
            diffs = sum(1 for a, b in zip(k1, k2) if a != b)
            print(f"    {diffs} byte diffs out of {len(k1)}")
            if diffs > 0 and len(k1) > 0:
                pct = diffs / len(k1) * 100
                print(f"    ({pct:.1f}% different)")


def inspect_espresso_weights(model_path, label):
    """Inspect the espresso weights file format."""
    weights_path = os.path.join(model_path, 'model.espresso.weights')
    if not os.path.exists(weights_path):
        return
    size = os.path.getsize(weights_path)
    with open(weights_path, 'rb') as f:
        data = f.read(min(128, size))
    print(f"\n  {label} weights: {size} bytes")
    print(f"    Header (64B): {data[:64].hex()}")
    if size > 64:
        # Check if FP32 or packed INT8
        sample = data[64:68] if len(data) > 67 else b''
        if sample:
            fp32_val = struct.unpack('<f', sample)[0]
            print(f"    First weight bytes: {sample.hex()} (as FP32: {fp32_val:.6f})")


def main():
    print("=== B16: Data Format Probing ===\n")

    paths = generate_models()

    # Inspect espresso weight formats before compilation
    for label, path in paths.items():
        if os.path.isdir(path):
            inspect_espresso_weights(path, label)

    # Compile each and capture .hwx
    hwx_files = {}
    for label, path in paths.items():
        if not os.path.isdir(path):
            print(f"\n--- Skipping {label} (path missing: {path}) ---")
            continue
        hwx_path = f'/tmp/b16_{label}.hwx'
        print(f"\n--- Compiling {label}: {path} ---")
        if compile_and_capture(path, hwx_path):
            hwx_files[label] = parse_hwx(hwx_path)
            print(f"  OK: {hwx_files[label]['size']} bytes, ncmds={hwx_files[label]['ncmds']}")
        else:
            print(f"  FAILED to capture .hwx")

    # Diff pairs
    pairs = [
        ('fp16_ct', 'int8_ct', 'FP16_ct', 'INT8_ct'),
        ('fp16_ct', 'int4_ct', 'FP16_ct', 'INT4_ct'),
        ('fp16_ct', 'fp16_esp', 'FP16_ct', 'FP16_espresso'),
    ]
    for k1, k2, l1, l2 in pairs:
        if k1 in hwx_files and k2 in hwx_files:
            diff_hwx(hwx_files[k1], hwx_files[k2], l1, l2)

    # Summary
    print(f"\n{'='*60}")
    print("=== B16 Summary ===")
    print(f"{'='*60}")
    for label, info in hwx_files.items():
        print(f"\n  {label}:")
        print(f"    Total: {info['size']} bytes, {info['ncmds']} load cmds")
        print(f"    Load cmd types: {[hex(lc['cmd']) for lc in info['load_cmds']]}")
        for sec in info['sections']:
            print(f"    {sec['segment']}.{sec['section']}: offset=0x{sec['offset']:X} size={sec['size']}")

    print("\n=== B16 COMPLETE ===")


if __name__ == '__main__':
    main()
