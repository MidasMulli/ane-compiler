#!/usr/bin/env python3
"""B11: aned hang investigation for large conv models (in_ch x out_ch > ~1.3M)

Tests:
1. Reproduce hang with 768x3072 conv via espresso generator
2. Test coremltools NeuralNetworkBuilder alternative
3. Test split workaround (3x 768->1024)
4. Binary search for the channel product limit
"""

import sys, os, subprocess, time, signal
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.compiler import gen_conv_mlmodelc

CAPTURE_TOOL = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'capture_and_eval')
TIMEOUT = 15  # seconds per compilation attempt


def try_compile(model_path, label, timeout=TIMEOUT):
    """Try to compile a .mlmodelc, return (success, time_taken, output)"""
    print(f"\n--- {label} ---")
    print(f"  Model: {model_path}")

    if not os.path.exists(CAPTURE_TOOL):
        print(f"  ERROR: {CAPTURE_TOOL} not found")
        return False, 0, "tool missing"

    hwx_path = f'/tmp/b11_{label}.hwx'
    start = time.time()
    try:
        result = subprocess.run(
            [CAPTURE_TOOL, model_path, hwx_path],
            capture_output=True, text=True, timeout=timeout)
        elapsed = time.time() - start
        success = os.path.exists(hwx_path) and os.path.getsize(hwx_path) > 0
        print(f"  Time: {elapsed:.1f}s")
        print(f"  Result: {'PASS' if success else 'FAIL'}")
        if result.stderr:
            # Truncate long output
            stderr = result.stderr.strip()
            if len(stderr) > 500:
                stderr = stderr[:500] + "..."
            print(f"  stderr: {stderr}")
        return success, elapsed, result.stderr
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start
        print(f"  TIMEOUT after {elapsed:.1f}s (limit={timeout}s)")
        # Kill any lingering aned processes that might be hung
        # (Don't actually kill aned -- it's a system daemon)
        return False, elapsed, "TIMEOUT"


def gen_espresso_model(in_ch, out_ch, path):
    """Generate .mlmodelc via our espresso generator."""
    np.random.seed(42)
    W = np.random.randn(out_ch, in_ch).astype(np.float32)
    gen_conv_mlmodelc(path, W, in_ch, out_ch)
    return path


def gen_coremltools_model(in_ch, out_ch, path):
    """Generate .mlmodelc via coremltools NeuralNetworkBuilder."""
    import coremltools as ct
    from coremltools.models.neural_network import NeuralNetworkBuilder
    import shutil

    np.random.seed(42)
    W = np.random.randn(out_ch, in_ch).astype(np.float32)

    builder = NeuralNetworkBuilder(
        input_features=[('input', ct.models.datatypes.Array(in_ch))],
        output_features=[('output', ct.models.datatypes.Array(out_ch))])
    builder.add_inner_product('fc', W, None, in_ch, out_ch,
                              has_bias=False,
                              input_name='input', output_name='output')
    model = ct.models.MLModel(builder.spec)
    mlmodel_path = path.replace('.mlmodelc', '.mlmodel')
    model.save(mlmodel_path)
    if os.path.exists(path):
        shutil.rmtree(path)
    import subprocess
    subprocess.run(['xcrun', 'coremlcompiler', 'compile', mlmodel_path,
                    os.path.dirname(path)], check=True, capture_output=True)
    return path


def gen_coremltools_conv2d_model(in_ch, out_ch, path):
    """Generate .mlmodelc via coremltools using explicit conv2d layer."""
    import coremltools as ct
    from coremltools.models.neural_network import NeuralNetworkBuilder
    import shutil

    np.random.seed(42)
    W = np.random.randn(out_ch, in_ch, 1, 1).astype(np.float32)

    builder = NeuralNetworkBuilder(
        input_features=[('input', ct.models.datatypes.Array(in_ch, 1, 1))],
        output_features=[('output', ct.models.datatypes.Array(out_ch, 1, 1))])
    builder.add_convolution('conv', kernel_channels=in_ch, output_channels=out_ch,
                           height=1, width=1,
                           stride_height=1, stride_width=1,
                           border_mode='valid',
                           groups=1,
                           W=W, b=None, has_bias=False,
                           input_name='input', output_name='output')
    model = ct.models.MLModel(builder.spec)
    mlmodel_path = path.replace('.mlmodelc', '.mlmodel')
    model.save(mlmodel_path)
    if os.path.exists(path):
        shutil.rmtree(path)
    import subprocess
    subprocess.run(['xcrun', 'coremlcompiler', 'compile', mlmodel_path,
                    os.path.dirname(path)], check=True, capture_output=True)
    return path


def gen_torch_mlprogram(in_ch, out_ch, path):
    """Generate .mlmodelc via PyTorch -> coremltools ML Program conversion."""
    try:
        import torch
        import coremltools as ct

        np.random.seed(42)

        class LinearModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = torch.nn.Linear(in_ch, out_ch, bias=False)
                self.fc.weight.data = torch.from_numpy(
                    np.random.randn(out_ch, in_ch).astype(np.float32))
            def forward(self, x):
                return self.fc(x)

        model = LinearModel()
        traced = torch.jit.trace(model, torch.randn(1, in_ch))
        ct_model = ct.convert(traced,
                              inputs=[ct.TensorType(shape=(1, in_ch))],
                              compute_precision=ct.precision.FLOAT16)
        import shutil
        pkg_path = path.replace('.mlmodelc', '.mlpackage')
        ct_model.save(pkg_path)
        if os.path.exists(path):
            shutil.rmtree(path)
        subprocess.run(['xcrun', 'coremlcompiler', 'compile', pkg_path,
                        os.path.dirname(path)], check=True, capture_output=True)
        return path
    except Exception as e:
        print(f"  ML Program generation failed: {e}")
        return None


def main():
    print("=== B11: aned Hang Investigation ===")
    print(f"Target: 768x3072 conv (product={768*3072:,})")
    print(f"Known limit: ~1.3M channel product\n")

    results = {}

    # PHASE 1: Reproduce hang with espresso generator
    print("=" * 60)
    print("PHASE 1: Reproduce hang (espresso generator)")
    print("=" * 60)

    # Small reference that should work
    path = gen_espresso_model(64, 64, '/tmp/b11_esp_64x64.mlmodelc')
    ok, t, _ = try_compile(path, 'esp_64x64')
    results['esp_64x64'] = (ok, t, 64*64)

    # Medium that should work
    path = gen_espresso_model(256, 256, '/tmp/b11_esp_256x256.mlmodelc')
    ok, t, _ = try_compile(path, 'esp_256x256')
    results['esp_256x256'] = (ok, t, 256*256)

    # 768x768 (product=589824 < 1.3M)
    path = gen_espresso_model(768, 768, '/tmp/b11_esp_768x768.mlmodelc')
    ok, t, _ = try_compile(path, 'esp_768x768')
    results['esp_768x768'] = (ok, t, 768*768)

    # 768x1024 (product=786432)
    path = gen_espresso_model(768, 1024, '/tmp/b11_esp_768x1024.mlmodelc')
    ok, t, _ = try_compile(path, 'esp_768x1024')
    results['esp_768x1024'] = (ok, t, 768*1024)

    # 1024x1024 (product=1048576, near limit)
    path = gen_espresso_model(1024, 1024, '/tmp/b11_esp_1024x1024.mlmodelc')
    ok, t, _ = try_compile(path, 'esp_1024x1024')
    results['esp_1024x1024'] = (ok, t, 1024*1024)

    # 768x2048 (product=1572864, above limit)
    path = gen_espresso_model(768, 2048, '/tmp/b11_esp_768x2048.mlmodelc')
    ok, t, _ = try_compile(path, 'esp_768x2048')
    results['esp_768x2048'] = (ok, t, 768*2048)

    # THE TARGET: 768x3072 (product=2359296)
    path = gen_espresso_model(768, 3072, '/tmp/b11_esp_768x3072.mlmodelc')
    ok, t, _ = try_compile(path, 'esp_768x3072')
    results['esp_768x3072'] = (ok, t, 768*3072)

    # PHASE 2: coremltools alternative paths
    print("\n" + "=" * 60)
    print("PHASE 2: coremltools paths (NeuralNetworkBuilder)")
    print("=" * 60)

    # Reference small
    path = gen_coremltools_model(64, 64, '/tmp/b11_ct_64x64.mlmodelc')
    ok, t, _ = try_compile(path, 'ct_ip_64x64')
    results['ct_ip_64x64'] = (ok, t, 64*64)

    # Target size via inner_product
    path = gen_coremltools_model(768, 3072, '/tmp/b11_ct_ip_768x3072.mlmodelc')
    ok, t, _ = try_compile(path, 'ct_ip_768x3072')
    results['ct_ip_768x3072'] = (ok, t, 768*3072)

    # Target size via conv2d
    path = gen_coremltools_conv2d_model(768, 3072, '/tmp/b11_ct_conv_768x3072.mlmodelc')
    ok, t, _ = try_compile(path, 'ct_conv_768x3072')
    results['ct_conv_768x3072'] = (ok, t, 768*3072)

    # ML Program path (torch conversion)
    print("\n--- ML Program path (torch -> coremltools) ---")
    path = gen_torch_mlprogram(768, 3072, '/tmp/b11_mlp_768x3072.mlmodelc')
    if path:
        ok, t, _ = try_compile(path, 'mlp_768x3072')
        results['mlp_768x3072'] = (ok, t, 768*3072)

    # PHASE 3: Split workaround
    print("\n" + "=" * 60)
    print("PHASE 3: Split workaround (3x 768->1024)")
    print("=" * 60)

    for i in range(3):
        path = gen_espresso_model(768, 1024, f'/tmp/b11_split_{i}.mlmodelc')
        ok, t, _ = try_compile(path, f'split_{i}_768x1024')
        results[f'split_{i}'] = (ok, t, 768*1024)

    # Also test coremltools split
    for i in range(3):
        path = gen_coremltools_model(768, 1024, f'/tmp/b11_ct_split_{i}.mlmodelc')
        ok, t, _ = try_compile(path, f'ct_split_{i}_768x1024')
        results[f'ct_split_{i}'] = (ok, t, 768*1024)

    # PHASE 4: Binary search for limit (espresso path)
    print("\n" + "=" * 60)
    print("PHASE 4: Binary search for channel product limit")
    print("=" * 60)

    # Test powers of 2 for out_ch with in_ch=768
    for out_ch in [1024, 1152, 1280, 1408, 1536, 1664, 1792, 2048]:
        path = gen_espresso_model(768, out_ch, f'/tmp/b11_search_768x{out_ch}.mlmodelc')
        ok, t, _ = try_compile(path, f'search_768x{out_ch}')
        results[f'search_768x{out_ch}'] = (ok, t, 768*out_ch)

    # PHASE 5: Compare metadata between espresso and coremltools
    print("\n" + "=" * 60)
    print("PHASE 5: Metadata comparison")
    print("=" * 60)

    import json

    for label, path in [('espresso', '/tmp/b11_esp_768x3072.mlmodelc'),
                         ('ct_ip', '/tmp/b11_ct_ip_768x3072.mlmodelc'),
                         ('ct_conv', '/tmp/b11_ct_conv_768x3072.mlmodelc')]:
        print(f"\n--- {label} ---")
        net_path = os.path.join(path, 'model.espresso.net')
        shape_path = os.path.join(path, 'model.espresso.shape')
        meta_path = os.path.join(path, 'metadata.json')
        weights_path = os.path.join(path, 'model.espresso.weights')

        if os.path.exists(net_path):
            with open(net_path) as f:
                net = json.load(f)
            print(f"  net: format_version={net.get('format_version')}, "
                  f"layers={len(net.get('layers', []))}")
            for layer in net.get('layers', []):
                print(f"    layer: type={layer.get('type')}, name={layer.get('name')}")
                # Print key fields
                for key in ['K', 'C', 'Nx', 'Ny', 'n_groups', 'has_biases',
                            'nB', 'nC', 'nH', 'nW']:
                    if key in layer:
                        print(f"      {key}={layer[key]}")

        if os.path.exists(shape_path):
            with open(shape_path) as f:
                shapes = json.load(f)
            print(f"  shapes: {json.dumps(shapes, indent=4)[:300]}")

        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            print(f"  meta: specVersion={meta.get('specificationVersion')}")

        if os.path.exists(weights_path):
            size = os.path.getsize(weights_path)
            print(f"  weights: {size} bytes")
            # Read header
            with open(weights_path, 'rb') as f:
                header = f.read(min(64, size))
            print(f"  weights header: {header[:32].hex()}")

    # SUMMARY
    print("\n" + "=" * 60)
    print("=== B11 Summary ===")
    print("=" * 60)
    print(f"{'Label':<30s} {'Pass':<6s} {'Time':<8s} {'Product':<12s}")
    print("-" * 60)
    for label, (ok, t, prod) in sorted(results.items(), key=lambda x: x[1][2]):
        status = "PASS" if ok else ("HANG" if t >= TIMEOUT-1 else "FAIL")
        print(f"{label:<30s} {status:<6s} {t:<8.1f} {prod:<12,d}")

    print("\n=== B11 COMPLETE ===")


if __name__ == '__main__':
    main()
