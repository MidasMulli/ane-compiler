#!/usr/bin/env python3
"""
Measure ANE softmax dispatch cost at attention-relevant dimensions.

Builds standalone softmax models via MIL IR, dispatches via ct.predict
with CPU_AND_NE compute units (same path as production Llama-1B).

Tests at dimensions matching attention score vectors:
  - kv_len values: 1, 16, 64, 128, 256, 512, 1024, 2048
  - n_heads = 32 (Llama-1B Q heads)

For each: measure dispatch latency per softmax call and compare
against the C/Accelerate softmax in llama_cpu_ops.c.

Copyright 2026 Nick Lo. MIT License.
"""

import os
import sys
import time
import ctypes
import numpy as np
import warnings
warnings.filterwarnings('ignore')


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BUILD_DIR = '/tmp/bench_ane_softmax'
LIB_PATH = os.path.join(SCRIPT_DIR, 'libllama_cpu_ops.dylib')


def build_softmax_model(dim, save_dir):
    """Build a MIL IR softmax model for a given dimension."""
    import coremltools as ct
    from coremltools.converters.mil import Builder as mb
    from coremltools.converters.mil.mil import types

    path = os.path.join(save_dir, f'softmax_{dim}.mlpackage')
    if os.path.exists(path):
        return ct.models.MLModel(path, compute_units=ct.ComputeUnit.CPU_AND_NE)

    @mb.program(input_specs=[
        mb.TensorSpec(shape=(1, dim, 1, 1), dtype=types.fp16),
    ])
    def softmax_prog(x):
        xf = mb.reshape(x=x, shape=[1, dim])
        sm = mb.softmax(x=xf, axis=1)
        return mb.reshape(x=sm, shape=[1, dim, 1, 1])

    model = ct.convert(
        softmax_prog,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.iOS18,
    )
    model.save(path)
    return ct.models.MLModel(path, compute_units=ct.ComputeUnit.CPU_AND_NE)


def build_full_attention_score_model(n_heads, kv_len, head_dim, save_dir):
    """Build a model that does softmax on attention scores: [n_heads, kv_len] -> softmax per head.

    This is closer to what a real attention-on-ANE dispatch would look like:
    n_heads independent softmax operations over kv_len values.
    """
    import coremltools as ct
    from coremltools.converters.mil import Builder as mb
    from coremltools.converters.mil.mil import types

    path = os.path.join(save_dir, f'attn_softmax_{n_heads}h_{kv_len}kv.mlpackage')
    if os.path.exists(path):
        return ct.models.MLModel(path, compute_units=ct.ComputeUnit.CPU_AND_NE)

    # Shape: [1, n_heads, 1, kv_len] — ANE NCHW format
    @mb.program(input_specs=[
        mb.TensorSpec(shape=(1, n_heads, 1, kv_len), dtype=types.fp16),
    ])
    def attn_softmax(scores):
        # Softmax over the last dimension (kv_len)
        sm = mb.softmax(x=scores, axis=-1)
        return sm

    model = ct.convert(
        attn_softmax,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.iOS18,
    )
    model.save(path)
    return ct.models.MLModel(path, compute_units=ct.ComputeUnit.CPU_AND_NE)


def bench_ane_softmax(model, input_shape, input_name='x', n_iters=200):
    """Benchmark ANE softmax dispatch."""
    np.random.seed(42)
    x = np.random.randn(*input_shape).astype(np.float16).astype(np.float32)

    # Warmup
    for _ in range(20):
        model.predict({input_name: x})

    times = []
    for _ in range(n_iters):
        t0 = time.perf_counter_ns()
        model.predict({input_name: x})
        t1 = time.perf_counter_ns()
        times.append(t1 - t0)

    return times


def bench_cpu_softmax_via_c(lib, n_heads, kv_len, n_iters=200):
    """Benchmark C/Accelerate softmax at attention-relevant sizes.
    Simulates: for each of 32 heads, softmax over kv_len scores."""
    np.random.seed(42)
    scores = np.random.randn(n_heads, kv_len).astype(np.float32)
    scores = np.ascontiguousarray(scores)

    lib.llama_softmax.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.llama_softmax.restype = None

    # Warmup
    for _ in range(20):
        tmp = scores.copy()
        for h in range(n_heads):
            lib.llama_softmax(tmp[h:h+1].ctypes.data, kv_len)

    times = []
    for _ in range(n_iters):
        tmp = scores.copy()
        t0 = time.perf_counter_ns()
        for h in range(n_heads):
            lib.llama_softmax(tmp[h:h+1].ctypes.data, kv_len)
        t1 = time.perf_counter_ns()
        times.append(t1 - t0)

    return times


def verify_softmax_correctness(model, input_shape, input_name='x'):
    """Verify ANE softmax matches numpy reference."""
    np.random.seed(123)
    x = np.random.randn(*input_shape).astype(np.float16)
    x_f32 = x.astype(np.float32)

    # Numpy reference
    if len(input_shape) == 4:
        if input_shape[2] == 1 and input_shape[3] > 1:
            # [1, n_heads, 1, kv_len] — softmax over last dim
            from scipy.special import softmax as sp_softmax
            ref = sp_softmax(x_f32, axis=-1)
        else:
            xf = x_f32.reshape(1, -1)
            exp_x = np.exp(xf - np.max(xf, axis=1, keepdims=True))
            ref = (exp_x / np.sum(exp_x, axis=1, keepdims=True)).reshape(input_shape)
    else:
        xf = x_f32.reshape(1, -1)
        exp_x = np.exp(xf - np.max(xf, axis=1, keepdims=True))
        ref = (exp_x / np.sum(exp_x, axis=1, keepdims=True)).reshape(input_shape)

    # ANE
    result = model.predict({input_name: x_f32})
    ane_out = list(result.values())[0].astype(np.float32)

    max_diff = np.max(np.abs(ane_out - ref))
    mean_diff = np.mean(np.abs(ane_out - ref))
    return max_diff, mean_diff


def main():
    import coremltools as ct

    os.makedirs(BUILD_DIR, exist_ok=True)

    # Load C library for CPU comparison
    lib = None
    if os.path.exists(LIB_PATH):
        lib = ctypes.CDLL(LIB_PATH)

    n_heads = 32
    head_dim = 64
    kv_lengths = [1, 16, 64, 128, 256, 512, 1024, 2048]

    print("=" * 70)
    print("ANE SOFTMAX DISPATCH COST — MEASUREMENT")
    print(f"Llama-1B attention: {n_heads} heads, head_dim={head_dim}")
    print("=" * 70)

    # Part 1: Simple 1D softmax at various dims (baseline)
    print("\n--- Part 1: Simple softmax(dim) via MIL IR ---")
    print(f"  {'dim':>8} {'median_us':>10} {'p5_us':>10} {'p95_us':>10} {'correct':>10}")
    print(f"  {'-'*8} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

    for dim in [64, 128, 256, 512, 1024, 2048]:
        print(f"  Building softmax dim={dim}...", end='\r')
        model = build_softmax_model(dim, BUILD_DIR)
        times = bench_ane_softmax(model, (1, dim, 1, 1))
        us = np.array(times) / 1000.0
        max_diff, mean_diff = verify_softmax_correctness(model, (1, dim, 1, 1))
        correct = "PASS" if max_diff < 1e-2 else f"FAIL({max_diff:.4f})"
        print(f"  {dim:>8} {np.median(us):>10.1f} {np.percentile(us, 5):>10.1f} "
              f"{np.percentile(us, 95):>10.1f} {correct:>10}")

    # Part 2: Attention-shaped softmax: [1, n_heads, 1, kv_len]
    print(f"\n--- Part 2: Attention softmax [{n_heads} heads x kv_len] ---")
    print(f"  {'kv_len':>8} {'ane_med_us':>10} {'ane_p95_us':>10} {'cpu_med_us':>10} "
          f"{'ratio':>8} {'correct':>10}")
    print(f"  {'-'*8} {'-'*10} {'-'*10} {'-'*10} {'-'*8} {'-'*10}")

    for kv_len in kv_lengths:
        if kv_len < 2:
            # MIL softmax needs at least 2 elements
            print(f"  {kv_len:>8} {'skip':>10} {'':>10} {'':>10} {'':>8} {'':>10}")
            continue

        print(f"  Building attn softmax kv={kv_len}...", end='\r')
        model = build_full_attention_score_model(n_heads, kv_len, head_dim, BUILD_DIR)

        # ANE timing
        times = bench_ane_softmax(model, (1, n_heads, 1, kv_len), input_name='scores')
        ane_us = np.array(times) / 1000.0

        # CPU timing (if lib available)
        cpu_med = 0
        if lib:
            cpu_times = bench_cpu_softmax_via_c(lib, n_heads, kv_len)
            cpu_us = np.array(cpu_times) / 1000.0
            cpu_med = np.median(cpu_us)

        # Correctness
        try:
            max_diff, mean_diff = verify_softmax_correctness(
                model, (1, n_heads, 1, kv_len), input_name='scores')
            correct = "PASS" if max_diff < 1e-2 else f"FAIL({max_diff:.4f})"
        except Exception as e:
            correct = f"ERR"

        ane_med = np.median(ane_us)
        ratio = f"{ane_med/cpu_med:.1f}x" if cpu_med > 0 else "n/a"

        print(f"  {kv_len:>8} {ane_med:>10.1f} {np.percentile(ane_us, 95):>10.1f} "
              f"{cpu_med:>10.1f} {ratio:>8} {correct:>10}")

    # Part 3: Cost of N softmax dispatches vs 1 fused
    print(f"\n--- Part 3: Dispatch overhead summary ---")
    print(f"  ANE dispatch floor (prior measurement): 93 us")
    print(f"  16 layers × 1 softmax dispatch each = 16 × 93 = 1,488 us minimum")
    print(f"  Compare against CPU attention total from bench_cpu_attention.py")

    print("\n" + "=" * 70)
    print("RAW DATA ABOVE — INTERPRETATION SEPARATE")
    print("=" * 70)


if __name__ == '__main__':
    main()
