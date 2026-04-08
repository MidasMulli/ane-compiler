#!/usr/bin/env python3
"""
Measure CPU attention latency in isolation.

Instruments the C/Accelerate GQA attention call with high-resolution
timestamps. Tests at multiple KV cache lengths to understand scaling.

Llama-3.2-1B: 32 Q heads, 8 KV heads, head_dim=64, dim=2048.

Copyright 2026 Nick Lo. MIT License.
"""

import os
import sys
import time
import ctypes
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LIB_PATH = os.path.join(SCRIPT_DIR, 'libllama_cpu_ops.dylib')


def load_c_lib():
    if not os.path.exists(LIB_PATH):
        print(f"ERROR: {LIB_PATH} not found. Build first.")
        sys.exit(1)
    lib = ctypes.CDLL(LIB_PATH)
    lib.llama_gqa_attention.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ]
    lib.llama_gqa_attention.restype = None
    lib.llama_rope.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_double,
    ]
    lib.llama_rope.restype = None
    lib.llama_rms_norm.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_float,
    ]
    lib.llama_rms_norm.restype = None
    return lib


def bench_attention(lib, n_heads, n_kv_heads, head_dim, kv_len, n_iters=200):
    """Benchmark GQA attention at a specific KV cache length."""
    dim = n_heads * head_dim

    # Create realistic FP16 data
    np.random.seed(42)
    q = np.random.randn(dim).astype(np.float16)
    k_cache = np.random.randn(kv_len, n_kv_heads, head_dim).astype(np.float16)
    v_cache = np.random.randn(kv_len, n_kv_heads, head_dim).astype(np.float16)
    out = np.empty(dim, dtype=np.float16)

    # Ensure contiguous
    q = np.ascontiguousarray(q)
    k_flat = np.ascontiguousarray(k_cache.ravel())
    v_flat = np.ascontiguousarray(v_cache.ravel())

    # Warmup
    for _ in range(20):
        lib.llama_gqa_attention(
            q.ctypes.data, k_flat.ctypes.data, v_flat.ctypes.data, out.ctypes.data,
            n_heads, n_kv_heads, head_dim, kv_len)

    # Timed
    times = []
    for _ in range(n_iters):
        t0 = time.perf_counter_ns()
        lib.llama_gqa_attention(
            q.ctypes.data, k_flat.ctypes.data, v_flat.ctypes.data, out.ctypes.data,
            n_heads, n_kv_heads, head_dim, kv_len)
        t1 = time.perf_counter_ns()
        times.append(t1 - t0)

    return times


def bench_rope(lib, n_heads, n_kv_heads, head_dim, n_iters=200):
    """Benchmark RoPE at a fixed position."""
    np.random.seed(42)
    q = np.random.randn(n_heads * head_dim).astype(np.float16)
    k = np.random.randn(n_kv_heads * head_dim).astype(np.float16)
    q_out = np.empty_like(q)
    k_out = np.empty_like(k)

    q = np.ascontiguousarray(q)
    k = np.ascontiguousarray(k)

    # Warmup
    for _ in range(20):
        lib.llama_rope(
            q.ctypes.data, k.ctypes.data, q_out.ctypes.data, k_out.ctypes.data,
            n_heads, n_kv_heads, head_dim, 100, ctypes.c_double(500000.0))

    times = []
    for _ in range(n_iters):
        t0 = time.perf_counter_ns()
        lib.llama_rope(
            q.ctypes.data, k.ctypes.data, q_out.ctypes.data, k_out.ctypes.data,
            n_heads, n_kv_heads, head_dim, 100, ctypes.c_double(500000.0))
        t1 = time.perf_counter_ns()
        times.append(t1 - t0)

    return times


def bench_rms_norm(lib, dim, n_iters=200):
    """Benchmark RMSNorm."""
    np.random.seed(42)
    x = np.random.randn(dim).astype(np.float16)
    w = np.random.randn(dim).astype(np.float32)
    out = np.empty(dim, dtype=np.float16)

    x = np.ascontiguousarray(x)
    w = np.ascontiguousarray(w)

    for _ in range(20):
        lib.llama_rms_norm(x.ctypes.data, w.ctypes.data, out.ctypes.data, dim, ctypes.c_float(1e-5))

    times = []
    for _ in range(n_iters):
        t0 = time.perf_counter_ns()
        lib.llama_rms_norm(x.ctypes.data, w.ctypes.data, out.ctypes.data, dim, ctypes.c_float(1e-5))
        t1 = time.perf_counter_ns()
        times.append(t1 - t0)

    return times


def main():
    lib = load_c_lib()

    # Llama-3.2-1B config
    n_heads = 32
    n_kv_heads = 8
    head_dim = 64
    dim = 2048
    n_layers = 16

    print("=" * 70)
    print("CPU ATTENTION LATENCY — ISOLATION BENCHMARK")
    print(f"Llama-3.2-1B: {n_heads}Q/{n_kv_heads}KV heads, dim={dim}, {n_layers} layers")
    print("=" * 70)

    # Benchmark RMSNorm (for comparison)
    print("\n--- RMSNorm (C/Accelerate) ---")
    rms_times = bench_rms_norm(lib, dim)
    rms_us = np.array(rms_times) / 1000.0
    print(f"  Median: {np.median(rms_us):.1f} us")
    print(f"  P95:    {np.percentile(rms_us, 95):.1f} us")

    # Benchmark RoPE (for comparison)
    print("\n--- RoPE (C/Accelerate) ---")
    rope_times = bench_rope(lib, n_heads, n_kv_heads, head_dim)
    rope_us = np.array(rope_times) / 1000.0
    print(f"  Median: {np.median(rope_us):.1f} us")
    print(f"  P95:    {np.percentile(rope_us, 95):.1f} us")

    # Benchmark GQA attention at multiple KV lengths
    print("\n--- GQA Attention (C/Accelerate) ---")
    print(f"  {'kv_len':>8} {'median_us':>10} {'p5_us':>10} {'p95_us':>10} {'x16_layers_ms':>14}")
    print(f"  {'-'*8} {'-'*10} {'-'*10} {'-'*10} {'-'*14}")

    kv_lengths = [1, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]

    for kv_len in kv_lengths:
        times = bench_attention(lib, n_heads, n_kv_heads, head_dim, kv_len)
        us = np.array(times) / 1000.0
        med = np.median(us)
        p5 = np.percentile(us, 5)
        p95 = np.percentile(us, 95)
        total_16_ms = med * n_layers / 1000.0
        print(f"  {kv_len:>8} {med:>10.1f} {p5:>10.1f} {p95:>10.1f} {total_16_ms:>14.3f}")

    # Full forward pass simulation: RMSNorm + RoPE + attention for all 16 layers
    print("\n--- Full CPU ops per token (16 layers) ---")
    # Test at typical decode positions
    for kv_len in [16, 64, 128, 256, 512]:
        np.random.seed(42)
        q = np.random.randn(dim).astype(np.float16)
        k_cache = np.random.randn(kv_len, n_kv_heads, head_dim).astype(np.float16)
        v_cache = np.random.randn(kv_len, n_kv_heads, head_dim).astype(np.float16)
        out = np.empty(dim, dtype=np.float16)
        rms_x = np.random.randn(dim).astype(np.float16)
        rms_w = np.random.randn(dim).astype(np.float32)
        rms_out = np.empty(dim, dtype=np.float16)
        q_rope_out = np.empty(n_heads * head_dim, dtype=np.float16)
        k_rope_in = np.random.randn(n_kv_heads * head_dim).astype(np.float16)
        k_rope_out = np.empty(n_kv_heads * head_dim, dtype=np.float16)

        q = np.ascontiguousarray(q)
        k_flat = np.ascontiguousarray(k_cache.ravel())
        v_flat = np.ascontiguousarray(v_cache.ravel())
        rms_x = np.ascontiguousarray(rms_x)
        rms_w = np.ascontiguousarray(rms_w)
        k_rope_in = np.ascontiguousarray(k_rope_in)

        # Warmup
        for _ in range(10):
            for li in range(n_layers):
                lib.llama_rms_norm(rms_x.ctypes.data, rms_w.ctypes.data, rms_out.ctypes.data, dim, ctypes.c_float(1e-5))
                lib.llama_rope(q.ctypes.data, k_rope_in.ctypes.data, q_rope_out.ctypes.data, k_rope_out.ctypes.data,
                    n_heads, n_kv_heads, head_dim, 100, ctypes.c_double(500000.0))
                lib.llama_gqa_attention(q.ctypes.data, k_flat.ctypes.data, v_flat.ctypes.data, out.ctypes.data,
                    n_heads, n_kv_heads, head_dim, kv_len)

        # Timed: all CPU ops for 16 layers
        n_iters = 100
        total_times = []
        attn_only_times = []
        for _ in range(n_iters):
            t_total_start = time.perf_counter_ns()
            t_attn_accum = 0
            for li in range(n_layers):
                lib.llama_rms_norm(rms_x.ctypes.data, rms_w.ctypes.data, rms_out.ctypes.data, dim, ctypes.c_float(1e-5))
                lib.llama_rope(q.ctypes.data, k_rope_in.ctypes.data, q_rope_out.ctypes.data, k_rope_out.ctypes.data,
                    n_heads, n_kv_heads, head_dim, 100, ctypes.c_double(500000.0))
                t_a0 = time.perf_counter_ns()
                lib.llama_gqa_attention(q.ctypes.data, k_flat.ctypes.data, v_flat.ctypes.data, out.ctypes.data,
                    n_heads, n_kv_heads, head_dim, kv_len)
                t_a1 = time.perf_counter_ns()
                t_attn_accum += (t_a1 - t_a0)
            t_total_end = time.perf_counter_ns()
            total_times.append((t_total_end - t_total_start))
            attn_only_times.append(t_attn_accum)

        total_ms = np.median(total_times) / 1e6
        attn_ms = np.median(attn_only_times) / 1e6
        other_ms = total_ms - attn_ms
        print(f"  kv_len={kv_len:>4}: total_cpu={total_ms:.3f}ms  "
              f"attn={attn_ms:.3f}ms ({attn_ms/total_ms*100:.0f}%)  "
              f"rope+rms={other_ms:.3f}ms ({other_ms/total_ms*100:.0f}%)")

    print("\n" + "=" * 70)
    print("INTERPRETATION NOTES (raw data above, analysis separate)")
    print("=" * 70)
    print("  Compare attention_total_ms against ANE dispatch floor:")
    print("  ANE dispatch floor = 93us. 16 layers = 16 * 93us = 1.49ms minimum.")
    print("  If CPU attention < 1.49ms, ANE can't beat it at dispatch level alone.")


if __name__ == '__main__':
    main()
