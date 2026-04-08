#!/usr/bin/env python3
"""
Project Chimera: Per-operation profiling of Llama 70B transformer layer.
Profiles GPU (MLX), CPU (numpy/Accelerate/AMX) across multiple context lengths.

Hardware: Apple M5 Pro, 64GB, 20 GPU cores, 16-core ANE, 307 GB/s DRAM
Model: Llama 3.3-70B-Instruct-4bit
"""

import mlx.core as mx
import numpy as np
import time
import json
import sys

# ─── Llama 3.3 70B specs ───
DIM = 8192
N_HEADS = 64
N_KV_HEADS = 8
HEAD_DIM = 128
FFN_DIM = 28672
GQA_RATIO = N_HEADS // N_KV_HEADS  # 8

SEQ_LENS = [256, 1024, 4096]
N_WARMUP = 20
N_ITERS = 100


def median(vals):
    s = sorted(vals)
    return s[len(s) // 2]


def p10(vals):
    s = sorted(vals)
    return s[len(s) // 10]


# ═══════════════════════════════════════════════════════════════════════
#  GPU (MLX) PROFILING
# ═══════════════════════════════════════════════════════════════════════

def profile_gpu():
    print("\n" + "=" * 70)
    print("  GPU (MLX) PROFILING — Apple M5 Pro, 20 cores")
    print("=" * 70)

    results = {}

    # --- Weight matrices (FP16 for compute profiling) ---
    # QKV fused: Q(8192x8192) + K(1024x8192) + V(1024x8192) = (10240, 8192)
    W_q = mx.random.normal((DIM, DIM)).astype(mx.float16)
    W_k = mx.random.normal((N_KV_HEADS * HEAD_DIM, DIM)).astype(mx.float16)  # 1024 x 8192
    W_v = mx.random.normal((N_KV_HEADS * HEAD_DIM, DIM)).astype(mx.float16)  # 1024 x 8192
    W_o = mx.random.normal((DIM, DIM)).astype(mx.float16)
    W_gate = mx.random.normal((FFN_DIM, DIM)).astype(mx.float16)
    W_up = mx.random.normal((FFN_DIM, DIM)).astype(mx.float16)
    W_down = mx.random.normal((DIM, FFN_DIM)).astype(mx.float16)
    rms_w = mx.ones((DIM,)).astype(mx.float16)

    x = mx.random.normal((1, DIM)).astype(mx.float16)
    mx.eval(W_q, W_k, W_v, W_o, W_gate, W_up, W_down, rms_w, x)

    # --- Q4 quantized weight profiling ---
    # MLX quantized matmul uses mx.quantize
    print("\n--- Quantized (Q4) weight profiling ---")

    W_q_q, W_q_s, W_q_b = mx.quantize(W_q, bits=4, group_size=64)
    W_k_q, W_k_s, W_k_b = mx.quantize(W_k, bits=4, group_size=64)
    W_v_q, W_v_s, W_v_b = mx.quantize(W_v, bits=4, group_size=64)
    W_o_q, W_o_s, W_o_b = mx.quantize(W_o, bits=4, group_size=64)
    W_gate_q, W_gate_s, W_gate_b = mx.quantize(W_gate, bits=4, group_size=64)
    W_up_q, W_up_s, W_up_b = mx.quantize(W_up, bits=4, group_size=64)
    W_down_q, W_down_s, W_down_b = mx.quantize(W_down, bits=4, group_size=64)
    mx.eval(W_q_q, W_q_s, W_q_b, W_k_q, W_k_s, W_k_b, W_v_q, W_v_s, W_v_b,
            W_o_q, W_o_s, W_o_b, W_gate_q, W_gate_s, W_gate_b,
            W_up_q, W_up_s, W_up_b, W_down_q, W_down_s, W_down_b)

    def bench_gpu(label, fn, n_warmup=N_WARMUP, n_iters=N_ITERS):
        for _ in range(n_warmup):
            fn(); mx.synchronize()
        times = []
        for _ in range(n_iters):
            t0 = time.perf_counter()
            fn()
            mx.synchronize()
            times.append((time.perf_counter() - t0) * 1000)
        med = median(times)
        p = p10(times)
        print(f"  {label:40s}: median {med:8.3f} ms   p10 {p:8.3f} ms")
        return {"median_ms": round(med, 4), "p10_ms": round(p, 4)}

    # === RMSNorm ===
    print("\n--- RMSNorm ---")
    def rmsnorm_fp16():
        n = x * mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + 1e-5) * rms_w
        mx.eval(n)
    results["rmsnorm_gpu_fp16"] = bench_gpu("RMSNorm (FP16)", rmsnorm_fp16)

    # === Q Projection ===
    print("\n--- Attention Projections (FP16) ---")
    def q_proj_fp16():
        y = x @ W_q.T; mx.eval(y)
    results["q_proj_gpu_fp16"] = bench_gpu("Q proj 8192x8192 (FP16)", q_proj_fp16)

    def k_proj_fp16():
        y = x @ W_k.T; mx.eval(y)
    results["k_proj_gpu_fp16"] = bench_gpu("K proj 1024x8192 (FP16)", k_proj_fp16)

    def v_proj_fp16():
        y = x @ W_v.T; mx.eval(y)
    results["v_proj_gpu_fp16"] = bench_gpu("V proj 1024x8192 (FP16)", v_proj_fp16)

    def o_proj_fp16():
        y = x @ W_o.T; mx.eval(y)
    results["o_proj_gpu_fp16"] = bench_gpu("O proj 8192x8192 (FP16)", o_proj_fp16)

    # === Q4 Projections ===
    print("\n--- Attention Projections (Q4) ---")
    def q_proj_q4():
        y = mx.quantized_matmul(x, W_q_q, W_q_s, W_q_b, bits=4, group_size=64)
        mx.eval(y)
    results["q_proj_gpu_q4"] = bench_gpu("Q proj 8192x8192 (Q4)", q_proj_q4)

    def k_proj_q4():
        y = mx.quantized_matmul(x, W_k_q, W_k_s, W_k_b, bits=4, group_size=64)
        mx.eval(y)
    results["k_proj_gpu_q4"] = bench_gpu("K proj 1024x8192 (Q4)", k_proj_q4)

    def v_proj_q4():
        y = mx.quantized_matmul(x, W_v_q, W_v_s, W_v_b, bits=4, group_size=64)
        mx.eval(y)
    results["v_proj_gpu_q4"] = bench_gpu("V proj 1024x8192 (Q4)", v_proj_q4)

    def o_proj_q4():
        y = mx.quantized_matmul(x, W_o_q, W_o_s, W_o_b, bits=4, group_size=64)
        mx.eval(y)
    results["o_proj_gpu_q4"] = bench_gpu("O proj 8192x8192 (Q4)", o_proj_q4)

    # === FFN (FP16) ===
    print("\n--- FFN (FP16) ---")
    def ffn_fp16():
        g = x @ W_gate.T
        u = x @ W_up.T
        h = mx.sigmoid(g) * g * u  # SiLU = sigmoid(x)*x
        y = h @ W_down.T
        mx.eval(y)
    results["ffn_gpu_fp16"] = bench_gpu("FFN full (FP16)", ffn_fp16)

    def ffn_gate_fp16():
        g = x @ W_gate.T; mx.eval(g)
    results["ffn_gate_gpu_fp16"] = bench_gpu("FFN gate 28672x8192 (FP16)", ffn_gate_fp16)

    def ffn_up_fp16():
        u = x @ W_up.T; mx.eval(u)
    results["ffn_up_gpu_fp16"] = bench_gpu("FFN up 28672x8192 (FP16)", ffn_up_fp16)

    def ffn_down_fp16():
        h = mx.random.normal((1, FFN_DIM)).astype(mx.float16)
        mx.eval(h)
        def inner():
            y = h @ W_down.T; mx.eval(y)
        return inner
    _fn = ffn_down_fp16()
    results["ffn_down_gpu_fp16"] = bench_gpu("FFN down 8192x28672 (FP16)", _fn)

    # === FFN (Q4) ===
    print("\n--- FFN (Q4) ---")
    def ffn_q4():
        g = mx.quantized_matmul(x, W_gate_q, W_gate_s, W_gate_b, bits=4, group_size=64)
        u = mx.quantized_matmul(x, W_up_q, W_up_s, W_up_b, bits=4, group_size=64)
        h = mx.sigmoid(g) * g * u
        y = mx.quantized_matmul(h, W_down_q, W_down_s, W_down_b, bits=4, group_size=64)
        mx.eval(y)
    results["ffn_gpu_q4"] = bench_gpu("FFN full (Q4)", ffn_q4)

    def ffn_gate_q4():
        g = mx.quantized_matmul(x, W_gate_q, W_gate_s, W_gate_b, bits=4, group_size=64)
        mx.eval(g)
    results["ffn_gate_gpu_q4"] = bench_gpu("FFN gate 28672x8192 (Q4)", ffn_gate_q4)

    def ffn_up_q4():
        u = mx.quantized_matmul(x, W_up_q, W_up_s, W_up_b, bits=4, group_size=64)
        mx.eval(u)
    results["ffn_up_gpu_q4"] = bench_gpu("FFN up 28672x8192 (Q4)", ffn_up_q4)

    h_buf = mx.random.normal((1, FFN_DIM)).astype(mx.float16)
    mx.eval(h_buf)
    def ffn_down_q4():
        y = mx.quantized_matmul(h_buf, W_down_q, W_down_s, W_down_b, bits=4, group_size=64)
        mx.eval(y)
    results["ffn_down_gpu_q4"] = bench_gpu("FFN down 8192x28672 (Q4)", ffn_down_q4)

    # === SiLU activation (elementwise) ===
    print("\n--- Activations ---")
    act_in = mx.random.normal((1, FFN_DIM)).astype(mx.float16)
    mx.eval(act_in)
    def silu_gpu():
        y = mx.sigmoid(act_in) * act_in
        mx.eval(y)
    results["silu_gpu"] = bench_gpu("SiLU (28672)", silu_gpu)

    def silu_mul_gpu():
        u = mx.random.normal((1, FFN_DIM)).astype(mx.float16)
        mx.eval(u)
        def inner():
            g = mx.sigmoid(act_in) * act_in * u
            mx.eval(g)
        return inner
    _fn = silu_mul_gpu()
    results["silu_mul_gpu"] = bench_gpu("SiLU * up (28672)", _fn)

    # === Attention at various context lengths ===
    for seq_len in SEQ_LENS:
        print(f"\n--- Attention (seq_len={seq_len}) ---")
        Q = mx.random.normal((1, N_HEADS, 1, HEAD_DIM)).astype(mx.float16)
        K = mx.random.normal((1, N_KV_HEADS, seq_len, HEAD_DIM)).astype(mx.float16)
        V = mx.random.normal((1, N_KV_HEADS, seq_len, HEAD_DIM)).astype(mx.float16)
        mx.eval(Q, K, V)

        # GQA expand + manual attention
        K_rep = mx.repeat(K, GQA_RATIO, axis=1)
        V_rep = mx.repeat(V, GQA_RATIO, axis=1)
        mx.eval(K_rep, V_rep)

        def attn_full(Q=Q, K_rep=K_rep, V_rep=V_rep):
            scores = (Q @ mx.transpose(K_rep, (0, 1, 3, 2))) / (HEAD_DIM ** 0.5)
            weights = mx.softmax(scores, axis=-1)
            out = weights @ V_rep
            out = out.reshape(1, -1)
            mx.eval(out)

        results[f"attn_full_gpu_seq{seq_len}"] = bench_gpu(
            f"Attention full (seq={seq_len})", attn_full)

        # Score computation only
        def attn_scores(Q=Q, K_rep=K_rep):
            scores = (Q @ mx.transpose(K_rep, (0, 1, 3, 2))) / (HEAD_DIM ** 0.5)
            mx.eval(scores)

        results[f"attn_scores_gpu_seq{seq_len}"] = bench_gpu(
            f"  QK scores (seq={seq_len})", attn_scores)

        # Softmax only
        scores_buf = mx.random.normal((1, N_HEADS, 1, seq_len)).astype(mx.float16)
        mx.eval(scores_buf)
        def attn_softmax(s=scores_buf):
            w = mx.softmax(s, axis=-1)
            mx.eval(w)

        results[f"attn_softmax_gpu_seq{seq_len}"] = bench_gpu(
            f"  Softmax (seq={seq_len})", attn_softmax)

        # Weighted sum only
        weights_buf = mx.random.normal((1, N_HEADS, 1, seq_len)).astype(mx.float16)
        mx.eval(weights_buf)
        def attn_wv(w=weights_buf, V_rep=V_rep):
            out = w @ V_rep
            mx.eval(out)

        results[f"attn_wv_gpu_seq{seq_len}"] = bench_gpu(
            f"  Weighted V (seq={seq_len})", attn_wv)

        # GQA repeat cost
        def gqa_repeat(K=K, V=V):
            Kr = mx.repeat(K, GQA_RATIO, axis=1)
            Vr = mx.repeat(V, GQA_RATIO, axis=1)
            mx.eval(Kr, Vr)

        results[f"gqa_repeat_gpu_seq{seq_len}"] = bench_gpu(
            f"  GQA repeat K+V (seq={seq_len})", gqa_repeat)

    # === RoPE ===
    print("\n--- RoPE ---")
    q_rope = mx.random.normal((1, N_HEADS, 1, HEAD_DIM)).astype(mx.float16)
    k_rope = mx.random.normal((1, N_KV_HEADS, 1, HEAD_DIM)).astype(mx.float16)
    cos_cache = mx.random.normal((1, 1, 1, HEAD_DIM)).astype(mx.float16)
    sin_cache = mx.random.normal((1, 1, 1, HEAD_DIM)).astype(mx.float16)
    mx.eval(q_rope, k_rope, cos_cache, sin_cache)

    def apply_rope(t=q_rope, cos=cos_cache, sin=sin_cache):
        t1 = t[..., :HEAD_DIM // 2]
        t2 = t[..., HEAD_DIM // 2:]
        out = mx.concatenate([t1 * cos[..., :HEAD_DIM // 2] - t2 * sin[..., :HEAD_DIM // 2],
                               t2 * cos[..., HEAD_DIM // 2:] + t1 * sin[..., HEAD_DIM // 2:]], axis=-1)
        mx.eval(out)

    results["rope_gpu"] = bench_gpu("RoPE Q+K", apply_rope)

    # === Residual add ===
    print("\n--- Residual ---")
    r1 = mx.random.normal((1, DIM)).astype(mx.float16)
    r2 = mx.random.normal((1, DIM)).astype(mx.float16)
    mx.eval(r1, r2)
    def residual_add(a=r1, b=r2):
        y = a + b; mx.eval(y)
    results["residual_add_gpu"] = bench_gpu("Residual add (8192)", residual_add)

    return results


# ═══════════════════════════════════════════════════════════════════════
#  CPU (Numpy / Accelerate / AMX) PROFILING
# ═══════════════════════════════════════════════════════════════════════

def profile_cpu():
    print("\n" + "=" * 70)
    print("  CPU (Numpy/Accelerate/AMX) PROFILING")
    print("=" * 70)

    results = {}

    def bench_cpu(label, fn, n_warmup=N_WARMUP, n_iters=N_ITERS):
        for _ in range(n_warmup):
            fn()
        times = []
        for _ in range(n_iters):
            t0 = time.perf_counter()
            fn()
            times.append((time.perf_counter() - t0) * 1000)
        med = median(times)
        p = p10(times)
        print(f"  {label:40s}: median {med:8.3f} ms   p10 {p:8.3f} ms")
        return {"median_ms": round(med, 4), "p10_ms": round(p, 4)}

    # FP32 weights (AMX uses FP32)
    W_q = np.random.randn(DIM, DIM).astype(np.float32)
    W_k = np.random.randn(N_KV_HEADS * HEAD_DIM, DIM).astype(np.float32)
    W_v = np.random.randn(N_KV_HEADS * HEAD_DIM, DIM).astype(np.float32)
    W_o = np.random.randn(DIM, DIM).astype(np.float32)
    W_gate = np.random.randn(FFN_DIM, DIM).astype(np.float32)
    W_up = np.random.randn(FFN_DIM, DIM).astype(np.float32)
    W_down = np.random.randn(DIM, FFN_DIM).astype(np.float32)
    rms_w = np.ones(DIM, dtype=np.float32)
    x = np.random.randn(1, DIM).astype(np.float32)

    # Also FP16
    W_q_f16 = W_q.astype(np.float16)
    W_k_f16 = W_k.astype(np.float16)
    W_o_f16 = W_o.astype(np.float16)
    W_gate_f16 = W_gate.astype(np.float16)
    W_up_f16 = W_up.astype(np.float16)
    W_down_f16 = W_down.astype(np.float16)
    x_f16 = x.astype(np.float16)

    # === RMSNorm ===
    print("\n--- RMSNorm ---")
    def rmsnorm_cpu():
        n = x * (1.0 / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + 1e-5)) * rms_w
        return n
    results["rmsnorm_cpu_fp32"] = bench_cpu("RMSNorm (FP32)", rmsnorm_cpu)

    # === Projections FP32 ===
    print("\n--- Attention Projections (FP32 / AMX) ---")
    def q_proj_cpu():
        return x @ W_q.T
    results["q_proj_cpu_fp32"] = bench_cpu("Q proj 8192x8192 (FP32)", q_proj_cpu)

    def k_proj_cpu():
        return x @ W_k.T
    results["k_proj_cpu_fp32"] = bench_cpu("K proj 1024x8192 (FP32)", k_proj_cpu)

    def o_proj_cpu():
        return x @ W_o.T
    results["o_proj_cpu_fp32"] = bench_cpu("O proj 8192x8192 (FP32)", o_proj_cpu)

    # === FFN FP32 ===
    print("\n--- FFN (FP32 / AMX) ---")
    def ffn_gate_cpu():
        return x @ W_gate.T
    results["ffn_gate_cpu_fp32"] = bench_cpu("FFN gate 28672x8192 (FP32)", ffn_gate_cpu)

    def ffn_up_cpu():
        return x @ W_up.T
    results["ffn_up_cpu_fp32"] = bench_cpu("FFN up 28672x8192 (FP32)", ffn_up_cpu)

    h_cpu = np.random.randn(1, FFN_DIM).astype(np.float32)
    def ffn_down_cpu():
        return h_cpu @ W_down.T
    results["ffn_down_cpu_fp32"] = bench_cpu("FFN down 8192x28672 (FP32)", ffn_down_cpu)

    def ffn_full_cpu():
        g = x @ W_gate.T
        u = x @ W_up.T
        h = (1.0 / (1.0 + np.exp(-g))) * g * u  # SiLU
        return h @ W_down.T
    results["ffn_full_cpu_fp32"] = bench_cpu("FFN full (FP32)", ffn_full_cpu)

    # === Projections FP16 ===
    print("\n--- Attention Projections (FP16) ---")
    def q_proj_cpu_f16():
        return x_f16 @ W_q_f16.T
    results["q_proj_cpu_fp16"] = bench_cpu("Q proj 8192x8192 (FP16)", q_proj_cpu_f16)

    # === Attention ===
    for seq_len in SEQ_LENS:
        print(f"\n--- Attention (seq_len={seq_len}, FP32) ---")
        Q_np = np.random.randn(1, N_HEADS, 1, HEAD_DIM).astype(np.float32)
        K_np = np.random.randn(1, N_HEADS, seq_len, HEAD_DIM).astype(np.float32)  # already expanded
        V_np = np.random.randn(1, N_HEADS, seq_len, HEAD_DIM).astype(np.float32)

        def attn_cpu(Q=Q_np, K=K_np, V=V_np, sl=seq_len):
            scores = (Q @ K.transpose(0, 1, 3, 2)) / (HEAD_DIM ** 0.5)
            # simple softmax
            exp_s = np.exp(scores - scores.max(axis=-1, keepdims=True))
            weights = exp_s / exp_s.sum(axis=-1, keepdims=True)
            out = weights @ V
            return out.reshape(1, -1)

        results[f"attn_full_cpu_seq{seq_len}"] = bench_cpu(
            f"Attention full (seq={seq_len})", attn_cpu)

    # === Activations ===
    print("\n--- Activations ---")
    act_np = np.random.randn(1, FFN_DIM).astype(np.float32)
    def silu_cpu():
        return (1.0 / (1.0 + np.exp(-act_np))) * act_np
    results["silu_cpu_fp32"] = bench_cpu("SiLU (28672, FP32)", silu_cpu)

    # === Residual ===
    print("\n--- Residual ---")
    r1 = np.random.randn(1, DIM).astype(np.float32)
    r2 = np.random.randn(1, DIM).astype(np.float32)
    def residual_cpu():
        return r1 + r2
    results["residual_add_cpu"] = bench_cpu("Residual add (8192)", residual_cpu)

    return results


# ═══════════════════════════════════════════════════════════════════════
#  REPORT GENERATION
# ═══════════════════════════════════════════════════════════════════════

def generate_report(gpu_results, cpu_results):
    print("\n" + "=" * 70)
    print("  PROJECT CHIMERA — OPERATION TIMING REPORT")
    print("  Llama 3.3 70B, M5 Pro 64GB, batch=1 decode")
    print("=" * 70)

    # Header
    print(f"\n{'Operation':<45} {'GPU Q4':>10} {'GPU FP16':>10} {'CPU FP32':>10} {'Fastest':>10}")
    print("-" * 90)

    rows = [
        ("RMSNorm",
         gpu_results.get("rmsnorm_gpu_fp16", {}),
         gpu_results.get("rmsnorm_gpu_fp16", {}),
         cpu_results.get("rmsnorm_cpu_fp32", {})),
        ("Q proj (8192x8192)",
         gpu_results.get("q_proj_gpu_q4", {}),
         gpu_results.get("q_proj_gpu_fp16", {}),
         cpu_results.get("q_proj_cpu_fp32", {})),
        ("K proj (1024x8192)",
         gpu_results.get("k_proj_gpu_q4", {}),
         gpu_results.get("k_proj_gpu_fp16", {}),
         cpu_results.get("k_proj_cpu_fp32", {})),
        ("V proj (1024x8192)",
         gpu_results.get("v_proj_gpu_q4", {}),
         gpu_results.get("v_proj_gpu_fp16", {}),
         cpu_results.get("k_proj_cpu_fp32", {})),
        ("O proj (8192x8192)",
         gpu_results.get("o_proj_gpu_q4", {}),
         gpu_results.get("o_proj_gpu_fp16", {}),
         cpu_results.get("o_proj_cpu_fp32", {})),
        ("FFN gate (28672x8192)",
         gpu_results.get("ffn_gate_gpu_q4", {}),
         gpu_results.get("ffn_gate_gpu_fp16", {}),
         cpu_results.get("ffn_gate_cpu_fp32", {})),
        ("FFN up (28672x8192)",
         gpu_results.get("ffn_up_gpu_q4", {}),
         gpu_results.get("ffn_up_gpu_fp16", {}),
         cpu_results.get("ffn_up_cpu_fp32", {})),
        ("FFN down (8192x28672)",
         gpu_results.get("ffn_down_gpu_q4", {}),
         gpu_results.get("ffn_down_gpu_fp16", {}),
         cpu_results.get("ffn_down_cpu_fp32", {})),
        ("FFN full",
         gpu_results.get("ffn_gpu_q4", {}),
         gpu_results.get("ffn_gpu_fp16", {}),
         cpu_results.get("ffn_full_cpu_fp32", {})),
        ("SiLU (28672)",
         gpu_results.get("silu_gpu", {}),
         gpu_results.get("silu_gpu", {}),
         cpu_results.get("silu_cpu_fp32", {})),
        ("SiLU*up (28672)",
         gpu_results.get("silu_mul_gpu", {}),
         gpu_results.get("silu_mul_gpu", {}),
         {},),
        ("Residual add",
         gpu_results.get("residual_add_gpu", {}),
         gpu_results.get("residual_add_gpu", {}),
         cpu_results.get("residual_add_cpu", {})),
    ]

    for label, q4, fp16, cpu in rows:
        q4_ms = q4.get("median_ms", float('inf'))
        fp16_ms = fp16.get("median_ms", float('inf'))
        cpu_ms = cpu.get("median_ms", float('inf'))
        best = min(q4_ms, fp16_ms, cpu_ms)
        if best == float('inf'):
            fastest = "N/A"
        elif best == q4_ms:
            fastest = "GPU Q4"
        elif best == fp16_ms:
            fastest = "GPU FP16"
        else:
            fastest = "CPU"
        q4_s = f"{q4_ms:.3f}" if q4_ms < 1e6 else "N/A"
        fp16_s = f"{fp16_ms:.3f}" if fp16_ms < 1e6 else "N/A"
        cpu_s = f"{cpu_ms:.3f}" if cpu_ms < 1e6 else "N/A"
        mark = " ***" if fastest != "N/A" else ""
        print(f"  {label:<43} {q4_s:>10} {fp16_s:>10} {cpu_s:>10} {fastest:>10}")

    # Attention rows by sequence length
    print()
    print(f"{'Attention by seq_len':<45} {'GPU FP16':>10} {'CPU FP32':>10} {'Fastest':>10}")
    print("-" * 80)
    for seq_len in SEQ_LENS:
        for sub_label, gpu_key, cpu_key in [
            (f"Full attention (seq={seq_len})",
             f"attn_full_gpu_seq{seq_len}", f"attn_full_cpu_seq{seq_len}"),
            (f"  QK scores (seq={seq_len})",
             f"attn_scores_gpu_seq{seq_len}", None),
            (f"  Softmax (seq={seq_len})",
             f"attn_softmax_gpu_seq{seq_len}", None),
            (f"  Weighted V (seq={seq_len})",
             f"attn_wv_gpu_seq{seq_len}", None),
            (f"  GQA repeat (seq={seq_len})",
             f"gqa_repeat_gpu_seq{seq_len}", None),
        ]:
            gpu_ms = gpu_results.get(gpu_key, {}).get("median_ms", float('inf'))
            cpu_ms = cpu_results.get(cpu_key, {}).get("median_ms", float('inf')) if cpu_key else float('inf')
            gpu_s = f"{gpu_ms:.3f}" if gpu_ms < 1e6 else "N/A"
            cpu_s = f"{cpu_ms:.3f}" if cpu_ms < 1e6 else "N/A"
            if gpu_ms <= cpu_ms:
                fastest = "GPU"
            elif cpu_ms < float('inf'):
                fastest = "CPU"
            else:
                fastest = "GPU"
            print(f"  {sub_label:<43} {gpu_s:>10} {cpu_s:>10} {fastest:>10}")
        print()

    # === Per-layer total estimate ===
    print("\n" + "=" * 70)
    print("  PER-LAYER TOTAL ESTIMATES (batch=1 decode, median)")
    print("=" * 70)

    # GPU Q4 full layer
    layer_q4 = (
        gpu_results.get("rmsnorm_gpu_fp16", {}).get("median_ms", 0) * 2 +  # pre-attn + pre-ffn
        gpu_results.get("q_proj_gpu_q4", {}).get("median_ms", 0) +
        gpu_results.get("k_proj_gpu_q4", {}).get("median_ms", 0) +
        gpu_results.get("v_proj_gpu_q4", {}).get("median_ms", 0) +
        gpu_results.get("o_proj_gpu_q4", {}).get("median_ms", 0) +
        gpu_results.get("attn_full_gpu_seq1024", {}).get("median_ms", 0) +
        gpu_results.get("ffn_gpu_q4", {}).get("median_ms", 0) +
        gpu_results.get("residual_add_gpu", {}).get("median_ms", 0) * 2
    )
    print(f"\n  GPU Q4 full layer (seq=1024):  {layer_q4:.3f} ms")
    print(f"  -> 80 layers total:           {layer_q4 * 80:.1f} ms")
    print(f"  -> tok/s estimate:            {1000 / (layer_q4 * 80):.1f}")

    # CPU full layer
    layer_cpu = (
        cpu_results.get("rmsnorm_cpu_fp32", {}).get("median_ms", 0) * 2 +
        cpu_results.get("q_proj_cpu_fp32", {}).get("median_ms", 0) +
        cpu_results.get("k_proj_cpu_fp32", {}).get("median_ms", 0) * 2 +  # K+V ~ same
        cpu_results.get("o_proj_cpu_fp32", {}).get("median_ms", 0) +
        cpu_results.get("attn_full_cpu_seq1024", {}).get("median_ms", 0) +
        cpu_results.get("ffn_full_cpu_fp32", {}).get("median_ms", 0) +
        cpu_results.get("residual_add_cpu", {}).get("median_ms", 0) * 2
    )
    print(f"\n  CPU FP32 full layer (seq=1024): {layer_cpu:.3f} ms")
    print(f"  -> 80 layers total:            {layer_cpu * 80:.1f} ms")
    print(f"  -> tok/s estimate:             {1000 / (layer_cpu * 80):.1f}" if layer_cpu > 0 else "  -> N/A")

    # === Bandwidth analysis ===
    print("\n" + "=" * 70)
    print("  BANDWIDTH ANALYSIS")
    print("=" * 70)

    # Q4 weight bytes per layer
    attn_weights_q4 = (DIM * DIM + 2 * N_KV_HEADS * HEAD_DIM * DIM + DIM * DIM) * 0.5  # bytes
    ffn_weights_q4 = (2 * FFN_DIM * DIM + DIM * FFN_DIM) * 0.5
    total_q4_per_layer = attn_weights_q4 + ffn_weights_q4
    total_q4_80 = total_q4_per_layer * 80

    print(f"\n  Q4 weight bytes per layer:  {total_q4_per_layer / 1e6:.1f} MB")
    print(f"  Q4 weight bytes 80 layers:  {total_q4_80 / 1e9:.2f} GB")
    print(f"  DRAM bandwidth:             307 GB/s")
    print(f"  Theoretical min time (BW):  {total_q4_80 / 307e9 * 1000:.1f} ms")
    print(f"  -> Theoretical max tok/s:   {1.0 / (total_q4_80 / 307e9):.1f}")

    return {"gpu": gpu_results, "cpu": cpu_results}


# ═══════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Project Chimera: Llama 70B Per-Operation Profiler")
    print(f"Hardware: Apple M5 Pro, 64GB, 307 GB/s DRAM")
    print(f"MLX version: {mx.__version__}")
    print(f"Numpy version: {np.__version__} (Accelerate/AMX)")
    print(f"Iterations: {N_ITERS} (warmup: {N_WARMUP})")

    gpu_results = profile_gpu()
    cpu_results = profile_cpu()

    all_results = generate_report(gpu_results, cpu_results)

    # Save raw JSON
    out_path = "/Users/midas/Desktop/cowork/ane-compiler/chimera_profile_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nRaw results saved to: {out_path}")
