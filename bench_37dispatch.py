#!/usr/bin/env python3
"""
Benchmark: GPT-2 37-dispatch generation with correctness verification.

Measures per-dispatch latency, CPU attention time, total tok/s,
and effective bandwidth. Compares to PyTorch reference.

Copyright 2026 Nick Lo. MIT License.
"""

import os
import sys
import time
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from model_loader import GPT2Model
from first_token import compile_all_ops, MODEL_PATH, layernorm_cpu
from generate import (ANEDispatcher, forward_layer_decode, embed, lm_head,
                      generate, softmax_cpu)
from kv_cache import KVCache


def main():
    print("=" * 70)
    print("GPT-2 117M: 37-DISPATCH BENCHMARK")
    print("=" * 70)

    from transformers import GPT2Tokenizer

    # Load
    model = GPT2Model.from_safetensors(MODEL_PATH)
    tokenizer = GPT2Tokenizer.from_pretrained('openai-community/gpt2')
    config = model.config
    print(f"Model loaded: {config.n_layer}L, {config.n_embd}d, {config.n_head}h")

    # Compile
    build_dir = '/tmp/gpt2_first_token_fused'
    compiled = compile_all_ops(model, build_dir, mode='fused')
    n_ops = len(compiled)
    print(f"Compiled: {n_ops} ops")

    # Launch
    disp = ANEDispatcher(compiled, quiet=True)
    disp.start()
    print("Dispatcher ready")

    # === CORRECTNESS ===
    print(f"\n{'='*70}")
    print("CORRECTNESS: 10-token match vs PyTorch")
    print("=" * 70)

    import torch
    from transformers import GPT2LMHeadModel
    pt_model = GPT2LMHeadModel.from_pretrained('openai-community/gpt2')
    pt_model.eval()

    prompt = "The capital of France"
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    with torch.no_grad():
        output = pt_model.generate(input_ids, max_new_tokens=10, do_sample=False)
    pt_tokens = output[0].tolist()

    prompt_tokens = tokenizer.encode(prompt)
    ane_tokens = generate(model, disp, prompt_tokens, max_new_tokens=10,
                           mode='fused')

    all_match = True
    for i in range(len(pt_tokens)):
        if i < len(ane_tokens):
            match = ane_tokens[i] == pt_tokens[i]
            if not match: all_match = False
            tok = tokenizer.decode([pt_tokens[i]])
            print(f"  pos {i:2d}: PT={pt_tokens[i]:6d} ANE={ane_tokens[i]:6d} "
                  f"{'OK' if match else 'MISS':4s} \"{tok}\"")

    print(f"\n{'PASS' if all_match else 'FAIL'}: "
          f"{sum(1 for i in range(min(len(ane_tokens),len(pt_tokens))) if ane_tokens[i]==pt_tokens[i])}"
          f"/{len(pt_tokens)} match")
    print(f"ANE: \"{tokenizer.decode(ane_tokens)}\"")
    print(f"PT:  \"{tokenizer.decode(pt_tokens)}\"")

    disp.stop()

    if not all_match:
        print("Stopping -- correctness failed")
        return

    # === BENCHMARK ===
    print(f"\n{'='*70}")
    print("BENCHMARK: throughput measurement")
    print("=" * 70)

    disp2 = ANEDispatcher(compiled, quiet=True)
    disp2.start()

    kv = KVCache(config.n_layer, config.n_head, config.head_dim)

    # Warmup
    for w in range(10):
        x = embed(model, 0, w)
        for li in range(config.n_layer):
            x = forward_layer_decode(li, x, model, disp2, kv, mode='fused')
        lm_head(x, model, disp2)

    # Measure per-token latency
    n_measure = 100
    times_total = []
    times_ane = []
    times_cpu = []

    for step in range(n_measure):
        pos = 10 + step
        t_total_start = time.perf_counter()

        x = embed(model, 0, pos)
        t_ane_total = 0.0
        t_cpu_total = 0.0

        for li in range(config.n_layer):
            L = model.layers[li]
            pfx = f'L{li}'

            # LN1 (CPU)
            t_cpu = time.perf_counter()
            ln1_out = layernorm_cpu(x, L.ln_1_weight, L.ln_1_bias,
                                    config.layer_norm_epsilon)
            t_cpu_total += time.perf_counter() - t_cpu

            # QKV (ANE)
            t_ane = time.perf_counter()
            qkv = disp2.dispatch(f'{pfx}_qkv_proj', ln1_out)
            t_ane_total += time.perf_counter() - t_ane

            q = qkv[:config.n_embd]
            k = qkv[config.n_embd:2*config.n_embd]
            v = qkv[2*config.n_embd:]
            q_heads = q.reshape(config.n_head, config.head_dim)
            k_heads = k.reshape(config.n_head, config.head_dim)
            v_heads = v.reshape(config.n_head, config.head_dim)
            kv.append(li, k_heads[np.newaxis], v_heads[np.newaxis])

            # Attention (CPU)
            t_cpu = time.perf_counter()
            cached_k, cached_v = kv.get(li)
            scale = np.float32(1.0 / np.sqrt(config.head_dim))
            attn_output = np.zeros(config.n_embd, dtype=np.float32)
            for h in range(config.n_head):
                q_h = q_heads[h].astype(np.float32)
                k_h = cached_k[:, h, :].astype(np.float32)
                v_h = cached_v[:, h, :].astype(np.float32)
                scores = (q_h @ k_h.T) * scale
                weights = softmax_cpu(scores)
                attn_output[h*config.head_dim:(h+1)*config.head_dim] = weights @ v_h
            attn_output = attn_output.astype(np.float16)
            t_cpu_total += time.perf_counter() - t_cpu

            # O_proj (ANE)
            t_ane = time.perf_counter()
            o_out = disp2.dispatch(f'{pfx}_o_proj', attn_output)
            t_ane_total += time.perf_counter() - t_ane

            # Residual + LN2 (CPU)
            t_cpu = time.perf_counter()
            r1 = (x.astype(np.float32) + o_out.astype(np.float32)).astype(np.float16)
            ln2_out = layernorm_cpu(r1, L.ln_2_weight, L.ln_2_bias,
                                    config.layer_norm_epsilon)
            t_cpu_total += time.perf_counter() - t_cpu

            # Fused FFN (ANE)
            t_ane = time.perf_counter()
            ffn_out = disp2.dispatch(f'{pfx}_fused_ffn', ln2_out)
            t_ane_total += time.perf_counter() - t_ane

            # Residual (CPU)
            t_cpu = time.perf_counter()
            x = (r1.astype(np.float32) + ffn_out.astype(np.float32)).astype(np.float16)
            t_cpu_total += time.perf_counter() - t_cpu

        # Final LN + lm_head
        t_cpu = time.perf_counter()
        x_ln = layernorm_cpu(x, model.ln_f_weight, model.ln_f_bias,
                              config.layer_norm_epsilon)
        t_cpu_total += time.perf_counter() - t_cpu

        t_ane = time.perf_counter()
        logits = disp2.dispatch('lm_head', x_ln)
        t_ane_total += time.perf_counter() - t_ane

        t_total = (time.perf_counter() - t_total_start) * 1000
        times_total.append(t_total)
        times_ane.append(t_ane_total * 1000)
        times_cpu.append(t_cpu_total * 1000)

    disp2.stop()

    # Stats
    times_total.sort()
    times_ane.sort()
    times_cpu.sort()

    med_total = times_total[n_measure // 2]
    med_ane = times_ane[n_measure // 2]
    med_cpu = times_cpu[n_measure // 2]
    tps = 1000.0 / med_total

    # Weight sizes
    per_layer = (768*2304 + 768*768 + 768*3072 + 3072*768) * 2  # FP16
    total_weight = per_layer * 12 + 768 * 50257 * 2
    eff_bw = (total_weight / 1e9) / (med_total / 1000)
    ane_bw = (total_weight / 1e9) / (med_ane / 1000)

    print(f"\nMeasured over {n_measure} tokens (seq_len 10-{10+n_measure}):")
    print(f"\n{'Metric':<30} {'Value':>10}")
    print("-" * 42)
    print(f"{'Total per token':<30} {med_total:>8.2f} ms")
    print(f"{'ANE dispatches per token':<30} {med_ane:>8.2f} ms")
    print(f"{'CPU ops per token':<30} {med_cpu:>8.2f} ms")
    print(f"{'Overhead (total-ANE-CPU)':<30} {med_total-med_ane-med_cpu:>8.2f} ms")
    print(f"{'Throughput':<30} {tps:>8.1f} tok/s")
    print(f"{'ANE dispatches':<30} {37:>8}")
    print(f"{'Model weights (FP16)':<30} {total_weight/1e6:>7.1f} MB")
    print(f"{'Effective bandwidth (total)':<30} {eff_bw:>7.1f} GB/s")
    print(f"{'ANE bandwidth (ANE only)':<30} {ane_bw:>7.1f} GB/s")

    print(f"\nPercentile breakdown (ms):")
    print(f"{'':>15} {'p5':>8} {'p25':>8} {'p50':>8} {'p75':>8} {'p95':>8}")
    for name, times in [('Total', times_total), ('ANE', times_ane), ('CPU', times_cpu)]:
        p5 = times[int(n_measure * 0.05)]
        p25 = times[int(n_measure * 0.25)]
        p50 = times[n_measure // 2]
        p75 = times[int(n_measure * 0.75)]
        p95 = times[int(n_measure * 0.95)]
        print(f"{name:>15} {p5:>8.2f} {p25:>8.2f} {p50:>8.2f} {p75:>8.2f} {p95:>8.2f}")

    print(f"\nCPU breakdown per token:")
    print(f"  LayerNorm (25x): ~{med_cpu * 0.4:.2f} ms (estimate)")
    print(f"  Attention (12x): ~{med_cpu * 0.5:.2f} ms (estimate)")
    print(f"  Residuals (24x): ~{med_cpu * 0.1:.2f} ms (estimate)")

    print(f"\n{'='*70}")
    print("COMPARISON TO PRIOR MEASUREMENTS")
    print("-" * 70)
    print(f"  37-dispatch:  {med_total:.2f} ms/tok, {tps:.1f} tok/s (this run)")
    print(f"  37-dispatch:  ~7.2 ms/tok, ~139 tok/s (prior session)")
    print(f"  1-dispatch:   1.59 ms/tok (bench_fusion_depth, V->O shortcut)")
    print(f"  1-dispatch BW: 137.9 GB/s (single dispatch, no attention)")
    print("=" * 70)


if __name__ == '__main__':
    main()
