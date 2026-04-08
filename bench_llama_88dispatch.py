#!/usr/bin/env python3
"""
Benchmark: Llama-1B 88-dispatch generation with correctness verification.

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
from llama_loader import (LlamaModel, compile_llama_unfused,
                          forward_layer_decode_unfused, rms_norm_cpu,
                          lm_head_dispatch)
from generate import ANEDispatcher
from kv_cache import KVCache


LLAMA_PATH = os.path.expanduser(
    "~/.cache/huggingface/hub/models--unsloth--Llama-3.2-1B-Instruct/"
    "snapshots/5a8abab4a5d6f164389b1079fb721cfab8d7126c/"
)


def main():
    print("=" * 70)
    print("LLAMA-3.2-1B: 88-DISPATCH BENCHMARK")
    print("=" * 70)

    if not os.path.exists(LLAMA_PATH):
        print(f"Model not found at {LLAMA_PATH}")
        return

    # Load
    t0 = time.time()
    model = LlamaModel.from_safetensors(LLAMA_PATH)
    config = model.config
    print(f"Loaded in {time.time()-t0:.1f}s: {config.n_layers}L, "
          f"{config.hidden_size}d, {config.n_heads}Q/{config.n_kv_heads}KV")

    # Compile
    build_dir = '/tmp/llama_1b_88dispatch'
    t0 = time.time()
    compiled = compile_llama_unfused(model, build_dir)
    print(f"Compiled in {time.time()-t0:.1f}s")

    # Launch
    dispatch_dict = {k: v for k, v in compiled.items() if not k.startswith('_')}
    disp = ANEDispatcher(dispatch_dict, quiet=True)
    disp.start()
    print("Dispatcher ready")

    # === CORRECTNESS ===
    print(f"\n{'='*70}")
    print("CORRECTNESS: PyTorch reference comparison")
    print("=" * 70)

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("unsloth/Llama-3.2-1B-Instruct")
        pt_model = AutoModelForCausalLM.from_pretrained("unsloth/Llama-3.2-1B-Instruct",
                                                         torch_dtype=torch.float32)
        pt_model.eval()

        prompt = "The capital of France"
        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        with torch.no_grad():
            output = pt_model.generate(input_ids, max_new_tokens=10, do_sample=False)
        pt_tokens = output[0].tolist()
        print(f"PyTorch: {tokenizer.decode(pt_tokens)}")

        # ANE generation
        prompt_tokens = tokenizer.encode(prompt)
        kv = KVCache(config.n_layers, config.n_kv_heads, config.head_dim)
        generated = list(prompt_tokens)

        for pos, tid in enumerate(prompt_tokens[:-1]):
            x = model.embed_tokens[tid].astype(np.float16)
            for li in range(config.n_layers):
                x = forward_layer_decode_unfused(li, x, model, disp, kv, pos)

        x = model.embed_tokens[prompt_tokens[-1]].astype(np.float16)
        for li in range(config.n_layers):
            x = forward_layer_decode_unfused(li, x, model, disp, kv,
                                              len(prompt_tokens) - 1)
        x_norm = rms_norm_cpu(x, model.norm_weight, config.rms_norm_eps)
        logits = lm_head_dispatch(x_norm, compiled, disp)
        next_token = int(np.argmax(logits.astype(np.float32)))
        generated.append(next_token)

        for step in range(9):
            pos = len(generated) - 1
            x = model.embed_tokens[next_token].astype(np.float16)
            for li in range(config.n_layers):
                x = forward_layer_decode_unfused(li, x, model, disp, kv, pos)
            x_norm = rms_norm_cpu(x, model.norm_weight, config.rms_norm_eps)
            logits = lm_head_dispatch(x_norm, compiled, disp)
            next_token = int(np.argmax(logits.astype(np.float32)))
            generated.append(next_token)

        ane_text = tokenizer.decode(generated)
        pt_text = tokenizer.decode(pt_tokens)
        print(f"ANE:     {ane_text}")

        matches = sum(1 for i in range(min(len(generated), len(pt_tokens)))
                      if generated[i] == pt_tokens[i])
        print(f"Match: {matches}/{len(pt_tokens)}")

    except Exception as e:
        print(f"PyTorch comparison failed: {e}")
        import traceback; traceback.print_exc()

    # === BENCHMARK ===
    print(f"\n{'='*70}")
    print("BENCHMARK: throughput measurement")
    print("=" * 70)

    kv2 = KVCache(config.n_layers, config.n_kv_heads, config.head_dim)

    # Warmup
    for w in range(5):
        x = model.embed_tokens[0].astype(np.float16)
        for li in range(config.n_layers):
            x = forward_layer_decode_unfused(li, x, model, disp, kv2, w)
        x_norm = rms_norm_cpu(x, model.norm_weight, config.rms_norm_eps)
        lm_head_dispatch(x_norm, compiled, disp)

    # Measure
    n_measure = 50
    times = []
    for step in range(n_measure):
        pos = 5 + step
        t0 = time.perf_counter()
        x = model.embed_tokens[0].astype(np.float16)
        for li in range(config.n_layers):
            x = forward_layer_decode_unfused(li, x, model, disp, kv2, pos)
        x_norm = rms_norm_cpu(x, model.norm_weight, config.rms_norm_eps)
        lm_head_dispatch(x_norm, compiled, disp)
        times.append((time.perf_counter() - t0) * 1000)

    disp.stop()

    times.sort()
    med = times[n_measure // 2]
    tps = 1000.0 / med

    # Weight sizes (FP16)
    per_layer = (2048*3072 + 2048*2048 + 2048*8192 + 2048*8192 + 8192*2048) * 2
    lm_head_w = 2048 * 128256 * 2
    total_weight = per_layer * 16 + lm_head_w
    eff_bw = (total_weight / 1e9) / (med / 1000)

    print(f"\n{'Metric':<30} {'Value':>10}")
    print("-" * 42)
    print(f"{'Total per token':<30} {med:>8.2f} ms")
    print(f"{'Throughput':<30} {tps:>8.1f} tok/s")
    print(f"{'ANE dispatches':<30} {88:>8}")
    print(f"{'Model weights (FP16)':<30} {total_weight/1e6:>7.1f} MB")
    print(f"{'Effective bandwidth':<30} {eff_bw:>7.1f} GB/s")

    print(f"\nPercentile (ms):")
    p5 = times[int(n_measure * 0.05)]
    p50 = times[n_measure // 2]
    p95 = times[int(n_measure * 0.95)]
    print(f"  p5={p5:.2f}  p50={p50:.2f}  p95={p95:.2f}")

    print(f"\n{'='*70}")
    print(f"Llama-1B: {tps:.1f} tok/s at 88 dispatches")
    print(f"Prior measurement: 28.3 tok/s (from session log)")
    print("=" * 70)


if __name__ == '__main__':
    main()
