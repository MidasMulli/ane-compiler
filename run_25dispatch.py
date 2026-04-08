#!/usr/bin/env python3
"""
25-dispatch GPT-2 generation via MIL IR 2-input fusion.

Architecture per layer (2 dispatches, was 3):
  1. ANE: QKV_proj (768→2304, bias) — same as 37-dispatch
  2. CPU: attention (split QKV, KV cache, Q@K^T, softmax, @V, concat)
  3. ANE: fused_post_attn(attn_out, x) — 2-input MIL IR:
     O_proj + residual_1 + LN2 + FFN_up + GELU + FFN_down + residual_2

Total per token: 12 layers × 2 dispatches + 1 lm_head = 25 dispatches
vs 37-dispatch: 12 × 3 + 1 = 37 dispatches

Uses coremltools ct.convert for MIL IR compilation.
Uses _ANEClient for direct ANE dispatch (same as ane_generate.m).

Copyright 2026 Nick Lo. MIT License.
"""

import argparse
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from model_loader import GPT2Model
from first_token import layernorm_cpu, gelu_new_cpu, compile_all_ops, MODEL_PATH
from kv_cache import KVCache
from generate import ANEDispatcher, softmax_cpu, embed, lm_head

BUILD_DIR_37 = '/tmp/gpt2_first_token'
BUILD_DIR_25 = '/tmp/gpt2_mil_fused'


def build_fused_models(model):
    """Build 12 fused post-attention MIL IR models via ct.convert."""
    import coremltools as ct
    from coremltools.converters.mil import Builder as mb
    from coremltools.converters.mil.mil import types

    os.makedirs(BUILD_DIR_25, exist_ok=True)
    fused_models = {}

    for i in range(model.config.n_layer):
        mlpackage_path = os.path.join(BUILD_DIR_25, f'layer_{i}_post_attn.mlpackage')

        if os.path.exists(mlpackage_path):
            # Load pre-built model
            fused_models[i] = ct.models.MLModel(
                mlpackage_path, compute_units=ct.ComputeUnit.CPU_AND_NE)
            continue

        layer = model.layers[i]

        W_o = layer.W_o.astype(np.float16)
        b_o = layer.c_proj_bias.astype(np.float16)
        ln2_w = layer.ln_2_weight.astype(np.float16)
        ln2_b = layer.ln_2_bias.astype(np.float16)
        W_up = layer.W_fc.astype(np.float16)
        b_up = layer.c_fc_bias.astype(np.float16)
        W_down = layer.W_fc_down.astype(np.float16)
        b_down = layer.c_proj_ffn_bias.astype(np.float16)
        epsilon = np.float16(1e-5)

        @mb.program(input_specs=[
            mb.TensorSpec(shape=(1, 768, 1, 1), dtype=types.fp16),  # attn_out
            mb.TensorSpec(shape=(1, 768, 1, 1), dtype=types.fp16),  # x (skip)
        ])
        def fused_post_attn(attn_out, x):
            attn_flat = mb.reshape(x=attn_out, shape=[1, 768])
            x_flat = mb.reshape(x=x, shape=[1, 768])
            o_out = mb.linear(x=attn_flat, weight=mb.const(val=W_o), bias=mb.const(val=b_o))
            r1 = mb.add(x=o_out, y=x_flat)
            ln2_out = mb.layer_norm(x=r1, axes=[1], gamma=mb.const(val=ln2_w),
                                     beta=mb.const(val=ln2_b), epsilon=epsilon)
            ffn_up = mb.linear(x=ln2_out, weight=mb.const(val=W_up), bias=mb.const(val=b_up))
            gelu_out = mb.gelu(x=ffn_up, mode="TANH_APPROXIMATION")
            ffn_down = mb.linear(x=gelu_out, weight=mb.const(val=W_down), bias=mb.const(val=b_down))
            output = mb.add(x=ffn_down, y=r1)
            output_4d = mb.reshape(x=output, shape=[1, 768, 1, 1])
            return output_4d

        ct_model = ct.convert(fused_post_attn, compute_units=ct.ComputeUnit.CPU_AND_NE,
                               minimum_deployment_target=ct.target.iOS18)
        ct_model.save(mlpackage_path)
        fused_models[i] = ct_model

    return fused_models


def forward_layer_25d(layer_idx, x_fp16, model, dispatcher, kv_cache,
                      fused_models):
    """Forward pass through one GPT-2 layer with fused post-attention.

    Dispatches:
      1. QKV projection (ANE, via pipe dispatcher)
      2. Attention (CPU)
      3. Fused post-attention (ANE, via ct.predict — 2-input)
    """
    L = model.layers[layer_idx]
    dim = model.config.n_embd
    n_heads = model.config.n_head
    head_dim = model.config.head_dim
    pfx = f'L{layer_idx}'

    # 1. LayerNorm 1 (CPU)
    ln1_out = layernorm_cpu(x_fp16, L.ln_1_weight, L.ln_1_bias,
                            model.config.layer_norm_epsilon)

    # 2. QKV projection (ANE, single dispatch)
    qkv = dispatcher.dispatch(f'{pfx}_qkv_proj', ln1_out)
    q = qkv[:dim]
    k = qkv[dim:2*dim]
    v = qkv[2*dim:]

    # 3. Multi-head attention (CPU)
    q_heads = q.reshape(n_heads, head_dim)
    k_heads = k.reshape(n_heads, head_dim)
    v_heads = v.reshape(n_heads, head_dim)
    kv_cache.append(layer_idx, k_heads[np.newaxis], v_heads[np.newaxis])

    cached_k, cached_v = kv_cache.get(layer_idx)
    scale = np.float32(1.0 / np.sqrt(head_dim))
    attn_output = np.zeros(dim, dtype=np.float32)
    for h in range(n_heads):
        q_h = q_heads[h].astype(np.float32)
        k_h = cached_k[:, h, :].astype(np.float32)
        v_h = cached_v[:, h, :].astype(np.float32)
        scores = (q_h @ k_h.T) * scale
        weights = softmax_cpu(scores)
        attn_output[h * head_dim:(h + 1) * head_dim] = weights @ v_h
    attn_out = attn_output.astype(np.float16)

    # 4. Fused post-attention (ANE, 2-input MIL IR model via ct.predict)
    # Inputs: attn_out (attention output), x_fp16 (skip connection)
    fused = fused_models[layer_idx]
    result = fused.predict({
        'attn_out': attn_out.reshape(1, 768, 1, 1).astype(np.float32),
        'x': x_fp16.reshape(1, 768, 1, 1).astype(np.float32),
    })
    out_key = list(result.keys())[0]
    output = result[out_key].flatten().astype(np.float16)

    return output


def forward_layer_37d(layer_idx, x_fp16, model, dispatcher, kv_cache):
    """Forward pass — 37-dispatch baseline (from generate.py)."""
    from generate import forward_layer_decode
    return forward_layer_decode(layer_idx, x_fp16, model, dispatcher, kv_cache,
                                mode='fused')


def generate_tokens(model, dispatcher, fused_models, prompt_tokens,
                    max_new_tokens=10, mode='25d'):
    """Generate tokens with either 25 or 37 dispatch paths."""
    config = model.config
    kv_cache = KVCache(config.n_layer, config.n_head, config.head_dim)
    generated = list(prompt_tokens)

    forward_fn = forward_layer_25d if mode == '25d' else forward_layer_37d

    # Prefill
    for pos, token_id in enumerate(prompt_tokens[:-1]):
        x = embed(model, token_id, pos)
        for li in range(config.n_layer):
            if mode == '25d':
                x = forward_fn(li, x, model, dispatcher, kv_cache, fused_models)
            else:
                x = forward_fn(li, x, model, dispatcher, kv_cache)

    # Last prompt token -> first generated token
    last_pos = len(prompt_tokens) - 1
    x = embed(model, prompt_tokens[-1], last_pos)
    for li in range(config.n_layer):
        if mode == '25d':
            x = forward_fn(li, x, model, dispatcher, kv_cache, fused_models)
        else:
            x = forward_fn(li, x, model, dispatcher, kv_cache)
    logits = lm_head(x, model, dispatcher)
    next_token = int(np.argmax(logits.astype(np.float32)))
    generated.append(next_token)

    # Generation loop
    t_start = time.time()
    for step in range(max_new_tokens - 1):
        pos = len(generated) - 1
        x = embed(model, next_token, pos)
        for li in range(config.n_layer):
            if mode == '25d':
                x = forward_fn(li, x, model, dispatcher, kv_cache, fused_models)
            else:
                x = forward_fn(li, x, model, dispatcher, kv_cache)
        logits = lm_head(x, model, dispatcher)
        next_token = int(np.argmax(logits.astype(np.float32)))
        generated.append(next_token)

    gen_time = time.time() - t_start
    return generated, gen_time


def main():
    parser = argparse.ArgumentParser(description='25-dispatch GPT-2 generation')
    parser.add_argument('--prompt', default='The capital of France is',
                        help='Input prompt')
    parser.add_argument('--tokens', type=int, default=10,
                        help='Number of tokens to generate')
    parser.add_argument('--compare', action='store_true',
                        help='Also run 37-dispatch for comparison')
    args = parser.parse_args()

    print("=" * 70)
    print("25-DISPATCH GPT-2 GENERATION")
    print("MIL IR 2-input fusion: O_proj + res + LN2 + FFN = 1 dispatch")
    print("=" * 70)

    # Load model
    print("\n[1/5] Loading GPT-2...")
    t0 = time.time()
    model = GPT2Model.from_safetensors(MODEL_PATH)
    print(f"  Loaded in {time.time()-t0:.2f}s")

    # Build fused models
    print("\n[2/5] Building fused MIL IR models (12 layers)...")
    t0 = time.time()
    fused_models = build_fused_models(model)
    print(f"  Built in {time.time()-t0:.2f}s")

    # Compile 37-dispatch ops (for QKV + lm_head)
    print("\n[3/5] Compiling QKV + lm_head (37-dispatch path)...")
    t0 = time.time()
    compiled_37 = compile_all_ops(model, BUILD_DIR_37, mode='fused')
    print(f"  Compiled in {time.time()-t0:.2f}s")

    # Start dispatcher (for QKV + lm_head only in 25d mode)
    print("\n[4/5] Starting ANE dispatcher...")
    dispatcher = ANEDispatcher(compiled_37, quiet=True)
    dispatcher.start()

    # Tokenize
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained('gpt2')
        prompt_tokens = tokenizer.encode(args.prompt)
    except ImportError:
        prompt_tokens = [464]
        tokenizer = None

    print(f"  Prompt: '{args.prompt}'")
    print(f"  Tokens: {prompt_tokens}")

    # PyTorch reference
    print("\n[5/5] PyTorch reference...")
    try:
        import torch
        from transformers import GPT2LMHeadModel
        pt_model = GPT2LMHeadModel.from_pretrained('openai-community/gpt2')
        pt_model.eval()
        input_ids = tokenizer.encode(args.prompt, return_tensors="pt") if tokenizer else torch.tensor([[464]])
        with torch.no_grad():
            output = pt_model.generate(input_ids, max_new_tokens=args.tokens, do_sample=False)
        pt_tokens = output[0].tolist()
        pt_text = tokenizer.decode(pt_tokens) if tokenizer else str(pt_tokens)
        print(f"  PyTorch: {pt_tokens}")
        print(f"  Text:    \"{pt_text}\"")
    except Exception as e:
        print(f"  PyTorch reference failed: {e}")
        pt_tokens = None

    # ─── 25-dispatch generation ───
    print(f"\n{'=' * 70}")
    print("25-DISPATCH GENERATION")
    print(f"{'=' * 70}")
    t0 = time.time()
    ane_25d, gen_time_25d = generate_tokens(
        model, dispatcher, fused_models, prompt_tokens,
        max_new_tokens=args.tokens, mode='25d')
    total_time_25d = time.time() - t0
    n_gen = len(ane_25d) - len(prompt_tokens)

    if tokenizer:
        ane_25d_text = tokenizer.decode(ane_25d)
    else:
        ane_25d_text = str(ane_25d)

    print(f"  Tokens: {ane_25d}")
    print(f"  Text:   \"{ane_25d_text}\"")
    print(f"  Generation time: {gen_time_25d:.3f}s ({n_gen} decode tokens)")
    if gen_time_25d > 0:
        tps_25d = (n_gen - 1) / gen_time_25d  # gen_time starts after first token
        print(f"  Decode tok/s: {tps_25d:.1f}")

    # Kill test
    if pt_tokens:
        match = ane_25d == pt_tokens
        print(f"\n  Kill test (vs PyTorch): {'PASS' if match else 'FAIL'}")
        if not match:
            for i in range(min(len(ane_25d), len(pt_tokens))):
                if i < len(ane_25d) and i < len(pt_tokens):
                    m = "OK" if ane_25d[i] == pt_tokens[i] else "MISMATCH"
                    tok = tokenizer.decode([pt_tokens[i]]) if tokenizer else str(pt_tokens[i])
                    print(f"    pos {i:2d}: 25d={ane_25d[i]:6d} PT={pt_tokens[i]:6d} {m} \"{tok}\"")

    # ─── 37-dispatch comparison ───
    if args.compare:
        print(f"\n{'=' * 70}")
        print("37-DISPATCH GENERATION (baseline)")
        print(f"{'=' * 70}")
        t0 = time.time()
        ane_37d, gen_time_37d = generate_tokens(
            model, dispatcher, None, prompt_tokens,
            max_new_tokens=args.tokens, mode='37d')
        total_time_37d = time.time() - t0

        if tokenizer:
            ane_37d_text = tokenizer.decode(ane_37d)
        else:
            ane_37d_text = str(ane_37d)

        print(f"  Tokens: {ane_37d}")
        print(f"  Text:   \"{ane_37d_text}\"")
        print(f"  Generation time: {gen_time_37d:.3f}s")
        if gen_time_37d > 0:
            tps_37d = (n_gen - 1) / gen_time_37d
            print(f"  Decode tok/s: {tps_37d:.1f}")

        # Token match
        match_37 = ane_25d == ane_37d
        print(f"\n  25d vs 37d match: {'YES' if match_37 else 'NO'}")

    # ─── Summary ───
    print(f"\n{'=' * 70}")
    print("PERFORMANCE SUMMARY")
    print(f"{'=' * 70}")
    print(f"  25-dispatch: {gen_time_25d:.3f}s for {n_gen-1} decode tokens")
    if gen_time_25d > 0:
        tps = (n_gen - 1) / gen_time_25d
        print(f"  25-dispatch tok/s: {tps:.1f}")
    if args.compare and gen_time_37d > 0:
        tps37 = (n_gen - 1) / gen_time_37d
        print(f"  37-dispatch tok/s: {tps37:.1f}")
        if tps > 0 and tps37 > 0:
            speedup = tps / tps37
            print(f"  Speedup: {speedup:.2f}x")

    # Cleanup
    dispatcher.stop()


if __name__ == "__main__":
    main()
