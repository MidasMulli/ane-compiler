#!/usr/bin/env python3
"""
25-dispatch GPT-2 generation — fully via coremltools ct.predict.

All ANE dispatches go through CoreML framework (ct.predict), not pipe tool.
This eliminates pipe tool IPC overhead for all dispatches.

Architecture per layer (2 ANE dispatches):
  1. ANE: QKV_proj (ct.predict, 1 input, 768->2304)
  2. CPU: attention
  3. ANE: fused_post_attn (ct.predict, 2 inputs, O_proj+res+LN2+FFN)

Total: 12 * 2 + 1 (lm_head) = 25 dispatches

Copyright 2026 Nick Lo. MIT License.
"""

import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from model_loader import GPT2Model
from first_token import layernorm_cpu, MODEL_PATH
from kv_cache import KVCache
from generate import softmax_cpu

BUILD_DIR = '/tmp/gpt2_mil_fused'


def build_all_mil_models(model):
    """Build all 25 MIL IR models via coremltools."""
    import coremltools as ct
    from coremltools.converters.mil import Builder as mb
    from coremltools.converters.mil.mil import types

    os.makedirs(BUILD_DIR, exist_ok=True)
    models = {}

    for i in range(model.config.n_layer):
        L = model.layers[i]

        # QKV model
        qkv_path = os.path.join(BUILD_DIR, f'layer_{i}_qkv.mlpackage')
        if not os.path.exists(qkv_path):
            W_qkv = L.c_attn_weight.T.copy().astype(np.float16)
            b_qkv = L.c_attn_bias.astype(np.float16)

            @mb.program(input_specs=[
                mb.TensorSpec(shape=(1, 768, 1, 1), dtype=types.fp16),
            ])
            def qkv_model(x):
                x_flat = mb.reshape(x=x, shape=[1, 768])
                qkv = mb.linear(x=x_flat, weight=mb.const(val=W_qkv),
                                bias=mb.const(val=b_qkv))
                return mb.reshape(x=qkv, shape=[1, 2304, 1, 1])

            ct_model = ct.convert(qkv_model, compute_units=ct.ComputeUnit.CPU_AND_NE,
                                   minimum_deployment_target=ct.target.iOS18)
            ct_model.save(qkv_path)

        models[f'L{i}_qkv'] = ct.models.MLModel(
            qkv_path, compute_units=ct.ComputeUnit.CPU_AND_NE)

        # Fused post-attention model
        post_path = os.path.join(BUILD_DIR, f'layer_{i}_post_attn.mlpackage')
        if not os.path.exists(post_path):
            W_o = L.W_o.astype(np.float16)
            b_o = L.c_proj_bias.astype(np.float16)
            ln2_w = L.ln_2_weight.astype(np.float16)
            ln2_b = L.ln_2_bias.astype(np.float16)
            W_up = L.W_fc.astype(np.float16)
            b_up = L.c_fc_bias.astype(np.float16)
            W_down = L.W_fc_down.astype(np.float16)
            b_down = L.c_proj_ffn_bias.astype(np.float16)
            epsilon = np.float16(1e-5)

            @mb.program(input_specs=[
                mb.TensorSpec(shape=(1, 768, 1, 1), dtype=types.fp16),
                mb.TensorSpec(shape=(1, 768, 1, 1), dtype=types.fp16),
            ])
            def post_attn_model(attn_out, x):
                af = mb.reshape(x=attn_out, shape=[1, 768])
                xf = mb.reshape(x=x, shape=[1, 768])
                o = mb.linear(x=af, weight=mb.const(val=W_o), bias=mb.const(val=b_o))
                r1 = mb.add(x=o, y=xf)
                ln = mb.layer_norm(x=r1, axes=[1], gamma=mb.const(val=ln2_w),
                                    beta=mb.const(val=ln2_b), epsilon=epsilon)
                up = mb.linear(x=ln, weight=mb.const(val=W_up), bias=mb.const(val=b_up))
                g = mb.gelu(x=up, mode="TANH_APPROXIMATION")
                dn = mb.linear(x=g, weight=mb.const(val=W_down), bias=mb.const(val=b_down))
                out = mb.add(x=dn, y=r1)
                return mb.reshape(x=out, shape=[1, 768, 1, 1])

            ct_model = ct.convert(post_attn_model, compute_units=ct.ComputeUnit.CPU_AND_NE,
                                   minimum_deployment_target=ct.target.iOS18)
            ct_model.save(post_path)

        models[f'L{i}_post'] = ct.models.MLModel(
            post_path, compute_units=ct.ComputeUnit.CPU_AND_NE)

    # LM head
    lm_path = os.path.join(BUILD_DIR, 'lm_head.mlpackage')
    if not os.path.exists(lm_path):
        W_lm = model.wte.astype(np.float16)

        @mb.program(input_specs=[
            mb.TensorSpec(shape=(1, 768, 1, 1), dtype=types.fp16),
        ])
        def lm_model(x):
            xf = mb.reshape(x=x, shape=[1, 768])
            logits = mb.linear(x=xf, weight=mb.const(val=W_lm))
            return mb.reshape(x=logits, shape=[1, 50257, 1, 1])

        ct_model = ct.convert(lm_model, compute_units=ct.ComputeUnit.CPU_AND_NE,
                               minimum_deployment_target=ct.target.iOS18)
        ct_model.save(lm_path)

    models['lm_head'] = ct.models.MLModel(
        lm_path, compute_units=ct.ComputeUnit.CPU_AND_NE)

    return models


def generate_ct(model, ct_models, prompt_tokens, max_new_tokens=10):
    """Generation loop using ct.predict for all dispatches."""
    config = model.config
    kv = KVCache(config.n_layer, config.n_head, config.head_dim)
    generated = list(prompt_tokens)

    def forward_token(token_id, pos):
        # Embedding
        x_f32 = model.wte[token_id].astype(np.float32) + model.wpe[pos].astype(np.float32)
        x_fp16 = x_f32.astype(np.float16)

        for li in range(config.n_layer):
            L = model.layers[li]

            # LN1 (CPU)
            ln1_out = layernorm_cpu(x_fp16, L.ln_1_weight, L.ln_1_bias,
                                    config.layer_norm_epsilon)

            # QKV (ANE via ct.predict)
            qkv_result = ct_models[f'L{li}_qkv'].predict({
                'x': ln1_out.reshape(1, 768, 1, 1).astype(np.float32)
            })
            qkv = list(qkv_result.values())[0].flatten().astype(np.float16)
            q, k, v = qkv[:768], qkv[768:1536], qkv[1536:]

            # Attention (CPU)
            q_h = q.reshape(12, 64)
            k_h = k.reshape(12, 64)
            v_h = v.reshape(12, 64)
            kv.append(li, k_h[np.newaxis], v_h[np.newaxis])
            ck, cv = kv.get(li)
            scale = np.float32(1.0 / np.sqrt(64))
            attn_out = np.zeros(768, dtype=np.float32)
            for h in range(12):
                qh = q_h[h].astype(np.float32)
                kh = ck[:, h, :].astype(np.float32)
                vh = cv[:, h, :].astype(np.float32)
                scores = (qh @ kh.T) * scale
                weights = softmax_cpu(scores)
                attn_out[h*64:(h+1)*64] = weights @ vh

            # Fused post-attention (ANE via ct.predict)
            post_result = ct_models[f'L{li}_post'].predict({
                'attn_out': attn_out.astype(np.float32).reshape(1, 768, 1, 1),
                'x': x_fp16.reshape(1, 768, 1, 1).astype(np.float32),
            })
            x_fp16 = list(post_result.values())[0].flatten().astype(np.float16)

        # Final LN + LM head
        x_ln = layernorm_cpu(x_fp16, model.ln_f_weight, model.ln_f_bias,
                             config.layer_norm_epsilon)
        lm_result = ct_models['lm_head'].predict({
            'x': x_ln.reshape(1, 768, 1, 1).astype(np.float32)
        })
        logits = list(lm_result.values())[0].flatten()
        return int(np.argmax(logits.astype(np.float32)))

    # Prefill
    for pos, tok in enumerate(prompt_tokens[:-1]):
        forward_token(tok, pos)

    # First generated token
    next_tok = forward_token(prompt_tokens[-1], len(prompt_tokens) - 1)
    generated.append(next_tok)

    # Generation loop
    t_start = time.perf_counter()
    for _ in range(max_new_tokens - 1):
        pos = len(generated) - 1
        next_tok = forward_token(next_tok, pos)
        generated.append(next_tok)
    gen_time = time.perf_counter() - t_start

    return generated, gen_time


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt', default='The capital of France is')
    parser.add_argument('--tokens', type=int, default=50)
    args = parser.parse_args()

    print("=" * 70)
    print("25-DISPATCH GPT-2 — FULL ct.predict PATH")
    print("=" * 70)

    print("\n[1/4] Loading GPT-2...")
    model = GPT2Model.from_safetensors(MODEL_PATH)

    print("[2/4] Building MIL IR models...")
    t0 = time.time()
    ct_models = build_all_mil_models(model)
    print(f"  Built {len(ct_models)} models in {time.time()-t0:.1f}s")

    # Tokenize
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained('gpt2')
        prompt_tokens = tokenizer.encode(args.prompt)
    except ImportError:
        prompt_tokens = [464]
        tokenizer = None

    # PyTorch reference
    print("[3/4] PyTorch reference...")
    try:
        import torch
        from transformers import GPT2LMHeadModel
        pt_model = GPT2LMHeadModel.from_pretrained('openai-community/gpt2')
        pt_model.eval()
        input_ids = tokenizer.encode(args.prompt, return_tensors="pt")
        with torch.no_grad():
            output = pt_model.generate(input_ids, max_new_tokens=args.tokens, do_sample=False)
        pt_tokens = output[0].tolist()
    except Exception as e:
        print(f"  Failed: {e}")
        pt_tokens = None

    # Generate
    print(f"[4/4] Generating {args.tokens} tokens...")
    generated, gen_time = generate_ct(model, ct_models, prompt_tokens,
                                       max_new_tokens=args.tokens)
    n_gen = len(generated) - len(prompt_tokens)

    if tokenizer:
        text = tokenizer.decode(generated)
    else:
        text = str(generated)

    print(f"\n  Text: \"{text}\"")
    print(f"  Decode time: {gen_time:.3f}s ({n_gen-1} tokens after first)")
    if gen_time > 0:
        tps = (n_gen - 1) / gen_time
        print(f"  Decode tok/s: {tps:.1f}")

    if pt_tokens:
        match = generated == pt_tokens
        n_match = sum(1 for a, b in zip(generated, pt_tokens) if a == b)
        print(f"  Kill test: {'PASS' if match else f'PARTIAL {n_match}/{len(pt_tokens)}'}")
        if not match and tokenizer:
            for i in range(min(len(generated), len(pt_tokens))):
                if generated[i] != pt_tokens[i]:
                    print(f"  First mismatch at pos {i}: gen={generated[i]} pt={pt_tokens[i]}")
                    break

    # Summary table
    print(f"\n{'=' * 70}")
    print("RESULTS")
    print(f"{'=' * 70}")
    if gen_time > 0:
        tps = (n_gen - 1) / gen_time
        print(f"  Config          | Dispatches | tok/s   | vs C baseline")
        print(f"  37 serial C     | 37         | 137.7   | baseline")
        print(f"  25 ct.predict   | 25         | {tps:7.1f} | {tps/137.7:.2f}x")


if __name__ == "__main__":
    main()
