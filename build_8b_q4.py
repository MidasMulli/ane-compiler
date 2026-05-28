#!/usr/bin/env python3
"""
Build Llama-3.1-8B-Instruct Q4 pipeline for ANE (Phase 2 of Q4
production build, M101 CA directive 2026-05-27T19-15-19).

Phase 1 (2026-05-27T19:38:40Z) validated the V3 recipe (per_channel
int4 + macOS26) at single-layer FFN scale:
  Q4 routes to ANE under VERIFIED_ENERGY (cpu_e_ratio 0.137,
  wall_ratio 0.381, ANE energy 5696 units).
  Wall p50: Q4 1.247 ms vs FP16 2.408 ms = 1.93x speedup.
  Probe-projected 1.90x match within 1.7% relative error.

Phase 2 adapts build_8b_q8.py to the V3 recipe at full-8B scale.

Architecture (preserved from Q8 build):
  - 32 layers x (pre_attn + post_attn) = 64 ANE dispatches
  - CPU: RoPE, GQA attention (32Q/8KV, head_dim=128), KV cache
  - lm_head: chunked (128256 / 16032 = 8 chunks) -- FP16 per CA §1
    mixed-precision footprint (preserved for embedding + lm_head +
    layer_norms; ONLY linear weights inside layers are Q4)
  - Total: 72 dispatches (UNCHANGED from Q8; Q4 should not alter
    dispatch shape per CA §3 gate)

Quantization changes vs build_8b_q8.py:
  - pre_attn: linear_quantize_weights mode=linear_symmetric,
    dtype=int4 (was int8), granularity=per_channel (same)
  - post_attn: same recipe
  - lm_head: FP16, NOT quantized (was Q8 per_channel; per CA §1)
  - compile target: iOS26 (was iOS18; THE requirement that unlocks
    Q4 on ANE per the V3 probe + Phase 1 validation)

Mixed-precision footprint:
  - Embedding (FP32 numpy on CPU): unchanged
  - Layer norms (FP32 numpy on CPU): unchanged
  - lm_head: FP16 mlpackage (NOT quantized)
  - Linear weights inside pre_attn + post_attn: Q4 per_channel

Expected:
  - 3.6 GB on disk (vs Q8's 7.2 GB)
  - ~18.1 tok/s per CA §1 + Phase 1 projection
  - 72 dispatches per token (load-bearing gate)

BUILD_DIR is persistent (per CC §3 R4 mitigation): on Q4 build
failure mid-flight, intermediate .mlpackage files survive across
process restarts, harness skips finished layers.

Copyright 2026 Nick Lo. MIT License.
"""

import os
import sys
import time
import numpy as np
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from llama_loader import LlamaModel, LlamaConfig
from kv_cache import KVCache

BUILD_DIR = '/Users/midas/Desktop/cowork/models/llama-8b-q4-ane'
MODEL_PATH = os.path.expanduser(
    "~/.cache/huggingface/hub/models--unsloth--Meta-Llama-3.1-8B-Instruct/"
    "snapshots/a2856192dd7c25b842431f39c179a6c2c2f627d1/"
)


def build_pre_attn(layer, layer_idx, config, save_dir):
    """Fused RMSNorm + QKV projection. Q4 per_channel iOS26."""
    import coremltools as ct
    from coremltools.converters.mil import Builder as mb
    from coremltools.converters.mil.mil import types
    from coremltools.optimize.coreml import (
        OpLinearQuantizerConfig, OptimizationConfig, linear_quantize_weights)

    dim = config.hidden_size
    qkv_out = dim + 2 * (config.n_kv_heads * config.head_dim)

    rms_w = layer.input_layernorm_weight.astype(np.float16)
    W_qkv = layer.W_qkv.astype(np.float16)
    eps_val = np.float32(config.rms_norm_eps)

    path = os.path.join(save_dir, f'L{layer_idx}_pre_q4.mlpackage')
    if os.path.exists(path):
        return ct.models.MLModel(path, compute_units=ct.ComputeUnit.CPU_AND_NE)

    @mb.program(input_specs=[
        mb.TensorSpec(shape=(1, dim, 1, 1), dtype=types.fp16),
    ])
    def pre_attn(x):
        xf = mb.reshape(x=x, shape=[1, dim])
        xf_f32 = mb.cast(x=xf, dtype="fp32")
        sq = mb.mul(x=xf_f32, y=xf_f32)
        mean_sq = mb.reduce_mean(x=sq, axes=[1], keep_dims=True)
        eps_c = mb.const(val=np.array([[eps_val]], dtype=np.float32))
        sum_eps = mb.add(x=mean_sq, y=eps_c)
        rms_inv = mb.rsqrt(x=sum_eps)
        normed = mb.mul(x=xf_f32, y=rms_inv)
        normed_f16 = mb.cast(x=normed, dtype="fp16")
        rms_w_c = mb.const(val=rms_w.reshape(1, dim))
        scaled = mb.mul(x=normed_f16, y=rms_w_c)
        qkv = mb.linear(x=scaled, weight=mb.const(val=W_qkv))
        return mb.reshape(x=qkv, shape=[1, qkv_out, 1, 1])

    fp16_model = ct.convert(pre_attn,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.iOS26)

    q4_config = OptimizationConfig(global_config=OpLinearQuantizerConfig(
        mode="linear_symmetric", dtype="int4", granularity="per_channel"))
    q4_model = linear_quantize_weights(fp16_model, config=q4_config)
    q4_model.save(path)
    return ct.models.MLModel(path, compute_units=ct.ComputeUnit.CPU_AND_NE)


def build_post_attn(layer, layer_idx, config, save_dir):
    """Fused O-proj + residual + RMSNorm + SwiGLU FFN + residual.
    Q4 per_channel iOS26."""
    import coremltools as ct
    from coremltools.converters.mil import Builder as mb
    from coremltools.converters.mil.mil import types
    from coremltools.optimize.coreml import (
        OpLinearQuantizerConfig, OptimizationConfig, linear_quantize_weights)

    dim = config.hidden_size
    ffn_dim = config.intermediate_size

    W_o = layer.o_proj_weight.astype(np.float16)
    rms_w = layer.post_attention_layernorm_weight.astype(np.float16)
    W_gate = layer.gate_proj_weight.astype(np.float16)
    W_up = layer.up_proj_weight.astype(np.float16)
    W_down = layer.down_proj_weight.astype(np.float16)
    eps_val = np.float32(config.rms_norm_eps)

    path = os.path.join(save_dir, f'L{layer_idx}_post_q4.mlpackage')
    if os.path.exists(path):
        return ct.models.MLModel(path, compute_units=ct.ComputeUnit.CPU_AND_NE)

    @mb.program(input_specs=[
        mb.TensorSpec(shape=(1, dim, 1, 1), dtype=types.fp16),
        mb.TensorSpec(shape=(1, dim, 1, 1), dtype=types.fp16),
    ])
    def post_attn(attn_out, x):
        af = mb.reshape(x=attn_out, shape=[1, dim])
        xf = mb.reshape(x=x, shape=[1, dim])
        o_out = mb.linear(x=af, weight=mb.const(val=W_o))
        r1 = mb.add(x=o_out, y=xf)
        r1_f32 = mb.cast(x=r1, dtype="fp32")
        sq = mb.mul(x=r1_f32, y=r1_f32)
        mean_sq = mb.reduce_mean(x=sq, axes=[1], keep_dims=True)
        eps_c = mb.const(val=np.array([[eps_val]], dtype=np.float32))
        sum_eps = mb.add(x=mean_sq, y=eps_c)
        rms_inv = mb.rsqrt(x=sum_eps)
        normed = mb.mul(x=r1_f32, y=rms_inv)
        normed_f16 = mb.cast(x=normed, dtype="fp16")
        rms_w_c = mb.const(val=rms_w.reshape(1, dim))
        ln_out = mb.mul(x=normed_f16, y=rms_w_c)
        gate = mb.linear(x=ln_out, weight=mb.const(val=W_gate))
        up = mb.linear(x=ln_out, weight=mb.const(val=W_up))
        gate_sig = mb.sigmoid(x=gate)
        gate_silu = mb.mul(x=gate, y=gate_sig)
        swiglu = mb.mul(x=gate_silu, y=up)
        down = mb.linear(x=swiglu, weight=mb.const(val=W_down))
        output = mb.add(x=down, y=r1)
        return mb.reshape(x=output, shape=[1, dim, 1, 1])

    fp16_model = ct.convert(post_attn,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.iOS26)

    q4_config = OptimizationConfig(global_config=OpLinearQuantizerConfig(
        mode="linear_symmetric", dtype="int4", granularity="per_channel"))
    q4_model = linear_quantize_weights(fp16_model, config=q4_config)
    q4_model.save(path)
    return ct.models.MLModel(path, compute_units=ct.ComputeUnit.CPU_AND_NE)


def build_lm_head_chunks(model, save_dir):
    """Build lm_head chunk models.

    Per CA §1 mixed-precision footprint: lm_head is FP16, NOT
    quantized. Q4 quantization is linear-only inside transformer
    layers; embedding + lm_head + layer_norms stay FP16.
    """
    import coremltools as ct
    from coremltools.converters.mil import Builder as mb
    from coremltools.converters.mil.mil import types

    dim = model.config.hidden_size
    total_out = model.config.vocab_size
    chunk_size = 16032
    # Use separate lm_head weight if available (8B+), otherwise tied to embed_tokens
    if model.lm_head_weight is not None:
        W_full = model.lm_head_weight.astype(np.float16)
    else:
        W_full = model.embed_tokens.astype(np.float16)

    def _build_chunk(w, co, p):
        @mb.program(input_specs=[mb.TensorSpec(shape=(1, dim, 1, 1), dtype=types.fp16)])
        def prog(x):
            xf = mb.reshape(x=x, shape=[1, dim])
            logits = mb.linear(x=xf, weight=mb.const(val=w))
            return mb.reshape(x=logits, shape=[1, co, 1, 1])
        fp16_m = ct.convert(prog, compute_units=ct.ComputeUnit.CPU_AND_NE,
                            minimum_deployment_target=ct.target.iOS26)
        fp16_m.save(p)

    models = {}
    for i, start in enumerate(range(0, total_out, chunk_size)):
        end = min(start + chunk_size, total_out)
        W_chunk = W_full[start:end, :].copy()
        path = os.path.join(save_dir, f'lm_head_{i}_fp16.mlpackage')
        if not os.path.exists(path):
            _build_chunk(W_chunk, end - start, path)
        models[f'lm_head_{i}'] = ct.models.MLModel(
            path, compute_units=ct.ComputeUnit.CPU_AND_NE)

    return models


# CPU ops (numpy, no C library dependency)

def rms_norm(x_fp16, weight, eps):
    x32 = x_fp16.astype(np.float32)
    ms = np.mean(x32 ** 2)
    return (x32 / np.sqrt(ms + eps) * weight.astype(np.float32)).astype(np.float16)


def _llama3_rope_freqs(head_dim, theta, rope_scaling):
    """Llama 3 extended RoPE: frequency-dependent scaling."""
    factor = rope_scaling["factor"]
    low_freq_factor = rope_scaling["low_freq_factor"]
    high_freq_factor = rope_scaling["high_freq_factor"]
    orig_max_pos = rope_scaling["original_max_position_embeddings"]

    freqs = 1.0 / (theta ** (np.arange(0, head_dim, 2, dtype=np.float64) / head_dim))
    low_freq_wavelen = orig_max_pos / low_freq_factor
    high_freq_wavelen = orig_max_pos / high_freq_factor

    new_freqs = []
    for freq in freqs:
        wavelen = 2 * np.pi / freq
        if wavelen < high_freq_wavelen:
            new_freqs.append(freq)
        elif wavelen > low_freq_wavelen:
            new_freqs.append(freq / factor)
        else:
            smooth = (orig_max_pos / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
            new_freqs.append((1 - smooth) * freq / factor + smooth * freq)
    return np.array(new_freqs, dtype=np.float64)


def rope(q, k, pos, config, rope_scaling=None):
    """RoPE with optional Llama 3 frequency scaling."""
    head_dim = config.head_dim
    half = head_dim // 2

    if rope_scaling and rope_scaling.get("rope_type") == "llama3":
        freqs = _llama3_rope_freqs(head_dim, config.rope_theta, rope_scaling)
    else:
        freqs = 1.0 / (config.rope_theta ** (np.arange(0, half, dtype=np.float64) * 2 / head_dim))

    angles = pos * freqs
    cos_v = np.cos(angles).astype(np.float32)
    sin_v = np.sin(angles).astype(np.float32)
    cos_full = np.concatenate([cos_v, cos_v])
    sin_full = np.concatenate([sin_v, sin_v])

    def apply(x_heads):
        result = np.empty_like(x_heads)
        for h in range(x_heads.shape[0]):
            x = x_heads[h].astype(np.float32)
            rot = np.concatenate([-x[half:], x[:half]])
            result[h] = (x * cos_full + rot * sin_full).astype(np.float16)
        return result

    return apply(q), apply(k)


def gqa_attention(q, cached_k, cached_v, config):
    """GQA attention on CPU. q=[n_heads, head_dim], cached_k/v=[seq, n_kv, hd]."""
    n_heads = config.n_heads
    n_kv = config.n_kv_heads
    n_rep = config.n_rep
    hd = config.head_dim
    seq = cached_k.shape[0]
    scale = 1.0 / np.sqrt(float(hd))

    out = np.zeros(n_heads * hd, dtype=np.float32)
    for h in range(n_heads):
        kv_h = h // n_rep
        q_h = q[h].astype(np.float32)
        scores = np.zeros(seq, dtype=np.float32)
        for s in range(seq):
            scores[s] = np.dot(q_h, cached_k[s, kv_h].astype(np.float32)) * scale
        scores -= scores.max()
        exp_s = np.exp(scores)
        scores = exp_s / exp_s.sum()
        for s in range(seq):
            out[h*hd:(h+1)*hd] += scores[s] * cached_v[s, kv_h].astype(np.float32)

    return out.astype(np.float16)


def main():
    import coremltools as ct

    os.makedirs(BUILD_DIR, exist_ok=True)

    print("=" * 60)
    print("8B Q4 PIPELINE BUILD (V3 recipe per_channel int4 iOS26)")
    print("=" * 60)

    print("\n[1/5] Loading Llama-3.1-8B-Instruct...")
    t0 = time.time()
    model = LlamaModel.from_safetensors(MODEL_PATH)
    c = model.config
    print(f"  Loaded in {time.time()-t0:.1f}s")
    print(f"  {c.hidden_size}h, {c.n_layers}L, {c.n_heads}Q/{c.n_kv_heads}KV, hd={c.head_dim}")
    print(f"  FFN={c.intermediate_size}, vocab={c.vocab_size}")

    print(f"\n[2/5] Building {c.n_layers * 2} layer models + lm_head (Q4 + FP16 lm_head)...")
    t0 = time.time()
    ct_models = {}

    for i in range(c.n_layers):
        ct_models[f'L{i}_pre'] = build_pre_attn(model.layers[i], i, c, BUILD_DIR)
        ct_models[f'L{i}_post'] = build_post_attn(model.layers[i], i, c, BUILD_DIR)
        if (i + 1) % 4 == 0:
            print(f"    Layers 0-{i} done ({time.time()-t0:.0f}s)")

    lm_models = build_lm_head_chunks(model, BUILD_DIR)
    ct_models.update(lm_models)
    build_time = time.time() - t0
    print(f"  Built {len(ct_models)} models in {build_time:.0f}s")

    import subprocess
    total_kb = int(subprocess.check_output(
        ["du", "-sk", BUILD_DIR]).split()[0])
    print(f"  Total Q4 on disk: {total_kb/1024:.0f}MB")

    print(f"\n[3/5] Loading tokenizer...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('unsloth/Meta-Llama-3.1-8B-Instruct')

    embed_fp32 = model.embed_tokens.astype(np.float32)
    layer_rms1_w = [model.layers[i].input_layernorm_weight for i in range(c.n_layers)]
    layer_rms2_w = [model.layers[i].post_attention_layernorm_weight for i in range(c.n_layers)]
    final_rms_w = model.norm_weight

    dim = c.hidden_size
    n_heads = c.n_heads
    n_kv = c.n_kv_heads
    hd = c.head_dim

    def forward_token(token_id, pos, kv):
        x = embed_fp32[token_id].copy()  # FP32

        for li in range(c.n_layers):
            pre_result = ct_models[f'L{li}_pre'].predict({
                'x': x.reshape(1, dim, 1, 1).astype(np.float32)})
            qkv = list(pre_result.values())[0].flatten()

            q = qkv[:dim].reshape(n_heads, hd)
            k = qkv[dim:dim + n_kv * hd].astype(np.float16).reshape(n_kv, hd)
            v = qkv[dim + n_kv * hd:].astype(np.float16).reshape(n_kv, hd)

            q_fp16 = q.astype(np.float16).reshape(n_heads, hd)
            q_fp16, k = rope(q_fp16, k, pos, c, rope_scaling=model.rope_scaling)

            kv.append(li, k[np.newaxis], v[np.newaxis])
            cached_k, cached_v = kv.get(li)

            attn_out = gqa_attention(q_fp16, cached_k, cached_v, c)

            post_result = ct_models[f'L{li}_post'].predict({
                'attn_out': attn_out.reshape(1, dim, 1, 1).astype(np.float32),
                'x': x.reshape(1, dim, 1, 1).astype(np.float32),
            })
            x = list(post_result.values())[0].flatten().astype(np.float32)

        x_fp16 = x.astype(np.float16)
        x_norm = rms_norm(x_fp16, final_rms_w, c.rms_norm_eps)

        logits = np.empty(c.vocab_size, dtype=np.float32)
        offset = 0
        for j in range(len(lm_models)):
            lm_result = ct_models[f'lm_head_{j}'].predict({
                'x': x_norm.reshape(1, dim, 1, 1).astype(np.float32)})
            chunk = list(lm_result.values())[0].flatten().astype(np.float32)
            logits[offset:offset + len(chunk)] = chunk
            offset += len(chunk)

        return int(logits.argmax())

    print(f"\n[4/5] Testing generation...")
    prompts = [
        "The capital of France is",
        "Machine learning is a field of",
        "The largest planet in our solar system is",
        "In 2024, the most popular programming language was",
        "The speed of light is approximately",
    ]

    for prompt in prompts:
        tokens = tokenizer.encode(prompt, add_special_tokens=False)
        kv = KVCache(c.n_layers, n_kv, hd)
        for pos, tok in enumerate(tokens[:-1]):
            forward_token(tok, pos, kv)
        t0 = time.perf_counter()
        next_tok = forward_token(tokens[-1], len(tokens) - 1, kv)
        gen_time = time.perf_counter() - t0
        decoded = tokenizer.decode([next_tok])
        print(f"  \"{prompt}\" -> \"{decoded}\" ({gen_time*1000:.0f}ms)")

    print(f"\n[5/5] Throughput benchmark (20 tokens)...")
    prompt = "The meaning of life is"
    tokens = tokenizer.encode(prompt, add_special_tokens=False)
    kv = KVCache(c.n_layers, n_kv, hd)
    for pos, tok in enumerate(tokens[:-1]):
        forward_token(tok, pos, kv)
    next_tok = forward_token(tokens[-1], len(tokens) - 1, kv)
    generated = [next_tok]
    t_start = time.perf_counter()
    for i in range(19):
        pos = len(tokens) + i
        next_tok = forward_token(next_tok, pos, kv)
        generated.append(next_tok)
    t_total = time.perf_counter() - t_start

    tps = 19 / t_total
    text = tokenizer.decode(generated)
    print(f"  Generated: \"{text[:100]}\"")
    print(f"  19 tokens in {t_total:.1f}s = {tps:.1f} tok/s")

    import resource
    mem_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024
    print(f"  Process memory: {mem_mb:.0f}MB")

    print(f"\n{'='*60}")
    print(f"8B Q4 PIPELINE: {tps:.1f} tok/s, {total_kb/1024:.0f}MB on disk")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
