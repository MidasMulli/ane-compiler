#!/usr/bin/env python3
"""
Build Llama-3.1-8B-Instruct Q8 pipeline for ANE.

Same architecture as Llama-1B 25d+C (run_llama_fused.py), scaled to 8B:
  - 32 layers × (pre_attn + post_attn) = 64 ANE dispatches
  - CPU: RoPE, GQA attention (32Q/8KV, head_dim=128), KV cache
  - lm_head: chunked (128256 / 16032 = 8 chunks)
  - Total: 72 dispatches

Q8 quantization via coremltools linear_quantize_weights.
Expected: ~5.8GB total, ~11.4 tok/s on ANE.

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

BUILD_DIR = '/tmp/llama_8b_q8'
MODEL_PATH = os.path.expanduser(
    "~/.cache/huggingface/hub/models--unsloth--Meta-Llama-3.1-8B-Instruct/"
    "snapshots/a2856192dd7c25b842431f39c179a6c2c2f627d1/"
)


def build_pre_attn(layer, layer_idx, config, save_dir):
    """Fused RMSNorm + QKV projection."""
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

    path = os.path.join(save_dir, f'L{layer_idx}_pre_q8.mlpackage')
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
        minimum_deployment_target=ct.target.iOS18)

    q8_config = OptimizationConfig(global_config=OpLinearQuantizerConfig(
        mode="linear_symmetric", dtype="int8", granularity="per_channel"))
    q8_model = linear_quantize_weights(fp16_model, config=q8_config)
    q8_model.save(path)
    return ct.models.MLModel(path, compute_units=ct.ComputeUnit.CPU_AND_NE)


def build_post_attn(layer, layer_idx, config, save_dir):
    """Fused O-proj + residual + RMSNorm + SwiGLU FFN + residual."""
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

    path = os.path.join(save_dir, f'L{layer_idx}_post_q8.mlpackage')
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
        minimum_deployment_target=ct.target.iOS18)

    q8_config = OptimizationConfig(global_config=OpLinearQuantizerConfig(
        mode="linear_symmetric", dtype="int8", granularity="per_channel"))
    q8_model = linear_quantize_weights(fp16_model, config=q8_config)
    q8_model.save(path)
    return ct.models.MLModel(path, compute_units=ct.ComputeUnit.CPU_AND_NE)


def build_lm_head_chunks(model, save_dir):
    """Build lm_head chunk models, Q8."""
    import coremltools as ct
    from coremltools.converters.mil import Builder as mb
    from coremltools.converters.mil.mil import types
    from coremltools.optimize.coreml import (
        OpLinearQuantizerConfig, OptimizationConfig, linear_quantize_weights)

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
                            minimum_deployment_target=ct.target.iOS18)
        q8_cfg = OptimizationConfig(global_config=OpLinearQuantizerConfig(
            mode="linear_symmetric", dtype="int8", granularity="per_channel"))
        linear_quantize_weights(fp16_m, config=q8_cfg).save(p)

    models = {}
    for i, start in enumerate(range(0, total_out, chunk_size)):
        end = min(start + chunk_size, total_out)
        W_chunk = W_full[start:end, :].copy()
        path = os.path.join(save_dir, f'lm_head_{i}_q8.mlpackage')
        if not os.path.exists(path):
            _build_chunk(W_chunk, end - start, path)
        models[f'lm_head_{i}'] = ct.models.MLModel(
            path, compute_units=ct.ComputeUnit.CPU_AND_NE)

    return models


# ── CPU ops (numpy, no C library dependency) ──

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
        # softmax
        scores -= scores.max()
        exp_s = np.exp(scores)
        scores = exp_s / exp_s.sum()
        # weighted sum
        for s in range(seq):
            out[h*hd:(h+1)*hd] += scores[s] * cached_v[s, kv_h].astype(np.float32)

    return out.astype(np.float16)


def main():
    import coremltools as ct

    os.makedirs(BUILD_DIR, exist_ok=True)

    # ── Load model ──
    print("=" * 60)
    print("8B Q8 PIPELINE BUILD")
    print("=" * 60)

    print("\n[1/5] Loading Llama-3.1-8B-Instruct...")
    t0 = time.time()
    model = LlamaModel.from_safetensors(MODEL_PATH)
    c = model.config
    print(f"  Loaded in {time.time()-t0:.1f}s")
    print(f"  {c.hidden_size}h, {c.n_layers}L, {c.n_heads}Q/{c.n_kv_heads}KV, hd={c.head_dim}")
    print(f"  FFN={c.intermediate_size}, vocab={c.vocab_size}")

    # ── Build all MIL IR models ──
    print(f"\n[2/5] Building {c.n_layers * 2} layer models + lm_head (Q8)...")
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

    # Model size on disk
    import subprocess
    total_kb = int(subprocess.check_output(
        ["du", "-sk", BUILD_DIR]).split()[0])
    print(f"  Total Q8 on disk: {total_kb/1024:.0f}MB")

    # ── Generation loop ──
    print(f"\n[3/5] Loading tokenizer...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('unsloth/Meta-Llama-3.1-8B-Instruct')

    # Pre-convert weights for CPU ops
    embed_fp32 = model.embed_tokens.astype(np.float32)
    layer_rms1_w = [model.layers[i].input_layernorm_weight for i in range(c.n_layers)]
    layer_rms2_w = [model.layers[i].post_attention_layernorm_weight for i in range(c.n_layers)]
    final_rms_w = model.norm_weight

    dim = c.hidden_size
    n_heads = c.n_heads
    n_kv = c.n_kv_heads
    hd = c.head_dim

    def forward_token(token_id, pos, kv):
        # Embedding — keep residual stream in FP32 throughout
        x = embed_fp32[token_id].copy()  # FP32

        for li in range(c.n_layers):
            # Pre-attn (ANE): RMSNorm + QKV
            # ct.predict accepts FP32, handles FP16 cast internally
            pre_result = ct_models[f'L{li}_pre'].predict({
                'x': x.reshape(1, dim, 1, 1).astype(np.float32)})
            qkv = list(pre_result.values())[0].flatten()  # FP32 from ct.predict

            # Split QKV — keep FP16 for KV cache (memory), FP32 for Q (compute)
            q = qkv[:dim].reshape(n_heads, hd)  # FP32
            k = qkv[dim:dim + n_kv * hd].astype(np.float16).reshape(n_kv, hd)
            v = qkv[dim + n_kv * hd:].astype(np.float16).reshape(n_kv, hd)

            # RoPE (CPU, FP32 internally)
            q_fp16 = q.astype(np.float16).reshape(n_heads, hd)
            q_fp16, k = rope(q_fp16, k, pos, c, rope_scaling=model.rope_scaling)

            # KV cache (FP16)
            kv.append(li, k[np.newaxis], v[np.newaxis])
            cached_k, cached_v = kv.get(li)

            # GQA attention (CPU, FP32 internally, returns FP16)
            attn_out = gqa_attention(q_fp16, cached_k, cached_v, c)

            # Post-attn (ANE): O-proj + residual + RMSNorm + FFN + residual
            # Pass x as FP32 — residual stream stays high precision
            post_result = ct_models[f'L{li}_post'].predict({
                'attn_out': attn_out.reshape(1, dim, 1, 1).astype(np.float32),
                'x': x.reshape(1, dim, 1, 1).astype(np.float32),
            })
            x = list(post_result.values())[0].flatten().astype(np.float32)  # FP32

        # Final RMSNorm (CPU, FP32 -> FP16 for output)
        x_fp16 = x.astype(np.float16)
        x_norm = rms_norm(x_fp16, final_rms_w, c.rms_norm_eps)

        # lm_head (ANE, chunked)
        logits = np.empty(c.vocab_size, dtype=np.float32)
        offset = 0
        for j in range(len(lm_models)):
            lm_result = ct_models[f'lm_head_{j}'].predict({
                'x': x_norm.reshape(1, dim, 1, 1).astype(np.float32)})
            chunk = list(lm_result.values())[0].flatten().astype(np.float32)
            logits[offset:offset + len(chunk)] = chunk
            offset += len(chunk)

        return int(logits.argmax())

    # ── Test generation ──
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

        # Prefill
        for pos, tok in enumerate(tokens[:-1]):
            forward_token(tok, pos, kv)

        # Generate 1 token
        t0 = time.perf_counter()
        next_tok = forward_token(tokens[-1], len(tokens) - 1, kv)
        gen_time = time.perf_counter() - t0

        decoded = tokenizer.decode([next_tok])
        print(f"  \"{prompt}\" → \"{decoded}\" ({gen_time*1000:.0f}ms)")

    # ── Throughput benchmark ──
    print(f"\n[5/5] Throughput benchmark (20 tokens)...")
    prompt = "The meaning of life is"
    tokens = tokenizer.encode(prompt, add_special_tokens=False)
    kv = KVCache(c.n_layers, n_kv, hd)

    # Prefill
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

    # Memory check
    import resource
    mem_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024
    print(f"  Process memory: {mem_mb:.0f}MB")

    print(f"\n{'='*60}")
    print(f"8B Q8 PIPELINE: {tps:.1f} tok/s, {total_kb/1024:.0f}MB on disk")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
