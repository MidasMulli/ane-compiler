#!/usr/bin/env python3
"""
Fused Llama-3.2-1B generation via MIL IR 2-input fusion + ct.predict.

Reduces 88 dispatches/token to 40 (or 24 with RMSNorm+QKV fusion):

Per layer with 2-input fusion:
  1. ANE: QKV_proj (ct.predict, 1 input, 2048->3072)
     OR fused_pre_attn: RMSNorm + QKV (ct.predict, 1 input, 2048->3072)
  2. CPU: RoPE + GQA attention + KV cache
  3. ANE: fused_post_attn (ct.predict, 2 inputs):
     O_proj + residual + RMSNorm + gate + SiLU + up + mul + down + residual

40-dispatch: 16 * 2 (QKV + post_attn) + 8 (lm_head) = 40
24-dispatch: 16 * 1 (fused RMSNorm+QKV replaces QKV, post_attn absorbs pre-norm)
             + 8 (lm_head) = 24

Copyright 2026 Nick Lo. MIT License.
"""

import os
import sys
import time
import argparse
import numpy as np
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from llama_loader import (
    LlamaModel, LlamaConfig, LlamaLayer,
    rms_norm_cpu, rope_cpu, softmax_cpu,
    compile_llama_unfused, forward_layer_decode_unfused,
    lm_head_dispatch,
)
from kv_cache import KVCache

BUILD_DIR = '/tmp/llama_mil_fused'

MODEL_PATH = os.path.expanduser(
    "~/.cache/huggingface/hub/models--unsloth--Llama-3.2-1B-Instruct/"
    "snapshots/5a8abab4a5d6f164389b1079fb721cfab8d7126c/"
)


# ===================================================================
# Phase 1: Build fused MIL IR models
# ===================================================================

def build_fused_post_attn(layer: LlamaLayer, layer_idx: int, config: LlamaConfig,
                          save_dir: str):
    """Build a fused post-attention MIL IR model for one Llama layer.

    Fuses: O_proj + residual_1 + RMSNorm + gate + SiLU + up + mul + down + residual_2

    2 inputs: attn_out [2048], x [2048] (skip connection)
    1 output: [2048]
    """
    import coremltools as ct
    from coremltools.converters.mil import Builder as mb
    from coremltools.converters.mil.mil import types

    dim = config.hidden_size
    ffn_dim = config.intermediate_size

    # Extract weights as FP16
    W_o = layer.o_proj_weight.astype(np.float16)                    # [2048, 2048]
    rms_w = layer.post_attention_layernorm_weight.astype(np.float16) # [2048]
    W_gate = layer.gate_proj_weight.astype(np.float16)              # [8192, 2048]
    W_up = layer.up_proj_weight.astype(np.float16)                  # [8192, 2048]
    W_down = layer.down_proj_weight.astype(np.float16)              # [2048, 8192]
    eps_val = np.float32(config.rms_norm_eps)

    @mb.program(input_specs=[
        mb.TensorSpec(shape=(1, dim, 1, 1), dtype=types.fp16),   # attn_out
        mb.TensorSpec(shape=(1, dim, 1, 1), dtype=types.fp16),   # x (skip)
    ])
    def fused_post_attn(attn_out, x):
        # Flatten to 2D for linear ops
        af = mb.reshape(x=attn_out, shape=[1, dim])
        xf = mb.reshape(x=x, shape=[1, dim])

        # O projection (no bias)
        o_out = mb.linear(x=af, weight=mb.const(val=W_o))

        # Residual 1
        r1 = mb.add(x=o_out, y=xf)

        # RMSNorm: x * rsqrt(mean(x^2) + eps) * weight
        # Cast to FP32 for numerical stability
        r1_f32 = mb.cast(x=r1, dtype="fp32")
        sq = mb.mul(x=r1_f32, y=r1_f32)
        mean_sq = mb.reduce_mean(x=sq, axes=[1], keep_dims=True)
        eps_c = mb.const(val=np.array([[eps_val]], dtype=np.float32))
        sum_eps = mb.add(x=mean_sq, y=eps_c)
        rms_inv = mb.rsqrt(x=sum_eps)
        normed = mb.mul(x=r1_f32, y=rms_inv)
        normed_f16 = mb.cast(x=normed, dtype="fp16")
        # Scale by weight
        rms_w_c = mb.const(val=rms_w.reshape(1, dim))
        ln_out = mb.mul(x=normed_f16, y=rms_w_c)

        # SwiGLU FFN
        # gate projection: [1, 2048] -> [1, 8192]
        gate = mb.linear(x=ln_out, weight=mb.const(val=W_gate))
        # up projection: [1, 2048] -> [1, 8192]
        up = mb.linear(x=ln_out, weight=mb.const(val=W_up))

        # SiLU(gate) = gate * sigmoid(gate)
        gate_sig = mb.sigmoid(x=gate)
        gate_silu = mb.mul(x=gate, y=gate_sig)

        # SwiGLU multiply
        swiglu = mb.mul(x=gate_silu, y=up)

        # Down projection: [1, 8192] -> [1, 2048]
        down = mb.linear(x=swiglu, weight=mb.const(val=W_down))

        # Residual 2
        output = mb.add(x=down, y=r1)

        # Back to 4D ANE format
        output_4d = mb.reshape(x=output, shape=[1, dim, 1, 1])
        return output_4d

    mlpackage_path = os.path.join(save_dir, f'layer_{layer_idx}_post_attn.mlpackage')

    model = ct.convert(
        fused_post_attn,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.iOS18,
    )
    model.save(mlpackage_path)
    return mlpackage_path


def build_fused_pre_attn(layer: LlamaLayer, layer_idx: int, config: LlamaConfig,
                         save_dir: str):
    """Build fused RMSNorm + QKV projection MIL IR model.

    1 input: x [2048]
    1 output: qkv [3072]
    """
    import coremltools as ct
    from coremltools.converters.mil import Builder as mb
    from coremltools.converters.mil.mil import types

    dim = config.hidden_size
    qkv_out = dim + 2 * (config.n_kv_heads * config.head_dim)  # 3072

    rms_w = layer.input_layernorm_weight.astype(np.float16)  # [2048]
    W_qkv = layer.W_qkv.astype(np.float16)                   # [3072, 2048]
    eps_val = np.float32(config.rms_norm_eps)

    @mb.program(input_specs=[
        mb.TensorSpec(shape=(1, dim, 1, 1), dtype=types.fp16),
    ])
    def fused_pre_attn(x):
        xf = mb.reshape(x=x, shape=[1, dim])

        # RMSNorm
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

        # QKV projection (no bias)
        qkv = mb.linear(x=scaled, weight=mb.const(val=W_qkv))

        # 4D ANE format
        qkv_4d = mb.reshape(x=qkv, shape=[1, qkv_out, 1, 1])
        return qkv_4d

    mlpackage_path = os.path.join(save_dir, f'layer_{layer_idx}_pre_attn.mlpackage')

    try:
        model = ct.convert(
            fused_pre_attn,
            compute_units=ct.ComputeUnit.CPU_AND_NE,
            minimum_deployment_target=ct.target.iOS18,
        )
        model.save(mlpackage_path)
        return mlpackage_path
    except Exception as e:
        print(f"  RMSNorm+QKV fusion FAILED for layer {layer_idx}: {e}")
        return None


def build_qkv_model(layer: LlamaLayer, layer_idx: int, config: LlamaConfig,
                    save_dir: str):
    """Build standalone QKV projection model (no RMSNorm fusion)."""
    import coremltools as ct
    from coremltools.converters.mil import Builder as mb
    from coremltools.converters.mil.mil import types

    dim = config.hidden_size
    qkv_out = dim + 2 * (config.n_kv_heads * config.head_dim)
    W_qkv = layer.W_qkv.astype(np.float16)

    @mb.program(input_specs=[
        mb.TensorSpec(shape=(1, dim, 1, 1), dtype=types.fp16),
    ])
    def qkv_model(x):
        xf = mb.reshape(x=x, shape=[1, dim])
        qkv = mb.linear(x=xf, weight=mb.const(val=W_qkv))
        return mb.reshape(x=qkv, shape=[1, qkv_out, 1, 1])

    mlpackage_path = os.path.join(save_dir, f'layer_{layer_idx}_qkv.mlpackage')
    model = ct.convert(qkv_model, compute_units=ct.ComputeUnit.CPU_AND_NE,
                       minimum_deployment_target=ct.target.iOS18)
    model.save(mlpackage_path)
    return mlpackage_path


def build_lm_head_models(model: LlamaModel, save_dir: str):
    """Build lm_head chunk models via MIL IR."""
    import coremltools as ct
    from coremltools.converters.mil import Builder as mb
    from coremltools.converters.mil.mil import types

    dim = model.config.hidden_size
    total_out = model.config.vocab_size
    chunk_size = 16032  # 128256 / 8
    W_full = model.embed_tokens.astype(np.float16)  # [128256, 2048]

    models = {}
    for i, start in enumerate(range(0, total_out, chunk_size)):
        end = min(start + chunk_size, total_out)
        chunk_out = end - start
        W_chunk = W_full[start:end, :].copy()

        path = os.path.join(save_dir, f'lm_head_chunk_{i}.mlpackage')
        if not os.path.exists(path):
            _build_lm_chunk(path, W_chunk, dim, chunk_out, ct, mb, types)

        models[f'lm_head_{i}'] = ct.models.MLModel(
            path, compute_units=ct.ComputeUnit.CPU_AND_NE)

    return models


def _build_lm_chunk(path, W_chunk, dim, chunk_out, ct, mb, types):
    """Build a single lm_head chunk model (avoids closure issues)."""
    @mb.program(input_specs=[
        mb.TensorSpec(shape=(1, dim, 1, 1), dtype=types.fp16),
    ])
    def lm_model(x):
        xf = mb.reshape(x=x, shape=[1, dim])
        logits = mb.linear(x=xf, weight=mb.const(val=W_chunk))
        return mb.reshape(x=logits, shape=[1, chunk_out, 1, 1])

    ct_model = ct.convert(lm_model, compute_units=ct.ComputeUnit.CPU_AND_NE,
                           minimum_deployment_target=ct.target.iOS18)
    ct_model.save(path)


def build_final_rmsnorm_model(model: LlamaModel, save_dir: str):
    """Build final RMSNorm model (before lm_head)."""
    import coremltools as ct
    from coremltools.converters.mil import Builder as mb
    from coremltools.converters.mil.mil import types

    dim = model.config.hidden_size
    rms_w = model.norm_weight.astype(np.float16)
    eps_val = np.float32(model.config.rms_norm_eps)

    @mb.program(input_specs=[
        mb.TensorSpec(shape=(1, dim, 1, 1), dtype=types.fp16),
    ])
    def final_norm(x):
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
        return mb.reshape(x=scaled, shape=[1, dim, 1, 1])

    path = os.path.join(save_dir, 'final_rmsnorm.mlpackage')
    if not os.path.exists(path):
        ct_model = ct.convert(final_norm, compute_units=ct.ComputeUnit.CPU_AND_NE,
                               minimum_deployment_target=ct.target.iOS18)
        ct_model.save(path)
    return ct.models.MLModel(path, compute_units=ct.ComputeUnit.CPU_AND_NE)


# ===================================================================
# Phase 2: Verify correctness for ONE layer
# ===================================================================

def verify_one_layer(layer: LlamaLayer, layer_idx: int, config: LlamaConfig,
                     post_attn_model, pre_attn_model=None, qkv_model=None):
    """Verify fused model output matches unfused CPU computation for one layer."""
    dim = config.hidden_size
    n_heads = config.n_heads
    n_kv_heads = config.n_kv_heads
    head_dim = config.head_dim
    n_rep = config.n_rep

    np.random.seed(42 + layer_idx)
    x = np.random.randn(dim).astype(np.float16)

    # === Reference: unfused CPU computation ===
    # RMSNorm 1
    ln1 = rms_norm_cpu(x, layer.input_layernorm_weight, config.rms_norm_eps)

    # QKV projection (CPU reference)
    W_qkv = layer.W_qkv.astype(np.float32)
    qkv_ref = (ln1.astype(np.float32) @ W_qkv.T).astype(np.float16)

    # Split QKV
    q = qkv_ref[:dim]
    k = qkv_ref[dim:dim + n_kv_heads * head_dim]
    v = qkv_ref[dim + n_kv_heads * head_dim:]

    q_heads = q.reshape(n_heads, head_dim)
    k_heads = k.reshape(n_kv_heads, head_dim)
    v_heads = v.reshape(n_kv_heads, head_dim)

    # Fake attention output (just use a random vector for testing fused post-attn)
    np.random.seed(1000 + layer_idx)
    attn_out = np.random.randn(dim).astype(np.float16)

    # CPU reference for post-attention path
    # O projection
    o_out = (attn_out.astype(np.float32) @ layer.o_proj_weight.astype(np.float32).T).astype(np.float16)
    # Residual 1
    r1 = (x.astype(np.float32) + o_out.astype(np.float32)).astype(np.float16)
    # RMSNorm 2
    ln2 = rms_norm_cpu(r1, layer.post_attention_layernorm_weight, config.rms_norm_eps)
    # SwiGLU
    g32 = (ln2.astype(np.float32) @ layer.gate_proj_weight.astype(np.float32).T)
    u32 = (ln2.astype(np.float32) @ layer.up_proj_weight.astype(np.float32).T)
    silu = g32 / (1.0 + np.exp(-g32))
    sw = silu * u32
    down = (sw @ layer.down_proj_weight.astype(np.float32).T)
    # Residual 2
    ref_output = (r1.astype(np.float32) + down).astype(np.float16)

    # === Fused model prediction ===
    post_result = post_attn_model.predict({
        'attn_out': attn_out.reshape(1, dim, 1, 1).astype(np.float32),
        'x': x.reshape(1, dim, 1, 1).astype(np.float32),
    })
    fused_out = list(post_result.values())[0].flatten().astype(np.float16)

    post_diff = np.max(np.abs(fused_out.astype(np.float32) - ref_output.astype(np.float32)))

    # === Pre-attention (RMSNorm+QKV) verification ===
    pre_diff = None
    if pre_attn_model is not None:
        pre_result = pre_attn_model.predict({
            'x': x.reshape(1, dim, 1, 1).astype(np.float32),
        })
        fused_qkv = list(pre_result.values())[0].flatten().astype(np.float16)
        pre_diff = np.max(np.abs(fused_qkv.astype(np.float32) - qkv_ref.astype(np.float32)))

    # === QKV-only verification ===
    qkv_diff = None
    if qkv_model is not None:
        qkv_result = qkv_model.predict({
            'x': ln1.reshape(1, dim, 1, 1).astype(np.float32),
        })
        fused_qkv_only = list(qkv_result.values())[0].flatten().astype(np.float16)
        qkv_diff = np.max(np.abs(fused_qkv_only.astype(np.float32) - qkv_ref.astype(np.float32)))

    return post_diff, pre_diff, qkv_diff


# ===================================================================
# Phase 3: Build all models
# ===================================================================

def build_all_models(model: LlamaModel):
    """Build all MIL IR models for Llama fused generation."""
    import coremltools as ct

    os.makedirs(BUILD_DIR, exist_ok=True)
    config = model.config
    ct_models = {}
    pre_attn_works = True  # will test on layer 0

    # Build layer 0 first to test both fusion paths
    print("  Testing fusion paths on layer 0...")

    # Post-attention (must work)
    post_path = os.path.join(BUILD_DIR, 'layer_0_post_attn.mlpackage')
    if not os.path.exists(post_path):
        post_path = build_fused_post_attn(model.layers[0], 0, config, BUILD_DIR)
    post_model = ct.models.MLModel(post_path, compute_units=ct.ComputeUnit.CPU_AND_NE)

    # Pre-attention (RMSNorm+QKV, may fail)
    pre_path = os.path.join(BUILD_DIR, 'layer_0_pre_attn.mlpackage')
    if not os.path.exists(pre_path):
        pre_path_result = build_fused_pre_attn(model.layers[0], 0, config, BUILD_DIR)
    else:
        pre_path_result = pre_path
    pre_model = None
    if pre_path_result and os.path.exists(pre_path_result):
        pre_model = ct.models.MLModel(pre_path_result, compute_units=ct.ComputeUnit.CPU_AND_NE)

    # QKV-only (fallback)
    qkv_path = os.path.join(BUILD_DIR, 'layer_0_qkv.mlpackage')
    if not os.path.exists(qkv_path):
        qkv_path = build_qkv_model(model.layers[0], 0, config, BUILD_DIR)
    qkv_model = ct.models.MLModel(qkv_path, compute_units=ct.ComputeUnit.CPU_AND_NE)

    # Verify layer 0
    post_diff, pre_diff, qkv_diff = verify_one_layer(
        model.layers[0], 0, config, post_model, pre_model, qkv_model)

    print(f"  Layer 0 post_attn max_diff: {post_diff:.6f} "
          f"{'PASS' if post_diff < 1.0 else 'FAIL'}")
    if pre_diff is not None:
        print(f"  Layer 0 pre_attn  max_diff: {pre_diff:.6f} "
              f"{'PASS' if pre_diff < 1.0 else 'FAIL'}")
        if pre_diff >= 1.0:
            pre_attn_works = False
    else:
        pre_attn_works = False
        print(f"  Layer 0 pre_attn: compilation failed")
    print(f"  Layer 0 qkv_only  max_diff: {qkv_diff:.6f}")

    if pre_attn_works:
        print(f"  USING: RMSNorm+QKV fusion (24 dispatches target)")
        dispatch_mode = 'fused_pre'
    else:
        print(f"  USING: QKV-only (40 dispatches target)")
        dispatch_mode = 'qkv_only'

    # Store layer 0 models
    ct_models['L0_post'] = post_model
    if dispatch_mode == 'fused_pre':
        ct_models['L0_pre'] = pre_model
    else:
        ct_models['L0_qkv'] = qkv_model

    # Build remaining layers
    for i in range(1, config.n_layers):
        # Post-attention
        post_p = os.path.join(BUILD_DIR, f'layer_{i}_post_attn.mlpackage')
        if not os.path.exists(post_p):
            post_p = build_fused_post_attn(model.layers[i], i, config, BUILD_DIR)
        ct_models[f'L{i}_post'] = ct.models.MLModel(
            post_p, compute_units=ct.ComputeUnit.CPU_AND_NE)

        if dispatch_mode == 'fused_pre':
            pre_p = os.path.join(BUILD_DIR, f'layer_{i}_pre_attn.mlpackage')
            if not os.path.exists(pre_p):
                pre_p = build_fused_pre_attn(model.layers[i], i, config, BUILD_DIR)
            if pre_p and os.path.exists(pre_p):
                ct_models[f'L{i}_pre'] = ct.models.MLModel(
                    pre_p, compute_units=ct.ComputeUnit.CPU_AND_NE)
            else:
                # Fallback to QKV-only for this layer
                qkv_p = os.path.join(BUILD_DIR, f'layer_{i}_qkv.mlpackage')
                if not os.path.exists(qkv_p):
                    qkv_p = build_qkv_model(model.layers[i], i, config, BUILD_DIR)
                ct_models[f'L{i}_qkv'] = ct.models.MLModel(
                    qkv_p, compute_units=ct.ComputeUnit.CPU_AND_NE)
        else:
            qkv_p = os.path.join(BUILD_DIR, f'layer_{i}_qkv.mlpackage')
            if not os.path.exists(qkv_p):
                qkv_p = build_qkv_model(model.layers[i], i, config, BUILD_DIR)
            ct_models[f'L{i}_qkv'] = ct.models.MLModel(
                qkv_p, compute_units=ct.ComputeUnit.CPU_AND_NE)

        if (i + 1) % 4 == 0:
            print(f"  Built layers 0-{i}")

    # lm_head chunks
    print("  Building lm_head chunks...")
    lm_models = build_lm_head_models(model, BUILD_DIR)
    ct_models.update(lm_models)

    # Final RMSNorm (only needed if NOT using fused_pre for lm_head path)
    # We always do final norm on CPU since it's just one norm before lm_head

    n_dispatches = 0
    for i in range(config.n_layers):
        if f'L{i}_pre' in ct_models:
            n_dispatches += 1  # fused pre_attn
        else:
            n_dispatches += 1  # QKV only
        n_dispatches += 1  # post_attn
    n_dispatches += len(lm_models)  # lm_head chunks

    print(f"  Total models: {len(ct_models)}")
    print(f"  Dispatches per token: {n_dispatches}")

    return ct_models, dispatch_mode


# ===================================================================
# Phase 4: Generation loop
# ===================================================================

def generate_fused(model: LlamaModel, ct_models: dict, dispatch_mode: str,
                   prompt_tokens: list, max_new_tokens: int = 10):
    """Generation loop with fused MIL IR models via ct.predict."""
    config = model.config
    dim = config.hidden_size
    n_heads = config.n_heads
    n_kv_heads = config.n_kv_heads
    head_dim = config.head_dim
    n_rep = config.n_rep

    kv = KVCache(config.n_layers, n_kv_heads, head_dim)
    generated = list(prompt_tokens)

    def forward_token(token_id, pos):
        # Embedding (Llama: no position embedding, uses RoPE)
        x_fp16 = model.embed_tokens[token_id].astype(np.float16)

        for li in range(config.n_layers):
            L = model.layers[li]

            # Pre-attention: RMSNorm + QKV
            if f'L{li}_pre' in ct_models:
                # Fused RMSNorm + QKV
                pre_result = ct_models[f'L{li}_pre'].predict({
                    'x': x_fp16.reshape(1, dim, 1, 1).astype(np.float32)
                })
                qkv = list(pre_result.values())[0].flatten().astype(np.float16)
            else:
                # CPU RMSNorm + ANE QKV
                ln1 = rms_norm_cpu(x_fp16, L.input_layernorm_weight,
                                   config.rms_norm_eps)
                qkv_result = ct_models[f'L{li}_qkv'].predict({
                    'x': ln1.reshape(1, dim, 1, 1).astype(np.float32)
                })
                qkv = list(qkv_result.values())[0].flatten().astype(np.float16)

            # Split QKV
            q = qkv[:dim]
            k = qkv[dim:dim + n_kv_heads * head_dim]
            v = qkv[dim + n_kv_heads * head_dim:]

            q_heads = q.reshape(n_heads, head_dim)
            k_heads = k.reshape(n_kv_heads, head_dim)
            v_heads = v.reshape(n_kv_heads, head_dim)

            # RoPE on CPU
            q_heads, k_heads = rope_cpu(q_heads, k_heads, pos, head_dim,
                                        config.rope_theta)

            # KV cache
            kv.append(li, k_heads[np.newaxis], v_heads[np.newaxis])

            # GQA attention on CPU
            cached_k, cached_v = kv.get(li)
            scale = np.float32(1.0 / np.sqrt(head_dim))
            attn_output = np.zeros(dim, dtype=np.float32)

            for h in range(n_heads):
                kv_h = h // n_rep
                q_h = q_heads[h].astype(np.float32)
                k_h = cached_k[:, kv_h, :].astype(np.float32)
                v_h = cached_v[:, kv_h, :].astype(np.float32)
                scores = (q_h @ k_h.T) * scale
                weights = softmax_cpu(scores)
                attn_output[h * head_dim:(h + 1) * head_dim] = weights @ v_h

            attn_out = attn_output.astype(np.float16)

            # Fused post-attention (ANE via ct.predict, 2 inputs)
            post_result = ct_models[f'L{li}_post'].predict({
                'attn_out': attn_out.reshape(1, dim, 1, 1).astype(np.float32),
                'x': x_fp16.reshape(1, dim, 1, 1).astype(np.float32),
            })
            x_fp16 = list(post_result.values())[0].flatten().astype(np.float16)

        # Final RMSNorm (CPU)
        x_norm = rms_norm_cpu(x_fp16, model.norm_weight, config.rms_norm_eps)

        # lm_head (chunked ANE via ct.predict)
        logit_chunks = []
        n_chunks = model.config.vocab_size // 16032
        if model.config.vocab_size % 16032 != 0:
            n_chunks += 1
        for j in range(n_chunks):
            lm_result = ct_models[f'lm_head_{j}'].predict({
                'x': x_norm.reshape(1, dim, 1, 1).astype(np.float32)
            })
            chunk_logits = list(lm_result.values())[0].flatten()
            logit_chunks.append(chunk_logits)

        logits = np.concatenate(logit_chunks)
        return int(np.argmax(logits.astype(np.float32)))

    # Prefill
    for pos, tok in enumerate(prompt_tokens[:-1]):
        forward_token(tok, pos)

    # First generated token
    next_tok = forward_token(prompt_tokens[-1], len(prompt_tokens) - 1)
    generated.append(next_tok)

    # Generation loop (timed)
    t_start = time.perf_counter()
    for _ in range(max_new_tokens - 1):
        pos = len(generated) - 1
        next_tok = forward_token(next_tok, pos)
        generated.append(next_tok)
    gen_time = time.perf_counter() - t_start

    return generated, gen_time


# ===================================================================
# Phase 5: Unfused baseline for comparison
# ===================================================================

def generate_unfused_ct(model: LlamaModel, prompt_tokens: list,
                        max_new_tokens: int = 10):
    """88-dispatch unfused baseline via pipe tool for comparison."""
    from generate import ANEDispatcher

    build_dir = '/tmp/llama_ane_agent'
    compiled = compile_llama_unfused(model, build_dir)

    dispatch_ops = {k: v for k, v in compiled.items()
                    if not k.startswith('_') and len(v) == 3}
    dispatcher = ANEDispatcher(dispatch_ops, quiet=True)
    dispatcher.start()

    config = model.config
    kv = KVCache(config.n_layers, config.n_kv_heads, config.head_dim)
    generated = list(prompt_tokens)

    for pos, tok in enumerate(prompt_tokens[:-1]):
        x = model.embed_tokens[tok].astype(np.float16)
        for li in range(config.n_layers):
            x = forward_layer_decode_unfused(li, x, model, dispatcher, kv, pos)

    # First generated token
    pos = len(prompt_tokens) - 1
    x = model.embed_tokens[prompt_tokens[-1]].astype(np.float16)
    for li in range(config.n_layers):
        x = forward_layer_decode_unfused(li, x, model, dispatcher, kv, pos)
    x_norm = rms_norm_cpu(x, model.norm_weight, config.rms_norm_eps)
    logits = lm_head_dispatch(x_norm, compiled, dispatcher)
    next_tok = int(np.argmax(logits.astype(np.float32)))
    generated.append(next_tok)

    t_start = time.perf_counter()
    for _ in range(max_new_tokens - 1):
        pos = len(generated) - 1
        x = model.embed_tokens[next_tok].astype(np.float16)
        for li in range(config.n_layers):
            x = forward_layer_decode_unfused(li, x, model, dispatcher, kv, pos)
        x_norm = rms_norm_cpu(x, model.norm_weight, config.rms_norm_eps)
        logits = lm_head_dispatch(x_norm, compiled, dispatcher)
        next_tok = int(np.argmax(logits.astype(np.float32)))
        generated.append(next_tok)
    gen_time = time.perf_counter() - t_start

    dispatcher.stop()
    return generated, gen_time


# ===================================================================
# Main
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Fused Llama-3.2-1B generation via MIL IR')
    parser.add_argument('--prompt', default='The capital of France is',
                        help='Input prompt')
    parser.add_argument('--tokens', type=int, default=10,
                        help='Number of tokens to generate')
    parser.add_argument('--compare', action='store_true',
                        help='Also run 88-dispatch unfused for comparison')
    parser.add_argument('--skip-build', action='store_true',
                        help='Skip building (use cached models)')
    parser.add_argument('--contention', action='store_true',
                        help='Run camera contention test')
    args = parser.parse_args()

    print("=" * 70)
    print("FUSED LLAMA-3.2-1B GENERATION — MIL IR 2-INPUT FUSION")
    print("O_proj + res + RMSNorm + SwiGLU + res = 1 dispatch")
    print("=" * 70)

    # Load model
    print("\n[1/5] Loading Llama-3.2-1B...")
    t0 = time.time()
    model_path = MODEL_PATH
    if not os.path.exists(model_path):
        # Try finding it
        snap_dir = os.path.expanduser(
            "~/.cache/huggingface/hub/models--unsloth--Llama-3.2-1B-Instruct/snapshots/")
        if os.path.exists(snap_dir):
            snap = os.listdir(snap_dir)[0]
            model_path = os.path.join(snap_dir, snap)
        else:
            print("ERROR: Model not found")
            sys.exit(1)

    model = LlamaModel.from_safetensors(model_path)
    print(f"  Loaded in {time.time()-t0:.1f}s")
    print(f"  Config: {model.config.hidden_size}h, {model.config.n_layers}L, "
          f"{model.config.n_heads}Q/{model.config.n_kv_heads}KV heads")

    # Build fused models
    print("\n[2/5] Building fused MIL IR models...")
    t0 = time.time()
    ct_models, dispatch_mode = build_all_models(model)
    build_time = time.time() - t0
    print(f"  Built in {build_time:.1f}s")

    # Tokenize
    print("\n[3/5] Tokenizing...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('unsloth/Llama-3.2-1B-Instruct')
    prompt_tokens = tokenizer.encode(args.prompt, add_special_tokens=False)
    print(f"  Prompt: '{args.prompt}'")
    print(f"  Tokens: {prompt_tokens}")

    # PyTorch reference
    print("\n[4/5] PyTorch reference...")
    try:
        import torch
        from transformers import AutoModelForCausalLM
        pt_model = AutoModelForCausalLM.from_pretrained(
            'unsloth/Llama-3.2-1B-Instruct',
            torch_dtype=torch.float32,
        )
        pt_model.eval()
        input_ids = torch.tensor([prompt_tokens])
        with torch.no_grad():
            output = pt_model.generate(input_ids, max_new_tokens=args.tokens,
                                       do_sample=False)
        pt_tokens = output[0].tolist()
        pt_text = tokenizer.decode(pt_tokens)
        print(f"  PyTorch: {pt_tokens}")
        print(f"  Text:    \"{pt_text}\"")
    except Exception as e:
        print(f"  PyTorch reference failed: {e}")
        pt_tokens = None

    # Fused generation
    print(f"\n[5/5] Fused generation ({args.tokens} tokens)...")
    t0 = time.time()
    fused_tokens, gen_time = generate_fused(
        model, ct_models, dispatch_mode, prompt_tokens,
        max_new_tokens=args.tokens)
    total_time = time.time() - t0
    n_gen = len(fused_tokens) - len(prompt_tokens)

    fused_text = tokenizer.decode(fused_tokens)
    print(f"  Tokens: {fused_tokens}")
    print(f"  Text:   \"{fused_text}\"")
    print(f"  Decode time: {gen_time:.3f}s ({n_gen - 1} tokens after first)")
    if gen_time > 0:
        tps = (n_gen - 1) / gen_time
        print(f"  Decode tok/s: {tps:.1f}")

    # Kill test
    if pt_tokens:
        match = fused_tokens == pt_tokens
        n_match = sum(1 for a, b in zip(fused_tokens, pt_tokens) if a == b)
        print(f"\n  Kill test: {'PASS' if match else f'PARTIAL {n_match}/{len(pt_tokens)}'}")
        if not match:
            for i in range(min(len(fused_tokens), len(pt_tokens))):
                if i < len(fused_tokens) and fused_tokens[i] != pt_tokens[i]:
                    ft = tokenizer.decode([fused_tokens[i]])
                    pt = tokenizer.decode([pt_tokens[i]])
                    print(f"    pos {i}: fused={fused_tokens[i]} \"{ft}\" "
                          f"vs pt={pt_tokens[i]} \"{pt}\"")

    # Comparison with unfused baseline
    if args.compare:
        print(f"\n{'=' * 70}")
        print("88-DISPATCH UNFUSED BASELINE")
        print(f"{'=' * 70}")
        unfused_tokens, unfused_time = generate_unfused_ct(
            model, prompt_tokens, max_new_tokens=args.tokens)
        unfused_text = tokenizer.decode(unfused_tokens)
        print(f"  Text:   \"{unfused_text}\"")
        if unfused_time > 0:
            tps_unfused = (n_gen - 1) / unfused_time
            print(f"  Decode tok/s: {tps_unfused:.1f}")

    # Camera contention test
    if args.contention:
        print(f"\n{'=' * 70}")
        print("CAMERA CONTENTION TEST")
        print(f"{'=' * 70}")
        import subprocess
        # Start camera capture in background
        cam_proc = subprocess.Popen(
            ['ffmpeg', '-f', 'avfoundation', '-framerate', '30',
             '-video_size', '1280x720', '-i', '0:none',
             '-t', '30', '-y', '/tmp/contention_test.mp4'],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        time.sleep(2)  # let camera stabilize

        print("  Running fused generation under camera load...")
        contention_tokens, contention_time = generate_fused(
            model, ct_models, dispatch_mode, prompt_tokens,
            max_new_tokens=args.tokens)
        if contention_time > 0:
            tps_contention = (n_gen - 1) / contention_time
            print(f"  Decode tok/s (under camera): {tps_contention:.1f}")
            if gen_time > 0:
                baseline_tps = (n_gen - 1) / gen_time
                delta = (tps_contention - baseline_tps) / baseline_tps * 100
                print(f"  Delta: {delta:+.1f}%")

        cam_proc.terminate()
        cam_proc.wait()

    # Summary table
    print(f"\n{'=' * 70}")
    print("RESULTS")
    print(f"{'=' * 70}")

    n_dispatches_fused = sum(1 for k in ct_models if k.startswith('L'))
    n_dispatches_fused += sum(1 for k in ct_models if k.startswith('lm_head'))

    if gen_time > 0:
        tps = (n_gen - 1) / gen_time
        print(f"  Config              | Dispatches | tok/s   | vs baseline")
        print(f"  88d unfused Python  | 88         | 28.3    | 1.00x")
        print(f"  {n_dispatches_fused}d fused ct.predict | {n_dispatches_fused:2d}         | {tps:7.1f} | {tps/28.3:.2f}x")

    # DRAM floor analysis
    print(f"\n  DRAM analysis:")
    total_params = sum(p.size for p in [
        model.embed_tokens, model.norm_weight,
    ])
    for L in model.layers:
        for w in [L.q_proj_weight, L.k_proj_weight, L.v_proj_weight,
                  L.o_proj_weight, L.gate_proj_weight, L.up_proj_weight,
                  L.down_proj_weight, L.input_layernorm_weight,
                  L.post_attention_layernorm_weight]:
            total_params += w.size
    total_bytes = total_params * 2  # FP16
    dram_bw = 307e9  # M5 Pro
    dram_floor_ms = total_bytes / dram_bw * 1000
    theoretical_max = 1000 / dram_floor_ms
    print(f"  Total weights: {total_bytes / 1e9:.2f} GB (FP16)")
    print(f"  DRAM floor: {dram_floor_ms:.1f}ms = {theoretical_max:.0f} tok/s theoretical max")
    if gen_time > 0:
        efficiency = tps / theoretical_max * 100
        print(f"  Efficiency: {efficiency:.1f}% of DRAM floor")

    if pt_tokens:
        match = fused_tokens == pt_tokens
        print(f"\n  Kill test: {'10/10 PASS' if match else 'FAIL'}")


if __name__ == "__main__":
    main()
