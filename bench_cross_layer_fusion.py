#!/usr/bin/env python3
"""
Cross-layer post-attention fusion for Llama-3.2-1B.

Fuses post_attn_i + pre_attn_(i+1) into a single ANE dispatch, reducing
dispatches from 40 to 25 per token. The key insight from GPT-2 SRAM
pipelining: fusion depth directly determines effective ANE bandwidth
(37-dispatch=34 GB/s, 1-dispatch=138 GB/s). Every eliminated dispatch
boundary is a chance to keep intermediates in SRAM instead of round-
tripping through DRAM.

Current pipeline (40 dispatches):
  Per layer (x16):
    1. ANE: fused_pre_attn  (RMSNorm_i + QKV_i)            -> 1 dispatch
    2. CPU: RoPE + GQA attention + KV cache
    3. ANE: fused_post_attn (O_proj_i + res + RMSNorm_i_post
                             + SwiGLU_i + res)               -> 1 dispatch
  Plus: 8 lm_head chunks = 8 dispatches
  Total: 16*2 + 8 = 40

Cross-layer fused pipeline (25 dispatches):
  Layer 0:
    1. ANE: fused_pre_attn_0 (standalone)                    -> 1 dispatch
    2. CPU: RoPE + attention + KV cache
  Layers 0-14 (15 cross-layer pairs):
    3. ANE: cross_layer_i (post_attn_i + pre_attn_(i+1))    -> 1 dispatch
    4. CPU: RoPE + attention + KV cache for layer i+1
  Layer 15:
    5. ANE: fused_post_attn_15 (standalone)                  -> 1 dispatch
  Plus: 8 lm_head chunks
  Total: 1 + 15 + 1 + 8 = 25

Each cross-layer dispatch fuses:
  Inputs: attn_out_i [2048], x_i [2048]
  Ops: O_proj_i + residual_i + RMSNorm_i_post + SwiGLU_i(gate+SiLU+up+mul+down)
       + residual_i + RMSNorm_(i+1)_pre + QKV_(i+1)
  Output: qkv_(i+1) [3072]

This is valid because CPU attention MUST happen between layers (it needs
qkv output and produces attn_out), so the maximum fusable unit between
two CPU attention stages is exactly: post_attn_i + pre_attn_(i+1).

Expected bandwidth improvement:
  40 dispatches: each dispatch boundary forces SRAM->DRAM->SRAM roundtrip
  25 dispatches: 15 fewer boundaries, intermediates stay in 16MB ANE SRAM
  The cross-layer model is ~2x the weight of either standalone model, so
  we lose nothing on weight bandwidth (weights stream once either way).
  We GAIN by keeping the dim=2048 activation vector in SRAM across what
  was previously 2 separate dispatches.

  Predicted: 37.5% fewer dispatches -> 15-25% tok/s improvement
  (sublinear because lm_head's 8 dispatches are unchanged and CPU
  attention time is unchanged)

Kill test: 10/10 token exact match vs run_llama_fused.py before any
throughput claim.

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
)
from kv_cache import KVCache

BUILD_DIR = '/tmp/llama_cross_layer_fused'

MODEL_PATH = os.path.expanduser(
    "~/.cache/huggingface/hub/models--unsloth--Llama-3.2-1B-Instruct/"
    "snapshots/5a8abab4a5d6f164389b1079fb721cfab8d7126c/"
)


# ===================================================================
# Cross-layer fusion model builder
# ===================================================================

def build_cross_layer_model(layer_i: LlamaLayer, layer_next: LlamaLayer,
                            layer_idx: int, config: LlamaConfig,
                            save_dir: str):
    """Build a fused post_attn_i + pre_attn_(i+1) MIL IR model.

    Fuses: O_proj_i + residual_i + RMSNorm_i_post + SwiGLU_i
           + residual_i + RMSNorm_(i+1)_pre + QKV_(i+1)

    2 inputs:  attn_out_i [2048], x_i [2048]
    1 output:  qkv_(i+1) [3072]

    Plus a second output: x_(i+1) [2048] (the hidden state after layer i,
    needed as the residual input for post_attn_(i+1) or the next cross-layer
    model).

    Actually we need x_(i+1) for the NEXT cross-layer model's 'x' input.
    So this model has 2 outputs:
      - qkv_(i+1): [3072] for CPU attention in layer i+1
      - x_(i+1):   [2048] the hidden state (residual output of layer i)
    """
    import coremltools as ct
    from coremltools.converters.mil import Builder as mb
    from coremltools.converters.mil.mil import types

    dim = config.hidden_size        # 2048
    ffn_dim = config.intermediate_size  # 8192
    qkv_out = dim + 2 * (config.n_kv_heads * config.head_dim)  # 3072
    eps_val = np.float32(config.rms_norm_eps)

    # ---- Layer i weights (post-attention) ----
    W_o_i = layer_i.o_proj_weight.astype(np.float16)                        # [2048, 2048]
    rms_post_i = layer_i.post_attention_layernorm_weight.astype(np.float16) # [2048]
    W_gate_i = layer_i.gate_proj_weight.astype(np.float16)                  # [8192, 2048]
    W_up_i = layer_i.up_proj_weight.astype(np.float16)                      # [8192, 2048]
    W_down_i = layer_i.down_proj_weight.astype(np.float16)                  # [2048, 8192]

    # ---- Layer i+1 weights (pre-attention) ----
    rms_pre_next = layer_next.input_layernorm_weight.astype(np.float16)     # [2048]
    W_qkv_next = layer_next.W_qkv.astype(np.float16)                       # [3072, 2048]

    @mb.program(input_specs=[
        mb.TensorSpec(shape=(1, dim, 1, 1), dtype=types.fp16),   # attn_out_i
        mb.TensorSpec(shape=(1, dim, 1, 1), dtype=types.fp16),   # x_i (residual)
    ])
    def cross_layer(attn_out, x):
        # === Flatten to 2D ===
        af = mb.reshape(x=attn_out, shape=[1, dim])
        xf = mb.reshape(x=x, shape=[1, dim])

        # === Post-attention for layer i ===

        # O projection (no bias)
        o_out = mb.linear(x=af, weight=mb.const(val=W_o_i))

        # Residual 1: O_proj + x
        r1 = mb.add(x=o_out, y=xf)

        # RMSNorm (post-attention layer i)
        r1_f32 = mb.cast(x=r1, dtype="fp32")
        sq = mb.mul(x=r1_f32, y=r1_f32)
        mean_sq = mb.reduce_mean(x=sq, axes=[1], keep_dims=True)
        eps_c = mb.const(val=np.array([[eps_val]], dtype=np.float32))
        sum_eps = mb.add(x=mean_sq, y=eps_c)
        rms_inv = mb.rsqrt(x=sum_eps)
        normed = mb.mul(x=r1_f32, y=rms_inv)
        normed_f16 = mb.cast(x=normed, dtype="fp16")
        rms_post_w = mb.const(val=rms_post_i.reshape(1, dim))
        ln_out = mb.mul(x=normed_f16, y=rms_post_w)

        # SwiGLU FFN for layer i
        gate = mb.linear(x=ln_out, weight=mb.const(val=W_gate_i))
        up = mb.linear(x=ln_out, weight=mb.const(val=W_up_i))
        gate_sig = mb.sigmoid(x=gate)
        gate_silu = mb.mul(x=gate, y=gate_sig)
        swiglu = mb.mul(x=gate_silu, y=up)
        down = mb.linear(x=swiglu, weight=mb.const(val=W_down_i))

        # Residual 2: FFN output + r1
        x_next = mb.add(x=down, y=r1)

        # === Pre-attention for layer i+1 ===

        # RMSNorm (pre-attention layer i+1)
        xn_f32 = mb.cast(x=x_next, dtype="fp32")
        sq2 = mb.mul(x=xn_f32, y=xn_f32)
        mean_sq2 = mb.reduce_mean(x=sq2, axes=[1], keep_dims=True)
        eps_c2 = mb.const(val=np.array([[eps_val]], dtype=np.float32))
        sum_eps2 = mb.add(x=mean_sq2, y=eps_c2)
        rms_inv2 = mb.rsqrt(x=sum_eps2)
        normed2 = mb.mul(x=xn_f32, y=rms_inv2)
        normed2_f16 = mb.cast(x=normed2, dtype="fp16")
        rms_pre_w = mb.const(val=rms_pre_next.reshape(1, dim))
        ln2_out = mb.mul(x=normed2_f16, y=rms_pre_w)

        # QKV projection for layer i+1 (no bias)
        qkv = mb.linear(x=ln2_out, weight=mb.const(val=W_qkv_next))

        # === Outputs in 4D ANE format ===
        qkv_4d = mb.reshape(x=qkv, shape=[1, qkv_out, 1, 1])
        x_next_4d = mb.reshape(x=x_next, shape=[1, dim, 1, 1])

        return qkv_4d, x_next_4d

    mlpackage_path = os.path.join(save_dir,
                                   f'cross_layer_{layer_idx}_{layer_idx+1}.mlpackage')

    model = ct.convert(
        cross_layer,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.iOS18,
    )
    model.save(mlpackage_path)
    return mlpackage_path


def build_standalone_pre_attn(layer: LlamaLayer, layer_idx: int,
                              config: LlamaConfig, save_dir: str):
    """Build standalone fused RMSNorm + QKV for layer 0 (no preceding post_attn).

    1 input: x [2048]
    1 output: qkv [3072]

    Identical to build_fused_pre_attn in run_llama_fused.py.
    """
    import coremltools as ct
    from coremltools.converters.mil import Builder as mb
    from coremltools.converters.mil.mil import types

    dim = config.hidden_size
    qkv_out = dim + 2 * (config.n_kv_heads * config.head_dim)  # 3072
    eps_val = np.float32(config.rms_norm_eps)

    rms_w = layer.input_layernorm_weight.astype(np.float16)
    W_qkv = layer.W_qkv.astype(np.float16)

    @mb.program(input_specs=[
        mb.TensorSpec(shape=(1, dim, 1, 1), dtype=types.fp16),
    ])
    def pre_attn(x):
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

        # QKV
        qkv = mb.linear(x=scaled, weight=mb.const(val=W_qkv))
        return mb.reshape(x=qkv, shape=[1, qkv_out, 1, 1])

    mlpackage_path = os.path.join(save_dir, f'pre_attn_{layer_idx}.mlpackage')
    ct_model = ct.convert(pre_attn, compute_units=ct.ComputeUnit.CPU_AND_NE,
                          minimum_deployment_target=ct.target.iOS18)
    ct_model.save(mlpackage_path)
    return mlpackage_path


def build_standalone_post_attn(layer: LlamaLayer, layer_idx: int,
                               config: LlamaConfig, save_dir: str):
    """Build standalone post_attn for layer 15 (no following pre_attn).

    2 inputs: attn_out [2048], x [2048]
    1 output: hidden [2048]

    Identical to build_fused_post_attn in run_llama_fused.py.
    """
    import coremltools as ct
    from coremltools.converters.mil import Builder as mb
    from coremltools.converters.mil.mil import types

    dim = config.hidden_size
    ffn_dim = config.intermediate_size
    eps_val = np.float32(config.rms_norm_eps)

    W_o = layer.o_proj_weight.astype(np.float16)
    rms_w = layer.post_attention_layernorm_weight.astype(np.float16)
    W_gate = layer.gate_proj_weight.astype(np.float16)
    W_up = layer.up_proj_weight.astype(np.float16)
    W_down = layer.down_proj_weight.astype(np.float16)

    @mb.program(input_specs=[
        mb.TensorSpec(shape=(1, dim, 1, 1), dtype=types.fp16),   # attn_out
        mb.TensorSpec(shape=(1, dim, 1, 1), dtype=types.fp16),   # x (skip)
    ])
    def post_attn(attn_out, x):
        af = mb.reshape(x=attn_out, shape=[1, dim])
        xf = mb.reshape(x=x, shape=[1, dim])

        o_out = mb.linear(x=af, weight=mb.const(val=W_o))
        r1 = mb.add(x=o_out, y=xf)

        # RMSNorm
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

        # SwiGLU
        gate = mb.linear(x=ln_out, weight=mb.const(val=W_gate))
        up = mb.linear(x=ln_out, weight=mb.const(val=W_up))
        gate_sig = mb.sigmoid(x=gate)
        gate_silu = mb.mul(x=gate, y=gate_sig)
        swiglu = mb.mul(x=gate_silu, y=up)
        down = mb.linear(x=swiglu, weight=mb.const(val=W_down))

        output = mb.add(x=down, y=r1)
        return mb.reshape(x=output, shape=[1, dim, 1, 1])

    mlpackage_path = os.path.join(save_dir, f'post_attn_{layer_idx}.mlpackage')
    ct_model = ct.convert(post_attn, compute_units=ct.ComputeUnit.CPU_AND_NE,
                          minimum_deployment_target=ct.target.iOS18)
    ct_model.save(mlpackage_path)
    return mlpackage_path


def build_lm_head_models(model: LlamaModel, save_dir: str):
    """Build lm_head chunk models via MIL IR (same as run_llama_fused.py)."""
    import coremltools as ct
    from coremltools.converters.mil import Builder as mb
    from coremltools.converters.mil.mil import types

    dim = model.config.hidden_size
    total_out = model.config.vocab_size
    chunk_size = 16032  # 128256 / 8
    W_full = model.embed_tokens.astype(np.float16)

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
    """Build a single lm_head chunk model."""
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


# ===================================================================
# Correctness verification
# ===================================================================

def verify_cross_layer_model(layer_i: LlamaLayer, layer_next: LlamaLayer,
                             layer_idx: int, config: LlamaConfig,
                             cross_model) -> tuple:
    """Verify cross-layer fused model vs sequential CPU reference.

    Returns: (qkv_max_diff, x_next_max_diff)
    """
    dim = config.hidden_size
    n_heads = config.n_heads
    n_kv_heads = config.n_kv_heads
    head_dim = config.head_dim
    eps = config.rms_norm_eps

    np.random.seed(42 + layer_idx)
    attn_out = np.random.randn(dim).astype(np.float16)
    x = np.random.randn(dim).astype(np.float16)

    # --- CPU reference: post_attn_i ---
    # O projection
    o_out = (attn_out.astype(np.float32) @
             layer_i.o_proj_weight.astype(np.float32).T)
    # Residual 1
    r1 = x.astype(np.float32) + o_out
    # RMSNorm post
    rms = np.sqrt(np.mean(r1 ** 2) + eps)
    ln_out = (r1 / rms) * layer_i.post_attention_layernorm_weight.astype(np.float32)
    # SwiGLU
    gate = ln_out @ layer_i.gate_proj_weight.astype(np.float32).T
    up = ln_out @ layer_i.up_proj_weight.astype(np.float32).T
    silu = gate / (1.0 + np.exp(-gate))
    sw = silu * up
    down = sw @ layer_i.down_proj_weight.astype(np.float32).T
    # Residual 2
    x_next_ref = (r1 + down).astype(np.float16)

    # --- CPU reference: pre_attn_(i+1) ---
    xn_f32 = x_next_ref.astype(np.float32)
    rms2 = np.sqrt(np.mean(xn_f32 ** 2) + eps)
    ln2_out = (xn_f32 / rms2) * layer_next.input_layernorm_weight.astype(np.float32)
    W_qkv = layer_next.W_qkv.astype(np.float32)
    qkv_ref = (ln2_out @ W_qkv.T).astype(np.float16)

    # --- Fused model ---
    result = cross_model.predict({
        'attn_out': attn_out.reshape(1, dim, 1, 1).astype(np.float32),
        'x': x.reshape(1, dim, 1, 1).astype(np.float32),
    })

    # Multi-output: get both outputs
    result_keys = sorted(result.keys())
    # The outputs are named by MIL — find qkv (3072) and x_next (2048) by shape
    out_arrays = {}
    for k in result_keys:
        arr = result[k].flatten()
        out_arrays[len(arr)] = arr

    fused_qkv = out_arrays[dim + 2 * (n_kv_heads * head_dim)].astype(np.float16)
    fused_x_next = out_arrays[dim].astype(np.float16)

    qkv_diff = np.max(np.abs(fused_qkv.astype(np.float32) -
                               qkv_ref.astype(np.float32)))
    x_diff = np.max(np.abs(fused_x_next.astype(np.float32) -
                            x_next_ref.astype(np.float32)))

    return qkv_diff, x_diff


def verify_full_pipeline(model: LlamaModel, ct_models: dict,
                         cross_models: dict, prompt_tokens: list,
                         reference_tokens: list) -> bool:
    """Kill test: compare cross-layer output tokens vs reference tokens.

    Returns True if ALL tokens match.
    """
    config = model.config
    gen_tokens = generate_cross_layer(model, ct_models, cross_models,
                                      prompt_tokens, max_new_tokens=len(reference_tokens))
    match_count = 0
    for i, (gen, ref) in enumerate(zip(gen_tokens[len(prompt_tokens):],
                                        reference_tokens)):
        match = gen == ref
        if match:
            match_count += 1
        else:
            print(f"  Token {i}: gen={gen} ref={ref} MISMATCH")
    total = len(reference_tokens)
    print(f"  Kill test: {match_count}/{total} tokens match")
    return match_count == total


# ===================================================================
# Build all models
# ===================================================================

def build_all_models(model: LlamaModel):
    """Build the full 25-dispatch cross-layer fused model set.

    Returns:
      ct_models: dict of loaded ct.models.MLModel instances
      cross_model_keys: list of cross-layer model keys for dispatch counting
    """
    import coremltools as ct

    os.makedirs(BUILD_DIR, exist_ok=True)
    config = model.config
    ct_models = {}

    # 1. Standalone pre_attn for layer 0
    print("  Building standalone pre_attn_0...")
    pre_path = os.path.join(BUILD_DIR, 'pre_attn_0.mlpackage')
    if not os.path.exists(pre_path):
        pre_path = build_standalone_pre_attn(model.layers[0], 0, config, BUILD_DIR)
    ct_models['pre_attn_0'] = ct.models.MLModel(
        pre_path, compute_units=ct.ComputeUnit.CPU_AND_NE)

    # 2. Cross-layer models for layers 0-14 (15 pairs)
    cross_keys = []
    for i in range(config.n_layers - 1):  # 0..14
        key = f'cross_{i}_{i+1}'
        cross_keys.append(key)
        mlpkg_path = os.path.join(BUILD_DIR,
                                   f'cross_layer_{i}_{i+1}.mlpackage')
        if not os.path.exists(mlpkg_path):
            t0 = time.time()
            mlpkg_path = build_cross_layer_model(
                model.layers[i], model.layers[i+1], i, config, BUILD_DIR)
            elapsed = time.time() - t0
            print(f"  Cross-layer {i}->{i+1}: built in {elapsed:.1f}s")
        else:
            print(f"  Cross-layer {i}->{i+1}: cached")

        ct_models[key] = ct.models.MLModel(
            mlpkg_path, compute_units=ct.ComputeUnit.CPU_AND_NE)

    # 3. Standalone post_attn for layer 15 (final layer)
    last = config.n_layers - 1
    print(f"  Building standalone post_attn_{last}...")
    post_path = os.path.join(BUILD_DIR, f'post_attn_{last}.mlpackage')
    if not os.path.exists(post_path):
        post_path = build_standalone_post_attn(
            model.layers[last], last, config, BUILD_DIR)
    ct_models[f'post_attn_{last}'] = ct.models.MLModel(
        post_path, compute_units=ct.ComputeUnit.CPU_AND_NE)

    # 4. lm_head chunks (8 models)
    print("  Building lm_head chunks...")
    lm_models = build_lm_head_models(model, BUILD_DIR)
    ct_models.update(lm_models)

    # Dispatch count
    n_dispatches = 1 + len(cross_keys) + 1 + len(lm_models)
    print(f"\n  Total models: {len(ct_models)}")
    print(f"  Dispatches per token: {n_dispatches}")
    print(f"    1 (pre_attn_0) + {len(cross_keys)} (cross-layer) + "
          f"1 (post_attn_{last}) + {len(lm_models)} (lm_head) = {n_dispatches}")

    return ct_models, cross_keys


# ===================================================================
# Correctness verification phase
# ===================================================================

def verify_all_cross_layer(model: LlamaModel, ct_models: dict):
    """Verify all 15 cross-layer models against CPU reference."""
    import coremltools as ct
    config = model.config
    all_pass = True

    for i in range(config.n_layers - 1):
        key = f'cross_{i}_{i+1}'
        qkv_diff, x_diff = verify_cross_layer_model(
            model.layers[i], model.layers[i+1], i, config, ct_models[key])
        status = "PASS" if qkv_diff < 2.0 and x_diff < 2.0 else "FAIL"
        if status == "FAIL":
            all_pass = False
        print(f"  Cross {i:2d}->{i+1:2d}: qkv_diff={qkv_diff:.4f} "
              f"x_diff={x_diff:.4f} [{status}]")

    return all_pass


# ===================================================================
# Generation loop (25 dispatches)
# ===================================================================

def generate_cross_layer(model: LlamaModel, ct_models: dict,
                         cross_keys: list, prompt_tokens: list,
                         max_new_tokens: int = 10):
    """Generation loop with cross-layer fusion (25 dispatches/token).

    Pipeline per token:
      1. ANE: pre_attn_0 (x -> qkv_0)
      2. CPU: attention layer 0
      3. For i in 0..14:
         a. ANE: cross_{i}_{i+1} (attn_out_i, x_i -> qkv_{i+1}, x_{i+1})
         b. CPU: attention layer i+1
      4. ANE: post_attn_15 (attn_out_15, x_15 -> hidden)
      5. CPU: final RMSNorm
      6. ANE: 8x lm_head chunks
      7. CPU: argmax
    """
    config = model.config
    dim = config.hidden_size
    n_heads = config.n_heads
    n_kv_heads = config.n_kv_heads
    head_dim = config.head_dim
    n_rep = config.n_rep
    n_layers = config.n_layers

    kv = KVCache(n_layers, n_kv_heads, head_dim)
    generated = list(prompt_tokens)

    def do_attention_cpu(qkv_flat, layer_idx, pos):
        """CPU attention: split QKV, RoPE, KV cache, GQA attention.

        Args:
            qkv_flat: [3072] FP16 (Q[2048] + K[512] + V[512])
            layer_idx: layer index
            pos: sequence position

        Returns:
            attn_out: [2048] FP16
        """
        q = qkv_flat[:dim]
        k = qkv_flat[dim:dim + n_kv_heads * head_dim]
        v = qkv_flat[dim + n_kv_heads * head_dim:]

        q_heads = q.reshape(n_heads, head_dim)
        k_heads = k.reshape(n_kv_heads, head_dim)
        v_heads = v.reshape(n_kv_heads, head_dim)

        # RoPE
        q_heads, k_heads = rope_cpu(q_heads, k_heads, pos, head_dim,
                                     config.rope_theta)

        # KV cache
        kv.append(layer_idx, k_heads[np.newaxis], v_heads[np.newaxis])

        # GQA attention
        cached_k, cached_v = kv.get(layer_idx)
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

        return attn_output.astype(np.float16)

    def forward_token(token_id, pos):
        """Forward one token through the cross-layer fused pipeline."""
        # Embedding
        x_fp16 = model.embed_tokens[token_id].astype(np.float16)

        # === Layer 0: standalone pre_attn ===
        pre_result = ct_models['pre_attn_0'].predict({
            'x': x_fp16.reshape(1, dim, 1, 1).astype(np.float32)
        })
        qkv_0 = list(pre_result.values())[0].flatten().astype(np.float16)

        # CPU attention layer 0
        attn_out_0 = do_attention_cpu(qkv_0, 0, pos)

        # === Cross-layer dispatches: layers 0->1, 1->2, ..., 14->15 ===
        attn_out = attn_out_0
        x_residual = x_fp16  # residual for layer 0

        for i in range(n_layers - 1):  # i = 0..14
            key = f'cross_{i}_{i+1}'
            result = ct_models[key].predict({
                'attn_out': attn_out.reshape(1, dim, 1, 1).astype(np.float32),
                'x': x_residual.reshape(1, dim, 1, 1).astype(np.float32),
            })

            # Parse 2 outputs: qkv [3072] and x_next [2048]
            result_keys = sorted(result.keys())
            out_arrays = {}
            for k in result_keys:
                arr = result[k].flatten()
                out_arrays[len(arr)] = arr

            qkv_next = out_arrays[dim + 2 * (n_kv_heads * head_dim)].astype(np.float16)
            x_next = out_arrays[dim].astype(np.float16)

            # CPU attention for layer i+1
            attn_out = do_attention_cpu(qkv_next, i + 1, pos)
            x_residual = x_next

        # === Layer 15: standalone post_attn ===
        last = n_layers - 1
        post_result = ct_models[f'post_attn_{last}'].predict({
            'attn_out': attn_out.reshape(1, dim, 1, 1).astype(np.float32),
            'x': x_residual.reshape(1, dim, 1, 1).astype(np.float32),
        })
        x_fp16 = list(post_result.values())[0].flatten().astype(np.float16)

        # Final RMSNorm (CPU)
        x_norm = rms_norm_cpu(x_fp16, model.norm_weight, config.rms_norm_eps)

        # lm_head (8 chunked ANE dispatches)
        logit_chunks = []
        n_chunks = config.vocab_size // 16032
        if config.vocab_size % 16032 != 0:
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

    # First generated token (not timed)
    next_tok = forward_token(prompt_tokens[-1], len(prompt_tokens) - 1)
    generated.append(next_tok)

    # Generation loop (timed)
    t_start = time.perf_counter()
    for step in range(max_new_tokens - 1):
        pos = len(generated) - 1
        next_tok = forward_token(next_tok, pos)
        generated.append(next_tok)
    gen_time = time.perf_counter() - t_start

    return generated, gen_time


# ===================================================================
# Reference: generate with run_llama_fused.py's 40-dispatch pipeline
# ===================================================================

def generate_reference_40d(model: LlamaModel, prompt_tokens: list,
                           max_new_tokens: int = 10):
    """Run the existing 40-dispatch fused pipeline for comparison.

    Imports and calls the build/generate functions from run_llama_fused.py.
    """
    # Import the 40-dispatch pipeline
    sys.path.insert(0, os.path.dirname(__file__))
    from run_llama_fused import build_all_models as build_40d, generate_fused

    ct_models_40d, dispatch_mode_40d = build_40d(model)
    gen_tokens, gen_time = generate_fused(model, ct_models_40d, dispatch_mode_40d,
                                           prompt_tokens, max_new_tokens)
    return gen_tokens, gen_time


# ===================================================================
# Main
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Cross-layer fusion benchmark for Llama-3.2-1B')
    parser.add_argument('--prompt', default='The capital of France is',
                        help='Input prompt')
    parser.add_argument('--tokens', type=int, default=10,
                        help='Number of tokens to generate')
    parser.add_argument('--skip-build', action='store_true',
                        help='Use cached models')
    parser.add_argument('--skip-verify', action='store_true',
                        help='Skip per-model correctness verification')
    parser.add_argument('--compare', action='store_true',
                        help='Also run 40-dispatch baseline for A/B comparison')
    args = parser.parse_args()

    print("=" * 70)
    print("CROSS-LAYER FUSION BENCHMARK — LLAMA-3.2-1B")
    print("post_attn_i + pre_attn_(i+1) fused into single dispatch")
    print("Target: 40 -> 25 dispatches/token (37.5% reduction)")
    print("=" * 70)

    # Tokenize
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        prompt_tokens = tokenizer.encode(args.prompt)
        has_tokenizer = True
    except Exception:
        # Fallback: hardcoded "The capital of France is"
        prompt_tokens = [791, 6864, 315, 9822, 374]
        has_tokenizer = False

    # Load model
    print(f"\n[1/6] Loading Llama-3.2-1B from {MODEL_PATH}...")
    t0 = time.time()
    model_path = MODEL_PATH
    if not os.path.exists(model_path):
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

    # Build 25-dispatch cross-layer models
    print(f"\n[2/6] Building cross-layer fused models...")
    t0 = time.time()
    ct_models, cross_keys = build_all_models(model)
    print(f"  Build time: {time.time()-t0:.1f}s")

    # Per-model correctness verification
    if not args.skip_verify:
        print(f"\n[3/6] Verifying cross-layer model correctness...")
        all_pass = verify_all_cross_layer(model, ct_models)
        if not all_pass:
            print("\n  CORRECTNESS FAILURE — aborting generation")
            sys.exit(1)
        print(f"  All cross-layer models PASS")
    else:
        print(f"\n[3/6] Skipping per-model verification (--skip-verify)")

    # Generate with 25-dispatch pipeline
    print(f"\n[4/6] Generating {args.tokens} tokens (25 dispatches/token)...")
    print(f"  Prompt: '{args.prompt}'")
    gen_tokens_25d, gen_time_25d = generate_cross_layer(
        model, ct_models, cross_keys, prompt_tokens, args.tokens)
    tok_s_25d = (args.tokens - 1) / gen_time_25d if gen_time_25d > 0 else 0

    if has_tokenizer:
        output_text = tokenizer.decode(gen_tokens_25d, skip_special_tokens=True)
    else:
        output_text = f"[token IDs: {gen_tokens_25d}]"

    print(f"  Output: {output_text}")
    print(f"  Time: {gen_time_25d:.3f}s for {args.tokens-1} tokens")
    print(f"  Speed: {tok_s_25d:.1f} tok/s (25 dispatches)")

    # Generate reference with 40-dispatch pipeline (kill test)
    if args.compare:
        print(f"\n[5/6] Running 40-dispatch baseline for comparison...")
        gen_tokens_40d, gen_time_40d = generate_reference_40d(
            model, prompt_tokens, args.tokens)
        tok_s_40d = (args.tokens - 1) / gen_time_40d if gen_time_40d > 0 else 0

        if has_tokenizer:
            ref_text = tokenizer.decode(gen_tokens_40d, skip_special_tokens=True)
        else:
            ref_text = f"[token IDs: {gen_tokens_40d}]"

        print(f"  Output: {ref_text}")
        print(f"  Speed: {tok_s_40d:.1f} tok/s (40 dispatches)")

        # Kill test: token-by-token match
        print(f"\n[6/6] Kill test: 25-dispatch vs 40-dispatch token match...")
        gen_25 = gen_tokens_25d[len(prompt_tokens):]
        gen_40 = gen_tokens_40d[len(prompt_tokens):]
        match_count = 0
        for i, (t25, t40) in enumerate(zip(gen_25, gen_40)):
            if t25 == t40:
                match_count += 1
            else:
                print(f"  Token {i}: 25d={t25} 40d={t40} MISMATCH")
        total = min(len(gen_25), len(gen_40))
        print(f"  Kill test: {match_count}/{total} tokens match")

        # Summary
        print(f"\n{'='*70}")
        print(f"RESULTS")
        print(f"{'='*70}")
        print(f"  40-dispatch (baseline): {tok_s_40d:.1f} tok/s")
        print(f"  25-dispatch (cross-layer): {tok_s_25d:.1f} tok/s")
        if tok_s_40d > 0:
            speedup = tok_s_25d / tok_s_40d
            print(f"  Speedup: {speedup:.2f}x")
        print(f"  Token match: {match_count}/{total}")
        print(f"  Dispatch reduction: 40 -> 25 (-37.5%)")
        if tok_s_40d > 0:
            # Estimate effective bandwidth improvement
            # At 40d: 41 tok/s baseline = 103 GB/s effective
            # Bandwidth = tok/s * bytes_per_token
            # bytes_per_token = sum of all weight matrices read per token
            bw_40d = tok_s_40d * (103 / 41.0)  # scale from known 41 tok/s = 103 GB/s
            bw_25d = tok_s_25d * (103 / 41.0)
            print(f"  Effective bandwidth: {bw_40d:.0f} -> {bw_25d:.0f} GB/s "
                  f"(est. from 41 tok/s = 103 GB/s baseline)")
    else:
        print(f"\n[5/6] Skipping 40-dispatch comparison (use --compare)")
        print(f"\n[6/6] Summary")

    print(f"\n{'='*70}")
    print(f"CROSS-LAYER FUSION SUMMARY")
    print(f"{'='*70}")
    print(f"  Architecture: post_attn_i + pre_attn_(i+1) in single dispatch")
    print(f"  Dispatch count: 25 (was 40)")
    print(f"    - 1 standalone pre_attn_0")
    print(f"    - 15 cross-layer (post_i + pre_(i+1))")
    print(f"    - 1 standalone post_attn_15")
    print(f"    - 8 lm_head chunks")
    print(f"  Speed: {tok_s_25d:.1f} tok/s")
    print(f"  Hypothesis: SRAM pipelining within cross-layer dispatch keeps")
    print(f"  dim=2048 intermediates in 16MB ANE SRAM across what were")
    print(f"  previously 2 separate DRAM-flushing dispatch boundaries.")
    print(f"  Per GPT-2 data: 37d=34 GB/s, 1d=138 GB/s (4.06x).")
    print(f"  25d should approach ~120-140 GB/s effective for the fused ops.")


if __name__ == "__main__":
    main()
