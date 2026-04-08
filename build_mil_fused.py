#!/usr/bin/env python3
"""
Build 25-dispatch GPT-2 fused MIL IR models.

For each of the 12 GPT-2 layers, builds a 2-input MIL IR model that fuses:
  O_proj(attn_out) + residual + LN2 + FFN_up + GELU + FFN_down + residual

This replaces 4 CPU ops + 2 ANE dispatches with 1 ANE dispatch.
37 dispatches -> 25 dispatches per token.

Architecture:
  Input 1: attn_out [768] — attention output (from CPU attention)
  Input 2: x [768] — skip connection (pre-attention hidden state)
  Output: residual_2 [768] — final hidden state for this layer

Uses coremltools MIL builder for weight embedding + ct.convert for compilation.

Copyright 2026 Nick Lo. MIT License.
"""

import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from model_loader import GPT2Model

MODEL_PATH = os.path.expanduser(
    "~/.cache/huggingface/hub/models--openai-community--gpt2/"
    "snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e/model.safetensors"
)

BUILD_DIR = '/tmp/gpt2_mil_fused'


def build_fused_post_attn(layer, layer_idx, save_dir):
    """Build a fused post-attention MIL IR model for one GPT-2 layer.

    Fuses: O_proj + residual_1 + LN2 + FFN_up + GELU + FFN_down + residual_2

    Args:
        layer: GPT2Layer with weights
        layer_idx: layer index (for naming)
        save_dir: directory to save .mlpackage

    Returns:
        (mlpackage_path, correctness_verified)
    """
    import coremltools as ct
    from coremltools.converters.mil import Builder as mb
    from coremltools.converters.mil.mil import types

    # Extract weights (already in [out, in] format from properties)
    W_o = layer.W_o.astype(np.float16)        # [768, 768]
    b_o = layer.c_proj_bias.astype(np.float16) # [768]

    ln2_w = layer.ln_2_weight.astype(np.float16)  # [768]
    ln2_b = layer.ln_2_bias.astype(np.float16)    # [768]

    W_up = layer.W_fc.astype(np.float16)           # [3072, 768]
    b_up = layer.c_fc_bias.astype(np.float16)      # [3072]

    W_down = layer.W_fc_down.astype(np.float16)    # [768, 3072]
    b_down = layer.c_proj_ffn_bias.astype(np.float16)  # [768]

    epsilon = np.float16(1e-5)

    @mb.program(input_specs=[
        mb.TensorSpec(shape=(1, 768, 1, 1), dtype=types.fp16),  # attn_out
        mb.TensorSpec(shape=(1, 768, 1, 1), dtype=types.fp16),  # x (skip)
    ])
    def fused_post_attn(attn_out, x):
        # Reshape inputs from 4D ANE format to 2D for linear
        attn_flat = mb.reshape(x=attn_out, shape=[1, 768])
        x_flat = mb.reshape(x=x, shape=[1, 768])

        # O projection: linear 768->768 + bias
        o_out = mb.linear(x=attn_flat, weight=mb.const(val=W_o), bias=mb.const(val=b_o))

        # Residual 1: O_proj_out + x
        r1 = mb.add(x=o_out, y=x_flat)

        # LayerNorm 2: layer_norm(r1)
        ln2_out = mb.layer_norm(
            x=r1, axes=[1],
            gamma=mb.const(val=ln2_w),
            beta=mb.const(val=ln2_b),
            epsilon=epsilon
        )

        # FFN up: linear 768->3072 + bias
        ffn_up = mb.linear(x=ln2_out, weight=mb.const(val=W_up), bias=mb.const(val=b_up))

        # GELU (tanh approximation — GPT-2's gelu_new)
        gelu_out = mb.gelu(x=ffn_up, mode="TANH_APPROXIMATION")

        # FFN down: linear 3072->768 + bias
        ffn_down = mb.linear(x=gelu_out, weight=mb.const(val=W_down), bias=mb.const(val=b_down))

        # Residual 2: FFN_down_out + r1
        output = mb.add(x=ffn_down, y=r1)

        # Reshape back to 4D ANE format
        output_4d = mb.reshape(x=output, shape=[1, 768, 1, 1])

        return output_4d

    # Convert to CoreML
    mlpackage_path = os.path.join(save_dir, f'layer_{layer_idx}_post_attn.mlpackage')

    model = ct.convert(
        fused_post_attn,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.iOS18,
    )
    model.save(mlpackage_path)

    return mlpackage_path


def build_fused_pre_attn(layer, layer_idx, save_dir):
    """Build a fused pre-attention MIL IR model (LN1 + QKV_proj).

    Single input: x [768]
    Output: qkv [2304]

    If this compiles on ANE, we eliminate the CPU LN1 dispatch.
    """
    import coremltools as ct
    from coremltools.converters.mil import Builder as mb
    from coremltools.converters.mil.mil import types

    ln1_w = layer.ln_1_weight.astype(np.float16)
    ln1_b = layer.ln_1_bias.astype(np.float16)
    W_qkv = layer.c_attn_weight.T.copy().astype(np.float16)  # [2304, 768]
    b_qkv = layer.c_attn_bias.astype(np.float16)  # [2304]
    epsilon = np.float16(1e-5)

    @mb.program(input_specs=[
        mb.TensorSpec(shape=(1, 768, 1, 1), dtype=types.fp16),
    ])
    def fused_pre_attn(x):
        x_flat = mb.reshape(x=x, shape=[1, 768])

        # LayerNorm 1
        ln1_out = mb.layer_norm(
            x=x_flat, axes=[1],
            gamma=mb.const(val=ln1_w),
            beta=mb.const(val=ln1_b),
            epsilon=epsilon
        )

        # QKV projection
        qkv = mb.linear(x=ln1_out, weight=mb.const(val=W_qkv), bias=mb.const(val=b_qkv))

        # Reshape to 4D ANE format
        qkv_4d = mb.reshape(x=qkv, shape=[1, 2304, 1, 1])

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
        print(f"  LN1+QKV fusion FAILED for layer {layer_idx}: {e}")
        return None


def verify_fused_post_attn(layer, mlpackage_path, layer_idx):
    """Verify fused model output matches sequential computation."""
    import coremltools as ct

    np.random.seed(42 + layer_idx)
    attn_out = np.random.randn(768).astype(np.float16)
    x = np.random.randn(768).astype(np.float16)

    # Sequential reference computation (FP32 for precision)
    W_o = layer.W_o.astype(np.float32)
    b_o = layer.c_proj_bias.astype(np.float32)
    ln2_w = layer.ln_2_weight.astype(np.float32)
    ln2_b = layer.ln_2_bias.astype(np.float32)
    W_up = layer.W_fc.astype(np.float32)
    b_up = layer.c_fc_bias.astype(np.float32)
    W_down = layer.W_fc_down.astype(np.float32)
    b_down = layer.c_proj_ffn_bias.astype(np.float32)

    # O projection
    o_out = attn_out.astype(np.float32) @ W_o.T + b_o

    # Residual 1
    r1 = x.astype(np.float32) + o_out

    # LN2
    mean = r1.mean()
    var = ((r1 - mean) ** 2).mean()
    ln2_out = (r1 - mean) / np.sqrt(var + 1e-5)
    ln2_out = ln2_out * ln2_w + ln2_b

    # FFN up
    ffn_up = ln2_out @ W_up.T + b_up

    # GELU (tanh approx)
    ffn_gelu = 0.5 * ffn_up * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (ffn_up + 0.044715 * ffn_up**3)))

    # FFN down
    ffn_down = ffn_gelu @ W_down.T + b_down

    # Residual 2
    ref_output = (r1 + ffn_down).astype(np.float16)

    # Run fused model
    model = ct.models.MLModel(mlpackage_path, compute_units=ct.ComputeUnit.CPU_AND_NE)

    result = model.predict({
        'attn_out': attn_out.reshape(1, 768, 1, 1),
        'x': x.reshape(1, 768, 1, 1),
    })

    out_key = list(result.keys())[0]
    fused_output = result[out_key].flatten().astype(np.float16)

    max_diff = np.max(np.abs(fused_output.astype(np.float32) - ref_output.astype(np.float32)))
    return max_diff, fused_output[:5], ref_output[:5]


def verify_fused_pre_attn(layer, mlpackage_path, layer_idx):
    """Verify fused LN1+QKV model output."""
    import coremltools as ct

    np.random.seed(42 + layer_idx)
    x = np.random.randn(768).astype(np.float16)

    # Reference
    x_f32 = x.astype(np.float32)
    mean = x_f32.mean()
    var = ((x_f32 - mean) ** 2).mean()
    ln1_out = (x_f32 - mean) / np.sqrt(var + 1e-5)
    ln1_out = ln1_out * layer.ln_1_weight.astype(np.float32) + layer.ln_1_bias.astype(np.float32)

    W_qkv = layer.c_attn_weight.T.copy().astype(np.float32)
    b_qkv = layer.c_attn_bias.astype(np.float32)
    ref_qkv = (ln1_out @ W_qkv.T + b_qkv).astype(np.float16)

    # Run fused model
    model = ct.models.MLModel(mlpackage_path, compute_units=ct.ComputeUnit.CPU_AND_NE)
    result = model.predict({'x': x.reshape(1, 768, 1, 1)})
    out_key = list(result.keys())[0]
    fused_qkv = result[out_key].flatten().astype(np.float16)

    max_diff = np.max(np.abs(fused_qkv.astype(np.float32) - ref_qkv.astype(np.float32)))
    return max_diff, fused_qkv[:5], ref_qkv[:5]


def main():
    print("=" * 70)
    print("BUILD MIL FUSED MODELS — 25-DISPATCH GPT-2")
    print("Fusing: O_proj + residual + LN2 + FFN into single 2-input dispatch")
    print("=" * 70)

    # Load model
    print("\n[1/4] Loading GPT-2 from safetensors...")
    t0 = time.time()
    model = GPT2Model.from_safetensors(MODEL_PATH)
    print(f"  Loaded in {time.time()-t0:.2f}s")

    os.makedirs(BUILD_DIR, exist_ok=True)

    # Phase 1: Build fused post-attention models
    print("\n[2/4] Building fused post-attention models (12 layers)...")
    post_attn_paths = {}
    for i in range(model.config.n_layer):
        t0 = time.time()
        path = build_fused_post_attn(model.layers[i], i, BUILD_DIR)
        elapsed = time.time() - t0
        post_attn_paths[i] = path
        print(f"  Layer {i:2d}: built in {elapsed:.1f}s -> {os.path.basename(path)}")

    # Phase 2: Verify correctness
    print("\n[3/4] Verifying correctness (12 layers)...")
    all_pass = True
    for i in range(model.config.n_layer):
        max_diff, fused_5, ref_5 = verify_fused_post_attn(
            model.layers[i], post_attn_paths[i], i)
        status = "PASS" if max_diff < 1.0 else "FAIL"
        if max_diff >= 1.0:
            all_pass = False
        print(f"  Layer {i:2d}: max_diff={max_diff:.4f} [{status}]")
        if i == 0:
            print(f"    Fused:  {[f'{v:.4f}' for v in fused_5]}")
            print(f"    Ref:    {[f'{v:.4f}' for v in ref_5]}")

    # Phase 3: Try fused pre-attention (LN1 + QKV)
    print("\n[4/4] Attempting LN1 + QKV fusion (Phase 2)...")
    pre_attn_path = build_fused_pre_attn(model.layers[0], 0, BUILD_DIR)
    if pre_attn_path:
        max_diff, fused_5, ref_5 = verify_fused_pre_attn(
            model.layers[0], pre_attn_path, 0)
        print(f"  Layer 0 LN1+QKV: max_diff={max_diff:.4f}")
        print(f"    Fused: {[f'{v:.4f}' for v in fused_5]}")
        print(f"    Ref:   {[f'{v:.4f}' for v in ref_5]}")
        if max_diff < 1.0:
            print("  LN1+QKV fusion WORKS — could reduce to ~13 dispatches")
        else:
            print("  LN1+QKV fusion: correctness issue")
    else:
        print("  LN1+QKV fusion: compilation failed (expected — espresso hung at 2304)")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Post-attention fusion: {'ALL PASS' if all_pass else 'SOME FAILED'}")
    print(f"  Models saved to: {BUILD_DIR}")
    if all_pass:
        print(f"  Next: run_25dispatch.py for end-to-end generation")

    return all_pass


if __name__ == "__main__":
    main()
