#!/usr/bin/env python3
"""
First token on open ANE stack.

Loads GPT-2 from safetensors, compiles every layer via ane-compiler,
dispatches via ane-dispatch (ane_eval_binary), generates one correct token.
Zero Apple frameworks in the inference path (only aned for .hwx compilation).

For seq_len=1 (single token), attention simplifies:
  softmax(Q@K^T / sqrt(d)) @ V = softmax(scalar) @ V = 1.0 * V = V
  So attention output = O_proj(V_proj(ln1(x))) + x

Full layer: ln1 → V_proj → O_proj → residual → ln2 → fc_up → GELU → fc_down → residual

Bias is fused into .hwx files — ANE computes W@x+b in a single dispatch.
No separate CPU bias-add ops needed (Gate 2 finding).

Copyright 2026 Nick Lo. MIT License.
"""

import os
import sys
import struct
import subprocess
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from compiler import (gen_conv_mlmodelc, gen_softmax_mlmodelc,
                      gen_layernorm_mlmodelc, gen_fused_ffn_mlmodelc)
from model_loader import GPT2Model, GPT2Config, gen_gelu_mlmodelc

# Paths
MODEL_PATH = os.path.expanduser(
    "~/.cache/huggingface/hub/models--openai-community--gpt2/"
    "snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e/model.safetensors"
)
EVAL_BINARY = os.path.join(os.path.dirname(__file__), 'tests', 'ane_eval_binary')
BUILD_DIR = '/tmp/gpt2_first_token'


def dispatch_ane(mlmodelc_path: str, input_fp16: np.ndarray,
                 in_ch: int, out_ch: int) -> np.ndarray:
    """Dispatch a single op on ANE via ane_eval_binary.

    Args:
        mlmodelc_path: path to compiled .mlmodelc
        input_fp16: input data as FP16 numpy array
        in_ch: input channel count
        out_ch: output channel count

    Returns:
        Output as FP16 numpy array
    """
    input_bytes = input_fp16.astype(np.float16).tobytes()
    result = subprocess.run(
        [EVAL_BINARY, mlmodelc_path, str(in_ch), str(out_ch)],
        input=input_bytes, capture_output=True, timeout=30,
    )
    if b'OK' not in result.stderr:
        raise RuntimeError(f"ANE dispatch failed for {mlmodelc_path}: {result.stderr.decode()}")
    return np.frombuffer(result.stdout, dtype=np.float16)


def layernorm_cpu(x: np.ndarray, weight: np.ndarray, bias: np.ndarray,
                  eps: float = 1e-5) -> np.ndarray:
    """CPU LayerNorm (FP32 for accuracy, output as FP16)."""
    x_f32 = x.astype(np.float32)
    mean = x_f32.mean()
    var = ((x_f32 - mean) ** 2).mean()
    normed = (x_f32 - mean) / np.sqrt(var + eps)
    result = normed * weight.astype(np.float32) + bias.astype(np.float32)
    return result.astype(np.float16)


def gelu_new_cpu(x: np.ndarray) -> np.ndarray:
    """CPU gelu_new (GPT-2's GELU variant)."""
    x_f32 = x.astype(np.float32)
    result = 0.5 * x_f32 * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x_f32 + 0.044715 * x_f32**3)))
    return result.astype(np.float16)


def compile_all_ops(model: GPT2Model, build_dir: str, mode: str = 'fused',
                    trace=None):
    """Pre-compile all .mlmodelc bundles for the full model.

    Args:
        mode: 'fused'  — QKV combined + fused FFN with ANE GELU (37 dispatches)
              'exact'  — QKV combined + separate FFN with CPU GELU (49 dispatches)
              'unfused' — separate Q/K/V + separate FFN (73 dispatches)
        trace: optional TraceLogger for verbose output
    """
    os.makedirs(build_dir, exist_ok=True)
    config = model.config
    dim = config.n_embd
    ffn_dim = config.n_inner
    fuse_qkv = mode in ('fused', 'exact')
    fuse_ffn = mode == 'fused'

    compiled = {}
    total_ops = 0

    # Count total ops for trace header
    if trace:
        n_per_layer = (1 if fuse_qkv else 3) + 1 + (1 if fuse_ffn else 2)
        n_total = n_per_layer * config.n_layer + 1  # +1 for lm_head
        trace.trace_compile_start(n_total)

    for i in range(config.n_layer):
        layer_dir = os.path.join(build_dir, f'layer_{i}')
        os.makedirs(layer_dir, exist_ok=True)
        L = model.layers[i]

        if fuse_qkv:
            path = os.path.join(layer_dir, 'qkv_proj.mlmodelc')
            if not os.path.exists(path):
                W_qkv = L.c_attn_weight.T.copy()
                gen_conv_mlmodelc(path, W_qkv.astype(np.float32), dim, 2304,
                                  bias=L.c_attn_bias.astype(np.float32), name='qkv_proj')
            compiled[f'L{i}_qkv_proj'] = (path, dim, 2304)
            if trace:
                trace.trace_compile_op(f'L{i}_qkv_proj', dim, 2304, 'conv', 'bias fused')
            total_ops += 1
        else:
            for name, weight, bias in [
                ('q_proj', L.W_q, L.bias_q),
                ('k_proj', L.W_k, L.bias_k),
                ('v_proj', L.W_v, L.bias_v),
            ]:
                path = os.path.join(layer_dir, f'{name}.mlmodelc')
                if not os.path.exists(path):
                    gen_conv_mlmodelc(path, weight.astype(np.float32), dim, dim,
                                      bias=bias.astype(np.float32), name=name)
                compiled[f'L{i}_{name}'] = (path, dim, dim)
                if trace:
                    trace.trace_compile_op(f'L{i}_{name}', dim, dim, 'conv', 'bias fused')
                total_ops += 1

        path = os.path.join(layer_dir, 'o_proj.mlmodelc')
        if not os.path.exists(path):
            gen_conv_mlmodelc(path, L.W_o.astype(np.float32), dim, dim,
                              bias=L.c_proj_bias.astype(np.float32), name='o_proj')
        compiled[f'L{i}_o_proj'] = (path, dim, dim)
        if trace:
            trace.trace_compile_op(f'L{i}_o_proj', dim, dim, 'conv', 'bias fused')
        total_ops += 1

        if fuse_ffn:
            path = os.path.join(layer_dir, 'fused_ffn.mlmodelc')
            if not os.path.exists(path):
                gen_fused_ffn_mlmodelc(
                    path,
                    W_up=L.W_fc.astype(np.float32),
                    bias_up=L.c_fc_bias.astype(np.float32),
                    W_down=L.W_fc_down.astype(np.float32),
                    bias_down=L.c_proj_ffn_bias.astype(np.float32),
                    in_ch=dim, hidden_ch=ffn_dim, out_ch=dim,
                )
            compiled[f'L{i}_fused_ffn'] = (path, dim, dim)
            if trace:
                trace.trace_compile_op(f'L{i}_fused_ffn', dim, dim,
                                       f'conv+GELU+conv', f'{dim}->{ffn_dim}->{dim}, hardware PWL')
            total_ops += 1
        else:
            path = os.path.join(layer_dir, 'fc_up.mlmodelc')
            if not os.path.exists(path):
                gen_conv_mlmodelc(path, L.W_fc.astype(np.float32), dim, ffn_dim,
                                  bias=L.c_fc_bias.astype(np.float32), name='fc_up')
            compiled[f'L{i}_fc_up'] = (path, dim, ffn_dim)
            if trace:
                trace.trace_compile_op(f'L{i}_fc_up', dim, ffn_dim, 'conv', 'bias fused')
            total_ops += 1

            path = os.path.join(layer_dir, 'fc_down.mlmodelc')
            if not os.path.exists(path):
                gen_conv_mlmodelc(path, L.W_fc_down.astype(np.float32), ffn_dim, dim,
                                  bias=L.c_proj_ffn_bias.astype(np.float32), name='fc_down')
            compiled[f'L{i}_fc_down'] = (path, ffn_dim, dim)
            if trace:
                trace.trace_compile_op(f'L{i}_fc_down', ffn_dim, dim, 'conv', 'bias fused')
            total_ops += 1

    # lm_head (tied to embedding weights)
    lm_head_path = os.path.join(build_dir, 'lm_head.mlmodelc')
    if not os.path.exists(lm_head_path):
        gen_conv_mlmodelc(lm_head_path, model.wte.astype(np.float32), dim, config.vocab_size, name='lm_head')
    compiled['lm_head'] = (lm_head_path, dim, config.vocab_size)
    if trace:
        trace.trace_compile_op('lm_head', dim, config.vocab_size, 'conv', 'no bias')
    total_ops += 1

    return compiled


def forward_layer(layer_idx: int, x: np.ndarray, model: GPT2Model,
                  compiled: dict, use_ane_conv: bool = True) -> np.ndarray:
    """Forward pass through one GPT-2 layer.

    For seq_len=1: attention = O_proj(V_proj(ln1(x))) + x
    (softmax of single element = 1.0, so output = V directly)
    Bias is fused into ANE dispatch — no separate CPU bias-add needed.
    """
    L = model.layers[layer_idx]
    dim = model.config.n_embd
    pfx = f'L{layer_idx}'

    # 1. LayerNorm 1 (CPU — includes weight/bias which ANE layernorm doesn't handle)
    ln1_out = layernorm_cpu(x, L.ln_1_weight, L.ln_1_bias, model.config.layer_norm_epsilon)

    # 2. Attention (seq_len=1 shortcut)
    if use_ane_conv:
        # V projection on ANE (bias fused into .hwx)
        path, ic, oc = compiled[f'{pfx}_v_proj']
        v = dispatch_ane(path, ln1_out, ic, oc)

        # O projection on ANE (bias fused into .hwx)
        path, ic, oc = compiled[f'{pfx}_o_proj']
        attn_out = dispatch_ane(path, v, ic, oc)
    else:
        # CPU fallback
        v = (L.W_v.astype(np.float16) @ ln1_out).astype(np.float16) + L.bias_v.astype(np.float16)
        attn_out = (L.W_o.astype(np.float16) @ v).astype(np.float16) + L.c_proj_bias.astype(np.float16)

    # 3. Residual 1
    r1 = (x.astype(np.float32) + attn_out.astype(np.float32)).astype(np.float16)

    # 4. LayerNorm 2
    ln2_out = layernorm_cpu(r1, L.ln_2_weight, L.ln_2_bias, model.config.layer_norm_epsilon)

    # 5. FFN up (bias fused into .hwx)
    if use_ane_conv:
        path, ic, oc = compiled[f'{pfx}_fc_up']
        fc_up = dispatch_ane(path, ln2_out, ic, oc)
    else:
        fc_up = (L.W_fc.astype(np.float16) @ ln2_out).astype(np.float16) + L.c_fc_bias.astype(np.float16)

    # 6. GELU (CPU — gelu_new has tanh approximation, ANE has erf-GELU)
    gelu_out = gelu_new_cpu(fc_up)

    # 7. FFN down (bias fused into .hwx)
    if use_ane_conv:
        path, ic, oc = compiled[f'{pfx}_fc_down']
        fc_down = dispatch_ane(path, gelu_out, ic, oc)
    else:
        fc_down = (L.W_fc_down.astype(np.float16) @ gelu_out).astype(np.float16) + L.c_proj_ffn_bias.astype(np.float16)

    # 8. Residual 2
    output = (r1.astype(np.float32) + fc_down.astype(np.float32)).astype(np.float16)
    return output


def forward_pass(token_id: int, model: GPT2Model, compiled: dict,
                 use_ane: bool = True) -> np.ndarray:
    """Full forward pass: embed → 12 layers → ln_f → lm_head → logits."""
    config = model.config

    # 1. Embedding: token + position
    token_emb = model.wte[token_id].astype(np.float16)  # [768]
    pos_emb = model.wpe[0].astype(np.float16)            # position 0
    x = (token_emb.astype(np.float32) + pos_emb.astype(np.float32)).astype(np.float16)

    # 2. 12 transformer layers
    for i in range(config.n_layer):
        x = forward_layer(i, x, model, compiled, use_ane_conv=use_ane)
        if i == 0:
            print(f"  Layer 0 output (first 8): {x[:8]}")

    # 3. Final LayerNorm
    x = layernorm_cpu(x, model.ln_f_weight, model.ln_f_bias, config.layer_norm_epsilon)

    # 4. lm_head (language model head)
    if use_ane:
        path, ic, oc = compiled['lm_head']
        logits = dispatch_ane(path, x, ic, oc)
    else:
        logits = (model.wte.astype(np.float16) @ x).astype(np.float16)

    return logits


def pytorch_reference(token_id: int) -> np.ndarray:
    """PyTorch reference forward pass for verification."""
    import torch
    from transformers import GPT2LMHeadModel

    model = GPT2LMHeadModel.from_pretrained('openai-community/gpt2')
    model.eval()

    with torch.no_grad():
        input_ids = torch.tensor([[token_id]])
        outputs = model(input_ids)
        logits = outputs.logits[0, 0]  # [vocab_size]

    return logits.numpy()


def main():
    print("=" * 60)
    print("FIRST TOKEN ON OPEN ANE STACK")
    print("GPT-2 117M — safetensors → ane-compiler → ane-dispatch")
    print("=" * 60)

    # Load model
    print("\n[1/5] Loading GPT-2 from safetensors...")
    t0 = time.time()
    model = GPT2Model.from_safetensors(MODEL_PATH)
    print(f"  Loaded in {time.time()-t0:.2f}s")

    # Compile all ops
    print("\n[2/5] Compiling .mlmodelc bundles...")
    t0 = time.time()
    compiled = compile_all_ops(model, BUILD_DIR)
    print(f"  Compiled in {time.time()-t0:.2f}s")

    # CPU reference first
    print("\n[3/5] CPU reference forward pass...")
    token_id = 0  # Use token 0 ("<|endoftext|>" in GPT-2)
    t0 = time.time()
    cpu_logits = forward_pass(token_id, model, compiled, use_ane=False)
    print(f"  CPU forward in {time.time()-t0:.2f}s")
    cpu_top5 = np.argsort(cpu_logits.astype(np.float32))[-5:][::-1]
    print(f"  CPU top-5 tokens: {cpu_top5}")
    print(f"  CPU top-1 logit: {float(cpu_logits[cpu_top5[0]]):.4f}")

    # ANE forward pass
    print("\n[4/5] ANE forward pass...")
    t0 = time.time()
    ane_logits = forward_pass(token_id, model, compiled, use_ane=True)
    print(f"  ANE forward in {time.time()-t0:.2f}s")
    ane_top5 = np.argsort(ane_logits.astype(np.float32))[-5:][::-1]
    print(f"  ANE top-5 tokens: {ane_top5}")
    print(f"  ANE top-1 logit: {float(ane_logits[ane_top5[0]]):.4f}")

    # Compare
    print("\n[5/5] Verification...")
    diff = np.abs(ane_logits.astype(np.float32) - cpu_logits.astype(np.float32))
    max_diff = diff.max()
    mean_diff = diff.mean()
    top1_match = ane_top5[0] == cpu_top5[0]

    print(f"  Max logit diff:  {max_diff:.4f}")
    print(f"  Mean logit diff: {mean_diff:.4f}")
    print(f"  Top-1 match:     {'YES' if top1_match else 'NO'}")
    print(f"  ANE top-1:       token {ane_top5[0]}")
    print(f"  CPU top-1:       token {cpu_top5[0]}")

    # PyTorch reference
    print("\n[Bonus] PyTorch reference...")
    try:
        pt_logits = pytorch_reference(token_id)
        pt_top5 = np.argsort(pt_logits)[-5:][::-1]
        pt_match = ane_top5[0] == pt_top5[0]
        pt_diff = np.abs(ane_logits.astype(np.float32) - pt_logits.astype(np.float32))
        print(f"  PyTorch top-5: {pt_top5}")
        print(f"  ANE vs PyTorch top-1 match: {'YES' if pt_match else 'NO'}")
        print(f"  ANE vs PyTorch max diff: {pt_diff.max():.4f}")
    except Exception as e:
        print(f"  PyTorch reference failed: {e}")

    # Final verdict
    print("\n" + "=" * 60)
    if top1_match:
        print("*** FIRST TOKEN: PASS ***")
        print(f"Token {ane_top5[0]} generated correctly on ANE")
    else:
        print("*** FIRST TOKEN: INVESTIGATING ***")
        print(f"ANE predicted {ane_top5[0]}, CPU predicted {cpu_top5[0]}")
    print("=" * 60)


if __name__ == "__main__":
    main()
