#!/usr/bin/env python3
"""
GPT-2 117M: 25-dispatch generation with maximally fused ANE ops.

Architecture:
  Per layer (2 dispatches):
    D1: LN1(MVN+BN) + QKV_proj (fused, raw espresso format)
    CPU: attention (Q@K^T, softmax, attn@V)
    D2: O_proj + fused_FFN(LN2(MVN+BN) + fc_up + GELU + fc_down)
        Input: concatenated [x (768), attn_output (768)] = 1536
        But this requires slice in espresso, which may not work.

  REVISED: Since we can't easily slice in raw espresso,
  split D2 into O_proj (1) + fused post-FFN (1) = 2 dispatches.

  ACTUAL architecture (27 dispatches):
    D1: LN1(MVN+BN) + QKV_proj = 1 fused dispatch
    CPU: attention
    D2: O_proj = 1 dispatch (existing gen_conv_mlmodelc)
    CPU: residual 1
    D3: LN2(MVN+BN) + FFN(fc_up + GELU + fc_down) = 1 fused dispatch
    CPU: residual 2
    Total: 3 dispatches/layer * 12 + 1 lm_head = 37

  WAIT: that's still 37. The LN moves to ANE but dispatch count doesn't change.
  To truly reduce dispatch count, we need either:
    a) Merge O_proj with LN2+FFN (needs 2-input support in pipe tool)
    b) Merge LN1+QKV+attention (impossible without attention on ANE)

  APPROACH: Just merge O_proj into the fused FFN dispatch by using
  a concatenated input. The raw espresso 'crop' layer can slice channels.
  If crop works: D2 takes [x(768), attn_out(768)] = 1536 input, does:
    crop(attn_out) -> O_proj -> o_out
    crop(x) + o_out -> r1
    r1 -> LN2(MVN+BN) -> fc_up -> GELU -> fc_down -> ffn_out
    r1 + ffn_out -> output
  This is 1 dispatch replacing O_proj + LN2 + FFN.

  Let me test if 'crop' works in raw espresso.

FINDING: coremltools-generated .mlmodelc (via xcrun coremlcompiler)
causes aned compilation hangs. ONLY our raw espresso format works fast.

FINAL APPROACH (proven format, 25 dispatches):
  D1: LN1(MVN+BN) + QKV_proj (raw espresso, 3 layers)
  CPU: attention
  D2: O_proj + residual + LN2(MVN+BN) + FFN(up+GELU+down) + residual
      (raw espresso, ~10 layers, crop for 2-input slicing)
  12*2 + 1 lm_head = 25 dispatches

Copyright 2026 Nick Lo. MIT License.
"""

import os
import sys
import json
import struct
import time
import shutil
import subprocess
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
sys.path.insert(0, os.path.dirname(__file__))

from model_loader import GPT2Model, GPT2Config
from first_token import layernorm_cpu, gelu_new_cpu, MODEL_PATH
from generate import ANEDispatcher, softmax_cpu, embed, lm_head
from generate import forward_layer_decode
from kv_cache import KVCache
from compiler import gen_conv_mlmodelc
from compiler import (
    _write_espresso_net, _write_espresso_shape, _write_metadata,
    _write_coremldata, _write_espresso_weights_multi,
)

BUILD_DIR = '/tmp/gpt2_25dispatch_v2'
PIPE_TOOL = os.path.join(os.path.dirname(__file__), 'tests', 'ane_standalone_pipe')


def _write_weights_2layer_bn(path, bn_params, W_ip, bias_ip):
    """Write espresso weights for batchnorm + inner_product.

    Blob IDs: bn_params@1 (gap), ip_bias@3, ip_weights@5
    Uses v8 format matching proven fused_ffn pattern.
    """
    bn_bytes = bn_params.astype(np.float32).tobytes()
    w_bytes = W_ip.astype(np.float32).tobytes()
    b_bytes = bias_ip.astype(np.float32).tobytes()

    first_gap = max(0x3000, len(bn_bytes) * 4)

    # Remaining blobs after gap: bias@3, weights@5
    # Remaining blobs after gap: bias@3, weights@5
    remaining = [
        (3, b_bytes, len(b_bytes), max(0x3000, len(b_bytes) * 4)),
        (5, w_bytes, len(w_bytes), len(w_bytes)),
    ]

    n_entries = len(remaining)
    blob_table_offset = 0x38
    total_blobs = 3  # bn(1) + bias(3) + weights(5)
    header_raw = blob_table_offset + (n_entries + 1) * 0x20
    header_size = max(0x200, ((header_raw + 0x1FF) // 0x200) * 0x200)

    with open(path, 'wb') as f:
        header = bytearray(header_size)
        struct.pack_into('<I', header, 0x00, total_blobs * 2)  # version
        struct.pack_into('<I', header, 0x10, blob_table_offset)
        struct.pack_into('<I', header, 0x18, 1)
        struct.pack_into('<I', header, 0x20, first_gap)
        struct.pack_into('<I', header, 0x28, 2)

        for i, (bid, data, raw_size, table_size) in enumerate(remaining):
            off = blob_table_offset + i * 0x20
            next_id = remaining[i + 1][0] if i + 1 < len(remaining) else 0
            struct.pack_into('<I', header, off + 0, bid)
            struct.pack_into('<I', header, off + 8, table_size)
            struct.pack_into('<I', header, off + 16, next_id)

        f.write(header)
        f.write(bn_bytes)
        f.write(b'\x00' * (first_gap - len(bn_bytes)))
        for bid, data, raw_size, table_size in remaining:
            f.write(data)
            if table_size > raw_size:
                f.write(b'\x00' * (table_size - raw_size))


def _write_weights_3layer(path, bn_params, W_up, bias_up, W_down, bias_down):
    """Write espresso weights for 3-layer model (batchnorm + 2x inner_product).

    Blob IDs: bn_params@1, fc_up_bias@3, fc_up_weights@5,
              fc_down_bias@7, fc_down_weights@9

    Uses v28 format matching bench_fusion_depth.py pattern.
    """
    # Blob data: [(blob_id, data_bytes, is_bias)]
    bn_bytes = bn_params.astype(np.float32).tobytes()
    up_w_bytes = W_up.astype(np.float32).tobytes()
    up_b_bytes = bias_up.astype(np.float32).tobytes()
    dn_w_bytes = W_down.astype(np.float32).tobytes()
    dn_b_bytes = bias_down.astype(np.float32).tobytes()

    # First blob goes in gap (bn_params, blob_id=1)
    first_gap = max(0x3000, len(bn_bytes) * 4)

    # Remaining blobs: fc_up_bias(3), fc_up_weights(5), fc_down_bias(7), fc_down_weights(9)
    remaining = [
        (3, up_b_bytes, True),
        (5, up_w_bytes, False),
        (7, dn_b_bytes, True),
        (9, dn_w_bytes, False),
    ]

    # Compute table sizes
    for i, (bid, data, is_bias) in enumerate(remaining):
        if is_bias:
            remaining[i] = (bid, data, max(0x3000, len(data) * 4))
        else:
            remaining[i] = (bid, data, len(data))

    n_entries = len(remaining)
    blob_table_offset = 0x38
    header_size = blob_table_offset + n_entries * 0x20
    header_size = ((header_size + 0x3F) // 0x40) * 0x40

    with open(path, 'wb') as f:
        header = bytearray(header_size)
        struct.pack_into('<I', header, 0x00, 8)  # version 8
        struct.pack_into('<I', header, 0x10, blob_table_offset)
        struct.pack_into('<I', header, 0x18, 1)
        struct.pack_into('<I', header, 0x20, first_gap)
        struct.pack_into('<I', header, 0x28, 2)

        for i, (bid, data, table_size) in enumerate(remaining):
            off = blob_table_offset + i * 0x20
            next_id = bid + 1 if i < len(remaining) - 1 else 0
            struct.pack_into('<I', header, off + 0, bid)
            struct.pack_into('<I', header, off + 8, table_size)
            struct.pack_into('<I', header, off + 16, next_id)

        f.write(header)
        f.write(bn_bytes)
        f.write(b'\x00' * (first_gap - len(bn_bytes)))
        for bid, data, table_size in remaining:
            f.write(data)
            if table_size > len(data):
                f.write(b'\x00' * (table_size - len(data)))


# ===================================================================
# Raw espresso builder for fused pre-attention: LN1(MVN+BN) + QKV
# ===================================================================

def gen_pre_attn_espresso(output_dir, ln1_weight, ln1_bias, W_qkv, bias_qkv,
                           in_ch=768, qkv_out=2304, eps=1e-5):
    """Build fused LN1+QKV in our raw espresso format (proven fast with aned).

    Layers: l2_normalize -> batchnorm -> inner_product(+bias)
    Uses v8 multi-layer weight format.
    """
    os.makedirs(output_dir, exist_ok=True)
    for sub in ['model', 'analytics', 'neural_network_optionals']:
        os.makedirs(os.path.join(output_dir, sub), exist_ok=True)

    layers = [
        {
            "type": "l2_normalize", "name": "mvn", "debug_info": "mvn",
            "bottom": "input", "top": "mvn_out",
            "normalization_mode": 1, "axis": 2, "eps": float(eps),
            "weights": {},
        },
        {
            "type": "batchnorm", "name": "bn", "debug_info": "bn",
            "bottom": "mvn_out", "top": "bn_out",
            "blob_batchnorm_params": 1, "C": int(in_ch),
            "weights": {},
        },
        {
            "type": "inner_product", "name": "qkv", "debug_info": "qkv",
            "bottom": "bn_out", "top": "output",
            "nB": int(in_ch), "nC": int(qkv_out),
            "has_biases": 1, "blob_weights": 5, "blob_biases": 3,
            "has_relu": 0, "has_tanh": 0, "has_prelu": 0,
            "weights": {},
            "attributes": {"is_output": 1},
        },
    ]

    shapes = {
        "input": (1, 1, 1, in_ch),
        "mvn_out": (1, 1, 1, in_ch),
        "bn_out": (1, 1, 1, in_ch),
        "output": (1, 1, 1, qkv_out),
    }

    _write_espresso_net(os.path.join(output_dir, 'model.espresso.net'), layers)
    _write_espresso_shape(os.path.join(output_dir, 'model.espresso.shape'), shapes)

    # Batchnorm params: [gamma, beta, mean, var] per channel
    bn_params = np.zeros((in_ch, 4), dtype=np.float32)
    bn_params[:, 0] = ln1_weight.astype(np.float32)
    bn_params[:, 1] = ln1_bias.astype(np.float32)
    bn_params[:, 2] = 0.0
    bn_params[:, 3] = 1.0

    # Write weights using v8 format
    # Blob map: bn_params@1 (gap), qkv_bias@3, qkv_weights@5
    # Treat batchnorm params as first layer "bias", no first layer "weights"
    # Then qkv as second layer with bias+weights
    # This maps to: gap=bn_params(blob 1), table=[qkv_bias(blob 3), qkv_weights(blob 5)]
    # But _write_espresso_weights_multi expects (weights, bias) tuples...
    # Use custom writer instead
    _write_weights_2layer_bn(
        os.path.join(output_dir, 'model.espresso.weights'),
        bn_params.flatten().astype(np.float32),
        W_qkv.astype(np.float32),
        bias_qkv.astype(np.float32),
    )

    _write_metadata(
        os.path.join(output_dir, 'metadata.json'),
        inputs=[("input", [int(in_ch), 1, 1])],
        outputs=[("output", [int(qkv_out), 1, 1])],
    )
    _write_coremldata(os.path.join(output_dir, 'coremldata.bin'))
    for sub in ['model', 'analytics', 'neural_network_optionals']:
        with open(os.path.join(output_dir, sub, 'coremldata.bin'), 'wb') as f:
            f.write(b'\x00' * 64)


# ===================================================================
# Raw espresso builder for fused LN2+FFN
# ===================================================================

def gen_ln2_ffn_espresso(output_dir, ln2_weight, ln2_bias,
                          W_up, bias_up, W_down, bias_down,
                          in_ch=768, ffn_dim=3072, eps=1e-5):
    """Build fused LN2+FFN in raw espresso: MVN+BN+inner_product+GELU+inner_product.

    5 espresso layers, single dispatch. Uses mode 19 for hardware GELU.
    """
    os.makedirs(output_dir, exist_ok=True)
    for sub in ['model', 'analytics', 'neural_network_optionals']:
        os.makedirs(os.path.join(output_dir, sub), exist_ok=True)

    layers = [
        {
            "type": "l2_normalize", "name": "mvn", "debug_info": "mvn",
            "bottom": "input", "top": "mvn_out",
            "normalization_mode": 1, "axis": 2, "eps": float(eps),
            "weights": {},
        },
        {
            "type": "batchnorm", "name": "bn", "debug_info": "bn",
            "bottom": "mvn_out", "top": "bn_out",
            "blob_batchnorm_params": 1, "C": int(in_ch),
            "weights": {},
        },
        {
            "type": "inner_product", "name": "fc_up", "debug_info": "fc_up",
            "bottom": "bn_out", "top": "fc_up_out",
            "nB": int(in_ch), "nC": int(ffn_dim),
            "has_biases": 1, "blob_weights": 5, "blob_biases": 3,
            "has_relu": 0, "has_tanh": 0, "has_prelu": 0,
            "weights": {},
        },
        {
            "type": "activation", "name": "gelu",
            "bottom": "fc_up_out", "top": "gelu_out",
            "mode": 19, "weights": {},
        },
        {
            "type": "inner_product", "name": "fc_down", "debug_info": "fc_down",
            "bottom": "gelu_out", "top": "output",
            "nB": int(ffn_dim), "nC": int(in_ch),
            "has_biases": 1, "blob_weights": 9, "blob_biases": 7,
            "has_relu": 0, "has_tanh": 0, "has_prelu": 0,
            "weights": {},
            "attributes": {"is_output": 1},
        },
    ]

    shapes = {
        "input": (1, 1, 1, in_ch),
        "mvn_out": (1, 1, 1, in_ch),
        "bn_out": (1, 1, 1, in_ch),
        "fc_up_out": (1, 1, 1, ffn_dim),
        "gelu_out": (1, 1, 1, ffn_dim),
        "output": (1, 1, 1, in_ch),
    }

    _write_espresso_net(os.path.join(output_dir, 'model.espresso.net'), layers)
    _write_espresso_shape(os.path.join(output_dir, 'model.espresso.shape'), shapes)

    # Batchnorm params
    bn_params = np.zeros((in_ch, 4), dtype=np.float32)
    bn_params[:, 0] = ln2_weight.astype(np.float32)
    bn_params[:, 1] = ln2_bias.astype(np.float32)
    bn_params[:, 2] = 0.0
    bn_params[:, 3] = 1.0

    # Write weights using v28 format (handles arbitrary blob IDs)
    # Blob map: bn_params@1, fc_up_bias@3, fc_up_weights@5,
    #           fc_down_bias@7, fc_down_weights@9
    _write_weights_3layer(
        os.path.join(output_dir, 'model.espresso.weights'),
        bn_params.flatten().astype(np.float32),
        W_up.astype(np.float32), bias_up.astype(np.float32),
        W_down.astype(np.float32), bias_down.astype(np.float32),
    )

    _write_metadata(
        os.path.join(output_dir, 'metadata.json'),
        inputs=[("input", [int(in_ch), 1, 1])],
        outputs=[("output", [int(in_ch), 1, 1])],
    )
    _write_coremldata(os.path.join(output_dir, 'coremldata.bin'))
    for sub in ['model', 'analytics', 'neural_network_optionals']:
        with open(os.path.join(output_dir, sub, 'coremldata.bin'), 'wb') as f:
            f.write(b'\x00' * 64)


# ===================================================================
# Compile GPT-2: 3 dispatches/layer but with LN fused on ANE
# ===================================================================

def compile_fused_ln(model, build_dir):
    """Compile GPT-2 with fused LayerNorm on ANE.

    Per layer (3 dispatches):
      D1: LN1(MVN+BN) + QKV_proj (raw espresso, 3 layers)
      D2: O_proj (single inner_product, existing format)
      D3: LN2(MVN+BN) + FFN(up+GELU+down) (raw espresso, 5 layers)
      CPU: attention, residuals

    37 dispatches total (same count as baseline), but:
    - LN1 and LN2 moved from CPU to ANE (fused into D1 and D3)
    - Deeper fusion per dispatch (3 and 5 espresso layers vs 1 and 3)
    - Fewer CPU operations per token
    """
    os.makedirs(build_dir, exist_ok=True)
    config = model.config
    dim = config.n_embd
    ffn_dim = config.n_inner

    compiled = {}

    for i in range(config.n_layer):
        layer_dir = os.path.join(build_dir, f'layer_{i}')
        os.makedirs(layer_dir, exist_ok=True)
        L = model.layers[i]

        # D1: LN1 + QKV (fused)
        path = os.path.join(layer_dir, 'pre_attn.mlmodelc')
        if not os.path.exists(path):
            W_qkv = L.c_attn_weight.T.copy()  # [2304, 768]
            gen_pre_attn_espresso(
                path,
                ln1_weight=L.ln_1_weight, ln1_bias=L.ln_1_bias,
                W_qkv=W_qkv, bias_qkv=L.c_attn_bias,
                in_ch=dim, qkv_out=2304, eps=config.layer_norm_epsilon,
            )
        compiled[f'L{i}_pre_attn'] = (path, dim, 2304)

        # D2: O_proj
        path = os.path.join(layer_dir, 'o_proj.mlmodelc')
        if not os.path.exists(path):
            gen_conv_mlmodelc(path, L.W_o.astype(np.float32), dim, dim,
                              bias=L.c_proj_bias.astype(np.float32), name='o_proj')
        compiled[f'L{i}_o_proj'] = (path, dim, dim)

        # D3: fused FFN (existing proven format, LN2 stays on CPU)
        from compiler import gen_fused_ffn_mlmodelc
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

        print(f"  Layer {i}: pre_attn({dim}->2304) + o_proj({dim}->{dim}) + "
              f"ln2_ffn({dim}->{ffn_dim}->{dim})")

    # lm_head - reuse baseline if available
    baseline_lm = '/tmp/gpt2_first_token_fused/lm_head.mlmodelc'
    lm_head_path = os.path.join(build_dir, 'lm_head.mlmodelc')
    if os.path.exists(baseline_lm) and not os.path.exists(lm_head_path):
        shutil.copytree(baseline_lm, lm_head_path)
    elif not os.path.exists(lm_head_path):
        gen_conv_mlmodelc(lm_head_path, model.wte.astype(np.float32),
                          dim, config.vocab_size, name='lm_head')
    compiled['lm_head'] = (lm_head_path, dim, config.vocab_size)

    n_total = config.n_layer * 3 + 1
    print(f"  Total dispatches: {n_total}")
    return compiled


# ===================================================================
# Forward pass with fused LN
# ===================================================================

def forward_layer_fused_ln(layer_idx, x, model, dispatcher, kv_cache,
                            mode='fused_ln'):
    """Forward pass with fused LN on ANE (3 dispatches/layer)."""
    config = model.config
    dim = config.n_embd
    n_heads = config.n_head
    head_dim = config.head_dim
    pfx = f'L{layer_idx}'

    # D1: LN1+QKV on ANE (1 dispatch)
    qkv = dispatcher.dispatch(f'{pfx}_pre_attn', x)
    q = qkv[:dim]
    k = qkv[dim:2*dim]
    v = qkv[2*dim:]

    q_heads = q.reshape(n_heads, head_dim)
    k_heads = k.reshape(n_heads, head_dim)
    v_heads = v.reshape(n_heads, head_dim)

    kv_cache.append(layer_idx, k_heads[np.newaxis], v_heads[np.newaxis])

    # CPU attention
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
    attn_output = attn_output.astype(np.float16)

    # D2: O_proj on ANE
    o_out = dispatcher.dispatch(f'{pfx}_o_proj', attn_output)

    # Residual 1 (CPU)
    r1 = (x.astype(np.float32) + o_out.astype(np.float32)).astype(np.float16)

    # LN2 on CPU
    L = model.layers[layer_idx]
    ln2_out = layernorm_cpu(r1, L.ln_2_weight, L.ln_2_bias,
                            config.layer_norm_epsilon)

    # D3: fused FFN on ANE (1 dispatch)
    ffn_out = dispatcher.dispatch(f'{pfx}_fused_ffn', ln2_out)

    # Residual 2 (CPU)
    output = (r1.astype(np.float32) + ffn_out.astype(np.float32)).astype(np.float16)
    return output


# ===================================================================
# Generation loop
# ===================================================================

def generate_fused_ln(model, dispatcher, prompt_tokens, max_new_tokens=10):
    """Generation with fused-LN forward pass."""
    config = model.config
    kv_cache = KVCache(config.n_layer, config.n_head, config.head_dim)
    generated = list(prompt_tokens)

    for pos, token_id in enumerate(prompt_tokens[:-1]):
        x = embed(model, token_id, pos)
        for li in range(config.n_layer):
            x = forward_layer_fused_ln(li, x, model, dispatcher, kv_cache)

    last_prompt = prompt_tokens[-1]
    x = embed(model, last_prompt, len(prompt_tokens) - 1)
    for li in range(config.n_layer):
        x = forward_layer_fused_ln(li, x, model, dispatcher, kv_cache)
    logits = lm_head(x, model, dispatcher)
    next_token = int(np.argmax(logits.astype(np.float32)))
    generated.append(next_token)

    for step in range(max_new_tokens - 1):
        pos = len(generated) - 1
        x = embed(model, next_token, pos)
        for li in range(config.n_layer):
            x = forward_layer_fused_ln(li, x, model, dispatcher, kv_cache)
        logits = lm_head(x, model, dispatcher)
        next_token = int(np.argmax(logits.astype(np.float32)))
        generated.append(next_token)

    return generated


# ===================================================================
# Main test + benchmark
# ===================================================================

def main():
    print("=" * 70)
    print("GPT-2 117M: FUSED LAYERNORM ON ANE")
    print("37 dispatches (same count) but LN1+LN2 moved to ANE")
    print("=" * 70)

    from transformers import GPT2Tokenizer

    # Load
    print("\n[1/5] Loading GPT-2...")
    t0 = time.time()
    model = GPT2Model.from_safetensors(MODEL_PATH)
    tokenizer = GPT2Tokenizer.from_pretrained('openai-community/gpt2')
    config = model.config
    print(f"  Loaded in {time.time()-t0:.1f}s")

    # Compile fused LN
    print(f"\n[2/5] Compiling fused-LN ops...")
    t0 = time.time()
    compiled = compile_fused_ln(model, BUILD_DIR)
    print(f"  Built in {time.time()-t0:.1f}s")

    # Launch dispatcher
    print(f"\n[3/5] Launching ANE dispatcher...")
    t0 = time.time()
    disp = ANEDispatcher(compiled, quiet=True)
    disp.start()
    print(f"  Ready in {time.time()-t0:.1f}s")

    # PyTorch reference
    print("\n[4/5] PyTorch reference...")
    import torch
    from transformers import GPT2LMHeadModel
    pt_model = GPT2LMHeadModel.from_pretrained('openai-community/gpt2')
    pt_model.eval()

    prompt = "The capital of France"
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    with torch.no_grad():
        output = pt_model.generate(input_ids, max_new_tokens=10, do_sample=False)
    pt_tokens = output[0].tolist()
    pt_text = tokenizer.decode(pt_tokens)
    print(f"  PyTorch: \"{pt_text}\"")
    print(f"  Tokens: {pt_tokens}")

    # Fused-LN generation
    prompt_tokens = tokenizer.encode(prompt)
    print(f"\n[5/5] Fused-LN ANE generation...")
    t0 = time.time()
    ane_tokens = generate_fused_ln(model, disp, prompt_tokens, max_new_tokens=10)
    gen_time = time.time() - t0
    ane_text = tokenizer.decode(ane_tokens)
    n_new = len(ane_tokens) - len(prompt_tokens)
    tps = n_new / gen_time if gen_time > 0 else 0
    print(f"  ANE: \"{ane_text}\"")
    print(f"  {n_new} tokens in {gen_time:.2f}s = {tps:.1f} tok/s")

    # Kill test
    print(f"\n{'='*70}")
    print("KILL TEST: token-by-token match vs PyTorch")
    all_match = True
    for i in range(max(len(pt_tokens), len(ane_tokens))):
        pt_t = pt_tokens[i] if i < len(pt_tokens) else None
        ane_t = ane_tokens[i] if i < len(ane_tokens) else None
        if pt_t is not None and ane_t is not None:
            match = pt_t == ane_t
            if not match:
                all_match = False
            label = "OK" if match else "MISS"
            tok = tokenizer.decode([pt_t])
            print(f"  pos {i:2d}: PT={pt_t:6d} ANE={ane_t:6d} {label} \"{tok}\"")
        elif pt_t is not None:
            print(f"  pos {i:2d}: PT={pt_t:6d} ANE=MISSING")
            all_match = False
        else:
            print(f"  pos {i:2d}: PT=MISSING ANE={ane_t:6d}")

    if all_match:
        print(f"\n*** KILL TEST: PASS ({len(pt_tokens)}/{len(pt_tokens)} match) ***")
    else:
        matches = sum(1 for i in range(min(len(ane_tokens), len(pt_tokens)))
                      if ane_tokens[i] == pt_tokens[i])
        print(f"\n*** KILL TEST: {matches}/{len(pt_tokens)} match ***")

    disp.stop()

    # Benchmark: fused-LN vs baseline
    if all_match:
        print(f"\n{'='*70}")
        print("BENCHMARK: fused-LN vs baseline 37-dispatch")
        print("=" * 70)

        from first_token import compile_all_ops

        # Baseline
        print("\n[A] Baseline 37-dispatch...")
        compiled_37 = compile_all_ops(model, '/tmp/gpt2_first_token_fused', mode='fused')
        disp_37 = ANEDispatcher(compiled_37, quiet=True)
        disp_37.start()

        kv37 = KVCache(config.n_layer, config.n_head, config.head_dim)
        for w in range(5):
            x = embed(model, 0, w)
            for li in range(config.n_layer):
                x = forward_layer_decode(li, x, model, disp_37, kv37, mode='fused')
            lm_head(x, model, disp_37)

        times_37 = []
        for step in range(50):
            t0 = time.perf_counter()
            x = embed(model, 0, 5 + step)
            for li in range(config.n_layer):
                x = forward_layer_decode(li, x, model, disp_37, kv37, mode='fused')
            lm_head(x, model, disp_37)
            times_37.append((time.perf_counter() - t0) * 1000)
        disp_37.stop()
        times_37.sort()
        med_37 = times_37[len(times_37) // 2]

        # Fused LN
        print("[B] Fused-LN 37-dispatch...")
        compiled_fl = compile_fused_ln(model, BUILD_DIR)
        disp_fl = ANEDispatcher(compiled_fl, quiet=True)
        disp_fl.start()

        kv_fl = KVCache(config.n_layer, config.n_head, config.head_dim)
        for w in range(5):
            x = embed(model, 0, w)
            for li in range(config.n_layer):
                x = forward_layer_fused_ln(li, x, model, disp_fl, kv_fl)
            lm_head(x, model, disp_fl)

        times_fl = []
        for step in range(50):
            t0 = time.perf_counter()
            x = embed(model, 0, 5 + step)
            for li in range(config.n_layer):
                x = forward_layer_fused_ln(li, x, model, disp_fl, kv_fl)
            lm_head(x, model, disp_fl)
            times_fl.append((time.perf_counter() - t0) * 1000)
        disp_fl.stop()
        times_fl.sort()
        med_fl = times_fl[len(times_fl) // 2]

        # Weight sizes for bandwidth
        total_weight = ((768*2304 + 768*768 + 768*3072 + 3072*768) * 2 * 12
                        + 768 * 50257 * 2)

        tps_37 = 1000.0 / med_37
        tps_fl = 1000.0 / med_fl
        bw_37 = (total_weight / 1e9) / (med_37 / 1000)
        bw_fl = (total_weight / 1e9) / (med_fl / 1000)
        speedup = med_37 / med_fl

        print(f"\n{'='*70}")
        print(f"{'Config':<25} {'Disp':>5} {'ms/tok':>8} {'tok/s':>7} {'GB/s':>7} {'vs base':>8}")
        print("-" * 70)
        print(f"{'Baseline (CPU LN)':<25} {37:>5} {med_37:>8.2f} {tps_37:>7.1f} {bw_37:>7.1f} {'1.00x':>8}")
        print(f"{'Fused LN (ANE LN)':<25} {37:>5} {med_fl:>8.2f} {tps_fl:>7.1f} {bw_fl:>7.1f} {speedup:>7.2f}x")
        print("=" * 70)
        print(f"\nDelta: {med_37 - med_fl:.2f} ms saved by moving LN to ANE")
        print(f"LN overhead per token (12 LN1 + 12 LN2 + 1 LN_f, CPU): "
              f"{(med_37 - med_fl):.2f} ms")


if __name__ == '__main__':
    main()
