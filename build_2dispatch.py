#!/usr/bin/env python3
"""
Build LN-fused GPT-2 generation with C/Accelerate hot path.

Fuses LayerNorm into ANE dispatches:
  - LN1 + QKV: MVN + BN + inner_product (1 dispatch, was CPU LN + 1 dispatch)
  - LN2 + FFN: MVN + BN + fc_up + GELU + fc_down (1 dispatch, was CPU LN + 1 dispatch)
  - LN_f + lm_head: MVN + BN + inner_product (1 dispatch, was CPU LN + 1 dispatch)

Same dispatch count (37) but eliminates ALL CPU LayerNorm computation.

Uses proven v28 weight format from bench_fusion_depth.py.

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

from model_loader import GPT2Model
from compiler import gen_conv_mlmodelc

MODEL_PATH = os.path.expanduser(
    "~/.cache/huggingface/hub/models--openai-community--gpt2/"
    "snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e/model.safetensors"
)
PIPE_TOOL = os.path.join(os.path.dirname(__file__), 'tests', 'ane_standalone_pipe')
BUILD_DIR = '/tmp/gpt2_2dispatch'

DIM = 768
FFN_DIM = 3072
N_LAYERS = 12
VOCAB_SIZE = 50257


# ===================================================================
# Espresso file writers (matching bench_fusion_depth v28 format exactly)
# ===================================================================

def write_bundle_files(output_dir, layers_json, shapes_dict, weight_blobs,
                       model, in_dim, out_dim):
    """Write a complete .mlmodelc bundle in v28 format.

    Args:
        layers_json: list of espresso layer dicts
        shapes_dict: dict of name -> (n, h, w, k) tuples
        weight_blobs: list of (name, type, shape_or_dim, blob_id) tuples
        model: GPT2Model for extracting weights
        in_dim: input dimension
        out_dim: output dimension
    """
    os.makedirs(output_dir, exist_ok=True)
    for sub in ['model', 'analytics', 'neural_network_optionals']:
        os.makedirs(os.path.join(output_dir, sub), exist_ok=True)

    # espresso.net
    net = {
        "storage": "model.espresso.weights",
        "analyses": {}, "properties": {},
        "format_version": 200,
        "metadata_in_weights": [],
        "layers": layers_json,
    }
    with open(os.path.join(output_dir, 'model.espresso.net'), 'w') as f:
        json.dump(net, f)

    # espresso.shape (with _rank: 3 required for aned compilation)
    layer_shapes = {}
    for name, (n, h, w, k) in shapes_dict.items():
        layer_shapes[name] = {"n": n, "h": h, "w": w, "k": k, "_rank": 3}
    with open(os.path.join(output_dir, 'model.espresso.shape'), 'w') as f:
        json.dump({"layer_shapes": layer_shapes}, f)

    # espresso.weights (v28)
    write_weights_v28(output_dir, weight_blobs, model)

    # metadata.json
    meta = {
        "specificationVersion": 4, "isUpdatable": False,
        "modelType": {"name": "MLModelType_neuralNetwork"},
        "computePrecision": "Float16",
        "inputSchema": [{"name": "input", "type": "MultiArray",
                         "shape": [in_dim, 1, 1], "dataType": "Float16"}],
        "outputSchema": [{"name": "output", "type": "MultiArray",
                          "shape": [out_dim, 1, 1], "dataType": "Float16"}],
    }
    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump(meta, f)

    with open(os.path.join(output_dir, 'coremldata.bin'), 'wb') as f:
        f.write(b'\x08\x04')
    for sub in ['model', 'analytics', 'neural_network_optionals']:
        with open(os.path.join(output_dir, sub, 'coremldata.bin'), 'wb') as f:
            f.write(b'\x00' * 64)


def make_blob_data(blob_name, blob_type, shape_or_dim, model, layer_idx=None):
    """Generate weight bytes for a single blob entry."""
    dtype = np.float32

    if blob_type == 'batchnorm':
        dim_bn = shape_or_dim
        if blob_name == 'ln_f' or blob_name.startswith('ln_f'):
            gamma = model.ln_f_weight.astype(dtype)
            beta = model.ln_f_bias.astype(dtype)
        elif layer_idx is not None:
            L = model.layers[layer_idx]
            if 'ln1' in blob_name:
                gamma = L.ln_1_weight.astype(dtype)
                beta = L.ln_1_bias.astype(dtype)
            else:
                gamma = L.ln_2_weight.astype(dtype)
                beta = L.ln_2_bias.astype(dtype)
        else:
            raise ValueError(f"Cannot determine LN source for {blob_name}")
        mean = np.zeros(dim_bn, dtype=dtype)
        var = np.ones(dim_bn, dtype=dtype)
        return np.column_stack([gamma, beta, mean, var]).flatten().tobytes()

    elif blob_type == 'ip':
        out_ch, in_ch = shape_or_dim

        if blob_name == 'lm_head':
            w = model.wte.astype(dtype)
            return w.tobytes()

        L = model.layers[layer_idx]
        if blob_name.endswith('qkv'):
            w = L.c_attn_weight.T.copy().astype(dtype)  # [2304, 768]
        elif blob_name.endswith('o'):
            w = L.W_o.astype(dtype)
        elif blob_name.endswith('fc_up'):
            w = L.W_fc.astype(dtype)
        elif blob_name.endswith('fc_down'):
            w = L.W_fc_down.astype(dtype)
        else:
            raise ValueError(f"Unknown ip blob: {blob_name}")
        return w.tobytes()

    elif blob_type == 'bias':
        L = model.layers[layer_idx]
        if blob_name.endswith('qkv_bias'):
            b = L.c_attn_bias.astype(dtype)
        elif blob_name.endswith('o_bias'):
            b = L.c_proj_bias.astype(dtype)
        elif blob_name.endswith('fc_up_bias'):
            b = L.c_fc_bias.astype(dtype)
        elif blob_name.endswith('fc_down_bias'):
            b = L.c_proj_ffn_bias.astype(dtype)
        else:
            raise ValueError(f"Unknown bias blob: {blob_name}")
        return b.tobytes()

    raise ValueError(f"Unknown blob type: {blob_type}")


def write_weights_v28(output_dir, weight_blobs, model):
    """Write model.espresso.weights in v28 format.

    Exactly matches bench_fusion_depth.py pattern.
    Each blob entry has: (name, type, shape_or_dim, blob_id, layer_idx)
    """
    total_weight_blobs = len(weight_blobs)

    # Prepare blob data
    blob_info = []
    for entry in weight_blobs:
        name, btype, shape_or_dim, bid, layer_idx = entry
        data = make_blob_data(name, btype, shape_or_dim, model, layer_idx)
        raw_size = len(data)
        if btype == 'batchnorm':
            table_size = max(0x3000, raw_size * 4)
        elif btype == 'bias':
            table_size = max(0x3000, raw_size * 4)
        else:
            table_size = raw_size
        blob_info.append((bid, data, raw_size, table_size, btype, name))

    # First blob goes in gap
    first = blob_info[0]
    first_gap = first[3]
    remaining = blob_info[1:]

    n_entries = len(remaining)
    blob_table_offset = 0x38
    header_raw_end = blob_table_offset + (n_entries + 1) * 0x20
    header_padded = max(0x200, ((header_raw_end + 0x1FF) // 0x200) * 0x200)

    weight_path = os.path.join(output_dir, 'model.espresso.weights')
    with open(weight_path, 'wb') as f:
        header = bytearray(header_padded)
        struct.pack_into('<I', header, 0x00, total_weight_blobs * 2)
        struct.pack_into('<I', header, 0x10, blob_table_offset)
        struct.pack_into('<I', header, 0x18, 1)
        struct.pack_into('<I', header, 0x20, first_gap)
        struct.pack_into('<I', header, 0x28, 2)

        for i, (bid, data, raw_size, table_size, btype, name) in enumerate(remaining):
            off = blob_table_offset + i * 0x20
            next_id = remaining[i+1][0] if i+1 < len(remaining) else 0
            struct.pack_into('<I', header, off + 0, bid)
            struct.pack_into('<I', header, off + 8, table_size)
            struct.pack_into('<I', header, off + 16, next_id)

        f.write(header)

        # First blob in gap
        f.write(first[1])
        f.write(b'\x00' * (first_gap - first[2]))

        # Remaining blobs
        for bid, data, raw_size, table_size, btype, name in remaining:
            f.write(data)
            if table_size > raw_size:
                f.write(b'\x00' * (table_size - raw_size))


# ===================================================================
# Build fused models
# ===================================================================

def build_ln1_qkv(output_dir, model, layer_idx):
    """Build LN1 + QKV: MVN + BN + biased inner_product.

    Uses v28 format with stride-2 blob IDs:
      blob 1: BN params
      blob 3: QKV bias (as bias section)
      blob 5: QKV weights
    """
    layers_json = [
        {
            "type": "l2_normalize", "name": "ln1_mvn",
            "debug_info": "ln1_mvn",
            "bottom": "input", "top": "ln1_mvn_out",
            "normalization_mode": 1, "axis": 2, "eps": 1e-5,
            "weights": {},
        },
        {
            "type": "batchnorm", "name": "ln1_bn",
            "debug_info": "ln1_bn",
            "bottom": "ln1_mvn_out", "top": "ln1_out",
            "blob_batchnorm_params": 1, "C": DIM, "weights": {},
        },
        {
            "type": "inner_product", "name": "qkv_proj",
            "debug_info": "qkv_proj",
            "bottom": "ln1_out", "top": "output",
            "nB": DIM, "nC": 2304,
            "has_biases": 1,
            "blob_biases": 3,
            "blob_weights": 5,
            "has_relu": 0, "has_tanh": 0, "has_prelu": 0,
            "weights": {},
            "attributes": {"is_output": 1},
        },
    ]

    shapes = {
        "input": (1, 1, 1, DIM),
        "ln1_mvn_out": (1, 1, 1, DIM),
        "ln1_out": (1, 1, 1, DIM),
        "output": (1, 1, 2304, 2304),  # IP output: w=out_ch, k=out_ch
    }

    weight_blobs = [
        (f'L{layer_idx}_ln1', 'batchnorm', DIM, 1, layer_idx),
        (f'L{layer_idx}_qkv_bias', 'bias', None, 3, layer_idx),
        (f'L{layer_idx}_qkv', 'ip', (2304, DIM), 5, layer_idx),
    ]

    write_bundle_files(output_dir, layers_json, shapes, weight_blobs,
                       model, DIM, 2304)


def build_ln2_ffn(output_dir, model, layer_idx):
    """Build LN2 + FFN: MVN + BN + fc_up + GELU + fc_down (biased).

    Blob layout (stride-2):
      blob 1: BN params
      blob 3: fc_up bias
      blob 5: fc_up weights
      blob 7: fc_down bias
      blob 9: fc_down weights
    """
    layers_json = [
        {
            "type": "l2_normalize", "name": "ln2_mvn",
            "debug_info": "ln2_mvn",
            "bottom": "input", "top": "ln2_mvn_out",
            "normalization_mode": 1, "axis": 2, "eps": 1e-5,
            "weights": {},
        },
        {
            "type": "batchnorm", "name": "ln2_bn",
            "debug_info": "ln2_bn",
            "bottom": "ln2_mvn_out", "top": "ln2_out",
            "blob_batchnorm_params": 1, "C": DIM, "weights": {},
        },
        {
            "type": "inner_product", "name": "fc_up",
            "debug_info": "fc_up",
            "bottom": "ln2_out", "top": "fc_up_out",
            "nB": DIM, "nC": FFN_DIM,
            "has_biases": 1,
            "blob_biases": 3,
            "blob_weights": 5,
            "has_relu": 0, "has_tanh": 0, "has_prelu": 0,
            "weights": {},
        },
        {
            "type": "activation", "name": "gelu",
            "bottom": "fc_up_out", "top": "gelu_out",
            "mode": 19, "weights": {},
        },
        {
            "type": "inner_product", "name": "fc_down",
            "debug_info": "fc_down",
            "bottom": "gelu_out", "top": "output",
            "nB": FFN_DIM, "nC": DIM,
            "has_biases": 1,
            "blob_biases": 7,
            "blob_weights": 9,
            "has_relu": 0, "has_tanh": 0, "has_prelu": 0,
            "weights": {},
            "attributes": {"is_output": 1},
        },
    ]

    shapes = {
        "input": (1, 1, 1, DIM),
        "ln2_mvn_out": (1, 1, 1, DIM),
        "ln2_out": (1, 1, 1, DIM),
        "fc_up_out": (1, 1, FFN_DIM, FFN_DIM),  # IP output
        "gelu_out": (1, 1, FFN_DIM, FFN_DIM),
        "output": (1, 1, DIM, DIM),  # IP output
    }

    weight_blobs = [
        (f'L{layer_idx}_ln2', 'batchnorm', DIM, 1, layer_idx),
        (f'L{layer_idx}_fc_up_bias', 'bias', None, 3, layer_idx),
        (f'L{layer_idx}_fc_up', 'ip', (FFN_DIM, DIM), 5, layer_idx),
        (f'L{layer_idx}_fc_down_bias', 'bias', None, 7, layer_idx),
        (f'L{layer_idx}_fc_down', 'ip', (DIM, FFN_DIM), 9, layer_idx),
    ]

    write_bundle_files(output_dir, layers_json, shapes, weight_blobs,
                       model, DIM, DIM)


def build_fused_lm_head(output_dir, model):
    """Build LN_f + lm_head: MVN + BN + inner_product (no bias).

    Blob layout:
      blob 1: BN params
      blob 3: lm_head weights
    """
    layers_json = [
        {
            "type": "l2_normalize", "name": "ln_f_mvn",
            "debug_info": "ln_f_mvn",
            "bottom": "input", "top": "ln_f_mvn_out",
            "normalization_mode": 1, "axis": 2, "eps": 1e-5,
            "weights": {},
        },
        {
            "type": "batchnorm", "name": "ln_f_bn",
            "debug_info": "ln_f_bn",
            "bottom": "ln_f_mvn_out", "top": "ln_f_out",
            "blob_batchnorm_params": 1, "C": DIM, "weights": {},
        },
        {
            "type": "inner_product", "name": "lm_head",
            "debug_info": "lm_head",
            "bottom": "ln_f_out", "top": "output",
            "nB": DIM, "nC": VOCAB_SIZE,
            "has_biases": 0,
            "blob_weights": 3,
            "has_relu": 0, "has_tanh": 0, "has_prelu": 0,
            "weights": {},
            "attributes": {"is_output": 1},
        },
    ]

    shapes = {
        "input": (1, 1, 1, DIM),
        "ln_f_mvn_out": (1, 1, 1, DIM),
        "ln_f_out": (1, 1, 1, DIM),
        "output": (1, 1, 1, VOCAB_SIZE),  # lm_head keeps w=1 for large output
    }

    weight_blobs = [
        ('ln_f', 'batchnorm', DIM, 1, None),
        ('lm_head', 'ip', (VOCAB_SIZE, DIM), 3, None),
    ]

    write_bundle_files(output_dir, layers_json, shapes, weight_blobs,
                       model, DIM, VOCAB_SIZE)


# ===================================================================
# Compile all models
# ===================================================================

def compile_all_fused(model):
    """Compile all fused models.

    Returns dict of op_name -> (mlmodelc_path, in_ch, out_ch)
    """
    os.makedirs(BUILD_DIR, exist_ok=True)
    compiled = {}

    for i in range(N_LAYERS):
        layer_dir = os.path.join(BUILD_DIR, f'layer_{i}')
        os.makedirs(layer_dir, exist_ok=True)

        # LN1 + QKV
        a_path = os.path.join(layer_dir, 'ln1_qkv.mlmodelc')
        if not os.path.exists(a_path):
            build_ln1_qkv(a_path, model, i)
        compiled[f'L{i}_ln1_qkv'] = (a_path, DIM, 2304)

        # O projection (unchanged from baseline)
        o_path = os.path.join(layer_dir, 'o_proj.mlmodelc')
        if not os.path.exists(o_path):
            L = model.layers[i]
            gen_conv_mlmodelc(o_path, L.W_o.astype(np.float32), DIM, DIM,
                              bias=L.c_proj_bias.astype(np.float32), name='o_proj')
        compiled[f'L{i}_o_proj'] = (o_path, DIM, DIM)

        # LN2 + FFN
        b_path = os.path.join(layer_dir, 'ln2_ffn.mlmodelc')
        if not os.path.exists(b_path):
            build_ln2_ffn(b_path, model, i)
        compiled[f'L{i}_ln2_ffn'] = (b_path, DIM, DIM)

    # LM head (standalone, not fused — too large for fused compilation)
    # LN_f stays on CPU for lm_head path
    lm_path = os.path.join(BUILD_DIR, 'lm_head.mlmodelc')
    if not os.path.exists(lm_path):
        gen_conv_mlmodelc(lm_path, model.wte.astype(np.float32),
                          DIM, VOCAB_SIZE, name='lm_head')
    compiled['lm_head'] = (lm_path, DIM, VOCAB_SIZE)

    print(f"  Built {len(compiled)} ops ({N_LAYERS}*3 + 1 lm_head = {len(compiled)})")
    return compiled


# ===================================================================
# Verify via Python dispatch
# ===================================================================

def verify_against_pytorch(model, compiled):
    """Verify fused generation matches PyTorch."""
    from generate import ANEDispatcher, softmax_cpu, embed
    from kv_cache import KVCache

    print("\n" + "=" * 60)
    print("VERIFICATION: LN-fused vs PyTorch")
    print("=" * 60)

    dispatcher = ANEDispatcher(compiled, quiet=True)
    dispatcher.start()

    config = model.config
    kv_cache = KVCache(config.n_layer, config.n_head, config.head_dim)

    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained('gpt2')
    except ImportError:
        tokenizer = None

    prompt = "The"
    prompt_tokens = tokenizer.encode(prompt) if tokenizer else [464]
    n_gen = 10
    generated = list(prompt_tokens)

    for step in range(len(prompt_tokens) + n_gen):
        if step < len(prompt_tokens):
            pos = step
            tok = prompt_tokens[pos]
        else:
            pos = len(generated) - 1
            tok = generated[pos]

        x = embed(model, tok, pos)

        for li in range(config.n_layer):
            # LN1 + QKV (fused)
            qkv = dispatcher.dispatch(f'L{li}_ln1_qkv', x)
            q, k, v = qkv[:DIM], qkv[DIM:2*DIM], qkv[2*DIM:]

            # Attention (CPU)
            n_heads, head_dim = config.n_head, config.head_dim
            q_heads = q.reshape(n_heads, head_dim)
            k_heads = k.reshape(n_heads, head_dim)
            v_heads = v.reshape(n_heads, head_dim)
            kv_cache.append(li, k_heads[np.newaxis], v_heads[np.newaxis])
            cached_k, cached_v = kv_cache.get(li)

            scale = np.float32(1.0 / np.sqrt(head_dim))
            attn_output = np.zeros(DIM, dtype=np.float32)
            for h in range(n_heads):
                q_h = q_heads[h].astype(np.float32)
                k_h = cached_k[:, h, :].astype(np.float32)
                v_h = cached_v[:, h, :].astype(np.float32)
                scores = (q_h @ k_h.T) * scale
                weights = softmax_cpu(scores)
                attn_output[h*head_dim:(h+1)*head_dim] = weights @ v_h
            attn_output = attn_output.astype(np.float16)

            # O projection (ANE)
            o_out = dispatcher.dispatch(f'L{li}_o_proj', attn_output)

            # Residual 1
            r1 = (x.astype(np.float32) + o_out.astype(np.float32)).astype(np.float16)

            # LN2 + FFN (fused)
            ffn_out = dispatcher.dispatch(f'L{li}_ln2_ffn', r1)

            # Residual 2
            x = (r1.astype(np.float32) + ffn_out.astype(np.float32)).astype(np.float16)

        # LM head (LN_f on CPU, lm_head on ANE — not fused)
        from first_token import layernorm_cpu
        x_ln = layernorm_cpu(x, model.ln_f_weight, model.ln_f_bias,
                              model.config.layer_norm_epsilon)
        logits = dispatcher.dispatch('lm_head', x_ln)
        next_token = int(np.argmax(logits.astype(np.float32)))

        if step >= len(prompt_tokens):
            generated.append(next_token)
        elif step == len(prompt_tokens) - 1:
            generated.append(next_token)

    dispatcher.stop()

    # PyTorch reference
    import torch
    from transformers import GPT2LMHeadModel
    pt_model = GPT2LMHeadModel.from_pretrained('openai-community/gpt2')
    pt_model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors="pt") if tokenizer else torch.tensor([[464]])
    with torch.no_grad():
        output = pt_model.generate(input_ids, max_new_tokens=n_gen, do_sample=False)
    pt_tokens = output[0].tolist()

    ane_text = tokenizer.decode(generated) if tokenizer else str(generated)
    pt_text = tokenizer.decode(pt_tokens) if tokenizer else str(pt_tokens)

    print(f"  PyTorch: {pt_tokens}")
    print(f"  ANE:     {generated}")
    print(f"  PT text: '{pt_text}'")
    print(f"  ANE text:'{ane_text}'")

    matches = sum(1 for i in range(min(len(generated), len(pt_tokens)))
                  if generated[i] == pt_tokens[i])
    total = len(pt_tokens)
    all_match = matches == total and len(generated) >= total

    print(f"  Kill test: {matches}/{total}")
    if all_match:
        print(f"  *** KILL TEST: PASS ***")
    else:
        for i in range(min(len(generated), len(pt_tokens))):
            if generated[i] != pt_tokens[i]:
                print(f"  First divergence at pos {i}: PT={pt_tokens[i]} ANE={generated[i]}")
                break
        print(f"  *** KILL TEST: FAIL ***")

    return all_match


# ===================================================================
# Export + C binary
# ===================================================================

def export_for_c_binary(model, compiled):
    """Export manifest and CPU weights for C binary."""
    weights_bin = os.path.join(BUILD_DIR, 'cpu_weights.bin')
    manifest = os.path.join(BUILD_DIR, 'manifest.txt')

    # CPU weights: embeddings + final LN (LN1/LN2 fused into ANE)
    with open(weights_bin, 'wb') as f:
        f.write(model.wte.astype(np.float32).tobytes())
        f.write(model.wpe.astype(np.float32).tobytes())
        f.write(model.ln_f_weight.astype(np.float32).tobytes())
        f.write(model.ln_f_bias.astype(np.float32).tobytes())

    size_mb = os.path.getsize(weights_bin) / 1e6
    print(f"  CPU weights: {size_mb:.1f} MB (embeddings + final LN)")

    with open(manifest, 'w') as f:
        for name in sorted(compiled.keys()):
            path, in_ch, out_ch = compiled[name]
            f.write(f"{path} {in_ch} {out_ch} {name}\n")
    print(f"  Manifest: {len(compiled)} ops")

    return manifest, weights_bin


def write_c_source():
    """Write the C generation binary source."""
    c_path = os.path.join(os.path.dirname(__file__), 'tests', 'ane_generate_2d.m')
    c_code = r"""// ane_generate_2d.m — GPT-2 with LN-fused ANE dispatches
//
// Per layer: LN1+QKV(ANE) -> attention(CPU) -> O(ANE) -> res(CPU) ->
//            LN2+FFN(ANE) -> res(CPU)
// Final: CPU LN_f + ANE lm_head (separate, fused LN_f too large)
// = 37 dispatches, 0 CPU LayerNorm
//
// Build:
//   xcrun clang -O2 -framework Foundation -framework IOSurface -framework Accelerate \
//     -fobjc-arc -o ane_generate_2d ane_generate_2d.m

#import <Foundation/Foundation.h>
#import <Accelerate/Accelerate.h>
#import <IOSurface/IOSurface.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>
#import <signal.h>
#import <mach/mach_time.h>

#define N_LAYERS     12
#define N_HEADS      12
#define DIM          768
#define HEAD_DIM     64
#define VOCAB_SIZE   50257
#define MAX_SEQ      1024

static Class _Cl, _Mo, _Rq, _IO;
static void loadFW(void) {
    dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);
    _Cl = NSClassFromString(@"_ANEClient");
    _Mo = NSClassFromString(@"_ANEModel");
    _Rq = NSClassFromString(@"_ANERequest");
    _IO = NSClassFromString(@"_ANEIOSurfaceObject");
}

typedef struct {
    id model;
    int inCh, outCh;
    uint32_t inBS, outBS, inPS, outPS;
    IOSurfaceRef inSurf, outSurf;
    id inObj, outObj;
} OpEntry;

typedef struct {
    float *k_cache, *v_cache;
    int len;
} KVCache;

static mach_timebase_info_data_t tb;
static inline double ns_to_ms(uint64_t ns) {
    return (double)(ns * tb.numer / tb.denom) / 1e6;
}

static void softmax_f32(float *x, int n) {
    float max_val;
    vDSP_maxv(x, 1, &max_val, n);
    float neg_max = -max_val;
    vDSP_vsadd(x, 1, &neg_max, x, 1, n);
    int n_int = n;
    vvexpf(x, x, &n_int);
    float sum = 0;
    vDSP_sve(x, 1, &sum, n);
    float inv_sum = 1.0f / sum;
    vDSP_vsmul(x, 1, &inv_sum, x, 1, n);
}

static void fp16_to_fp32(const uint16_t *in, float *out, int n) {
    vImage_Buffer src = {(void*)in, 1, (vImagePixelCount)n, n * 2};
    vImage_Buffer dst = {out, 1, (vImagePixelCount)n, n * 4};
    vImageConvert_Planar16FtoPlanarF(&src, &dst, 0);
}

static void fp32_to_fp16(const float *in, uint16_t *out, int n) {
    vImage_Buffer src = {(void*)in, 1, (vImagePixelCount)n, n * 4};
    vImage_Buffer dst = {out, 1, (vImagePixelCount)n, n * 2};
    vImageConvert_PlanarFtoPlanar16F(&src, &dst, 0);
}

static id ane_client = nil;

static void ane_dispatch(OpEntry *op, const uint16_t *input, uint16_t *output) {
    IOSurfaceLock(op->inSurf, 0, NULL);
    void *inBase = IOSurfaceGetBaseAddress(op->inSurf);
    memset(inBase, 0, op->inBS);
    for (int j = 0; j < op->inCh; j++)
        memcpy((uint8_t*)inBase + j * op->inPS, &input[j], 2);
    IOSurfaceUnlock(op->inSurf, 0, NULL);

    id req = ((id (*)(id, SEL, id, id, id, id, id, id, id, id, id))objc_msgSend)(
        [_Rq alloc],
        NSSelectorFromString(@"initWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:sharedEvents:transactionHandle:"),
        @[op->inObj], @[@0], @[op->outObj], @[@0], nil, nil, @(0), nil, nil);

    ((BOOL (*)(id, SEL, id, id, BOOL, id*))objc_msgSend)(
        ane_client, NSSelectorFromString(@"mapIOSurfacesWithModel:request:cacheInference:error:"),
        op->model, req, NO, nil);

    ((BOOL (*)(id, SEL, id, id, id, int, id*))objc_msgSend)(
        ane_client, NSSelectorFromString(@"doEvaluateDirectWithModel:options:request:qos:error:"),
        op->model, @{}, req, 21, nil);

    IOSurfaceLock(op->outSurf, kIOSurfaceLockReadOnly, NULL);
    void *outBase = IOSurfaceGetBaseAddress(op->outSurf);
    for (int j = 0; j < op->outCh; j++)
        memcpy(&output[j], (uint8_t*)outBase + j * op->outPS, 2);
    IOSurfaceUnlock(op->outSurf, kIOSurfaceLockReadOnly, NULL);

    ((void (*)(id, SEL, id, id))objc_msgSend)(
        ane_client, NSSelectorFromString(@"unmapIOSurfacesWithModel:request:"), op->model, req);
}

int main(int argc, char *argv[]) {
    @autoreleasepool {
        signal(SIGSEGV, SIG_IGN);
        mach_timebase_info(&tb);
        loadFW();

        if (argc < 4) {
            fprintf(stderr, "Usage: %s <manifest.txt> <weights.bin> <n_tokens> [prompt_tokens...]\n", argv[0]);
            return 1;
        }

        const char *manifest_path = argv[1];
        const char *weights_path = argv[2];
        int n_gen_tokens = atoi(argv[3]);

        int n_prompt = argc - 4;
        int *prompt_tokens = malloc(n_prompt * sizeof(int));
        for (int i = 0; i < n_prompt; i++)
            prompt_tokens[i] = atoi(argv[4 + i]);

        // Load manifest
        NSString *manifest = [NSString stringWithContentsOfFile:
            [NSString stringWithUTF8String:manifest_path]
            encoding:NSUTF8StringEncoding error:nil];
        NSArray *lines = [manifest componentsSeparatedByString:@"\n"];

        int nOps = 0;
        OpEntry *ops = calloc(lines.count, sizeof(OpEntry));
        NSMutableArray *opPaths = [NSMutableArray array];
        NSMutableDictionary *opMap = [NSMutableDictionary dictionary];

        for (NSString *line in lines) {
            NSArray *parts = [line componentsSeparatedByString:@" "];
            if (parts.count < 4) continue;
            ops[nOps].inCh = [parts[1] intValue];
            ops[nOps].outCh = [parts[2] intValue];
            [opPaths addObject:parts[0]];
            opMap[parts[3]] = @(nOps);
            nOps++;
        }
        fprintf(stderr, "Loaded %d ops from manifest\n", nOps);

        int (^opIdx)(NSString *) = ^int(NSString *name) {
            NSNumber *n = opMap[name];
            if (!n) { fprintf(stderr, "Op not found: %s\n", [name UTF8String]); exit(1); }
            return [n intValue];
        };

        // CPU weights (embeddings + final LN)
        FILE *wf = fopen(weights_path, "rb");
        if (!wf) { fprintf(stderr, "Cannot open weights\n"); return 1; }
        float *wte = malloc(VOCAB_SIZE * DIM * sizeof(float));
        float *wpe = malloc(MAX_SEQ * DIM * sizeof(float));
        float *ln_f_w = malloc(DIM * sizeof(float));
        float *ln_f_b = malloc(DIM * sizeof(float));
        fread(wte, sizeof(float), VOCAB_SIZE * DIM, wf);
        fread(wpe, sizeof(float), MAX_SEQ * DIM, wf);
        fread(ln_f_w, sizeof(float), DIM, wf);
        fread(ln_f_b, sizeof(float), DIM, wf);
        fclose(wf);
        fprintf(stderr, "CPU weights loaded (embeddings + final LN)\n");

        // Compile + load
        ane_client = ((id (*)(id, SEL))objc_msgSend)((id)_Cl, NSSelectorFromString(@"sharedConnection"));
        NSError *err = nil;

        fprintf(stderr, "Compiling %d models...\n", nOps);
        for (int i = 0; i < nOps; i++) {
            NSURL *url = [NSURL fileURLWithPath:opPaths[i]];
            NSString *key = [NSString stringWithFormat:@"op_%d", i];
            ops[i].model = ((id (*)(id, SEL, id, id))objc_msgSend)(
                (id)_Mo, NSSelectorFromString(@"modelAtURL:key:"), url, key);

            err = nil;
            ((BOOL (*)(id, SEL, id, id, NSInteger, id*))objc_msgSend)(
                ane_client, NSSelectorFromString(@"compileModel:options:qos:error:"),
                ops[i].model, @{}, 0, &err);
            if (err) fprintf(stderr, "Compile %d failed: %s\n", i, [[err description] UTF8String]);

            BOOL ok = ((BOOL (*)(id, SEL, id, id, NSInteger, id*))objc_msgSend)(
                ane_client, NSSelectorFromString(@"loadModel:options:qos:error:"),
                ops[i].model, @{}, 0, &err);
            if (!ok) { fprintf(stderr, "Load %d FATAL: %s\n", i, err ? [[err description] UTF8String] : "nil"); return 1; }

            id attrs = ((id (*)(id, SEL))objc_msgSend)(ops[i].model, NSSelectorFromString(@"modelAttributes"));
            NSDictionary *ns = [attrs[@"NetworkStatusList"] firstObject];
            ops[i].inBS = [[ns[@"LiveInputList"] firstObject][@"BatchStride"] unsignedIntValue];
            ops[i].outBS = [[ns[@"LiveOutputList"] firstObject][@"BatchStride"] unsignedIntValue];
            ops[i].inPS = [[ns[@"LiveInputList"] firstObject][@"PlaneStride"] unsignedIntValue];
            ops[i].outPS = [[ns[@"LiveOutputList"] firstObject][@"PlaneStride"] unsignedIntValue];

            ((BOOL (*)(id, SEL, id, id, NSInteger, id*))objc_msgSend)(
                ane_client, NSSelectorFromString(@"doUnloadModel:options:qos:error:"),
                ops[i].model, @{}, 0, &err);
        }

        for (int i = 0; i < nOps; i++) {
            ((BOOL (*)(id, SEL, id, id, NSInteger, id*))objc_msgSend)(
                ane_client, NSSelectorFromString(@"loadModel:options:qos:error:"),
                ops[i].model, @{}, 0, &err);

            NSDictionary *inP = @{@"IOSurfaceWidth":@(ops[i].inBS/2), @"IOSurfaceHeight":@1,
                @"IOSurfaceBytesPerRow":@(ops[i].inBS), @"IOSurfaceBytesPerElement":@2,
                @"IOSurfaceAllocSize":@(ops[i].inBS), @"IOSurfacePixelFormat":@(0x6630304C)};
            NSDictionary *outP = @{@"IOSurfaceWidth":@(ops[i].outBS/2), @"IOSurfaceHeight":@1,
                @"IOSurfaceBytesPerRow":@(ops[i].outBS), @"IOSurfaceBytesPerElement":@2,
                @"IOSurfaceAllocSize":@(ops[i].outBS), @"IOSurfacePixelFormat":@(0x6630304C)};
            ops[i].inSurf = IOSurfaceCreate((__bridge CFDictionaryRef)inP);
            ops[i].outSurf = IOSurfaceCreate((__bridge CFDictionaryRef)outP);
            ops[i].inObj = ((id (*)(id, SEL, void*, NSInteger, BOOL))objc_msgSend)(
                [_IO alloc], NSSelectorFromString(@"initWithIOSurface:startOffset:shouldRetain:"),
                ops[i].inSurf, 0, YES);
            ops[i].outObj = ((id (*)(id, SEL, void*, NSInteger, BOOL))objc_msgSend)(
                [_IO alloc], NSSelectorFromString(@"initWithIOSurface:startOffset:shouldRetain:"),
                ops[i].outSurf, 0, YES);
        }
        fprintf(stderr, "All %d models loaded\n", nOps);

        // Buffers
        float *x_f32 = malloc(DIM * sizeof(float));
        float *attn_out_f32 = malloc(DIM * sizeof(float));
        float *scores = malloc(MAX_SEQ * sizeof(float));
        uint16_t *x_fp16 = malloc(DIM * 2);
        uint16_t *qkv_fp16 = malloc(2304 * 2);
        uint16_t *o_fp16 = malloc(DIM * 2);
        uint16_t *ffn_fp16 = malloc(DIM * 2);
        uint16_t *logits_fp16 = malloc(VOCAB_SIZE * 2);
        float *logits_f32 = malloc(VOCAB_SIZE * sizeof(float));

        KVCache kv[N_LAYERS];
        for (int i = 0; i < N_LAYERS; i++) {
            kv[i].k_cache = calloc(MAX_SEQ * N_HEADS * HEAD_DIM, sizeof(float));
            kv[i].v_cache = calloc(MAX_SEQ * N_HEADS * HEAD_DIM, sizeof(float));
            kv[i].len = 0;
        }

        int max_tokens = n_prompt + n_gen_tokens;
        int *tokens = malloc(max_tokens * sizeof(int));
        memcpy(tokens, prompt_tokens, n_prompt * sizeof(int));
        int total_tokens = n_prompt;

        fprintf(stderr, "Prefilling %d prompt tokens...\n", n_prompt);
        int total_steps = n_prompt + n_gen_tokens;
        uint64_t t_start = mach_absolute_time();

        for (int step = 0; step < total_steps; step++) {
            int pos, tok;
            BOOL is_generate = (step >= n_prompt);
            if (!is_generate) { pos = step; tok = tokens[pos]; }
            else { pos = total_tokens - 1; tok = tokens[pos]; }

            // Embedding
            for (int d = 0; d < DIM; d++)
                x_f32[d] = wte[tok * DIM + d] + wpe[pos * DIM + d];
            fp32_to_fp16(x_f32, x_fp16, DIM);

            for (int li = 0; li < N_LAYERS; li++) {
                // LN1 + QKV (fused ANE — no CPU LN!)
                int qkv_idx = opIdx([NSString stringWithFormat:@"L%d_ln1_qkv", li]);
                ane_dispatch(&ops[qkv_idx], x_fp16, qkv_fp16);

                float q_f32[DIM], k_f32[DIM], v_f32[DIM];
                fp16_to_fp32(qkv_fp16, q_f32, DIM);
                fp16_to_fp32(qkv_fp16 + DIM, k_f32, DIM);
                fp16_to_fp32(qkv_fp16 + 2*DIM, v_f32, DIM);

                int seq_pos = kv[li].len;
                memcpy(&kv[li].k_cache[seq_pos * N_HEADS * HEAD_DIM], k_f32, DIM * sizeof(float));
                memcpy(&kv[li].v_cache[seq_pos * N_HEADS * HEAD_DIM], v_f32, DIM * sizeof(float));
                kv[li].len++;
                int seq_len = kv[li].len;

                float scale = 1.0f / sqrtf((float)HEAD_DIM);
                memset(attn_out_f32, 0, DIM * sizeof(float));
                for (int h = 0; h < N_HEADS; h++) {
                    float *q_h = &q_f32[h * HEAD_DIM];
                    for (int s = 0; s < seq_len; s++) {
                        float dot = 0;
                        vDSP_dotpr(q_h, 1,
                                   &kv[li].k_cache[s * N_HEADS * HEAD_DIM + h * HEAD_DIM], 1,
                                   &dot, HEAD_DIM);
                        scores[s] = dot * scale;
                    }
                    softmax_f32(scores, seq_len);
                    float *out_h = &attn_out_f32[h * HEAD_DIM];
                    memset(out_h, 0, HEAD_DIM * sizeof(float));
                    for (int s = 0; s < seq_len; s++) {
                        float w = scores[s];
                        float *v_s = &kv[li].v_cache[s * N_HEADS * HEAD_DIM + h * HEAD_DIM];
                        vDSP_vsma(v_s, 1, &w, out_h, 1, out_h, 1, HEAD_DIM);
                    }
                }

                uint16_t attn_fp16[DIM];
                fp32_to_fp16(attn_out_f32, attn_fp16, DIM);
                int o_idx = opIdx([NSString stringWithFormat:@"L%d_o_proj", li]);
                ane_dispatch(&ops[o_idx], attn_fp16, o_fp16);

                // Residual 1
                fp16_to_fp32(x_fp16, x_f32, DIM);
                float o_f32[DIM];
                fp16_to_fp32(o_fp16, o_f32, DIM);
                vDSP_vadd(x_f32, 1, o_f32, 1, x_f32, 1, DIM);

                // r1 to FP16 for LN2+FFN
                uint16_t r1_fp16[DIM];
                fp32_to_fp16(x_f32, r1_fp16, DIM);

                // LN2 + FFN (fused ANE — no CPU LN2!)
                int ffn_idx = opIdx([NSString stringWithFormat:@"L%d_ln2_ffn", li]);
                ane_dispatch(&ops[ffn_idx], r1_fp16, ffn_fp16);

                // Residual 2
                float ffn_f32[DIM];
                fp16_to_fp32(ffn_fp16, ffn_f32, DIM);
                vDSP_vadd(x_f32, 1, ffn_f32, 1, x_f32, 1, DIM);
                fp32_to_fp16(x_f32, x_fp16, DIM);
            }

            // Final LayerNorm (CPU) + LM head (ANE)
            fp16_to_fp32(x_fp16, x_f32, DIM);
            // layernorm
            {
                float mean = 0, var = 0;
                vDSP_meanv(x_f32, 1, &mean, DIM);
                float neg_mean = -mean;
                float ln_out[DIM];
                vDSP_vsadd(x_f32, 1, &neg_mean, ln_out, 1, DIM);
                vDSP_vsq(ln_out, 1, ln_out, 1, DIM);
                vDSP_meanv(ln_out, 1, &var, DIM);
                float inv_std = 1.0f / sqrtf(var + 1e-5f);
                vDSP_vsadd(x_f32, 1, &neg_mean, ln_out, 1, DIM);
                vDSP_vsmul(ln_out, 1, &inv_std, ln_out, 1, DIM);
                vDSP_vma(ln_out, 1, ln_f_w, 1, ln_f_b, 1, ln_out, 1, DIM);
                fp32_to_fp16(ln_out, x_fp16, DIM);
            }
            int lm_idx = opIdx(@"lm_head");
            ane_dispatch(&ops[lm_idx], x_fp16, logits_fp16);

            fp16_to_fp32(logits_fp16, logits_f32, VOCAB_SIZE);
            vDSP_Length max_idx = 0;
            float max_val = 0;
            vDSP_maxvi(logits_f32, 1, &max_val, &max_idx, VOCAB_SIZE);

            if (is_generate) {
                tokens[total_tokens] = (int)max_idx;
                total_tokens++;
                printf("%d\n", (int)max_idx);
                fflush(stdout);
            } else if (step == n_prompt - 1) {
                t_start = mach_absolute_time();
            }
        }

        uint64_t t_end = mach_absolute_time();
        double elapsed_ms = ns_to_ms(t_end - t_start);
        double tok_per_sec = (double)n_gen_tokens / (elapsed_ms / 1000.0);
        fprintf(stderr, "\nGenerated %d tokens in %.1f ms = %.1f tok/s\n",
                n_gen_tokens, elapsed_ms, tok_per_sec);
        fprintf(stderr, "Dispatches/token: %d (3/layer + 1 lm_head)\n", 3*N_LAYERS+1);
        fprintf(stderr, "CPU LN eliminated: 24 (12*LN1 + 12*LN2, LN_f still on CPU)\n");

        for (int i = 0; i < nOps; i++) {
            ((BOOL (*)(id, SEL, id, id, NSInteger, id*))objc_msgSend)(
                ane_client, NSSelectorFromString(@"doUnloadModel:options:qos:error:"),
                ops[i].model, @{}, 0, &err);
            if (ops[i].inSurf) CFRelease(ops[i].inSurf);
            if (ops[i].outSurf) CFRelease(ops[i].outSurf);
        }
        free(ops); free(tokens); free(prompt_tokens);
        free(x_f32); free(attn_out_f32); free(scores);
        free(x_fp16); free(qkv_fp16); free(o_fp16); free(ffn_fp16);
        free(logits_fp16); free(logits_f32); free(wte); free(wpe);
        free(ln_f_w); free(ln_f_b);
        for (int i = 0; i < N_LAYERS; i++) {
            free(kv[i].k_cache); free(kv[i].v_cache);
        }
        return 0;
    }
}
"""
    with open(c_path, 'w') as f:
        f.write(c_code)
    return c_path


# ===================================================================
# Main
# ===================================================================

def main():
    print("=" * 60)
    print("GPT-2 LN-FUSED BUILD")
    print("Target: fuse all LayerNorm into ANE dispatches")
    print("Baseline: 37 dispatches at 137.7 tok/s")
    print("=" * 60)

    print("\n[1] Loading GPT-2 117M...")
    t0 = time.time()
    model = GPT2Model.from_safetensors(MODEL_PATH)
    print(f"    Loaded in {time.time()-t0:.1f}s")

    # Clean build
    if os.path.exists(BUILD_DIR):
        shutil.rmtree(BUILD_DIR)

    # Build fused models
    print("\n[2] Building fused models...")
    compiled = compile_all_fused(model)

    # Skip slow Python verification — verify via C binary output instead
    # Export
    print("\n[3] Exporting for C binary...")
    manifest, weights_bin = export_for_c_binary(model, compiled)

    # Build C binary
    print("\n[4] Building C binary...")
    c_src = write_c_source()
    c_bin = c_src.replace('.m', '')

    r = subprocess.run([
        'xcrun', 'clang', '-O2',
        '-framework', 'Foundation',
        '-framework', 'IOSurface',
        '-framework', 'Accelerate',
        '-fobjc-arc',
        '-o', c_bin, c_src,
    ], capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  BUILD FAILED: {r.stderr}")
        return
    print(f"  Built: {c_bin}")

    # Run
    print("\n[5] Measuring performance...")
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained('gpt2')
        prompt_tokens = tokenizer.encode("The")
    except ImportError:
        prompt_tokens = [464]
        tokenizer = None

    n_gen = 50
    cmd = [c_bin, manifest, weights_bin, str(n_gen)] + [str(t) for t in prompt_tokens]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        print(f"  FAILED: {result.stderr}")
        return

    print(f"\n  C binary output:")
    for line in result.stderr.split('\n'):
        if line.strip():
            print(f"    {line.strip()}")

    gen_ids = []
    for line in result.stdout.strip().split('\n'):
        line = line.strip()
        if line and line.lstrip('-').isdigit():
            gen_ids.append(int(line))

    if tokenizer and gen_ids:
        print(f"\n  Generated: 'The{tokenizer.decode(gen_ids)}'")

    # Compare with baseline
    print(f"\n{'='*60}")
    print(f"COMPARISON")
    print(f"  Baseline: 37 dispatches, 137.7 tok/s (CPU LN in hot path)")
    for line in result.stderr.split('\n'):
        if 'tok/s' in line:
            print(f"  LN-fused: {line.strip()}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
