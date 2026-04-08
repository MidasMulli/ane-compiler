#!/usr/bin/env python3
"""
Build Llama-3.1-8B-Instruct as a SINGLE espresso graph (bypassing protobuf limits).

Strategy:
- Generate .mlmodelc directly in espresso format (JSON net + binary weights)
- No coremltools protobuf serialization (2GB limit bypass)
- CoreML loads the .mlmodelc and aned compiles to .hwx at runtime

For seq_len=1 decode: attention simplifies to V→VO (softmax of scalar = 1.0)

Copyright 2026 Nick Lo. MIT License.
"""

import os
import sys
import json
import struct
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))


def _make_blob_data(name, blob_type, shape_or_dim, weights, dtype):
    """Generate binary data for a single weight blob."""
    if blob_type == 'batchnorm':
        dim_bn = shape_or_dim
        if weights and name in weights:
            gamma = weights[name].astype(np.float32)
        else:
            gamma = np.ones(dim_bn, dtype=np.float32)
        beta = np.zeros(dim_bn, dtype=np.float32)
        mean = np.zeros(dim_bn, dtype=np.float32)
        var = np.ones(dim_bn, dtype=np.float32)
        bn_data = np.column_stack([gamma, beta, mean, var]).flatten()
        return bn_data.astype(dtype).tobytes()
    else:
        out_ch, in_ch = shape_or_dim
        if weights and name in weights:
            w = weights[name].reshape(out_ch, in_ch).astype(dtype)
        else:
            w = (np.random.randn(out_ch, in_ch) * 0.01).astype(dtype)
        return w.tobytes()


def generate_llama_espresso(output_dir, n_layers, dim=4096, ffn_dim=14336,
                             n_kv_heads=8, head_dim=128, weights=None,
                             use_fp16_weights=False):
    """Generate .mlmodelc in raw espresso format for Llama-8B.

    Bypasses coremltools protobuf 2GB limit by writing espresso format directly.

    Args:
        output_dir: .mlmodelc output path
        n_layers: number of transformer layers
        dim: hidden dimension (4096)
        ffn_dim: FFN intermediate dimension (14336)
        n_kv_heads: number of KV heads (8)
        head_dim: head dimension (128)
        weights: dict mapping weight names to numpy arrays (FP32)
        use_fp16_weights: if True, store weights as FP16 (halves file size)
    """
    kv_dim = n_kv_heads * head_dim  # 1024

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'model'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'analytics'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'neural_network_optionals'), exist_ok=True)

    # ---------------------------------------------------------------
    # Build espresso layer graph
    # ---------------------------------------------------------------
    layers = []
    shapes = {}
    blob_idx = 1  # blob indices: 1, 3, 5, ...  (odd numbers)
    weight_blobs = []  # (name, shape, blob_id)

    # Input shape
    shapes['input'] = {"n": 1, "h": 1, "w": 1, "k": dim}

    for i in range(n_layers):
        pfx = f'L{i}'
        prev = 'input' if i == 0 else f'L{i-1}_out'

        # --- RMSNorm 1: MVN ---
        layers.append({
            "type": "l2_normalize",
            "name": f"{pfx}_ln1_mvn",
            "debug_info": f"{pfx}_ln1_mvn",
            "bottom": prev,
            "top": f"{pfx}_ln1_mvn_out",
            "normalization_mode": 1,
            "axis": 2,
            "eps": 1e-5,
            "weights": {},
        })
        shapes[f"{pfx}_ln1_mvn_out"] = {"n": 1, "h": 1, "w": 1, "k": dim}

        # --- RMSNorm 1: Batchnorm (scale) ---
        layers.append({
            "type": "batchnorm",
            "name": f"{pfx}_ln1_bn",
            "debug_info": f"{pfx}_ln1_bn",
            "bottom": f"{pfx}_ln1_mvn_out",
            "top": f"{pfx}_ln1_out",
            "blob_batchnorm_params": blob_idx,
            "C": dim,
            "weights": {},
        })
        weight_blobs.append((f'L{i}_ln1_bn', 'batchnorm', dim, blob_idx))
        blob_idx += 2
        shapes[f"{pfx}_ln1_out"] = {"n": 1, "h": 1, "w": 1, "k": dim}

        # --- V projection: dim -> kv_dim ---
        layers.append({
            "type": "inner_product",
            "name": f"{pfx}_v",
            "debug_info": f"{pfx}_v",
            "bottom": f"{pfx}_ln1_out",
            "top": f"{pfx}_v_out",
            "nB": dim,
            "nC": kv_dim,
            "has_biases": 0,
            "blob_weights": blob_idx,
            "has_relu": 0, "has_tanh": 0, "has_prelu": 0,
            "weights": {},
        })
        weight_blobs.append((f'L{i}_v_proj', 'inner_product', (kv_dim, dim), blob_idx))
        blob_idx += 2
        shapes[f"{pfx}_v_out"] = {"n": 1, "h": 1, "w": 1, "k": kv_dim}

        # --- VO projection: kv_dim -> dim ---
        layers.append({
            "type": "inner_product",
            "name": f"{pfx}_vo",
            "debug_info": f"{pfx}_vo",
            "bottom": f"{pfx}_v_out",
            "top": f"{pfx}_attn_out",
            "nB": kv_dim,
            "nC": dim,
            "has_biases": 0,
            "blob_weights": blob_idx,
            "has_relu": 0, "has_tanh": 0, "has_prelu": 0,
            "weights": {},
        })
        weight_blobs.append((f'L{i}_vo_proj', 'inner_product', (dim, kv_dim), blob_idx))
        blob_idx += 2
        shapes[f"{pfx}_attn_out"] = {"n": 1, "h": 1, "w": 1, "k": dim}

        # --- Residual 1 ---
        layers.append({
            "type": "elementwise",
            "name": f"{pfx}_res1",
            "bottom": f"{prev},{pfx}_attn_out",
            "top": f"{pfx}_r1_out",
            "operation": 0,  # ADD
            "alpha": 1, "beta": 0, "fused_relu": 0,
            "weights": {},
        })
        shapes[f"{pfx}_r1_out"] = {"n": 1, "h": 1, "w": 1, "k": dim}

        # --- RMSNorm 2: MVN ---
        layers.append({
            "type": "l2_normalize",
            "name": f"{pfx}_ln2_mvn",
            "debug_info": f"{pfx}_ln2_mvn",
            "bottom": f"{pfx}_r1_out",
            "top": f"{pfx}_ln2_mvn_out",
            "normalization_mode": 1,
            "axis": 2,
            "eps": 1e-5,
            "weights": {},
        })
        shapes[f"{pfx}_ln2_mvn_out"] = {"n": 1, "h": 1, "w": 1, "k": dim}

        # --- RMSNorm 2: Batchnorm (scale) ---
        layers.append({
            "type": "batchnorm",
            "name": f"{pfx}_ln2_bn",
            "debug_info": f"{pfx}_ln2_bn",
            "bottom": f"{pfx}_ln2_mvn_out",
            "top": f"{pfx}_ln2_out",
            "blob_batchnorm_params": blob_idx,
            "C": dim,
            "weights": {},
        })
        weight_blobs.append((f'L{i}_ln2_bn', 'batchnorm', dim, blob_idx))
        blob_idx += 2
        shapes[f"{pfx}_ln2_out"] = {"n": 1, "h": 1, "w": 1, "k": dim}

        # --- Gate projection: dim -> ffn_dim ---
        layers.append({
            "type": "inner_product",
            "name": f"{pfx}_gate",
            "debug_info": f"{pfx}_gate",
            "bottom": f"{pfx}_ln2_out",
            "top": f"{pfx}_gate_out",
            "nB": dim,
            "nC": ffn_dim,
            "has_biases": 0,
            "blob_weights": blob_idx,
            "has_relu": 0, "has_tanh": 0, "has_prelu": 0,
            "weights": {},
        })
        weight_blobs.append((f'L{i}_gate_proj', 'inner_product', (ffn_dim, dim), blob_idx))
        blob_idx += 2
        shapes[f"{pfx}_gate_out"] = {"n": 1, "h": 1, "w": 1, "k": ffn_dim}

        # --- Sigmoid (for SiLU) ---
        # espresso mode 3 = sigmoid (coremltools encoding)
        layers.append({
            "type": "activation",
            "name": f"{pfx}_sigmoid",
            "bottom": f"{pfx}_gate_out",
            "top": f"{pfx}_sig_out",
            "mode": 3,  # sigmoid in espresso encoding
            "weights": {},
        })
        shapes[f"{pfx}_sig_out"] = {"n": 1, "h": 1, "w": 1, "k": ffn_dim}

        # --- SiLU multiply: gate * sigmoid(gate) ---
        layers.append({
            "type": "elementwise",
            "name": f"{pfx}_silu",
            "bottom": f"{pfx}_gate_out,{pfx}_sig_out",
            "top": f"{pfx}_silu_out",
            "operation": 1,  # MUL
            "alpha": 1, "beta": 0, "fused_relu": 0,
            "weights": {},
        })
        shapes[f"{pfx}_silu_out"] = {"n": 1, "h": 1, "w": 1, "k": ffn_dim}

        # --- Up projection: dim -> ffn_dim ---
        layers.append({
            "type": "inner_product",
            "name": f"{pfx}_up",
            "debug_info": f"{pfx}_up",
            "bottom": f"{pfx}_ln2_out",
            "top": f"{pfx}_up_out",
            "nB": dim,
            "nC": ffn_dim,
            "has_biases": 0,
            "blob_weights": blob_idx,
            "has_relu": 0, "has_tanh": 0, "has_prelu": 0,
            "weights": {},
        })
        weight_blobs.append((f'L{i}_up_proj', 'inner_product', (ffn_dim, dim), blob_idx))
        blob_idx += 2
        shapes[f"{pfx}_up_out"] = {"n": 1, "h": 1, "w": 1, "k": ffn_dim}

        # --- SwiGLU multiply: SiLU(gate) * up ---
        layers.append({
            "type": "elementwise",
            "name": f"{pfx}_swiglu",
            "bottom": f"{pfx}_silu_out,{pfx}_up_out",
            "top": f"{pfx}_sw_out",
            "operation": 1,  # MUL
            "alpha": 1, "beta": 0, "fused_relu": 0,
            "weights": {},
        })
        shapes[f"{pfx}_sw_out"] = {"n": 1, "h": 1, "w": 1, "k": ffn_dim}

        # --- Down projection: ffn_dim -> dim ---
        layers.append({
            "type": "inner_product",
            "name": f"{pfx}_down",
            "debug_info": f"{pfx}_down",
            "bottom": f"{pfx}_sw_out",
            "top": f"{pfx}_ffn_out",
            "nB": ffn_dim,
            "nC": dim,
            "has_biases": 0,
            "blob_weights": blob_idx,
            "has_relu": 0, "has_tanh": 0, "has_prelu": 0,
            "weights": {},
        })
        weight_blobs.append((f'L{i}_down_proj', 'inner_product', (dim, ffn_dim), blob_idx))
        blob_idx += 2
        shapes[f"{pfx}_ffn_out"] = {"n": 1, "h": 1, "w": 1, "k": dim}

        # --- Residual 2 ---
        layers.append({
            "type": "elementwise",
            "name": f"{pfx}_res2",
            "bottom": f"{pfx}_r1_out,{pfx}_ffn_out",
            "top": f"{pfx}_out",
            "operation": 0,  # ADD
            "alpha": 1, "beta": 0, "fused_relu": 0,
            "weights": {},
        })
        shapes[f"{pfx}_out"] = {"n": 1, "h": 1, "w": 1, "k": dim}

    # Mark last layer as output
    last_layer_name = f'L{n_layers-1}_out'
    # Add identity to map to 'output'
    layers.append({
        "type": "activation",
        "name": "final_identity",
        "bottom": last_layer_name,
        "top": "output",
        "mode": 6,  # linear (espresso mode 6)
        "beta": 0,
        "weights": {},
        "attributes": {"is_output": 1},
    })
    shapes["output"] = {"n": 1, "h": 1, "w": 1, "k": dim}

    print(f"  Espresso layers: {len(layers)}")
    print(f"  Weight blobs: {len(weight_blobs)}")
    print(f"  Max blob_id: {blob_idx - 2}")

    # ---------------------------------------------------------------
    # Write model.espresso.net (JSON)
    # ---------------------------------------------------------------
    net = {
        "storage": "model.espresso.weights",
        "analyses": {},
        "properties": {},
        "format_version": 200,
        "metadata_in_weights": [],
        "layers": layers,
    }
    with open(os.path.join(output_dir, 'model.espresso.net'), 'w') as f:
        json.dump(net, f)

    # ---------------------------------------------------------------
    # Write model.espresso.shape (JSON)
    # ---------------------------------------------------------------
    with open(os.path.join(output_dir, 'model.espresso.shape'), 'w') as f:
        json.dump({"layer_shapes": shapes}, f)

    # ---------------------------------------------------------------
    # Write model.espresso.weights (binary, v28 format)
    # ---------------------------------------------------------------
    # v28 format (matches coremltools output):
    #   Header: version=28, blob_table_offset=0x38
    #   Blob table: N entries, each 0x20 bytes: (blob_id, 0, size, 0, next_id, 0, 0, 0)
    #   Header padded to 0x200 alignment
    #   Data: first_gap (first batchnorm blob) + sequential remaining blobs

    dtype = np.float16 if use_fp16_weights else np.float32
    bytes_per_elem = 2 if use_fp16_weights else 4

    print(f"  Writing weights ({dtype.__name__}, v28 format)...")
    t0 = time.time()

    weight_path = os.path.join(output_dir, 'model.espresso.weights')

    # Calculate blob sizes
    # Key format rules (from coremltools v28 format):
    # 1. header[0] = total_weight_blobs * 2
    # 2. Batchnorm blobs stored at 4x their raw data size in blob table
    # 3. next_id in blob table = blob_id + 1 (sequential, not skipping)
    total_weight_blobs = len(weight_blobs)

    blob_info = []  # (name, blob_type, shape_or_dim, blob_id, raw_size, table_size)
    for name, blob_type, shape_or_dim, bid in weight_blobs:
        if blob_type == 'batchnorm':
            raw_size = shape_or_dim * 4 * bytes_per_elem
            # Batchnorm table entry size = 4x raw size (coremltools convention)
            table_size = raw_size * 4
        else:
            out_ch, in_ch = shape_or_dim
            raw_size = out_ch * in_ch * bytes_per_elem
            table_size = raw_size
        blob_info.append((name, blob_type, shape_or_dim, bid, raw_size, table_size))

    # First blob is batchnorm (goes in the "gap")
    first_raw_size = blob_info[0][4]
    first_gap = blob_info[0][5]  # 4x raw size for batchnorm

    # Remaining blobs
    remaining_blobs = blob_info[1:]
    total_remaining = sum(b[5] for b in remaining_blobs)

    # Blob table: one entry per remaining blob, plus a zero-terminator entry
    n_entries = len(remaining_blobs)
    blob_table_offset = 0x38
    # Add 0x20 for a zero-terminator entry after the last blob table entry
    header_raw_end = blob_table_offset + (n_entries + 1) * 0x20
    # Pad header to 0x200 alignment minimum
    header_padded = max(0x200, ((header_raw_end + 0x1FF) // 0x200) * 0x200)

    total_file = header_padded + first_gap + total_remaining
    print(f"  Total weight file: {total_file/1e9:.2f} GB")
    print(f"  Header: {header_padded} bytes, gap: {first_gap}, "
          f"remaining: {total_remaining/1e9:.2f} GB, blobs: {total_weight_blobs}")

    with open(weight_path, 'wb') as f:
        header = bytearray(header_padded)
        # header[0] = total_weight_blobs * 2 (coremltools convention)
        struct.pack_into('<I', header, 0x00, total_weight_blobs * 2)
        struct.pack_into('<I', header, 0x10, blob_table_offset)
        struct.pack_into('<I', header, 0x18, 1)  # num blob groups
        struct.pack_into('<I', header, 0x20, first_gap)  # first gap size
        struct.pack_into('<I', header, 0x28, 2)  # num data entries

        # Blob table entries for remaining blobs
        for i, (name, blob_type, shape_or_dim, bid, raw_size, table_size) in enumerate(remaining_blobs):
            off = blob_table_offset + i * 0x20
            # next_id = bid + 1 for non-last entries, 0 for last
            next_id = bid + 1 if i + 1 < len(remaining_blobs) else 0
            struct.pack_into('<I', header, off + 0, bid)
            struct.pack_into('<I', header, off + 8, table_size)
            struct.pack_into('<I', header, off + 16, next_id)

        f.write(header)

        # First blob (batchnorm in gap)
        name, blob_type, shape_or_dim, bid, raw_size, table_size = blob_info[0]
        blob_data = _make_blob_data(name, blob_type, shape_or_dim, weights, dtype)
        f.write(blob_data)
        f.write(b'\x00' * (first_gap - len(blob_data)))

        # Remaining blobs
        for idx, (name, blob_type, shape_or_dim, bid, raw_size, table_size) in enumerate(remaining_blobs):
            blob_data = _make_blob_data(name, blob_type, shape_or_dim, weights, dtype)
            f.write(blob_data)
            # Pad batchnorm blobs to their table_size
            if table_size > len(blob_data):
                f.write(b'\x00' * (table_size - len(blob_data)))
            if (idx + 1) % 10 == 0:
                print(f"    Written {idx+1}/{len(remaining_blobs)} blobs...")

    dt = time.time() - t0
    actual_size = os.path.getsize(weight_path)
    print(f"  Weights written: {actual_size/1e9:.2f} GB in {dt:.1f}s")

    # ---------------------------------------------------------------
    # Write metadata.json
    # ---------------------------------------------------------------
    meta = {
        "specificationVersion": 4,
        "isUpdatable": False,
        "modelType": {"name": "MLModelType_neuralNetwork"},
        "computePrecision": "Float16",
        "inputSchema": [{"name": "input", "type": "MultiArray",
                          "shape": [dim, 1, 1], "dataType": "Float16"}],
        "outputSchema": [{"name": "output", "type": "MultiArray",
                           "shape": [dim, 1, 1], "dataType": "Float16"}],
    }
    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump(meta, f)

    # ---------------------------------------------------------------
    # Write coremldata.bin (minimal protobuf stub)
    # ---------------------------------------------------------------
    with open(os.path.join(output_dir, 'coremldata.bin'), 'wb') as f:
        f.write(b'\x08\x04')

    for sub in ['model', 'analytics', 'neural_network_optionals']:
        with open(os.path.join(output_dir, sub, 'coremldata.bin'), 'wb') as f:
            f.write(b'\x00' * 64)

    print(f"  .mlmodelc generated at {output_dir}")
    return output_dir


def build_with_real_weights(n_layers=32, output_dir='/tmp/llama8b_espresso',
                             use_fp16=True):
    """Build with real Llama-8B weights."""
    model_path = os.path.expanduser(
        '~/.cache/huggingface/hub/models--unsloth--Meta-Llama-3.1-8B-Instruct/'
        'snapshots/a2856192dd7c25b842431f39c179a6c2c2f627d1'
    )

    print(f"Loading Llama-8B weights from {model_path}...")
    t0 = time.time()
    from llama_loader import LlamaModel
    model = LlamaModel.from_safetensors(model_path)
    dt = time.time() - t0
    print(f"  Loaded in {dt:.1f}s")

    # Build weight dict
    weights = {}
    for i in range(min(n_layers, model.config.n_layers)):
        L = model.layers[i]
        weights[f'L{i}_ln1_bn'] = L.input_layernorm_weight
        weights[f'L{i}_ln2_bn'] = L.post_attention_layernorm_weight
        weights[f'L{i}_v_proj'] = L.v_proj_weight

        # Combined V→O projection
        n_rep = model.config.n_rep
        kv_dim = model.config.n_kv_heads * model.config.head_dim
        expand = np.zeros((model.config.hidden_size, kv_dim), dtype=np.float32)
        for kv_h in range(model.config.n_kv_heads):
            for r in range(n_rep):
                q_h = kv_h * n_rep + r
                expand[q_h * model.config.head_dim:(q_h+1) * model.config.head_dim,
                       kv_h * model.config.head_dim:(kv_h+1) * model.config.head_dim] = \
                    np.eye(model.config.head_dim, dtype=np.float32)
        vo_weight = L.o_proj_weight @ expand
        weights[f'L{i}_vo_proj'] = vo_weight

        weights[f'L{i}_gate_proj'] = L.gate_proj_weight
        weights[f'L{i}_up_proj'] = L.up_proj_weight
        weights[f'L{i}_down_proj'] = L.down_proj_weight

    return generate_llama_espresso(
        output_dir, n_layers,
        dim=model.config.hidden_size,
        ffn_dim=model.config.intermediate_size,
        n_kv_heads=model.config.n_kv_heads,
        head_dim=model.config.head_dim,
        weights=weights,
        use_fp16_weights=use_fp16,
    )


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--layers', type=int, default=1)
    parser.add_argument('--real-weights', action='store_true')
    parser.add_argument('--fp16', action='store_true', default=True)
    parser.add_argument('--fp32', action='store_true')
    parser.add_argument('--output-dir', default='/tmp/llama8b_espresso')
    args = parser.parse_args()

    use_fp16 = not args.fp32

    if args.real_weights:
        result = build_with_real_weights(args.layers, args.output_dir, use_fp16)
    else:
        result = generate_llama_espresso(
            args.output_dir + '.mlmodelc', args.layers,
            use_fp16_weights=use_fp16,
        )

    print(f"\nOutput: {result}")
