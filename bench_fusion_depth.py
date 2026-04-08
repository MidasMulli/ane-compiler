#!/usr/bin/env python3
"""
Benchmark: GPT-2 fusion depth vs throughput.

Thesis: fusion depth determines SRAM residency, which determines
effective bandwidth, which determines tok/s.

Builds progressively deeper fused models in raw espresso format:
  Config A: 1 layer  (V->O + FFN fused) = 1 dispatch
  Config B: 3 layers fused = 1 dispatch
  Config C: 6 layers fused = 1 dispatch
  Config D: 12 layers fused = 1 dispatch
  Config E: 12 layers + lm_head = 1 dispatch (entire model)

For throughput testing: no biases (matches Llama 8B format which is proven).
V->O shortcut: attention = O_proj(V_proj(ln1(x))) + x

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

MODEL_PATH = os.path.expanduser(
    "~/.cache/huggingface/hub/models--openai-community--gpt2/"
    "snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e/model.safetensors"
)
PIPE_TOOL = os.path.join(os.path.dirname(__file__), 'tests', 'ane_standalone_pipe')
BUILD_DIR = '/tmp/gpt2_fusion_bench'


# ===================================================================
# Build fused GPT-2 in raw espresso format (no coremltools)
# ===================================================================

def build_fused_gpt2_espresso(output_dir, n_layers, model,
                               include_lm_head=False):
    """Build fused GPT-2 model in raw espresso format.

    Uses the same pattern as the proven Llama 8B espresso builder:
    - Unbiased inner_product layers (bias not needed for throughput test)
    - v28 weight format with stride-2 blob indices
    - GELU via espresso mode 19
    - MVN + batchnorm for LayerNorm

    Each layer:
      LN1(MVN+BN) -> V_proj -> O_proj -> residual ->
      LN2(MVN+BN) -> fc_up -> GELU -> fc_down -> residual
    """
    dim = 768
    ffn_dim = 3072
    vocab_size = 50257

    os.makedirs(output_dir, exist_ok=True)
    for sub in ['model', 'analytics', 'neural_network_optionals']:
        os.makedirs(os.path.join(output_dir, sub), exist_ok=True)

    layers_json = []
    shapes = {}
    blob_idx = 1
    weight_blobs = []  # (name, type, shape_or_dim, blob_id)

    shapes['input'] = {"n": 1, "h": 1, "w": 1, "k": dim}

    for i in range(n_layers):
        pfx = f'L{i}'
        prev = 'input' if i == 0 else f'L{i-1}_out'

        # --- LN1: MVN ---
        layers_json.append({
            "type": "l2_normalize", "name": f"{pfx}_ln1_mvn",
            "debug_info": f"{pfx}_ln1_mvn",
            "bottom": prev, "top": f"{pfx}_ln1_mvn_out",
            "normalization_mode": 1, "axis": 2, "eps": 1e-5,
            "weights": {},
        })
        shapes[f"{pfx}_ln1_mvn_out"] = {"n": 1, "h": 1, "w": 1, "k": dim}

        # --- LN1: BN (affine) ---
        layers_json.append({
            "type": "batchnorm", "name": f"{pfx}_ln1_bn",
            "debug_info": f"{pfx}_ln1_bn",
            "bottom": f"{pfx}_ln1_mvn_out", "top": f"{pfx}_ln1_out",
            "blob_batchnorm_params": blob_idx, "C": dim, "weights": {},
        })
        weight_blobs.append((f'{pfx}_ln1_bn', 'batchnorm', dim, blob_idx))
        blob_idx += 2
        shapes[f"{pfx}_ln1_out"] = {"n": 1, "h": 1, "w": 1, "k": dim}

        # --- V projection: dim -> dim (no bias for throughput test) ---
        layers_json.append({
            "type": "inner_product", "name": f"{pfx}_v",
            "debug_info": f"{pfx}_v",
            "bottom": f"{pfx}_ln1_out", "top": f"{pfx}_v_out",
            "nB": dim, "nC": dim, "has_biases": 0,
            "blob_weights": blob_idx,
            "has_relu": 0, "has_tanh": 0, "has_prelu": 0,
            "weights": {},
        })
        weight_blobs.append((f'{pfx}_v', 'ip', (dim, dim), blob_idx))
        blob_idx += 2
        shapes[f"{pfx}_v_out"] = {"n": 1, "h": 1, "w": 1, "k": dim}

        # --- O projection: dim -> dim ---
        layers_json.append({
            "type": "inner_product", "name": f"{pfx}_o",
            "debug_info": f"{pfx}_o",
            "bottom": f"{pfx}_v_out", "top": f"{pfx}_attn_out",
            "nB": dim, "nC": dim, "has_biases": 0,
            "blob_weights": blob_idx,
            "has_relu": 0, "has_tanh": 0, "has_prelu": 0,
            "weights": {},
        })
        weight_blobs.append((f'{pfx}_o', 'ip', (dim, dim), blob_idx))
        blob_idx += 2
        shapes[f"{pfx}_attn_out"] = {"n": 1, "h": 1, "w": 1, "k": dim}

        # --- Residual 1 ---
        layers_json.append({
            "type": "elementwise", "name": f"{pfx}_res1",
            "bottom": f"{prev},{pfx}_attn_out",
            "top": f"{pfx}_r1_out",
            "operation": 0, "alpha": 1, "beta": 0, "fused_relu": 0,
            "weights": {},
        })
        shapes[f"{pfx}_r1_out"] = {"n": 1, "h": 1, "w": 1, "k": dim}

        # --- LN2: MVN ---
        layers_json.append({
            "type": "l2_normalize", "name": f"{pfx}_ln2_mvn",
            "debug_info": f"{pfx}_ln2_mvn",
            "bottom": f"{pfx}_r1_out", "top": f"{pfx}_ln2_mvn_out",
            "normalization_mode": 1, "axis": 2, "eps": 1e-5,
            "weights": {},
        })
        shapes[f"{pfx}_ln2_mvn_out"] = {"n": 1, "h": 1, "w": 1, "k": dim}

        # --- LN2: BN ---
        layers_json.append({
            "type": "batchnorm", "name": f"{pfx}_ln2_bn",
            "debug_info": f"{pfx}_ln2_bn",
            "bottom": f"{pfx}_ln2_mvn_out", "top": f"{pfx}_ln2_out",
            "blob_batchnorm_params": blob_idx, "C": dim, "weights": {},
        })
        weight_blobs.append((f'{pfx}_ln2_bn', 'batchnorm', dim, blob_idx))
        blob_idx += 2
        shapes[f"{pfx}_ln2_out"] = {"n": 1, "h": 1, "w": 1, "k": dim}

        # --- FFN up: dim -> ffn_dim ---
        layers_json.append({
            "type": "inner_product", "name": f"{pfx}_fc_up",
            "debug_info": f"{pfx}_fc_up",
            "bottom": f"{pfx}_ln2_out", "top": f"{pfx}_fc_up_out",
            "nB": dim, "nC": ffn_dim, "has_biases": 0,
            "blob_weights": blob_idx,
            "has_relu": 0, "has_tanh": 0, "has_prelu": 0,
            "weights": {},
        })
        weight_blobs.append((f'{pfx}_fc_up', 'ip', (ffn_dim, dim), blob_idx))
        blob_idx += 2
        shapes[f"{pfx}_fc_up_out"] = {"n": 1, "h": 1, "w": 1, "k": ffn_dim}

        # --- GELU (mode 19 = erf-GELU hardware PWL) ---
        layers_json.append({
            "type": "activation", "name": f"{pfx}_gelu",
            "bottom": f"{pfx}_fc_up_out", "top": f"{pfx}_gelu_out",
            "mode": 19, "weights": {},
        })
        shapes[f"{pfx}_gelu_out"] = {"n": 1, "h": 1, "w": 1, "k": ffn_dim}

        # --- FFN down: ffn_dim -> dim ---
        layers_json.append({
            "type": "inner_product", "name": f"{pfx}_fc_down",
            "debug_info": f"{pfx}_fc_down",
            "bottom": f"{pfx}_gelu_out", "top": f"{pfx}_ffn_out",
            "nB": ffn_dim, "nC": dim, "has_biases": 0,
            "blob_weights": blob_idx,
            "has_relu": 0, "has_tanh": 0, "has_prelu": 0,
            "weights": {},
        })
        weight_blobs.append((f'{pfx}_fc_down', 'ip', (dim, ffn_dim), blob_idx))
        blob_idx += 2
        shapes[f"{pfx}_ffn_out"] = {"n": 1, "h": 1, "w": 1, "k": dim}

        # --- Residual 2 ---
        layers_json.append({
            "type": "elementwise", "name": f"{pfx}_res2",
            "bottom": f"{pfx}_r1_out,{pfx}_ffn_out",
            "top": f"{pfx}_out",
            "operation": 0, "alpha": 1, "beta": 0, "fused_relu": 0,
            "weights": {},
        })
        shapes[f"{pfx}_out"] = {"n": 1, "h": 1, "w": 1, "k": dim}

    last_out = f'L{n_layers-1}_out'

    if include_lm_head:
        # Final LN: MVN
        layers_json.append({
            "type": "l2_normalize", "name": "ln_f_mvn",
            "debug_info": "ln_f_mvn",
            "bottom": last_out, "top": "ln_f_mvn_out",
            "normalization_mode": 1, "axis": 2, "eps": 1e-5,
            "weights": {},
        })
        shapes["ln_f_mvn_out"] = {"n": 1, "h": 1, "w": 1, "k": dim}

        # Final LN: BN
        layers_json.append({
            "type": "batchnorm", "name": "ln_f_bn",
            "debug_info": "ln_f_bn",
            "bottom": "ln_f_mvn_out", "top": "ln_f_out",
            "blob_batchnorm_params": blob_idx, "C": dim, "weights": {},
        })
        weight_blobs.append(('ln_f_bn', 'batchnorm', dim, blob_idx))
        blob_idx += 2
        shapes["ln_f_out"] = {"n": 1, "h": 1, "w": 1, "k": dim}

        # lm_head: dim -> vocab_size
        layers_json.append({
            "type": "inner_product", "name": "lm_head",
            "debug_info": "lm_head",
            "bottom": "ln_f_out", "top": "output",
            "nB": dim, "nC": vocab_size, "has_biases": 0,
            "blob_weights": blob_idx,
            "has_relu": 0, "has_tanh": 0, "has_prelu": 0,
            "weights": {},
            "attributes": {"is_output": 1},
        })
        weight_blobs.append(('lm_head', 'ip', (vocab_size, dim), blob_idx))
        blob_idx += 2
        shapes["output"] = {"n": 1, "h": 1, "w": 1, "k": vocab_size}
        out_dim = vocab_size
    else:
        layers_json.append({
            "type": "activation", "name": "final_identity",
            "bottom": last_out, "top": "output",
            "mode": 6, "beta": 0,
            "weights": {},
            "attributes": {"is_output": 1},
        })
        shapes["output"] = {"n": 1, "h": 1, "w": 1, "k": dim}
        out_dim = dim

    # Write espresso net
    net = {
        "storage": "model.espresso.weights",
        "analyses": {}, "properties": {},
        "format_version": 200,
        "metadata_in_weights": [],
        "layers": layers_json,
    }
    with open(os.path.join(output_dir, 'model.espresso.net'), 'w') as f:
        json.dump(net, f)

    with open(os.path.join(output_dir, 'model.espresso.shape'), 'w') as f:
        json.dump({"layer_shapes": shapes}, f)

    # Write weights (v28 format, matching proven Llama 8B pattern)
    _write_weights_v28(output_dir, weight_blobs, model, n_layers,
                       include_lm_head)

    # Metadata + stubs
    meta = {
        "specificationVersion": 4, "isUpdatable": False,
        "modelType": {"name": "MLModelType_neuralNetwork"},
        "computePrecision": "Float16",
        "inputSchema": [{"name": "input", "type": "MultiArray",
                         "shape": [dim, 1, 1], "dataType": "Float16"}],
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

    n_espresso = len(layers_json)
    print(f"    Espresso layers: {n_espresso}, weight blobs: {len(weight_blobs)}")
    return out_dim


def _make_blob_data(name, btype, shape_or_dim, model, n_layers,
                    include_lm_head):
    """Generate weight bytes for a single blob entry."""
    dtype = np.float32

    if btype == 'batchnorm':
        dim_bn = shape_or_dim
        # Parse which batchnorm
        if name == 'ln_f_bn':
            gamma = model.ln_f_weight.astype(dtype)
            beta = model.ln_f_bias.astype(dtype)
        else:
            parts = name.split('_')
            layer_idx = int(parts[0][1:])
            ln_num = parts[1]  # 'ln1' or 'ln2'
            L = model.layers[layer_idx]
            if ln_num == 'ln1':
                gamma = L.ln_1_weight.astype(dtype)
                beta = L.ln_1_bias.astype(dtype)
            else:
                gamma = L.ln_2_weight.astype(dtype)
                beta = L.ln_2_bias.astype(dtype)
        mean = np.zeros(dim_bn, dtype=dtype)
        var = np.ones(dim_bn, dtype=dtype)
        return np.column_stack([gamma, beta, mean, var]).flatten().tobytes()

    elif btype == 'ip':
        out_ch, in_ch = shape_or_dim

        if name == 'lm_head':
            w = model.wte.astype(dtype)
            return w.tobytes()

        parts = name.split('_')
        layer_idx = int(parts[0][1:])
        op = '_'.join(parts[1:])
        L = model.layers[layer_idx]

        if op == 'v':
            w = L.W_v.astype(dtype)
        elif op == 'o':
            w = L.W_o.astype(dtype)
        elif op == 'fc_up':
            w = L.W_fc.astype(dtype)
        elif op == 'fc_down':
            w = L.W_fc_down.astype(dtype)
        else:
            raise ValueError(f"Unknown op: {op}")
        return w.tobytes()

    raise ValueError(f"Unknown btype: {btype}")


def _write_weights_v28(output_dir, weight_blobs, model, n_layers,
                       include_lm_head):
    """Write model.espresso.weights in v28 format.

    Exactly matches the pattern from build_llama8b_espresso.py which
    is proven to compile on ANE.
    """
    dtype = np.float32
    total_weight_blobs = len(weight_blobs)

    # Prepare blob info
    blob_info = []
    for name, btype, shape_or_dim, bid in weight_blobs:
        data = _make_blob_data(name, btype, shape_or_dim, model,
                               n_layers, include_lm_head)
        raw_size = len(data)
        if btype == 'batchnorm':
            table_size = max(0x3000, raw_size * 4)
        else:
            table_size = raw_size
        blob_info.append((bid, data, raw_size, table_size, btype, name))

    # First blob goes in gap
    first = blob_info[0]
    first_gap = first[3]  # table_size for first blob
    remaining = blob_info[1:]

    # Blob table
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

    wsize = os.path.getsize(weight_path)
    print(f"    Weight file: {wsize/1e6:.1f} MB")
    return wsize


# ===================================================================
# Dispatch + measurement
# ===================================================================

def dispatch_single(mlmodelc_path, in_ch, out_ch, input_fp16, n_iter=50):
    """Dispatch a single model N times and measure latency."""
    manifest_path = '/tmp/fusion_bench_manifest.txt'
    with open(manifest_path, 'w') as f:
        f.write(f"{mlmodelc_path} {in_ch} {out_ch}\n")

    proc = subprocess.Popen(
        [PIPE_TOOL, manifest_path],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Wait for READY_FOR_SWAP
    while True:
        line = proc.stdout.readline().decode().strip()
        if line == 'READY_FOR_SWAP':
            break
        if proc.poll() is not None:
            err = proc.stderr.read().decode()
            raise RuntimeError(f"Pipe tool exited: {err}")

    # Send GO
    proc.stdin.write(b"GO\n")
    proc.stdin.flush()

    while True:
        line = proc.stdout.readline().decode().strip()
        if line == 'DISPATCH_READY':
            break
        if proc.poll() is not None:
            err = proc.stderr.read().decode()
            raise RuntimeError(f"Pipe tool exited during load: {err}")

    input_bytes = input_fp16.astype(np.float16).tobytes()
    times = []

    # Warmup
    for _ in range(5):
        proc.stdin.write(b"D 0\n")
        proc.stdin.write(input_bytes)
        proc.stdin.flush()
        out = proc.stdout.read(out_ch * 2)

    # Measurement
    for _ in range(n_iter):
        t0 = time.perf_counter()
        proc.stdin.write(b"D 0\n")
        proc.stdin.write(input_bytes)
        proc.stdin.flush()
        out = proc.stdout.read(out_ch * 2)
        dt = time.perf_counter() - t0
        times.append(dt * 1000)

    # Quit
    proc.stdin.write(b"Q\n")
    proc.stdin.flush()
    try:
        proc.wait(timeout=5)
    except:
        proc.kill()

    times.sort()
    median = times[len(times) // 2]
    return median, times


def measure_baseline_37(model, n_iter=30):
    """Measure current 37-dispatch baseline tok/s."""
    from first_token import compile_all_ops
    from generate import ANEDispatcher, forward_layer_decode, embed, lm_head
    from kv_cache import KVCache

    build_dir = '/tmp/gpt2_first_token_fused'
    compiled = compile_all_ops(model, build_dir, mode='fused')

    dispatcher = ANEDispatcher(compiled, quiet=True)
    dispatcher.start()

    config = model.config
    kv_cache = KVCache(config.n_layer, config.n_head, config.head_dim)

    # Warmup
    for warm in range(3):
        x = embed(model, 0, warm)
        for layer_i in range(config.n_layer):
            x = forward_layer_decode(layer_i, x, model, dispatcher,
                                     kv_cache, mode='fused')
        logits = lm_head(x, model, dispatcher)

    # Measure
    times = []
    for step in range(n_iter):
        pos = 3 + step
        t0 = time.perf_counter()
        x = embed(model, 0, pos)
        for layer_i in range(config.n_layer):
            x = forward_layer_decode(layer_i, x, model, dispatcher,
                                     kv_cache, mode='fused')
        logits = lm_head(x, model, dispatcher)
        dt = time.perf_counter() - t0
        times.append(dt * 1000)

    dispatcher.stop()

    times.sort()
    median = times[len(times) // 2]
    tps = 1000.0 / median
    return median, tps, times


# ===================================================================
# Main
# ===================================================================

def main():
    print("=" * 70)
    print("GPT-2 FUSION DEPTH BENCHMARK")
    print("Thesis: deeper fusion -> SRAM residency -> higher eff bandwidth")
    print("=" * 70)

    print("\n[1] Loading GPT-2 weights...")
    t0 = time.time()
    model = GPT2Model.from_safetensors(MODEL_PATH)
    print(f"    Loaded in {time.time()-t0:.1f}s")

    dim = 768
    ffn_dim = 3072

    # Weight bytes per layer (FP16): V(768x768) + O(768x768) +
    # fc_up(768x3072) + fc_down(3072x768) = 11.8MB
    bytes_per_layer_fp16 = (768*768 + 768*768 + 768*3072 + 3072*768) * 2
    total_model_fp16 = bytes_per_layer_fp16 * 12
    lm_head_bytes = 768 * 50257 * 2

    print(f"    Weights per layer (FP16): {bytes_per_layer_fp16/1e6:.1f} MB")
    print(f"    Total 12L (FP16): {total_model_fp16/1e6:.1f} MB")
    print(f"    lm_head (FP16): {lm_head_bytes/1e6:.1f} MB")

    os.makedirs(BUILD_DIR, exist_ok=True)

    configs = [
        ("1L",   1,  False),
        ("3L",   3,  False),
        ("6L",   6,  False),
        ("12L",  12, False),
        ("12L+head", 12, True),
    ]

    results = []

    for label, n_layers, include_lm_head in configs:
        print(f"\n{'='*70}")
        print(f"[BUILD] {label} ({n_layers}L, "
              f"{'with' if include_lm_head else 'no'} lm_head)")
        print(f"{'='*70}")

        outdir = os.path.join(BUILD_DIR, f'{label}.mlmodelc')
        weight_fp16 = bytes_per_layer_fp16 * n_layers
        if include_lm_head:
            weight_fp16 += lm_head_bytes

        if os.path.exists(outdir):
            shutil.rmtree(outdir)

        try:
            t0 = time.time()
            out_dim = build_fused_gpt2_espresso(
                outdir, n_layers, model, include_lm_head=include_lm_head)
            build_time = time.time() - t0
            print(f"    Built in {build_time:.1f}s")
            print(f"    Model weights (FP16 equiv): {weight_fp16/1e6:.1f} MB")

            print(f"    Compiling+dispatching on ANE...")
            x = np.random.randn(dim).astype(np.float16)

            median_ms, times = dispatch_single(
                outdir, dim, out_dim, x, n_iter=50)

            eff_bw = (weight_fp16 / 1e9) / (median_ms / 1000)

            print(f"    OK: median {median_ms:.2f} ms, "
                  f"p5/p95 {times[2]:.2f}/{times[-3]:.2f} ms, "
                  f"eff BW {eff_bw:.1f} GB/s")

            results.append({
                'label': label, 'n_layers': n_layers,
                'include_lm_head': include_lm_head,
                'dispatches': 1,
                'weight_mb': weight_fp16 / 1e6,
                'median_ms': median_ms,
                'p5_ms': times[2], 'p95_ms': times[-3],
                'eff_bw': eff_bw, 'status': 'OK',
            })

        except Exception as e:
            import traceback; traceback.print_exc()
            results.append({
                'label': label, 'n_layers': n_layers,
                'include_lm_head': include_lm_head,
                'dispatches': 1,
                'weight_mb': weight_fp16 / 1e6,
                'median_ms': None, 'p5_ms': None, 'p95_ms': None,
                'eff_bw': None, 'status': f'FAIL: {str(e)[:55]}',
            })

    # Baseline
    print(f"\n{'='*70}")
    print("[BASELINE] 37-dispatch current config")
    print(f"{'='*70}")

    try:
        base_median, base_tps, base_times = measure_baseline_37(model, n_iter=30)
        total_weight = total_model_fp16 + lm_head_bytes
        base_eff_bw = (total_weight / 1e9) / (base_median / 1000)
        print(f"    Median: {base_median:.2f} ms, {base_tps:.1f} tok/s, "
              f"eff BW {base_eff_bw:.1f} GB/s")

        results.insert(0, {
            'label': '37-fused (baseline)',
            'n_layers': 12, 'include_lm_head': True,
            'dispatches': 37,
            'weight_mb': total_weight / 1e6,
            'median_ms': base_median,
            'p5_ms': base_times[1], 'p95_ms': base_times[-2],
            'eff_bw': base_eff_bw, 'status': 'OK',
        })
    except Exception as e:
        print(f"    FAILED: {e}")
        results.insert(0, {
            'label': '37-fused (baseline)',
            'n_layers': 12, 'include_lm_head': True,
            'dispatches': 37,
            'weight_mb': (total_model_fp16 + lm_head_bytes) / 1e6,
            'median_ms': None, 'p5_ms': None, 'p95_ms': None,
            'eff_bw': None, 'status': f'FAIL: {str(e)[:55]}',
        })

    # Results table
    print(f"\n\n{'='*95}")
    print("RESULTS TABLE")
    print(f"{'='*95}")
    print(f"{'Config':<22} {'Disp':>5} {'Wt MB':>7} {'ms':>8} "
          f"{'p5':>7} {'p95':>7} {'GB/s':>7} {'tok/s':>7} {'Status':<12}")
    print("-" * 95)

    for r in results:
        ms = f"{r['median_ms']:.2f}" if r['median_ms'] else "---"
        p5 = f"{r['p5_ms']:.2f}" if r['p5_ms'] else "---"
        p95 = f"{r['p95_ms']:.2f}" if r['p95_ms'] else "---"
        bw = f"{r['eff_bw']:.1f}" if r['eff_bw'] else "---"
        tps = f"{1000/r['median_ms']:.1f}" if r['median_ms'] else "---"
        print(f"{r['label']:<22} {r['dispatches']:>5} {r['weight_mb']:>7.1f} "
              f"{ms:>8} {p5:>7} {p95:>7} {bw:>7} {tps:>7} {r['status']:<12}")

    print(f"{'='*95}")

    # Analysis
    ok_results = [r for r in results if r['median_ms'] is not None]
    if len(ok_results) >= 2:
        print(f"\nANALYSIS:")
        baseline = next((r for r in ok_results if r['dispatches'] == 37), None)
        fused = [r for r in ok_results if r['dispatches'] == 1]

        if baseline and fused:
            best = min(fused, key=lambda r: r['median_ms'])
            print(f"  Best fused: {best['label']} ({best['median_ms']:.2f} ms)")
            print(f"  Baseline: {baseline['median_ms']:.2f} ms")
            speedup = baseline['median_ms'] / best['median_ms']
            bw_ratio = best['eff_bw'] / baseline['eff_bw']
            print(f"  Speedup: {speedup:.2f}x")
            print(f"  BW improvement: {bw_ratio:.2f}x")

        if len(fused) >= 2:
            print(f"\n  Fusion depth scaling:")
            for r in sorted(fused, key=lambda r: r['n_layers']):
                print(f"    {r['label']:>15}: {r['eff_bw']:.1f} GB/s, "
                      f"{r['median_ms']:.2f} ms")

            print(f"\n  SRAM residency test:")
            sf = sorted(fused, key=lambda r: r['weight_mb'])
            for i in range(1, len(sf)):
                p, c = sf[i-1], sf[i]
                wr = c['weight_mb'] / p['weight_mb']
                tr = c['median_ms'] / p['median_ms']
                verdict = ("SUB-LINEAR (SRAM)" if tr < wr * 0.8
                           else "SUPER-LINEAR (cliff)" if tr > wr * 1.2
                           else "LINEAR (DRAM)")
                print(f"    {p['label']} -> {c['label']}: "
                      f"wt {wr:.1f}x, time {tr:.1f}x -> {verdict}")

    # ---------------------------------------------------------------
    # Phase 4: Contention comparison
    # ---------------------------------------------------------------
    # Compare degradation of 12L fused (1 dispatch) vs 37 baseline
    # under concurrent CoreML model dispatch
    print(f"\n{'='*70}")
    print("[CONTENTION] 12L fused vs 37-dispatch under ANE load")
    print(f"{'='*70}")

    # Build a simple contention model (single conv, dispatched in a loop)
    contention_dir = os.path.join(BUILD_DIR, 'contention.mlmodelc')
    if not os.path.exists(contention_dir):
        from compiler import gen_conv_mlmodelc
        w = np.random.randn(768, 768).astype(np.float32) * 0.01
        gen_conv_mlmodelc(contention_dir, w, 768, 768, name='contention')

    # Measure 12L fused under contention
    fused_12l_dir = os.path.join(BUILD_DIR, '12L.mlmodelc')
    if os.path.exists(fused_12l_dir):
        print("    Measuring 12L fused under contention...")
        # Start contention process in background
        contention_manifest = '/tmp/contention_manifest.txt'
        with open(contention_manifest, 'w') as f:
            f.write(f"{contention_dir} 768 768\n")

        contention_proc = subprocess.Popen(
            [PIPE_TOOL, contention_manifest],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        # Wait for ready
        while True:
            line = contention_proc.stdout.readline().decode().strip()
            if line == 'READY_FOR_SWAP':
                break
            if contention_proc.poll() is not None:
                break
        contention_proc.stdin.write(b"GO\n")
        contention_proc.stdin.flush()
        while True:
            line = contention_proc.stdout.readline().decode().strip()
            if line == 'DISPATCH_READY':
                break
            if contention_proc.poll() is not None:
                break

        # Start contention loop in background thread
        import threading
        contention_running = True
        def contention_loop():
            x = np.random.randn(768).astype(np.float16)
            while contention_running:
                try:
                    contention_proc.stdin.write(b"D 0\n")
                    contention_proc.stdin.write(x.tobytes())
                    contention_proc.stdin.flush()
                    contention_proc.stdout.read(768 * 2)
                except:
                    break

        ct_thread = threading.Thread(target=contention_loop, daemon=True)
        ct_thread.start()
        time.sleep(0.5)  # let contention stabilize

        # Measure 12L under contention
        try:
            x = np.random.randn(dim).astype(np.float16)
            cont_median, cont_times = dispatch_single(
                fused_12l_dir, dim, dim, x, n_iter=50)

            # Find solo result
            solo_12l = next((r for r in results
                             if r['label'] == '12L' and r['median_ms']), None)
            if solo_12l:
                degradation = (cont_median - solo_12l['median_ms']) / solo_12l['median_ms'] * 100
                print(f"    12L solo:       {solo_12l['median_ms']:.2f} ms")
                print(f"    12L contention: {cont_median:.2f} ms")
                print(f"    Degradation:    {degradation:+.1f}%")
        except Exception as e:
            print(f"    12L contention test failed: {e}")

        contention_running = False
        try:
            contention_proc.stdin.write(b"Q\n")
            contention_proc.stdin.flush()
            contention_proc.wait(timeout=3)
        except:
            contention_proc.kill()

        # Measure baseline under contention
        print("    Measuring 37-dispatch baseline under contention...")
        contention_proc2 = subprocess.Popen(
            [PIPE_TOOL, contention_manifest],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        while True:
            line = contention_proc2.stdout.readline().decode().strip()
            if line == 'READY_FOR_SWAP':
                break
            if contention_proc2.poll() is not None:
                break
        contention_proc2.stdin.write(b"GO\n")
        contention_proc2.stdin.flush()
        while True:
            line = contention_proc2.stdout.readline().decode().strip()
            if line == 'DISPATCH_READY':
                break
            if contention_proc2.poll() is not None:
                break

        contention_running = True
        def contention_loop2():
            x = np.random.randn(768).astype(np.float16)
            while contention_running:
                try:
                    contention_proc2.stdin.write(b"D 0\n")
                    contention_proc2.stdin.write(x.tobytes())
                    contention_proc2.stdin.flush()
                    contention_proc2.stdout.read(768 * 2)
                except:
                    break

        ct_thread2 = threading.Thread(target=contention_loop2, daemon=True)
        ct_thread2.start()
        time.sleep(0.5)

        try:
            cont_base_median, cont_base_tps, cont_base_times = \
                measure_baseline_37(model, n_iter=20)
            base_solo = next((r for r in results
                              if r['label'] == '37-fused (baseline)' and r['median_ms']), None)
            if base_solo:
                degradation_base = (cont_base_median - base_solo['median_ms']) / base_solo['median_ms'] * 100
                print(f"    37-disp solo:       {base_solo['median_ms']:.2f} ms")
                print(f"    37-disp contention: {cont_base_median:.2f} ms")
                print(f"    Degradation:        {degradation_base:+.1f}%")
        except Exception as e:
            print(f"    37-dispatch contention test failed: {e}")

        contention_running = False
        try:
            contention_proc2.stdin.write(b"Q\n")
            contention_proc2.stdin.flush()
            contention_proc2.wait(timeout=3)
        except:
            contention_proc2.kill()

    print(f"\n{'='*95}")
    print("DONE")


if __name__ == '__main__':
    main()
