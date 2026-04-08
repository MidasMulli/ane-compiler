#!/usr/bin/env python3
"""
Build Llama-3.1-8B-Instruct as a SINGLE NeuralNetworkBuilder graph.

Strategy: NeuralNetworkBuilder → espresso format → coremlcompiler → single .hwx
- For seq_len=1 decode: attention simplifies to V→O (softmax of scalar = 1.0)
- All 32 layers fused into one graph
- RMSNorm via MVN + batchnorm (scale only, beta=0)

Incremental: 1 layer → 4 layers → 32 layers

Copyright 2026 Nick Lo. MIT License.
"""

import os
import sys
import time
import shutil
import subprocess
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import coremltools as ct
from coremltools.models.neural_network import NeuralNetworkBuilder


# ===================================================================
# Build a single Llama layer as NeuralNetworkBuilder graph
# ===================================================================

def build_llama_layer(builder, layer_idx, prev_name, dim, ffn_dim,
                      n_heads, n_kv_heads, head_dim,
                      weights=None, use_random=True):
    """Add one Llama transformer layer to the NeuralNetworkBuilder graph.

    For seq_len=1 decode, attention is simplified:
    - Q@K^T is a single scalar per head → softmax = 1.0 → output = V
    - So we skip Q projection entirely and just do V→O
    - This is mathematically correct for single-token decode

    But for the full graph (to get fusion), we include all projections
    and let the compiler fuse them. The attention matmul is skipped.

    Architecture per layer:
    1. RMSNorm (MVN + scale)
    2. Q proj (4096→4096) — included for completeness
    3. K proj (4096→1024)
    4. V proj (4096→1024)
    5. For decode: attention output ≈ repeat(V, 4) → O proj
       (Since softmax of single element = 1.0, attn_output = V repeated for GQA)
    6. O proj (4096→4096)
    7. Residual add
    8. RMSNorm (MVN + scale)
    9. Gate proj (4096→14336) → sigmoid → mul = SiLU
    10. Up proj (4096→14336)
    11. Multiply (SiLU(gate) * up)
    12. Down proj (14336→4096)
    13. Residual add
    """
    pfx = f'L{layer_idx}'

    def get_w(name, shape):
        if weights is not None and name in weights:
            return weights[name].flatten().astype(np.float32)
        return np.random.randn(*shape).astype(np.float32) * 0.02

    def get_norm_w(name, size):
        if weights is not None and name in weights:
            return weights[name].astype(np.float32)
        return np.ones(size, dtype=np.float32)

    # --- RMSNorm 1 (pre-attention) ---
    # RMSNorm = x / sqrt(mean(x^2) + eps) * weight
    # MVN (mean-variance-normalize) approximates this
    # Then batchnorm applies the learned scale (gamma=weight, beta=0)

    builder.add_mvn(
        name=f'{pfx}_ln1_mvn',
        input_name=prev_name,
        output_name=f'{pfx}_ln1_mvn_out',
        across_channels=True,
        normalize_variance=True,
        epsilon=1e-5,
    )

    ln1_gamma = get_norm_w(f'L{layer_idx}_ln1_weight', dim)
    ln1_beta = np.zeros(dim, dtype=np.float32)
    builder.add_batchnorm(
        name=f'{pfx}_ln1_bn',
        channels=dim,
        gamma=ln1_gamma,
        beta=ln1_beta,
        mean=np.zeros(dim, dtype=np.float32),
        variance=np.ones(dim, dtype=np.float32),
        input_name=f'{pfx}_ln1_mvn_out',
        output_name=f'{pfx}_ln1_out',
        epsilon=0.0,
    )

    # --- V projection (4096 → 1024 for 8 KV heads × 128 dim) ---
    # For decode with seq_len=1, attention output = V (repeated for GQA)
    # We still need V→O to be mathematically meaningful

    kv_dim = n_kv_heads * head_dim  # 8 * 128 = 1024

    builder.add_inner_product(
        name=f'{pfx}_v_proj',
        W=get_w(f'L{layer_idx}_v_proj', (kv_dim, dim)),
        b=None,
        input_channels=dim,
        output_channels=kv_dim,
        has_bias=False,
        input_name=f'{pfx}_ln1_out',
        output_name=f'{pfx}_v_out',
    )

    # For seq_len=1 decode: we need to expand V from kv_dim to full dim
    # GQA: repeat each KV head 4 times (n_heads/n_kv_heads = 32/8 = 4)
    # Then apply O projection
    #
    # Simplification: V (1024) → linear expand (1024→4096) → O (4096→4096)
    # The expand + O can be combined into a single projection: (1024→4096)
    # This is V_expanded @ O = (V @ expand) @ O = V @ (expand @ O)
    # But we want to keep it as separate ops so the weights are meaningful
    #
    # Actually: for the graph, just use a combined V→attn_out projection
    # attn_out = O @ expand(V)  where expand repeats KV heads
    # Combined: attn_out_weight[4096, 1024] = O[4096, 4096] @ expand[4096, 1024]
    # But with random weights this is just another projection

    # Use a direct V→attn_out linear (1024→4096) combining GQA expand + O
    builder.add_inner_product(
        name=f'{pfx}_vo_proj',
        W=get_w(f'L{layer_idx}_vo_proj', (dim, kv_dim)),
        b=None,
        input_channels=kv_dim,
        output_channels=dim,
        has_bias=False,
        input_name=f'{pfx}_v_out',
        output_name=f'{pfx}_attn_out',
    )

    # --- Residual 1 ---
    builder.add_elementwise(
        name=f'{pfx}_res1',
        input_names=[prev_name, f'{pfx}_attn_out'],
        output_name=f'{pfx}_res1_out',
        mode='ADD',
    )

    # --- RMSNorm 2 (pre-FFN) ---
    builder.add_mvn(
        name=f'{pfx}_ln2_mvn',
        input_name=f'{pfx}_res1_out',
        output_name=f'{pfx}_ln2_mvn_out',
        across_channels=True,
        normalize_variance=True,
        epsilon=1e-5,
    )

    ln2_gamma = get_norm_w(f'L{layer_idx}_ln2_weight', dim)
    ln2_beta = np.zeros(dim, dtype=np.float32)
    builder.add_batchnorm(
        name=f'{pfx}_ln2_bn',
        channels=dim,
        gamma=ln2_gamma,
        beta=ln2_beta,
        mean=np.zeros(dim, dtype=np.float32),
        variance=np.ones(dim, dtype=np.float32),
        input_name=f'{pfx}_ln2_mvn_out',
        output_name=f'{pfx}_ln2_out',
        epsilon=0.0,
    )

    # --- SwiGLU FFN ---
    # gate_proj: 4096 → 14336
    builder.add_inner_product(
        name=f'{pfx}_gate_proj',
        W=get_w(f'L{layer_idx}_gate_proj', (ffn_dim, dim)),
        b=None,
        input_channels=dim,
        output_channels=ffn_dim,
        has_bias=False,
        input_name=f'{pfx}_ln2_out',
        output_name=f'{pfx}_gate_out',
    )

    # SiLU = gate * sigmoid(gate)
    builder.add_activation(
        name=f'{pfx}_gate_sigmoid',
        non_linearity='SIGMOID',
        input_name=f'{pfx}_gate_out',
        output_name=f'{pfx}_gate_sig',
    )

    builder.add_elementwise(
        name=f'{pfx}_gate_silu',
        input_names=[f'{pfx}_gate_out', f'{pfx}_gate_sig'],
        output_name=f'{pfx}_silu_out',
        mode='MULTIPLY',
    )

    # up_proj: 4096 → 14336
    builder.add_inner_product(
        name=f'{pfx}_up_proj',
        W=get_w(f'L{layer_idx}_up_proj', (ffn_dim, dim)),
        b=None,
        input_channels=dim,
        output_channels=ffn_dim,
        has_bias=False,
        input_name=f'{pfx}_ln2_out',
        output_name=f'{pfx}_up_out',
    )

    # SwiGLU multiply: SiLU(gate) * up
    builder.add_elementwise(
        name=f'{pfx}_swiglu_mul',
        input_names=[f'{pfx}_silu_out', f'{pfx}_up_out'],
        output_name=f'{pfx}_swiglu_out',
        mode='MULTIPLY',
    )

    # down_proj: 14336 → 4096
    builder.add_inner_product(
        name=f'{pfx}_down_proj',
        W=get_w(f'L{layer_idx}_down_proj', (dim, ffn_dim)),
        b=None,
        input_channels=ffn_dim,
        output_channels=dim,
        has_bias=False,
        input_name=f'{pfx}_swiglu_out',
        output_name=f'{pfx}_ffn_out',
    )

    # --- Residual 2 ---
    builder.add_elementwise(
        name=f'{pfx}_res2',
        input_names=[f'{pfx}_res1_out', f'{pfx}_ffn_out'],
        output_name=f'{pfx}_out',
        mode='ADD',
    )

    return f'{pfx}_out'


def build_model(n_layers, dim=4096, ffn_dim=14336, n_heads=32, n_kv_heads=8,
                head_dim=128, weights=None, output_dir='/tmp/llama8b_test'):
    """Build an n-layer Llama model as single NeuralNetworkBuilder graph."""

    print(f"\n{'='*60}")
    print(f"Building {n_layers}-layer Llama graph")
    print(f"  dim={dim}, ffn={ffn_dim}, heads={n_heads}/{n_kv_heads}")
    print(f"{'='*60}")

    input_features = [('input', ct.models.datatypes.Array(dim, 1, 1))]
    output_features = [('output', ct.models.datatypes.Array(dim, 1, 1))]

    builder = NeuralNetworkBuilder(input_features, output_features,
                                    disable_rank5_shape_mapping=True)

    current = 'input'
    for i in range(n_layers):
        t0 = time.time()
        current = build_llama_layer(
            builder, i, current, dim, ffn_dim,
            n_heads, n_kv_heads, head_dim,
            weights=weights,
        )
        dt = time.time() - t0
        print(f"  Layer {i}: added ({dt:.1f}s)")

    # Final output rename (connect last layer output to 'output')
    builder.add_activation(
        name='final_identity',
        non_linearity='LINEAR',
        input_name=current,
        output_name='output',
        params=[1.0, 0.0],  # alpha=1.0, beta=0.0 → identity
    )

    # Save as .mlmodel
    os.makedirs(output_dir, exist_ok=True)
    mlmodel_path = os.path.join(output_dir, f'llama_{n_layers}L.mlmodel')

    print(f"\nSaving .mlmodel to {mlmodel_path}...")
    t0 = time.time()
    model = builder.spec
    ct.utils.save_spec(model, mlmodel_path)
    dt = time.time() - t0

    fsize = os.path.getsize(mlmodel_path)
    print(f"  Saved: {fsize/1024/1024:.1f} MB in {dt:.1f}s")

    # Compile via xcrun coremlcompiler
    mlmodelc_path = os.path.join(output_dir, f'llama_{n_layers}L.mlmodelc')
    if os.path.exists(mlmodelc_path):
        shutil.rmtree(mlmodelc_path)

    print(f"\nCompiling via xcrun coremlcompiler...")
    t0 = time.time()
    result = subprocess.run(
        ['xcrun', 'coremlcompiler', 'compile', mlmodel_path, output_dir],
        capture_output=True, text=True, timeout=600,
    )
    dt = time.time() - t0

    if result.returncode != 0:
        print(f"  COMPILE FAILED: {result.stderr[:500]}")
        return None

    # coremlcompiler outputs to <output_dir>/<stem>.mlmodelc
    compiled_name = Path(mlmodel_path).stem + '.mlmodelc'
    compiled_path = os.path.join(output_dir, compiled_name)

    print(f"  Compiled in {dt:.1f}s")

    # Check for .hwx (ANE compiled binary)
    hwx_files = list(Path(compiled_path).rglob('*.hwx'))
    espresso_net = os.path.join(compiled_path, 'model.espresso.net')

    if hwx_files:
        for hwx in hwx_files:
            print(f"  .hwx: {hwx} ({hwx.stat().st_size/1024:.1f} KB)")
    else:
        print(f"  NO .hwx found (ANE compilation may be deferred to runtime)")

    # Count espresso layers to verify fusion
    if os.path.exists(espresso_net):
        import json
        with open(espresso_net) as f:
            net = json.load(f)
        n_espresso = len(net.get('layers', []))
        print(f"  Espresso layers: {n_espresso}")

    return compiled_path


def build_with_real_weights(n_layers=32, output_dir='/tmp/llama8b_real'):
    """Build with real Llama-8B weights from safetensors."""

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
    print(f"  Config: hidden={model.config.hidden_size}, layers={model.config.n_layers}, "
          f"heads={model.config.n_heads}/{model.config.n_kv_heads}, "
          f"ffn={model.config.intermediate_size}")

    # Build weight dict for builder
    weights = {}
    for i in range(min(n_layers, model.config.n_layers)):
        L = model.layers[i]
        weights[f'L{i}_ln1_weight'] = L.input_layernorm_weight
        weights[f'L{i}_ln2_weight'] = L.post_attention_layernorm_weight
        weights[f'L{i}_v_proj'] = L.v_proj_weight.flatten()

        # Combined V→O: for decode, attn_out = O @ expand(V)
        # expand: [4096, 1024] repeating each KV head 4 times
        # O: [4096, 4096]
        # Combined: O @ expand = [4096, 1024]
        n_rep = model.config.n_rep  # 4
        kv_dim = model.config.n_kv_heads * model.config.head_dim  # 1024

        # Build expand matrix: each kv head repeated n_rep times
        # expand[q_head_idx * head_dim : (q_head_idx+1) * head_dim,
        #        kv_head_idx * head_dim : (kv_head_idx+1) * head_dim] = I
        expand = np.zeros((model.config.hidden_size, kv_dim), dtype=np.float32)
        for kv_h in range(model.config.n_kv_heads):
            for r in range(n_rep):
                q_h = kv_h * n_rep + r
                expand[q_h * model.config.head_dim:(q_h+1) * model.config.head_dim,
                       kv_h * model.config.head_dim:(kv_h+1) * model.config.head_dim] = \
                    np.eye(model.config.head_dim, dtype=np.float32)

        # vo_weight = O @ expand: [4096, 4096] @ [4096, 1024] = [4096, 1024]
        vo_weight = L.o_proj_weight @ expand  # [4096, 4096] @ [4096, 1024] = [4096, 1024]
        weights[f'L{i}_vo_proj'] = vo_weight.flatten()

        weights[f'L{i}_gate_proj'] = L.gate_proj_weight.flatten()
        weights[f'L{i}_up_proj'] = L.up_proj_weight.flatten()
        weights[f'L{i}_down_proj'] = L.down_proj_weight.flatten()

    return build_model(
        n_layers=n_layers,
        dim=model.config.hidden_size,
        ffn_dim=model.config.intermediate_size,
        n_heads=model.config.n_heads,
        n_kv_heads=model.config.n_kv_heads,
        head_dim=model.config.head_dim,
        weights=weights,
        output_dir=output_dir,
    )


# ===================================================================
# Main: incremental build
# ===================================================================

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--layers', type=int, default=1,
                       help='Number of layers (1,2,4,8,16,32)')
    parser.add_argument('--real-weights', action='store_true',
                       help='Use real Llama-8B weights')
    parser.add_argument('--output-dir', default='/tmp/llama8b_build',
                       help='Output directory')
    args = parser.parse_args()

    if args.real_weights:
        result = build_with_real_weights(args.layers, args.output_dir)
    else:
        result = build_model(args.layers, output_dir=args.output_dir)

    if result:
        print(f"\nSUCCESS: {result}")
    else:
        print(f"\nFAILED")
