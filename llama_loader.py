#!/usr/bin/env python3
"""
Llama-3.2-1B-Instruct model loader and ANE compilation pipeline.

Loads Llama weights from safetensors (BF16), maps to layer roles,
and generates per-op .mlmodelc bundles for ANE dispatch.

Architecture differences vs GPT-2:
  - RMSNorm (no bias, no mean subtraction)
  - GQA: 32 Q heads, 8 KV heads (head_dim=64)
  - SwiGLU FFN: gate*SiLU(gate_proj) * up_proj, then down_proj
  - RoPE (theta=500000) instead of absolute position embeddings
  - No bias on any projection
  - Tied embeddings (lm_head = embed_tokens)
  - lm_head 2048->128256 needs output-dimension splitting

SwiGLU fusion: single ANE dispatch via coremltools NeuralNetworkBuilder.
  gate_proj -> sigmoid -> mul (=SiLU) -> mul(up_proj) -> down_proj
  Compiles to 1 .mlmodelc via xcrun coremlcompiler.

Copyright 2026 Nick Lo. MIT License.
"""

import os
import sys
import time
import subprocess
import shutil
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from compiler import gen_conv_mlmodelc


# ===================================================================
# Llama Config
# ===================================================================

@dataclass
class LlamaConfig:
    hidden_size: int = 2048
    n_layers: int = 16
    n_heads: int = 32         # query heads
    n_kv_heads: int = 8       # key/value heads (GQA)
    head_dim: int = 64        # hidden_size // n_heads
    intermediate_size: int = 8192
    vocab_size: int = 128256
    rms_norm_eps: float = 1e-5
    rope_theta: float = 500000.0
    max_position: int = 131072

    # GQA derived
    @property
    def n_rep(self) -> int:
        """Number of times to repeat KV heads to match Q heads."""
        return self.n_heads // self.n_kv_heads  # 32/8 = 4

    # aned compilation limit: in_ch * out_ch < ~1.3M
    aned_param_limit: int = 1_200_000

    def needs_split(self, in_ch: int, out_ch: int) -> bool:
        return in_ch * out_ch > self.aned_param_limit


# ===================================================================
# Safetensors Loader (BF16 via torch)
# ===================================================================

def load_safetensors_bf16(model_path: str) -> Dict[str, np.ndarray]:
    """Load all tensors from BF16 safetensors file(s), convert to FP32 numpy.

    Supports both single-file and multi-shard safetensors.
    If model_path is a directory, loads all shards.
    """
    from safetensors.torch import load_file
    import torch, glob

    if os.path.isdir(model_path):
        shards = sorted(glob.glob(os.path.join(model_path, "model*.safetensors")))
    elif os.path.isfile(model_path):
        shards = [model_path]
    else:
        raise FileNotFoundError(f"No safetensors found at {model_path}")

    tensors = {}
    for shard in shards:
        tensors_torch = load_file(shard)
        for key, val in tensors_torch.items():
            tensors[key] = val.float().numpy()
    return tensors


# ===================================================================
# Llama Layer dataclass
# ===================================================================

@dataclass
class LlamaLayer:
    """Weights for a single Llama transformer layer."""
    layer_idx: int
    # RMSNorm (pre-attention)
    input_layernorm_weight: np.ndarray    # [2048]
    # Attention projections (no bias)
    q_proj_weight: np.ndarray             # [2048, 2048] = [n_heads*head_dim, hidden]
    k_proj_weight: np.ndarray             # [512, 2048]  = [n_kv_heads*head_dim, hidden]
    v_proj_weight: np.ndarray             # [512, 2048]
    o_proj_weight: np.ndarray             # [2048, 2048]
    # RMSNorm (pre-FFN)
    post_attention_layernorm_weight: np.ndarray  # [2048]
    # SwiGLU FFN (no bias)
    gate_proj_weight: np.ndarray          # [8192, 2048]
    up_proj_weight: np.ndarray            # [8192, 2048]
    down_proj_weight: np.ndarray          # [2048, 8192]

    @property
    def W_qkv(self) -> np.ndarray:
        """Combined QKV: [2048+512+512=3072, 2048] in [out, in] format."""
        return np.concatenate([
            self.q_proj_weight,  # [2048, 2048]
            self.k_proj_weight,  # [512, 2048]
            self.v_proj_weight,  # [512, 2048]
        ], axis=0)

    @property
    def W_o(self) -> np.ndarray:
        """Output projection: [2048, 2048] already in [out, in] format."""
        return self.o_proj_weight


# ===================================================================
# Llama Model
# ===================================================================

@dataclass
class LlamaModel:
    """Complete Llama model weights (supports 1B, 3B, 8B+)."""
    config: LlamaConfig
    embed_tokens: np.ndarray         # [vocab, hidden] token embeddings
    layers: List[LlamaLayer]
    norm_weight: np.ndarray          # [hidden] final RMSNorm
    lm_head_weight: Optional[np.ndarray] = None  # [vocab, hidden] if not tied
    rope_scaling: Optional[dict] = None  # Llama 3 RoPE scaling config

    @classmethod
    def from_safetensors(cls, path: str) -> 'LlamaModel':
        """Load Llama from safetensors file (BF16 -> FP32)."""
        tensors = load_safetensors_bf16(path)

        # Infer config from weight shapes (supports 1B, 8B, 14B, etc.)
        embed = tensors["model.embed_tokens.weight"]
        q0 = tensors["model.layers.0.self_attn.q_proj.weight"]
        k0 = tensors["model.layers.0.self_attn.k_proj.weight"]
        gate0 = tensors["model.layers.0.mlp.gate_proj.weight"]
        n_layers = max(int(k.split('.')[2]) for k in tensors if k.startswith("model.layers.")) + 1
        hidden = q0.shape[1]

        # Try config.json first (if path is a directory)
        config_path = os.path.join(path, 'config.json') if os.path.isdir(path) else None
        if config_path and os.path.exists(config_path):
            import json
            cfg = json.load(open(config_path))
            config = LlamaConfig(
                hidden_size=cfg['hidden_size'],
                n_layers=cfg['num_hidden_layers'],
                n_heads=cfg['num_attention_heads'],
                n_kv_heads=cfg.get('num_key_value_heads', cfg['num_attention_heads']),
                head_dim=cfg.get('head_dim', cfg['hidden_size'] // cfg['num_attention_heads']),
                intermediate_size=cfg['intermediate_size'],
                vocab_size=cfg['vocab_size'],
                rope_theta=cfg.get('rope_theta', 500000.0),
                rms_norm_eps=cfg.get('rms_norm_eps', 1e-5),
            )
        else:
            # Infer from weight shapes
            head_dim = 64 if hidden <= 2048 else 128
            config = LlamaConfig(
                hidden_size=hidden,
                n_layers=n_layers,
                n_heads=q0.shape[0] // head_dim,
                n_kv_heads=k0.shape[0] // head_dim,
                head_dim=head_dim,
                intermediate_size=gate0.shape[0],
                vocab_size=embed.shape[0],
            )

        layers = []
        for i in range(config.n_layers):
            prefix = f"model.layers.{i}"
            layers.append(LlamaLayer(
                layer_idx=i,
                input_layernorm_weight=tensors[f"{prefix}.input_layernorm.weight"],
                q_proj_weight=tensors[f"{prefix}.self_attn.q_proj.weight"],
                k_proj_weight=tensors[f"{prefix}.self_attn.k_proj.weight"],
                v_proj_weight=tensors[f"{prefix}.self_attn.v_proj.weight"],
                o_proj_weight=tensors[f"{prefix}.self_attn.o_proj.weight"],
                post_attention_layernorm_weight=tensors[f"{prefix}.post_attention_layernorm.weight"],
                gate_proj_weight=tensors[f"{prefix}.mlp.gate_proj.weight"],
                up_proj_weight=tensors[f"{prefix}.mlp.up_proj.weight"],
                down_proj_weight=tensors[f"{prefix}.mlp.down_proj.weight"],
            ))

        # Separate lm_head weight if not tied (8B+)
        lm_head_w = None
        if "lm_head.weight" in tensors:
            lm_head_w = tensors["lm_head.weight"]

        # RoPE scaling config (Llama 3 extended RoPE)
        rope_scaling = None
        if config_path and os.path.exists(config_path):
            rope_scaling = cfg.get('rope_scaling', None)

        return cls(
            config=config,
            embed_tokens=tensors["model.embed_tokens.weight"],
            layers=layers,
            norm_weight=tensors["model.norm.weight"],
            lm_head_weight=lm_head_w,
            rope_scaling=rope_scaling,
        )


# ===================================================================
# SwiGLU fused .mlmodelc via coremltools NeuralNetworkBuilder
# ===================================================================

def gen_swiglu_mlmodelc(output_dir: str,
                        W_gate: np.ndarray, W_up: np.ndarray,
                        W_down: np.ndarray,
                        in_ch: int, hidden_ch: int, out_ch: int):
    """Generate fused SwiGLU .mlmodelc via coremltools NeuralNetworkBuilder.

    SwiGLU(x) = down_proj(SiLU(gate_proj(x)) * up_proj(x))
    SiLU(x) = x * sigmoid(x)

    Graph:
      input -> gate_proj (inner_product) -> sigmoid -> mul(input) = SiLU
      input -> up_proj (inner_product)
      SiLU_out * up_out -> down_proj (inner_product) -> output

    All compiled into a single ANE dispatch via NeuralNetworkBuilder.

    Args:
        W_gate: gate projection [hidden_ch, in_ch] FP32
        W_up:   up projection   [hidden_ch, in_ch] FP32
        W_down: down projection [out_ch, hidden_ch] FP32
    """
    import coremltools as ct
    from coremltools.models.neural_network import NeuralNetworkBuilder

    input_features = [('input', ct.models.datatypes.Array(in_ch, 1, 1))]
    output_features = [('output', ct.models.datatypes.Array(out_ch, 1, 1))]

    builder = NeuralNetworkBuilder(input_features, output_features,
                                    disable_rank5_shape_mapping=True)

    # gate_proj: input -> gate_out [hidden_ch]
    builder.add_inner_product(
        name='gate_proj',
        W=W_gate.flatten().astype(np.float32),
        b=None,
        input_channels=in_ch,
        output_channels=hidden_ch,
        has_bias=False,
        input_name='input',
        output_name='gate_out',
    )

    # sigmoid(gate_out) -> gate_sigmoid
    builder.add_activation(
        name='gate_sigmoid',
        non_linearity='SIGMOID',
        input_name='gate_out',
        output_name='gate_sigmoid_out',
    )

    # SiLU = gate_out * sigmoid(gate_out)
    builder.add_elementwise(
        name='gate_silu',
        input_names=['gate_out', 'gate_sigmoid_out'],
        output_name='silu_out',
        mode='MULTIPLY',
    )

    # up_proj: input -> up_out [hidden_ch]
    builder.add_inner_product(
        name='up_proj',
        W=W_up.flatten().astype(np.float32),
        b=None,
        input_channels=in_ch,
        output_channels=hidden_ch,
        has_bias=False,
        input_name='input',
        output_name='up_out',
    )

    # SwiGLU multiply: SiLU(gate) * up
    builder.add_elementwise(
        name='swiglu_mul',
        input_names=['silu_out', 'up_out'],
        output_name='swiglu_out',
        mode='MULTIPLY',
    )

    # down_proj: swiglu_out -> output [out_ch]
    builder.add_inner_product(
        name='down_proj',
        W=W_down.flatten().astype(np.float32),
        b=None,
        input_channels=hidden_ch,
        output_channels=out_ch,
        has_bias=False,
        input_name='swiglu_out',
        output_name='output',
    )

    # Save as .mlmodel, then compile to .mlmodelc via xcrun
    tmp_mlmodel = output_dir + '.mlmodel'
    model = builder.spec
    ct.utils.save_spec(model, tmp_mlmodel)

    # Compile to .mlmodelc
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    result = subprocess.run(
        ['xcrun', 'coremlcompiler', 'compile', tmp_mlmodel, os.path.dirname(output_dir)],
        capture_output=True, text=True, timeout=60,
    )
    if result.returncode != 0:
        raise RuntimeError(f"coremlcompiler failed: {result.stderr}")

    # coremlcompiler outputs to <parent>/<stem>.mlmodelc
    compiled_name = Path(tmp_mlmodel).stem + '.mlmodelc'
    compiled_path = os.path.join(os.path.dirname(output_dir), compiled_name)
    if compiled_path != output_dir:
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.rename(compiled_path, output_dir)

    # Cleanup temp .mlmodel
    if os.path.exists(tmp_mlmodel):
        os.unlink(tmp_mlmodel)


# ===================================================================
# lm_head splitting (128256 output dim)
# ===================================================================

def gen_lm_head_chunks(build_dir: str, weight: np.ndarray,
                       in_ch: int, total_out: int,
                       chunk_size: int = 16032) -> List[Tuple[str, int, int, int]]:
    """Generate split lm_head .mlmodelc bundles.

    Splits the output dimension into chunks to stay under aned param limit.
    128256 / 8 = 16032 (already mult of 16).

    Args:
        weight: [total_out, in_ch] = [128256, 2048] FP32
        chunk_size: output channels per chunk (default 16032)

    Returns:
        List of (mlmodelc_path, in_ch, chunk_out_ch, start_idx) tuples.
        Outputs should be concatenated along output dimension.
    """
    chunks = []
    for i, start in enumerate(range(0, total_out, chunk_size)):
        end = min(start + chunk_size, total_out)
        chunk_out = end - start
        sub_w = weight[start:end, :].copy()

        path = os.path.join(build_dir, f'lm_head_chunk_{i}.mlmodelc')
        if not os.path.exists(path):
            gen_conv_mlmodelc(path, sub_w.astype(np.float32), in_ch, chunk_out,
                              name=f'lm_head_{i}')
        chunks.append((path, in_ch, chunk_out, start))

    return chunks


# ===================================================================
# Compile all Llama ops
# ===================================================================

def compile_llama_ops(model: LlamaModel, build_dir: str) -> dict:
    """Pre-compile all .mlmodelc bundles for the full Llama model.

    Per layer (16 layers):
      - qkv_proj: combined Q+K+V (2048 -> 3072), no bias
      - o_proj: output projection (2048 -> 2048), no bias
      - fused_swiglu: gate+SiLU+up+mul+down as single graph (2048 -> 2048)

    lm_head: tied to embed_tokens, split into 8 chunks of 16032

    Returns:
        dict mapping op_name -> (mlmodelc_path, in_ch, out_ch)
        For lm_head chunks: 'lm_head_chunk_N' -> (path, in_ch, chunk_out_ch)
        Plus 'lm_head_chunks' -> list of (path, in_ch, chunk_out, start_idx)
    """
    os.makedirs(build_dir, exist_ok=True)
    config = model.config
    dim = config.hidden_size
    ffn_dim = config.intermediate_size
    qkv_out = dim + 2 * (config.n_kv_heads * config.head_dim)  # 2048 + 512 + 512 = 3072

    compiled = {}
    total_ops = 0

    for i in range(config.n_layers):
        layer_dir = os.path.join(build_dir, f'layer_{i}')
        os.makedirs(layer_dir, exist_ok=True)
        L = model.layers[i]

        # Combined QKV projection (no bias)
        path = os.path.join(layer_dir, 'qkv_proj.mlmodelc')
        if not os.path.exists(path):
            W_qkv = L.W_qkv.astype(np.float32)  # [3072, 2048]
            gen_conv_mlmodelc(path, W_qkv, dim, qkv_out, name='qkv_proj')
        compiled[f'L{i}_qkv_proj'] = (path, dim, qkv_out)
        total_ops += 1

        # O projection (no bias)
        path = os.path.join(layer_dir, 'o_proj.mlmodelc')
        if not os.path.exists(path):
            gen_conv_mlmodelc(path, L.W_o.astype(np.float32), dim, dim, name='o_proj')
        compiled[f'L{i}_o_proj'] = (path, dim, dim)
        total_ops += 1

        # Fused SwiGLU FFN (single .mlmodelc via NeuralNetworkBuilder)
        path = os.path.join(layer_dir, 'fused_swiglu.mlmodelc')
        if not os.path.exists(path):
            gen_swiglu_mlmodelc(
                path,
                W_gate=L.gate_proj_weight.astype(np.float32),
                W_up=L.up_proj_weight.astype(np.float32),
                W_down=L.down_proj_weight.astype(np.float32),
                in_ch=dim, hidden_ch=ffn_dim, out_ch=dim,
            )
        compiled[f'L{i}_fused_swiglu'] = (path, dim, dim)
        total_ops += 1

        print(f"  Layer {i}: qkv({dim}->{qkv_out}) + o({dim}->{dim}) + "
              f"swiglu({dim}->{ffn_dim}->{dim})")

    # lm_head: tied to embed_tokens, split into chunks
    lm_head_dir = os.path.join(build_dir, 'lm_head')
    os.makedirs(lm_head_dir, exist_ok=True)
    lm_head_weight = model.embed_tokens.astype(np.float32)  # [128256, 2048]

    chunk_size = 16032  # 128256 / 8 = 16032, mult of 16
    chunks = gen_lm_head_chunks(lm_head_dir, lm_head_weight, dim,
                                 config.vocab_size, chunk_size)
    for j, (path, ic, oc, start) in enumerate(chunks):
        compiled[f'lm_head_chunk_{j}'] = (path, ic, oc)
        total_ops += 1

    # Store chunk metadata separately (not a 3-tuple, so keep out of
    # the main dict that ANEDispatcher iterates)
    compiled['_lm_head_chunks'] = chunks

    print(f"  lm_head: {len(chunks)} chunks of ~{chunk_size} "
          f"(total {config.vocab_size})")
    print(f"  Total ops: {total_ops}")

    return compiled


# ===================================================================
# UNFUSED SwiGLU compilation (14x faster than fused)
# ===================================================================

def compile_llama_unfused(model: LlamaModel, build_dir: str) -> dict:
    """Compile Llama ops with SEPARATE gate/up/down projections.

    The unfused SwiGLU path uses 3 separate ANE dispatches + CPU SiLU:
      gate = dispatch(L{i}_gate, ln2)     # 2048 -> 8192
      up   = dispatch(L{i}_up, ln2)       # 2048 -> 8192
      silu = gate * sigmoid(gate)          # CPU FP32
      sw   = silu * up                     # CPU FP32
      down = dispatch(L{i}_down, sw)       # 8192 -> 2048

    This is 14x faster than the fused NeuralNetworkBuilder path because
    the individual convolutions stay within ANE's fast dispatch path,
    while the fused graph hits espresso's slow multi-pass compilation.

    Per layer: qkv_proj + o_proj + gate + up + down = 5 ops
    Total: 5 * 16 layers + 8 lm_head chunks = 88 ops

    Returns:
        dict mapping op_name -> (mlmodelc_path, in_ch, out_ch)
    """
    os.makedirs(build_dir, exist_ok=True)
    config = model.config
    dim = config.hidden_size
    ffn_dim = config.intermediate_size
    qkv_out = dim + 2 * (config.n_kv_heads * config.head_dim)

    compiled = {}
    total_ops = 0

    for i in range(config.n_layers):
        layer_dir = os.path.join(build_dir, f'layer_{i}')
        os.makedirs(layer_dir, exist_ok=True)
        L = model.layers[i]

        # Combined QKV projection (no bias)
        path = os.path.join(layer_dir, 'qkv_proj.mlmodelc')
        if not os.path.exists(path):
            W_qkv = L.W_qkv.astype(np.float32)
            gen_conv_mlmodelc(path, W_qkv, dim, qkv_out, name='qkv_proj')
        compiled[f'L{i}_qkv_proj'] = (path, dim, qkv_out)
        total_ops += 1

        # O projection (no bias)
        path = os.path.join(layer_dir, 'o_proj.mlmodelc')
        if not os.path.exists(path):
            gen_conv_mlmodelc(path, L.W_o.astype(np.float32), dim, dim,
                              name='o_proj')
        compiled[f'L{i}_o_proj'] = (path, dim, dim)
        total_ops += 1

        # Separate gate projection: 2048 -> 8192
        path = os.path.join(layer_dir, 'gate_proj.mlmodelc')
        if not os.path.exists(path):
            gen_conv_mlmodelc(path, L.gate_proj_weight.astype(np.float32),
                              dim, ffn_dim, name='gate_proj')
        compiled[f'L{i}_gate'] = (path, dim, ffn_dim)
        total_ops += 1

        # Separate up projection: 2048 -> 8192
        path = os.path.join(layer_dir, 'up_proj.mlmodelc')
        if not os.path.exists(path):
            gen_conv_mlmodelc(path, L.up_proj_weight.astype(np.float32),
                              dim, ffn_dim, name='up_proj')
        compiled[f'L{i}_up'] = (path, dim, ffn_dim)
        total_ops += 1

        # Separate down projection: 8192 -> 2048
        path = os.path.join(layer_dir, 'down_proj.mlmodelc')
        if not os.path.exists(path):
            gen_conv_mlmodelc(path, L.down_proj_weight.astype(np.float32),
                              ffn_dim, dim, name='down_proj')
        compiled[f'L{i}_down'] = (path, ffn_dim, dim)
        total_ops += 1

        print(f"  Layer {i}: qkv({dim}->{qkv_out}) + o({dim}->{dim}) + "
              f"gate({dim}->{ffn_dim}) + up({dim}->{ffn_dim}) + "
              f"down({ffn_dim}->{dim})")

    # lm_head: tied to embed_tokens, split into chunks
    lm_head_dir = os.path.join(build_dir, 'lm_head')
    os.makedirs(lm_head_dir, exist_ok=True)
    lm_head_weight = model.embed_tokens.astype(np.float32)

    chunk_size = 16032
    chunks = gen_lm_head_chunks(lm_head_dir, lm_head_weight, dim,
                                 config.vocab_size, chunk_size)
    for j, (path, ic, oc, start) in enumerate(chunks):
        compiled[f'lm_head_chunk_{j}'] = (path, ic, oc)
        total_ops += 1

    compiled['_lm_head_chunks'] = chunks

    print(f"  lm_head: {len(chunks)} chunks of ~{chunk_size} "
          f"(total {config.vocab_size})")
    print(f"  Total ops: {total_ops}")

    return compiled


def forward_layer_decode_unfused(layer_idx: int, x: np.ndarray,
                                  model: LlamaModel, dispatcher,
                                  kv_cache, position: int) -> np.ndarray:
    """Forward pass with UNFUSED SwiGLU (3 separate dispatches + CPU SiLU).

    Proven at 28.5 tok/s. 14x faster than fused NeuralNetworkBuilder path.
    """
    L = model.layers[layer_idx]
    config = model.config
    dim = config.hidden_size
    n_heads = config.n_heads
    n_kv_heads = config.n_kv_heads
    head_dim = config.head_dim
    n_rep = config.n_rep
    pfx = f'L{layer_idx}'

    # 1. RMSNorm (pre-attention)
    ln1_out = rms_norm_cpu(x, L.input_layernorm_weight, config.rms_norm_eps)

    # 2. QKV projection on ANE
    qkv = dispatcher.dispatch(f'{pfx}_qkv_proj', ln1_out)

    q = qkv[:dim]
    k = qkv[dim:dim + n_kv_heads * head_dim]
    v = qkv[dim + n_kv_heads * head_dim:]

    q_heads = q.reshape(n_heads, head_dim)
    k_heads = k.reshape(n_kv_heads, head_dim)
    v_heads = v.reshape(n_kv_heads, head_dim)

    # 3. RoPE on CPU
    q_heads, k_heads = rope_cpu(q_heads, k_heads, position, head_dim,
                                 config.rope_theta)

    # 4. KV cache
    kv_cache.append(layer_idx,
                    k_heads[np.newaxis],
                    v_heads[np.newaxis])

    # 5. GQA attention on CPU
    cached_k, cached_v = kv_cache.get(layer_idx)
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

    attn_output = attn_output.astype(np.float16)

    # 6. O projection on ANE
    o_out = dispatcher.dispatch(f'{pfx}_o_proj', attn_output)

    # 7. Residual 1
    r1 = (x.astype(np.float32) + o_out.astype(np.float32)).astype(np.float16)

    # 8. RMSNorm (pre-FFN)
    ln2 = rms_norm_cpu(r1, L.post_attention_layernorm_weight,
                        config.rms_norm_eps)

    # 9. UNFUSED SwiGLU: gate/up on ANE, SiLU on CPU, down on ANE
    gate = dispatcher.dispatch(f'{pfx}_gate', ln2)
    up = dispatcher.dispatch(f'{pfx}_up', ln2)

    # CPU SiLU: x * sigmoid(x), in FP32 for accuracy
    g32 = gate.astype(np.float32)
    silu = (g32 / (1.0 + np.exp(-g32))).astype(np.float16)
    sw = (silu.astype(np.float32) * up.astype(np.float32)).astype(np.float16)

    down = dispatcher.dispatch(f'{pfx}_down', sw)

    # 10. Residual 2
    output = (r1.astype(np.float32) + down.astype(np.float32)).astype(np.float16)
    return output


# ===================================================================
# CPU helper functions
# ===================================================================

def rms_norm_cpu(x: np.ndarray, weight: np.ndarray,
                 eps: float = 1e-5) -> np.ndarray:
    """RMSNorm on CPU (FP32 for accuracy, output as FP16).

    RMSNorm(x) = x * rsqrt(mean(x^2) + eps) * weight
    No mean subtraction, no bias (unlike LayerNorm).
    """
    x_f32 = x.astype(np.float32)
    rms = np.sqrt(np.mean(x_f32 ** 2) + eps)
    normed = x_f32 / rms
    result = normed * weight.astype(np.float32)
    return result.astype(np.float16)


def rope_cpu(q: np.ndarray, k: np.ndarray, position: int,
             head_dim: int, theta: float = 500000.0) -> Tuple[np.ndarray, np.ndarray]:
    """Apply Rotary Position Embedding on CPU.

    q: [n_heads, head_dim] FP16
    k: [n_kv_heads, head_dim] FP16
    Returns: (q_rotated, k_rotated) as FP16

    RoPE: for each pair (x_2i, x_2i+1), apply rotation by angle
    position * theta^(-2i/head_dim).
    """
    q_f32 = q.astype(np.float32)
    k_f32 = k.astype(np.float32)

    # Compute rotation angles
    half_dim = head_dim // 2
    freqs = 1.0 / (theta ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))
    angles = position * freqs  # [half_dim]

    cos_vals = np.cos(angles)  # [half_dim]
    sin_vals = np.sin(angles)  # [half_dim]

    def apply_rope(x):
        # x: [n_heads, head_dim]
        # Llama uses rotate_half: split into first half / second half
        x1 = x[:, :half_dim]   # [n_heads, half_dim]
        x2 = x[:, half_dim:]   # [n_heads, half_dim]
        # rotate_half: [-x2, x1]
        rotated = np.concatenate([-x2, x1], axis=-1)
        # Apply: x * cos + rotate_half(x) * sin
        cos_full = np.concatenate([cos_vals, cos_vals])  # [head_dim]
        sin_full = np.concatenate([sin_vals, sin_vals])  # [head_dim]
        return x * cos_full + rotated * sin_full

    q_rot = apply_rope(q_f32).astype(np.float16)
    k_rot = apply_rope(k_f32).astype(np.float16)
    return q_rot, k_rot


def softmax_cpu(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax in FP32."""
    x_f32 = x.astype(np.float32)
    x_f32 = x_f32 - x_f32.max()
    exp_x = np.exp(x_f32)
    return (exp_x / exp_x.sum()).astype(np.float32)


# ===================================================================
# Forward pass for a single Llama layer (decode, seq_len=1)
# ===================================================================

def forward_layer_decode(layer_idx: int, x: np.ndarray, model: LlamaModel,
                         dispatcher, kv_cache, position: int) -> np.ndarray:
    """Forward pass through one Llama layer with GQA attention on CPU.

    Args:
        layer_idx: which layer
        x: [hidden_size] FP16 input
        model: LlamaModel with weights
        dispatcher: ANEDispatcher (has .dispatch(name, input) method)
        kv_cache: KVCache instance
        position: absolute position in sequence

    Returns:
        [hidden_size] FP16 output
    """
    L = model.layers[layer_idx]
    config = model.config
    dim = config.hidden_size
    n_heads = config.n_heads
    n_kv_heads = config.n_kv_heads
    head_dim = config.head_dim
    n_rep = config.n_rep
    pfx = f'L{layer_idx}'

    # 1. RMSNorm (pre-attention)
    ln1_out = rms_norm_cpu(x, L.input_layernorm_weight, config.rms_norm_eps)

    # 2. QKV projection on ANE (combined, no bias)
    qkv = dispatcher.dispatch(f'{pfx}_qkv_proj', ln1_out)

    # Split QKV
    q = qkv[:dim]                              # [2048]
    k = qkv[dim:dim + n_kv_heads * head_dim]   # [512]
    v = qkv[dim + n_kv_heads * head_dim:]      # [512]

    # Reshape to multi-head
    q_heads = q.reshape(n_heads, head_dim)       # [32, 64]
    k_heads = k.reshape(n_kv_heads, head_dim)    # [8, 64]
    v_heads = v.reshape(n_kv_heads, head_dim)    # [8, 64]

    # 3. RoPE on CPU
    q_heads, k_heads = rope_cpu(q_heads, k_heads, position, head_dim,
                                 config.rope_theta)

    # 4. KV cache append
    kv_cache.append(layer_idx,
                    k_heads[np.newaxis],   # [1, n_kv_heads, head_dim]
                    v_heads[np.newaxis])    # [1, n_kv_heads, head_dim]

    # 5. GQA attention on CPU
    cached_k, cached_v = kv_cache.get(layer_idx)
    # cached_k: [seq_len, n_kv_heads, head_dim]
    # cached_v: [seq_len, n_kv_heads, head_dim]
    scale = np.float32(1.0 / np.sqrt(head_dim))
    attn_output = np.zeros(dim, dtype=np.float32)

    for h in range(n_heads):
        kv_h = h // n_rep  # map Q head to KV head (GQA)
        q_h = q_heads[h].astype(np.float32)              # [head_dim]
        k_h = cached_k[:, kv_h, :].astype(np.float32)    # [seq_len, head_dim]
        v_h = cached_v[:, kv_h, :].astype(np.float32)    # [seq_len, head_dim]

        scores = (q_h @ k_h.T) * scale                   # [seq_len]
        weights = softmax_cpu(scores)                      # [seq_len]
        attn_output[h * head_dim:(h + 1) * head_dim] = weights @ v_h

    attn_output = attn_output.astype(np.float16)

    # 6. O projection on ANE (no bias)
    o_out = dispatcher.dispatch(f'{pfx}_o_proj', attn_output)

    # 7. Residual 1
    r1 = (x.astype(np.float32) + o_out.astype(np.float32)).astype(np.float16)

    # 8. RMSNorm (pre-FFN)
    ln2_out = rms_norm_cpu(r1, L.post_attention_layernorm_weight,
                            config.rms_norm_eps)

    # 9. Fused SwiGLU FFN on ANE (single dispatch)
    ffn_out = dispatcher.dispatch(f'{pfx}_fused_swiglu', ln2_out)

    # 10. Residual 2
    output = (r1.astype(np.float32) + ffn_out.astype(np.float32)).astype(np.float16)
    return output


# ===================================================================
# lm_head dispatch (chunked)
# ===================================================================

def lm_head_dispatch(x: np.ndarray, compiled: dict,
                     dispatcher) -> np.ndarray:
    """Dispatch lm_head as chunked convolutions, concatenate on CPU.

    Args:
        x: [hidden_size] FP16 (after final RMSNorm)
        compiled: dict with 'lm_head_chunks' entry
        dispatcher: ANEDispatcher

    Returns:
        [vocab_size] FP16 logits
    """
    chunks = compiled['_lm_head_chunks']
    outputs = []
    for j, (path, ic, oc, start) in enumerate(chunks):
        chunk_out = dispatcher.dispatch(f'lm_head_chunk_{j}', x)
        outputs.append(chunk_out)
    return np.concatenate(outputs)


# ===================================================================
# Test: load, compile, verify ANE dispatch
# ===================================================================

def main():
    MODEL_PATH = os.path.expanduser(
        "~/.cache/huggingface/hub/models--unsloth--Llama-3.2-1B-Instruct/"
        "snapshots/5a8abab4a5d6f164389b1079fb721cfab8d7126c/model.safetensors"
    )
    BUILD_DIR = '/tmp/llama_1b_ane'

    print("=" * 60)
    print("LLAMA-3.2-1B-INSTRUCT ANE COMPILATION PIPELINE")
    print("=" * 60)

    # 1. Load model
    print("\n[1/4] Loading Llama from safetensors (BF16 -> FP32)...")
    t0 = time.time()
    model = LlamaModel.from_safetensors(MODEL_PATH)
    load_time = time.time() - t0
    print(f"  Loaded in {load_time:.2f}s")
    print(f"  Layers: {len(model.layers)}")
    print(f"  Embedding: {model.embed_tokens.shape}")
    print(f"  Norm: {model.norm_weight.shape}")

    # Print layer 0 weight shapes
    L0 = model.layers[0]
    print(f"\n  Layer 0 weights:")
    print(f"    q_proj:    {L0.q_proj_weight.shape}")
    print(f"    k_proj:    {L0.k_proj_weight.shape}")
    print(f"    v_proj:    {L0.v_proj_weight.shape}")
    print(f"    o_proj:    {L0.o_proj_weight.shape}")
    print(f"    gate_proj: {L0.gate_proj_weight.shape}")
    print(f"    up_proj:   {L0.up_proj_weight.shape}")
    print(f"    down_proj: {L0.down_proj_weight.shape}")
    print(f"    W_qkv:     {L0.W_qkv.shape}")

    # Splitting analysis
    config = model.config
    print(f"\n  Conv splitting analysis:")
    for name, shape in [
        ("QKV combined", (3072, 2048)),
        ("O proj", (2048, 2048)),
        ("gate_proj", (8192, 2048)),
        ("up_proj", (8192, 2048)),
        ("down_proj", (2048, 8192)),
        ("lm_head", (128256, 2048)),
    ]:
        out_ch, in_ch = shape
        needs = config.needs_split(in_ch, out_ch)
        params = in_ch * out_ch
        print(f"    {name:14s}: {in_ch}x{out_ch} = {params:>12,d} params -> "
              f"{'SPLIT NEEDED' if needs else 'OK'}")

    # 2. Compile all ops
    print(f"\n[2/4] Compiling .mlmodelc bundles to {BUILD_DIR}...")
    t0 = time.time()
    compiled = compile_llama_ops(model, BUILD_DIR)
    compile_time = time.time() - t0
    print(f"  Compiled in {compile_time:.2f}s")

    # Count ops
    n_per_layer = 3  # qkv + o + swiglu
    n_lm_head = len(compiled['_lm_head_chunks'])
    n_total = n_per_layer * config.n_layers + n_lm_head
    print(f"  Total .mlmodelc bundles: {n_total}")
    print(f"    Per layer: {n_per_layer} (qkv + o_proj + fused_swiglu)")
    print(f"    lm_head:   {n_lm_head} chunks")

    # 3. Verify with random input dispatch
    print(f"\n[3/4] Verifying ANE dispatch with random input...")

    # Import the dispatcher from generate.py
    from generate import ANEDispatcher

    # Filter out metadata keys (ANEDispatcher expects all values to be 3-tuples)
    dispatch_dict = {k: v for k, v in compiled.items() if not k.startswith('_')}

    t0 = time.time()
    dispatcher = ANEDispatcher(dispatch_dict, quiet=True)
    dispatcher.start()
    dispatch_time = time.time() - t0
    print(f"  Dispatcher ready in {dispatch_time:.2f}s")

    # Test single layer forward
    rng = np.random.RandomState(42)
    x_test = rng.randn(config.hidden_size).astype(np.float16)

    # Create KV cache for test
    kv_cache_test = __import__('kv_cache').KVCache(
        config.n_layers, config.n_kv_heads, config.head_dim)

    print(f"  Running layer 0 forward (decode, position=0)...")
    t0 = time.time()
    y = forward_layer_decode(0, x_test, model, dispatcher, kv_cache_test,
                              position=0)
    layer_time = time.time() - t0
    print(f"    Output shape: {y.shape}, dtype: {y.dtype}")
    print(f"    Output range: [{float(y.min()):.4f}, {float(y.max()):.4f}]")
    print(f"    Layer time: {layer_time*1000:.1f}ms")
    print(f"    Not all zeros: {np.any(y != 0)}")

    # Test lm_head
    print(f"  Running lm_head (chunked, {n_lm_head} dispatches)...")
    x_norm = rms_norm_cpu(x_test, model.norm_weight, config.rms_norm_eps)
    t0 = time.time()
    logits = lm_head_dispatch(x_norm, compiled, dispatcher)
    lm_time = time.time() - t0
    print(f"    Logits shape: {logits.shape}, dtype: {logits.dtype}")
    print(f"    Logits range: [{float(logits.min()):.4f}, {float(logits.max()):.4f}]")
    print(f"    lm_head time: {lm_time*1000:.1f}ms")
    top5 = np.argsort(logits.astype(np.float32))[-5:][::-1]
    print(f"    Top-5 tokens (random input): {top5}")

    # 4. CPU reference for layer 0
    print(f"\n[4/4] CPU reference comparison (layer 0)...")
    L0 = model.layers[0]
    x_f16 = x_test.copy()

    # RMSNorm
    ln1 = rms_norm_cpu(x_f16, L0.input_layernorm_weight, config.rms_norm_eps)

    # QKV on CPU
    qkv_cpu = (L0.W_qkv.astype(np.float16) @ ln1.astype(np.float16)).astype(np.float16)

    # QKV on ANE (already dispatched in forward_layer_decode, verify shape)
    qkv_ane = dispatcher.dispatch('L0_qkv_proj', ln1)

    diff = np.abs(qkv_cpu.astype(np.float32) - qkv_ane.astype(np.float32))
    max_diff = diff.max()
    mean_diff = diff.mean()
    print(f"  QKV proj max diff (ANE vs CPU): {max_diff:.6f}")
    print(f"  QKV proj mean diff:             {mean_diff:.6f}")
    print(f"  QKV match quality: {'GOOD' if max_diff < 1.0 else 'CHECK'}")

    dispatcher.stop()

    # Summary
    print(f"\n{'=' * 60}")
    print("COMPILATION PIPELINE SUMMARY")
    print(f"  Model load:      {load_time:.2f}s")
    print(f"  Compile:         {compile_time:.2f}s")
    print(f"  Dispatcher init: {dispatch_time:.2f}s")
    print(f"  Layer forward:   {layer_time*1000:.1f}ms")
    print(f"  lm_head:         {lm_time*1000:.1f}ms")
    print(f"  Dispatches/token: {n_per_layer * config.n_layers + n_lm_head} "
          f"({n_per_layer}/layer x {config.n_layers} layers + {n_lm_head} lm_head)")
    print(f"  ANE ops: QKV(no bias) + O(no bias) + fused SwiGLU + chunked lm_head")
    print(f"  CPU ops: RMSNorm + RoPE + GQA attention + residuals")
    print("=" * 60)


if __name__ == "__main__":
    main()
