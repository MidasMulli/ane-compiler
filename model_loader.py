#!/usr/bin/env python3
"""
GPT-2 model loader for ane-compiler.

Loads GPT-2 weights from safetensors, maps to layer roles,
repacks to ANE tile layout, and generates per-op .mlmodelc bundles.

Weight format notes:
  - GPT-2 uses Conv1D: weight shape [in_features, out_features]
  - ANE conv1x1 expects [out_channels, in_channels]
  - So all projection weights need .T before packing

Architecture:
  Per layer:
    ln_1 → c_attn (QKV combined) → attention → c_proj → residual_add →
    ln_2 → c_fc → GELU → c_proj_ffn → residual_add

  For seq_len=1 first token, attention collapses:
    softmax(scalar) = 1.0, so output = O_proj(V_proj(ln1(x)))

Copyright 2026 Nick Lo. MIT License.
"""

import os
import sys
import struct
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

# ANE weight packing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from emitter import WeightPacker
from compiler import gen_conv_mlmodelc, gen_softmax_mlmodelc, gen_layernorm_mlmodelc


# ═══════════════════════════════════════════════════════════════
# GPT-2 Config
# ═══════════════════════════════════════════════════════════════

@dataclass
class GPT2Config:
    n_layer: int = 12
    n_embd: int = 768
    n_head: int = 12
    head_dim: int = 64      # n_embd // n_head
    n_inner: int = 3072      # 4 * n_embd
    vocab_size: int = 50257
    n_positions: int = 1024
    layer_norm_epsilon: float = 1e-5
    activation: str = "gelu_new"

    # aned compilation limit: in_ch * out_ch < ~1.3M
    # 768 * 3072 = 2.36M > limit → need splitting
    aned_param_limit: int = 1_200_000

    def needs_split(self, in_ch: int, out_ch: int) -> bool:
        return in_ch * out_ch > self.aned_param_limit


# ═══════════════════════════════════════════════════════════════
# Safetensors Loader
# ═══════════════════════════════════════════════════════════════

def load_safetensors(model_path: str) -> Dict[str, np.ndarray]:
    """Load all tensors from a safetensors file."""
    from safetensors import safe_open
    tensors = {}
    with safe_open(model_path, framework="numpy") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    return tensors


@dataclass
class GPT2Layer:
    """Weights for a single GPT-2 transformer layer."""
    layer_idx: int
    # LayerNorm 1 (pre-attention)
    ln_1_weight: np.ndarray    # [768]
    ln_1_bias: np.ndarray      # [768]
    # Attention
    c_attn_weight: np.ndarray  # [768, 2304] Conv1D format
    c_attn_bias: np.ndarray    # [2304]
    c_proj_weight: np.ndarray  # [768, 768] Conv1D format
    c_proj_bias: np.ndarray    # [768]
    causal_mask: np.ndarray    # [1, 1, 1024, 1024]
    # LayerNorm 2 (pre-FFN)
    ln_2_weight: np.ndarray    # [768]
    ln_2_bias: np.ndarray      # [768]
    # FFN
    c_fc_weight: np.ndarray    # [768, 3072] Conv1D format
    c_fc_bias: np.ndarray      # [3072]
    c_proj_ffn_weight: np.ndarray  # [3072, 768] Conv1D format
    c_proj_ffn_bias: np.ndarray    # [768]

    @property
    def W_q(self) -> np.ndarray:
        """Q projection: [out=768, in=768]"""
        return self.c_attn_weight[:, :768].T.copy()

    @property
    def W_k(self) -> np.ndarray:
        """K projection: [out=768, in=768]"""
        return self.c_attn_weight[:, 768:1536].T.copy()

    @property
    def W_v(self) -> np.ndarray:
        """V projection: [out=768, in=768]"""
        return self.c_attn_weight[:, 1536:2304].T.copy()

    @property
    def W_o(self) -> np.ndarray:
        """Output projection: [out=768, in=768]"""
        return self.c_proj_weight.T.copy()

    @property
    def bias_q(self) -> np.ndarray:
        return self.c_attn_bias[:768]

    @property
    def bias_k(self) -> np.ndarray:
        return self.c_attn_bias[768:1536]

    @property
    def bias_v(self) -> np.ndarray:
        return self.c_attn_bias[1536:2304]

    @property
    def W_fc(self) -> np.ndarray:
        """FFN up: [out=3072, in=768]"""
        return self.c_fc_weight.T.copy()

    @property
    def W_fc_down(self) -> np.ndarray:
        """FFN down: [out=768, in=3072]"""
        return self.c_proj_ffn_weight.T.copy()


@dataclass
class GPT2Model:
    """Complete GPT-2 model weights."""
    config: GPT2Config
    wte: np.ndarray            # [50257, 768] token embeddings
    wpe: np.ndarray            # [1024, 768] position embeddings
    layers: List[GPT2Layer]
    ln_f_weight: np.ndarray    # [768] final layernorm
    ln_f_bias: np.ndarray      # [768]

    @classmethod
    def from_safetensors(cls, path: str) -> 'GPT2Model':
        """Load GPT-2 from safetensors file."""
        tensors = load_safetensors(path)
        config = GPT2Config()

        layers = []
        for i in range(config.n_layer):
            prefix = f"h.{i}"
            layers.append(GPT2Layer(
                layer_idx=i,
                ln_1_weight=tensors[f"{prefix}.ln_1.weight"],
                ln_1_bias=tensors[f"{prefix}.ln_1.bias"],
                c_attn_weight=tensors[f"{prefix}.attn.c_attn.weight"],
                c_attn_bias=tensors[f"{prefix}.attn.c_attn.bias"],
                c_proj_weight=tensors[f"{prefix}.attn.c_proj.weight"],
                c_proj_bias=tensors[f"{prefix}.attn.c_proj.bias"],
                causal_mask=tensors[f"{prefix}.attn.bias"],
                ln_2_weight=tensors[f"{prefix}.ln_2.weight"],
                ln_2_bias=tensors[f"{prefix}.ln_2.bias"],
                c_fc_weight=tensors[f"{prefix}.mlp.c_fc.weight"],
                c_fc_bias=tensors[f"{prefix}.mlp.c_fc.bias"],
                c_proj_ffn_weight=tensors[f"{prefix}.mlp.c_proj.weight"],
                c_proj_ffn_bias=tensors[f"{prefix}.mlp.c_proj.bias"],
            ))

        return cls(
            config=config,
            wte=tensors["wte.weight"],
            wpe=tensors["wpe.weight"],
            layers=layers,
            ln_f_weight=tensors["ln_f.weight"],
            ln_f_bias=tensors["ln_f.bias"],
        )


# ═══════════════════════════════════════════════════════════════
# Conv splitting for large dimensions (aned hang workaround)
# ═══════════════════════════════════════════════════════════════

def split_conv_output(weight: np.ndarray, max_params: int = 1_200_000
                      ) -> List[Tuple[np.ndarray, int, int]]:
    """Split a conv along output dimension to stay under aned param limit.

    Weight shape: [out_ch, in_ch]
    Returns list of (sub_weight, in_ch, out_ch) tuples.
    Results are concatenated along output dimension.
    """
    out_ch, in_ch = weight.shape
    if in_ch * out_ch <= max_params:
        return [(weight, in_ch, out_ch)]

    # Find chunk size that stays under limit
    chunk_out = max_params // in_ch
    # Round down to multiple of 16 (ANE alignment)
    chunk_out = (chunk_out // 16) * 16
    if chunk_out < 16:
        chunk_out = 16

    chunks = []
    for start in range(0, out_ch, chunk_out):
        end = min(start + chunk_out, out_ch)
        sub_w = weight[start:end, :]
        chunks.append((sub_w.copy(), in_ch, end - start))

    return chunks


def split_conv_input(weight: np.ndarray, max_params: int = 1_200_000
                     ) -> List[Tuple[np.ndarray, int, int, int]]:
    """Split a conv along input dimension to stay under aned param limit.

    Weight shape: [out_ch, in_ch]
    Returns list of (sub_weight, in_ch, out_ch, input_start) tuples.
    Results must be SUMMED (not concatenated).
    """
    out_ch, in_ch = weight.shape
    if in_ch * out_ch <= max_params:
        return [(weight, in_ch, out_ch, 0)]

    chunk_in = max_params // out_ch
    chunk_in = (chunk_in // 16) * 16
    if chunk_in < 16:
        chunk_in = 16

    chunks = []
    for start in range(0, in_ch, chunk_in):
        end = min(start + chunk_in, in_ch)
        sub_w = weight[:, start:end]
        chunks.append((sub_w.copy(), end - start, out_ch, start))

    return chunks


# ═══════════════════════════════════════════════════════════════
# .mlmodelc Generation for GPT-2 ops
# ═══════════════════════════════════════════════════════════════

def gen_gelu_mlmodelc(output_dir: str, dim: int):
    """Generate .mlmodelc for GELU activation (mode 19 = erf-GELU PWL)."""
    from compiler import generate_mlmodelc
    layer = {
        "type": "activation",
        "name": "gelu",
        "bottom": "input",
        "top": "output",
        "mode": 19,  # GELU exact (erf) — closest to gelu_new for FP16
        "weights": {},
        "attributes": {"is_output": 1},
    }
    shapes = {
        "input": (1, 1, dim, dim),
        "output": (1, 1, dim, dim),
    }
    generate_mlmodelc(
        output_dir, [layer], shapes, [],
        inputs=[("input", [dim, 1, 1])],
        outputs=[("output", [dim, 1, 1])],
    )


# ═══════════════════════════════════════════════════════════════
# Kill Test: Single conv dispatch with real weights
# ═══════════════════════════════════════════════════════════════

def kill_test_conv(model: GPT2Model, capture_tool: str, output_dir: str):
    """Kill test: Q projection from layer 0, dispatch on ANE, compare vs NumPy.

    Steps:
      1. Extract Q weight from layer 0 (768x768)
      2. Generate .mlmodelc
      3. Compile via capture_and_eval
      4. Dispatch with test input
      5. Compare output against NumPy matmul
    """
    import subprocess

    layer = model.layers[0]
    W_q = layer.W_q  # [768, 768] in [out, in] format

    # Generate .mlmodelc
    mlmodelc_path = os.path.join(output_dir, "q_proj_test.mlmodelc")
    gen_conv_mlmodelc(mlmodelc_path, W_q.astype(np.float32), 768, 768)
    print(f"Generated .mlmodelc: {mlmodelc_path}")

    # Generate test input (random FP16 values)
    rng = np.random.RandomState(42)
    x_fp32 = rng.randn(768).astype(np.float32)
    x_fp16 = x_fp32.astype(np.float16)

    # Reference output (NumPy)
    # Conv1x1 computes: y = W @ x
    ref_output = (W_q.astype(np.float16) @ x_fp16.astype(np.float16)).astype(np.float16)

    print(f"Reference output (first 8): {ref_output[:8]}")
    print(f"Reference output dtype: {ref_output.dtype}")

    # Compile via capture_and_eval
    hwx_path = os.path.join(output_dir, "q_proj_test.hwx")
    print(f"\nCompiling via {capture_tool}...")
    result = subprocess.run(
        [capture_tool, mlmodelc_path, hwx_path],
        capture_output=True, text=True, timeout=30,
    )
    if result.returncode != 0:
        print(f"  COMPILE FAILED: {result.stderr}")
        return False

    if os.path.exists(hwx_path):
        print(f"  .hwx captured: {os.path.getsize(hwx_path)} bytes")
    else:
        print("  .hwx NOT captured (check ANE cache)")

    # TODO: Dispatch via ane-dispatch and compare
    # For now, verify the .mlmodelc generates and compiles
    return True


if __name__ == "__main__":
    MODEL_PATH = os.path.expanduser(
        "~/.cache/huggingface/hub/models--openai-community--gpt2/"
        "snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e/model.safetensors"
    )

    print("Loading GPT-2 from safetensors...")
    model = GPT2Model.from_safetensors(MODEL_PATH)
    print(f"  Layers: {len(model.layers)}")
    print(f"  Embedding: {model.wte.shape}")
    print(f"  Position: {model.wpe.shape}")

    # Print layer 0 weight shapes
    L0 = model.layers[0]
    print(f"\nLayer 0 weights:")
    print(f"  W_q: {L0.W_q.shape} (extracted from c_attn)")
    print(f"  W_k: {L0.W_k.shape}")
    print(f"  W_v: {L0.W_v.shape}")
    print(f"  W_o: {L0.W_o.shape}")
    print(f"  W_fc (up): {L0.W_fc.shape}")
    print(f"  W_fc_down: {L0.W_fc_down.shape}")

    # Check which convs need splitting
    config = model.config
    print(f"\nConv splitting analysis:")
    for name, shape in [
        ("Q proj", (768, 768)),
        ("K proj", (768, 768)),
        ("V proj", (768, 768)),
        ("O proj", (768, 768)),
        ("FFN up", (3072, 768)),
        ("FFN down", (768, 3072)),
    ]:
        out_ch, in_ch = shape
        needs = config.needs_split(in_ch, out_ch)
        params = in_ch * out_ch
        print(f"  {name:12s}: {in_ch}x{out_ch} = {params:>10,d} params → "
              f"{'SPLIT NEEDED' if needs else 'OK'}")
