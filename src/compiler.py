#!/usr/bin/env python3
"""
ane-compiler v1.0: Compile transformer layers for Apple Neural Engine.

This module generates .mlmodelc bundles (espresso format) from model
definitions, bypassing CoreML entirely. The ANE daemon (aned) compiles
these to .hwx binaries which execute on the 16-core ANE hardware.

Architecture:
    TransformerLayerConfig (weights + dimensions)
        ↓
    compile_layer() → generates per-op .mlmodelc bundles
        ↓
    ExecutionPlan (op ordering + IOSurface routing)
        ↓
    ane-dispatch executes chain on ANE hardware

No coremltools dependency. No CoreML framework. Pure espresso format generation.

Copyright 2026 Nick Lo. MIT License.
"""

import os
import json
import struct
import shutil
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
from enum import Enum


# ═══════════════════════════════════════════════════════════════
# .mlmodelc bundle generator (espresso format)
# ═══════════════════════════════════════════════════════════════

def _write_espresso_net(path: str, layers: List[dict], weight_file: str = "model.espresso.weights"):
    """Write model.espresso.net (JSON layer graph)."""
    net = {
        "storage": weight_file,
        "analyses": {},
        "properties": {},
        "format_version": 200,
        "metadata_in_weights": [],
        "layers": layers,
    }
    with open(path, 'w') as f:
        json.dump(net, f)


def _write_espresso_shape(path: str, shapes: Dict[str, Tuple[int, int, int, int]]):
    """Write model.espresso.shape (JSON tensor dimensions in NHWK format)."""
    layer_shapes = {}
    for name, (n, h, w, k) in shapes.items():
        layer_shapes[name] = {"n": n, "h": h, "w": w, "k": k}
    with open(path, 'w') as f:
        json.dump({"layer_shapes": layer_shapes}, f)


def _write_espresso_weights(path: str, blobs: List[np.ndarray]):
    """Write model.espresso.weights (binary: version header + FP32 weight data).

    For models with weights (conv): 0x02 header + metadata + FP32 data at offset 0x40.
    For models without weights (softmax, layernorm): 8 zero bytes.
    """
    if not blobs:
        with open(path, 'wb') as f:
            f.write(b'\x00' * 8)
        return

    total_bytes = sum(b.size * 4 for b in blobs)
    with open(path, 'wb') as f:
        # Header (matches coremltools format)
        header = bytearray(0x40)
        struct.pack_into('<I', header, 0x00, 2)       # version
        struct.pack_into('<I', header, 0x10, 0x18)     # offset to blob table
        struct.pack_into('<I', header, 0x18, 1)        # num blobs
        struct.pack_into('<I', header, 0x20, total_bytes)  # total weight bytes
        f.write(header)
        # Weight data at offset 0x40
        for blob in blobs:
            f.write(blob.astype(np.float32).tobytes())


def _write_metadata(path: str, inputs: List[Tuple[str, List[int]]],
                    outputs: List[Tuple[str, List[int]]]):
    """Write metadata.json (CoreML schema — minimal valid version)."""
    def make_schema(name, shape):
        return {
            "name": name,
            "type": "MultiArray",
            "shape": shape,
            "dataType": "Float16",
        }

    meta = {
        "specificationVersion": 4,
        "isUpdatable": False,
        "modelType": {"name": "MLModelType_neuralNetwork"},
        "computePrecision": "Float16",
        "inputSchema": [make_schema(n, s) for n, s in inputs],
        "outputSchema": [make_schema(n, s) for n, s in outputs],
    }
    with open(path, 'w') as f:
        json.dump(meta, f)


def _write_coremldata(path: str):
    """Write minimal coremldata.bin (protobuf stub)."""
    # Minimal valid protobuf that passes CoreML validation
    # Field 1 (specificationVersion) = 4
    with open(path, 'wb') as f:
        f.write(b'\x08\x04')  # varint field 1 = 4


def generate_mlmodelc(output_dir: str, layers: List[dict],
                      shapes: Dict[str, Tuple[int, int, int, int]],
                      weight_blobs: List[np.ndarray],
                      inputs: List[Tuple[str, List[int]]],
                      outputs: List[Tuple[str, List[int]]]):
    """Generate a complete .mlmodelc bundle from scratch.

    No coremltools dependency. Produces espresso-format .mlmodelc that
    aned can compile directly to ANE .hwx.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'model'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'analytics'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'neural_network_optionals'), exist_ok=True)

    _write_espresso_net(os.path.join(output_dir, 'model.espresso.net'), layers)
    _write_espresso_shape(os.path.join(output_dir, 'model.espresso.shape'), shapes)
    _write_espresso_weights(os.path.join(output_dir, 'model.espresso.weights'), weight_blobs)
    _write_metadata(os.path.join(output_dir, 'metadata.json'), inputs, outputs)
    _write_coremldata(os.path.join(output_dir, 'coremldata.bin'))

    # Stub files required by CoreML
    for sub in ['model', 'analytics', 'neural_network_optionals']:
        stub = os.path.join(output_dir, sub, 'coremldata.bin')
        with open(stub, 'wb') as f:
            f.write(b'\x00' * 64)


# ═══════════════════════════════════════════════════════════════
# Per-op .mlmodelc generators
# ═══════════════════════════════════════════════════════════════

def gen_conv_mlmodelc(output_dir: str, weights: np.ndarray,
                      in_ch: int, out_ch: int,
                      fused_relu: bool = False, name: str = "conv"):
    """Generate .mlmodelc for a conv1x1 (linear projection)."""
    layer = {
        "type": "convolution",
        "name": name,
        "debug_info": name,
        "bottom": "input",
        "top": "output",
        "K": int(in_ch),
        "C": int(out_ch),
        "Nx": 1, "Ny": 1,
        "n_groups": 1,
        "n_parallel": 1,
        "has_biases": 0,
        "has_batch_norm": 0,
        "blob_weights": 1,
        "pad_t": 0, "pad_b": 0, "pad_l": 0, "pad_r": 0,
        "pad_mode": 0, "pad_fill_mode": 0, "pad_value": 0,
        "stride_x": 1, "stride_y": 1,
        "fused_relu": 1 if fused_relu else 0,
        "fused_tanh": 0,
        "weights": {},
        "attributes": {"is_output": 1},
    }
    shapes = {
        "input": (1, 1, in_ch, in_ch),
        "output": (1, 1, out_ch, out_ch),
    }
    generate_mlmodelc(
        output_dir, [layer], shapes, [weights],
        inputs=[("input", [int(in_ch), 1, 1])],
        outputs=[("output", [int(out_ch), 1, 1])],
    )


def gen_softmax_mlmodelc(output_dir: str, dim: int):
    """Generate .mlmodelc for softmax."""
    layer = {
        "type": "softmax",
        "name": "softmax",
        "bottom": "input",
        "top": "output",
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


def gen_layernorm_mlmodelc(output_dir: str, dim: int, epsilon: float = 1e-5):
    """Generate .mlmodelc for layer normalization (l2_normalize with normalization_mode=1)."""
    layer = {
        "type": "l2_normalize",
        "name": "mvn",
        "debug_info": "mvn",
        "bottom": "input",
        "top": "output",
        "normalization_mode": 1,
        "axis": 2,
        "eps": float(epsilon),
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


def gen_add_mlmodelc(output_dir: str, dim: int):
    """Generate .mlmodelc for two-input elementwise add (residual)."""
    layer = {
        "type": "elementwise",
        "name": "add",
        "bottom": "input_a,input_b",
        "operation": 0,  # ADD
        "weights": {},
        "attributes": {"is_output": 1},
    }
    shapes = {
        "input_a": (1, 1, dim, dim),
        "input_b": (1, 1, dim, dim),
        "add": (1, 1, dim, dim),
    }
    generate_mlmodelc(
        output_dir, [layer], shapes, [],
        inputs=[("input_a", [dim, 1, 1]), ("input_b", [dim, 1, 1])],
        outputs=[("output", [dim, 1, 1])],
    )


# ═══════════════════════════════════════════════════════════════
# Execution plan
# ═══════════════════════════════════════════════════════════════

class OpType(Enum):
    CONV = "conv"
    SOFTMAX = "softmax"
    LAYERNORM = "layernorm"
    ADD_CPU = "add_cpu"  # CPU residual add


@dataclass
class Op:
    """A single operation in the execution plan."""
    name: str
    op_type: OpType
    mlmodelc_path: str
    input_names: List[str]
    output_name: str
    in_channels: int
    out_channels: int


@dataclass
class ExecutionPlan:
    """Ordered list of operations with IOSurface routing."""
    ops: List[Op]
    buffer_names: List[str]  # all named IOSurface buffers
    input_name: str
    output_name: str

    def summary(self) -> str:
        lines = [f"Execution plan: {len(self.ops)} ops"]
        ane_count = sum(1 for o in self.ops if o.op_type != OpType.ADD_CPU)
        cpu_count = len(self.ops) - ane_count
        lines.append(f"  ANE dispatches: {ane_count}")
        lines.append(f"  CPU operations: {cpu_count}")
        lines.append(f"  IOSurface buffers: {len(self.buffer_names)}")
        lines.append("")
        for i, op in enumerate(self.ops):
            device = "CPU" if op.op_type == OpType.ADD_CPU else "ANE"
            ins = ", ".join(op.input_names)
            lines.append(f"  [{i:2d}] {device:3s} {op.name:20s} ({ins}) → {op.output_name}")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
# compile_layer(): the main API
# ═══════════════════════════════════════════════════════════════

@dataclass
class TransformerLayerConfig:
    """Configuration for a transformer layer."""
    hidden_dim: int
    n_heads: int
    head_dim: int
    ffn_dim: int
    activation: str = "relu"  # "relu", "silu", "gelu", or "custom_pwl"
    weights: Dict[str, np.ndarray] = field(default_factory=dict)
    # Required weight keys: W_q, W_k, W_v, W_o, W_gate, W_down
    # Optional: W_qk (matmul proxy), W_sv (matmul proxy)


def compile_layer(config: TransformerLayerConfig,
                  output_dir: str) -> ExecutionPlan:
    """Compile a transformer layer to per-op .mlmodelc bundles.

    Generates 11+ .mlmodelc files in output_dir, one per ANE dispatch.
    Returns an ExecutionPlan describing the operation ordering and
    IOSurface routing for ane-dispatch.

    Args:
        config: transformer layer configuration with weights
        output_dir: directory to write .mlmodelc bundles

    Returns:
        ExecutionPlan for ane-dispatch execution
    """
    os.makedirs(output_dir, exist_ok=True)
    dim = config.hidden_dim
    ffn_dim = config.ffn_dim
    W = config.weights
    fused_act = config.activation == "relu"  # only relu can be fused in espresso

    ops = []
    buffers = {"x", "output"}

    def add_conv(name, w_key, in_ch, out_ch, input_name, output_name, fused=False):
        path = os.path.join(output_dir, f"{name}.mlmodelc")
        gen_conv_mlmodelc(path, W[w_key], in_ch, out_ch, fused_relu=fused, name=name)
        ops.append(Op(name, OpType.CONV, path, [input_name], output_name, in_ch, out_ch))
        buffers.add(input_name)
        buffers.add(output_name)

    def add_softmax(name, d, input_name, output_name):
        path = os.path.join(output_dir, f"{name}.mlmodelc")
        gen_softmax_mlmodelc(path, d)
        ops.append(Op(name, OpType.SOFTMAX, path, [input_name], output_name, d, d))
        buffers.add(input_name)
        buffers.add(output_name)

    def add_layernorm(name, d, input_name, output_name):
        path = os.path.join(output_dir, f"{name}.mlmodelc")
        gen_layernorm_mlmodelc(path, d)
        ops.append(Op(name, OpType.LAYERNORM, path, [input_name], output_name, d, d))
        buffers.add(input_name)
        buffers.add(output_name)

    def add_residual(name, a, b, output_name):
        ops.append(Op(name, OpType.ADD_CPU, "", [a, b], output_name, dim, dim))
        buffers.add(output_name)

    # 1. LayerNorm 1
    add_layernorm("ln1", dim, "x", "ln1_out")

    # 2-4. QKV projections (parallel, all from ln1_out)
    add_conv("q_proj", "W_q", dim, dim, "ln1_out", "q")
    add_conv("k_proj", "W_k", dim, dim, "ln1_out", "k")
    add_conv("v_proj", "W_v", dim, dim, "ln1_out", "v")

    # 5. Q@K^T proxy (linear transform of Q)
    add_conv("qk_matmul", "W_qk", dim, dim, "q", "qk")

    # 6. Softmax
    add_softmax("attn_softmax", dim, "qk", "scores")

    # 7. Scores@V proxy (linear transform of scores)
    add_conv("sv_matmul", "W_sv", dim, dim, "scores", "sv")

    # 8. Output projection
    add_conv("o_proj", "W_o", dim, dim, "sv", "attn_out")

    # 9. Residual 1 (CPU)
    add_residual("residual1", "x", "attn_out", "r1")

    # 10. LayerNorm 2
    add_layernorm("ln2", dim, "r1", "ln2_out")

    # 11. FFN gate (with fused activation if relu)
    add_conv("ffn_gate", "W_gate", dim, ffn_dim, "ln2_out", "gate_out",
             fused=fused_act)

    # 12. FFN down
    add_conv("ffn_down", "W_down", ffn_dim, dim, "gate_out", "ffn_out")

    # 13. Residual 2 (CPU)
    add_residual("residual2", "r1", "ffn_out", "output")

    return ExecutionPlan(
        ops=ops,
        buffer_names=sorted(buffers),
        input_name="x",
        output_name="output",
    )
