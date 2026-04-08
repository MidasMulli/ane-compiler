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


def _write_espresso_weights(path: str, blobs: List[np.ndarray],
                            bias: Optional[np.ndarray] = None):
    """Write model.espresso.weights (binary weight data).

    For models with weights (conv): header + FP32 data.
    For models without weights (softmax, layernorm): 8 zero bytes.

    When bias is provided, uses v4 format which places bias before weights
    with proper alignment. Without bias, uses v2 format (weights at 0x40).
    """
    if not blobs:
        with open(path, 'wb') as f:
            f.write(b'\x00' * 8)
        return

    weight_bytes = sum(b.size * 4 for b in blobs)

    if bias is not None:
        # v4 format: bias before weights with 0x1000-aligned sections
        bias_data = bias.astype(np.float32).tobytes()
        # Bias section is padded to accommodate the v4 blob table structure.
        # The data offset field (0x20) specifies the gap between bias and weights.
        # Observed pattern: bias at 0x80, weights at 0x80 + data_offset.
        # data_offset is the bias section size rounded up to 0x1000 alignment,
        # with a minimum of 0x3000 for small bias vectors.
        bias_section = max(0x3000, ((len(bias_data) + 0xFFF) // 0x1000) * 0x1000)

        with open(path, 'wb') as f:
            header = bytearray(0x80)
            struct.pack_into('<I', header, 0x00, 4)           # version 4
            struct.pack_into('<I', header, 0x10, 0x38)        # blob table offset
            struct.pack_into('<I', header, 0x18, 1)           # num blobs
            struct.pack_into('<I', header, 0x20, bias_section) # data offset (bias→weights gap)
            struct.pack_into('<I', header, 0x28, 2)           # num data entries
            struct.pack_into('<I', header, 0x38, 3)           # blob_weights type
            struct.pack_into('<I', header, 0x40, weight_bytes) # weight data size
            f.write(header)

            # Bias data at 0x80
            f.write(bias_data)
            # Pad to weights start
            f.write(b'\x00' * (bias_section - len(bias_data)))
            # Weight data
            for blob in blobs:
                f.write(blob.astype(np.float32).tobytes())
    else:
        # v2 format: weights at 0x40
        with open(path, 'wb') as f:
            header = bytearray(0x40)
            struct.pack_into('<I', header, 0x00, 2)       # version
            struct.pack_into('<I', header, 0x10, 0x18)     # offset to blob table
            struct.pack_into('<I', header, 0x18, 1)        # num blobs
            struct.pack_into('<I', header, 0x20, weight_bytes)  # total weight bytes
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
                      outputs: List[Tuple[str, List[int]]],
                      bias: Optional[np.ndarray] = None):
    """Generate a complete .mlmodelc bundle from scratch.

    No coremltools dependency. Produces espresso-format .mlmodelc that
    aned can compile directly to ANE .hwx.

    Args:
        bias: Optional bias vector (1D array). When provided, uses v4
              weight format and sets has_biases=1 in the layer config.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'model'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'analytics'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'neural_network_optionals'), exist_ok=True)

    _write_espresso_net(os.path.join(output_dir, 'model.espresso.net'), layers)
    _write_espresso_shape(os.path.join(output_dir, 'model.espresso.shape'), shapes)
    _write_espresso_weights(os.path.join(output_dir, 'model.espresso.weights'),
                            weight_blobs, bias=bias)
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
                      bias: Optional[np.ndarray] = None,
                      fused_relu: bool = False, name: str = "conv"):
    """Generate .mlmodelc for a conv1x1 (linear projection).

    Uses inner_product layer type (matches coremltools output) for correct
    non-square dimension support. The convolution type only works for
    square (in_ch == out_ch) configurations with our espresso format.

    Args:
        weights: Weight matrix [out_ch, in_ch] as FP32.
        bias: Optional bias vector [out_ch] as FP32. When provided,
              bias is fused into the .hwx — ANE computes W@x+b in one dispatch.
    """
    has_bias = bias is not None
    layer = {
        "type": "inner_product",
        "name": name,
        "debug_info": name,
        "bottom": "input",
        "top": "output",
        "nB": int(in_ch),
        "nC": int(out_ch),
        "has_biases": 1 if has_bias else 0,
        "blob_weights": 3 if has_bias else 1,
        "has_relu": 1 if fused_relu else 0,
        "has_tanh": 0,
        "has_prelu": 0,
        "weights": {},
        "attributes": {"is_output": 1},
    }
    if has_bias:
        layer["blob_biases"] = 1
    shapes = {
        "input": (1, 1, 1, in_ch),
        "output": (1, 1, 1, out_ch),
    }
    generate_mlmodelc(
        output_dir, [layer], shapes, [weights],
        inputs=[("input", [int(in_ch), 1, 1])],
        outputs=[("output", [int(out_ch), 1, 1])],
        bias=bias,
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
    """Generate .mlmodelc for layer normalization (l2_normalize with normalization_mode=1).

    Shape uses K=dim (channel dimension) matching coremltools MVN output.
    aned compiles this to a 4-pass pipeline (572B __text at dim=768).
    """
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
        "input": (1, 1, 1, dim),
        "output": (1, 1, 1, dim),
    }
    generate_mlmodelc(
        output_dir, [layer], shapes, [],
        inputs=[("input", [dim, 1, 1])],
        outputs=[("output", [dim, 1, 1])],
    )


def gen_batchnorm_mlmodelc(output_dir: str, dim: int,
                           gamma: np.ndarray, beta: np.ndarray,
                           name: str = "bn"):
    """Generate .mlmodelc for batchnorm (scale+bias, used for LN affine params).

    Uses batchnorm layer with mean=0, variance=1, epsilon=0 to implement
    a per-channel affine transform: output = gamma * input + beta.

    When combined with a preceding l2_normalize (MVN) dispatch, this
    implements LayerNorm with learned affine parameters (elementwise_affine=True).

    ANE compiles this to a single dispatch: 320B __text + 3072B __const (dim=768).
    The __const stores [beta/gamma (FP16), gamma (FP16)] — hardware computes
    output = gamma * (input + beta/gamma) = gamma*input + beta.

    Args:
        output_dir: path for .mlmodelc output
        dim: channel dimension (must match gamma/beta length)
        gamma: per-channel scale [dim] (FP32)
        beta: per-channel bias [dim] (FP32)
        name: layer name
    """
    layer = {
        "type": "batchnorm",
        "name": name,
        "debug_info": name,
        "bottom": "input",
        "top": "output",
        "blob_batchnorm_params": 1,
        "C": int(dim),
        "weights": {},
        "attributes": {"is_output": 1},
    }
    shapes = {
        "input": (1, 1, 1, dim),
        "output": (1, 1, 1, dim),
    }
    # Batchnorm weights: interleaved [gamma[i], beta[i], mean[i], var[i]] per channel
    bn_params = np.zeros((dim, 4), dtype=np.float32)
    bn_params[:, 0] = gamma.astype(np.float32)
    bn_params[:, 1] = beta.astype(np.float32)
    bn_params[:, 2] = 0.0   # mean
    bn_params[:, 3] = 1.0   # variance (eps=0 so sqrt(var+eps)=1)
    generate_mlmodelc(
        output_dir, [layer], shapes, [bn_params.flatten()],
        inputs=[("input", [int(dim), 1, 1])],
        outputs=[("output", [int(dim), 1, 1])],
    )


def _write_espresso_weights_multi(path: str,
                                   layers: List[Tuple[np.ndarray, Optional[np.ndarray]]]):
    """Write model.espresso.weights for multi-layer models.

    Version 8 format matching coremltools output. Layout:
      Header (0xC0) + [bias1, pad, weights1, bias2, pad, weights2, ...]

    Blob table has 2*n_layers - 1 entries:
      weights1 (id=3), bias2_section (id=5), weights2 (id=7), ...
    First bias gap stored in header[0x20]. Subsequent bias gaps in table entries.

    Blob indices: bias=1,5,9,...  weights=3,7,11,...  (stride 4 per layer).
    v8 bias_section formula: max(0x3000, 4 * bias_data_size).
    """
    weighted = [(w, b) for w, b in layers if w is not None]
    n_layers = len(weighted)

    # Compute bias sections (v8 formula: 4x raw bias size, min 0x3000)
    data_sections = []
    for w, b in weighted:
        w_bytes = w.astype(np.float32).tobytes()
        if b is not None:
            b_bytes = b.astype(np.float32).tobytes()
            bias_section = max(0x3000, 4 * len(b_bytes))
        else:
            b_bytes = b''
            bias_section = 0
        data_sections.append((w_bytes, b_bytes, bias_section))

    # Blob table: weights entries + bias section entries for layers 2+
    # Entry layout: [blob_id(+0), pad(+4), size(+8), pad(+12), next_id(+16), pad(+20..+31)]
    # Total entries: n_layers (weights) + (n_layers - 1) (bias sections) = 2*n_layers - 1
    n_entries = 2 * n_layers - 1
    blob_table_offset = 0x38
    header_size = blob_table_offset + n_entries * 0x20
    # Pad header to 0x40 alignment (matches v4 and coremltools convention)
    header_size = ((header_size + 0x3F) // 0x40) * 0x40

    with open(path, 'wb') as f:
        header = bytearray(header_size)
        struct.pack_into('<I', header, 0x00, 8)               # version 8
        struct.pack_into('<I', header, 0x10, blob_table_offset)
        struct.pack_into('<I', header, 0x18, 1)               # num blob groups
        struct.pack_into('<I', header, 0x20, data_sections[0][2])  # first bias gap
        struct.pack_into('<I', header, 0x28, 2)               # num data entries

        # Build blob table: interleave weight entries and bias section entries
        entry_idx = 0
        for i in range(n_layers):
            # Weight entry for layer i
            off = blob_table_offset + entry_idx * 0x20
            w_blob_id = 3 + i * 4          # 3, 7, 11, ...
            struct.pack_into('<I', header, off, w_blob_id)
            struct.pack_into('<I', header, off + 8, len(data_sections[i][0]))
            if i < n_layers - 1:
                struct.pack_into('<I', header, off + 16, w_blob_id + 1)
            entry_idx += 1

            # Bias section entry for layer i+1 (not for last layer)
            if i < n_layers - 1:
                off = blob_table_offset + entry_idx * 0x20
                b_blob_id = 5 + i * 4      # 5, 9, 13, ...
                struct.pack_into('<I', header, off, b_blob_id)
                struct.pack_into('<I', header, off + 8, data_sections[i + 1][2])
                struct.pack_into('<I', header, off + 16, b_blob_id + 1)
                entry_idx += 1

        f.write(header)

        # Data: interleaved [bias, pad, weights] per layer
        for w_bytes, b_bytes, bias_section in data_sections:
            if b_bytes:
                f.write(b_bytes)
                f.write(b'\x00' * (bias_section - len(b_bytes)))
            f.write(w_bytes)


def gen_fused_ffn_mlmodelc(output_dir: str, W_up: np.ndarray, bias_up: np.ndarray,
                           W_down: np.ndarray, bias_down: np.ndarray,
                           in_ch: int, hidden_ch: int, out_ch: int):
    """Generate fused FFN .mlmodelc: inner_product(bias) → GELU → inner_product(bias).

    Pure espresso format — no coremltools dependency. aned compiles and fuses
    the multi-layer graph into a single .hwx with hardware GELU (mode 19).

    Args:
        W_up: Gate/up projection [hidden_ch, in_ch] FP32
        bias_up: Gate bias [hidden_ch] FP32
        W_down: Down projection [out_ch, hidden_ch] FP32
        bias_down: Down bias [out_ch] FP32
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'model'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'analytics'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'neural_network_optionals'), exist_ok=True)

    # Espresso layer graph: inner_product → activation(GELU) → inner_product
    layers = [
        {
            "type": "inner_product",
            "name": "fc_up",
            "debug_info": "fc_up",
            "bottom": "input",
            "top": "fc_up_out",
            "nB": int(in_ch),
            "nC": int(hidden_ch),
            "has_biases": 1,
            "blob_weights": 3,
            "blob_biases": 1,
            "has_relu": 0,
            "has_tanh": 0,
            "has_prelu": 0,
            "weights": {},
        },
        {
            "type": "activation",
            "name": "gelu",
            "bottom": "fc_up_out",
            "top": "gelu_out",
            "mode": 19,
            "weights": {},
        },
        {
            "type": "inner_product",
            "name": "fc_down",
            "debug_info": "fc_down",
            "bottom": "gelu_out",
            "top": "output",
            "nB": int(hidden_ch),
            "nC": int(out_ch),
            "has_biases": 1,
            "blob_weights": 7,
            "blob_biases": 5,
            "has_relu": 0,
            "has_tanh": 0,
            "has_prelu": 0,
            "weights": {},
            "attributes": {"is_output": 1},
        },
    ]

    shapes = {
        "input": (1, 1, 1, in_ch),
        "fc_up_out": (1, 1, 1, hidden_ch),
        "gelu_out": (1, 1, 1, hidden_ch),
        "output": (1, 1, 1, out_ch),
    }

    # Write espresso.net
    _write_espresso_net(os.path.join(output_dir, 'model.espresso.net'), layers)

    # Write espresso.shape
    _write_espresso_shape(os.path.join(output_dir, 'model.espresso.shape'), shapes)

    # Write multi-layer weights (version 8)
    _write_espresso_weights_multi(
        os.path.join(output_dir, 'model.espresso.weights'),
        [(W_up.astype(np.float32), bias_up.astype(np.float32)),
         (W_down.astype(np.float32), bias_down.astype(np.float32))],
    )

    # Write metadata
    _write_metadata(
        os.path.join(output_dir, 'metadata.json'),
        inputs=[("input", [int(in_ch), 1, 1])],
        outputs=[("output", [int(out_ch), 1, 1])],
    )
    _write_coremldata(os.path.join(output_dir, 'coremldata.bin'))

    # Stub files
    for sub in ['model', 'analytics', 'neural_network_optionals']:
        with open(os.path.join(output_dir, sub, 'coremldata.bin'), 'wb') as f:
            f.write(b'\x00' * 64)


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
    BATCHNORM = "batchnorm"  # scale+bias (affine params for LN)
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
    # Optional affine LN keys: ln1_gamma, ln1_beta, ln2_gamma, ln2_beta
    # When present, LayerNorm uses affine params (MVN + batchnorm dispatch)


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

    def add_layernorm(name, d, input_name, output_name,
                      gamma_key=None, beta_key=None):
        """Add LayerNorm (MVN) and optional affine (batchnorm) dispatches.

        When gamma_key/beta_key are provided and present in config.weights,
        generates two dispatches: MVN → batchnorm (gamma*x + beta).
        Otherwise generates a single MVN dispatch.
        """
        # MVN dispatch (mean-variance normalization)
        mvn_output = output_name if gamma_key is None else f"{name}_mvn"
        path = os.path.join(output_dir, f"{name}.mlmodelc")
        gen_layernorm_mlmodelc(path, d)
        ops.append(Op(name, OpType.LAYERNORM, path, [input_name], mvn_output, d, d))
        buffers.add(input_name)
        buffers.add(mvn_output)

        # Affine dispatch (scale + bias via batchnorm)
        if gamma_key and gamma_key in W:
            bn_name = f"{name}_affine"
            bn_path = os.path.join(output_dir, f"{bn_name}.mlmodelc")
            gen_batchnorm_mlmodelc(bn_path, d, W[gamma_key], W[beta_key])
            ops.append(Op(bn_name, OpType.BATCHNORM, bn_path,
                         [mvn_output], output_name, d, d))
            buffers.add(output_name)

    def add_residual(name, a, b, output_name):
        ops.append(Op(name, OpType.ADD_CPU, "", [a, b], output_name, dim, dim))
        buffers.add(output_name)

    # 1. LayerNorm 1 (with affine if gamma/beta provided)
    add_layernorm("ln1", dim, "x", "ln1_out",
                  gamma_key="ln1_gamma", beta_key="ln1_beta")

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

    # 10. LayerNorm 2 (with affine if gamma/beta provided)
    add_layernorm("ln2", dim, "r1", "ln2_out",
                  gamma_key="ln2_gamma", beta_key="ln2_beta")

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
