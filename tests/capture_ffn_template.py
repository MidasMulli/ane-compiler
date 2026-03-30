#!/usr/bin/env python3
"""Capture a fused FFN .hwx template from coremltools compilation.

Compiles a conv→SiLU→conv model through CoreML's ANE compiler,
captures the resulting .hwx from the ANE cache, and saves it as
a template for ane-compiler emission.

Also captures individual conv_silu and conv_only references for
comparison.
"""

import os
import sys
import time
import glob
import shutil
import struct
import numpy as np
import coremltools as ct
from coremltools.models.neural_network import NeuralNetworkBuilder

CACHE_BASE = "/Library/Caches/com.apple.aned"
TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), '..', 'templates')


def find_latest_hwx(before_ts=None):
    """Find the most recently modified .hwx in ANE cache."""
    pattern = os.path.join(CACHE_BASE, "**", "model.hwx")
    candidates = []
    for p in glob.glob(pattern, recursive=True):
        mtime = os.path.getmtime(p)
        if before_ts is None or mtime > before_ts:
            candidates.append((mtime, p))
    candidates.sort(reverse=True)
    return candidates


def compile_and_capture(model, name, out_dir):
    """Compile a CoreML model and capture the ANE .hwx."""
    # Timestamp before compilation
    ts = time.time()

    # Save and predict (triggers ANE compilation + cache write)
    path = f"/tmp/ane_compiler_{name}.mlpackage"
    model.save(path)

    # Predict to trigger ANE compilation + cache write
    # Use the model spec to determine input shapes
    spec = model.get_spec()
    inputs = {}
    for inp in spec.description.input:
        if inp.type.HasField('multiArrayType'):
            shape = tuple(inp.type.multiArrayType.shape)
            inputs[inp.name] = np.random.randn(*shape).astype(np.float32)
        elif inp.type.HasField('tensorType'):
            shape = tuple(d.constant.size for d in inp.type.tensorType.shape.dimensions)
            inputs[inp.name] = np.random.randn(*shape).astype(np.float32)

    model.predict(inputs)
    time.sleep(0.5)  # Wait for cache write

    # Find new .hwx files
    new_hwx = find_latest_hwx(ts)
    if not new_hwx:
        print(f"  WARNING: No .hwx captured for {name} (model likely ran on GPU)")
        return None

    # Copy the most recent one
    src = new_hwx[0][1]
    dst = os.path.join(out_dir, f"{name}.hwx")
    shutil.copy2(src, dst)

    data = open(dst, 'rb').read()
    ncmds = struct.unpack_from('<I', data, 0x10)[0]
    # Find __text size
    text_sz = 0
    kern0_sz = 0
    offset = 32
    for _ in range(min(ncmds, 50)):
        if offset + 8 > len(data):
            break
        cmd = struct.unpack_from('<I', data, offset)[0]
        cmdsize = struct.unpack_from('<I', data, offset + 4)[0]
        if cmdsize == 0:
            break
        if cmd == 0x19:
            segname = data[offset+8:offset+24].split(b'\x00')[0].decode('ascii')
            nsects = struct.unpack_from('<I', data, offset + 64)[0]
            for s in range(min(nsects, 10)):
                s_off = offset + 72 + s * 80
                if s_off + 80 > len(data):
                    break
                sectname = data[s_off:s_off+16].split(b'\x00')[0].decode('ascii')
                size = struct.unpack_from('<Q', data, s_off + 40)[0]
                if segname == '__TEXT' and sectname == '__text':
                    text_sz = size
                elif segname == '__KERN_0':
                    kern0_sz = size
        offset += cmdsize

    print(f"  {name}: {len(data)}B, ncmds={ncmds}, __text={text_sz}B, __kern0={kern0_sz}B")
    return dst


def build_ffn_model(in_ch=8, hidden_ch=8, out_ch=8):
    """Build a conv→SiLU→conv FFN model via NeuralNetworkBuilder.

    This creates: input → conv1x1 (gate) → SiLU → conv1x1 (down) → output
    Using fixed known weights for reproducibility.
    """
    np.random.seed(42)

    builder = NeuralNetworkBuilder(
        input_features=[('input', ct.models.datatypes.Array(in_ch, 1, 1))],
        output_features=[('output', ct.models.datatypes.Array(out_ch, 1, 1))],
    )

    # Gate projection: [in_ch] → [hidden_ch]
    gate_w = np.random.randn(hidden_ch, in_ch, 1, 1).astype(np.float32) * 0.1
    builder.add_convolution(
        name='gate',
        kernel_channels=in_ch,
        output_channels=hidden_ch,
        height=1, width=1,
        stride_height=1, stride_width=1,
        border_mode='valid',
        groups=1,
        W=gate_w,
        b=None, has_bias=False,
        input_name='input',
        output_name='gate_out'
    )

    # SiLU activation
    # CoreML NeuralNetwork doesn't have native SiLU, use sigmoid * x
    builder.add_activation(
        name='sigmoid_gate',
        non_linearity='SIGMOID',
        input_name='gate_out',
        output_name='sigmoid_out'
    )
    builder.add_elementwise(
        name='silu',
        mode='MULTIPLY',
        input_names=['gate_out', 'sigmoid_out'],
        output_name='silu_out'
    )

    # Down projection: [hidden_ch] → [out_ch]
    down_w = np.random.randn(out_ch, hidden_ch, 1, 1).astype(np.float32) * 0.1
    builder.add_convolution(
        name='down',
        kernel_channels=hidden_ch,
        output_channels=out_ch,
        height=1, width=1,
        stride_height=1, stride_width=1,
        border_mode='valid',
        groups=1,
        W=down_w,
        b=None, has_bias=False,
        input_name='silu_out',
        output_name='output'
    )

    model = ct.models.MLModel(builder.spec)
    return model, gate_w, down_w


def build_ffn_mlprogram(in_ch=64, hidden_ch=64, out_ch=64):
    """Build FFN as mlprogram (more likely to trigger ANE fusion)."""
    import torch
    import torch.nn as nn

    class FFN(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate = nn.Linear(in_ch, hidden_ch, bias=False)
            self.down = nn.Linear(hidden_ch, out_ch, bias=False)
            self.act = nn.SiLU()

        def forward(self, x):
            return self.down(self.act(self.gate(x)))

    model = FFN()
    model.eval()

    # Fixed weights for reproducibility
    torch.manual_seed(42)
    nn.init.normal_(model.gate.weight, std=0.1)
    nn.init.normal_(model.down.weight, std=0.1)

    example = torch.randn(1, in_ch)
    traced = torch.jit.trace(model, example)

    ct_model = ct.convert(
        traced,
        inputs=[ct.TensorType(name="input", shape=(1, in_ch))],
        outputs=[ct.TensorType(name="output")],
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.macOS15,
    )

    gate_w = model.gate.weight.detach().numpy()
    down_w = model.down.weight.detach().numpy()
    return ct_model, gate_w, down_w


def main():
    os.makedirs(TEMPLATE_DIR, exist_ok=True)
    print(f"Template output: {TEMPLATE_DIR}")
    print(f"ANE cache: {CACHE_BASE}")
    print()

    # Try mlprogram path first (better ANE fusion)
    print("=== FFN via mlprogram (64x64x64) ===")
    try:
        model, gate_w, down_w = build_ffn_mlprogram(64, 64, 64)
        result = compile_and_capture(model, "ffn_silu_64x64x64_mlp", TEMPLATE_DIR)
        if result:
            print(f"  Captured: {result}")
            # Save reference weights
            np.savez(os.path.join(TEMPLATE_DIR, "ffn_silu_64x64x64_weights.npz"),
                     gate=gate_w, down=down_w)
    except Exception as e:
        print(f"  mlprogram failed: {e}")

    # Also try NeuralNetwork path (8x8x8 to maximize ANE chance)
    print("\n=== FFN via NeuralNetwork (8x8x8) ===")
    try:
        model, gate_w, down_w = build_ffn_model(8, 8, 8)
        result = compile_and_capture(model, "ffn_silu_8x8x8_nn", TEMPLATE_DIR)
        if result:
            print(f"  Captured: {result}")
            np.savez(os.path.join(TEMPLATE_DIR, "ffn_silu_8x8x8_weights.npz"),
                     gate=gate_w, down=down_w)
    except Exception as e:
        print(f"  NeuralNetwork failed: {e}")

    # Single conv+silu for reference
    print("\n=== Single conv+SiLU (64x64) ===")
    try:
        import torch
        import torch.nn as nn

        class ConvSiLU(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Linear(64, 64, bias=False)
                self.act = nn.SiLU()
            def forward(self, x):
                return self.act(self.conv(x))

        m = ConvSiLU()
        m.eval()
        torch.manual_seed(42)
        nn.init.normal_(m.conv.weight, std=0.1)

        traced = torch.jit.trace(m, torch.randn(1, 64))
        ct_model = ct.convert(
            traced,
            inputs=[ct.TensorType(name="input", shape=(1, 64))],
            outputs=[ct.TensorType(name="output")],
            compute_units=ct.ComputeUnit.ALL,
            minimum_deployment_target=ct.target.macOS15,
        )
        compile_and_capture(ct_model, "conv_silu_64x64_ref", TEMPLATE_DIR)
    except Exception as e:
        print(f"  Single conv+SiLU failed: {e}")

    # Single conv (identity) for reference
    print("\n=== Single conv only (64x64) ===")
    try:
        class ConvOnly(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Linear(64, 64, bias=False)
            def forward(self, x):
                return self.conv(x)

        m = ConvOnly()
        m.eval()
        torch.manual_seed(42)
        nn.init.normal_(m.conv.weight, std=0.1)

        traced = torch.jit.trace(m, torch.randn(1, 64))
        ct_model = ct.convert(
            traced,
            inputs=[ct.TensorType(name="input", shape=(1, 64))],
            outputs=[ct.TensorType(name="output")],
            compute_units=ct.ComputeUnit.ALL,
            minimum_deployment_target=ct.target.macOS15,
        )
        compile_and_capture(ct_model, "conv_only_64x64_ref", TEMPLATE_DIR)
    except Exception as e:
        print(f"  Single conv failed: {e}")

    print("\n=== Done ===")
    print(f"Templates in {TEMPLATE_DIR}:")
    for f in sorted(os.listdir(TEMPLATE_DIR)):
        sz = os.path.getsize(os.path.join(TEMPLATE_DIR, f))
        print(f"  {f}: {sz:,} bytes")


if __name__ == '__main__':
    main()
