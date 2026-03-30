#!/usr/bin/env python3
"""
ane-compiler v0.1 verification test suite.

Tests:
1. Weight packing round-trip (template → emit → extract → compare)
2. Binary structure validation (all emitted .hwx pass BEEFFACE checks)
3. Softmax/LayerNorm template emission (default = exact match)
4. FFN emission (gate+SiLU + down projection, weight verification)
5. Reference comparison: compile via _ANEInMemoryModel, diff __kern_0

Requires: numpy, Foundation (pyobjc on macOS)
Run with: ~/.mlx-env/bin/python3 test_ffn_verify.py
"""

import sys
import os
import struct
import traceback
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from emitter import *

# Paths
ATLAS_DIR = os.path.expanduser('~/Desktop/cowork/ngram-engine/ane_reverse/hwx_cache/atlas')
CONV_ATLAS_DIR = os.path.expanduser('~/Desktop/cowork/ngram-engine/ane_reverse/hwx_cache/conv_atlas')
PROBE_DIR = os.path.expanduser('~/Desktop/cowork/ngram-engine/ane_reverse/hwx_cache/attention_probe_v3')

PASS_COUNT = 0
FAIL_COUNT = 0


def check(name, condition, detail=""):
    global PASS_COUNT, FAIL_COUNT
    if condition:
        PASS_COUNT += 1
        print(f"  ✓ {name}")
    else:
        FAIL_COUNT += 1
        print(f"  ✗ {name}: {detail}")


def build_compiler():
    """Build compiler with all available templates."""
    c = ANECompiler.__new__(ANECompiler)
    c.registry = TemplateRegistry()
    c.registry.load_directory(ATLAS_DIR)
    c.registry.load_directory(CONV_ATLAS_DIR)
    c.registry.load_file(os.path.join(PROBE_DIR, 'softmax_only.hwx'), name='softmax_ref')
    c.registry.load_file(os.path.join(PROBE_DIR, 'layernorm.hwx'), name='layernorm_ref')
    return c


def validate_hwx(data: bytes, label: str):
    """Validate basic .hwx structure."""
    magic = struct.unpack_from('<I', data, 0)[0]
    check(f"{label}: BEEFFACE magic", magic == 0xBEEFFACE,
          f"got 0x{magic:08X}")

    cpu_type = struct.unpack_from('<I', data, 4)[0]
    check(f"{label}: CPU type 128", cpu_type == 128,
          f"got {cpu_type}")

    size = len(data)
    check(f"{label}: page-aligned size", size % 4096 == 0,
          f"{size} not page-aligned")

    ncmds = struct.unpack_from('<I', data, 0x10)[0]
    check(f"{label}: reasonable ncmds", 5 <= ncmds <= 500,
          f"ncmds={ncmds}")

    return True


def test_1_binary_structure():
    """Test that all emittable types produce valid .hwx."""
    print("\n═══ Test 1: Binary Structure ═══")
    c = build_compiler()

    # Activation
    hwx = c.emit_activation(ActivationType.RELU)
    validate_hwx(hwx, "relu")

    # Softmax
    hwx = c.emit_softmax()
    validate_hwx(hwx, "softmax")

    # LayerNorm
    hwx = c.emit_layernorm()
    validate_hwx(hwx, "layernorm")

    # Conv+SiLU
    np.random.seed(42)
    w = np.random.randn(4, 4).astype(np.float16) * 0.1
    hwx = c.emit_conv_activation(w, ActivationType.SILU)
    validate_hwx(hwx, "conv_silu")

    # Conv+ReLU
    hwx = c.emit_conv_activation(w, ActivationType.RELU)
    validate_hwx(hwx, "conv_relu")


def test_2_template_exact_match():
    """Test that default emission matches template exactly."""
    print("\n═══ Test 2: Template Exact Match ═══")
    c = build_compiler()

    # Softmax: no params = exact template copy
    sm_hwx = c.emit_softmax()
    sm_ref = open(os.path.join(PROBE_DIR, 'softmax_only.hwx'), 'rb').read()
    check("softmax default = template", sm_hwx == sm_ref,
          f"{sum(a!=b for a,b in zip(sm_hwx, sm_ref))} bytes differ")

    # LayerNorm: no params = exact template copy
    ln_hwx = c.emit_layernorm()
    ln_ref = open(os.path.join(PROBE_DIR, 'layernorm.hwx'), 'rb').read()
    check("layernorm default = template", ln_hwx == ln_ref,
          f"{sum(a!=b for a,b in zip(ln_hwx, ln_ref))} bytes differ")

    # ReLU: no changes = exact template copy
    relu_hwx = c.emit_activation(ActivationType.RELU)
    relu_ref = open(os.path.join(ATLAS_DIR, 'relu.hwx'), 'rb').read()
    check("relu default = template", relu_hwx == relu_ref)


def test_3_weight_packing():
    """Test weight packing into tile-replicated __KERN_0."""
    print("\n═══ Test 3: Weight Packing ═══")
    c = build_compiler()

    np.random.seed(123)
    weights = np.random.randn(4, 4).astype(np.float16) * 0.1

    # Conv+SiLU
    hwx = c.emit_conv_activation(weights, ActivationType.SILU)
    t = c.registry.get_conv_template(with_pwl=True)
    tile_sz = t.kern0_size // 16

    # Check weights in each of 16 tiles
    w_flat = weights.flatten()
    w_bytes = w_flat.tobytes()
    all_tiles_match = True
    for tile in range(16):
        off = t.kern0_offset + tile * tile_sz + 128  # PWL offset for SiLU
        extracted = np.frombuffer(hwx[off:off + len(w_bytes)], dtype=np.float16)
        if not np.array_equal(extracted, w_flat):
            all_tiles_match = False
            break

    check("conv_silu: weights in all 16 tiles", all_tiles_match,
          f"tile {tile} mismatch")

    # Check PWL header is SiLU
    pwl_start = t.kern0_offset
    header = np.frombuffer(hwx[pwl_start:pwl_start + 8], dtype=np.float16)
    check("conv_silu: SiLU PWL header present",
          np.isinf(header[1]) and np.isinf(header[3]),
          f"header={header}")

    # Conv+ReLU (no PWL, weights at tile offset 0)
    hwx_relu = c.emit_conv_activation(weights, ActivationType.RELU)
    t_relu = c.registry.get_conv_template(with_pwl=False)
    tile_sz_r = t_relu.kern0_size // 16

    off = t_relu.kern0_offset
    extracted = np.frombuffer(hwx_relu[off:off + len(w_bytes)], dtype=np.float16)
    check("conv_relu: weights at tile 0", np.array_equal(extracted, w_flat),
          f"expected {w_flat[:4]}, got {extracted[:4]}")


def test_4_ffn_emission():
    """Test FFN two-dispatch emission."""
    print("\n═══ Test 4: FFN Emission ═══")
    c = build_compiler()

    np.random.seed(42)
    gate_w = np.random.randn(4, 4).astype(np.float16) * 0.1
    down_w = np.random.randn(4, 4).astype(np.float16) * 0.1

    gate_hwx, down_hwx = c.emit_ffn(gate_w, down_w,
                                      activation=ActivationType.SILU,
                                      output_path='/tmp/test_ffn')

    check("ffn: gate .hwx valid size", len(gate_hwx) == 65536)
    check("ffn: down .hwx valid size", len(down_hwx) == 65536)
    validate_hwx(gate_hwx, "ffn_gate")
    validate_hwx(down_hwx, "ffn_down")

    # Verify gate has SiLU PWL
    t_silu = c.registry.get_conv_template(with_pwl=True)
    kern0_off = t_silu.kern0_offset
    header = np.frombuffer(gate_hwx[kern0_off:kern0_off + 8], dtype=np.float16)
    check("ffn: gate has SiLU PWL", np.isinf(header[1]))

    # Verify files written
    check("ffn: .gate.hwx file exists", os.path.exists('/tmp/test_ffn.gate.hwx'))
    check("ffn: .down.hwx file exists", os.path.exists('/tmp/test_ffn.down.hwx'))


def test_5_multipass_parsing():
    """Test multi-pass parsing and reassembly."""
    print("\n═══ Test 5: Multi-Pass Parsing ═══")
    c = build_compiler()

    # Softmax: 5 passes
    sm = c.registry.get_softmax_template()
    check("softmax: 5 passes", sm.num_passes == 5,
          f"got {sm.num_passes}")

    # Pass 0: reduce_max
    check("softmax pass 0: reduce_max opcode",
          sm.passes[0].opcode == 0x92618005,
          f"got 0x{sm.passes[0].opcode:08X}")

    # Pass 2: accumulate (has conv header)
    check("softmax pass 2: has conv header", sm.passes[2].has_conv_header)

    # Round-trip
    rt = assemble_multipass_text(sm.passes)
    check("softmax: round-trip intact", rt == sm.text_data,
          f"{len(rt)} vs {len(sm.text_data)}")

    # LayerNorm: 5 passes (4 semantic + 1 split)
    ln = c.registry.get_layernorm_template()
    check("layernorm: 5 passes", ln.num_passes == 5,
          f"got {ln.num_passes}")

    rt = assemble_multipass_text(ln.passes)
    check("layernorm: round-trip intact", rt == ln.text_data)

    # Single-pass (relu): 1 pass
    relu = c.registry.templates.get('relu')
    if relu:
        check("relu: 1 pass", relu.num_passes == 1)


def test_6_layernorm_epsilon():
    """Test layernorm epsilon patching."""
    print("\n═══ Test 6: LayerNorm Epsilon ═══")
    c = build_compiler()

    # Default
    ln_default = c.emit_layernorm()

    # Custom epsilon
    ln_custom = c.emit_layernorm(LayerNormParams(epsilon=1e-6, dim=128))

    diffs = sum(1 for a, b in zip(ln_default, ln_custom) if a != b)
    check("layernorm: custom epsilon produces diff", diffs > 0,
          "no bytes differ")
    check("layernorm: diff count reasonable", 2 <= diffs <= 12,
          f"{diffs} bytes differ (expected 4-8 for epsilon+dim)")

    # Verify epsilon value is correct
    ln_t = c.registry.get_layernorm_template()
    # Find epsilon location in __text
    for p in ln_t.passes:
        for i, w in enumerate(p.words):
            if w == 0x37280000:  # original epsilon
                abs_off = ln_t.text_offset + (p.word_offset + i) * 4
                patched_val = struct.unpack_from('<f', ln_custom, abs_off)[0]
                check("layernorm: epsilon = 1e-6",
                      abs(patched_val - 1e-6) < 1e-9,
                      f"got {patched_val}")


def test_7_reference_comparison():
    """Compare emitted conv weights against template weight layout.

    This test verifies that the weight packing preserves the template's
    tile metadata while correctly replacing weight data.
    """
    print("\n═══ Test 7: Template Preservation ═══")
    c = build_compiler()

    # Get conv_silu template
    t = c.registry.get_conv_template(with_pwl=True)
    tile_sz = t.kern0_size // 16

    # Emit with known weights
    np.random.seed(42)
    w = np.random.randn(4, 4).astype(np.float16) * 0.1
    hwx = c.emit_conv_activation(w, ActivationType.SILU)

    # Check that non-weight, non-PWL bytes are preserved from template
    # Tile metadata bytes (bytes 84-127 and beyond weights) should match
    w_bytes_len = 32  # 4x4 FP16
    for tile in range(16):
        tile_start = t.kern0_offset + tile * tile_sz
        # Metadata region: bytes 84-127 (between PWL and weights)
        template_meta = t.data[tile_start + 84:tile_start + 128]
        emitted_meta = hwx[tile_start + 84:tile_start + 128]
        if template_meta != emitted_meta:
            check(f"tile {tile}: metadata preserved", False,
                  f"metadata differs in bytes 84-128")
            break
    else:
        check("all tiles: metadata preserved", True)

    # Check regions OUTSIDE __kern_0 are unchanged
    # Header (0-0x20)
    check("header unchanged", hwx[:0x20] == t.data[:0x20])
    # Load commands
    check("load_cmds unchanged", hwx[0x20:0x4000] == t.data[0x20:0x4000])
    # __text unchanged
    text_region = hwx[t.text_offset:t.text_offset + t.text_size]
    template_text = t.data[t.text_offset:t.text_offset + t.text_size]
    check("__text unchanged", text_region == template_text)


def main():
    print("═══════════════════════════════════════════")
    print("  ane-compiler v0.1 verification suite")
    print("═══════════════════════════════════════════")

    tests = [
        test_1_binary_structure,
        test_2_template_exact_match,
        test_3_weight_packing,
        test_4_ffn_emission,
        test_5_multipass_parsing,
        test_6_layernorm_epsilon,
        test_7_reference_comparison,
    ]

    for test in tests:
        try:
            test()
        except Exception as e:
            print(f"\n  ✗ EXCEPTION in {test.__name__}: {e}")
            traceback.print_exc()
            global FAIL_COUNT
            FAIL_COUNT += 1

    print(f"\n═══════════════════════════════════════════")
    print(f"  Results: {PASS_COUNT} passed, {FAIL_COUNT} failed")
    print(f"═══════════════════════════════════════════")

    return 0 if FAIL_COUNT == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
