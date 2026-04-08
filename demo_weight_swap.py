#!/usr/bin/env python3
"""
Weight Swap Demo: Zero-Recompile Weight Switching on ANE

Demonstrates that weights can be changed between dispatches without
recompilation. Two models are pre-compiled with different weights,
then dispatched alternately. The "swap" is switching which model index
gets dispatched -- both are pre-compiled, switching is instant.

SIP ON:  pre-compile weight variants, dispatch-switch is instant
SIP OFF: modify weights in mmap'd .hwx, next dispatch uses new weights

Usage:
    python demo_weight_swap.py

Copyright 2026 Nick Lo. MIT License.
"""

import os
import sys
import time
import subprocess
import shutil
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from compiler import gen_conv_mlmodelc

PIPE_TOOL = os.path.join(os.path.dirname(__file__), 'tests', 'ane_standalone_pipe')
BUILD_DIR = '/tmp/weight_swap_demo'
MODEL_PATH = os.path.expanduser(
    "~/.cache/huggingface/hub/models--openai-community--gpt2/"
    "snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e/model.safetensors"
)


def load_gpt2_fc_up_weights():
    """Load GPT-2 layer 0 c_fc (fc_up) weights from safetensors."""
    from safetensors import safe_open
    with safe_open(MODEL_PATH, framework="numpy") as f:
        # c_fc_weight is [768, 3072] Conv1D format -> transpose to [3072, 768]
        W = f.get_tensor("h.0.mlp.c_fc.weight").T.copy()
        b = f.get_tensor("h.0.mlp.c_fc.bias").copy()
    return W, b


def compile_model_pair():
    """Compile two models: original weights and 2x-scaled weights."""
    os.makedirs(BUILD_DIR, exist_ok=True)

    W_orig, b_orig = load_gpt2_fc_up_weights()
    in_ch, out_ch = W_orig.shape[1], W_orig.shape[0]  # 768, 3072

    # Model A: original weights
    path_a = os.path.join(BUILD_DIR, 'model_a.mlmodelc')
    if os.path.exists(path_a):
        shutil.rmtree(path_a)
    gen_conv_mlmodelc(path_a, W_orig.astype(np.float32),
                      in_ch, out_ch, bias=b_orig.astype(np.float32),
                      name="fc_up_orig")

    # Model B: weights scaled by 2x
    W_scaled = W_orig * 2.0
    b_scaled = b_orig * 2.0
    path_b = os.path.join(BUILD_DIR, 'model_b.mlmodelc')
    if os.path.exists(path_b):
        shutil.rmtree(path_b)
    gen_conv_mlmodelc(path_b, W_scaled.astype(np.float32),
                      in_ch, out_ch, bias=b_scaled.astype(np.float32),
                      name="fc_up_2x")

    return path_a, path_b, in_ch, out_ch


def dispatch_pair(path_a, path_b, in_ch, out_ch, input_fp16, n_swaps=1000):
    """Dispatch both models via pipe tool, measure swap time.

    Loads both models into the pipe tool simultaneously, then alternates
    dispatch between them. The swap is just changing the model index --
    zero recompilation, both models are pre-loaded in ANE.
    """
    manifest_path = os.path.join(BUILD_DIR, 'manifest.txt')
    with open(manifest_path, 'w') as f:
        f.write(f"{path_a} {in_ch} {out_ch}\n")
        f.write(f"{path_b} {in_ch} {out_ch}\n")

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
            raise RuntimeError(f"Pipe tool exited during compile: {err}")

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
    out_size = out_ch * 2

    def dispatch(idx):
        cmd = f"D {idx}\n".encode()
        proc.stdin.write(cmd)
        proc.stdin.write(input_bytes)
        proc.stdin.flush()
        out_bytes = proc.stdout.read(out_size)
        if len(out_bytes) != out_size:
            raise RuntimeError(f"Short read: got {len(out_bytes)}, expected {out_size}")
        return np.frombuffer(out_bytes, dtype=np.float16).copy()

    # Warm up both models
    _ = dispatch(0)
    _ = dispatch(1)

    # Single dispatch of each for comparison
    out_a = dispatch(0)
    out_b = dispatch(1)

    # Benchmark: alternate dispatches to measure swap time
    # Each iteration dispatches A then B, so "swap" is the index change
    t0 = time.perf_counter()
    for _ in range(n_swaps):
        _ = dispatch(0)
        _ = dispatch(1)
    t_total = time.perf_counter() - t0

    # Also time single-model dispatch for baseline
    t0_single = time.perf_counter()
    for _ in range(n_swaps * 2):
        _ = dispatch(0)
    t_single = time.perf_counter() - t0_single

    # Clean up
    try:
        proc.stdin.write(b"Q\n")
        proc.stdin.flush()
        proc.wait(timeout=5)
    except Exception:
        proc.kill()
        try:
            proc.wait(timeout=2)
        except Exception:
            pass

    swap_overhead_ms = (t_total - t_single) / n_swaps * 1000
    per_dispatch_ms = t_single / (n_swaps * 2) * 1000
    alternating_ms = t_total / (n_swaps * 2) * 1000

    return out_a, out_b, swap_overhead_ms, per_dispatch_ms, alternating_ms


def print_sip_off_instructions():
    """Print the LLDB mmap swap path for SIP-OFF."""
    print()
    print("  " + "=" * 58)
    print("  SIP OFF: In-Flight Weight Swap via LLDB")
    print("  " + "=" * 58)
    print()
    print("  Weights are plaintext FP16 in the __KERN_0 segment of .hwx")
    print("  Layout: 16 tiles, 32-channel sub-blocks, column-major")
    print("  (fully decoded by ane-compiler emitter)")
    print()
    print("  Swap procedure:")
    print("    1. aned compiles .mlmodelc -> .hwx (mmap'd in aned address space)")
    print("    2. LLDB attach to aned, find mmap'd .hwx buffer")
    print("    3. Overwrite __KERN_0 weight data in-place (memcpy)")
    print("    4. Next dispatch uses new weights immediately")
    print("    5. No recompilation, no reload, no model re-creation")
    print()
    print("  Use case: LoRA hot-swap")
    print("    - Base model compiled once (expensive: 768x3072 = 2.36M params)")
    print("    - LoRA delta applied as weight patch to mmap'd buffer")
    print("    - Swap latency: ~memcpy time (~50us for 4.7MB fc_up weights)")
    print("    - vs recompilation: ~200ms per op through aned")
    print()
    print("  Weight layout in __KERN_0 (conv/inner_product):")
    print("    Offset 0x0000: PWL table (128 bytes, for fused activation)")
    print("    Offset 0x0080: Weight data starts")
    print("    Per tile: 32-channel sub-blocks, column-major within block")
    print("    16 tiles total (hardware-enforced, not configurable)")
    print("    FP16 values, directly patchable")
    print()


def main():
    print()
    print("=" * 62)
    print("  Weight Swap Demo: Zero-Recompile Weight Switching")
    print("=" * 62)
    print()

    # Check pipe tool exists
    if not os.path.exists(PIPE_TOOL):
        print(f"  ERROR: pipe tool not found at {PIPE_TOOL}")
        print(f"  Build with: cd tests && make ane_standalone_pipe")
        sys.exit(1)

    # Check model exists
    if not os.path.exists(MODEL_PATH):
        print(f"  ERROR: GPT-2 safetensors not found at {MODEL_PATH}")
        sys.exit(1)

    # Step 1: Compile two models with different weights
    print("  Compiling two weight variants...")
    t0 = time.time()
    path_a, path_b, in_ch, out_ch = compile_model_pair()
    t_compile = time.time() - t0
    print(f"  Compiled 2 models in {t_compile:.2f}s")
    print()

    # Create deterministic input (GPT-2 layer 0 would see embedding output)
    np.random.seed(42)
    input_fp16 = np.random.randn(in_ch).astype(np.float16)

    # Step 2-5: Dispatch both, compare outputs
    print("  Loading both models on ANE...")
    n_swaps = 500
    out_a, out_b, swap_overhead_ms, per_dispatch_ms, alternating_ms = \
        dispatch_pair(path_a, path_b, in_ch, out_ch, input_fp16, n_swaps=n_swaps)

    # Results
    print()
    print("  " + "-" * 58)
    print(f"  Model A: GPT-2 L0 fc_up (original weights)")
    print(f"  Model B: GPT-2 L0 fc_up (weights * 2.0)")
    print()
    print(f"  Dispatch A: output[0:4] = {[round(float(v), 4) for v in out_a[:4]]}")
    print(f"  Dispatch B: output[0:4] = {[round(float(v), 4) for v in out_b[:4]]}")
    print()

    # Verify outputs are different
    max_diff = float(np.max(np.abs(out_a.astype(np.float32) - out_b.astype(np.float32))))
    ratio = np.mean(np.abs(out_b.astype(np.float32))) / max(np.mean(np.abs(out_a.astype(np.float32))), 1e-10)
    outputs_different = max_diff > 0.01

    print(f"  Outputs different: {'YES' if outputs_different else 'NO'} "
          f"(max diff = {max_diff:.4f}, mean |B|/|A| = {ratio:.2f}x)")
    print()

    # Verify 2x scaling relationship
    # With bias*2 and weight*2, output should be ~2x for linear: W*x + b
    ratio_check = out_b.astype(np.float32) / np.where(np.abs(out_a.astype(np.float32)) > 0.01,
                                                       out_a.astype(np.float32), np.nan)
    valid_ratios = ratio_check[np.isfinite(ratio_check)]
    if len(valid_ratios) > 0:
        mean_ratio = float(np.nanmean(valid_ratios))
        print(f"  Scaling check: B/A ratio = {mean_ratio:.3f}x (expected 2.0x)")
    print()

    # Timing
    print(f"  Timing ({n_swaps} alternating dispatches):")
    print(f"    Single-model dispatch: {per_dispatch_ms:.3f}ms/dispatch")
    print(f"    Alternating dispatch:  {alternating_ms:.3f}ms/dispatch")
    print(f"    Swap overhead:         {swap_overhead_ms:.4f}ms "
          f"(pre-compiled, no recompilation)")
    print()

    print(f"  SIP ON:  pre-compile weight variants -> instant dispatch switch")
    print(f"  SIP OFF: modify weights in mmap'd .hwx -> instant hot-swap")
    print("  " + "-" * 58)

    # SIP OFF instructions
    print_sip_off_instructions()


if __name__ == '__main__':
    main()
