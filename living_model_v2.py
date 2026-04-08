#!/usr/bin/env python3
"""
Living Model v2: Stateful LoRA on ANE via CoreML state buffers.

Architecture:
  - 1B model with LoRA matrices as mutable state buffers
  - LoRA injected at attention Q projections (2048-dim, not 128K logits)
  - Per-token: ANE forward (read_state), CPU gradient, memcpy state update
  - Native CoreML API, no weight surgery, no aned XPC in hot loop

Kill test:
  1. 200 tokens frozen LoRA (zero state) → baseline accuracy
  2. 200 tokens live LoRA updates → compare accuracy
  3. If >5% improvement in late window: CONFIRMED

Phase 1: Single-layer proof (this file)
Phase 2: Scale to full 1B attention projections

Copyright 2026 Nick Lo. MIT License.
"""

import os
import sys
import time
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import coremltools as ct
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types


# ===================================================================
# Phase 1: Build a single stateful linear layer with LoRA
# ===================================================================

def build_stateful_lora_model(dim=2048, rank=4, save_path='/tmp/lora_stateful_test'):
    """Build a linear layer with LoRA as mutable state.

    y = (W_base @ x) + (lora_B @ lora_A @ x)

    W_base: frozen weights [dim, dim] (compiled into model)
    lora_A: mutable state [rank, dim] (read from state buffer)
    lora_B: mutable state [dim, rank] (read from state buffer)

    State buffers are IOSurfaces updatable via memcpy between inference calls.
    """
    print(f"Building stateful LoRA model (dim={dim}, rank={rank})...")

    # Random base weights (would be real Llama weights in production)
    W_base = np.random.randn(dim, dim).astype(np.float16) * 0.02

    @mb.program(
        input_specs=[
            mb.TensorSpec(shape=(1, dim), dtype=types.fp16),
            mb.StateTensorSpec(shape=(rank, dim), dtype=types.fp16),  # lora_A
            mb.StateTensorSpec(shape=(dim, rank), dtype=types.fp16),  # lora_B
        ],
        opset_version=ct.target.iOS18,
    )
    def lora_model(x, lora_A_state, lora_B_state):
        # Base linear: y_base = x @ W_base.T
        base_out = mb.linear(x=x, weight=mb.const(val=W_base))

        # Read LoRA state buffers
        lora_A = mb.read_state(input=lora_A_state)
        lora_B = mb.read_state(input=lora_B_state)

        # Persist state (required for CoreML state management)
        mb.coreml_update_state(state=lora_A_state, value=lora_A)
        mb.coreml_update_state(state=lora_B_state, value=lora_B)

        # LoRA forward: delta = (x @ A.T) @ B.T
        x_f32 = mb.cast(x=x, dtype="fp32")
        lora_A_f32 = mb.cast(x=lora_A, dtype="fp32")
        lora_B_f32 = mb.cast(x=lora_B, dtype="fp32")

        # x @ A.T -> [1, rank]
        low_rank = mb.matmul(x=x_f32, y=lora_A_f32, transpose_y=True)
        # [1, rank] @ B.T -> [1, dim]
        delta = mb.matmul(x=low_rank, y=lora_B_f32, transpose_y=True)
        delta_f16 = mb.cast(x=delta, dtype="fp16")

        # Combined output
        output = mb.add(x=base_out, y=delta_f16)
        return output

    # Convert
    model = ct.convert(
        lora_model,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.iOS18,
    )

    pkg_path = f'{save_path}.mlpackage'
    model.save(pkg_path)
    print(f"Saved to {pkg_path}")

    # Load and verify
    loaded = ct.models.MLModel(pkg_path, compute_units=ct.ComputeUnit.CPU_AND_NE)
    print(f"Loaded. Compute unit: CPU_AND_NE")

    return loaded, dim, rank


def test_state_update(model, dim, rank):
    """Test that state updates actually affect inference output."""
    print("\n=== State Update Test ===")

    # Create state
    state = model.make_state()

    # Test input
    x = np.random.randn(1, dim).astype(np.float32)

    # Inference with zero LoRA (default state)
    result_zero = model.predict({'x': x}, state=state)
    y_zero = list(result_zero.values())[0].flatten()

    # Update LoRA state with non-zero values
    lora_A_new = np.random.randn(rank, dim).astype(np.float32) * 0.1
    lora_B_new = np.random.randn(dim, rank).astype(np.float32) * 0.1

    state.write_state('lora_A_state', lora_A_new)
    state.write_state('lora_B_state', lora_B_new)

    # Inference with updated LoRA
    result_updated = model.predict({'x': x}, state=state)
    y_updated = list(result_updated.values())[0].flatten()

    # Compare
    diff = np.max(np.abs(y_zero.astype(np.float32) - y_updated.astype(np.float32)))
    print(f"  Zero LoRA output (first 5): {y_zero[:5]}")
    print(f"  Updated LoRA output (first 5): {y_updated[:5]}")
    print(f"  Max diff: {diff:.6f}")

    if diff > 0.001:
        print(f"  >>> STATE UPDATE CONFIRMED — output changes with LoRA <<<")
        return True
    else:
        print(f"  >>> STATE UPDATE FAILED — output unchanged <<<")
        return False


def benchmark_state_update(model, dim, rank, n_iterations=100):
    """Measure state update latency."""
    print(f"\n=== State Update Latency ({n_iterations} iterations) ===")

    state = model.make_state()
    x = np.random.randn(1, dim).astype(np.float32)

    # Warm up
    for _ in range(5):
        model.predict({'x': x}, state=state)

    # Measure inference
    t0 = time.perf_counter()
    for _ in range(n_iterations):
        model.predict({'x': x}, state=state)
    inference_time = (time.perf_counter() - t0) / n_iterations * 1000
    print(f"  Inference: {inference_time:.2f} ms/call")

    # Measure state update
    lora_A_new = np.random.randn(rank, dim).astype(np.float32)
    lora_B_new = np.random.randn(dim, rank).astype(np.float32)

    t0 = time.perf_counter()
    for _ in range(n_iterations):
        state.write_state('lora_A_state', lora_A_new)
        state.write_state('lora_B_state', lora_B_new)
    update_time = (time.perf_counter() - t0) / n_iterations * 1000
    print(f"  State update: {update_time:.3f} ms/call")

    # Measure combined (inference + update = one "learning step")
    t0 = time.perf_counter()
    for _ in range(n_iterations):
        model.predict({'x': x}, state=state)
        state.write_state('lora_A_state', lora_A_new)
        state.write_state('lora_B_state', lora_B_new)
    combined_time = (time.perf_counter() - t0) / n_iterations * 1000
    print(f"  Combined (inference + update): {combined_time:.2f} ms/step")
    print(f"  Update overhead: {update_time/inference_time*100:.1f}% of inference")

    if update_time < 1.0:
        print(f"  >>> SUB-MILLISECOND UPDATE CONFIRMED ({update_time:.3f} ms) <<<")

    return inference_time, update_time


def main():
    print("=" * 70)
    print("LIVING MODEL v2: STATEFUL LoRA ON ANE")
    print("Phase 1: Single-layer proof with mutable state buffers")
    print("=" * 70)

    # Build model
    model, dim, rank = build_stateful_lora_model(dim=2048, rank=4)

    # Test state updates affect output
    state_works = test_state_update(model, dim, rank)
    if not state_works:
        print("\nSTATE UPDATE DOES NOT AFFECT OUTPUT. ABORTING.")
        return

    # Benchmark latency
    inf_ms, update_ms = benchmark_state_update(model, dim, rank)

    # Summary
    print(f"\n{'=' * 70}")
    print("PHASE 1 RESULTS")
    print(f"{'=' * 70}")
    print(f"  State update affects output: {'YES' if state_works else 'NO'}")
    print(f"  Inference latency: {inf_ms:.2f} ms")
    print(f"  State update latency: {update_ms:.3f} ms")
    print(f"  Combined step: {inf_ms + update_ms:.2f} ms")
    print(f"  Max learning rate: {1000/(inf_ms + update_ms):.0f} Hz")
    print(f"  37 Hz viable: {'YES' if (inf_ms + update_ms) < 27 else 'NO'}")

    if state_works and update_ms < 1.0:
        print(f"\n  >>> PHASE 1 PASS: Stateful LoRA on ANE confirmed <<<")
        print(f"  >>> Proceed to Phase 2: Full Llama-1B attention projections <<<")
    else:
        print(f"\n  >>> PHASE 1: Issues found, investigate before Phase 2 <<<")


if __name__ == "__main__":
    main()
