#!/usr/bin/env python3
"""
Living Model v2 Kill Test: Stateful LoRA at attention Q projection.

Phase 1 proved: state updates work on ANE at 6μs.
This test: full 200-token prediction accuracy comparison with diagnostics.

Diagnostics:
  1. LoRA weight magnitude every 10 tokens (growth/oscillation/plateau)
  2. Per-window accuracy (4x50-token windows)
  3. Top-5 predictions per position (rank tracking)
  4. Two learning rates (1e-4 and 1e-3)

Architecture:
  - Single linear layer (simulating Q projection): dim=2048
  - LoRA rank 4 as mutable state buffers
  - 70B generates 200 tokens of ground truth
  - Per-token: forward → compare to 70B → gradient → state update

Copyright 2026 Nick Lo. MIT License.
"""

import json
import os
import sys
import time
import urllib.request
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import coremltools as ct
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types

LLM_URL = "http://127.0.0.1:8899/v1/chat/completions"


# ===================================================================
# Build stateful model
# ===================================================================

def build_lora_model(dim=2048, rank=4):
    """Build a linear layer with LoRA as stateful buffers.

    Simulates the Q projection of attention layer 0.
    y = W_base @ x + (lora_B @ lora_A @ x)
    """
    W_base = np.random.randn(dim, dim).astype(np.float16) * 0.02

    @mb.program(
        input_specs=[
            mb.TensorSpec(shape=(1, dim), dtype=types.fp16),
            mb.StateTensorSpec(shape=(rank, dim), dtype=types.fp16),   # lora_A
            mb.StateTensorSpec(shape=(dim, rank), dtype=types.fp16),   # lora_B
        ],
        opset_version=ct.target.iOS18,
    )
    def model(x, lora_A_state, lora_B_state):
        base_out = mb.linear(x=x, weight=mb.const(val=W_base))

        lora_A = mb.read_state(input=lora_A_state)
        lora_B = mb.read_state(input=lora_B_state)
        mb.coreml_update_state(state=lora_A_state, value=lora_A)
        mb.coreml_update_state(state=lora_B_state, value=lora_B)

        x32 = mb.cast(x=x, dtype="fp32")
        A32 = mb.cast(x=lora_A, dtype="fp32")
        B32 = mb.cast(x=lora_B, dtype="fp32")

        low = mb.matmul(x=x32, y=A32, transpose_y=True)   # [1, rank]
        delta = mb.matmul(x=low, y=B32, transpose_y=True)  # [1, dim]
        delta16 = mb.cast(x=delta, dtype="fp16")

        return mb.add(x=base_out, y=delta16)

    ct_model = ct.convert(model, compute_units=ct.ComputeUnit.CPU_AND_NE,
                           minimum_deployment_target=ct.target.iOS18)
    path = '/tmp/living_model_v2_killtest.mlpackage'
    ct_model.save(path)
    loaded = ct.models.MLModel(path, compute_units=ct.ComputeUnit.CPU_AND_NE)
    return loaded, W_base, dim, rank


# ===================================================================
# Generate ground truth
# ===================================================================

def generate_ground_truth(n_tokens=200):
    payload = json.dumps({
        "model": "mlx-community/Llama-3.3-70B-Instruct-4bit",
        "messages": [{"role": "user", "content":
            "Explain the Apple Neural Engine hardware architecture in detail. "
            "Cover register maps, dispatch mechanisms, SRAM organization, "
            "and kext interfaces."}],
        "max_tokens": n_tokens, "temperature": 0.3,
        "repetition_penalty": 1.1,
    }).encode()
    req = urllib.request.Request(LLM_URL, data=payload,
        headers={"Content-Type": "application/json"})
    resp = urllib.request.urlopen(req, timeout=300)
    data = json.loads(resp.read())
    return data["choices"][0]["message"]["content"]


# ===================================================================
# LoRA gradient computation (CPU)
# ===================================================================

class LoRAGradient:
    """Compute gradients for LoRA A and B matrices."""

    def __init__(self, dim, rank):
        self.dim = dim
        self.rank = rank

    def compute(self, x, target_idx, logits, lora_A, lora_B, W_base, lr):
        """Compute gradient and return updated A, B.

        x: [dim] input hidden state
        target_idx: scalar, correct token index (simulated)
        logits: [dim] output logits
        lora_A: [rank, dim]
        lora_B: [dim, rank]
        W_base: [dim, dim]
        lr: learning rate

        For this test, "target" is the ground truth next embedding vector.
        We minimize MSE between model output and target.
        """
        x = x.astype(np.float32).reshape(1, -1)  # [1, dim]

        # Forward: y = W_base @ x.T + B @ A @ x.T
        base = x @ W_base.T  # [1, dim]
        low = x @ lora_A.T   # [1, rank]
        delta = low @ lora_B.T  # [1, dim]
        y = base + delta  # [1, dim]

        # Target: we want y to match the ground truth embedding
        target = np.zeros_like(y)
        target[0, target_idx % self.dim] = 1.0  # one-hot at target position

        # MSE gradient: dL/dy = 2*(y - target) / dim
        grad_y = 2.0 * (y - target) / self.dim  # [1, dim]

        # Backprop through B: dL/dB = grad_y.T @ low
        grad_B = grad_y.T @ low  # [dim, rank]

        # Backprop through A: dL/dA = (grad_y @ B).T @ x = low_grad.T @ x
        grad_low = grad_y @ lora_B  # [1, rank]
        grad_A = grad_low.T @ x  # [rank, dim]

        # SGD update
        new_A = lora_A - lr * grad_A
        new_B = lora_B - lr * grad_B

        return new_A.astype(np.float32), new_B.astype(np.float32)


# ===================================================================
# Run one variant
# ===================================================================

def run_variant(model, W_base, dim, rank, gt_tokens, tokenizer,
                lr, variant_name):
    """Run 200 tokens, return full diagnostics."""
    state = model.make_state()
    grad = LoRAGradient(dim, rank)
    n = min(len(gt_tokens), 200)

    # Initialize LoRA to zero
    lora_A = np.zeros((rank, dim), dtype=np.float32)
    lora_B = np.zeros((dim, rank), dtype=np.float32)

    if variant_name != "frozen":
        state.write_state('lora_A_state', lora_A)
        state.write_state('lora_B_state', lora_B)

    # Embedding table (random, consistent across variants)
    np.random.seed(42)
    embed = np.random.randn(max(gt_tokens) + 1, dim).astype(np.float32) * 0.02

    # Diagnostics
    top1_correct = 0
    top5_correct = 0
    total = 0
    window_correct = [0, 0, 0, 0]  # 4 windows of 50
    weight_magnitudes = []
    rank_history = []  # where does the correct answer rank?

    t0 = time.perf_counter()

    for pos in range(n - 1):
        token = gt_tokens[pos]
        target = gt_tokens[pos + 1]

        # Get input embedding
        x = embed[token].reshape(1, dim).astype(np.float32)

        # Forward through stateful model
        result = model.predict({'x': x}, state=state)
        logits = list(result.values())[0].flatten().astype(np.float32)

        # Top-5 predictions
        top5_idx = np.argsort(logits)[-5:][::-1]
        predicted = top5_idx[0]

        # Track accuracy
        if predicted == target % dim:
            top1_correct += 1
        if (target % dim) in top5_idx:
            top5_correct += 1

        # Track rank of correct answer
        sorted_idx = np.argsort(logits)[::-1]
        correct_rank = np.where(sorted_idx == (target % dim))[0]
        rank_val = correct_rank[0] if len(correct_rank) > 0 else dim
        rank_history.append(int(rank_val))

        # Window tracking
        window_idx = min(pos // 50, 3)
        if predicted == target % dim:
            window_correct[window_idx] += 1

        total += 1

        # LoRA update (skip for frozen)
        if variant_name != "frozen" and lr > 0:
            new_A, new_B = grad.compute(
                x.flatten(), target, logits, lora_A, lora_B,
                W_base.astype(np.float32), lr)
            lora_A = new_A
            lora_B = new_B
            state.write_state('lora_A_state', lora_A)
            state.write_state('lora_B_state', lora_B)

        # Log weight magnitude every 10 tokens
        if (pos + 1) % 10 == 0:
            mag_A = np.linalg.norm(lora_A)
            mag_B = np.linalg.norm(lora_B)
            weight_magnitudes.append({
                'pos': pos + 1,
                'mag_A': float(mag_A),
                'mag_B': float(mag_B),
                'mag_total': float(mag_A + mag_B),
            })

    elapsed = time.perf_counter() - t0

    # Window accuracy (each 50 tokens)
    window_sizes = [min(50, n - 1 - i * 50) for i in range(4)]
    window_acc = [window_correct[i] / max(1, window_sizes[i]) * 100
                  for i in range(4)]

    # Median rank in each window
    window_ranks = []
    for w in range(4):
        start = w * 50
        end = min(start + 50, len(rank_history))
        if start < end:
            window_ranks.append(float(np.median(rank_history[start:end])))
        else:
            window_ranks.append(0)

    return {
        'variant': variant_name,
        'lr': lr,
        'top1_acc': top1_correct / total * 100,
        'top5_acc': top5_correct / total * 100,
        'total': total,
        'elapsed': elapsed,
        'window_acc': window_acc,
        'window_ranks': window_ranks,
        'weight_magnitudes': weight_magnitudes,
        'final_rank_median': float(np.median(rank_history)),
    }


# ===================================================================
# Main
# ===================================================================

def main():
    print("=" * 74)
    print("LIVING MODEL v2 KILL TEST")
    print("Stateful LoRA at attention Q projection, 200 tokens, full diagnostics")
    print("=" * 74)

    # Generate ground truth
    print("\n[1/4] Generating 200 tokens via 70B Q4...")
    gt_text = generate_ground_truth(200)
    print(f"  Generated: {gt_text[:80]}...")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('unsloth/Llama-3.2-1B-Instruct')
    gt_tokens = tokenizer.encode(gt_text, add_special_tokens=False)[:200]
    print(f"  Tokens: {len(gt_tokens)}")

    # Build model
    print("\n[2/4] Building stateful LoRA model...")
    model, W_base, dim, rank = build_lora_model(dim=2048, rank=4)
    print(f"  dim={dim}, rank={rank}")

    # Run variants
    print("\n[3/4] Running kill test variants...")

    variants = [
        ("frozen", 0),
        ("lr=1e-4", 1e-4),
        ("lr=1e-3", 1e-3),
        ("lr=1e-2", 1e-2),
    ]

    results = []
    for name, lr in variants:
        print(f"\n  --- {name} ---")
        r = run_variant(model, W_base, dim, rank, gt_tokens, tokenizer,
                        lr, name)
        results.append(r)
        print(f"  Top-1: {r['top1_acc']:.1f}% | Top-5: {r['top5_acc']:.1f}% | "
              f"Median rank: {r['final_rank_median']:.0f} | {r['elapsed']:.1f}s")
        print(f"  Windows: {['%.1f%%' % w for w in r['window_acc']]}")

    # Full report
    print(f"\n\n{'=' * 74}")
    print("[4/4] RESULTS")
    print(f"{'=' * 74}")

    # Table 1: Overall accuracy
    print(f"\n{'Variant':<12} | {'Top-1':>6} | {'Top-5':>6} | {'Med Rank':>8} | {'Time':>5}")
    print("-" * 50)
    for r in results:
        print(f"{r['variant']:<12} | {r['top1_acc']:>5.1f}% | {r['top5_acc']:>5.1f}% | "
              f"{r['final_rank_median']:>8.0f} | {r['elapsed']:>4.1f}s")

    # Table 2: Per-window accuracy
    print(f"\n{'Variant':<12} | {'1-50':>7} | {'51-100':>7} | {'101-150':>7} | {'151-200':>7}")
    print("-" * 50)
    for r in results:
        wa = r['window_acc']
        print(f"{r['variant']:<12} | {wa[0]:>6.1f}% | {wa[1]:>6.1f}% | "
              f"{wa[2]:>6.1f}% | {wa[3]:>6.1f}%")

    # Table 3: Per-window median rank
    print(f"\n{'Variant':<12} | {'1-50':>7} | {'51-100':>7} | {'101-150':>7} | {'151-200':>7}")
    print("-" * 50)
    for r in results:
        wr = r['window_ranks']
        print(f"{r['variant']:<12} | {wr[0]:>7.0f} | {wr[1]:>7.0f} | "
              f"{wr[2]:>7.0f} | {wr[3]:>7.0f}")

    # Table 4: Weight magnitude trajectory (for each non-frozen variant)
    print(f"\nWeight magnitude trajectory:")
    for r in results:
        if r['variant'] == 'frozen':
            continue
        mags = r['weight_magnitudes']
        if not mags:
            continue
        traj = [f"{m['mag_total']:.4f}" for m in mags]
        # Classify trajectory
        vals = [m['mag_total'] for m in mags]
        if len(vals) > 3:
            growth = vals[-1] - vals[0]
            oscillation = np.std(np.diff(vals))
            if vals[-1] < 0.001:
                pattern = "PLATEAU (near zero — gradient too weak)"
            elif oscillation > abs(growth) * 0.5:
                pattern = "OSCILLATING (lr too high?)"
            elif growth > 0:
                pattern = "GROWING (adaptation accumulating)"
            else:
                pattern = "SHRINKING"
        else:
            pattern = "insufficient data"

        print(f"\n  {r['variant']}: {pattern}")
        print(f"    Every 10 tok: {' → '.join(traj[:10])}")

    # Kill test evaluation
    print(f"\n{'=' * 74}")
    frozen = results[0]
    best_live = max(results[1:], key=lambda r: r['top1_acc'])
    delta = best_live['top1_acc'] - frozen['top1_acc']

    # Check late window specifically
    best_late = max(results[1:], key=lambda r: r['window_acc'][3])
    late_delta = best_late['window_acc'][3] - frozen['window_acc'][3]

    # Check rank improvement
    best_rank = min(results[1:], key=lambda r: r['final_rank_median'])
    rank_delta = frozen['final_rank_median'] - best_rank['final_rank_median']

    print(f"Overall: frozen={frozen['top1_acc']:.1f}%, best={best_live['top1_acc']:.1f}% "
          f"({best_live['variant']}), delta={delta:+.1f}%")
    print(f"Late window (151-200): frozen={frozen['window_acc'][3]:.1f}%, "
          f"best={best_late['window_acc'][3]:.1f}% ({best_late['variant']}), "
          f"delta={late_delta:+.1f}%")
    print(f"Median rank: frozen={frozen['final_rank_median']:.0f}, "
          f"best={best_rank['final_rank_median']:.0f} ({best_rank['variant']}), "
          f"improvement={rank_delta:+.0f} positions")

    if delta > 5.0 or late_delta > 10.0:
        print(f"\n>>> KILL TEST: CONFIRMED <<<")
    elif delta > 2.0 or late_delta > 5.0 or rank_delta > 50:
        print(f"\n>>> KILL TEST: PROMISING — signal present, needs tuning <<<")
    elif rank_delta > 10:
        print(f"\n>>> KILL TEST: WEAK SIGNAL — rank improves but can't flip argmax <<<")
        print(f"    Try rank=8 or rank=16 for more magnitude.")
    else:
        print(f"\n>>> KILL TEST: PARKED — no measurable improvement <<<")
    print(f"{'=' * 74}")


if __name__ == "__main__":
    main()
