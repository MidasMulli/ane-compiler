#!/usr/bin/env python3
"""
Living Model v2 Kill Test — REAL Llama-1B weights.

Uses Llama-1B's actual Q projection (layer 0) + embedding table.
LoRA at Q projection as CoreML mutable state buffers.
200 tokens of 70B ground truth. Full diagnostics.

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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
sys.path.insert(0, os.path.dirname(__file__))

import coremltools as ct
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types

from llama_loader import LlamaModel

LLM_URL = "http://127.0.0.1:8899/v1/chat/completions"
MODEL_PATH = os.path.expanduser(
    "~/.cache/huggingface/hub/models--unsloth--Llama-3.2-1B-Instruct/"
    "snapshots/5a8abab4a5d6f164389b1079fb721cfab8d7126c/")


def build_stateful_q_proj(llama, layer_idx=0, rank=4):
    """Build Llama layer 0's Q projection as stateful CoreML model with LoRA.

    y = W_q @ x + lora_B @ lora_A @ x

    W_q: [2048, 2048] from Llama layer 0 (frozen)
    lora_A: [rank, 2048] mutable state
    lora_B: [2048, rank] mutable state
    """
    layer = llama.layers[layer_idx]
    dim = llama.config.hidden_size  # 2048
    W_q = layer.q_proj_weight.astype(np.float16)  # [2048, 2048]

    print(f"  W_q shape: {W_q.shape}, dtype: {W_q.dtype}")
    print(f"  dim={dim}, rank={rank}")

    @mb.program(
        input_specs=[
            mb.TensorSpec(shape=(1, dim), dtype=types.fp16),
            mb.StateTensorSpec(shape=(rank, dim), dtype=types.fp16),
            mb.StateTensorSpec(shape=(dim, rank), dtype=types.fp16),
        ],
        opset_version=ct.target.iOS18,
    )
    def q_proj_lora(x, lora_A_state, lora_B_state):
        # Base Q projection (frozen Llama weights)
        base_out = mb.linear(x=x, weight=mb.const(val=W_q))

        # Read LoRA state
        lora_A = mb.read_state(input=lora_A_state)
        lora_B = mb.read_state(input=lora_B_state)
        mb.coreml_update_state(state=lora_A_state, value=lora_A)
        mb.coreml_update_state(state=lora_B_state, value=lora_B)

        # LoRA: delta = (x @ A.T) @ B.T
        x32 = mb.cast(x=x, dtype="fp32")
        A32 = mb.cast(x=lora_A, dtype="fp32")
        B32 = mb.cast(x=lora_B, dtype="fp32")
        low = mb.matmul(x=x32, y=A32, transpose_y=True)
        delta = mb.matmul(x=low, y=B32, transpose_y=True)
        delta16 = mb.cast(x=delta, dtype="fp16")

        return mb.add(x=base_out, y=delta16)

    path = '/tmp/llama_q_proj_lora.mlpackage'
    ct_model = ct.convert(q_proj_lora, compute_units=ct.ComputeUnit.CPU_AND_NE,
                           minimum_deployment_target=ct.target.iOS18)
    ct_model.save(path)
    loaded = ct.models.MLModel(path, compute_units=ct.ComputeUnit.CPU_AND_NE)
    print(f"  Stateful: {loaded._is_stateful()}")
    return loaded, W_q.astype(np.float32), dim, rank


def generate_ground_truth(n_tokens=220):
    """Get 70B to generate domain-specific content."""
    payload = json.dumps({
        "model": "mlx-community/Llama-3.3-70B-Instruct-4bit",
        "messages": [{"role": "user", "content":
            "Explain the Apple Neural Engine hardware architecture. "
            "Cover the 16-core tile structure, 17-stage pipeline, "
            "opcode encoding, SRAM organization, dispatch mechanism, "
            "and IOKit kext interface. Be specific with numbers."}],
        "max_tokens": n_tokens, "temperature": 0.3,
        "repetition_penalty": 1.1,
    }).encode()
    req = urllib.request.Request(LLM_URL, data=payload,
        headers={"Content-Type": "application/json"})
    resp = urllib.request.urlopen(req, timeout=300)
    data = json.loads(resp.read())
    return data["choices"][0]["message"]["content"]


def run_variant(model, W_q, embed_table, dim, rank, gt_tokens,
                lr, variant_name):
    """Run 200 tokens with full diagnostics."""
    from llama_loader import rms_norm_cpu
    config_eps = 1e-5

    state = model.make_state()
    n = min(len(gt_tokens) - 1, 199)

    # Get RMSNorm weights for layer 0 input norm
    llama = LlamaModel.from_safetensors(MODEL_PATH)
    ln_w = llama.layers[0].input_layernorm_weight

    # Init LoRA
    lora_A = np.zeros((rank, dim), dtype=np.float32)
    lora_B = np.zeros((dim, rank), dtype=np.float32)
    state.write_state('lora_A_state', lora_A)
    state.write_state('lora_B_state', lora_B)

    # Tracking
    top1_correct = 0
    total = 0
    window_correct = [0, 0, 0, 0]
    weight_mags = []
    rank_history = []

    # For "accuracy" we check if the Q projection output, projected back
    # through the embedding table, predicts the next token.
    # This is a proxy: Q_proj output → dot with all embeddings → argmax
    # Not a full LM forward pass, but tests if LoRA improves Q's
    # representation of the input for next-token prediction.

    t0 = time.perf_counter()

    for pos in range(n):
        token = gt_tokens[pos]
        target = gt_tokens[pos + 1]

        # Input: embed(token) → RMSNorm (layer 0 pre-attention)
        x_raw = embed_table[token].astype(np.float16)
        x_norm = rms_norm_cpu(x_raw, ln_w, config_eps)
        x = x_norm.astype(np.float32).reshape(1, dim)

        # Forward through stateful Q projection
        result = model.predict({'x': x}, state=state)
        q_out = list(result.values())[0].flatten().astype(np.float32)

        # Project Q output back to vocab via embed table dot product
        # logits[i] = q_out · embed[i]
        logits = embed_table.astype(np.float32) @ q_out  # [vocab_size]

        # Top-5
        top5_idx = np.argsort(logits)[-5:][::-1]
        predicted = top5_idx[0]

        if predicted == target:
            top1_correct += 1
        total += 1

        # Window tracking
        w = min(pos // 50, 3)
        if predicted == target:
            window_correct[w] += 1

        # Rank of correct token
        sorted_idx = np.argsort(logits)[::-1]
        correct_pos = np.where(sorted_idx == target)[0]
        rank_val = int(correct_pos[0]) if len(correct_pos) > 0 else len(logits)
        rank_history.append(rank_val)

        # Gradient + state update (skip for frozen)
        if variant_name != "frozen" and lr > 0:
            # Target: minimize cross-entropy of logits vs target token
            # Softmax
            logits_shifted = logits - logits.max()
            exp_l = np.exp(np.clip(logits_shifted, -30, 30))
            probs = exp_l / exp_l.sum()

            # Gradient of CE w.r.t. q_out:
            # dL/dq = embed.T @ (probs - one_hot)  [dim]
            grad_probs = probs.copy()
            grad_probs[target] -= 1.0
            grad_q = embed_table.astype(np.float32).T @ grad_probs  # [dim]

            # Backprop through LoRA:
            # q_out = W_q @ x + B @ A @ x
            # dL/dB = grad_q.reshape(dim,1) @ (A @ x.T).reshape(1,rank)
            x_flat = x.flatten()
            low = lora_A @ x_flat  # [rank]
            grad_B = np.outer(grad_q, low)  # [dim, rank]

            # dL/dA = (B.T @ grad_q).reshape(rank,1) @ x.reshape(1,dim)
            grad_low = lora_B.T @ grad_q  # [rank]
            grad_A = np.outer(grad_low, x_flat)  # [rank, dim]

            # SGD
            lora_A -= lr * grad_A
            lora_B -= lr * grad_B

            # Write updated state
            state.write_state('lora_A_state', lora_A)
            state.write_state('lora_B_state', lora_B)

        # Weight magnitude every 10 tokens
        if (pos + 1) % 10 == 0:
            mag = float(np.linalg.norm(lora_A) + np.linalg.norm(lora_B))
            weight_mags.append({'pos': pos + 1, 'mag': mag})

    elapsed = time.perf_counter() - t0

    # Window accuracy
    window_sizes = [min(50, n - i * 50) for i in range(4)]
    window_acc = [window_correct[i] / max(1, window_sizes[i]) * 100
                  for i in range(4)]

    # Window median ranks
    window_ranks = []
    for w in range(4):
        s, e = w * 50, min((w + 1) * 50, len(rank_history))
        window_ranks.append(float(np.median(rank_history[s:e])) if s < e else 0)

    return {
        'variant': variant_name, 'lr': lr,
        'top1_acc': top1_correct / total * 100 if total else 0,
        'total': total, 'elapsed': elapsed,
        'window_acc': window_acc,
        'window_ranks': window_ranks,
        'weight_mags': weight_mags,
        'median_rank': float(np.median(rank_history)),
    }


def main():
    print("=" * 74)
    print("LIVING MODEL v2 KILL TEST — REAL LLAMA-1B WEIGHTS")
    print("Q projection (layer 0) + embedding table + 70B ground truth")
    print("=" * 74)

    # Load Llama
    print("\n[1/4] Loading Llama-1B...")
    llama = LlamaModel.from_safetensors(MODEL_PATH)
    embed = llama.embed_tokens  # [128256, 2048]
    print(f"  Loaded. dim={llama.config.hidden_size}, vocab={llama.config.vocab_size}")

    # Generate ground truth
    print("\n[2/4] Generating 200 tokens via 70B Q4...")
    gt_text = generate_ground_truth(220)
    print(f"  Text: {gt_text[:80]}...")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('unsloth/Llama-3.2-1B-Instruct')
    gt_tokens = tokenizer.encode(gt_text, add_special_tokens=False)[:201]
    print(f"  Tokens: {len(gt_tokens)}")

    # Build stateful model
    print("\n[3/4] Building stateful Q projection with LoRA...")
    model, W_q, dim, rank = build_stateful_q_proj(llama, layer_idx=0, rank=4)

    # Run variants
    print("\n[4/4] Running kill test...")

    variants = [
        ("frozen", 0),
        ("lr=1e-4", 1e-4),
        ("lr=1e-3", 1e-3),
        ("lr=1e-2", 1e-2),
    ]

    results = []
    for name, lr in variants:
        print(f"\n  --- {name} ---")
        r = run_variant(model, W_q, embed, dim, rank, gt_tokens, lr, name)
        results.append(r)
        print(f"  Top-1: {r['top1_acc']:.1f}% | Med rank: {r['median_rank']:.0f} | "
              f"{r['elapsed']:.1f}s")
        print(f"  Windows: {['%.1f%%' % w for w in r['window_acc']]}")
        print(f"  Win ranks: {['%.0f' % w for w in r['window_ranks']]}")

    # Report
    print(f"\n\n{'=' * 74}")
    print("RESULTS")
    print(f"{'=' * 74}")

    print(f"\n{'Variant':<12} | {'Top-1':>6} | {'Med Rank':>8} | {'Time':>5}")
    print("-" * 40)
    for r in results:
        print(f"{r['variant']:<12} | {r['top1_acc']:>5.1f}% | "
              f"{r['median_rank']:>8.0f} | {r['elapsed']:>4.1f}s")

    print(f"\nPer-window accuracy:")
    print(f"{'Variant':<12} | {'1-50':>7} | {'51-100':>7} | {'101-150':>7} | {'151-200':>7}")
    print("-" * 55)
    for r in results:
        wa = r['window_acc']
        print(f"{r['variant']:<12} | {wa[0]:>6.1f}% | {wa[1]:>6.1f}% | "
              f"{wa[2]:>6.1f}% | {wa[3]:>6.1f}%")

    print(f"\nPer-window median rank (lower = better):")
    print(f"{'Variant':<12} | {'1-50':>7} | {'51-100':>7} | {'101-150':>7} | {'151-200':>7}")
    print("-" * 55)
    for r in results:
        wr = r['window_ranks']
        print(f"{r['variant']:<12} | {wr[0]:>7.0f} | {wr[1]:>7.0f} | "
              f"{wr[2]:>7.0f} | {wr[3]:>7.0f}")

    print(f"\nWeight magnitude trajectory:")
    for r in results:
        if r['variant'] == 'frozen':
            continue
        mags = r['weight_mags']
        if not mags:
            continue
        vals = [m['mag'] for m in mags]
        traj = [f"{v:.4f}" for v in vals[:10]]
        if len(vals) > 3:
            growth = vals[-1] - vals[0]
            osc = np.std(np.diff(vals)) if len(vals) > 1 else 0
            if vals[-1] < 0.001:
                pat = "PLATEAU NEAR ZERO"
            elif osc > abs(growth) * 0.5 and growth != 0:
                pat = "OSCILLATING"
            elif growth > 0:
                pat = "GROWING"
            else:
                pat = "SHRINKING"
        else:
            pat = "?"
        print(f"  {r['variant']}: {pat}")
        print(f"    {' → '.join(traj)}")

    # Kill test
    print(f"\n{'=' * 74}")
    frozen = results[0]
    best = max(results[1:], key=lambda r: r['top1_acc'])
    delta = best['top1_acc'] - frozen['top1_acc']

    best_late = max(results[1:], key=lambda r: r['window_acc'][3])
    late_delta = best_late['window_acc'][3] - frozen['window_acc'][3]

    best_rank = min(results[1:], key=lambda r: r['median_rank'])
    rank_imp = frozen['median_rank'] - best_rank['median_rank']

    print(f"Top-1: frozen={frozen['top1_acc']:.1f}%, best={best['top1_acc']:.1f}% "
          f"({best['variant']}), delta={delta:+.1f}%")
    print(f"Late window: frozen={frozen['window_acc'][3]:.1f}%, "
          f"best={best_late['window_acc'][3]:.1f}% ({best_late['variant']}), "
          f"delta={late_delta:+.1f}%")
    print(f"Median rank: frozen={frozen['median_rank']:.0f}, "
          f"best={best_rank['median_rank']:.0f} ({best_rank['variant']}), "
          f"improvement={rank_imp:+.0f}")

    if delta > 5.0 or late_delta > 10.0:
        print(f"\n>>> KILL TEST: CONFIRMED <<<")
    elif delta > 2.0 or late_delta > 5.0 or rank_imp > 100:
        print(f"\n>>> KILL TEST: PROMISING <<<")
    elif rank_imp > 20:
        print(f"\n>>> KILL TEST: WEAK SIGNAL — rank improves, can't flip argmax <<<")
    else:
        print(f"\n>>> KILL TEST: PARKED <<<")
    print(f"{'=' * 74}")


if __name__ == "__main__":
    main()
