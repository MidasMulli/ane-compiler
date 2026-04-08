#!/usr/bin/env python3
"""
Phase 3A Revival: Living Model with warmup variant.

Previous kill test showed +0% overall BUT 76% accuracy at tokens 150-175
vs 55% frozen. The model learned late — early random-init penalty canceled.

This test: pre-adapt the LoRA on 50 tokens of domain context BEFORE
the conversation starts. If the 76% window extends to the full
conversation, the adaptation is real and fast enough to matter.

Three variants tested:
  A) Frozen baseline (same as before)
  B) Warmup: lr=0 for first 50 tokens, then lr=0.01
  C) Pre-adapt: run 50 tokens of separate domain text through LoRA
     updates first, THEN run the test conversation with live updates

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

from living_model_test import (
    LlamaInference, LoRACorrection, generate_ground_truth, MODEL_PATH
)

LLM_URL = "http://127.0.0.1:8899/v1/chat/completions"


def generate_domain_primer(n_tokens=60):
    """Generate a separate domain-relevant text for pre-adaptation."""
    payload = json.dumps({
        "model": "mlx-community/Llama-3.3-70B-Instruct-3bit",
        "messages": [{"role": "user", "content":
            "Describe the ANE dispatch mechanism: IOKit selectors, "
            "IOSurface buffer mapping, the _ANEClient API, and how "
            "doEvaluateDirectWithModel works at the kext level."}],
        "max_tokens": n_tokens, "temperature": 0.3,
        "repetition_penalty": 1.1,
    }).encode()
    req = urllib.request.Request(LLM_URL, data=payload,
        headers={"Content-Type": "application/json"})
    resp = urllib.request.urlopen(req, timeout=120)
    data = json.loads(resp.read())
    return data["choices"][0]["message"]["content"]


def run_variant(llama, gt_tokens, lora, variant_name, warmup_tokens=0):
    """Run a single variant and return per-window accuracy."""
    llama.reset_kv()
    n_tokens = len(gt_tokens)
    correct = 0
    total = 0
    window_size = 25
    windows = []
    window_correct = 0

    t0 = time.time()
    for pos in range(n_tokens - 1):
        token = gt_tokens[pos]
        target = gt_tokens[pos + 1]

        base_logits, hidden = llama.forward_one(token, pos)

        if lora is not None:
            corrected = lora.correct(hidden, base_logits)
            predicted = int(np.argmax(corrected))

            # Update LoRA (skip if in warmup-freeze zone)
            if variant_name != "warmup" or pos >= warmup_tokens:
                lora.update(hidden, target, corrected)
        else:
            predicted = int(np.argmax(base_logits))

        if predicted == target:
            correct += 1
            window_correct += 1
        total += 1

        if total % window_size == 0:
            windows.append(window_correct / window_size * 100)
            window_correct = 0

    # Final partial window
    if total % window_size != 0:
        remaining = total % window_size
        windows.append(window_correct / remaining * 100)

    elapsed = time.time() - t0
    acc = correct / total * 100
    tps = total / elapsed

    return {
        'name': variant_name,
        'accuracy': acc,
        'correct': correct,
        'total': total,
        'tps': tps,
        'elapsed': elapsed,
        'windows': windows,
    }


def main():
    print("=" * 70)
    print("PHASE 3A REVIVAL: LIVING MODEL — WARMUP VARIANTS")
    print("=" * 70)

    # Generate ground truth
    print("\n[1/6] Generating test conversation via 70B (200 tokens)...")
    gt_text = generate_ground_truth(200)
    print(f"  Generated: {gt_text[:80]}...")

    # Generate domain primer (separate text for pre-adaptation)
    print("\n[2/6] Generating domain primer via 70B (60 tokens)...")
    primer_text = generate_domain_primer(60)
    print(f"  Primer: {primer_text[:80]}...")

    # Load 1B
    print("\n[3/6] Loading 1B...")
    llama = LlamaInference(MODEL_PATH)

    gt_tokens = llama.tokenizer.encode(gt_text, add_special_tokens=False)[:200]
    primer_tokens = llama.tokenizer.encode(primer_text, add_special_tokens=False)[:60]
    print(f"  Test tokens: {len(gt_tokens)}, Primer tokens: {len(primer_tokens)}")

    # === Variant A: Frozen baseline ===
    print(f"\n[4/6] Variant A: Frozen baseline...")
    result_a = run_variant(llama, gt_tokens, None, "frozen")
    print(f"  Accuracy: {result_a['accuracy']:.1f}% ({result_a['tps']:.1f} tok/s)")
    print(f"  Windows: {['%.0f' % w for w in result_a['windows']]}")

    # === Variant B: Warmup (lr=0 for 50 tokens, then lr=0.01) ===
    print(f"\n[5/6] Variant B: Warmup (lr=0 for 50 tokens, then lr=0.01)...")
    lora_b = LoRACorrection(
        hidden_dim=llama.config.hidden_size,
        vocab_size=llama.config.vocab_size,
        rank=4, lr=0.01)
    result_b = run_variant(llama, gt_tokens, lora_b, "warmup", warmup_tokens=50)
    print(f"  Accuracy: {result_b['accuracy']:.1f}% ({result_b['tps']:.1f} tok/s)")
    print(f"  Windows: {['%.0f' % w for w in result_b['windows']]}")

    # === Variant C: Pre-adapt on primer, then test with live updates ===
    print(f"\n[6/6] Variant C: Pre-adapt on {len(primer_tokens)} primer tokens...")
    lora_c = LoRACorrection(
        hidden_dim=llama.config.hidden_size,
        vocab_size=llama.config.vocab_size,
        rank=4, lr=0.01)

    # Pre-adapt: run primer through 1B with LoRA updates
    llama.reset_kv()
    for pos in range(len(primer_tokens) - 1):
        token = primer_tokens[pos]
        target = primer_tokens[pos + 1]
        base_logits, hidden = llama.forward_one(token, pos)
        corrected = lora_c.correct(hidden, base_logits)
        lora_c.update(hidden, target, corrected)
    print(f"  Pre-adapted on {len(primer_tokens)-1} tokens")

    # Now test on the actual conversation
    result_c = run_variant(llama, gt_tokens, lora_c, "pre-adapt")
    print(f"  Accuracy: {result_c['accuracy']:.1f}% ({result_c['tps']:.1f} tok/s)")
    print(f"  Windows: {['%.0f' % w for w in result_c['windows']]}")

    # === Results ===
    print(f"\n{'=' * 70}")
    print("RESULTS")
    print(f"{'=' * 70}")

    results = [result_a, result_b, result_c]
    print(f"\n{'Variant':<20} | {'Accuracy':>8} | {'Delta':>7} | {'tok/s':>6}")
    print("-" * 50)
    for r in results:
        delta = r['accuracy'] - result_a['accuracy']
        print(f"{r['name']:<20} | {r['accuracy']:>7.1f}% | {delta:>+6.1f}% | {r['tps']:>5.1f}")

    # Window trajectory comparison
    print(f"\nPer-window accuracy (each window = 25 tokens):")
    max_windows = max(len(r['windows']) for r in results)
    header = f"{'Window':<8}"
    for r in results:
        header += f" | {r['name']:>10}"
    print(header)
    print("-" * len(header))
    for i in range(max_windows):
        row = f"{'%d-%d' % (i*25+1, (i+1)*25):<8}"
        for r in results:
            if i < len(r['windows']):
                row += f" | {r['windows'][i]:>9.0f}%"
            else:
                row += f" | {'':>10}"
        print(row)

    # Kill test evaluation
    print(f"\n{'=' * 70}")
    best = max(results[1:], key=lambda r: r['accuracy'])
    delta = best['accuracy'] - result_a['accuracy']
    if delta > 5.0:
        print(f"KILL TEST: CONFIRMED — {best['name']} at +{delta:.1f}%")
        print(f"The model learns during inference and the warmup eliminates early penalty.")
    elif delta > 2.0:
        print(f"KILL TEST: PROMISING — {best['name']} at +{delta:.1f}% (below 5% threshold)")
        print(f"Try: rank=8, lr=0.005, longer primer, or target attention projections.")
    elif delta > 0:
        print(f"KILL TEST: PARKED — {best['name']} at +{delta:.1f}% (minimal improvement)")
    else:
        print(f"KILL TEST: PARKED — no variant improved over frozen baseline")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
