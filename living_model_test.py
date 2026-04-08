#!/usr/bin/env python3
"""
Phase 3A: The Living Model — test-time training during inference.

Kill test: does a 1B model improve its next-token prediction accuracy
during a conversation by applying live LoRA updates after each token?

Architecture:
  - 1B runs forward on ANE (frozen weights, 50 tok/s)
  - After final RMSNorm, hidden state [2048] is extracted
  - lm_head logits computed on ANE (base prediction)
  - CPU applies LoRA correction: delta = hidden @ A @ B
    where A: [2048, rank], B: [rank, vocab_or_subset]
  - After 70B reveals actual token, CPU computes gradient
    and updates A, B via SGD
  - Next token: 1B base + updated LoRA correction

Kill test protocol:
  1. Generate 200 tokens of ANE register conversation via 70B
  2. Frozen baseline: run 1B, record top-1 match rate vs 70B
  3. Live LoRA: run 1B + LoRA updates, record top-1 match rate
  4. If accuracy improves >5%: CONFIRMED
  5. If <5% or degrades: PARKED

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

from llama_loader import LlamaModel, rms_norm_cpu, rope_cpu, softmax_cpu
from kv_cache import KVCache

MODEL_PATH = os.path.expanduser(
    "~/.cache/huggingface/hub/models--unsloth--Llama-3.2-1B-Instruct/"
    "snapshots/5a8abab4a5d6f164389b1079fb721cfab8d7126c/"
)
LLM_URL = "http://127.0.0.1:8899/v1/chat/completions"


# ===================================================================
# Step 1: Generate ground truth conversation via 70B
# ===================================================================

def generate_ground_truth(n_tokens=200):
    """Get 70B to generate a conversation about ANE registers."""
    prompt = (
        "Explain in detail the Apple Neural Engine hardware architecture. "
        "Cover the register map, dispatch mechanism, the 16-core tile structure, "
        "the 17-stage pipeline, opcode encoding format, SRAM organization, "
        "and how the kext interfaces with the hardware via IOKit selectors. "
        "Be specific with hex addresses, bit fields, and timing numbers."
    )
    payload = json.dumps({
        "model": "mlx-community/Llama-3.3-70B-Instruct-3bit",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": n_tokens, "temperature": 0.3,
        "repetition_penalty": 1.1,
    }).encode()
    req = urllib.request.Request(LLM_URL, data=payload,
        headers={"Content-Type": "application/json"})
    resp = urllib.request.urlopen(req, timeout=300)
    data = json.loads(resp.read())
    text = data["choices"][0]["message"]["content"]
    return text


# ===================================================================
# Step 2: 1B forward pass with hidden state extraction
# ===================================================================

class LlamaInference:
    """1B inference with hidden state access for LoRA injection."""

    def __init__(self, model_path):
        print("  Loading 1B model...")
        self.model = LlamaModel.from_safetensors(model_path)
        self.config = self.model.config

        print("  Building MIL IR models...")
        from run_llama_fused import build_all_models
        self.ct_models, self.dispatch_mode = build_all_models(self.model)

        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            'unsloth/Llama-3.2-1B-Instruct')

        self.kv = None

    def reset_kv(self):
        cfg = self.config
        self.kv = KVCache(cfg.n_layers, cfg.n_kv_heads, cfg.head_dim)

    def forward_one(self, token_id, position):
        """Forward pass returning (logits, hidden_state_before_lm_head).

        hidden_state is the [2048] FP16 vector after final RMSNorm,
        before lm_head projection. This is where LoRA correction is applied.
        """
        config = self.config
        dim = config.hidden_size
        n_heads = config.n_heads
        n_kv_heads = config.n_kv_heads
        head_dim = config.head_dim
        n_rep = config.n_rep

        x_fp16 = self.model.embed_tokens[int(token_id)].astype(np.float16)

        for li in range(config.n_layers):
            L = self.model.layers[li]
            if f'L{li}_pre' in self.ct_models:
                pre_result = self.ct_models[f'L{li}_pre'].predict({
                    'x': x_fp16.reshape(1, dim, 1, 1).astype(np.float32)})
                qkv = list(pre_result.values())[0].flatten().astype(np.float16)
            else:
                ln1 = rms_norm_cpu(x_fp16, L.input_layernorm_weight,
                                   config.rms_norm_eps)
                qkv_result = self.ct_models[f'L{li}_qkv'].predict({
                    'x': ln1.reshape(1, dim, 1, 1).astype(np.float32)})
                qkv = list(qkv_result.values())[0].flatten().astype(np.float16)

            q = qkv[:dim].reshape(n_heads, head_dim)
            k = qkv[dim:dim + n_kv_heads * head_dim].reshape(n_kv_heads, head_dim)
            v = qkv[dim + n_kv_heads * head_dim:].reshape(n_kv_heads, head_dim)
            q, k = rope_cpu(q, k, position, head_dim, config.rope_theta)
            self.kv.append(li, k[np.newaxis], v[np.newaxis])
            cached_k, cached_v = self.kv.get(li)
            scale = np.float32(1.0 / np.sqrt(head_dim))
            attn_output = np.zeros(dim, dtype=np.float32)
            for h in range(n_heads):
                kv_h = h // n_rep
                q_h = q[h].astype(np.float32)
                k_h = cached_k[:, kv_h, :].astype(np.float32)
                v_h = cached_v[:, kv_h, :].astype(np.float32)
                scores = (q_h @ k_h.T) * scale
                weights = softmax_cpu(scores)
                attn_output[h * head_dim:(h + 1) * head_dim] = weights @ v_h
            attn_out = attn_output.astype(np.float16)

            post_result = self.ct_models[f'L{li}_post'].predict({
                'attn_out': attn_out.reshape(1, dim, 1, 1).astype(np.float32),
                'x': x_fp16.reshape(1, dim, 1, 1).astype(np.float32),
            })
            x_fp16 = list(post_result.values())[0].flatten().astype(np.float16)

        # Final RMSNorm → hidden state (this is what LoRA corrects)
        hidden = rms_norm_cpu(x_fp16, self.model.norm_weight,
                              config.rms_norm_eps)

        # lm_head via ANE chunks
        logit_chunks = []
        n_chunks = config.vocab_size // 16032
        if config.vocab_size % 16032 != 0:
            n_chunks += 1
        for j in range(n_chunks):
            lm_result = self.ct_models[f'lm_head_{j}'].predict({
                'x': hidden.reshape(1, dim, 1, 1).astype(np.float32)})
            logit_chunks.append(list(lm_result.values())[0].flatten())

        logits = np.concatenate(logit_chunks).astype(np.float32)
        return logits, hidden.astype(np.float32)


# ===================================================================
# Step 3: LoRA module (CPU, rank-4)
# ===================================================================

class LoRACorrection:
    """Rank-4 LoRA applied to hidden state → logit correction.

    Instead of modifying lm_head weights (128K x 2048 = too large),
    we project hidden [2048] through A[2048,rank] → B[rank, vocab_subset].

    For efficiency, we only correct the top-K logit positions identified
    by the base model. This keeps the LoRA matmul small.
    """

    def __init__(self, hidden_dim=2048, vocab_size=128256, rank=4, lr=0.01,
                 top_k=1000):
        self.rank = rank
        self.lr = lr
        self.top_k = top_k
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size

        # Full vocab LoRA: A[hidden, rank], B[rank, vocab]
        # But 128K vocab makes B huge. Use top-K subset tracking.
        # A projects hidden → low-rank space
        # B projects low-rank → correction for each vocab token
        self.A = np.zeros((hidden_dim, rank), dtype=np.float32)
        self.B = np.zeros((rank, vocab_size), dtype=np.float32)

        # Track which tokens have been seen for gradient updates
        self._seen_tokens = set()

    def correct(self, hidden, base_logits):
        """Apply LoRA correction to logits.

        Args:
            hidden: [hidden_dim] FP32
            base_logits: [vocab_size] FP32

        Returns:
            corrected_logits: [vocab_size] FP32
        """
        # LoRA forward: hidden @ A @ B → [vocab_size]
        low_rank = hidden @ self.A  # [rank]
        delta = low_rank @ self.B   # [vocab_size]
        return base_logits + delta

    def update(self, hidden, target_token_id, corrected_logits):
        """SGD update after seeing the true token from 70B.

        Simple cross-entropy gradient on the corrected logits.
        Only updates the columns of B that correspond to frequently
        seen tokens (sparse update for efficiency).
        """
        self._seen_tokens.add(target_token_id)

        # Softmax of corrected logits
        logits = corrected_logits - corrected_logits.max()
        exp_logits = np.exp(logits.clip(-30, 30))
        probs = exp_logits / exp_logits.sum()

        # Gradient of cross-entropy: probs - one_hot(target)
        grad_logits = probs.copy()
        grad_logits[target_token_id] -= 1.0  # [vocab_size]

        # Backprop through B: grad_B = low_rank.T @ grad_logits
        low_rank = hidden @ self.A  # [rank]
        grad_B = np.outer(low_rank, grad_logits)  # [rank, vocab_size]

        # Backprop through A: grad_A = hidden.T @ (grad_logits @ B.T)
        grad_low_rank = grad_logits @ self.B.T  # [rank]
        grad_A = np.outer(hidden, grad_low_rank)  # [hidden_dim, rank]

        # SGD update
        self.A -= self.lr * grad_A
        self.B -= self.lr * grad_B


# ===================================================================
# Step 4: Kill test
# ===================================================================

def run_kill_test():
    print("=" * 70)
    print("PHASE 3A: THE LIVING MODEL — TEST-TIME TRAINING")
    print("Kill test: does live LoRA improve 1B prediction accuracy?")
    print("=" * 70)

    # Generate ground truth
    print("\n[1/5] Generating 200 tokens of ANE content via 70B...")
    t0 = time.time()
    gt_text = generate_ground_truth(200)
    print(f"  Generated in {time.time()-t0:.1f}s")
    print(f"  Text: {gt_text[:100]}...")

    # Load 1B
    print("\n[2/5] Loading 1B model...")
    llama = LlamaInference(MODEL_PATH)

    # Tokenize ground truth
    gt_tokens = llama.tokenizer.encode(gt_text, add_special_tokens=False)
    n_tokens = min(len(gt_tokens), 200)
    gt_tokens = gt_tokens[:n_tokens]
    print(f"  Ground truth: {n_tokens} tokens")

    # === FROZEN BASELINE ===
    print(f"\n[3/5] Frozen baseline ({n_tokens} tokens)...")
    llama.reset_kv()
    frozen_correct = 0
    frozen_total = 0

    t0 = time.time()
    for pos in range(n_tokens - 1):
        token = gt_tokens[pos]
        target = gt_tokens[pos + 1]

        logits, hidden = llama.forward_one(token, pos)
        predicted = int(np.argmax(logits))

        if predicted == target:
            frozen_correct += 1
        frozen_total += 1

        if (pos + 1) % 50 == 0:
            acc = frozen_correct / frozen_total * 100
            elapsed = time.time() - t0
            tps = frozen_total / elapsed
            print(f"    pos {pos+1}: {frozen_correct}/{frozen_total} = {acc:.1f}% "
                  f"({tps:.1f} tok/s)")

    frozen_acc = frozen_correct / frozen_total * 100
    frozen_time = time.time() - t0
    print(f"  Frozen: {frozen_correct}/{frozen_total} = {frozen_acc:.1f}% "
          f"in {frozen_time:.1f}s ({frozen_total/frozen_time:.1f} tok/s)")

    # === LIVE LORA ===
    print(f"\n[4/5] Live LoRA (rank=4, lr=0.01, {n_tokens} tokens)...")
    llama.reset_kv()
    lora = LoRACorrection(
        hidden_dim=llama.config.hidden_size,
        vocab_size=llama.config.vocab_size,
        rank=4, lr=0.01)

    lora_correct = 0
    lora_total = 0
    # Track accuracy in windows for trajectory
    window_size = 25
    window_correct = 0

    t0 = time.time()
    for pos in range(n_tokens - 1):
        token = gt_tokens[pos]
        target = gt_tokens[pos + 1]

        # 1B forward (ANE)
        base_logits, hidden = llama.forward_one(token, pos)

        # LoRA correction (CPU)
        corrected_logits = lora.correct(hidden, base_logits)
        predicted = int(np.argmax(corrected_logits))

        if predicted == target:
            lora_correct += 1
            window_correct += 1
        lora_total += 1

        # Update LoRA with ground truth (CPU backward)
        lora.update(hidden, target, corrected_logits)

        if (pos + 1) % window_size == 0:
            window_acc = window_correct / window_size * 100
            total_acc = lora_correct / lora_total * 100
            elapsed = time.time() - t0
            tps = lora_total / elapsed
            print(f"    pos {pos+1}: window={window_acc:.1f}% total={total_acc:.1f}% "
                  f"({tps:.1f} tok/s)")
            window_correct = 0

    lora_acc = lora_correct / lora_total * 100
    lora_time = time.time() - t0
    print(f"  LoRA: {lora_correct}/{lora_total} = {lora_acc:.1f}% "
          f"in {lora_time:.1f}s ({lora_total/lora_time:.1f} tok/s)")

    # === RESULTS ===
    print(f"\n[5/5] Results")
    print(f"{'=' * 70}")
    delta = lora_acc - frozen_acc
    print(f"  Frozen baseline: {frozen_acc:.1f}%")
    print(f"  Live LoRA:       {lora_acc:.1f}%")
    print(f"  Delta:           {delta:+.1f}%")
    print(f"  Frozen speed:    {frozen_total/frozen_time:.1f} tok/s")
    print(f"  LoRA speed:      {lora_total/lora_time:.1f} tok/s")
    print(f"  LoRA overhead:   {(lora_time/frozen_time - 1)*100:.1f}%")
    print(f"")

    if delta > 5.0:
        print(f"  KILL TEST: CONFIRMED (+{delta:.1f}% > 5% threshold)")
        print(f"  The model learns during inference.")
    elif delta > 0:
        print(f"  KILL TEST: PARKED (+{delta:.1f}% < 5% threshold)")
        print(f"  Positive but below threshold. Investigate: rank, lr, update frequency.")
    else:
        print(f"  KILL TEST: PARKED ({delta:.1f}% — no improvement or degradation)")
        print(f"  LoRA correction not learning effectively. Check: lr too high/low,")
        print(f"  rank insufficient, catastrophic forgetting, or base model too weak.")

    print(f"{'=' * 70}")

    return {
        'frozen_acc': frozen_acc,
        'lora_acc': lora_acc,
        'delta': delta,
        'n_tokens': n_tokens,
        'frozen_tps': frozen_total / frozen_time,
        'lora_tps': lora_total / lora_time,
    }


if __name__ == "__main__":
    run_kill_test()
