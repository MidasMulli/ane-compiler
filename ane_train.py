#!/usr/bin/env python3
"""
On-device fine-tuning on Apple Neural Engine.

Forward pass on ANE (guaranteed execution).
Backward pass on CPU (numpy gradient).
Weight update via recompile (20ms per step).

SIP ON. Stock macOS. No hacks.

Usage:
  python ane_train.py --prompt "The capital of France is" --target " Paris"
  python ane_train.py --data training_pairs.jsonl --steps 100 --lr 0.001

Copyright 2026 Nick Lo. MIT License.
"""

import argparse
import os
import sys
import time
import shutil
import numpy as np
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))


class ANETrainer:
    """Fine-tune a single projection layer on ANE.

    The training loop:
      1. Forward pass on ANE (guaranteed execution, 0.23ms)
      2. Compute loss on CPU (cross-entropy)
      3. Compute gradient on CPU (numpy matmul)
      4. Apply gradient to weights
      5. Write new .mlmodelc with updated weights (1ms)
      6. aned recompiles (20ms)
      7. Reload into pipe tool
      8. Repeat

    This trains ONE projection at a time (LoRA-style).
    The projection being trained is recompiled each step.
    Other projections remain static in the pipe tool.
    """

    def __init__(self, model_path=None, build_dir='/tmp/ane_train'):
        from llama_loader import LlamaModel, compile_llama_unfused
        from generate import ANEDispatcher
        from kv_cache import KVCache
        from transformers import AutoTokenizer

        if model_path is None:
            model_path = os.path.expanduser(
                "~/.cache/huggingface/hub/models--unsloth--Llama-3.2-1B-Instruct/"
                "snapshots/5a8abab4a5d6f164389b1079fb721cfab8d7126c/model.safetensors")

        self.model = LlamaModel.from_safetensors(model_path)
        self.config = self.model.config
        self.tokenizer = AutoTokenizer.from_pretrained('unsloth/Llama-3.2-1B-Instruct')
        self.build_dir = build_dir

        # Compile all ops (static, don't change during training)
        self.compiled = compile_llama_unfused(self.model, build_dir)
        self.dispatch_ops = {k: v for k, v in self.compiled.items()
                             if not k.startswith('_') and len(v) == 3}

        # Load dispatcher
        self.dispatcher = ANEDispatcher(self.dispatch_ops, quiet=True)
        self.dispatcher.start()

        self.kv_cache = KVCache(
            self.config.n_layers, self.config.n_kv_heads, self.config.head_dim)

    def forward_token(self, token_id, position):
        """Forward pass for one token through all layers on ANE."""
        from llama_loader import forward_layer_decode_unfused, rms_norm_cpu

        x = self.model.embed_tokens[int(token_id)].astype(np.float16)

        for li in range(self.config.n_layers):
            x = forward_layer_decode_unfused(
                li, x, self.model, self.dispatcher, self.kv_cache, position)

        x = rms_norm_cpu(x, self.model.norm_weight, self.config.rms_norm_eps)
        logits = np.concatenate([
            self.dispatcher.dispatch(f'lm_head_chunk_{i}', x) for i in range(8)])

        return logits, x  # logits for loss, hidden state for gradient

    def compute_loss(self, logits, target_id):
        """Cross-entropy loss."""
        logits_f32 = logits.astype(np.float32)
        # Stable softmax
        logits_f32 -= logits_f32.max()
        exp_logits = np.exp(logits_f32)
        probs = exp_logits / exp_logits.sum()
        loss = -np.log(probs[int(target_id)] + 1e-10)
        return loss, probs

    def compute_gradient_lm_head(self, probs, target_id, hidden_state):
        """Gradient of cross-entropy w.r.t. lm_head weights.

        dL/dW = (probs - one_hot) ⊗ hidden_state
        This is the outer product of the error signal and the input.
        """
        error = probs.copy()
        error[int(target_id)] -= 1.0  # probs - one_hot

        # Gradient: error (vocab_size,) × hidden (hidden_size,) = (vocab_size, hidden_size)
        # But lm_head is tied to embed_tokens, so we update embed_tokens
        grad = np.outer(error, hidden_state.astype(np.float32))
        return grad

    def train_step(self, prompt_tokens, target_token, lr=0.001):
        """One training step: forward → loss → gradient → update → recompile.

        Updates the embedding/lm_head weights (tied).
        """
        from compiler import gen_conv_mlmodelc
        from llama_loader import gen_lm_head_chunks

        # Reset KV cache
        self.kv_cache.reset()

        # Forward: process all prompt tokens
        for pos, tok in enumerate(prompt_tokens):
            logits, hidden = self.forward_token(tok, pos)

        # Loss
        loss, probs = self.compute_loss(logits, target_token)

        # Gradient w.r.t. lm_head (= embed_tokens, tied)
        grad = self.compute_gradient_lm_head(probs, target_token, hidden)

        # Apply gradient (SGD)
        # embed_tokens shape: [vocab_size, hidden_size]
        self.model.embed_tokens -= lr * grad.astype(self.model.embed_tokens.dtype)

        # Recompile lm_head chunks with updated weights
        t0 = time.time()
        lm_chunks = gen_lm_head_chunks(
            self.build_dir,
            self.model.embed_tokens.astype(np.float32),
            self.config.hidden_size,
            self.config.vocab_size)

        # Update compiled dict
        for path, ic, oc, offset in lm_chunks:
            name = f'lm_head_chunk_{offset // 16032}'
            self.compiled[name] = (path, ic, oc)

        # Reload dispatcher with updated lm_head
        self.dispatcher.stop()
        self.dispatch_ops = {k: v for k, v in self.compiled.items()
                             if not k.startswith('_') and len(v) == 3}
        self.dispatcher = type(self.dispatcher)(self.dispatch_ops, quiet=True)
        self.dispatcher.start()
        t_recompile = time.time() - t0

        return loss, t_recompile

    def train(self, prompt, target, steps=10, lr=0.001):
        """Train on a single prompt→target pair."""
        prompt_tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
        target_id = self.tokenizer.encode(target, add_special_tokens=False)[0]

        print(f"Training: '{prompt}' → '{self.tokenizer.decode([target_id])}'")
        print(f"Target token: {target_id}")
        print(f"Steps: {steps}, LR: {lr}")
        print()

        for step in range(steps):
            t0 = time.time()
            loss, t_recompile = self.train_step(prompt_tokens, target_id, lr)
            t_total = time.time() - t0

            # Check: what does the model predict now?
            self.kv_cache.reset()
            for pos, tok in enumerate(prompt_tokens):
                logits, _ = self.forward_token(tok, pos)
            pred_token = int(np.argmax(logits.astype(np.float32)))
            pred_text = self.tokenizer.decode([pred_token])

            correct = "✓" if pred_token == target_id else "✗"
            print(f"  step {step+1:3d}: loss={loss:.4f}  pred='{pred_text}' {correct}  "
                  f"recompile={t_recompile*1000:.0f}ms  total={t_total*1000:.0f}ms")

            if pred_token == target_id and loss < 0.1:
                print(f"\n  Converged at step {step+1}!")
                break

    def stop(self):
        if self.dispatcher:
            self.dispatcher.stop()


def main():
    parser = argparse.ArgumentParser(description='On-device fine-tuning on ANE')
    parser.add_argument('--prompt', default='The capital of France is',
                        help='Training prompt')
    parser.add_argument('--target', default=' Paris',
                        help='Target completion')
    parser.add_argument('--steps', type=int, default=20, help='Training steps')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    args = parser.parse_args()

    print("=" * 50)
    print("  ANE On-Device Training")
    print("  Forward: ANE (guaranteed execution)")
    print("  Backward: CPU (numpy gradient)")
    print("  Update: recompile (20ms/step)")
    print("  SIP: ON  |  GPU: idle")
    print("=" * 50)
    print()

    trainer = ANETrainer()
    trainer.train(args.prompt, args.target, args.steps, args.lr)
    trainer.stop()


if __name__ == '__main__':
    main()
