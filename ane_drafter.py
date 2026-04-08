#!/usr/bin/env python3
"""
ANE Drafter for speculative decoding.

Runs Llama-3.2-1B-Instruct on Apple Neural Engine via ane-compiler.
Same interface as CPUDrafter in q3_spec_decode_server.py.
Zero GPU contention — independent silicon, independent bandwidth.

Usage:
    drafter = ANEDrafter()
    drafter.prefill(prompt_tokens)
    drafter.feed(first_token)

    draft_token = drafter.draft_one()  # ~27ms on ANE
    drafter.feed(accepted_token)
    drafter.correct(n_reject, correction_token)

Integration with spec decode server:
    During the 70B GPU verify cycle (~94ms), the ANE drafter
    generates ~3 draft tokens. These are free to verify due to
    the batch verification plateau (K=10-32 flat cost).

Copyright 2026 Nick Lo. MIT License.
"""

import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))


class ANEDrafter:
    """Llama-3.2-1B on ANE for speculative decoding.

    Same interface as CPUDrafter:
        prefill(tokens) — initialize with prompt
        draft_one() — generate one token
        feed(token) — accept token into KV cache
        correct(n_reject, correction) — rollback and correct
    """

    def __init__(self, model_path=None, build_dir='/tmp/llama_1b_ane_unfused'):
        """Load Llama-1B and compile for ANE dispatch (unfused SwiGLU).

        Uses the unfused SwiGLU path: 3 separate gate/up/down dispatches
        with CPU SiLU. 14x faster than the fused NeuralNetworkBuilder path.
        88 total ops (5/layer * 16 layers + 8 lm_head chunks).
        """
        from llama_loader import LlamaModel, compile_llama_unfused
        from generate import ANEDispatcher

        if model_path is None:
            model_path = os.path.expanduser(
                "~/.cache/huggingface/hub/models--unsloth--Llama-3.2-1B-Instruct/"
                "snapshots/5a8abab4a5d6f164389b1079fb721cfab8d7126c/model.safetensors"
            )

        t0 = time.time()
        self.model = LlamaModel.from_safetensors(model_path)
        self.config = self.model.config
        t_load = time.time() - t0

        t0 = time.time()
        self.compiled = compile_llama_unfused(self.model, build_dir)
        t_compile = time.time() - t0

        # Filter out metadata keys (e.g. _lm_head_chunks has 4-element tuples)
        dispatch_ops = {k: v for k, v in self.compiled.items()
                        if not k.startswith('_') and len(v) == 3}

        t0 = time.time()
        self.dispatcher = ANEDispatcher(dispatch_ops, quiet=True)
        self.dispatcher.start()
        t_dispatch = time.time() - t0

        # KV cache
        from kv_cache import KVCache
        self.kv_cache = KVCache(
            self.config.n_layers, self.config.n_kv_heads, self.config.head_dim)

        self.n_tokens = 0  # current sequence length
        self._last_logits = None

        print(f"ANE Drafter ready: load={t_load:.1f}s compile={t_compile:.1f}s "
              f"dispatch={t_dispatch:.1f}s ({len(dispatch_ops)} ops, unfused SwiGLU)")

    def _forward_one(self, token_id: int, position: int) -> np.ndarray:
        """Forward pass for a single token through all layers (unfused SwiGLU)."""
        from llama_loader import (forward_layer_decode_unfused,
                                   rms_norm_cpu, lm_head_dispatch)

        # Embedding: token lookup (no position embedding — Llama uses RoPE)
        x = self.model.embed_tokens[token_id].astype(np.float16)

        for layer_i in range(self.config.n_layers):
            x = forward_layer_decode_unfused(
                layer_i, x, self.model, self.dispatcher, self.kv_cache,
                position=position)

        # Final RMSNorm + lm_head (chunked)
        x = rms_norm_cpu(x, self.model.norm_weight, self.config.rms_norm_eps)
        logits = lm_head_dispatch(x, self.compiled, self.dispatcher)

        return logits

    def prefill(self, tokens):
        """Process prompt tokens to populate KV cache."""
        self.kv_cache.reset()
        self.n_tokens = 0

        for pos, tok in enumerate(tokens):
            logits = self._forward_one(tok, pos)
            self.n_tokens += 1

        self._last_logits = logits

    def draft_one(self) -> int:
        """Generate one draft token from current state."""
        if self._last_logits is None:
            raise RuntimeError("Must call prefill() or feed() before draft_one()")

        # Greedy sample from last logits
        token = int(np.argmax(self._last_logits.astype(np.float32)))

        # Forward the drafted token to advance the state
        logits = self._forward_one(token, self.n_tokens)
        self.n_tokens += 1
        self._last_logits = logits

        return token

    def feed(self, token: int):
        """Accept an externally-generated token (from verifier)."""
        logits = self._forward_one(token, self.n_tokens)
        self.n_tokens += 1
        self._last_logits = logits

    def correct(self, n_reject: int, correction: int):
        """Rollback n_reject tokens and feed correction."""
        if n_reject > 0:
            # Trim KV cache
            new_len = self.n_tokens - n_reject
            for layer_i in range(self.config.n_layers):
                self.kv_cache.k_cache[layer_i] = \
                    self.kv_cache.k_cache[layer_i][:new_len]
                self.kv_cache.v_cache[layer_i] = \
                    self.kv_cache.v_cache[layer_i][:new_len]
            self.n_tokens = new_len

        self.feed(correction)

    def stop(self):
        """Release ANE resources."""
        if self.dispatcher:
            self.dispatcher.stop()
            self.dispatcher = None


def benchmark():
    """Standalone benchmark of ANE drafter speed."""
    print("ANE Drafter Benchmark")
    print("=" * 50)

    drafter = ANEDrafter()

    # Use Llama tokenizer for a test prompt
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('unsloth/Llama-3.2-1B-Instruct')

    prompt = "The capital of France is"
    tokens = tokenizer.encode(prompt, add_special_tokens=False)

    print(f"\nPrefilling {len(tokens)} tokens...")
    t0 = time.time()
    drafter.prefill(tokens)
    print(f"Prefill: {time.time()-t0:.2f}s")

    # Draft tokens and measure speed
    print(f"\nDrafting 20 tokens...")
    draft_tokens = []
    t0 = time.time()
    for i in range(20):
        tok = drafter.draft_one()
        draft_tokens.append(tok)
    elapsed = time.time() - t0

    text = tokenizer.decode(draft_tokens)
    tps = len(draft_tokens) / elapsed
    per_tok = elapsed / len(draft_tokens) * 1000

    print(f"Generated: {text}")
    print(f"Speed: {tps:.1f} tok/s ({per_tok:.1f}ms/tok)")
    print(f"Dispatches/token: {len(drafter.compiled)}")
    print(f"GPU cost: 0%")

    drafter.stop()


if __name__ == '__main__':
    benchmark()
