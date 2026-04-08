#!/usr/bin/env python3
"""
8B Q8 ANE Extractor for Subconscious memory extraction.

Loads 72 Q8 CoreML models once, runs extraction via ct.predict.
Zero GPU contention. ~7.9 tok/s decode.

Usage:
    extractor = ANEExtractor8B()
    extractor.load()  # ~350s cold, ~20s warm
    facts = extractor.extract("conversation text here")
    # Returns list of {"text": ..., "type": ..., "entities": [...]}

Copyright 2026 Nick Lo. MIT License.
"""

import os
import sys
import re
import json
import time
import ctypes
import logging
import numpy as np

log = logging.getLogger("ane_extractor_8b")

# Main 25 close: wire the existing libllama_cpu_ops.dylib (built and tested
# in bench_cpu_attention.py / run_llama_fused_c.py / bench_combined_stack.py)
# into the production extractor. R5 measured 78x speedup on the GQA
# attention call (3.36 ms → 0.043 ms at seq=64); projected end-to-end 8B
# prompt encoding from 6.3 → 19.1 tok/s. Correctness 1.5e-5 vs Python
# (FP16 round-trip floor). Contention with 70B verifier +2.7% (noise).
_LIB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "libllama_cpu_ops.dylib")
_C_LIB = None


def _load_c_lib():
    """Lazy-load the GQA C kernel. Returns None if the dylib is missing
    so callers can fall back to the Python loop."""
    global _C_LIB
    if _C_LIB is not None:
        return _C_LIB
    if not os.path.exists(_LIB_PATH):
        log.warning(f"libllama_cpu_ops.dylib not found at {_LIB_PATH}; "
                    f"using slow Python attention fallback")
        return None
    try:
        lib = ctypes.CDLL(_LIB_PATH)
        lib.llama_gqa_attention.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ]
        lib.llama_gqa_attention.restype = None
        _C_LIB = lib
        log.info("loaded libllama_cpu_ops.dylib (C+vDSP attention)")
        return _C_LIB
    except Exception as e:
        log.warning(f"failed to load libllama_cpu_ops.dylib: {e}")
        return None

BUILD_DIR = '/tmp/llama_8b_q8'
MODEL_PATH = os.path.expanduser(
    "~/.cache/huggingface/hub/models--unsloth--Meta-Llama-3.1-8B-Instruct/"
    "snapshots/a2856192dd7c25b842431f39c179a6c2c2f627d1/")

EXTRACTION_PROMPT = """Extract every important fact from this text. Include:
- Measurements and numbers (exactly as stated)
- Decisions ("we decided...", "killed because...", "parked...")
- Architectural insights ("X is actually Y", "the real role is...")
- Relationships ("X feeds Y", "X replaces Y")
- Preferences ("always do X", "never do Y")

Each fact is one complete sentence. Classify each as:
  quantitative, decision, preference, relationship, conceptual, or fact.

Format each line as: [TYPE] fact sentence

TEXT:
---
{text}
---

Extracted facts:
- """


def _llama3_rope_freqs(head_dim, theta, rope_scaling):
    factor = rope_scaling["factor"]
    low_freq_factor = rope_scaling["low_freq_factor"]
    high_freq_factor = rope_scaling["high_freq_factor"]
    orig_max_pos = rope_scaling["original_max_position_embeddings"]
    freqs = 1.0 / (theta ** (np.arange(0, head_dim, 2, dtype=np.float64) / head_dim))
    low_wavelen = orig_max_pos / low_freq_factor
    high_wavelen = orig_max_pos / high_freq_factor
    new = []
    for freq in freqs:
        wl = 2 * np.pi / freq
        if wl < high_wavelen:
            new.append(freq)
        elif wl > low_wavelen:
            new.append(freq / factor)
        else:
            s = (orig_max_pos / wl - low_freq_factor) / (high_freq_factor - low_freq_factor)
            new.append((1 - s) * freq / factor + s * freq)
    return np.array(new, dtype=np.float64)


class ANEExtractor8B:
    def __init__(self, build_dir=BUILD_DIR, model_path=MODEL_PATH):
        self.build_dir = build_dir
        self.model_path = model_path
        self._loaded = False
        self.ct_models = {}

    def load(self):
        if self._loaded:
            return
        import coremltools as ct

        log.info("Loading 8B Q8 extractor (%s)...", self.build_dir)
        t0 = time.time()

        # Config
        cfg = json.load(open(f"{self.model_path}/config.json"))
        self.dim = cfg["hidden_size"]
        self.n_layers = cfg["num_hidden_layers"]
        self.n_heads = cfg["num_attention_heads"]
        self.n_kv = cfg["num_key_value_heads"]
        self.hd = self.dim // self.n_heads
        self.vocab = cfg["vocab_size"]
        self.rope_scaling = cfg.get("rope_scaling")
        self.rope_theta = cfg.get("rope_theta", 500000.0)
        self.rms_eps = cfg.get("rms_norm_eps", 1e-5)
        self.n_rep = self.n_heads // self.n_kv

        # Minimal weights (FP16 to save memory)
        from safetensors.torch import load_file
        s1 = load_file(f"{self.model_path}/model-00001-of-00004.safetensors")
        self.embed = s1["model.embed_tokens.weight"].half().numpy()
        self.norm_w = s1.get("model.norm.weight")
        if self.norm_w is not None:
            self.norm_w = self.norm_w.float().numpy()
        del s1

        s4 = load_file(f"{self.model_path}/model-00004-of-00004.safetensors")
        if self.norm_w is None and "model.norm.weight" in s4:
            self.norm_w = s4["model.norm.weight"].float().numpy()
        del s4

        # Tokenizer
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            'unsloth/Meta-Llama-3.1-8B-Instruct')

        # CoreML models
        for i in range(self.n_layers):
            self.ct_models[f'L{i}_pre'] = ct.models.MLModel(
                f'{self.build_dir}/L{i}_pre_q8.mlpackage',
                compute_units=ct.ComputeUnit.CPU_AND_NE)
            self.ct_models[f'L{i}_post'] = ct.models.MLModel(
                f'{self.build_dir}/L{i}_post_q8.mlpackage',
                compute_units=ct.ComputeUnit.CPU_AND_NE)
        self.n_lm = 0
        while os.path.exists(f'{self.build_dir}/lm_head_{self.n_lm}_q8.mlpackage'):
            self.ct_models[f'lm_head_{self.n_lm}'] = ct.models.MLModel(
                f'{self.build_dir}/lm_head_{self.n_lm}_q8.mlpackage',
                compute_units=ct.ComputeUnit.CPU_AND_NE)
            self.n_lm += 1

        self._loaded = True
        log.info("8B Q8 loaded: %d models in %.0fs", len(self.ct_models), time.time() - t0)

    def _rms_norm(self, x_fp16, w, eps):
        x32 = x_fp16.astype(np.float32)
        ms = np.mean(x32 ** 2)
        return (x32 / np.sqrt(ms + eps) * w.astype(np.float32)).astype(np.float16)

    def _rope(self, q, k, pos):
        hd = self.hd
        half = hd // 2
        if self.rope_scaling and self.rope_scaling.get("rope_type") == "llama3":
            freqs = _llama3_rope_freqs(hd, self.rope_theta, self.rope_scaling)
        else:
            freqs = 1.0 / (self.rope_theta ** (np.arange(0, half, dtype=np.float64) * 2 / hd))
        angles = pos * freqs
        cos_full = np.concatenate([np.cos(angles), np.cos(angles)]).astype(np.float32)
        sin_full = np.concatenate([np.sin(angles), np.sin(angles)]).astype(np.float32)
        def apply(x_heads):
            result = np.empty_like(x_heads)
            for h in range(x_heads.shape[0]):
                x = x_heads[h].astype(np.float32)
                rot = np.concatenate([-x[half:], x[:half]])
                result[h] = (x * cos_full + rot * sin_full).astype(np.float16)
            return result
        return apply(q), apply(k)

    def _gqa_attention(self, q, cached_k, cached_v):
        # Main 25 close: prefer C+vDSP kernel via libllama_cpu_ops.dylib
        # (78x faster than Python loop, R5 measured). Falls back to the
        # Python implementation if the dylib is missing.
        lib = _load_c_lib()
        if lib is not None:
            seq = cached_k.shape[0]
            q_in = np.ascontiguousarray(q.astype(np.float16).ravel())
            k_in = np.ascontiguousarray(cached_k.astype(np.float16).ravel())
            v_in = np.ascontiguousarray(cached_v.astype(np.float16).ravel())
            out = np.empty(self.n_heads * self.hd, dtype=np.float16)
            lib.llama_gqa_attention(
                q_in.ctypes.data, k_in.ctypes.data, v_in.ctypes.data,
                out.ctypes.data,
                self.n_heads, self.n_kv, self.hd, seq,
            )
            return out

        # Fallback: original Python loop
        seq = cached_k.shape[0]
        scale = 1.0 / np.sqrt(float(self.hd))
        out = np.zeros(self.n_heads * self.hd, dtype=np.float32)
        for h in range(self.n_heads):
            kv_h = h // self.n_rep
            q_h = q[h].astype(np.float32)
            scores = np.array([np.dot(q_h, cached_k[s, kv_h].astype(np.float32)) * scale
                              for s in range(seq)])
            scores -= scores.max()
            exp_s = np.exp(scores)
            scores = exp_s / exp_s.sum()
            for s in range(seq):
                out[h*self.hd:(h+1)*self.hd] += scores[s] * cached_v[s, kv_h].astype(np.float32)
        return out.astype(np.float16)

    def _forward_token(self, token_id, pos, kv):
        x = self.embed[token_id].astype(np.float32)
        dim = self.dim
        for li in range(self.n_layers):
            pre_r = list(self.ct_models[f'L{li}_pre'].predict({
                'x': x.reshape(1, dim, 1, 1).astype(np.float32)}).values())[0].flatten()
            q = pre_r[:dim].astype(np.float16).reshape(self.n_heads, self.hd)
            k = pre_r[dim:dim+self.n_kv*self.hd].astype(np.float16).reshape(self.n_kv, self.hd)
            v = pre_r[dim+self.n_kv*self.hd:].astype(np.float16).reshape(self.n_kv, self.hd)
            q, k = self._rope(q, k, pos)
            kv.append(li, k[np.newaxis], v[np.newaxis])
            ck, cv = kv.get(li)
            attn = self._gqa_attention(q, ck, cv)
            post_r = list(self.ct_models[f'L{li}_post'].predict({
                'attn_out': attn.reshape(1, dim, 1, 1).astype(np.float32),
                'x': x.reshape(1, dim, 1, 1).astype(np.float32)}).values())[0].flatten()
            x = post_r.astype(np.float32)
        x_norm = self._rms_norm(x.astype(np.float16), self.norm_w, self.rms_eps)
        logits = np.empty(self.vocab, dtype=np.float32)
        off = 0
        for j in range(self.n_lm):
            ch = list(self.ct_models[f'lm_head_{j}'].predict({
                'x': x_norm.reshape(1, dim, 1, 1).astype(np.float32)}).values())[0].flatten()
            logits[off:off+len(ch)] = ch.astype(np.float32)
            off += len(ch)
        return int(logits.argmax())

    def _forward_hidden(self, token_id, pos, kv):
        """Like _forward_token but returns the normalized hidden state (pre-lm_head)."""
        x = self.embed[token_id].astype(np.float32)
        dim = self.dim
        for li in range(self.n_layers):
            pre_r = list(self.ct_models[f'L{li}_pre'].predict({
                'x': x.reshape(1, dim, 1, 1).astype(np.float32)}).values())[0].flatten()
            q = pre_r[:dim].astype(np.float16).reshape(self.n_heads, self.hd)
            k = pre_r[dim:dim+self.n_kv*self.hd].astype(np.float16).reshape(self.n_kv, self.hd)
            v = pre_r[dim+self.n_kv*self.hd:].astype(np.float16).reshape(self.n_kv, self.hd)
            q, k = self._rope(q, k, pos)
            kv.append(li, k[np.newaxis], v[np.newaxis])
            ck, cv = kv.get(li)
            attn = self._gqa_attention(q, ck, cv)
            post_r = list(self.ct_models[f'L{li}_post'].predict({
                'attn_out': attn.reshape(1, dim, 1, 1).astype(np.float32),
                'x': x.reshape(1, dim, 1, 1).astype(np.float32)}).values())[0].flatten()
            x = post_r.astype(np.float32)
        x_norm = self._rms_norm(x.astype(np.float16), self.norm_w, self.rms_eps)
        return x_norm.astype(np.float32)

    def embed_text(self, text, pooling="mean", max_input_tokens=512):
        """Truncated forward pass: skip lm_head, return pooled hidden state.

        pooling: "mean" (default) or "last".
        Returns 1D numpy array of length self.dim.
        """
        if not self._loaded:
            self.load()

        sys.path.insert(0, os.path.dirname(__file__))
        from kv_cache import KVCache

        tokens = self.tokenizer.encode(text, add_special_tokens=True)
        if len(tokens) > max_input_tokens:
            tokens = tokens[:max_input_tokens]

        kv = KVCache(self.n_layers, self.n_kv, self.hd)
        hiddens = []
        for pos, tok in enumerate(tokens):
            h = self._forward_hidden(tok, pos, kv)
            hiddens.append(h)

        H = np.stack(hiddens, axis=0)  # (seq, dim)
        if pooling == "last":
            vec = H[-1]
        else:
            vec = H.mean(axis=0)
            if not np.isfinite(vec).all() or np.allclose(vec, 0):
                log.warning("mean-pool produced NaN/zero, falling back to last-token")
                vec = H[-1]
        return vec.astype(np.float32), len(tokens)

    def generate(self, prompt_text, max_tokens=400):
        sys.path.insert(0, os.path.dirname(__file__))
        from kv_cache import KVCache

        messages = [{"role": "user", "content": prompt_text}]
        input_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        tokens = self.tokenizer.encode(input_text, add_special_tokens=False)

        kv = KVCache(self.n_layers, self.n_kv, self.hd)
        for pos, tok in enumerate(tokens[:-1]):
            self._forward_token(tok, pos, kv)
        next_tok = self._forward_token(tokens[-1], len(tokens) - 1, kv)
        generated = [next_tok]
        for i in range(max_tokens - 1):
            if next_tok == self.tokenizer.eos_token_id:
                break
            next_tok = self._forward_token(next_tok, len(tokens) + i, kv)
            generated.append(next_tok)
        return self.tokenizer.decode(generated, skip_special_tokens=True)

    def extract(self, text, role="user"):
        """Extract facts from text via 8B on ANE.

        Returns list of dicts: {"text", "type", "entities", "extraction_source"}
        """
        if not self._loaded:
            self.load()

        prompt = EXTRACTION_PROMPT.format(text=text[:2000])
        t0 = time.perf_counter()
        response = self.generate(prompt, max_tokens=400)
        elapsed = time.perf_counter() - t0
        log.info("8B extraction: %.1fs for %d chars", elapsed, len(text))

        # Parse [TYPE] fact lines and plain bullet points
        facts = []
        for line in response.split("\n"):
            line = line.strip()
            # Strip bullet prefix
            for pfx in ["- ", "* ", "• ", "· "]:
                if line.startswith(pfx):
                    line = line[len(pfx):]
                    break
            else:
                m = re.match(r'^\d+[\.\)]\s+', line)
                if m:
                    line = line[m.end():]
                elif line.startswith(("•", "-", "*")):
                    line = line[1:].strip()
                else:
                    if len(line) < 20:
                        continue

            # Extract [TYPE] prefix if present
            fact_type = "general"
            type_match = re.match(r'^\[(\w+)\]\s*', line)
            if type_match:
                t = type_match.group(1).lower()
                if t in ("quantitative", "decision", "preference",
                         "relationship", "conceptual", "fact"):
                    fact_type = t
                line = line[type_match.end():]

            content = line.strip().rstrip(".")
            if len(content) >= 15:
                facts.append({
                    "text": content,
                    "type": fact_type,
                    "entities": [],
                    "extraction_source": "ane_8b",
                })

        return facts


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ext = ANEExtractor8B()
    ext.load()

    test = ("The Llama-1B model achieves 50.2 tok/s on ANE with 25 dispatches. "
            "We decided to park the Living Model experiment because three experiments "
            "showed no adaptation headroom. The user prefers measurement before interpretation.")

    facts = ext.extract(test)
    print(f"\nExtracted {len(facts)} facts:")
    for f in facts:
        print(f"  [{f['type']:>12}] {f['text'][:100]}")
