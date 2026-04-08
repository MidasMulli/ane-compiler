#!/usr/bin/env python3
"""
Fused Llama-3.2-1B generation — C/Accelerate CPU ops + ANE via ct.predict.

Replaces all numpy CPU ops (RMSNorm, RoPE, GQA attention, softmax, embedding,
argmax) with ctypes calls to libllama_cpu_ops.dylib (Accelerate framework).
ANE dispatch stays in Python via ct.predict (MIL IR 2-input fusion models).

This is the hybrid approach: C for CPU hot path, Python for ANE dispatch.
Analogous to how tests/ane_generate.m works for GPT-2, but adapted for
Llama's MIL IR 2-input fusion models that require ct.predict.

Build the C library first:
  cd ane-compiler
  xcrun clang -O2 -shared -framework Accelerate -fobjc-arc \
    -o libllama_cpu_ops.dylib llama_cpu_ops.c

Usage:
  python run_llama_fused_c.py --prompt "The capital of France is" --tokens 10
  python run_llama_fused_c.py --compare  # also run pure-Python for A/B

Copyright 2026 Nick Lo. MIT License.
"""

import os
import sys
import time
import ctypes
import argparse
import numpy as np
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from llama_loader import LlamaModel, LlamaConfig
from kv_cache import KVCache

# ===================================================================
# C library bindings
# ===================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LIB_PATH = os.path.join(SCRIPT_DIR, 'libllama_cpu_ops.dylib')

BUILD_DIR = '/tmp/llama_mil_fused'

MODEL_PATH = os.path.expanduser(
    "~/.cache/huggingface/hub/models--unsloth--Llama-3.2-1B-Instruct/"
    "snapshots/5a8abab4a5d6f164389b1079fb721cfab8d7126c/"
)


def load_c_lib():
    """Load the Accelerate-backed C shared library."""
    if not os.path.exists(LIB_PATH):
        print(f"ERROR: {LIB_PATH} not found. Build it first:")
        print(f"  xcrun clang -O2 -shared -framework Accelerate -fobjc-arc "
              f"-o libllama_cpu_ops.dylib llama_cpu_ops.c")
        sys.exit(1)

    lib = ctypes.CDLL(LIB_PATH)

    # llama_rms_norm(x_fp16, weight_fp32, out_fp16, dim, eps)
    lib.llama_rms_norm.argtypes = [
        ctypes.c_void_p,  # x_fp16
        ctypes.c_void_p,  # weight_fp32
        ctypes.c_void_p,  # out_fp16
        ctypes.c_int,     # dim
        ctypes.c_float,   # eps
    ]
    lib.llama_rms_norm.restype = None

    # llama_rope(q_fp16, k_fp16, q_out, k_out, n_q, n_kv, head_dim, pos, theta)
    lib.llama_rope.argtypes = [
        ctypes.c_void_p,  # q_fp16
        ctypes.c_void_p,  # k_fp16
        ctypes.c_void_p,  # q_out_fp16
        ctypes.c_void_p,  # k_out_fp16
        ctypes.c_int,     # n_q_heads
        ctypes.c_int,     # n_kv_heads
        ctypes.c_int,     # head_dim
        ctypes.c_int,     # position
        ctypes.c_double,  # theta
    ]
    lib.llama_rope.restype = None

    # llama_gqa_attention(q, cached_k, cached_v, out, n_heads, n_kv, hd, seq)
    lib.llama_gqa_attention.argtypes = [
        ctypes.c_void_p,  # q_fp16
        ctypes.c_void_p,  # cached_k_fp16
        ctypes.c_void_p,  # cached_v_fp16
        ctypes.c_void_p,  # out_fp16
        ctypes.c_int,     # n_heads
        ctypes.c_int,     # n_kv_heads
        ctypes.c_int,     # head_dim
        ctypes.c_int,     # seq_len
    ]
    lib.llama_gqa_attention.restype = None

    # llama_softmax(x, n) — used internally, exposed for testing
    lib.llama_softmax.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.llama_softmax.restype = None

    # llama_embedding(embed_table, token_id, out_fp16, dim)
    lib.llama_embedding.argtypes = [
        ctypes.c_void_p,  # embed_table (fp32)
        ctypes.c_int,     # token_id
        ctypes.c_void_p,  # out_fp16
        ctypes.c_int,     # dim
    ]
    lib.llama_embedding.restype = None

    # llama_argmax(logits_fp32, vocab_size) -> int
    lib.llama_argmax.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.llama_argmax.restype = ctypes.c_int

    # llama_fp16_to_fp32(in, out, n)
    lib.llama_fp16_to_fp32.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
    lib.llama_fp16_to_fp32.restype = None

    # llama_fp32_to_fp16(in, out, n)
    lib.llama_fp32_to_fp16.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
    lib.llama_fp32_to_fp16.restype = None

    # llama_concat_logits_fp16_to_fp32(chunk_fp16, logits_fp32, offset, chunk_size)
    lib.llama_concat_logits_fp16_to_fp32.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    lib.llama_concat_logits_fp16_to_fp32.restype = None

    return lib


# ===================================================================
# C-backed CPU operations (thin Python wrappers)
# ===================================================================

class CCPUOps:
    """Wrapper around C library for Llama CPU operations."""

    def __init__(self, lib, config: LlamaConfig):
        self.lib = lib
        self.config = config
        self.dim = config.hidden_size
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim
        self.n_rep = config.n_rep

        # Pre-allocate reusable buffers
        self._rms_out = np.empty(self.dim, dtype=np.float16)
        self._q_rope_out = np.empty(self.n_heads * self.head_dim, dtype=np.float16)
        self._k_rope_out = np.empty(self.n_kv_heads * self.head_dim, dtype=np.float16)
        self._attn_out = np.empty(self.dim, dtype=np.float16)
        self._embed_out = np.empty(self.dim, dtype=np.float16)
        self._logits_f32 = np.empty(config.vocab_size, dtype=np.float32)

    def rms_norm(self, x_fp16: np.ndarray, weight_fp32: np.ndarray,
                 eps: float) -> np.ndarray:
        """RMSNorm via C/Accelerate. Returns FP16."""
        self.lib.llama_rms_norm(
            x_fp16.ctypes.data,
            weight_fp32.ctypes.data,
            self._rms_out.ctypes.data,
            self.dim,
            ctypes.c_float(eps),
        )
        return self._rms_out

    def rope(self, q_fp16: np.ndarray, k_fp16: np.ndarray,
             position: int):
        """RoPE via C/Accelerate. Returns (q_rotated, k_rotated) FP16."""
        # q is [n_heads, head_dim], k is [n_kv_heads, head_dim] — contiguous
        q_flat = np.ascontiguousarray(q_fp16.ravel())
        k_flat = np.ascontiguousarray(k_fp16.ravel())

        self.lib.llama_rope(
            q_flat.ctypes.data,
            k_flat.ctypes.data,
            self._q_rope_out.ctypes.data,
            self._k_rope_out.ctypes.data,
            self.n_heads,
            self.n_kv_heads,
            self.head_dim,
            position,
            ctypes.c_double(self.config.rope_theta),
        )
        return (self._q_rope_out.reshape(self.n_heads, self.head_dim),
                self._k_rope_out.reshape(self.n_kv_heads, self.head_dim))

    def gqa_attention(self, q_fp16: np.ndarray,
                      cached_k_fp16: np.ndarray,
                      cached_v_fp16: np.ndarray) -> np.ndarray:
        """GQA attention via C/Accelerate. Returns FP16 [dim]."""
        seq_len = cached_k_fp16.shape[0]
        # Ensure contiguous
        q_flat = np.ascontiguousarray(q_fp16.ravel())
        k_flat = np.ascontiguousarray(cached_k_fp16.ravel())
        v_flat = np.ascontiguousarray(cached_v_fp16.ravel())

        self.lib.llama_gqa_attention(
            q_flat.ctypes.data,
            k_flat.ctypes.data,
            v_flat.ctypes.data,
            self._attn_out.ctypes.data,
            self.n_heads,
            self.n_kv_heads,
            self.head_dim,
            seq_len,
        )
        return self._attn_out

    def embedding(self, embed_table_fp32: np.ndarray,
                  token_id: int) -> np.ndarray:
        """Embedding lookup via C. Returns FP16 [dim]."""
        self.lib.llama_embedding(
            embed_table_fp32.ctypes.data,
            token_id,
            self._embed_out.ctypes.data,
            self.dim,
        )
        return self._embed_out.copy()  # copy because caller may store it

    def argmax(self, logits_fp32: np.ndarray) -> int:
        """Argmax via C/vDSP_maxvi."""
        return self.lib.llama_argmax(
            logits_fp32.ctypes.data,
            len(logits_fp32),
        )


# ===================================================================
# Build fused models (reuse from run_llama_fused.py)
# ===================================================================

def build_all_models(model: LlamaModel):
    """Build all MIL IR models. Reuses run_llama_fused.py logic."""
    # Import the build functions from run_llama_fused
    from run_llama_fused import build_all_models as _build_all
    return _build_all(model)


# ===================================================================
# C-backed generation loop
# ===================================================================

def generate_fused_c(model: LlamaModel, ct_models: dict, dispatch_mode: str,
                     prompt_tokens: list, max_new_tokens: int, lib):
    """Generation loop: C CPU ops + ANE via ct.predict.

    Identical logic to generate_fused() in run_llama_fused.py but all
    numpy CPU ops replaced with C/Accelerate calls.
    """
    config = model.config
    dim = config.hidden_size
    n_heads = config.n_heads
    n_kv_heads = config.n_kv_heads
    head_dim = config.head_dim
    n_rep = config.n_rep

    # C-backed CPU ops
    cops = CCPUOps(lib, config)

    # Pre-convert embedding table to contiguous FP32
    embed_fp32 = np.ascontiguousarray(model.embed_tokens.astype(np.float32))

    # Pre-convert layer norm weights to contiguous FP32
    layer_rms1_w = []
    layer_rms2_w = []
    for li in range(config.n_layers):
        layer_rms1_w.append(
            np.ascontiguousarray(model.layers[li].input_layernorm_weight.astype(np.float32)))
        layer_rms2_w.append(
            np.ascontiguousarray(model.layers[li].post_attention_layernorm_weight.astype(np.float32)))
    final_rms_w = np.ascontiguousarray(model.norm_weight.astype(np.float32))

    # KV cache — use FP16 numpy arrays matching kv_cache.py layout
    kv = KVCache(config.n_layers, n_kv_heads, head_dim)

    # Logits buffer (reusable)
    logits_f32 = np.empty(config.vocab_size, dtype=np.float32)

    generated = list(prompt_tokens)

    def forward_token(token_id, pos):
        # Embedding (C: FP32 lookup → FP16)
        x_fp16 = cops.embedding(embed_fp32, token_id)

        for li in range(config.n_layers):
            L = model.layers[li]

            # Pre-attention: RMSNorm + QKV
            if f'L{li}_pre' in ct_models:
                # Fused RMSNorm + QKV on ANE
                pre_result = ct_models[f'L{li}_pre'].predict({
                    'x': x_fp16.reshape(1, dim, 1, 1).astype(np.float32)
                })
                qkv = list(pre_result.values())[0].flatten().astype(np.float16)
            else:
                # RMSNorm on C, QKV on ANE
                ln1 = cops.rms_norm(x_fp16, layer_rms1_w[li], config.rms_norm_eps)
                qkv_result = ct_models[f'L{li}_qkv'].predict({
                    'x': ln1.reshape(1, dim, 1, 1).astype(np.float32)
                })
                qkv = list(qkv_result.values())[0].flatten().astype(np.float16)

            # Split QKV
            q = qkv[:dim]
            k = qkv[dim:dim + n_kv_heads * head_dim]
            v = qkv[dim + n_kv_heads * head_dim:]

            q_heads = q.reshape(n_heads, head_dim)
            k_heads = k.reshape(n_kv_heads, head_dim)
            v_heads = v.reshape(n_kv_heads, head_dim)

            # RoPE (C/Accelerate)
            q_heads, k_heads = cops.rope(q_heads, k_heads, pos)

            # KV cache append
            kv.append(li, k_heads[np.newaxis], v_heads[np.newaxis])

            # GQA attention (C/Accelerate)
            cached_k, cached_v = kv.get(li)
            attn_out = cops.gqa_attention(q_heads, cached_k, cached_v)

            # Fused post-attention (ANE via ct.predict, 2 inputs)
            post_result = ct_models[f'L{li}_post'].predict({
                'attn_out': attn_out.reshape(1, dim, 1, 1).astype(np.float32),
                'x': x_fp16.reshape(1, dim, 1, 1).astype(np.float32),
            })
            x_fp16 = list(post_result.values())[0].flatten().astype(np.float16)

        # Final RMSNorm (C/Accelerate)
        x_norm = cops.rms_norm(x_fp16, final_rms_w, config.rms_norm_eps)

        # lm_head (chunked ANE via ct.predict)
        n_chunks = config.vocab_size // 16032
        if config.vocab_size % 16032 != 0:
            n_chunks += 1

        offset = 0
        for j in range(n_chunks):
            lm_result = ct_models[f'lm_head_{j}'].predict({
                'x': x_norm.reshape(1, dim, 1, 1).astype(np.float32)
            })
            chunk_vals = list(lm_result.values())[0].flatten()
            chunk_size = len(chunk_vals)
            # Direct copy into logits buffer (already FP32 from ct.predict)
            logits_f32[offset:offset + chunk_size] = chunk_vals.astype(np.float32)
            offset += chunk_size

        # Argmax (C/vDSP)
        return cops.argmax(logits_f32)

    # Prefill
    for pos, tok in enumerate(prompt_tokens[:-1]):
        forward_token(tok, pos)

    # First generated token
    next_tok = forward_token(prompt_tokens[-1], len(prompt_tokens) - 1)
    generated.append(next_tok)

    # Generation loop (timed)
    t_start = time.perf_counter()
    for _ in range(max_new_tokens - 1):
        pos = len(generated) - 1
        next_tok = forward_token(next_tok, pos)
        generated.append(next_tok)
    gen_time = time.perf_counter() - t_start

    return generated, gen_time


# ===================================================================
# Main
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Fused Llama-3.2-1B — C/Accelerate CPU + ANE ct.predict')
    parser.add_argument('--prompt', default='The capital of France is',
                        help='Input prompt')
    parser.add_argument('--tokens', type=int, default=10,
                        help='Number of tokens to generate')
    parser.add_argument('--compare', action='store_true',
                        help='Also run pure-Python for A/B comparison')
    parser.add_argument('--skip-build', action='store_true',
                        help='Skip building (use cached models)')
    args = parser.parse_args()

    print("=" * 70)
    print("FUSED LLAMA-3.2-1B — C/ACCELERATE CPU + ANE ct.predict")
    print("RMSNorm/RoPE/GQA/softmax/argmax via Accelerate framework")
    print("=" * 70)

    # Load C library
    print("\n[0/5] Loading C library...")
    lib = load_c_lib()
    print(f"  Loaded: {LIB_PATH}")

    # Load model
    print("\n[1/5] Loading Llama-3.2-1B...")
    t0 = time.time()
    model_path = MODEL_PATH
    if not os.path.exists(model_path):
        snap_dir = os.path.expanduser(
            "~/.cache/huggingface/hub/models--unsloth--Llama-3.2-1B-Instruct/snapshots/")
        if os.path.exists(snap_dir):
            snap = os.listdir(snap_dir)[0]
            model_path = os.path.join(snap_dir, snap)
        else:
            print("ERROR: Model not found")
            sys.exit(1)

    model = LlamaModel.from_safetensors(model_path)
    print(f"  Loaded in {time.time()-t0:.1f}s")
    print(f"  Config: {model.config.hidden_size}h, {model.config.n_layers}L, "
          f"{model.config.n_heads}Q/{model.config.n_kv_heads}KV heads")

    # Build fused models
    print("\n[2/5] Building fused MIL IR models...")
    t0 = time.time()
    ct_models, dispatch_mode = build_all_models(model)
    build_time = time.time() - t0
    print(f"  Built in {build_time:.1f}s")

    # Tokenize
    print("\n[3/5] Tokenizing...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('unsloth/Llama-3.2-1B-Instruct')
    prompt_tokens = tokenizer.encode(args.prompt, add_special_tokens=False)
    print(f"  Prompt: '{args.prompt}'")
    print(f"  Tokens: {prompt_tokens}")

    # PyTorch reference
    print("\n[4/5] PyTorch reference...")
    pt_tokens = None
    try:
        import torch
        from transformers import AutoModelForCausalLM
        pt_model = AutoModelForCausalLM.from_pretrained(
            'unsloth/Llama-3.2-1B-Instruct',
            torch_dtype=torch.float32,
        )
        pt_model.eval()
        input_ids = torch.tensor([prompt_tokens])
        with torch.no_grad():
            output = pt_model.generate(input_ids, max_new_tokens=args.tokens,
                                       do_sample=False)
        pt_tokens = output[0].tolist()
        pt_text = tokenizer.decode(pt_tokens)
        print(f"  PyTorch: {pt_tokens}")
        print(f"  Text:    \"{pt_text}\"")
    except Exception as e:
        print(f"  PyTorch reference failed: {e}")

    # C-backed generation
    print(f"\n[5/5] C-backed fused generation ({args.tokens} tokens)...")
    t0 = time.time()
    c_tokens, gen_time = generate_fused_c(
        model, ct_models, dispatch_mode, prompt_tokens,
        max_new_tokens=args.tokens, lib=lib)
    total_time = time.time() - t0
    n_gen = len(c_tokens) - len(prompt_tokens)

    c_text = tokenizer.decode(c_tokens)
    print(f"  Tokens: {c_tokens}")
    print(f"  Text:   \"{c_text}\"")
    print(f"  Decode time: {gen_time:.3f}s ({n_gen - 1} tokens after first)")
    if gen_time > 0:
        tps_c = (n_gen - 1) / gen_time
        print(f"  Decode tok/s: {tps_c:.1f}")

    # Kill test vs PyTorch
    if pt_tokens:
        match = c_tokens == pt_tokens
        n_match = sum(1 for a, b in zip(c_tokens, pt_tokens) if a == b)
        print(f"\n  Kill test vs PyTorch: "
              f"{'PASS' if match else f'PARTIAL {n_match}/{len(pt_tokens)}'}")
        if not match:
            for i in range(min(len(c_tokens), len(pt_tokens))):
                if i < len(c_tokens) and c_tokens[i] != pt_tokens[i]:
                    ct_ = tokenizer.decode([c_tokens[i]])
                    pt_ = tokenizer.decode([pt_tokens[i]])
                    print(f"    pos {i}: c={c_tokens[i]} \"{ct_}\" "
                          f"vs pt={pt_tokens[i]} \"{pt_}\"")

    # A/B comparison with pure-Python
    if args.compare:
        print(f"\n{'=' * 70}")
        print("PURE-PYTHON BASELINE (numpy CPU ops)")
        print(f"{'=' * 70}")
        from run_llama_fused import generate_fused as generate_python
        py_tokens, py_time = generate_python(
            model, ct_models, dispatch_mode, prompt_tokens,
            max_new_tokens=args.tokens)
        py_text = tokenizer.decode(py_tokens)
        print(f"  Tokens: {py_tokens}")
        print(f"  Text:   \"{py_text}\"")
        if py_time > 0:
            tps_py = (n_gen - 1) / py_time
            print(f"  Decode tok/s: {tps_py:.1f}")

        # Cross-check: C vs Python
        match_cp = c_tokens == py_tokens
        print(f"\n  C vs Python match: "
              f"{'PASS' if match_cp else 'MISMATCH'}")

        if gen_time > 0 and py_time > 0:
            speedup = tps_c / tps_py
            print(f"\n  Speedup: {speedup:.2f}x ({tps_c:.1f} vs {tps_py:.1f} tok/s)")

    # Summary
    print(f"\n{'=' * 70}")
    print("RESULTS")
    print(f"{'=' * 70}")

    n_dispatches = sum(1 for k in ct_models if k.startswith('L'))
    n_dispatches += sum(1 for k in ct_models if k.startswith('lm_head'))

    if gen_time > 0:
        tps = (n_gen - 1) / gen_time
        print(f"  Pipeline:  C/Accelerate CPU + ANE ct.predict ({n_dispatches}d)")
        print(f"  Decode:    {tps:.1f} tok/s")
        print(f"  CPU ops:   RMSNorm, RoPE, GQA attention (all via Accelerate)")
        print(f"  ANE ops:   fused post-attn, QKV/pre-attn, lm_head chunks")

    if pt_tokens:
        match = c_tokens == pt_tokens
        n_match = sum(1 for a, b in zip(c_tokens, pt_tokens) if a == b)
        total_check = min(len(c_tokens), len(pt_tokens))
        print(f"  Kill test: {n_match}/{total_check} "
              f"{'PASS' if match else 'PARTIAL'}")


if __name__ == "__main__":
    main()
