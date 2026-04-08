#!/usr/bin/env python3
"""
Combined stack benchmark: C CPU ops + cross-layer fusion.

4 configs, 5 runs each, per-component profiling, kill test.

Configs:
  A) Python/numpy + 40 dispatches (baseline)
  B) C/Accelerate + 40 dispatches
  C) Python/numpy + 25 dispatches (cross-layer fusion)
  D) C/Accelerate + 25 dispatches (combined)

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
sys.path.insert(0, os.path.dirname(__file__))

from llama_loader import (
    LlamaModel, LlamaConfig, LlamaLayer,
    rms_norm_cpu, rope_cpu, softmax_cpu,
)
from kv_cache import KVCache

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LIB_PATH = os.path.join(SCRIPT_DIR, 'libllama_cpu_ops.dylib')

MODEL_PATH = os.path.expanduser(
    "~/.cache/huggingface/hub/models--unsloth--Llama-3.2-1B-Instruct/"
    "snapshots/5a8abab4a5d6f164389b1079fb721cfab8d7126c/"
)

N_RUNS = 5
N_TOKENS = 10
PROMPT = "The capital of France is"
PROFILE_TOKENS = 50


# ===================================================================
# C library loader (from run_llama_fused_c.py)
# ===================================================================

def load_c_lib():
    lib = ctypes.CDLL(LIB_PATH)
    lib.llama_rms_norm.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_float]
    lib.llama_rms_norm.restype = None
    lib.llama_rope.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_double]
    lib.llama_rope.restype = None
    lib.llama_gqa_attention.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
    lib.llama_gqa_attention.restype = None
    lib.llama_softmax.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.llama_softmax.restype = None
    lib.llama_embedding.argtypes = [
        ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int]
    lib.llama_embedding.restype = None
    lib.llama_argmax.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.llama_argmax.restype = ctypes.c_int
    lib.llama_fp16_to_fp32.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
    lib.llama_fp16_to_fp32.restype = None
    lib.llama_fp32_to_fp16.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
    lib.llama_fp32_to_fp16.restype = None
    return lib


# ===================================================================
# CPU ops: Python/numpy vs C/Accelerate
# ===================================================================

class PythonCPUOps:
    """numpy-based CPU ops (reference)."""
    def __init__(self, config):
        self.config = config
        self.dim = config.hidden_size

    def rms_norm(self, x_fp16, weight_fp32, eps):
        return rms_norm_cpu(x_fp16, weight_fp32, eps)

    def rope(self, q_heads, k_heads, pos):
        return rope_cpu(q_heads, k_heads, pos, self.config.head_dim,
                        self.config.rope_theta)

    def gqa_attention(self, q_heads, cached_k, cached_v):
        config = self.config
        scale = np.float32(1.0 / np.sqrt(config.head_dim))
        attn_output = np.zeros(self.dim, dtype=np.float32)
        for h in range(config.n_heads):
            kv_h = h // config.n_rep
            q_h = q_heads[h].astype(np.float32)
            k_h = cached_k[:, kv_h, :].astype(np.float32)
            v_h = cached_v[:, kv_h, :].astype(np.float32)
            scores = (q_h @ k_h.T) * scale
            weights = softmax_cpu(scores)
            attn_output[h * config.head_dim:(h + 1) * config.head_dim] = weights @ v_h
        return attn_output.astype(np.float16)

    def embedding(self, embed_fp32, token_id):
        return embed_fp32[token_id].astype(np.float16)

    def argmax(self, logits_fp32):
        return int(np.argmax(logits_fp32))


class CCPUOps:
    """C/Accelerate-based CPU ops."""
    def __init__(self, lib, config):
        self.lib = lib
        self.config = config
        self.dim = config.hidden_size
        self._rms_out = np.empty(self.dim, dtype=np.float16)
        self._q_rope_out = np.empty(config.n_heads * config.head_dim, dtype=np.float16)
        self._k_rope_out = np.empty(config.n_kv_heads * config.head_dim, dtype=np.float16)
        self._attn_out = np.empty(self.dim, dtype=np.float16)
        self._embed_out = np.empty(self.dim, dtype=np.float16)

    def rms_norm(self, x_fp16, weight_fp32, eps):
        self.lib.llama_rms_norm(
            x_fp16.ctypes.data, weight_fp32.ctypes.data,
            self._rms_out.ctypes.data, self.dim, ctypes.c_float(eps))
        return self._rms_out

    def rope(self, q_heads, k_heads, pos):
        q_flat = np.ascontiguousarray(q_heads.ravel())
        k_flat = np.ascontiguousarray(k_heads.ravel())
        self.lib.llama_rope(
            q_flat.ctypes.data, k_flat.ctypes.data,
            self._q_rope_out.ctypes.data, self._k_rope_out.ctypes.data,
            self.config.n_heads, self.config.n_kv_heads, self.config.head_dim,
            pos, ctypes.c_double(self.config.rope_theta))
        return (self._q_rope_out.reshape(self.config.n_heads, self.config.head_dim),
                self._k_rope_out.reshape(self.config.n_kv_heads, self.config.head_dim))

    def gqa_attention(self, q_heads, cached_k, cached_v):
        seq_len = cached_k.shape[0]
        q_flat = np.ascontiguousarray(q_heads.ravel())
        k_flat = np.ascontiguousarray(cached_k.ravel())
        v_flat = np.ascontiguousarray(cached_v.ravel())
        self.lib.llama_gqa_attention(
            q_flat.ctypes.data, k_flat.ctypes.data, v_flat.ctypes.data,
            self._attn_out.ctypes.data,
            self.config.n_heads, self.config.n_kv_heads,
            self.config.head_dim, seq_len)
        return self._attn_out

    def embedding(self, embed_fp32, token_id):
        self.lib.llama_embedding(
            embed_fp32.ctypes.data, token_id,
            self._embed_out.ctypes.data, self.dim)
        return self._embed_out.copy()

    def argmax(self, logits_fp32):
        return self.lib.llama_argmax(logits_fp32.ctypes.data, len(logits_fp32))


# ===================================================================
# Generation engines
# ===================================================================

def gen_40d(model, ct_models_40d, dispatch_mode, ops, prompt_tokens,
            max_new_tokens, profile=False):
    """40-dispatch generation. ops = PythonCPUOps or CCPUOps."""
    config = model.config
    dim = config.hidden_size
    n_heads = config.n_heads
    n_kv_heads = config.n_kv_heads
    head_dim = config.head_dim
    n_rep = config.n_rep

    embed_fp32 = np.ascontiguousarray(model.embed_tokens.astype(np.float32))
    layer_rms1_w = [np.ascontiguousarray(model.layers[i].input_layernorm_weight.astype(np.float32))
                    for i in range(config.n_layers)]
    final_rms_w = np.ascontiguousarray(model.norm_weight.astype(np.float32))

    kv = KVCache(config.n_layers, n_kv_heads, head_dim)
    generated = list(prompt_tokens)
    logits_f32 = np.empty(config.vocab_size, dtype=np.float32)

    cpu_times = []
    ane_times = []
    lm_times = []

    def forward_token(token_id, pos, do_profile=False):
        t_cpu = 0.0
        t_ane = 0.0
        t_lm = 0.0

        tc = time.perf_counter()
        x_fp16 = ops.embedding(embed_fp32, token_id)
        if do_profile:
            t_cpu += time.perf_counter() - tc

        for li in range(config.n_layers):
            L = model.layers[li]

            if f'L{li}_pre' in ct_models_40d:
                ta = time.perf_counter()
                pre_result = ct_models_40d[f'L{li}_pre'].predict({
                    'x': x_fp16.reshape(1, dim, 1, 1).astype(np.float32)})
                qkv = list(pre_result.values())[0].flatten().astype(np.float16)
                if do_profile:
                    t_ane += time.perf_counter() - ta
            else:
                tc = time.perf_counter()
                ln1 = ops.rms_norm(x_fp16, layer_rms1_w[li], config.rms_norm_eps)
                if do_profile:
                    t_cpu += time.perf_counter() - tc
                ta = time.perf_counter()
                qkv_result = ct_models_40d[f'L{li}_qkv'].predict({
                    'x': ln1.reshape(1, dim, 1, 1).astype(np.float32)})
                qkv = list(qkv_result.values())[0].flatten().astype(np.float16)
                if do_profile:
                    t_ane += time.perf_counter() - ta

            tc = time.perf_counter()
            q = qkv[:dim].reshape(n_heads, head_dim)
            k = qkv[dim:dim + n_kv_heads * head_dim].reshape(n_kv_heads, head_dim)
            v = qkv[dim + n_kv_heads * head_dim:].reshape(n_kv_heads, head_dim)
            q, k = ops.rope(q, k, pos)
            kv.append(li, k[np.newaxis], v[np.newaxis])
            cached_k, cached_v = kv.get(li)
            attn_out = ops.gqa_attention(q, cached_k, cached_v)
            if do_profile:
                t_cpu += time.perf_counter() - tc

            ta = time.perf_counter()
            post_result = ct_models_40d[f'L{li}_post'].predict({
                'attn_out': attn_out.reshape(1, dim, 1, 1).astype(np.float32),
                'x': x_fp16.reshape(1, dim, 1, 1).astype(np.float32),
            })
            x_fp16 = list(post_result.values())[0].flatten().astype(np.float16)
            if do_profile:
                t_ane += time.perf_counter() - ta

        tc = time.perf_counter()
        x_norm = ops.rms_norm(x_fp16, final_rms_w, config.rms_norm_eps)
        if do_profile:
            t_cpu += time.perf_counter() - tc

        tl = time.perf_counter()
        n_chunks = config.vocab_size // 16032
        if config.vocab_size % 16032 != 0:
            n_chunks += 1
        offset = 0
        for j in range(n_chunks):
            lm_result = ct_models_40d[f'lm_head_{j}'].predict({
                'x': x_norm.reshape(1, dim, 1, 1).astype(np.float32)})
            chunk_vals = list(lm_result.values())[0].flatten()
            chunk_size = len(chunk_vals)
            logits_f32[offset:offset + chunk_size] = chunk_vals.astype(np.float32)
            offset += chunk_size
        if do_profile:
            t_lm = time.perf_counter() - tl

        tc2 = time.perf_counter()
        tok = ops.argmax(logits_f32)
        if do_profile:
            t_cpu += time.perf_counter() - tc2

        if do_profile:
            cpu_times.append(t_cpu)
            ane_times.append(t_ane)
            lm_times.append(t_lm)

        return tok

    # Prefill
    for pos, tok in enumerate(prompt_tokens[:-1]):
        forward_token(tok, pos)

    next_tok = forward_token(prompt_tokens[-1], len(prompt_tokens) - 1)
    generated.append(next_tok)

    t_start = time.perf_counter()
    for _ in range(max_new_tokens - 1):
        pos = len(generated) - 1
        next_tok = forward_token(next_tok, pos, do_profile=profile)
        generated.append(next_tok)
    gen_time = time.perf_counter() - t_start

    profile_data = None
    if profile and cpu_times:
        profile_data = {
            'cpu_ms': np.median(cpu_times) * 1000,
            'ane_ms': np.median(ane_times) * 1000,
            'lm_ms': np.median(lm_times) * 1000,
        }

    return generated, gen_time, profile_data


def gen_25d(model, ct_models_25d, cross_keys, ops, prompt_tokens,
            max_new_tokens, profile=False):
    """25-dispatch cross-layer generation. ops = PythonCPUOps or CCPUOps."""
    config = model.config
    dim = config.hidden_size
    n_heads = config.n_heads
    n_kv_heads = config.n_kv_heads
    head_dim = config.head_dim
    n_rep = config.n_rep
    n_layers = config.n_layers

    embed_fp32 = np.ascontiguousarray(model.embed_tokens.astype(np.float32))
    final_rms_w = np.ascontiguousarray(model.norm_weight.astype(np.float32))

    kv = KVCache(n_layers, n_kv_heads, head_dim)
    generated = list(prompt_tokens)
    logits_f32 = np.empty(config.vocab_size, dtype=np.float32)

    cpu_times = []
    ane_times = []
    lm_times = []

    def do_attention_cpu(qkv_flat, layer_idx, pos):
        q = qkv_flat[:dim].reshape(n_heads, head_dim)
        k = qkv_flat[dim:dim + n_kv_heads * head_dim].reshape(n_kv_heads, head_dim)
        v = qkv_flat[dim + n_kv_heads * head_dim:].reshape(n_kv_heads, head_dim)
        q, k = ops.rope(q, k, pos)
        kv.append(layer_idx, k[np.newaxis], v[np.newaxis])
        cached_k, cached_v = kv.get(layer_idx)
        attn_out = ops.gqa_attention(q, cached_k, cached_v)
        return attn_out

    def forward_token(token_id, pos, do_profile=False):
        t_cpu = 0.0
        t_ane = 0.0
        t_lm = 0.0

        tc = time.perf_counter()
        x_fp16 = ops.embedding(embed_fp32, token_id)
        if do_profile:
            t_cpu += time.perf_counter() - tc

        # Layer 0: standalone pre_attn
        ta = time.perf_counter()
        pre_result = ct_models_25d['pre_attn_0'].predict({
            'x': x_fp16.reshape(1, dim, 1, 1).astype(np.float32)})
        qkv_0 = list(pre_result.values())[0].flatten().astype(np.float16)
        if do_profile:
            t_ane += time.perf_counter() - ta

        # CPU attention layer 0
        tc = time.perf_counter()
        attn_out = do_attention_cpu(qkv_0, 0, pos)
        if do_profile:
            t_cpu += time.perf_counter() - tc

        x_residual = x_fp16

        # Cross-layer dispatches: 0->1, 1->2, ..., 14->15
        for i in range(n_layers - 1):
            key = f'cross_{i}_{i+1}'
            ta = time.perf_counter()
            result = ct_models_25d[key].predict({
                'attn_out': attn_out.reshape(1, dim, 1, 1).astype(np.float32),
                'x': x_residual.reshape(1, dim, 1, 1).astype(np.float32),
            })
            # Parse 2 outputs by shape
            out_arrays = {}
            for k in result:
                arr = result[k].flatten()
                out_arrays[len(arr)] = arr
            qkv_next = out_arrays[dim + 2 * (n_kv_heads * head_dim)].astype(np.float16)
            x_next = out_arrays[dim].astype(np.float16)
            if do_profile:
                t_ane += time.perf_counter() - ta

            tc = time.perf_counter()
            attn_out = do_attention_cpu(qkv_next, i + 1, pos)
            if do_profile:
                t_cpu += time.perf_counter() - tc

            x_residual = x_next

        # Layer 15: standalone post_attn
        last = n_layers - 1
        ta = time.perf_counter()
        post_result = ct_models_25d[f'post_attn_{last}'].predict({
            'attn_out': attn_out.reshape(1, dim, 1, 1).astype(np.float32),
            'x': x_residual.reshape(1, dim, 1, 1).astype(np.float32),
        })
        x_fp16 = list(post_result.values())[0].flatten().astype(np.float16)
        if do_profile:
            t_ane += time.perf_counter() - ta

        tc = time.perf_counter()
        x_norm = ops.rms_norm(x_fp16, final_rms_w, config.rms_norm_eps)
        if do_profile:
            t_cpu += time.perf_counter() - tc

        tl = time.perf_counter()
        n_chunks = config.vocab_size // 16032
        if config.vocab_size % 16032 != 0:
            n_chunks += 1
        offset = 0
        for j in range(n_chunks):
            lm_result = ct_models_25d[f'lm_head_{j}'].predict({
                'x': x_norm.reshape(1, dim, 1, 1).astype(np.float32)})
            chunk_vals = list(lm_result.values())[0].flatten()
            chunk_size = len(chunk_vals)
            logits_f32[offset:offset + chunk_size] = chunk_vals.astype(np.float32)
            offset += chunk_size
        if do_profile:
            t_lm = time.perf_counter() - tl

        tc2 = time.perf_counter()
        tok = ops.argmax(logits_f32)
        if do_profile:
            t_cpu += time.perf_counter() - tc2

        if do_profile:
            cpu_times.append(t_cpu)
            ane_times.append(t_ane)
            lm_times.append(t_lm)

        return tok

    # Prefill
    for pos, tok in enumerate(prompt_tokens[:-1]):
        forward_token(tok, pos)

    next_tok = forward_token(prompt_tokens[-1], len(prompt_tokens) - 1)
    generated.append(next_tok)

    t_start = time.perf_counter()
    for _ in range(max_new_tokens - 1):
        pos = len(generated) - 1
        next_tok = forward_token(next_tok, pos, do_profile=profile)
        generated.append(next_tok)
    gen_time = time.perf_counter() - t_start

    profile_data = None
    if profile and cpu_times:
        profile_data = {
            'cpu_ms': np.median(cpu_times) * 1000,
            'ane_ms': np.median(ane_times) * 1000,
            'lm_ms': np.median(lm_times) * 1000,
        }

    return generated, gen_time, profile_data


# ===================================================================
# Benchmark runner
# ===================================================================

def run_config(name, gen_fn, n_runs, prompt_tokens, n_tokens):
    """Run a config n_runs times, return list of tok/s values."""
    results = []
    for r in range(n_runs):
        tokens, gen_time, _ = gen_fn(prompt_tokens, n_tokens, profile=False)
        tps = (n_tokens - 1) / gen_time if gen_time > 0 else 0
        results.append(tps)
        print(f"  {name} run {r+1}/{n_runs}: {tps:.1f} tok/s")
    return results, tokens


def run_profile(name, gen_fn, prompt_tokens, profile_tokens):
    """Run once with profiling for per-component timing."""
    tokens, gen_time, prof = gen_fn(prompt_tokens, profile_tokens, profile=True)
    return prof


def main():
    print("=" * 74)
    print("COMBINED STACK BENCHMARK — C CPU ops + Cross-layer fusion")
    print(f"Prompt: '{PROMPT}' | {N_TOKENS} tokens | {N_RUNS} runs each")
    print("=" * 74)

    # Load model
    print("\n[LOAD] Model...")
    t0 = time.time()
    model_path = MODEL_PATH
    if not os.path.exists(model_path):
        snap_dir = os.path.expanduser(
            "~/.cache/huggingface/hub/models--unsloth--Llama-3.2-1B-Instruct/snapshots/")
        if os.path.exists(snap_dir):
            snap = os.listdir(snap_dir)[0]
            model_path = os.path.join(snap_dir, snap)
        else:
            print("ERROR: Model not found"); sys.exit(1)
    model = LlamaModel.from_safetensors(model_path)
    print(f"  Loaded in {time.time()-t0:.1f}s")

    # Tokenize
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('unsloth/Llama-3.2-1B-Instruct')
    prompt_tokens = tokenizer.encode(PROMPT, add_special_tokens=False)
    print(f"  Prompt tokens: {prompt_tokens}")

    # PyTorch reference
    print("\n[REF] PyTorch reference...")
    import torch
    from transformers import AutoModelForCausalLM
    pt_model = AutoModelForCausalLM.from_pretrained(
        'unsloth/Llama-3.2-1B-Instruct', torch_dtype=torch.float32)
    pt_model.eval()
    with torch.no_grad():
        output = pt_model.generate(torch.tensor([prompt_tokens]),
                                   max_new_tokens=N_TOKENS, do_sample=False)
    pt_tokens = output[0].tolist()
    print(f"  PyTorch: {tokenizer.decode(pt_tokens)}")
    del pt_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Load C library
    print("\n[LOAD] C library...")
    lib = load_c_lib()

    # Build 40-dispatch models
    print("\n[BUILD] 40-dispatch MIL IR models...")
    from run_llama_fused import build_all_models as build_40d_models
    ct_models_40d, dispatch_mode_40d = build_40d_models(model)

    # Build 25-dispatch cross-layer models
    print("\n[BUILD] 25-dispatch cross-layer models...")
    from bench_cross_layer_fusion import build_all_models as build_25d_models
    ct_models_25d, cross_keys = build_25d_models(model)

    # Create ops instances
    py_ops = PythonCPUOps(model.config)
    c_ops = CCPUOps(lib, model.config)

    # Wrap generation functions for uniform interface
    def make_40d_fn(ops_inst):
        def fn(prompt_tokens, n_tokens, profile=False):
            return gen_40d(model, ct_models_40d, dispatch_mode_40d, ops_inst,
                          prompt_tokens, n_tokens, profile)
        return fn

    def make_25d_fn(ops_inst):
        def fn(prompt_tokens, n_tokens, profile=False):
            return gen_25d(model, ct_models_25d, cross_keys, ops_inst,
                          prompt_tokens, n_tokens, profile)
        return fn

    configs = [
        ("Python/40d", make_40d_fn(py_ops), 40),
        ("C ops/40d", make_40d_fn(c_ops), 40),
        ("Python/25d", make_25d_fn(py_ops), 25),
        ("Combined/25d", make_25d_fn(c_ops), 25),
    ]

    # ─── Step 1: 5 runs each ───
    print("\n" + "=" * 74)
    print("STEP 1: Throughput (5 runs each)")
    print("=" * 74)

    all_results = {}
    all_tokens = {}
    for name, gen_fn, dispatches in configs:
        print(f"\n  --- {name} ({dispatches}d) ---")
        results, tokens = run_config(name, gen_fn, N_RUNS, prompt_tokens, N_TOKENS)
        median_tps = sorted(results)[N_RUNS // 2]
        all_results[name] = {
            'runs': results,
            'median': median_tps,
            'dispatches': dispatches,
        }
        all_tokens[name] = tokens
        print(f"  Median: {median_tps:.1f} tok/s")

    # ─── Step 3: Per-component profiling ───
    print("\n" + "=" * 74)
    print(f"STEP 3: Per-component profiling ({PROFILE_TOKENS} tokens)")
    print("=" * 74)

    for name, gen_fn, dispatches in configs:
        print(f"\n  --- {name} ---")
        prof = run_profile(name, gen_fn, prompt_tokens, PROFILE_TOKENS)
        if prof:
            all_results[name]['cpu_ms'] = prof['cpu_ms']
            all_results[name]['ane_ms'] = prof['ane_ms']
            all_results[name]['lm_ms'] = prof['lm_ms']
            print(f"  CPU: {prof['cpu_ms']:.2f} ms  ANE: {prof['ane_ms']:.2f} ms  "
                  f"lm_head: {prof['lm_ms']:.2f} ms")

    # ─── Step 4: Kill test ───
    print("\n" + "=" * 74)
    print("STEP 4: Kill test (all configs vs PyTorch)")
    print("=" * 74)

    all_kill = {}
    for name, gen_fn, dispatches in configs:
        gen_toks = all_tokens[name]
        match = gen_toks == pt_tokens
        n_match = sum(1 for a, b in zip(gen_toks, pt_tokens) if a == b)
        total = min(len(gen_toks), len(pt_tokens))
        status = f"{n_match}/{total} PASS" if match else f"{n_match}/{total} FAIL"
        all_kill[name] = status
        print(f"  {name}: {status}")
        if not match:
            for i in range(total):
                if gen_toks[i] != pt_tokens[i]:
                    print(f"    pos {i}: got {gen_toks[i]} expected {pt_tokens[i]}")

    # ─── Deliverable: Results table ───
    print("\n" + "=" * 74)
    print("RESULTS TABLE")
    print("=" * 74)
    print(f"{'Config':<16} | {'Disp':>4} | {'Median tok/s':>12} | "
          f"{'CPU ms':>7} | {'ANE ms':>7} | {'lm_head ms':>10} | {'Kill test'}")
    print("-" * 74)
    for name, gen_fn, dispatches in configs:
        r = all_results[name]
        cpu = f"{r.get('cpu_ms', 0):.2f}" if 'cpu_ms' in r else "—"
        ane = f"{r.get('ane_ms', 0):.2f}" if 'ane_ms' in r else "—"
        lm = f"{r.get('lm_ms', 0):.2f}" if 'lm_ms' in r else "—"
        kt = all_kill.get(name, "—")
        print(f"{name:<16} | {r['dispatches']:>4} | {r['median']:>12.1f} | "
              f"{cpu:>7} | {ane:>7} | {lm:>10} | {kt}")
    print("-" * 74)

    # Raw runs
    print("\nRaw tok/s (all runs):")
    for name, gen_fn, dispatches in configs:
        runs = all_results[name]['runs']
        runs_str = ", ".join(f"{r:.1f}" for r in runs)
        print(f"  {name}: [{runs_str}]")


if __name__ == "__main__":
    main()
