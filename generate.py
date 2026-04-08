#!/usr/bin/env python3
"""
Autoregressive text generation for GPT-2 117M on Apple Neural Engine.

Architecture:
  - ANE: Q/K/V/O projections, FFN up/down, lm_head (via ane_standalone_pipe)
  - CPU: LayerNorm, GELU, attention (Q@K^T, softmax, attn@V), residuals
  - KV cache: CPU numpy arrays, appended per token

Uses ane_standalone_pipe in dispatch-loop mode: all 85 models compiled
and loaded ONCE, then dispatched repeatedly via "D <idx>" commands.
This eliminates the subprocess-per-dispatch overhead (was 0.42 tok/s).

Kill test: 10 tokens from "The" must match PyTorch greedy generation exactly.

Copyright 2026 Nick Lo. MIT License.
"""

import os
import sys
import time
import subprocess
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from model_loader import GPT2Model, GPT2Config
from first_token import (
    layernorm_cpu, gelu_new_cpu,
    compile_all_ops, MODEL_PATH, BUILD_DIR,
)
from kv_cache import KVCache

PIPE_TOOL = os.path.join(os.path.dirname(__file__), 'tests', 'ane_standalone_pipe')


# ===================================================================
# Softmax (CPU, FP32 for numerical stability)
# ===================================================================

def softmax_cpu(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax in FP32."""
    x_f32 = x.astype(np.float32)
    x_f32 = x_f32 - x_f32.max()
    exp_x = np.exp(x_f32)
    return exp_x / exp_x.sum()


# ===================================================================
# ANE Dispatch Server (persistent pipe tool)
# ===================================================================

class ANEDispatcher:
    """Persistent ANE dispatch server using ane_standalone_pipe.

    Protocol:
      1. Write manifest listing all (mlmodelc_path, in_ch, out_ch) tuples
      2. Launch pipe tool, wait for READY_FOR_SWAP
      3. Send GO, wait for DISPATCH_READY (all models loaded + IOSurfaces allocated)
      4. Send "D <idx>\n" + in_ch*2 bytes input, read out_ch*2 bytes output
      5. Send "Q\n" to quit
    """

    def __init__(self, compiled: dict, quiet: bool = False,
                 hwx_overrides: dict = None):
        """Initialize dispatcher with compiled model dict.

        Args:
            compiled: dict mapping op_name -> (mlmodelc_path, in_ch, out_ch)
            quiet: suppress status messages
            hwx_overrides: dict mapping op_name -> hwx_bytes. When provided,
                swaps the aned-compiled .hwx with emitted .hwx at READY_FOR_SWAP.
                This makes the emitter the compiler — aned only loads, never compiles
                the production weights.
        """
        self.compiled = compiled
        self.proc = None
        self.op_names = []  # ordered list of op names
        self.op_index = {}  # op_name -> manifest index
        self.op_info = {}   # op_name -> (in_ch, out_ch)
        self.quiet = quiet
        self.hwx_overrides = hwx_overrides or {}

    def start(self):
        """Write manifest, launch pipe tool, wait for ready."""
        self.op_names = sorted(self.compiled.keys())
        manifest_path = '/tmp/gpt2_generate_manifest.txt'

        with open(manifest_path, 'w') as f:
            for i, name in enumerate(self.op_names):
                path, in_ch, out_ch = self.compiled[name]
                f.write(f"{path} {in_ch} {out_ch}\n")
                self.op_index[name] = i
                self.op_info[name] = (in_ch, out_ch)

        if not self.quiet:
            print(f"  Manifest: {len(self.op_names)} models")

        self.proc = subprocess.Popen(
            [PIPE_TOOL, manifest_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Wait for READY_FOR_SWAP, collecting cache paths
        cache_paths = {}
        while True:
            line = self.proc.stdout.readline().decode().strip()
            if line == 'READY_FOR_SWAP':
                break
            if line.startswith('CACHE:'):
                # Format: CACHE:<idx>:<path>
                parts = line.split(':', 2)
                idx = int(parts[1])
                cache_paths[idx] = parts[2]
            if self.proc.poll() is not None:
                err = self.proc.stderr.read().decode()
                raise RuntimeError(f"Pipe tool exited during compile: {err}")

        # Swap emitted .hwx into cache
        swapped = 0
        if self.hwx_overrides:
            import tempfile
            for name, hwx_bytes in self.hwx_overrides.items():
                idx = self.op_index.get(name)
                if idx is not None and idx in cache_paths:
                    cache_hwx = cache_paths[idx]
                    with tempfile.NamedTemporaryFile(suffix='.hwx', delete=False) as tf:
                        tf.write(hwx_bytes)
                        tmp_path = tf.name
                    subprocess.run(['sudo', '-n', 'cp', tmp_path, cache_hwx],
                                   capture_output=True, check=True)
                    os.unlink(tmp_path)
                    swapped += 1

            # loadModel in the same process uses in-memory compiled state,
            # ignoring the cache swap. Kill the pipe tool + aned, then relaunch.
            # The new pipe tool creates fresh model handles → compileModel checks
            # cache by .mlmodelc hash → finds our swapped .hwx → loads it.
            try:
                self.proc.stdin.write(b"Q\n")
                self.proc.stdin.flush()
                self.proc.wait(timeout=3)
            except Exception:
                self.proc.kill()
                try: self.proc.wait(timeout=2)
                except: pass

            subprocess.run(['sudo', '-n', 'killall', 'aned'],
                           capture_output=True)
            time.sleep(1.5)

            if not self.quiet:
                print(f"  Swapped {swapped}/{len(self.hwx_overrides)} .hwx + killall aned")

            # Relaunch pipe tool: new process, new model handles, new aned connection.
            # compileModel checks cache — finds entry with our swapped .hwx.
            self.proc = subprocess.Popen(
                [PIPE_TOOL, manifest_path],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            while True:
                line = self.proc.stdout.readline().decode().strip()
                if line == 'READY_FOR_SWAP':
                    break
                if self.proc.poll() is not None:
                    err = self.proc.stderr.read().decode()
                    raise RuntimeError(f"Pipe tool relaunch failed: {err}")

        # Send GO to enter dispatch phase
        self.proc.stdin.write(b"GO\n")
        self.proc.stdin.flush()

        while True:
            line = self.proc.stdout.readline().decode().strip()
            if line == 'DISPATCH_READY':
                break
            if self.proc.poll() is not None:
                err = self.proc.stderr.read().decode()
                raise RuntimeError(f"Pipe tool exited during load: {err}")

        if not self.quiet:
            src = "emitter" if swapped > 0 else "aned"
            print(f"  Ready: {len(self.op_names)} models loaded ({src})")

    def dispatch(self, op_name: str, input_fp16: np.ndarray) -> np.ndarray:
        """Dispatch a single op on ANE.

        Args:
            op_name: operation name (e.g. 'L0_q_proj', 'lm_head')
            input_fp16: input data as FP16 numpy array

        Returns:
            Output as FP16 numpy array
        """
        idx = self.op_index[op_name]
        in_ch, out_ch = self.op_info[op_name]

        # Send dispatch command
        cmd = f"D {idx}\n".encode()
        self.proc.stdin.write(cmd)
        self.proc.stdin.write(input_fp16.astype(np.float16).tobytes())
        self.proc.stdin.flush()

        # Read output
        out_bytes = self.proc.stdout.read(out_ch * 2)
        if len(out_bytes) != out_ch * 2:
            raise RuntimeError(f"Short read for {op_name}: got {len(out_bytes)}, expected {out_ch * 2}")
        return np.frombuffer(out_bytes, dtype=np.float16).copy()

    def stop(self):
        """Send quit command and wait for process to exit."""
        if self.proc and self.proc.poll() is None:
            try:
                self.proc.stdin.write(b"Q\n")
                self.proc.stdin.flush()
                self.proc.wait(timeout=10)
            except Exception:
                self.proc.kill()
        self.proc = None


# ===================================================================
# Decode forward pass (single token with KV cache)
# ===================================================================

def forward_layer_decode(layer_idx: int, x: np.ndarray, model: GPT2Model,
                         dispatcher: ANEDispatcher, kv_cache: KVCache,
                         mode: str = 'fused', trace=None) -> np.ndarray:
    """Forward pass through one GPT-2 layer with KV cache.

    Modes:
      'fused'   — QKV combined (1), O (1), fused FFN with ANE GELU (1) = 3 dispatches
      'exact'   — QKV combined (1), O (1), fc_up (1) + CPU GELU + fc_down (1) = 4 dispatches
      'unfused' — Q+K+V (3), O (1), fc_up (1) + CPU GELU + fc_down (1) = 6 dispatches

    Args:
        trace: optional TraceLogger. When provided, records per-op timing and
               device placement. Zero-cost when None.
    """
    L = model.layers[layer_idx]
    dim = model.config.n_embd
    n_heads = model.config.n_head
    head_dim = model.config.head_dim
    pfx = f'L{layer_idx}'
    fuse_qkv = mode in ('fused', 'exact')
    fuse_ffn = mode == 'fused'
    _trace = trace is not None and trace.enabled
    layer_ops = [] if _trace else None

    # 1. LayerNorm 1 (CPU)
    ln1_out = layernorm_cpu(x, L.ln_1_weight, L.ln_1_bias,
                            model.config.layer_norm_epsilon)
    if _trace:
        layer_ops.append(('LN1', 'CPU', None))

    if fuse_qkv:
        if _trace:
            t0 = time.perf_counter()
        qkv = dispatcher.dispatch(f'{pfx}_qkv_proj', ln1_out)
        if _trace:
            layer_ops.append(('QKV', 'ANE', (time.perf_counter() - t0) * 1000))
        q = qkv[:dim]
        k = qkv[dim:2*dim]
        v = qkv[2*dim:]
    else:
        if _trace:
            t0 = time.perf_counter()
        q = dispatcher.dispatch(f'{pfx}_q_proj', ln1_out)
        if _trace:
            layer_ops.append(('Q', 'ANE', (time.perf_counter() - t0) * 1000))
            t0 = time.perf_counter()
        k = dispatcher.dispatch(f'{pfx}_k_proj', ln1_out)
        if _trace:
            layer_ops.append(('K', 'ANE', (time.perf_counter() - t0) * 1000))
            t0 = time.perf_counter()
        v = dispatcher.dispatch(f'{pfx}_v_proj', ln1_out)
        if _trace:
            layer_ops.append(('V', 'ANE', (time.perf_counter() - t0) * 1000))

    # 3. Reshape to multi-head
    q_heads = q.reshape(n_heads, head_dim)
    k_heads = k.reshape(n_heads, head_dim)
    v_heads = v.reshape(n_heads, head_dim)

    # 4. KV cache append
    kv_cache.append(layer_idx, k_heads[np.newaxis], v_heads[np.newaxis])

    # 5. Attention on CPU
    if _trace:
        t0 = time.perf_counter()
    cached_k, cached_v = kv_cache.get(layer_idx)
    scale = np.float32(1.0 / np.sqrt(head_dim))
    attn_output = np.zeros(dim, dtype=np.float32)
    for h in range(n_heads):
        q_h = q_heads[h].astype(np.float32)
        k_h = cached_k[:, h, :].astype(np.float32)
        v_h = cached_v[:, h, :].astype(np.float32)
        scores = (q_h @ k_h.T) * scale
        weights = softmax_cpu(scores)
        attn_output[h * head_dim:(h + 1) * head_dim] = weights @ v_h
    attn_output = attn_output.astype(np.float16)
    if _trace:
        layer_ops.append((f'attn({n_heads}h)', 'CPU', (time.perf_counter() - t0) * 1000))

    # 6. O projection on ANE
    if _trace:
        t0 = time.perf_counter()
    o_out = dispatcher.dispatch(f'{pfx}_o_proj', attn_output)
    if _trace:
        layer_ops.append(('O', 'ANE', (time.perf_counter() - t0) * 1000))

    # 7. Residual 1
    r1 = (x.astype(np.float32) + o_out.astype(np.float32)).astype(np.float16)
    if _trace:
        layer_ops.append(('res', 'CPU', None))

    # 8. LayerNorm 2 (CPU)
    ln2_out = layernorm_cpu(r1, L.ln_2_weight, L.ln_2_bias,
                            model.config.layer_norm_epsilon)
    if _trace:
        layer_ops.append(('LN2', 'CPU', None))

    # 9. FFN
    if fuse_ffn:
        if _trace:
            t0 = time.perf_counter()
        ffn_out = dispatcher.dispatch(f'{pfx}_fused_ffn', ln2_out)
        if _trace:
            layer_ops.append(('FFN', 'ANE', (time.perf_counter() - t0) * 1000))
    else:
        if _trace:
            t0 = time.perf_counter()
        fc_up = dispatcher.dispatch(f'{pfx}_fc_up', ln2_out)
        if _trace:
            layer_ops.append(('fc_up', 'ANE', (time.perf_counter() - t0) * 1000))
        gelu_out = gelu_new_cpu(fc_up)
        if _trace:
            layer_ops.append(('GELU', 'CPU', None))
            t0 = time.perf_counter()
        ffn_out = dispatcher.dispatch(f'{pfx}_fc_down', gelu_out)
        if _trace:
            layer_ops.append(('fc_down', 'ANE', (time.perf_counter() - t0) * 1000))

    # 10. Residual 2
    output = (r1.astype(np.float32) + ffn_out.astype(np.float32)).astype(np.float16)
    if _trace:
        layer_ops.append(('res', 'CPU', None))
        trace.trace_layer(layer_idx, layer_ops)

    return output


# ===================================================================
# Embedding + LM head
# ===================================================================

def embed(model: GPT2Model, token_id: int, position: int) -> np.ndarray:
    """Token + position embedding for GPT-2."""
    tok_emb = model.wte[token_id].astype(np.float32)
    pos_emb = model.wpe[position].astype(np.float32)
    return (tok_emb + pos_emb).astype(np.float16)


def lm_head(x: np.ndarray, model: GPT2Model, dispatcher: ANEDispatcher) -> np.ndarray:
    """Final layernorm + logit projection."""
    x = layernorm_cpu(x, model.ln_f_weight, model.ln_f_bias,
                      model.config.layer_norm_epsilon)
    logits = dispatcher.dispatch('lm_head', x)
    return logits


# ===================================================================
# Generation loop
# ===================================================================

def generate(model: GPT2Model, dispatcher: ANEDispatcher, prompt_tokens: list,
             max_new_tokens: int = 10, mode: str = 'fused',
             trace=None, tokenizer=None) -> list:
    """Autoregressive generation with KV cache and persistent ANE dispatch.

    Args:
        mode: 'fused' (ANE GELU, 37 dispatches), 'exact' (CPU GELU, 49),
              'unfused' (no fusion, 73)
        trace: optional TraceLogger for verbose output
        tokenizer: optional tokenizer for trace token decoding
    """
    config = model.config
    kv_cache = KVCache(config.n_layer, config.n_head, config.head_dim)
    generated = list(prompt_tokens)
    _trace = trace is not None and trace.enabled

    def _tok_str(tid):
        if tokenizer is not None:
            return tokenizer.decode([tid])
        return str(tid)

    # Prefill (all but last prompt token without trace)
    for pos, token_id in enumerate(prompt_tokens[:-1]):
        x = embed(model, token_id, pos)
        for layer_i in range(config.n_layer):
            x = forward_layer_decode(layer_i, x, model, dispatcher, kv_cache,
                                     mode=mode)

    # Last prompt token with trace (produces first generated token)
    last_prompt = prompt_tokens[-1]
    last_pos = len(prompt_tokens) - 1
    if _trace:
        trace.trace_token_start(1, last_prompt, _tok_str(last_prompt))
    x = embed(model, last_prompt, last_pos)
    for layer_i in range(config.n_layer):
        x = forward_layer_decode(layer_i, x, model, dispatcher, kv_cache,
                                 mode=mode, trace=trace if _trace else None)

    logits = lm_head(x, model, dispatcher)
    next_token = int(np.argmax(logits.astype(np.float32)))
    generated.append(next_token)

    if _trace:
        trace.trace_token_end(1, next_token, _tok_str(next_token))

    for step in range(max_new_tokens - 1):
        token_idx = step + 2
        if _trace:
            trace.trace_token_start(token_idx, next_token, _tok_str(next_token))

        pos = len(generated) - 1
        x = embed(model, next_token, pos)
        for layer_i in range(config.n_layer):
            x = forward_layer_decode(layer_i, x, model, dispatcher, kv_cache,
                                     mode=mode, trace=trace if _trace else None)
        logits = lm_head(x, model, dispatcher)
        next_token = int(np.argmax(logits.astype(np.float32)))
        generated.append(next_token)

        if _trace:
            trace.trace_token_end(token_idx, next_token, _tok_str(next_token))

    return generated


# ===================================================================
# Benchmark at different sequence lengths
# ===================================================================

def benchmark_seq_lens(model: GPT2Model, dispatcher: ANEDispatcher,
                       tokenizer, seq_lens: list):
    """Benchmark generation at different prompt lengths."""
    print("\n" + "=" * 60)
    print("BENCHMARK: tok/s at different prompt lengths")
    print("=" * 60)

    # Use a long prompt we can slice
    long_prompt = ("The capital of France is Paris. The capital of Germany is "
                   "Berlin. The capital of Japan is Tokyo. The capital of Italy "
                   "is Rome. The capital of Spain is Madrid. The capital of "
                   "China is Beijing. The capital of Russia is Moscow. The "
                   "capital of India is New Delhi. The capital of Brazil is "
                   "Brasilia. The capital of Australia is Canberra.")
    long_tokens = tokenizer.encode(long_prompt)

    max_new = 5  # generate 5 tokens for each test
    results = []

    for seq_len in seq_lens:
        if seq_len > len(long_tokens):
            print(f"\n  seq_len={seq_len}: SKIP (prompt too short, have {len(long_tokens)} tokens)")
            continue

        prompt_tokens = long_tokens[:seq_len]
        prompt_text = tokenizer.decode(prompt_tokens)

        # Warm up KV cache fresh
        config = model.config
        kv_cache = KVCache(config.n_layer, config.n_head, config.head_dim)

        # Time the full generation
        t0 = time.time()

        # Prefill
        for pos, token_id in enumerate(prompt_tokens):
            x = embed(model, token_id, pos)
            for layer_i in range(config.n_layer):
                x = forward_layer_decode(layer_i, x, model, dispatcher, kv_cache)
        prefill_time = time.time() - t0

        # Decode
        logits = lm_head(x, model, dispatcher)
        next_token = int(np.argmax(logits.astype(np.float32)))
        generated = [next_token]

        t_decode_start = time.time()
        for step in range(max_new - 1):
            pos = seq_len + len(generated) - 1
            x = embed(model, next_token, pos)
            for layer_i in range(config.n_layer):
                x = forward_layer_decode(layer_i, x, model, dispatcher, kv_cache)
            logits = lm_head(x, model, dispatcher)
            next_token = int(np.argmax(logits.astype(np.float32)))
            generated.append(next_token)
        decode_time = time.time() - t_decode_start

        total_time = time.time() - t0
        decode_toks = max_new - 1  # first token is from prefill
        prefill_tps = seq_len / prefill_time if prefill_time > 0 else 0
        decode_tps = decode_toks / decode_time if decode_time > 0 else 0
        overall_tps = max_new / total_time if total_time > 0 else 0

        gen_text = tokenizer.decode(generated)
        results.append((seq_len, prefill_time, decode_time, total_time,
                        prefill_tps, decode_tps, overall_tps))

        print(f"\n  seq_len={seq_len}:")
        print(f"    Prompt: \"{prompt_text[:50]}...\"")
        print(f"    Generated: \"{gen_text}\"")
        print(f"    Prefill: {prefill_time:.3f}s ({prefill_tps:.1f} tok/s for {seq_len} tokens)")
        print(f"    Decode:  {decode_time:.3f}s ({decode_tps:.2f} tok/s for {decode_toks} tokens)")
        print(f"    Overall: {total_time:.3f}s ({overall_tps:.2f} tok/s for {max_new} new tokens)")
        per_token_ms = (decode_time / decode_toks * 1000) if decode_toks > 0 else 0
        print(f"    Per decode token: {per_token_ms:.1f}ms")

    if results:
        print(f"\n{'=' * 60}")
        print(f"{'seq_len':>8} {'prefill':>10} {'decode':>10} {'total':>10} {'decode_tps':>12}")
        print(f"{'':>8} {'(s)':>10} {'(s)':>10} {'(s)':>10} {'(tok/s)':>12}")
        print("-" * 60)
        for sl, pt, dt, tt, pts, dts, ots in results:
            print(f"{sl:>8} {pt:>10.3f} {dt:>10.3f} {tt:>10.3f} {dts:>12.2f}")
        print("=" * 60)


# ===================================================================
# Main: kill test + benchmark
# ===================================================================

def main():
    print("=" * 60)
    print("GPT-2 117M AUTOREGRESSIVE GENERATION ON ANE")
    print("Persistent pipe dispatch (compile once, dispatch many)")
    print("=" * 60)

    # -- Load tokenizer --
    from transformers import GPT2Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('openai-community/gpt2')

    # -- Load model --
    print("\n[1/6] Loading GPT-2 from safetensors...")
    t0 = time.time()
    model = GPT2Model.from_safetensors(MODEL_PATH)
    print(f"  Loaded in {time.time() - t0:.2f}s")

    # -- Compile all ops --
    print("\n[2/6] Compiling .mlmodelc bundles...")
    t0 = time.time()
    compiled = compile_all_ops(model, BUILD_DIR)
    print(f"  Compiled in {time.time() - t0:.2f}s")

    # -- Launch persistent ANE dispatcher --
    print("\n[3/6] Launching ANE dispatch server...")
    t0 = time.time()
    dispatcher = ANEDispatcher(compiled)
    dispatcher.start()
    print(f"  Ready in {time.time() - t0:.2f}s")

    # -- PyTorch reference --
    print("\n[4/6] PyTorch reference generation...")
    import torch
    from transformers import GPT2LMHeadModel
    pt_model = GPT2LMHeadModel.from_pretrained('openai-community/gpt2')
    pt_model.eval()

    prompt = "The"
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    with torch.no_grad():
        output = pt_model.generate(input_ids, max_new_tokens=10, do_sample=False)
    pt_tokens = output[0].tolist()
    pt_text = tokenizer.decode(pt_tokens)
    print(f"  Prompt: \"{prompt}\"")
    print(f"  PyTorch tokens: {pt_tokens}")
    print(f"  PyTorch text:   \"{pt_text}\"")

    # -- ANE generation --
    prompt_tokens = tokenizer.encode(prompt)
    print(f"\n[5/6] ANE generation (prompt tokens: {prompt_tokens})...")
    t0 = time.time()
    ane_tokens = generate(model, dispatcher, prompt_tokens, max_new_tokens=10)
    gen_time = time.time() - t0
    ane_text = tokenizer.decode(ane_tokens)
    print(f"  ANE tokens: {ane_tokens}")
    print(f"  ANE text:   \"{ane_text}\"")
    print(f"  Total time: {gen_time:.2f}s")

    n_generated = len(ane_tokens) - len(prompt_tokens)
    print(f"  Generated {n_generated} tokens in {gen_time:.2f}s")
    if gen_time > 0:
        print(f"  Throughput: {n_generated / gen_time:.2f} tok/s")

    # -- Kill test: all 10 generated tokens must match --
    print(f"\n[6/6] KILL TEST: 10-token match vs PyTorch")
    print(f"  PyTorch: {pt_tokens}")
    print(f"  ANE:     {ane_tokens}")

    all_match = True
    for i in range(len(pt_tokens)):
        if i < len(ane_tokens):
            match = ane_tokens[i] == pt_tokens[i]
            label = "MATCH" if match else "MISMATCH"
            tok_str = tokenizer.decode([pt_tokens[i]])
            print(f"    pos {i:2d}: PT={pt_tokens[i]:6d} ANE={ane_tokens[i]:6d} "
                  f"{label} \"{tok_str}\"")
            if not match:
                all_match = False
        else:
            print(f"    pos {i:2d}: PT={pt_tokens[i]:6d} ANE=MISSING")
            all_match = False

    print(f"\n{'=' * 60}")
    if all_match and len(ane_tokens) >= len(pt_tokens):
        print("*** KILL TEST: PASS -- ALL 10 TOKENS MATCH ***")
        print(f"Generated: \"{ane_text}\"")
    else:
        matches = sum(1 for i in range(min(len(ane_tokens), len(pt_tokens)))
                      if ane_tokens[i] == pt_tokens[i])
        total = len(pt_tokens)
        print(f"*** KILL TEST: {matches}/{total} tokens match ***")
        if matches < total:
            for i in range(min(len(ane_tokens), len(pt_tokens))):
                if ane_tokens[i] != pt_tokens[i]:
                    print(f"First divergence at position {i}: "
                          f"PT={pt_tokens[i]} (\"{tokenizer.decode([pt_tokens[i]])}\") "
                          f"ANE={ane_tokens[i]} (\"{tokenizer.decode([ane_tokens[i]])}\")")
                    break

    # -- Benchmark: tok/s --
    dispatches_per_token = 7 * 12 + 1  # 7 ANE ops/layer * 12 layers + lm_head
    print(f"\n{'=' * 60}")
    print("PERFORMANCE SUMMARY")
    print(f"  Total generation: {gen_time:.2f}s for {n_generated} tokens")
    if gen_time > 0:
        print(f"  Average: {n_generated / gen_time:.2f} tok/s")
        per_token = gen_time / n_generated
        print(f"  Per token: {per_token * 1000:.1f}ms")
        print(f"  ANE dispatches per token: {dispatches_per_token}")
        print(f"  Dispatch method: persistent pipe (compile once, dispatch many)")
        print(f"  Estimated dispatch overhead: {dispatches_per_token * 0.093:.1f}ms "
              f"(at 93us floor)")
    print("=" * 60)

    # -- TASK A2: Benchmark at different seq_lens --
    if all_match:
        benchmark_seq_lens(model, dispatcher, tokenizer, [16, 32, 64, 128])

    # Cleanup
    dispatcher.stop()


if __name__ == "__main__":
    main()
