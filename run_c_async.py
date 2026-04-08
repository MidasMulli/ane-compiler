#!/usr/bin/env python3
"""
GPT-2 async generation benchmark.

Tests three modes:
  serial — original map+eval+unmap per dispatch
  reuse  — IOSurface map once at startup, eval only per dispatch
  async  — IOSurface reuse + split dispatch (start/wait) with overlap

Usage:
  python run_c_async.py --test       # correctness test (10 tokens, all modes)
  python run_c_async.py --bench      # throughput benchmark (50 tokens, all modes)
  python run_c_async.py --mode reuse # single mode

Copyright 2026 Nick Lo. MIT License.
"""

import argparse
import os
import sys
import time
import subprocess
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from model_loader import GPT2Model
from first_token import compile_all_ops, MODEL_PATH

BUILD_DIR = '/tmp/gpt2_c_gen'
WEIGHTS_BIN = '/tmp/gpt2_c_gen/cpu_weights.bin'
MANIFEST = '/tmp/gpt2_c_gen/manifest.txt'
C_BINARY_ASYNC = os.path.join(os.path.dirname(__file__), 'tests', 'ane_generate_async')
C_BINARY_BASELINE = os.path.join(os.path.dirname(__file__), 'tests', 'ane_generate')
C_SOURCE_ASYNC = os.path.join(os.path.dirname(__file__), 'tests', 'ane_generate_async.m')


def build_if_needed():
    if not os.path.exists(C_BINARY_ASYNC) or \
       os.path.getmtime(C_SOURCE_ASYNC) > os.path.getmtime(C_BINARY_ASYNC):
        print("Building async binary...")
        r = subprocess.run([
            'xcrun', 'clang', '-O2',
            '-framework', 'Foundation',
            '-framework', 'IOSurface',
            '-framework', 'Accelerate',
            '-fobjc-arc',
            '-o', C_BINARY_ASYNC,
            C_SOURCE_ASYNC,
        ], capture_output=True, text=True)
        if r.returncode != 0:
            print(f"Build failed:\n{r.stderr}")
            sys.exit(1)
    else:
        print(f"Async binary up to date")


def ensure_data():
    """Make sure manifest + weights exist."""
    if os.path.exists(MANIFEST) and os.path.exists(WEIGHTS_BIN):
        return

    print("Preparing data (one-time)...")
    model = GPT2Model.from_safetensors(MODEL_PATH)
    compiled = compile_all_ops(model, BUILD_DIR, mode='fused')

    # Export CPU weights
    with open(WEIGHTS_BIN, 'wb') as f:
        f.write(model.wte.astype(np.float32).tobytes())
        f.write(model.wpe.astype(np.float32).tobytes())
        for i in range(model.config.n_layer):
            L = model.layers[i]
            f.write(L.ln_1_weight.astype(np.float32).tobytes())
            f.write(L.ln_1_bias.astype(np.float32).tobytes())
            f.write(L.ln_2_weight.astype(np.float32).tobytes())
            f.write(L.ln_2_bias.astype(np.float32).tobytes())
        f.write(model.ln_f_weight.astype(np.float32).tobytes())
        f.write(model.ln_f_bias.astype(np.float32).tobytes())

    # Write manifest
    with open(MANIFEST, 'w') as f:
        for name in sorted(compiled.keys()):
            path, in_ch, out_ch = compiled[name]
            f.write(f"{path} {in_ch} {out_ch} {name}\n")


def run_mode(binary, mode, prompt_tokens, n_tokens, mode_flag=True):
    """Run generation binary in specified mode, return (token_ids, stderr_text)."""
    cmd = [binary, MANIFEST, WEIGHTS_BIN, str(n_tokens)]
    if mode_flag:
        cmd += ['--mode', mode]
    cmd += [str(t) for t in prompt_tokens]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        print(f"  FAILED ({mode}):\n{result.stderr}")
        return None, result.stderr

    token_ids = []
    for line in result.stdout.strip().split('\n'):
        line = line.strip()
        if line and line.lstrip('-').isdigit():
            token_ids.append(int(line))

    return token_ids, result.stderr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true', help='Correctness test')
    parser.add_argument('--bench', action='store_true', help='Throughput benchmark')
    parser.add_argument('--mode', default=None, help='Single mode: serial|reuse|async')
    parser.add_argument('--tokens', type=int, default=None)
    parser.add_argument('--prompt', default='The capital of France is')
    args = parser.parse_args()

    if not args.test and not args.bench and not args.mode:
        args.test = True
        args.bench = True

    build_if_needed()
    ensure_data()

    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained('gpt2')
    except ImportError:
        tokenizer = None

    # ─── Correctness Test ───
    if args.test:
        print("\n" + "=" * 60)
        print("CORRECTNESS TEST: 10 tokens from 'The capital of France is'")
        print("=" * 60)

        prompt = 'The capital of France is'
        if tokenizer:
            prompt_tokens = tokenizer.encode(prompt)
        else:
            prompt_tokens = [464, 3139, 286, 4881, 318]

        # Baseline
        print("\n[baseline] Original serial binary...")
        base_ids, base_stderr = run_mode(C_BINARY_BASELINE, '', prompt_tokens, 10, mode_flag=False)
        if base_ids:
            if tokenizer:
                print(f"  Output: '{tokenizer.decode(base_ids)}'")
            print(f"  Tokens: {base_ids}")
            for line in base_stderr.split('\n'):
                if 'tok/s' in line:
                    print(f"  {line.strip()}")

        for mode in ['serial', 'reuse', 'async', 'cached', 'cache_yes']:
            print(f"\n[{mode}] Async binary, --mode {mode}...")
            ids, stderr = run_mode(C_BINARY_ASYNC, mode, prompt_tokens, 10)
            if ids:
                match = ids == base_ids
                if tokenizer:
                    print(f"  Output: '{tokenizer.decode(ids)}'")
                print(f"  Tokens: {ids}")
                print(f"  Match baseline: {'PASS' if match else 'FAIL'}")
                if not match:
                    print(f"  EXPECTED: {base_ids}")
                    print(f"  GOT:      {ids}")
                for line in stderr.split('\n'):
                    if 'tok/s' in line or 'ANE:' in line or 'Per-token' in line:
                        print(f"  {line.strip()}")

    # ─── Throughput Benchmark ───
    if args.bench:
        n = args.tokens or 50
        print("\n" + "=" * 60)
        print(f"THROUGHPUT BENCHMARK: {n} tokens from 'The'")
        print("=" * 60)

        prompt_tokens = [464]  # "The"

        # Baseline
        print("\n[baseline] Original serial binary...")
        _, base_stderr = run_mode(C_BINARY_BASELINE, '', prompt_tokens, n, mode_flag=False)
        for line in base_stderr.split('\n'):
            if 'tok/s' in line:
                print(f"  {line.strip()}")

        for mode in ['serial', 'reuse', 'async', 'cached', 'cache_yes']:
            print(f"\n[{mode}] Async binary, --mode {mode}...")
            _, stderr = run_mode(C_BINARY_ASYNC, mode, prompt_tokens, n)
            for line in stderr.split('\n'):
                if 'tok/s' in line or 'ANE:' in line or 'Per-token' in line:
                    print(f"  {line.strip()}")

    # ─── Single mode ───
    if args.mode and not args.test and not args.bench:
        n = args.tokens or 10
        if tokenizer:
            prompt_tokens = tokenizer.encode(args.prompt)
        else:
            prompt_tokens = [464]

        print(f"\nRunning --mode {args.mode}, {n} tokens...")
        ids, stderr = run_mode(C_BINARY_ASYNC, args.mode, prompt_tokens, n)
        if ids and tokenizer:
            print(f"Output: '{args.prompt}{tokenizer.decode(ids)}'")
        print(stderr)


if __name__ == '__main__':
    main()
