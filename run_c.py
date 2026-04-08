#!/usr/bin/env python3
"""
GPT-2 generation via C/Accelerate hot path.

Python handles one-time startup (model loading, compilation).
C binary handles every per-token operation (zero Python in hot path).

Usage:
  python run_c.py --prompt "The capital of France" --tokens 10
  python run_c.py --build-only  # just compile, don't run

Copyright 2026 Nick Lo. MIT License.
"""

import argparse
import os
import sys
import time
import struct
import subprocess
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from model_loader import GPT2Model
from first_token import compile_all_ops, MODEL_PATH

BUILD_DIR = '/tmp/gpt2_c_gen'
WEIGHTS_BIN = '/tmp/gpt2_c_gen/cpu_weights.bin'
MANIFEST = '/tmp/gpt2_c_gen/manifest.txt'
C_BINARY = os.path.join(os.path.dirname(__file__), 'tests', 'ane_generate')
C_SOURCE = os.path.join(os.path.dirname(__file__), 'tests', 'ane_generate.m')


def build_c_binary():
    """Compile the C generation binary."""
    print("Building C generation binary...")
    r = subprocess.run([
        'xcrun', 'clang', '-O2',
        '-framework', 'Foundation',
        '-framework', 'IOSurface',
        '-framework', 'Accelerate',
        '-fobjc-arc',
        '-o', C_BINARY,
        C_SOURCE,
    ], capture_output=True, text=True)
    if r.returncode != 0:
        print(f"Build failed:\n{r.stderr}")
        sys.exit(1)
    print(f"  Built: {C_BINARY}")


def export_cpu_weights(model):
    """Export CPU-side weights (embeddings, layernorms) as binary."""
    print("Exporting CPU weights...")
    with open(WEIGHTS_BIN, 'wb') as f:
        # wte: [50257, 768] float32
        f.write(model.wte.astype(np.float32).tobytes())
        # wpe: [1024, 768] float32
        f.write(model.wpe.astype(np.float32).tobytes())
        # Per layer: ln1_w, ln1_b, ln2_w, ln2_b (all [768] float32)
        for i in range(model.config.n_layer):
            L = model.layers[i]
            f.write(L.ln_1_weight.astype(np.float32).tobytes())
            f.write(L.ln_1_bias.astype(np.float32).tobytes())
            f.write(L.ln_2_weight.astype(np.float32).tobytes())
            f.write(L.ln_2_bias.astype(np.float32).tobytes())
        # Final LN
        f.write(model.ln_f_weight.astype(np.float32).tobytes())
        f.write(model.ln_f_bias.astype(np.float32).tobytes())

    size_mb = os.path.getsize(WEIGHTS_BIN) / 1e6
    print(f"  Exported: {WEIGHTS_BIN} ({size_mb:.1f} MB)")


def write_manifest(compiled):
    """Write manifest file with op names for the C binary."""
    with open(MANIFEST, 'w') as f:
        for name in sorted(compiled.keys()):
            path, in_ch, out_ch = compiled[name]
            # Format: path in_ch out_ch op_name
            f.write(f"{path} {in_ch} {out_ch} {name}\n")
    print(f"  Manifest: {len(compiled)} ops -> {MANIFEST}")


def main():
    parser = argparse.ArgumentParser(description='GPT-2 generation via C hot path')
    parser.add_argument('--prompt', default='The capital of France is',
                        help='Input prompt')
    parser.add_argument('--tokens', type=int, default=10,
                        help='Number of tokens to generate')
    parser.add_argument('--build-only', action='store_true',
                        help='Just compile, do not run')
    parser.add_argument('--compare', action='store_true',
                        help='Also run Python path for comparison')
    args = parser.parse_args()

    # Build C binary
    if not os.path.exists(C_BINARY) or \
       os.path.getmtime(C_SOURCE) > os.path.getmtime(C_BINARY):
        build_c_binary()
    else:
        print(f"C binary up to date: {C_BINARY}")

    # Load model
    print("Loading GPT-2 117M...")
    model = GPT2Model.from_safetensors(MODEL_PATH)

    # Compile ANE ops
    print("Compiling ANE ops (fused mode)...")
    compiled = compile_all_ops(model, BUILD_DIR, mode='fused')
    print(f"  {len(compiled)} ops compiled")

    # Export CPU weights
    export_cpu_weights(model)

    # Write manifest
    write_manifest(compiled)

    if args.build_only:
        print("Build complete. Run with: python run_c.py --prompt 'text'")
        return

    # Tokenize
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained('gpt2')
        prompt_tokens = tokenizer.encode(args.prompt)
    except ImportError:
        # Fallback: hardcode "The" = 464
        prompt_tokens = [464]
        tokenizer = None

    print(f"\nPrompt: '{args.prompt}'")
    print(f"Tokens: {prompt_tokens}")
    print(f"Generating {args.tokens} tokens via C hot path...")
    print()

    # Run C binary
    cmd = [C_BINARY, MANIFEST, WEIGHTS_BIN, str(args.tokens)] + \
          [str(t) for t in prompt_tokens]

    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"C binary failed:\n{result.stderr}")
        sys.exit(1)

    # Parse output token IDs
    gen_token_ids = []
    for line in result.stdout.strip().split('\n'):
        line = line.strip()
        if line and line.lstrip('-').isdigit():
            gen_token_ids.append(int(line))

    # Decode
    if tokenizer:
        gen_text = tokenizer.decode(gen_token_ids)
        full_text = args.prompt + gen_text
    else:
        gen_text = str(gen_token_ids)
        full_text = gen_text

    # Parse stderr for timing
    for line in result.stderr.split('\n'):
        if 'tok/s' in line:
            print(f"  C: {line.strip()}")

    print(f"  Output: '{full_text}'")
    print(f"  Token IDs: {gen_token_ids}")
    print(f"  Wall time: {elapsed:.2f}s")
    print()

    # Compare with Python path
    if args.compare:
        print("=" * 50)
        print("Python path comparison:")
        from generate import ANEDispatcher, forward_layer_decode, embed, lm_head, softmax_cpu
        from kv_cache import KVCache

        disp = ANEDispatcher(compiled, quiet=True)
        disp.start()

        kv_cache = KVCache(model.config.n_layer, model.config.n_head,
                           model.config.head_dim)
        generated = list(prompt_tokens)

        t0 = time.time()
        for pos in range(len(prompt_tokens) - 1):
            x = embed(model, prompt_tokens[pos], pos)
            for li in range(model.config.n_layer):
                x = forward_layer_decode(li, x, model, disp, kv_cache, mode='fused')
            _ = lm_head(x, model, disp)

        for _ in range(args.tokens):
            pos = len(generated) - 1
            x = embed(model, generated[-1], pos)
            for li in range(model.config.n_layer):
                x = forward_layer_decode(li, x, model, disp, kv_cache, mode='fused')
            logits = lm_head(x, model, disp)
            next_tok = int(np.argmax(logits.astype(np.float32)))
            generated.append(next_tok)

        py_elapsed = time.time() - t0
        py_tps = args.tokens / py_elapsed
        py_tokens = generated[len(prompt_tokens):]

        if tokenizer:
            py_text = tokenizer.decode(py_tokens)
        else:
            py_text = str(py_tokens)

        print(f"  Python: {py_tps:.1f} tok/s, output: '{py_text}'")
        print(f"  Match: {gen_token_ids == py_tokens}")

        disp.stop()


if __name__ == '__main__':
    main()
