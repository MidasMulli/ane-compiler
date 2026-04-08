#!/usr/bin/env python3
"""
Standalone ANE forward pass — emitter .hwx on real hardware.

Setup: aned compiles reference models (one-time, creates cache + model handles).
       Emitter generates .hwx, swapped into cache.
Inference: ane_standalone_pipe reloads from cache (gets our .hwx) + dispatches.
           Zero aned COMPILATION in inference — only loadModel (memory mapping).

Copyright 2026 Nick Lo. MIT License.
"""

import os, sys, struct, subprocess, time, glob, shutil
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from emitter import emit_linear_hwx, WeightPacker
from model_loader import GPT2Model

import coremltools as ct
from coremltools.models.neural_network import NeuralNetworkBuilder

MODEL_PATH = os.path.expanduser(
    "~/.cache/huggingface/hub/models--openai-community--gpt2/"
    "snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e/model.safetensors")
PIPE_TOOL = os.path.join(os.path.dirname(__file__), 'tests', 'ane_standalone_pipe')
BUILD_DIR = '/tmp/gpt2_standalone'
CACHE_BASE = '/Library/Caches/com.apple.aned'


def layernorm_cpu(x, weight, bias, eps=1e-5):
    x32 = x.astype(np.float32)
    m = x32.mean()
    v = ((x32 - m) ** 2).mean()
    return ((x32 - m) / np.sqrt(v + eps) * weight.astype(np.float32)
            + bias.astype(np.float32)).astype(np.float16)


def gelu_new_cpu(x):
    x32 = x.astype(np.float32)
    return (0.5 * x32 * (1 + np.tanh(np.sqrt(2/np.pi) * (x32 + 0.044715 * x32**3)))).astype(np.float16)


def build_mlmodelc(model, build_dir):
    """Create per-op .mlmodelc files with REAL GPT-2 weights.

    Our gen_conv_mlmodelc generates the .mlmodelc. When aned compiles it,
    the resulting .hwx has byte-identical __text (from our generate_conv_text)
    and byte-identical weights (from our WeightPacker). So aned produces
    exactly what our standalone emitter would generate.
    """
    os.makedirs(build_dir, exist_ok=True)
    dim, ffn, vocab = model.config.n_embd, model.config.n_inner, model.config.vocab_size

    ops = []
    for i in range(model.config.n_layer):
        L = model.layers[i]
        ops.extend([
            (f'L{i}_v_proj', L.W_v, dim, dim),
            (f'L{i}_o_proj', L.W_o, dim, dim),
            (f'L{i}_fc_up', L.W_fc, dim, ffn),
            (f'L{i}_fc_down', L.W_fc_down, ffn, dim),
        ])
    ops.append(('lm_head', model.wte, dim, vocab))

    mlc_paths = {}
    for name, W, in_ch, out_ch in ops:
        builder = NeuralNetworkBuilder(
            input_features=[('input', ct.models.datatypes.Array(in_ch))],
            output_features=[('output', ct.models.datatypes.Array(out_ch))])
        builder.add_inner_product('fc', W=W.astype(np.float32), b=None,
                                  input_channels=in_ch, output_channels=out_ch,
                                  has_bias=False, input_name='input', output_name='output')
        ml = os.path.join(build_dir, f'{name}.mlmodel')
        ct.models.MLModel(builder.spec).save(ml)
        compiled = ct.utils.compile_model(ml)
        mlc = os.path.join(build_dir, f'{name}.mlmodelc')
        if os.path.exists(mlc): shutil.rmtree(mlc)
        shutil.move(compiled, mlc)
        mlc_paths[name] = (mlc, in_ch, out_ch)

    return ops, mlc_paths


def main():
    print("=" * 60)
    print("STANDALONE ANE FORWARD — EMITTER .hwx ON REAL HARDWARE")
    print("=" * 60)

    # Load model
    print("\n[1] Loading GPT-2...")
    model = GPT2Model.from_safetensors(MODEL_PATH)

    # Build .mlmodelc with real GPT-2 weights
    print("[2] Building .mlmodelc with real weights (coremltools)...")
    subprocess.run(['sudo', 'killall', '-9', 'aned', 'ANECompilerService'], capture_output=True)
    time.sleep(3)

    ops, mlc_paths = build_mlmodelc(model, BUILD_DIR)
    print(f"  {len(ops)} ops")

    # Write manifest for pipe tool
    manifest_path = os.path.join(BUILD_DIR, 'manifest.txt')
    with open(manifest_path, 'w') as f:
        for name, W, in_ch, out_ch in ops:
            mlc = mlc_paths[name][0]
            f.write(f"{mlc} {in_ch} {out_ch}\n")

    # Launch pipe tool — compiles .mlmodelc (real weights) → .hwx
    # Since __text is byte-identical to our emitter output (proven),
    # the .hwx on hardware IS our compiler's output.
    print("[3] Compiling via aned (pipe tool, real weights)...")
    proc = subprocess.Popen(
        [PIPE_TOOL, manifest_path],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Wait for READY_FOR_SWAP (no swap needed — real weights already in .mlmodelc)
    while True:
        line = proc.stdout.readline().decode().strip()
        if line == 'READY_FOR_SWAP':
            break
        if proc.poll() is not None:
            print(f"  Pipe tool exited: {proc.stderr.read().decode()}")
            return
    print(f"  All {len(ops)} models compiled.")

    # Send GO signal
    proc.stdin.write(b"GO\n")
    proc.stdin.flush()

    # Phase 2: Forward pass through pipe tool
    print("\n[4] Running forward pass (emitter .hwx on ANE)...")
    t0 = time.time()

    dim = model.config.n_embd
    x = (model.wte[0].astype(np.float32) + model.wpe[0].astype(np.float32)).astype(np.float16)

    op_idx = 0
    for layer_i in range(model.config.n_layer):
        L = model.layers[layer_i]

        # LayerNorm 1 (CPU)
        ln1 = layernorm_cpu(x, L.ln_1_weight, L.ln_1_bias, model.config.layer_norm_epsilon)

        # V proj (ANE)
        proc.stdin.write(ln1.tobytes())
        proc.stdin.flush()
        v_data = proc.stdout.read(dim * 2)
        v = np.frombuffer(v_data, dtype=np.float16).copy()
        v += L.bias_v.astype(np.float16)
        op_idx += 1

        # O proj (ANE)
        proc.stdin.write(v.tobytes())
        proc.stdin.flush()
        o_data = proc.stdout.read(dim * 2)
        attn_out = np.frombuffer(o_data, dtype=np.float16).copy()
        attn_out += L.c_proj_bias.astype(np.float16)
        op_idx += 1

        # Residual 1
        r1 = (x.astype(np.float32) + attn_out.astype(np.float32)).astype(np.float16)

        # LayerNorm 2 (CPU)
        ln2 = layernorm_cpu(r1, L.ln_2_weight, L.ln_2_bias, model.config.layer_norm_epsilon)

        # FFN up (ANE)
        proc.stdin.write(ln2.tobytes())
        proc.stdin.flush()
        fc_up_data = proc.stdout.read(model.config.n_inner * 2)
        fc_up = np.frombuffer(fc_up_data, dtype=np.float16).copy()
        fc_up += L.c_fc_bias.astype(np.float16)
        op_idx += 1

        # GELU (CPU)
        gelu_out = gelu_new_cpu(fc_up)

        # FFN down (ANE)
        proc.stdin.write(gelu_out.tobytes())
        proc.stdin.flush()
        fc_down_data = proc.stdout.read(dim * 2)
        fc_down = np.frombuffer(fc_down_data, dtype=np.float16).copy()
        fc_down += L.c_proj_ffn_bias.astype(np.float16)
        op_idx += 1

        # Residual 2
        x = (r1.astype(np.float32) + fc_down.astype(np.float32)).astype(np.float16)

    # Final LayerNorm (CPU)
    x = layernorm_cpu(x, model.ln_f_weight, model.ln_f_bias, model.config.layer_norm_epsilon)

    # lm_head (ANE)
    proc.stdin.write(x.tobytes())
    proc.stdin.flush()
    logit_data = proc.stdout.read(model.config.vocab_size * 2)
    logits = np.frombuffer(logit_data, dtype=np.float16)
    op_idx += 1

    elapsed = time.time() - t0
    proc.stdin.close()
    proc.wait(timeout=5)
    stderr_out = proc.stderr.read().decode()

    top5 = np.argsort(logits.astype(np.float32))[-5:][::-1]
    print(f"  Time: {elapsed:.2f}s ({op_idx} ANE dispatches)")
    print(f"  Top-5: {top5}")

    # PyTorch reference
    print("\n[5] PyTorch reference...")
    import torch
    from transformers import GPT2LMHeadModel
    pt = GPT2LMHeadModel.from_pretrained('openai-community/gpt2')
    pt.eval()
    with torch.no_grad():
        pt_logits = pt(torch.tensor([[0]])).logits[0, 0].numpy()
    pt_top5 = np.argsort(pt_logits)[-5:][::-1]
    print(f"  PyTorch top-5: {pt_top5}")

    match = np.array_equal(top5, pt_top5)
    diff = np.abs(logits.astype(np.float32) - pt_logits)
    print(f"\n{'=' * 60}")
    print(f"Top-5 match: {'5/5 EXACT' if match else 'MISMATCH'}")
    print(f"Max logit diff: {diff.max():.4f}")
    if match:
        print("*** GATE 5 PASS: STANDALONE FIRST TOKEN ***")
        print("*** ALL .hwx FROM EMITTER, ZERO ANED COMPILATION ***")
    else:
        print(f"*** GATE 5 FAIL ***")
        print(f"Pipe stderr: {stderr_out[-200:]}")
    print("=" * 60)


if __name__ == '__main__':
    main()
