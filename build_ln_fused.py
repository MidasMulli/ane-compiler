#!/usr/bin/env python3
"""
Build GPT-2 with LN2 fused into FFN via coremltools.

Current baseline: 37 dispatches (QKV + O + fused_FFN per layer + lm_head)
  - CPU: LN1, LN2, attention, residuals, LN_f
  - ANE: QKV, O, fused_FFN(up+GELU+down), lm_head

This build: 37 dispatches (QKV + O + fused_LN2FFN per layer + lm_head)
  - CPU: LN1, attention, residuals, LN_f (LN2 eliminated!)
  - ANE: QKV, O, fused_LN2FFN(MVN+BN+up+GELU+down), lm_head

Same dispatch count, 12 fewer CPU LayerNorm ops.

Copyright 2026 Nick Lo. MIT License.
"""

import os
import sys
import json
import time
import shutil
import subprocess
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
sys.path.insert(0, os.path.dirname(__file__))

from model_loader import GPT2Model
from compiler import gen_conv_mlmodelc
from first_token import compile_all_ops, MODEL_PATH

BUILD_DIR = '/tmp/gpt2_ln_fused'
DIM = 768
FFN_DIM = 3072
N_LAYERS = 12
VOCAB_SIZE = 50257


def build_fused_ln2_ffn_ct(output_dir, layer, layer_idx):
    """Build LN2+FFN via coremltools (proven format)."""
    import coremltools as ct
    from coremltools.models.neural_network import NeuralNetworkBuilder
    from coremltools.models.datatypes import Array

    L = layer
    input_features = [('input', Array(DIM, 1, 1))]
    output_features = [('output', Array(DIM, 1, 1))]
    builder = NeuralNetworkBuilder(input_features, output_features,
                                    disable_rank5_shape_mapping=True)

    builder.add_mvn('mvn', 'input', 'mvn_out', across_channels=True,
                    normalize_variance=True, epsilon=1e-5)

    builder.add_batchnorm('bn', DIM,
        L.ln_2_weight.astype(np.float32),
        L.ln_2_bias.astype(np.float32),
        np.zeros(DIM, dtype=np.float32),
        np.ones(DIM, dtype=np.float32),
        input_name='mvn_out', output_name='bn_out', epsilon=0)

    builder.add_inner_product('fc_up',
        L.W_fc.astype(np.float32),
        L.c_fc_bias.astype(np.float32),
        DIM, FFN_DIM, has_bias=True,
        input_name='bn_out', output_name='fc_up_out')

    # GELU via coremltools — will be mode 19 in espresso
    builder.add_gelu('gelu', input_name='fc_up_out', output_name='gelu_out')

    builder.add_inner_product('fc_down',
        L.W_fc_down.astype(np.float32),
        L.c_proj_ffn_bias.astype(np.float32),
        FFN_DIM, DIM, has_bias=True,
        input_name='gelu_out', output_name='output')

    spec = builder.spec
    spec.specificationVersion = 4

    # Save as mlpackage then compile to mlmodelc
    pkg_path = output_dir + '.mlpackage'
    if os.path.exists(pkg_path):
        shutil.rmtree(pkg_path)
    model = ct.models.MLModel(spec, compute_units=ct.ComputeUnit.ALL)
    model.save(pkg_path)

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    r = subprocess.run(['xcrun', 'coremlcompiler', 'compile', pkg_path,
                        os.path.dirname(output_dir)],
                       capture_output=True, text=True, timeout=60)
    if r.returncode != 0:
        raise RuntimeError(f"coremlcompiler failed: {r.stderr[:200]}")

    # Rename if needed
    expected = os.path.join(os.path.dirname(output_dir),
                            os.path.basename(pkg_path).replace('.mlpackage', '.mlmodelc'))
    if expected != output_dir and os.path.exists(expected):
        os.rename(expected, output_dir)

    # Cleanup mlpackage
    if os.path.exists(pkg_path):
        shutil.rmtree(pkg_path)


def compile_all_ln_fused(model):
    """Compile all models with LN2 fused into FFN."""
    os.makedirs(BUILD_DIR, exist_ok=True)
    compiled = {}

    for i in range(N_LAYERS):
        layer_dir = os.path.join(BUILD_DIR, f'layer_{i}')
        os.makedirs(layer_dir, exist_ok=True)
        L = model.layers[i]

        # QKV projection (same as baseline)
        qkv_path = os.path.join(layer_dir, 'qkv_proj.mlmodelc')
        if not os.path.exists(qkv_path):
            W_qkv = L.c_attn_weight.T.copy()
            gen_conv_mlmodelc(qkv_path, W_qkv.astype(np.float32), DIM, 2304,
                              bias=L.c_attn_bias.astype(np.float32), name='qkv_proj')
        compiled[f'L{i}_qkv_proj'] = (qkv_path, DIM, 2304)

        # O projection (same as baseline)
        o_path = os.path.join(layer_dir, 'o_proj.mlmodelc')
        if not os.path.exists(o_path):
            gen_conv_mlmodelc(o_path, L.W_o.astype(np.float32), DIM, DIM,
                              bias=L.c_proj_bias.astype(np.float32), name='o_proj')
        compiled[f'L{i}_o_proj'] = (o_path, DIM, DIM)

        # LN2 + FFN (fused via coremltools)
        ffn_path = os.path.join(layer_dir, 'ln2_fused_ffn.mlmodelc')
        if not os.path.exists(ffn_path):
            build_fused_ln2_ffn_ct(ffn_path, L, i)
        compiled[f'L{i}_fused_ffn'] = (ffn_path, DIM, DIM)

        print(f"  L{i}: QKV + O + LN2+FFN(fused)")

    # lm_head (same as baseline)
    lm_path = os.path.join(BUILD_DIR, 'lm_head.mlmodelc')
    if not os.path.exists(lm_path):
        gen_conv_mlmodelc(lm_path, model.wte.astype(np.float32),
                          DIM, VOCAB_SIZE, name='lm_head')
    compiled['lm_head'] = (lm_path, DIM, VOCAB_SIZE)

    print(f"\n  Total: {len(compiled)} ops")
    return compiled


def export_for_c(model, compiled):
    """Export manifest and weights."""
    manifest = os.path.join(BUILD_DIR, 'manifest.txt')
    weights_bin = os.path.join(BUILD_DIR, 'cpu_weights.bin')

    with open(manifest, 'w') as f:
        for name in sorted(compiled.keys()):
            path, in_ch, out_ch = compiled[name]
            f.write(f"{path} {in_ch} {out_ch} {name}\n")

    # CPU weights: wte, wpe, ln1 (per layer), ln_f
    # LN2 is NOW fused into ANE — no longer needed on CPU
    with open(weights_bin, 'wb') as f:
        f.write(model.wte.astype(np.float32).tobytes())
        f.write(model.wpe.astype(np.float32).tobytes())
        for i in range(N_LAYERS):
            L = model.layers[i]
            f.write(L.ln_1_weight.astype(np.float32).tobytes())
            f.write(L.ln_1_bias.astype(np.float32).tobytes())
        f.write(model.ln_f_weight.astype(np.float32).tobytes())
        f.write(model.ln_f_bias.astype(np.float32).tobytes())

    size_mb = os.path.getsize(weights_bin) / 1e6
    print(f"  CPU weights: {size_mb:.1f} MB")
    print(f"  (No LN2 weights — fused into ANE)")
    return manifest, weights_bin


def main():
    print("=" * 60)
    print("GPT-2 LN2-FUSED BUILD")
    print("Fuse LayerNorm2 into FFN dispatch (eliminate CPU LN2)")
    print("Baseline: 37 dispatches at 137.7 tok/s")
    print("=" * 60)

    print("\n[1] Loading GPT-2 117M...")
    model = GPT2Model.from_safetensors(MODEL_PATH)

    if os.path.exists(BUILD_DIR):
        shutil.rmtree(BUILD_DIR)

    print("\n[2] Building fused models via coremltools...")
    t0 = time.time()
    compiled = compile_all_ln_fused(model)
    print(f"  Built in {time.time()-t0:.1f}s")

    print("\n[3] Exporting...")
    manifest, weights_bin = export_for_c(model, compiled)

    # The C binary is the SAME as the baseline (ane_generate.m) because:
    # - Same number of dispatches (37)
    # - Same op names (L{i}_qkv_proj, L{i}_o_proj, L{i}_fused_ffn, lm_head)
    # - Same CPU weight format (LN1 per layer, LN_f)
    # - LN2 is implicitly done by the fused FFN dispatch
    #
    # The ONLY difference: the fused_ffn models now include MVN+BN at the start.
    # The C binary feeds them the POST-RESIDUAL signal (r1) instead of the
    # POST-LN2 signal. So we need to skip the CPU LN2 step.

    c_source = os.path.join(os.path.dirname(__file__), 'tests', 'ane_generate.m')
    c_binary = os.path.join(os.path.dirname(__file__), 'tests', 'ane_generate')

    # Build C binary (reuse existing ane_generate.m but we need a modified version
    # that skips CPU LN2)
    c_source_2 = os.path.join(os.path.dirname(__file__), 'tests', 'ane_generate_ln.m')
    write_c_binary_ln_fused(c_source_2)
    c_binary_2 = c_source_2.replace('.m', '')

    print("\n[4] Building C binary...")
    r = subprocess.run([
        'xcrun', 'clang', '-O2',
        '-framework', 'Foundation',
        '-framework', 'IOSurface',
        '-framework', 'Accelerate',
        '-fobjc-arc',
        '-o', c_binary_2, c_source_2,
    ], capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  BUILD FAILED: {r.stderr}")
        return

    # Run with different token counts
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained('gpt2')
        prompt_tokens = tokenizer.encode("The")
    except ImportError:
        prompt_tokens = [464]
        tokenizer = None

    # First: 10 tokens for correctness check
    print("\n[5] Correctness check (10 tokens)...")
    cmd = [c_binary_2, manifest, weights_bin, '10'] + [str(t) for t in prompt_tokens]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

    if result.returncode != 0:
        print(f"  FAILED: {result.stderr[:500]}")
        return

    for line in result.stderr.split('\n'):
        if line.strip():
            print(f"    {line.strip()}")

    gen_ids = []
    for line in result.stdout.strip().split('\n'):
        line = line.strip()
        if line and line.lstrip('-').isdigit():
            gen_ids.append(int(line))

    # Check against PyTorch
    import torch
    from transformers import GPT2LMHeadModel
    pt_model = GPT2LMHeadModel.from_pretrained('openai-community/gpt2')
    pt_model.eval()
    input_ids = torch.tensor([prompt_tokens])
    with torch.no_grad():
        output = pt_model.generate(input_ids, max_new_tokens=10, do_sample=False)
    pt_tokens = output[0].tolist()[len(prompt_tokens):]

    print(f"\n  PyTorch:  {pt_tokens}")
    print(f"  ANE LN2f: {gen_ids[:10]}")
    matches = sum(1 for i in range(min(len(gen_ids), len(pt_tokens)))
                  if gen_ids[i] == pt_tokens[i])
    print(f"  Match: {matches}/{len(pt_tokens)}")

    if tokenizer:
        print(f"  Text: 'The{tokenizer.decode(gen_ids[:10])}'")

    # Then: 50 tokens for performance
    print("\n[6] Performance (50 tokens)...")
    cmd = [c_binary_2, manifest, weights_bin, '50'] + [str(t) for t in prompt_tokens]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    for line in result.stderr.split('\n'):
        if 'tok/s' in line or 'Generated' in line or 'eliminated' in line:
            print(f"    {line.strip()}")

    # Run baseline for comparison
    print("\n[7] Baseline comparison (50 tokens)...")
    baseline_compiled = compile_all_ops(model, '/tmp/gpt2_baseline_cmp', mode='fused')
    baseline_manifest = os.path.join('/tmp/gpt2_baseline_cmp', 'manifest.txt')
    baseline_weights = os.path.join('/tmp/gpt2_baseline_cmp', 'cpu_weights.bin')

    with open(baseline_manifest, 'w') as f:
        for name in sorted(baseline_compiled.keys()):
            path, in_ch, out_ch = baseline_compiled[name]
            f.write(f"{path} {in_ch} {out_ch} {name}\n")

    from run_c import export_cpu_weights
    export_cpu_weights(model)

    baseline_c = os.path.join(os.path.dirname(__file__), 'tests', 'ane_generate')
    # Build if needed
    baseline_src = os.path.join(os.path.dirname(__file__), 'tests', 'ane_generate.m')
    if not os.path.exists(baseline_c) or \
       os.path.getmtime(baseline_src) > os.path.getmtime(baseline_c):
        subprocess.run([
            'xcrun', 'clang', '-O2',
            '-framework', 'Foundation', '-framework', 'IOSurface',
            '-framework', 'Accelerate', '-fobjc-arc',
            '-o', baseline_c, baseline_src,
        ], capture_output=True, text=True)

    cmd_base = [baseline_c, '/tmp/gpt2_c_gen/manifest.txt',
                '/tmp/gpt2_c_gen/cpu_weights.bin', '50'] + [str(t) for t in prompt_tokens]
    result_base = subprocess.run(cmd_base, capture_output=True, text=True, timeout=300)
    for line in result_base.stderr.split('\n'):
        if 'tok/s' in line or 'Generated' in line:
            print(f"    Baseline: {line.strip()}")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    for line in result.stderr.split('\n'):
        if 'tok/s' in line:
            print(f"  LN2-fused:  {line.strip()}")
    for line in result_base.stderr.split('\n'):
        if 'tok/s' in line:
            print(f"  Baseline:   {line.strip()}")
    print(f"  Dispatches: 37 (both configs)")
    print(f"  CPU LN2 ops eliminated: 12 (one per layer)")
    print(f"{'='*60}")


def write_c_binary_ln_fused(output_path):
    """Write C binary that skips CPU LN2 (fused into ANE FFN dispatch)."""
    # This is identical to ane_generate.m BUT:
    # 1. Skips CPU LayerNorm2 before FFN dispatch
    # 2. Feeds post-residual1 directly to fused FFN (which does MVN+BN internally)
    # 3. Still does CPU LN1 before QKV

    with open(os.path.join(os.path.dirname(__file__), 'tests', 'ane_generate.m')) as f:
        source = f.read()

    # Replace the LN2 + FFN section
    old_ln2_ffn = """                // LayerNorm 2
                layernorm(x_f32, W.ln2_w[li], W.ln2_b[li], ln_out, DIM);
                uint16_t ln2_fp16[DIM];
                fp32_to_fp16(ln_out, ln2_fp16, DIM);

                // Fused FFN (ANE): up + GELU + down in one dispatch
                int ffn_idx = opIdx([NSString stringWithFormat:@"L%d_fused_ffn", li]);
                ane_dispatch(&ops[ffn_idx], ln2_fp16, ffn_fp16);"""

    new_ln2_ffn = """                // Fused LN2+FFN (ANE): MVN + BN + up + GELU + down
                // CPU LN2 ELIMINATED — fused into ANE dispatch
                uint16_t r1_fp16[DIM];
                fp32_to_fp16(x_f32, r1_fp16, DIM);
                int ffn_idx = opIdx([NSString stringWithFormat:@"L%d_fused_ffn", li]);
                ane_dispatch(&ops[ffn_idx], r1_fp16, ffn_fp16);"""

    source = source.replace(old_ln2_ffn, new_ln2_ffn)

    # Remove ln2 weight loading since we don't need them
    # Actually keep them — they're still in the weights file for compatibility
    # The CPU weight format matches the baseline (has LN2 weights but we just don't use them)

    with open(output_path, 'w') as f:
        f.write(source)


if __name__ == '__main__':
    main()
