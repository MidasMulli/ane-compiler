#!/usr/bin/env python3
"""
Build Llama-3.1-8B-Instruct as 4 fused dispatch groups (8 layers each).

Each group of 8 layers compiles to a single ANE dispatch via espresso format.
Total: 4 dispatches for all 32 layers (vs 168 unfused dispatches previously).

For seq_len=1 decode: attention simplified to V→VO (softmax of scalar = 1.0)
Weight format: raw espresso v28 (bypasses protobuf 2GB limit)
Limit: ~6GB per dispatch group (8 layers at dim=4096 = 5.9GB FP32)

Copyright 2026 Nick Lo. MIT License.
"""

import os
import sys
import time
import shutil
import numpy as np
import subprocess

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
sys.path.insert(0, os.path.dirname(__file__))

from build_llama8b_espresso import generate_llama_espresso


MODEL_PATH = os.path.expanduser(
    '~/.cache/huggingface/hub/models--unsloth--Meta-Llama-3.1-8B-Instruct/'
    'snapshots/a2856192dd7c25b842431f39c179a6c2c2f627d1'
)

EVAL_BINARY = os.path.join(os.path.dirname(__file__), 'tests', 'ane_eval_binary')

LAYERS_PER_GROUP = 8
N_GROUPS = 4  # 32 / 8 = 4


def build_all_groups(build_dir='/tmp/llama8b_fused', load_weights=True):
    """Build 4 groups of 8 layers each as fused espresso .mlmodelc."""

    os.makedirs(build_dir, exist_ok=True)

    # Load model weights
    if load_weights:
        print(f"Loading Llama-8B weights from {MODEL_PATH}...")
        t0 = time.time()
        from llama_loader import LlamaModel
        model = LlamaModel.from_safetensors(MODEL_PATH)
        dt = time.time() - t0
        print(f"  Loaded in {dt:.1f}s")
        print(f"  Config: hidden={model.config.hidden_size}, layers={model.config.n_layers}")
    else:
        model = None

    dim = 4096
    ffn_dim = 14336
    n_kv_heads = 8
    head_dim = 128
    kv_dim = n_kv_heads * head_dim  # 1024

    results = []
    total_t0 = time.time()

    for group in range(N_GROUPS):
        start_layer = group * LAYERS_PER_GROUP
        end_layer = start_layer + LAYERS_PER_GROUP
        print(f"\n{'='*60}")
        print(f"Group {group}: Layers {start_layer}-{end_layer-1}")
        print(f"{'='*60}")

        # Build weight dict for this group
        weights = {}
        if model:
            for i in range(LAYERS_PER_GROUP):
                global_layer = start_layer + i
                L = model.layers[global_layer]

                weights[f'L{i}_ln1_bn'] = L.input_layernorm_weight
                weights[f'L{i}_ln2_bn'] = L.post_attention_layernorm_weight
                weights[f'L{i}_v_proj'] = L.v_proj_weight

                # Combined V→O projection
                n_rep = model.config.n_rep  # 4
                expand = np.zeros((dim, kv_dim), dtype=np.float32)
                for kv_h in range(n_kv_heads):
                    for r in range(n_rep):
                        q_h = kv_h * n_rep + r
                        expand[q_h * head_dim:(q_h+1) * head_dim,
                               kv_h * head_dim:(kv_h+1) * head_dim] = \
                            np.eye(head_dim, dtype=np.float32)
                vo_weight = L.o_proj_weight @ expand  # [4096, 1024]
                weights[f'L{i}_vo_proj'] = vo_weight

                weights[f'L{i}_gate_proj'] = L.gate_proj_weight
                weights[f'L{i}_up_proj'] = L.up_proj_weight
                weights[f'L{i}_down_proj'] = L.down_proj_weight

        outdir = os.path.join(build_dir, f'group_{group}.mlmodelc')
        if os.path.exists(outdir):
            shutil.rmtree(outdir)

        t0 = time.time()
        generate_llama_espresso(
            outdir, n_layers=LAYERS_PER_GROUP,
            dim=dim, ffn_dim=ffn_dim,
            n_kv_heads=n_kv_heads, head_dim=head_dim,
            weights=weights if model else None,
            use_fp16_weights=False,
        )
        gen_time = time.time() - t0

        wsize = os.path.getsize(os.path.join(outdir, 'model.espresso.weights'))
        print(f"  Generated in {gen_time:.1f}s, weights: {wsize/1e9:.2f}GB")

        results.append({
            'group': group,
            'start_layer': start_layer,
            'end_layer': end_layer,
            'mlmodelc_path': outdir,
            'weight_size_gb': wsize / 1e9,
        })

    total_time = time.time() - total_t0
    print(f"\n{'='*60}")
    print(f"BUILD COMPLETE: {len(results)} groups in {total_time:.0f}s")
    for r in results:
        print(f"  Group {r['group']}: L{r['start_layer']}-{r['end_layer']-1}, "
              f"{r['weight_size_gb']:.2f}GB")
    print(f"{'='*60}")

    return results


def compile_and_verify(build_dir='/tmp/llama8b_fused'):
    """Compile all groups on ANE and verify dispatch count."""
    import objc
    from Foundation import NSURL, NSDictionary

    objc.loadBundle('AppleNeuralEngine', globals(),
        bundle_path='/System/Library/PrivateFrameworks/AppleNeuralEngine.framework')
    ANEClient = objc.lookUpClass('_ANEClient')
    ANEModel = objc.lookUpClass('_ANEModel')
    client = ANEClient.sharedConnection()

    for group in range(N_GROUPS):
        outdir = os.path.join(build_dir, f'group_{group}.mlmodelc')
        if not os.path.exists(outdir):
            print(f"  Group {group}: NOT FOUND")
            continue

        url = NSURL.fileURLWithPath_(outdir)
        model = ANEModel.modelAtURL_key_(url, 'default')

        t0 = time.time()
        ok = client.compileModel_options_qos_error_(model, {}, 0, None)
        compile_time = time.time() - t0

        if ok:
            ok2 = client.loadModel_options_qos_error_(model, {}, 0, None)
            if ok2:
                attrs = model.modelAttributes()
                desc = attrs.get('ANEFModelDescription', {})
                procs = desc.get('ANEFModelProcedures', [])
                print(f"  Group {group}: {len(procs)} dispatch, compile={compile_time:.1f}s")
            else:
                print(f"  Group {group}: compile OK, load FAILED")
        else:
            print(f"  Group {group}: compile FAILED ({compile_time:.1f}s)")


def dispatch_test(build_dir='/tmp/llama8b_fused'):
    """Dispatch all 4 groups sequentially with test input."""
    dim = 4096

    # Start with random input
    x = np.random.randn(dim).astype(np.float16)

    total_t0 = time.time()
    for group in range(N_GROUPS):
        outdir = os.path.join(build_dir, f'group_{group}.mlmodelc')

        t0 = time.time()
        result = subprocess.run(
            [EVAL_BINARY, outdir, str(dim), str(dim)],
            input=x.tobytes(),
            capture_output=True,
            timeout=120,
        )
        dt = time.time() - t0

        if result.returncode == 0 and result.stdout:
            out = np.frombuffer(result.stdout[:dim*2], dtype=np.float16)
            print(f"  Group {group}: {dt*1000:.0f}ms, "
                  f"range=[{out.min():.4f}, {out.max():.4f}], "
                  f"non-zero={np.count_nonzero(out)}/{len(out)}")
            x = out  # Feed output to next group
        else:
            print(f"  Group {group}: FAILED (rc={result.returncode})")
            print(f"    stderr: {result.stderr.decode()[:200]}")
            break

    total_time = time.time() - total_t0
    print(f"\n  Total: {total_time*1000:.0f}ms for 32 layers = {1/total_time:.1f} tok/s")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--build', action='store_true', help='Build all groups')
    parser.add_argument('--compile', action='store_true', help='Compile and verify dispatch count')
    parser.add_argument('--dispatch', action='store_true', help='Dispatch test')
    parser.add_argument('--no-weights', action='store_true', help='Use random weights')
    parser.add_argument('--build-dir', default='/tmp/llama8b_fused')
    args = parser.parse_args()

    if args.build:
        build_all_groups(args.build_dir, load_weights=not args.no_weights)

    if args.compile:
        print("\nCompiling on ANE:")
        compile_and_verify(args.build_dir)

    if args.dispatch:
        print("\nDispatching test:")
        dispatch_test(args.build_dir)

    if not any([args.build, args.compile, args.dispatch]):
        # Default: build with random weights + compile + dispatch
        build_all_groups(args.build_dir, load_weights=False)
        print("\nCompiling on ANE:")
        compile_and_verify(args.build_dir)
        print("\nDispatching test:")
        dispatch_test(args.build_dir)
