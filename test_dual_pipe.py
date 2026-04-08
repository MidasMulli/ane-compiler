#!/usr/bin/env python3
"""
Test: Is the ANE 125-op concurrent program limit per-connection or per-system?

If per-connection: two separate pipe tool instances can each load half of the
8B model's 168 ops. This unlocks 8B on ANE via dual-pipe architecture.

If per-system: both instances share a single 125-op budget. 8B is dead on ANE.

Test protocol:
  1. Compile all 168 ops for Llama-3.1-8B
  2. Split into Group A (layers 0-15 = 80 ops + lm_head 8 ops = 88) and
     Group B (layers 16-31 = 80 ops)
  3. Launch TWO pipe tool instances simultaneously
  4. If BOTH reach DISPATCH_READY: per-connection (8B alive)
  5. If either crashes: per-system (8B dead)
"""

import os
import sys
import time
import subprocess
import threading

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

PIPE_TOOL = os.path.join(os.path.dirname(__file__), 'tests', 'ane_standalone_pipe')
MODEL_PATH = os.path.expanduser(
    '~/.cache/huggingface/hub/models--unsloth--Meta-Llama-3.1-8B-Instruct/'
    'snapshots/a2856192dd7c25b842431f39c179a6c2c2f627d1/')
BUILD_DIR = '/tmp/llama_8b_ane_32L'


def compile_8b():
    """Compile all 168 ops for Llama-3.1-8B.

    If BUILD_DIR already has compiled .mlmodelc bundles, reuse them.
    Otherwise compile from safetensors.
    """
    print("=" * 60)
    print("STEP 1: Loading & compiling Llama-3.1-8B (168 ops)")
    print("=" * 60)

    # Check if already compiled
    existing = os.path.exists(os.path.join(BUILD_DIR, 'layer_31', 'down_proj.mlmodelc'))
    if existing:
        print("  Reusing existing compiled ops from", BUILD_DIR)
        return reconstruct_compiled(BUILD_DIR)

    from llama_loader import LlamaModel, compile_llama_unfused
    model = LlamaModel.from_safetensors(MODEL_PATH)
    print(f"  Config: {model.config.n_layers} layers, "
          f"hidden={model.config.hidden_size}, "
          f"ffn={model.config.intermediate_size}")

    compiled = compile_llama_unfused(model, BUILD_DIR)

    # Filter to real ops only
    ops = {k: v for k, v in compiled.items()
           if not k.startswith('_') and len(v) == 3}
    print(f"\n  Total compiled ops: {len(ops)}")
    return ops


def reconstruct_compiled(build_dir):
    """Reconstruct compiled dict from existing build directory.

    Uses known Llama-3.1-8B dimensions:
      hidden=4096, n_kv_heads=8, head_dim=128, ffn=14336
      qkv_out = 4096 + 2*8*128 = 6144
    """
    dim = 4096
    qkv_out = 6144  # 4096 + 2*1024
    ffn_dim = 14336

    ops = {}
    for i in range(32):
        layer_dir = os.path.join(build_dir, f'layer_{i}')
        ops[f'L{i}_qkv_proj'] = (os.path.join(layer_dir, 'qkv_proj.mlmodelc'), dim, qkv_out)
        ops[f'L{i}_o_proj'] = (os.path.join(layer_dir, 'o_proj.mlmodelc'), dim, dim)
        ops[f'L{i}_gate'] = (os.path.join(layer_dir, 'gate_proj.mlmodelc'), dim, ffn_dim)
        ops[f'L{i}_up'] = (os.path.join(layer_dir, 'up_proj.mlmodelc'), dim, ffn_dim)
        ops[f'L{i}_down'] = (os.path.join(layer_dir, 'down_proj.mlmodelc'), ffn_dim, dim)

    # lm_head chunks
    lm_head_dir = os.path.join(build_dir, 'lm_head')
    chunk_size = 16032
    vocab = 128256
    for j in range(8):
        start = j * chunk_size
        end = min(start + chunk_size, vocab)
        oc = end - start
        path = os.path.join(lm_head_dir, f'lm_head_chunk_{j}.mlmodelc')
        ops[f'lm_head_chunk_{j}'] = (path, dim, oc)

    print(f"  Reconstructed {len(ops)} ops from {build_dir}")
    return ops


def split_ops(ops):
    """Split into Group A (layers 0-15 + lm_head) and Group B (layers 16-31)."""
    group_a = {}
    group_b = {}

    for name, info in ops.items():
        if 'lm_head' in name:
            group_a[name] = info
        elif name.startswith('L'):
            try:
                layer_num = int(name.split('_')[0][1:])
                if layer_num < 16:
                    group_a[name] = info
                else:
                    group_b[name] = info
            except ValueError:
                group_a[name] = info
        else:
            group_a[name] = info

    return group_a, group_b


def write_manifest(ops, path):
    """Write manifest file for pipe tool."""
    names = sorted(ops.keys())
    with open(path, 'w') as f:
        for name in names:
            mlc_path, in_ch, out_ch = ops[name]
            f.write(f"{mlc_path} {in_ch} {out_ch}\n")
    return names


def run_pipe(label, manifest_path, result):
    """Run a single pipe tool instance. Stores result in dict."""
    result['label'] = label
    result['status'] = 'STARTING'
    result['stderr_lines'] = []
    result['stdout_lines'] = []

    try:
        proc = subprocess.Popen(
            [PIPE_TOOL, manifest_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        result['proc'] = proc
        result['pid'] = proc.pid
        print(f"  [{label}] PID {proc.pid} launched")

        # Collect stderr in background
        def read_stderr():
            while True:
                line = proc.stderr.readline()
                if not line:
                    break
                decoded = line.decode().strip()
                result['stderr_lines'].append(decoded)
                print(f"  [{label}] stderr: {decoded}")

        stderr_thread = threading.Thread(target=read_stderr, daemon=True)
        stderr_thread.start()

        # Wait for READY_FOR_SWAP
        result['status'] = 'COMPILING'
        t0 = time.time()
        while True:
            line = proc.stdout.readline().decode().strip()
            if not line and proc.poll() is not None:
                result['status'] = 'CRASHED_DURING_COMPILE'
                result['exit_code'] = proc.returncode
                stderr_thread.join(timeout=2)
                result['error'] = '\n'.join(result['stderr_lines'][-10:])
                return
            result['stdout_lines'].append(line)
            if line == 'READY_FOR_SWAP':
                result['compile_time'] = time.time() - t0
                result['status'] = 'READY_FOR_SWAP'
                print(f"  [{label}] READY_FOR_SWAP ({result['compile_time']:.1f}s)")
                return

    except Exception as e:
        result['status'] = 'EXCEPTION'
        result['error'] = str(e)


def send_go_and_wait(result):
    """Send GO to a pipe tool and wait for DISPATCH_READY."""
    label = result['label']
    proc = result['proc']

    try:
        proc.stdin.write(b"GO\n")
        proc.stdin.flush()

        t0 = time.time()
        while True:
            line = proc.stdout.readline().decode().strip()
            if not line and proc.poll() is not None:
                result['status'] = 'CRASHED_DURING_LOAD'
                result['exit_code'] = proc.returncode
                result['error'] = '\n'.join(result['stderr_lines'][-10:])
                return
            result['stdout_lines'].append(line)
            if line == 'DISPATCH_READY':
                result['load_time'] = time.time() - t0
                result['status'] = 'DISPATCH_READY'
                print(f"  [{label}] DISPATCH_READY ({result['load_time']:.1f}s)")
                return

    except Exception as e:
        result['status'] = 'EXCEPTION'
        result['error'] = str(e)


def cleanup(result):
    """Clean up a pipe tool process."""
    proc = result.get('proc')
    if proc and proc.poll() is None:
        try:
            proc.stdin.write(b"Q\n")
            proc.stdin.flush()
            proc.wait(timeout=3)
        except Exception:
            proc.kill()
            try:
                proc.wait(timeout=2)
            except Exception:
                pass


def main():
    # Step 1: Compile
    ops = compile_8b()

    # Step 2: Split
    group_a, group_b = split_ops(ops)
    print(f"\n{'=' * 60}")
    print(f"STEP 2: Split ops")
    print(f"  Group A (layers 0-15 + lm_head): {len(group_a)} ops")
    print(f"  Group B (layers 16-31):           {len(group_b)} ops")
    print(f"  Total:                             {len(group_a) + len(group_b)} ops")
    print(f"{'=' * 60}")

    # Step 3: Write manifests
    manifest_a = '/tmp/manifest_a.txt'
    manifest_b = '/tmp/manifest_b.txt'
    names_a = write_manifest(group_a, manifest_a)
    names_b = write_manifest(group_b, manifest_b)

    # Step 4: Launch BOTH simultaneously
    print(f"\n{'=' * 60}")
    print(f"STEP 3: Launch TWO pipe tool instances simultaneously")
    print(f"{'=' * 60}")

    result_a = {}
    result_b = {}

    thread_a = threading.Thread(target=run_pipe,
                                args=('GROUP_A', manifest_a, result_a))
    thread_b = threading.Thread(target=run_pipe,
                                args=('GROUP_B', manifest_b, result_b))

    thread_a.start()
    thread_b.start()

    # Wait for both to finish compile phase (timeout 10 min)
    thread_a.join(timeout=600)
    thread_b.join(timeout=600)

    print(f"\n  Group A status: {result_a.get('status')}")
    print(f"  Group B status: {result_b.get('status')}")

    # Check if either crashed during compile
    if result_a['status'] != 'READY_FOR_SWAP':
        print(f"\n  GROUP A CRASHED DURING COMPILE")
        print(f"  Exit code: {result_a.get('exit_code')}")
        print(f"  Error: {result_a.get('error', 'unknown')}")
        cleanup(result_b)
        return report_result('CRASHED_COMPILE', result_a, result_b)

    if result_b['status'] != 'READY_FOR_SWAP':
        print(f"\n  GROUP B CRASHED DURING COMPILE")
        print(f"  Exit code: {result_b.get('exit_code')}")
        print(f"  Error: {result_b.get('error', 'unknown')}")
        cleanup(result_a)
        return report_result('CRASHED_COMPILE', result_a, result_b)

    # Step 5: Send GO to BOTH simultaneously
    print(f"\n{'=' * 60}")
    print(f"STEP 4: Send GO to both — this is the critical test")
    print(f"{'=' * 60}")

    thread_go_a = threading.Thread(target=send_go_and_wait, args=(result_a,))
    thread_go_b = threading.Thread(target=send_go_and_wait, args=(result_b,))

    thread_go_a.start()
    thread_go_b.start()

    thread_go_a.join(timeout=120)
    thread_go_b.join(timeout=120)

    print(f"\n  Group A status: {result_a.get('status')}")
    print(f"  Group B status: {result_b.get('status')}")

    # Step 6: Report
    if (result_a['status'] == 'DISPATCH_READY' and
            result_b['status'] == 'DISPATCH_READY'):
        total_loaded = len(group_a) + len(group_b)
        print(f"\n{'=' * 60}")
        print(f"RESULT: PER-CONNECTION. {total_loaded} ops loaded across 2 pipes.")
        print(f"  Group A: {len(group_a)} ops LOADED")
        print(f"  Group B: {len(group_b)} ops LOADED")
        print(f"  8B IS ALIVE on ANE via dual-pipe architecture.")
        print(f"{'=' * 60}")
    else:
        print(f"\n{'=' * 60}")
        print(f"RESULT: PER-SYSTEM (or other failure).")
        if result_a['status'] != 'DISPATCH_READY':
            print(f"  Group A FAILED: {result_a.get('error', result_a['status'])}")
        if result_b['status'] != 'DISPATCH_READY':
            print(f"  Group B FAILED: {result_b.get('error', result_b['status'])}")
        print(f"  8B IS DEAD on ANE.")
        print(f"{'=' * 60}")

    # Cleanup
    cleanup(result_a)
    cleanup(result_b)


def report_result(phase, result_a, result_b):
    print(f"\n{'=' * 60}")
    print(f"RESULT: FAILED during {phase}")
    for label, r in [('A', result_a), ('B', result_b)]:
        print(f"  Group {label}: {r.get('status')} "
              f"(exit={r.get('exit_code', 'N/A')})")
        if r.get('error'):
            print(f"    Error: {r['error'][:200]}")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
