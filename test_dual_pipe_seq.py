#!/usr/bin/env python3
"""
Sequential dual-pipe test: compile A, then compile B, then load both.

The simultaneous test failed because Group B's load failed at op 63
while Group A was also compiling. This tests whether the failure is
due to concurrent aned access or a true system limit.

Protocol:
  1. Launch pipe A, wait for READY_FOR_SWAP (all 88 ops compiled+unloaded)
  2. THEN launch pipe B, wait for READY_FOR_SWAP (all 80 ops compiled+unloaded)
  3. Send GO to BOTH simultaneously (both try to loadModel all their ops)
  4. Check if both reach DISPATCH_READY
"""

import os
import sys
import time
import subprocess
import threading

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

PIPE_TOOL = os.path.join(os.path.dirname(__file__), 'tests', 'ane_standalone_pipe')
BUILD_DIR = '/tmp/llama_8b_ane_32L'


def reconstruct_compiled(build_dir):
    dim = 4096
    qkv_out = 6144
    ffn_dim = 14336
    ops = {}
    for i in range(32):
        layer_dir = os.path.join(build_dir, f'layer_{i}')
        ops[f'L{i}_qkv_proj'] = (os.path.join(layer_dir, 'qkv_proj.mlmodelc'), dim, qkv_out)
        ops[f'L{i}_o_proj'] = (os.path.join(layer_dir, 'o_proj.mlmodelc'), dim, dim)
        ops[f'L{i}_gate'] = (os.path.join(layer_dir, 'gate_proj.mlmodelc'), dim, ffn_dim)
        ops[f'L{i}_up'] = (os.path.join(layer_dir, 'up_proj.mlmodelc'), dim, ffn_dim)
        ops[f'L{i}_down'] = (os.path.join(layer_dir, 'down_proj.mlmodelc'), ffn_dim, dim)

    lm_head_dir = os.path.join(build_dir, 'lm_head')
    chunk_size = 16032
    vocab = 128256
    for j in range(8):
        start = j * chunk_size
        end = min(start + chunk_size, vocab)
        oc = end - start
        path = os.path.join(lm_head_dir, f'lm_head_chunk_{j}.mlmodelc')
        ops[f'lm_head_chunk_{j}'] = (path, dim, oc)
    return ops


def split_ops(ops):
    group_a = {}
    group_b = {}
    for name, info in ops.items():
        if 'lm_head' in name:
            group_a[name] = info
        elif name.startswith('L'):
            layer_num = int(name.split('_')[0][1:])
            if layer_num < 16:
                group_a[name] = info
            else:
                group_b[name] = info
        else:
            group_a[name] = info
    return group_a, group_b


def write_manifest(ops, path):
    names = sorted(ops.keys())
    with open(path, 'w') as f:
        for name in names:
            mlc_path, in_ch, out_ch = ops[name]
            f.write(f"{mlc_path} {in_ch} {out_ch}\n")
    return names


def launch_and_wait_compile(label, manifest_path, timeout=600):
    """Launch pipe tool and wait for READY_FOR_SWAP."""
    proc = subprocess.Popen(
        [PIPE_TOOL, manifest_path],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    print(f"  [{label}] PID {proc.pid} launched")

    stderr_lines = []

    def read_stderr():
        while True:
            line = proc.stderr.readline()
            if not line:
                break
            decoded = line.decode().strip()
            stderr_lines.append(decoded)
            print(f"  [{label}] stderr: {decoded}")

    t = threading.Thread(target=read_stderr, daemon=True)
    t.start()

    t0 = time.time()
    stdout_lines = []
    while time.time() - t0 < timeout:
        line = proc.stdout.readline().decode().strip()
        if not line and proc.poll() is not None:
            t.join(timeout=2)
            return None, proc, stderr_lines, stdout_lines
        stdout_lines.append(line)
        if line == 'READY_FOR_SWAP':
            elapsed = time.time() - t0
            print(f"  [{label}] READY_FOR_SWAP ({elapsed:.1f}s)")
            return proc, proc, stderr_lines, stdout_lines

    return None, proc, stderr_lines, stdout_lines


def send_go(label, proc, stderr_lines):
    """Send GO and wait for DISPATCH_READY."""
    proc.stdin.write(b"GO\n")
    proc.stdin.flush()

    t0 = time.time()
    while time.time() - t0 < 120:
        line = proc.stdout.readline().decode().strip()
        if not line and proc.poll() is not None:
            return False, f"Crashed. Last stderr: {stderr_lines[-5:]}"
        if line == 'DISPATCH_READY':
            elapsed = time.time() - t0
            print(f"  [{label}] DISPATCH_READY ({elapsed:.1f}s)")
            return True, None
    return False, "Timeout"


def cleanup(proc):
    if proc and proc.poll() is None:
        try:
            proc.stdin.write(b"Q\n")
            proc.stdin.flush()
            proc.wait(timeout=3)
        except:
            proc.kill()
            try: proc.wait(timeout=2)
            except: pass


def main():
    ops = reconstruct_compiled(BUILD_DIR)
    group_a, group_b = split_ops(ops)
    print(f"Group A: {len(group_a)} ops (layers 0-15 + lm_head)")
    print(f"Group B: {len(group_b)} ops (layers 16-31)")

    manifest_a = '/tmp/manifest_a.txt'
    manifest_b = '/tmp/manifest_b.txt'
    write_manifest(group_a, manifest_a)
    write_manifest(group_b, manifest_b)

    # Step 1: Compile A fully (sequential)
    print(f"\n{'='*60}")
    print("STEP 1: Compile Group A (88 ops) — wait for completion")
    print(f"{'='*60}")
    proc_a, _, stderr_a, stdout_a = launch_and_wait_compile('A', manifest_a)
    if proc_a is None:
        print("  Group A FAILED during compile!")
        print(f"  Stderr: {stderr_a[-5:]}")
        return

    # Step 2: Compile B fully (sequential — A is done compiling, holding handles)
    print(f"\n{'='*60}")
    print("STEP 2: Compile Group B (80 ops) — A is idle at READY_FOR_SWAP")
    print(f"{'='*60}")
    proc_b, _, stderr_b, stdout_b = launch_and_wait_compile('B', manifest_b)
    if proc_b is None:
        print("  Group B FAILED during compile!")
        print(f"  Stderr: {stderr_b[-5:]}")
        cleanup(proc_a)
        return

    # Step 3: Send GO to BOTH simultaneously
    print(f"\n{'='*60}")
    print("STEP 3: Send GO to both — load all ops simultaneously")
    print(f"{'='*60}")

    results = [None, None]

    def go_a():
        results[0] = send_go('A', proc_a, stderr_a)

    def go_b():
        results[1] = send_go('B', proc_b, stderr_b)

    ta = threading.Thread(target=go_a)
    tb = threading.Thread(target=go_b)
    ta.start()
    tb.start()
    ta.join(timeout=120)
    tb.join(timeout=120)

    ok_a, err_a = results[0]
    ok_b, err_b = results[1]

    # Report
    print(f"\n{'='*60}")
    if ok_a and ok_b:
        print(f"RESULT: PER-CONNECTION. 168 ops loaded across 2 pipes.")
        print(f"  Group A: 88 ops LOADED")
        print(f"  Group B: 80 ops LOADED")
        print(f"  8B IS ALIVE on ANE via dual-pipe architecture.")
    else:
        print(f"RESULT: PER-SYSTEM (or contention failure).")
        if not ok_a:
            print(f"  Group A FAILED during load: {err_a}")
        if not ok_b:
            print(f"  Group B FAILED during load: {err_b}")
        if ok_a and not ok_b:
            print(f"  A loaded 88 ops. B failed. System limit likely ~125.")
        elif not ok_a and ok_b:
            print(f"  B loaded 80 ops. A failed at 88.")
        else:
            print(f"  Both failed.")
        print(f"  8B IS DEAD on ANE via dual-pipe.")
    print(f"{'='*60}")

    cleanup(proc_a)
    cleanup(proc_b)


if __name__ == '__main__':
    main()
