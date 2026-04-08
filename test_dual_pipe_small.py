#!/usr/bin/env python3
"""
Definitive per-connection vs per-system test with controlled sizes.

Test matrix:
  Test 1: A=60, B=60 (120 total, under 125 limit)
  Test 2: A=88, B=40 (128 total, at/near limit)
  Test 3: A=88, B=80 (168 total, over limit)

If Test 1 passes and Test 3 fails: per-system limit confirmed.
If Test 1 fails: per-connection limit at <60 (contradicts 88 single-pipe).
"""

import os
import sys
import time
import subprocess
import threading

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

PIPE_TOOL = os.path.join(os.path.dirname(__file__), 'tests', 'ane_standalone_pipe')
BUILD_DIR = '/tmp/llama_8b_ane_32L'


def reconstruct_all():
    dim = 4096; qkv_out = 6144; ffn_dim = 14336
    ops = {}
    for i in range(32):
        d = os.path.join(BUILD_DIR, f'layer_{i}')
        ops[f'L{i}_qkv_proj'] = (os.path.join(d, 'qkv_proj.mlmodelc'), dim, qkv_out)
        ops[f'L{i}_o_proj'] = (os.path.join(d, 'o_proj.mlmodelc'), dim, dim)
        ops[f'L{i}_gate'] = (os.path.join(d, 'gate_proj.mlmodelc'), dim, ffn_dim)
        ops[f'L{i}_up'] = (os.path.join(d, 'up_proj.mlmodelc'), dim, ffn_dim)
        ops[f'L{i}_down'] = (os.path.join(d, 'down_proj.mlmodelc'), ffn_dim, dim)
    lm_head_dir = os.path.join(BUILD_DIR, 'lm_head')
    for j in range(8):
        start = j * 16032
        oc = min(16032, 128256 - start)
        ops[f'lm_head_chunk_{j}'] = (os.path.join(lm_head_dir, f'lm_head_chunk_{j}.mlmodelc'), dim, oc)
    return ops


def take_n(ops, n):
    """Take first n ops (sorted by name)."""
    names = sorted(ops.keys())[:n]
    return {k: ops[k] for k in names}


def write_manifest(ops, path):
    names = sorted(ops.keys())
    with open(path, 'w') as f:
        for name in names:
            p, ic, oc = ops[name]
            f.write(f"{p} {ic} {oc}\n")


def run_pipe(label, manifest_path, timeout=300):
    """Run pipe tool through compile and load phases."""
    proc = subprocess.Popen(
        [PIPE_TOOL, manifest_path],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    stderr_lines = []
    def drain_stderr():
        while True:
            line = proc.stderr.readline()
            if not line: break
            stderr_lines.append(line.decode().strip())

    t = threading.Thread(target=drain_stderr, daemon=True)
    t.start()

    # Wait for READY_FOR_SWAP
    t0 = time.time()
    while time.time() - t0 < timeout:
        line = proc.stdout.readline().decode().strip()
        if not line and proc.poll() is not None:
            t.join(timeout=2)
            fail_line = [s for s in stderr_lines if 'failed' in s.lower()]
            return {'status': 'COMPILE_FAIL', 'error': fail_line or stderr_lines[-3:],
                    'proc': proc}
        if line == 'READY_FOR_SWAP':
            break
    else:
        return {'status': 'TIMEOUT_COMPILE', 'proc': proc}

    compile_time = time.time() - t0
    return {'status': 'READY', 'compile_time': compile_time,
            'proc': proc, 'stderr': stderr_lines}


def send_go(result, label, timeout=120):
    """Send GO and wait for DISPATCH_READY."""
    proc = result['proc']
    proc.stdin.write(b"GO\n")
    proc.stdin.flush()

    t0 = time.time()
    while time.time() - t0 < timeout:
        line = proc.stdout.readline().decode().strip()
        if not line and proc.poll() is not None:
            stderr = result.get('stderr', [])
            fail = [s for s in stderr if 'failed' in s.lower()]
            return False, fail or stderr[-3:]
        if line == 'DISPATCH_READY':
            return True, time.time() - t0
    return False, 'TIMEOUT'


def cleanup(result):
    proc = result.get('proc')
    if proc and proc.poll() is None:
        try:
            proc.stdin.write(b"Q\n"); proc.stdin.flush()
            proc.wait(timeout=3)
        except:
            proc.kill()
            try: proc.wait(timeout=2)
            except: pass


def run_test(label, n_a, n_b, all_ops):
    """Run a single dual-pipe test with n_a + n_b ops."""
    print(f"\n{'='*60}")
    print(f"TEST: {label} — A={n_a} ops, B={n_b} ops, total={n_a+n_b}")
    print(f"{'='*60}")

    sorted_names = sorted(all_ops.keys())
    ops_a = {k: all_ops[k] for k in sorted_names[:n_a]}
    ops_b = {k: all_ops[k] for k in sorted_names[n_a:n_a+n_b]}

    ma = '/tmp/manifest_test_a.txt'
    mb = '/tmp/manifest_test_b.txt'
    write_manifest(ops_a, ma)
    write_manifest(ops_b, mb)

    # Sequential compile: A first, then B
    print(f"  Compiling A ({n_a} ops)...")
    ra = run_pipe('A', ma)
    if ra['status'] != 'READY':
        print(f"  A FAILED compile: {ra.get('error', ra['status'])}")
        cleanup(ra)
        return 'A_COMPILE_FAIL'

    print(f"  A ready ({ra['compile_time']:.1f}s). Compiling B ({n_b} ops)...")
    rb = run_pipe('B', mb)
    if rb['status'] != 'READY':
        print(f"  B FAILED compile: {rb.get('error', rb['status'])}")
        cleanup(ra); cleanup(rb)
        return 'B_COMPILE_FAIL'

    print(f"  B ready ({rb['compile_time']:.1f}s). Sending GO to both...")

    # Simultaneous GO
    results = [None, None]
    def go_a(): results[0] = send_go(ra, 'A')
    def go_b(): results[1] = send_go(rb, 'B')

    ta = threading.Thread(target=go_a)
    tb = threading.Thread(target=go_b)
    ta.start(); tb.start()
    ta.join(timeout=120); tb.join(timeout=120)

    ok_a, info_a = results[0]
    ok_b, info_b = results[1]

    if ok_a and ok_b:
        print(f"  BOTH LOADED. A={n_a}, B={n_b}, total={n_a+n_b}")
        print(f"  Load times: A={info_a:.1f}s, B={info_b:.1f}s")
        result = 'PASS'
    else:
        if not ok_a: print(f"  A FAILED load: {info_a}")
        if not ok_b: print(f"  B FAILED load: {info_b}")
        result = 'FAIL'

    cleanup(ra); cleanup(rb)
    return result


def main():
    all_ops = reconstruct_all()
    print(f"Total available ops: {len(all_ops)}")

    results = {}

    # Test 1: well under limit
    results['60+60=120'] = run_test('Under limit', 60, 60, all_ops)

    # Test 2: at limit
    results['64+64=128'] = run_test('At limit', 64, 64, all_ops)

    # Test 3: over limit (if test 2 passes)
    if results['64+64=128'] == 'PASS':
        results['88+80=168'] = run_test('Over limit (8B)', 88, 80, all_ops)
    else:
        # Binary search: find the crossover
        results['50+50=100'] = run_test('Lower', 50, 50, all_ops)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for label, result in results.items():
        print(f"  {label}: {result}")

    # Interpret
    if results.get('60+60=120') == 'PASS' and results.get('88+80=168') == 'FAIL':
        print("\n  VERDICT: PER-SYSTEM limit ~125-128 program handles.")
        print("  8B (168 ops) is DEAD via dual-pipe.")
    elif results.get('88+80=168') == 'PASS':
        print("\n  VERDICT: PER-CONNECTION. 8B is ALIVE via dual-pipe!")
    elif results.get('60+60=120') == 'PASS' and results.get('64+64=128') != 'PASS':
        print(f"\n  VERDICT: PER-SYSTEM limit between 120-128.")
        print("  8B (168 ops) is DEAD via dual-pipe.")
    elif results.get('60+60=120') != 'PASS':
        print("\n  VERDICT: Unexpected — even 60+60 fails. Investigate.")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
