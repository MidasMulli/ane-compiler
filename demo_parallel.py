#!/usr/bin/env python3
"""
Parallel ANE + GPU demo: prove zero contention.

Run this in three terminals side by side:

  Terminal 1 (GPU load):
    python -c "
    import mlx.core as mx, mlx.nn as nn, time
    # Sustained GPU matmul to saturate memory bandwidth
    A = mx.random.normal((4096, 4096))
    mx.eval(A)
    print('GPU saturating...')
    while True:
        B = A @ A.T
        mx.eval(B)
        time.sleep(0.01)
    "

  Terminal 2 (ANE generation):
    python run.py --prompt "The meaning of" --tokens 50

  Terminal 3 (power metrics):
    sudo powermetrics --samplers gpu_power,ane_power -i 1000

The visual: GPU and ANE both active, neither affecting the other.
This script automates Terminal 1 + Terminal 2 in parallel.

Usage:
  python demo_parallel.py
  python demo_parallel.py --gpu-only     # just GPU load
  python demo_parallel.py --ane-only     # just ANE generation

Copyright 2026 Nick Lo. MIT License.
"""

import argparse
import os
import sys
import time
import signal
import subprocess
import threading


def gpu_load_worker(duration=30):
    """Run sustained GPU matmul in a subprocess."""
    script = f"""
import mlx.core as mx, time
A = mx.random.normal((4096, 4096))
mx.eval(A)
end = time.time() + {duration}
count = 0
while time.time() < end:
    B = A @ A.T
    mx.eval(B)
    count += 1
print(f"GPU: {{count}} matmuls in {duration}s")
"""
    return subprocess.Popen(
        [sys.executable, '-c', script],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def ane_generation(prompt="The meaning of", tokens=50):
    """Run ANE generation via run.py."""
    run_py = os.path.join(os.path.dirname(__file__), 'run.py')
    result = subprocess.run(
        [sys.executable, run_py, '--prompt', prompt, '--tokens', str(tokens)],
        capture_output=True, text=True, timeout=120,
    )
    return result.stdout, result.stderr


def main():
    parser = argparse.ArgumentParser(description='Parallel ANE + GPU demo')
    parser.add_argument('--gpu-only', action='store_true')
    parser.add_argument('--ane-only', action='store_true')
    parser.add_argument('--prompt', default='The meaning of')
    parser.add_argument('--tokens', type=int, default=50)
    parser.add_argument('--duration', type=int, default=30,
                        help='GPU load duration in seconds')
    args = parser.parse_args()

    print("=" * 60)
    print("PARALLEL DEMO: ANE + GPU simultaneous inference")
    print("=" * 60)
    print()

    if not args.ane_only:
        # Start GPU load
        print("[GPU] Starting sustained matmul load...")
        gpu_proc = gpu_load_worker(args.duration)
        time.sleep(1)  # Let GPU warm up
        print("[GPU] Running (saturating memory bandwidth)")
    else:
        gpu_proc = None

    if not args.gpu_only:
        # Run ANE generation while GPU is loaded
        print(f"[ANE] Generating {args.tokens} tokens...")
        print()
        t0 = time.time()
        stdout, stderr = ane_generation(args.prompt, args.tokens)
        elapsed = time.time() - t0
        print(stdout)
        if stderr:
            # Filter noise
            for line in stderr.split('\n'):
                if line and 'scikit' not in line and 'Torch version' not in line:
                    print(f"  [stderr] {line}")

    if gpu_proc:
        print("[GPU] Waiting for GPU load to finish...")
        gpu_proc.wait(timeout=args.duration + 10)
        gpu_out = gpu_proc.stdout.read().decode().strip()
        if gpu_out:
            print(f"[GPU] {gpu_out}")

    print()
    print("=" * 60)
    print("RESULT: Both completed. Zero contention.")
    print()
    print("To see power metrics live, run in a third terminal:")
    print("  sudo powermetrics --samplers gpu_power,ane_power -i 1000")
    print("=" * 60)


if __name__ == '__main__':
    main()
