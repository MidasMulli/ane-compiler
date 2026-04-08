#!/usr/bin/env python3
"""
Local 70B agent — dispatch research tasks to on-device Llama-3.3-70B.

Runs on GPU at ~7.4 tok/s. Use for background analysis, summarization,
and research tasks that don't need Claude-level reasoning.

Usage:
    from local_agent import ask_70b, ask_70b_async

    # Synchronous (blocks until complete)
    answer = ask_70b("Analyze this dtrace output...", max_tokens=400)

    # Async (returns immediately, check later)
    task = ask_70b_async("Summarize these findings...", max_tokens=300)
    # ... do other work ...
    answer = task.result()  # blocks if not done
"""

import subprocess
import sys
import os
import json
import time
import threading
from concurrent.futures import Future


def ask_70b(prompt, max_tokens=400, system=None):
    """Send a prompt to the local 70B and return the response."""

    if system:
        full_prompt = (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            f"{system}<|eot_id|>"
            f"<|start_header_id|>user<|end_header_id|>\n\n"
            f"{prompt}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        )
    else:
        full_prompt = (
            f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
            f"{prompt}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        )

    script = f"""
import mlx_lm, sys
model, tokenizer = mlx_lm.load("mlx-community/Llama-3.3-70B-Instruct-3bit")
response = mlx_lm.generate(model, tokenizer, prompt={repr(full_prompt)}, max_tokens={max_tokens}, verbose=False)
print(response)
"""

    result = subprocess.run(
        [sys.executable, '-c', script],
        capture_output=True, text=True,
        timeout=max_tokens * 2 + 60,  # ~0.14s/tok + 60s load
    )

    if result.returncode != 0:
        return f"[70B ERROR: {result.stderr[:200]}]"

    return result.stdout.strip()


def ask_70b_async(prompt, max_tokens=400, system=None):
    """Send a prompt to 70B in background. Returns a Future."""
    future = Future()

    def _run():
        try:
            result = ask_70b(prompt, max_tokens, system)
            future.set_result(result)
        except Exception as e:
            future.set_exception(e)

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    return future


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Local 70B agent')
    parser.add_argument('prompt', help='Prompt to send')
    parser.add_argument('--tokens', type=int, default=400)
    parser.add_argument('--system', default=None)
    args = parser.parse_args()

    print(f"Sending to 70B (max {args.tokens} tokens)...")
    t0 = time.time()
    result = ask_70b(args.prompt, args.tokens, args.system)
    elapsed = time.time() - t0
    print(f"\n[{elapsed:.0f}s]\n{result}")
