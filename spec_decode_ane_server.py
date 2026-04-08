#!/usr/bin/env python3
"""
Q3 3.3 70B + 1B CPU Drafter + 1B ANE Drafter — Spec Decode Server v4
=====================================================================
Three drafters on three compute units:
  - N-gram:     software (fires first, dominates structured text)
  - CPU 1B:     llama.cpp Q8 on CPU/AMX (K=1, ~10ms)
  - ANE 1B:     ane-compiler on Neural Engine (K=1, ~35ms, zero GPU cost)

When n-gram misses, BOTH CPU and ANE produce 1 draft each.
Verify [last_tok, cpu_draft, ane_draft] as batch of 3 in ONE GPU pass.

Port: 8898 (does not conflict with existing 8899 server).

Copyright 2026 Nick Lo. MIT License.
"""
import json, time, uuid, sys, os, re, threading
from collections import deque
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
import numpy as np

sys.path.insert(0, "/Users/midas/.mlx-env/lib/python3.11/site-packages")
sys.path.insert(0, os.path.dirname(__file__))

import mlx.core as mx
import mlx_lm
from mlx_lm.models.cache import make_prompt_cache, trim_prompt_cache
from llama_cpp import Llama

MODEL_ID = "mlx-community/Llama-3.3-70B-Instruct-3bit"
DRAFT_GGUF = "/Users/midas/models/Llama-3.2-1B-GGUF/Llama-3.2-1B-Instruct-Q8_0.gguf"
PORT = 8898
NGRAM_N = 4
NGRAM_K = 16


class NGramDrafter:
    def __init__(self, n=4):
        self.n = n
        self.table = {}
        self.total_hits = 0
        self.total_lookups = 0

    def update(self, tokens):
        if isinstance(tokens, mx.array):
            tokens = tokens.tolist()
        for i in range(len(tokens) - self.n):
            key = tuple(tokens[i:i + self.n])
            next_tok = tokens[i + self.n]
            if key not in self.table:
                self.table[key] = {}
            self.table[key][next_tok] = self.table[key].get(next_tok, 0) + 1

    def draft(self, context, K):
        if isinstance(context, mx.array):
            context = context.tolist()
        drafts, hit_count = [], 0
        ctx = list(context[-self.n:])
        for _ in range(K):
            key = tuple(ctx[-self.n:])
            self.total_lookups += 1
            if key in self.table:
                candidates = self.table[key]
                tok = max(candidates, key=candidates.get)
                hit_count += 1
                self.total_hits += 1
            else:
                return drafts, hit_count
            drafts.append(tok)
            ctx.append(tok)
        return drafts, hit_count

    @property
    def hit_rate(self):
        return self.total_hits / max(1, self.total_lookups)

    @property
    def size(self):
        return len(self.table)


class CPUDrafter:
    def __init__(self, gguf_path):
        self.llm = Llama(model_path=gguf_path, n_gpu_layers=0, n_ctx=4096,
                         n_threads=2, verbose=False)

    def prefill(self, tokens):
        self.llm.reset()
        self.llm.eval(tokens)

    def draft_one(self):
        tok = self.llm.sample(temp=0.0, top_k=1)
        self.llm.eval([tok])
        return tok

    def feed(self, token):
        self.llm.eval([token])

    def correct(self, n_reject, correction):
        if n_reject > 0:
            new_len = self.llm.n_tokens - n_reject
            self.llm._ctx.kv_cache_seq_rm(0, new_len, -1)
            self.llm.n_tokens = new_len
        self.llm.eval([correction])


class Engine:
    def __init__(self):
        print(f"Loading {MODEL_ID} (GPU)...", flush=True)
        self.model, self.tokenizer = mlx_lm.load(MODEL_ID)
        mx.eval(self.model.parameters())
        print(f"  {len(self.model.model.layers)} layers, "
              f"{mx.get_active_memory()/1e9:.1f} GB", flush=True)

        print(f"Loading 1B drafter (CPU)...", flush=True)
        self.cpu_drafter = CPUDrafter(DRAFT_GGUF)
        print(f"  1B on CPU ready", flush=True)

        print(f"Loading 1B drafter (ANE)...", flush=True)
        from ane_drafter import ANEDrafter
        self.ane_drafter = ANEDrafter()
        print(f"  1B on ANE ready", flush=True)

        self.ngram = NGramDrafter(n=NGRAM_N)
        self.request_count = 0
        self.ngram_window = deque(maxlen=20)
        self.ngram_suppress = False
        self.suppress_threshold = 0.25
        self.reenable_threshold = 0.35

    def generate(self, messages, max_tokens=256):
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        input_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        eos_id = self.tokenizer.eos_token_id

        # Prefill 70B
        cache = make_prompt_cache(self.model)
        logits = self.model(mx.array([input_ids]), cache=cache)
        mx.eval(logits)
        first_tok = mx.argmax(logits[:, -1, :], axis=-1).item()

        # Prefill 1B CPU
        self.cpu_drafter.prefill(input_ids)
        self.cpu_drafter.feed(first_tok)

        # Prefill 1B ANE
        self.ane_drafter.prefill(input_ids)
        self.ane_drafter.feed(first_tok)

        generated = [first_tok]
        all_context = list(input_ids) + [first_tok]
        last_tok = first_tok

        stats = {'ngram_drafted': 0, 'ngram_accepted': 0,
                 'cpu_drafted': 0, 'cpu_accepted': 0,
                 'ane_drafted': 0, 'ane_accepted': 0,
                 'cycles': 0}
        self.ngram_window.clear()
        self.ngram_suppress = False

        t0 = time.perf_counter()

        while len(generated) < max_tokens:
            if last_tok == eos_id:
                break

            remaining = max_tokens - len(generated)
            if remaining <= 0:
                break

            # -- Adaptive N-gram suppression --
            if len(self.ngram_window) >= 10:
                ngram_rate = sum(self.ngram_window) / len(self.ngram_window)
                if ngram_rate < self.suppress_threshold:
                    self.ngram_suppress = True
                elif ngram_rate > self.reenable_threshold:
                    self.ngram_suppress = False

            # -- Try N-gram first (unless suppressed) --
            ngram_k = min(NGRAM_K, remaining)
            if not self.ngram_suppress:
                draft_tokens, hits = self.ngram.draft(all_context, ngram_k)
            else:
                draft_tokens, hits = [], 0

            if draft_tokens:
                stats['ngram_drafted'] += len(draft_tokens)
                verify_mx = mx.array([[last_tok] + draft_tokens])
                vlogits = self.model(verify_mx, cache=cache)
                mx.eval(vlogits)

                accepted = 0
                for i in range(len(draft_tokens)):
                    pred = mx.argmax(vlogits[0, i, :], axis=-1).item()
                    if pred == draft_tokens[i]:
                        accepted += 1
                    else:
                        to_trim = len(draft_tokens) + 1 - (accepted + 1)
                        trim_prompt_cache(cache, to_trim)
                        for j in range(accepted):
                            generated.append(draft_tokens[j])
                            all_context.append(draft_tokens[j])
                            self.cpu_drafter.feed(draft_tokens[j])
                            self.ane_drafter.feed(draft_tokens[j])
                        generated.append(pred)
                        all_context.append(pred)
                        self.cpu_drafter.feed(pred)
                        self.ane_drafter.feed(pred)
                        last_tok = pred
                        stats['ngram_accepted'] += accepted
                        break
                else:
                    bonus = mx.argmax(vlogits[0, len(draft_tokens), :],
                                      axis=-1).item()
                    for j in range(len(draft_tokens)):
                        generated.append(draft_tokens[j])
                        all_context.append(draft_tokens[j])
                        self.cpu_drafter.feed(draft_tokens[j])
                        self.ane_drafter.feed(draft_tokens[j])
                    generated.append(bonus)
                    all_context.append(bonus)
                    self.cpu_drafter.feed(bonus)
                    self.ane_drafter.feed(bonus)
                    last_tok = bonus
                    stats['ngram_accepted'] += len(draft_tokens)

                self.ngram.update(
                    all_context[max(0, len(all_context) - NGRAM_N - 5):])
                self.ngram_window.append(1 if accepted > 0 else 0)
                stats['cycles'] += 1
                continue

            # -- CPU K=1 + ANE K=1: both draft the SAME next position --
            # CPU drafts first (~10ms). ANE drafts in background thread
            # during GPU verify (~139ms), so its 35ms is fully hidden.
            # Both drafters see the same context and independently predict
            # the next token. ANE runs on dedicated silicon (zero GPU cost).
            cpu_draft = self.cpu_drafter.draft_one()
            stats['cpu_drafted'] += 1

            # Start ANE draft in background (runs during GPU verify)
            ane_result = [None]
            def _ane_draft():
                ane_result[0] = self.ane_drafter.draft_one()
            ane_thread = threading.Thread(target=_ane_draft)
            ane_thread.start()
            stats['ane_drafted'] += 1

            # Verify CPU draft: feed [last_tok, cpu_draft] to 70B (batch of 2)
            # logits[0] = 70B prediction after last_tok -> verify cpu_draft
            # logits[1] = 70B prediction after cpu_draft -> bonus if match
            verify_mx = mx.array([[last_tok, cpu_draft]])
            vlogits = self.model(verify_mx, cache=cache)
            mx.eval(vlogits)

            # Wait for ANE draft (should already be done — 35ms < 139ms GPU)
            ane_thread.join()
            ane_draft = ane_result[0]

            target_0 = mx.argmax(vlogits[0, 0, :], axis=-1).item()

            if target_0 == cpu_draft:
                # CPU MATCH: 2 tokens (draft + bonus)
                bonus = mx.argmax(vlogits[0, 1, :], axis=-1).item()
                generated.append(cpu_draft)
                all_context.append(cpu_draft)
                generated.append(bonus)
                all_context.append(bonus)
                # CPU already advanced past cpu_draft. Feed bonus.
                self.cpu_drafter.feed(bonus)
                # ANE drafted same position. Correct it to match accepted.
                self.ane_drafter.correct(1, cpu_draft)
                self.ane_drafter.feed(bonus)
                last_tok = bonus
                stats['cpu_accepted'] += 1
            elif ane_draft is not None and target_0 == ane_draft:
                # CPU missed, but ANE got it right!
                # GPU cache has [last_tok, cpu_draft] — wrong draft in cache.
                # Trim cpu_draft, re-verify with ane_draft for bonus.
                trim_prompt_cache(cache, 1)
                re_verify = mx.array([[last_tok, ane_draft]])
                re_logits = self.model(re_verify, cache=cache)
                mx.eval(re_logits)
                bonus = mx.argmax(re_logits[0, 1, :], axis=-1).item()

                generated.append(ane_draft)
                all_context.append(ane_draft)
                generated.append(bonus)
                all_context.append(bonus)
                # CPU had cpu_draft wrong. Correct to ane_draft + bonus.
                self.cpu_drafter.correct(1, ane_draft)
                self.cpu_drafter.feed(bonus)
                # ANE already advanced past ane_draft. Feed bonus.
                self.ane_drafter.feed(bonus)
                last_tok = bonus
                stats['ane_accepted'] += 1
            else:
                # BOTH MISSED: target_0 is 70B's correction
                trim_prompt_cache(cache, 1)
                generated.append(target_0)
                all_context.append(target_0)
                # CPU has cpu_draft wrong. Correct.
                self.cpu_drafter.correct(1, target_0)
                # ANE has ane_draft wrong. Correct.
                self.ane_drafter.correct(1, target_0)
                last_tok = target_0

            self.ngram.update(
                all_context[max(0, len(all_context) - NGRAM_N - 3):])
            stats['cycles'] += 1

        elapsed = time.perf_counter() - t0
        tps = len(generated) / elapsed if elapsed > 0 else 0
        text = self.tokenizer.decode(generated)
        for eos in ("<|eot_id|>", "<|end_of_text|>", "</s>",
                    "<|start_header_id|>", "<|end_header_id|>",
                    "<|begin_of_text|>"):
            text = text.replace(eos, "")
        text = re.sub(r'\s*(assistant|user|system)\s*$', '', text).strip()
        self.request_count += 1

        total_d = stats['ngram_drafted'] + stats['cpu_drafted'] + stats['ane_drafted']
        total_a = stats['ngram_accepted'] + stats['cpu_accepted'] + stats['ane_accepted']

        return {
            "text": text, "tokens": len(generated),
            "tps": round(tps, 1), "elapsed": round(elapsed, 2),
            "accept_rate": round(total_a / max(1, total_d) * 100, 1),
            **stats, "ngram_table_size": self.ngram.size,
            "request_num": self.request_count,
        }


engine = None
engine_lock = threading.Lock()


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path in ("/v1/models", "/v1/models/"):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(json.dumps({"object": "list", "data": [
                {"id": MODEL_ID, "object": "model",
                 "owned_by": "local"}]}).encode())
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path == "/v1/config":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length))
            if "suppress_threshold" in body:
                engine.suppress_threshold = body["suppress_threshold"]
            if "reenable_threshold" in body:
                engine.reenable_threshold = body["reenable_threshold"]
            resp = {"suppress_threshold": engine.suppress_threshold,
                    "reenable_threshold": engine.reenable_threshold}
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(json.dumps(resp).encode())
        elif self.path == "/v1/chat/completions":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length))
            with engine_lock:
                result = engine.generate(
                    body.get("messages", []),
                    max_tokens=body.get("max_tokens", 1024))
            response = {
                "id": f"chatcmpl-{uuid.uuid4()}",
                "object": "chat.completion",
                "model": MODEL_ID, "created": int(time.time()),
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant",
                                         "content": result["text"]}}],
                "usage": {"completion_tokens": result["tokens"]},
                "x_spec_decode": {k: v for k, v in result.items()
                                  if k != "text"},
            }
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())
        else:
            self.send_error(404)

    def log_message(self, fmt, *args):
        pass


def benchmark_mode():
    prompts = [
        "Explain step by step how backpropagation computes gradients in a neural network",
        "Write a Python function that finds the longest palindromic substring",
        "What are the key differences between TCP and UDP protocols",
        "Prove that the square root of 2 is irrational",
        "Explain how a B-tree maintains balance during insertions",
        "What happens at the hardware level when a CPU encounters a branch misprediction",
        "Derive the formula for compound interest from first principles",
        "Compare gradient descent, Adam, and SGD with momentum",
        "Explain why quicksort has O(n log n) average but O(n^2) worst case",
        "How does the transformer attention mechanism scale with sequence length",
    ]

    print(f"\n{'='*70}")
    print(f"  BENCHMARK: 10 reasoning prompts, 100 tokens each")
    print(f"  3-drafter: N-gram + CPU K=1 + ANE K=1. ONE GPU pass per cycle.")
    print(f"{'='*70}")
    print(f"{'#':>3} {'tok/s':>7} {'Tokens':>7} {'NGram':>10} "
          f"{'CPU':>10} {'ANE':>10} {'Accept':>7}")
    print("-" * 70)

    all_tps = []
    for i, prompt in enumerate(prompts):
        r = engine.generate([{"role": "user", "content": prompt}],
                            max_tokens=100)
        all_tps.append(r['tps'])
        ng = (f"{r['ngram_accepted']}/{r['ngram_drafted']}"
              if r['ngram_drafted'] > 0 else "--")
        cpu = (f"{r['cpu_accepted']}/{r['cpu_drafted']}"
               if r['cpu_drafted'] > 0 else "--")
        ane = (f"{r['ane_accepted']}/{r['ane_drafted']}"
               if r['ane_drafted'] > 0 else "--")
        print(f"{i+1:>3} {r['tps']:>6.1f} {r['tokens']:>7} {ng:>10} "
              f"{cpu:>10} {ane:>10} {r['accept_rate']:>6.1f}%")

    avg = np.mean(all_tps)
    print(f"\n  Average: {avg:.1f} tok/s")
    print(f"  vs Q4 3.1 baseline (6.9): {avg/6.9:.2f}x")
    print(f"  vs Q3 3.3 baseline (8.8): {avg/8.8:.2f}x")
    print(f"  vs CPU-only server (10.6): {avg/10.6:.2f}x")

    if avg >= 12.0:
        print(f"\n  >>> STRETCH: {avg:.1f} >= 12.0 tok/s. SHIP. <<<")
    elif avg >= 10.0:
        print(f"\n  >>> SHIP GATE PASSED: {avg:.1f} >= 10.0 tok/s. <<<")
    else:
        print(f"\n  >>> Below 10.0: {avg:.1f} tok/s. <<<")


def main():
    global engine
    engine = Engine()
    if "--benchmark" in sys.argv:
        benchmark_mode()
        return
    server = ThreadedHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"  http://127.0.0.1:{PORT}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
