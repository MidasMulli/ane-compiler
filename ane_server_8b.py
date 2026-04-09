#!/usr/bin/env python3
"""
8B Q8 ANE Extraction Server.

Loads the 72 Q8 CoreML models and serves extraction via HTTP on :8891.
Replaces the 1B ANE server for Subconscious extraction.

Endpoints:
  POST /analyze  {"prompt": "...", "max_tokens": 400} -> {"result": "..."}
  GET  /health   -> {"status": "ok", "model": "8B-Q8", "tok_s": 7.9}

Copyright 2026 Nick Lo. MIT License.
"""

import json
import logging
import time
import sys
import os
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
import threading

sys.path.insert(0, os.path.dirname(__file__))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("ane_server_8b")

_extractor = None
_lock = threading.Lock()
_tasks_completed = 0
_start_time = 0
# Main 35 close: expose the same in-flight fields as the 72B server so the
# subconscious_daemon's health monitor can detect when the ANE is actively
# generating and the visualization can light up the 8B "thinking" dot.
_status = "idle"  # "idle" | "generating"
_current_request_started_at = None
_current_request_path = None
_current_request_n_chars = 0


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # Suppress default HTTP logging

    def do_GET(self):
        if self.path == "/health":
            uptime = time.time() - _start_time
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            payload = {
                # Daemon health monitor reads `status` to detect in-flight
                # state. "ok" is the legacy idle marker; "generating" trips
                # the upstream_thinking_started event in the daemon parser.
                "status": _status if _status != "idle" else "ok",
                "model": "Llama-3.1-8B-Instruct-Q8",
                "backend": "ane-compiler-q8",
                "uptime": round(uptime, 1),
                "tasks_completed": _tasks_completed,
                "current_request_started_at": _current_request_started_at,
                "current_request_path": _current_request_path,
                "current_request_n_chars": _current_request_n_chars,
            }
            self.wfile.write(json.dumps(payload).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        global _tasks_completed, _status, _current_request_started_at
        global _current_request_path, _current_request_n_chars
        if self.path == "/v1/embed":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            data = json.loads(body)
            text = data.get("text", "")
            pooling = data.get("pooling", "mean")

            with _lock:
                _status = "generating"
                _current_request_started_at = time.time()
                _current_request_path = "/v1/embed"
                _current_request_n_chars = len(text)
                t0 = time.perf_counter()
                try:
                    vec, n_tokens = _extractor.embed_text(text, pooling=pooling)
                    elapsed_ms = (time.perf_counter() - t0) * 1000
                    _tasks_completed += 1
                finally:
                    _status = "idle"
                    _current_request_started_at = None
                    _current_request_path = None
                    _current_request_n_chars = 0

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({
                "embedding": vec.tolist(),
                "latency_ms": round(elapsed_ms, 2),
                "dim": int(vec.shape[0]),
                "n_tokens": int(n_tokens),
                "pooling": pooling,
            }).encode())
            return

        if self.path == "/analyze":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            data = json.loads(body)
            prompt = data.get("prompt", "")
            max_tokens = data.get("max_tokens", 400)

            with _lock:
                _status = "generating"
                _current_request_started_at = time.time()
                _current_request_path = "/analyze"
                _current_request_n_chars = len(prompt)
                t0 = time.perf_counter()
                try:
                    result = _extractor.generate(prompt, max_tokens=max_tokens)
                    elapsed_ms = (time.perf_counter() - t0) * 1000
                    _tasks_completed += 1
                finally:
                    _status = "idle"
                    _current_request_started_at = None
                    _current_request_path = None
                    _current_request_n_chars = 0

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({
                "result": result,
                "elapsed_ms": round(elapsed_ms, 1),
            }).encode())
        else:
            self.send_response(404)
            self.end_headers()


def main():
    global _extractor, _start_time

    port = 8891
    log.info("Starting 8B Q8 ANE Extraction Server on :%d", port)

    from ane_extractor_8b import ANEExtractor8B
    _extractor = ANEExtractor8B()
    _extractor.load()

    _start_time = time.time()

    class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
        daemon_threads = True

    server = ThreadedHTTPServer(("0.0.0.0", port), Handler)
    log.info("Ready on http://localhost:%d", port)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("Shutting down")
        server.shutdown()


if __name__ == "__main__":
    main()
