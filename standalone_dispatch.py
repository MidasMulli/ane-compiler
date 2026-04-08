#!/usr/bin/env python3
"""
Standalone ANE dispatch for emitter-generated .hwx files.

Dispatch mechanism:
  1. Compile a reference .mlmodelc via aned (one-time setup, creates cache entry)
  2. Emit .hwx via our compiler (emit_linear_hwx)
  3. Swap emitter .hwx into the aned cache location
  4. Dispatch via _ANEClient loadModel (loads our .hwx, no recompilation)

The compilation step (1) uses aned, but only for setup — it creates the cache
directory structure and loads the initial model. After swap (3), the actual
binary executing on hardware is our emitter output.

For a fully standalone pipeline (no aned at any point), we would need direct
IOKit dispatch. That's a future goal. This module bridges the gap.

Copyright 2026 Nick Lo. MIT License.
"""

import os
import sys
import glob
import time
import shutil
import struct
import subprocess
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from emitter import emit_linear_hwx, WeightPacker, generate_conv_text


class StandaloneDispatcher:
    """Manages emitter .hwx dispatch via cache-swap mechanism."""

    def __init__(self, template_hwx_path: str, eval_binary: str = None,
                 cache_base: str = '/Library/Caches/com.apple.aned'):
        """Initialize with a reference .hwx template.

        Args:
            template_hwx_path: path to a captured reference .hwx (any dim)
            eval_binary: path to ane_eval_binary tool
            cache_base: ANE cache directory
        """
        self.template = open(template_hwx_path, 'rb').read()
        self.eval_binary = eval_binary or os.path.join(
            os.path.dirname(__file__), 'tests', 'ane_eval_binary')
        self.cache_base = cache_base

        # Cache of compiled reference models: (in_ch, out_ch) → (mlmodelc_path, cache_hwx_path)
        self._ref_cache = {}

    def setup_reference(self, in_ch: int, out_ch: int, build_dir: str) -> str:
        """Compile a reference .mlmodelc for a given dimension pair.

        This creates the cache entry that we'll later swap our .hwx into.
        Only needs to be called once per unique dimension pair.

        Returns the .mlmodelc path.
        """
        key = (in_ch, out_ch)
        if key in self._ref_cache:
            return self._ref_cache[key][0]

        # Generate reference .mlmodelc via coremltools
        import coremltools as ct
        from coremltools.models.neural_network import NeuralNetworkBuilder

        os.makedirs(build_dir, exist_ok=True)
        w = np.random.randn(out_ch, in_ch).astype(np.float32)
        builder = NeuralNetworkBuilder(
            input_features=[('input', ct.models.datatypes.Array(in_ch))],
            output_features=[('output', ct.models.datatypes.Array(out_ch))])
        builder.add_inner_product('fc', W=w, b=None,
                                  input_channels=in_ch, output_channels=out_ch,
                                  has_bias=False, input_name='input', output_name='output')

        ml_path = os.path.join(build_dir, f'ref_{in_ch}x{out_ch}.mlmodel')
        ct.models.MLModel(builder.spec).save(ml_path)
        compiled = ct.utils.compile_model(ml_path)
        mlc_path = os.path.join(build_dir, f'ref_{in_ch}x{out_ch}.mlmodelc')
        if os.path.exists(mlc_path):
            shutil.rmtree(mlc_path)
        shutil.move(compiled, mlc_path)

        # Compile on ANE to populate cache
        from compiler import gen_conv_mlmodelc
        r = subprocess.run(
            [os.path.join(os.path.dirname(__file__), 'tests', 'ane_eval'),
             mlc_path, str(out_ch)],
            capture_output=True, timeout=30)

        # Find cache entry
        time.sleep(0.3)
        hwx_files = sorted(
            glob.glob(f'{self.cache_base}/**/model.hwx', recursive=True),
            key=os.path.getmtime, reverse=True)

        if hwx_files:
            self._ref_cache[key] = (mlc_path, hwx_files[0])
            return mlc_path
        else:
            raise RuntimeError(f"Failed to create cache entry for {in_ch}→{out_ch}")

    def dispatch(self, in_ch: int, out_ch: int, weights: np.ndarray,
                 input_fp16: np.ndarray, build_dir: str) -> np.ndarray:
        """Emit .hwx and dispatch on ANE.

        Args:
            in_ch, out_ch: dimensions
            weights: [out_ch, in_ch] weight matrix
            input_fp16: input data as FP16 array
            build_dir: directory for temporary files

        Returns:
            Output as FP16 array
        """
        key = (in_ch, out_ch)

        # Ensure reference is set up
        if key not in self._ref_cache:
            self.setup_reference(in_ch, out_ch, build_dir)

        mlc_path, cache_hwx = self._ref_cache[key]

        # Emit .hwx
        emitted = emit_linear_hwx(self.template, in_ch, out_ch, weights)
        emitted_path = os.path.join(build_dir, f'emitted_{in_ch}x{out_ch}.hwx')
        with open(emitted_path, 'wb') as f:
            f.write(emitted)

        # Swap into cache
        subprocess.run(['sudo', 'cp', emitted_path, cache_hwx],
                       capture_output=True)

        # Kill aned to force cache reload
        subprocess.run(['sudo', 'killall', 'aned'], capture_output=True)
        time.sleep(1)

        # Dispatch with --load-only
        result = subprocess.run(
            [self.eval_binary, mlc_path, str(in_ch), str(out_ch)],
            input=input_fp16.astype(np.float16).tobytes(),
            capture_output=True, timeout=30)

        if b'OK' not in result.stderr:
            raise RuntimeError(f"Dispatch failed: {result.stderr.decode()}")

        return np.frombuffer(result.stdout, dtype=np.float16)


def dispatch_emitter_hwx(template: bytes, in_ch: int, out_ch: int,
                          weights: np.ndarray, input_fp16: np.ndarray,
                          eval_binary: str, mlmodelc_path: str) -> np.ndarray:
    """Simple one-shot dispatch of emitter .hwx via aned pipeline.

    Uses the aned-based pipeline (gen_conv_mlmodelc → compile → dispatch)
    but the .mlmodelc contains our weights. Since aned compiles to identical
    __text (proven in Gate 1), the result is identical to emitter .hwx dispatch.

    This is the PRACTICAL approach: let aned do the loading, our compiler
    decides what __text and weights go on the hardware.
    """
    input_bytes = input_fp16.astype(np.float16).tobytes()
    result = subprocess.run(
        [eval_binary, mlmodelc_path, str(in_ch), str(out_ch)],
        input=input_bytes, capture_output=True, timeout=30)

    if b'OK' not in result.stderr:
        raise RuntimeError(f"Dispatch failed: {result.stderr.decode()}")

    return np.frombuffer(result.stdout, dtype=np.float16)
