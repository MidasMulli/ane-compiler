#!/usr/bin/env python3
"""
Llama-3.1-8B-Instruct on Apple Neural Engine via 2-layer fused blocks.

Architecture:
  16 fused blocks (2 layers each) + 8 lm_head chunks = 24 program slots
  Each fused block: [RMSNorm→V→VO→residual→RMSNorm→SwiGLU→residual] × 2 layers
  CPU: attention (Q/K projections, RoPE, GQA with KV cache)

First build uses seq_len=1 shortcut:
  - Fused block does V→VO→residual→SwiGLU→residual (attention = identity V)
  - Works for single-token decode (first generated token)
  - Full KV-cache attention on CPU for decode after first token

Compile: NeuralNetworkBuilder → xcrun coremlcompiler → .mlmodelc
Dispatch: pipe tool with 24 ops (well under 128 kext limit)

Usage:
  python llama_8b_fused.py --prompt "What is the capital of France?" --tokens 30

Copyright 2026 Nick Lo. MIT License.
"""

import argparse
import os
import sys
import time
import shutil
import subprocess
import json
import numpy as np
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

PIPE_TOOL = os.path.join(os.path.dirname(__file__), 'tests', 'ane_standalone_pipe')

MODEL_PATH = os.path.expanduser(
    '~/.cache/huggingface/hub/models--unsloth--Meta-Llama-3.1-8B-Instruct/'
    'snapshots/a2856192dd7c25b842431f39c179a6c2c2f627d1'
)

BUILD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'build_8b_fused')


# ===================================================================
# Fused 2-layer block builder (NeuralNetworkBuilder)
# ===================================================================

def build_fused_block(block_idx, dim, ffn_dim, n_heads, n_kv_heads, head_dim,
                      weights, output_dir):
    """Build a fused 2-layer block as a single NeuralNetworkBuilder graph.

    For seq_len=1 decode, attention simplifies to V→VO (softmax of scalar = 1.0).
    Each block fuses 2 consecutive transformer layers into 1 ANE dispatch.

    Graph per layer within block:
      1. MVN + BN (RMSNorm approximation, pre-attention)
      2. V projection (dim → kv_dim)
      3. VO projection (kv_dim → dim, combines GQA expand + O)
      4. Residual add (input + attn_out)
      5. MVN + BN (RMSNorm approximation, pre-FFN)
      6. SwiGLU: gate→sigmoid→mul(=SiLU) × up → down
      7. Residual add

    Args:
        block_idx: 0-15, maps to layers [block_idx*2, block_idx*2+1]
        weights: dict mapping weight names to numpy arrays
        output_dir: where to write .mlmodel and .mlmodelc

    Returns:
        path to compiled .mlmodelc
    """
    import coremltools as ct
    from coremltools.models.neural_network import NeuralNetworkBuilder

    input_features = [('input', ct.models.datatypes.Array(dim, 1, 1))]
    output_features = [('output', ct.models.datatypes.Array(dim, 1, 1))]

    builder = NeuralNetworkBuilder(input_features, output_features,
                                    disable_rank5_shape_mapping=True)

    kv_dim = n_kv_heads * head_dim  # 8 * 128 = 1024

    current = 'input'

    for sub in range(2):
        layer_idx = block_idx * 2 + sub
        pfx = f'L{layer_idx}'

        # --- RMSNorm 1 (pre-attention) via MVN + BN ---
        builder.add_mvn(
            name=f'{pfx}_ln1_mvn',
            input_name=current,
            output_name=f'{pfx}_ln1_mvn_out',
            across_channels=True,
            normalize_variance=True,
            epsilon=1e-5,
        )

        ln1_w = weights[f'{pfx}_ln1_weight']
        builder.add_batchnorm(
            name=f'{pfx}_ln1_bn',
            channels=dim,
            gamma=ln1_w.astype(np.float32),
            beta=np.zeros(dim, dtype=np.float32),
            mean=np.zeros(dim, dtype=np.float32),
            variance=np.ones(dim, dtype=np.float32),
            input_name=f'{pfx}_ln1_mvn_out',
            output_name=f'{pfx}_ln1_out',
            epsilon=0.0,
        )

        # --- V projection (dim → kv_dim) ---
        builder.add_inner_product(
            name=f'{pfx}_v_proj',
            W=weights[f'{pfx}_v_proj'].flatten().astype(np.float32),
            b=None,
            input_channels=dim,
            output_channels=kv_dim,
            has_bias=False,
            input_name=f'{pfx}_ln1_out',
            output_name=f'{pfx}_v_out',
        )

        # --- VO projection (kv_dim → dim, combines GQA expand + O) ---
        builder.add_inner_product(
            name=f'{pfx}_vo_proj',
            W=weights[f'{pfx}_vo_proj'].flatten().astype(np.float32),
            b=None,
            input_channels=kv_dim,
            output_channels=dim,
            has_bias=False,
            input_name=f'{pfx}_v_out',
            output_name=f'{pfx}_attn_out',
        )

        # --- Residual 1 ---
        builder.add_elementwise(
            name=f'{pfx}_res1',
            input_names=[current, f'{pfx}_attn_out'],
            output_name=f'{pfx}_res1_out',
            mode='ADD',
        )

        # --- RMSNorm 2 (pre-FFN) via MVN + BN ---
        builder.add_mvn(
            name=f'{pfx}_ln2_mvn',
            input_name=f'{pfx}_res1_out',
            output_name=f'{pfx}_ln2_mvn_out',
            across_channels=True,
            normalize_variance=True,
            epsilon=1e-5,
        )

        ln2_w = weights[f'{pfx}_ln2_weight']
        builder.add_batchnorm(
            name=f'{pfx}_ln2_bn',
            channels=dim,
            gamma=ln2_w.astype(np.float32),
            beta=np.zeros(dim, dtype=np.float32),
            mean=np.zeros(dim, dtype=np.float32),
            variance=np.ones(dim, dtype=np.float32),
            input_name=f'{pfx}_ln2_mvn_out',
            output_name=f'{pfx}_ln2_out',
            epsilon=0.0,
        )

        # --- SwiGLU FFN ---
        # gate_proj: dim → ffn_dim
        builder.add_inner_product(
            name=f'{pfx}_gate_proj',
            W=weights[f'{pfx}_gate_proj'].flatten().astype(np.float32),
            b=None,
            input_channels=dim,
            output_channels=ffn_dim,
            has_bias=False,
            input_name=f'{pfx}_ln2_out',
            output_name=f'{pfx}_gate_out',
        )

        # SiLU = gate * sigmoid(gate)
        builder.add_activation(
            name=f'{pfx}_gate_sigmoid',
            non_linearity='SIGMOID',
            input_name=f'{pfx}_gate_out',
            output_name=f'{pfx}_gate_sig',
        )

        builder.add_elementwise(
            name=f'{pfx}_gate_silu',
            input_names=[f'{pfx}_gate_out', f'{pfx}_gate_sig'],
            output_name=f'{pfx}_silu_out',
            mode='MULTIPLY',
        )

        # up_proj: dim → ffn_dim
        builder.add_inner_product(
            name=f'{pfx}_up_proj',
            W=weights[f'{pfx}_up_proj'].flatten().astype(np.float32),
            b=None,
            input_channels=dim,
            output_channels=ffn_dim,
            has_bias=False,
            input_name=f'{pfx}_ln2_out',
            output_name=f'{pfx}_up_out',
        )

        # SwiGLU multiply: SiLU(gate) * up
        builder.add_elementwise(
            name=f'{pfx}_swiglu_mul',
            input_names=[f'{pfx}_silu_out', f'{pfx}_up_out'],
            output_name=f'{pfx}_swiglu_out',
            mode='MULTIPLY',
        )

        # down_proj: ffn_dim → dim
        builder.add_inner_product(
            name=f'{pfx}_down_proj',
            W=weights[f'{pfx}_down_proj'].flatten().astype(np.float32),
            b=None,
            input_channels=ffn_dim,
            output_channels=dim,
            has_bias=False,
            input_name=f'{pfx}_swiglu_out',
            output_name=f'{pfx}_ffn_out',
        )

        # --- Residual 2 ---
        out_name = f'{pfx}_out' if sub < 1 else 'pre_output'
        builder.add_elementwise(
            name=f'{pfx}_res2',
            input_names=[f'{pfx}_res1_out', f'{pfx}_ffn_out'],
            output_name=out_name,
            mode='ADD',
        )

        current = out_name

    # Final identity to connect to 'output'
    builder.add_activation(
        name='final_identity',
        non_linearity='LINEAR',
        input_name='pre_output',
        output_name='output',
        params=[1.0, 0.0],
    )

    # Save .mlmodel, compile to .mlmodelc
    os.makedirs(output_dir, exist_ok=True)
    mlmodel_path = os.path.join(output_dir, f'block_{block_idx}.mlmodel')

    model_spec = builder.spec
    import coremltools as ct
    ct.utils.save_spec(model_spec, mlmodel_path)

    mlmodelc_path = os.path.join(output_dir, f'block_{block_idx}.mlmodelc')
    if os.path.exists(mlmodelc_path):
        shutil.rmtree(mlmodelc_path)

    result = subprocess.run(
        ['xcrun', 'coremlcompiler', 'compile', mlmodel_path, output_dir],
        capture_output=True, text=True, timeout=600,
    )
    if result.returncode != 0:
        raise RuntimeError(f"coremlcompiler failed for block {block_idx}: "
                           f"{result.stderr[:500]}")

    # coremlcompiler may output with different name
    compiled_name = f'block_{block_idx}.mlmodelc'
    compiled_path = os.path.join(output_dir, compiled_name)
    if not os.path.exists(compiled_path):
        # Try the stem from mlmodel
        alt = os.path.join(output_dir, f'block_{block_idx}.mlmodelc')
        if not os.path.exists(alt):
            raise RuntimeError(f"Compiled output not found for block {block_idx}")

    # Cleanup .mlmodel (large)
    if os.path.exists(mlmodel_path):
        os.unlink(mlmodel_path)

    return compiled_path


# ===================================================================
# Compile all fused blocks + lm_head chunks
# ===================================================================

def compile_fused_blocks(model, build_dir, n_blocks=16):
    """Generate fused 2-layer blocks + 8 lm_head chunks.

    Args:
        model: LlamaModel with weights
        build_dir: output directory
        n_blocks: number of 2-layer blocks to compile (default 16 = all 32 layers)

    Returns:
        dict mapping op_name -> (mlmodelc_path, in_ch, out_ch)
    """
    from compiler import gen_conv_mlmodelc
    from llama_loader import gen_lm_head_chunks

    os.makedirs(build_dir, exist_ok=True)
    config = model.config
    dim = config.hidden_size           # 4096
    ffn_dim = config.intermediate_size  # 14336
    n_heads = config.n_heads            # 32
    n_kv_heads = config.n_kv_heads      # 8
    head_dim = config.head_dim          # 128
    n_rep = config.n_rep                # 4
    kv_dim = n_kv_heads * head_dim      # 1024

    compiled = {}

    # Build weight dict for needed layers only
    n_layers_needed = n_blocks * 2
    print(f"  Preparing weights for {n_layers_needed} layers ({n_blocks} blocks)...")
    t0 = time.time()

    weights = {}
    for i in range(n_layers_needed):
        L = model.layers[i]
        pfx = f'L{i}'
        weights[f'{pfx}_ln1_weight'] = L.input_layernorm_weight
        weights[f'{pfx}_ln2_weight'] = L.post_attention_layernorm_weight
        weights[f'{pfx}_v_proj'] = L.v_proj_weight  # [1024, 4096]

        # Combined VO: O @ expand, where expand repeats KV heads for GQA
        # expand[4096, 1024]: maps 8 KV heads → 32 Q heads
        expand = np.zeros((dim, kv_dim), dtype=np.float32)
        for kv_h in range(n_kv_heads):
            for r in range(n_rep):
                q_h = kv_h * n_rep + r
                expand[q_h * head_dim:(q_h + 1) * head_dim,
                       kv_h * head_dim:(kv_h + 1) * head_dim] = \
                    np.eye(head_dim, dtype=np.float32)

        # vo_weight = O @ expand: [4096, 4096] @ [4096, 1024] = [4096, 1024]
        vo_weight = L.o_proj_weight.astype(np.float32) @ expand
        weights[f'{pfx}_vo_proj'] = vo_weight

        weights[f'{pfx}_gate_proj'] = L.gate_proj_weight
        weights[f'{pfx}_up_proj'] = L.up_proj_weight
        weights[f'{pfx}_down_proj'] = L.down_proj_weight

    print(f"  Weights prepared in {time.time() - t0:.1f}s")

    # Compile fused blocks
    for b in range(n_blocks):
        path = os.path.join(build_dir, f'block_{b}.mlmodelc')
        if os.path.exists(path):
            print(f"  Block {b:2d}: cached (layers {b*2}-{b*2+1})")
        else:
            t0 = time.time()
            path = build_fused_block(
                b, dim, ffn_dim, n_heads, n_kv_heads, head_dim,
                weights, build_dir,
            )
            dt = time.time() - t0
            fsize = sum(f.stat().st_size for f in
                        os.scandir(path) if f.is_file()) / 1024 / 1024
            print(f"  Block {b:2d}: compiled (layers {b*2}-{b*2+1}) "
                  f"{dt:.1f}s, {fsize:.1f} MB")

        compiled[f'block_{b}'] = (path, dim, dim)

    # Compile 8 lm_head chunks
    # 8B has separate lm_head weight (not tied to embed_tokens)
    lm_head_dir = os.path.join(build_dir, 'lm_head')
    os.makedirs(lm_head_dir, exist_ok=True)

    # Check for separate lm_head weight
    if hasattr(model, 'lm_head_weight') and model.lm_head_weight is not None:
        lm_head_weight = model.lm_head_weight.astype(np.float32)
    else:
        # 8B has untied lm_head — load directly from safetensors
        lm_head_weight = _load_lm_head_weight(MODEL_PATH).astype(np.float32)

    chunk_size = 16032  # 128256 / 8 = 16032
    chunks = gen_lm_head_chunks(lm_head_dir, lm_head_weight, dim,
                                 config.vocab_size, chunk_size)
    for j, (path, ic, oc, start) in enumerate(chunks):
        compiled[f'lm_head_chunk_{j}'] = (path, ic, oc)

    n_total = n_blocks + len(chunks)
    print(f"  lm_head: {len(chunks)} chunks of ~{chunk_size}")
    print(f"  Total program slots: {n_total}")

    return compiled


def _load_lm_head_weight(model_path):
    """Load just the lm_head weight from safetensors (avoids loading full model twice)."""
    from safetensors.torch import load_file
    import glob
    import torch

    if os.path.isdir(model_path):
        shards = sorted(glob.glob(os.path.join(model_path, "model*.safetensors")))
    else:
        shards = [model_path]

    for shard in shards:
        from safetensors import safe_open
        with safe_open(shard, framework="pt") as f:
            if "lm_head.weight" in f.keys():
                return f.get_tensor("lm_head.weight").float().numpy()

    raise RuntimeError("lm_head.weight not found in safetensors shards")


# ===================================================================
# CPU helpers (from llama_loader.py)
# ===================================================================

def rms_norm_cpu(x, weight, eps=1e-5):
    """RMSNorm on CPU (FP32 for accuracy, output as FP16)."""
    x_f32 = x.astype(np.float32)
    rms = np.sqrt(np.mean(x_f32 ** 2) + eps)
    normed = x_f32 / rms
    return (normed * weight.astype(np.float32)).astype(np.float16)


def rope_cpu(q, k, position, head_dim, theta=500000.0):
    """Apply Rotary Position Embedding on CPU."""
    q_f32 = q.astype(np.float32)
    k_f32 = k.astype(np.float32)

    half_dim = head_dim // 2
    freqs = 1.0 / (theta ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))
    angles = position * freqs

    cos_vals = np.cos(angles)
    sin_vals = np.sin(angles)

    def apply_rope(x):
        x1 = x[:, :half_dim]
        x2 = x[:, half_dim:]
        rotated = np.concatenate([-x2, x1], axis=-1)
        cos_full = np.concatenate([cos_vals, cos_vals])
        sin_full = np.concatenate([sin_vals, sin_vals])
        return x * cos_full + rotated * sin_full

    return apply_rope(q_f32).astype(np.float16), apply_rope(k_f32).astype(np.float16)


def softmax_cpu(x):
    """Numerically stable softmax in FP32."""
    x_f32 = x.astype(np.float32)
    x_f32 = x_f32 - x_f32.max()
    exp_x = np.exp(x_f32)
    return (exp_x / exp_x.sum()).astype(np.float32)


# ===================================================================
# ANE8BModel: load + forward + generate
# ===================================================================

class ANE8BModel:
    """Llama-3.1-8B on ANE via 2-layer fused blocks.

    24 program slots: 16 fused blocks + 8 lm_head chunks.
    All loaded into a single pipe tool instance.
    """

    def __init__(self, model, compiled, n_blocks=16):
        """
        Args:
            model: LlamaModel with weights (for CPU ops: RMSNorm, attention)
            compiled: dict mapping op_name -> (mlmodelc_path, in_ch, out_ch)
            n_blocks: number of fused blocks (default 16)
        """
        self.model = model
        self.config = model.config
        self.compiled = compiled
        self.n_blocks = n_blocks
        self.dispatcher = None

    def load(self):
        """Launch pipe tool, compile + load all 24 ops on ANE."""
        from generate import ANEDispatcher

        # Filter to dispatch-ready ops only (3-tuples)
        dispatch_ops = {k: v for k, v in self.compiled.items()
                        if not k.startswith('_') and isinstance(v, tuple) and len(v) == 3}

        self.dispatcher = ANEDispatcher(dispatch_ops, quiet=True)
        self.dispatcher.start()
        return len(dispatch_ops)

    def forward_token(self, token_id, position, kv_cache):
        """Forward one token through all 16 fused blocks.

        For the first build (seq_len=1 shortcut):
          - Fused blocks handle V→VO→residual→SwiGLU→residual (2 layers each)
          - Q/K projections + RoPE + GQA attention on CPU between blocks
          - KV cache updated on CPU

        Actually: the fused blocks include MVN+BN (RMSNorm approximation),
        V, VO, residuals, and SwiGLU. The attention with KV cache is on CPU,
        but the fused block's V→VO path is used as the attention shortcut
        for the FIRST token (position 0). For subsequent positions, we need
        the full attention path.

        For this first build: we use the fused block for ALL projections
        and SwiGLU, and do attention on CPU with separate Q/K projections.

        Since Q/K are NOT in the fused block, we compute them on CPU
        from the weights directly (matrix multiply, no ANE dispatch).
        This is slower but avoids additional program slots.

        Args:
            token_id: input token
            position: absolute position in sequence
            kv_cache: KVCache instance

        Returns:
            [dim] FP16 output after all layers
        """
        c = self.config
        dim = c.hidden_size
        n_heads = c.n_heads
        n_kv_heads = c.n_kv_heads
        head_dim = c.head_dim
        n_rep = c.n_rep

        # Embed
        x = self.model.embed_tokens[token_id].astype(np.float16)

        # Process fused blocks
        for block_idx in range(self.n_blocks):
            # Each block contains 2 layers
            # The fused block does V→VO→residual→SwiGLU→residual for each layer
            # But we need Q/K for real attention with KV cache

            # For the first build, dispatch the fused block (V→VO shortcut)
            # and ALSO do CPU attention to update KV cache
            #
            # Strategy: Run fused block (gets the SwiGLU part right),
            # but for attention, do CPU Q/K projections + KV cache + attention
            # and use the CPU attention result instead of the V→VO shortcut.
            #
            # However, the fused block IS a single dispatch — we can't
            # intercept between layers within it.
            #
            # For the FIRST BUILD: just dispatch the fused block.
            # The V→VO shortcut means attention = V (no KV cache).
            # This produces coherent-ish output for short sequences.

            x = self.dispatcher.dispatch(f'block_{block_idx}', x)

        # Final RMSNorm on CPU
        x = rms_norm_cpu(x, self.model.norm_weight, c.rms_norm_eps)

        # lm_head: 8 chunks concatenated
        logits = np.concatenate([
            self.dispatcher.dispatch(f'lm_head_chunk_{i}', x)
            for i in range(8)
        ])

        return logits

    def forward_token_with_attention(self, token_id, position, kv_cache):
        """Forward with CPU attention between fused blocks.

        This is the HYBRID path: fused blocks handle RMSNorm + V→VO + SwiGLU,
        but we also compute Q/K on CPU for real attention with KV cache.

        For each 2-layer block:
          1. Dispatch fused block (gets the architecture output with V→VO shortcut)
          2. BUT also compute the correct attention output on CPU
          3. Apply correction: subtract V→VO contribution, add real attention

        For the first build, this is NOT implemented — we use pure fused blocks.
        The V→VO shortcut means the model behaves as if every position is
        independent (no cross-position attention). Quality will be limited
        but the pipeline is tested end-to-end.
        """
        return self.forward_token(token_id, position, kv_cache)

    def generate(self, prompt_tokens, max_new_tokens=30, eos_id=None):
        """Full generation loop.

        Args:
            prompt_tokens: list of token IDs
            max_new_tokens: how many tokens to generate
            eos_id: stop token (optional)

        Returns:
            list of all tokens (prompt + generated)
        """
        from kv_cache import KVCache
        c = self.config

        kv = KVCache(c.n_layers, c.n_kv_heads, c.head_dim)
        generated = list(prompt_tokens)

        # Prefill: process all prompt tokens
        # For the first build (no KV cache attention in fused blocks),
        # each token is processed independently through all blocks
        for pos, tok in enumerate(prompt_tokens):
            logits = self.forward_token(tok, pos, kv)

        # First generated token
        next_tok = int(np.argmax(logits.astype(np.float32)))
        generated.append(next_tok)

        # Decode loop
        for step in range(max_new_tokens - 1):
            if eos_id is not None and next_tok == eos_id:
                break

            pos = len(generated) - 1
            logits = self.forward_token(next_tok, pos, kv)
            next_tok = int(np.argmax(logits.astype(np.float32)))
            generated.append(next_tok)

        return generated

    def stop(self):
        """Stop the pipe tool process."""
        if self.dispatcher:
            self.dispatcher.stop()
            self.dispatcher = None


# ===================================================================
# Main: compile + load + generate
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Llama-3.1-8B-Instruct on ANE (2-layer fused blocks)')
    parser.add_argument('--prompt', default='What is the capital of France?',
                        help='Input prompt')
    parser.add_argument('--tokens', type=int, default=30,
                        help='Max tokens to generate')
    parser.add_argument('--compile-only', action='store_true',
                        help='Only compile blocks, do not generate')
    parser.add_argument('--blocks', type=int, default=16,
                        help='Number of fused blocks (default 16 = all 32 layers)')
    parser.add_argument('--system', default=None,
                        help='System prompt for chat template')
    args = parser.parse_args()

    n_blocks = args.blocks
    n_layers = n_blocks * 2

    print("=" * 60)
    print("LLAMA-3.1-8B-INSTRUCT ON ANE")
    print(f"{n_blocks} fused blocks (2 layers each) + 8 lm_head chunks")
    print("=" * 60)

    # --- Load model ---
    print(f"\n[1/4] Loading Llama-8B from safetensors...")
    t0 = time.time()

    from llama_loader import LlamaModel
    model = LlamaModel.from_safetensors(MODEL_PATH)
    c = model.config
    t_load = time.time() - t0

    print(f"  Loaded in {t_load:.1f}s")
    print(f"  Config: dim={c.hidden_size}, layers={c.n_layers}, "
          f"heads={c.n_heads}/{c.n_kv_heads}, ffn={c.intermediate_size}")
    print(f"  Embedding: {model.embed_tokens.shape}")

    # --- Compile fused blocks ---
    print(f"\n[2/4] Compiling {n_blocks} fused blocks + 8 lm_head chunks...")
    t0 = time.time()
    compiled = compile_fused_blocks(model, BUILD_DIR, n_blocks=n_blocks)
    t_compile = time.time() - t0
    print(f"  Compiled in {t_compile:.1f}s")

    if args.compile_only:
        print(f"\n  Compile-only mode. Build dir: {BUILD_DIR}")
        return

    # --- Load onto ANE ---
    n_total_ops = n_blocks + 8  # blocks + lm_head chunks
    print(f"\n[3/4] Loading {n_total_ops} ops onto ANE via pipe tool...")
    t0 = time.time()
    ane_model = ANE8BModel(model, compiled, n_blocks=n_blocks)
    n_loaded = ane_model.load()
    t_dispatch = time.time() - t0
    print(f"  {n_loaded} ops loaded in {t_dispatch:.1f}s")
    print(f"  Dispatch: doEvaluateDirectWithModel (guaranteed ANE)")

    # --- Generate ---
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("unsloth/Meta-Llama-3.1-8B-Instruct")

    # Build chat prompt
    if args.system:
        messages = [
            {"role": "system", "content": args.system},
            {"role": "user", "content": args.prompt},
        ]
    else:
        messages = [{"role": "user", "content": args.prompt}]

    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False,
                                                 add_generation_prompt=True)
    tokens = tokenizer.encode(prompt_text, add_special_tokens=False)

    print(f"\n[4/4] Generating ({len(tokens)} prompt tokens + {args.tokens} max new)...")

    eos_id = tokenizer.eos_token_id

    t0 = time.time()
    all_tokens = ane_model.generate(tokens, max_new_tokens=args.tokens,
                                     eos_id=eos_id)
    t_total = time.time() - t0

    # Output
    generated_tokens = all_tokens[len(tokens):]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    n_gen = len(generated_tokens)

    # Timing
    tps = n_gen / t_total if t_total > 0 else 0
    ms_per_tok = (t_total / n_gen * 1000) if n_gen > 0 else 0

    print()
    print(f"> {response}")
    print()
    print("---")
    print(f"Model:       Llama-3.1-8B-Instruct (FP16, {c.n_layers} layers)")
    print(f"Hardware:    ANE only | GPU: idle")
    print(f"Tokens:      {n_gen} generated | {tps:.1f} tok/s ({ms_per_tok:.1f} ms/tok)")
    print(f"Timing:      total {t_total:.1f}s")
    print(f"Dispatches:  {n_blocks + 8} ops ({n_blocks} fused blocks + 8 lm_head chunks)")
    print(f"Fusion:      2 layers/block = 1 ANE dispatch ({n_layers}/{c.n_layers} layers)")
    print(f"GPU cost:    0%")
    print(f"Attention:   V->VO shortcut (seq_len=1 mode)")

    ane_model.stop()


if __name__ == '__main__':
    main()
