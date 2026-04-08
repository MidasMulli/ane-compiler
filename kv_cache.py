#!/usr/bin/env python3
"""
KV cache for GPT-2 autoregressive generation on ANE.

Simple CPU-side numpy KV cache. K/V vectors are appended after each
token's Q/K/V projections. Attention is computed on CPU using the
cached K/V and the current Q.

GPT-2: 12 layers, 12 heads, 64 head_dim, absolute positional embeddings.

Copyright 2026 Nick Lo. MIT License.
"""

import numpy as np


class KVCache:
    """Per-layer KV cache stored as CPU numpy arrays.

    Shapes:
        k_cache[layer]: [seq_len, n_heads, head_dim] FP16
        v_cache[layer]: [seq_len, n_heads, head_dim] FP16
    """

    def __init__(self, n_layers: int, n_heads: int, head_dim: int):
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.head_dim = head_dim
        # Start empty — grow via concatenation
        self.k_cache = [np.zeros((0, n_heads, head_dim), dtype=np.float16)
                        for _ in range(n_layers)]
        self.v_cache = [np.zeros((0, n_heads, head_dim), dtype=np.float16)
                        for _ in range(n_layers)]

    def append(self, layer_idx: int, new_k: np.ndarray, new_v: np.ndarray):
        """Append new K/V vectors to cache for a layer.

        Args:
            layer_idx: transformer layer index
            new_k: [1, n_heads, head_dim] FP16
            new_v: [1, n_heads, head_dim] FP16
        """
        self.k_cache[layer_idx] = np.concatenate(
            [self.k_cache[layer_idx], new_k], axis=0)
        self.v_cache[layer_idx] = np.concatenate(
            [self.v_cache[layer_idx], new_v], axis=0)

    def get(self, layer_idx: int):
        """Return full K/V cache for a layer.

        Returns:
            (k, v) where k.shape = v.shape = [seq_len, n_heads, head_dim]
        """
        return self.k_cache[layer_idx], self.v_cache[layer_idx]

    @property
    def seq_len(self) -> int:
        """Current sequence length (same across all layers)."""
        return self.k_cache[0].shape[0]

    def reset(self):
        """Clear all cached K/V entries."""
        for i in range(self.n_layers):
            self.k_cache[i] = np.zeros((0, self.n_heads, self.head_dim),
                                       dtype=np.float16)
            self.v_cache[i] = np.zeros((0, self.n_heads, self.head_dim),
                                       dtype=np.float16)
