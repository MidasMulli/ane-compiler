#!/usr/bin/env python3
"""
ane-compiler: Emit ANE .hwx binaries from model definitions.

This module emits BEEFFACE Zin binaries (.hwx) that the Apple Neural Engine
executes directly. It uses template-based emission: reference .hwx binaries
(compiled by Apple's ANE compiler) provide the structural template, and the
emitter fills in weights, activations, and graph-specific configuration.

Architecture:
    Model definition (graph + weights)
        ↓
    Graph partitioner (split into ANE-compatible subgraphs)
        ↓
    Template selector (pick .hwx template for each subgraph)
        ↓
    Weight packer (FP16 layout into __KERN_0)
        ↓
    Activation encoder (48K instruction stream or 64K PWL table)
        ↓
    HWX writer (assemble final binary)

Supported operations:
    - Linear (1x1 conv): any channel count, identity or custom weights
    - ReLU, abs, tanh, sigmoid (48K activation class)
    - SiLU, GELU, custom PWL (64K activation class)
    - Fused linear → activation → linear (single .hwx, graph fusion)

Requirements:
    - Template .hwx files (from Apple ANE compiler output)
    - ane-dispatch (for runtime loading and execution)

Copyright 2026 Nick Lo. MIT License.
"""

import os
import sys
import struct
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Union
from enum import Enum


# ═══════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════

BEEFFACE = 0xBEEFFACE
ANE_CPU_TYPE = 128
H17G_SUBTYPE = 9

# Multi-pass markers
PASS_BOUNDARY = 0x00FFF800      # Marks start of new pipeline pass in __text
PASS_BOUNDARY_ALT = 0x00FFF860  # Alternate boundary (seen in pass 4 of layernorm)
PASS_BOUNDARY_ALT2 = 0x00FFF868 # First-pass header variant
OUTPUT_DMA = 0x83119640         # Output writeback opcode
PROGRAM_TERM1 = 0x22001440      # Program terminator word 1
PROGRAM_TERM2 = 0x01040021      # Program terminator word 2
NUM_ANE_CORES = 16              # H17G/H17S have 16 cores

# ═══════════════════════════════════════════════════════════════
# Parameterized conv __text generation (no template needed)
# ═══════════════════════════════════════════════════════════════

# 117-word __text template for conv1x1 (all dims ≥64, output ≠ 128 or input > 128)
# Fixed words extracted from compiler output at 8 dimension pairs.
# Parameterized words filled by generate_conv_text().
_CONV_TEXT_FIXED = [
    0x00000001, 0x00000000, 0x00000000, 0x00000000,  # W[0-3]
    0x00710000, None,       0x04000068, 0x00000000,  # W[4-7]    W[5]=param
    0x00FFF868, 0x00000000, 0x00000000, 0x00000000,  # W[8-11]
    0x00050009, 0xFFC01540, 0x000102C0, 0x00000021,  # W[12-15]
    0x00000021, 0x00000021, 0x00000021, 0x00000021,  # W[16-19]  tile desc header
    0x00000021, 0x00000021, 0x00000021, 0x00000021,  # W[20-23]
    0x00031551, 0x00000021, 0x00000021, 0x00000021,  # W[24-27]
    0x00000021, 0x00000021, 0x00000021, 0x00000021,  # W[28-31]
    0x000F1559, None,       None,       None,         # W[32-35]  W[33-47]=tile offsets
    None,       None,       None,       None,         # W[36-39]
    None,       None,       None,       None,         # W[40-43]
    None,       None,       None,       None,         # W[44-47]
    None,       None,       None,       None,         # W[48-51]  W[48-63]=tile stride
    None,       None,       None,       None,         # W[52-55]
    None,       None,       None,       None,         # W[56-59]
    None,       None,       None,       None,         # W[60-63]
    0x00010001, 0x00000001, 0x00000001, None,         # W[64-67]  W[67]=ic
    0x93418005, 0x00000001, 0x00000001, None,         # W[68-71]  W[71]=oc
    0x00000001, None,       None,       0x00200000,   # W[72-75]  W[73,74]=param
    0x80081342, 0x000000CE, 0x00000040, 0x0000135A,   # W[76-79]
    0x01002031, 0x00009041, 0x00500172, 0x00500130,   # W[80-83]
    0x98039045, 0x00000010, None,       None,         # W[84-87]  W[86-88]=ic_stride
    None,       0x0050017A, None,       0x80049240,   # W[88-91]  W[90]=ic_stride
    0x10000082, 0x00103C00, 0x00003C00, 0x80801445,   # W[92-95]
    0x00000040, 0x01302031, 0x83109640, 0x00000012,   # W[96-99]
    0x00A000A0, 0x00000000, 0x007F0000, 0x20010701,   # W[100-103]
    0x24009544, 0x00000000, 0x00000000, 0x23009344,   # W[104-107]
    0x00000000, 0x00000000, 0x23809442, 0x00000000,   # W[108-111]
    0x00000000, 0x22001340, 0x00000021, 0x22001440,   # W[112-115]
    0x01000021,                                        # W[116]
]


def generate_conv_text(in_ch: int, out_ch: int) -> bytes:
    """Generate conv1x1 __text microcode from dimensions alone.

    Produces the 117-word (468-byte) ANE pipeline program for a
    conv1x1 operation. No template .hwx needed.

    Formulas decoded from 8 compiler captures (64→64 through 1024→1024):
      W[5]      = tile_size // 4096
      W[33-47]  = tile_size * (i - 32)  [16-core tile offset table]
      W[48-63]  = tile_size             [uniform tile stride]
      W[67]     = in_ch
      W[71]     = out_ch
      W[73]     = 0x200004 (fixed for dims ≤ 512; 0x240004 at 1024)
      W[74]     = min(ceil(log2(out_ch / 16)), 5)
      W[86-88,90] = in_ch * 16

    Args:
        in_ch: input channels (must be ≥ 64, multiple of 16)
        out_ch: output channels (must be ≥ 64, multiple of 16, ≠ 128 unless in_ch > 128)

    Returns:
        468 bytes of __text microcode
    """
    import math

    # Tile sizes: first tile gets ceil, rest get floor (handles non-multiple-of-16 out_ch)
    oc_per_tile_first = -(-out_ch // NUM_ANE_CORES)  # ceil(out_ch / 16)
    oc_per_tile_rest = out_ch // NUM_ANE_CORES         # floor(out_ch / 16)
    tile_first = in_ch * oc_per_tile_first * 2
    tile_rest = in_ch * oc_per_tile_rest * 2
    ic_stride = in_ch * NUM_ANE_CORES  # = in_ch * 16

    words = list(_CONV_TEXT_FIXED)

    # W[5]: tile pages (uses original uniform tile_size formula, floor division)
    tile_size_uniform = in_ch * out_ch * 2 // NUM_ANE_CORES
    words[5] = 0 if tile_size_uniform < 4096 else tile_size_uniform // 4096

    # W[33]-W[47]: tile offset table
    # offset[i] = tile_first + i * tile_rest
    for i in range(16):
        words[33 + i] = tile_first + i * tile_rest

    # W[48]-W[63]: tile stride (first tile gets ceil, rest get floor)
    words[48] = tile_first
    for i in range(1, 16):
        words[48 + i] = tile_rest

    # W[67]: input channels
    words[67] = in_ch

    # W[71]: output channels
    words[71] = out_ch

    # W[73]: pipeline config (0x200004 for ic ≤ 640; 0x244404 at ic ≥ 768)
    words[73] = 0x200004 if in_ch <= 640 else 0x244404

    # W[74]: pipeline depth = min(ceil(log2(out_ch / 16)), 5)
    words[74] = min(math.ceil(math.log2(out_ch / 16)), 5)

    # W[86,87,88,90]: input DMA stride (ic*16; +0x30 at ic ≥ 768)
    ic_dma = ic_stride if in_ch <= 640 else ic_stride + 0x30
    words[86] = ic_dma
    words[87] = ic_stride  # W[87] stays clean
    words[88] = ic_stride  # W[88] stays clean
    words[90] = ic_dma     # W[90] matches W[86]

    return struct.pack(f'<{len(words)}I', *words)


def generate_conv_text_int8(in_ch: int, out_ch: int) -> bytes:
    """Generate INT8 conv1x1 __text microcode from dimensions alone.

    Produces the 117-word (468-byte) ANE pipeline program for an INT8
    conv1x1 operation. Same structure as FP16 conv with two key changes:

    From system .hwx analysis (VideoProcessing frame enhancers):
      W[68] = 0x93488005 (INT8 conv opcode, was 0x93418005 FP16)
      W[71] = 0x5042A0C3 (INT8 dequant config, was out_ch)
               0x50 = UINT8 kernel format selector
               0x042A0C3 = dequantization scale/zero-point config

    Tile sizes are computed for INT8 weight density (1 byte/element vs 2).
    Hardware decompresses INT8 weights to FP16 for computation.

    Args:
        in_ch: input channels (must be >= 64, multiple of 16)
        out_ch: output channels (must be >= 64, multiple of 16)

    Returns:
        468 bytes of __text microcode
    """
    import math

    # INT8: 1 byte per weight element (vs 2 for FP16)
    NC = NUM_ANE_CORES
    oc_per_tile_first = -(-out_ch // NC)
    oc_per_tile_rest = out_ch // NC
    tile_first = in_ch * oc_per_tile_first * 1   # 1 byte per INT8 weight
    tile_rest = in_ch * oc_per_tile_rest * 1
    ic_stride = in_ch * NC

    words = list(_CONV_TEXT_FIXED)

    # W[5]: tile pages (INT8 weights = half the size)
    tile_size_uniform = in_ch * out_ch * 1 // NC
    words[5] = 0 if tile_size_uniform < 4096 else tile_size_uniform // 4096

    # W[33]-W[47]: tile offset table (INT8 density)
    for i in range(16):
        words[33 + i] = tile_first + i * tile_rest

    # W[48]-W[63]: tile stride (INT8 density)
    words[48] = tile_first
    for i in range(1, 16):
        words[48 + i] = tile_rest

    # W[67]: input channels (unchanged)
    words[67] = in_ch

    # W[68]: INT8 conv opcode (key change #1)
    words[68] = 0x93488005

    # W[71]: INT8 dequant config word (key change #2 — replaces out_ch)
    words[71] = 0x5042A0C3

    # W[73]: pipeline config
    words[73] = 0x200004 if in_ch <= 640 else 0x244404

    # W[74]: pipeline depth
    words[74] = min(math.ceil(math.log2(out_ch / 16)), 5)

    # W[86,87,88,90]: input DMA stride
    ic_dma = ic_stride if in_ch <= 640 else ic_stride + 0x30
    words[86] = ic_dma
    words[87] = ic_stride
    words[88] = ic_stride
    words[90] = ic_dma

    return struct.pack(f'<{len(words)}I', *words)


# ═══════════════════════════════════════════════════════════════
# Parameterized softmax __text generation
# ═══════════════════════════════════════════════════════════════

# 257-word template for softmax (dims 32, 64, 128, 512 — all except 256 which is 253w)
# 22 parameterized words, 235 fixed. PWL tables in __KERN_0 are dimension-independent.
_SOFTMAX_TEXT_FIXED = None  # Loaded lazily from reference

_LAYERNORM_TEXT_FIXED = None  # Loaded lazily from reference

def generate_layernorm_text(dim: int, epsilon: float = 1e-5,
                            reference_hwx_path: Optional[str] = None) -> bytes:
    """Generate layernorm __text microcode from channel dimension.

    LayerNorm decomposes into 4-5 ANE pipeline passes (mean → variance →
    normalize → scale+shift). The 143-word (572-byte) __text is parameterized
    by channel dim and epsilon.

    Formulas decoded from 4 captures (dim=32, 64, 128, 512):
      W[15,19,59,63,65,94,98] = dim
      W[41,79]                = FP32(1/dim)
      W[81]                   = dim // 512
      W[115]                  = dim*16 + (0x10 if dim < 64 else 0x20)
      W[116,117]              = dim*16
      W[120]                  = dim*16 + (0x50 if dim < 64 else 0x60)

    Note: epsilon is embedded as FP32 at W[78] (default 0x37390000 ≈ 1.1e-5).

    Args:
        dim: channel dimension (≥ 32, ≠ 256 which uses different template)
        epsilon: layernorm epsilon (default 1e-5)
        reference_hwx_path: path to reference layernorm .hwx for template
    """
    global _LAYERNORM_TEXT_FIXED
    if _LAYERNORM_TEXT_FIXED is None:
        if reference_hwx_path:
            data = Path(reference_hwx_path).read_bytes()
            # Find __text
            ncmds = struct.unpack_from('<I', data, 0x10)[0]
            offset = 32
            for _ in range(ncmds):
                cmd = struct.unpack_from('<I', data, offset)[0]
                cmdsize = struct.unpack_from('<I', data, offset + 4)[0]
                if cmd == 0x19:
                    segname = data[offset+8:offset+24].split(b'\x00')[0].decode('ascii')
                    nsects = struct.unpack_from('<I', data, offset+64)[0]
                    for s in range(nsects):
                        s_off = offset + 72 + s * 80
                        sectname = data[s_off:s_off+16].split(b'\x00')[0].decode('ascii')
                        size = struct.unpack_from('<Q', data, s_off+40)[0]
                        foff = struct.unpack_from('<I', data, s_off+48)[0]
                        if segname == '__TEXT' and sectname == '__text':
                            _LAYERNORM_TEXT_FIXED = list(struct.unpack(
                                f'<{size//4}I', data[foff:foff+size]))
                offset += cmdsize
        else:
            raise ValueError("First call requires reference_hwx_path")

    words = list(_LAYERNORM_TEXT_FIXED)

    # Channel dimension
    for wi in [15, 19, 59, 63, 65, 94, 98]:
        words[wi] = dim

    # FP32(1/dim) — reduction scale
    inv_dim = struct.unpack('<I', struct.pack('<f', 1.0 / dim))[0]
    words[41] = inv_dim
    words[79] = inv_dim

    # Page config
    words[81] = dim // 512

    # DMA strides
    offset_small = 0x10 if dim < 64 else 0x20
    words[115] = dim * 16 + offset_small
    words[116] = dim * 16
    words[117] = dim * 16
    words[120] = dim * 16 + offset_small + 0x40

    # Epsilon (FP32 at W[78]) — compiler uses 0x37390000 (≈1.1e-5) for eps=1e-5
    if abs(epsilon - 1e-5) < 1e-8:
        words[78] = 0x37390000  # match compiler default exactly
    else:
        words[78] = struct.unpack('<I', struct.pack('<f', epsilon))[0]

    return struct.pack(f'<{len(words)}I', *words)


def generate_softmax_text(dim: int, reference_hwx_path: Optional[str] = None) -> bytes:
    """Generate softmax __text microcode from channel dimension alone.

    Softmax decomposes into 5 ANE pipeline passes:
      1. reduce_max  2. exp(x-max)  3. reduce_sum  4. reciprocal  5. multiply

    The 257-word (1028-byte) __text is parameterized by channel dim only.
    PWL tables (exp + reciprocal) in __KERN_0 are dimension-independent.

    Formulas decoded from 4 compiler captures (dim=32, 64, 128, 512):
      W[5]          = dim // 512
      W[16,64,68,120,124,182,222,226] = dim
      W[21]         = (dim << 16) | 0x4000
      W[35,78]      = dim * 18 + 16
      W[37,38,39,81,134,197,236] = dim * 16
      W[41,80]      = dim * 18
      W[239]        = dim * 16 + 16

    Args:
        dim: channel dimension (must be power of 2, ≥ 32, ≠ 256)
        reference_hwx_path: path to a reference softmax .hwx for the template

    Returns:
        1028 bytes of __text microcode
    """
    global _SOFTMAX_TEXT_FIXED
    if _SOFTMAX_TEXT_FIXED is None:
        if reference_hwx_path:
            data = Path(reference_hwx_path).read_bytes()
            _SOFTMAX_TEXT_FIXED = list(struct.unpack('<257I', data[0x4000:0x4000 + 1028]))
        else:
            raise ValueError("First call to generate_softmax_text requires reference_hwx_path")

    words = list(_SOFTMAX_TEXT_FIXED)

    # Apply parameterization
    words[5] = dim // 512

    for wi in [16, 64, 68, 120, 124, 182, 222, 226]:
        words[wi] = dim

    words[21] = (dim << 16) | 0x4000

    for wi in [35, 78]:
        words[wi] = dim * 18 + 16

    for wi in [37, 38, 39, 81, 134, 197, 236]:
        words[wi] = dim * 16

    for wi in [41, 80]:
        words[wi] = dim * 18

    words[239] = dim * 16 + 16

    return struct.pack(f'<{len(words)}I', *words)


# File regions for template-based emission
class Region(Enum):
    HEADER = "header"           # 0x0000-0x001F (32 bytes, fixed)
    LOAD_CMDS = "load_cmds"     # 0x0020-0x2F2F (variable)
    SYMTAB = "symtab"           # after load_cmds (tile descriptor copy)
    TEXT = "text"               # 0x4000+ (__text kernel program)
    CONST = "const"             # after __text (__const pipeline config)
    KERN0 = "kern0"             # 0xC000+ (weights + PWL + tile replication)
    METADATA = "metadata"       # compiler info (non-functional)


# ═══════════════════════════════════════════════════════════════
# Activation encoding
# ═══════════════════════════════════════════════════════════════

class ActivationType(Enum):
    """Supported activation functions."""
    RELU = "relu"
    ABS = "abs"
    TANH = "tanh"
    SIGMOID = "sigmoid"
    SILU = "silu"
    GELU = "gelu"
    GELU_TANH = "gelu_tanh"
    LINEAR = "linear"
    CUSTOM_PWL = "custom_pwl"


@dataclass
class PWLTable:
    """Piecewise linear lookup table for 64K activation class.

    Format: 84 bytes = 42 FP16 values
        [0-3]: header (left_bound, right_bound_or_inf, center_x, center_y_or_inf)
        [4-36]: 33 breakpoint y-values
        [37-41]: footer (residuals, asymptotic_slope, metadata)
    """
    header: np.ndarray      # 4 FP16 values
    breakpoints: np.ndarray  # 33 FP16 values
    footer: np.ndarray       # 5 FP16 values

    def to_bytes(self) -> bytes:
        """Pack to 84-byte binary representation."""
        return (self.header.astype(np.float16).tobytes() +
                self.breakpoints.astype(np.float16).tobytes() +
                self.footer.astype(np.float16).tobytes())

    @classmethod
    def from_bytes(cls, data: bytes) -> 'PWLTable':
        """Parse from 84-byte binary."""
        values = np.frombuffer(data[:84], dtype=np.float16)
        return cls(
            header=values[0:4].copy(),
            breakpoints=values[4:37].copy(),
            footer=values[37:42].copy()
        )

    @classmethod
    def extract_from_hwx(cls, hwx_path: str) -> 'PWLTable':
        """Extract PWL table from a 64K .hwx file at offset 0xC000."""
        data = Path(hwx_path).read_bytes()
        return cls.from_bytes(data[0xC000:0xC054])


# ═══════════════════════════════════════════════════════════════
# Multi-pass __text parsing
# ═══════════════════════════════════════════════════════════════

@dataclass
class PipelinePass:
    """A single ANE pipeline pass within a multi-pass __text program.

    Each pass is a contiguous block of instruction words that configures one
    traversal through the ANE's 17-stage fixed-function pipeline. Passes are
    delimited by PASS_BOUNDARY (0x00FFF800) markers.

    The first pass has no boundary marker (it starts at __text offset 0).
    Subsequent passes start with a 4-word header: [pass_info, 0, 0, 0]
    followed by the boundary marker.
    """
    index: int                  # 0-based pass index
    words: List[int]            # Raw instruction words
    opcode: int = 0             # Primary opcode (0x????8005 pattern)
    word_offset: int = 0        # Word offset within __text
    has_conv_header: bool = False  # Has 0xFFC01540 conv pipeline header
    pwl_ref_offset: int = -1    # Offset into __KERN_0 for PWL data (-1 = none)

    def to_bytes(self) -> bytes:
        return struct.pack(f'<{len(self.words)}I', *self.words)

    @property
    def byte_size(self) -> int:
        return len(self.words) * 4


def parse_multipass_text(text_data: bytes) -> List[PipelinePass]:
    """Parse a multi-pass __text program into individual passes.

    Each pass (except pass 0) starts with a 4-word header where
    word[0] = 0x00XX000Y (Y = 1-based pass index in low nibble),
    word[1] = 0. The header is followed by a boundary marker
    (0x00FFF800 or variant like 0x00FFD800, 0x00FFF860).

    Returns list of PipelinePass objects.
    """
    nwords = len(text_data) // 4
    words = list(struct.unpack(f'<{nwords}I', text_data))

    # Find pass header positions by looking for the 0x00XX000Y pattern
    # where Y = pass index (1+) and following word is 0
    pass_starts = []
    for i in range(len(words) - 1):
        w = words[i]
        # Pattern: 0x00XX00YY where XX > 0, YY = pass index 1-15
        idx = w & 0xFF
        if (idx >= 1 and idx <= 15 and
            (w >> 24) == 0 and
            (w >> 16) & 0xFF > 0 and
            (w >> 8) & 0xFF == 0 and
            words[i + 1] == 0 and
            i + 4 < len(words)):
            # Verify: word at i+4 looks like a boundary marker (0x00FF????)
            boundary = words[i + 4]
            if (boundary & 0xFFFF0000) == 0x00FF0000:
                pass_starts.append((i, idx))

    if not pass_starts:
        # Single-pass program
        opcode = _find_opcode(words)
        return [PipelinePass(index=0, words=words, opcode=opcode, word_offset=0)]

    passes = []

    # First pass: from word 0 to the first pass header
    first_end = pass_starts[0][0]
    first_words = words[:first_end]
    passes.append(PipelinePass(
        index=0, words=first_words, opcode=_find_opcode(first_words),
        word_offset=0))

    # Subsequent passes: from each header to the next header (or end)
    for pi, (start, idx) in enumerate(pass_starts):
        end = pass_starts[pi + 1][0] if pi + 1 < len(pass_starts) else nwords
        pass_words = words[start:end]
        opcode = _find_opcode(pass_words)
        has_conv = 0xFFC01540 in pass_words
        passes.append(PipelinePass(
            index=idx, words=pass_words, opcode=opcode,
            word_offset=start, has_conv_header=has_conv))

    return passes


def _find_opcode(words: List[int]) -> int:
    """Find the primary opcode in an instruction word sequence.
    ANE opcodes share the 0x8005 suffix in the low 16 bits."""
    for w in words:
        if w & 0xFFFF == 0x8005:
            return w
    return 0


def assemble_multipass_text(passes: List[PipelinePass]) -> bytes:
    """Reassemble individual passes into a complete __text program."""
    result = bytearray()
    for p in passes:
        result.extend(p.to_bytes())
    return bytes(result)


# ═══════════════════════════════════════════════════════════════
# Softmax / LayerNorm emission parameters
# ═══════════════════════════════════════════════════════════════

@dataclass
class SoftmaxParams:
    """Parameters for softmax emission.

    Softmax decomposes into 5 ANE passes:
      1. reduce_max(x)
      2. exp(x - max)   [PWL: exp table]
      3. reduce_sum(exp)
      4. reciprocal(sum) [PWL: reciprocal table]
      5. exp * reciprocal

    The two PWL tables (128 bytes each) live in __KERN_0.
    """
    exp_pwl: Optional[bytes] = None         # 128-byte exp PWL (None = use template)
    reciprocal_pwl: Optional[bytes] = None  # 128-byte reciprocal PWL (None = use template)
    dim: int = 64                           # Channel dimension

    def get_kern0(self, template_kern0: bytes) -> bytes:
        """Build __KERN_0 with optional PWL overrides."""
        buf = bytearray(template_kern0)
        if self.exp_pwl:
            buf[0:128] = self.exp_pwl
        if self.reciprocal_pwl:
            buf[128:256] = self.reciprocal_pwl
        return bytes(buf)


@dataclass
class LayerNormParams:
    """Parameters for layernorm emission.

    LayerNorm decomposes into 4 ANE passes:
      1. mean(x)         [reduce + scale by 1/dim]
      2. variance(x-mean) [abs opcode repurposed]
      3. normalize        [(x-mean)/sqrt(var+eps)]
      4. scale + shift    [gamma * normalized + beta, + output DMA]

    Constants (epsilon, 1/dim) are embedded as FP32 literals in the instruction stream.
    No __KERN_0 needed — gamma/beta are identity transforms in the template.
    """
    epsilon: float = 1e-5       # LayerNorm epsilon
    dim: int = 64               # Channel dimension (affects 1/dim constant)

    @property
    def inv_dim_fp32_bytes(self) -> bytes:
        """1/dim as FP32 bytes (for pass 1 reduction scale)."""
        return struct.pack('<f', 1.0 / self.dim)

    @property
    def epsilon_fp32_bytes(self) -> bytes:
        """Epsilon as FP32 bytes (embedded in pass 3)."""
        return struct.pack('<f', self.epsilon)


@dataclass
class BatchNormParams:
    """Parameters for batchnorm emission (affine scale+bias for LayerNorm).

    Batchnorm with mean=0, variance=1, epsilon=0 implements:
        output = gamma * input + beta

    ANE batchnorm at dim=768 compiles to a single dispatch:
      - __text: 320B (80 words), opcode 0x9D41
      - __const: 3072B = [beta/gamma (768 FP16), gamma (768 FP16)]

    Hardware computes: output = gamma * (input + beta/gamma)
    This is mathematically equivalent to gamma*input + beta but uses
    a single fused multiply-add per element.

    Used after MVN dispatch to implement LayerNorm with elementwise_affine=True.
    """
    gamma: np.ndarray   # Per-channel scale [dim] (FP32)
    beta: np.ndarray    # Per-channel bias [dim] (FP32)

    @property
    def dim(self) -> int:
        return len(self.gamma)

    def pack_const(self) -> bytes:
        """Pack gamma/beta into ANE batchnorm __const format.

        Layout: [beta/gamma (dim FP16), gamma (dim FP16)]
        Total: dim * 4 bytes
        """
        ratio = (self.beta / self.gamma).astype(np.float16)
        gamma_fp16 = self.gamma.astype(np.float16)
        return ratio.tobytes() + gamma_fp16.tobytes()


# ═══════════════════════════════════════════════════════════════
# Template registry
# ═══════════════════════════════════════════════════════════════

@dataclass
class HWXTemplate:
    """A reference .hwx binary that serves as a structural template.

    Templates provide the fixed structure (headers, load commands, pipeline config).
    The emitter replaces variable regions (weights, activations, tile descriptors).
    """
    name: str
    path: str
    data: bytes
    template_class: str  # "48k_activation", "64k_activation", "conv", "conv_fused"

    # Parsed regions
    text_offset: int = 0
    text_size: int = 0
    const_offset: int = 0
    const_size: int = 0
    kern0_offset: int = 0
    kern0_size: int = 0
    symtab_offset: int = 0
    symtab_size: int = 0
    ncmds: int = 0
    file_size: int = 0

    # Activation data extracted from this template
    text_data: bytes = b''
    pwl_data: bytes = b''

    # Multi-pass data (parsed lazily)
    _passes: Optional[List[PipelinePass]] = field(default=None, repr=False)

    @property
    def passes(self) -> List[PipelinePass]:
        """Parse __text into pipeline passes (cached)."""
        if self._passes is None:
            self._passes = parse_multipass_text(self.text_data)
        return self._passes

    @property
    def num_passes(self) -> int:
        return len(self.passes)

    @classmethod
    def load(cls, path: str, name: str = "", template_class: str = "auto") -> 'HWXTemplate':
        """Load a .hwx file as a template."""
        data = Path(path).read_bytes()
        t = cls(name=name or Path(path).stem, path=path, data=data,
                template_class=template_class)
        t.file_size = len(data)
        t._parse()
        return t

    def _parse(self):
        """Parse the template's structure."""
        magic = struct.unpack_from('<I', self.data, 0)[0]
        if magic != BEEFFACE:
            raise ValueError(f"Not a BEEFFACE binary: 0x{magic:08X}")

        self.ncmds = struct.unpack_from('<I', self.data, 0x10)[0]

        offset = 32
        for _ in range(self.ncmds):
            cmd = struct.unpack_from('<I', self.data, offset)[0]
            cmdsize = struct.unpack_from('<I', self.data, offset + 4)[0]

            if cmd == 0x19:  # LC_SEGMENT_64
                segname = self.data[offset+8:offset+24].split(b'\x00')[0].decode('ascii')
                nsects = struct.unpack_from('<I', self.data, offset+56+8)[0]

                for s in range(nsects):
                    s_off = offset + 72 + s * 80
                    sectname = self.data[s_off:s_off+16].split(b'\x00')[0].decode('ascii')
                    size = struct.unpack_from('<Q', self.data, s_off + 40)[0]
                    foff = struct.unpack_from('<I', self.data, s_off + 48)[0]

                    if segname == '__TEXT' and sectname == '__text':
                        self.text_offset = foff
                        self.text_size = size
                    elif segname == '__TEXT' and sectname == '__const':
                        self.const_offset = foff
                        self.const_size = size
                    elif segname == '__KERN_0' and sectname == '__kern_0':
                        self.kern0_offset = foff
                        self.kern0_size = size

            elif cmd == 0x02:  # LC_SYMTAB
                self.symtab_offset = struct.unpack_from('<I', self.data, offset + 8)[0]
                nsyms = struct.unpack_from('<I', self.data, offset + 12)[0]
                stroff = struct.unpack_from('<I', self.data, offset + 16)[0]
                strsize = struct.unpack_from('<I', self.data, offset + 20)[0]
                self.symtab_size = (stroff + strsize) - self.symtab_offset

            offset += cmdsize

        # Extract activation data
        self.text_data = self.data[self.text_offset:self.text_offset + self.text_size]
        if self.file_size > 0xC000:
            self.pwl_data = self.data[0xC000:0xC054]

        # Auto-detect template class
        if self.template_class == "auto":
            if self.kern0_offset > 0 and self.kern0_size == 256 and self.text_size > 800:
                # 5-pass softmax: 256B PWL (exp+reciprocal), >800B __text
                self.template_class = "softmax"
            elif self.kern0_offset > 0:
                if self.text_size > 400:
                    self.template_class = "conv_fused"
                else:
                    self.template_class = "conv"
            elif self.text_size > 400 and self.file_size == 49152:
                # 4-pass layernorm: large __text, no __KERN_0, 48K file
                self.template_class = "layernorm"
            elif self.text_size == 320 and self.kern0_offset == 0 and self.const_size > 0:
                # Batchnorm: 320B __text, no __KERN_0, has __const
                self.template_class = "batchnorm"
            elif self.file_size == 65536:
                self.template_class = "64k_activation"
            else:
                self.template_class = "48k_activation"


class TemplateRegistry:
    """Registry of available .hwx templates.

    Templates are loaded from a directory of compiler-generated .hwx files.
    Each template provides the structural skeleton for emitting a specific
    type of ANE operation.
    """

    def __init__(self):
        self.templates: Dict[str, HWXTemplate] = {}
        self._activation_48k: Dict[str, HWXTemplate] = {}
        self._activation_64k: Dict[str, HWXTemplate] = {}
        self._conv: Dict[str, HWXTemplate] = {}
        self._softmax: Dict[str, HWXTemplate] = {}
        self._layernorm: Dict[str, HWXTemplate] = {}
        self._batchnorm: Dict[str, HWXTemplate] = {}

    def load_directory(self, path: str):
        """Load all .hwx files from a directory as templates."""
        p = Path(path)
        for hwx in sorted(p.glob('*.hwx')):
            try:
                t = HWXTemplate.load(str(hwx))
                self.templates[t.name] = t
                self._categorize(t)
            except Exception:
                pass  # Skip unparseable files

    def load_file(self, path: str, name: str = "", template_class: str = "auto"):
        """Load a single .hwx file as a template."""
        t = HWXTemplate.load(path, name=name, template_class=template_class)
        self.templates[t.name] = t
        self._categorize(t)

    def _categorize(self, t: HWXTemplate):
        if t.template_class == '48k_activation':
            self._activation_48k[t.name] = t
        elif t.template_class == '64k_activation':
            self._activation_64k[t.name] = t
        elif t.template_class in ('conv', 'conv_fused'):
            self._conv[t.name] = t
        elif t.template_class == 'softmax':
            self._softmax[t.name] = t
        elif t.template_class == 'layernorm':
            self._layernorm[t.name] = t
        elif t.template_class == 'batchnorm':
            self._batchnorm[t.name] = t

    def get_activation_template(self, activation: ActivationType) -> HWXTemplate:
        """Get the best template for an activation function."""
        name_map = {
            ActivationType.RELU: 'relu',
            ActivationType.ABS: 'abs',
            ActivationType.SIGMOID: 'sigmoid',
            ActivationType.SILU: 'silu_native_65536',
            ActivationType.GELU: 'gelu_exact',
            ActivationType.GELU_TANH: 'gelu_tanh',
        }
        target = name_map.get(activation)

        # Try exact match
        if target and target in self.templates:
            return self.templates[target]

        # Fallback: any template of the right class
        if activation in (ActivationType.SILU, ActivationType.GELU,
                         ActivationType.GELU_TANH, ActivationType.CUSTOM_PWL):
            if self._activation_64k:
                return next(iter(self._activation_64k.values()))

        if self._activation_48k:
            return next(iter(self._activation_48k.values()))

        raise ValueError(f"No template available for {activation}")

    def get_conv_template(self, with_pwl: bool = False) -> HWXTemplate:
        """Get a conv template (fused conv+activation).
        If with_pwl=True, prefer a template with PWL activation (SiLU/sigmoid)."""
        if self._conv:
            if with_pwl:
                for name in ('conv_silu_mil_65536', 'conv_sigmoid_mil_65536'):
                    if name in self._conv:
                        return self._conv[name]
            if 'conv_relu_mil_65536' in self._conv:
                return self._conv['conv_relu_mil_65536']
            return next(iter(self._conv.values()))
        raise ValueError("No conv template available")

    def get_softmax_template(self) -> HWXTemplate:
        """Get a softmax template (5-pass multi-pass)."""
        if self._softmax:
            return next(iter(self._softmax.values()))
        raise ValueError("No softmax template available. "
                        "Load a softmax .hwx (e.g. from attention_probe_v3/)")

    def get_layernorm_template(self) -> HWXTemplate:
        """Get a layernorm template (4-pass multi-pass)."""
        if self._layernorm:
            return next(iter(self._layernorm.values()))
        raise ValueError("No layernorm template available. "
                        "Load a layernorm .hwx (e.g. from attention_probe_v3/)")

    def get_batchnorm_template(self) -> HWXTemplate:
        """Get a batchnorm template (single-pass scale+bias).

        Batchnorm is used for LayerNorm affine params (gamma*x + beta).
        Template: 320B __text + 3072B __const at dim=768.
        """
        if self._batchnorm:
            return next(iter(self._batchnorm.values()))
        raise ValueError("No batchnorm template available. "
                        "Load a batchnorm .hwx captured from aned.")


# ═══════════════════════════════════════════════════════════════
# Weight packer
# ═══════════════════════════════════════════════════════════════

class WeightPacker:
    """Pack model weights into ANE __KERN_0 format.

    The ANE weight layout depends on the output channel count:

    For out_ch >= 16 (production scale):
        Weights are partitioned across 16 ANE cores. No tile replication.
        __KERN_0 size = out_ch * in_ch * 2 bytes exactly.
        Layout: ref_w.reshape(16, out_ch//16, in_ch).transpose(0, 2, 1).flatten()
        i.e., [core_group, in_ch, sub_out_ch] interleave.

    For out_ch < 16 (small models, atlas templates):
        Tile-replicated format with padding. Template-specific layout.
        Each tile = kern0_size // 16, weights padded to PlaneStride boundaries.

    Verified against Apple ANE compiler output at 64→64, 64→128,
    128→256, and 256→512 dimensions (sentinel weight round-trip).
    """
    NUM_CORES = 16

    @staticmethod
    def pack_conv1x1(weights: np.ndarray) -> bytes:
        """Pack conv1x1 weights into ANE __KERN_0 layout.

        The ANE uses a 16-core tiled layout with 32-channel sub-blocks
        organized in "stripes" across tiles.

        Layout: rows are assigned to tiles in 32-row stripes.
        Stripe s contains rows [s*512 : (s+1)*512], distributed as
        32 consecutive rows per tile. A remainder stripe at the end
        handles non-multiple-of-32 tile sizes, with tile 0 getting
        ceil and tiles 1-15 getting floor of the remainder.

        Verified byte-identical to Apple's ANE compiler at:
        768x768, 768x3072, 3072x768, 768x50257 (all GPT-2 dims).

        Args:
            weights: [out_channels, in_channels] weight matrix

        Returns:
            Packed bytes for __KERN_0 section
        """
        NC = WeightPacker.NUM_CORES
        w = weights.astype(np.float16)
        out_ch, in_ch = w.shape

        if out_ch < NC:
            return w.flatten().tobytes()

        sub_size = 32
        stripe_size = NC * sub_size  # 512

        # Per-tile output channel counts (tile 0 gets ceil, rest floor)
        oc_first = -(-out_ch // NC)   # ceil
        oc_rest = out_ch // NC         # floor

        # Full stripes (all tiles get 32 rows each)
        n_full = oc_rest // sub_size
        # Remainder per tile after full stripes
        rem_first = oc_first - n_full * sub_size
        rem_rest = oc_rest - n_full * sub_size

        result = []
        for t in range(NC):
            # Full 32-row sub-blocks from each stripe
            for s in range(n_full):
                row_start = t * sub_size + s * stripe_size
                result.append(w[row_start:row_start + sub_size, :].flatten(order='F'))

            # Remainder sub-block from partial stripe
            rem = rem_first if t == 0 else rem_rest
            if rem > 0:
                rem_stripe_start = n_full * stripe_size
                if t == 0:
                    row_start = rem_stripe_start
                else:
                    row_start = rem_stripe_start + rem_first + (t - 1) * rem_rest
                result.append(w[row_start:row_start + rem, :].flatten(order='F'))

        return np.concatenate(result).tobytes()

    @staticmethod
    def unpack_conv1x1(kern0_data: bytes, out_ch: int, in_ch: int) -> np.ndarray:
        """Unpack weights from ANE __KERN_0 layout back to [out_ch, in_ch].

        Inverse of pack_conv1x1.
        """
        hw = np.frombuffer(kern0_data[:out_ch * in_ch * 2], dtype=np.float16)

        if out_ch < WeightPacker.NUM_CORES or out_ch % WeightPacker.NUM_CORES != 0:
            return hw.reshape(out_ch, in_ch)

        block = out_ch // WeightPacker.NUM_CORES
        sub1 = min(32, block)
        sub2 = block - sub1
        split_point = WeightPacker.NUM_CORES * sub1

        w = np.zeros((out_ch, in_ch), dtype=np.float16)
        offset = 0
        for t in range(WeightPacker.NUM_CORES):
            # First sub-block
            chunk_sz = sub1 * in_ch
            chunk = hw[offset:offset + chunk_sz].reshape(in_ch, sub1, order='C')
            w[t * sub1:(t + 1) * sub1, :] = chunk.T
            offset += chunk_sz
            # Second sub-block
            if sub2 > 0:
                chunk_sz = sub2 * in_ch
                chunk = hw[offset:offset + chunk_sz].reshape(in_ch, sub2, order='C')
                w[split_point + t * sub2:split_point + (t + 1) * sub2, :] = chunk.T
                offset += chunk_sz

        return w

    @staticmethod
    def pack_conv1x1_int8(weights: np.ndarray,
                          scale: Optional[np.ndarray] = None,
                          zero_point: Optional[np.ndarray] = None
                          ) -> Tuple[bytes, np.ndarray, np.ndarray]:
        """Pack conv1x1 weights as INT8 into ANE __KERN_0 layout.

        Quantizes FP16/FP32 weights to UINT8 using per-channel symmetric
        quantization (matching Apple's INT8 format), then packs in the
        same 16-core tiled layout as FP16 but at 1 byte per element.

        The ANE hardware decompresses INT8 to FP16 for computation using
        the dequant config encoded in __text W[71] = 0x5042A0C3.

        Args:
            weights: [out_channels, in_channels] weight matrix (FP32 or FP16)
            scale: optional per-channel scale [out_channels] (computed if None)
            zero_point: optional per-channel zero point [out_channels] (computed if None)

        Returns:
            Tuple of (packed_bytes, scale, zero_point) where:
              packed_bytes: INT8 packed weights for __KERN_0
              scale: per-channel dequant scale [out_channels]
              zero_point: per-channel zero point [out_channels]
        """
        NC = WeightPacker.NUM_CORES
        w = weights.astype(np.float32)
        out_ch, in_ch = w.shape

        # Per-channel symmetric quantization
        if scale is None or zero_point is None:
            w_min = w.min(axis=1)
            w_max = w.max(axis=1)
            scale = (w_max - w_min) / 255.0
            scale = np.where(scale == 0, 1.0, scale)  # avoid div by zero
            zero_point = np.round(-w_min / scale).astype(np.uint8)

        # Quantize to UINT8
        w_int8 = np.clip(np.round(w / scale[:, None] + zero_point[:, None].astype(np.float32)),
                         0, 255).astype(np.uint8)

        if out_ch < NC:
            return w_int8.flatten().tobytes(), scale, zero_point

        sub_size = 32
        stripe_size = NC * sub_size

        oc_first = -(-out_ch // NC)
        oc_rest = out_ch // NC
        n_full = oc_rest // sub_size
        rem_first = oc_first - n_full * sub_size
        rem_rest = oc_rest - n_full * sub_size

        result = []
        for t in range(NC):
            for s in range(n_full):
                row_start = t * sub_size + s * stripe_size
                # Column-major packing (same as FP16 but 1 byte/element)
                result.append(w_int8[row_start:row_start + sub_size, :].flatten(order='F'))

            rem = rem_first if t == 0 else rem_rest
            if rem > 0:
                rem_stripe_start = n_full * stripe_size
                if t == 0:
                    row_start = rem_stripe_start
                else:
                    row_start = rem_stripe_start + rem_first + (t - 1) * rem_rest
                result.append(w_int8[row_start:row_start + rem, :].flatten(order='F'))

        packed = np.concatenate(result).tobytes()
        return packed, scale.astype(np.float32), zero_point.astype(np.uint8)

    @staticmethod
    def compute_kern0_size(out_ch: int, in_ch: int) -> int:
        """Compute the __KERN_0 section size needed for given dimensions."""
        return out_ch * in_ch * 2  # FP16

    @staticmethod
    def compute_kern0_size_int8(out_ch: int, in_ch: int) -> int:
        """Compute the __KERN_0 section size for INT8 weights."""
        return out_ch * in_ch  # 1 byte per element

    @staticmethod
    def pack_into_template(template_kern0: bytes, weights: np.ndarray,
                           pwl: Optional['PWLTable'] = None) -> bytes:
        """Pack weights into a template __KERN_0 (for small/atlas models).

        For small models (out_ch < 16) that use tile-replicated templates,
        this preserves tile metadata and replaces weight regions.
        For production models, use pack_conv1x1() directly.
        """
        kern0_size = len(template_kern0)
        out_ch, in_ch = weights.shape

        if out_ch >= WeightPacker.NUM_CORES and out_ch % WeightPacker.NUM_CORES == 0:
            # Production layout: just pack directly
            packed = WeightPacker.pack_conv1x1(weights)
            if len(packed) <= kern0_size:
                buf = bytearray(kern0_size)
                buf[:len(packed)] = packed
                return bytes(buf)
            else:
                return packed  # kern0 needs to be larger than template

        # Small model: tile-replicated packing
        tile_size = kern0_size // WeightPacker.NUM_CORES
        buf = bytearray(template_kern0)
        w_bytes = weights.astype(np.float16).flatten().tobytes()

        weight_offset = 128 if pwl is not None else 0
        pwl_bytes = pwl.to_bytes() if pwl else b''

        for t in range(WeightPacker.NUM_CORES):
            tile_start = t * tile_size
            w_start = tile_start + weight_offset
            w_end = w_start + len(w_bytes)
            if w_end <= len(buf):
                buf[w_start:w_end] = w_bytes
            if pwl_bytes:
                buf[tile_start:tile_start + len(pwl_bytes)] = pwl_bytes

        return bytes(buf)


# ═══════════════════════════════════════════════════════════════
# Full .hwx emission from template (no aned required)
# ═══════════════════════════════════════════════════════════════

def _round_up(x: int, alignment: int) -> int:
    return (x + alignment - 1) & ~(alignment - 1)


def emit_linear_hwx(template_hwx: bytes, in_ch: int, out_ch: int,
                     weights: np.ndarray) -> bytes:
    """Emit a complete .hwx for a linear projection (inner_product/conv1x1).

    Takes a captured reference .hwx as structural template and patches
    all dimension-dependent fields. Generates __text from scratch via
    generate_conv_text(). Packs weights via WeightPacker.pack_conv1x1().

    The template must be a valid inner_product .hwx captured from aned.
    Any dimension works — all dimension-dependent fields are patched.

    Args:
        template_hwx: bytes of a reference .hwx (any inner_product/conv)
        in_ch: input channels
        out_ch: output channels
        weights: [out_ch, in_ch] weight matrix (FP32 or FP16)

    Returns:
        Complete .hwx bytes ready for ANE dispatch
    """
    PAGE = 0x4000

    # Compute layout
    fvmlib0_vmsize = _round_up(in_ch * 64, PAGE)
    fvmlib1_vmsize = _round_up(out_ch * 64, PAGE)
    text_vmaddr = 0x30000000 + fvmlib0_vmsize + fvmlib1_vmsize
    kern0_vmaddr = text_vmaddr + 0x8000
    fvmlib1_vmaddr = 0x30000000 + fvmlib0_vmsize

    weight_data = WeightPacker.pack_conv1x1(weights.reshape(out_ch, in_ch))
    kern0_filesize = _round_up(len(weight_data), PAGE)
    total_filesize = 0xC000 + kern0_filesize

    # Build output buffer from template
    buf = bytearray(total_filesize)
    copy_len = min(len(template_hwx), total_filesize)
    buf[:copy_len] = template_hwx[:copy_len]

    # --- Extract template's original vmaddrs for replacement ---
    ncmds = struct.unpack_from('<I', template_hwx, 0x10)[0]
    tmpl_text_va = tmpl_fv1_va = tmpl_kern0_va = 0
    tmpl_in_ch = tmpl_out_ch = 0
    fv_idx = 0
    off = 32
    seg_lc = []
    thread_lc = []
    for _ in range(ncmds):
        cmd = struct.unpack_from('<I', template_hwx, off)[0]
        cs = struct.unpack_from('<I', template_hwx, off + 4)[0]
        if cmd == 0x19:
            sn = bytes(template_hwx[off+8:off+24]).split(b'\x00')[0].decode('ascii')
            va = struct.unpack_from('<Q', template_hwx, off + 24)[0]
            vs = struct.unpack_from('<Q', template_hwx, off + 32)[0]
            seg_lc.append((off, cs, sn, va, vs))
            if sn == '__FVMLIB':
                if fv_idx == 0:
                    tmpl_in_ch = vs // 64
                elif fv_idx == 1:
                    tmpl_fv1_va = va
                    tmpl_out_ch = vs // 64
                fv_idx += 1
            elif sn == '__TEXT':
                tmpl_text_va = va
            elif sn == '__KERN_0':
                tmpl_kern0_va = va
        elif cmd == 0x04:
            thread_lc.append((off, cs))
        off += cs

    # --- Patch LC_SEGMENT_64 ---
    fv_idx = 0
    for seg_off, seg_cs, sn, old_va, old_vs in seg_lc:
        if sn == '__FVMLIB':
            if fv_idx == 0:
                struct.pack_into('<Q', buf, seg_off + 24, 0x30000000)
                struct.pack_into('<Q', buf, seg_off + 32, fvmlib0_vmsize)
                ns = struct.unpack_from('<I', buf, seg_off + 64)[0]
                if ns > 0:
                    struct.pack_into('<Q', buf, seg_off + 72 + 32, 0x30000000)
                    struct.pack_into('<Q', buf, seg_off + 72 + 40, in_ch * 64)
            else:
                struct.pack_into('<Q', buf, seg_off + 24, fvmlib1_vmaddr)
                struct.pack_into('<Q', buf, seg_off + 32, fvmlib1_vmsize)
                ns = struct.unpack_from('<I', buf, seg_off + 64)[0]
                if ns > 0:
                    struct.pack_into('<Q', buf, seg_off + 72 + 32, fvmlib1_vmaddr)
                    struct.pack_into('<Q', buf, seg_off + 72 + 40, out_ch * 64)
            fv_idx += 1
        elif sn == '__TEXT':
            struct.pack_into('<Q', buf, seg_off + 24, text_vmaddr)
            for s in range(struct.unpack_from('<I', buf, seg_off + 64)[0]):
                s_base = seg_off + 72 + s * 80
                sname = bytes(buf[s_base:s_base+16]).split(b'\x00')[0].decode('ascii')
                if sname == '__text':
                    struct.pack_into('<Q', buf, s_base + 32, text_vmaddr)
                elif sname == '__const':
                    struct.pack_into('<Q', buf, s_base + 32, text_vmaddr + 0x200)
        elif sn == '__KERN_0':
            struct.pack_into('<Q', buf, seg_off + 24, kern0_vmaddr)
            struct.pack_into('<Q', buf, seg_off + 32, kern0_filesize)
            struct.pack_into('<Q', buf, seg_off + 48, kern0_filesize)
            ns = struct.unpack_from('<I', buf, seg_off + 64)[0]
            if ns > 0:
                struct.pack_into('<Q', buf, seg_off + 72 + 32, kern0_vmaddr)
                struct.pack_into('<Q', buf, seg_off + 72 + 40, len(weight_data))
        elif sn == '__LINKEDIT':
            struct.pack_into('<Q', buf, seg_off + 24, kern0_vmaddr + kern0_filesize)

    # --- Patch LC_THREAD commands ---
    if len(thread_lc) >= 3:
        # Thread #0: replace vmaddr references
        t0_off, t0_cs = thread_lc[0]
        nw = (t0_cs - 8) // 4
        words = list(struct.unpack(f'<{nw}I', buf[t0_off+8:t0_off+t0_cs]))
        tile_size = in_ch * out_ch * 2 // NUM_ANE_CORES
        tile_pages = 0 if tile_size < 4096 else (tile_size + 4095) // 4096
        tmpl_tile = (tmpl_in_ch * tmpl_out_ch * 2 // NUM_ANE_CORES + 4095) // 4096

        for i in range(nw):
            v = words[i]
            if v == tmpl_text_va:
                words[i] = text_vmaddr
            elif v == tmpl_text_va + 0x200:
                words[i] = text_vmaddr + 0x200
            elif v == tmpl_fv1_va:
                words[i] = fvmlib1_vmaddr
            elif v == tmpl_kern0_va:
                words[i] = kern0_vmaddr
            elif v == tmpl_tile and tmpl_tile > 0:
                words[i] = tile_pages
        struct.pack_into(f'<{nw}I', buf, t0_off + 8, *words)

        # Thread #1: input descriptor
        t1_off, t1_cs = thread_lc[1]
        nw1 = (t1_cs - 8) // 4
        tmpl_w1 = struct.unpack(f'<{nw1}I', template_hwx[t1_off+8:t1_off+t1_cs])
        words1 = list(struct.unpack(f'<{nw1}I', buf[t1_off+8:t1_off+t1_cs]))
        for i in range(nw1):
            if tmpl_w1[i] == tmpl_in_ch:
                words1[i] = in_ch
            elif tmpl_w1[i] == tmpl_in_ch * 64:
                words1[i] = in_ch * 64
        struct.pack_into(f'<{nw1}I', buf, t1_off + 8, *words1)

        # Thread #2: output descriptor
        t2_off, t2_cs = thread_lc[2]
        nw2 = (t2_cs - 8) // 4
        tmpl_w2 = struct.unpack(f'<{nw2}I', template_hwx[t2_off+8:t2_off+t2_cs])
        words2 = list(struct.unpack(f'<{nw2}I', buf[t2_off+8:t2_off+t2_cs]))
        for i in range(nw2):
            if tmpl_w2[i] == tmpl_out_ch:
                words2[i] = out_ch
            elif tmpl_w2[i] == tmpl_out_ch * 64:
                words2[i] = out_ch * 64
        struct.pack_into(f'<{nw2}I', buf, t2_off + 8, *words2)

    # --- Write __text (from scratch) ---
    text = generate_conv_text(in_ch, out_ch)
    buf[0x4000:0x4000 + len(text)] = text

    # --- Write weights ---
    buf[0xC000:0xC000 + len(weight_data)] = weight_data

    return bytes(buf)


def emit_linear_hwx_int8(template_hwx: bytes, in_ch: int, out_ch: int,
                          weights: np.ndarray) -> Tuple[bytes, np.ndarray, np.ndarray]:
    """Emit a complete .hwx for an INT8 linear projection.

    Same structure as emit_linear_hwx() but with INT8 quantization:
    - __text uses opcode 0x9348 (INT8 conv) instead of 0x9341 (FP16 conv)
    - W[71] = 0x5042A0C3 (INT8 dequant config) instead of out_ch
    - __KERN_0 stores INT8 weights at half the FP16 size
    - Tile sizes adjusted for 1 byte/element density

    The ANE hardware decompresses INT8 to FP16 for computation.
    Prior measurement: INT8 has zero compute speedup (1.005x = noise).
    Benefit is DMA bandwidth: 2x fewer bytes transferred per dispatch.

    Args:
        template_hwx: bytes of a reference .hwx (any inner_product/conv)
        in_ch: input channels
        out_ch: output channels
        weights: [out_ch, in_ch] weight matrix (FP32 or FP16)

    Returns:
        Tuple of (hwx_bytes, scale, zero_point) where:
          hwx_bytes: complete .hwx ready for ANE dispatch
          scale: per-channel dequant scale [out_channels]
          zero_point: per-channel zero point [out_channels]
    """
    PAGE = 0x4000

    # Compute layout (same FVMLIB sizes as FP16 — IO is always FP16)
    fvmlib0_vmsize = _round_up(in_ch * 64, PAGE)
    fvmlib1_vmsize = _round_up(out_ch * 64, PAGE)
    text_vmaddr = 0x30000000 + fvmlib0_vmsize + fvmlib1_vmsize
    kern0_vmaddr = text_vmaddr + 0x8000
    fvmlib1_vmaddr = 0x30000000 + fvmlib0_vmsize

    # INT8 weights: half the size
    weight_data, scale, zero_point = WeightPacker.pack_conv1x1_int8(
        weights.reshape(out_ch, in_ch))
    kern0_filesize = _round_up(len(weight_data), PAGE)
    total_filesize = 0xC000 + kern0_filesize

    # Build output buffer from template
    buf = bytearray(total_filesize)
    copy_len = min(len(template_hwx), total_filesize)
    buf[:copy_len] = template_hwx[:copy_len]

    # --- Extract template's original vmaddrs for replacement ---
    ncmds = struct.unpack_from('<I', template_hwx, 0x10)[0]
    tmpl_text_va = tmpl_fv1_va = tmpl_kern0_va = 0
    tmpl_in_ch = tmpl_out_ch = 0
    fv_idx = 0
    off = 32
    seg_lc = []
    thread_lc = []
    for _ in range(ncmds):
        cmd = struct.unpack_from('<I', template_hwx, off)[0]
        cs = struct.unpack_from('<I', template_hwx, off + 4)[0]
        if cmd == 0x19:
            sn = bytes(template_hwx[off+8:off+24]).split(b'\x00')[0].decode('ascii')
            va = struct.unpack_from('<Q', template_hwx, off + 24)[0]
            vs = struct.unpack_from('<Q', template_hwx, off + 32)[0]
            seg_lc.append((off, cs, sn, va, vs))
            if sn == '__FVMLIB':
                if fv_idx == 0:
                    tmpl_in_ch = vs // 64
                elif fv_idx == 1:
                    tmpl_fv1_va = va
                    tmpl_out_ch = vs // 64
                fv_idx += 1
            elif sn == '__TEXT':
                tmpl_text_va = va
            elif sn == '__KERN_0':
                tmpl_kern0_va = va
        elif cmd == 0x04:
            thread_lc.append((off, cs))
        off += cs

    # --- Patch LC_SEGMENT_64 ---
    fv_idx = 0
    for seg_off, seg_cs, sn, old_va, old_vs in seg_lc:
        if sn == '__FVMLIB':
            if fv_idx == 0:
                struct.pack_into('<Q', buf, seg_off + 24, 0x30000000)
                struct.pack_into('<Q', buf, seg_off + 32, fvmlib0_vmsize)
                ns = struct.unpack_from('<I', buf, seg_off + 64)[0]
                if ns > 0:
                    struct.pack_into('<Q', buf, seg_off + 72 + 32, 0x30000000)
                    struct.pack_into('<Q', buf, seg_off + 72 + 40, in_ch * 64)
            else:
                struct.pack_into('<Q', buf, seg_off + 24, fvmlib1_vmaddr)
                struct.pack_into('<Q', buf, seg_off + 32, fvmlib1_vmsize)
                ns = struct.unpack_from('<I', buf, seg_off + 64)[0]
                if ns > 0:
                    struct.pack_into('<Q', buf, seg_off + 72 + 32, fvmlib1_vmaddr)
                    struct.pack_into('<Q', buf, seg_off + 72 + 40, out_ch * 64)
            fv_idx += 1
        elif sn == '__TEXT':
            struct.pack_into('<Q', buf, seg_off + 24, text_vmaddr)
            for s in range(struct.unpack_from('<I', buf, seg_off + 64)[0]):
                s_base = seg_off + 72 + s * 80
                sname = bytes(buf[s_base:s_base+16]).split(b'\x00')[0].decode('ascii')
                if sname == '__text':
                    struct.pack_into('<Q', buf, s_base + 32, text_vmaddr)
                elif sname == '__const':
                    struct.pack_into('<Q', buf, s_base + 32, text_vmaddr + 0x200)
        elif sn == '__KERN_0':
            struct.pack_into('<Q', buf, seg_off + 24, kern0_vmaddr)
            struct.pack_into('<Q', buf, seg_off + 32, kern0_filesize)
            struct.pack_into('<Q', buf, seg_off + 48, kern0_filesize)
            ns = struct.unpack_from('<I', buf, seg_off + 64)[0]
            if ns > 0:
                struct.pack_into('<Q', buf, seg_off + 72 + 32, kern0_vmaddr)
                struct.pack_into('<Q', buf, seg_off + 72 + 40, len(weight_data))
        elif sn == '__LINKEDIT':
            struct.pack_into('<Q', buf, seg_off + 24, kern0_vmaddr + kern0_filesize)

    # --- Patch LC_THREAD commands ---
    if len(thread_lc) >= 3:
        t0_off, t0_cs = thread_lc[0]
        nw = (t0_cs - 8) // 4
        words = list(struct.unpack(f'<{nw}I', buf[t0_off+8:t0_off+t0_cs]))
        tile_size = in_ch * out_ch * 1 // NUM_ANE_CORES  # INT8: 1 byte/element
        tile_pages = 0 if tile_size < 4096 else (tile_size + 4095) // 4096
        tmpl_tile = (tmpl_in_ch * tmpl_out_ch * 2 // NUM_ANE_CORES + 4095) // 4096

        for i in range(nw):
            v = words[i]
            if v == tmpl_text_va:
                words[i] = text_vmaddr
            elif v == tmpl_text_va + 0x200:
                words[i] = text_vmaddr + 0x200
            elif v == tmpl_fv1_va:
                words[i] = fvmlib1_vmaddr
            elif v == tmpl_kern0_va:
                words[i] = kern0_vmaddr
            elif v == tmpl_tile and tmpl_tile > 0:
                words[i] = tile_pages
        struct.pack_into(f'<{nw}I', buf, t0_off + 8, *words)

        # Thread #1: input descriptor
        t1_off, t1_cs = thread_lc[1]
        nw1 = (t1_cs - 8) // 4
        tmpl_w1 = struct.unpack(f'<{nw1}I', template_hwx[t1_off+8:t1_off+t1_cs])
        words1 = list(struct.unpack(f'<{nw1}I', buf[t1_off+8:t1_off+t1_cs]))
        for i in range(nw1):
            if tmpl_w1[i] == tmpl_in_ch:
                words1[i] = in_ch
            elif tmpl_w1[i] == tmpl_in_ch * 64:
                words1[i] = in_ch * 64
        struct.pack_into(f'<{nw1}I', buf, t1_off + 8, *words1)

        # Thread #2: output descriptor
        t2_off, t2_cs = thread_lc[2]
        nw2 = (t2_cs - 8) // 4
        tmpl_w2 = struct.unpack(f'<{nw2}I', template_hwx[t2_off+8:t2_off+t2_cs])
        words2 = list(struct.unpack(f'<{nw2}I', buf[t2_off+8:t2_off+t2_cs]))
        for i in range(nw2):
            if tmpl_w2[i] == tmpl_out_ch:
                words2[i] = out_ch
            elif tmpl_w2[i] == tmpl_out_ch * 64:
                words2[i] = out_ch * 64
        struct.pack_into(f'<{nw2}I', buf, t2_off + 8, *words2)

    # --- Write INT8 __text (from scratch) ---
    text = generate_conv_text_int8(in_ch, out_ch)
    buf[0x4000:0x4000 + len(text)] = text

    # --- Write INT8 weights ---
    buf[0xC000:0xC000 + len(weight_data)] = weight_data

    return bytes(buf), scale, zero_point


# ═══════════════════════════════════════════════════════════════
# HWX Writer
# ═══════════════════════════════════════════════════════════════

class HWXWriter:
    """Assemble a complete .hwx binary from template + modifications.

    The writer starts from a template binary and replaces specific regions:
    - __text: kernel program (activation encoding)
    - __kern_0: weights + PWL
    - symtab: tile descriptor copy
    - Header bytes: kernel count references

    The template provides everything else: Mach-O header, segment layout,
    LC_THREAD descriptors, compiler metadata, pipeline config.
    """

    def __init__(self, template: HWXTemplate):
        self.template = template
        self.output = bytearray(template.data)

    def set_text(self, text_data: bytes):
        """Replace __text kernel program."""
        if len(text_data) != self.template.text_size:
            raise ValueError(
                f"__text size mismatch: template={self.template.text_size}, "
                f"new={len(text_data)}. Use a template with matching __text size.")
        self.output[self.template.text_offset:
                    self.template.text_offset + len(text_data)] = text_data

    def set_kern0(self, kern0_data: bytes):
        """Replace __kern_0 weights/PWL data."""
        off = self.template.kern0_offset
        self.output[off:off + len(kern0_data)] = kern0_data

    def set_tile_replication(self, data: bytes):
        """Replace the full 0xC000-to-EOF region (weights + PWL + tile copies)."""
        self.output[0xC000:0xC000 + len(data)] = data

    def set_pwl_table(self, pwl: PWLTable):
        """Replace only the 84-byte PWL table at 0xC000."""
        pwl_bytes = pwl.to_bytes()
        self.output[0xC000:0xC000 + len(pwl_bytes)] = pwl_bytes

    def set_symtab_tiles(self, symtab_data: bytes):
        """Replace symtab region (contains tile descriptor copy)."""
        off = self.template.symtab_offset
        self.output[off:off + len(symtab_data)] = symtab_data

    def patch_fp32_at(self, word_index: int, value: float):
        """Patch a FP32 constant in __text at the given word index.
        Used for layernorm epsilon and 1/dim constants."""
        off = self.template.text_offset + word_index * 4
        struct.pack_into('<f', self.output, off, value)

    def patch_kern0_region(self, offset: int, data: bytes):
        """Patch a region within __KERN_0."""
        abs_off = self.template.kern0_offset + offset
        self.output[abs_off:abs_off + len(data)] = data

    def build(self) -> bytes:
        """Return the assembled .hwx binary."""
        return bytes(self.output)

    def write(self, path: str):
        """Write the assembled .hwx to a file."""
        Path(path).write_bytes(self.build())


# ═══════════════════════════════════════════════════════════════
# High-level emitter
# ═══════════════════════════════════════════════════════════════

class ANECompiler:
    """High-level compiler: model definition → .hwx binary.

    Usage:
        compiler = ANECompiler('/path/to/templates/')

        # Emit a simple activation
        hwx = compiler.emit_activation(ActivationType.SILU)

        # Emit a conv + activation (FFN layer)
        hwx = compiler.emit_conv_activation(
            weights=np.random.randn(8, 8).astype(np.float16),
            activation=ActivationType.SILU
        )

        # Emit with custom PWL activation
        pwl = PWLTable.extract_from_hwx('reference_silu.hwx')
        hwx = compiler.emit_conv_activation(weights=w, activation_pwl=pwl)
    """

    def __init__(self, template_dir: str):
        self.registry = TemplateRegistry()
        self.registry.load_directory(template_dir)

    def emit_activation(self, activation: ActivationType,
                        output_path: Optional[str] = None) -> bytes:
        """Emit a standalone activation .hwx (no weights)."""
        template = self.registry.get_activation_template(activation)
        writer = HWXWriter(template)

        # For 64K class with different PWL: swap the table
        if (activation == ActivationType.CUSTOM_PWL and
            template.template_class == '64k_activation'):
            raise ValueError("Custom PWL requires a PWLTable argument. "
                           "Use emit_activation_pwl() instead.")

        result = writer.build()
        if output_path:
            writer.write(output_path)
        return result

    def emit_activation_pwl(self, pwl: PWLTable,
                            output_path: Optional[str] = None) -> bytes:
        """Emit a 64K activation with custom PWL table."""
        # Use any 64K template — the __text is shared
        template = self.registry.get_activation_template(ActivationType.SILU)
        writer = HWXWriter(template)
        writer.set_pwl_table(pwl)

        result = writer.build()
        if output_path:
            writer.write(output_path)
        return result

    def emit_conv_activation(self, weights: np.ndarray,
                             activation: ActivationType = ActivationType.RELU,
                             activation_pwl: Optional[PWLTable] = None,
                             output_path: Optional[str] = None) -> bytes:
        """Emit a fused conv + activation .hwx.

        This produces a single kernel that computes:
            output = activation(conv1x1(input, weights))

        For production dimensions (out_ch >= 16), weights are packed in
        16-core partitioned layout. The .hwx file is resized to fit.

        Args:
            weights: [out_channels, in_channels] weight matrix
            activation: activation type (default ReLU)
            activation_pwl: custom PWL table (for CUSTOM_PWL activation)
            output_path: optional file path to write

        Returns:
            Complete .hwx binary
        """
        out_ch, in_ch = weights.shape
        needs_pwl = activation in (ActivationType.SILU, ActivationType.GELU,
                                   ActivationType.SIGMOID, ActivationType.CUSTOM_PWL)

        conv_template = self.registry.get_conv_template(with_pwl=needs_pwl)

        # Determine PWL table
        pwl = activation_pwl
        if pwl is None and needs_pwl:
            target_template = self.registry.get_activation_template(activation)
            if target_template.pwl_data:
                pwl = PWLTable.from_bytes(target_template.pwl_data)

        # For production dimensions, pack weights in 16-core layout
        if out_ch >= 16 and out_ch % 16 == 0:
            kern0_data = WeightPacker.pack_conv1x1(weights)
        else:
            template_kern0 = conv_template.data[
                conv_template.kern0_offset:
                conv_template.kern0_offset + conv_template.kern0_size]
            kern0_data = WeightPacker.pack_into_template(template_kern0, weights, pwl)

        writer = HWXWriter(conv_template)

        # If kern0 is larger than template, we need to resize the file
        needed_kern0 = len(kern0_data)
        template_kern0_size = conv_template.kern0_size
        if needed_kern0 > template_kern0_size:
            # Extend the output buffer to accommodate larger __KERN_0
            # __KERN_0 starts at kern0_offset, extends to kern0_offset + needed_kern0
            # File must be page-aligned (4096-byte boundaries)
            new_end = conv_template.kern0_offset + needed_kern0
            new_size = ((new_end + 4095) // 4096) * 4096
            extended = bytearray(new_size)
            extended[:len(conv_template.data)] = conv_template.data
            writer.output = extended

            # Update __KERN_0 section size in the load command
            self._patch_kern0_size(writer.output, needed_kern0, new_size)

        writer.set_kern0(kern0_data)

        result = writer.build()
        if output_path:
            Path(output_path).write_bytes(result)
        return result

    @staticmethod
    def _patch_kern0_size(buf: bytearray, new_kern0_size: int, new_file_size: int):
        """Patch the __KERN_0 segment/section size and file size in load commands."""
        ncmds = struct.unpack_from('<I', buf, 0x10)[0]
        offset = 32
        for _ in range(ncmds):
            cmd = struct.unpack_from('<I', buf, offset)[0]
            cmdsize = struct.unpack_from('<I', buf, offset + 4)[0]
            if cmd == 0x19:  # LC_SEGMENT_64
                segname = buf[offset+8:offset+24].split(b'\x00')[0].decode('ascii')
                if segname == '__KERN_0':
                    # Patch segment vmsize and filesize
                    struct.pack_into('<Q', buf, offset + 40, new_kern0_size)  # vmsize
                    struct.pack_into('<Q', buf, offset + 48, new_kern0_size)  # filesize
                    # Patch section size
                    nsects = struct.unpack_from('<I', buf, offset + 64)[0]
                    for s in range(nsects):
                        s_off = offset + 72 + s * 80
                        struct.pack_into('<Q', buf, s_off + 40, new_kern0_size)  # size
            offset += cmdsize

    def emit_softmax(self, params: Optional[SoftmaxParams] = None,
                     output_path: Optional[str] = None) -> bytes:
        """Emit a softmax .hwx (5-pass: reduce_max → exp → sum → reciprocal → multiply).

        The template provides the complete 5-pass __text program. The only
        variable data is the two PWL tables in __KERN_0 (exp and reciprocal).
        """
        template = self.registry.get_softmax_template()
        writer = HWXWriter(template)

        if params:
            kern0 = params.get_kern0(
                template.data[template.kern0_offset:
                              template.kern0_offset + template.kern0_size])
            writer.set_kern0(kern0)

        result = writer.build()
        if output_path:
            writer.write(output_path)
        return result

    def emit_layernorm(self, params: Optional[LayerNormParams] = None,
                       output_path: Optional[str] = None) -> bytes:
        """Emit a layernorm .hwx (4-pass: mean → variance → normalize → scale+shift).

        The template provides the complete 4-pass __text program.
        Constants (epsilon, 1/dim) are FP32 literals embedded in the instruction stream.
        """
        template = self.registry.get_layernorm_template()
        writer = HWXWriter(template)

        if params:
            # Patch epsilon in pass 3 (word 105 in the original template)
            # Find the epsilon constant (0x37280000 = ~1e-5 FP32)
            passes = template.passes
            for p in passes:
                for i, w in enumerate(p.words):
                    if w == 0x37280000:  # default epsilon
                        abs_word = p.word_offset + i
                        writer.patch_fp32_at(abs_word, params.epsilon)
                    elif w == 0x3C800000:  # 1/dim (0.015625 = 1/64)
                        abs_word = p.word_offset + i
                        inv_dim = 1.0 / params.dim
                        writer.patch_fp32_at(abs_word, inv_dim)

        result = writer.build()
        if output_path:
            writer.write(output_path)
        return result

    def emit_batchnorm(self, params: BatchNormParams,
                       output_path: Optional[str] = None) -> bytes:
        """Emit a batchnorm .hwx (single-pass scale+bias).

        Implements: output = gamma * input + beta

        Used after MVN dispatch to add learned affine parameters to LayerNorm.
        Template-based: replaces __const data with packed gamma/beta.

        The batchnorm template has:
          - __text: 320B (80 words), opcode 0x9D41 (multi-pass with const data)
          - __const: dim*4 bytes = [beta/gamma (FP16), gamma (FP16)]

        Args:
            params: BatchNormParams with gamma and beta arrays
            output_path: optional path to write .hwx

        Returns:
            Complete .hwx bytes ready for ANE dispatch
        """
        template = self.registry.get_batchnorm_template()
        writer = HWXWriter(template)

        # Pack gamma/beta into __const format
        const_data = params.pack_const()

        # Find __const offset and size in the template
        # __const starts at text_offset + text_size (rounded up to alignment)
        ncmds = struct.unpack_from('<I', template.data, 0x10)[0]
        offset = 32
        const_off = 0
        for _ in range(ncmds):
            cmd = struct.unpack_from('<I', template.data, offset)[0]
            cmdsize = struct.unpack_from('<I', template.data, offset + 4)[0]
            if cmd == 0x19:  # LC_SEGMENT_64
                segname = template.data[offset+8:offset+24].split(b'\x00')[0].decode('ascii')
                nsects = struct.unpack_from('<I', template.data, offset+64)[0]
                for s in range(nsects):
                    s_off = offset + 72 + s * 80
                    sectname = template.data[s_off:s_off+16].split(b'\x00')[0].decode('ascii')
                    sect_foff = struct.unpack_from('<I', template.data, s_off+48)[0]
                    if segname == '__TEXT' and sectname == '__const':
                        const_off = sect_foff
            offset += cmdsize

        if const_off == 0:
            raise ValueError("Batchnorm template has no __const section")

        # Write packed gamma/beta to __const
        writer.output[const_off:const_off + len(const_data)] = const_data

        result = writer.build()
        if output_path:
            Path(output_path).write_bytes(result)
        return result

    def emit_layernorm_affine(self, gamma: np.ndarray, beta: np.ndarray,
                              epsilon: float = 1e-5,
                              output_path: Optional[str] = None) -> Tuple[bytes, bytes]:
        """Emit LayerNorm with learned affine parameters as two .hwx files.

        Implements: output = gamma * LayerNorm(input, eps) + beta

        Returns two .hwx files:
          1. MVN .hwx (mean-variance normalization)
          2. Batchnorm .hwx (gamma * x + beta)

        Chain them via ane-dispatch IOSurface routing.

        Args:
            gamma: per-channel scale [dim] (FP32)
            beta: per-channel bias [dim] (FP32)
            epsilon: LayerNorm epsilon
            output_path: base path (writes .mvn.hwx and .affine.hwx)

        Returns:
            Tuple (mvn_hwx_bytes, affine_hwx_bytes)
        """
        dim = len(gamma)

        # 1. MVN .hwx
        ln_params = LayerNormParams(epsilon=epsilon, dim=dim)
        mvn_hwx = self.emit_layernorm(params=ln_params)

        # 2. Batchnorm .hwx (affine: gamma*x + beta)
        bn_params = BatchNormParams(gamma=gamma, beta=beta)
        affine_hwx = self.emit_batchnorm(params=bn_params)

        if output_path:
            base = Path(output_path)
            base.with_suffix('.mvn.hwx').write_bytes(mvn_hwx)
            base.with_suffix('.affine.hwx').write_bytes(affine_hwx)

        return mvn_hwx, affine_hwx

    def emit_ffn(self, weights_gate: np.ndarray, weights_down: np.ndarray,
                 activation: ActivationType = ActivationType.SILU,
                 activation_pwl: Optional[PWLTable] = None,
                 output_path: Optional[str] = None) -> Tuple[bytes, bytes]:
        """Emit an FFN as two .hwx files: (gate+activation, down_projection).

        Computes: output = down_proj(activation(gate_proj(input)))

        Each .hwx is a single ANE dispatch. Chain them via ane-dispatch
        with IOSurface routing (output of gate feeds input of down).

        Args:
            weights_gate: [hidden, input] gate/up projection weights
            weights_down: [output, hidden] down projection weights
            activation: activation between the two linears (default SiLU)
            activation_pwl: custom PWL table (for CUSTOM_PWL)
            output_path: base path (writes .gate.hwx and .down.hwx)

        Returns:
            Tuple (gate_hwx_bytes, down_hwx_bytes)
        """
        gate_hwx = self.emit_conv_activation(
            weights=weights_gate,
            activation=activation,
            activation_pwl=activation_pwl)

        down_hwx = self.emit_conv_activation(
            weights=weights_down,
            activation=ActivationType.RELU)

        if output_path:
            base = Path(output_path)
            base.with_suffix('.gate.hwx').write_bytes(gate_hwx)
            base.with_suffix('.down.hwx').write_bytes(down_hwx)

        return gate_hwx, down_hwx

    def list_templates(self) -> Dict[str, str]:
        """List all available templates and their classes."""
        return {name: f"{t.template_class} ({t.text_size}B __text, "
                      f"{t.num_passes} pass{'es' if t.num_passes > 1 else ''})"
                for name, t in self.registry.templates.items()}


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='ANE Compiler: emit .hwx binaries from model definitions')
    parser.add_argument('--templates', required=True,
                       help='Path to template .hwx directory (or multiple, comma-separated)')
    parser.add_argument('--template-file', action='append', default=[],
                       help='Load individual template .hwx (repeatable)')
    parser.add_argument('--output', '-o', help='Output .hwx path')
    parser.add_argument('--list-templates', action='store_true',
                       help='List available templates')

    sub = parser.add_subparsers(dest='mode')

    # Activation mode
    act_p = sub.add_parser('activation', help='Emit standalone activation')
    act_p.add_argument('type', choices=[a.value for a in ActivationType])

    # Softmax mode
    sub.add_parser('softmax', help='Emit softmax (5-pass)')

    # LayerNorm mode
    ln_p = sub.add_parser('layernorm', help='Emit layernorm (4-pass)')
    ln_p.add_argument('--epsilon', type=float, default=1e-5)
    ln_p.add_argument('--dim', type=int, default=64)

    # Conv mode
    conv_p = sub.add_parser('conv', help='Emit conv + activation')
    conv_p.add_argument('--weights', required=True, help='Weights .npy file')
    conv_p.add_argument('--activation', default='relu',
                       choices=[a.value for a in ActivationType])

    # Batchnorm mode (for LN affine params)
    bn_p = sub.add_parser('batchnorm', help='Emit batchnorm (scale+bias)')
    bn_p.add_argument('--gamma', required=True, help='Gamma weights .npy')
    bn_p.add_argument('--beta', required=True, help='Beta weights .npy')

    # LayerNorm+affine mode (MVN + batchnorm)
    lna_p = sub.add_parser('layernorm-affine', help='Emit layernorm with affine (MVN + batchnorm)')
    lna_p.add_argument('--gamma', required=True, help='Gamma weights .npy')
    lna_p.add_argument('--beta', required=True, help='Beta weights .npy')
    lna_p.add_argument('--epsilon', type=float, default=1e-5)

    # FFN mode
    ffn_p = sub.add_parser('ffn', help='Emit FFN (gate+activation + down)')
    ffn_p.add_argument('--gate-weights', required=True, help='Gate weights .npy')
    ffn_p.add_argument('--down-weights', required=True, help='Down weights .npy')
    ffn_p.add_argument('--activation', default='silu',
                       choices=[a.value for a in ActivationType])

    args = parser.parse_args()

    # Build compiler
    compiler = ANECompiler.__new__(ANECompiler)
    compiler.registry = TemplateRegistry()
    for d in args.templates.split(','):
        d = d.strip()
        if os.path.isdir(d):
            compiler.registry.load_directory(d)
    for f in args.template_file:
        compiler.registry.load_file(f)

    if args.list_templates:
        print("Available templates:")
        for name, info in compiler.list_templates().items():
            print(f"  {name}: {info}")
        sys.exit(0)

    if not args.output:
        parser.error("--output is required for emission")

    if args.mode == 'activation':
        act = ActivationType(args.type)
        hwx = compiler.emit_activation(act, args.output)
        print(f"Emitted {act.value} activation: {len(hwx)} bytes → {args.output}")

    elif args.mode == 'softmax':
        hwx = compiler.emit_softmax(output_path=args.output)
        print(f"Emitted softmax (5-pass): {len(hwx)} bytes → {args.output}")

    elif args.mode == 'layernorm':
        params = LayerNormParams(epsilon=args.epsilon, dim=args.dim)
        hwx = compiler.emit_layernorm(params, output_path=args.output)
        print(f"Emitted layernorm (eps={args.epsilon}, dim={args.dim}): "
              f"{len(hwx)} bytes → {args.output}")

    elif args.mode == 'conv':
        w = np.load(args.weights)
        act = ActivationType(args.activation)
        hwx = compiler.emit_conv_activation(w, act, output_path=args.output)
        print(f"Emitted conv+{act.value}: {len(hwx)} bytes → {args.output}")

    elif args.mode == 'batchnorm':
        gamma = np.load(args.gamma)
        beta = np.load(args.beta)
        params = BatchNormParams(gamma=gamma, beta=beta)
        hwx = compiler.emit_batchnorm(params, output_path=args.output)
        print(f"Emitted batchnorm (dim={len(gamma)}): {len(hwx)} bytes → {args.output}")

    elif args.mode == 'layernorm-affine':
        gamma = np.load(args.gamma)
        beta = np.load(args.beta)
        mvn_hwx, affine_hwx = compiler.emit_layernorm_affine(
            gamma, beta, epsilon=args.epsilon, output_path=args.output)
        print(f"Emitted layernorm+affine (dim={len(gamma)}, eps={args.epsilon}): "
              f"mvn={len(mvn_hwx)}B, affine={len(affine_hwx)}B → {args.output}")

    elif args.mode == 'ffn':
        gate_w = np.load(args.gate_weights)
        down_w = np.load(args.down_weights)
        act = ActivationType(args.activation)
        gate_hwx, down_hwx = compiler.emit_ffn(
            gate_w, down_w, activation=act, output_path=args.output)
        print(f"Emitted FFN ({act.value}): "
              f"gate={len(gate_hwx)}B, down={len(down_hwx)}B → {args.output}")

    else:
        parser.print_help()
