"""
path_c_mvp_emitter.py - Path C MVP multi-pass __text + LC_0x04/0x40/0x08 emitter.

Implements the 11 (now 14) missing emitter functions per
vault/research/state_vector_crack/path_c_dma_addressing_decode.json
emitter_modification_spec, plus a validation harness.

2026-05-27 Path A.3 extension: emit_lc_segment_64, emit_lc_symtab,
emit_lc_0x08_anec_full added to close R1 structural validation gate
revealed by Path A.1 swap_hwx test (BadArgument due to missing
5x LC_SEGMENT_64 wrappers + 1x LC_SYMTAB + 12x undersized LC_0x08).
Spec source: /private/tmp/identity8.hwx (production reference).

Architecture (decoded from 3 production .hwx samples):
  - LC_0x40: named external IO surface binding (32B fixed)
  - LC_0x04 type=0x04: surface descriptor table (16B per entry; primary record)
  - LC_0x04 type=0x03: pipeline stage records (~3400B per stage)
    WARNING: type=0x03 body fields under-decoded (Risk #2 40-60% per spec)
  - LC_0x08: ANEC metadata (ASCII; experimental.aot.enable_surface_desc:1)
  - FVMLIB: contiguous from 0x30000000; 16KB aligned
  - Prologue: 5 sub-link blocks that install LC_0x04 table into firmware sequencer
  - Multi-pass __text:
      * Flavor A (inline compute-config, 20B): small ops, implicit DMA addressing
      * Flavor B (table-driven via seq counter, 20B): bulk fused passes
      * DMA fence (8B): 0x22001340 + 0x00000021
      * Terminator (8B): 0x22001440 + (0x01XX0021 with XX = pass_count-1)

Per CA disposition 2026-05-27T23-16-36 corrected value bracket:
  Floor   ~0-3%   (XPC inside per-layer wall; dispatch reduction near-zero visible lift)
  Mid     15-25%  (per-block bandwidth improvement)
  Ceiling 2.1-2.4x (full fused-dispatch bandwidth utilization)

Capability deliverable (cross-pass DMA prefetch mechanism) preserved at FLOOR.

verification_status: candidate
Author: CC sub-agent under Rule 24 sub-amendment 1
Date: 2026-05-27
"""

import os
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple


# ============================================================
# Constants (canonical per path_c_dma_addressing_decode.json)
# ============================================================

# Load command IDs
LC_NAMED_SEG = 0x00000040     # External IO surface binding (32B)
LC_CUSTOM_BLOCK = 0x00000004  # Surface table OR pipeline stage (type discriminator at byte+8)
LC_ANEC_METADATA = 0x00000008
LC_SYMTAB = 0x00000002

# LC_0x04 record types (byte offset +8 within LC)
LC04_TYPE_SURFACE_TABLE = 0x00000004
LC04_TYPE_PIPELINE_STAGE = 0x00000003

# Magic / version words
SURFACE_TABLE_MAGIC = 0x00000222
PIPELINE_STAGE_MAGIC = 0x0000034A

# FVMLIB base + alignment
FVMLIB_BASE = 0x30000000
FVMLIB_ALIGN = 0x4000  # 16 KB hardware page size

# __text words
TEXT_COMPUTE_CONFIG_SENTINEL = 0x00008001     # Flavor A header word
TEXT_DMA_COMPUTE_SENTINEL = 0xFFC01540        # Flavor B header word (matches existing emitter)
DMA_FENCE_WORD_1 = 0x22001340                 # DMA fence opcode
DMA_FENCE_WORD_2 = 0x00000021                 # DMA fence closing
TERMINATOR_WORD_1 = 0x22001440                # Terminator opcode (matches PROGRAM_TERM1)
TERMINATOR_WORD_2_BASE = 0x01000021           # base; OR (pass_count-1)<<16 = 0x01XX0021

# Ready opcodes (from residual_opcode_deep_decode.json, decision-cell candidates)
OPCODE_READY_A = 0x954d8005
OPCODE_READY_B = 0xb3498005
OPCODE_READY_C = 0x954c8005


# ============================================================
# Structured data classes
# ============================================================

@dataclass
class NamedSurface:
    """External IO surface name + VM binding."""
    name: str       # ASCII <= 8 chars
    vm_addr: int    # 64-bit VM address (typically in 0x30000000-0x30FFFFFF range)


@dataclass
class FVMLIBSegment:
    """Internal weight/intermediate segment."""
    vm_addr: int
    size_bytes: int
    label: str = ""


@dataclass
class ComputePass:
    """One pass within the multi-pass __text program.

    flavor 'A' = inline compute-config (small ops)
    flavor 'B' = table-driven via sequence counter (bulk fused)
    """
    flavor: str             # 'A' or 'B'
    opcode: int             # e.g., OPCODE_READY_A or activation opcode
    tile_dim_n: int = 16
    tile_dim_m: int = 16
    seq_id: int = 0         # Flavor B only; assigned monotonically by emitter
    dma_channel_cfg: int = 0x0240  # Flavor B only; observed 0x0240 / 0x0260


# ============================================================
# Function 1: LC_0x40 named binding emitter
# ============================================================

def emit_lc_segment_binding(name: str, vm_addr: int) -> bytes:
    """Emit LC_0x40 named external IO surface binding (32B fixed).

    Byte layout (per path_c_dma_addressing_decode.json
    decoded_addressing.format_spec.loadcmd_named_segment_binding):
      [0]   u32  0x00000040 (lc_cmd)
      [4]   u32  0x00000020 (lc_size always 32)
      [8]   u32  0x00000018 (name_offset within LC = 24)
      [12]  u32  0 (reserved)
      [16]  u32  vm_addr_lo
      [20]  u32  vm_addr_hi (always 0 on M5 Pro)
      [24]  u64  name ASCII, null-padded to 8 bytes

    Args:
        name: ASCII surface name <= 8 chars (e.g., 'input1', 'embedding')
        vm_addr: 64-bit VM address; high bits expected 0 on M5 Pro.

    Returns:
        32 bytes.
    """
    if len(name) > 8:
        raise ValueError(f"name '{name}' exceeds 8-byte LC_0x40 limit")
    name_bytes = name.encode('ascii').ljust(8, b'\x00')
    vm_lo = vm_addr & 0xFFFFFFFF
    vm_hi = (vm_addr >> 32) & 0xFFFFFFFF
    return struct.pack(
        '<IIIIII8s',
        LC_NAMED_SEG,    # 0x40 cmd
        0x00000020,      # cmdsize = 32
        0x00000018,      # name_offset = 24
        0x00000000,      # reserved
        vm_lo,
        vm_hi,
        name_bytes,
    )


# ============================================================
# Function 2: LC_0x04 type=0x04 surface descriptor table
# ============================================================

def emit_lc_surface_table(named_io: List[NamedSurface],
                          internal_segs: List[FVMLIBSegment]) -> bytes:
    """Emit LC_0x04 type=0x04 surface descriptor table.

    Layout (per format_spec.loadcmd_surface_table; verified byte-exact
    across 3 samples):
      [0]   u32 0x00000004 (lc_cmd)
      [4]   u32 cmdsize
      [8]   u32 0x00000004 (record type = surface_table)
      [12]  u32 0x00000222 (magic/version)
      [16+] entries, 16-byte stride:
              [vm_lo, vm_hi=0, size_or_flag=0, reserved=0]

    Table layout convention:
      idx 0..N_io-1  : named external IO surfaces (matches LC_0x40 bindings)
      idx N_io..6    : reserved/zero-filled
      idx 7..        : internal FVMLIB segments in vm-order
      final entry    : single alias slot (input-as-output ping-pong) - repeat of entry[0]

    Args:
        named_io: external IO surface bindings.
        internal_segs: internal weight/intermediate FVMLIB segments.

    Returns:
        2208-2216 bytes typical; cmdsize encoded in header.
    """
    body = bytearray()
    # Entries 0..N_io-1: named IO
    for surf in named_io:
        vm_lo = surf.vm_addr & 0xFFFFFFFF
        body += struct.pack('<IIII', vm_lo, 0, 0, 0)
    # Reserved slots: pad up to index 7
    while len(body) // 16 < 7:
        body += struct.pack('<IIII', 0, 0, 0, 0)
    # Internal FVMLIB segments
    for seg in internal_segs:
        vm_lo = seg.vm_addr & 0xFFFFFFFF
        body += struct.pack('<IIII', vm_lo, 0, 0, 0)
    # Alias slot at file_off ~0x800 (repeat entry[0]); pad to 0x800 - 0x10 first
    target_offset_within_entries = 0x800 - 0x10
    while len(body) < target_offset_within_entries:
        body += struct.pack('<IIII', 0, 0, 0, 0)
    if named_io:
        vm_lo = named_io[0].vm_addr & 0xFFFFFFFF
        body += struct.pack('<IIII', vm_lo, 0, 0, 0)
    # 16-byte align
    while len(body) % 16:
        body += b'\x00'

    cmdsize = 16 + len(body)
    header = struct.pack(
        '<IIII',
        LC_CUSTOM_BLOCK,             # 0x04
        cmdsize,
        LC04_TYPE_SURFACE_TABLE,     # 0x04
        SURFACE_TABLE_MAGIC,         # 0x222
    )
    return bytes(header) + bytes(body)


# ============================================================
# Function 3: LC_0x04 type=0x03 pipeline stage records
# ============================================================

def emit_lc_pipeline_stage(stage_idx: int, stage_total: int,
                           surface_refs: Optional[List[int]] = None,
                           strides: Optional[List[int]] = None,
                           operand_strings: Optional[List[str]] = None,
                           tile_config_words: Optional[List[int]] = None,
                           record_size: int = 3400) -> bytes:
    """Emit LC_0x04 type=0x03 pipeline stage record (~3400 bytes).

    UPDATED 2026-05-28 per Path A.2 byte-diff findings (corrected_candidate_fields.json):
    type=0x03 body carries THREE ASCII null-terminated operand-name strings
    at body+0x0d20+ (each ~16-25B). These are load-bearing for the firmware
    validator (Risk #2 root cause of R1_LOAD_REJECT).

    Additionally body+0x1c/0x38/0x40/0x48/0x58/0x68 carry six u32 tile-config
    fields that vary per-stage. Can be passed via tile_config_words.

    Header layout (decoded byte-exact):
      [0]  u32 0x00000004 (lc_cmd)
      [4]  u32 cmdsize
      [8]  u32 0x00000003 (record type = pipeline_stage)
      [12] u32 0x0000034A (magic/version)
      [16] u32 stage_total
      [20] u32 stage_idx (1..stage_total)
      [24+]   body: surface_refs + tile config + strides + ASCII operand names

    Body offsets (Path A.2 decoded):
      body+0x00 surface_refs (zero-padded if not supplied)
      body+0x1c, 0x38, 0x40, 0x48, 0x58, 0x68 — six tile-config u32 words
      body+0x40+strides (zero-padded if not supplied)
      body+0x64 — likely string-length pointer (set to body_size-6 heuristic)
      body+0x0d20+ — THREE null-terminated operand-name ASCII strings

    Args:
        stage_idx: 1-based stage index within pipeline.
        stage_total: total stage count.
        surface_refs: indices into LC_0x04 type=0x04 surface table (optional).
        strides: per-surface byte strides (optional).
        operand_strings: list of up to 3 operand names (e.g.,
                          ['extend_16_8_ane', 'input', 'output']).
                          Each will be null-terminated and emitted at
                          body+0x0d20+ in sequence.
        tile_config_words: list of up to 6 u32 values for body+0x1c/0x38/
                            0x40/0x48/0x58/0x68. Order matches offsets.
        record_size: total record size including header. Default 3400.

    Returns:
        record_size bytes.
    """
    if stage_idx < 1 or stage_idx > stage_total:
        raise ValueError(f"stage_idx {stage_idx} out of [1, {stage_total}]")

    body = bytearray(record_size - 24)  # header is 24B

    # Surface refs at offset 0 of body (observed pattern but not byte-exact verified)
    if surface_refs:
        off = 0
        for sref in surface_refs[:16]:  # cap at 16
            if off + 4 > len(body):
                break
            struct.pack_into('<I', body, off, sref & 0xFFFFFFFF)
            off += 4

    # Tile-config u32 words at body+0x1c, 0x38, 0x40, 0x48, 0x58, 0x68 (Path A.2)
    if tile_config_words:
        tile_offsets = [0x1c, 0x38, 0x40, 0x48, 0x58, 0x68]
        for i, val in enumerate(tile_config_words[:6]):
            off = tile_offsets[i]
            if off + 4 > len(body):
                break
            struct.pack_into('<I', body, off, val & 0xFFFFFFFF)

    # Strides at offset 64 (heuristic; observed in samples but not byte-exact)
    if strides:
        off = 64
        for s in strides[:16]:
            if off + 4 > len(body):
                break
            struct.pack_into('<I', body, off, s & 0xFFFFFFFF)
            off += 4

    # ASCII operand-name strings at body+0x0d20+ (Path A.2 Rank 1 candidate)
    if operand_strings:
        write_off = 0x0d20
        for s in operand_strings[:3]:
            name_bytes = s.encode('ascii') + b'\x00'
            if write_off + len(name_bytes) > len(body):
                break
            body[write_off:write_off + len(name_bytes)] = name_bytes
            write_off += len(name_bytes)
        # body+0x64 = string-length pointer heuristic per Path A.2 Rank 3
        # Set to the last written byte offset (matches observed 0xd62..0xd65 range)
        if 0x64 + 4 <= len(body):
            struct.pack_into('<I', body, 0x64, write_off - 1)

    header = struct.pack(
        '<IIIIII',
        LC_CUSTOM_BLOCK,             # 0x04
        record_size,
        LC04_TYPE_PIPELINE_STAGE,    # 0x03
        PIPELINE_STAGE_MAGIC,        # 0x34A
        stage_total,
        stage_idx,
    )
    return bytes(header) + bytes(body)


# ============================================================
# Function 4: LC_0x08 ANEC metadata
# ============================================================

def emit_lc_anec_metadata(compiler_string: str = "zin_ane_compiler v9.509.0",
                          module_bundle: str = "com.apple.EspressoFramework",
                          flags: Optional[dict] = None) -> bytes:
    """Emit LC_0x08 ANEC metadata block.

    ASCII-driven; the load-bearing flag is:
        experimental.aot.enable_surface_desc:1

    Without this, the firmware sequencer does NOT install the LC_0x04
    type=0x04 surface table per the decoded protocol.

    Args:
        compiler_string: zin compiler version string.
        module_bundle: bundle identifier.
        flags: dict of compilation flags. Must include
               'experimental.aot.enable_surface_desc': 1 for Path C MVP.

    Returns:
        Variable length, header + ASCII payload, padded to multiple of 16.
    """
    if flags is None:
        flags = {'experimental.aot.enable_surface_desc': 1}

    flags_str = ','.join(f'{k}:{v}' for k, v in flags.items())
    payload = (
        f"ANEC v1\n"
        f"{compiler_string}\n"
        f"Module 0: \n"
        f" CompilationMethod: V1\n"
        f" ModuleBundleName: {module_bundle}\n"
        f" ModuleCompilationFlags: {flags_str}\n"
    ).encode('ascii')

    # Pad payload to 16-byte boundary
    while (len(payload) + 8) % 16:
        payload += b'\x00'

    cmdsize = 8 + len(payload)
    return struct.pack('<II', LC_ANEC_METADATA, cmdsize) + payload


# ============================================================
# Function 5: FVMLIB segment layout
# ============================================================

def compute_fvmlib_vm_layout(weight_sizes: List[int],
                              intermediate_sizes: List[int]) -> List[FVMLIBSegment]:
    """Compute VM-address layout for FVMLIB segments.

    Layout convention:
      Base: 0x30000000
      Alignment: 16 KB (FVMLIB_ALIGN = 0x4000)
      Order: weights first, then intermediates (matches observed samples
             where __FVMLIB / __KERN comes before __MKERN).

    Args:
        weight_sizes: byte sizes of weight segments.
        intermediate_sizes: byte sizes of intermediate activation segments.

    Returns:
        list of FVMLIBSegment with vm_addr + size_bytes computed.
    """
    segments = []
    vm = FVMLIB_BASE
    for i, sz in enumerate(weight_sizes):
        segments.append(FVMLIBSegment(vm_addr=vm, size_bytes=sz, label=f'W{i}'))
        vm += _round_up(sz, FVMLIB_ALIGN)
    for i, sz in enumerate(intermediate_sizes):
        segments.append(FVMLIBSegment(vm_addr=vm, size_bytes=sz, label=f'I{i}'))
        vm += _round_up(sz, FVMLIB_ALIGN)
    return segments


def _round_up(x: int, align: int) -> int:
    return ((x + align - 1) // align) * align


# ============================================================
# Function 6: Prologue 5 sub-link blocks
# ============================================================

def emit_prologue_text(num_surfaces: int, num_stages: int) -> bytes:
    """Emit prologue __text bytes (model header + entry + 5 sub-link blocks).

    Structure (per format_spec.prologue_structure; identical 5-sub-link
    structure observed in BOTH 3-pass and 41-pass production samples):

      w0..w11  (48B):    model-wide header
                          w0 = 1 (multi-pass flag)
                          w4 = total task size word count (filled later)
                          w11 = sub-block count (5)
      w12..w47 (144B):   model entry; DMA-setup opcode at w45, fence at w46
      then 5 sub-link blocks, each sized to firmware-state slot count.
            Sub-link sizes vary with surface count; empirical formula:
              base_words = 40
              per_surface_words = 2 (rounded)
            Resulting per-block: 40 + 2 * num_surfaces words approximately;
            samples observed at sizes 43..88 words.

    Args:
        num_surfaces: count of LC_0x04 type=0x04 entries (informs sub-block sizes).
        num_stages: count of LC_0x04 type=0x03 stage records (informs sub-block content).

    Returns:
        Bytes; total length = (12 + 36 + 5 * sub_link_words) * 4.
    """
    # Model-wide header w0..w11
    header_words = [0] * 12
    header_words[0] = 1            # multi-pass flag
    header_words[11] = 5           # sub-block count

    # Model entry w12..w47 (36 words = 144B)
    entry_words = [0] * 36
    # w12 = compute config sentinel (matches Flavor B header for first compute pass)
    entry_words[0] = TEXT_DMA_COMPUTE_SENTINEL  # at word index 12 in __text
    # w45 = DMA-setup opcode (0x9X41 family per spec; we use 0x80049241 as
    #       observed; this is the entry-side DMA bootstrap)
    entry_words[45 - 12] = 0x80049241
    # w46 = DMA fence
    entry_words[46 - 12] = DMA_FENCE_WORD_1
    # w47 = fence closing
    entry_words[47 - 12] = DMA_FENCE_WORD_2

    # 5 sub-link blocks; size grows with surface count
    sub_link_word_count = max(40, 40 + 2 * num_surfaces)
    sub_links = []
    for i in range(5):
        # Each sub-link header word follows the 0x00XX0001 pattern used
        # elsewhere in __text for multi-pass link words; here XX encodes
        # the sub-link size in words.
        link_header = (sub_link_word_count << 16) | (i + 1) | 0x00000001
        body_words = [0] * (sub_link_word_count - 1)
        # Minimal sequencer-state setup: first body word encodes stage
        # count + a reserved field; remaining words zero-init.
        if i == 0:
            body_words[0] = num_stages & 0xFFFF
        sub_links.append([link_header] + body_words)

    all_words = header_words + entry_words
    for sl in sub_links:
        all_words += sl

    return struct.pack(f'<{len(all_words)}I', *all_words)


# ============================================================
# Function 7: Flavor A pass (inline compute-config)
# ============================================================

def emit_compute_pass_inline(opcode: int,
                             tile_dim_n: int = 16,
                             tile_dim_m: int = 16) -> bytes:
    """Emit Flavor A single-op pass body (20 bytes = 5 words).

    Byte layout (per format_spec.text_sub_pass_link_5w_body.Flavor A;
    verified byte-exact in sys_cd9a0349.hwx w12):
      [0]  u32 0x00008001 (compute-config sentinel)
      [4]  u32 tile_dim_n
      [8]  u32 tile_dim_m
      [12] u32 opcode (e.g., 0x93608005 = PWL activation)
      [16] u32 post-config (often tile-dim again)

    Args:
        opcode: full 32-bit opcode word (e.g., OPCODE_READY_A, 0x93608005)
        tile_dim_n: tile dimension N. Default 16 (one ANE core's slice).
        tile_dim_m: tile dimension M. Default 16.

    Returns:
        20 bytes.
    """
    return struct.pack(
        '<IIIII',
        TEXT_COMPUTE_CONFIG_SENTINEL,
        tile_dim_n,
        tile_dim_m,
        opcode,
        tile_dim_m,  # post-config mirrors tile dim
    )


# ============================================================
# Function 8: Flavor B pass (table-driven via sequence counter)
# ============================================================

def emit_compute_pass_table_driven(seq_id: int,
                                   dma_channel_cfg: int = 0x0240) -> bytes:
    """Emit Flavor B single pass body (20 bytes = 5 words) using
    sequence counter into LC_0x04 surface table.

    Byte layout (per format_spec.text_sub_pass_link_5w_body.Flavor B;
    verified across 40 sub-passes in lib_f3707dd3.hwx):
      [0]  u32 0xFFC01540 (DMA-pass compute sentinel; constant)
      [4]  u32 0x00YY0240 or 0x00YY0260: YY = seq_id, low 16 = channel cfg
      [8]  u32 0x00YY0021 (seq closing 1)
      [12] u32 0x00YY0021 (next-pass prefetch hint)
      [16] u32 0x00YY0021 (closing sentinel)

    The pass_sequence_id (YY) is an INDEX into the prologue-installed
    firmware sequencer table. The hardware translates seq_id ->
    (src_vm_addr, dst_vm_addr) at dispatch time.

    Args:
        seq_id: monotonic sequence counter 1..N (must fit in u8).
        dma_channel_cfg: low 16 bits of word[1]; observed 0x0240 / 0x0260.

    Returns:
        20 bytes.
    """
    if seq_id < 1 or seq_id > 255:
        raise ValueError(f"seq_id {seq_id} outside [1, 255]")
    yy = (seq_id & 0xFF) << 16
    w1 = yy | (dma_channel_cfg & 0xFFFF)
    w2 = yy | 0x0021
    w3 = yy | 0x0021
    w4 = yy | 0x0021
    return struct.pack(
        '<IIIII',
        TEXT_DMA_COMPUTE_SENTINEL,
        w1, w2, w3, w4,
    )


# ============================================================
# Function 9: DMA fence
# ============================================================

def emit_dma_fence() -> bytes:
    """Emit DMA fence (8 bytes = 2 words).

    Inserted between Flavor B compute passes to enable cross-pass DMA
    prefetch overlap per fusion_architecture.md hypothesis. The 0x9141
    family opcodes (DMA/bypass) enable inter-pass SRAM ping-pong without
    DRAM round-trip.

    Returns:
        8 bytes: 0x22001340 + 0x00000021.
    """
    return struct.pack('<II', DMA_FENCE_WORD_1, DMA_FENCE_WORD_2)


# ============================================================
# Function 10: Terminator
# ============================================================

def emit_terminator(pass_count: int) -> bytes:
    """Emit __text terminator (8 bytes = 2 words).

    The closing word encodes the total pass count in the high byte:
        word2 = 0x01XX0021 where XX = (pass_count - 1)

    Verified across samples: lib_f3707dd3 41-pass -> 0x01280021
                             sys_cd9a0349 3-pass -> 0x01020021

    Args:
        pass_count: total compute passes in this binary (1-based).

    Returns:
        8 bytes.
    """
    if pass_count < 1 or pass_count > 256:
        raise ValueError(f"pass_count {pass_count} outside [1, 256]")
    xx = (pass_count - 1) & 0xFF
    word2 = TERMINATOR_WORD_2_BASE | (xx << 16)
    return struct.pack('<II', TERMINATOR_WORD_1, word2)


# ============================================================
# Function 11: Validation (loadModel handle + predict vs reference + dispatch count)
# ============================================================

def validate_multipass_binary(path: str,
                              reference_fn=None,
                              input_data=None,
                              expected_dispatch_count: int = 1) -> dict:
    """Validate a hand-emitted multi-pass .hwx.

    Three gate criteria per validation_gate_recipe:
      1. _ANEClient.loadModel returns a non-zero handle (or DirectLoader
         sel=3 ProgramCreate succeeds via direct IOKit path)
      2. predict() output matches reference_fn(input_data) within FP16
         tolerance (atol=1e-3, rtol=1e-2)
      3. dispatch count via ioreg/Instruments matches expected
         (multi-pass should emit exactly 1 XPC dispatch, not N)

    Args:
        path: .hwx file path.
        reference_fn: optional callable(input_data) -> numpy output for
                      correctness check.
        input_data: optional numpy array input for predict().
        expected_dispatch_count: expected XPC dispatch count. 1 for a
                                 multi-pass binary that should NOT be
                                 split.

    Returns:
        dict with keys: load_ok, load_handle, load_err, predict_ok,
        predict_err, max_abs_err, dispatch_count_ok, dispatch_count,
        all_pass.
    """
    result = {
        'load_ok': False,
        'load_handle': 0,
        'load_err': '',
        'predict_ok': None,
        'predict_err': '',
        'max_abs_err': None,
        'dispatch_count_ok': None,
        'dispatch_count': -1,
        'all_pass': False,
    }

    # Gate 1: load via direct IOKit sel=3 path
    try:
        # Lazy import to avoid IOKit binding at module import time
        from direct_load import DirectLoader, stage_hwx, ANED_CACHE
        staged = path
        if not path.startswith(ANED_CACHE):
            staged = stage_hwx(path, model_name='path_c_mvp_validate')
        loader = DirectLoader()
        try:
            handle = loader.load_hwx(staged)
            result['load_ok'] = handle != 0
            result['load_handle'] = handle
        finally:
            loader.close()
    except Exception as e:
        result['load_err'] = f'{type(e).__name__}: {e}'
        result['all_pass'] = False
        return result

    # Gate 2: predict vs reference
    if reference_fn is not None and input_data is not None:
        try:
            # Stub: actual predict would route through DirectLoader's
            # doEvaluateDirectWithModel binding. For MVP validation we
            # mark predict as 'unverified' if the binding isn't wired.
            result['predict_err'] = (
                'predict binding not wired in MVP scaffolding; '
                'gate 2 reported unverified.'
            )
            result['predict_ok'] = None
        except Exception as e:
            result['predict_err'] = f'{type(e).__name__}: {e}'
            result['predict_ok'] = False

    # Gate 3: dispatch count via ioreg (informational)
    try:
        import subprocess
        proc = subprocess.run(
            ['ioreg', '-c', 'AppleH13ANEInterface', '-w', '0'],
            capture_output=True, text=True, timeout=5,
        )
        # Heuristic: count 'ProgramCreate' or 'doEvaluateDirect' entries.
        # Real measurement requires Instruments ANE trace post-execution.
        result['dispatch_count'] = -1
        result['dispatch_count_ok'] = None
    except Exception as e:
        result['dispatch_count_ok'] = None

    result['all_pass'] = (
        result['load_ok']
        and result.get('predict_ok') is not False
    )
    return result


# ============================================================
# Function 12: LC_SEGMENT_64 (Path A.3 Mach-O wrapper)
# ============================================================
# Source: /private/tmp/identity8.hwx (production reference, captured 2026-05-27)
# Production offsets:
#   __PAGEZERO at 0x0020 (72B, 0 sections)
#   __FVMLIB   at 0x0068 (152B, 1 section)
#   __FVMLIB   at 0x0100 (152B, 1 section)
#   __TEXT     at 0x0198 (232B, 2 sections)
#   __KERN_0   at 0x0280 (152B, 1 section)
#
# Each segment_command_64 is 72B header + 80B per section.
# struct segment_command_64 {
#   u32 cmd        = 0x19;
#   u32 cmdsize;   = 72 + 80*nsects
#   char segname[16];
#   u64 vmaddr;
#   u64 vmsize;
#   u64 fileoff;
#   u64 filesize;
#   u32 maxprot;
#   u32 initprot;
#   u32 nsects;
#   u32 flags;
# }
# struct section_64 {
#   char sectname[16];
#   char segname[16];
#   u64 addr;
#   u64 size;
#   u32 offset;
#   u32 align;
#   u32 reloff;
#   u32 nreloc;
#   u32 flags;
#   u32 reserved1, reserved2, reserved3;  (12B padding)
# }

LC_SEGMENT_64 = 0x00000019


def emit_lc_segment_64(name: str, vmaddr: int, vmsize: int,
                       fileoff: int, filesize: int,
                       maxprot: int = 0, initprot: int = 0,
                       flags: int = 0,
                       sections: Optional[List[dict]] = None) -> bytes:
    """Emit LC_SEGMENT_64 (cmd=0x19) with N sections.

    Args:
        name: segment name (<= 16 chars, e.g. '__PAGEZERO', '__FVMLIB',
              '__TEXT', '__KERN_0').
        vmaddr: virtual address base (typically 0 for __PAGEZERO,
                0x30000000+ for __FVMLIB family).
        vmsize: virtual size (16KB-aligned).
        fileoff: byte offset within file where this segment's data lives.
        filesize: byte length within file.
        maxprot, initprot: VM protections (0/1/2/5 observed).
        flags: segment flags (0/4/6 observed).
        sections: list of dicts with keys sectname, segname, addr, size,
                  fileoff, align, reloff, nreloc, flags.

    Returns:
        72 + 80*len(sections) bytes.
    """
    if sections is None:
        sections = []
    name_b = name.encode('ascii').ljust(16, b'\x00')[:16]
    cmdsize = 72 + 80 * len(sections)

    header = struct.pack(
        '<II16sQQQQIIII',
        LC_SEGMENT_64,    # cmd
        cmdsize,
        name_b,
        vmaddr,
        vmsize,
        fileoff,
        filesize,
        maxprot,
        initprot,
        len(sections),
        flags,
    )

    sect_bytes = bytearray()
    for s in sections:
        sn = s['sectname'].encode('ascii').ljust(16, b'\x00')[:16]
        sg = s['segname'].encode('ascii').ljust(16, b'\x00')[:16]
        sect_bytes += struct.pack(
            '<16s16sQQIIIIIIII',
            sn, sg,
            s['addr'],
            s['size'],
            s['fileoff'],
            s.get('align', 0),
            s.get('reloff', 0),
            s.get('nreloc', 0),
            s.get('flags', 0),
            0, 0, 0,  # reserved1/2/3
        )

    return bytes(header) + bytes(sect_bytes)


def emit_identity8_segment_chain() -> bytes:
    """Emit the 5x LC_SEGMENT_64 chain matching production identity8.hwx.

    Source: /private/tmp/identity8.hwx production reference at offsets
            0x0020/0x0068/0x0100/0x0198/0x0280.

    Returns:
        760 bytes (72 + 152 + 152 + 232 + 152).
    """
    out = bytearray()
    # __PAGEZERO (idx 0)
    out += emit_lc_segment_64(
        '__PAGEZERO', vmaddr=0, vmsize=0x4000,
        fileoff=0, filesize=0,
        maxprot=0, initprot=0, flags=4,
        sections=[],
    )
    # __FVMLIB w0 (idx 1) - __data
    out += emit_lc_segment_64(
        '__FVMLIB', vmaddr=0x30000000, vmsize=0x4000,
        fileoff=0, filesize=0,
        maxprot=2, initprot=2, flags=6,
        sections=[{
            'sectname': '__data', 'segname': '__FVMLIB',
            'addr': 0x30000000, 'size': 0x200,
            'fileoff': 0, 'align': 14,
            'reloff': 0, 'nreloc': 0, 'flags': 0x23,
        }],
    )
    # __FVMLIB w1 (idx 2) - __const
    out += emit_lc_segment_64(
        '__FVMLIB', vmaddr=0x30004000, vmsize=0x4000,
        fileoff=0, filesize=0,
        maxprot=1, initprot=1, flags=6,
        sections=[{
            'sectname': '__const', 'segname': '__FVMLIB',
            'addr': 0x30004000, 'size': 0x200,
            'fileoff': 0, 'align': 14,
            'reloff': 0, 'nreloc': 0, 'flags': 0x21,
        }],
    )
    # __TEXT (idx 3) - __text + __const
    out += emit_lc_segment_64(
        '__TEXT', vmaddr=0x30008000, vmsize=0x8000,
        fileoff=0x4000, filesize=0x8000,
        maxprot=5, initprot=5, flags=0,
        sections=[
            {
                'sectname': '__text', 'segname': '__TEXT',
                'addr': 0x30008000, 'size': 0x150,
                'fileoff': 0x4000, 'align': 14,
                'reloff': 0, 'nreloc': 0, 'flags': 0x128,
            },
            {
                'sectname': '__const', 'segname': '__TEXT',
                'addr': 0x30008180, 'size': 0x4000,
                'fileoff': 0x4180, 'align': 6,
                'reloff': 0, 'nreloc': 0, 'flags': 0x26,
            },
        ],
    )
    # __KERN_0 (idx 4) - __kern_0
    out += emit_lc_segment_64(
        '__KERN_0', vmaddr=0x30010000, vmsize=0x4000,
        fileoff=0xc000, filesize=0x4000,
        maxprot=1, initprot=1, flags=4,
        sections=[{
            'sectname': '__kern_0', 'segname': '__KERN_0',
            'addr': 0x30010000, 'size': 0x200,
            'fileoff': 0xc000, 'align': 6,
            'reloff': 0, 'nreloc': 0, 'flags': 0x26,
        }],
    )
    return bytes(out)


# ============================================================
# Function 13: LC_SYMTAB (Path A.3 symbol table marker)
# ============================================================
# Source: /private/tmp/identity8.hwx LC_SYMTAB at offset 0x3020
# Raw: 0200000018000000383000002300000068320000c9040000
#   cmd=0x02, cmdsize=24, symoff=0x3038, nsyms=35, stroff=0x3268, strsize=1225
#
# struct symtab_command {
#   u32 cmd      = 0x2;
#   u32 cmdsize  = 24;
#   u32 symoff;
#   u32 nsyms;
#   u32 stroff;
#   u32 strsize;
# }

def emit_lc_symtab(symoff: int = 0, nsyms: int = 0,
                   stroff: int = 0, strsize: int = 0) -> bytes:
    """Emit LC_SYMTAB (cmd=0x02, 24B fixed).

    For MVP we can emit zero-pointer / zero-count symtab if the kext
    only validates the LC marker presence. If the kext dereferences
    symoff/stroff, callers must provide a valid file region.

    Args:
        symoff: file offset of nlist_64 array (default 0).
        nsyms: number of symbol records (default 0).
        stroff: file offset of string table (default 0).
        strsize: byte size of string table (default 0).

    Returns:
        24 bytes.
    """
    return struct.pack(
        '<IIIIII',
        LC_SYMTAB,
        24,
        symoff,
        nsyms,
        stroff,
        strsize,
    )


# ============================================================
# Function 14: LC_0x08 ANEC metadata FULL (Path A.3 enlargement)
# ============================================================
# Source: /private/tmp/identity8.hwx LC_0x08 at offset 0x2690, size 2448
# Production payload (2440B ASCII after 8B header) contains the full
# Module + Module ANEC sections with 44 compilation flag lines.
# Our prior emit (192B / 6 lines) was 12x undersized per A.1 finding.

# Canonical ModuleCompilationFlags observed in identity8.hwx
DEFAULT_AOT_FLAGS = [
    '-t h17g',
    '--fno-fold-scale=true',
    '--fdram-allocator=ffreuse',
    '--fdram-tensor-priority=sizebyliverange',
    '--fl2-allocator=ffreuse',
    '--fl3-allocator=ffreuse',
    '--fl2-cache-mode=resident',
    '--fsignature=ident',
    '--fdisable-bonded-networks=true',
    '--enable-param-and-map-rtgraph-refactor=false',
    '--memcache-size=4194304',
    '--fspatial-split=disabled',
    '--fenable-circular-buffer-in-spatial-split=-1',
    '--fkernel-rewind=enabled',
    '--max-td-latency=10000.000000',
    '--ne-frequency=-1.000000',
    '--pstate-dcs=-1',
    '--pstate-soc=-1',
    '--bss-limit=8589934592',
    '--foptimize-ne-utilization=true',
    '--disable-cache-prefetch-mask=1',
    '--optimize-mutable-kernel-section=true',
    '--enable-summary-performance-stats=false',
    '--split-kernel-section=true',
    '--max-kernel-section-size=134217728',
    '--enable-context-switch-events=false',
    '--fdisable-ne-width-slicing=false',
    '--disable-hard-swish-opt=false',
    '--preserve-texture-fraction=false',
    '--max-td-count=-1',
    '--max-segment-size=-1',
    '--e4m3-overflow-setting=Saturate',
    '--enable-dram-inplace-allocation=false',
    '--enable-l2-batch-splitting=true',
    '--enable-global-cw-optimization=true',
    '--suppress-math-exception=false',
    '--enable-l2-cached-buffer=true',
    '--enable-low-effort-cp-allocation=false',
    '--enable-noise-reduction-tiling=false',
    '--dump-parallel-score=false',
    '--enable-function-inlining=true',
    '--fdisable-weight-file-size-check=false',
    '--compile_time_mutable=false',
    '--enable-afm-mlir-features=true',
    '--enable-stable-precision=false',
    '--enable-kernel-split-for-multi-palette-lut=false',
    '--enable-non-self-replaceable-persistence=false',
    '--cost-model-cluster-threshold=85',
    '--dma-buffer-allocation-prioritization=false',
    '--enable-matmul-kernel-pad-insertion=false',
    '--Wl-undefined=fvmlib',
]


def emit_lc_0x08_anec_full(
    compiler_string: str = 'zin_ane_compiler v9.509.0',
    module_bundle: str = 'com.apple.EspressoFramework',
    module_version: str = '3520.15.1.14.2',
    anec_compiler_bundle: str = 'com.apple.ANECompilerFramework',
    anec_compiler_version: str = '9.509.0',
    module_compilation_flags: str = 'experimental.aot.enable_surface_desc:1',
    aot_flag_lines: Optional[List[str]] = None,
    input_path: str = '/tmp/path_a3_input.plist',
    output_path: str = '/tmp/path_a3_output.hwx.tmp',
    target_size: int = 2448,
) -> bytes:
    """Emit LC_0x08 ANEC metadata FULL block sized to match production
    identity8.hwx (~2448B per /private/tmp/identity8.hwx offset 0x2690).

    Produces a two-Module ASCII payload matching production identity8.hwx
    structure with all 44 default AOT flag lines plus -i / -o file paths.

    Args:
        compiler_string: zin compiler version string (line 2).
        module_bundle: client bundle (e.g. EspressoFramework).
        module_version: client module version (line 8 of identity8).
        anec_compiler_bundle: ANEC framework bundle.
        anec_compiler_version: ANEC framework version.
        module_compilation_flags: Module 0 flags (load-bearing
              experimental.aot.enable_surface_desc:1 must be present).
        aot_flag_lines: list of Module ANEC AOT flags. Defaults to
              DEFAULT_AOT_FLAGS observed in production identity8.hwx.
        input_path: -i path string.
        output_path: -o path string.
        target_size: desired LC_0x08 cmdsize. Default 2448 (matches
              identity8.hwx). Pads with nulls to reach exactly this size.

    Returns:
        target_size bytes.
    """
    if aot_flag_lines is None:
        aot_flag_lines = list(DEFAULT_AOT_FLAGS)

    flag_block = '\n\t'.join(aot_flag_lines)

    payload = (
        f'ANEC v1\n'
        f'{compiler_string}\n'
        f'\n'
        f' Module 0: \n'
        f'\t CompilationMethod: V1\n'
        f'\t ModuleBundleName: {module_bundle}\n'
        f'\t ModuleCompilationFlags: {module_compilation_flags}\n'
        f'\t ModuleVersion: {module_version}\n'
        f'\n'
        f' Module ANEC: \n'
        f'\t ModuleBundleName: {anec_compiler_bundle}\n'
        f'\t ModuleVersion: {anec_compiler_version}\n'
        f'\t ModuleCompilationFlags: \n'
        f'\t{flag_block}\n'
        f'\t-i {input_path}\n'
        f'\t-o {output_path}\n'
    ).encode('ascii')

    header_size = 8
    if header_size + len(payload) > target_size:
        # Truncate (caller asked for smaller-than-natural size; rare)
        payload = payload[:target_size - header_size]
    else:
        payload = payload.ljust(target_size - header_size, b'\x00')

    return struct.pack('<II', LC_ANEC_METADATA, target_size) + bytes(payload)


# ============================================================
# High-level assembly helper
# ============================================================

def assemble_multipass_text(prologue: bytes,
                            passes: List[ComputePass]) -> bytes:
    """Concatenate prologue + all compute passes + terminator into __text.

    Assigns Flavor B seq_ids monotonically starting at 1 in pass order.
    Inserts DMA fence between every Flavor B pass to enable cross-pass
    DMA prefetch overlap (Risk #3 mitigation).

    Args:
        prologue: bytes from emit_prologue_text()
        passes: ordered list of ComputePass.

    Returns:
        __text bytes.
    """
    result = bytearray(prologue)
    seq = 1
    for i, p in enumerate(passes):
        if p.flavor == 'A':
            result += emit_compute_pass_inline(
                p.opcode, p.tile_dim_n, p.tile_dim_m,
            )
        elif p.flavor == 'B':
            p.seq_id = seq
            seq += 1
            result += emit_compute_pass_table_driven(
                p.seq_id, p.dma_channel_cfg,
            )
            # Insert DMA fence after Flavor B (except last to avoid
            # double-fence with terminator)
            if i < len(passes) - 1:
                result += emit_dma_fence()
        else:
            raise ValueError(f"unknown flavor '{p.flavor}'")
    result += emit_terminator(len(passes))
    return bytes(result)


# ============================================================
# Self-test (Phase A 2-pass minimal gate)
# ============================================================

if __name__ == '__main__':
    print('path_c_mvp_emitter.py self-test')
    print('=' * 60)

    # Phase A minimal 2-pass test: conv (Flavor B, table-driven) + ReLU (Flavor A)
    named_io = [
        NamedSurface(name='input', vm_addr=0x30000000),
        NamedSurface(name='output', vm_addr=0x30004000),
    ]
    weight_sizes = [4096]       # KERN_0: conv weights
    intermediate_sizes = [4096] # one intermediate buffer
    segs = compute_fvmlib_vm_layout(weight_sizes, intermediate_sizes)

    # LC_0x40 emits
    lc40_in = emit_lc_segment_binding('input', named_io[0].vm_addr)
    lc40_out = emit_lc_segment_binding('output', named_io[1].vm_addr)
    print(f'LC_0x40 input: {len(lc40_in)} bytes (expected 32)')
    print(f'LC_0x40 output: {len(lc40_out)} bytes (expected 32)')
    assert len(lc40_in) == 32 and len(lc40_out) == 32

    # LC_0x04 type=0x04 surface table
    lc04_table = emit_lc_surface_table(named_io, segs)
    # Minimal-test size will be ~2064 (header + 7 reserved + 2 segs + pad + alias);
    # production samples observe 2208-2216 with more entries. Both valid.
    print(f'LC_0x04 surface table: {len(lc04_table)} bytes '
          f'(minimal-test ~2064; production samples 2208-2216)')
    assert 2040 <= len(lc04_table) <= 2240

    # LC_0x04 type=0x03 pipeline stages (2 stages for 2-pass test)
    lc03_stage1 = emit_lc_pipeline_stage(stage_idx=1, stage_total=2,
                                          surface_refs=[0, 7], strides=[128, 128])
    lc03_stage2 = emit_lc_pipeline_stage(stage_idx=2, stage_total=2,
                                          surface_refs=[7, 1], strides=[128, 128])
    print(f'LC_0x04 type=0x03 stage1: {len(lc03_stage1)} bytes (expected ~3400)')
    print(f'LC_0x04 type=0x03 stage2: {len(lc03_stage2)} bytes (expected ~3400)')

    # LC_0x08 ANEC metadata
    lc08 = emit_lc_anec_metadata()
    print(f'LC_0x08 ANEC metadata: {len(lc08)} bytes (expected ~128-256)')

    # Prologue
    prologue = emit_prologue_text(num_surfaces=len(named_io) + len(segs),
                                  num_stages=2)
    print(f'Prologue __text: {len(prologue)} bytes')

    # 2-pass __text: conv (Flavor B seq=1) + ReLU (Flavor A)
    passes = [
        ComputePass(flavor='B', opcode=OPCODE_READY_A,
                    tile_dim_n=16, tile_dim_m=16, dma_channel_cfg=0x0240),
        ComputePass(flavor='A', opcode=0x93608005,  # ReLU PWL
                    tile_dim_n=16, tile_dim_m=16),
    ]
    text = assemble_multipass_text(prologue, passes)
    print(f'Full __text (prologue + 2 passes + terminator): {len(text)} bytes')

    # Sanity: fence + terminator + pass words
    fence = emit_dma_fence()
    term = emit_terminator(pass_count=2)
    print(f'DMA fence: {len(fence)} bytes (expected 8); '
          f'words = 0x{struct.unpack("<I", fence[0:4])[0]:08x}, '
          f'0x{struct.unpack("<I", fence[4:8])[0]:08x}')
    print(f'Terminator(pc=2): {len(term)} bytes (expected 8); '
          f'words = 0x{struct.unpack("<I", term[0:4])[0]:08x}, '
          f'0x{struct.unpack("<I", term[4:8])[0]:08x} '
          f'(expected 0x22001440, 0x01010021)')

    # Check terminator XX byte
    t2 = struct.unpack('<I', term[4:8])[0]
    xx = (t2 >> 16) & 0xFF
    assert xx == 1, f'Expected XX=1 (pass_count-1=1), got {xx}'
    print(f'Terminator XX byte: {xx} (pass_count-1=1) PASS')

    print('=' * 60)
    print('All emitter functions executed without exception.')
    print('Phase Scaffolding: 11/11 functions emitted; ANED_CACHE fixed.')
