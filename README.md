# ane-compiler

Compile machine learning models for the Apple Neural Engine, driving the ANE through custom MIL IR rather than a hand-written CoreML model. Two modes:

- **SIP ON**: fused-subgraph execution via `_ANEInMemoryModel` + MIL IR. 37 fused subgraphs from a 73-op GPT-2. Guaranteed ANE execution, validated by `doEvaluateDirectWithModel`. (This path still uses Apple's `aned` compiler under the hood: custom MIL is submitted and `aned` emits the multi-pass binary. The contribution is the MIL the compiler generates and the per-layer dispatch control around it, not an aned bypass.)
- **SIP OFF**: direct `.hwx` manipulation. The emitted `.hwx` is byte-identical to aned output. An LLDB in-flight swap intercepts `ProgramCreate` and overwrites the mmap'd `.hwx` before the kext reads it, giving opcode/dequant/DMA-schedule control over what executes. Note: a fully user-side IOKit dispatch that bypasses `aned` is not available on macOS 26 (the kext path is gated); this mode operates on the aned-mediated artifact, not against it.

---

## Production results

| Model | Dispatches | Throughput | Hardware | Notes |
|---|---|---|---|---|
| **GPT-2 117M** | 25 (fused from 73 ops) | **229 tok/s** | ANE via `_ANEInMemoryModel` | Custom MIL activations (Mish, GELU-tanh, squared ReLU) |
| **Llama 3.2-1B** | 25 (25d+C combined stack) | **50.2 tok/s** | ANE | Cross-layer fusion: post_attn + pre_attn = 40 → 25 dispatches |
| **Llama 3.1-8B Q8** (`bench_combined_stack.py`) | 72 | **9.66 tok/s** | ANE | Single-stream at Q8, the deployment precision. Canonical, placement-verified (IOReport, n_a_reads=0, reproduced 9.53/9.66) on the 72-dispatch residual-capture *instrument* build, measured against a ~19.4 tok/s Q8 bandwidth floor (the gap is the dispatch tax of the instrumented topology, not a model limit). FP32 residual accumulation (FP16 fails past 16 layers at dim 4096), Llama 3 RoPE scaling. The earlier 7.9 figure is retired (silent-offload-confounded). A Q4 fused-multipass build reaches a higher ~13.4 tok/s at the lower precision (candidate; next row). |
| **Llama 3.1-8B fused attention** (`bench_cross_layer_fusion.py`) | 40 (MIL IR) | ~13.4 tok/s (Q4, candidate) | ANE (CPU_AND_NE) | Full attention incl. activation×activation matmul + softmax, SIP ON. The canonical claims here are correctness: fused-block dispatch count 40 and 5/5 prompt top-1 match vs PyTorch reference. The ~13.4 tok/s throughput is a **Q4 fused-multipass build** (registry `llama_3_1_8b_ane.fused_multipass_tok_s`, candidate, VERIFIED_ENERGY + ANEMLL, with real attention): a higher rate at the lower Q4 precision, explicitly NOT the Q8 deployment number (9.66, prior row). |
| **Neuron 80M** | 5 | **1,064 tok/s** | ANE SRAM | FFN-only domain classifier, 905 µs/dispatch, 98.7% accuracy |

**Cross-accelerator contention is model-dependent, not zero** (see Hardware Characterization below). The honest split: ANE-side cost is small and model-invariant (+1.4% Llama 70B, +0.38% Gemma 4 31B), while GPU-side cost is verifier-scale-dependent (−4.7% Llama 70B, −20.1% Gemma 4 31B). Separately, a *concurrent ANE monitor* costs ~2.2% GPU throughput per GB of weight-streaming footprint it places (small monitor ~2–4%, a full-8B placed monitor ~16%; from a footprint sweep in the research program, not independently published). This is not "zero contention."

---

## CPU acceleration kernels (`llama_cpu_ops.c`)

`libllama_cpu_ops.dylib` ships fused C/Accelerate kernels for the parts of an LLM forward pass that don't run on the ANE:

- `llama_gqa_attention`: fused QK^T → softmax → V via vDSP/BLAS, replacing a Python NumPy loop.
- `llama_rope`: plain RoPE (no scaling) via vDSP. Use a wrapper that supplies precomputed cos/sin tables for Llama-3-style scaling.
- `llama_rms_norm`: fused RMSNorm via `vDSP_meanvv` + `vvrsqrtf`.

These kernels move the non-ANE parts of the forward pass off the Python path. *(The per-kernel speedup multiple and the end-to-end prompt-encode tok/s delta from an earlier build are not yet registered in the canonical measurement registry and so are not quoted here as verified numbers.)*

---

## What it does (architecture)

The compiler walks a fused-graph IR and emits one of two outputs:

1. `.mlmodelc` packages with custom MIL ops, loadable by `_ANEInMemoryModel.compileWithQoS:` and dispatchable via `loadWithQoS:` + `requestWithInputs:`.
2. Raw `.hwx` Mach-O kernel images, byte-identical to what aned produces, ready for direct kext load via the IOKit `H11ANEIn` user client (`sel=3 ProgramCreate`).

The MIL IR path is the practical one: it works under SIP ON, doesn't need kext loads, and handles 14 elementwise op primitives plus all the standard transformer ops (linear, layer_norm, gelu, softmax, matmul, gather). The `.hwx` direct path exists as proof that the compiler matches Apple's binary format.

**Scope note (what is and isn't unique here).** Multi-pass `__text`/`__KERN_0` emission for a complex MIL program is *not* unique to this project: Apple's own `aned`, driven through the public `coremltools` path, emits the same multi-pass binary. The throughput numbers above are reproductions of what the production CoreML/aned stack already achieves on this hardware, presented as a characterization baseline, not as a faster-than-Apple compiler. The two capabilities this repo adds on top of that stack are (1) **per-layer residual capture and inject** at arbitrary layer boundaries (the instrument that underpins the cross-instance coupling research below), and (2) **opcode-level dispatch plus dequant/DMA-scheduling control** via the SIP-off path. Those are the load-bearing contributions; the tok/s figures are context, not the claim.

The `bench_combined_stack.py` measurement (42.2 → 50.2 tok/s on Llama-1B) and the `bench_cross_layer_fusion.py` measurement (40 → 25 dispatches via post-attn + pre-attn fusion) are the experimental evidence that fusion *depth*, not channel-count tuning, is the optimization lever for small models on this hardware.

The compiler emits per-layer dispatch artifacts that support residual capture and inject at arbitrary layer boundaries. This primitive underpins separate research on cross-instance LLM coupling.

---

## Hardware constraints (measured)

- **93 µs dispatch floor** on M5 Pro (XPC overhead). Below dim 1024, all latency is dispatch-bound, not compute-bound.
- **dim≈2048 compute crossover**: above this, compute time equals dispatch time.
- **DMA stride regime change at ic=768**: discrete binary threshold in the compiled `__text` section. Not yet measured for latency impact.
- **128-program slot exclave wall**: the kext refuses to allocate more than 128 program objects per ANE client. Hardware-enforced.
- **16-tile fixed channel partition**: work is sliced into 16 equal `(ic*oc*2)/16`-byte tile slabs at compile time. Hardware-validated; tile descriptors are cryptographically checked. Not user-tunable.

---

## Hardware Characterization

Measurements from the ANE research program. Each headline number is reproducible with the
in-repo bench script named beside it; the private research registry these were logged in is not
published, so nothing below leans on it:

- **Q8 = ANE deployment precision.** Q8 finishes a layer in ~97.4% of FP16 wall time at 50.4% memory cost. This is *same wall-time, not free dequant*: Q8 moves half the bytes but feeds them through the on-ANE int→fp16 dequant pipeline at roughly half the effective bandwidth (~80 GB/s vs ~155 GB/s FP16), so the two land at parity. Q4 pays a 31% latency penalty.
- **FP32 internal accumulation.** ANE reduction network accumulates in FP32 with full mantissa (bit-exact on overflow probe). The FP32 between-dispatch requirement (measured on this stack: the inter-dispatch residual stream needs FP32 to avoid accumulation drift) is specific to that stream, not ANE hardware.
- **Cross-accelerator contention.** Not zero. ANE-side cost is small and model-invariant: +0.38% (Gemma 4 31B), +1.4% (Llama 70B). GPU-side cost is verifier-scale-dependent: −4.7% (Llama 70B), −20.1% (Gemma 4 31B) Separately, a concurrent ANE *monitor* costs ~2.2% GPU throughput per GB of weight-streaming footprint it places: a small placed monitor ~2–4%, a full-8B placed monitor ~16% (footprint sweep; research-program measurement, not independently published). The DMA path isolation bounds the *idle* case; an actively-streaming concurrent ANE workload does take measurable GPU throughput.
- **Bidirectional SharedEvents.** Both GPU→ANE and ANE→GPU hardware event signaling confirmed working. See `ane-dispatch/examples/gpu_ane_sync.m`.
- **GQA tile bottleneck.** 72% of on-ANE predicted cost is GQA head-repeat data materialization. Skip-tile fix (Q-group matmul) eliminates it with bitwise-identical output. −6% per-block ANE latency.
- **53 ISA opcodes catalogued.** 8 emitted, 45 additional mapped with decoded control words.

---

## Living Model

`living_model_*.py` tracks whether a model's weights can be edited in place between dispatches. Current state (two distinct results, kept separate):

- **FP16 weight patch-execute: reproduced twice, independently (once by a second, clean session).** A reversible edit to the weights inside the live multi-pass FP16 block (the 838 MB / 7-pass compiled blob) reaches the executing substrate: it changes the ANE block output and then reverts byte-identical, confirmed by an independent reproduction in a clean session. The patched program reloads via the CoreML cache read (~6 ms re-mmap, not a full recompile). An earlier "small single-pass only" bound on patch-execute was refuted by this result for FP16 multipass.
- **Per-channel gain-write: a working primitive, but a MAGNITUDE knob, not a behavioral STEERING knob.** A per-channel gain delta applies exactly on real FP16 Llama-3.1-8B L31 weights (places + executes + reversibly moves the argmax token; cos ≈ 0.9999997 on the gain subset). But the gain-induced basin-shift does **not** exceed a norm-matched uniform null (null ≥ gain at every differing level, in both runs). So this is a controllable magnitude lever on the live weights, **not** a means of steering the model toward a chosen behavior. Do not read it as behavioral control.
- **Arbitrary low-rank weight delta (continuous-LoRA-style B@A): REFUTED.** A general rank-2 update does not reproduce on the substrate (repro cos ≈ 0.118); the int8 tile-interleave permutation needed for an arbitrary delta is not decoded. Only the gain-shaped (magnitude) subset is reachable.
- **int4 / Q4 same-topology: not on the ANE.** The quantized path is CPU-routed on macOS 26.3, so patch-control is moot there.

The mid-dispatch DRAM-overwrite framing from earlier runs was a methodology artifact (it patched an FP16-stage decoy cache entry that int4 execution never reads); the corrected result above supersedes it. Net: the weight-write reaches the live FP16 substrate (a verified instrument primitive); it is a magnitude substrate, not a steering one.

---

## Project Chimera

`project_chimera_*.py` measured ANE↔GPU handoff cost at or below the noise floor (~0.1 ms/dispatch via zero-copy IOSurface-backed activation transfer; not literally zero, but negligible against per-layer compute). Conclusion: handoff is not the bottleneck; single-accelerator compute dominates handoff overhead at every interesting model scale. Parked, with the measurement preserved as the reason future split-compute proposals start out skeptical.

---

## Related work

This compiler emits ANE bytecode for macOS-side execution on H17 (M5 Pro). The Asahi Linux community has built complementary infrastructure for ANE execution on Linux without Apple toolchain dependencies:

- [allbilly/ane](https://github.com/allbilly/ane): H13 ANE ops via pure Python register programming on Asahi Linux
- [eiln/ane](https://github.com/eiln/ane): Linux kernel driver for the ANE
- [tinygrad's accel/ane](https://github.com/tinygrad/tinygrad/tree/v0.10.3/extra/accel/ane/): original H13 reverse engineering
- [Maynard Handley's vol7 ANE](https://github.com/name99-org/AArch64-Explore/blob/main/vol7%20ANE.nb.pdf): architectural analysis

---

## Related repos

- [orion-ane](https://github.com/MidasMulli/cognitive-stack-ane): Midas cognitive agent + Subconscious memory system that uses these models
- [subconscious](https://github.com/MidasMulli/subconscious): the cognitive memory loops as a separate package
- [ane-dispatch](https://github.com/MidasMulli/ane-dispatch): direct ANE dispatch + SharedEvents (a ~37% per-dispatch latency reduction was observed in informal testing; indicative, not benchmarked)
- [ane-toolkit](https://github.com/MidasMulli/ane-toolkit): ANE binary-format (H17) reverse engineering + PWL activation deployment
- [ane-perf](https://github.com/MidasMulli/ane-perf): ANE hardware performance characterization via IOReport histograms

## License

MIT.
