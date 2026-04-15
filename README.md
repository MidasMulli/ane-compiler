# ane-compiler

Compile machine learning models for the Apple Neural Engine without going through Apple's CoreML / aned compiler service. Two modes:

- **SIP ON** — fused-subgraph execution via `_ANEInMemoryModel` + MIL IR. 37 fused subgraphs from a 73-op GPT-2. Guaranteed ANE execution, validated by `doEvaluateDirectWithModel`.
- **SIP OFF** — direct `.hwx` emission. Byte-identical to aned output. LLDB in-flight swap intercepts `sel=3 ProgramCreate` and overwrites the mmap'd `.hwx` before the kext reads it. Demo runs as a single command.

---

## Production results

| Model | Dispatches | Throughput | Hardware | Notes |
|---|---|---|---|---|
| **GPT-2 117M** | 25 (fused from 73 ops) | **229 tok/s** | ANE via `_ANEInMemoryModel` | Custom MIL activations (Mish, GELU-tanh, squared ReLU) |
| **Llama 3.2-1B** | 25 (25d+C combined stack) | **50.2 tok/s** | ANE | Cross-layer fusion: post_attn + pre_attn = 40 → 25 dispatches |
| **Llama 3.1-8B Q8** | 72 | **7.9 tok/s** | ANE | FP32 residual accumulation (FP16 fails past 16 layers at dim 4096), Llama 3 RoPE scaling |
| **Llama 3.1-8B fused attention** | 32 (MIL IR) | **3.56 ms/block** | ANE (CPU_AND_NE) | Full attention incl. activation×activation matmul + softmax. SIP ON. 5/5 top-1 match vs PyTorch reference. Mechanism demo — does not beat production 72d throughput. |
| **Neuron 80M** | 5 | **1,064 tok/s** | ANE SRAM | FFN-only domain classifier, 905 µs/dispatch, 98.7% accuracy |

**Cross-accelerator contention is model-dependent** (see Hardware Characterization below). On GPT-2, 143 tok/s saturated vs 145 tok/s idle = −1.2% (noise floor). ANE-side contention stays inside the noise floor across verifier swaps; GPU-side contention scales with verifier decode cadence.

---

## CPU acceleration kernels (`llama_cpu_ops.c`)

`libllama_cpu_ops.dylib` ships fused C/Accelerate kernels for the parts of an LLM forward pass that don't run on the ANE:

- `llama_gqa_attention` — fused QK^T → softmax → V via vDSP/BLAS. **78× faster** than the equivalent Python NumPy loop on a 64×64 fp16 conv.
- `llama_rope` — plain RoPE (no scaling) via vDSP. Use a wrapper that supplies precomputed cos/sin tables for Llama-3-style scaling.
- `llama_rms_norm` — fused RMSNorm via `vDSP_meanvv` + `vvrsqrtf`.

End-to-end measurement on the production 8B prompt-encode path: **6.23 → 9.9 tok/s (+59%)** after wiring `llama_gqa_attention` into `ane_extractor_8b.py:_gqa_attention`.

---

## What it does (architecture)

The compiler walks a fused-graph IR and emits one of two outputs:

1. `.mlmodelc` packages with custom MIL ops, loadable by `_ANEInMemoryModel.compileWithQoS:` and dispatchable via `loadWithQoS:` + `requestWithInputs:`.
2. Raw `.hwx` Mach-O kernel images, byte-identical to what aned produces, ready for direct kext load via the IOKit `H11ANEIn` user client (`sel=3 ProgramCreate`).

The MIL IR path is the practical one — it works under SIP ON, doesn't need kext loads, and handles 14 elementwise op primitives plus all the standard transformer ops (linear, layer_norm, gelu, softmax, matmul, gather). The `.hwx` direct path exists as proof that the compiler matches Apple's binary format.

The `bench_combined_stack.py` measurement (42.2 → 50.2 tok/s on Llama-1B) and the `bench_cross_layer_fusion.py` measurement (40 → 25 dispatches via post-attn + pre-attn fusion) are the experimental evidence that fusion *depth* — not channel-count tuning — is the optimization lever for small models on this hardware.

---

## Hardware constraints (measured)

- **93 µs dispatch floor** on M5 Pro (XPC overhead). Below dim 1024, all latency is dispatch-bound, not compute-bound.
- **dim≈2048 compute crossover** — above this, compute time equals dispatch time.
- **DMA stride regime change at ic=768** — discrete binary threshold in the compiled `__text` section. Documented in vault notes; not yet measured for latency impact.
- **128-program slot exclave wall** — the kext refuses to allocate more than 128 program objects per ANE client. Hardware-enforced.
- **16-tile fixed channel partition** — work is sliced into 16 equal `(ic*oc*2)/16`-byte tile slabs at compile time. Hardware-validated; tile descriptors are cryptographically checked. Not user-tunable.

---

## Hardware Characterization

Measurements from the ANE research program, registered in `data/measurement_registry.json`:

- **Q8 = ANE deployment precision.** 97.4% of FP16 per-layer throughput at 50.4% memory cost. Q4 pays 31% latency penalty.
- **FP32 internal accumulation.** ANE reduction network accumulates in FP32 with full mantissa (bit-exact on overflow probe). The FP32 between-dispatch requirement (§3.2 of Paper 1) is specific to the inter-dispatch residual stream, not ANE hardware.
- **Cross-accelerator contention.** ANE DMA path is physically isolated from GPU. ANE-side contention: +0.38% (Gemma 4), +1.4% (Llama 70B). GPU-side: model-dependent (−4.7% Llama 70B, −20.1% Gemma 4 31B).
- **Bidirectional SharedEvents.** Both GPU→ANE and ANE→GPU hardware event signaling confirmed working. See `ane-dispatch/examples/gpu_ane_sync.m`.
- **GQA tile bottleneck.** 72% of on-ANE predicted cost is GQA head-repeat data materialization. Skip-tile fix (Q-group matmul) eliminates it with bitwise-identical output. −6% per-block ANE latency.
- **53 ISA opcodes catalogued.** 8 emitted, 45 additional mapped with decoded control words. Full catalog in `vault/ane-reverse/`.

---

## Living Model

`living_model_*.py` is the parked LoRA-during-inference experiment. Three runs measured no overall adaptation headroom (+0%) but the 76% prediction window at tokens 150–175 (vs 55% frozen baseline) showed real signal in early noise — flagged but parked. The Main 26 weight intervention revival (see commit history) opens a different question worth re-examining: now that mid-dispatch DRAM weight modification works through the fresh-`compileWithQoS:` path at ~10 ms/probe, layer-wise ablation studies become tractable.

---

## Project Chimera

`project_chimera_*.py` measured ANE↔GPU handoff cost as **0 µs** (zero, not "small"). Conclusion: handoff is not the bottleneck; GPU Q4 dominates compute at the relevant model sizes. The lesson is that single-accelerator throughput dominates handoff overhead at every interesting model scale. Parked, with the measurement preserved as the reason future split-compute proposals start out skeptical.

---

## Related repos

- [orion-ane](https://github.com/MidasMulli/orion-ane) — Midas cognitive agent + Subconscious memory system that uses these models
- [subconscious](https://github.com/MidasMulli/subconscious) — the cognitive memory loops as a separate package
- [ane-dispatch](https://github.com/MidasMulli/ane-dispatch) — direct ANE dispatch + SharedEvents (37% faster than CoreML)
- [ane-toolkit](https://github.com/MidasMulli/ane-toolkit) — IOKit protocol decoder + Mach-O `.hwx` tooling
- [ane-perf](https://github.com/MidasMulli/ane-perf) — ANE hardware performance characterization via IOReport histograms

## License

MIT.
