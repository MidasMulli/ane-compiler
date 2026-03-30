# ane-compiler

Compile transformer layers for Apple Neural Engine. No CoreML. No coremltools. SIP-on compatible. M1-M5.

## What this does

Generates `.mlmodelc` bundles (espresso format) from Python weight matrices, which Apple's ANE daemon compiles to hardware binaries. Also provides parameterized microcode generation for conv, softmax, and layernorm — byte-identical to Apple's compiler output at arbitrary dimensions.

**Custom activation support**: inject ANY piecewise-linear activation (33 breakpoints) into the ANE pipeline. MISH, x²/16, or your own function — things CoreML/coremltools cannot produce.

## Architecture

```
Python weights + config
        ↓
    compiler.py → generates per-op .mlmodelc (espresso format)
        ↓
    _ANEClient.compileModel → ANE daemon produces .hwx
        ↓
    Multi-dispatch chain via ane-dispatch (IOSurface routing)
        ↓
    ANE hardware execution
```

Full transformer layer = 13 dispatches (11 ANE + 2 CPU residual add):
```
x → LN1 → Q_proj → K_proj → V_proj → QK_matmul → softmax
  → SV_matmul → O_proj → add(x, attn) → LN2 → FFN_gate(act)
  → FFN_down → add(r1, ffn) → output
```

## Quick start

```python
from compiler import compile_layer, TransformerLayerConfig
import numpy as np

config = TransformerLayerConfig(
    hidden_dim=64, n_heads=1, head_dim=64, ffn_dim=128,
    activation="relu",
    weights={
        "W_q": np.random.randn(64, 64).astype(np.float32) * 0.1,
        "W_k": np.random.randn(64, 64).astype(np.float32) * 0.1,
        "W_v": np.random.randn(64, 64).astype(np.float32) * 0.1,
        "W_qk": np.random.randn(64, 64).astype(np.float32) * 0.1,
        "W_sv": np.random.randn(64, 64).astype(np.float32) * 0.1,
        "W_o": np.random.randn(64, 64).astype(np.float32) * 0.1,
        "W_gate": np.random.randn(128, 64).astype(np.float32) * 0.1,
        "W_down": np.random.randn(64, 128).astype(np.float32) * 0.1,
    }
)

plan = compile_layer(config, output_dir="/tmp/my_layer")
print(plan.summary())
# Execute via: ane-dispatch multi-model chain (see tests/transformer_layer.m)
```

## Custom activation (the differentiator)

CoreML supports ~26 fixed activation modes. ane-compiler lets you run ANY activation via custom PWL tables:

```python
from emitter import PWLTable
import numpy as np

# Define MISH: f(x) = x * tanh(softplus(x))
x = np.linspace(-10, 10, 33)
y = (x * np.tanh(np.log1p(np.exp(x)))).astype(np.float16)

# Build 84-byte PWL table
header = np.array([-10.0, np.inf, 0.0, np.inf], dtype=np.float16)
footer = np.array([0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float16)
mish_pwl = PWLTable(header=header, breakpoints=y, footer=footer)

# Inject into .hwx at offset 0xC000 (replaces default activation)
# ANE executes MISH — something CoreML cannot produce
```

Verified on hardware: MISH activation on ANE, all 64 channels differ from SiLU baseline.

## Parameterized microcode generation

No template .hwx needed for standard dimensions:

```python
from emitter import generate_conv_text, generate_softmax_text, generate_layernorm_text

# Generate __text microcode from dimensions alone
conv_text = generate_conv_text(in_ch=384, out_ch=768)     # 468 bytes
sm_text = generate_softmax_text(dim=192, reference_hwx_path="softmax_ref.hwx")
ln_text = generate_layernorm_text(dim=192, reference_hwx_path="layernorm_ref.hwx")
```

All byte-identical to Apple's ANE compiler output at novel (never-captured) dimensions.

## Verification status

| Check | Status | Method |
|-------|--------|--------|
| Conv __text parameterization | **PASS** | Byte-identical at 8 dims + kill test 384→768 |
| Softmax __text parameterization | **PASS** | Byte-identical at 4 dims + kill test dim=192 |
| LayerNorm __text parameterization | **PASS** | Byte-identical at 4 dims + kill test dim=192 |
| Weight packing (16-core layout) | **PASS** | Byte-identical at 6 dims incl. non-power-of-2 |
| Per-op hardware execution | **PASS** | max diff < 1e-3 (conv, softmax, layernorm) |
| 7-op attention chain | **PASS** | max diff 1.22e-04, 0/64 mismatches > 1e-3 |
| Full transformer layer (13 dispatches) | **PASS** | Architecturally correct, all ops execute on ANE |
| Custom MISH activation | **PASS** | 64/64 channels differ from SiLU, PWL injection works |
| .mlmodelc generation (no coremltools) | **PASS** | Conv/softmax/LN compile on ANE from generated bundles |

## Limitations

- **Single-op compilation**: ANE daemon compiles one espresso layer per .mlmodelc. Multi-op models fail via `_ANEClient`. Attention is 7 separate dispatches chained via IOSurface.
- **Per-dispatch overhead**: ~93µs per ANE dispatch. Full transformer layer ≈ 13 × 93µs ≈ 1.2ms. Apple's fused 48-pass .hwx avoids this but requires internal compilation path.
- **Residual add on CPU**: Two-input elementwise add compiles on ANE but IOSurface reuse across models needs further work. CPU fallback is reliable.
- **Channel dims must be multiples of 16** for production weight packing.
- **Softmax/LayerNorm dim=256**: uses different __text template (excluded from parameterization).
- **No bias support**: conv layers are bias=False only.
- **FP16 precision**: accumulated error across 13 dispatches can reach ~5e-2 on pathological inputs (near-uniform → layernorm amplifies). Per-op accuracy is < 1e-3.
- **H17G only** (M1-M5 ANE generation).

## Comparison

| Feature | ane-compiler | Orion (maderix) | Apple CoreML |
|---------|-------------|----------------|--------------|
| Binary-level control | ✓ (__text + __KERN_0) | ✗ | ✗ |
| Custom activations | ✓ (33-pt PWL) | ✗ | ✗ (26 fixed modes) |
| No CoreML dependency | ✓ | ✗ | N/A |
| Multi-op fusion | ✗ (multi-dispatch) | ✗ | ✓ (48+ passes) |
| SharedEvents | ✓ (via ane-dispatch) | ✗ (listed unexplored) | ✗ |
| Weight layout decoded | ✓ (16-core, 32-ch sub-blocks) | ✗ | Internal |
| Transformer layer | ✓ (13-dispatch chain) | ✗ | ✓ (fused) |

## Related

- [ane-dispatch](https://github.com/MidasMulli/ane-dispatch) — Direct ANE dispatch without CoreML
- [ane-toolkit](https://github.com/MidasMulli/ane-toolkit) — H17 binary format research + PWL deployment

## License

MIT
