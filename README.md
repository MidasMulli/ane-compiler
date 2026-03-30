# ane-compiler

Emit Apple Neural Engine `.hwx` binaries from model definitions. No CoreML. No SIP-off. Works on M1-M5.

## What this does

The ANE executes BEEFFACE Zin binaries (`.hwx` files). Apple's compiler generates these from CoreML models. This project generates them directly from weight matrices and operation definitions — bypassing CoreML entirely.

Template-based emission: reference `.hwx` files (from Apple's compiler) provide the structural skeleton. The emitter replaces weights, activation tables, and configurable parameters while preserving the ANE-specific tile metadata, pipeline configuration, and I/O descriptors.

## Supported operations

| Operation | Binary class | Passes | Template needed |
|-----------|-------------|--------|-----------------|
| ReLU, abs, tanh | 48K activation | 1 | `relu.hwx` etc. |
| SiLU, GELU, sigmoid | 64K PWL activation | 1 | `silu_native_65536.hwx` etc. |
| Conv1x1 (linear projection) | 64K conv | 1 | `conv_relu_mil_65536.hwx` |
| Conv1x1 + SiLU/GELU | 64K conv+PWL | 1 | `conv_silu_mil_65536.hwx` |
| Softmax | 64K multi-pass | 5 | `softmax_only.hwx` |
| LayerNorm | 48K multi-pass | 4-5 | `layernorm.hwx` |
| FFN (linear→SiLU→linear) | Two 64K .hwx | 1+1 | conv_silu + conv_relu |

## Usage

```python
from emitter import ANECompiler, ActivationType, LayerNormParams
import numpy as np

# Point at directory of template .hwx files
compiler = ANECompiler('/path/to/templates/')

# Emit a standalone activation
compiler.emit_activation(ActivationType.RELU, output_path='relu.hwx')

# Emit a linear projection with SiLU activation
weights = np.random.randn(64, 64).astype(np.float16)
compiler.emit_conv_activation(weights, ActivationType.SILU, output_path='gate.hwx')

# Emit a full FFN: linear → SiLU → linear
gate_w = np.random.randn(64, 64).astype(np.float16)
down_w = np.random.randn(64, 64).astype(np.float16)
gate_hwx, down_hwx = compiler.emit_ffn(gate_w, down_w, output_path='ffn')
# Produces ffn.gate.hwx and ffn.down.hwx

# Emit softmax (5-pass ANE pipeline)
compiler.emit_softmax(output_path='softmax.hwx')

# Emit layernorm with custom epsilon
compiler.emit_layernorm(
    LayerNormParams(epsilon=1e-6, dim=128),
    output_path='layernorm.hwx'
)
```

## Loading emitted .hwx

Use [ane-dispatch](https://github.com/MidasMulli/ane-dispatch) to load and execute emitted `.hwx` files on ANE hardware:

```objc
ANEModel *model = [ANEModel modelWithCompiledURL:url error:&err];
[model prepareWithError:&err];

ANEBuffer *input = [ANEBuffer bufferWithShape:@[@1, @64, @1, @1] dtype:ANEDtypeFloat16];
ANEBuffer *output = [ANEBuffer bufferWithShape:@[@1, @64, @1, @1] dtype:ANEDtypeFloat16];

ANERequest *req = [ANERequest requestWithInputs:@[input] outputs:@[output]];
[[ANEDispatch shared] evaluate:model request:req error:&err];
```

## Template preparation

Templates are compiler-generated `.hwx` files that provide the structural skeleton. Capture them by compiling simple CoreML models and extracting from the ANE cache:

```
/Library/Caches/com.apple.aned/{build}/ModelAssetsCache/{process}/{hash}/model.hwx
```

Or use the `_ANEInMemoryModel` MIL compilation path (see `tests/capture_ffn_template.py`).

Required templates for full functionality:
- Activation atlas: `relu.hwx`, `silu_native_65536.hwx`, etc.
- Conv atlas: `conv_relu_mil_65536.hwx`, `conv_silu_mil_65536.hwx`, etc.
- Multi-pass: `softmax_only.hwx`, `layernorm.hwx`

## .hwx binary format

```
BEEFFACE Zin binary (Mach-O variant)
├── Header (32 bytes): magic, cpu_type=128, subtype=9 (H17G)
├── Load commands: LC_SEGMENT_64, LC_THREAD, LC_SYMTAB, LC_CMD_0x40
├── __PAGEZERO: guard page
├── __FVMLIB: I/O buffer descriptors (IOSurface refs)
├── __TEXT.__text: ANE pipeline microcode (1-5 passes)
├── __TEXT.__const: pipeline configuration (16K)
├── __KERN_0.__kern_0: weights (FP16), 16-core partitioned layout
└── __LINKEDIT: symbol table
```

## Weight layout

For `out_ch >= 16`: weights are partitioned across 16 ANE cores (not replicated).

```python
# ANE layout (verified byte-identical to Apple's compiler at 64-512 dims):
hw = weights.reshape(16, out_ch // 16, in_ch).transpose(0, 2, 1).flatten()

# Inverse:
weights = hw.reshape(16, in_ch, out_ch // 16).transpose(0, 2, 1).reshape(out_ch, in_ch)
```

`__KERN_0` size = `out_ch * in_ch * 2` bytes. File size scales accordingly (page-aligned).

For `out_ch < 16` (small atlas models): tile-replicated with padding.

## Architecture

- **17-stage fixed-function pipeline**: Operations = stage enable/disable combinations, not opcodes
- **Multi-pass programs**: Complex ops (softmax, layernorm) decompose into sequential pipeline passes
- **16-core weight partitioning**: Output channels split across 16 ANE cores
- **PWL activation tables**: 84-byte piecewise-linear lookup (33 breakpoints)

## Limitations

- **Template-based only**: You need Apple's compiler to generate template `.hwx` files first. This tool fills in weights and parameters — it doesn't generate the pipeline microcode from scratch.
- **No direct .hwx loading**: ANE always recompiles from `.mlmodelc` — the disk cache is write-only. Hardware verification done by patching `.espresso.weights` in the `.mlmodelc`, which makes the compiler produce a `.hwx` with our weight layout. Output matches Python reference within FP16 precision.
- **Channel dimensions must be multiples of 16** for production weight packing. Smaller dimensions use the tile-replicated atlas template format.
- **Single conv per .hwx**: FFN is emitted as two separate `.hwx` files (gate+activation, down projection), chained via ane-dispatch. Fused multi-conv `.hwx` emission is not yet supported.
- **Softmax/LayerNorm dimensions fixed to template**: The multi-pass programs carry dimension-specific microcode. Emitting softmax/layernorm for dimensions different from the template requires a new template capture.
- **H17G only (M1-M5)**: The binary format is specific to the H17G/H17S ANE generation. Earlier generations may differ.
- **No bias support yet**: Conv emission is bias=False only.

## Verification status

| Check | Status | Method |
|-------|--------|--------|
| Weight layout (16-core partition) | **PASS** | Byte-identical to compiler at 64→64, 64→128, 128→256, 256→512 |
| Weight round-trip (pack→unpack) | **PASS** | 5/5 dimensions up to 1024→512 |
| Softmax 5-pass parse+reassemble | **PASS** | Byte-identical round-trip |
| LayerNorm 5-pass parse+reassemble | **PASS** | Byte-identical round-trip |
| Emitted .hwx structure | **PASS** | BEEFFACE magic, page alignment, ncmds, all 53 structural tests |
| Emitted .hwx = compiler .hwx (same weights) | **PASS** | 0 byte diffs at 64x64 |
| Hardware execution (custom weights on ANE) | **PASS** | Patched .mlmodelc weights → compiler → ANE → output matches Python (max diff 2.19e-04, 0/64 > 1e-3). __KERN_0 byte-identical to compiler. |

## Requirements

- Python 3.9+
- numpy
- macOS 15+ (for ANE execution via ane-dispatch)
- Apple Silicon M1-M5
- Apple's ANE compiler output (template .hwx files)

## Related

- [ane-dispatch](https://github.com/MidasMulli/ane-dispatch) — Direct ANE dispatch without CoreML
- [ane-toolkit](https://github.com/MidasMulli/ane-toolkit) — H17 binary format research + PWL deployment

## License

MIT
