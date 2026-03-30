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
├── __KERN_0.__kern_0: weights (FP16) + PWL tables, 16-tile replicated
└── __LINKEDIT: symbol table
```

## Architecture

- **17-stage fixed-function pipeline**: Operations = stage enable/disable combinations, not opcodes
- **Multi-pass programs**: Complex ops (softmax, layernorm) decompose into sequential pipeline passes
- **16-tile replication**: Weights replicated across 16 ANE cores
- **PWL activation tables**: 84-byte piecewise-linear lookup (33 breakpoints)

## Requirements

- Python 3.9+
- numpy
- macOS 15+ (for ANE execution via ane-dispatch)
- Apple Silicon M1-M5

## Related

- [ane-dispatch](https://github.com/MidasMulli/ane-dispatch) — Direct ANE dispatch without CoreML
- [ane-toolkit](https://github.com/MidasMulli/ane-toolkit) — H17 binary format research + PWL deployment

## License

MIT
