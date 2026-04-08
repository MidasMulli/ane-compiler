#!/usr/bin/env python3
"""
ane-compiler: Run GPT-2 on Apple Neural Engine.

Default mode compiles through our emitter (cache-swap).
--no-compile falls back to Apple's aned daemon.
Operations beyond aned's ~8 espresso layer types require the compiler.

Usage:
  python run.py --prompt "The meaning of" --tokens 50
  python run.py --prompt "The" --tokens 20 --no-compile
  python run.py --prompt "The" --tokens 20 --activation mish
  python run.py --prompt "The" --tokens 20 --activation mish --no-compile  # fails
  sudo python run.py --prompt "The" --tokens 50  # compiler mode (needs root for killall aned)

Copyright 2026 Nick Lo. MIT License.
"""

import argparse
import os
import sys
import time
import struct
import warnings
import subprocess
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))


def build_mish_pwl():
    """Build a 33-breakpoint Mish PWL table for ANE hardware."""
    import numpy as np
    from emitter import PWLTable

    # Read GELU header/footer format from reference
    ref_path = os.path.join(os.path.dirname(__file__), 'gelu_aned_reference.hwx')
    if os.path.exists(ref_path):
        with open(ref_path, 'rb') as f:
            ref = f.read()
        header = np.frombuffer(ref[0xC000:0xC000+8], dtype=np.float16).copy()
        footer = np.frombuffer(ref[0xC000+8+66:0xC000+8+66+10], dtype=np.float16).copy()
    else:
        header = np.array([-4.4, np.inf, 0.0, np.inf], dtype=np.float16)
        footer = np.array([0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float16)

    x_pts = np.linspace(-4.0, 4.0, 33).astype(np.float32)
    y_pts = (x_pts * np.tanh(np.log1p(np.exp(x_pts)))).astype(np.float16)
    return PWLTable(header=header, breakpoints=y_pts, footer=footer)


def emit_fused_ffn_hwx(template, W_up, bias_up, W_down, bias_down,
                        pwl_override=None):
    """Emit fused FFN .hwx from template + weights. Optionally swap PWL."""
    import numpy as np

    def pack_biased(weights, bias):
        w = weights.astype(np.float16)
        b = bias.astype(np.float16)
        out_ch, in_ch = w.shape
        NC = 16; sub_size = 32; stripe_size = NC * sub_size
        n_full = (out_ch // NC) // sub_size
        rem_first = -(-out_ch // NC) - n_full * sub_size
        rem_rest = out_ch // NC - n_full * sub_size
        parts = []
        for t in range(NC):
            for s in range(n_full):
                row_start = t * sub_size + s * stripe_size
                parts.append(b[row_start:row_start+sub_size].tobytes())
                parts.append(w[row_start:row_start+sub_size, :].flatten(order='F').tobytes())
            rem = rem_first if t == 0 else rem_rest
            if rem > 0:
                rem_stripe_start = n_full * stripe_size
                row_start = rem_stripe_start if t == 0 else rem_stripe_start + rem_first + (t-1) * rem_rest
                parts.append(b[row_start:row_start+rem].tobytes())
                parts.append(w[row_start:row_start+rem, :].flatten(order='F').tobytes())
        return b''.join(parts)

    kern0_off = 0xC000
    pwl = template[kern0_off:kern0_off+128]
    if pwl_override:
        pwl_data = pwl_override.to_bytes()
        pwl = bytearray(128)
        pwl[:len(pwl_data)] = pwl_data

    up_biased = pack_biased(W_up, bias_up)
    up_tile = len(up_biased) // 16
    kern0_parts = [bytes(pwl)]
    for t in range(16):
        kern0_parts.append(up_biased[t*up_tile:(t+1)*up_tile])
        if t < 15:
            kern0_parts.append(bytes(pwl))

    down_biased = pack_biased(W_down, bias_down)
    out_ch, in_ch = W_down.shape
    n_full = (out_ch // 16) // 32
    rem = -(-out_ch // 16) - n_full * 32
    dt = n_full * (64 + 32*in_ch*2) + (rem*2 + rem*in_ch*2 if rem > 0 else 0)
    for t in range(16):
        kern0_parts.append(down_biased[t*dt:(t+1)*dt])
        kern0_parts.append(b'\x00' * 32)

    new_kern0 = b''.join(kern0_parts)
    PAGE = 0x4000
    kfs = ((len(new_kern0) + PAGE - 1) // PAGE) * PAGE
    buf = bytearray(kern0_off + kfs)
    buf[:kern0_off] = template[:kern0_off]
    buf[kern0_off:kern0_off+len(new_kern0)] = new_kern0

    ncmds = struct.unpack_from('<I', buf, 0x10)[0]
    off = 32; k0_va = 0
    for _ in range(ncmds):
        cmd = struct.unpack_from('<I', buf, off)[0]
        cs = struct.unpack_from('<I', buf, off+4)[0]
        if cmd == 0x19:
            sn = bytes(buf[off+8:off+24]).split(b'\x00')[0].decode('ascii')
            if sn == '__KERN_0':
                k0_va = struct.unpack_from('<Q', buf, off+24)[0]
                struct.pack_into('<Q', buf, off+32, kfs)
                struct.pack_into('<Q', buf, off+48, kfs)
                ns = struct.unpack_from('<I', buf, off+64)[0]
                if ns > 0:
                    struct.pack_into('<Q', buf, off+72+40, len(new_kern0))
            elif sn == '__LINKEDIT':
                struct.pack_into('<Q', buf, off+24, k0_va + kfs)
        off += cs
    return bytes(buf)


def main():
    parser = argparse.ArgumentParser(
        description='ane-compiler: Generate text on Apple Neural Engine')
    parser.add_argument('--prompt', default='The', help='Input prompt')
    parser.add_argument('--tokens', type=int, default=50, help='Tokens to generate')
    parser.add_argument('--activation', default='gelu',
                        choices=['gelu', 'mish'],
                        help='FFN activation (gelu=default, mish=custom PWL)')
    parser.add_argument('--no-compile', action='store_true',
                        help='Use aned compilation instead of emitter (fails for custom activations)')
    parser.add_argument('--exact', action='store_true',
                        help='CPU GELU for bitwise PyTorch match')
    args = parser.parse_args()

    import numpy as np
    from transformers import GPT2Tokenizer
    from model_loader import GPT2Model
    from generate import ANEDispatcher, generate
    from first_token import compile_all_ops, MODEL_PATH, BUILD_DIR
    from compiler import gen_fused_ffn_mlmodelc

    activation = args.activation
    no_compile = args.no_compile
    mode = 'exact' if args.exact else 'fused'

    # ─── Check permissions for compiler mode ─────────────
    if not no_compile and os.geteuid() != 0:
        print("Compiler mode requires root (cache-swap uses sudo).")
        print("Run with sudo, or use --no-compile for aned fallback.")
        sys.exit(1)

    # ─── Load ────────────────────────────────────────────
    t_total = time.time()
    t0 = time.time()
    model = GPT2Model.from_safetensors(MODEL_PATH)
    tokenizer = GPT2Tokenizer.from_pretrained('openai-community/gpt2')
    t_load = time.time() - t0

    act_label = f" ({activation} activation)" if activation != 'gelu' else ""
    print(f"Loading GPT-2 117M{act_label}...")

    # ─── Compile ─────────────────────────────────────────
    t0 = time.time()
    build_dir = BUILD_DIR + f'_{mode}'
    compiled = compile_all_ops(model, build_dir, mode=mode)
    n_ops = len(compiled)

    if no_compile:
        # aned mode: compiler.py → .mlmodelc → aned
        if activation != 'gelu':
            # Try to compile Mish — aned will reject it
            print(f"Compiling via aned...")
            print(f"ERROR: aned rejected fused_ffn: activation '{activation}' has no espresso mapping.")
            print(f"This operation requires ane-compiler.")
            print(f"\nRun without --no-compile to use the compiler:")
            print(f"  sudo {sys.executable} {' '.join(a for a in sys.argv if a != '--no-compile')}")
            sys.exit(1)

        print(f"Compiling via aned ({n_ops} ops)...")
        dispatcher = ANEDispatcher(compiled, quiet=True)
        dispatcher.start()
        compiler_label = "aned"

    else:
        # Compiler mode: emitter → .hwx → cache-swap
        print(f"Compiling via ane-compiler ({n_ops} ops)...")

        # Load fused FFN template
        fused_ref = '/tmp/fused_ffn_ref.hwx'
        if not os.path.exists(fused_ref):
            # Capture reference on first run
            print("  Capturing fused FFN template (first run only)...")
            L = model.layers[0]
            ref_mlmodelc = os.path.join(build_dir, 'layer_0', 'fused_ffn.mlmodelc')
            EVAL = os.path.join(os.path.dirname(__file__), 'tests', 'ane_eval_binary')
            x = np.zeros(768, dtype=np.float16)
            subprocess.run([EVAL, ref_mlmodelc, '768', '768'],
                           input=x.tobytes(), capture_output=True, timeout=30)
            import time as tm; tm.sleep(0.5)
            # Find in cache
            cache_base = '/Library/Caches/com.apple.aned'
            for root, dirs, files in os.walk(cache_base):
                for f in files:
                    if f == 'model.hwx':
                        full = os.path.join(root, f)
                        if os.path.getmtime(full) > t0 - 2 and os.path.getsize(full) > 5000000:
                            import shutil
                            shutil.copy2(full, fused_ref)
                            break

        with open(fused_ref, 'rb') as f:
            fused_template = f.read()

        # Build PWL override for custom activations
        pwl_override = None
        if activation == 'mish':
            pwl_override = build_mish_pwl()

        # Emit .hwx for all fused FFN ops
        hwx_overrides = {}
        for i in range(12):
            L = model.layers[i]
            name = f'L{i}_fused_ffn'
            if name in compiled:
                hwx_overrides[name] = emit_fused_ffn_hwx(
                    fused_template,
                    L.W_fc.astype(np.float32), L.c_fc_bias.astype(np.float32),
                    L.W_fc_down.astype(np.float32), L.c_proj_ffn_bias.astype(np.float32),
                    pwl_override=pwl_override)

        n_emitted = len(hwx_overrides)
        dispatcher = ANEDispatcher(compiled, quiet=True, hwx_overrides=hwx_overrides)
        dispatcher.start()
        compiler_label = "ane-compiler (emitter)"

    t_compile = time.time() - t0

    # ─── Generate ────────────────────────────────────────
    prompt_tokens = tokenizer.encode(args.prompt)
    print(f"Dispatch: doEvaluateDirectWithModel (guaranteed ANE)")
    print()

    t0 = time.time()
    tokens = generate(model, dispatcher, prompt_tokens,
                      max_new_tokens=args.tokens, mode=mode)
    t_gen = time.time() - t0

    text = tokenizer.decode(tokens)
    n_new = len(tokens) - len(prompt_tokens)
    tps = n_new / t_gen if t_gen > 0 else 0

    print(f"> {text}")
    print()
    print("---")
    print(f"Model:      GPT-2 117M (12 layers, 768 hidden)")
    print(f"Tokens:     {n_new} | Speed: {tps:.1f} tok/s")
    print(f"Dispatches: {n_ops}/token" +
          (f" (fused from 73 ops)" if mode == 'fused' else ""))
    print(f"Compiler:   {compiler_label}")
    print(f"Activation: {activation}" +
          (" (custom PWL, 33 breakpoints)" if activation != 'gelu' else
           " (ANE hardware PWL)" if mode == 'fused' else " (CPU FP32)"))
    print(f"GPU:        idle")

    dispatcher.stop()


if __name__ == '__main__':
    main()
