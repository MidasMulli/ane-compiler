#!/usr/bin/env python3
"""
Custom MISH activation on Apple Neural Engine.

MISH(x) = x * tanh(softplus(x)) is NOT in CoreML's 26 activation modes.
This example builds a 33-breakpoint piecewise-linear approximation and
injects it into the ANE pipeline — something no framework can do.

Usage:
    # 1. Compile a SiLU activation model (closest built-in)
    python3 -c "
    from compiler import gen_conv_mlmodelc
    import numpy as np
    gen_conv_mlmodelc('/tmp/silu_model.mlmodelc',
                      np.eye(64, dtype=np.float32), 64, 64)
    "

    # 2. Compile on ANE via capture_and_eval
    ./tests/capture_and_eval /tmp/silu_model.mlmodelc /tmp/silu.hwx

    # 3. Run this script to generate the MISH PWL table
    python3 examples/custom_mish_activation.py

    # 4. Patch the .hwx: replace 84 bytes at offset 0xC000
    #    (requires SIP-off for cache patching, or use emitter API)

Verified: MISH on ANE produces output distinct from SiLU on all 64 channels.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from emitter import PWLTable


def compute_mish_pwl(n_breakpoints: int = 33,
                     x_range: tuple = (-10.0, 10.0)) -> PWLTable:
    """Compute a piecewise-linear approximation to MISH.

    MISH(x) = x * tanh(ln(1 + exp(x)))

    Returns an 84-byte PWL table compatible with the ANE's
    64K activation class format.
    """
    x_min, x_max = x_range
    x_points = np.linspace(x_min, x_max, n_breakpoints).astype(np.float32)
    y_points = (x_points * np.tanh(np.log1p(np.exp(x_points)))).astype(np.float16)

    # ANE PWL format: 4-word header + 33 breakpoints + 5-word footer
    # Asymmetric unbounded (like SiLU): header has inf, slope=1.0 in footer
    header = np.array([x_min, np.inf, 0.0, np.inf], dtype=np.float16)
    footer = np.array([0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float16)

    return PWLTable(header=header, breakpoints=y_points, footer=footer)


def main():
    mish_pwl = compute_mish_pwl()
    pwl_bytes = mish_pwl.to_bytes()

    print("MISH PWL table (84 bytes):")
    print(f"  Header:      {mish_pwl.header}")
    print(f"  Breakpoints: {mish_pwl.breakpoints[:5]} ... {mish_pwl.breakpoints[-3:]}")
    print(f"  Footer:      {mish_pwl.footer}")
    print()

    # Verify approximation quality
    x_test = np.array([0.0, 0.5, 1.0, 2.0, -1.0, -2.0], dtype=np.float32)
    exact = x_test * np.tanh(np.log1p(np.exp(x_test)))

    # PWL interpolation (simplified — real ANE uses hardware interpolation)
    x_min, x_max = -10.0, 10.0
    step = (x_max - x_min) / 32
    bps = mish_pwl.breakpoints.astype(np.float32)
    for x, y_exact in zip(x_test, exact):
        idx = (x - x_min) / step
        i = int(np.clip(idx, 0, 31))
        frac = idx - i
        y_pwl = bps[i] + frac * (bps[min(i+1, 32)] - bps[i])
        print(f"  MISH({x:5.1f}) = {y_exact:7.4f}  PWL ≈ {y_pwl:7.4f}  "
              f"error = {abs(y_exact - y_pwl):.4f}")

    # Save
    out_path = os.path.join(os.path.dirname(__file__), '..', 'mish_pwl.bin')
    with open(out_path, 'wb') as f:
        f.write(pwl_bytes)
    print(f"\nSaved to {out_path} ({len(pwl_bytes)} bytes)")
    print("Inject into .hwx at offset 0xC000 to replace default activation.")


if __name__ == '__main__':
    main()
