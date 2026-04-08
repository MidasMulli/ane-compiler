#!/usr/bin/env python3
"""
SIP-OFF Demo: Custom Mish activation on Apple Neural Engine via LLDB .hwx swap.

Single-command automated demo. Run as: sudo python demo_sip_off.py

This script demonstrates the full ane-compiler pipeline for injecting a custom
activation function (Mish) that Apple's ANE compiler does not support natively.

Approach: LLDB in-flight .hwx swap (fully automated)
    1. Emit a GELU activation .hwx via aned (mode 19, dim=64)
    2. Build a Mish PWL table (33 breakpoints, [-4,4] range)
    3. Clone the GELU .hwx and swap the 84-byte PWL at __KERN_0 offset with Mish
    4. Write LLDB Python module to /tmp/, launch LLDB attached to aned
    5. Wait for LLDB ready, trigger dispatch from main process
    6. Read output, compare against CPU Mish reference, detach LLDB

Requirements:
    - SIP OFF (csrutil disable from Recovery)
    - Must run as root: sudo python demo_sip_off.py
    - Python: ~/.mlx-env/bin/python3
    - ane_eval_binary compiled at tests/ane_eval_binary

Copyright 2026 Nick Lo. MIT License.
"""

import os
import sys
import struct
import subprocess
import shutil
import time
import signal
import threading
import numpy as np
from pathlib import Path

sys.path.insert(0, 'src')
from compiler import gen_conv_mlmodelc, generate_mlmodelc
from emitter import PWLTable

# ═══════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EVAL_BINARY = os.path.join(SCRIPT_DIR, 'tests', 'ane_eval_binary')
WORK_DIR = '/tmp/mish_sip_off_demo'
GELU_MLMODELC = os.path.join(WORK_DIR, 'gelu_64.mlmodelc')
CUSTOM_HWX_PATH = os.path.join(WORK_DIR, 'mish_custom.hwx')
LLDB_MODULE_PATH = os.path.join(WORK_DIR, 'mish_swap.py')
TRIGGER_SCRIPT_PATH = os.path.join(WORK_DIR, 'trigger_dispatch.py')
REFERENCE_PATH = os.path.join(WORK_DIR, 'mish_reference.npy')
DIM = 64
PWL_OFFSET = 0xC000  # __KERN_0 offset in .hwx where 84-byte PWL table lives
PWL_SIZE = 84

ANE_CACHE = '/Library/Caches/com.apple.aned'
BUILD_VERSION = '25E246'


# ═══════════════════════════════════════════════════════════════
# Step 1: Generate GELU reference .hwx
# ═══════════════════════════════════════════════════════════════

def gen_gelu_mlmodelc(output_dir: str, dim: int):
    """Generate .mlmodelc for GELU activation (mode 19 = erf-GELU PWL)."""
    layer = {
        "type": "activation",
        "name": "gelu",
        "bottom": "input",
        "top": "output",
        "mode": 19,
        "weights": {},
        "attributes": {"is_output": 1},
    }
    shapes = {
        "input": (1, 1, dim, dim),
        "output": (1, 1, dim, dim),
    }
    generate_mlmodelc(
        output_dir, [layer], shapes, [],
        inputs=[("input", [dim, 1, 1])],
        outputs=[("output", [dim, 1, 1])],
    )


def step1_generate_gelu():
    """Generate GELU .mlmodelc and compile it on ANE to get a reference .hwx."""
    print("=" * 70)
    print("STEP 1: Generate GELU activation model (mode 19, dim=64)")
    print("=" * 70)

    os.makedirs(WORK_DIR, exist_ok=True)

    # Generate .mlmodelc for GELU activation
    if os.path.exists(GELU_MLMODELC):
        shutil.rmtree(GELU_MLMODELC)
    gen_gelu_mlmodelc(GELU_MLMODELC, DIM)
    print(f"  Generated: {GELU_MLMODELC}")

    # Kill aned to force fresh compilation
    subprocess.run(['killall', 'aned'], capture_output=True)
    time.sleep(1)

    # Dispatch on ANE to trigger aned compilation (creates .hwx in cache)
    test_input = np.random.randn(DIM).astype(np.float16)
    result = subprocess.run(
        [EVAL_BINARY, GELU_MLMODELC, str(DIM), str(DIM)],
        input=test_input.tobytes(), capture_output=True, timeout=30,
    )
    if result.returncode != 0:
        print(f"  WARNING: ane_eval_binary returned {result.returncode}")
        print(f"  stderr: {result.stderr.decode()[:200]}")
    else:
        print(f"  GELU compiled and dispatched on ANE")

    # Find the compiled .hwx in aned cache
    hwx_path = find_cached_hwx(GELU_MLMODELC)
    if hwx_path:
        print(f"  Cached .hwx: {hwx_path}")
        hwx_size = os.path.getsize(hwx_path)
        print(f"  .hwx size: {hwx_size} bytes")

        # Read and verify PWL at offset 0xC000
        with open(hwx_path, 'rb') as f:
            f.seek(PWL_OFFSET)
            pwl_data = f.read(PWL_SIZE)
        gelu_pwl = PWLTable.from_bytes(pwl_data)
        print(f"  GELU PWL header: {gelu_pwl.header}")
        print(f"  GELU PWL breakpoints[0:5]: {gelu_pwl.breakpoints[:5]}")
        return hwx_path
    else:
        print("  WARNING: Could not find cached .hwx")
        print("  The LLDB swap approach does not need a cached file --")
        print("  it intercepts the .hwx in aned's memory during dispatch.")
        return None


def find_cached_hwx(mlmodelc_path: str) -> str:
    """Find the most recently modified .hwx in aned cache."""
    cache_dir = os.path.join(ANE_CACHE, BUILD_VERSION, 'ModelAssetsCache')
    if not os.path.exists(cache_dir):
        return None

    # Walk cache looking for model.hwx files, return newest
    newest = None
    newest_mtime = 0
    for root, dirs, files in os.walk(cache_dir):
        for f in files:
            if f == 'model.hwx':
                path = os.path.join(root, f)
                mtime = os.path.getmtime(path)
                if mtime > newest_mtime:
                    newest_mtime = mtime
                    newest = path
    return newest


# ═══════════════════════════════════════════════════════════════
# Step 2: Build Mish PWL table
# ═══════════════════════════════════════════════════════════════

def step2_build_mish_pwl(gelu_hwx_path: str) -> PWLTable:
    """Build Mish PWL using GELU .hwx header/footer (keeps HW range consistent)."""
    print()
    print("=" * 70)
    print("STEP 2: Build Mish PWL table (reusing GELU header for HW range)")
    print("=" * 70)

    # Read the original GELU PWL to get the header/footer the hardware expects
    if gelu_hwx_path and os.path.exists(gelu_hwx_path):
        with open(gelu_hwx_path, 'rb') as f:
            f.seek(PWL_OFFSET)
            gelu_pwl_data = f.read(PWL_SIZE)
        gelu_pwl = PWLTable.from_bytes(gelu_pwl_data)
        # Use GELU's header range for breakpoint generation
        x_min = float(gelu_pwl.header[0])  # -4.4 for erf-GELU
        header = gelu_pwl.header.copy()
        footer = gelu_pwl.footer.copy()
        print(f"  Using GELU header: {header} (x_min={x_min})")
    else:
        x_min = -4.0
        header = np.array([x_min, np.inf, 0.0, np.inf], dtype=np.float16)
        footer = np.array([0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float16)
        print(f"  No GELU .hwx, using default header (x_min={x_min})")

    # The PWL is asymmetric unbounded: x_max is implicit from header[1]=inf
    # For the breakpoints, we need 33 points from x_min to the effective max
    # GELU uses x_min=-4.4, and with 33 points the step determines x_max
    # The hardware's effective x_max = x_min + 32*step
    # From GELU: step = 8.8/32 = 0.275, x_max = -4.4 + 32*0.275 = 4.4
    x_max = -x_min  # Symmetric: if x_min=-4.4, x_max=4.4
    x_points = np.linspace(x_min, x_max, 33).astype(np.float32)

    # Mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
    y_points = (x_points * np.tanh(np.log1p(np.exp(x_points)))).astype(np.float16)

    mish_pwl = PWLTable(header=header, breakpoints=y_points, footer=footer)
    pwl_bytes = mish_pwl.to_bytes()

    print(f"  Range: [{x_min}, {x_max}]")
    print(f"  Breakpoints: {len(y_points)}")
    print(f"  PWL size: {len(pwl_bytes)} bytes")
    print(f"  Header: {mish_pwl.header}")
    print(f"  First 5 breakpoints: {mish_pwl.breakpoints[:5]}")
    print(f"  Last 3 breakpoints:  {mish_pwl.breakpoints[-3:]}")
    print(f"  Footer: {mish_pwl.footer}")

    # Verify approximation quality
    print()
    print("  PWL approximation quality:")
    x_test = np.array([0.0, 0.5, 1.0, 2.0, -1.0, -2.0, x_min, x_max], dtype=np.float32)
    exact = x_test * np.tanh(np.log1p(np.exp(x_test)))
    bps = mish_pwl.breakpoints.astype(np.float32)
    step = (x_max - x_min) / 32.0
    for x, y_exact in zip(x_test, exact):
        idx = (x - x_min) / step
        i = int(np.clip(idx, 0, 31))
        frac = idx - i
        y_pwl = bps[i] + frac * (bps[min(i + 1, 32)] - bps[i])
        print(f"    Mish({x:5.1f}) = {y_exact:7.4f}  PWL = {y_pwl:7.4f}  "
              f"err = {abs(y_exact - y_pwl):.4f}")

    return mish_pwl


# ═══════════════════════════════════════════════════════════════
# Step 3: Create custom .hwx with Mish PWL
# ═══════════════════════════════════════════════════════════════

def step3_create_custom_hwx(gelu_hwx_path: str, mish_pwl: PWLTable) -> str:
    """Clone GELU .hwx and swap PWL table at 0xC000 with Mish breakpoints."""
    print()
    print("=" * 70)
    print("STEP 3: Create custom .hwx (GELU template + Mish PWL)")
    print("=" * 70)

    if not gelu_hwx_path or not os.path.exists(gelu_hwx_path):
        print("  ERROR: No GELU .hwx available to use as template")
        print("  Using fallback: will rely on LLDB to swap during next dispatch")
        return None

    # Read GELU .hwx
    with open(gelu_hwx_path, 'rb') as f:
        hwx_data = bytearray(f.read())

    original_size = len(hwx_data)
    print(f"  Template .hwx: {original_size} bytes")

    # Verify we have enough data for PWL at 0xC000
    if len(hwx_data) < PWL_OFFSET + PWL_SIZE:
        print(f"  ERROR: .hwx too small ({len(hwx_data)} < {PWL_OFFSET + PWL_SIZE})")
        return None

    # Show original PWL (GELU)
    original_pwl = hwx_data[PWL_OFFSET:PWL_OFFSET + PWL_SIZE]
    print(f"  Original PWL at 0x{PWL_OFFSET:X}: {original_pwl[:16].hex()}...")

    # Swap in Mish PWL
    mish_bytes = mish_pwl.to_bytes()
    assert len(mish_bytes) == PWL_SIZE, f"PWL size mismatch: {len(mish_bytes)} != {PWL_SIZE}"
    hwx_data[PWL_OFFSET:PWL_OFFSET + PWL_SIZE] = mish_bytes

    # Verify swap
    print(f"  Mish PWL at 0x{PWL_OFFSET:X}:  {hwx_data[PWL_OFFSET:PWL_OFFSET + 16].hex()}...")

    # Write custom .hwx (SAME SIZE as original -- critical for LLDB swap)
    assert len(hwx_data) == original_size, "Size changed during patch!"
    with open(CUSTOM_HWX_PATH, 'wb') as f:
        f.write(hwx_data)

    print(f"  Written: {CUSTOM_HWX_PATH} ({len(hwx_data)} bytes)")
    print(f"  Size match: {len(hwx_data)} == {original_size} (REQUIRED for LLDB swap)")

    return CUSTOM_HWX_PATH


# ═══════════════════════════════════════════════════════════════
# Step 4: Write LLDB swap module to /tmp/
# ═══════════════════════════════════════════════════════════════

def step4_write_lldb_module(custom_hwx_path: str):
    """Write the LLDB Python module for in-flight .hwx swap."""
    print()
    print("=" * 70)
    print("STEP 4: Write LLDB swap module")
    print("=" * 70)

    hwx_load_path = custom_hwx_path or CUSTOM_HWX_PATH

    # The LLDB module writes a sentinel file when the swap fires,
    # so the main process can detect success without parsing LLDB output.
    lldb_script = f'''#!/usr/bin/env python3
"""LLDB module: intercept sel=3 and swap .hwx with Mish version."""

import lldb
import struct as pystruct
import os
import json

CUSTOM_HWX = "{hwx_load_path}"
SENTINEL = "{WORK_DIR}/mish_swap_result.json"
SWAP_COUNT = 0

def intercept(frame, bp_loc, dict):
    global SWAP_COUNT
    proc = frame.GetThread().GetProcess()
    e = lldb.SBError()

    sel = frame.FindRegister('x1').GetValueAsUnsigned()
    if sel != 3:
        return False

    x2 = frame.FindRegister('x2').GetValueAsUnsigned()
    wrapper = proc.ReadMemory(x2, 32, e)
    if not wrapper:
        print(f"[MISH] Failed to read wrapper at 0x{{x2:X}}: {{e}}")
        return False

    inner_ptr = pystruct.unpack_from('<Q', wrapper, 0)[0]
    inner = proc.ReadMemory(inner_ptr, 16, e)
    if not inner:
        print(f"[MISH] Failed to read inner struct at 0x{{inner_ptr:X}}: {{e}}")
        return False

    hwx_addr = pystruct.unpack_from('<Q', inner, 0)[0]
    hwx_size = pystruct.unpack_from('<Q', inner, 8)[0]

    print(f"\\n[MISH] === SEL=3 INTERCEPTED ===")
    print(f"[MISH] hwx_addr = 0x{{hwx_addr:X}}")
    print(f"[MISH] hwx_size = 0x{{hwx_size:X}} ({{hwx_size}} bytes)")

    hwx_head = proc.ReadMemory(hwx_addr, 16, e)
    if hwx_head:
        print(f"[MISH] hwx header: {{hwx_head.hex()}}")

    try:
        custom = open(CUSTOM_HWX, 'rb').read()
    except Exception as ex:
        print(f"[MISH] Failed to read {{CUSTOM_HWX}}: {{ex}}")
        return False

    if len(custom) != hwx_size:
        print(f"[MISH] SIZE MISMATCH: custom={{len(custom)}} vs original={{hwx_size}}")
        # Write failure sentinel
        with open(SENTINEL, 'w') as sf:
            json.dump({{"status": "size_mismatch", "custom": len(custom), "original": hwx_size}}, sf)
        return False

    written = proc.WriteMemory(hwx_addr, custom, e)
    if e.Success():
        SWAP_COUNT += 1
        print(f"[MISH] OVERWROTE {{written}} bytes (swap #{{SWAP_COUNT}})")
        print(f"[MISH] Mish PWL is now at 0x{{hwx_addr + 0xC000:X}}")

        verify = proc.ReadMemory(hwx_addr + 0xC000, 8, e)
        if verify:
            print(f"[MISH] PWL verify: {{verify.hex()}}")

        # Write success sentinel so main process knows swap fired
        with open(SENTINEL, 'w') as sf:
            json.dump({{
                "status": "ok",
                "swap_count": SWAP_COUNT,
                "hwx_addr": f"0x{{hwx_addr:X}}",
                "hwx_size": hwx_size,
                "written": written,
            }}, sf)
    else:
        print(f"[MISH] WRITE FAILED: {{e}}")
        with open(SENTINEL, 'w') as sf:
            json.dump({{"status": "write_failed", "error": str(e)}}, sf)

    return False


def setup(debugger, command, result, internal_dict):
    target = debugger.GetSelectedTarget()
    bp = target.BreakpointCreateByName("IOConnectCallStructMethod")
    bp.SetScriptCallbackFunction("mish_swap.intercept")
    bp.SetAutoContinue(True)
    # Write a ready sentinel so main process knows breakpoint is armed
    with open("{WORK_DIR}/lldb_ready", 'w') as f:
        f.write("ready")
    print(f"[MISH] Breakpoint set ({{bp.GetNumLocations()}} locations)")
    print(f"[MISH] Custom .hwx: {{CUSTOM_HWX}}")
    print(f"[MISH] Armed and waiting for sel=3...")


def __lldb_init_module(debugger, internal_dict):
    debugger.HandleCommand('command script add -f mish_swap.setup mish_setup')
    print("[MISH] Module loaded. Run 'mish_setup' to activate.")
'''

    with open(LLDB_MODULE_PATH, 'w') as f:
        f.write(lldb_script)
    os.chmod(LLDB_MODULE_PATH, 0o755)
    print(f"  Written: {LLDB_MODULE_PATH}")


# ═══════════════════════════════════════════════════════════════
# Step 5: Automated LLDB + dispatch + verify (single process)
# ═══════════════════════════════════════════════════════════════

def mish_cpu(x):
    """CPU reference: Mish(x) = x * tanh(softplus(x))"""
    x32 = x.astype(np.float32)
    return (x32 * np.tanh(np.log1p(np.exp(x32)))).astype(np.float16)


def gelu_cpu(x):
    """CPU reference: GELU(x) = x * 0.5 * (1 + erf(x/sqrt(2)))"""
    from scipy.special import erf
    x32 = x.astype(np.float32)
    return (x32 * 0.5 * (1 + erf(x32 / np.sqrt(2)))).astype(np.float16)


def step5_run_automated(custom_hwx_path: str):
    """Launch LLDB, trigger dispatch, verify output -- all automated."""
    import json

    print()
    print("=" * 70)
    print("STEP 5: Automated LLDB attach + dispatch + verify")
    print("=" * 70)

    # Clean sentinel files
    sentinel_ready = os.path.join(WORK_DIR, 'lldb_ready')
    sentinel_result = os.path.join(WORK_DIR, 'mish_swap_result.json')
    for f in [sentinel_ready, sentinel_result]:
        if os.path.exists(f):
            os.remove(f)

    # Capture GELU baseline (no LLDB) for comparison
    x_test = np.linspace(-4.0, 4.0, DIM).astype(np.float16)
    print()
    print("  [5a] Capturing GELU baseline (no LLDB)...")
    subprocess.run(['killall', 'aned'], capture_output=True)
    time.sleep(1)
    baseline_result = subprocess.run(
        [EVAL_BINARY, GELU_MLMODELC, str(DIM), str(DIM)],
        input=x_test.tobytes(), capture_output=True, timeout=15,
    )
    if baseline_result.returncode == 0 and len(baseline_result.stdout) >= DIM * 2:
        gelu_baseline = np.frombuffer(baseline_result.stdout, dtype=np.float16)[:DIM]
        print(f"  GELU baseline captured: {gelu_baseline.shape}")
        print(f"  GELU[0]={gelu_baseline[0]:.4f}  GELU[-1]={gelu_baseline[-1]:.4f}")
    else:
        gelu_baseline = None
        print("  WARNING: Could not capture GELU baseline")
    time.sleep(0.5)

    # Find aned PID
    pid_result = subprocess.run(['pgrep', '-x', 'aned'], capture_output=True, text=True)
    if pid_result.returncode != 0:
        print("  ERROR: aned is not running. Cannot attach LLDB.")
        sys.exit(1)
    aned_pid = pid_result.stdout.strip().split('\n')[0]
    print(f"  aned PID: {aned_pid}")

    # Build LLDB command sequence
    # The key: import module, run setup (arms breakpoint + writes ready sentinel),
    # then continue (unpauses aned so it can process dispatches).
    lldb_input = '\n'.join([
        f'command script import {LLDB_MODULE_PATH}',
        'mish_setup',
        'continue',
    ]) + '\n'

    # Launch LLDB attached to aned
    print("  [5b] Launching LLDB attached to aned...")
    lldb_proc = subprocess.Popen(
        ['lldb', '-p', aned_pid],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    # Feed commands to LLDB (non-blocking write)
    def feed_lldb():
        try:
            time.sleep(2)  # Wait for LLDB to attach and show prompt
            lldb_proc.stdin.write(lldb_input)
            lldb_proc.stdin.flush()
        except Exception as ex:
            print(f"  WARNING: Failed to feed LLDB commands: {ex}")

    feeder = threading.Thread(target=feed_lldb, daemon=True)
    feeder.start()

    # Wait for ready sentinel (LLDB has armed breakpoint + continued)
    print("  [5c] Waiting for LLDB to arm breakpoint...")
    deadline = time.time() + 15
    ready = False
    while time.time() < deadline:
        if os.path.exists(sentinel_ready):
            ready = True
            break
        time.sleep(0.3)

    if not ready:
        print("  ERROR: LLDB did not arm breakpoint within 15s.")
        print("  LLDB output so far:")
        # Try to read what LLDB produced
        lldb_proc.stdin.close()
        try:
            out, _ = lldb_proc.communicate(timeout=3)
            print(out[-2000:] if len(out) > 2000 else out)
        except Exception:
            lldb_proc.kill()
        print()
        print_manual_fallback()
        sys.exit(1)

    # Give LLDB a moment to reach 'continue' (after mish_setup)
    time.sleep(1)
    print("  LLDB armed and aned resumed.")

    # Trigger the dispatch -- this is the ANE eval that LLDB will intercept
    print()
    print("  [5d] Triggering ANE dispatch (LLDB will swap .hwx)...")

    dispatch_result = subprocess.run(
        [EVAL_BINARY, GELU_MLMODELC, str(DIM), str(DIM)],
        input=x_test.tobytes(), capture_output=True, timeout=30,
    )

    # Wait a moment for LLDB callback to write sentinel
    time.sleep(1)

    # Read LLDB swap result
    swap_ok = False
    if os.path.exists(sentinel_result):
        with open(sentinel_result) as f:
            swap_result = json.load(f)
        print(f"  LLDB swap result: {swap_result}")
        swap_ok = swap_result.get('status') == 'ok'
    else:
        print("  WARNING: No swap sentinel found. LLDB may not have intercepted sel=3.")
        print("  (This can happen if aned used cached .hwx and skipped recompilation.)")

    # Detach LLDB
    print()
    print("  [5e] Detaching LLDB...")
    try:
        lldb_proc.stdin.write('detach\nquit\n')
        lldb_proc.stdin.flush()
        lldb_proc.wait(timeout=5)
    except Exception:
        lldb_proc.kill()
        lldb_proc.wait()
    print("  LLDB detached.")

    # Read LLDB full output for diagnostics
    # (already closed stdin, read stdout)
    lldb_output = ""
    try:
        lldb_output = lldb_proc.stdout.read()
    except Exception:
        pass

    # Show LLDB [MISH] lines
    mish_lines = [l for l in lldb_output.split('\n') if '[MISH]' in l]
    if mish_lines:
        print()
        print("  LLDB intercept log:")
        for line in mish_lines:
            print(f"    {line.strip()}")

    # Verify dispatch output
    print()
    print("=" * 70)
    print("STEP 6: Verification")
    print("=" * 70)

    if dispatch_result.returncode != 0:
        print(f"  DISPATCH FAILED (rc={dispatch_result.returncode})")
        print(f"  stderr: {dispatch_result.stderr.decode()[:300]}")
        print()
        print_manual_fallback()
        return

    ane_output = np.frombuffer(dispatch_result.stdout, dtype=np.float16)
    if len(ane_output) < DIM:
        print(f"  WARNING: Expected {DIM} values, got {len(ane_output)}")
        if len(ane_output) == 0:
            print("  No output data.")
            print()
            print_manual_fallback()
            return
    ane_output = ane_output[:DIM]

    # CPU references
    mish_expected = mish_cpu(x_test)
    gelu_expected = gelu_cpu(x_test)

    mish_diff = np.abs(ane_output.astype(np.float32) - mish_expected.astype(np.float32))
    gelu_diff = np.abs(ane_output.astype(np.float32) - gelu_expected.astype(np.float32))

    mish_max = mish_diff.max()
    gelu_max = gelu_diff.max()
    mish_mean = mish_diff.mean()
    gelu_mean = gelu_diff.mean()

    print()
    print(f"  vs Mish CPU:  max_err={mish_max:.6f}  mean_err={mish_mean:.6f}")
    print(f"  vs GELU CPU:  max_err={gelu_max:.6f}  mean_err={gelu_mean:.6f}")

    # Compare against GELU baseline (most reliable test: did output change?)
    if gelu_baseline is not None:
        baseline_diff = np.abs(ane_output.astype(np.float32) - gelu_baseline.astype(np.float32))
        baseline_max = baseline_diff.max()
        baseline_mean = baseline_diff.mean()
        print(f"  vs GELU baseline: max_diff={baseline_max:.6f}  mean_diff={baseline_mean:.6f}")
        output_changed = baseline_max > 0.001
    else:
        output_changed = None

    print()
    if swap_ok and output_changed:
        print("  RESULT: .hwx swap CONFIRMED -- ANE output differs from GELU baseline")
        print("  >>> IN-FLIGHT .hwx MODIFICATION ON ANE HARDWARE CONFIRMED <<<")
    elif swap_ok and output_changed is False:
        print("  RESULT: Swap fired but output unchanged vs GELU baseline.")
        print("  The kext may have already DMA'd the .hwx before the overwrite.")
    elif mish_max < gelu_max and mish_max < 0.01:
        print("  RESULT: ANE output matches MISH (not GELU)")
        print("  >>> CUSTOM ACTIVATION ON ANE HARDWARE CONFIRMED <<<")
    elif mish_max < 0.01:
        print("  RESULT: ANE output matches Mish (close to GELU too, expected at small x)")
    elif gelu_max < mish_max:
        print("  RESULT: ANE output matches GELU (swap may not have fired)")
    else:
        print("  RESULT: ANE output matches neither cleanly. Check LLDB log.")

    # Sample comparison
    print()
    header = "  x        ANE       Mish_ref  GELU_ref"
    if gelu_baseline is not None:
        header += "  Baseline"
    print(header)
    indices = [0, DIM // 4, DIM // 2, 3 * DIM // 4, DIM - 1]
    for i in indices:
        line = (f"  {x_test[i]:6.3f}    {ane_output[i]:7.4f}    "
                f"{mish_expected[i]:7.4f}    {gelu_expected[i]:7.4f}")
        if gelu_baseline is not None:
            line += f"    {gelu_baseline[i]:7.4f}"
        print(line)

    # Save results
    np.save(os.path.join(WORK_DIR, 'ane_output.npy'), ane_output)
    np.save(os.path.join(WORK_DIR, 'test_input.npy'), x_test)
    print(f"\n  Saved: {WORK_DIR}/ane_output.npy")

    print()
    print("=" * 70)
    print("FILES")
    print("=" * 70)
    for p in [GELU_MLMODELC, CUSTOM_HWX_PATH, LLDB_MODULE_PATH,
              sentinel_result]:
        exists = "OK" if os.path.exists(p) else "MISSING"
        print(f"  [{exists}] {p}")


def print_manual_fallback():
    """Print 2-terminal fallback instructions if automation fails."""
    print("=" * 70)
    print("MANUAL FALLBACK (2-terminal mode)")
    print("=" * 70)
    print()
    print("Automated LLDB control failed. Run manually in 2 terminals:")
    print()
    print("--- TERMINAL 1 ---")
    print("  sudo lldb -n aned")
    print(f"  (lldb) command script import {LLDB_MODULE_PATH}")
    print("  (lldb) mish_setup")
    print("  (lldb) continue")
    print()
    print("--- TERMINAL 2 ---")
    x_test_path = os.path.join(WORK_DIR, 'test_input.npy')
    print(f"  cd {SCRIPT_DIR}")
    print(f"  {sys.executable} -c \"")
    print(f"  import subprocess, numpy as np")
    print(f"  x = np.linspace(-4,4,{DIM}).astype(np.float16)")
    print(f"  r = subprocess.run(['{EVAL_BINARY}', '{GELU_MLMODELC}', '{DIM}', '{DIM}'],")
    print(f"      input=x.tobytes(), capture_output=True, timeout=30)")
    print(f"  o = np.frombuffer(r.stdout, dtype=np.float16)")
    print(f"  m = (x.astype('f4')*np.tanh(np.log1p(np.exp(x.astype('f4'))))).astype('f2')")
    print(f"  print('max_err vs Mish:', abs(o.astype('f4')-m.astype('f4')).max())\"")
    print()
    print("Then Ctrl-C in Terminal 1 to detach LLDB.")
    print()


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    print()
    print("  ANE-COMPILER: SIP-OFF DEMO (automated)")
    print("  Custom Mish Activation on Apple Neural Engine")
    print("  LLDB In-Flight .hwx Swap -- Single Command")
    print()

    # ── Check SIP ──
    sip = subprocess.run(['csrutil', 'status'], capture_output=True, text=True)
    if 'enabled' in sip.stdout.lower():
        print("ERROR: System Integrity Protection is ENABLED.")
        print("This demo requires SIP OFF to attach LLDB to aned.")
        print()
        print("To disable SIP:")
        print("  1. Restart into Recovery Mode (hold power button)")
        print("  2. Open Terminal from Utilities menu")
        print("  3. Run: csrutil disable")
        print("  4. Restart")
        sys.exit(1)
    print("  SIP: disabled (OK)")

    # ── Check root ──
    if os.geteuid() != 0:
        print("ERROR: Must run as root (LLDB attach requires root).")
        print(f"  sudo {sys.executable} {os.path.abspath(__file__)}")
        sys.exit(1)
    print("  Root: yes (OK)")

    # ── Check eval binary ──
    if not os.path.exists(EVAL_BINARY):
        print(f"ERROR: ane_eval_binary not found at {EVAL_BINARY}")
        print("Build it: cd tests && xcrun clang -framework Foundation "
              "-framework IOSurface -fobjc-arc -o ane_eval_binary ane_eval_binary.m")
        sys.exit(1)
    print(f"  Eval binary: {EVAL_BINARY} (OK)")
    print()

    # Step 1: Generate GELU .hwx (template)
    gelu_hwx = step1_generate_gelu()

    # Step 2: Build Mish PWL (using GELU header for HW-consistent range)
    mish_pwl = step2_build_mish_pwl(gelu_hwx)

    # Step 3: Create custom .hwx
    custom_hwx = step3_create_custom_hwx(gelu_hwx, mish_pwl)

    # Step 4: Write LLDB module
    step4_write_lldb_module(custom_hwx)

    # Step 5: Automated LLDB + dispatch + verify
    step5_run_automated(custom_hwx)

    print()
    print("  Demo complete.")


if __name__ == '__main__':
    main()
