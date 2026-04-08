#!/usr/bin/env python3
"""
MIL Custom Activation Test via _ANEInMemoryModel

Tests whether MIL IR can compile activations that espresso rejects.
Espresso only accepts 12 activation modes (0-3,5,6,8,19,21,22,23,25).
MIL has compositional ops: mul, sigmoid, tanh, softplus — can they compose
Mish = x * tanh(softplus(x)) on ANE?

Uses proven compile_and_run() from true_5_0_kill.py.
"""
import objc, os, plistlib, ctypes, shutil, time
import numpy as np
from Foundation import *

objc.loadBundle('AppleNeuralEngine', globals(),
    bundle_path='/System/Library/PrivateFrameworks/AppleNeuralEngine.framework')

ANEInMemoryModel = objc.lookUpClass('_ANEInMemoryModel')
ANEInMemoryModelDescriptor = objc.lookUpClass('_ANEInMemoryModelDescriptor')
ANERequest = objc.lookUpClass('_ANERequest')
ANEIOSurfaceObject = objc.lookUpClass('_ANEIOSurfaceObject')

IOSL = ctypes.cdll.LoadLibrary('/System/Library/Frameworks/IOSurface.framework/IOSurface')
for fn, rt, at in [
    ('IOSurfaceCreate', ctypes.c_void_p, [ctypes.c_void_p]),
    ('IOSurfaceLock', ctypes.c_int, [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p]),
    ('IOSurfaceUnlock', ctypes.c_int, [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p]),
    ('IOSurfaceGetBaseAddress', ctypes.c_void_p, [ctypes.c_void_p]),
]:
    getattr(IOSL, fn).restype = rt
    getattr(IOSL, fn).argtypes = at

objc.registerMetaDataForSelector(b'_ANEIOSurfaceObject',
    b'initWithIOSurface:startOffset:shouldRetain:',
    {'arguments': {2: {'type': b'^v'}}})


def compile_and_run(mil_text, test_vals, label="test"):
    """Compile MIL, run on ANE, return (output_list, error_string)."""
    print(f"  [{label}] Compiling MIL ({len(mil_text)} bytes)...")

    ns_net = NSData.dataWithBytes_length_(mil_text, len(mil_text))
    opts = plistlib.dumps({'h17g': {'EnableLowEffortCPAllocation': True}}, fmt=plistlib.FMT_BINARY)
    ns_opts = NSData.dataWithBytes_length_(opts, len(opts))
    desc = ANEInMemoryModelDescriptor.alloc().initWithNetworkText_weights_optionsPlist_isMILModel_(
        ns_net, NSDictionary.dictionary(), ns_opts, True)
    model = ANEInMemoryModel.alloc().initWithDesctiptor_(desc)
    model.purgeCompiledModel()
    model.saveModelFiles()
    lmp = model.localModelPath()
    print(f"  [{label}] Local model path: {lmp}")

    # List files in model dir
    if lmp and os.path.isdir(lmp):
        for f in os.listdir(lmp):
            fp = os.path.join(lmp, f)
            sz = os.path.getsize(fp) if os.path.isfile(fp) else 0
            print(f"    {f}: {sz} bytes")

    # Copy net.plist -> model.mil so compiler finds it
    net_plist = os.path.join(lmp, 'net.plist')
    model_mil = os.path.join(lmp, 'model.mil')
    if os.path.exists(net_plist):
        shutil.copy2(net_plist, model_mil)
        print(f"  [{label}] Copied net.plist -> model.mil")

    # COMPILE
    print(f"  [{label}] Calling compileWithQoS...")
    err_ptr = objc.nil
    try:
        ok = model.compileWithQoS_options_error_(0, NSDictionary.dictionary(), None)
    except Exception as e:
        print(f"  [{label}] COMPILE EXCEPTION: {e}")
        return None, f"compile exception: {e}"

    if not ok:
        # Try to get error details
        print(f"  [{label}] COMPILE FAILED (returned False)")
        # Check if hwx was generated
        if lmp and os.path.isdir(lmp):
            for f in os.listdir(lmp):
                fp = os.path.join(lmp, f)
                sz = os.path.getsize(fp) if os.path.isfile(fp) else 0
                print(f"    POST-COMPILE {f}: {sz} bytes")
        return None, "compile failed"
    print(f"  [{label}] COMPILE SUCCESS")

    # Check post-compile files
    if lmp and os.path.isdir(lmp):
        for f in os.listdir(lmp):
            fp = os.path.join(lmp, f)
            sz = os.path.getsize(fp) if os.path.isfile(fp) else 0
            print(f"    POST-COMPILE {f}: {sz} bytes")

    # LOAD
    print(f"  [{label}] Calling loadWithQoS...")
    ok = model.loadWithQoS_options_error_(0, NSDictionary.dictionary(), None)
    if not ok:
        print(f"  [{label}] LOAD FAILED")
        return None, "load failed"
    print(f"  [{label}] LOAD SUCCESS")

    # Get model attributes for IOSurface sizing
    attrs = model.modelAttributes()
    if not attrs:
        print(f"  [{label}] No model attributes")
        model.unloadWithQoS_error_(0, None)
        return None, "no attributes"

    ns = attrs['NetworkStatusList'][0]
    in_info = ns['LiveInputList'][0]
    out_info = ns['LiveOutputList'][0]
    ps = int(in_info['PlaneStride'])
    bs = int(in_info['BatchStride'])
    ch = int(in_info['Channels'])
    dtype_str = str(in_info['Type'])
    dtype = np.float16 if dtype_str == 'Float16' else np.float32
    elem = 2 if dtype == np.float16 else 4

    print(f"  [{label}] PlaneStride={ps}, BatchStride={bs}, Channels={ch}, Type={dtype_str}")

    def mk_surf():
        props = NSMutableDictionary.dictionary()
        props.setObject_forKey_(NSNumber.numberWithInt_(bs // 2), 'IOSurfaceWidth')
        props.setObject_forKey_(NSNumber.numberWithInt_(1), 'IOSurfaceHeight')
        props.setObject_forKey_(NSNumber.numberWithInt_(bs), 'IOSurfaceBytesPerRow')
        props.setObject_forKey_(NSNumber.numberWithInt_(2), 'IOSurfaceBytesPerElement')
        props.setObject_forKey_(NSNumber.numberWithInt_(bs), 'IOSurfaceAllocSize')
        props.setObject_forKey_(NSNumber.numberWithInt_(0x6630304C), 'IOSurfacePixelFormat')
        return IOSL.IOSurfaceCreate(objc.pyobjc_id(props))

    in_ref = mk_surf()
    out_ref = mk_surf()

    IOSL.IOSurfaceLock(in_ref, 0, None)
    base = IOSL.IOSurfaceGetBaseAddress(in_ref)
    ctypes.memset(base, 0, bs)
    for i, v in enumerate(test_vals):
        if i >= ch: break
        val = np.array([v], dtype=dtype)
        ctypes.memmove(base + i * ps, val.tobytes(), elem)
    IOSL.IOSurfaceUnlock(in_ref, 0, None)

    in_obj = ANEIOSurfaceObject.alloc().initWithIOSurface_startOffset_shouldRetain_(in_ref, 0, True)
    out_obj = ANEIOSurfaceObject.alloc().initWithIOSurface_startOffset_shouldRetain_(out_ref, 0, True)
    req = ANERequest.alloc().initWithInputs_inputIndices_outputs_outputIndices_weightsBuffer_perfStats_procedureIndex_sharedEvents_transactionHandle_(
        NSArray.arrayWithObject_(in_obj), NSArray.arrayWithObject_(NSNumber.numberWithInt_(0)),
        NSArray.arrayWithObject_(out_obj), NSArray.arrayWithObject_(NSNumber.numberWithInt_(0)),
        None, None, NSNumber.numberWithInt_(0), None, None)

    map_ok = model.mapIOSurfacesWithRequest_cacheInference_error_(req, False, None)
    if not map_ok:
        model.unloadWithQoS_error_(0, None)
        return None, "map failed"

    # EVALUATE
    print(f"  [{label}] Calling evaluateWithQoS...")
    t0 = time.perf_counter()
    eval_ok = model.evaluateWithQoS_options_request_error_(0, None, req, None)
    t1 = time.perf_counter()
    if not eval_ok:
        model.unmapIOSurfacesWithRequest_(req)
        model.unloadWithQoS_error_(0, None)
        print(f"  [{label}] EVAL FAILED")
        return None, "eval failed"

    latency_us = (t1 - t0) * 1e6
    print(f"  [{label}] EVAL SUCCESS ({latency_us:.0f} µs)")

    # Read output
    IOSL.IOSurfaceLock(out_ref, 1, None)
    out_base = IOSL.IOSurfaceGetBaseAddress(out_ref)
    output = []
    for i in range(min(len(test_vals), ch)):
        raw = (ctypes.c_uint8 * elem)()
        ctypes.memmove(raw, out_base + i * ps, elem)
        val = np.frombuffer(bytes(raw), dtype=dtype)[0]
        output.append(float(val))
    IOSL.IOSurfaceUnlock(out_ref, 1, None)

    model.unmapIOSurfacesWithRequest_(req)
    model.unloadWithQoS_error_(0, None)
    return output, None


def benchmark(mil_text, test_vals, label, n_iters=100):
    """Benchmark dispatch latency."""
    ns_net = NSData.dataWithBytes_length_(mil_text, len(mil_text))
    opts = plistlib.dumps({'h17g': {'EnableLowEffortCPAllocation': True}}, fmt=plistlib.FMT_BINARY)
    ns_opts = NSData.dataWithBytes_length_(opts, len(opts))
    desc = ANEInMemoryModelDescriptor.alloc().initWithNetworkText_weights_optionsPlist_isMILModel_(
        ns_net, NSDictionary.dictionary(), ns_opts, True)
    model = ANEInMemoryModel.alloc().initWithDesctiptor_(desc)
    model.purgeCompiledModel()
    model.saveModelFiles()
    lmp = model.localModelPath()
    if lmp:
        net_plist = os.path.join(lmp, 'net.plist')
        model_mil = os.path.join(lmp, 'model.mil')
        if os.path.exists(net_plist):
            shutil.copy2(net_plist, model_mil)

    ok = model.compileWithQoS_options_error_(0, NSDictionary.dictionary(), None)
    if not ok:
        return None, "compile failed"
    ok = model.loadWithQoS_options_error_(0, NSDictionary.dictionary(), None)
    if not ok:
        return None, "load failed"

    attrs = model.modelAttributes()
    ns = attrs['NetworkStatusList'][0]
    in_info = ns['LiveInputList'][0]
    ps = int(in_info['PlaneStride'])
    bs = int(in_info['BatchStride'])
    ch = int(in_info['Channels'])
    dtype = np.float16

    def mk_surf():
        props = NSMutableDictionary.dictionary()
        props.setObject_forKey_(NSNumber.numberWithInt_(bs // 2), 'IOSurfaceWidth')
        props.setObject_forKey_(NSNumber.numberWithInt_(1), 'IOSurfaceHeight')
        props.setObject_forKey_(NSNumber.numberWithInt_(bs), 'IOSurfaceBytesPerRow')
        props.setObject_forKey_(NSNumber.numberWithInt_(2), 'IOSurfaceBytesPerElement')
        props.setObject_forKey_(NSNumber.numberWithInt_(bs), 'IOSurfaceAllocSize')
        props.setObject_forKey_(NSNumber.numberWithInt_(0x6630304C), 'IOSurfacePixelFormat')
        return IOSL.IOSurfaceCreate(objc.pyobjc_id(props))

    in_ref = mk_surf()
    out_ref = mk_surf()

    IOSL.IOSurfaceLock(in_ref, 0, None)
    base = IOSL.IOSurfaceGetBaseAddress(in_ref)
    ctypes.memset(base, 0, bs)
    for i, v in enumerate(test_vals):
        if i >= ch: break
        val = np.array([v], dtype=dtype)
        ctypes.memmove(base + i * ps, val.tobytes(), 2)
    IOSL.IOSurfaceUnlock(in_ref, 0, None)

    in_obj = ANEIOSurfaceObject.alloc().initWithIOSurface_startOffset_shouldRetain_(in_ref, 0, True)
    out_obj = ANEIOSurfaceObject.alloc().initWithIOSurface_startOffset_shouldRetain_(out_ref, 0, True)
    req = ANERequest.alloc().initWithInputs_inputIndices_outputs_outputIndices_weightsBuffer_perfStats_procedureIndex_sharedEvents_transactionHandle_(
        NSArray.arrayWithObject_(in_obj), NSArray.arrayWithObject_(NSNumber.numberWithInt_(0)),
        NSArray.arrayWithObject_(out_obj), NSArray.arrayWithObject_(NSNumber.numberWithInt_(0)),
        None, None, NSNumber.numberWithInt_(0), None, None)

    model.mapIOSurfacesWithRequest_cacheInference_error_(req, False, None)

    # Warmup
    for _ in range(5):
        model.evaluateWithQoS_options_request_error_(0, None, req, None)

    # Benchmark
    times = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        model.evaluateWithQoS_options_request_error_(0, None, req, None)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1e6)

    model.unmapIOSurfacesWithRequest_(req)
    model.unloadWithQoS_error_(0, None)

    return times, None


BUILD_INFO = b'dict<string, string>({{"coremlc-component-MIL", "3510.2.1"}, {"coremlc-version", "3500.32.1"}, {"coremltools-component-milinternal", ""}, {"coremltools-version", "9.0"}})'

def make_mil(body_lines):
    """Build a MIL program with given body lines."""
    header = b'''program(1.3)
[buildInfo = ''' + BUILD_INFO + b''']
{
    func main<ios18>(tensor<fp16, [1, 64, 1, 1]> x) {
'''
    footer = b'''
    } -> (output);
}'''
    body = b'\n'.join(b'            ' + line for line in body_lines)
    return header + body + footer


def main():
    print("=" * 70)
    print("MIL CUSTOM ACTIVATION TEST")
    print("Can MIL IR compile activations that espresso rejects?")
    print("=" * 70)

    test_input = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]
    results = {}

    # ================================================================
    # TEST 1: ReLU (baseline — espresso mode 1 accepts this)
    # ================================================================
    RELU_MIL = make_mil([
        b'tensor<fp16, [1, 64, 1, 1]> output = relu(x = x)[name = string("output")];',
    ])

    print(f"\n{'=' * 70}")
    print("TEST 1: relu(x) — baseline (espresso mode 1)")
    print(f"{'=' * 70}")

    relu_expected = [max(0, v) for v in test_input]
    out, err = compile_and_run(RELU_MIL, test_input, "relu")
    if err:
        print(f"  RESULT: FAIL — {err}")
        results['relu'] = (False, err)
    else:
        max_err = max(abs(a - b) for a, b in zip(out, relu_expected))
        print(f"  Output:   {[round(v, 4) for v in out]}")
        print(f"  Expected: {relu_expected}")
        print(f"  Max error: {max_err:.6f}")
        results['relu'] = (max_err < 0.01, out)

    # ================================================================
    # TEST 2: SiLU = x * sigmoid(x)
    # espresso mode 25 accepts SiLU — but let's see if MIL composes it
    # ================================================================
    SILU_MIL = make_mil([
        b'tensor<fp16, [1, 64, 1, 1]> sig = sigmoid(x = x)[name = string("sig")];',
        b'tensor<fp16, [1, 64, 1, 1]> output = mul(x = x, y = sig)[name = string("output")];',
    ])

    print(f"\n{'=' * 70}")
    print("TEST 2: x * sigmoid(x) — SiLU composed from primitives")
    print("  (espresso mode 25 does this as a single op)")
    print(f"{'=' * 70}")

    def silu(x):
        return x / (1.0 + np.exp(-x))
    silu_expected = [float(silu(v)) for v in test_input]

    out, err = compile_and_run(SILU_MIL, test_input, "silu_composed")
    if err:
        print(f"  RESULT: FAIL — {err}")
        results['silu_composed'] = (False, err)
    else:
        max_err = max(abs(a - b) for a, b in zip(out, silu_expected))
        print(f"  Output:   {[round(v, 4) for v in out]}")
        print(f"  Expected: {[round(v, 4) for v in silu_expected]}")
        print(f"  Max error: {max_err:.6f}")
        results['silu_composed'] = (max_err < 0.05, out)

    # ================================================================
    # TEST 3: Mish = x * tanh(softplus(x))
    # softplus(x) = log(1 + exp(x))
    # This is NOT in any espresso mode. If this compiles, it's novel.
    # ================================================================

    # Try with softplus first (coremltools MIL has softplus)
    MISH_SOFTPLUS_MIL = make_mil([
        b'tensor<fp16, [1, 64, 1, 1]> sp = softplus(x = x)[name = string("sp")];',
        b'tensor<fp16, [1, 64, 1, 1]> th = tanh(x = sp)[name = string("th")];',
        b'tensor<fp16, [1, 64, 1, 1]> output = mul(x = x, y = th)[name = string("output")];',
    ])

    print(f"\n{'=' * 70}")
    print("TEST 3a: x * tanh(softplus(x)) — Mish via softplus op")
    print("  NOT in any espresso activation mode")
    print(f"{'=' * 70}")

    def mish(x):
        sp = np.log(1.0 + np.exp(x))
        return x * np.tanh(sp)
    mish_expected = [float(mish(v)) for v in test_input]

    out, err = compile_and_run(MISH_SOFTPLUS_MIL, test_input, "mish_softplus")
    if err:
        print(f"  RESULT: FAIL — {err}")
        results['mish_softplus'] = (False, err)
    else:
        max_err = max(abs(a - b) for a, b in zip(out, mish_expected))
        print(f"  Output:   {[round(v, 4) for v in out]}")
        print(f"  Expected: {[round(v, 4) for v in mish_expected]}")
        print(f"  Max error: {max_err:.6f}")
        results['mish_softplus'] = (max_err < 0.1, out)

    # ================================================================
    # TEST 3b: Mish decomposed: x * tanh(log(1 + exp(x)))
    # In case softplus isn't recognized, decompose fully
    # ================================================================
    MISH_DECOMPOSED_MIL = make_mil([
        b'tensor<fp16, [1, 64, 1, 1]> ex = exp(x = x)[name = string("ex")];',
        b'tensor<fp16, []> one = const()[name = string("one"), val = tensor<fp16, []>(1.0)];',
        b'tensor<fp16, [1, 64, 1, 1]> ep1 = add(x = ex, y = one)[name = string("ep1")];',
        b'tensor<fp16, [1, 64, 1, 1]> lg = log(x = ep1)[name = string("lg")];',
        b'tensor<fp16, [1, 64, 1, 1]> th = tanh(x = lg)[name = string("th")];',
        b'tensor<fp16, [1, 64, 1, 1]> output = mul(x = x, y = th)[name = string("output")];',
    ])

    print(f"\n{'=' * 70}")
    print("TEST 3b: x * tanh(log(1 + exp(x))) — Mish fully decomposed")
    print(f"{'=' * 70}")

    out, err = compile_and_run(MISH_DECOMPOSED_MIL, test_input, "mish_decomposed")
    if err:
        print(f"  RESULT: FAIL — {err}")
        results['mish_decomposed'] = (False, err)
    else:
        max_err = max(abs(a - b) for a, b in zip(out, mish_expected))
        print(f"  Output:   {[round(v, 4) for v in out]}")
        print(f"  Expected: {[round(v, 4) for v in mish_expected]}")
        print(f"  Max error: {max_err:.6f}")
        results['mish_decomposed'] = (max_err < 0.1, out)

    # ================================================================
    # TEST 4: GELU (tanh approximation) — not in espresso modes
    # gelu_tanh(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    # ================================================================
    GELU_TANH_MIL = make_mil([
        b'tensor<fp16, [1, 64, 1, 1]> x2 = mul(x = x, y = x)[name = string("x2")];',
        b'tensor<fp16, [1, 64, 1, 1]> x3 = mul(x = x, y = x2)[name = string("x3")];',
        b'tensor<fp16, []> c1 = const()[name = string("c1"), val = tensor<fp16, []>(0.044715)];',
        b'tensor<fp16, [1, 64, 1, 1]> cx3 = mul(x = c1, y = x3)[name = string("cx3")];',
        b'tensor<fp16, [1, 64, 1, 1]> xp = add(x = x, y = cx3)[name = string("xp")];',
        b'tensor<fp16, []> c2 = const()[name = string("c2"), val = tensor<fp16, []>(0.7978845608)];',
        b'tensor<fp16, [1, 64, 1, 1]> inner = mul(x = c2, y = xp)[name = string("inner")];',
        b'tensor<fp16, [1, 64, 1, 1]> th = tanh(x = inner)[name = string("th")];',
        b'tensor<fp16, []> one = const()[name = string("one"), val = tensor<fp16, []>(1.0)];',
        b'tensor<fp16, [1, 64, 1, 1]> tp1 = add(x = th, y = one)[name = string("tp1")];',
        b'tensor<fp16, []> half = const()[name = string("half"), val = tensor<fp16, []>(0.5)];',
        b'tensor<fp16, [1, 64, 1, 1]> hx = mul(x = half, y = x)[name = string("hx")];',
        b'tensor<fp16, [1, 64, 1, 1]> output = mul(x = hx, y = tp1)[name = string("output")];',
    ])

    print(f"\n{'=' * 70}")
    print("TEST 4: GELU (tanh approximation) — composed from primitives")
    print("  (espresso modes 19/21/22 do erf-based GELU, not tanh-based)")
    print(f"{'=' * 70}")

    def gelu_tanh(x):
        return 0.5 * x * (1.0 + np.tanh(0.7978845608 * (x + 0.044715 * x**3)))
    gelu_expected = [float(gelu_tanh(v)) for v in test_input]

    out, err = compile_and_run(GELU_TANH_MIL, test_input, "gelu_tanh")
    if err:
        print(f"  RESULT: FAIL — {err}")
        results['gelu_tanh'] = (False, err)
    else:
        max_err = max(abs(a - b) for a, b in zip(out, gelu_expected))
        print(f"  Output:   {[round(v, 4) for v in out]}")
        print(f"  Expected: {[round(v, 4) for v in gelu_expected]}")
        print(f"  Max error: {max_err:.6f}")
        results['gelu_tanh'] = (max_err < 0.1, out)

    # ================================================================
    # TEST 5: Squared ReLU — ReLU(x)^2
    # Used in some modern architectures. Not an espresso mode.
    # ================================================================
    SQRELU_MIL = make_mil([
        b'tensor<fp16, [1, 64, 1, 1]> r = relu(x = x)[name = string("r")];',
        b'tensor<fp16, [1, 64, 1, 1]> output = mul(x = r, y = r)[name = string("output")];',
    ])

    print(f"\n{'=' * 70}")
    print("TEST 5: ReLU(x)^2 — squared ReLU")
    print(f"{'=' * 70}")

    sqrelu_expected = [max(0, v)**2 for v in test_input]

    out, err = compile_and_run(SQRELU_MIL, test_input, "sqrelu")
    if err:
        print(f"  RESULT: FAIL — {err}")
        results['sqrelu'] = (False, err)
    else:
        max_err = max(abs(a - b) for a, b in zip(out, sqrelu_expected))
        print(f"  Output:   {[round(v, 4) for v in out]}")
        print(f"  Expected: {sqrelu_expected}")
        print(f"  Max error: {max_err:.6f}")
        results['sqrelu'] = (max_err < 0.01, out)

    # ================================================================
    # SUMMARY
    # ================================================================
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")

    for name, (passed, detail) in results.items():
        if passed:
            status = "PASS"
        elif isinstance(detail, str):
            status = f"FAIL ({detail})"
        else:
            status = "FAIL (wrong output)"
        print(f"  {name}: {status}")

    passed_count = sum(1 for _, (p, _) in results.items() if p)
    print(f"\n  Passed: {passed_count}/{len(results)}")

    # ================================================================
    # BENCHMARK any that passed
    # ================================================================
    benchmark_targets = {}
    if results.get('mish_softplus', (False,))[0]:
        benchmark_targets['mish_softplus'] = MISH_SOFTPLUS_MIL
    if results.get('mish_decomposed', (False,))[0]:
        benchmark_targets['mish_decomposed'] = MISH_DECOMPOSED_MIL
    if results.get('silu_composed', (False,))[0]:
        benchmark_targets['silu_composed'] = SILU_MIL
    if results.get('gelu_tanh', (False,))[0]:
        benchmark_targets['gelu_tanh'] = GELU_TANH_MIL
    if results.get('sqrelu', (False,))[0]:
        benchmark_targets['sqrelu'] = SQRELU_MIL

    if benchmark_targets:
        print(f"\n{'=' * 70}")
        print("BENCHMARKS (100 iterations)")
        print(f"{'=' * 70}")
        for name, mil in benchmark_targets.items():
            times, err = benchmark(mil, test_input, name)
            if err:
                print(f"  {name}: BENCH FAILED ({err})")
            else:
                med = np.median(times)
                p5 = np.percentile(times, 5)
                p95 = np.percentile(times, 95)
                print(f"  {name}: median={med:.0f}µs, p5={p5:.0f}µs, p95={p95:.0f}µs")

    # ================================================================
    # Check .hwx structure for any compiled model
    # ================================================================
    if any(p for p, _ in results.values()):
        print(f"\n{'=' * 70}")
        print("HWX ANALYSIS (checking dispatch count)")
        print(f"{'=' * 70}")

        # Recompile Mish to inspect .hwx
        for name, mil in [('mish_softplus', MISH_SOFTPLUS_MIL),
                          ('mish_decomposed', MISH_DECOMPOSED_MIL),
                          ('silu_composed', SILU_MIL)]:
            if not results.get(name, (False,))[0]:
                continue
            ns_net = NSData.dataWithBytes_length_(mil, len(mil))
            opts_data = plistlib.dumps({'h17g': {'EnableLowEffortCPAllocation': True}}, fmt=plistlib.FMT_BINARY)
            ns_opts = NSData.dataWithBytes_length_(opts_data, len(opts_data))
            desc = ANEInMemoryModelDescriptor.alloc().initWithNetworkText_weights_optionsPlist_isMILModel_(
                ns_net, NSDictionary.dictionary(), ns_opts, True)
            mdl = ANEInMemoryModel.alloc().initWithDesctiptor_(desc)
            mdl.purgeCompiledModel()
            mdl.saveModelFiles()
            lmp = mdl.localModelPath()
            if lmp and os.path.exists(os.path.join(lmp, 'net.plist')):
                shutil.copy2(os.path.join(lmp, 'net.plist'), os.path.join(lmp, 'model.mil'))
            mdl.compileWithQoS_options_error_(0, NSDictionary.dictionary(), None)

            # Find .hwx
            if lmp and os.path.isdir(lmp):
                for f in sorted(os.listdir(lmp)):
                    if f.endswith('.hwx'):
                        hwx_path = os.path.join(lmp, f)
                        sz = os.path.getsize(hwx_path)
                        print(f"  {name}: {f} = {sz} bytes")
                        # Read Mach-O header to count load commands
                        with open(hwx_path, 'rb') as fh:
                            magic = fh.read(4)
                            if magic == b'\xcf\xfa\xed\xfe':
                                fh.seek(16)  # ncmds at offset 16
                                ncmds = int.from_bytes(fh.read(4), 'little')
                                sizeofcmds = int.from_bytes(fh.read(4), 'little')
                                print(f"    ncmds={ncmds}, sizeofcmds={sizeofcmds}")
                                # ncmds roughly indicates dispatch count
                                # Single-pass: ~11 ncmds
                                # Multi-pass: more
            mdl.unloadWithQoS_error_(0, None)


if __name__ == "__main__":
    main()
