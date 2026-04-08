#!/usr/bin/env python3
"""
MIL IR Fusion Boundary Test
Maps the maximum graph size aned can compile into a single ANE dispatch.

Incremental tests from single matmul up to full transformer layer.
"""
import objc, os, plistlib, ctypes, shutil, time, struct
import numpy as np
from Foundation import *

objc.loadBundle('AppleNeuralEngine', globals(),
    bundle_path='/System/Library/PrivateFrameworks/AppleNeuralEngine.framework')

ANEInMemoryModel = objc.lookUpClass('_ANEInMemoryModel')
ANEInMemoryModelDescriptor = objc.lookUpClass('_ANEInMemoryModelDescriptor')

BUILD_INFO = b'dict<string, string>({{"coremlc-component-MIL", "3510.2.1"}, {"coremlc-version", "3500.32.1"}, {"coremltools-component-milinternal", ""}, {"coremltools-version", "9.0"}})'


def compile_mil(mil_text, label="test"):
    """Compile MIL text, return (model, attrs, lmp, hwx_size, compile_time) or (None, error_str, ...)."""
    print(f"\n{'='*70}")
    print(f"TEST: {label}")
    print(f"{'='*70}")

    ns_net = NSData.dataWithBytes_length_(mil_text, len(mil_text))
    opts = plistlib.dumps({'h17g': {'EnableLowEffortCPAllocation': True}}, fmt=plistlib.FMT_BINARY)
    ns_opts = NSData.dataWithBytes_length_(opts, len(opts))
    desc = ANEInMemoryModelDescriptor.alloc().initWithNetworkText_weights_optionsPlist_isMILModel_(
        ns_net, NSDictionary.dictionary(), ns_opts, True)
    model = ANEInMemoryModel.alloc().initWithDesctiptor_(desc)
    model.purgeCompiledModel()
    model.saveModelFiles()
    lmp = model.localModelPath()

    # Copy net.plist -> model.mil
    if lmp and os.path.isdir(lmp):
        net_plist = os.path.join(lmp, 'net.plist')
        model_mil = os.path.join(lmp, 'model.mil')
        if os.path.exists(net_plist):
            shutil.copy2(net_plist, model_mil)

    # Compile
    t0 = time.perf_counter()
    try:
        ok = model.compileWithQoS_options_error_(0, NSDictionary.dictionary(), None)
    except Exception as e:
        print(f"  COMPILE EXCEPTION: {e}")
        return None, str(e), None, 0, 0

    compile_time = time.perf_counter() - t0

    if not ok:
        print(f"  COMPILE FAILED ({compile_time*1000:.0f}ms)")
        # Check files
        if lmp and os.path.isdir(lmp):
            for f in sorted(os.listdir(lmp)):
                fp = os.path.join(lmp, f)
                sz = os.path.getsize(fp) if os.path.isfile(fp) else 0
                print(f"    {f}: {sz} bytes")
        return None, "compile failed", lmp, 0, compile_time

    print(f"  COMPILE SUCCESS ({compile_time*1000:.0f}ms)")

    # Find .hwx size
    hwx_size = 0
    if lmp and os.path.isdir(lmp):
        for f in sorted(os.listdir(lmp)):
            fp = os.path.join(lmp, f)
            sz = os.path.getsize(fp) if os.path.isfile(fp) else 0
            print(f"    {f}: {sz} bytes")
            if f.endswith('.hwx'):
                hwx_size = sz

    # Load to get attributes
    ok = model.loadWithQoS_options_error_(0, NSDictionary.dictionary(), None)
    if not ok:
        print(f"  LOAD FAILED (can't get attributes)")
        return None, "load failed", lmp, hwx_size, compile_time

    attrs = model.modelAttributes()
    if attrs:
        try:
            desc_dict = attrs.get('ANEFModelDescription', {})
            procedures = desc_dict.get('ANEFModelProcedures', [])
            print(f"  PROCEDURES (dispatches): {len(procedures)}")
            for i, p in enumerate(procedures):
                if hasattr(p, 'keys'):
                    print(f"    Procedure {i}: {dict(p)}")
                else:
                    print(f"    Procedure {i}: {p}")
        except Exception as e:
            print(f"  Could not read procedures: {e}")

        # Also check NetworkStatusList
        try:
            ns_list = attrs.get('NetworkStatusList', [])
            print(f"  NetworkStatus entries: {len(ns_list)}")
            for i, ns in enumerate(ns_list):
                ins = ns.get('LiveInputList', [])
                outs = ns.get('LiveOutputList', [])
                print(f"    NS[{i}]: {len(ins)} inputs, {len(outs)} outputs")
        except Exception as e:
            print(f"  Could not read NetworkStatusList: {e}")
    else:
        print(f"  No model attributes returned")

    model.unloadWithQoS_error_(0, None)
    return model, attrs, lmp, hwx_size, compile_time


def make_weight_const(name, rows, cols):
    """Generate MIL const for a weight matrix with small random values."""
    # Use deterministic small values
    np.random.seed(hash(name) % 2**32)
    w = np.random.randn(rows, cols).astype(np.float16) * 0.01
    # Format as MIL tensor literal
    vals = ', '.join(f'{v:.4f}' for v in w.flatten())
    return f'tensor<fp16, [{rows}, {cols}]>({vals})'


def make_weight_const_bytes(name, shape):
    """Generate const declaration for weights."""
    shape_str = ', '.join(str(s) for s in shape)
    # For large weights, use default-initialized (zeros)
    # MIL accepts const() with just name and val
    np.random.seed(abs(hash(name)) % 2**32)
    total = 1
    for s in shape:
        total *= s
    if total <= 4096:
        w = np.random.randn(total).astype(np.float16) * 0.01
        vals = ', '.join(f'{float(v):.6f}' for v in w)
        return f'tensor<fp16, [{shape_str}]> {name} = const()[name = string("{name}"), val = tensor<fp16, [{shape_str}]>({vals})];'
    else:
        # For large tensors, use all 0.01 to keep MIL text manageable
        vals = ', '.join(['0.01'] * total)
        return f'tensor<fp16, [{shape_str}]> {name} = const()[name = string("{name}"), val = tensor<fp16, [{shape_str}]>({vals})];'


def make_scalar_const(name, val):
    return f'tensor<fp16, []> {name} = const()[name = string("{name}"), val = tensor<fp16, []>({val})];'


def build_mil(func_body, input_shape="[1, 64, 1, 1]", output_name="output"):
    """Build complete MIL program."""
    header = f'''program(1.3)
[buildInfo = {BUILD_INFO.decode()}]
{{
    func main<ios18>(tensor<fp16, {input_shape}> x) {{
'''
    footer = f'''
    }} -> ({output_name});
}}'''
    body = '\n'.join('        ' + line for line in func_body)
    return (header + body + footer).encode()


# ============================================================
# TEST 1: Single linear (matmul with weights)
# ============================================================
def test1_single_linear():
    DIM = 64
    lines = [
        make_weight_const_bytes("w", [DIM, DIM]),
        f'tensor<fp16, [1, {DIM}, 1, 1]> output = linear(x = x, weight = w)[name = string("output")];',
    ]
    mil = build_mil(lines)
    return compile_mil(mil, "TEST 1: Single linear (matmul) [64x64]")


# ============================================================
# TEST 2: Linear + SiLU
# ============================================================
def test2_linear_silu():
    DIM = 64
    lines = [
        make_weight_const_bytes("w", [DIM, DIM]),
        f'tensor<fp16, [1, {DIM}, 1, 1]> lin = linear(x = x, weight = w)[name = string("lin")];',
        f'tensor<fp16, [1, {DIM}, 1, 1]> output = silu(x = lin)[name = string("output")];',
    ]
    mil = build_mil(lines)
    return compile_mil(mil, "TEST 2: Linear + SiLU")


# ============================================================
# TEST 3: FFN (linear + SiLU + linear)
# ============================================================
def test3_ffn():
    DIM = 64
    lines = [
        make_weight_const_bytes("w1", [DIM, DIM]),
        make_weight_const_bytes("w2", [DIM, DIM]),
        f'tensor<fp16, [1, {DIM}, 1, 1]> h1 = linear(x = x, weight = w1)[name = string("h1")];',
        f'tensor<fp16, [1, {DIM}, 1, 1]> h2 = silu(x = h1)[name = string("h2")];',
        f'tensor<fp16, [1, {DIM}, 1, 1]> output = linear(x = h2, weight = w2)[name = string("output")];',
    ]
    mil = build_mil(lines)
    return compile_mil(mil, "TEST 3: FFN (linear->SiLU->linear)")


# ============================================================
# TEST 4: SwiGLU (gate + silu + up + mul + down)
# ============================================================
def test4_swiglu():
    DIM = 64
    HIDDEN = 128  # typical: hidden > dim
    lines = [
        make_weight_const_bytes("w_gate", [HIDDEN, DIM]),
        make_weight_const_bytes("w_up", [HIDDEN, DIM]),
        make_weight_const_bytes("w_down", [DIM, HIDDEN]),
        f'tensor<fp16, [1, {HIDDEN}, 1, 1]> gate = linear(x = x, weight = w_gate)[name = string("gate")];',
        f'tensor<fp16, [1, {HIDDEN}, 1, 1]> gate_act = silu(x = gate)[name = string("gate_act")];',
        f'tensor<fp16, [1, {HIDDEN}, 1, 1]> up = linear(x = x, weight = w_up)[name = string("up")];',
        f'tensor<fp16, [1, {HIDDEN}, 1, 1]> gated = mul(x = gate_act, y = up)[name = string("gated")];',
        f'tensor<fp16, [1, {DIM}, 1, 1]> output = linear(x = gated, weight = w_down)[name = string("output")];',
    ]
    mil = build_mil(lines)
    return compile_mil(mil, "TEST 4: SwiGLU (5 ops: gate+silu+up+mul+down)")


# ============================================================
# TEST 5: RMSNorm
# ============================================================
def test5_rmsnorm():
    DIM = 64
    lines = [
        # x^2
        f'tensor<fp16, [1, {DIM}, 1, 1]> x2 = mul(x = x, y = x)[name = string("x2")];',
        # mean(x^2) - reduce over channels
        f'tensor<fp16, [1, 1, 1, 1]> var = reduce_mean(x = x2, axes = [1], keep_dims = true)[name = string("var")];',
        # add eps
        make_scalar_const("eps", "1e-6"),
        f'tensor<fp16, [1, 1, 1, 1]> var_eps = add(x = var, y = eps)[name = string("var_eps")];',
        # rsqrt
        f'tensor<fp16, [1, 1, 1, 1]> inv_std = rsqrt(x = var_eps)[name = string("inv_std")];',
        # normalize
        f'tensor<fp16, [1, {DIM}, 1, 1]> output = mul(x = x, y = inv_std)[name = string("output")];',
    ]
    mil = build_mil(lines)
    result = compile_mil(mil, "TEST 5a: RMSNorm (mul->reduce_mean->rsqrt->mul)")

    if result[0] is None:
        # Try sqrt + real_div instead of rsqrt
        print("\n  Trying alternative: sqrt + real_div...")
        lines_alt = [
            f'tensor<fp16, [1, {DIM}, 1, 1]> x2 = mul(x = x, y = x)[name = string("x2")];',
            f'tensor<fp16, [1, 1, 1, 1]> var = reduce_mean(x = x2, axes = [1], keep_dims = true)[name = string("var")];',
            make_scalar_const("eps", "1e-6"),
            f'tensor<fp16, [1, 1, 1, 1]> var_eps = add(x = var, y = eps)[name = string("var_eps")];',
            f'tensor<fp16, [1, 1, 1, 1]> std = sqrt(x = var_eps)[name = string("std")];',
            f'tensor<fp16, [1, {DIM}, 1, 1]> output = real_div(x = x, y = std)[name = string("output")];',
        ]
        mil_alt = build_mil(lines_alt)
        result = compile_mil(mil_alt, "TEST 5b: RMSNorm (sqrt + real_div)")

    return result


# ============================================================
# TEST 6: Attention block
# ============================================================
def test6_attention():
    DIM = 64
    HEADS = 4
    HEAD_DIM = DIM // HEADS  # 16

    lines = [
        # Q, K, V projections
        make_weight_const_bytes("wq", [DIM, DIM]),
        make_weight_const_bytes("wk", [DIM, DIM]),
        make_weight_const_bytes("wv", [DIM, DIM]),
        make_weight_const_bytes("wo", [DIM, DIM]),

        # Q = linear(x, wq)
        f'tensor<fp16, [1, {DIM}, 1, 1]> q = linear(x = x, weight = wq)[name = string("q")];',
        # K = linear(x, wk)
        f'tensor<fp16, [1, {DIM}, 1, 1]> k = linear(x = x, weight = wk)[name = string("k")];',
        # V = linear(x, wv)
        f'tensor<fp16, [1, {DIM}, 1, 1]> v = linear(x = x, weight = wv)[name = string("v")];',

        # Reshape for attention: [1, DIM, 1, 1] -> [1, HEADS, HEAD_DIM, 1]
        # MIL reshape op
        f'tensor<int32, [4]> qkv_shape = const()[name = string("qkv_shape"), val = tensor<int32, [4]>(1, {HEADS}, {HEAD_DIM}, 1)];',
        f'tensor<fp16, [1, {HEADS}, {HEAD_DIM}, 1]> q_r = reshape(x = q, shape = qkv_shape)[name = string("q_r")];',
        f'tensor<fp16, [1, {HEADS}, {HEAD_DIM}, 1]> k_r = reshape(x = k, shape = qkv_shape)[name = string("k_r")];',
        f'tensor<fp16, [1, {HEADS}, {HEAD_DIM}, 1]> v_r = reshape(x = v, shape = qkv_shape)[name = string("v_r")];',

        # For single-token attention, Q@K^T is just dot product per head
        # attn_scores = matmul(q_r, k_r^T)
        f'tensor<fp16, [1, {HEADS}, 1, {HEAD_DIM}]> k_t = transpose(x = k_r, perm = [0, 1, 3, 2])[name = string("k_t")];',
        f'tensor<fp16, [1, {HEADS}, {HEAD_DIM}, {HEAD_DIM}]> scores = matmul(x = q_r, y = k_t)[name = string("scores")];',

        # Scale
        make_scalar_const("scale", f"{1.0/HEAD_DIM**0.5:.6f}"),
        f'tensor<fp16, [1, {HEADS}, {HEAD_DIM}, {HEAD_DIM}]> scaled = mul(x = scores, y = scale)[name = string("scaled")];',

        # Softmax
        f'tensor<fp16, [1, {HEADS}, {HEAD_DIM}, {HEAD_DIM}]> attn = softmax(x = scaled, axis = -1)[name = string("attn")];',

        # attn @ V
        f'tensor<fp16, [1, {HEADS}, {HEAD_DIM}, 1]> ctx = matmul(x = attn, y = v_r)[name = string("ctx")];',

        # Reshape back: [1, HEADS, HEAD_DIM, 1] -> [1, DIM, 1, 1]
        f'tensor<int32, [4]> out_shape = const()[name = string("out_shape"), val = tensor<int32, [4]>(1, {DIM}, 1, 1)];',
        f'tensor<fp16, [1, {DIM}, 1, 1]> ctx_flat = reshape(x = ctx, shape = out_shape)[name = string("ctx_flat")];',

        # Output projection
        f'tensor<fp16, [1, {DIM}, 1, 1]> output = linear(x = ctx_flat, weight = wo)[name = string("output")];',
    ]
    mil = build_mil(lines)
    return compile_mil(mil, "TEST 6: Full Attention (QKV proj + matmul + softmax + O proj)")


# ============================================================
# TEST 7: Full transformer layer (RMSNorm + Attention + Residual + RMSNorm + SwiGLU + Residual)
# ============================================================
def test7_full_layer():
    DIM = 64
    HIDDEN = 128
    HEADS = 4
    HEAD_DIM = DIM // HEADS

    lines = [
        # Weights
        make_weight_const_bytes("wq", [DIM, DIM]),
        make_weight_const_bytes("wk", [DIM, DIM]),
        make_weight_const_bytes("wv", [DIM, DIM]),
        make_weight_const_bytes("wo", [DIM, DIM]),
        make_weight_const_bytes("w_gate", [HIDDEN, DIM]),
        make_weight_const_bytes("w_up", [HIDDEN, DIM]),
        make_weight_const_bytes("w_down", [DIM, HIDDEN]),

        # === RMSNorm 1 ===
        f'tensor<fp16, [1, {DIM}, 1, 1]> x2 = mul(x = x, y = x)[name = string("x2")];',
        f'tensor<fp16, [1, 1, 1, 1]> var1 = reduce_mean(x = x2, axes = [1], keep_dims = true)[name = string("var1")];',
        make_scalar_const("eps1", "1e-6"),
        f'tensor<fp16, [1, 1, 1, 1]> var1_eps = add(x = var1, y = eps1)[name = string("var1_eps")];',
        f'tensor<fp16, [1, 1, 1, 1]> inv_std1 = rsqrt(x = var1_eps)[name = string("inv_std1")];',
        f'tensor<fp16, [1, {DIM}, 1, 1]> norm1 = mul(x = x, y = inv_std1)[name = string("norm1")];',

        # === Attention ===
        f'tensor<fp16, [1, {DIM}, 1, 1]> q = linear(x = norm1, weight = wq)[name = string("q")];',
        f'tensor<fp16, [1, {DIM}, 1, 1]> k = linear(x = norm1, weight = wk)[name = string("k")];',
        f'tensor<fp16, [1, {DIM}, 1, 1]> v = linear(x = norm1, weight = wv)[name = string("v")];',

        f'tensor<int32, [4]> qkv_shape = const()[name = string("qkv_shape"), val = tensor<int32, [4]>(1, {HEADS}, {HEAD_DIM}, 1)];',
        f'tensor<fp16, [1, {HEADS}, {HEAD_DIM}, 1]> q_r = reshape(x = q, shape = qkv_shape)[name = string("q_r")];',
        f'tensor<fp16, [1, {HEADS}, {HEAD_DIM}, 1]> k_r = reshape(x = k, shape = qkv_shape)[name = string("k_r")];',
        f'tensor<fp16, [1, {HEADS}, {HEAD_DIM}, 1]> v_r = reshape(x = v, shape = qkv_shape)[name = string("v_r")];',

        f'tensor<fp16, [1, {HEADS}, 1, {HEAD_DIM}]> k_t = transpose(x = k_r, perm = [0, 1, 3, 2])[name = string("k_t")];',
        f'tensor<fp16, [1, {HEADS}, {HEAD_DIM}, {HEAD_DIM}]> scores = matmul(x = q_r, y = k_t)[name = string("scores")];',
        make_scalar_const("scale", f"{1.0/HEAD_DIM**0.5:.6f}"),
        f'tensor<fp16, [1, {HEADS}, {HEAD_DIM}, {HEAD_DIM}]> scaled = mul(x = scores, y = scale)[name = string("scaled")];',
        f'tensor<fp16, [1, {HEADS}, {HEAD_DIM}, {HEAD_DIM}]> attn_w = softmax(x = scaled, axis = -1)[name = string("attn_w")];',
        f'tensor<fp16, [1, {HEADS}, {HEAD_DIM}, 1]> ctx = matmul(x = attn_w, y = v_r)[name = string("ctx")];',

        f'tensor<int32, [4]> out_shape = const()[name = string("out_shape"), val = tensor<int32, [4]>(1, {DIM}, 1, 1)];',
        f'tensor<fp16, [1, {DIM}, 1, 1]> ctx_flat = reshape(x = ctx, shape = out_shape)[name = string("ctx_flat")];',
        f'tensor<fp16, [1, {DIM}, 1, 1]> attn_out = linear(x = ctx_flat, weight = wo)[name = string("attn_out")];',

        # === Residual 1 ===
        f'tensor<fp16, [1, {DIM}, 1, 1]> res1 = add(x = x, y = attn_out)[name = string("res1")];',

        # === RMSNorm 2 ===
        f'tensor<fp16, [1, {DIM}, 1, 1]> res1_sq = mul(x = res1, y = res1)[name = string("res1_sq")];',
        f'tensor<fp16, [1, 1, 1, 1]> var2 = reduce_mean(x = res1_sq, axes = [1], keep_dims = true)[name = string("var2")];',
        make_scalar_const("eps2", "1e-6"),
        f'tensor<fp16, [1, 1, 1, 1]> var2_eps = add(x = var2, y = eps2)[name = string("var2_eps")];',
        f'tensor<fp16, [1, 1, 1, 1]> inv_std2 = rsqrt(x = var2_eps)[name = string("inv_std2")];',
        f'tensor<fp16, [1, {DIM}, 1, 1]> norm2 = mul(x = res1, y = inv_std2)[name = string("norm2")];',

        # === SwiGLU FFN ===
        f'tensor<fp16, [1, {HIDDEN}, 1, 1]> gate = linear(x = norm2, weight = w_gate)[name = string("gate")];',
        f'tensor<fp16, [1, {HIDDEN}, 1, 1]> gate_act = silu(x = gate)[name = string("gate_act")];',
        f'tensor<fp16, [1, {HIDDEN}, 1, 1]> up = linear(x = norm2, weight = w_up)[name = string("up")];',
        f'tensor<fp16, [1, {HIDDEN}, 1, 1]> gated = mul(x = gate_act, y = up)[name = string("gated")];',
        f'tensor<fp16, [1, {DIM}, 1, 1]> ffn_out = linear(x = gated, weight = w_down)[name = string("ffn_out")];',

        # === Residual 2 ===
        f'tensor<fp16, [1, {DIM}, 1, 1]> output = add(x = res1, y = ffn_out)[name = string("output")];',
    ]
    mil = build_mil(lines)
    return compile_mil(mil, "TEST 7: Full Transformer Layer (RMSNorm+Attn+Res+RMSNorm+SwiGLU+Res)")


# ============================================================
# TEST 8: Two transformer layers
# ============================================================
def test8_two_layers():
    DIM = 64
    HIDDEN = 128
    HEADS = 4
    HEAD_DIM = DIM // HEADS

    def make_layer(prefix, input_var):
        """Generate one transformer layer, return (lines, output_var_name)."""
        p = prefix
        lines = [
            make_weight_const_bytes(f"{p}_wq", [DIM, DIM]),
            make_weight_const_bytes(f"{p}_wk", [DIM, DIM]),
            make_weight_const_bytes(f"{p}_wv", [DIM, DIM]),
            make_weight_const_bytes(f"{p}_wo", [DIM, DIM]),
            make_weight_const_bytes(f"{p}_wg", [HIDDEN, DIM]),
            make_weight_const_bytes(f"{p}_wu", [HIDDEN, DIM]),
            make_weight_const_bytes(f"{p}_wd", [DIM, HIDDEN]),

            # RMSNorm
            f'tensor<fp16, [1, {DIM}, 1, 1]> {p}_x2 = mul(x = {input_var}, y = {input_var})[name = string("{p}_x2")];',
            f'tensor<fp16, [1, 1, 1, 1]> {p}_var = reduce_mean(x = {p}_x2, axes = [1], keep_dims = true)[name = string("{p}_var")];',
            make_scalar_const(f"{p}_eps1", "1e-6"),
            f'tensor<fp16, [1, 1, 1, 1]> {p}_ve = add(x = {p}_var, y = {p}_eps1)[name = string("{p}_ve")];',
            f'tensor<fp16, [1, 1, 1, 1]> {p}_is = rsqrt(x = {p}_ve)[name = string("{p}_is")];',
            f'tensor<fp16, [1, {DIM}, 1, 1]> {p}_n1 = mul(x = {input_var}, y = {p}_is)[name = string("{p}_n1")];',

            # Attention
            f'tensor<fp16, [1, {DIM}, 1, 1]> {p}_q = linear(x = {p}_n1, weight = {p}_wq)[name = string("{p}_q")];',
            f'tensor<fp16, [1, {DIM}, 1, 1]> {p}_k = linear(x = {p}_n1, weight = {p}_wk)[name = string("{p}_k")];',
            f'tensor<fp16, [1, {DIM}, 1, 1]> {p}_v = linear(x = {p}_n1, weight = {p}_wv)[name = string("{p}_v")];',

            f'tensor<int32, [4]> {p}_qs = const()[name = string("{p}_qs"), val = tensor<int32, [4]>(1, {HEADS}, {HEAD_DIM}, 1)];',
            f'tensor<fp16, [1, {HEADS}, {HEAD_DIM}, 1]> {p}_qr = reshape(x = {p}_q, shape = {p}_qs)[name = string("{p}_qr")];',
            f'tensor<fp16, [1, {HEADS}, {HEAD_DIM}, 1]> {p}_kr = reshape(x = {p}_k, shape = {p}_qs)[name = string("{p}_kr")];',
            f'tensor<fp16, [1, {HEADS}, {HEAD_DIM}, 1]> {p}_vr = reshape(x = {p}_v, shape = {p}_qs)[name = string("{p}_vr")];',

            f'tensor<fp16, [1, {HEADS}, 1, {HEAD_DIM}]> {p}_kt = transpose(x = {p}_kr, perm = [0, 1, 3, 2])[name = string("{p}_kt")];',
            f'tensor<fp16, [1, {HEADS}, {HEAD_DIM}, {HEAD_DIM}]> {p}_sc = matmul(x = {p}_qr, y = {p}_kt)[name = string("{p}_sc")];',
            make_scalar_const(f"{p}_scl", f"{1.0/HEAD_DIM**0.5:.6f}"),
            f'tensor<fp16, [1, {HEADS}, {HEAD_DIM}, {HEAD_DIM}]> {p}_ss = mul(x = {p}_sc, y = {p}_scl)[name = string("{p}_ss")];',
            f'tensor<fp16, [1, {HEADS}, {HEAD_DIM}, {HEAD_DIM}]> {p}_aw = softmax(x = {p}_ss, axis = -1)[name = string("{p}_aw")];',
            f'tensor<fp16, [1, {HEADS}, {HEAD_DIM}, 1]> {p}_ct = matmul(x = {p}_aw, y = {p}_vr)[name = string("{p}_ct")];',

            f'tensor<int32, [4]> {p}_os = const()[name = string("{p}_os"), val = tensor<int32, [4]>(1, {DIM}, 1, 1)];',
            f'tensor<fp16, [1, {DIM}, 1, 1]> {p}_cf = reshape(x = {p}_ct, shape = {p}_os)[name = string("{p}_cf")];',
            f'tensor<fp16, [1, {DIM}, 1, 1]> {p}_ao = linear(x = {p}_cf, weight = {p}_wo)[name = string("{p}_ao")];',

            # Residual 1
            f'tensor<fp16, [1, {DIM}, 1, 1]> {p}_r1 = add(x = {input_var}, y = {p}_ao)[name = string("{p}_r1")];',

            # RMSNorm 2
            f'tensor<fp16, [1, {DIM}, 1, 1]> {p}_r1s = mul(x = {p}_r1, y = {p}_r1)[name = string("{p}_r1s")];',
            f'tensor<fp16, [1, 1, 1, 1]> {p}_v2 = reduce_mean(x = {p}_r1s, axes = [1], keep_dims = true)[name = string("{p}_v2")];',
            make_scalar_const(f"{p}_eps2", "1e-6"),
            f'tensor<fp16, [1, 1, 1, 1]> {p}_v2e = add(x = {p}_v2, y = {p}_eps2)[name = string("{p}_v2e")];',
            f'tensor<fp16, [1, 1, 1, 1]> {p}_is2 = rsqrt(x = {p}_v2e)[name = string("{p}_is2")];',
            f'tensor<fp16, [1, {DIM}, 1, 1]> {p}_n2 = mul(x = {p}_r1, y = {p}_is2)[name = string("{p}_n2")];',

            # SwiGLU
            f'tensor<fp16, [1, {HIDDEN}, 1, 1]> {p}_g = linear(x = {p}_n2, weight = {p}_wg)[name = string("{p}_g")];',
            f'tensor<fp16, [1, {HIDDEN}, 1, 1]> {p}_ga = silu(x = {p}_g)[name = string("{p}_ga")];',
            f'tensor<fp16, [1, {HIDDEN}, 1, 1]> {p}_u = linear(x = {p}_n2, weight = {p}_wu)[name = string("{p}_u")];',
            f'tensor<fp16, [1, {HIDDEN}, 1, 1]> {p}_gd = mul(x = {p}_ga, y = {p}_u)[name = string("{p}_gd")];',
            f'tensor<fp16, [1, {DIM}, 1, 1]> {p}_fo = linear(x = {p}_gd, weight = {p}_wd)[name = string("{p}_fo")];',

            # Residual 2
            f'tensor<fp16, [1, {DIM}, 1, 1]> {p}_out = add(x = {p}_r1, y = {p}_fo)[name = string("{p}_out")];',
        ]
        return lines, f'{p}_out'

    l1_lines, l1_out = make_layer("l1", "x")
    l2_lines, l2_out = make_layer("l2", l1_out)

    # Rename final output
    all_lines = l1_lines + l2_lines
    # The last line should output the final var
    all_lines.append(
        f'tensor<fp16, [1, {DIM}, 1, 1]> output = identity(x = {l2_out})[name = string("output")];'
    )

    mil = build_mil(all_lines)
    return compile_mil(mil, "TEST 8: Two Transformer Layers")


# ============================================================
# MAIN
# ============================================================
def main():
    print("MIL IR FUSION BOUNDARY TEST")
    print("How large a graph can aned compile into a single ANE dispatch?")
    print("=" * 70)

    results = {}

    # Run tests sequentially
    for test_fn, name in [
        (test1_single_linear, "1_linear"),
        (test2_linear_silu, "2_linear_silu"),
        (test3_ffn, "3_ffn"),
        (test4_swiglu, "4_swiglu"),
        (test5_rmsnorm, "5_rmsnorm"),
        (test6_attention, "6_attention"),
        (test7_full_layer, "7_full_layer"),
        (test8_two_layers, "8_two_layers"),
    ]:
        try:
            model, info, lmp, hwx_size, compile_time = test_fn()
            compiled = model is not None
            results[name] = {
                'compiled': compiled,
                'hwx_size': hwx_size,
                'compile_time_ms': compile_time * 1000,
                'info': info if not compiled else 'OK',
            }
            if compiled and hasattr(info, 'get'):
                try:
                    procs = info.get('ANEFModelDescription', {}).get('ANEFModelProcedures', [])
                    results[name]['procedures'] = len(procs)
                except:
                    pass
        except Exception as e:
            print(f"\n  EXCEPTION in {name}: {e}")
            import traceback
            traceback.print_exc()
            results[name] = {'compiled': False, 'info': str(e), 'hwx_size': 0, 'compile_time_ms': 0}

    # Summary
    print(f"\n{'='*70}")
    print("FUSION BOUNDARY SUMMARY")
    print(f"{'='*70}")
    print(f"{'Test':<25} {'Compiled':<10} {'Procedures':<12} {'HWX Size':<12} {'Compile ms':<12}")
    print("-" * 70)
    for name, r in results.items():
        compiled = "YES" if r['compiled'] else "NO"
        procs = str(r.get('procedures', '?'))
        hwx = f"{r['hwx_size']:,}" if r['hwx_size'] else "-"
        ct = f"{r['compile_time_ms']:.0f}"
        print(f"{name:<25} {compiled:<10} {procs:<12} {hwx:<12} {ct:<12}")
        if not r['compiled']:
            info = r.get('info', '')
            if isinstance(info, str) and info != 'OK':
                print(f"  -> {info[:100]}")


if __name__ == "__main__":
    main()
