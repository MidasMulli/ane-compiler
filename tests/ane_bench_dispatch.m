// ane_bench_dispatch.m — A/B dispatch mode benchmark within one process
//
// Tests request reuse + cacheInference flags in a controlled environment.
// Runs the same generation loop multiple times with different dispatch strategies.
// All modes share the same compiled models, IOSurfaces, and KV caches (reset between runs).
//
// Build:
//   xcrun clang -O2 -framework Foundation -framework IOSurface -framework Accelerate \
//     -fobjc-arc -o ane_bench_dispatch ane_bench_dispatch.m

#import <Foundation/Foundation.h>
#import <Accelerate/Accelerate.h>
#import <IOSurface/IOSurface.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>
#import <signal.h>
#import <mach/mach_time.h>

#define N_LAYERS     12
#define N_HEADS      12
#define DIM          768
#define HEAD_DIM     64
#define FFN_DIM      3072
#define VOCAB_SIZE   50257
#define MAX_SEQ      1024
#define LN_EPS       1e-5f

static Class _Cl, _Mo, _Rq, _IO;
static void loadFW(void) {
    dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);
    _Cl = NSClassFromString(@"_ANEClient");
    _Mo = NSClassFromString(@"_ANEModel");
    _Rq = NSClassFromString(@"_ANERequest");
    _IO = NSClassFromString(@"_ANEIOSurfaceObject");
}

typedef struct {
    id model;
    int inCh, outCh;
    uint32_t inBS, outBS, inPS, outPS;
    IOSurfaceRef inSurf, outSurf;
    id inObj, outObj;
    id request;  // Pre-built request (reused)
} OpEntry;

typedef struct {
    float *wte, *wpe;
    float *ln1_w[N_LAYERS], *ln1_b[N_LAYERS];
    float *ln2_w[N_LAYERS], *ln2_b[N_LAYERS];
    float *ln_f_w, *ln_f_b;
} CPUWeights;

typedef struct {
    float *k_cache, *v_cache;
    int len;
} KVCache;

static mach_timebase_info_data_t tb;
static inline double ns_to_ms(uint64_t ns) {
    return (double)(ns * tb.numer / tb.denom) / 1e6;
}

static void layernorm(const float *x, const float *w, const float *b, float *out, int n) {
    float mean = 0, var = 0;
    vDSP_meanv(x, 1, &mean, n);
    float neg_mean = -mean;
    vDSP_vsadd(x, 1, &neg_mean, out, 1, n);
    vDSP_vsq(out, 1, out, 1, n);
    vDSP_meanv(out, 1, &var, n);
    float inv_std = 1.0f / sqrtf(var + LN_EPS);
    vDSP_vsadd(x, 1, &neg_mean, out, 1, n);
    vDSP_vsmul(out, 1, &inv_std, out, 1, n);
    vDSP_vma(out, 1, w, 1, b, 1, out, 1, n);
}

static void softmax_f32(float *x, int n) {
    float max_val;
    vDSP_maxv(x, 1, &max_val, n);
    float neg_max = -max_val;
    vDSP_vsadd(x, 1, &neg_max, x, 1, n);
    int n_int = n;
    vvexpf(x, x, &n_int);
    float sum = 0;
    vDSP_sve(x, 1, &sum, n);
    float inv_sum = 1.0f / sum;
    vDSP_vsmul(x, 1, &inv_sum, x, 1, n);
}

static void fp16_to_fp32(const uint16_t *in, float *out, int n) {
    vImage_Buffer src = {(void*)in, 1, (vImagePixelCount)n, n * 2};
    vImage_Buffer dst = {out, 1, (vImagePixelCount)n, n * 4};
    vImageConvert_Planar16FtoPlanarF(&src, &dst, 0);
}

static void fp32_to_fp16(const float *in, uint16_t *out, int n) {
    vImage_Buffer src = {(void*)in, 1, (vImagePixelCount)n, n * 4};
    vImage_Buffer dst = {out, 1, (vImagePixelCount)n, n * 2};
    vImageConvert_PlanarFtoPlanar16F(&src, &dst, 0);
}

static id ane_client = nil;

// Dispatch mode A: original (fresh request, map+eval+unmap, cacheInference:NO)
static void dispatch_original(OpEntry *op, const uint16_t *input, uint16_t *output) {
    IOSurfaceLock(op->inSurf, 0, NULL);
    void *inBase = IOSurfaceGetBaseAddress(op->inSurf);
    memset(inBase, 0, op->inBS);
    for (int j = 0; j < op->inCh; j++)
        memcpy((uint8_t*)inBase + j * op->inPS, &input[j], 2);
    IOSurfaceUnlock(op->inSurf, 0, NULL);

    // Fresh request every dispatch (matches original ane_generate.m)
    id req = ((id (*)(id, SEL, id, id, id, id, id, id, id, id, id))objc_msgSend)(
        [_Rq alloc],
        NSSelectorFromString(@"initWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:sharedEvents:transactionHandle:"),
        @[op->inObj], @[@0], @[op->outObj], @[@0], nil, nil, @(0), nil, nil);

    ((BOOL (*)(id, SEL, id, id, BOOL, id*))objc_msgSend)(
        ane_client, NSSelectorFromString(@"mapIOSurfacesWithModel:request:cacheInference:error:"),
        op->model, req, NO, nil);
    ((BOOL (*)(id, SEL, id, id, id, int, id*))objc_msgSend)(
        ane_client, NSSelectorFromString(@"doEvaluateDirectWithModel:options:request:qos:error:"),
        op->model, @{}, req, 21, nil);

    IOSurfaceLock(op->outSurf, kIOSurfaceLockReadOnly, NULL);
    void *outBase = IOSurfaceGetBaseAddress(op->outSurf);
    for (int j = 0; j < op->outCh; j++)
        memcpy(&output[j], (uint8_t*)outBase + j * op->outPS, 2);
    IOSurfaceUnlock(op->outSurf, kIOSurfaceLockReadOnly, NULL);

    ((void (*)(id, SEL, id, id))objc_msgSend)(
        ane_client, NSSelectorFromString(@"unmapIOSurfacesWithModel:request:"), op->model, req);
}

// Dispatch mode B: request reuse (pre-built request, map+eval+unmap, cacheInference:NO)
static void dispatch_req_reuse(OpEntry *op, const uint16_t *input, uint16_t *output) {
    IOSurfaceLock(op->inSurf, 0, NULL);
    void *inBase = IOSurfaceGetBaseAddress(op->inSurf);
    memset(inBase, 0, op->inBS);
    for (int j = 0; j < op->inCh; j++)
        memcpy((uint8_t*)inBase + j * op->inPS, &input[j], 2);
    IOSurfaceUnlock(op->inSurf, 0, NULL);

    ((BOOL (*)(id, SEL, id, id, BOOL, id*))objc_msgSend)(
        ane_client, NSSelectorFromString(@"mapIOSurfacesWithModel:request:cacheInference:error:"),
        op->model, op->request, NO, nil);
    ((BOOL (*)(id, SEL, id, id, id, int, id*))objc_msgSend)(
        ane_client, NSSelectorFromString(@"doEvaluateDirectWithModel:options:request:qos:error:"),
        op->model, @{}, op->request, 21, nil);

    IOSurfaceLock(op->outSurf, kIOSurfaceLockReadOnly, NULL);
    void *outBase = IOSurfaceGetBaseAddress(op->outSurf);
    for (int j = 0; j < op->outCh; j++)
        memcpy(&output[j], (uint8_t*)outBase + j * op->outPS, 2);
    IOSurfaceUnlock(op->outSurf, kIOSurfaceLockReadOnly, NULL);

    ((void (*)(id, SEL, id, id))objc_msgSend)(
        ane_client, NSSelectorFromString(@"unmapIOSurfacesWithModel:request:"), op->model, op->request);
}

// Dispatch mode C: request reuse + cacheInference:YES
static void dispatch_cache_yes(OpEntry *op, const uint16_t *input, uint16_t *output) {
    IOSurfaceLock(op->inSurf, 0, NULL);
    void *inBase = IOSurfaceGetBaseAddress(op->inSurf);
    memset(inBase, 0, op->inBS);
    for (int j = 0; j < op->inCh; j++)
        memcpy((uint8_t*)inBase + j * op->inPS, &input[j], 2);
    IOSurfaceUnlock(op->inSurf, 0, NULL);

    ((BOOL (*)(id, SEL, id, id, BOOL, id*))objc_msgSend)(
        ane_client, NSSelectorFromString(@"mapIOSurfacesWithModel:request:cacheInference:error:"),
        op->model, op->request, YES, nil);
    ((BOOL (*)(id, SEL, id, id, id, int, id*))objc_msgSend)(
        ane_client, NSSelectorFromString(@"doEvaluateDirectWithModel:options:request:qos:error:"),
        op->model, @{}, op->request, 21, nil);

    IOSurfaceLock(op->outSurf, kIOSurfaceLockReadOnly, NULL);
    void *outBase = IOSurfaceGetBaseAddress(op->outSurf);
    for (int j = 0; j < op->outCh; j++)
        memcpy(&output[j], (uint8_t*)outBase + j * op->outPS, 2);
    IOSurfaceUnlock(op->outSurf, kIOSurfaceLockReadOnly, NULL);

    ((void (*)(id, SEL, id, id))objc_msgSend)(
        ane_client, NSSelectorFromString(@"unmapIOSurfacesWithModel:request:"), op->model, op->request);
}

// Dispatch mode D: request reuse + cacheInference:YES + NO unmap
static void dispatch_no_unmap(OpEntry *op, const uint16_t *input, uint16_t *output) {
    IOSurfaceLock(op->inSurf, 0, NULL);
    void *inBase = IOSurfaceGetBaseAddress(op->inSurf);
    memset(inBase, 0, op->inBS);
    for (int j = 0; j < op->inCh; j++)
        memcpy((uint8_t*)inBase + j * op->inPS, &input[j], 2);
    IOSurfaceUnlock(op->inSurf, 0, NULL);

    ((BOOL (*)(id, SEL, id, id, BOOL, id*))objc_msgSend)(
        ane_client, NSSelectorFromString(@"mapIOSurfacesWithModel:request:cacheInference:error:"),
        op->model, op->request, YES, nil);
    ((BOOL (*)(id, SEL, id, id, id, int, id*))objc_msgSend)(
        ane_client, NSSelectorFromString(@"doEvaluateDirectWithModel:options:request:qos:error:"),
        op->model, @{}, op->request, 21, nil);

    IOSurfaceLock(op->outSurf, kIOSurfaceLockReadOnly, NULL);
    void *outBase = IOSurfaceGetBaseAddress(op->outSurf);
    for (int j = 0; j < op->outCh; j++)
        memcpy(&output[j], (uint8_t*)outBase + j * op->outPS, 2);
    IOSurfaceUnlock(op->outSurf, kIOSurfaceLockReadOnly, NULL);

    // No unmap
}

// Dispatch mode E: eval only (pre-mapped IOSurfaces)
static void dispatch_eval_only(OpEntry *op, const uint16_t *input, uint16_t *output) {
    IOSurfaceLock(op->inSurf, 0, NULL);
    void *inBase = IOSurfaceGetBaseAddress(op->inSurf);
    memset(inBase, 0, op->inBS);
    for (int j = 0; j < op->inCh; j++)
        memcpy((uint8_t*)inBase + j * op->inPS, &input[j], 2);
    IOSurfaceUnlock(op->inSurf, 0, NULL);

    ((BOOL (*)(id, SEL, id, id, id, int, id*))objc_msgSend)(
        ane_client, NSSelectorFromString(@"doEvaluateDirectWithModel:options:request:qos:error:"),
        op->model, @{}, op->request, 21, nil);

    IOSurfaceLock(op->outSurf, kIOSurfaceLockReadOnly, NULL);
    void *outBase = IOSurfaceGetBaseAddress(op->outSurf);
    for (int j = 0; j < op->outCh; j++)
        memcpy(&output[j], (uint8_t*)outBase + j * op->outPS, 2);
    IOSurfaceUnlock(op->outSurf, kIOSurfaceLockReadOnly, NULL);
}

// Run generation loop with a given dispatch function, return elapsed ms
static double run_generation(
    OpEntry *ops, CPUWeights *W, KVCache *kv,
    int *prompt_tokens, int n_prompt, int n_gen,
    void (*dispatch_fn)(OpEntry *, const uint16_t *, uint16_t *),
    int (^opIdx)(NSString *),
    int *out_tokens)
{
    float *x_f32 = malloc(DIM * sizeof(float));
    float *ln_out = malloc(DIM * sizeof(float));
    float *attn_out_f32 = malloc(DIM * sizeof(float));
    float *scores = malloc(MAX_SEQ * sizeof(float));
    uint16_t *x_fp16 = malloc(DIM * 2);
    uint16_t *qkv_fp16 = malloc(2304 * 2);
    uint16_t *o_fp16 = malloc(DIM * 2);
    uint16_t *ffn_fp16 = malloc(DIM * 2);
    uint16_t *logits_fp16 = malloc(VOCAB_SIZE * 2);
    float *logits_f32 = malloc(VOCAB_SIZE * sizeof(float));

    // Reset KV caches
    for (int i = 0; i < N_LAYERS; i++) {
        memset(kv[i].k_cache, 0, MAX_SEQ * N_HEADS * HEAD_DIM * sizeof(float));
        memset(kv[i].v_cache, 0, MAX_SEQ * N_HEADS * HEAD_DIM * sizeof(float));
        kv[i].len = 0;
    }

    int max_tokens = n_prompt + n_gen;
    int *tokens = malloc(max_tokens * sizeof(int));
    memcpy(tokens, prompt_tokens, n_prompt * sizeof(int));
    int total_tokens = n_prompt;
    int total_steps = n_prompt + n_gen;

    uint64_t t_start = mach_absolute_time();

    for (int step = 0; step < total_steps; step++) {
        int pos, tok;
        BOOL is_generate = (step >= n_prompt);
        if (!is_generate) { pos = step; tok = tokens[pos]; }
        else { pos = total_tokens - 1; tok = tokens[pos]; }

        for (int d = 0; d < DIM; d++)
            x_f32[d] = W->wte[tok * DIM + d] + W->wpe[pos * DIM + d];
        fp32_to_fp16(x_f32, x_fp16, DIM);

        for (int li = 0; li < N_LAYERS; li++) {
            fp16_to_fp32(x_fp16, x_f32, DIM);
            layernorm(x_f32, W->ln1_w[li], W->ln1_b[li], ln_out, DIM);
            uint16_t ln1_fp16[DIM];
            fp32_to_fp16(ln_out, ln1_fp16, DIM);

            int qkv_idx = opIdx([NSString stringWithFormat:@"L%d_qkv_proj", li]);
            dispatch_fn(&ops[qkv_idx], ln1_fp16, qkv_fp16);

            float q_f32[DIM], k_f32[DIM], v_f32[DIM];
            fp16_to_fp32(qkv_fp16, q_f32, DIM);
            fp16_to_fp32(qkv_fp16 + DIM, k_f32, DIM);
            fp16_to_fp32(qkv_fp16 + 2*DIM, v_f32, DIM);

            int seq_pos = kv[li].len;
            memcpy(&kv[li].k_cache[seq_pos * N_HEADS * HEAD_DIM], k_f32, DIM * sizeof(float));
            memcpy(&kv[li].v_cache[seq_pos * N_HEADS * HEAD_DIM], v_f32, DIM * sizeof(float));
            kv[li].len++;
            int seq_len = kv[li].len;

            float scale = 1.0f / sqrtf((float)HEAD_DIM);
            memset(attn_out_f32, 0, DIM * sizeof(float));
            for (int h = 0; h < N_HEADS; h++) {
                float *q_h = &q_f32[h * HEAD_DIM];
                for (int s = 0; s < seq_len; s++) {
                    float dot = 0;
                    vDSP_dotpr(q_h, 1,
                               &kv[li].k_cache[s * N_HEADS * HEAD_DIM + h * HEAD_DIM], 1,
                               &dot, HEAD_DIM);
                    scores[s] = dot * scale;
                }
                softmax_f32(scores, seq_len);
                float *out_h = &attn_out_f32[h * HEAD_DIM];
                memset(out_h, 0, HEAD_DIM * sizeof(float));
                for (int s = 0; s < seq_len; s++) {
                    float w = scores[s];
                    float *v_s = &kv[li].v_cache[s * N_HEADS * HEAD_DIM + h * HEAD_DIM];
                    vDSP_vsma(v_s, 1, &w, out_h, 1, out_h, 1, HEAD_DIM);
                }
            }

            uint16_t attn_fp16[DIM];
            fp32_to_fp16(attn_out_f32, attn_fp16, DIM);
            int o_idx = opIdx([NSString stringWithFormat:@"L%d_o_proj", li]);
            dispatch_fn(&ops[o_idx], attn_fp16, o_fp16);

            fp16_to_fp32(x_fp16, x_f32, DIM);
            float o_f32[DIM];
            fp16_to_fp32(o_fp16, o_f32, DIM);
            vDSP_vadd(x_f32, 1, o_f32, 1, x_f32, 1, DIM);

            layernorm(x_f32, W->ln2_w[li], W->ln2_b[li], ln_out, DIM);
            uint16_t ln2_fp16[DIM];
            fp32_to_fp16(ln_out, ln2_fp16, DIM);

            int ffn_idx = opIdx([NSString stringWithFormat:@"L%d_fused_ffn", li]);
            dispatch_fn(&ops[ffn_idx], ln2_fp16, ffn_fp16);

            float ffn_f32[DIM];
            fp16_to_fp32(ffn_fp16, ffn_f32, DIM);
            vDSP_vadd(x_f32, 1, ffn_f32, 1, x_f32, 1, DIM);
            fp32_to_fp16(x_f32, x_fp16, DIM);
        }

        fp16_to_fp32(x_fp16, x_f32, DIM);
        layernorm(x_f32, W->ln_f_w, W->ln_f_b, ln_out, DIM);
        uint16_t lnf_fp16[DIM];
        fp32_to_fp16(ln_out, lnf_fp16, DIM);
        int lm_idx = opIdx(@"lm_head");
        dispatch_fn(&ops[lm_idx], lnf_fp16, logits_fp16);

        fp16_to_fp32(logits_fp16, logits_f32, VOCAB_SIZE);
        vDSP_Length max_idx = 0;
        float max_val = 0;
        vDSP_maxvi(logits_f32, 1, &max_val, &max_idx, VOCAB_SIZE);

        if (is_generate) {
            tokens[total_tokens] = (int)max_idx;
            total_tokens++;
            if (out_tokens) out_tokens[step - n_prompt] = (int)max_idx;
        } else if (step == n_prompt - 1) {
            t_start = mach_absolute_time();
        }
    }

    uint64_t t_end = mach_absolute_time();
    double elapsed_ms = ns_to_ms(t_end - t_start);

    free(x_f32); free(ln_out); free(attn_out_f32); free(scores);
    free(x_fp16); free(qkv_fp16); free(o_fp16); free(ffn_fp16);
    free(logits_fp16); free(logits_f32); free(tokens);
    return elapsed_ms;
}

int main(int argc, char *argv[]) {
    @autoreleasepool {
        signal(SIGSEGV, SIG_IGN);
        mach_timebase_info(&tb);
        loadFW();

        if (argc < 4) {
            fprintf(stderr, "Usage: %s <manifest.txt> <weights.bin> <n_tokens> [prompt_tokens...]\n", argv[0]);
            return 1;
        }

        const char *manifest_path = argv[1];
        const char *weights_path = argv[2];
        int n_gen = atoi(argv[3]);
        int n_prompt = argc - 4;
        int *prompt_tokens = malloc(n_prompt * sizeof(int));
        for (int i = 0; i < n_prompt; i++)
            prompt_tokens[i] = atoi(argv[4 + i]);

        // Load manifest
        NSString *manifest = [NSString stringWithContentsOfFile:
            [NSString stringWithUTF8String:manifest_path]
            encoding:NSUTF8StringEncoding error:nil];
        NSArray *lines = [manifest componentsSeparatedByString:@"\n"];

        int nOps = 0;
        OpEntry *ops = calloc(lines.count, sizeof(OpEntry));
        NSMutableArray *opPaths = [NSMutableArray array];
        NSMutableDictionary *opMap = [NSMutableDictionary dictionary];

        for (NSString *line in lines) {
            NSArray *parts = [line componentsSeparatedByString:@" "];
            if (parts.count < 4) continue;
            ops[nOps].inCh = [parts[1] intValue];
            ops[nOps].outCh = [parts[2] intValue];
            [opPaths addObject:parts[0]];
            opMap[parts[3]] = @(nOps);
            nOps++;
        }

        int (^opIdx)(NSString *) = ^int(NSString *name) {
            NSNumber *n = opMap[name];
            if (!n) { fprintf(stderr, "Op not found: %s\n", [name UTF8String]); exit(1); }
            return [n intValue];
        };

        // Load CPU weights
        FILE *wf = fopen(weights_path, "rb");
        CPUWeights W;
        W.wte = malloc(VOCAB_SIZE * DIM * sizeof(float));
        W.wpe = malloc(MAX_SEQ * DIM * sizeof(float));
        fread(W.wte, sizeof(float), VOCAB_SIZE * DIM, wf);
        fread(W.wpe, sizeof(float), MAX_SEQ * DIM, wf);
        for (int i = 0; i < N_LAYERS; i++) {
            W.ln1_w[i] = malloc(DIM * sizeof(float));
            W.ln1_b[i] = malloc(DIM * sizeof(float));
            W.ln2_w[i] = malloc(DIM * sizeof(float));
            W.ln2_b[i] = malloc(DIM * sizeof(float));
            fread(W.ln1_w[i], sizeof(float), DIM, wf);
            fread(W.ln1_b[i], sizeof(float), DIM, wf);
            fread(W.ln2_w[i], sizeof(float), DIM, wf);
            fread(W.ln2_b[i], sizeof(float), DIM, wf);
        }
        W.ln_f_w = malloc(DIM * sizeof(float));
        W.ln_f_b = malloc(DIM * sizeof(float));
        fread(W.ln_f_w, sizeof(float), DIM, wf);
        fread(W.ln_f_b, sizeof(float), DIM, wf);
        fclose(wf);

        // Compile + load all ANE models
        ane_client = ((id (*)(id, SEL))objc_msgSend)((id)_Cl, NSSelectorFromString(@"sharedConnection"));
        NSError *err = nil;

        for (int i = 0; i < nOps; i++) {
            NSURL *url = [NSURL fileURLWithPath:opPaths[i]];
            NSString *key = [NSString stringWithFormat:@"op_%d", i];
            ops[i].model = ((id (*)(id, SEL, id, id))objc_msgSend)(
                (id)_Mo, NSSelectorFromString(@"modelAtURL:key:"), url, key);
            ((BOOL (*)(id, SEL, id, id, NSInteger, id*))objc_msgSend)(
                ane_client, NSSelectorFromString(@"compileModel:options:qos:error:"),
                ops[i].model, @{}, 0, &err);
            ((BOOL (*)(id, SEL, id, id, NSInteger, id*))objc_msgSend)(
                ane_client, NSSelectorFromString(@"loadModel:options:qos:error:"),
                ops[i].model, @{}, 0, &err);
            id attrs = ((id (*)(id, SEL))objc_msgSend)(ops[i].model, NSSelectorFromString(@"modelAttributes"));
            NSDictionary *ns = [attrs[@"NetworkStatusList"] firstObject];
            ops[i].inBS = [[ns[@"LiveInputList"] firstObject][@"BatchStride"] unsignedIntValue];
            ops[i].outBS = [[ns[@"LiveOutputList"] firstObject][@"BatchStride"] unsignedIntValue];
            ops[i].inPS = [[ns[@"LiveInputList"] firstObject][@"PlaneStride"] unsignedIntValue];
            ops[i].outPS = [[ns[@"LiveOutputList"] firstObject][@"PlaneStride"] unsignedIntValue];
            ((BOOL (*)(id, SEL, id, id, NSInteger, id*))objc_msgSend)(
                ane_client, NSSelectorFromString(@"doUnloadModel:options:qos:error:"),
                ops[i].model, @{}, 0, &err);
        }

        // Load all + allocate IOSurfaces + pre-build requests
        for (int i = 0; i < nOps; i++) {
            ((BOOL (*)(id, SEL, id, id, NSInteger, id*))objc_msgSend)(
                ane_client, NSSelectorFromString(@"loadModel:options:qos:error:"),
                ops[i].model, @{}, 0, &err);

            NSDictionary *inP = @{@"IOSurfaceWidth":@(ops[i].inBS/2), @"IOSurfaceHeight":@1,
                @"IOSurfaceBytesPerRow":@(ops[i].inBS), @"IOSurfaceBytesPerElement":@2,
                @"IOSurfaceAllocSize":@(ops[i].inBS), @"IOSurfacePixelFormat":@(0x6630304C)};
            NSDictionary *outP = @{@"IOSurfaceWidth":@(ops[i].outBS/2), @"IOSurfaceHeight":@1,
                @"IOSurfaceBytesPerRow":@(ops[i].outBS), @"IOSurfaceBytesPerElement":@2,
                @"IOSurfaceAllocSize":@(ops[i].outBS), @"IOSurfacePixelFormat":@(0x6630304C)};
            ops[i].inSurf = IOSurfaceCreate((__bridge CFDictionaryRef)inP);
            ops[i].outSurf = IOSurfaceCreate((__bridge CFDictionaryRef)outP);

            ops[i].inObj = ((id (*)(id, SEL, void*, NSInteger, BOOL))objc_msgSend)(
                [_IO alloc], NSSelectorFromString(@"initWithIOSurface:startOffset:shouldRetain:"),
                ops[i].inSurf, 0, YES);
            ops[i].outObj = ((id (*)(id, SEL, void*, NSInteger, BOOL))objc_msgSend)(
                [_IO alloc], NSSelectorFromString(@"initWithIOSurface:startOffset:shouldRetain:"),
                ops[i].outSurf, 0, YES);

            ops[i].request = ((id (*)(id, SEL, id, id, id, id, id, id, id, id, id))objc_msgSend)(
                [_Rq alloc],
                NSSelectorFromString(@"initWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:sharedEvents:transactionHandle:"),
                @[ops[i].inObj], @[@0], @[ops[i].outObj], @[@0], nil, nil, @(0), nil, nil);
        }

        fprintf(stderr, "All %d models loaded. Benchmarking %d gen tokens...\n", nOps, n_gen);

        // Allocate KV caches (shared, reset between runs)
        KVCache kv[N_LAYERS];
        for (int i = 0; i < N_LAYERS; i++) {
            kv[i].k_cache = calloc(MAX_SEQ * N_HEADS * HEAD_DIM, sizeof(float));
            kv[i].v_cache = calloc(MAX_SEQ * N_HEADS * HEAD_DIM, sizeof(float));
            kv[i].len = 0;
        }

        int *ref_tokens = calloc(n_gen, sizeof(int));
        int *test_tokens = calloc(n_gen, sizeof(int));

        // Pre-map all IOSurfaces for mode E
        for (int i = 0; i < nOps; i++) {
            ((BOOL (*)(id, SEL, id, id, BOOL, id*))objc_msgSend)(
                ane_client, NSSelectorFromString(@"mapIOSurfacesWithModel:request:cacheInference:error:"),
                ops[i].model, ops[i].request, NO, nil);
        }

        // ─── Warmup (mode A, discard) ───
        fprintf(stderr, "Warmup...\n");
        run_generation(ops, &W, kv, prompt_tokens, n_prompt, n_gen,
                      dispatch_original, opIdx, NULL);

        // ─── Mode A: original (fresh request per dispatch) ───
        double ms_A = run_generation(ops, &W, kv, prompt_tokens, n_prompt, n_gen,
                                     dispatch_original, opIdx, ref_tokens);
        fprintf(stderr, "A) original:    %6.1f ms = %5.1f tok/s\n", ms_A, n_gen/(ms_A/1000.0));

        // ─── Mode B: request reuse ───
        double ms_B = run_generation(ops, &W, kv, prompt_tokens, n_prompt, n_gen,
                                     dispatch_req_reuse, opIdx, test_tokens);
        int match_B = memcmp(ref_tokens, test_tokens, n_gen * sizeof(int)) == 0;
        fprintf(stderr, "B) req_reuse:   %6.1f ms = %5.1f tok/s  [%s]\n",
                ms_B, n_gen/(ms_B/1000.0), match_B ? "MATCH" : "MISMATCH");

        // ─── Mode C: request reuse + cacheInference:YES ───
        double ms_C = run_generation(ops, &W, kv, prompt_tokens, n_prompt, n_gen,
                                     dispatch_cache_yes, opIdx, test_tokens);
        int match_C = memcmp(ref_tokens, test_tokens, n_gen * sizeof(int)) == 0;
        fprintf(stderr, "C) cache_yes:   %6.1f ms = %5.1f tok/s  [%s]\n",
                ms_C, n_gen/(ms_C/1000.0), match_C ? "MATCH" : "MISMATCH");

        // ─── Mode D: no unmap ───
        double ms_D = run_generation(ops, &W, kv, prompt_tokens, n_prompt, n_gen,
                                     dispatch_no_unmap, opIdx, test_tokens);
        int match_D = memcmp(ref_tokens, test_tokens, n_gen * sizeof(int)) == 0;
        fprintf(stderr, "D) no_unmap:    %6.1f ms = %5.1f tok/s  [%s]\n",
                ms_D, n_gen/(ms_D/1000.0), match_D ? "MATCH" : "MISMATCH");

        // ─── Mode E: eval only (pre-mapped) ───
        // Unmap first, then re-map for clean state
        for (int i = 0; i < nOps; i++) {
            ((void (*)(id, SEL, id, id))objc_msgSend)(
                ane_client, NSSelectorFromString(@"unmapIOSurfacesWithModel:request:"),
                ops[i].model, ops[i].request);
        }
        for (int i = 0; i < nOps; i++) {
            ((BOOL (*)(id, SEL, id, id, BOOL, id*))objc_msgSend)(
                ane_client, NSSelectorFromString(@"mapIOSurfacesWithModel:request:cacheInference:error:"),
                ops[i].model, ops[i].request, YES, nil);
        }
        double ms_E = run_generation(ops, &W, kv, prompt_tokens, n_prompt, n_gen,
                                     dispatch_eval_only, opIdx, test_tokens);
        int match_E = memcmp(ref_tokens, test_tokens, n_gen * sizeof(int)) == 0;
        fprintf(stderr, "E) eval_only:   %6.1f ms = %5.1f tok/s  [%s]\n",
                ms_E, n_gen/(ms_E/1000.0), match_E ? "MATCH" : "MISMATCH");

        // ─── Summary ───
        fprintf(stderr, "\n--- Summary ---\n");
        fprintf(stderr, "A) original:  %.1f tok/s (baseline)\n", n_gen/(ms_A/1000.0));
        fprintf(stderr, "B) req_reuse: %.1f tok/s (%+.1f%%)\n",
                n_gen/(ms_B/1000.0), (ms_A/ms_B - 1)*100);
        fprintf(stderr, "C) cache_yes: %.1f tok/s (%+.1f%%)\n",
                n_gen/(ms_C/1000.0), (ms_A/ms_C - 1)*100);
        fprintf(stderr, "D) no_unmap:  %.1f tok/s (%+.1f%%)\n",
                n_gen/(ms_D/1000.0), (ms_A/ms_D - 1)*100);
        fprintf(stderr, "E) eval_only: %.1f tok/s (%+.1f%%)\n",
                n_gen/(ms_E/1000.0), (ms_A/ms_E - 1)*100);

        // Print ref tokens for verification
        fprintf(stderr, "\nRef tokens: ");
        for (int i = 0; i < n_gen && i < 10; i++)
            fprintf(stderr, "%d ", ref_tokens[i]);
        fprintf(stderr, "...\n");

        // Cleanup
        for (int i = 0; i < nOps; i++) {
            ((void (*)(id, SEL, id, id))objc_msgSend)(
                ane_client, NSSelectorFromString(@"unmapIOSurfacesWithModel:request:"),
                ops[i].model, ops[i].request);
            ((BOOL (*)(id, SEL, id, id, NSInteger, id*))objc_msgSend)(
                ane_client, NSSelectorFromString(@"doUnloadModel:options:qos:error:"),
                ops[i].model, @{}, 0, &err);
            if (ops[i].inSurf) CFRelease(ops[i].inSurf);
            if (ops[i].outSurf) CFRelease(ops[i].outSurf);
        }
        free(ops); free(prompt_tokens); free(ref_tokens); free(test_tokens);
        for (int i = 0; i < N_LAYERS; i++) {
            free(kv[i].k_cache); free(kv[i].v_cache);
        }

        return 0;
    }
}
