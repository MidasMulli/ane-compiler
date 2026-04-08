// ane_generate.m — Full GPT-2 generation loop in C/Obj-C
//
// Zero Python in the hot path. All CPU ops via Accelerate framework.
// ANE dispatch via _ANEClient doEvaluateDirectWithModel (direct kext port).
//
// Usage:
//   1. Python compiles .mlmodelc files (one-time)
//   2. Python writes manifest + weights binary
//   3. This binary loads models, generates tokens, prints output
//
// Build:
//   xcrun clang -O2 -framework Foundation -framework IOSurface -framework Accelerate \
//     -fobjc-arc -o ane_generate ane_generate.m
//
// Copyright 2026 Nick Lo. MIT License.

#import <Foundation/Foundation.h>
#import <Accelerate/Accelerate.h>
#import <IOSurface/IOSurface.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>
#import <signal.h>
#import <mach/mach_time.h>

// ─── Configuration ───
#define N_LAYERS     12
#define N_HEADS      12
#define DIM          768
#define HEAD_DIM     64
#define FFN_DIM      3072
#define VOCAB_SIZE   50257
#define MAX_SEQ      1024
#define LN_EPS       1e-5f

// ─── ANE Framework ───
static Class _Cl, _Mo, _Rq, _IO;
static void loadFW(void) {
    dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);
    _Cl = NSClassFromString(@"_ANEClient");
    _Mo = NSClassFromString(@"_ANEModel");
    _Rq = NSClassFromString(@"_ANERequest");
    _IO = NSClassFromString(@"_ANEIOSurfaceObject");
}

// ─── Op entry (one per ANE model) ───
typedef struct {
    id model;
    int inCh, outCh;
    uint32_t inBS, outBS, inPS, outPS;
    IOSurfaceRef inSurf, outSurf;
    id inObj, outObj;
} OpEntry;

// ─── Model weights (CPU) ───
typedef struct {
    float *wte;         // [VOCAB_SIZE, DIM]
    float *wpe;         // [MAX_SEQ, DIM]
    // Per layer
    float *ln1_w[N_LAYERS], *ln1_b[N_LAYERS];
    float *ln2_w[N_LAYERS], *ln2_b[N_LAYERS];
    // Final LN
    float *ln_f_w, *ln_f_b;
} CPUWeights;

// ─── KV Cache ───
typedef struct {
    float *k_cache; // [MAX_SEQ, N_HEADS, HEAD_DIM]
    float *v_cache; // [MAX_SEQ, N_HEADS, HEAD_DIM]
    int len;
} KVCache;

// ─── Timing ───
static mach_timebase_info_data_t tb;
static inline double ns_to_ms(uint64_t ns) {
    return (double)(ns * tb.numer / tb.denom) / 1e6;
}

// ─── LayerNorm (Accelerate) ───
static void layernorm(const float *x, const float *w, const float *b,
                      float *out, int n) {
    float mean = 0, var = 0;
    vDSP_meanv(x, 1, &mean, n);
    // var = mean((x - mean)^2)
    float neg_mean = -mean;
    vDSP_vsadd(x, 1, &neg_mean, out, 1, n);  // out = x - mean
    vDSP_vsq(out, 1, out, 1, n);               // out = (x-mean)^2
    vDSP_meanv(out, 1, &var, n);
    float inv_std = 1.0f / sqrtf(var + LN_EPS);
    // out = (x - mean) * inv_std
    vDSP_vsadd(x, 1, &neg_mean, out, 1, n);
    vDSP_vsmul(out, 1, &inv_std, out, 1, n);
    // out = out * w + b
    vDSP_vma(out, 1, w, 1, b, 1, out, 1, n);
}

// ─── Softmax (Accelerate) ───
static void softmax_f32(float *x, int n) {
    float max_val;
    vDSP_maxv(x, 1, &max_val, n);
    float neg_max = -max_val;
    vDSP_vsadd(x, 1, &neg_max, x, 1, n);
    // exp
    int n_int = n;
    vvexpf(x, x, &n_int);
    float sum = 0;
    vDSP_sve(x, 1, &sum, n);
    float inv_sum = 1.0f / sum;
    vDSP_vsmul(x, 1, &inv_sum, x, 1, n);
}

// ─── FP16 ↔ FP32 conversion ───
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

// ─── ANE Dispatch ───
static id ane_client = nil;

static void ane_dispatch(OpEntry *op, const uint16_t *input, uint16_t *output) {
    // Fill input IOSurface
    IOSurfaceLock(op->inSurf, 0, NULL);
    void *inBase = IOSurfaceGetBaseAddress(op->inSurf);
    memset(inBase, 0, op->inBS);
    for (int j = 0; j < op->inCh; j++)
        memcpy((uint8_t*)inBase + j * op->inPS, &input[j], 2);
    IOSurfaceUnlock(op->inSurf, 0, NULL);

    // Build request
    id req = ((id (*)(id, SEL, id, id, id, id, id, id, id, id, id))objc_msgSend)(
        [_Rq alloc],
        NSSelectorFromString(@"initWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:sharedEvents:transactionHandle:"),
        @[op->inObj], @[@0], @[op->outObj], @[@0], nil, nil, @(0), nil, nil);

    // Map
    ((BOOL (*)(id, SEL, id, id, BOOL, id*))objc_msgSend)(
        ane_client, NSSelectorFromString(@"mapIOSurfacesWithModel:request:cacheInference:error:"),
        op->model, req, NO, nil);

    // Execute
    ((BOOL (*)(id, SEL, id, id, id, int, id*))objc_msgSend)(
        ane_client, NSSelectorFromString(@"doEvaluateDirectWithModel:options:request:qos:error:"),
        op->model, @{}, req, 21, nil);

    // Read output
    IOSurfaceLock(op->outSurf, kIOSurfaceLockReadOnly, NULL);
    void *outBase = IOSurfaceGetBaseAddress(op->outSurf);
    for (int j = 0; j < op->outCh; j++)
        memcpy(&output[j], (uint8_t*)outBase + j * op->outPS, 2);
    IOSurfaceUnlock(op->outSurf, kIOSurfaceLockReadOnly, NULL);

    // Unmap
    ((void (*)(id, SEL, id, id))objc_msgSend)(
        ane_client, NSSelectorFromString(@"unmapIOSurfacesWithModel:request:"), op->model, req);
}

// ─── Op indices (sorted alphabetically by Python — must match manifest order) ───
// The manifest is sorted by op name. We need to know which index maps to which op.
// Python generates the manifest and this mapping.

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
        int n_gen_tokens = atoi(argv[3]);

        // Parse prompt tokens from argv
        int n_prompt = argc - 4;
        int *prompt_tokens = malloc(n_prompt * sizeof(int));
        for (int i = 0; i < n_prompt; i++)
            prompt_tokens[i] = atoi(argv[4 + i]);

        // ─── Load manifest ───
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
            opMap[parts[3]] = @(nOps);  // op_name -> index
            nOps++;
        }
        fprintf(stderr, "Loaded %d ops from manifest\n", nOps);

        // Helper to get op index
        int (^opIdx)(NSString *) = ^int(NSString *name) {
            NSNumber *n = opMap[name];
            if (!n) { fprintf(stderr, "Op not found: %s\n", [name UTF8String]); exit(1); }
            return [n intValue];
        };

        // ─── Load CPU weights ───
        FILE *wf = fopen(weights_path, "rb");
        if (!wf) { fprintf(stderr, "Cannot open weights: %s\n", weights_path); return 1; }

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
        fprintf(stderr, "CPU weights loaded\n");

        // ─── Compile + Load all ANE models ───
        ane_client = ((id (*)(id, SEL))objc_msgSend)((id)_Cl, NSSelectorFromString(@"sharedConnection"));
        NSError *err = nil;

        fprintf(stderr, "Compiling %d models...\n", nOps);
        for (int i = 0; i < nOps; i++) {
            NSURL *url = [NSURL fileURLWithPath:opPaths[i]];
            NSString *key = [NSString stringWithFormat:@"op_%d", i];
            ops[i].model = ((id (*)(id, SEL, id, id))objc_msgSend)(
                (id)_Mo, NSSelectorFromString(@"modelAtURL:key:"), url, key);

            err = nil;
            ((BOOL (*)(id, SEL, id, id, NSInteger, id*))objc_msgSend)(
                ane_client, NSSelectorFromString(@"compileModel:options:qos:error:"),
                ops[i].model, @{}, 0, &err);
            if (err) { fprintf(stderr, "Compile %d failed: %s\n", i, [[err description] UTF8String]); }

            BOOL ok = ((BOOL (*)(id, SEL, id, id, NSInteger, id*))objc_msgSend)(
                ane_client, NSSelectorFromString(@"loadModel:options:qos:error:"),
                ops[i].model, @{}, 0, &err);
            if (!ok) { fprintf(stderr, "Load %d failed: %s\n", i, err ? [[err description] UTF8String] : "nil"); return 1; }

            // Get buffer info
            id attrs = ((id (*)(id, SEL))objc_msgSend)(ops[i].model, NSSelectorFromString(@"modelAttributes"));
            NSDictionary *ns = [attrs[@"NetworkStatusList"] firstObject];
            ops[i].inBS = [[ns[@"LiveInputList"] firstObject][@"BatchStride"] unsignedIntValue];
            ops[i].outBS = [[ns[@"LiveOutputList"] firstObject][@"BatchStride"] unsignedIntValue];
            ops[i].inPS = [[ns[@"LiveInputList"] firstObject][@"PlaneStride"] unsignedIntValue];
            ops[i].outPS = [[ns[@"LiveOutputList"] firstObject][@"PlaneStride"] unsignedIntValue];

            // Unload after getting info (re-load in dispatch phase)
            ((BOOL (*)(id, SEL, id, id, NSInteger, id*))objc_msgSend)(
                ane_client, NSSelectorFromString(@"doUnloadModel:options:qos:error:"),
                ops[i].model, @{}, 0, &err);
        }

        // Load all + allocate IOSurfaces
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
        }
        fprintf(stderr, "All %d models loaded, ready for generation\n", nOps);

        // ─── Allocate working buffers ───
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

        // KV caches
        KVCache kv[N_LAYERS];
        for (int i = 0; i < N_LAYERS; i++) {
            kv[i].k_cache = calloc(MAX_SEQ * N_HEADS * HEAD_DIM, sizeof(float));
            kv[i].v_cache = calloc(MAX_SEQ * N_HEADS * HEAD_DIM, sizeof(float));
            kv[i].len = 0;
        }

        // ─── Generation loop ───
        int max_tokens = n_prompt + n_gen_tokens;
        int *tokens = malloc(max_tokens * sizeof(int));
        memcpy(tokens, prompt_tokens, n_prompt * sizeof(int));
        int total_tokens = n_prompt;

        // Prefill: process all prompt tokens through KV cache
        fprintf(stderr, "Prefilling %d prompt tokens...\n", n_prompt);
        int total_steps = n_prompt + n_gen_tokens;  // prefill + generate
        uint64_t t_start = mach_absolute_time();

        for (int step = 0; step < total_steps; step++) {
            int pos, tok;
            BOOL is_generate = (step >= n_prompt);
            if (!is_generate) {
                pos = step;
                tok = tokens[pos];
            } else {
                pos = total_tokens - 1;
                tok = tokens[pos];
            }

            // ─── Embedding (CPU) ───
            for (int d = 0; d < DIM; d++)
                x_f32[d] = W.wte[tok * DIM + d] + W.wpe[pos * DIM + d];
            fp32_to_fp16(x_f32, x_fp16, DIM);

            // ─── 12 transformer layers ───
            for (int li = 0; li < N_LAYERS; li++) {
                // LayerNorm 1
                fp16_to_fp32(x_fp16, x_f32, DIM);
                layernorm(x_f32, W.ln1_w[li], W.ln1_b[li], ln_out, DIM);
                uint16_t ln1_fp16[DIM];
                fp32_to_fp16(ln_out, ln1_fp16, DIM);

                // QKV projection (ANE)
                int qkv_idx = opIdx([NSString stringWithFormat:@"L%d_qkv_proj", li]);
                ane_dispatch(&ops[qkv_idx], ln1_fp16, qkv_fp16);

                // Split QKV and convert to F32
                float q_f32[DIM], k_f32[DIM], v_f32[DIM];
                fp16_to_fp32(qkv_fp16, q_f32, DIM);
                fp16_to_fp32(qkv_fp16 + DIM, k_f32, DIM);
                fp16_to_fp32(qkv_fp16 + 2*DIM, v_f32, DIM);

                // KV cache append
                int seq_pos = kv[li].len;
                memcpy(&kv[li].k_cache[seq_pos * N_HEADS * HEAD_DIM], k_f32, DIM * sizeof(float));
                memcpy(&kv[li].v_cache[seq_pos * N_HEADS * HEAD_DIM], v_f32, DIM * sizeof(float));
                kv[li].len++;
                int seq_len = kv[li].len;

                // Multi-head attention (CPU, Accelerate BLAS)
                float scale = 1.0f / sqrtf((float)HEAD_DIM);
                memset(attn_out_f32, 0, DIM * sizeof(float));

                for (int h = 0; h < N_HEADS; h++) {
                    float *q_h = &q_f32[h * HEAD_DIM];           // [HEAD_DIM]

                    // Build K matrix for this head: [seq_len, HEAD_DIM]
                    // K is stored as [seq, heads, head_dim]
                    // scores = Q @ K^T (1xD @ DxS = 1xS)
                    for (int s = 0; s < seq_len; s++) {
                        float dot = 0;
                        vDSP_dotpr(q_h, 1,
                                   &kv[li].k_cache[s * N_HEADS * HEAD_DIM + h * HEAD_DIM], 1,
                                   &dot, HEAD_DIM);
                        scores[s] = dot * scale;
                    }

                    // Softmax
                    softmax_f32(scores, seq_len);

                    // attn_out[h] = scores @ V (1xS @ SxD = 1xD)
                    float *out_h = &attn_out_f32[h * HEAD_DIM];
                    memset(out_h, 0, HEAD_DIM * sizeof(float));
                    for (int s = 0; s < seq_len; s++) {
                        float w = scores[s];
                        float *v_s = &kv[li].v_cache[s * N_HEADS * HEAD_DIM + h * HEAD_DIM];
                        vDSP_vsma(v_s, 1, &w, out_h, 1, out_h, 1, HEAD_DIM);
                    }
                }

                // O projection (ANE)
                uint16_t attn_fp16[DIM];
                fp32_to_fp16(attn_out_f32, attn_fp16, DIM);
                int o_idx = opIdx([NSString stringWithFormat:@"L%d_o_proj", li]);
                ane_dispatch(&ops[o_idx], attn_fp16, o_fp16);

                // Residual 1: x = x + o_out
                fp16_to_fp32(x_fp16, x_f32, DIM);
                float o_f32[DIM];
                fp16_to_fp32(o_fp16, o_f32, DIM);
                vDSP_vadd(x_f32, 1, o_f32, 1, x_f32, 1, DIM);

                // LayerNorm 2
                layernorm(x_f32, W.ln2_w[li], W.ln2_b[li], ln_out, DIM);
                uint16_t ln2_fp16[DIM];
                fp32_to_fp16(ln_out, ln2_fp16, DIM);

                // Fused FFN (ANE): up + GELU + down in one dispatch
                int ffn_idx = opIdx([NSString stringWithFormat:@"L%d_fused_ffn", li]);
                ane_dispatch(&ops[ffn_idx], ln2_fp16, ffn_fp16);

                // Residual 2: x = x + ffn_out
                float ffn_f32[DIM];
                fp16_to_fp32(ffn_fp16, ffn_f32, DIM);
                vDSP_vadd(x_f32, 1, ffn_f32, 1, x_f32, 1, DIM);
                fp32_to_fp16(x_f32, x_fp16, DIM);
            }

            // ─── Final LayerNorm + LM Head ───
            fp16_to_fp32(x_fp16, x_f32, DIM);
            layernorm(x_f32, W.ln_f_w, W.ln_f_b, ln_out, DIM);
            uint16_t lnf_fp16[DIM];
            fp32_to_fp16(ln_out, lnf_fp16, DIM);

            int lm_idx = opIdx(@"lm_head");
            ane_dispatch(&ops[lm_idx], lnf_fp16, logits_fp16);

            // ─── Argmax ───
            fp16_to_fp32(logits_fp16, logits_f32, VOCAB_SIZE);
            vDSP_Length max_idx = 0;
            float max_val = 0;
            vDSP_maxvi(logits_f32, 1, &max_val, &max_idx, VOCAB_SIZE);

            if (is_generate) {
                tokens[total_tokens] = (int)max_idx;
                total_tokens++;
                // Print token ID for Python to decode
                printf("%d\n", (int)max_idx);
                fflush(stdout);
            } else {
                // Prefill: don't emit, just populate KV cache
                // For the last prefill token, store the predicted next token
                if (step == n_prompt - 1) {
                    // Start timing generation from here
                    t_start = mach_absolute_time();
                }
            }
        }

        uint64_t t_end = mach_absolute_time();
        double elapsed_ms = ns_to_ms(t_end - t_start);
        double tok_per_sec = (double)n_gen_tokens / (elapsed_ms / 1000.0);
        fprintf(stderr, "\nGenerated %d tokens in %.1f ms = %.1f tok/s\n",
                n_gen_tokens, elapsed_ms, tok_per_sec);

        // ─── Cleanup ───
        for (int i = 0; i < nOps; i++) {
            ((BOOL (*)(id, SEL, id, id, NSInteger, id*))objc_msgSend)(
                ane_client, NSSelectorFromString(@"doUnloadModel:options:qos:error:"),
                ops[i].model, @{}, 0, &err);
            if (ops[i].inSurf) CFRelease(ops[i].inSurf);
            if (ops[i].outSurf) CFRelease(ops[i].outSurf);
        }
        free(ops);
        free(tokens);
        free(prompt_tokens);
        free(x_f32); free(ln_out); free(attn_out_f32); free(scores);
        free(x_fp16); free(qkv_fp16); free(o_fp16); free(ffn_fp16);
        free(logits_fp16); free(logits_f32);
        for (int i = 0; i < N_LAYERS; i++) {
            free(kv[i].k_cache); free(kv[i].v_cache);
        }

        return 0;
    }
}
