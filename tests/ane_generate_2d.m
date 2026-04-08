// ane_generate_2d.m — GPT-2 with LN-fused ANE dispatches
//
// Per layer: LN1+QKV(ANE) -> attention(CPU) -> O(ANE) -> res(CPU) ->
//            LN2+FFN(ANE) -> res(CPU)
// Final: CPU LN_f + ANE lm_head (separate, fused LN_f too large)
// = 37 dispatches, 0 CPU LayerNorm
//
// Build:
//   xcrun clang -O2 -framework Foundation -framework IOSurface -framework Accelerate \
//     -fobjc-arc -o ane_generate_2d ane_generate_2d.m

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
#define VOCAB_SIZE   50257
#define MAX_SEQ      1024

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
} OpEntry;

typedef struct {
    float *k_cache, *v_cache;
    int len;
} KVCache;

static mach_timebase_info_data_t tb;
static inline double ns_to_ms(uint64_t ns) {
    return (double)(ns * tb.numer / tb.denom) / 1e6;
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

static void ane_dispatch(OpEntry *op, const uint16_t *input, uint16_t *output) {
    IOSurfaceLock(op->inSurf, 0, NULL);
    void *inBase = IOSurfaceGetBaseAddress(op->inSurf);
    memset(inBase, 0, op->inBS);
    for (int j = 0; j < op->inCh; j++)
        memcpy((uint8_t*)inBase + j * op->inPS, &input[j], 2);
    IOSurfaceUnlock(op->inSurf, 0, NULL);

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
        fprintf(stderr, "Loaded %d ops from manifest\n", nOps);

        int (^opIdx)(NSString *) = ^int(NSString *name) {
            NSNumber *n = opMap[name];
            if (!n) { fprintf(stderr, "Op not found: %s\n", [name UTF8String]); exit(1); }
            return [n intValue];
        };

        // CPU weights (embeddings + final LN)
        FILE *wf = fopen(weights_path, "rb");
        if (!wf) { fprintf(stderr, "Cannot open weights\n"); return 1; }
        float *wte = malloc(VOCAB_SIZE * DIM * sizeof(float));
        float *wpe = malloc(MAX_SEQ * DIM * sizeof(float));
        float *ln_f_w = malloc(DIM * sizeof(float));
        float *ln_f_b = malloc(DIM * sizeof(float));
        fread(wte, sizeof(float), VOCAB_SIZE * DIM, wf);
        fread(wpe, sizeof(float), MAX_SEQ * DIM, wf);
        fread(ln_f_w, sizeof(float), DIM, wf);
        fread(ln_f_b, sizeof(float), DIM, wf);
        fclose(wf);
        fprintf(stderr, "CPU weights loaded (embeddings + final LN)\n");

        // Compile + load
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
            if (err) fprintf(stderr, "Compile %d failed: %s\n", i, [[err description] UTF8String]);

            BOOL ok = ((BOOL (*)(id, SEL, id, id, NSInteger, id*))objc_msgSend)(
                ane_client, NSSelectorFromString(@"loadModel:options:qos:error:"),
                ops[i].model, @{}, 0, &err);
            if (!ok) { fprintf(stderr, "Load %d FATAL: %s\n", i, err ? [[err description] UTF8String] : "nil"); return 1; }

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
        fprintf(stderr, "All %d models loaded\n", nOps);

        // Buffers
        float *x_f32 = malloc(DIM * sizeof(float));
        float *attn_out_f32 = malloc(DIM * sizeof(float));
        float *scores = malloc(MAX_SEQ * sizeof(float));
        uint16_t *x_fp16 = malloc(DIM * 2);
        uint16_t *qkv_fp16 = malloc(2304 * 2);
        uint16_t *o_fp16 = malloc(DIM * 2);
        uint16_t *ffn_fp16 = malloc(DIM * 2);
        uint16_t *logits_fp16 = malloc(VOCAB_SIZE * 2);
        float *logits_f32 = malloc(VOCAB_SIZE * sizeof(float));

        KVCache kv[N_LAYERS];
        for (int i = 0; i < N_LAYERS; i++) {
            kv[i].k_cache = calloc(MAX_SEQ * N_HEADS * HEAD_DIM, sizeof(float));
            kv[i].v_cache = calloc(MAX_SEQ * N_HEADS * HEAD_DIM, sizeof(float));
            kv[i].len = 0;
        }

        int max_tokens = n_prompt + n_gen_tokens;
        int *tokens = malloc(max_tokens * sizeof(int));
        memcpy(tokens, prompt_tokens, n_prompt * sizeof(int));
        int total_tokens = n_prompt;

        fprintf(stderr, "Prefilling %d prompt tokens...\n", n_prompt);
        int total_steps = n_prompt + n_gen_tokens;
        uint64_t t_start = mach_absolute_time();

        for (int step = 0; step < total_steps; step++) {
            int pos, tok;
            BOOL is_generate = (step >= n_prompt);
            if (!is_generate) { pos = step; tok = tokens[pos]; }
            else { pos = total_tokens - 1; tok = tokens[pos]; }

            // Embedding
            for (int d = 0; d < DIM; d++)
                x_f32[d] = wte[tok * DIM + d] + wpe[pos * DIM + d];
            fp32_to_fp16(x_f32, x_fp16, DIM);

            for (int li = 0; li < N_LAYERS; li++) {
                // LN1 + QKV (fused ANE — no CPU LN!)
                int qkv_idx = opIdx([NSString stringWithFormat:@"L%d_ln1_qkv", li]);
                ane_dispatch(&ops[qkv_idx], x_fp16, qkv_fp16);

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
                ane_dispatch(&ops[o_idx], attn_fp16, o_fp16);

                // Residual 1
                fp16_to_fp32(x_fp16, x_f32, DIM);
                float o_f32[DIM];
                fp16_to_fp32(o_fp16, o_f32, DIM);
                vDSP_vadd(x_f32, 1, o_f32, 1, x_f32, 1, DIM);

                // r1 to FP16 for LN2+FFN
                uint16_t r1_fp16[DIM];
                fp32_to_fp16(x_f32, r1_fp16, DIM);

                // LN2 + FFN (fused ANE — no CPU LN2!)
                int ffn_idx = opIdx([NSString stringWithFormat:@"L%d_ln2_ffn", li]);
                ane_dispatch(&ops[ffn_idx], r1_fp16, ffn_fp16);

                // Residual 2
                float ffn_f32[DIM];
                fp16_to_fp32(ffn_fp16, ffn_f32, DIM);
                vDSP_vadd(x_f32, 1, ffn_f32, 1, x_f32, 1, DIM);
                fp32_to_fp16(x_f32, x_fp16, DIM);
            }

            // Final LayerNorm (CPU) + LM head (ANE)
            fp16_to_fp32(x_fp16, x_f32, DIM);
            // layernorm
            {
                float mean = 0, var = 0;
                vDSP_meanv(x_f32, 1, &mean, DIM);
                float neg_mean = -mean;
                float ln_out[DIM];
                vDSP_vsadd(x_f32, 1, &neg_mean, ln_out, 1, DIM);
                vDSP_vsq(ln_out, 1, ln_out, 1, DIM);
                vDSP_meanv(ln_out, 1, &var, DIM);
                float inv_std = 1.0f / sqrtf(var + 1e-5f);
                vDSP_vsadd(x_f32, 1, &neg_mean, ln_out, 1, DIM);
                vDSP_vsmul(ln_out, 1, &inv_std, ln_out, 1, DIM);
                vDSP_vma(ln_out, 1, ln_f_w, 1, ln_f_b, 1, ln_out, 1, DIM);
                fp32_to_fp16(ln_out, x_fp16, DIM);
            }
            int lm_idx = opIdx(@"lm_head");
            ane_dispatch(&ops[lm_idx], x_fp16, logits_fp16);

            fp16_to_fp32(logits_fp16, logits_f32, VOCAB_SIZE);
            vDSP_Length max_idx = 0;
            float max_val = 0;
            vDSP_maxvi(logits_f32, 1, &max_val, &max_idx, VOCAB_SIZE);

            if (is_generate) {
                tokens[total_tokens] = (int)max_idx;
                total_tokens++;
                printf("%d\n", (int)max_idx);
                fflush(stdout);
            } else if (step == n_prompt - 1) {
                t_start = mach_absolute_time();
            }
        }

        uint64_t t_end = mach_absolute_time();
        double elapsed_ms = ns_to_ms(t_end - t_start);
        double tok_per_sec = (double)n_gen_tokens / (elapsed_ms / 1000.0);
        fprintf(stderr, "\nGenerated %d tokens in %.1f ms = %.1f tok/s\n",
                n_gen_tokens, elapsed_ms, tok_per_sec);
        fprintf(stderr, "Dispatches/token: %d (3/layer + 1 lm_head)\n", 3*N_LAYERS+1);
        fprintf(stderr, "CPU LN eliminated: 24 (12*LN1 + 12*LN2, LN_f still on CPU)\n");

        for (int i = 0; i < nOps; i++) {
            ((BOOL (*)(id, SEL, id, id, NSInteger, id*))objc_msgSend)(
                ane_client, NSSelectorFromString(@"doUnloadModel:options:qos:error:"),
                ops[i].model, @{}, 0, &err);
            if (ops[i].inSurf) CFRelease(ops[i].inSurf);
            if (ops[i].outSurf) CFRelease(ops[i].outSurf);
        }
        free(ops); free(tokens); free(prompt_tokens);
        free(x_f32); free(attn_out_f32); free(scores);
        free(x_fp16); free(qkv_fp16); free(o_fp16); free(ffn_fp16);
        free(logits_fp16); free(logits_f32); free(wte); free(wpe);
        free(ln_f_w); free(ln_f_b);
        for (int i = 0; i < N_LAYERS; i++) {
            free(kv[i].k_cache); free(kv[i].v_cache);
        }
        return 0;
    }
}
