// attention_chain.m — Execute 7-op attention chain on ANE via multi-dispatch
//
// Build: xcrun clang -framework Foundation -framework IOSurface -fobjc-arc \
//        -o attention_chain attention_chain.m
// Run: ./attention_chain <models_dir> <channels>
//
// Models dir must contain: q_proj.mlmodelc, k_proj.mlmodelc, v_proj.mlmodelc,
//   qk_matmul.mlmodelc, attn_softmax.mlmodelc, sv_matmul.mlmodelc, o_proj.mlmodelc

#import <Foundation/Foundation.h>
#import <dlfcn.h>
#import <signal.h>
#import <IOSurface/IOSurface.h>
#import <objc/runtime.h>
#import <objc/message.h>

static Class _ANEClientCls, _ANEModelCls, _ANERequestCls, _ANEIOSurfaceObjectCls;

static void loadFW(void) {
    dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);
    _ANEClientCls = NSClassFromString(@"_ANEClient");
    _ANEModelCls = NSClassFromString(@"_ANEModel");
    _ANERequestCls = NSClassFromString(@"_ANERequest");
    _ANEIOSurfaceObjectCls = NSClassFromString(@"_ANEIOSurfaceObject");
}

static IOSurfaceRef makeSurface(uint32_t channels) {
    uint32_t batchStride = channels * 64;  // PlaneStride=64
    if (batchStride < 4096) batchStride = 4096;
    NSDictionary *props = @{
        @"IOSurfaceWidth": @(batchStride / 2),
        @"IOSurfaceHeight": @1,
        @"IOSurfaceBytesPerRow": @(batchStride),
        @"IOSurfaceBytesPerElement": @2,
        @"IOSurfaceAllocSize": @(batchStride),
        @"IOSurfacePixelFormat": @(0x6630304C),
    };
    return IOSurfaceCreate((__bridge CFDictionaryRef)props);
}

static void fillOnes(IOSurfaceRef surf, int channels) {
    IOSurfaceLock(surf, 0, NULL);
    void *base = IOSurfaceGetBaseAddress(surf);
    uint32_t allocSize = (uint32_t)IOSurfaceGetAllocSize(surf);
    memset(base, 0, allocSize);
    uint16_t one = 0x3C00;
    for (int i = 0; i < channels; i++) {
        memcpy((uint8_t*)base + i * 64, &one, 2);
    }
    IOSurfaceUnlock(surf, 0, NULL);
}

static void copySurface(IOSurfaceRef dst, IOSurfaceRef src, uint32_t size) {
    IOSurfaceLock(src, kIOSurfaceLockReadOnly, NULL);
    IOSurfaceLock(dst, 0, NULL);
    void *srcBase = IOSurfaceGetBaseAddress(src);
    void *dstBase = IOSurfaceGetBaseAddress(dst);
    memcpy(dstBase, srcBase, size);
    IOSurfaceUnlock(dst, 0, NULL);
    IOSurfaceUnlock(src, kIOSurfaceLockReadOnly, NULL);
}

static BOOL evaluateModel(id client, id model, IOSurfaceRef input, IOSurfaceRef output) {
    id inObj = ((id (*)(id, SEL, void*, NSInteger, BOOL))objc_msgSend)(
        [_ANEIOSurfaceObjectCls alloc],
        NSSelectorFromString(@"initWithIOSurface:startOffset:shouldRetain:"),
        input, 0, YES);
    id outObj = ((id (*)(id, SEL, void*, NSInteger, BOOL))objc_msgSend)(
        [_ANEIOSurfaceObjectCls alloc],
        NSSelectorFromString(@"initWithIOSurface:startOffset:shouldRetain:"),
        output, 0, YES);

    id req = ((id (*)(id, SEL, id, id, id, id, id, id, id, id, id))objc_msgSend)(
        [_ANERequestCls alloc],
        NSSelectorFromString(@"initWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:sharedEvents:transactionHandle:"),
        @[inObj], @[@0], @[outObj], @[@0], nil, nil, @(0), nil, nil);

    BOOL mapOK = ((BOOL (*)(id, SEL, id, id, BOOL, id*))objc_msgSend)(
        client, NSSelectorFromString(@"mapIOSurfacesWithModel:request:cacheInference:error:"),
        model, req, NO, nil);
    if (!mapOK) return NO;

    BOOL evalOK = ((BOOL (*)(id, SEL, id, id, id, int, id*))objc_msgSend)(
        client, NSSelectorFromString(@"doEvaluateDirectWithModel:options:request:qos:error:"),
        model, @{}, req, 21, nil);

    ((void (*)(id, SEL, id, id))objc_msgSend)(
        client, NSSelectorFromString(@"unmapIOSurfacesWithModel:request:"), model, req);

    return evalOK;
}

int main(int argc, char *argv[]) {
    @autoreleasepool {
        signal(SIGSEGV, SIG_IGN);
        if (argc < 3) {
            fprintf(stderr, "Usage: %s <models_dir> <channels>\n", argv[0]);
            return 1;
        }

        loadFW();
        NSString *dir = [NSString stringWithUTF8String:argv[1]];
        int ch = atoi(argv[2]);

        id client = ((id (*)(id, SEL))objc_msgSend)(
            (id)_ANEClientCls, NSSelectorFromString(@"sharedConnection"));

        // Load all 7 models
        NSArray *names = @[@"q_proj", @"k_proj", @"v_proj", @"qk_matmul",
                          @"attn_softmax", @"sv_matmul", @"o_proj"];
        NSMutableArray *models = [NSMutableArray array];
        for (NSString *name in names) {
            NSString *path = [dir stringByAppendingPathComponent:
                             [name stringByAppendingString:@".mlmodelc"]];
            NSURL *url = [NSURL fileURLWithPath:path];
            id model = ((id (*)(id, SEL, id, id))objc_msgSend)(
                (id)_ANEModelCls, NSSelectorFromString(@"modelAtURL:key:"), url, @"default");
            if (!model) { fprintf(stderr, "Failed: %s\n", [name UTF8String]); return 1; }

            // Compile + load
            ((BOOL (*)(id, SEL, id, id, NSInteger, id*))objc_msgSend)(
                client, NSSelectorFromString(@"compileModel:options:qos:error:"),
                model, @{}, 0, nil);
            BOOL loadOK = ((BOOL (*)(id, SEL, id, id, NSInteger, id*))objc_msgSend)(
                client, NSSelectorFromString(@"loadModel:options:qos:error:"),
                model, @{}, 0, nil);
            if (!loadOK) { fprintf(stderr, "Load failed: %s\n", [name UTF8String]); return 1; }
            [models addObject:model];
            fprintf(stderr, "  Loaded: %s\n", [name UTF8String]);
        }

        // Attention data flow:
        //   x → Q_proj → Q
        //   x → K_proj → K       (parallel with Q, same input x)
        //   x → V_proj → V       (parallel with Q, same input x)
        //   Q → QK_matmul → QK   (QK = W_qkt @ Q, proxy for Q@K^T)
        //   QK → softmax → scores
        //   scores → SV_matmul → SV  (SV = W_sv @ scores, proxy for scores@V)
        //   SV → O_proj → output
        //
        // IOSurface routing:
        //   surf_x     → q_proj     → surf_q
        //   surf_x     → k_proj     → surf_k     (k unused in this simplified chain)
        //   surf_x     → v_proj     → surf_v     (v unused in this simplified chain)
        //   surf_q     → qk_matmul  → surf_qk
        //   surf_qk    → softmax    → surf_scores
        //   surf_scores → sv_matmul → surf_sv
        //   surf_sv    → o_proj     → surf_out

        IOSurfaceRef surf_x = makeSurface(ch);
        IOSurfaceRef surf_q = makeSurface(ch);
        IOSurfaceRef surf_k = makeSurface(ch);
        IOSurfaceRef surf_v = makeSurface(ch);
        IOSurfaceRef surf_qk = makeSurface(ch);
        IOSurfaceRef surf_scores = makeSurface(ch);
        IOSurfaceRef surf_sv = makeSurface(ch);
        IOSurfaceRef surf_out = makeSurface(ch);

        fillOnes(surf_x, ch);

        // Q, K, V projections (all from x)
        IOSurfaceRef inputs[]  = {surf_x, surf_x, surf_x, surf_q, surf_qk, surf_scores, surf_sv};
        IOSurfaceRef outputs[] = {surf_q, surf_k, surf_v, surf_qk, surf_scores, surf_sv, surf_out};

        for (int i = 0; i < 7; i++) {
            BOOL ok = evaluateModel(client, models[i], inputs[i], outputs[i]);
            if (!ok) {
                fprintf(stderr, "Eval failed at step %d (%s)\n", i,
                        [[names objectAtIndex:i] UTF8String]);
                return 1;
            }
        }

        // Read final output
        IOSurfaceLock(surf_out, kIOSurfaceLockReadOnly, NULL);
        void *base = IOSurfaceGetBaseAddress(surf_out);
        for (int i = 0; i < ch; i++) {
            uint16_t fp16;
            memcpy(&fp16, (uint8_t*)base + i * 64, 2);
            printf("%04x ", fp16);
        }
        printf("\n");
        IOSurfaceUnlock(surf_out, kIOSurfaceLockReadOnly, NULL);

        // Cleanup
        for (int i = 0; i < 7; i++) {
            ((BOOL (*)(id, SEL, id, id, NSInteger, id*))objc_msgSend)(
                client, NSSelectorFromString(@"doUnloadModel:options:qos:error:"),
                models[i], @{}, 0, nil);
        }

        fprintf(stderr, "Chain complete.\n");
        return 0;
    }
}
