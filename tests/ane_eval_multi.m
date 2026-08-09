// ane_eval_multi.m — MULTI-INPUT / MULTI-OUTPUT direct dispatch (CA 17-06-56).
//
// WHY THIS EXISTS. `ane_eval_binary.m` hardcodes ONE input and ONE output:
//     @[inObj], @[@0], @[outObj], @[@0]
// and reads `[LiveInputList firstObject]` / `[LiveOutputList firstObject]`.
// The A2 composed graphs take 4 inputs and return 2, so that harness cannot drive
// them. The SELECTOR is already general -- initWithInputs:inputIndices:outputs:
// outputIndices: takes ARRAYS -- so this is a HARNESS extension, not a graph change,
// and it therefore does not spend the reformulation budget (CA §3).
//
// Everything else is byte-for-byte the proven path: _ANEModel modelAtURL:key:,
// _ANEClient compileModel: + loadModel:, mapIOSurfacesWithModel:, and
// doEvaluateDirectWithModel:options:request:qos:error: -- the construction-based
// placement guarantee (standing_rules.md:511; ane-compiler/README.md:5).
//
// Q3 DISCRIMINATOR, PRE-REGISTERED. Three ordered failure points, textually distinct,
// each with its own exit code, so a CONSTRUCTION refusal and a DISPATCH absence never
// read as the same zero:
//     exit 2  MODEL_FAILED  -- _ANEModel could not be constructed. No dispatch attempted.
//     exit 3  LOAD_FAILED   -- model exists, compile/load refused. No dispatch attempted.
//     exit 4  EVAL_FAILED   -- model LOADED, doEvaluateDirectWithModel returned NO.
//                             This is the only one that is a dispatch refusal.
// Usage: ane_eval_multi <model.mlmodelc> [--no-compile]
//   stdin : concatenated FP16 for every input, in LiveInputList order
//   stdout: concatenated FP16 for every output, in LiveOutputList order
//   stderr: one STATUS line, then the per-tensor geometry it used.
#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <objc/message.h>
#include <stdio.h>
#include <signal.h>
#include <dlfcn.h>
#include <string.h>

static Class _ANEClientCls, _ANEModelCls, _ANERequestCls, _ANEIOSurfaceObjectCls;

static void loadFW(void) {
    dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);
    _ANEClientCls = NSClassFromString(@"_ANEClient");
    _ANEModelCls = NSClassFromString(@"_ANEModel");
    _ANERequestCls = NSClassFromString(@"_ANERequest");
    _ANEIOSurfaceObjectCls = NSClassFromString(@"_ANEIOSurfaceObject");
}

static IOSurfaceRef mkSurface(uint32_t batchStride) {
    NSDictionary *p = @{@"IOSurfaceWidth":@(batchStride/2), @"IOSurfaceHeight":@1,
        @"IOSurfaceBytesPerRow":@(batchStride), @"IOSurfaceBytesPerElement":@2,
        @"IOSurfaceAllocSize":@(batchStride), @"IOSurfacePixelFormat":@(0x6630304C)};
    return IOSurfaceCreate((__bridge CFDictionaryRef)p);
}

int main(int argc, char *argv[]) {
    @autoreleasepool {
        signal(SIGSEGV, SIG_IGN);
        if (argc < 2) {
            fprintf(stderr, "Usage: %s <model.mlmodelc> [--no-compile]\n", argv[0]);
            return 1;
        }
        BOOL skipCompile = NO, dense = NO;
        for (int i = 1; i < argc; i++)
            if (strcmp(argv[i], "--no-compile") == 0) skipCompile = YES;
            else if (strcmp(argv[i], "--dense") == 0) dense = YES;

        loadFW();
        id client = ((id (*)(id, SEL))objc_msgSend)((id)_ANEClientCls, NSSelectorFromString(@"sharedConnection"));
        NSURL *url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:argv[1]]];
        id model = ((id (*)(id, SEL, id, id))objc_msgSend)((id)_ANEModelCls,
                        NSSelectorFromString(@"modelAtURL:key:"), url, @"default");
        if (!model) { fprintf(stderr, "STATUS=MODEL_FAILED\n"); return 2; }

        NSError *err = nil;
        if (!skipCompile)
            ((BOOL (*)(id, SEL, id, id, NSInteger, id*))objc_msgSend)(
                client, NSSelectorFromString(@"compileModel:options:qos:error:"), model, @{}, 0, &err);
        BOOL loadOK = ((BOOL (*)(id, SEL, id, id, NSInteger, id*))objc_msgSend)(
            client, NSSelectorFromString(@"loadModel:options:qos:error:"), model, @{}, 0, &err);
        if (!loadOK) {
            fprintf(stderr, "STATUS=LOAD_FAILED err=%s\n",
                    err ? [[err description] UTF8String] : "(nil)");
            return 3;
        }

        // ---- geometry for EVERY tensor, not just firstObject
        id attrs = ((id (*)(id, SEL))objc_msgSend)(model, NSSelectorFromString(@"modelAttributes"));
        NSDictionary *ns = [attrs[@"NetworkStatusList"] firstObject];
        NSArray *ins = ns[@"LiveInputList"], *outs = ns[@"LiveOutputList"];
        NSUInteger nIn = [ins count], nOut = [outs count];
        fprintf(stderr, "GEOM inputs=%lu outputs=%lu\n", (unsigned long)nIn, (unsigned long)nOut);

        NSMutableArray *inObjs = [NSMutableArray array], *inIdx = [NSMutableArray array];
        NSMutableArray *outObjs = [NSMutableArray array], *outIdx = [NSMutableArray array];
        IOSurfaceRef *outSurfs = calloc(nOut, sizeof(IOSurfaceRef));
        uint32_t *outBSs = calloc(nOut, sizeof(uint32_t)), *outPSs = calloc(nOut, sizeof(uint32_t));

        for (NSUInteger i = 0; i < nIn; i++) {
            uint32_t bs = [ins[i][@"BatchStride"] unsignedIntValue];
            uint32_t ps = [ins[i][@"PlaneStride"] unsignedIntValue];
            // DENSE (CA 17-22-53 §2): the surface holds the WHOLE tensor --
            // (bs/ps) planes x (ps/2) contiguous fp16 = bs/2 elements. The
            // 1-value-per-PlaneStride convention inherited from ane_eval_binary
            // wrote 1 element in every ps/2 and left the rest at the memset zero,
            // so 99.97% of each tensor was never stated. Dense states all of it.
            uint32_t nPl = ps ? bs / ps : 1;
            uint32_t perPl = ps ? ps / 2 : bs / 2;
            uint32_t n  = dense ? (nPl * perPl) : nPl;
            fprintf(stderr, "  in[%lu] BatchStride=%u PlaneStride=%u planes=%u perPlane=%u n=%u dense=%d\n",
                    (unsigned long)i, bs, ps, nPl, perPl, n, (int)dense);
            IOSurfaceRef s = mkSurface(bs);
            IOSurfaceLock(s, 0, NULL);
            void *base = IOSurfaceGetBaseAddress(s);
            memset(base, 0, bs);
            if (dense) {
                for (uint32_t pl = 0; pl < nPl; pl++)
                    if (fread((uint8_t*)base + (size_t)pl * ps, 2, perPl, stdin) != perPl)
                        fprintf(stderr, "  WARN short read in[%lu] plane %u\n", (unsigned long)i, pl);
            } else {
                for (uint32_t k = 0; k < n; k++) {
                    uint16_t v = 0;
                    if (fread(&v, 2, 1, stdin) != 1) v = 0;
                    memcpy((uint8_t*)base + (size_t)k * (ps ? ps : 2), &v, 2);
                }
            }
            IOSurfaceUnlock(s, 0, NULL);
            [inObjs addObject:((id (*)(id, SEL, void*, NSInteger, BOOL))objc_msgSend)(
                [_ANEIOSurfaceObjectCls alloc],
                NSSelectorFromString(@"initWithIOSurface:startOffset:shouldRetain:"), s, 0, YES)];
            [inIdx addObject:@(i)];
        }
        for (NSUInteger i = 0; i < nOut; i++) {
            uint32_t bs = [outs[i][@"BatchStride"] unsignedIntValue];
            outBSs[i] = bs; outPSs[i] = [outs[i][@"PlaneStride"] unsignedIntValue];
            fprintf(stderr, "  out[%lu] BatchStride=%u PlaneStride=%u\n", (unsigned long)i, bs, outPSs[i]);
            outSurfs[i] = mkSurface(bs);
            [outObjs addObject:((id (*)(id, SEL, void*, NSInteger, BOOL))objc_msgSend)(
                [_ANEIOSurfaceObjectCls alloc],
                NSSelectorFromString(@"initWithIOSurface:startOffset:shouldRetain:"), outSurfs[i], 0, YES)];
            [outIdx addObject:@(i)];
        }

        id req = ((id (*)(id, SEL, id, id, id, id, id, id, id, id, id))objc_msgSend)(
            [_ANERequestCls alloc],
            NSSelectorFromString(@"initWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:sharedEvents:transactionHandle:"),
            inObjs, inIdx, outObjs, outIdx, nil, nil, @(0), nil, nil);

        ((BOOL (*)(id, SEL, id, id, BOOL, id*))objc_msgSend)(
            client, NSSelectorFromString(@"mapIOSurfacesWithModel:request:cacheInference:error:"),
            model, req, NO, nil);

        NSError *eerr = nil;
        BOOL evalOK = ((BOOL (*)(id, SEL, id, id, id, int, id*))objc_msgSend)(
            client, NSSelectorFromString(@"doEvaluateDirectWithModel:options:request:qos:error:"),
            model, @{}, req, 21, &eerr);
        if (!evalOK) {
            fprintf(stderr, "STATUS=EVAL_FAILED err=%s\n",
                    eerr ? [[eerr description] UTF8String] : "(nil)");
            return 4;
        }

        for (NSUInteger i = 0; i < nOut; i++) {
            IOSurfaceLock(outSurfs[i], kIOSurfaceLockReadOnly, NULL);
            void *base = IOSurfaceGetBaseAddress(outSurfs[i]);
            uint32_t ps = outPSs[i] ? outPSs[i] : 2;
            uint32_t nPl = outBSs[i] / ps, perPl = ps / 2;
            if (dense) {
                for (uint32_t pl = 0; pl < nPl; pl++)
                    fwrite((uint8_t*)base + (size_t)pl * ps, 2, perPl, stdout);
            } else {
                for (uint32_t k = 0; k < nPl; k++)
                    fwrite((uint8_t*)base + (size_t)k * ps, 2, 1, stdout);
            }
            IOSurfaceUnlock(outSurfs[i], kIOSurfaceLockReadOnly, NULL);
        }
        fflush(stdout);
        fprintf(stderr, "STATUS=OK\n");
        return 0;
    }
}
