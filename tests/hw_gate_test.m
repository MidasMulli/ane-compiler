// hw_gate_test.m — Hardware execution gate for ane-compiler
//
// Evaluates a conv model on ANE, then swaps the cached .hwx with an emitted
// version and evaluates again. Compares output tensors.
//
// Build: xcrun clang -framework Foundation -framework IOSurface -fobjc-arc \
//        -o hw_gate_test hw_gate_test.m
// Run: ./hw_gate_test <model.mlmodelc> <emitted.hwx> <cache_hwx_path>

#import <Foundation/Foundation.h>
#import <dlfcn.h>
#import <signal.h>
#import <IOSurface/IOSurface.h>
#import <objc/runtime.h>
#import <objc/message.h>

static Class _ANEClientCls;
static Class _ANEModelCls;
static Class _ANERequestCls;
static Class _ANEIOSurfaceObjectCls;

static void loadFW(void) {
    dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);
    _ANEClientCls = NSClassFromString(@"_ANEClient");
    _ANEModelCls = NSClassFromString(@"_ANEModel");
    _ANERequestCls = NSClassFromString(@"_ANERequest");
    _ANEIOSurfaceObjectCls = NSClassFromString(@"_ANEIOSurfaceObject");
}

static IOSurfaceRef makeSurface(uint32_t allocSize) {
    NSDictionary *props = @{
        @"IOSurfaceWidth": @(allocSize / 2),
        @"IOSurfaceHeight": @1,
        @"IOSurfaceBytesPerRow": @(allocSize),
        @"IOSurfaceBytesPerElement": @2,
        @"IOSurfaceAllocSize": @(allocSize),
        @"IOSurfacePixelFormat": @(0x6630304C),
    };
    return IOSurfaceCreate((__bridge CFDictionaryRef)props);
}

static BOOL evaluate(id client, id model, float *inputValues, int inChannels,
                     float *outputValues, int outChannels) {
    // Get model attributes for buffer sizing
    id attrs = ((id (*)(id, SEL))objc_msgSend)(model, NSSelectorFromString(@"modelAttributes"));
    NSDictionary *ns = [attrs[@"NetworkStatusList"] firstObject];
    NSDictionary *inInfo = [ns[@"LiveInputList"] firstObject];
    NSDictionary *outInfo = [ns[@"LiveOutputList"] firstObject];

    uint32_t inBatchStride = [inInfo[@"BatchStride"] unsignedIntValue];
    uint32_t outBatchStride = [outInfo[@"BatchStride"] unsignedIntValue];
    uint32_t planeStride = [inInfo[@"PlaneStride"] unsignedIntValue];

    IOSurfaceRef inSurf = makeSurface(inBatchStride);
    IOSurfaceRef outSurf = makeSurface(outBatchStride);

    // Fill input
    IOSurfaceLock(inSurf, 0, NULL);
    void *inBase = IOSurfaceGetBaseAddress(inSurf);
    memset(inBase, 0, inBatchStride);
    for (int i = 0; i < inChannels; i++) {
        uint16_t fp16;
        float v = inputValues[i];
        // Simple float→fp16 conversion
        uint32_t bits;
        memcpy(&bits, &v, 4);
        uint32_t sign = (bits >> 31) & 1;
        int32_t exp = ((bits >> 23) & 0xFF) - 127;
        uint32_t mant = bits & 0x7FFFFF;
        if (exp > 15) { fp16 = (sign << 15) | 0x7C00; }
        else if (exp < -14) { fp16 = (sign << 15); }
        else { fp16 = (sign << 15) | ((exp + 15) << 10) | (mant >> 13); }
        memcpy((uint8_t*)inBase + i * planeStride, &fp16, 2);
    }
    IOSurfaceUnlock(inSurf, 0, NULL);

    // Build request
    id inObj = ((id (*)(id, SEL, void*, NSInteger, BOOL))objc_msgSend)(
        [_ANEIOSurfaceObjectCls alloc],
        NSSelectorFromString(@"initWithIOSurface:startOffset:shouldRetain:"),
        inSurf, 0, YES);
    id outObj = ((id (*)(id, SEL, void*, NSInteger, BOOL))objc_msgSend)(
        [_ANEIOSurfaceObjectCls alloc],
        NSSelectorFromString(@"initWithIOSurface:startOffset:shouldRetain:"),
        outSurf, 0, YES);

    id req = ((id (*)(id, SEL, id, id, id, id, id, id, id, id, id))objc_msgSend)(
        [_ANERequestCls alloc],
        NSSelectorFromString(@"initWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:sharedEvents:transactionHandle:"),
        @[inObj], @[@0], @[outObj], @[@0], nil, nil, @(0), nil, nil);

    // Map IOSurfaces (on _ANEClient, not _ANEModel)
    id aneClient = ((id (*)(id, SEL))objc_msgSend)(
        (id)_ANEClientCls, NSSelectorFromString(@"sharedConnection"));

    BOOL mapOK = ((BOOL (*)(id, SEL, id, id, BOOL, id*))objc_msgSend)(
        aneClient, NSSelectorFromString(@"mapIOSurfacesWithModel:request:cacheInference:error:"),
        model, req, NO, nil);
    if (!mapOK) { fprintf(stderr, "map failed\n"); return NO; }

    // Evaluate (doEvaluateDirectWithModel — 37% faster)
    BOOL evalOK = ((BOOL (*)(id, SEL, id, id, id, int, id*))objc_msgSend)(
        aneClient, NSSelectorFromString(@"doEvaluateDirectWithModel:options:request:qos:error:"),
        model, @{}, req, 21, nil);
    if (!evalOK) {
        ((void (*)(id, SEL, id, id))objc_msgSend)(
            aneClient, NSSelectorFromString(@"unmapIOSurfacesWithModel:request:"),
            model, req);
        fprintf(stderr, "eval failed\n");
        return NO;
    }

    // Read output
    IOSurfaceLock(outSurf, kIOSurfaceLockReadOnly, NULL);
    void *outBase = IOSurfaceGetBaseAddress(outSurf);
    for (int i = 0; i < outChannels; i++) {
        uint16_t fp16;
        memcpy(&fp16, (uint8_t*)outBase + i * planeStride, 2);
        // fp16 → float
        uint32_t sign = (fp16 >> 15) & 1;
        uint32_t exp = (fp16 >> 10) & 0x1F;
        uint32_t mant = fp16 & 0x3FF;
        float val;
        if (exp == 0) { val = (sign ? -1.0f : 1.0f) * mant / 1024.0f / 16384.0f; }
        else if (exp == 31) { val = (sign ? -1.0f : 1.0f) * INFINITY; }
        else {
            uint32_t fbits = (sign << 31) | ((exp - 15 + 127) << 23) | (mant << 13);
            memcpy(&val, &fbits, 4);
        }
        outputValues[i] = val;
    }
    IOSurfaceUnlock(outSurf, kIOSurfaceLockReadOnly, NULL);

    ((void (*)(id, SEL, id, id))objc_msgSend)(
        aneClient, NSSelectorFromString(@"unmapIOSurfacesWithModel:request:"),
        model, req);

    return YES;
}

int main(int argc, char *argv[]) {
    @autoreleasepool {
        signal(SIGSEGV, SIG_IGN);

        if (argc < 4) {
            fprintf(stderr, "Usage: %s <model.mlmodelc> <emitted.hwx> <cache_hwx_path>\n", argv[0]);
            return 1;
        }

        loadFW();

        NSString *modelPath = [NSString stringWithUTF8String:argv[1]];
        NSString *emittedHWX = [NSString stringWithUTF8String:argv[2]];
        NSString *cacheHWX = [NSString stringWithUTF8String:argv[3]];

        id client = ((id (*)(id, SEL))objc_msgSend)((id)_ANEClientCls, NSSelectorFromString(@"sharedConnection"));
        NSURL *url = [NSURL fileURLWithPath:modelPath];
        id model = ((id (*)(id, SEL, id, id))objc_msgSend)(
            (id)_ANEModelCls, NSSelectorFromString(@"modelAtURL:key:"), url, @"default");

        if (!model) { fprintf(stderr, "Model create failed\n"); return 1; }

        // Compile + load reference
        ((void (*)(id, SEL, id))objc_msgSend)(client, NSSelectorFromString(@"purgeCompiledModel:"), model);
        NSError *err = nil;
        BOOL ok = ((BOOL (*)(id, SEL, id, id, NSInteger, id*))objc_msgSend)(
            client, NSSelectorFromString(@"compileModel:options:qos:error:"), model, @{}, 0, &err);
        if (!ok) { fprintf(stderr, "Compile failed\n"); return 1; }
        ok = ((BOOL (*)(id, SEL, id, id, NSInteger, id*))objc_msgSend)(
            client, NSSelectorFromString(@"loadModel:options:qos:error:"), model, @{}, 0, &err);
        if (!ok) { fprintf(stderr, "Load failed\n"); return 1; }

        // Evaluate with reference .hwx (ones input)
        int channels = 64; // Adjust if needed
        float input[512];
        for (int i = 0; i < channels; i++) input[i] = 1.0f;

        float refOutput[512] = {0};
        float emittedOutput[512] = {0};

        ok = evaluate(client, model, input, channels, refOutput, channels);
        fprintf(stderr, "Reference eval: %s\n", ok ? "OK" : "FAILED");
        if (!ok) return 1;

        // Unload
        ((BOOL (*)(id, SEL, id, id, NSInteger, id*))objc_msgSend)(
            client, NSSelectorFromString(@"doUnloadModel:options:qos:error:"), model, @{}, 0, &err);

        // Swap .hwx in cache
        NSData *backup = [NSData dataWithContentsOfFile:cacheHWX];
        NSData *emitted = [NSData dataWithContentsOfFile:emittedHWX];
        if (!emitted) { fprintf(stderr, "Cannot read emitted .hwx\n"); return 1; }

        // Write emitted .hwx to cache location (DO NOT purge — we want the cache hit)
        [emitted writeToFile:cacheHWX atomically:NO];
        fprintf(stderr, "Swapped cache .hwx (%lu → %lu bytes)\n",
                (unsigned long)backup.length, (unsigned long)emitted.length);

        // Check if compiled model still exists in cache (should be YES)
        BOOL exists = ((BOOL (*)(id, SEL, id))objc_msgSend)(
            client, NSSelectorFromString(@"compiledModelExistsFor:"), model);
        fprintf(stderr, "compiledModelExists after swap: %s\n", exists ? "YES" : "NO");

        // Load directly (no compile — use the swapped cached .hwx)
        ok = ((BOOL (*)(id, SEL, id, id, NSInteger, id*))objc_msgSend)(
            client, NSSelectorFromString(@"loadModel:options:qos:error:"), model, @{}, 0, &err);
        fprintf(stderr, "Load with swapped .hwx: %s\n", ok ? "OK" : "FAILED");

        if (ok) {
            ok = evaluate(client, model, input, channels, emittedOutput, channels);
            fprintf(stderr, "Emitted eval: %s\n", ok ? "OK" : "FAILED");

            ((BOOL (*)(id, SEL, id, id, NSInteger, id*))objc_msgSend)(
                client, NSSelectorFromString(@"doUnloadModel:options:qos:error:"), model, @{}, 0, &err);
        }

        // Restore original
        [backup writeToFile:cacheHWX atomically:NO];
        fprintf(stderr, "Restored original .hwx\n");

        // Compare outputs
        if (ok) {
            printf("=== MEASUREMENT BLOCK ===\n");
            float maxDiff = 0, sumDiff = 0;
            int mismatches = 0;
            for (int i = 0; i < channels; i++) {
                float diff = fabsf(refOutput[i] - emittedOutput[i]);
                if (diff > maxDiff) maxDiff = diff;
                sumDiff += diff;
                if (diff > 1e-3) mismatches++;
            }
            printf("Channels: %d\n", channels);
            printf("Max abs diff: %.6e\n", maxDiff);
            printf("Mean abs diff: %.6e\n", sumDiff / channels);
            printf("Mismatches (>1e-3): %d/%d\n", mismatches, channels);
            printf("Reference first 8: ");
            for (int i = 0; i < 8; i++) printf("%.4f ", refOutput[i]);
            printf("\nEmitted first 8:   ");
            for (int i = 0; i < 8; i++) printf("%.4f ", emittedOutput[i]);
            printf("\n");
        }

        return 0;
    }
}
