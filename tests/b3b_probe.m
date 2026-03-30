// b3b_probe.m — Two-input elementwise W[51] selector probe
//
// Compiles a 2-input add .mlmodelc, then patches W[51] in the cached .hwx
// to test SUB, DIV, and other undiscovered operations.
//
// Build: xcrun clang -framework Foundation -framework IOSurface -fobjc-arc \
//        -o b3b_probe b3b_probe.m
// Run: ./b3b_probe <model.mlmodelc> <w51_hex_value>
//   e.g., ./b3b_probe /tmp/b3b/add_64.mlmodelc 0x00080010

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

static void fillSurface(IOSurfaceRef surf, float *values, int count, int planeStride) {
    IOSurfaceLock(surf, 0, NULL);
    void *base = IOSurfaceGetBaseAddress(surf);
    uint32_t allocSize = (uint32_t)IOSurfaceGetAllocSize(surf);
    memset(base, 0, allocSize);
    for (int i = 0; i < count; i++) {
        float v = values[i];
        uint32_t bits;
        memcpy(&bits, &v, 4);
        uint32_t sign = (bits >> 31) & 1;
        int32_t exp = ((bits >> 23) & 0xFF) - 127;
        uint32_t mant = bits & 0x7FFFFF;
        uint16_t fp16;
        if (exp > 15) { fp16 = (sign << 15) | 0x7C00; }
        else if (exp < -14) { fp16 = (sign << 15); }
        else { fp16 = (sign << 15) | ((exp + 15) << 10) | (mant >> 13); }
        memcpy((uint8_t*)base + i * planeStride, &fp16, 2);
    }
    IOSurfaceUnlock(surf, 0, NULL);
}

static void readSurface(IOSurfaceRef surf, float *values, int count, int planeStride) {
    IOSurfaceLock(surf, kIOSurfaceLockReadOnly, NULL);
    void *base = IOSurfaceGetBaseAddress(surf);
    for (int i = 0; i < count; i++) {
        uint16_t fp16;
        memcpy(&fp16, (uint8_t*)base + i * planeStride, 2);
        uint32_t sign = (fp16 >> 15) & 1;
        uint32_t exp16 = (fp16 >> 10) & 0x1F;
        uint32_t mant = fp16 & 0x3FF;
        float val;
        if (exp16 == 0) { val = (sign ? -1.0f : 1.0f) * mant / 1024.0f / 16384.0f; }
        else if (exp16 == 31) { val = mant ? NAN : (sign ? -INFINITY : INFINITY); }
        else {
            uint32_t fbits = (sign << 31) | ((exp16 - 15 + 127) << 23) | (mant << 13);
            memcpy(&val, &fbits, 4);
        }
        values[i] = val;
    }
    IOSurfaceUnlock(surf, kIOSurfaceLockReadOnly, NULL);
}

// Find most recent .hwx in ANE cache
static NSString *findRecentHWX(void) {
    NSString *cacheBase = @"/Library/Caches/com.apple.aned";
    NSFileManager *fm = [NSFileManager defaultManager];
    NSDate *cutoff = [NSDate dateWithTimeIntervalSinceNow:-30.0]; // last 30 seconds

    NSMutableArray *candidates = [NSMutableArray array];
    NSDirectoryEnumerator *enumerator = [fm enumeratorAtPath:cacheBase];
    NSString *path;
    while ((path = [enumerator nextObject])) {
        if ([path.pathExtension isEqualToString:@"hwx"]) {
            NSString *full = [cacheBase stringByAppendingPathComponent:path];
            NSDictionary *attrs = [fm attributesOfItemAtPath:full error:nil];
            NSDate *mod = attrs[NSFileModificationDate];
            if ([mod compare:cutoff] == NSOrderedDescending) {
                [candidates addObject:@{@"path": full, @"date": mod}];
            }
        }
    }
    if (candidates.count == 0) return nil;
    [candidates sortUsingComparator:^NSComparisonResult(id a, id b) {
        return [b[@"date"] compare:a[@"date"]];
    }];
    return candidates[0][@"path"];
}

static BOOL eval2Input(id client, id model, float *inputA, float *inputB,
                       int channels, float *output) {
    id attrs = ((id (*)(id, SEL))objc_msgSend)(model, NSSelectorFromString(@"modelAttributes"));
    NSDictionary *ns = [attrs[@"NetworkStatusList"] firstObject];
    NSArray *inputs = ns[@"LiveInputList"];
    NSArray *outputs = ns[@"LiveOutputList"];

    if (inputs.count < 2) {
        fprintf(stderr, "Model has %lu inputs, need 2\n", (unsigned long)inputs.count);
        return NO;
    }

    uint32_t inBatchStride0 = [inputs[0][@"BatchStride"] unsignedIntValue];
    uint32_t inBatchStride1 = [inputs[1][@"BatchStride"] unsignedIntValue];
    uint32_t outBatchStride = [outputs[0][@"BatchStride"] unsignedIntValue];
    uint32_t planeStride = [inputs[0][@"PlaneStride"] unsignedIntValue];

    IOSurfaceRef inSurf0 = makeSurface(inBatchStride0);
    IOSurfaceRef inSurf1 = makeSurface(inBatchStride1);
    IOSurfaceRef outSurf = makeSurface(outBatchStride);

    fillSurface(inSurf0, inputA, channels, planeStride);
    fillSurface(inSurf1, inputB, channels, planeStride);

    id inObj0 = ((id (*)(id, SEL, void*, NSInteger, BOOL))objc_msgSend)(
        [_ANEIOSurfaceObjectCls alloc],
        NSSelectorFromString(@"initWithIOSurface:startOffset:shouldRetain:"),
        inSurf0, 0, YES);
    id inObj1 = ((id (*)(id, SEL, void*, NSInteger, BOOL))objc_msgSend)(
        [_ANEIOSurfaceObjectCls alloc],
        NSSelectorFromString(@"initWithIOSurface:startOffset:shouldRetain:"),
        inSurf1, 0, YES);
    id outObj = ((id (*)(id, SEL, void*, NSInteger, BOOL))objc_msgSend)(
        [_ANEIOSurfaceObjectCls alloc],
        NSSelectorFromString(@"initWithIOSurface:startOffset:shouldRetain:"),
        outSurf, 0, YES);

    id req = ((id (*)(id, SEL, id, id, id, id, id, id, id, id, id))objc_msgSend)(
        [_ANERequestCls alloc],
        NSSelectorFromString(@"initWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:sharedEvents:transactionHandle:"),
        @[inObj0, inObj1], @[@0, @1], @[outObj], @[@0], nil, nil, @(0), nil, nil);

    BOOL mapOK = ((BOOL (*)(id, SEL, id, id, BOOL, id*))objc_msgSend)(
        client, NSSelectorFromString(@"mapIOSurfacesWithModel:request:cacheInference:error:"),
        model, req, NO, nil);
    if (!mapOK) { fprintf(stderr, "map failed\n"); return NO; }

    BOOL evalOK = ((BOOL (*)(id, SEL, id, id, id, int, id*))objc_msgSend)(
        client, NSSelectorFromString(@"doEvaluateDirectWithModel:options:request:qos:error:"),
        model, @{}, req, 21, nil);

    if (evalOK) {
        readSurface(outSurf, output, channels, planeStride);
    }

    ((void (*)(id, SEL, id, id))objc_msgSend)(
        client, NSSelectorFromString(@"unmapIOSurfacesWithModel:request:"),
        model, req);

    CFRelease(inSurf0);
    CFRelease(inSurf1);
    CFRelease(outSurf);

    return evalOK;
}

int main(int argc, char *argv[]) {
    @autoreleasepool {
        signal(SIGSEGV, SIG_IGN);

        if (argc < 3) {
            fprintf(stderr, "Usage: %s <model.mlmodelc> <w51_hex_value>\n", argv[0]);
            fprintf(stderr, "  Known: ADD=0x00080000 MUL=0x00080004 MAX=0x00080008 MIN=0x0008000C\n");
            fprintf(stderr, "  Test:  SUB=0x00080010? DIV=0x00080014?\n");
            return 1;
        }

        loadFW();

        NSString *modelPath = [NSString stringWithUTF8String:argv[1]];
        uint32_t w51Value = (uint32_t)strtoul(argv[2], NULL, 0);

        fprintf(stderr, "Model: %s\n", argv[1]);
        fprintf(stderr, "W[51] = 0x%08X\n", w51Value);

        id client = ((id (*)(id, SEL))objc_msgSend)(
            (id)_ANEClientCls, NSSelectorFromString(@"sharedConnection"));
        NSURL *url = [NSURL fileURLWithPath:modelPath];
        id model = ((id (*)(id, SEL, id, id))objc_msgSend)(
            (id)_ANEModelCls, NSSelectorFromString(@"modelAtURL:key:"), url, @"default");

        if (!model) { fprintf(stderr, "FAIL: model create\n"); return 1; }

        // Purge + compile to get fresh cache entry
        ((void (*)(id, SEL, id))objc_msgSend)(client, NSSelectorFromString(@"purgeCompiledModel:"), model);
        usleep(100000); // 100ms

        NSError *err = nil;
        BOOL ok = ((BOOL (*)(id, SEL, id, id, NSInteger, id*))objc_msgSend)(
            client, NSSelectorFromString(@"compileModel:options:qos:error:"), model, @{}, 0, &err);
        if (!ok) { fprintf(stderr, "FAIL: compile\n"); return 1; }
        fprintf(stderr, "Compile OK\n");

        // Find the cached .hwx
        NSString *cacheHWX = findRecentHWX();
        if (!cacheHWX) {
            fprintf(stderr, "FAIL: could not find cached .hwx\n");
            return 1;
        }
        fprintf(stderr, "Cache .hwx: %s\n", [cacheHWX UTF8String]);

        // Load and evaluate reference (ADD)
        ok = ((BOOL (*)(id, SEL, id, id, NSInteger, id*))objc_msgSend)(
            client, NSSelectorFromString(@"loadModel:options:qos:error:"), model, @{}, 0, &err);
        if (!ok) { fprintf(stderr, "FAIL: load reference\n"); return 1; }

        int channels = 64;
        float inputA[64], inputB[64];
        // Use known test values: A = [3.0, 6.0, 10.0, 2.0, ...], B = [1.0, 2.0, 5.0, 4.0, ...]
        for (int i = 0; i < channels; i++) {
            inputA[i] = 3.0f + (float)(i % 8) * 1.5f;
            inputB[i] = 1.0f + (float)((i + 3) % 7) * 0.5f;
        }

        float refOutput[64] = {0};
        ok = eval2Input(client, model, inputA, inputB, channels, refOutput);
        fprintf(stderr, "Reference (ADD) eval: %s\n", ok ? "OK" : "FAILED");

        // Unload
        ((BOOL (*)(id, SEL, id, id, NSInteger, id*))objc_msgSend)(
            client, NSSelectorFromString(@"doUnloadModel:options:qos:error:"), model, @{}, 0, &err);

        // Backup cache .hwx
        NSData *backup = [NSData dataWithContentsOfFile:cacheHWX];
        if (!backup) { fprintf(stderr, "FAIL: read cached .hwx\n"); return 1; }

        // Find __text offset in .hwx
        const uint8_t *hwxBytes = (const uint8_t *)[backup bytes];
        uint32_t ncmds = *(uint32_t *)(hwxBytes + 0x10);
        uint32_t textOff = 0, textSize = 0;
        uint32_t off = 32;
        for (uint32_t i = 0; i < ncmds; i++) {
            uint32_t cmd = *(uint32_t *)(hwxBytes + off);
            uint32_t cmdsize = *(uint32_t *)(hwxBytes + off + 4);
            if (cmd == 0x19) {
                char segname[17] = {0};
                memcpy(segname, hwxBytes + off + 8, 16);
                uint32_t nsects = *(uint32_t *)(hwxBytes + off + 64);
                for (uint32_t s = 0; s < nsects; s++) {
                    uint32_t soff = off + 72 + s * 80;
                    char sectname[17] = {0};
                    memcpy(sectname, hwxBytes + soff, 16);
                    if (strcmp(segname, "__TEXT") == 0 && strcmp(sectname, "__text") == 0) {
                        textSize = *(uint32_t *)(hwxBytes + soff + 40);
                        textOff = *(uint32_t *)(hwxBytes + soff + 48);
                    }
                }
            }
            off += cmdsize;
        }

        fprintf(stderr, "__text at 0x%04X, size=%u (W[51] at 0x%04X)\n",
                textOff, textSize, textOff + 51 * 4);

        // Verify current W[51] is ADD (0x00080000)
        uint32_t currentW51 = *(uint32_t *)(hwxBytes + textOff + 51 * 4);
        fprintf(stderr, "Current W[51] = 0x%08X (expected 0x00080000)\n", currentW51);

        if (currentW51 != 0x00080000) {
            fprintf(stderr, "WARNING: W[51] is not ADD baseline!\n");
        }

        // Patch W[51] to target value
        NSMutableData *patched = [NSMutableData dataWithData:backup];
        uint8_t *patchBytes = (uint8_t *)[patched mutableBytes];
        *(uint32_t *)(patchBytes + textOff + 51 * 4) = w51Value;

        // Write patched .hwx to cache
        [patched writeToFile:cacheHWX atomically:NO];
        fprintf(stderr, "Patched W[51] to 0x%08X in cache\n", w51Value);

        // Reload + evaluate with patched .hwx
        ok = ((BOOL (*)(id, SEL, id, id, NSInteger, id*))objc_msgSend)(
            client, NSSelectorFromString(@"loadModel:options:qos:error:"), model, @{}, 0, &err);
        fprintf(stderr, "Load with patched .hwx: %s\n", ok ? "OK" : "FAILED");

        float patchedOutput[64] = {0};
        BOOL patchEvalOK = NO;
        if (ok) {
            patchEvalOK = eval2Input(client, model, inputA, inputB, channels, patchedOutput);
            fprintf(stderr, "Patched eval: %s\n", patchEvalOK ? "OK" : "FAILED");

            ((BOOL (*)(id, SEL, id, id, NSInteger, id*))objc_msgSend)(
                client, NSSelectorFromString(@"doUnloadModel:options:qos:error:"), model, @{}, 0, &err);
        }

        // Restore original
        [backup writeToFile:cacheHWX atomically:NO];
        fprintf(stderr, "Restored original .hwx\n");

        // Output results as JSON
        printf("{\"w51\": \"0x%08X\", \"load_ok\": %s, \"eval_ok\": %s",
               w51Value, ok ? "true" : "false", patchEvalOK ? "true" : "false");

        if (patchEvalOK) {
            // Print first 16 values of each
            printf(", \"input_a\": [");
            for (int i = 0; i < 16; i++) printf("%s%.4f", i ? "," : "", inputA[i]);
            printf("], \"input_b\": [");
            for (int i = 0; i < 16; i++) printf("%s%.4f", i ? "," : "", inputB[i]);
            printf("], \"ref_add\": [");
            for (int i = 0; i < 16; i++) printf("%s%.4f", i ? "," : "", refOutput[i]);
            printf("], \"patched_out\": [");
            for (int i = 0; i < 16; i++) printf("%s%.4f", i ? "," : "", patchedOutput[i]);
            printf("]");

            // Compute expected operations
            printf(", \"expected_sub\": [");
            for (int i = 0; i < 16; i++) printf("%s%.4f", i ? "," : "", inputA[i] - inputB[i]);
            printf("], \"expected_div\": [");
            for (int i = 0; i < 16; i++) printf("%s%.4f", i ? "," : "", inputA[i] / inputB[i]);
            printf("], \"expected_pow\": [");
            for (int i = 0; i < 16; i++) printf("%s%.4f", i ? "," : "", powf(inputA[i], inputB[i]));
            printf("]");
        }
        printf("}\n");

        return 0;
    }
}
