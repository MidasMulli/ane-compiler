// ane_raw_dispatch.m — Minimal raw dispatch latency measurement
//
// Measures the absolute minimum dispatch latency achievable through
// the _ANEClient framework path, eliminating all pipe tool overhead.
//
// Build: xcrun clang -framework Foundation -framework IOSurface -fobjc-arc \
//        -O2 -o /tmp/ane_raw_dispatch /tmp/ane_raw_dispatch.m

#import <Foundation/Foundation.h>
#import <dlfcn.h>
#import <signal.h>
#import <IOSurface/IOSurface.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <mach/mach_time.h>

static int cmp_double(const void *a, const void *b) {
    double d = *(double*)a - *(double*)b;
    return d < 0 ? -1 : d > 0 ? 1 : 0;
}

static Class _Cl, _Mo, _Rq, _IO;
static void loadFW(void) {
    dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);
    _Cl = NSClassFromString(@"_ANEClient");
    _Mo = NSClassFromString(@"_ANEModel");
    _Rq = NSClassFromString(@"_ANERequest");
    _IO = NSClassFromString(@"_ANEIOSurfaceObject");
}

int main(int argc, char *argv[]) {
    @autoreleasepool {
        signal(SIGSEGV, SIG_IGN);

        if (argc < 2) {
            fprintf(stderr, "Usage: %s <model.mlmodelc> [in_ch] [out_ch] [iterations]\n", argv[0]);
            return 1;
        }

        loadFW();

        NSString *modelPath = [NSString stringWithUTF8String:argv[1]];
        int inCh = argc > 2 ? atoi(argv[2]) : 64;
        int outCh = argc > 3 ? atoi(argv[3]) : inCh;
        int iterations = argc > 4 ? atoi(argv[4]) : 1000;

        // Get timebase info
        mach_timebase_info_data_t timebase;
        mach_timebase_info(&timebase);

        id client = ((id (*)(id, SEL))objc_msgSend)((id)_Cl, NSSelectorFromString(@"sharedConnection"));

        NSURL *url = [NSURL fileURLWithPath:modelPath];
        id model = ((id (*)(id, SEL, id, id))objc_msgSend)(
            (id)_Mo, NSSelectorFromString(@"modelAtURL:key:"), url, @"default");
        if (!model) { fprintf(stderr, "Model create failed\n"); return 1; }

        NSError *err = nil;

        // Compile + load
        ((BOOL (*)(id, SEL, id, id, NSInteger, id*))objc_msgSend)(
            client, NSSelectorFromString(@"compileModel:options:qos:error:"),
            model, @{}, 0, &err);

        BOOL loadOK = ((BOOL (*)(id, SEL, id, id, NSInteger, id*))objc_msgSend)(
            client, NSSelectorFromString(@"loadModel:options:qos:error:"),
            model, @{}, 0, &err);
        if (!loadOK) { fprintf(stderr, "Load failed: %s\n", err ? [[err description] UTF8String] : "nil"); return 1; }

        // Get buffer info
        id attrs = ((id (*)(id, SEL))objc_msgSend)(model, NSSelectorFromString(@"modelAttributes"));
        NSDictionary *ns = [attrs[@"NetworkStatusList"] firstObject];
        uint32_t inBS = [[ns[@"LiveInputList"] firstObject][@"BatchStride"] unsignedIntValue];
        uint32_t outBS = [[ns[@"LiveOutputList"] firstObject][@"BatchStride"] unsignedIntValue];
        uint32_t inPS = [[ns[@"LiveInputList"] firstObject][@"PlaneStride"] unsignedIntValue];
        uint32_t outPS = [[ns[@"LiveOutputList"] firstObject][@"PlaneStride"] unsignedIntValue];

        fprintf(stderr, "Model loaded: inBS=%u outBS=%u inPS=%u outPS=%u\n", inBS, outBS, inPS, outPS);

        // Pre-allocate IOSurfaces
        NSDictionary *inProps = @{@"IOSurfaceWidth":@(inBS/2), @"IOSurfaceHeight":@1,
            @"IOSurfaceBytesPerRow":@(inBS), @"IOSurfaceBytesPerElement":@2,
            @"IOSurfaceAllocSize":@(inBS), @"IOSurfacePixelFormat":@(0x6630304C)};
        NSDictionary *outProps = @{@"IOSurfaceWidth":@(outBS/2), @"IOSurfaceHeight":@1,
            @"IOSurfaceBytesPerRow":@(outBS), @"IOSurfaceBytesPerElement":@2,
            @"IOSurfaceAllocSize":@(outBS), @"IOSurfacePixelFormat":@(0x6630304C)};

        IOSurfaceRef inSurf = IOSurfaceCreate((__bridge CFDictionaryRef)inProps);
        IOSurfaceRef outSurf = IOSurfaceCreate((__bridge CFDictionaryRef)outProps);

        id inObj = ((id (*)(id, SEL, void*, NSInteger, BOOL))objc_msgSend)(
            [_IO alloc], NSSelectorFromString(@"initWithIOSurface:startOffset:shouldRetain:"),
            inSurf, 0, YES);
        id outObj = ((id (*)(id, SEL, void*, NSInteger, BOOL))objc_msgSend)(
            [_IO alloc], NSSelectorFromString(@"initWithIOSurface:startOffset:shouldRetain:"),
            outSurf, 0, YES);

        // Fill input with random data once
        IOSurfaceLock(inSurf, 0, NULL);
        void *inBase = IOSurfaceGetBaseAddress(inSurf);
        for (int j = 0; j < inCh; j++) {
            uint16_t val = 0x3C00; // 1.0 in FP16
            memcpy((uint8_t*)inBase + j * inPS, &val, 2);
        }
        IOSurfaceUnlock(inSurf, 0, NULL);

        // Pre-build request ONCE
        id req = ((id (*)(id, SEL, id, id, id, id, id, id, id, id, id))objc_msgSend)(
            [_Rq alloc],
            NSSelectorFromString(@"initWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:sharedEvents:transactionHandle:"),
            @[inObj], @[@0], @[outObj], @[@0], nil, nil, @(0), nil, nil);

        // Map ONCE
        ((BOOL (*)(id, SEL, id, id, BOOL, id*))objc_msgSend)(
            client, NSSelectorFromString(@"mapIOSurfacesWithModel:request:cacheInference:error:"),
            model, req, NO, nil);

        fprintf(stderr, "Starting %d dispatches...\n", iterations);

        // Warmup
        for (int i = 0; i < 10; i++) {
            ((BOOL (*)(id, SEL, id, id, id, int, id*))objc_msgSend)(
                client, NSSelectorFromString(@"doEvaluateDirectWithModel:options:request:qos:error:"),
                model, @{}, req, 21, nil);
        }

        // === Measurement 1: Full dispatch including map/unmap ===
        double *lats_full = calloc(iterations, sizeof(double));
        for (int i = 0; i < iterations; i++) {
            // Remap each time (like the pipe tool does)
            ((void (*)(id, SEL, id, id))objc_msgSend)(
                client, NSSelectorFromString(@"unmapIOSurfacesWithModel:request:"), model, req);
            ((BOOL (*)(id, SEL, id, id, BOOL, id*))objc_msgSend)(
                client, NSSelectorFromString(@"mapIOSurfacesWithModel:request:cacheInference:error:"),
                model, req, NO, nil);

            uint64_t t0 = mach_absolute_time();
            BOOL ok = ((BOOL (*)(id, SEL, id, id, id, int, id*))objc_msgSend)(
                client, NSSelectorFromString(@"doEvaluateDirectWithModel:options:request:qos:error:"),
                model, @{}, req, 21, nil);
            uint64_t t1 = mach_absolute_time();

            if (!ok) { fprintf(stderr, "Eval failed at iteration %d\n", i); return 1; }
            lats_full[i] = (double)(t1 - t0) * timebase.numer / timebase.denom / 1000.0; // microseconds
        }

        // === Measurement 2: Pure dispatch (no remap) ===
        // Map once, dispatch many
        ((BOOL (*)(id, SEL, id, id, BOOL, id*))objc_msgSend)(
            client, NSSelectorFromString(@"mapIOSurfacesWithModel:request:cacheInference:error:"),
            model, req, NO, nil);

        double *lats_pure = calloc(iterations, sizeof(double));
        for (int i = 0; i < iterations; i++) {
            uint64_t t0 = mach_absolute_time();
            BOOL ok = ((BOOL (*)(id, SEL, id, id, id, int, id*))objc_msgSend)(
                client, NSSelectorFromString(@"doEvaluateDirectWithModel:options:request:qos:error:"),
                model, @{}, req, 21, nil);
            uint64_t t1 = mach_absolute_time();

            if (!ok) { fprintf(stderr, "Eval failed at pure iteration %d\n", i); return 1; }
            lats_pure[i] = (double)(t1 - t0) * timebase.numer / timebase.denom / 1000.0;
        }

        // === Measurement 3: IOKit direct timing via DTrace marker ===
        // Skip — need DTrace for that

        // Sort and print stats
        qsort(lats_full, iterations, sizeof(double), cmp_double);
        qsort(lats_pure, iterations, sizeof(double), cmp_double);

        double sum_full = 0, sum_pure = 0;
        for (int i = 0; i < iterations; i++) { sum_full += lats_full[i]; sum_pure += lats_pure[i]; }

        printf("=== Full dispatch (with remap per call) — %d iterations ===\n", iterations);
        printf("  min:  %.1f us\n", lats_full[0]);
        printf("  p50:  %.1f us\n", lats_full[iterations/2]);
        printf("  p95:  %.1f us\n", lats_full[(int)(iterations*0.95)]);
        printf("  p99:  %.1f us\n", lats_full[(int)(iterations*0.99)]);
        printf("  mean: %.1f us\n", sum_full / iterations);
        printf("  max:  %.1f us\n", lats_full[iterations-1]);

        printf("\n=== Pure dispatch (single map, no remap) — %d iterations ===\n", iterations);
        printf("  min:  %.1f us\n", lats_pure[0]);
        printf("  p50:  %.1f us\n", lats_pure[iterations/2]);
        printf("  p95:  %.1f us\n", lats_pure[(int)(iterations*0.95)]);
        printf("  p99:  %.1f us\n", lats_pure[(int)(iterations*0.99)]);
        printf("  mean: %.1f us\n", sum_pure / iterations);
        printf("  max:  %.1f us\n", lats_pure[iterations-1]);

        // Verify output
        IOSurfaceLock(outSurf, kIOSurfaceLockReadOnly, NULL);
        void *outBase = IOSurfaceGetBaseAddress(outSurf);
        printf("\nOutput sample (first 5 channels): ");
        for (int j = 0; j < 5 && j < outCh; j++) {
            uint16_t val;
            memcpy(&val, (uint8_t*)outBase + j * outPS, 2);
            // FP16 to float
            int sign = (val >> 15) & 1;
            int exp = (val >> 10) & 0x1f;
            int mant = val & 0x3ff;
            float f;
            if (exp == 0) f = (sign ? -1 : 1) * (mant / 1024.0f) * (1.0f / 16384.0f);
            else if (exp == 31) f = (sign ? -1 : 1) * (mant ? NAN : INFINITY);
            else f = (sign ? -1 : 1) * (1.0f + mant / 1024.0f) * powf(2.0f, exp - 15);
            printf("%.4f ", f);
        }
        printf("\n");
        IOSurfaceUnlock(outSurf, kIOSurfaceLockReadOnly, NULL);

        // Cleanup
        ((void (*)(id, SEL, id, id))objc_msgSend)(
            client, NSSelectorFromString(@"unmapIOSurfacesWithModel:request:"), model, req);
        ((BOOL (*)(id, SEL, id, id, NSInteger, id*))objc_msgSend)(
            client, NSSelectorFromString(@"doUnloadModel:options:qos:error:"),
            model, @{}, 0, &err);
        CFRelease(inSurf);
        CFRelease(outSurf);

        free(lats_full);
        free(lats_pure);
        return 0;
    }
}
