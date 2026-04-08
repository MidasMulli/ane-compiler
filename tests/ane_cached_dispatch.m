// ane_cached_dispatch.m — Test cached inference mode
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
        if (argc < 2) { fprintf(stderr, "Usage: %s <model.mlmodelc> [ch] [iters]\n", argv[0]); return 1; }
        loadFW();

        int ch = argc > 2 ? atoi(argv[2]) : 768;
        int iters = argc > 3 ? atoi(argv[3]) : 2000;

        mach_timebase_info_data_t tb;
        mach_timebase_info(&tb);

        id client = ((id (*)(id, SEL))objc_msgSend)((id)_Cl, NSSelectorFromString(@"sharedConnection"));
        NSURL *url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:argv[1]]];
        id model = ((id (*)(id, SEL, id, id))objc_msgSend)(
            (id)_Mo, NSSelectorFromString(@"modelAtURL:key:"), url, @"default");

        NSError *err = nil;
        ((BOOL (*)(id, SEL, id, id, NSInteger, id*))objc_msgSend)(
            client, NSSelectorFromString(@"compileModel:options:qos:error:"), model, @{}, 0, &err);
        ((BOOL (*)(id, SEL, id, id, NSInteger, id*))objc_msgSend)(
            client, NSSelectorFromString(@"loadModel:options:qos:error:"), model, @{}, 0, &err);

        id attrs = ((id (*)(id, SEL))objc_msgSend)(model, NSSelectorFromString(@"modelAttributes"));
        NSDictionary *ns = [attrs[@"NetworkStatusList"] firstObject];
        uint32_t inBS = [[ns[@"LiveInputList"] firstObject][@"BatchStride"] unsignedIntValue];
        uint32_t outBS = [[ns[@"LiveOutputList"] firstObject][@"BatchStride"] unsignedIntValue];

        IOSurfaceRef inSurf = IOSurfaceCreate((__bridge CFDictionaryRef)@{
            @"IOSurfaceWidth":@(inBS/2), @"IOSurfaceHeight":@1,
            @"IOSurfaceBytesPerRow":@(inBS), @"IOSurfaceBytesPerElement":@2,
            @"IOSurfaceAllocSize":@(inBS), @"IOSurfacePixelFormat":@(0x6630304C)});
        IOSurfaceRef outSurf = IOSurfaceCreate((__bridge CFDictionaryRef)@{
            @"IOSurfaceWidth":@(outBS/2), @"IOSurfaceHeight":@1,
            @"IOSurfaceBytesPerRow":@(outBS), @"IOSurfaceBytesPerElement":@2,
            @"IOSurfaceAllocSize":@(outBS), @"IOSurfacePixelFormat":@(0x6630304C)});

        id inObj = ((id (*)(id, SEL, void*, NSInteger, BOOL))objc_msgSend)(
            [_IO alloc], NSSelectorFromString(@"initWithIOSurface:startOffset:shouldRetain:"), inSurf, 0, YES);
        id outObj = ((id (*)(id, SEL, void*, NSInteger, BOOL))objc_msgSend)(
            [_IO alloc], NSSelectorFromString(@"initWithIOSurface:startOffset:shouldRetain:"), outSurf, 0, YES);

        id req = ((id (*)(id, SEL, id, id, id, id, id, id, id, id, id))objc_msgSend)(
            [_Rq alloc],
            NSSelectorFromString(@"initWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:sharedEvents:transactionHandle:"),
            @[inObj], @[@0], @[outObj], @[@0], nil, nil, @(0), nil, nil);

        // === Test 1: Map with cacheInference:YES, then dispatch many ===
        ((BOOL (*)(id, SEL, id, id, BOOL, id*))objc_msgSend)(
            client, NSSelectorFromString(@"mapIOSurfacesWithModel:request:cacheInference:error:"),
            model, req, YES, nil);

        // Warmup
        for (int i = 0; i < 50; i++) {
            ((BOOL (*)(id, SEL, id, id, id, int, id*))objc_msgSend)(
                client, NSSelectorFromString(@"doEvaluateDirectWithModel:options:request:qos:error:"),
                model, @{}, req, 21, nil);
        }

        double *lats = calloc(iters, sizeof(double));
        for (int i = 0; i < iters; i++) {
            uint64_t t0 = mach_absolute_time();
            ((BOOL (*)(id, SEL, id, id, id, int, id*))objc_msgSend)(
                client, NSSelectorFromString(@"doEvaluateDirectWithModel:options:request:qos:error:"),
                model, @{}, req, 21, nil);
            uint64_t t1 = mach_absolute_time();
            lats[i] = (double)(t1 - t0) * tb.numer / tb.denom / 1000.0;
        }

        qsort(lats, iters, sizeof(double), cmp_double);
        double sum = 0;
        for (int i = 0; i < iters; i++) sum += lats[i];

        printf("=== Cached inference (map once, cacheInference=YES) — %d iters ===\n", iters);
        printf("  min:  %.1f us\n", lats[0]);
        printf("  p50:  %.1f us\n", lats[iters/2]);
        printf("  p95:  %.1f us\n", lats[(int)(iters*0.95)]);
        printf("  p99:  %.1f us\n", lats[(int)(iters*0.99)]);
        printf("  mean: %.1f us\n", sum / iters);
        printf("  max:  %.1f us\n", lats[iters-1]);

        // === Test 2: map+eval+unmap each time ===
        ((void (*)(id, SEL, id, id))objc_msgSend)(
            client, NSSelectorFromString(@"unmapIOSurfacesWithModel:request:"), model, req);

        double *lats2 = calloc(iters, sizeof(double));
        for (int i = 0; i < iters; i++) {
            ((BOOL (*)(id, SEL, id, id, BOOL, id*))objc_msgSend)(
                client, NSSelectorFromString(@"mapIOSurfacesWithModel:request:cacheInference:error:"),
                model, req, NO, nil);
            uint64_t t0 = mach_absolute_time();
            ((BOOL (*)(id, SEL, id, id, id, int, id*))objc_msgSend)(
                client, NSSelectorFromString(@"doEvaluateDirectWithModel:options:request:qos:error:"),
                model, @{}, req, 21, nil);
            uint64_t t1 = mach_absolute_time();
            ((void (*)(id, SEL, id, id))objc_msgSend)(
                client, NSSelectorFromString(@"unmapIOSurfacesWithModel:request:"), model, req);
            lats2[i] = (double)(t1 - t0) * tb.numer / tb.denom / 1000.0;
        }

        qsort(lats2, iters, sizeof(double), cmp_double);
        sum = 0;
        for (int i = 0; i < iters; i++) sum += lats2[i];

        printf("\n=== Remap each time (cacheInference=NO) — %d iters ===\n", iters);
        printf("  min:  %.1f us\n", lats2[0]);
        printf("  p50:  %.1f us\n", lats2[iters/2]);
        printf("  p95:  %.1f us\n", lats2[(int)(iters*0.95)]);
        printf("  p99:  %.1f us\n", lats2[(int)(iters*0.99)]);
        printf("  mean: %.1f us\n", sum / iters);
        printf("  max:  %.1f us\n", lats2[iters-1]);

        // === Test 3: FULL path including map overhead in timing ===
        double *lats3 = calloc(iters, sizeof(double));
        for (int i = 0; i < iters; i++) {
            uint64_t t0 = mach_absolute_time();
            ((BOOL (*)(id, SEL, id, id, BOOL, id*))objc_msgSend)(
                client, NSSelectorFromString(@"mapIOSurfacesWithModel:request:cacheInference:error:"),
                model, req, NO, nil);
            ((BOOL (*)(id, SEL, id, id, id, int, id*))objc_msgSend)(
                client, NSSelectorFromString(@"doEvaluateDirectWithModel:options:request:qos:error:"),
                model, @{}, req, 21, nil);
            ((void (*)(id, SEL, id, id))objc_msgSend)(
                client, NSSelectorFromString(@"unmapIOSurfacesWithModel:request:"), model, req);
            uint64_t t1 = mach_absolute_time();
            lats3[i] = (double)(t1 - t0) * tb.numer / tb.denom / 1000.0;
        }

        qsort(lats3, iters, sizeof(double), cmp_double);
        sum = 0;
        for (int i = 0; i < iters; i++) sum += lats3[i];

        printf("\n=== Full map+eval+unmap timed together — %d iters ===\n", iters);
        printf("  min:  %.1f us\n", lats3[0]);
        printf("  p50:  %.1f us\n", lats3[iters/2]);
        printf("  p95:  %.1f us\n", lats3[(int)(iters*0.95)]);
        printf("  p99:  %.1f us\n", lats3[(int)(iters*0.99)]);
        printf("  mean: %.1f us\n", sum / iters);
        printf("  max:  %.1f us\n", lats3[iters-1]);

        // Cleanup
        ((BOOL (*)(id, SEL, id, id, NSInteger, id*))objc_msgSend)(
            client, NSSelectorFromString(@"doUnloadModel:options:qos:error:"), model, @{}, 0, &err);
        CFRelease(inSurf); CFRelease(outSurf);
        free(lats); free(lats2); free(lats3);
        return 0;
    }
}
