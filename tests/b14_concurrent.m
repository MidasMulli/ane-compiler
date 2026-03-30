// b14_concurrent.m — Measure sequential vs concurrent ANE dispatch
//
// Build: xcrun clang -framework Foundation -framework IOSurface -fobjc-arc \
//        -o b14_concurrent b14_concurrent.m
// Run: ./b14_concurrent <modelA.mlmodelc> <modelB.mlmodelc> <iterations>

#import <Foundation/Foundation.h>
#import <dlfcn.h>
#import <signal.h>
#import <IOSurface/IOSurface.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <mach/mach_time.h>
#import <dispatch/dispatch.h>

static Class _C, _M, _R, _IO;
static void loadFW(void) {
    dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);
    _C=NSClassFromString(@"_ANEClient"); _M=NSClassFromString(@"_ANEModel");
    _R=NSClassFromString(@"_ANERequest"); _IO=NSClassFromString(@"_ANEIOSurfaceObject");
}

static IOSurfaceRef mkSurf(uint32_t sz) {
    return IOSurfaceCreate((__bridge CFDictionaryRef)@{
        @"IOSurfaceWidth":@(sz/2),@"IOSurfaceHeight":@1,@"IOSurfaceBytesPerRow":@(sz),
        @"IOSurfaceBytesPerElement":@2,@"IOSurfaceAllocSize":@(sz),@"IOSurfacePixelFormat":@(0x6630304C)});
}

typedef struct {
    id client;
    id model;
    id request;
    IOSurfaceRef inSurf;
    IOSurfaceRef outSurf;
    uint32_t inBS;
    uint32_t outBS;
    uint32_t planeStride;
} ANEContext;

static ANEContext setupModel(id client, NSString *path) {
    ANEContext ctx = {0};
    ctx.client = client;

    NSURL *url = [NSURL fileURLWithPath:path];
    ctx.model = ((id(*)(id,SEL,id,id))objc_msgSend)(
        (id)_M, NSSelectorFromString(@"modelAtURL:key:"), url, @"b14");
    if (!ctx.model) { fprintf(stderr, "model create fail: %s\n", [path UTF8String]); return ctx; }

    ((void(*)(id,SEL,id))objc_msgSend)(client, NSSelectorFromString(@"purgeCompiledModel:"), ctx.model);
    usleep(200000);
    NSError *err = nil;
    ((BOOL(*)(id,SEL,id,id,NSInteger,id*))objc_msgSend)(
        client, NSSelectorFromString(@"compileModel:options:qos:error:"), ctx.model, @{}, 0, &err);
    ((BOOL(*)(id,SEL,id,id,NSInteger,id*))objc_msgSend)(
        client, NSSelectorFromString(@"loadModel:options:qos:error:"), ctx.model, @{}, 0, &err);

    id attrs = ((id(*)(id,SEL))objc_msgSend)(ctx.model, NSSelectorFromString(@"modelAttributes"));
    NSDictionary *ns = [attrs[@"NetworkStatusList"] firstObject];
    ctx.inBS = [ns[@"LiveInputList"][0][@"BatchStride"] unsignedIntValue];
    ctx.outBS = [ns[@"LiveOutputList"][0][@"BatchStride"] unsignedIntValue];
    ctx.planeStride = [ns[@"LiveInputList"][0][@"PlaneStride"] unsignedIntValue];

    ctx.inSurf = mkSurf(ctx.inBS);
    ctx.outSurf = mkSurf(ctx.outBS);

    // Fill input
    IOSurfaceLock(ctx.inSurf, 0, NULL);
    uint16_t *fp16 = (uint16_t *)IOSurfaceGetBaseAddress(ctx.inSurf);
    for (uint32_t i = 0; i < ctx.inBS/2; i++) fp16[i] = 0x3C00 + (i % 256);
    IOSurfaceUnlock(ctx.inSurf, 0, NULL);

    id oI = ((id(*)(id,SEL,void*,NSInteger,BOOL))objc_msgSend)([_IO alloc],
        NSSelectorFromString(@"initWithIOSurface:startOffset:shouldRetain:"), ctx.inSurf, 0, YES);
    id oO = ((id(*)(id,SEL,void*,NSInteger,BOOL))objc_msgSend)([_IO alloc],
        NSSelectorFromString(@"initWithIOSurface:startOffset:shouldRetain:"), ctx.outSurf, 0, YES);

    ctx.request = ((id(*)(id,SEL,id,id,id,id,id,id,id,id,id))objc_msgSend)(
        [_R alloc],
        NSSelectorFromString(@"initWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:sharedEvents:transactionHandle:"),
        @[oI], @[@0], @[oO], @[@0], nil, nil, @(0), nil, nil);

    ((BOOL(*)(id,SEL,id,id,BOOL,id*))objc_msgSend)(
        client, NSSelectorFromString(@"mapIOSurfacesWithModel:request:cacheInference:error:"),
        ctx.model, ctx.request, NO, nil);

    return ctx;
}

static BOOL evalOnce(ANEContext *ctx) {
    return ((BOOL(*)(id,SEL,id,id,id,int,id*))objc_msgSend)(
        ctx->client, NSSelectorFromString(@"doEvaluateDirectWithModel:options:request:qos:error:"),
        ctx->model, @{}, ctx->request, 21, nil);
}

static void cleanup(ANEContext *ctx) {
    ((void(*)(id,SEL,id,id))objc_msgSend)(ctx->client,
        NSSelectorFromString(@"unmapIOSurfacesWithModel:request:"), ctx->model, ctx->request);
    NSError *err = nil;
    ((BOOL(*)(id,SEL,id,id,NSInteger,id*))objc_msgSend)(
        ctx->client, NSSelectorFromString(@"doUnloadModel:options:qos:error:"), ctx->model, @{}, 0, &err);
    if (ctx->inSurf) CFRelease(ctx->inSurf);
    if (ctx->outSurf) CFRelease(ctx->outSurf);
}

int main(int argc, char *argv[]) {
    @autoreleasepool {
        signal(SIGSEGV, SIG_IGN);
        if (argc < 4) {
            fprintf(stderr, "Usage: %s <modelA.mlmodelc> <modelB.mlmodelc> <iterations>\n", argv[0]);
            return 1;
        }
        loadFW();

        NSString *pathA = [NSString stringWithUTF8String:argv[1]];
        NSString *pathB = [NSString stringWithUTF8String:argv[2]];
        int iters = atoi(argv[3]);

        id client = ((id(*)(id,SEL))objc_msgSend)((id)_C, NSSelectorFromString(@"sharedConnection"));

        fprintf(stderr, "Setting up models...\n");
        ANEContext ctxA = setupModel(client, pathA);
        ANEContext ctxB = setupModel(client, pathB);

        if (!ctxA.model || !ctxB.model) {
            fprintf(stderr, "Setup failed\n");
            return 1;
        }

        // Warmup
        for (int i = 0; i < 10; i++) { evalOnce(&ctxA); evalOnce(&ctxB); }

        mach_timebase_info_data_t tb;
        mach_timebase_info(&tb);

        // ============================================================
        // Test 1: Sequential A then B
        // ============================================================
        double *seq_times = malloc(iters * sizeof(double));
        for (int i = 0; i < iters; i++) {
            uint64_t s = mach_absolute_time();
            evalOnce(&ctxA);
            evalOnce(&ctxB);
            uint64_t e = mach_absolute_time();
            seq_times[i] = (double)(e-s) * tb.numer / tb.denom / 1e6;
        }

        // ============================================================
        // Test 2: Concurrent A and B (dispatch_group)
        // ============================================================
        double *con_times = malloc(iters * sizeof(double));
        dispatch_queue_t qA = dispatch_queue_create("ane.a", DISPATCH_QUEUE_SERIAL);
        dispatch_queue_t qB = dispatch_queue_create("ane.b", DISPATCH_QUEUE_SERIAL);

        for (int i = 0; i < iters; i++) {
            dispatch_group_t group = dispatch_group_create();
            uint64_t s = mach_absolute_time();

            dispatch_group_async(group, qA, ^{ evalOnce(&ctxA); });
            dispatch_group_async(group, qB, ^{ evalOnce(&ctxB); });
            dispatch_group_wait(group, DISPATCH_TIME_FOREVER);

            uint64_t e = mach_absolute_time();
            con_times[i] = (double)(e-s) * tb.numer / tb.denom / 1e6;
        }

        // ============================================================
        // Test 3: Single model A only (baseline)
        // ============================================================
        double *single_times = malloc(iters * sizeof(double));
        for (int i = 0; i < iters; i++) {
            uint64_t s = mach_absolute_time();
            evalOnce(&ctxA);
            uint64_t e = mach_absolute_time();
            single_times[i] = (double)(e-s) * tb.numer / tb.denom / 1e6;
        }

        // ============================================================
        // Test 4: Burst — fire N dispatches of A as fast as possible
        // ============================================================
        int burst_sizes[] = {1, 2, 4, 8, 16, 32};
        int n_bursts = 6;

        // Sort
        for (int t = 0; t < 3; t++) {
            double *arr = (t==0) ? seq_times : (t==1) ? con_times : single_times;
            for (int i = 0; i < iters-1; i++)
                for (int j = i+1; j < iters; j++)
                    if (arr[j] < arr[i]) { double tmp=arr[i]; arr[i]=arr[j]; arr[j]=tmp; }
        }

        printf("{\"iters\":%d", iters);
        printf(",\"single_median_ms\":%.4f", single_times[iters/2]);
        printf(",\"sequential_median_ms\":%.4f", seq_times[iters/2]);
        printf(",\"concurrent_median_ms\":%.4f", con_times[iters/2]);
        printf(",\"single_p5\":%.4f,\"single_p95\":%.4f", single_times[iters*5/100], single_times[iters*95/100]);
        printf(",\"sequential_p5\":%.4f,\"sequential_p95\":%.4f", seq_times[iters*5/100], seq_times[iters*95/100]);
        printf(",\"concurrent_p5\":%.4f,\"concurrent_p95\":%.4f", con_times[iters*5/100], con_times[iters*95/100]);

        // Burst test
        printf(",\"burst\":[");
        for (int b = 0; b < n_bursts; b++) {
            int N = burst_sizes[b];
            double burst_times[20];
            int burst_iters = 20;

            for (int i = 0; i < burst_iters; i++) {
                dispatch_group_t g = dispatch_group_create();
                dispatch_queue_t bq = dispatch_queue_create("ane.burst", DISPATCH_QUEUE_CONCURRENT);
                uint64_t s = mach_absolute_time();
                for (int j = 0; j < N; j++) {
                    dispatch_group_async(g, bq, ^{ evalOnce(&ctxA); });
                }
                dispatch_group_wait(g, DISPATCH_TIME_FOREVER);
                uint64_t e = mach_absolute_time();
                burst_times[i] = (double)(e-s) * tb.numer / tb.denom / 1e6;
            }
            // Sort
            for (int i = 0; i < burst_iters-1; i++)
                for (int j = i+1; j < burst_iters; j++)
                    if (burst_times[j] < burst_times[i]) { double t=burst_times[i]; burst_times[i]=burst_times[j]; burst_times[j]=t; }

            printf("%s{\"n\":%d,\"median_ms\":%.4f}", b?",":"", N, burst_times[burst_iters/2]);
        }
        printf("]");

        printf("}\n");

        free(seq_times); free(con_times); free(single_times);
        cleanup(&ctxA); cleanup(&ctxB);
        return 0;
    }
}
