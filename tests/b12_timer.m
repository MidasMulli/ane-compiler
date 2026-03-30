// b12_timer.m — Time single-input ANE model dispatch (N iterations)
// Build: xcrun clang -framework Foundation -framework IOSurface -fobjc-arc -o b12_timer b12_timer.m
// Run: ./b12_timer <model.mlmodelc> <iterations>
#import <Foundation/Foundation.h>
#import <dlfcn.h>
#import <signal.h>
#import <IOSurface/IOSurface.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <mach/mach_time.h>

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

int main(int argc, char *argv[]) {
    @autoreleasepool {
        signal(SIGSEGV, SIG_IGN);
        if (argc < 3) { fprintf(stderr, "Usage: %s <model.mlmodelc> <iterations>\n", argv[0]); return 1; }
        loadFW();

        NSString *path = [NSString stringWithUTF8String:argv[1]];
        int iters = atoi(argv[2]);

        id cli = ((id(*)(id,SEL))objc_msgSend)((id)_C, NSSelectorFromString(@"sharedConnection"));
        id mdl = ((id(*)(id,SEL,id,id))objc_msgSend)((id)_M,
            NSSelectorFromString(@"modelAtURL:key:"), [NSURL fileURLWithPath:path], @"timer");
        if (!mdl) { printf("{\"error\":\"model_create\"}\n"); return 0; }

        ((void(*)(id,SEL,id))objc_msgSend)(cli, NSSelectorFromString(@"purgeCompiledModel:"), mdl);
        usleep(200000);
        NSError *err = nil;
        BOOL ok = ((BOOL(*)(id,SEL,id,id,NSInteger,id*))objc_msgSend)(
            cli, NSSelectorFromString(@"compileModel:options:qos:error:"), mdl, @{}, 0, &err);
        if (!ok) { printf("{\"error\":\"compile\"}\n"); return 0; }
        ok = ((BOOL(*)(id,SEL,id,id,NSInteger,id*))objc_msgSend)(
            cli, NSSelectorFromString(@"loadModel:options:qos:error:"), mdl, @{}, 0, &err);
        if (!ok) { printf("{\"error\":\"load\"}\n"); return 0; }

        id attrs = ((id(*)(id,SEL))objc_msgSend)(mdl, NSSelectorFromString(@"modelAttributes"));
        NSDictionary *ns = [attrs[@"NetworkStatusList"] firstObject];
        NSArray *ins = ns[@"LiveInputList"];
        NSArray *outs = ns[@"LiveOutputList"];
        uint32_t inBS = [ins[0][@"BatchStride"] unsignedIntValue];
        uint32_t outBS = [outs[0][@"BatchStride"] unsignedIntValue];

        IOSurfaceRef sI = mkSurf(inBS), sO = mkSurf(outBS);
        // Fill input with random FP16
        IOSurfaceLock(sI, 0, NULL);
        void *base = IOSurfaceGetBaseAddress(sI);
        uint16_t *fp16 = (uint16_t *)base;
        for (uint32_t i = 0; i < inBS/2; i++) fp16[i] = 0x3C00 + (i % 1024); // ~1.0 + noise
        IOSurfaceUnlock(sI, 0, NULL);

        id oI = ((id(*)(id,SEL,void*,NSInteger,BOOL))objc_msgSend)([_IO alloc],
            NSSelectorFromString(@"initWithIOSurface:startOffset:shouldRetain:"), sI, 0, YES);
        id oO = ((id(*)(id,SEL,void*,NSInteger,BOOL))objc_msgSend)([_IO alloc],
            NSSelectorFromString(@"initWithIOSurface:startOffset:shouldRetain:"), sO, 0, YES);
        id req = ((id(*)(id,SEL,id,id,id,id,id,id,id,id,id))objc_msgSend)(
            [_R alloc],
            NSSelectorFromString(@"initWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:sharedEvents:transactionHandle:"),
            @[oI], @[@0], @[oO], @[@0], nil, nil, @(0), nil, nil);
        ((BOOL(*)(id,SEL,id,id,BOOL,id*))objc_msgSend)(
            cli, NSSelectorFromString(@"mapIOSurfacesWithModel:request:cacheInference:error:"),
            mdl, req, NO, nil);

        // Warmup
        for (int i = 0; i < 5; i++) {
            ((BOOL(*)(id,SEL,id,id,id,int,id*))objc_msgSend)(
                cli, NSSelectorFromString(@"doEvaluateDirectWithModel:options:request:qos:error:"),
                mdl, @{}, req, 21, nil);
        }

        // Timed iterations
        mach_timebase_info_data_t timebase;
        mach_timebase_info(&timebase);

        double *times = malloc(iters * sizeof(double));
        for (int i = 0; i < iters; i++) {
            uint64_t start = mach_absolute_time();
            ((BOOL(*)(id,SEL,id,id,id,int,id*))objc_msgSend)(
                cli, NSSelectorFromString(@"doEvaluateDirectWithModel:options:request:qos:error:"),
                mdl, @{}, req, 21, nil);
            uint64_t end = mach_absolute_time();
            times[i] = (double)(end - start) * timebase.numer / timebase.denom / 1e6; // ms
        }

        // Sort for percentiles
        for (int i = 0; i < iters - 1; i++)
            for (int j = i + 1; j < iters; j++)
                if (times[j] < times[i]) { double t = times[i]; times[i] = times[j]; times[j] = t; }

        double sum = 0;
        for (int i = 0; i < iters; i++) sum += times[i];

        printf("{\"model\":\"%s\",\"iters\":%d,\"median_ms\":%.4f,\"mean_ms\":%.4f,"
               "\"p5_ms\":%.4f,\"p95_ms\":%.4f,\"min_ms\":%.4f,\"max_ms\":%.4f}\n",
               argv[1], iters,
               times[iters/2], sum/iters,
               times[iters*5/100], times[iters*95/100],
               times[0], times[iters-1]);

        free(times);

        ((void(*)(id,SEL,id,id))objc_msgSend)(cli,
            NSSelectorFromString(@"unmapIOSurfacesWithModel:request:"), mdl, req);
        ((BOOL(*)(id,SEL,id,id,NSInteger,id*))objc_msgSend)(
            cli, NSSelectorFromString(@"doUnloadModel:options:qos:error:"), mdl, @{}, 0, &err);

        CFRelease(sI); CFRelease(sO);
        return 0;
    }
}
