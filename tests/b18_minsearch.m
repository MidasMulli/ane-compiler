// b18_minsearch.m — Fine-grained minimum IOSurface size search
// Build: xcrun clang -framework Foundation -framework IOSurface -fobjc-arc -o b18_minsearch b18_minsearch.m

#import <Foundation/Foundation.h>
#import <dlfcn.h>
#import <signal.h>
#import <IOSurface/IOSurface.h>
#import <objc/runtime.h>
#import <objc/message.h>

static Class _ANEClientCls, _ANEModelCls, _ANERequestCls, _ANEIOSurfaceObjectCls;
static id g_client, g_model;
static uint32_t g_inBS, g_outBS, g_ps;

static void loadFW(void) {
    dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);
    _ANEClientCls = NSClassFromString(@"_ANEClient");
    _ANEModelCls = NSClassFromString(@"_ANEModel");
    _ANERequestCls = NSClassFromString(@"_ANERequest");
    _ANEIOSurfaceObjectCls = NSClassFromString(@"_ANEIOSurfaceObject");
}

static int trySize(uint32_t sz, int channels) {
    NSDictionary *p = @{@"IOSurfaceWidth":@(sz), @"IOSurfaceHeight":@1,
        @"IOSurfaceBytesPerRow":@(sz), @"IOSurfaceBytesPerElement":@1,
        @"IOSurfaceAllocSize":@(sz), @"IOSurfacePixelFormat":@0};

    IOSurfaceRef inSurf = IOSurfaceCreate((__bridge CFDictionaryRef)p);
    IOSurfaceRef outSurf = IOSurfaceCreate((__bridge CFDictionaryRef)p);
    if (!inSurf || !outSurf) {
        if (inSurf) CFRelease(inSurf);
        if (outSurf) CFRelease(outSurf);
        return 0;
    }

    IOSurfaceLock(inSurf, 0, NULL);
    void *base = IOSurfaceGetBaseAddress(inSurf);
    size_t allocSz = IOSurfaceGetAllocSize(inSurf);
    memset(base, 0, allocSz);
    uint16_t one = 0x3C00;
    for (int i = 0; i < channels && (i * g_ps + 2) <= allocSz; i++)
        memcpy((uint8_t*)base + i * g_ps, &one, 2);
    IOSurfaceUnlock(inSurf, 0, NULL);

    id inObj = ((id (*)(id, SEL, void*, NSInteger, BOOL))objc_msgSend)(
        [_ANEIOSurfaceObjectCls alloc],
        NSSelectorFromString(@"initWithIOSurface:startOffset:shouldRetain:"),
        inSurf, 0, YES);
    id outObj = ((id (*)(id, SEL, void*, NSInteger, BOOL))objc_msgSend)(
        [_ANEIOSurfaceObjectCls alloc],
        NSSelectorFromString(@"initWithIOSurface:startOffset:shouldRetain:"),
        outSurf, 0, YES);

    if (!inObj || !outObj) { CFRelease(inSurf); CFRelease(outSurf); return 1; }

    id req = ((id (*)(id, SEL, id, id, id, id, id, id, id, id, id))objc_msgSend)(
        [_ANERequestCls alloc],
        NSSelectorFromString(@"initWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:sharedEvents:transactionHandle:"),
        @[inObj], @[@0], @[outObj], @[@0], nil, nil, @(0), nil, nil);

    if (!req) { CFRelease(inSurf); CFRelease(outSurf); return 1; }

    BOOL mapOK = ((BOOL (*)(id, SEL, id, id, BOOL, id*))objc_msgSend)(
        g_client, NSSelectorFromString(@"mapIOSurfacesWithModel:request:cacheInference:error:"),
        g_model, req, NO, nil);
    if (!mapOK) { CFRelease(inSurf); CFRelease(outSurf); return 1; }

    BOOL evalOK = ((BOOL (*)(id, SEL, id, id, id, int, id*))objc_msgSend)(
        g_client, NSSelectorFromString(@"doEvaluateDirectWithModel:options:request:qos:error:"),
        g_model, @{}, req, 21, nil);

    ((void (*)(id, SEL, id, id))objc_msgSend)(
        g_client, NSSelectorFromString(@"unmapIOSurfacesWithModel:request:"), g_model, req);

    if (!evalOK) { CFRelease(inSurf); CFRelease(outSurf); return 1; }

    IOSurfaceLock(outSurf, kIOSurfaceLockReadOnly, NULL);
    void *outBase = IOSurfaceGetBaseAddress(outSurf);
    size_t outAllocSz = IOSurfaceGetAllocSize(outSurf);
    int correct = 0;
    for (int i = 0; i < channels && (i * g_ps + 2) <= outAllocSz; i++) {
        uint16_t val;
        memcpy(&val, (uint8_t*)outBase + i * g_ps, 2);
        if (val == 0x3C00) correct++;
    }
    IOSurfaceUnlock(outSurf, kIOSurfaceLockReadOnly, NULL);

    CFRelease(inSurf);
    CFRelease(outSurf);
    return (correct == channels) ? 3 : 2;
}

int main(int argc, char *argv[]) {
    @autoreleasepool {
        signal(SIGSEGV, SIG_IGN);
        if (argc < 2) { fprintf(stderr, "Usage: %s <model.mlmodelc> [channels]\n", argv[0]); return 1; }
        loadFW();
        int channels = argc > 2 ? atoi(argv[2]) : 64;

        g_client = ((id (*)(id, SEL))objc_msgSend)((id)_ANEClientCls, NSSelectorFromString(@"sharedConnection"));
        NSURL *url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:argv[1]]];
        g_model = ((id (*)(id, SEL, id, id))objc_msgSend)((id)_ANEModelCls, NSSelectorFromString(@"modelAtURL:key:"), url, @"default");
        NSError *err = nil;
        ((BOOL (*)(id, SEL, id, id, NSInteger, id*))objc_msgSend)(
            g_client, NSSelectorFromString(@"compileModel:options:qos:error:"), g_model, @{}, 0, &err);
        ((BOOL (*)(id, SEL, id, id, NSInteger, id*))objc_msgSend)(
            g_client, NSSelectorFromString(@"loadModel:options:qos:error:"), g_model, @{}, 0, &err);

        id attrs = ((id (*)(id, SEL))objc_msgSend)(g_model, NSSelectorFromString(@"modelAttributes"));
        NSDictionary *ns = [attrs[@"NetworkStatusList"] firstObject];
        g_inBS = [[ns[@"LiveInputList"] firstObject][@"BatchStride"] unsignedIntValue];
        g_outBS = [[ns[@"LiveOutputList"] firstObject][@"BatchStride"] unsignedIntValue];
        g_ps = [[ns[@"LiveInputList"] firstObject][@"PlaneStride"] unsignedIntValue];

        printf("Model: inBS=%u outBS=%u ps=%u channels=%d\n", g_inBS, g_outBS, g_ps, channels);
        printf("Data requirement: %u bytes (channels * ps)\n\n", channels * g_ps);

        // Binary search between 4096 and 8192
        printf("=== Binary search: 4096-8192 (256-byte steps) ===\n");
        for (uint32_t sz = 4096; sz <= 8192; sz += 256) {
            int r = trySize(sz, channels);
            const char *rs = r == 0 ? "ALLOC_FAIL" : r == 1 ? "DISPATCH_FAIL" : r == 2 ? "WRONG" : "PASS";
            printf("  %5u: %s\n", sz, rs);
        }

        // Fine search around transition
        printf("\n=== Fine search: 2048-5120 (page-aligned, 256-byte steps) ===\n");
        for (uint32_t sz = 2048; sz <= 5120; sz += 256) {
            int r = trySize(sz, channels);
            const char *rs = r == 0 ? "ALLOC_FAIL" : r == 1 ? "DISPATCH_FAIL" : r == 2 ? "WRONG" : "PASS";
            printf("  %5u: %s\n", sz, rs);
        }

        // Test actual IOSurface alloc sizes (the kernel may round up)
        printf("\n=== Actual alloc sizes (kernel rounding) ===\n");
        for (uint32_t req = 1024; req <= 16384; req *= 2) {
            NSDictionary *p = @{@"IOSurfaceWidth":@(req), @"IOSurfaceHeight":@1,
                @"IOSurfaceBytesPerRow":@(req), @"IOSurfaceBytesPerElement":@1,
                @"IOSurfaceAllocSize":@(req), @"IOSurfacePixelFormat":@0};
            IOSurfaceRef s = IOSurfaceCreate((__bridge CFDictionaryRef)p);
            if (s) {
                size_t actual = IOSurfaceGetAllocSize(s);
                void *base = IOSurfaceGetBaseAddress(s);
                printf("  requested=%5u actual=%5zu base=%p (page-aligned=%s)\n",
                    req, actual, base, ((uintptr_t)base % 16384 == 0) ? "16K" :
                    ((uintptr_t)base % 4096 == 0) ? "4K" : "NO");
                CFRelease(s);
            } else {
                printf("  requested=%5u ALLOC_FAIL\n", req);
            }
        }

        ((BOOL (*)(id, SEL, id, id, NSInteger, id*))objc_msgSend)(
            g_client, NSSelectorFromString(@"doUnloadModel:options:qos:error:"), g_model, @{}, 0, &err);
        return 0;
    }
}
