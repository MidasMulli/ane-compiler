// b18_iosurface.m — IOSurface alignment/size constraint probing for ANE dispatch
//
// Tests: minimum alloc size, alignment requirements, BytesPerRow, pixel formats,
// Width/Height properties, and non-standard configurations.
//
// Build: xcrun clang -framework Foundation -framework IOSurface -fobjc-arc \
//        -o b18_iosurface b18_iosurface.m
// Run:   ./b18_iosurface <relu_model.mlmodelc> [channels=64]

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

// Load and prepare model, return client + model
static id g_client = nil;
static id g_model = nil;

static BOOL prepareModel(const char *path, int channels) {
    NSURL *url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:path]];
    g_client = ((id (*)(id, SEL))objc_msgSend)((id)_ANEClientCls, NSSelectorFromString(@"sharedConnection"));
    g_model = ((id (*)(id, SEL, id, id))objc_msgSend)((id)_ANEModelCls, NSSelectorFromString(@"modelAtURL:key:"), url, @"default");
    if (!g_model) return NO;

    NSError *err = nil;
    ((BOOL (*)(id, SEL, id, id, NSInteger, id*))objc_msgSend)(
        g_client, NSSelectorFromString(@"compileModel:options:qos:error:"), g_model, @{}, 0, &err);
    BOOL loadOK = ((BOOL (*)(id, SEL, id, id, NSInteger, id*))objc_msgSend)(
        g_client, NSSelectorFromString(@"loadModel:options:qos:error:"), g_model, @{}, 0, &err);
    return loadOK;
}

// Get BatchStride and PlaneStride from model attributes
static uint32_t g_inBS = 0, g_outBS = 0, g_ps = 0;
static void getModelStrides(void) {
    id attrs = ((id (*)(id, SEL))objc_msgSend)(g_model, NSSelectorFromString(@"modelAttributes"));
    NSDictionary *ns = [attrs[@"NetworkStatusList"] firstObject];
    g_inBS = [[ns[@"LiveInputList"] firstObject][@"BatchStride"] unsignedIntValue];
    g_outBS = [[ns[@"LiveOutputList"] firstObject][@"BatchStride"] unsignedIntValue];
    g_ps = [[ns[@"LiveInputList"] firstObject][@"PlaneStride"] unsignedIntValue];
    fprintf(stderr, "Model strides: inBS=%u outBS=%u ps=%u\n", g_inBS, g_outBS, g_ps);
}

// Try to dispatch with custom IOSurface properties
// Returns: 0=alloc_fail, 1=dispatch_fail, 2=dispatch_ok_wrong, 3=dispatch_ok_correct
static int tryDispatch(NSDictionary *inProps, NSDictionary *outProps, int channels) {
    IOSurfaceRef inSurf = IOSurfaceCreate((__bridge CFDictionaryRef)inProps);
    IOSurfaceRef outSurf = IOSurfaceCreate((__bridge CFDictionaryRef)outProps);

    if (!inSurf || !outSurf) {
        if (inSurf) CFRelease(inSurf);
        if (outSurf) CFRelease(outSurf);
        return 0; // alloc failed
    }

    // Fill input: 1.0 in FP16 at each channel's plane offset
    IOSurfaceLock(inSurf, 0, NULL);
    void *inBase = IOSurfaceGetBaseAddress(inSurf);
    size_t inSize = IOSurfaceGetAllocSize(inSurf);
    memset(inBase, 0, inSize);
    uint16_t one_fp16 = 0x3C00; // 1.0
    for (int i = 0; i < channels && (i * g_ps + 2) <= inSize; i++) {
        memcpy((uint8_t*)inBase + i * g_ps, &one_fp16, 2);
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

    if (!inObj || !outObj) {
        CFRelease(inSurf);
        CFRelease(outSurf);
        return 1;
    }

    id req = ((id (*)(id, SEL, id, id, id, id, id, id, id, id, id))objc_msgSend)(
        [_ANERequestCls alloc],
        NSSelectorFromString(@"initWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:sharedEvents:transactionHandle:"),
        @[inObj], @[@0], @[outObj], @[@0], nil, nil, @(0), nil, nil);

    if (!req) {
        CFRelease(inSurf);
        CFRelease(outSurf);
        return 1;
    }

    // Map
    NSError *err = nil;
    BOOL mapOK = ((BOOL (*)(id, SEL, id, id, BOOL, id*))objc_msgSend)(
        g_client, NSSelectorFromString(@"mapIOSurfacesWithModel:request:cacheInference:error:"),
        g_model, req, NO, &err);
    if (!mapOK) {
        CFRelease(inSurf);
        CFRelease(outSurf);
        return 1;
    }

    // Evaluate
    BOOL evalOK = ((BOOL (*)(id, SEL, id, id, id, int, id*))objc_msgSend)(
        g_client, NSSelectorFromString(@"doEvaluateDirectWithModel:options:request:qos:error:"),
        g_model, @{}, req, 21, &err);

    // Unmap
    ((void (*)(id, SEL, id, id))objc_msgSend)(
        g_client, NSSelectorFromString(@"unmapIOSurfacesWithModel:request:"), g_model, req);

    if (!evalOK) {
        CFRelease(inSurf);
        CFRelease(outSurf);
        return 1;
    }

    // Check output: ReLU(1.0) should be 1.0
    IOSurfaceLock(outSurf, kIOSurfaceLockReadOnly, NULL);
    void *outBase = IOSurfaceGetBaseAddress(outSurf);
    size_t outSize = IOSurfaceGetAllocSize(outSurf);
    int correct = 0;
    for (int i = 0; i < channels && (i * g_ps + 2) <= outSize; i++) {
        uint16_t val;
        memcpy(&val, (uint8_t*)outBase + i * g_ps, 2);
        if (val == 0x3C00) correct++;
    }
    IOSurfaceUnlock(outSurf, kIOSurfaceLockReadOnly, NULL);

    CFRelease(inSurf);
    CFRelease(outSurf);

    return (correct == channels) ? 3 : 2;
}

static const char *resultStr(int r) {
    switch(r) {
        case 0: return "ALLOC_FAIL";
        case 1: return "DISPATCH_FAIL";
        case 2: return "WRONG_OUTPUT";
        case 3: return "PASS";
        default: return "UNKNOWN";
    }
}

int main(int argc, char *argv[]) {
    @autoreleasepool {
        signal(SIGSEGV, SIG_IGN);
        if (argc < 2) {
            fprintf(stderr, "Usage: %s <relu_model.mlmodelc> [channels=64]\n", argv[0]);
            return 1;
        }

        loadFW();
        int channels = argc > 2 ? atoi(argv[2]) : 64;

        if (!prepareModel(argv[1], channels)) {
            fprintf(stderr, "FATAL: model prepare failed\n");
            return 1;
        }
        getModelStrides();

        // Reference: standard allocation (what ane_eval uses)
        NSDictionary *stdIn = @{@"IOSurfaceWidth":@(g_inBS/2), @"IOSurfaceHeight":@1,
            @"IOSurfaceBytesPerRow":@(g_inBS), @"IOSurfaceBytesPerElement":@2,
            @"IOSurfaceAllocSize":@(g_inBS), @"IOSurfacePixelFormat":@(0x6630304C)};
        NSDictionary *stdOut = @{@"IOSurfaceWidth":@(g_outBS/2), @"IOSurfaceHeight":@1,
            @"IOSurfaceBytesPerRow":@(g_outBS), @"IOSurfaceBytesPerElement":@2,
            @"IOSurfaceAllocSize":@(g_outBS), @"IOSurfacePixelFormat":@(0x6630304C)};

        printf("=== B18: IOSurface Alignment Probing ===\n");
        printf("Channels: %d  InBS: %u  OutBS: %u  PlaneStride: %u\n\n", channels, g_inBS, g_outBS, g_ps);

        // TEST 1: Reference (standard allocation)
        int r = tryDispatch(stdIn, stdOut, channels);
        printf("T01 Reference (standard)          : %s\n", resultStr(r));

        // TEST 2: Minimum size tests
        // 2a: Exactly 49152 bytes (the known minimum)
        {
            NSDictionary *p = @{@"IOSurfaceWidth":@(49152), @"IOSurfaceHeight":@1,
                @"IOSurfaceBytesPerRow":@(49152), @"IOSurfaceBytesPerElement":@1,
                @"IOSurfaceAllocSize":@(49152), @"IOSurfacePixelFormat":@0};
            r = tryDispatch(p, p, channels);
            printf("T02a 49152 bytes (minimum known)   : %s\n", resultStr(r));
        }

        // 2b: Half of minimum (24576)
        {
            NSDictionary *p = @{@"IOSurfaceWidth":@(24576), @"IOSurfaceHeight":@1,
                @"IOSurfaceBytesPerRow":@(24576), @"IOSurfaceBytesPerElement":@1,
                @"IOSurfaceAllocSize":@(24576), @"IOSurfacePixelFormat":@0};
            r = tryDispatch(p, p, channels);
            printf("T02b 24576 bytes (half minimum)    : %s\n", resultStr(r));
        }

        // 2c: 16384 bytes (16K)
        {
            NSDictionary *p = @{@"IOSurfaceWidth":@(16384), @"IOSurfaceHeight":@1,
                @"IOSurfaceBytesPerRow":@(16384), @"IOSurfaceBytesPerElement":@1,
                @"IOSurfaceAllocSize":@(16384), @"IOSurfacePixelFormat":@0};
            r = tryDispatch(p, p, channels);
            printf("T02c 16384 bytes (16K)             : %s\n", resultStr(r));
        }

        // 2d: 4096 bytes (4K page)
        {
            NSDictionary *p = @{@"IOSurfaceWidth":@(4096), @"IOSurfaceHeight":@1,
                @"IOSurfaceBytesPerRow":@(4096), @"IOSurfaceBytesPerElement":@1,
                @"IOSurfaceAllocSize":@(4096), @"IOSurfacePixelFormat":@0};
            r = tryDispatch(p, p, channels);
            printf("T02d 4096 bytes (4K page)          : %s\n", resultStr(r));
        }

        // 2e: 32768 bytes
        {
            NSDictionary *p = @{@"IOSurfaceWidth":@(32768), @"IOSurfaceHeight":@1,
                @"IOSurfaceBytesPerRow":@(32768), @"IOSurfaceBytesPerElement":@1,
                @"IOSurfaceAllocSize":@(32768), @"IOSurfacePixelFormat":@0};
            r = tryDispatch(p, p, channels);
            printf("T02e 32768 bytes (32K)             : %s\n", resultStr(r));
        }

        // 2f: Exact data size without padding (channels * ps)
        {
            uint32_t exact = channels * g_ps;
            NSDictionary *p = @{@"IOSurfaceWidth":@(exact), @"IOSurfaceHeight":@1,
                @"IOSurfaceBytesPerRow":@(exact), @"IOSurfaceBytesPerElement":@1,
                @"IOSurfaceAllocSize":@(exact), @"IOSurfacePixelFormat":@0};
            r = tryDispatch(p, p, channels);
            printf("T02f exact data size (%u)      : %s\n", exact, resultStr(r));
        }

        // TEST 3: Oversized buffers
        // 3a: 2x standard
        {
            NSDictionary *inP = @{@"IOSurfaceWidth":@(g_inBS), @"IOSurfaceHeight":@1,
                @"IOSurfaceBytesPerRow":@(g_inBS*2), @"IOSurfaceBytesPerElement":@2,
                @"IOSurfaceAllocSize":@(g_inBS*2), @"IOSurfacePixelFormat":@(0x6630304C)};
            NSDictionary *outP = @{@"IOSurfaceWidth":@(g_outBS), @"IOSurfaceHeight":@1,
                @"IOSurfaceBytesPerRow":@(g_outBS*2), @"IOSurfaceBytesPerElement":@2,
                @"IOSurfaceAllocSize":@(g_outBS*2), @"IOSurfacePixelFormat":@(0x6630304C)};
            r = tryDispatch(inP, outP, channels);
            printf("T03a 2x oversized                  : %s\n", resultStr(r));
        }

        // 3b: 1MB allocation
        {
            NSDictionary *p = @{@"IOSurfaceWidth":@(1048576/2), @"IOSurfaceHeight":@1,
                @"IOSurfaceBytesPerRow":@(1048576), @"IOSurfaceBytesPerElement":@2,
                @"IOSurfaceAllocSize":@(1048576), @"IOSurfacePixelFormat":@(0x6630304C)};
            r = tryDispatch(p, p, channels);
            printf("T03b 1MB allocation                : %s\n", resultStr(r));
        }

        // TEST 4: Alignment tests (non-power-of-2 sizes)
        // 4a: 49152 + 1 (non-aligned)
        {
            uint32_t sz = 49153;
            NSDictionary *p = @{@"IOSurfaceWidth":@(sz), @"IOSurfaceHeight":@1,
                @"IOSurfaceBytesPerRow":@(sz), @"IOSurfaceBytesPerElement":@1,
                @"IOSurfaceAllocSize":@(sz), @"IOSurfacePixelFormat":@0};
            r = tryDispatch(p, p, channels);
            printf("T04a 49153 bytes (non-aligned)     : %s\n", resultStr(r));
        }

        // 4b: 50000 (non-power-of-2)
        {
            uint32_t sz = 50000;
            NSDictionary *p = @{@"IOSurfaceWidth":@(sz), @"IOSurfaceHeight":@1,
                @"IOSurfaceBytesPerRow":@(sz), @"IOSurfaceBytesPerElement":@1,
                @"IOSurfaceAllocSize":@(sz), @"IOSurfacePixelFormat":@0};
            r = tryDispatch(p, p, channels);
            printf("T04b 50000 bytes (arbitrary)       : %s\n", resultStr(r));
        }

        // 4c: prime number close to 49K
        {
            uint32_t sz = 49157; // prime
            NSDictionary *p = @{@"IOSurfaceWidth":@(sz), @"IOSurfaceHeight":@1,
                @"IOSurfaceBytesPerRow":@(sz), @"IOSurfaceBytesPerElement":@1,
                @"IOSurfaceAllocSize":@(sz), @"IOSurfacePixelFormat":@0};
            r = tryDispatch(p, p, channels);
            printf("T04c 49157 bytes (prime)           : %s\n", resultStr(r));
        }

        // TEST 5: BytesPerRow alignment
        // 5a: Standard with non-matching BytesPerRow (half)
        {
            NSDictionary *inP = @{@"IOSurfaceWidth":@(g_inBS/2), @"IOSurfaceHeight":@1,
                @"IOSurfaceBytesPerRow":@(g_inBS/2), @"IOSurfaceBytesPerElement":@2,
                @"IOSurfaceAllocSize":@(g_inBS), @"IOSurfacePixelFormat":@(0x6630304C)};
            NSDictionary *outP = @{@"IOSurfaceWidth":@(g_outBS/2), @"IOSurfaceHeight":@1,
                @"IOSurfaceBytesPerRow":@(g_outBS/2), @"IOSurfaceBytesPerElement":@2,
                @"IOSurfaceAllocSize":@(g_outBS), @"IOSurfacePixelFormat":@(0x6630304C)};
            r = tryDispatch(inP, outP, channels);
            printf("T05a BytesPerRow=half              : %s\n", resultStr(r));
        }

        // 5b: BytesPerRow not aligned to 16
        {
            uint32_t bpr = g_inBS + 7;
            NSDictionary *inP = @{@"IOSurfaceWidth":@(g_inBS/2), @"IOSurfaceHeight":@1,
                @"IOSurfaceBytesPerRow":@(bpr), @"IOSurfaceBytesPerElement":@2,
                @"IOSurfaceAllocSize":@(bpr), @"IOSurfacePixelFormat":@(0x6630304C)};
            NSDictionary *outP = @{@"IOSurfaceWidth":@(g_outBS/2), @"IOSurfaceHeight":@1,
                @"IOSurfaceBytesPerRow":@(g_outBS+7), @"IOSurfaceBytesPerElement":@2,
                @"IOSurfaceAllocSize":@(g_outBS+7), @"IOSurfacePixelFormat":@(0x6630304C)};
            r = tryDispatch(inP, outP, channels);
            printf("T05b BytesPerRow=non-16-aligned    : %s\n", resultStr(r));
        }

        // TEST 6: Pixel format variations
        // 6a: PixelFormat 0 (raw bytes, used in ANEBuffer)
        {
            NSDictionary *inP = @{@"IOSurfaceWidth":@(g_inBS), @"IOSurfaceHeight":@1,
                @"IOSurfaceBytesPerRow":@(g_inBS), @"IOSurfaceBytesPerElement":@1,
                @"IOSurfaceAllocSize":@(g_inBS), @"IOSurfacePixelFormat":@0};
            NSDictionary *outP = @{@"IOSurfaceWidth":@(g_outBS), @"IOSurfaceHeight":@1,
                @"IOSurfaceBytesPerRow":@(g_outBS), @"IOSurfaceBytesPerElement":@1,
                @"IOSurfaceAllocSize":@(g_outBS), @"IOSurfacePixelFormat":@0};
            r = tryDispatch(inP, outP, channels);
            printf("T06a PixelFormat=0 (raw)           : %s\n", resultStr(r));
        }

        // 6b: PixelFormat 0x4c303068 ('L0h0' - FP16 linear)
        {
            NSDictionary *inP = @{@"IOSurfaceWidth":@(g_inBS), @"IOSurfaceHeight":@1,
                @"IOSurfaceBytesPerRow":@(g_inBS), @"IOSurfaceBytesPerElement":@1,
                @"IOSurfaceAllocSize":@(g_inBS), @"IOSurfacePixelFormat":@(0x4c303068)};
            NSDictionary *outP = @{@"IOSurfaceWidth":@(g_outBS), @"IOSurfaceHeight":@1,
                @"IOSurfaceBytesPerRow":@(g_outBS), @"IOSurfaceBytesPerElement":@1,
                @"IOSurfaceAllocSize":@(g_outBS), @"IOSurfacePixelFormat":@(0x4c303068)};
            r = tryDispatch(inP, outP, channels);
            printf("T06b PixelFormat=0x4c303068 (L0h0) : %s\n", resultStr(r));
        }

        // 6c: PixelFormat 0x6630304C ('f00L' - what ane_eval uses, little-endian of L00f)
        // (this is the standard one used by ane_eval)
        {
            NSDictionary *inP = @{@"IOSurfaceWidth":@(g_inBS/2), @"IOSurfaceHeight":@1,
                @"IOSurfaceBytesPerRow":@(g_inBS), @"IOSurfaceBytesPerElement":@2,
                @"IOSurfaceAllocSize":@(g_inBS), @"IOSurfacePixelFormat":@(0x6630304C)};
            NSDictionary *outP = @{@"IOSurfaceWidth":@(g_outBS/2), @"IOSurfaceHeight":@1,
                @"IOSurfaceBytesPerRow":@(g_outBS), @"IOSurfaceBytesPerElement":@2,
                @"IOSurfaceAllocSize":@(g_outBS), @"IOSurfacePixelFormat":@(0x6630304C)};
            r = tryDispatch(inP, outP, channels);
            printf("T06c PixelFormat=0x6630304C (std)  : %s\n", resultStr(r));
        }

        // 6d: PixelFormat 0x42475241 ('BGRA')
        {
            NSDictionary *inP = @{@"IOSurfaceWidth":@(g_inBS/4), @"IOSurfaceHeight":@1,
                @"IOSurfaceBytesPerRow":@(g_inBS), @"IOSurfaceBytesPerElement":@4,
                @"IOSurfaceAllocSize":@(g_inBS), @"IOSurfacePixelFormat":@(0x42475241)};
            NSDictionary *outP = @{@"IOSurfaceWidth":@(g_outBS/4), @"IOSurfaceHeight":@1,
                @"IOSurfaceBytesPerRow":@(g_outBS), @"IOSurfaceBytesPerElement":@4,
                @"IOSurfaceAllocSize":@(g_outBS), @"IOSurfacePixelFormat":@(0x42475241)};
            r = tryDispatch(inP, outP, channels);
            printf("T06d PixelFormat=BGRA              : %s\n", resultStr(r));
        }

        // 6e: PixelFormat 0x66 ('f' - half float single)
        {
            NSDictionary *inP = @{@"IOSurfaceWidth":@(g_inBS), @"IOSurfaceHeight":@1,
                @"IOSurfaceBytesPerRow":@(g_inBS), @"IOSurfaceBytesPerElement":@1,
                @"IOSurfaceAllocSize":@(g_inBS), @"IOSurfacePixelFormat":@(0x66)};
            NSDictionary *outP = @{@"IOSurfaceWidth":@(g_outBS), @"IOSurfaceHeight":@1,
                @"IOSurfaceBytesPerRow":@(g_outBS), @"IOSurfaceBytesPerElement":@1,
                @"IOSurfaceAllocSize":@(g_outBS), @"IOSurfacePixelFormat":@(0x66)};
            r = tryDispatch(inP, outP, channels);
            printf("T06e PixelFormat=0x66              : %s\n", resultStr(r));
        }

        // TEST 7: Width/Height variations (same AllocSize)
        // 7a: Width=1, Height=AllocSize
        {
            NSDictionary *inP = @{@"IOSurfaceWidth":@1, @"IOSurfaceHeight":@(g_inBS),
                @"IOSurfaceBytesPerRow":@(g_inBS), @"IOSurfaceBytesPerElement":@1,
                @"IOSurfaceAllocSize":@(g_inBS), @"IOSurfacePixelFormat":@0};
            NSDictionary *outP = @{@"IOSurfaceWidth":@1, @"IOSurfaceHeight":@(g_outBS),
                @"IOSurfaceBytesPerRow":@(g_outBS), @"IOSurfaceBytesPerElement":@1,
                @"IOSurfaceAllocSize":@(g_outBS), @"IOSurfacePixelFormat":@0};
            r = tryDispatch(inP, outP, channels);
            printf("T07a Width=1 Height=BS             : %s\n", resultStr(r));
        }

        // 7b: Channels x (ps/2) 2D layout
        {
            NSDictionary *inP = @{@"IOSurfaceWidth":@(g_ps/2), @"IOSurfaceHeight":@(channels),
                @"IOSurfaceBytesPerRow":@(g_ps), @"IOSurfaceBytesPerElement":@2,
                @"IOSurfaceAllocSize":@(g_inBS), @"IOSurfacePixelFormat":@(0x6630304C)};
            NSDictionary *outP = @{@"IOSurfaceWidth":@(g_ps/2), @"IOSurfaceHeight":@(channels),
                @"IOSurfaceBytesPerRow":@(g_ps), @"IOSurfaceBytesPerElement":@2,
                @"IOSurfaceAllocSize":@(g_outBS), @"IOSurfacePixelFormat":@(0x6630304C)};
            r = tryDispatch(inP, outP, channels);
            printf("T07b 2D layout (Ch x PlaneStride)  : %s\n", resultStr(r));
        }

        // 7c: Very tall (Width=1, Height=1, but full AllocSize)
        {
            NSDictionary *inP = @{@"IOSurfaceWidth":@1, @"IOSurfaceHeight":@1,
                @"IOSurfaceBytesPerRow":@1, @"IOSurfaceBytesPerElement":@1,
                @"IOSurfaceAllocSize":@(g_inBS), @"IOSurfacePixelFormat":@0};
            NSDictionary *outP = @{@"IOSurfaceWidth":@1, @"IOSurfaceHeight":@1,
                @"IOSurfaceBytesPerRow":@1, @"IOSurfaceBytesPerElement":@1,
                @"IOSurfaceAllocSize":@(g_outBS), @"IOSurfacePixelFormat":@0};
            r = tryDispatch(inP, outP, channels);
            printf("T07c Width=1 Height=1 (minimal)    : %s\n", resultStr(r));
        }

        // TEST 8: Mixed configs (in=standard, out=weird)
        {
            NSDictionary *outP = @{@"IOSurfaceWidth":@(g_outBS), @"IOSurfaceHeight":@1,
                @"IOSurfaceBytesPerRow":@(g_outBS), @"IOSurfaceBytesPerElement":@1,
                @"IOSurfaceAllocSize":@(g_outBS), @"IOSurfacePixelFormat":@0};
            r = tryDispatch(stdIn, outP, channels);
            printf("T08a std_in + raw_out              : %s\n", resultStr(r));
        }

        // TEST 9: Under-minimum with correct data area
        // Smaller than 49152 but enough for actual data
        {
            uint32_t sz = 8192; // 8K
            NSDictionary *p = @{@"IOSurfaceWidth":@(sz), @"IOSurfaceHeight":@1,
                @"IOSurfaceBytesPerRow":@(sz), @"IOSurfaceBytesPerElement":@1,
                @"IOSurfaceAllocSize":@(sz), @"IOSurfacePixelFormat":@0};
            r = tryDispatch(p, p, channels);
            printf("T09a 8192 bytes (8K)               : %s\n", resultStr(r));
        }
        {
            uint32_t sz = 2048;
            NSDictionary *p = @{@"IOSurfaceWidth":@(sz), @"IOSurfaceHeight":@1,
                @"IOSurfaceBytesPerRow":@(sz), @"IOSurfaceBytesPerElement":@1,
                @"IOSurfaceAllocSize":@(sz), @"IOSurfacePixelFormat":@0};
            r = tryDispatch(p, p, channels);
            printf("T09b 2048 bytes (2K)               : %s\n", resultStr(r));
        }
        {
            uint32_t sz = 1024;
            NSDictionary *p = @{@"IOSurfaceWidth":@(sz), @"IOSurfaceHeight":@1,
                @"IOSurfaceBytesPerRow":@(sz), @"IOSurfaceBytesPerElement":@1,
                @"IOSurfaceAllocSize":@(sz), @"IOSurfacePixelFormat":@0};
            r = tryDispatch(p, p, channels);
            printf("T09c 1024 bytes (1K)               : %s\n", resultStr(r));
        }

        // TEST 10: Page-size aligned binary search for minimum
        // Search between 1 page (16K on arm64) and 49152
        printf("\n=== Minimum Size Binary Search (16K granularity) ===\n");
        for (uint32_t sz = 16384; sz <= 65536; sz += 16384) {
            NSDictionary *p = @{@"IOSurfaceWidth":@(sz), @"IOSurfaceHeight":@1,
                @"IOSurfaceBytesPerRow":@(sz), @"IOSurfaceBytesPerElement":@1,
                @"IOSurfaceAllocSize":@(sz), @"IOSurfacePixelFormat":@0};
            r = tryDispatch(p, p, channels);
            printf("  %5u bytes: %s\n", sz, resultStr(r));
        }

        // Finer search around boundaries
        printf("\n=== Fine Size Search (4K granularity around 49152) ===\n");
        for (uint32_t sz = 40960; sz <= 57344; sz += 4096) {
            NSDictionary *p = @{@"IOSurfaceWidth":@(sz), @"IOSurfaceHeight":@1,
                @"IOSurfaceBytesPerRow":@(sz), @"IOSurfaceBytesPerElement":@1,
                @"IOSurfaceAllocSize":@(sz), @"IOSurfacePixelFormat":@0};
            r = tryDispatch(p, p, channels);
            printf("  %5u bytes: %s\n", sz, resultStr(r));
        }

        // Unload model
        NSError *err = nil;
        ((BOOL (*)(id, SEL, id, id, NSInteger, id*))objc_msgSend)(
            g_client, NSSelectorFromString(@"doUnloadModel:options:qos:error:"), g_model, @{}, 0, &err);

        printf("\n=== B18 COMPLETE ===\n");
        return 0;
    }
}
