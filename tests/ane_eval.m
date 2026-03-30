// ane_eval.m — Evaluate a model on ANE, print output as hex FP16
//
// Build: xcrun clang -framework Foundation -framework IOSurface -fobjc-arc -o ane_eval ane_eval.m
// Run: ./ane_eval <model.mlmodelc> [channels=64]
// Output: space-separated FP16 hex values to stdout

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

int main(int argc, char *argv[]) {
    @autoreleasepool {
        signal(SIGSEGV, SIG_IGN);
        if (argc < 2) { fprintf(stderr, "Usage: %s <model.mlmodelc> [channels]\n", argv[0]); return 1; }

        loadFW();
        int channels = argc > 2 ? atoi(argv[2]) : 64;

        id client = ((id (*)(id, SEL))objc_msgSend)((id)_ANEClientCls, NSSelectorFromString(@"sharedConnection"));
        NSURL *url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:argv[1]]];
        id model = ((id (*)(id, SEL, id, id))objc_msgSend)((id)_ANEModelCls, NSSelectorFromString(@"modelAtURL:key:"), url, @"default");
        if (!model) { fprintf(stderr, "Model failed\n"); return 1; }

        // Compile + load
        NSError *err = nil;
        ((BOOL (*)(id, SEL, id, id, NSInteger, id*))objc_msgSend)(
            client, NSSelectorFromString(@"compileModel:options:qos:error:"), model, @{}, 0, &err);
        BOOL loadOK = ((BOOL (*)(id, SEL, id, id, NSInteger, id*))objc_msgSend)(
            client, NSSelectorFromString(@"loadModel:options:qos:error:"), model, @{}, 0, &err);
        if (!loadOK) { fprintf(stderr, "Load failed\n"); return 1; }

        // Get buffer info
        id attrs = ((id (*)(id, SEL))objc_msgSend)(model, NSSelectorFromString(@"modelAttributes"));
        NSDictionary *ns = [attrs[@"NetworkStatusList"] firstObject];
        uint32_t inBS = [[ns[@"LiveInputList"] firstObject][@"BatchStride"] unsignedIntValue];
        uint32_t outBS = [[ns[@"LiveOutputList"] firstObject][@"BatchStride"] unsignedIntValue];
        uint32_t ps = [[ns[@"LiveInputList"] firstObject][@"PlaneStride"] unsignedIntValue];

        // Create IOSurfaces
        NSDictionary *inProps = @{@"IOSurfaceWidth":@(inBS/2), @"IOSurfaceHeight":@1,
            @"IOSurfaceBytesPerRow":@(inBS), @"IOSurfaceBytesPerElement":@2,
            @"IOSurfaceAllocSize":@(inBS), @"IOSurfacePixelFormat":@(0x6630304C)};
        NSDictionary *outProps = @{@"IOSurfaceWidth":@(outBS/2), @"IOSurfaceHeight":@1,
            @"IOSurfaceBytesPerRow":@(outBS), @"IOSurfaceBytesPerElement":@2,
            @"IOSurfaceAllocSize":@(outBS), @"IOSurfacePixelFormat":@(0x6630304C)};

        IOSurfaceRef inSurf = IOSurfaceCreate((__bridge CFDictionaryRef)inProps);
        IOSurfaceRef outSurf = IOSurfaceCreate((__bridge CFDictionaryRef)outProps);

        // Fill input: 1.0 in every channel
        IOSurfaceLock(inSurf, 0, NULL);
        void *inBase = IOSurfaceGetBaseAddress(inSurf);
        memset(inBase, 0, inBS);
        uint16_t one_fp16 = 0x3C00;
        for (int i = 0; i < channels; i++) {
            memcpy((uint8_t*)inBase + i * ps, &one_fp16, 2);
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

        // Map + evaluate
        ((BOOL (*)(id, SEL, id, id, BOOL, id*))objc_msgSend)(
            client, NSSelectorFromString(@"mapIOSurfacesWithModel:request:cacheInference:error:"),
            model, req, NO, nil);

        BOOL evalOK = ((BOOL (*)(id, SEL, id, id, id, int, id*))objc_msgSend)(
            client, NSSelectorFromString(@"doEvaluateDirectWithModel:options:request:qos:error:"),
            model, @{}, req, 21, nil);

        if (!evalOK) { fprintf(stderr, "Eval failed\n"); return 1; }

        // Read output
        IOSurfaceLock(outSurf, kIOSurfaceLockReadOnly, NULL);
        void *outBase = IOSurfaceGetBaseAddress(outSurf);
        for (int i = 0; i < channels; i++) {
            uint16_t fp16;
            memcpy(&fp16, (uint8_t*)outBase + i * ps, 2);
            printf("%04x ", fp16);
        }
        printf("\n");
        IOSurfaceUnlock(outSurf, kIOSurfaceLockReadOnly, NULL);

        // Cleanup
        ((void (*)(id, SEL, id, id))objc_msgSend)(
            client, NSSelectorFromString(@"unmapIOSurfacesWithModel:request:"), model, req);
        ((BOOL (*)(id, SEL, id, id, NSInteger, id*))objc_msgSend)(
            client, NSSelectorFromString(@"doUnloadModel:options:qos:error:"), model, @{}, 0, &err);

        fprintf(stderr, "OK\n");
        return 0;
    }
}
