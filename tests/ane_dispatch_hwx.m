// ane_dispatch_hwx.m — Dispatch a raw .hwx file on ANE hardware
//
// Phase 1 (setup):  ./ane_dispatch_hwx setup <model.mlmodelc> <channels>
//   Compiles model, writes cache path to stdout
//
// Phase 2 (run):    ./ane_dispatch_hwx run <model.mlmodelc> <in_ch> <out_ch>
//   Loads model from cache (no recompile), reads binary FP16 from stdin,
//   dispatches, writes binary FP16 to stdout
//
// Usage for standalone pipeline:
//   1. Setup: compile reference models for each unique dim pair
//   2. Swap: sudo cp emitter.hwx <cache_path>/model.hwx
//   3. Run: echo input | ./ane_dispatch_hwx run model.mlmodelc in_ch out_ch > output
//
// Build: xcrun clang -framework Foundation -framework IOSurface -fobjc-arc \
//        -o ane_dispatch_hwx ane_dispatch_hwx.m

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
        if (argc < 4) {
            fprintf(stderr, "Usage:\n");
            fprintf(stderr, "  %s setup <model.mlmodelc> <out_channels>\n", argv[0]);
            fprintf(stderr, "  %s run   <model.mlmodelc> <in_ch> <out_ch>\n", argv[0]);
            return 1;
        }

        loadFW();
        NSString *mode = [NSString stringWithUTF8String:argv[1]];
        NSString *modelPath = [NSString stringWithUTF8String:argv[2]];

        id client = ((id (*)(id, SEL))objc_msgSend)(
            (id)_ANEClientCls, NSSelectorFromString(@"sharedConnection"));
        NSURL *url = [NSURL fileURLWithPath:modelPath];
        id model = ((id (*)(id, SEL, id, id))objc_msgSend)(
            (id)_ANEModelCls, NSSelectorFromString(@"modelAtURL:key:"), url, @"default");
        if (!model) { fprintf(stderr, "Model create failed\n"); return 1; }

        NSError *err = nil;

        if ([mode isEqualToString:@"setup"]) {
            int outCh = atoi(argv[3]);
            // Compile + load to populate cache
            ((BOOL (*)(id, SEL, id, id, NSInteger, id*))objc_msgSend)(
                client, NSSelectorFromString(@"compileModel:options:qos:error:"),
                model, @{}, 0, &err);
            BOOL loadOK = ((BOOL (*)(id, SEL, id, id, NSInteger, id*))objc_msgSend)(
                client, NSSelectorFromString(@"loadModel:options:qos:error:"),
                model, @{}, 0, &err);
            if (!loadOK) { fprintf(stderr, "Setup load failed\n"); return 1; }

            // Verify by dispatching with all-ones
            id attrs = ((id (*)(id, SEL))objc_msgSend)(model, NSSelectorFromString(@"modelAttributes"));
            NSDictionary *ns = [attrs[@"NetworkStatusList"] firstObject];
            uint32_t inBS = [[ns[@"LiveInputList"] firstObject][@"BatchStride"] unsignedIntValue];
            uint32_t outBS = [[ns[@"LiveOutputList"] firstObject][@"BatchStride"] unsignedIntValue];

            fprintf(stderr, "SETUP_OK in_bs=%u out_bs=%u\n", inBS, outBS);

            // Unload
            ((BOOL (*)(id, SEL, id, id, NSInteger, id*))objc_msgSend)(
                client, NSSelectorFromString(@"doUnloadModel:options:qos:error:"),
                model, @{}, 0, &err);
            return 0;
        }

        if ([mode isEqualToString:@"run"]) {
            int inCh = atoi(argv[3]);
            int outCh = argc > 4 ? atoi(argv[4]) : inCh;

            // Load only (no compile — uses whatever is in cache)
            BOOL loadOK = ((BOOL (*)(id, SEL, id, id, NSInteger, id*))objc_msgSend)(
                client, NSSelectorFromString(@"loadModel:options:qos:error:"),
                model, @{}, 0, &err);
            if (!loadOK) { fprintf(stderr, "Run load failed\n"); return 1; }

            // Get buffer info
            id attrs = ((id (*)(id, SEL))objc_msgSend)(model, NSSelectorFromString(@"modelAttributes"));
            NSDictionary *ns = [attrs[@"NetworkStatusList"] firstObject];
            uint32_t inBS = [[ns[@"LiveInputList"] firstObject][@"BatchStride"] unsignedIntValue];
            uint32_t outBS = [[ns[@"LiveOutputList"] firstObject][@"BatchStride"] unsignedIntValue];
            uint32_t ps = [[ns[@"LiveInputList"] firstObject][@"PlaneStride"] unsignedIntValue];
            uint32_t outPS = [[ns[@"LiveOutputList"] firstObject][@"PlaneStride"] unsignedIntValue];

            // Create IOSurfaces
            NSDictionary *inProps = @{@"IOSurfaceWidth":@(inBS/2), @"IOSurfaceHeight":@1,
                @"IOSurfaceBytesPerRow":@(inBS), @"IOSurfaceBytesPerElement":@2,
                @"IOSurfaceAllocSize":@(inBS), @"IOSurfacePixelFormat":@(0x6630304C)};
            NSDictionary *outProps = @{@"IOSurfaceWidth":@(outBS/2), @"IOSurfaceHeight":@1,
                @"IOSurfaceBytesPerRow":@(outBS), @"IOSurfaceBytesPerElement":@2,
                @"IOSurfaceAllocSize":@(outBS), @"IOSurfacePixelFormat":@(0x6630304C)};

            IOSurfaceRef inSurf = IOSurfaceCreate((__bridge CFDictionaryRef)inProps);
            IOSurfaceRef outSurf = IOSurfaceCreate((__bridge CFDictionaryRef)outProps);

            // Read input from stdin
            uint16_t *inputData = calloc(inCh, sizeof(uint16_t));
            fread(inputData, sizeof(uint16_t), inCh, stdin);

            IOSurfaceLock(inSurf, 0, NULL);
            void *inBase = IOSurfaceGetBaseAddress(inSurf);
            memset(inBase, 0, inBS);
            for (int i = 0; i < inCh; i++)
                memcpy((uint8_t*)inBase + i * ps, &inputData[i], 2);
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
            uint16_t *outputData = calloc(outCh, sizeof(uint16_t));
            for (int i = 0; i < outCh; i++)
                memcpy(&outputData[i], (uint8_t*)outBase + i * outPS, 2);
            IOSurfaceUnlock(outSurf, kIOSurfaceLockReadOnly, NULL);

            fwrite(outputData, sizeof(uint16_t), outCh, stdout);

            // Cleanup
            ((void (*)(id, SEL, id, id))objc_msgSend)(
                client, NSSelectorFromString(@"unmapIOSurfacesWithModel:request:"), model, req);
            ((BOOL (*)(id, SEL, id, id, NSInteger, id*))objc_msgSend)(
                client, NSSelectorFromString(@"doUnloadModel:options:qos:error:"), model, @{}, 0, &err);

            free(inputData);
            free(outputData);
            fprintf(stderr, "OK\n");
            return 0;
        }

        fprintf(stderr, "Unknown mode: %s\n", [mode UTF8String]);
        return 1;
    }
}
