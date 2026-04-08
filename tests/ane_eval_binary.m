// ane_eval_binary.m — Evaluate a model on ANE with binary FP16 input/output
//
// Build: xcrun clang -framework Foundation -framework IOSurface -fobjc-arc -o ane_eval_binary ane_eval_binary.m
// Run: ./ane_eval_binary <model.mlmodelc> <in_channels> <out_channels> < input.bin > output.bin
// Input: binary FP16 data (in_channels * 2 bytes) from stdin
// Output: binary FP16 data (out_channels * 2 bytes) to stdout

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
            fprintf(stderr, "Usage: %s <model.mlmodelc> <in_channels> <out_channels>\n", argv[0]);
            fprintf(stderr, "  Reads binary FP16 from stdin, writes binary FP16 to stdout\n");
            return 1;
        }

        loadFW();
        int inChannels = atoi(argv[2]);
        int outChannels = atoi(argv[3]);

        id client = ((id (*)(id, SEL))objc_msgSend)((id)_ANEClientCls, NSSelectorFromString(@"sharedConnection"));
        NSURL *url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:argv[1]]];
        id model = ((id (*)(id, SEL, id, id))objc_msgSend)((id)_ANEModelCls, NSSelectorFromString(@"modelAtURL:key:"), url, @"default");
        if (!model) { fprintf(stderr, "Model failed\n"); return 1; }

        // Check for --no-compile flag
        BOOL skipCompile = NO;
        for (int i = 1; i < argc; i++) {
            if (strcmp(argv[i], "--no-compile") == 0) skipCompile = YES;
        }

        // Compile + load (or load-only with --no-compile)
        NSError *err = nil;
        if (!skipCompile) {
            ((BOOL (*)(id, SEL, id, id, NSInteger, id*))objc_msgSend)(
                client, NSSelectorFromString(@"compileModel:options:qos:error:"), model, @{}, 0, &err);
        }
        BOOL loadOK = ((BOOL (*)(id, SEL, id, id, NSInteger, id*))objc_msgSend)(
            client, NSSelectorFromString(@"loadModel:options:qos:error:"), model, @{}, 0, &err);
        if (!loadOK) { fprintf(stderr, "Load failed\n"); return 1; }

        // Get buffer info from model attributes
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

        // Read input from stdin: binary FP16, tiled into PlaneStride layout
        uint16_t *inputData = calloc(inChannels, sizeof(uint16_t));
        size_t bytesRead = fread(inputData, sizeof(uint16_t), inChannels, stdin);
        if (bytesRead != inChannels) {
            fprintf(stderr, "Expected %d FP16 values, got %zu\n", inChannels, bytesRead);
            return 1;
        }

        IOSurfaceLock(inSurf, 0, NULL);
        void *inBase = IOSurfaceGetBaseAddress(inSurf);
        memset(inBase, 0, inBS);
        // Pack: 1 value per PlaneStride offset (matches ane_eval behavior)
        for (int i = 0; i < inChannels; i++) {
            memcpy((uint8_t*)inBase + i * ps, &inputData[i], 2);
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

        // Read output: unpack from tiled layout to linear
        IOSurfaceLock(outSurf, kIOSurfaceLockReadOnly, NULL);
        void *outBase = IOSurfaceGetBaseAddress(outSurf);
        uint16_t *outputData = calloc(outChannels, sizeof(uint16_t));
        // Unpack: 1 value per PlaneStride offset (matches ane_eval behavior)
        for (int i = 0; i < outChannels; i++) {
            memcpy(&outputData[i], (uint8_t*)outBase + i * outPS, 2);
        }
        IOSurfaceUnlock(outSurf, kIOSurfaceLockReadOnly, NULL);

        // Write output to stdout as binary FP16
        fwrite(outputData, sizeof(uint16_t), outChannels, stdout);

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
}
