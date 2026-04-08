
// test_multi_input.m — Test if ANE supports 2-input models
#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>
#import <signal.h>

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
        loadFW();

        if (argc < 2) {
            fprintf(stderr, "Usage: %s <mlmodelc_path>\n", argv[0]);
            return 1;
        }

        id client = ((id (*)(id, SEL))objc_msgSend)((id)_Cl, NSSelectorFromString(@"sharedConnection"));
        NSURL *url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:argv[1]]];
        id model = ((id (*)(id, SEL, id, id))objc_msgSend)(
            (id)_Mo, NSSelectorFromString(@"modelAtURL:key:"), url, @"test_multi");

        NSError *err = nil;
        BOOL ok = ((BOOL (*)(id, SEL, id, id, NSInteger, id*))objc_msgSend)(
            client, NSSelectorFromString(@"compileModel:options:qos:error:"),
            model, @{}, 0, &err);
        if (err) {
            fprintf(stderr, "COMPILE_ERROR: %s\n", [[err description] UTF8String]);
            printf("COMPILE_FAIL\n");
            return 0;
        }
        fprintf(stderr, "Compile OK\n");

        ok = ((BOOL (*)(id, SEL, id, id, NSInteger, id*))objc_msgSend)(
            client, NSSelectorFromString(@"loadModel:options:qos:error:"),
            model, @{}, 0, &err);
        if (!ok) {
            fprintf(stderr, "LOAD_ERROR: %s\n", err ? [[err description] UTF8String] : "nil");
            printf("LOAD_FAIL\n");
            return 0;
        }
        fprintf(stderr, "Load OK\n");

        // Dump model attributes to see input/output info
        id attrs = ((id (*)(id, SEL))objc_msgSend)(model, NSSelectorFromString(@"modelAttributes"));
        NSArray *nsList = attrs[@"NetworkStatusList"];
        NSDictionary *ns = [nsList firstObject];

        NSArray *inputs = ns[@"LiveInputList"];
        NSArray *outputs = ns[@"LiveOutputList"];

        fprintf(stderr, "Inputs: %lu\n", (unsigned long)[inputs count]);
        for (NSDictionary *inp in inputs) {
            fprintf(stderr, "  BatchStride=%u PlaneStride=%u\n",
                    [inp[@"BatchStride"] unsignedIntValue],
                    [inp[@"PlaneStride"] unsignedIntValue]);
        }
        fprintf(stderr, "Outputs: %lu\n", (unsigned long)[outputs count]);
        for (NSDictionary *outp in outputs) {
            fprintf(stderr, "  BatchStride=%u PlaneStride=%u\n",
                    [outp[@"BatchStride"] unsignedIntValue],
                    [outp[@"PlaneStride"] unsignedIntValue]);
        }

        if ([inputs count] < 2) {
            fprintf(stderr, "Model has only %lu inputs (expected 2)\n", (unsigned long)[inputs count]);
            printf("SINGLE_INPUT\n");
            // Fall through to try dispatch anyway
        }

        // Allocate IOSurfaces for each input
        NSUInteger nInputs = [inputs count];
        IOSurfaceRef *inSurfs = calloc(nInputs, sizeof(IOSurfaceRef));
        NSMutableArray *inObjs = [NSMutableArray array];
        NSMutableArray *inIndices = [NSMutableArray array];

        for (NSUInteger j = 0; j < nInputs; j++) {
            NSDictionary *inp = inputs[j];
            uint32_t bs = [inp[@"BatchStride"] unsignedIntValue];
            NSDictionary *props = @{
                @"IOSurfaceWidth": @(bs/2),
                @"IOSurfaceHeight": @1,
                @"IOSurfaceBytesPerRow": @(bs),
                @"IOSurfaceBytesPerElement": @2,
                @"IOSurfaceAllocSize": @(bs),
                @"IOSurfacePixelFormat": @(0x6630304C),
            };
            inSurfs[j] = IOSurfaceCreate((__bridge CFDictionaryRef)props);
            [inObjs addObject:((id (*)(id, SEL, void*, NSInteger, BOOL))objc_msgSend)(
                [_IO alloc], NSSelectorFromString(@"initWithIOSurface:startOffset:shouldRetain:"),
                inSurfs[j], 0, YES)];
            [inIndices addObject:@(j)];
        }

        // Output IOSurface
        NSDictionary *outp = [outputs firstObject];
        uint32_t outBS = [outp[@"BatchStride"] unsignedIntValue];
        uint32_t outPS = [outp[@"PlaneStride"] unsignedIntValue];
        NSDictionary *outProps = @{
            @"IOSurfaceWidth": @(outBS/2),
            @"IOSurfaceHeight": @1,
            @"IOSurfaceBytesPerRow": @(outBS),
            @"IOSurfaceBytesPerElement": @2,
            @"IOSurfaceAllocSize": @(outBS),
            @"IOSurfacePixelFormat": @(0x6630304C),
        };
        IOSurfaceRef outSurf = IOSurfaceCreate((__bridge CFDictionaryRef)outProps);
        id outObj = ((id (*)(id, SEL, void*, NSInteger, BOOL))objc_msgSend)(
            [_IO alloc], NSSelectorFromString(@"initWithIOSurface:startOffset:shouldRetain:"),
            outSurf, 0, YES);

        // Read input data from stdin: dim*2 bytes for input_a, dim*2 for input_b
        int dim = atoi(argv[2]);
        uint16_t *data_a = malloc(dim * 2);
        uint16_t *data_b = malloc(dim * 2);
        fread(data_a, 2, dim, stdin);
        fread(data_b, 2, dim, stdin);

        // Fill input IOSurfaces
        for (NSUInteger j = 0; j < nInputs; j++) {
            NSDictionary *inp = inputs[j];
            uint32_t ps = [inp[@"PlaneStride"] unsignedIntValue];
            uint32_t bs = [inp[@"BatchStride"] unsignedIntValue];
            int nCh = bs / ps;
            uint16_t *data = (j == 0) ? data_a : data_b;

            IOSurfaceLock(inSurfs[j], 0, NULL);
            void *base = IOSurfaceGetBaseAddress(inSurfs[j]);
            memset(base, 0, bs);
            for (int c = 0; c < nCh && c < dim; c++)
                memcpy((uint8_t*)base + c * ps, &data[c], 2);
            IOSurfaceUnlock(inSurfs[j], 0, NULL);
        }

        // Build request with multiple inputs
        id req = ((id (*)(id, SEL, id, id, id, id, id, id, id, id, id))objc_msgSend)(
            [_Rq alloc],
            NSSelectorFromString(@"initWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:sharedEvents:transactionHandle:"),
            inObjs, inIndices, @[outObj], @[@0], nil, nil, @(0), nil, nil);

        // Map
        ok = ((BOOL (*)(id, SEL, id, id, BOOL, id*))objc_msgSend)(
            client, NSSelectorFromString(@"mapIOSurfacesWithModel:request:cacheInference:error:"),
            model, req, NO, &err);
        if (!ok) {
            fprintf(stderr, "MAP_ERROR: %s\n", err ? [[err description] UTF8String] : "nil");
            printf("MAP_FAIL\n");
            return 0;
        }

        // Execute
        ok = ((BOOL (*)(id, SEL, id, id, id, int, id*))objc_msgSend)(
            client, NSSelectorFromString(@"doEvaluateDirectWithModel:options:request:qos:error:"),
            model, @{}, req, 21, &err);
        if (!ok) {
            fprintf(stderr, "EXEC_ERROR: %s\n", err ? [[err description] UTF8String] : "nil");
            printf("EXEC_FAIL\n");
            return 0;
        }

        // Read output
        int outCh = outBS / outPS;
        uint16_t *out = malloc(outCh * 2);
        IOSurfaceLock(outSurf, kIOSurfaceLockReadOnly, NULL);
        void *outBase = IOSurfaceGetBaseAddress(outSurf);
        for (int c = 0; c < outCh; c++)
            memcpy(&out[c], (uint8_t*)outBase + c * outPS, 2);
        IOSurfaceUnlock(outSurf, kIOSurfaceLockReadOnly, NULL);

        // Write output to stdout
        fwrite(out, 2, dim, stdout);
        fflush(stdout);

        fprintf(stderr, "MULTI_INPUT_OK: dispatched with %lu inputs\n", (unsigned long)nInputs);
        printf("OK\n");

        // Cleanup
        ((BOOL (*)(id, SEL, id, id))objc_msgSend)(
            client, NSSelectorFromString(@"unmapIOSurfacesWithModel:request:"), model, req);
        ((BOOL (*)(id, SEL, id, id, NSInteger, id*))objc_msgSend)(
            client, NSSelectorFromString(@"doUnloadModel:options:qos:error:"),
            model, @{}, 0, &err);

        free(data_a); free(data_b); free(out);
        for (NSUInteger j = 0; j < nInputs; j++)
            if (inSurfs[j]) CFRelease(inSurfs[j]);
        if (outSurf) CFRelease(outSurf);
        free(inSurfs);

        return 0;
    }
}
