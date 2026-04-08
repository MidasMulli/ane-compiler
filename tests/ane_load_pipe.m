// ane_load_pipe.m — Load-only pipeline dispatch (no compileModel)
//
// VESTIGIAL: loadModel without compileModel does NOT load from disk cache.
// aned maintains compiled state internally; the .hwx cache files are write-through
// copies, not authoritative. This tool only works if compileModel was called in
// the same aned session (same process lifetime) for the same model.
//
// Kept for reference. The production dispatch path uses ane_standalone_pipe
// which calls compileModel + loadModel.
//
// Protocol:
//   1. Read manifest: (mlmodelc_path, in_ch, out_ch) per line
//   2. For each model: modelAtURL → loadModel (NO compileModel)
//   3. Allocate IOSurfaces, print DISPATCH_READY
//   4. Dispatch loop: "D <idx>\n" + input → output, "Q\n" → quit
//
// Build: xcrun clang -framework Foundation -framework IOSurface -fobjc-arc \
//        -o ane_load_pipe ane_load_pipe.m

#import <Foundation/Foundation.h>
#import <dlfcn.h>
#import <signal.h>
#import <IOSurface/IOSurface.h>
#import <objc/runtime.h>
#import <objc/message.h>

static Class _Cl, _Mo, _Rq, _IO;
static void loadFW(void) {
    dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);
    _Cl = NSClassFromString(@"_ANEClient");
    _Mo = NSClassFromString(@"_ANEModel");
    _Rq = NSClassFromString(@"_ANERequest");
    _IO = NSClassFromString(@"_ANEIOSurfaceObject");
}

typedef struct {
    id model;
    int inCh, outCh;
    uint32_t inBS, outBS, inPS, outPS;
} OpEntry;

int main(int argc, char *argv[]) {
    @autoreleasepool {
        signal(SIGSEGV, SIG_IGN);
        loadFW();

        if (argc < 2) {
            fprintf(stderr, "Usage: %s <manifest.txt>\n", argv[0]);
            return 1;
        }

        id client = ((id (*)(id, SEL))objc_msgSend)((id)_Cl, NSSelectorFromString(@"sharedConnection"));

        // Parse manifest
        NSString *manifest = [NSString stringWithContentsOfFile:[NSString stringWithUTF8String:argv[1]]
                                                       encoding:NSUTF8StringEncoding error:nil];
        NSArray *lines = [manifest componentsSeparatedByString:@"\n"];
        int nOps = 0;
        OpEntry *ops = calloc(lines.count, sizeof(OpEntry));

        for (NSString *line in lines) {
            NSArray *parts = [line componentsSeparatedByString:@" "];
            if (parts.count < 3) continue;
            ops[nOps].inCh = [parts[1] intValue];
            ops[nOps].outCh = [parts[2] intValue];

            NSURL *url = [NSURL fileURLWithPath:parts[0]];
            NSString *key = [NSString stringWithFormat:@"load_%d", nOps];
            ops[nOps].model = ((id (*)(id, SEL, id, id))objc_msgSend)(
                (id)_Mo, NSSelectorFromString(@"modelAtURL:key:"), url, key);
            if (!ops[nOps].model) { fprintf(stderr, "Model %d create failed\n", nOps); return 1; }
            nOps++;
        }

        // Load all models (NO compileModel — reads from aned cache)
        NSError *err = nil;
        IOSurfaceRef *inSurfs = calloc(nOps, sizeof(IOSurfaceRef));
        IOSurfaceRef *outSurfs = calloc(nOps, sizeof(IOSurfaceRef));
        NSMutableArray *inObjs = [NSMutableArray arrayWithCapacity:nOps];
        NSMutableArray *outObjs = [NSMutableArray arrayWithCapacity:nOps];

        for (int i = 0; i < nOps; i++) {
            BOOL loadOK = ((BOOL (*)(id, SEL, id, id, NSInteger, id*))objc_msgSend)(
                client, NSSelectorFromString(@"loadModel:options:qos:error:"),
                ops[i].model, @{}, 0, &err);
            if (!loadOK) {
                fprintf(stderr, "Load %d failed (cache miss? run compileModel first)\n", i);
                return 1;
            }

            id attrs = ((id (*)(id, SEL))objc_msgSend)(ops[i].model, NSSelectorFromString(@"modelAttributes"));
            NSDictionary *ns = [attrs[@"NetworkStatusList"] firstObject];
            ops[i].inBS = [[ns[@"LiveInputList"] firstObject][@"BatchStride"] unsignedIntValue];
            ops[i].outBS = [[ns[@"LiveOutputList"] firstObject][@"BatchStride"] unsignedIntValue];
            ops[i].inPS = [[ns[@"LiveInputList"] firstObject][@"PlaneStride"] unsignedIntValue];
            ops[i].outPS = [[ns[@"LiveOutputList"] firstObject][@"PlaneStride"] unsignedIntValue];

            // Allocate IOSurfaces
            uint32_t inBS = ops[i].inBS, outBS = ops[i].outBS;
            NSDictionary *inProps = @{@"IOSurfaceWidth":@(inBS/2), @"IOSurfaceHeight":@1,
                @"IOSurfaceBytesPerRow":@(inBS), @"IOSurfaceBytesPerElement":@2,
                @"IOSurfaceAllocSize":@(inBS), @"IOSurfacePixelFormat":@(0x6630304C)};
            NSDictionary *outProps = @{@"IOSurfaceWidth":@(outBS/2), @"IOSurfaceHeight":@1,
                @"IOSurfaceBytesPerRow":@(outBS), @"IOSurfaceBytesPerElement":@2,
                @"IOSurfaceAllocSize":@(outBS), @"IOSurfacePixelFormat":@(0x6630304C)};
            inSurfs[i] = IOSurfaceCreate((__bridge CFDictionaryRef)inProps);
            outSurfs[i] = IOSurfaceCreate((__bridge CFDictionaryRef)outProps);
            [inObjs addObject:((id (*)(id, SEL, void*, NSInteger, BOOL))objc_msgSend)(
                [_IO alloc], NSSelectorFromString(@"initWithIOSurface:startOffset:shouldRetain:"),
                inSurfs[i], 0, YES)];
            [outObjs addObject:((id (*)(id, SEL, void*, NSInteger, BOOL))objc_msgSend)(
                [_IO alloc], NSSelectorFromString(@"initWithIOSurface:startOffset:shouldRetain:"),
                outSurfs[i], 0, YES)];
        }

        fprintf(stderr, "Loaded %d models from cache\n", nOps);
        printf("DISPATCH_READY\n");
        fflush(stdout);

        // Dispatch loop
        char buf[64];
        while (fgets(buf, sizeof(buf), stdin)) {
            if (buf[0] == 'Q') break;
            if (buf[0] != 'D') continue;

            int idx = atoi(buf + 2);
            if (idx < 0 || idx >= nOps) { fprintf(stderr, "Bad idx %d\n", idx); return 1; }

            int inCh = ops[idx].inCh, outCh = ops[idx].outCh;
            uint32_t ps = ops[idx].inPS, outPS = ops[idx].outPS;

            uint16_t *inputData = calloc(inCh, sizeof(uint16_t));
            size_t got = fread(inputData, sizeof(uint16_t), inCh, stdin);
            if ((int)got != inCh) { fprintf(stderr, "Short read\n"); free(inputData); return 1; }

            IOSurfaceLock(inSurfs[idx], 0, NULL);
            void *inBase = IOSurfaceGetBaseAddress(inSurfs[idx]);
            memset(inBase, 0, ops[idx].inBS);
            for (int j = 0; j < inCh; j++)
                memcpy((uint8_t*)inBase + j * ps, &inputData[j], 2);
            IOSurfaceUnlock(inSurfs[idx], 0, NULL);

            id req = ((id (*)(id, SEL, id, id, id, id, id, id, id, id, id))objc_msgSend)(
                [_Rq alloc],
                NSSelectorFromString(@"initWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:sharedEvents:transactionHandle:"),
                @[inObjs[idx]], @[@0], @[outObjs[idx]], @[@0], nil, nil, @(0), nil, nil);

            ((BOOL (*)(id, SEL, id, id, BOOL, id*))objc_msgSend)(
                client, NSSelectorFromString(@"mapIOSurfacesWithModel:request:cacheInference:error:"),
                ops[idx].model, req, NO, nil);

            BOOL evalOK = ((BOOL (*)(id, SEL, id, id, id, int, id*))objc_msgSend)(
                client, NSSelectorFromString(@"doEvaluateDirectWithModel:options:request:qos:error:"),
                ops[idx].model, @{}, req, 21, nil);
            if (!evalOK) { fprintf(stderr, "Eval idx=%d failed\n", idx); free(inputData); return 1; }

            IOSurfaceLock(outSurfs[idx], kIOSurfaceLockReadOnly, NULL);
            void *outBase = IOSurfaceGetBaseAddress(outSurfs[idx]);
            uint16_t *outputData = calloc(outCh, sizeof(uint16_t));
            for (int j = 0; j < outCh; j++)
                memcpy(&outputData[j], (uint8_t*)outBase + j * outPS, 2);
            IOSurfaceUnlock(outSurfs[idx], kIOSurfaceLockReadOnly, NULL);

            fwrite(outputData, sizeof(uint16_t), outCh, stdout);
            fflush(stdout);

            ((void (*)(id, SEL, id, id))objc_msgSend)(
                client, NSSelectorFromString(@"unmapIOSurfacesWithModel:request:"), ops[idx].model, req);

            free(inputData);
            free(outputData);
        }

        // Cleanup
        for (int i = 0; i < nOps; i++) {
            ((BOOL (*)(id, SEL, id, id, NSInteger, id*))objc_msgSend)(
                client, NSSelectorFromString(@"doUnloadModel:options:qos:error:"),
                ops[i].model, @{}, 0, &err);
            if (inSurfs[i]) CFRelease(inSurfs[i]);
            if (outSurfs[i]) CFRelease(outSurfs[i]);
        }
        free(inSurfs); free(outSurfs); free(ops);
        return 0;
    }
}
