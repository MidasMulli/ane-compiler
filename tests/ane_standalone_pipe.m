// ane_standalone_pipe.m — Pipeline dispatch for standalone .hwx forward pass
//
// Reads a manifest of (mlmodelc_path, hwx_path, in_ch, out_ch) from stdin.
// Phase 1 (setup): compiles all models via _ANEClient (populates cache)
// Phase 2 (swap): reads "SWAP\n" signal, user swaps .hwx files externally
// Phase 3 (inference): for each op, reads binary FP16 input, dispatches,
//   writes binary FP16 output. Uses loadModel which reads from disk cache
//   (picking up the swapped .hwx).
//
// Build: xcrun clang -framework Foundation -framework IOSurface -fobjc-arc \
//        -o ane_standalone_pipe ane_standalone_pipe.m

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
    NSString *mlcPath;
    NSString *hwxPath;
    int inCh, outCh;
    uint32_t inBS, outBS, inPS, outPS;
} OpEntry;

int main(int argc, char *argv[]) {
    @autoreleasepool {
        signal(SIGSEGV, SIG_IGN);
        loadFW();

        id client = ((id (*)(id, SEL))objc_msgSend)((id)_Cl, NSSelectorFromString(@"sharedConnection"));

        // Read manifest from argv: pairs of (mlc_path in_ch out_ch)
        // Format: N ops, then N lines of "mlc_path in_ch out_ch"
        if (argc < 2) {
            fprintf(stderr, "Usage: %s <manifest.txt>\n", argv[0]);
            return 1;
        }

        // Parse manifest
        NSString *manifest = [NSString stringWithContentsOfFile:[NSString stringWithUTF8String:argv[1]]
                                                       encoding:NSUTF8StringEncoding error:nil];
        NSArray *lines = [manifest componentsSeparatedByString:@"\n"];
        int nOps = 0;
        OpEntry *ops = calloc(lines.count, sizeof(OpEntry));

        for (NSString *line in lines) {
            NSArray *parts = [line componentsSeparatedByString:@" "];
            if (parts.count < 3) continue;
            ops[nOps].mlcPath = parts[0];
            ops[nOps].inCh = [parts[1] intValue];
            ops[nOps].outCh = [parts[2] intValue];
            nOps++;
        }
        fprintf(stderr, "Loaded %d ops\n", nOps);

        // Phase 1: Compile all models
        fprintf(stderr, "PHASE1_COMPILING\n");
        NSError *err = nil;
        for (int i = 0; i < nOps; i++) {
            NSURL *url = [NSURL fileURLWithPath:ops[i].mlcPath];
            // Use a unique key per op so each gets its own model handle
            NSString *key = [NSString stringWithFormat:@"op_%d", i];
            ops[i].model = ((id (*)(id, SEL, id, id))objc_msgSend)(
                (id)_Mo, NSSelectorFromString(@"modelAtURL:key:"), url, key);
            if (!ops[i].model) { fprintf(stderr, "Model %d create failed\n", i); return 1; }

            err = nil;
            ((BOOL (*)(id, SEL, id, id, NSInteger, id*))objc_msgSend)(
                client, NSSelectorFromString(@"compileModel:options:qos:error:"),
                ops[i].model, @{}, 0, &err);
            if (err) { fprintf(stderr, "Compile %d failed: %s\n", i, [[err description] UTF8String]); err = nil; }

            // Get buffer info
            BOOL loadOK = ((BOOL (*)(id, SEL, id, id, NSInteger, id*))objc_msgSend)(
                client, NSSelectorFromString(@"loadModel:options:qos:error:"),
                ops[i].model, @{}, 0, &err);
            if (!loadOK) { fprintf(stderr, "Load %d failed: %s\n", i, err ? [[err description] UTF8String] : "nil error"); return 1; }

            id attrs = ((id (*)(id, SEL))objc_msgSend)(ops[i].model, NSSelectorFromString(@"modelAttributes"));
            NSDictionary *ns = [attrs[@"NetworkStatusList"] firstObject];
            ops[i].inBS = [[ns[@"LiveInputList"] firstObject][@"BatchStride"] unsignedIntValue];
            ops[i].outBS = [[ns[@"LiveOutputList"] firstObject][@"BatchStride"] unsignedIntValue];
            ops[i].inPS = [[ns[@"LiveInputList"] firstObject][@"PlaneStride"] unsignedIntValue];
            ops[i].outPS = [[ns[@"LiveOutputList"] firstObject][@"PlaneStride"] unsignedIntValue];

            // Find THIS model's cache .hwx by searching for .src file that matches our mlmodelc
            // The cache structure: .../hash1/hash2/model.hwx + model.src
            // model.src contains the path to the source .mlmodelc
            NSString *cacheBase = @"/Library/Caches/com.apple.aned";
            NSFileManager *fm = [NSFileManager defaultManager];
            NSDirectoryEnumerator *ce = [fm enumeratorAtPath:cacheBase];
            NSString *ep;
            NSString *foundHWX = nil;
            while ((ep = [ce nextObject])) {
                if ([ep.lastPathComponent isEqualToString:@"model.src"]) {
                    NSString *srcFull = [cacheBase stringByAppendingPathComponent:ep];
                    NSString *srcContent = [NSString stringWithContentsOfFile:srcFull
                                                                    encoding:NSUTF8StringEncoding error:nil];
                    if (srcContent && [srcContent containsString:ops[i].mlcPath.lastPathComponent]) {
                        NSString *dir = [srcFull stringByDeletingLastPathComponent];
                        foundHWX = [dir stringByAppendingPathComponent:@"model.hwx"];
                        break;
                    }
                }
            }
            if (foundHWX && [fm fileExistsAtPath:foundHWX]) {
                printf("CACHE:%d:%s\n", i, [foundHWX UTF8String]);
            } else {
                fprintf(stderr, "No cache found for op %d\n", i);
            }

            // Unload (free kernel memory, keep model handle)
            ((BOOL (*)(id, SEL, id, id, NSInteger, id*))objc_msgSend)(
                client, NSSelectorFromString(@"doUnloadModel:options:qos:error:"),
                ops[i].model, @{}, 0, &err);
        }
        fprintf(stderr, "PHASE1_DONE\n");
        fflush(stderr);

        fflush(stdout);

        // Signal ready for swap
        printf("READY_FOR_SWAP\n");
        fflush(stdout);

        // Wait for "GO" from stdin
        char buf[64];
        if (!fgets(buf, sizeof(buf), stdin)) return 1;
        fprintf(stderr, "PHASE2_INFERENCE\n");

        // Phase 2: Load all models and pre-allocate IOSurfaces
        // Then enter dispatch loop: "D <idx>\n" + input → output, "Q\n" → quit
        IOSurfaceRef *inSurfs = calloc(nOps, sizeof(IOSurfaceRef));
        IOSurfaceRef *outSurfs = calloc(nOps, sizeof(IOSurfaceRef));
        NSMutableArray *inObjs = [NSMutableArray arrayWithCapacity:nOps];
        NSMutableArray *outObjs = [NSMutableArray arrayWithCapacity:nOps];

        for (int i = 0; i < nOps; i++) {
            // Load model
            BOOL loadOK = ((BOOL (*)(id, SEL, id, id, NSInteger, id*))objc_msgSend)(
                client, NSSelectorFromString(@"loadModel:options:qos:error:"),
                ops[i].model, @{}, 0, &err);
            if (!loadOK) { fprintf(stderr, "Load %d failed\n", i); return 1; }

            uint32_t inBS = ops[i].inBS, outBS = ops[i].outBS;

            // Pre-allocate IOSurfaces
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
        fprintf(stderr, "All %d models loaded, IOSurfaces allocated\n", nOps);

        // Signal READY for dispatch loop
        printf("DISPATCH_READY\n");
        fflush(stdout);

        // Dispatch loop: read commands from stdin
        // "D <idx>\n" + in_ch*2 bytes → dispatch model #idx, write out_ch*2 bytes
        // "Q\n" → quit
        while (fgets(buf, sizeof(buf), stdin)) {
            if (buf[0] == 'Q') break;
            if (buf[0] != 'D') { fprintf(stderr, "Unknown cmd: %s", buf); continue; }

            int idx = atoi(buf + 2);
            if (idx < 0 || idx >= nOps) { fprintf(stderr, "Bad idx %d\n", idx); return 1; }

            int inCh = ops[idx].inCh, outCh = ops[idx].outCh;
            uint32_t ps = ops[idx].inPS, outPS = ops[idx].outPS;

            // Read input data
            uint16_t *inputData = calloc(inCh, sizeof(uint16_t));
            size_t got = fread(inputData, sizeof(uint16_t), inCh, stdin);
            if ((int)got != inCh) { fprintf(stderr, "Short read: %zu/%d\n", got, inCh); free(inputData); return 1; }

            // Fill input IOSurface
            IOSurfaceLock(inSurfs[idx], 0, NULL);
            void *inBase = IOSurfaceGetBaseAddress(inSurfs[idx]);
            memset(inBase, 0, ops[idx].inBS);
            for (int j = 0; j < inCh; j++)
                memcpy((uint8_t*)inBase + j * ps, &inputData[j], 2);
            IOSurfaceUnlock(inSurfs[idx], 0, NULL);

            // Build request
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

            // Read output from IOSurface
            IOSurfaceLock(outSurfs[idx], kIOSurfaceLockReadOnly, NULL);
            void *outBase = IOSurfaceGetBaseAddress(outSurfs[idx]);
            uint16_t *outputData = calloc(outCh, sizeof(uint16_t));
            for (int j = 0; j < outCh; j++)
                memcpy(&outputData[j], (uint8_t*)outBase + j * outPS, 2);
            IOSurfaceUnlock(outSurfs[idx], kIOSurfaceLockReadOnly, NULL);

            fwrite(outputData, sizeof(uint16_t), outCh, stdout);
            fflush(stdout);

            // Unmap (but don't unload — model stays loaded for reuse)
            ((void (*)(id, SEL, id, id))objc_msgSend)(
                client, NSSelectorFromString(@"unmapIOSurfacesWithModel:request:"), ops[idx].model, req);

            free(inputData);
            free(outputData);
        }

        // Cleanup: unload all models, release IOSurfaces
        for (int i = 0; i < nOps; i++) {
            ((BOOL (*)(id, SEL, id, id, NSInteger, id*))objc_msgSend)(
                client, NSSelectorFromString(@"doUnloadModel:options:qos:error:"),
                ops[i].model, @{}, 0, &err);
            if (inSurfs[i]) CFRelease(inSurfs[i]);
            if (outSurfs[i]) CFRelease(outSurfs[i]);
        }

        fprintf(stderr, "PHASE2_DONE\n");
        free(inSurfs); free(outSurfs);
        free(ops);
        return 0;
    }
}
