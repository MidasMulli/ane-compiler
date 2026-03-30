// b3b_verify.m — Verify whether _ANEClient loadModel recompiles over cache patch
//
// Build: xcrun clang -framework Foundation -framework IOSurface -fobjc-arc \
//        -o b3b_verify b3b_verify.m
// Run: ./b3b_verify <model.mlmodelc>

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

static NSString *findRecentHWX(void) {
    NSString *cacheBase = @"/Library/Caches/com.apple.aned";
    NSFileManager *fm = [NSFileManager defaultManager];
    NSDate *cutoff = [NSDate dateWithTimeIntervalSinceNow:-30.0];
    NSMutableArray *candidates = [NSMutableArray array];
    NSDirectoryEnumerator *enumerator = [fm enumeratorAtPath:cacheBase];
    NSString *path;
    while ((path = [enumerator nextObject])) {
        if ([path.pathExtension isEqualToString:@"hwx"]) {
            NSString *full = [cacheBase stringByAppendingPathComponent:path];
            NSDictionary *attrs = [fm attributesOfItemAtPath:full error:nil];
            NSDate *mod = attrs[NSFileModificationDate];
            if ([mod compare:cutoff] == NSOrderedDescending) {
                [candidates addObject:@{@"path": full, @"date": mod}];
            }
        }
    }
    if (candidates.count == 0) return nil;
    [candidates sortUsingComparator:^NSComparisonResult(id a, id b) {
        return [b[@"date"] compare:a[@"date"]];
    }];
    return candidates[0][@"path"];
}

static uint32_t readW51(NSString *hwxPath) {
    NSData *data = [NSData dataWithContentsOfFile:hwxPath];
    const uint8_t *bytes = (const uint8_t *)[data bytes];
    uint32_t ncmds = *(uint32_t *)(bytes + 0x10);
    uint32_t off = 32;
    for (uint32_t i = 0; i < ncmds; i++) {
        uint32_t cmd = *(uint32_t *)(bytes + off);
        uint32_t cmdsize = *(uint32_t *)(bytes + off + 4);
        if (cmd == 0x19) {
            char segname[17] = {0};
            memcpy(segname, bytes + off + 8, 16);
            uint32_t nsects = *(uint32_t *)(bytes + off + 64);
            for (uint32_t s = 0; s < nsects; s++) {
                uint32_t soff = off + 72 + s * 80;
                char sectname[17] = {0};
                memcpy(sectname, bytes + soff, 16);
                if (strcmp(segname, "__TEXT") == 0 && strcmp(sectname, "__text") == 0) {
                    uint32_t foff = *(uint32_t *)(bytes + soff + 48);
                    return *(uint32_t *)(bytes + foff + 51 * 4);
                }
            }
        }
        off += cmdsize;
    }
    return 0xDEADBEEF;
}

static IOSurfaceRef makeSurface(uint32_t allocSize) {
    NSDictionary *props = @{
        @"IOSurfaceWidth": @(allocSize / 2),
        @"IOSurfaceHeight": @1,
        @"IOSurfaceBytesPerRow": @(allocSize),
        @"IOSurfaceBytesPerElement": @2,
        @"IOSurfaceAllocSize": @(allocSize),
        @"IOSurfacePixelFormat": @(0x6630304C),
    };
    return IOSurfaceCreate((__bridge CFDictionaryRef)props);
}

static void fillFP16(IOSurfaceRef surf, float *values, int count, int planeStride) {
    IOSurfaceLock(surf, 0, NULL);
    void *base = IOSurfaceGetBaseAddress(surf);
    memset(base, 0, IOSurfaceGetAllocSize(surf));
    for (int i = 0; i < count; i++) {
        float v = values[i];
        uint32_t bits; memcpy(&bits, &v, 4);
        uint32_t sign = (bits >> 31) & 1;
        int32_t exp = ((bits >> 23) & 0xFF) - 127;
        uint32_t mant = bits & 0x7FFFFF;
        uint16_t fp16;
        if (exp > 15) fp16 = (sign << 15) | 0x7C00;
        else if (exp < -14) fp16 = (sign << 15);
        else fp16 = (sign << 15) | ((exp + 15) << 10) | (mant >> 13);
        memcpy((uint8_t*)base + i * planeStride, &fp16, 2);
    }
    IOSurfaceUnlock(surf, 0, NULL);
}

static float readFP16(void *base, int idx, int planeStride) {
    uint16_t fp16;
    memcpy(&fp16, (uint8_t*)base + idx * planeStride, 2);
    uint32_t sign = (fp16 >> 15) & 1;
    uint32_t e = (fp16 >> 10) & 0x1F;
    uint32_t m = fp16 & 0x3FF;
    if (e == 0) return (sign ? -1.0f : 1.0f) * m / 1024.0f / 16384.0f;
    if (e == 31) return m ? NAN : (sign ? -INFINITY : INFINITY);
    uint32_t fb = (sign << 31) | ((e - 15 + 127) << 23) | (m << 13);
    float val; memcpy(&val, &fb, 4); return val;
}

int main(int argc, char *argv[]) {
    @autoreleasepool {
        signal(SIGSEGV, SIG_IGN);
        if (argc < 2) { fprintf(stderr, "Usage: %s <model.mlmodelc>\n", argv[0]); return 1; }
        loadFW();

        NSString *modelPath = [NSString stringWithUTF8String:argv[1]];
        id client = ((id (*)(id, SEL))objc_msgSend)((id)_ANEClientCls, NSSelectorFromString(@"sharedConnection"));
        NSURL *url = [NSURL fileURLWithPath:modelPath];
        id model = ((id (*)(id, SEL, id, id))objc_msgSend)((id)_ANEModelCls, NSSelectorFromString(@"modelAtURL:key:"), url, @"verify");

        if (!model) { printf("MODEL_CREATE_FAILED\n"); return 1; }

        // Step 1: Purge + compile
        ((void (*)(id, SEL, id))objc_msgSend)(client, NSSelectorFromString(@"purgeCompiledModel:"), model);
        usleep(200000);
        NSError *err = nil;
        BOOL ok = ((BOOL (*)(id, SEL, id, id, NSInteger, id*))objc_msgSend)(
            client, NSSelectorFromString(@"compileModel:options:qos:error:"), model, @{}, 0, &err);
        if (!ok) { printf("COMPILE_FAILED\n"); return 1; }

        NSString *hwxPath = findRecentHWX();
        if (!hwxPath) { printf("HWX_NOT_FOUND\n"); return 1; }

        printf("Step 1 - After compile:    W[51] = 0x%08X\n", readW51(hwxPath));

        // Step 2: Load
        ok = ((BOOL (*)(id, SEL, id, id, NSInteger, id*))objc_msgSend)(
            client, NSSelectorFromString(@"loadModel:options:qos:error:"), model, @{}, 0, &err);
        printf("Step 2 - After load:       W[51] = 0x%08X (load=%s)\n", readW51(hwxPath), ok ? "OK" : "FAIL");

        // Step 3: Evaluate with reference data
        id attrs = ((id (*)(id, SEL))objc_msgSend)(model, NSSelectorFromString(@"modelAttributes"));
        NSDictionary *ns = [attrs[@"NetworkStatusList"] firstObject];
        NSArray *inputs = ns[@"LiveInputList"];
        NSArray *outputs = ns[@"LiveOutputList"];
        uint32_t inBS0 = [inputs[0][@"BatchStride"] unsignedIntValue];
        uint32_t inBS1 = [inputs[1][@"BatchStride"] unsignedIntValue];
        uint32_t outBS = [outputs[0][@"BatchStride"] unsignedIntValue];
        uint32_t ps = [inputs[0][@"PlaneStride"] unsignedIntValue];

        float inA[64], inB[64];
        for (int i = 0; i < 64; i++) { inA[i] = 3.0f + (i%8)*1.5f; inB[i] = 1.0f + ((i+3)%7)*0.5f; }

        IOSurfaceRef sA = makeSurface(inBS0), sB = makeSurface(inBS1), sO = makeSurface(outBS);
        fillFP16(sA, inA, 64, ps);
        fillFP16(sB, inB, 64, ps);

        id oA = ((id (*)(id, SEL, void*, NSInteger, BOOL))objc_msgSend)([_ANEIOSurfaceObjectCls alloc],
            NSSelectorFromString(@"initWithIOSurface:startOffset:shouldRetain:"), sA, 0, YES);
        id oB = ((id (*)(id, SEL, void*, NSInteger, BOOL))objc_msgSend)([_ANEIOSurfaceObjectCls alloc],
            NSSelectorFromString(@"initWithIOSurface:startOffset:shouldRetain:"), sB, 0, YES);
        id oO = ((id (*)(id, SEL, void*, NSInteger, BOOL))objc_msgSend)([_ANEIOSurfaceObjectCls alloc],
            NSSelectorFromString(@"initWithIOSurface:startOffset:shouldRetain:"), sO, 0, YES);

        id req = ((id (*)(id, SEL, id, id, id, id, id, id, id, id, id))objc_msgSend)(
            [_ANERequestCls alloc],
            NSSelectorFromString(@"initWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:sharedEvents:transactionHandle:"),
            @[oA, oB], @[@0, @1], @[oO], @[@0], nil, nil, @(0), nil, nil);

        ((BOOL (*)(id, SEL, id, id, BOOL, id*))objc_msgSend)(
            client, NSSelectorFromString(@"mapIOSurfacesWithModel:request:cacheInference:error:"),
            model, req, NO, nil);

        ok = ((BOOL (*)(id, SEL, id, id, id, int, id*))objc_msgSend)(
            client, NSSelectorFromString(@"doEvaluateDirectWithModel:options:request:qos:error:"),
            model, @{}, req, 21, nil);
        printf("Step 3 - Eval reference:   %s\n", ok ? "OK" : "FAIL");

        IOSurfaceLock(sO, kIOSurfaceLockReadOnly, NULL);
        void *oBase = IOSurfaceGetBaseAddress(sO);
        printf("  ref out[0..7]: ");
        for (int i = 0; i < 8; i++) printf("%.4f ", readFP16(oBase, i, ps));
        printf("\n");
        IOSurfaceUnlock(sO, kIOSurfaceLockReadOnly, NULL);

        ((void (*)(id, SEL, id, id))objc_msgSend)(client,
            NSSelectorFromString(@"unmapIOSurfacesWithModel:request:"), model, req);

        // Step 4: Unload
        ((BOOL (*)(id, SEL, id, id, NSInteger, id*))objc_msgSend)(
            client, NSSelectorFromString(@"doUnloadModel:options:qos:error:"), model, @{}, 0, &err);
        printf("Step 4 - After unload:     W[51] = 0x%08X\n", readW51(hwxPath));

        // Step 5: Patch W[51] to MUL
        NSMutableData *patched = [NSMutableData dataWithContentsOfFile:hwxPath];
        const uint8_t *pBytes = (const uint8_t *)[patched bytes];
        // Find text offset
        uint32_t ncmds2 = *(uint32_t *)(pBytes + 0x10);
        uint32_t off2 = 32, textOff2 = 0;
        for (uint32_t i = 0; i < ncmds2; i++) {
            uint32_t cmd = *(uint32_t *)(pBytes + off2);
            uint32_t cs = *(uint32_t *)(pBytes + off2 + 4);
            if (cmd == 0x19) {
                char seg[17]={0}; memcpy(seg, pBytes+off2+8, 16);
                uint32_t ns2 = *(uint32_t *)(pBytes + off2 + 64);
                for (uint32_t s = 0; s < ns2; s++) {
                    uint32_t so = off2+72+s*80;
                    char sn[17]={0}; memcpy(sn, pBytes+so, 16);
                    if (!strcmp(seg,"__TEXT") && !strcmp(sn,"__text"))
                        textOff2 = *(uint32_t *)(pBytes + so + 48);
                }
            }
            off2 += cs;
        }
        uint8_t *mutable = (uint8_t *)[patched mutableBytes];
        *(uint32_t *)(mutable + textOff2 + 51*4) = 0x00080004;  // MUL
        [patched writeToFile:hwxPath atomically:NO];
        printf("Step 5 - After patch:      W[51] = 0x%08X\n", readW51(hwxPath));

        // Step 6: Load WITHOUT recompile
        ok = ((BOOL (*)(id, SEL, id, id, NSInteger, id*))objc_msgSend)(
            client, NSSelectorFromString(@"loadModel:options:qos:error:"), model, @{}, 0, &err);
        printf("Step 6 - After re-load:    W[51] = 0x%08X (load=%s)\n", readW51(hwxPath), ok ? "OK" : "FAIL");

        // Step 7: Evaluate with patched .hwx
        IOSurfaceRef sO2 = makeSurface(outBS);
        fillFP16(sA, inA, 64, ps);
        fillFP16(sB, inB, 64, ps);

        id oO2 = ((id (*)(id, SEL, void*, NSInteger, BOOL))objc_msgSend)([_ANEIOSurfaceObjectCls alloc],
            NSSelectorFromString(@"initWithIOSurface:startOffset:shouldRetain:"), sO2, 0, YES);
        id req2 = ((id (*)(id, SEL, id, id, id, id, id, id, id, id, id))objc_msgSend)(
            [_ANERequestCls alloc],
            NSSelectorFromString(@"initWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:sharedEvents:transactionHandle:"),
            @[oA, oB], @[@0, @1], @[oO2], @[@0], nil, nil, @(0), nil, nil);
        ((BOOL (*)(id, SEL, id, id, BOOL, id*))objc_msgSend)(
            client, NSSelectorFromString(@"mapIOSurfacesWithModel:request:cacheInference:error:"),
            model, req2, NO, nil);
        ok = ((BOOL (*)(id, SEL, id, id, id, int, id*))objc_msgSend)(
            client, NSSelectorFromString(@"doEvaluateDirectWithModel:options:request:qos:error:"),
            model, @{}, req2, 21, nil);
        printf("Step 7 - Eval patched:     %s\n", ok ? "OK" : "FAIL");

        IOSurfaceLock(sO2, kIOSurfaceLockReadOnly, NULL);
        void *o2Base = IOSurfaceGetBaseAddress(sO2);
        printf("  pat out[0..7]: ");
        for (int i = 0; i < 8; i++) printf("%.4f ", readFP16(o2Base, i, ps));
        printf("\n  exp MUL[0..7]: ");
        for (int i = 0; i < 8; i++) printf("%.4f ", inA[i] * inB[i]);
        printf("\n  exp ADD[0..7]: ");
        for (int i = 0; i < 8; i++) printf("%.4f ", inA[i] + inB[i]);
        printf("\n");
        IOSurfaceUnlock(sO2, kIOSurfaceLockReadOnly, NULL);

        ((void (*)(id, SEL, id, id))objc_msgSend)(client,
            NSSelectorFromString(@"unmapIOSurfacesWithModel:request:"), model, req2);
        ((BOOL (*)(id, SEL, id, id, NSInteger, id*))objc_msgSend)(
            client, NSSelectorFromString(@"doUnloadModel:options:qos:error:"), model, @{}, 0, &err);

        // Restore
        printf("Step 8 - Final W[51]:      0x%08X\n", readW51(hwxPath));

        CFRelease(sA); CFRelease(sB); CFRelease(sO); CFRelease(sO2);
        return 0;
    }
}
