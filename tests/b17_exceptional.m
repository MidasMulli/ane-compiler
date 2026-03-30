// b17_exceptional.m — Test NaN/Inf/denormal behavior on ANE
//
// Build: xcrun clang -framework Foundation -framework IOSurface -fobjc-arc \
//        -o b17_exceptional b17_exceptional.m -lm
// Run: ./b17_exceptional <model.mlmodelc>

#import <Foundation/Foundation.h>
#import <dlfcn.h>
#import <signal.h>
#import <IOSurface/IOSurface.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <math.h>

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
        if (argc < 2) { fprintf(stderr, "Usage: %s <model.mlmodelc>\n", argv[0]); return 1; }
        loadFW();

        NSString *path = [NSString stringWithUTF8String:argv[1]];
        id cli = ((id(*)(id,SEL))objc_msgSend)((id)_C, NSSelectorFromString(@"sharedConnection"));
        id mdl = ((id(*)(id,SEL,id,id))objc_msgSend)((id)_M,
            NSSelectorFromString(@"modelAtURL:key:"), [NSURL fileURLWithPath:path], @"b17");
        if (!mdl) { printf("{\"error\":\"model\"}\n"); return 0; }

        ((void(*)(id,SEL,id))objc_msgSend)(cli, NSSelectorFromString(@"purgeCompiledModel:"), mdl);
        usleep(200000);
        NSError *err = nil;
        ((BOOL(*)(id,SEL,id,id,NSInteger,id*))objc_msgSend)(
            cli, NSSelectorFromString(@"compileModel:options:qos:error:"), mdl, @{}, 0, &err);
        ((BOOL(*)(id,SEL,id,id,NSInteger,id*))objc_msgSend)(
            cli, NSSelectorFromString(@"loadModel:options:qos:error:"), mdl, @{}, 0, &err);

        id attrs = ((id(*)(id,SEL))objc_msgSend)(mdl, NSSelectorFromString(@"modelAttributes"));
        NSDictionary *ns = [attrs[@"NetworkStatusList"] firstObject];
        uint32_t inBS = [ns[@"LiveInputList"][0][@"BatchStride"] unsignedIntValue];
        uint32_t outBS = [ns[@"LiveOutputList"][0][@"BatchStride"] unsignedIntValue];
        uint32_t ps = [ns[@"LiveInputList"][0][@"PlaneStride"] unsignedIntValue];

        IOSurfaceRef sI = mkSurf(inBS), sO = mkSurf(outBS);

        // Test values as raw FP16
        // NaN=0x7E00, Inf=0x7C00, -Inf=0xFC00, denormal=0x0001, -0=0x8000
        // FP16_MAX=0x7BFF (65504), -FP16_MAX=0xFBFF, smallest_normal=0x0400
        uint16_t test_fp16[] = {
            0x7E00, // NaN
            0x7C00, // +Inf
            0xFC00, // -Inf
            0x0001, // smallest denormal (5.96e-8)
            0x8000, // -0.0
            0x7BFF, // FP16_MAX (65504)
            0xFBFF, // -FP16_MAX
            0x0400, // smallest normal (6.1e-5)
        };
        const char *test_names[] = {
            "NaN", "+Inf", "-Inf", "denormal", "-0.0",
            "FP16_MAX", "-FP16_MAX", "smallest_normal"
        };
        int n_tests = 8;

        IOSurfaceLock(sI, 0, NULL);
        void *base = IOSurfaceGetBaseAddress(sI);
        memset(base, 0, inBS);
        for (int i = 0; i < n_tests; i++) {
            memcpy((uint8_t*)base + i * ps, &test_fp16[i], 2);
        }
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

        BOOL ok = ((BOOL(*)(id,SEL,id,id,id,int,id*))objc_msgSend)(
            cli, NSSelectorFromString(@"doEvaluateDirectWithModel:options:request:qos:error:"),
            mdl, @{}, req, 21, nil);

        printf("{\"eval_ok\":%s,\"model\":\"%s\"", ok?"true":"false", argv[1]);

        if (ok) {
            IOSurfaceLock(sO, kIOSurfaceLockReadOnly, NULL);
            void *oBase = IOSurfaceGetBaseAddress(sO);
            printf(",\"results\":[");
            for (int i = 0; i < n_tests; i++) {
                uint16_t out_fp16;
                memcpy(&out_fp16, (uint8_t*)oBase + i * ps, 2);

                // Classify
                uint32_t sign = (out_fp16 >> 15) & 1;
                uint32_t exp = (out_fp16 >> 10) & 0x1F;
                uint32_t mant = out_fp16 & 0x3FF;
                const char *cls;
                if (exp == 31 && mant != 0) cls = "NaN";
                else if (exp == 31 && mant == 0) cls = sign ? "-Inf" : "+Inf";
                else if (exp == 0 && mant == 0) cls = sign ? "-0" : "+0";
                else if (exp == 0) cls = "denormal";
                else cls = "normal";

                // Convert to float for display
                float val;
                if (exp == 0 && mant == 0) val = sign ? -0.0f : 0.0f;
                else if (exp == 31) val = mant ? NAN : (sign ? -INFINITY : INFINITY);
                else if (exp == 0) val = (sign ? -1.0f : 1.0f) * mant / 1024.0f / 16384.0f;
                else {
                    uint32_t fb = (sign<<31)|((exp-15+127)<<23)|(mant<<13);
                    memcpy(&val, &fb, 4);
                }

                printf("%s{\"input\":\"%s\",\"in_fp16\":\"0x%04X\",\"out_fp16\":\"0x%04X\","
                       "\"out_class\":\"%s\",\"out_val\":",
                       i?",":"", test_names[i], test_fp16[i], out_fp16, cls);
                if (isnan(val)) printf("\"NaN\"");
                else if (isinf(val)) printf("\"%s\"", val > 0 ? "+Inf" : "-Inf");
                else printf("%.6g", val);
                printf("}");
            }
            printf("]");
            IOSurfaceUnlock(sO, kIOSurfaceLockReadOnly, NULL);
        }
        printf("}\n");

        ((void(*)(id,SEL,id,id))objc_msgSend)(cli,
            NSSelectorFromString(@"unmapIOSurfacesWithModel:request:"), mdl, req);
        ((BOOL(*)(id,SEL,id,id,NSInteger,id*))objc_msgSend)(
            cli, NSSelectorFromString(@"doUnloadModel:options:qos:error:"), mdl, @{}, 0, &err);
        CFRelease(sI); CFRelease(sO);
        return 0;
    }
}
