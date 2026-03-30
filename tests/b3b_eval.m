// b3b_eval.m — Compile + evaluate 2-input elementwise model, print output
//
// Build: xcrun clang -framework Foundation -framework IOSurface -fobjc-arc \
//        -o b3b_eval b3b_eval.m
// Run: ./b3b_eval <model.mlmodelc>
// Outputs JSON with input_a, input_b, output arrays

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
    _C = NSClassFromString(@"_ANEClient");
    _M = NSClassFromString(@"_ANEModel");
    _R = NSClassFromString(@"_ANERequest");
    _IO = NSClassFromString(@"_ANEIOSurfaceObject");
}

static IOSurfaceRef mkSurf(uint32_t sz) {
    return IOSurfaceCreate((__bridge CFDictionaryRef)@{
        @"IOSurfaceWidth": @(sz/2), @"IOSurfaceHeight": @1,
        @"IOSurfaceBytesPerRow": @(sz), @"IOSurfaceBytesPerElement": @2,
        @"IOSurfaceAllocSize": @(sz), @"IOSurfacePixelFormat": @(0x6630304C),
    });
}

static uint16_t f2h(float v) {
    uint32_t b; memcpy(&b, &v, 4);
    uint32_t s = (b>>31)&1; int32_t e = ((b>>23)&0xFF)-127; uint32_t m = b&0x7FFFFF;
    if (e>15) return (s<<15)|0x7C00;
    if (e<-14) return (s<<15);
    return (s<<15)|((e+15)<<10)|(m>>13);
}

static float h2f(uint16_t fp16) {
    uint32_t s=(fp16>>15)&1, e=(fp16>>10)&0x1F, m=fp16&0x3FF;
    if (e==0) return (s?-1.0f:1.0f)*m/1024.0f/16384.0f;
    if (e==31) return m?NAN:(s?-INFINITY:INFINITY);
    uint32_t fb=(s<<31)|((e-15+127)<<23)|(m<<13);
    float v; memcpy(&v,&fb,4); return v;
}

int main(int argc, char *argv[]) {
    @autoreleasepool {
        signal(SIGSEGV, SIG_IGN);
        if (argc < 2) { fprintf(stderr, "Usage: %s <model.mlmodelc>\n", argv[0]); return 1; }
        loadFW();

        NSString *path = [NSString stringWithUTF8String:argv[1]];
        id cli = ((id(*)(id,SEL))objc_msgSend)((id)_C, NSSelectorFromString(@"sharedConnection"));
        id mdl = ((id(*)(id,SEL,id,id))objc_msgSend)((id)_M,
            NSSelectorFromString(@"modelAtURL:key:"),
            [NSURL fileURLWithPath:path], @"eval");
        if (!mdl) { printf("{\"eval_ok\":false,\"error\":\"model_create\"}\n"); return 0; }

        ((void(*)(id,SEL,id))objc_msgSend)(cli, NSSelectorFromString(@"purgeCompiledModel:"), mdl);
        usleep(200000);
        NSError *err=nil;
        BOOL ok = ((BOOL(*)(id,SEL,id,id,NSInteger,id*))objc_msgSend)(
            cli, NSSelectorFromString(@"compileModel:options:qos:error:"), mdl, @{}, 0, &err);
        if (!ok) { printf("{\"eval_ok\":false,\"error\":\"compile\"}\n"); return 0; }

        ok = ((BOOL(*)(id,SEL,id,id,NSInteger,id*))objc_msgSend)(
            cli, NSSelectorFromString(@"loadModel:options:qos:error:"), mdl, @{}, 0, &err);
        if (!ok) { printf("{\"eval_ok\":false,\"error\":\"load\"}\n"); return 0; }

        id attrs = ((id(*)(id,SEL))objc_msgSend)(mdl, NSSelectorFromString(@"modelAttributes"));
        NSDictionary *ns = [attrs[@"NetworkStatusList"] firstObject];
        NSArray *ins = ns[@"LiveInputList"];
        NSArray *outs = ns[@"LiveOutputList"];

        uint32_t bs0=[ins[0][@"BatchStride"] unsignedIntValue];
        uint32_t bs1=[ins[1][@"BatchStride"] unsignedIntValue];
        uint32_t bso=[outs[0][@"BatchStride"] unsignedIntValue];
        uint32_t ps=[ins[0][@"PlaneStride"] unsignedIntValue];

        int ch = 64;
        // Test vectors: diverse values including negatives
        float inA[64], inB[64];
        // A = [3.0, -2.0, 6.0, 0.5, 9.0, -1.0, 4.0, 8.0, repeated]
        float patA[] = {3.0f, -2.0f, 6.0f, 0.5f, 9.0f, -1.0f, 4.0f, 8.0f};
        float patB[] = {2.0f, 3.0f, -1.5f, 4.0f, 1.0f, -2.0f, 2.0f, 0.5f};
        for (int i = 0; i < ch; i++) { inA[i] = patA[i%8]; inB[i] = patB[i%8]; }

        IOSurfaceRef sA=mkSurf(bs0), sB=mkSurf(bs1), sO=mkSurf(bso);

        IOSurfaceLock(sA,0,NULL);
        void *bA=IOSurfaceGetBaseAddress(sA);
        memset(bA,0,bs0);
        for (int i=0;i<ch;i++) { uint16_t h=f2h(inA[i]); memcpy((uint8_t*)bA+i*ps,&h,2); }
        IOSurfaceUnlock(sA,0,NULL);

        IOSurfaceLock(sB,0,NULL);
        void *bB=IOSurfaceGetBaseAddress(sB);
        memset(bB,0,bs1);
        for (int i=0;i<ch;i++) { uint16_t h=f2h(inB[i]); memcpy((uint8_t*)bB+i*ps,&h,2); }
        IOSurfaceUnlock(sB,0,NULL);

        id oA=((id(*)(id,SEL,void*,NSInteger,BOOL))objc_msgSend)([_IO alloc],
            NSSelectorFromString(@"initWithIOSurface:startOffset:shouldRetain:"),sA,0,YES);
        id oB=((id(*)(id,SEL,void*,NSInteger,BOOL))objc_msgSend)([_IO alloc],
            NSSelectorFromString(@"initWithIOSurface:startOffset:shouldRetain:"),sB,0,YES);
        id oO=((id(*)(id,SEL,void*,NSInteger,BOOL))objc_msgSend)([_IO alloc],
            NSSelectorFromString(@"initWithIOSurface:startOffset:shouldRetain:"),sO,0,YES);

        id req=((id(*)(id,SEL,id,id,id,id,id,id,id,id,id))objc_msgSend)(
            [_R alloc],
            NSSelectorFromString(@"initWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:sharedEvents:transactionHandle:"),
            @[oA,oB],@[@0,@1],@[oO],@[@0],nil,nil,@(0),nil,nil);

        ((BOOL(*)(id,SEL,id,id,BOOL,id*))objc_msgSend)(
            cli,NSSelectorFromString(@"mapIOSurfacesWithModel:request:cacheInference:error:"),
            mdl,req,NO,nil);

        ok=((BOOL(*)(id,SEL,id,id,id,int,id*))objc_msgSend)(
            cli,NSSelectorFromString(@"doEvaluateDirectWithModel:options:request:qos:error:"),
            mdl,@{},req,21,nil);

        float out[64]={0};
        if (ok) {
            IOSurfaceLock(sO,kIOSurfaceLockReadOnly,NULL);
            void *bO=IOSurfaceGetBaseAddress(sO);
            for (int i=0;i<ch;i++) {
                uint16_t h; memcpy(&h,(uint8_t*)bO+i*ps,2);
                out[i]=h2f(h);
            }
            IOSurfaceUnlock(sO,kIOSurfaceLockReadOnly,NULL);
        }

        ((void(*)(id,SEL,id,id))objc_msgSend)(cli,
            NSSelectorFromString(@"unmapIOSurfacesWithModel:request:"),mdl,req);
        ((BOOL(*)(id,SEL,id,id,NSInteger,id*))objc_msgSend)(
            cli,NSSelectorFromString(@"doUnloadModel:options:qos:error:"),mdl,@{},0,&err);

        // JSON output
        printf("{\"eval_ok\":%s", ok?"true":"false");
        if (ok) {
            printf(",\"input_a\":[");
            for(int i=0;i<8;i++) printf("%s%.4f",i?",":"",inA[i]);
            printf("],\"input_b\":[");
            for(int i=0;i<8;i++) printf("%s%.4f",i?",":"",inB[i]);
            printf("],\"output\":[");
            for(int i=0;i<8;i++) printf("%s%.6f",i?",":"",out[i]);
            printf("]");
        }
        printf("}\n");

        CFRelease(sA); CFRelease(sB); CFRelease(sO);
        return 0;
    }
}
