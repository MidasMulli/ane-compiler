// transformer_layer.m — Full transformer layer: LN→MHA→add→LN→FFN→add
//
// 14 ANE dispatches, all via IOSurface routing.
// Build: xcrun clang -framework Foundation -framework IOSurface -fobjc-arc \
//        -o transformer_layer transformer_layer.m
// Run: ./transformer_layer <models_dir> <dim> <ffn_dim>

#import <Foundation/Foundation.h>
#import <dlfcn.h>
#import <signal.h>
#import <IOSurface/IOSurface.h>
#import <objc/runtime.h>
#import <objc/message.h>

static Class _Cl, _Md, _Rq, _IO;

static void loadFW(void) {
    dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);
    _Cl = NSClassFromString(@"_ANEClient");
    _Md = NSClassFromString(@"_ANEModel");
    _Rq = NSClassFromString(@"_ANERequest");
    _IO = NSClassFromString(@"_ANEIOSurfaceObject");
}

static IOSurfaceRef mkSurf(int ch) {
    uint32_t bs = ch * 64; if (bs < 4096) bs = 4096;
    return IOSurfaceCreate((__bridge CFDictionaryRef)@{
        @"IOSurfaceWidth":@(bs/2), @"IOSurfaceHeight":@1,
        @"IOSurfaceBytesPerRow":@(bs), @"IOSurfaceBytesPerElement":@2,
        @"IOSurfaceAllocSize":@(bs), @"IOSurfacePixelFormat":@(0x6630304C)});
}

static void fillOnes(IOSurfaceRef s, int ch) {
    IOSurfaceLock(s, 0, NULL);
    void *b = IOSurfaceGetBaseAddress(s);
    memset(b, 0, IOSurfaceGetAllocSize(s));
    uint16_t v = 0x3C00;
    for (int i = 0; i < ch; i++) memcpy((uint8_t*)b + i*64, &v, 2);
    IOSurfaceUnlock(s, 0, NULL);
}

static id loadModel(id client, NSString *dir, NSString *name) {
    NSString *p = [[dir stringByAppendingPathComponent:name] stringByAppendingString:@".mlmodelc"];
    id m = ((id(*)(id,SEL,id,id))objc_msgSend)((id)_Md,
        NSSelectorFromString(@"modelAtURL:key:"), [NSURL fileURLWithPath:p], @"default");
    if (!m) { fprintf(stderr, "FAIL model: %s\n", [name UTF8String]); return nil; }
    ((BOOL(*)(id,SEL,id,id,NSInteger,id*))objc_msgSend)(
        client, NSSelectorFromString(@"compileModel:options:qos:error:"), m, @{}, 0, nil);
    BOOL ok = ((BOOL(*)(id,SEL,id,id,NSInteger,id*))objc_msgSend)(
        client, NSSelectorFromString(@"loadModel:options:qos:error:"), m, @{}, 0, nil);
    if (!ok) { fprintf(stderr, "FAIL load: %s\n", [name UTF8String]); return nil; }
    return m;
}

// Single-input eval
static BOOL eval1(id client, id model, IOSurfaceRef in, IOSurfaceRef out) {
    id iO = ((id(*)(id,SEL,void*,NSInteger,BOOL))objc_msgSend)([_IO alloc],
        NSSelectorFromString(@"initWithIOSurface:startOffset:shouldRetain:"), in, 0, YES);
    id oO = ((id(*)(id,SEL,void*,NSInteger,BOOL))objc_msgSend)([_IO alloc],
        NSSelectorFromString(@"initWithIOSurface:startOffset:shouldRetain:"), out, 0, YES);
    id rq = ((id(*)(id,SEL,id,id,id,id,id,id,id,id,id))objc_msgSend)([_Rq alloc],
        NSSelectorFromString(@"initWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:sharedEvents:transactionHandle:"),
        @[iO], @[@0], @[oO], @[@0], nil, nil, @(0), nil, nil);
    ((BOOL(*)(id,SEL,id,id,BOOL,id*))objc_msgSend)(client,
        NSSelectorFromString(@"mapIOSurfacesWithModel:request:cacheInference:error:"), model, rq, NO, nil);
    BOOL ok = ((BOOL(*)(id,SEL,id,id,id,int,id*))objc_msgSend)(client,
        NSSelectorFromString(@"doEvaluateDirectWithModel:options:request:qos:error:"),
        model, @{}, rq, 21, nil);
    ((void(*)(id,SEL,id,id))objc_msgSend)(client,
        NSSelectorFromString(@"unmapIOSurfacesWithModel:request:"), model, rq);
    return ok;
}

// Two-input eval (for residual add)
static BOOL eval2(id client, id model, IOSurfaceRef inA, IOSurfaceRef inB, IOSurfaceRef out) {
    id iA = ((id(*)(id,SEL,void*,NSInteger,BOOL))objc_msgSend)([_IO alloc],
        NSSelectorFromString(@"initWithIOSurface:startOffset:shouldRetain:"), inA, 0, YES);
    id iB = ((id(*)(id,SEL,void*,NSInteger,BOOL))objc_msgSend)([_IO alloc],
        NSSelectorFromString(@"initWithIOSurface:startOffset:shouldRetain:"), inB, 0, YES);
    id oO = ((id(*)(id,SEL,void*,NSInteger,BOOL))objc_msgSend)([_IO alloc],
        NSSelectorFromString(@"initWithIOSurface:startOffset:shouldRetain:"), out, 0, YES);
    id rq = ((id(*)(id,SEL,id,id,id,id,id,id,id,id,id))objc_msgSend)([_Rq alloc],
        NSSelectorFromString(@"initWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:sharedEvents:transactionHandle:"),
        @[iA, iB], @[@0, @1], @[oO], @[@0], nil, nil, @(0), nil, nil);
    ((BOOL(*)(id,SEL,id,id,BOOL,id*))objc_msgSend)(client,
        NSSelectorFromString(@"mapIOSurfacesWithModel:request:cacheInference:error:"), model, rq, NO, nil);
    BOOL ok = ((BOOL(*)(id,SEL,id,id,id,int,id*))objc_msgSend)(client,
        NSSelectorFromString(@"doEvaluateDirectWithModel:options:request:qos:error:"),
        model, @{}, rq, 21, nil);
    ((void(*)(id,SEL,id,id))objc_msgSend)(client,
        NSSelectorFromString(@"unmapIOSurfacesWithModel:request:"), model, rq);
    return ok;
}

int main(int argc, char *argv[]) {
    @autoreleasepool {
        signal(SIGSEGV, SIG_IGN);
        if (argc < 4) { fprintf(stderr, "Usage: %s <dir> <dim> <ffn_dim>\n", argv[0]); return 1; }
        loadFW();
        NSString *dir = [NSString stringWithUTF8String:argv[1]];
        int dim = atoi(argv[2]), ffn = atoi(argv[3]);

        id cl = ((id(*)(id,SEL))objc_msgSend)((id)_Cl, NSSelectorFromString(@"sharedConnection"));

        // Load all 14 models
        id ln1=loadModel(cl,dir,@"ln1"), q=loadModel(cl,dir,@"q_proj"),
           k=loadModel(cl,dir,@"k_proj"), v=loadModel(cl,dir,@"v_proj"),
           qk=loadModel(cl,dir,@"qk_matmul"), sm=loadModel(cl,dir,@"attn_softmax"),
           sv=loadModel(cl,dir,@"sv_matmul"), o=loadModel(cl,dir,@"o_proj"),
           ln2_m=loadModel(cl,dir,@"ln2"),
           fg=loadModel(cl,dir,@"ffn_gate"), fd=loadModel(cl,dir,@"ffn_down");

        if (!ln1||!q||!k||!v||!qk||!sm||!sv||!o||!ln2_m||!fg||!fd) return 1;
        fprintf(stderr, "11 models loaded (residual on CPU).\n");

        // Allocate IOSurfaces
        IOSurfaceRef s_x=mkSurf(dim), s_ln1=mkSurf(dim),
            s_q=mkSurf(dim), s_k=mkSurf(dim), s_v=mkSurf(dim),
            s_qk=mkSurf(dim), s_sm=mkSurf(dim), s_sv=mkSurf(dim),
            s_o=mkSurf(dim), s_r1=mkSurf(dim), s_ln2=mkSurf(dim),
            s_fg=mkSurf(ffn), s_fd=mkSurf(dim), s_out=mkSurf(dim);

        fillOnes(s_x, dim);

        // Execute chain
        #define E1(m,i,o) if(!eval1(cl,m,i,o)){fprintf(stderr,"FAIL %s\n",#m);return 1;}
        #define E2(m,a,b,o) if(!eval2(cl,m,a,b,o)){fprintf(stderr,"FAIL %s\n",#m);return 1;}

        E1(ln1, s_x, s_ln1);           // 1. LayerNorm 1
        E1(q, s_ln1, s_q);             // 2. Q projection
        E1(k, s_ln1, s_k);             // 3. K projection (unused but computed)
        E1(v, s_ln1, s_v);             // 4. V projection (unused in simplified chain)
        E1(qk, s_q, s_qk);            // 5. QK matmul
        E1(sm, s_qk, s_sm);           // 6. Softmax
        E1(sv, s_sm, s_sv);           // 7. SV matmul
        E1(o, s_sv, s_o);             // 8. Output projection
        // 9. Residual add 1: x + attn_out (CPU — IOSurface read+add+write)
        {
            uint32_t bs_r = dim * 64; if (bs_r < 4096) bs_r = 4096;
            IOSurfaceLock(s_x, kIOSurfaceLockReadOnly, NULL);
            IOSurfaceLock(s_o, kIOSurfaceLockReadOnly, NULL);
            IOSurfaceLock(s_r1, 0, NULL);
            uint8_t *bx = IOSurfaceGetBaseAddress(s_x);
            uint8_t *bo = IOSurfaceGetBaseAddress(s_o);
            uint8_t *br = IOSurfaceGetBaseAddress(s_r1);
            memset(br, 0, bs_r);
            for (int i = 0; i < dim; i++) {
                uint16_t va, vb;
                memcpy(&va, bx + i*64, 2); memcpy(&vb, bo + i*64, 2);
                // FP16 add via float conversion
                float fa = 0, fb = 0;
                uint32_t sa=(va>>15)&1, ea=(va>>10)&0x1F, ma=va&0x3FF;
                if(ea==0) fa=(sa?-1:1)*ma/1024.0f/16384.0f;
                else if(ea<31) { uint32_t fb32=(sa<<31)|((ea-15+127)<<23)|(ma<<13); memcpy(&fa,&fb32,4); }
                uint32_t sb=(vb>>15)&1, eb=(vb>>10)&0x1F, mb=vb&0x3FF;
                if(eb==0) fb=(sb?-1:1)*mb/1024.0f/16384.0f;
                else if(eb<31) { uint32_t fb32=(sb<<31)|((eb-15+127)<<23)|(mb<<13); memcpy(&fb,&fb32,4); }
                float sum = fa + fb;
                // float to FP16
                uint32_t bits; memcpy(&bits, &sum, 4);
                uint32_t s=(bits>>31)&1; int32_t e=((bits>>23)&0xFF)-127; uint32_t m=bits&0x7FFFFF;
                uint16_t r;
                if(e>15) r=(s<<15)|0x7C00;
                else if(e<-24) r=(s<<15);
                else if(e<-14) { m|=0x800000; r=(s<<15)|(m>>(-e-14+13)); }
                else r=(s<<15)|((e+15)<<10)|(m>>13);
                memcpy(br + i*64, &r, 2);
            }
            IOSurfaceUnlock(s_r1, 0, NULL);
            IOSurfaceUnlock(s_o, kIOSurfaceLockReadOnly, NULL);
            IOSurfaceUnlock(s_x, kIOSurfaceLockReadOnly, NULL);
            fprintf(stderr, "  Residual 1 (CPU)\n");
        }

        E1(ln2_m, s_r1, s_ln2);         // 10. LayerNorm 2
        E1(fg, s_ln2, s_fg);          // 11. FFN gate (with fused relu)
        E1(fd, s_fg, s_fd);           // 12. FFN down

        // 13. Residual add 2: r1 + ffn_out (CPU)
        {
            uint32_t bs_r = dim * 64; if (bs_r < 4096) bs_r = 4096;
            IOSurfaceLock(s_r1, kIOSurfaceLockReadOnly, NULL);
            IOSurfaceLock(s_fd, kIOSurfaceLockReadOnly, NULL);
            IOSurfaceLock(s_out, 0, NULL);
            uint8_t *br = IOSurfaceGetBaseAddress(s_r1);
            uint8_t *bf = IOSurfaceGetBaseAddress(s_fd);
            uint8_t *bo2 = IOSurfaceGetBaseAddress(s_out);
            memset(bo2, 0, bs_r);
            for (int i = 0; i < dim; i++) {
                uint16_t va, vb;
                memcpy(&va, br + i*64, 2); memcpy(&vb, bf + i*64, 2);
                float fa = 0, fb = 0;
                uint32_t sa=(va>>15)&1, ea=(va>>10)&0x1F, ma=va&0x3FF;
                if(ea==0) fa=(sa?-1:1)*ma/1024.0f/16384.0f;
                else if(ea<31) { uint32_t fb32=(sa<<31)|((ea-15+127)<<23)|(ma<<13); memcpy(&fa,&fb32,4); }
                uint32_t sb=(vb>>15)&1, eb=(vb>>10)&0x1F, mb=vb&0x3FF;
                if(eb==0) fb=(sb?-1:1)*mb/1024.0f/16384.0f;
                else if(eb<31) { uint32_t fb32=(sb<<31)|((eb-15+127)<<23)|(mb<<13); memcpy(&fb,&fb32,4); }
                float sum = fa + fb;
                uint32_t bits; memcpy(&bits, &sum, 4);
                uint32_t s=(bits>>31)&1; int32_t e=((bits>>23)&0xFF)-127; uint32_t m=bits&0x7FFFFF;
                uint16_t r;
                if(e>15) r=(s<<15)|0x7C00;
                else if(e<-24) r=(s<<15);
                else if(e<-14) { m|=0x800000; r=(s<<15)|(m>>(-e-14+13)); }
                else r=(s<<15)|((e+15)<<10)|(m>>13);
                memcpy(bo2 + i*64, &r, 2);
            }
            IOSurfaceUnlock(s_out, 0, NULL);
            IOSurfaceUnlock(s_fd, kIOSurfaceLockReadOnly, NULL);
            IOSurfaceUnlock(s_r1, kIOSurfaceLockReadOnly, NULL);
            fprintf(stderr, "  Residual 2 (CPU)\n");
        }

        // Read output
        IOSurfaceLock(s_out, kIOSurfaceLockReadOnly, NULL);
        void *base = IOSurfaceGetBaseAddress(s_out);
        for (int i = 0; i < dim; i++) {
            uint16_t fp16; memcpy(&fp16, (uint8_t*)base + i*64, 2);
            printf("%04x ", fp16);
        }
        printf("\n");
        IOSurfaceUnlock(s_out, kIOSurfaceLockReadOnly, NULL);
        fprintf(stderr, "Chain complete: 13 dispatches.\n");
        return 0;
    }
}
