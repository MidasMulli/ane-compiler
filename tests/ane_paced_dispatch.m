// ane_paced_dispatch.m — ONE paced dispatch loop, absolute-deadline scheduled.
//
// ⛔ WHY THIS FILE EXISTS. `ane_rate_dispatch.m` applies its throttle ONLY inside Test 3; Tests 1
// and 2 run UNTHROTTLED first. Any external observer (a reasoner measured concurrently) therefore
// sees a BLEND of full-rate and throttled phases whose composition CHANGES with the throttle
// setting. That is an instrument defect, not a property of the ANE, and it is sufficient on its own
// to void every non-full-rate point of the 2026-08-20 ANE sweep.
//
// Two fixes here:
//   1. ONE loop. No unthrottled preamble beyond a fixed 50-dispatch warmup.
//   2. ABSOLUTE-DEADLINE pacing (next = start + i*period), not `usleep(period)` after each
//      dispatch. The relative form yields achieved = 1/(dispatch + period), which undershoots the
//      target by the dispatch time — 150 us at fp16, 475 us at fp32, i.e. the undershoot is itself
//      a function of the precision under test. nanosleep, never a spin: a busy-wait contends
//      through the CPU and is indistinguishable from the effect being measured.
//
// Runs for a WALL-CLOCK duration so the peripheral covers the reasoner's window exactly, and
// reports the ACHIEVED rate. Achieved is what gets reported downstream; target never is.
#import <Foundation/Foundation.h>
#import <dlfcn.h>
#import <signal.h>
#import <IOSurface/IOSurface.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <mach/mach_time.h>
#import <unistd.h>

static int cmp_double(const void *a, const void *b) {
    double d = *(double*)a - *(double*)b;
    return d < 0 ? -1 : d > 0 ? 1 : 0;
}

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
        if (argc < 2) {
            fprintf(stderr, "Usage: %s <model.mlmodelc> [ch] [seconds] [rate_hz] [report.json]\n", argv[0]);
            return 1;
        }
        loadFW();

        int    ch      = argc > 2 ? atoi(argv[2]) : 768;
        double seconds = argc > 3 ? atof(argv[3]) : 80.0;
        double rate    = argc > 4 ? atof(argv[4]) : 0.0;
        const char *report = argc > 5 ? argv[5] : NULL;
        (void)ch;

        mach_timebase_info_data_t tb;
        mach_timebase_info(&tb);
        const double NS_PER_TICK = (double)tb.numer / (double)tb.denom;
        const double TICKS_PER_S = 1e9 / NS_PER_TICK;

        id client = ((id (*)(id, SEL))objc_msgSend)((id)_Cl, NSSelectorFromString(@"sharedConnection"));
        NSURL *url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:argv[1]]];
        id model = ((id (*)(id, SEL, id, id))objc_msgSend)(
            (id)_Mo, NSSelectorFromString(@"modelAtURL:key:"), url, @"default");

        NSError *err = nil;
        ((BOOL (*)(id, SEL, id, id, NSInteger, id*))objc_msgSend)(
            client, NSSelectorFromString(@"compileModel:options:qos:error:"), model, @{}, 0, &err);
        ((BOOL (*)(id, SEL, id, id, NSInteger, id*))objc_msgSend)(
            client, NSSelectorFromString(@"loadModel:options:qos:error:"), model, @{}, 0, &err);

        id attrs = ((id (*)(id, SEL))objc_msgSend)(model, NSSelectorFromString(@"modelAttributes"));
        NSDictionary *ns = [attrs[@"NetworkStatusList"] firstObject];
        if (!ns) { fprintf(stderr, "⛔ no NetworkStatusList — model did not load on ANE\n"); return 2; }
        uint32_t inBS  = [[ns[@"LiveInputList"]  firstObject][@"BatchStride"] unsignedIntValue];
        uint32_t outBS = [[ns[@"LiveOutputList"] firstObject][@"BatchStride"] unsignedIntValue];
        if (!inBS || !outBS) { fprintf(stderr, "⛔ zero BatchStride — refusing to measure nothing\n"); return 2; }

        IOSurfaceRef inSurf = IOSurfaceCreate((__bridge CFDictionaryRef)@{
            @"IOSurfaceWidth":@(inBS/2), @"IOSurfaceHeight":@1,
            @"IOSurfaceBytesPerRow":@(inBS), @"IOSurfaceBytesPerElement":@2,
            @"IOSurfaceAllocSize":@(inBS), @"IOSurfacePixelFormat":@(0x6630304C)});
        IOSurfaceRef outSurf = IOSurfaceCreate((__bridge CFDictionaryRef)@{
            @"IOSurfaceWidth":@(outBS/2), @"IOSurfaceHeight":@1,
            @"IOSurfaceBytesPerRow":@(outBS), @"IOSurfaceBytesPerElement":@2,
            @"IOSurfaceAllocSize":@(outBS), @"IOSurfacePixelFormat":@(0x6630304C)});

        id inObj = ((id (*)(id, SEL, void*, NSInteger, BOOL))objc_msgSend)(
            [_IO alloc], NSSelectorFromString(@"initWithIOSurface:startOffset:shouldRetain:"), inSurf, 0, YES);
        id outObj = ((id (*)(id, SEL, void*, NSInteger, BOOL))objc_msgSend)(
            [_IO alloc], NSSelectorFromString(@"initWithIOSurface:startOffset:shouldRetain:"), outSurf, 0, YES);

        id req = ((id (*)(id, SEL, id, id, id, id, id, id, id, id, id))objc_msgSend)(
            [_Rq alloc],
            NSSelectorFromString(@"initWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:sharedEvents:transactionHandle:"),
            @[inObj], @[@0], @[outObj], @[@0], nil, nil, @(0), nil, nil);

        ((BOOL (*)(id, SEL, id, id, BOOL, id*))objc_msgSend)(
            client, NSSelectorFromString(@"mapIOSurfacesWithModel:request:cacheInference:error:"),
            model, req, YES, nil);

        for (int i = 0; i < 50; i++) {
            ((BOOL (*)(id, SEL, id, id, id, int, id*))objc_msgSend)(
                client, NSSelectorFromString(@"doEvaluateDirectWithModel:options:request:qos:error:"),
                model, @{}, req, 21, nil);
        }

        // === the ONE loop ===
        int cap = (int)((rate > 0 ? rate : 20000.0) * seconds * 1.2) + 1024;
        double *lats = calloc(cap, sizeof(double));
        uint64_t period_ticks = (rate > 0.0) ? (uint64_t)(TICKS_PER_S / rate) : 0;
        uint64_t t_start = mach_absolute_time();
        uint64_t t_end   = t_start + (uint64_t)(seconds * TICKS_PER_S);
        long n = 0;
        while (mach_absolute_time() < t_end && n < cap) {
            uint64_t t0 = mach_absolute_time();
            ((BOOL (*)(id, SEL, id, id, id, int, id*))objc_msgSend)(
                client, NSSelectorFromString(@"doEvaluateDirectWithModel:options:request:qos:error:"),
                model, @{}, req, 21, nil);
            uint64_t t1 = mach_absolute_time();
            lats[n] = (double)(t1 - t0) * NS_PER_TICK / 1000.0;
            n++;
            if (period_ticks) {
                uint64_t deadline = t_start + (uint64_t)n * period_ticks;   // ABSOLUTE
                uint64_t now = mach_absolute_time();
                if (deadline > now) {
                    uint64_t d_ns = (uint64_t)((deadline - now) * NS_PER_TICK);
                    struct timespec ts = { .tv_sec = (time_t)(d_ns / 1000000000ULL),
                                           .tv_nsec = (long)(d_ns % 1000000000ULL) };
                    nanosleep(&ts, NULL);
                }
            }
        }
        double elapsed = (double)(mach_absolute_time() - t_start) * NS_PER_TICK / 1e9;
        double achieved = n / elapsed;

        double *srt = malloc(n * sizeof(double));
        memcpy(srt, lats, n * sizeof(double));
        qsort(srt, n, sizeof(double), cmp_double);
        double sum = 0; for (long i = 0; i < n; i++) sum += srt[i];

        printf("dispatches %ld in %.2fs  ->  ACHIEVED %.1f/s (target %.1f)\n",
               n, elapsed, achieved, rate);
        printf("  lat p50 %.1f us   mean %.1f us   p95 %.1f us\n",
               srt[n/2], sum/n, srt[(long)(n*0.95)]);

        if (report) {
            FILE *f = fopen(report, "w");
            if (f) {
                fprintf(f, "{\n \"model\": \"%s\",\n \"target_rate\": %.1f,\n \"achieved_rate\": %.2f,\n"
                           " \"dispatches\": %ld,\n \"seconds\": %.3f,\n \"lat_p50_us\": %.2f,\n"
                           " \"lat_mean_us\": %.2f,\n \"lat_p95_us\": %.2f\n}\n",
                        argv[1], rate, achieved, n, elapsed, srt[n/2], sum/n, srt[(long)(n*0.95)]);
                fclose(f);
            }
        }

        ((BOOL (*)(id, SEL, id, id, NSInteger, id*))objc_msgSend)(
            client, NSSelectorFromString(@"doUnloadModel:options:qos:error:"), model, @{}, 0, &err);
        CFRelease(inSurf); CFRelease(outSurf);
        free(lats); free(srt);
        return 0;
    }
}
