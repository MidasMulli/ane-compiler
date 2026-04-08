// adapter_weight_test2.m — Full adapter weight hot-swap test
//
// Tests: compile base → load → create _ANEWeight → _ANEProcedureData →
//        _ANEModelInstanceParameters → doLoadModelNewInstance → time it
//
// Build + sign:
//   xcrun clang -O2 -framework Foundation -framework IOSurface -fobjc-arc \
//     -o adapter_weight_test2 adapter_weight_test2.m
//   codesign -s - --entitlements adapter_weight_test.entitlements -f adapter_weight_test2

#import <Foundation/Foundation.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>
#import <mach/mach_time.h>

static mach_timebase_info_data_t tb;
static double ns_to_ms(uint64_t ns) {
    return (double)(ns * tb.numer / tb.denom) / 1e6;
}

int main(int argc, char *argv[]) {
    @autoreleasepool {
        mach_timebase_info(&tb);

        dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);

        Class _Cl = NSClassFromString(@"_ANEClient");
        Class _Mo = NSClassFromString(@"_ANEModel");
        Class _Wt = NSClassFromString(@"_ANEWeight");
        Class _PD = NSClassFromString(@"_ANEProcedureData");
        Class _MIP = NSClassFromString(@"_ANEModelInstanceParameters");

        fprintf(stderr, "Classes: Client=%p Model=%p Weight=%p ProcData=%p MIP=%p\n",
                _Cl, _Mo, _Wt, _PD, _MIP);

        if (!_Cl || !_Mo || !_Wt || !_PD || !_MIP) {
            fprintf(stderr, "ERROR: Missing classes\n");
            return 1;
        }

        if (argc < 2) {
            fprintf(stderr, "Usage: %s <model.mlmodelc> [adapter_weights_file]\n", argv[0]);
            return 1;
        }

        id client = ((id (*)(id, SEL))objc_msgSend)((id)_Cl,
            NSSelectorFromString(@"sharedConnection"));

        NSString *modelPath = [NSString stringWithUTF8String:argv[1]];
        NSURL *modelURL = [NSURL fileURLWithPath:modelPath];

        // Step 1: Load base model
        fprintf(stderr, "\n=== Step 1: Compile + Load base model ===\n");
        id model = ((id (*)(id, SEL, id, id))objc_msgSend)(
            (id)_Mo, NSSelectorFromString(@"modelAtURL:key:"),
            modelURL, @"adapter_test_v2");

        NSError *err = nil;
        uint64_t t0 = mach_absolute_time();
        BOOL ok = ((BOOL (*)(id, SEL, id, id, NSInteger, id*))objc_msgSend)(
            client, NSSelectorFromString(@"compileModel:options:qos:error:"),
            model, @{}, 0, &err);
        fprintf(stderr, "Compile: %s (%.1f ms)\n", ok ? "OK" : "FAILED",
                ns_to_ms(mach_absolute_time() - t0));
        if (!ok) { fprintf(stderr, "Error: %s\n", [[err description] UTF8String]); return 1; }

        t0 = mach_absolute_time();
        ok = ((BOOL (*)(id, SEL, id, id, NSInteger, id*))objc_msgSend)(
            client, NSSelectorFromString(@"loadModel:options:qos:error:"),
            model, @{}, 0, &err);
        fprintf(stderr, "Load: %s (%.1f ms)\n", ok ? "OK" : "FAILED",
                ns_to_ms(mach_absolute_time() - t0));
        if (!ok) { fprintf(stderr, "Error: %s\n", [[err description] UTF8String]); return 1; }

        // Step 2: Create adapter weight objects
        fprintf(stderr, "\n=== Step 2: Create _ANEWeight + _ANEProcedureData + _ANEModelInstanceParameters ===\n");

        // Create weight pointing to model's own weights (same file, testing the path)
        NSString *weightsPath = [modelPath stringByAppendingPathComponent:@"model.espresso.weights"];
        if (argc >= 3) {
            weightsPath = [NSString stringWithUTF8String:argv[2]];
        }
        NSURL *weightsURL = [NSURL fileURLWithPath:weightsPath];
        fprintf(stderr, "Weight URL: %s\n", [weightsPath UTF8String]);

        id weight = ((id (*)(id, SEL, id, id))objc_msgSend)(
            (id)_Wt, NSSelectorFromString(@"weightWithSymbolAndURL:weightURL:"),
            @"fc1_weight", weightsURL);
        fprintf(stderr, "Weight: %s\n", [[weight description] UTF8String]);

        // Create ProcedureData with the weight
        id procData = ((id (*)(id, SEL, id, id))objc_msgSend)(
            (id)_PD, NSSelectorFromString(@"procedureDataWithSymbol:weightArray:"),
            @"main", @[weight]);
        fprintf(stderr, "ProcData: %s\n", [[procData description] UTF8String]);

        // Create ModelInstanceParameters
        id mip = ((id (*)(id, SEL, id, id))objc_msgSend)(
            (id)_MIP, NSSelectorFromString(@"withProcedureData:procedureArray:"),
            procData, @[procData]);
        fprintf(stderr, "MIP: %s\n", [[mip description] UTF8String]);

        // Step 3: Call doLoadModelNewInstance — THE CRITICAL TEST
        fprintf(stderr, "\n=== Step 3: doLoadModelNewInstance ===\n");

        err = nil;
        t0 = mach_absolute_time();
        @try {
            ok = ((BOOL (*)(id, SEL, id, id, id, NSInteger, id*))objc_msgSend)(
                client,
                NSSelectorFromString(@"doLoadModelNewInstance:options:modelInstParams:qos:error:"),
                model, @{}, mip, 0, &err);
            double ms = ns_to_ms(mach_absolute_time() - t0);

            fprintf(stderr, "doLoadModelNewInstance: %s (%.1f ms)\n",
                    ok ? "OK" : "FAILED", ms);
            if (err) fprintf(stderr, "  Error: %s\n", [[err description] UTF8String]);

            if (ok) {
                fprintf(stderr, "\n  *** TIMING: %.1f ms ***\n", ms);
                if (ms < 10.0) {
                    fprintf(stderr, "  >>> HOT-SWAP CONFIRMED (no recompilation) <<<\n");
                } else if (ms < 100.0) {
                    fprintf(stderr, "  >>> WARM-SWAP (partial reprocessing) <<<\n");
                } else {
                    fprintf(stderr, "  >>> COLD-SWAP (full recompilation) <<<\n");
                }
            }
        } @catch (NSException *e) {
            double ms = ns_to_ms(mach_absolute_time() - t0);
            fprintf(stderr, "EXCEPTION (%.1f ms): %s\n", ms, [[e description] UTF8String]);
            fprintf(stderr, "  Reason: %s\n", [[e reason] UTF8String]);
        }

        // Cleanup
        ((BOOL (*)(id, SEL, id, id, NSInteger, id*))objc_msgSend)(
            client, NSSelectorFromString(@"doUnloadModel:options:qos:error:"),
            model, @{}, 0, &err);

        fprintf(stderr, "\nDone.\n");
        return 0;
    }
}
