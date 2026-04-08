// adapter_weight_test.m — Test Apple's native adapter weight loading on ANE
//
// Tests the _ANEWeight + loadModelNewInstance path for zero-cost weight hot-swap.
//
// Build:
//   xcrun clang -O2 -framework Foundation -framework IOSurface -fobjc-arc \
//     -o adapter_weight_test adapter_weight_test.m
//
// Sign with adapter entitlement:
//   codesign -s - --entitlements adapter_weight_test.entitlements -f adapter_weight_test
//
// Copyright 2026 Nick Lo. MIT License.

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

        // Load ANE framework
        dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);

        Class _Cl = NSClassFromString(@"_ANEClient");
        Class _Mo = NSClassFromString(@"_ANEModel");
        Class _Wt = NSClassFromString(@"_ANEWeight");

        if (!_Cl) { fprintf(stderr, "ERROR: _ANEClient not found\n"); return 1; }
        if (!_Mo) { fprintf(stderr, "ERROR: _ANEModel not found\n"); return 1; }

        fprintf(stderr, "_ANEClient: %s\n", class_getName(_Cl));
        fprintf(stderr, "_ANEModel: %s\n", class_getName(_Mo));
        fprintf(stderr, "_ANEWeight: %s\n", _Wt ? class_getName(_Wt) : "NOT FOUND");

        // Enumerate _ANEWeight methods
        if (_Wt) {
            fprintf(stderr, "\n=== _ANEWeight class methods ===\n");
            unsigned int count = 0;
            Method *methods = class_copyMethodList(object_getClass(_Wt), &count);
            for (unsigned int i = 0; i < count; i++) {
                fprintf(stderr, "  + %s\n", sel_getName(method_getName(methods[i])));
            }
            free(methods);

            fprintf(stderr, "\n=== _ANEWeight instance methods ===\n");
            methods = class_copyMethodList(_Wt, &count);
            for (unsigned int i = 0; i < count; i++) {
                fprintf(stderr, "  - %s\n", sel_getName(method_getName(methods[i])));
            }
            free(methods);
        }

        // Enumerate _ANEClient methods related to adapter/weight/instance
        fprintf(stderr, "\n=== _ANEClient adapter-related methods ===\n");
        unsigned int count = 0;
        Method *methods = class_copyMethodList(_Cl, &count);
        for (unsigned int i = 0; i < count; i++) {
            const char *name = sel_getName(method_getName(methods[i]));
            if (strstr(name, "eight") || strstr(name, "dapter") ||
                strstr(name, "Instance") || strstr(name, "instance") ||
                strstr(name, "NewInstance") || strstr(name, "swap") ||
                strstr(name, "Swap")) {
                fprintf(stderr, "  - %s\n", name);
            }
        }
        free(methods);

        // Get shared connection
        id client = ((id (*)(id, SEL))objc_msgSend)((id)_Cl,
            NSSelectorFromString(@"sharedConnection"));
        fprintf(stderr, "\nClient: %s\n", [[client description] UTF8String]);

        // Check if we have the adapter entitlement
        // Try to see what happens when we call adapter-related methods

        if (argc < 2) {
            fprintf(stderr, "\nUsage: %s <model.mlmodelc> [weights_dir]\n", argv[0]);
            fprintf(stderr, "  Step 1: Run with just model path to verify compilation\n");
            fprintf(stderr, "  Step 2: Run with weights_dir to test adapter loading\n");
            return 0;
        }

        NSString *modelPath = [NSString stringWithUTF8String:argv[1]];
        NSURL *modelURL = [NSURL fileURLWithPath:modelPath];

        // Load model normally first
        fprintf(stderr, "\n=== Loading base model ===\n");
        NSString *key = @"adapter_test_base";
        id model = ((id (*)(id, SEL, id, id))objc_msgSend)(
            (id)_Mo, NSSelectorFromString(@"modelAtURL:key:"), modelURL, key);
        fprintf(stderr, "Model: %s\n", [[model description] UTF8String]);

        // Compile
        NSError *err = nil;
        uint64_t t0 = mach_absolute_time();
        BOOL ok = ((BOOL (*)(id, SEL, id, id, NSInteger, id*))objc_msgSend)(
            client, NSSelectorFromString(@"compileModel:options:qos:error:"),
            model, @{}, 0, &err);
        double compile_ms = ns_to_ms(mach_absolute_time() - t0);
        fprintf(stderr, "Compile: %s (%.1f ms)\n", ok ? "OK" : "FAILED", compile_ms);
        if (err) fprintf(stderr, "  Error: %s\n", [[err description] UTF8String]);

        // Load
        t0 = mach_absolute_time();
        ok = ((BOOL (*)(id, SEL, id, id, NSInteger, id*))objc_msgSend)(
            client, NSSelectorFromString(@"loadModel:options:qos:error:"),
            model, @{}, 0, &err);
        double load_ms = ns_to_ms(mach_absolute_time() - t0);
        fprintf(stderr, "Load: %s (%.1f ms)\n", ok ? "OK" : "FAILED", load_ms);
        if (err) fprintf(stderr, "  Error: %s\n", [[err description] UTF8String]);

        // Now try loadModelNewInstance
        fprintf(stderr, "\n=== Testing loadModelNewInstance ===\n");

        // Check if the method exists
        SEL newInstanceSel = NSSelectorFromString(
            @"doLoadModelNewInstance:options:modelInstParams:qos:error:");
        if ([client respondsToSelector:newInstanceSel]) {
            fprintf(stderr, "doLoadModelNewInstance: EXISTS\n");

            // Try calling with empty params to see what happens
            t0 = mach_absolute_time();
            err = nil;
            ok = ((BOOL (*)(id, SEL, id, id, id, NSInteger, id*))objc_msgSend)(
                client, newInstanceSel,
                model, @{}, @{}, 0, &err);
            double newInst_ms = ns_to_ms(mach_absolute_time() - t0);
            fprintf(stderr, "doLoadModelNewInstance: %s (%.1f ms)\n",
                    ok ? "OK" : "FAILED", newInst_ms);
            if (err) fprintf(stderr, "  Error: %s\n", [[err description] UTF8String]);

            // KEY MEASUREMENT: if newInst_ms < 10, it's hot-swap
            // if newInst_ms > 100, it's recompiling
            if (ok) {
                fprintf(stderr, "\n  *** TIMING: %.1f ms ***\n", newInst_ms);
                if (newInst_ms < 10.0) {
                    fprintf(stderr, "  VERDICT: HOT-SWAP (no recompilation)\n");
                } else if (newInst_ms < 100.0) {
                    fprintf(stderr, "  VERDICT: WARM-SWAP (partial recompilation?)\n");
                } else {
                    fprintf(stderr, "  VERDICT: COLD-SWAP (full recompilation)\n");
                }
            }
        } else {
            fprintf(stderr, "doLoadModelNewInstance: NOT FOUND\n");

            // Try alternative selectors
            NSArray *alts = @[
                @"loadModelNewInstance:options:modelInstParams:qos:error:",
                @"loadModel:options:modelInstanceParameters:qos:error:",
                @"doLoadModel:options:qos:modelInstanceParameters:error:",
            ];
            for (NSString *sel in alts) {
                if ([client respondsToSelector:NSSelectorFromString(sel)]) {
                    fprintf(stderr, "  FOUND alternative: %s\n", [sel UTF8String]);
                }
            }
        }

        // Try creating an _ANEWeight object if the class exists
        if (_Wt && argc >= 3) {
            fprintf(stderr, "\n=== Testing _ANEWeight ===\n");
            NSString *weightsPath = [NSString stringWithUTF8String:argv[2]];
            NSURL *weightsURL = [NSURL fileURLWithPath:weightsPath];

            // Try creating a weight object
            id weight = nil;
            @try {
                weight = ((id (*)(id, SEL, id, id))objc_msgSend)(
                    (id)_Wt,
                    NSSelectorFromString(@"weightWithSymbolAndURL:weightURL:"),
                    @"lm_head_weight", weightsURL);
                fprintf(stderr, "Weight object: %s\n",
                        weight ? [[weight description] UTF8String] : "nil");
            } @catch (NSException *e) {
                fprintf(stderr, "Weight creation failed: %s\n",
                        [[e description] UTF8String]);
            }
        }

        // Unload
        ((BOOL (*)(id, SEL, id, id, NSInteger, id*))objc_msgSend)(
            client, NSSelectorFromString(@"doUnloadModel:options:qos:error:"),
            model, @{}, 0, &err);

        fprintf(stderr, "\nDone.\n");
        return 0;
    }
}
