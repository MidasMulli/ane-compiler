// capture_and_eval.m — Load .mlmodelc via ane-dispatch, capture .hwx from ANE cache
//
// Build: clang -framework Foundation -framework IOSurface -framework Metal \
//        -I../../ane-dispatch/include -L../../ane-dispatch/src -lANEDispatch \
//        -o capture_and_eval capture_and_eval.m
//
// Run: ./capture_and_eval /path/to/model.mlmodelc [output.hwx]

#import <Foundation/Foundation.h>
#import <dlfcn.h>
#import <signal.h>
#import <sys/stat.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <IOSurface/IOSurface.h>

// Load ANE framework classes directly (no dependency on ANEDispatch.h wrapper)

static Class _ANEClientClass;
static Class _ANEModelClass;
static Class _ANERequestClass;
static Class _ANEIOSurfaceObjectClass;

static void loadFrameworks(void) {
    dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);
    _ANEClientClass = NSClassFromString(@"_ANEClient");
    _ANEModelClass = NSClassFromString(@"_ANEModel");
    _ANERequestClass = NSClassFromString(@"_ANERequest");
    _ANEIOSurfaceObjectClass = NSClassFromString(@"_ANEIOSurfaceObject");
}

// Find most recent .hwx in ANE cache
static NSString *findRecentHWX(NSTimeInterval maxAge) {
    NSString *cacheBase = @"/Library/Caches/com.apple.aned";
    NSFileManager *fm = [NSFileManager defaultManager];
    NSDate *cutoff = [NSDate dateWithTimeIntervalSinceNow:-maxAge];

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

    // Sort by date, newest first
    [candidates sortUsingComparator:^(NSDictionary *a, NSDictionary *b) {
        return [b[@"date"] compare:a[@"date"]];
    }];

    return candidates.count > 0 ? candidates[0][@"path"] : nil;
}

int main(int argc, char *argv[]) {
    @autoreleasepool {
        signal(SIGSEGV, SIG_IGN);  // XPC crash workaround

        if (argc < 2) {
            fprintf(stderr, "Usage: %s <model.mlmodelc> [output.hwx]\n", argv[0]);
            return 1;
        }

        loadFrameworks();

        NSString *modelPath = [NSString stringWithUTF8String:argv[1]];
        NSString *outputHWX = argc > 2 ? [NSString stringWithUTF8String:argv[2]] : nil;

        // Get shared client
        id client = ((id (*)(id, SEL))objc_msgSend)(
            (id)_ANEClientClass, NSSelectorFromString(@"sharedConnection"));
        if (!client) {
            fprintf(stderr, "Failed to get _ANEClient\n");
            return 1;
        }

        // Create model
        NSURL *url = [NSURL fileURLWithPath:modelPath];
        id model = ((id (*)(id, SEL, id, id))objc_msgSend)(
            (id)_ANEModelClass, NSSelectorFromString(@"modelAtURL:key:"), url, @"default");
        if (!model) {
            fprintf(stderr, "Failed to create _ANEModel\n");
            return 1;
        }
        fprintf(stderr, "Model created.\n");

        // Purge any cached version
        ((void (*)(id, SEL, id))objc_msgSend)(
            client, NSSelectorFromString(@"purgeCompiledModel:"), model);

        // Compile
        NSError *error = nil;
        BOOL compileOK = ((BOOL (*)(id, SEL, id, id, NSInteger, id*))objc_msgSend)(
            client, NSSelectorFromString(@"compileModel:options:qos:error:"),
            model, @{}, 0, &error);
        fprintf(stderr, "Compile: %s\n", compileOK ? "OK" : "FAILED");
        if (error) {
            fprintf(stderr, "Error: %s\n", [[error description] UTF8String]);
        }

        if (!compileOK) {
            // Try load (implicit compile)
            error = nil;
            BOOL loadOK = ((BOOL (*)(id, SEL, id, id, NSInteger, id*))objc_msgSend)(
                client, NSSelectorFromString(@"loadModel:options:qos:error:"),
                model, @{}, 0, &error);
            fprintf(stderr, "Load (implicit compile): %s\n", loadOK ? "OK" : "FAILED");
            if (error) fprintf(stderr, "Error: %s\n", [[error description] UTF8String]);
            if (!loadOK) return 1;

            // Unload
            ((BOOL (*)(id, SEL, id, id, NSInteger, id*))objc_msgSend)(
                client, NSSelectorFromString(@"doUnloadModel:options:qos:error:"),
                model, @{}, 0, &error);
        } else {
            // Also load to verify
            error = nil;
            BOOL loadOK = ((BOOL (*)(id, SEL, id, id, NSInteger, id*))objc_msgSend)(
                client, NSSelectorFromString(@"loadModel:options:qos:error:"),
                model, @{}, 0, &error);
            fprintf(stderr, "Load: %s\n", loadOK ? "OK" : "FAILED");

            if (loadOK) {
                // Get model attributes
                id attrs = ((id (*)(id, SEL))objc_msgSend)(
                    model, NSSelectorFromString(@"modelAttributes"));
                if (attrs) {
                    fprintf(stderr, "Model attributes: %s\n",
                            [[attrs description] UTF8String]);
                }

                // Unload
                ((BOOL (*)(id, SEL, id, id, NSInteger, id*))objc_msgSend)(
                    client, NSSelectorFromString(@"doUnloadModel:options:qos:error:"),
                    model, @{}, 0, &error);
            }
        }

        // Search for .hwx in ANE cache
        usleep(500000); // 500ms for cache write
        NSString *hwxPath = findRecentHWX(10.0);
        if (hwxPath) {
            NSData *data = [NSData dataWithContentsOfFile:hwxPath];
            uint32_t magic = 0;
            [data getBytes:&magic length:4];
            uint32_t ncmds = 0;
            [data getBytes:&ncmds range:NSMakeRange(0x10, 4)];

            fprintf(stderr, "CAPTURED: %s (%lu bytes, ncmds=%u)\n",
                    [hwxPath UTF8String], (unsigned long)data.length, ncmds);

            // Print key .hwx sections
            // Parse __KERN_0 size
            uint32_t offset = 32;
            for (uint32_t i = 0; i < ncmds && offset + 8 < data.length; i++) {
                uint32_t cmd, cmdsize;
                [data getBytes:&cmd range:NSMakeRange(offset, 4)];
                [data getBytes:&cmdsize range:NSMakeRange(offset+4, 4)];
                if (cmd == 0x19) { // LC_SEGMENT_64
                    char segname[17] = {0};
                    [data getBytes:segname range:NSMakeRange(offset+8, 16)];
                    uint32_t nsects = 0;
                    [data getBytes:&nsects range:NSMakeRange(offset+64, 4)];
                    for (uint32_t s = 0; s < nsects; s++) {
                        uint32_t s_off = offset + 72 + s * 80;
                        char sectname[17] = {0};
                        [data getBytes:sectname range:NSMakeRange(s_off, 16)];
                        uint64_t size = 0;
                        uint32_t foff = 0;
                        [data getBytes:&size range:NSMakeRange(s_off+40, 8)];
                        [data getBytes:&foff range:NSMakeRange(s_off+48, 4)];
                        fprintf(stderr, "  %s.%s: off=0x%X size=%llu\n",
                                segname, sectname, foff, size);
                    }
                }
                offset += cmdsize;
            }

            if (outputHWX) {
                [data writeToFile:outputHWX atomically:YES];
                fprintf(stderr, "Saved to: %s\n", [outputHWX UTF8String]);
            }

            // Print hex dump of first bytes to stdout for piping
            printf("size=%lu ncmds=%u\n", (unsigned long)data.length, ncmds);
        } else {
            fprintf(stderr, "No .hwx found in ANE cache\n");

            // Check if compiled model exists
            BOOL exists = ((BOOL (*)(id, SEL, id))objc_msgSend)(
                client, NSSelectorFromString(@"compiledModelExistsFor:"), model);
            fprintf(stderr, "compiledModelExists: %s\n", exists ? "YES" : "NO");
        }

        return 0;
    }
}
