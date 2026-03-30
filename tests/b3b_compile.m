// b3b_compile.m — Compile .mlmodelc and extract W[51] from cached .hwx
//
// Build: xcrun clang -framework Foundation -framework IOSurface -fobjc-arc \
//        -o b3b_compile b3b_compile.m
// Run: ./b3b_compile <model.mlmodelc>
// Output: JSON with compile status, hwx path, W[51] value, and __text dump

#import <Foundation/Foundation.h>
#import <dlfcn.h>
#import <signal.h>
#import <IOSurface/IOSurface.h>
#import <objc/runtime.h>
#import <objc/message.h>

static Class _ANEClientCls;
static Class _ANEModelCls;

static void loadFW(void) {
    dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);
    _ANEClientCls = NSClassFromString(@"_ANEClient");
    _ANEModelCls = NSClassFromString(@"_ANEModel");
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

int main(int argc, char *argv[]) {
    @autoreleasepool {
        signal(SIGSEGV, SIG_IGN);

        if (argc < 2) {
            fprintf(stderr, "Usage: %s <model.mlmodelc>\n", argv[0]);
            return 1;
        }

        loadFW();

        NSString *modelPath = [NSString stringWithUTF8String:argv[1]];

        id client = ((id (*)(id, SEL))objc_msgSend)(
            (id)_ANEClientCls, NSSelectorFromString(@"sharedConnection"));
        NSURL *url = [NSURL fileURLWithPath:modelPath];
        id model = ((id (*)(id, SEL, id, id))objc_msgSend)(
            (id)_ANEModelCls, NSSelectorFromString(@"modelAtURL:key:"), url, @"b3b");

        if (!model) {
            printf("{\"compile_ok\": false, \"error\": \"model_create_failed\"}\n");
            return 0;
        }

        // Purge old cache
        ((void (*)(id, SEL, id))objc_msgSend)(client, NSSelectorFromString(@"purgeCompiledModel:"), model);
        usleep(200000);

        NSError *err = nil;
        BOOL ok = ((BOOL (*)(id, SEL, id, id, NSInteger, id*))objc_msgSend)(
            client, NSSelectorFromString(@"compileModel:options:qos:error:"), model, @{}, 0, &err);

        if (!ok) {
            NSString *errStr = err ? [err localizedDescription] : @"unknown";
            printf("{\"compile_ok\": false, \"error\": \"%s\"}\n", [errStr UTF8String]);
            return 0;
        }

        // Find cached .hwx
        NSString *hwxPath = findRecentHWX();
        if (!hwxPath) {
            printf("{\"compile_ok\": true, \"error\": \"hwx_not_found\"}\n");
            return 0;
        }

        // Read .hwx and extract __text
        NSData *hwxData = [NSData dataWithContentsOfFile:hwxPath];
        const uint8_t *bytes = (const uint8_t *)[hwxData bytes];
        uint32_t ncmds = *(uint32_t *)(bytes + 0x10);
        uint32_t fileSize = (uint32_t)[hwxData length];

        uint32_t textOff = 0, textSize = 0;
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
                        textSize = *(uint32_t *)(bytes + soff + 40);
                        textOff = *(uint32_t *)(bytes + soff + 48);
                    }
                }
            }
            off += cmdsize;
        }

        // Extract W[51] and opcode W[19]
        uint32_t w51 = 0, w19 = 0;
        if (textOff > 0 && textSize >= 52 * 4) {
            w51 = *(uint32_t *)(bytes + textOff + 51 * 4);
        }
        if (textOff > 0 && textSize >= 20 * 4) {
            w19 = *(uint32_t *)(bytes + textOff + 19 * 4);
        }

        // Print JSON
        printf("{\"compile_ok\": true, \"ncmds\": %u, \"file_size\": %u, "
               "\"text_off\": %u, \"text_size\": %u, "
               "\"w19_opcode\": \"0x%08X\", \"w51_selector\": \"0x%08X\", "
               "\"hwx_path\": \"%s\"",
               ncmds, fileSize, textOff, textSize, w19, w51,
               [hwxPath UTF8String]);

        // Dump all __text words
        if (textOff > 0 && textSize > 0) {
            uint32_t nwords = textSize / 4;
            printf(", \"text_words\": [");
            for (uint32_t i = 0; i < nwords; i++) {
                uint32_t w = *(uint32_t *)(bytes + textOff + i * 4);
                printf("%s\"0x%08X\"", i ? "," : "", w);
            }
            printf("]");
        }

        printf("}\n");
        return 0;
    }
}
