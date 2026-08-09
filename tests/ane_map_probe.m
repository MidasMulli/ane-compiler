// CA `23-33-26` §C -- READ THE CONSUMER. What is LiveInputList actually enumerating?
// It reports 5 in / 4 out against the espresso net's declared 4 / 2, and wants
// 295,680 bytes against the four real inputs' 132,352 -- an unexplained 163,328.
// Dumps EVERY key of EVERY entry, plus the whole NetworkStatusList, rather than the
// four fields ane_eval_multi happens to read.
#import <Foundation/Foundation.h>
#import <objc/message.h>
#include <dlfcn.h>
static Class _ANEClientCls, _ANEModelCls;
static void loadFW(void) {
    dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);
    _ANEClientCls = NSClassFromString(@"_ANEClient");
    _ANEModelCls  = NSClassFromString(@"_ANEModel");
}
int main(int argc, char **argv) {
    @autoreleasepool {
        if (argc < 2) { fprintf(stderr, "usage: %s <model.mlmodelc>\n", argv[0]); return 1; }
        loadFW();
        id client = ((id (*)(id, SEL))objc_msgSend)((id)_ANEClientCls, NSSelectorFromString(@"sharedConnection"));
        NSURL *url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:argv[1]]];
        id model = ((id (*)(id, SEL, id, id))objc_msgSend)((id)_ANEModelCls,
                        NSSelectorFromString(@"modelAtURL:key:"), url, @"default");
        if (!model) { fprintf(stderr, "MODEL_FAILED\n"); return 2; }
        NSError *err = nil;
        ((BOOL (*)(id, SEL, id, id, NSInteger, id*))objc_msgSend)(
            client, NSSelectorFromString(@"compileModel:options:qos:error:"), model, @{}, 0, &err);
        BOOL ok = ((BOOL (*)(id, SEL, id, id, NSInteger, id*))objc_msgSend)(
            client, NSSelectorFromString(@"loadModel:options:qos:error:"), model, @{}, 0, &err);
        if (!ok) { fprintf(stderr, "LOAD_FAILED %s\n", err?[[err description] UTF8String]:"(nil)"); return 3; }
        id attrs = ((id (*)(id, SEL))objc_msgSend)(model, NSSelectorFromString(@"modelAttributes"));
        printf("=== modelAttributes TOP-LEVEL KEYS ===\n");
        for (id k in [attrs allKeys]) printf("  %s\n", [[k description] UTF8String]);
        printf("\n=== ANEFModelDescription ===\n");
        id afd = attrs[@"ANEFModelDescription"];
        if ([afd isKindOfClass:[NSDictionary class]]) {
            for (id k in [[afd allKeys] sortedArrayUsingSelector:@selector(compare:)]) {
                id v = afd[k];
                NSString *d = [v description];
                if ([d length] > 400) d = [[d substringToIndex:400] stringByAppendingString:@" ...(truncated)"];
                printf("  %-34s = %s\n", [[k description] UTF8String], [d UTF8String]);
            }
        } else printf("  (not a dictionary: %s)\n", [[afd description] UTF8String]);
        NSArray *nsl = attrs[@"NetworkStatusList"];
        printf("\n=== NetworkStatusList has %lu entry(ies) ===\n", (unsigned long)[nsl count]);
        NSDictionary *ns = [nsl firstObject];
        printf("=== NetworkStatusList[0] KEYS ===\n");
        for (id k in [ns allKeys]) printf("  %s\n", [[k description] UTF8String]);
        for (NSString *listName in @[@"LiveInputList", @"LiveOutputList"]) {
            NSArray *L = ns[listName];
            printf("\n=== %s : %lu entries ===\n", [listName UTF8String], (unsigned long)[L count]);
            for (NSUInteger i = 0; i < [L count]; i++) {
                printf("  [%lu]\n", (unsigned long)i);
                NSDictionary *e = L[i];
                for (id k in [[e allKeys] sortedArrayUsingSelector:@selector(compare:)])
                    printf("      %-24s = %s\n", [[k description] UTF8String],
                           [[e[k] description] UTF8String]);
            }
        }
        return 0;
    }
}
