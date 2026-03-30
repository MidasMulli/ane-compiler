// coreml_eval.m — Evaluate model via CoreML (uses ANE cache .hwx)
//
// Build: xcrun clang -framework Foundation -framework CoreML -fobjc-arc -o coreml_eval coreml_eval.m
// Run: ./coreml_eval <model.mlmodelc> [channels=64]
// Output: FP16 hex values to stdout

#import <Foundation/Foundation.h>
#import <CoreML/CoreML.h>

int main(int argc, char *argv[]) {
    @autoreleasepool {
        if (argc < 2) { fprintf(stderr, "Usage: %s <model.mlmodelc> [channels]\n", argv[0]); return 1; }

        int channels = argc > 2 ? atoi(argv[2]) : 64;
        NSString *path = [NSString stringWithUTF8String:argv[1]];
        NSURL *url = [NSURL fileURLWithPath:path];
        NSError *error = nil;

        // Configure for ANE
        MLModelConfiguration *config = [[MLModelConfiguration alloc] init];
        config.computeUnits = MLComputeUnitsAll;

        // Load compiled model
        MLModel *model = [MLModel modelWithContentsOfURL:url configuration:config error:&error];
        if (!model) { fprintf(stderr, "Load failed: %s\n", [[error description] UTF8String]); return 1; }

        // Create input: shape [C, 1, 1] (3D for NeuralNetwork v3)
        NSArray *shape = @[@(channels), @1, @1];
        MLMultiArray *inputArray = [[MLMultiArray alloc] initWithShape:shape
                                                             dataType:MLMultiArrayDataTypeFloat32
                                                                error:&error];
        if (!inputArray) { fprintf(stderr, "Input alloc failed\n"); return 1; }

        // Fill with 1.0
        for (int i = 0; i < channels; i++) {
            inputArray[i] = @1.0f;
        }

        // Build feature provider
        MLDictionaryFeatureProvider *provider = [[MLDictionaryFeatureProvider alloc]
            initWithDictionary:@{@"input": inputArray} error:&error];
        if (!provider) { fprintf(stderr, "Provider failed\n"); return 1; }

        // Predict
        id<MLFeatureProvider> result = [model predictionFromFeatures:provider error:&error];
        if (!result) { fprintf(stderr, "Predict failed: %s\n", [[error description] UTF8String]); return 1; }

        // Read output
        MLMultiArray *output = (MLMultiArray *)[[result featureValueForName:@"output"] multiArrayValue];
        if (!output) {
            // Try other output names
            for (NSString *name in @[@"output", @"output@output", @"var_6"]) {
                MLFeatureValue *fv = [result featureValueForName:name];
                if (fv) { output = [fv multiArrayValue]; break; }
            }
        }
        if (!output) { fprintf(stderr, "No output array found\n"); return 1; }

        // Print as FP16 hex (read as float, convert)
        for (int i = 0; i < channels && i < output.count; i++) {
            float val = output[i].floatValue;
            // float to fp16
            uint32_t bits;
            memcpy(&bits, &val, 4);
            uint32_t sign = (bits >> 31) & 1;
            int32_t exp = ((bits >> 23) & 0xFF) - 127;
            uint32_t mant = bits & 0x7FFFFF;
            uint16_t fp16;
            if (exp > 15) fp16 = (sign << 15) | 0x7C00;
            else if (exp < -24) fp16 = (sign << 15);
            else if (exp < -14) {
                mant |= 0x800000;
                int shift = -exp - 14 + 13;
                fp16 = (sign << 15) | (mant >> shift);
            } else {
                fp16 = (sign << 15) | ((exp + 15) << 10) | (mant >> 13);
            }
            printf("%04x ", fp16);
        }
        printf("\n");
        fprintf(stderr, "OK\n");
        return 0;
    }
}
