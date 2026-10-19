#import <Foundation/Foundation.h>
#include "CosyVoiceFailure.hpp"
#include <cassert>

static bool reject(CosyVoiceFailureOutput error, int cap) {
    @autoreleasepool {
        *error = CosyVoiceTokenLimitMessage(cap);
        return false;
    }
}
int main() {
    @autoreleasepool {
        // Same production out-parameter type and dynamic message factory,
        // read only AFTER the callee's inner autorelease pool has drained.
        for (int i=0;i<1000;++i) {
            NSString *error = nil;
            assert(!reject(&error,i%2 ? 160 : 105));
            assert([error containsString:i%2 ? @"160" : @"105"]);
            assert([error containsString:@"未截断"]);
            NSData *encoded = [error dataUsingEncoding:NSUTF8StringEncoding];
            assert(encoded.length>40);
        }
        puts("PASS: dynamic token-limit errors survive inner pool drain in production strong storage");
    }
}
