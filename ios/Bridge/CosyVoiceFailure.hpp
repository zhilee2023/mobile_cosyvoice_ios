#pragma once
#import <Foundation/Foundation.h>

// NSString ** defaults to __autoreleasing under ARC. A callee's inner pool
// would drain a dynamic error BEFORE caller writeback. Store directly into
// caller-owned strong storage instead; all callers use strong local strings.
using CosyVoiceFailureOutput = NSString *__strong *;
inline NSString *CosyVoiceTokenLimitMessage(int limit) {
    return [NSString stringWithFormat:@"短句超过%d speech token上限（约%.1f秒），未截断；请使用按句生成或拆短", limit, limit/25.0];
}
