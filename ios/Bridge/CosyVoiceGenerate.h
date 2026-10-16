#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

@interface CosyVoiceGenerate : NSObject
+ (NSDictionary<NSString *, id> *)runFixedTokenSampleAtDirectory:(NSString *)directory;
+ (NSDictionary<NSString *, id> *)runFixedTokenSampleAtDirectory:(NSString *)directory lowPrecisionFlow:(BOOL)lowPrecision;
+ (NSString *)generationLogPath;
+ (NSString *)generatedWavPath;
@end

NS_ASSUME_NONNULL_END
