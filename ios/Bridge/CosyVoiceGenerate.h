#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

// One batch owns exactly one 384-frame CPU High Flow. Calls are serialized;
// close after success, cancellation or error. LLM/HiFT are NOT retained here.
@interface CosyVoiceResidentFlow : NSObject
+ (NSDictionary<NSString *, id> *)beginAtDirectory:(NSString *)directory NS_SWIFT_NAME(begin(atDirectory:));
- (NSDictionary<NSString *, id> *)generateText:(NSString *)text;
- (NSDictionary<NSString *, id> *)close;
@end

@interface CosyVoiceGenerate : NSObject
+ (NSDictionary<NSString *, id> *)runFixedTokenSampleAtDirectory:(NSString *)directory;
+ (NSDictionary<NSString *, id> *)runFixedTokenSampleAtDirectory:(NSString *)directory lowPrecisionFlow:(BOOL)lowPrecision;
+ (NSDictionary<NSString *, id> *)generateText:(NSString *)text atDirectory:(NSString *)directory;
+ (NSDictionary<NSString *, id> *)generateText:(NSString *)text atDirectory:(NSString *)directory optimized:(BOOL)optimized cpuThreads:(NSInteger)threads;
+ (NSDictionary<NSString *, NSNumber *> *)memorySnapshot;
+ (NSDictionary<NSString *, id> *)probeHiFTPrecisionAtDirectory:(NSString *)directory;
+ (NSDictionary<NSString *, id> *)probeResidentFlowAtDirectory:(NSString *)directory;
+ (NSString *)hiftProbeLogPath;
+ (NSString *)residentProbeLogPath;
+ (NSString *)hiftReferenceWavPath;
+ (NSString *)hiftCandidateWavPath;
+ (NSString *)generationLogPath;
+ (NSString *)llmRuntimeLogPath;
+ (NSString *)generatedWavPath;
@end

NS_ASSUME_NONNULL_END
