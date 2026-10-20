#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

// One batch owns exactly one 384-frame CPU High Flow. Calls are serialized;
// close after success, cancellation or error. LLM/HiFT are NOT retained here.
@interface CosyVoiceResidentFlow : NSObject
+ (NSDictionary<NSString *, id> *)beginAtDirectory:(NSString *)directory NS_SWIFT_NAME(begin(atDirectory:));
- (NSDictionary<NSString *, id> *)generateText:(NSString *)text;
- (NSDictionary<NSString *, id> *)generateText:(NSString *)text lowPrecisionHiFT:(BOOL)low;
- (NSDictionary<NSString *, id> *)close;
@end

// One owner per batch. No static model pools; methods are serialized.
@interface CosyVoicePipeline : NSObject
+ (NSDictionary<NSString *, id> *)beginAtDirectory:(NSString *)directory mode:(NSString *)mode NS_SWIFT_NAME(begin(atDirectory:mode:));
- (NSDictionary<NSString *, id> *)generateText:(NSString *)text lowPrecisionHiFT:(BOOL)low;
- (NSDictionary<NSString *, id> *)prepareTexts:(NSArray<NSString *> *)texts;
- (NSDictionary<NSString *, id> *)generatePreparedWithLowPrecisionHiFT:(BOOL)low NS_SWIFT_NAME(generatePrepared(lowPrecisionHiFT:));
- (NSDictionary<NSString *, id> *)close;
@end

@interface CosyVoiceGenerate : NSObject
+ (NSDictionary<NSString *, id> *)runFixedTokenSampleAtDirectory:(NSString *)directory;
+ (NSDictionary<NSString *, id> *)runFixedTokenSampleAtDirectory:(NSString *)directory lowPrecisionFlow:(BOOL)lowPrecision;
+ (NSDictionary<NSString *, id> *)generateText:(NSString *)text atDirectory:(NSString *)directory;
+ (NSDictionary<NSString *, id> *)generateText:(NSString *)text atDirectory:(NSString *)directory optimized:(BOOL)optimized cpuThreads:(NSInteger)threads;
+ (NSDictionary<NSString *, id> *)generateText:(NSString *)text atDirectory:(NSString *)directory optimized:(BOOL)optimized cpuThreads:(NSInteger)threads lowPrecisionHiFT:(BOOL)low;
+ (NSDictionary<NSString *, id> *)generateText:(NSString *)text atDirectory:(NSString *)directory optimized:(BOOL)optimized cpuThreads:(NSInteger)threads lowPrecisionHiFT:(BOOL)low flowW8:(BOOL)useW8;
+ (NSDictionary<NSString *, NSNumber *> *)memorySnapshot;
// Isolated controlled routes. No automatic promotion to production settings.
+ (NSDictionary<NSString *, id> *)runExperiment:(NSString *)route atDirectory:(NSString *)directory;
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
