#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

@interface MNNProbe : NSObject
+ (NSDictionary<NSString *, id> *)probeConditionerAtDirectory:(NSString *)directory;
+ (NSDictionary<NSString *, id> *)probeFlowMetalMemoryAtDirectory:(NSString *)directory;
+ (NSDictionary<NSString *, id> *)probeHiFTCPUMemoryAtDirectory:(NSString *)directory;
+ (NSString *)memoryProbeLogPath;
@end

NS_ASSUME_NONNULL_END
