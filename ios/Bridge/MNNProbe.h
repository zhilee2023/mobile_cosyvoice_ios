#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

@interface MNNProbe : NSObject
+ (NSDictionary<NSString *, id> *)probeConditionerAtDirectory:(NSString *)directory;
@end

NS_ASSUME_NONNULL_END
