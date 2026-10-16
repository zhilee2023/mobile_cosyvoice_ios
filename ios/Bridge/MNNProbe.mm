#import "MNNProbe.h"
#import <MNN/Interpreter.hpp>
#import <mach/mach.h>
#import <os/proc.h>

#include <chrono>
#include <memory>

namespace {
double elapsedSeconds(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

uint64_t footprintBytes() {
    task_vm_info_data_t info = {};
    mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
    if (task_info(mach_task_self(), TASK_VM_INFO,
                  reinterpret_cast<task_info_t>(&info), &count) != KERN_SUCCESS) return 0;
    return info.phys_footprint;
}
}

@implementation MNNProbe

+ (NSDictionary<NSString *, id> *)probeConditionerAtDirectory:(NSString *)directory {
    @autoreleasepool {
        NSString *path = [directory stringByAppendingPathComponent:@"flow-conditioner.fp32.mnn"];
        if (![NSFileManager.defaultManager fileExistsAtPath:path]) {
            return @{ @"success": @NO, @"message": @"缺少 flow-conditioner.fp32.mnn" };
        }
        auto start = std::chrono::steady_clock::now();
        auto interpreter = std::unique_ptr<MNN::Interpreter, decltype(&MNN::Interpreter::destroy)>(
            MNN::Interpreter::createFromFile(path.fileSystemRepresentation), &MNN::Interpreter::destroy);
        if (!interpreter) return @{ @"success": @NO, @"message": @"MNN 无法解析 Conditioner 图" };
        double loadSeconds = elapsedSeconds(start);

        MNN::BackendConfig backend;
        backend.precision = MNN::BackendConfig::Precision_High;
        backend.memory = MNN::BackendConfig::Memory_Low;
        MNN::ScheduleConfig config;
        config.type = MNN_FORWARD_CPU;
        config.numThread = 2;
        config.backendConfig = &backend;
        start = std::chrono::steady_clock::now();
        MNN::Session *session = interpreter->createSession(config);
        if (!session) return @{ @"success": @NO, @"message": @"Conditioner CPU 会话创建失败" };
        double createSeconds = elapsedSeconds(start);
        uint64_t footprint = footprintBytes();
        uint64_t available = os_proc_available_memory();
        interpreter->releaseModel();
        interpreter->releaseSession(session);
        return @{ @"success": @YES, @"message": @"Conditioner CPU 会话创建成功；尚未前向推理",
                  @"loadSeconds": @(loadSeconds), @"createSeconds": @(createSeconds),
                  @"footprintMB": @(footprint / 1000000.0), @"availableMB": @(available / 1000000.0) };
    }
}

@end
