#import "MNNProbe.h"
#import <MNN/Interpreter.hpp>
#import <mach/mach.h>
#import <os/proc.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <limits>
#include <memory>
#include <thread>
#include <vector>

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

NSString *probeLogPath() {
    NSArray<NSURL *> *locations = [[NSFileManager defaultManager] URLsForDirectory:NSDocumentDirectory
                                                                         inDomains:NSUserDomainMask];
    return [locations.firstObject URLByAppendingPathComponent:@"last-memory-probe.log"].path;
}

void writeProbeLine(NSString *line, bool fresh) {
    NSString *path = probeLogPath();
    NSString *entry = [line stringByAppendingString:@"\n"];
    if (fresh) {
        [entry writeToFile:path atomically:YES encoding:NSUTF8StringEncoding error:nil];
        return;
    }
    NSFileHandle *handle = [NSFileHandle fileHandleForWritingAtPath:path];
    if (!handle) return;
    @try {
        [handle seekToEndOfFile];
        [handle writeData:[entry dataUsingEncoding:NSUTF8StringEncoding]];
        [handle synchronizeFile];
    } @finally {
        [handle closeFile];
    }
}

struct ProbeStats {
    std::chrono::steady_clock::time_point started = std::chrono::steady_clock::now();
    double peakFootprintMB = 0;
    double minAvailableMB = std::numeric_limits<double>::infinity();
    std::atomic<bool> sampling{true};
    std::atomic<uint64_t> sampledPeak{0};
    std::atomic<uint64_t> sampledMinimum{UINT64_MAX};
    std::thread worker;
    ProbeStats() : worker([this] {
        while (sampling.load()) {
            sample();
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
    }) { sample(); }
    ~ProbeStats() { stop(); }
    void sample() {
        uint64_t peak = footprintBytes(), available = os_proc_available_memory();
        uint64_t old = sampledPeak.load();
        while (old < peak && !sampledPeak.compare_exchange_weak(old, peak)) {}
        old = sampledMinimum.load();
        while (old > available && !sampledMinimum.compare_exchange_weak(old, available)) {}
    }
    void stop() {
        sample();
        sampling = false;
        if (worker.joinable()) worker.join();
        peakFootprintMB = std::max(peakFootprintMB, sampledPeak.load() / 1000000.0);
        minAvailableMB = std::min(minAvailableMB, sampledMinimum.load() / 1000000.0);
    }
};

void checkpoint(ProbeStats &stats, NSString *stage, bool fresh = false) {
    double footprint = footprintBytes() / 1000000.0;
    double available = os_proc_available_memory() / 1000000.0;
    stats.peakFootprintMB = std::max(stats.peakFootprintMB, footprint);
    stats.minAvailableMB = std::min(stats.minAvailableMB, available);
    writeProbeLine([NSString stringWithFormat:@"%.3f\t%@\tfootprint=%.0f MB\tavailable=%.0f MB",
                    elapsedSeconds(stats.started), stage, footprint, available], fresh);
}

NSDictionary<NSString *, id> *probeResult(bool success, NSString *message, ProbeStats &stats) {
    stats.stop();
    if (success && stats.minAvailableMB < 800.0) {
        success = false;
        message = @"20ms采样最低可用内存低于800 MB；GPU路线未通过，不允许前向";
    }
    return @{ @"success": @(success), @"message": message,
              @"peakFootprintMB": @(stats.peakFootprintMB),
              @"minAvailableMB": @(stats.minAvailableMB),
              @"logPath": probeLogPath() };
}

bool setInputShape(MNN::Interpreter *interpreter, MNN::Session *session,
                   const char *name, const std::vector<int> &shape) {
    MNN::Tensor *input = interpreter->getSessionInput(session, name);
    if (!input) return false;
    interpreter->resizeTensor(input, shape);
    return true;
}

enum class ProbeMode { flowMetal, flowMetalNoTune, flowMetalLowNoTune, flowCPU, hiftCPU };

NSDictionary<NSString *, id> *probeLargeSession(NSString *directory, ProbeMode mode) {
    @autoreleasepool {
        ProbeStats stats;
        const bool flow = mode != ProbeMode::hiftCPU;
        const bool metal = mode == ProbeMode::flowMetal || mode == ProbeMode::flowMetalNoTune || mode == ProbeMode::flowMetalLowNoTune;
        NSString *label = mode == ProbeMode::flowMetal ? @"Flow Metal High / 256 帧" :
            mode == ProbeMode::flowMetalNoTune ? @"Flow Metal High NoTune / 256 帧" :
            mode == ProbeMode::flowMetalLowNoTune ? @"Flow Metal Low NoTune / 256 帧" :
            mode == ProbeMode::flowCPU ? @"Flow CPU Low / 256 帧" : @"HiFT core CPU Low / 160 帧";
        checkpoint(stats, [@"开始：" stringByAppendingString:label], true);
        NSString *file = flow ? @"flow.cfg-student-2step.batch1.fp16.mnn" : @"hift-core.fp32.mnn";
        NSString *path = [directory stringByAppendingPathComponent:file];
        if (![[NSFileManager defaultManager] fileExistsAtPath:path]) {
            checkpoint(stats, @"停止：缺少模型图");
            return probeResult(false, [@"缺少 " stringByAppendingString:file], stats);
        }
        if (flow && ![[NSFileManager defaultManager] fileExistsAtPath:[path stringByAppendingString:@".weight"]]) {
            checkpoint(stats, @"停止：缺少 Flow 外部权重");
            return probeResult(false, @"缺少 Flow .mnn.weight", stats);
        }
        uint64_t available = os_proc_available_memory();
        uint64_t minimum = flow ? 2000000000ULL : 1000000000ULL;
        if (available < minimum) {
            checkpoint(stats, @"停止：创建会话前可用内存低于安全门槛");
            return probeResult(false, @"当前可用内存偏低，未加载大模型；请关闭其他 App 后重试", stats);
        }

        auto interpreter = std::unique_ptr<MNN::Interpreter, decltype(&MNN::Interpreter::destroy)>(
            MNN::Interpreter::createFromFile(path.fileSystemRepresentation), &MNN::Interpreter::destroy);
        if (!interpreter) {
            checkpoint(stats, @"停止：MNN 无法读取模型图");
            return probeResult(false, @"MNN 无法读取模型图", stats);
        }
        checkpoint(stats, @"图已读取；开始创建会话");

        MNN::BackendConfig backend;
        backend.precision = metal && mode != ProbeMode::flowMetalLowNoTune
            ? MNN::BackendConfig::Precision_High : MNN::BackendConfig::Precision_Low;
        backend.memory = MNN::BackendConfig::Memory_Low;
        MNN::ScheduleConfig config;
        config.type = metal ? MNN_FORWARD_METAL : MNN_FORWARD_CPU;
        config.backupType = MNN_FORWARD_CPU;
        config.numThread = metal ? 1 : 4;
        if (mode == ProbeMode::flowMetalNoTune || mode == ProbeMode::flowMetalLowNoTune) config.mode = MNN_GPU_TUNING_NONE;
        config.backendConfig = &backend;
        MNN::Session *session = interpreter->createSession(config);
        if (!session) {
            checkpoint(stats, @"停止：会话创建失败");
            return probeResult(false, @"MNN 会话创建失败；详见内存日志", stats);
        }
        checkpoint(stats, [@"会话已创建；开始设置真实测试形状：" stringByAppendingString:label]);
        if (metal && os_proc_available_memory() < 1000000000ULL) {
            checkpoint(stats, @"停止：Metal会话创建后不足1000 MB，未执行resize或前向");
            interpreter->releaseSession(session);
            return probeResult(false, @"Metal会话创建后余量不足，未执行resize", stats);
        }

        bool shapesOK = flow
            ? setInputShape(interpreter.get(), session, "x", {1, 80, 256})
              && setInputShape(interpreter.get(), session, "mask", {1, 1, 256})
              && setInputShape(interpreter.get(), session, "mu", {1, 80, 256})
              && setInputShape(interpreter.get(), session, "spks", {1, 80})
              && setInputShape(interpreter.get(), session, "cond", {1, 80, 256})
              && setInputShape(interpreter.get(), session, "t", {1})
            : setInputShape(interpreter.get(), session, "mel", {1, 80, 160})
              && setInputShape(interpreter.get(), session, "source_stft", {1, 18, 19201});
        if (shapesOK) {
            interpreter->resizeSession(session);
            checkpoint(stats, flow ? @"Flow 256 帧 resize 完成" : @"HiFT 160 帧 resize 完成");
            if (metal && (os_proc_available_memory() < 800000000ULL || stats.sampledMinimum.load() < 800000000ULL)) {
                checkpoint(stats, @"停止：Metal resize 后余量低于800 MB；不执行前向");
                interpreter->releaseSession(session);
                interpreter.reset();
                checkpoint(stats, @"低余量会话已释放");
                return probeResult(false, @"Metal 256帧 resize 未通过800 MB内存门槛", stats);
            }
        } else {
            checkpoint(stats, @"输入名称不匹配，跳过 resize");
        }
        interpreter->releaseModel();
        checkpoint(stats, @"原始模型已释放；尚未执行前向");
        interpreter->releaseSession(session);
        interpreter.reset();
        checkpoint(stats, @"会话与解释器已释放");
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        checkpoint(stats, @"释放后 0.5 秒");
        if (!shapesOK) return probeResult(false, @"模型输入名称不匹配；未执行前向", stats);
        return probeResult(true, @"256帧会话与resize通过800 MB门槛；未执行模型前向，不代表GPU整链可行", stats);
    }
}
}

@implementation MNNProbe

+ (NSString *)memoryProbeLogPath { return probeLogPath(); }

+ (NSDictionary<NSString *, id> *)probeFlowMetalMemoryAtDirectory:(NSString *)directory {
    return probeLargeSession(directory, ProbeMode::flowMetal);
}

+ (NSDictionary<NSString *, id> *)probeFlowMetalNoTuneMemoryAtDirectory:(NSString *)directory {
    return probeLargeSession(directory, ProbeMode::flowMetalNoTune);
}

+ (NSDictionary<NSString *, id> *)probeFlowCPULowMemoryAtDirectory:(NSString *)directory {
    return probeLargeSession(directory, ProbeMode::flowCPU);
}

+ (NSDictionary<NSString *, id> *)probeFlowMetalLowNoTuneMemoryAtDirectory:(NSString *)directory {
    return probeLargeSession(directory, ProbeMode::flowMetalLowNoTune);
}

+ (NSDictionary<NSString *, id> *)probeHiFTCPUMemoryAtDirectory:(NSString *)directory {
    return probeLargeSession(directory, ProbeMode::hiftCPU);
}

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
