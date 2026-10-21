#import "CosyVoiceGenerate.h"
#include "CosyVoiceDSP.hpp"
#include "CosyVoiceStableAttention.hpp"
#include "CosyVoiceReference.hpp"
#include "CosyVoiceProbeMetrics.hpp"
#include "CosyVoiceBudget.hpp"
#include "CosyVoiceFailure.hpp"
#include "CosyVoiceFlowW8.hpp"
#include "CosyVoiceOpProfile.hpp"
#import <MNN/Interpreter.hpp>
#import <MNN/llm/llm.hpp>
#import <mach/mach.h>
#import <os/proc.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstring>
#include <exception>
#include <fcntl.h>
#include <functional>
#include <iterator>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <string>
#include <thread>
#include <vector>
#include <unistd.h>
#include <sys/resource.h>

namespace {
constexpr int kPromptFrames = CosyVoiceBudget::promptFrames;
constexpr int kTargetTokens = 25;
constexpr int kMaxSpeechTokens = 160;
constexpr int kResidentFlowFrames = CosyVoiceBudget::residentFrames;
constexpr int kResidentSpeechTokens = CosyVoiceBudget::speechCapacity;
constexpr int kSpeechOffset = 151924;
constexpr int kSpeechEos = kSpeechOffset + 6562;
constexpr int kSampleRate = 24000;
constexpr int kHop = 480;
constexpr int kFft = 16;
constexpr double kPi = 3.14159265358979323846;
// First 25 speech tokens generated on PC by the published MNN LLM for its
// fixed Chinese text. This is NOT on-device text-to-speech or live streaming.
constexpr int kSpeechTokens[kTargetTokens] = {
    29, 2, 247, 132, 234, 1694, 4115, 4509, 4592, 4604,
    2402, 2294, 4480, 4426, 5656, 5347, 2189, 2241, 4428, 4429,
    2216, 4430, 5132, 5666, 5089
};
// Same real 74-token PC fixture as testdata/pc-real-74-speech-tokens.csv.
// Used only for paired, deterministic diagnostics, not current-text generation.
constexpr int kProbeSpeechTokens[] = {
    29,2,247,132,234,1694,4115,4509,4592,4604,2402,2294,4480,4426,5656,5347,2189,2241,4428,
    4429,2216,4430,5132,5666,5089,1442,1451,1937,1773,1854,4723,4582,4104,501,5073,2915,
    4697,4696,4669,2190,0,2046,2130,3809,1622,2162,4834,4968,1406,5806,5583,1693,1154,1475,
    1466,902,1154,1919,723,561,4887,4677,6384,5505,5570,5057,5759,3625,6538,6001,2353,2432,245,2
};

double secondsSince(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}
double processCPUSeconds() {
    rusage usage{};
    if (getrusage(RUSAGE_SELF, &usage) != 0) return -1;
    return usage.ru_utime.tv_sec + usage.ru_utime.tv_usec/1e6 +
           usage.ru_stime.tv_sec + usage.ru_stime.tv_usec/1e6;
}
NSString *documentPath(NSString *name) {
    NSURL *url = [NSFileManager.defaultManager URLsForDirectory:NSDocumentDirectory inDomains:NSUserDomainMask].firstObject;
    return [url URLByAppendingPathComponent:name].path;
}
uint64_t footprintBytes() {
    task_vm_info_data_t info = {};
    mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
    return task_info(mach_task_self(), TASK_VM_INFO, reinterpret_cast<task_info_t>(&info), &count) == KERN_SUCCESS
        ? info.phys_footprint : 0;
}
struct MemorySampler;
struct Trace {
    MemorySampler *sampler = nullptr;
    bool memoryFailure = false;
    NSString *fileName;
    explicit Trace(NSString *name = @"last-generation.log") : fileName(name) {}
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    double peakMB = 0;
    double minAvailableMB = 1e20;
    void record(NSString *message, bool fresh = false) {
        double footprint = footprintBytes() / 1e6;
        double available = os_proc_available_memory() / 1e6;
        peakMB = std::max(peakMB, footprint);
        minAvailableMB = std::min(minAvailableMB, available);
        NSString *line = [NSString stringWithFormat:@"%.3f\t%@\tfootprint=%.0f MB\tavailable=%.0f MB\n",
                          secondsSince(start), message, footprint, available];
        NSString *path = documentPath(fileName);
        if (fresh) {
            [line writeToFile:path atomically:YES encoding:NSUTF8StringEncoding error:nil];
        } else {
            NSFileHandle *handle = [NSFileHandle fileHandleForWritingAtPath:path];
            if (handle) {
                @try {
                    [handle seekToEndOfFile];
                    [handle writeData:[line dataUsingEncoding:NSUTF8StringEncoding]];
                    [handle synchronizeFile];
                } @finally { [handle closeFile]; }
            }
        }
    }
};
struct MemorySampler {
    std::atomic<bool> running{true};
    std::atomic<uint64_t> peak{0}, available{UINT64_MAX};
    std::thread worker;
    MemorySampler() : worker([this] {
        while (running.load()) { sample(); std::this_thread::sleep_for(std::chrono::milliseconds(20)); }
    }) { sample(); }
    void sample() {
        uint64_t p = footprintBytes(), a = os_proc_available_memory(), old = peak.load();
        while (old < p && !peak.compare_exchange_weak(old,p)) {}
        old = available.load();
        while (old > a && !available.compare_exchange_weak(old,a)) {}
    }
    void stop(Trace &trace) {
        sample(); running = false;
        if (worker.joinable()) worker.join();
        trace.peakMB = std::max(trace.peakMB, peak.load()/1e6);
        trace.minAvailableMB = std::min(trace.minAvailableMB, available.load()/1e6);
    }
    ~MemorySampler() { running=false; if(worker.joinable())worker.join(); }
};
NSDictionary<NSString *, id> *result(bool ok, NSString *message, Trace &trace) {
    if (trace.sampler) {
        trace.sampler->stop(trace);
        // Validate again AFTER the final sample; a late dip must not escape
        // through a successful return between the last phase check and stop.
        if (ok && !CosyVoiceBudget::accepts(os_proc_available_memory(),trace.sampler->available.load())) {
            ok = false;
            trace.memoryFailure = true;
            message = @"最终内存采样低于800 MB保护门槛，本次不计成功；请分享日志";
            trace.record(message);
        }
    }
    return @{@"success": @(ok), @"message": message, @"failureKind": trace.memoryFailure ? @"memory" : @"other", @"peakFootprintMB": @(trace.peakMB),
             @"minAvailableMB": @(trace.minAvailableMB), @"logPath": documentPath(trace.fileName),
             @"wavPath": documentPath(@"last-generated.wav")};
}
NSDictionary<NSString *, id> *memoryResult(NSString *message, Trace &trace) {
    trace.memoryFailure = true;
    return result(false,message,trace);
}
bool waitAvailable(uint64_t minimum, Trace &trace, NSString *phase) {
    auto start=std::chrono::steady_clock::now();
    int passed=0;
    while (secondsSince(start)<5.0) {
        passed=os_proc_available_memory()>=minimum ? passed+1 : 0;
        if (passed>=2) {
            trace.record([NSString stringWithFormat:@"%@内存门槛通过：至少%.0f MB，等待%.3f秒",phase,minimum/1e6,secondsSince(start)]);
            return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    trace.record([NSString stringWithFormat:@"%@等待5秒仍未达到%.0f MB，停止",phase,minimum/1e6]);
    trace.memoryFailure = true;
    return false;
}
bool sampledBudgetOK(Trace &trace) {
    auto current=os_proc_available_memory();
    bool ok = CosyVoiceBudget::accepts(current,trace.sampler?trace.sampler->available.load():current);
    if (!ok) trace.memoryFailure = true;
    return ok;
}
bool readFloats(NSString *directory, NSString *name, size_t count, std::vector<float> &output) {
    NSData *data = [NSData dataWithContentsOfFile:[directory stringByAppendingPathComponent:name]];
    if (!data || data.length != count * sizeof(float)) return false;
    output.resize(count);
    std::memcpy(output.data(), data.bytes, data.length);
    return true;
}
bool finite(const std::vector<float> &values) {
    return !values.empty() && std::all_of(values.begin(), values.end(), [](float v) { return std::isfinite(v); });
}
// MNN prints load failures to stdout/stderr rather than returning an error
// string. Capture to a regular file (no blocking pipe), restoring both streams
// on every return. Model actions are serialized by the UI.
struct NativeLogCapture {
    int savedOut = -1, savedErr = -1;
    bool active = false;
    NativeLogCapture() {
        int target = open(documentPath(@"last-llm-runtime.log").fileSystemRepresentation,
                          O_CREAT | O_TRUNC | O_WRONLY, 0600);
        if (target < 0) return;
        std::fflush(stdout);
        std::fflush(stderr);
        savedOut = dup(STDOUT_FILENO);
        savedErr = dup(STDERR_FILENO);
        if (savedOut >= 0 && savedErr >= 0) {
            active = dup2(target, STDOUT_FILENO) >= 0 && dup2(target, STDERR_FILENO) >= 0;
        }
        close(target);
    }
    void flush() { std::fflush(stdout); std::fflush(stderr); }
    void recordFailure(Trace &trace) {
        flush();
        NSString *log = [NSString stringWithContentsOfFile:documentPath(@"last-llm-runtime.log")
                                                encoding:NSUTF8StringEncoding error:nil];
        if (log.length) trace.record([@"MNN 原始输出（末尾）：\n" stringByAppendingString:
                                      [log substringFromIndex:log.length > 6000 ? log.length - 6000 : 0]]);
        else trace.record(@"MNN 未写出原始错误信息；请同时分享生成日志与 LLM 原始日志");
    }
    ~NativeLogCapture() {
        flush();
        if (savedOut >= 0) { dup2(savedOut, STDOUT_FILENO); close(savedOut); }
        if (savedErr >= 0) { dup2(savedErr, STDERR_FILENO); close(savedErr); }
    }
    NativeLogCapture(const NativeLogCapture &) = delete;
    NativeLogCapture &operator=(const NativeLogCapture &) = delete;
};
float peak(const std::vector<float> &values) {
    float value = 0;
    for (float sample : values) value = std::max(value, std::abs(sample));
    return value;
}

using Llm = MNN::Transformer::Llm;
using LlmOwner = std::unique_ptr<Llm, decltype(&Llm::destroy)>;
bool generateSpeechTokens(NSString *directory, NSString *text, const std::vector<int> &prompt,
                          Trace &trace, std::vector<int> &speech, CosyVoiceFailureOutput failure, int cpuThreads, bool *limitReached,
                          bool residentProbe = false, const std::function<bool()> &whileLlmAlive = {},
                          bool retainedFlow = false, int speechLimit = kMaxSpeechTokens,
                          LlmOwner *reusable = nullptr, NSMutableDictionary *metrics = nil, bool deterministic = false,
                          NSString *headExperiment = nil, int experimentSeed = 2307) {
    @autoreleasepool {
        NativeLogCapture nativeLog;
        if (!nativeLog.active) trace.record(@"警告：无法完整捕获 MNN 原始日志，仍将记录阶段与文件检查");
        NSData *source = [NSData dataWithContentsOfFile:[directory stringByAppendingPathComponent:@"config-cpu-cosyvoice-ras.json"]];
        NSDictionary *base = source ? [NSJSONSerialization JSONObjectWithData:source options:0 error:nil] : nil;
        if (![base isKindOfClass:NSDictionary.class]) {
            *failure = @"LLM 配置不存在或无法解析";
            return false;
        }
        NSMutableDictionary *config = [base mutableCopy];
        config[@"backend_type"] = @"cpu";
        config[@"thread_num"] = @(cpuThreads);
        config[@"precision"] = @"low";
        config[@"memory"] = @"low";
        config[@"sampler_type"] = @"mixed";
        config[@"mixed_samplers"] = @[@"penalty", @"topK", @"topP", @"temperature"];
        if (deterministic) config[@"mixed_samplers"] = @[@"penalty", @"greedy"];
        config[@"repetition_penalty"] = @1.15;
        config[@"max_new_tokens"] = @(speechLimit + 1);
        config[@"async"] = @NO;
        [config removeObjectForKey:@"cosyvoice_head_experiment"];
        [config removeObjectForKey:@"cosyvoice_experiment_seed"];
        if(headExperiment) {
            if(![headExperiment isEqual:@"full"]&&![headExperiment isEqual:@"reduced"]) {
                *failure=@"未知输出头模式";return false;
            }
            config[@"cosyvoice_head_experiment"]=headExperiment;
            config[@"cosyvoice_experiment_seed"]=@(experimentSeed);
            trace.record([NSString stringWithFormat:@"输出头端到端实验 %@；仅允许speech/EOS并恢复全局编号；seed=%d；%@、原始mixed采样",headExperiment,experimentSeed,reusable?@"批内复用权重/每句清KV和重置采样器":@"每句独立LLM"]);
        }
        // MNN 3.6.1 uses base_dir + filename verbatim; absolute filenames here
        // would duplicate the directory. Metadata and embedding need the same
        // explicit resolution as the graph, weight and tokenizer.
        config[@"base_dir"] = [directory.stringByStandardizingPath stringByAppendingString:@"/"];
        NSDictionary *defaults = @{@"llm_model": @"llm.mnn", @"llm_weight": @"llm.mnn.weight",
                                    @"tokenizer_file": @"tokenizer.mtok", @"llm_config": @"llm_config.json",
                                    @"embedding_file": @"embeddings_bf16.bin"};
        trace.record([NSString stringWithFormat:@"LLM 模型目录：%@；文件名使用相对路径", config[@"base_dir"]]);
        for (NSString *name in @[@"llm_config", @"llm_model", @"llm_weight", @"tokenizer_file", @"embedding_file"]) {
            id value = base[name] ?: defaults[name];
            if([headExperiment isEqual:@"reduced"]&&[name isEqual:@"llm_model"])value=@"llm-speech-head.mnn";
            if([headExperiment isEqual:@"reduced"]&&[name isEqual:@"llm_weight"])value=@"llm-speech-head.mnn.weight";
            if (![value isKindOfClass:NSString.class] || [value length] == 0 ||
                [value isAbsolutePath] || [[value pathComponents] containsObject:@".."]) {
                *failure = [NSString stringWithFormat:@"LLM %@ 必须是有效相对文件名", name];
                return false;
            }
            config[name] = value;
            NSString *resolved = [config[@"base_dir"] stringByAppendingString:value];
            NSDictionary *attributes = [NSFileManager.defaultManager attributesOfItemAtPath:resolved error:nil];
            unsigned long long size = [attributes[NSFileSize] unsignedLongLongValue];
            BOOL readable = [NSFileManager.defaultManager isReadableFileAtPath:resolved] &&
                            [attributes[NSFileType] isEqualToString:NSFileTypeRegular] && size > 0;
            trace.record([NSString stringWithFormat:@"LLM %@ → %@；%llu 字节；%@",
                          name, value, size, readable ? @"可读" : @"缺失或不可读"]);
            if (!readable) {
                *failure = [NSString stringWithFormat:@"LLM 文件缺失、为空或不可读：%@", resolved];
                return false;
            }
        }
        NSString *configPath = [directory stringByAppendingPathComponent:@"config-ios-runtime.json"];
        NSData *configData = [NSJSONSerialization dataWithJSONObject:config options:NSJSONWritingPrettyPrinted error:nil];
        if (!configData || ![configData writeToFile:configPath atomically:YES]) {
            *failure = @"无法保存手机 LLM 运行配置";
            return false;
        }
        trace.record(@"LLM CPU Low / Memory Low 开始加载；权重使用发布包原有混合量化");
        auto started = std::chrono::steady_clock::now();
        LlmOwner local(nullptr, &Llm::destroy);
        LlmOwner &owner = reusable ? *reusable : local;
        bool needsLoad = !owner;
        if (needsLoad && reusable && !waitAvailable(CosyVoiceBudget::reusableLlmBeforeLoad,trace,@"复用路线LLM加载前")) {
            *failure = @"LLM加载规划余量不足，未加载"; return false;
        }
        if (needsLoad) owner.reset(Llm::createLLM(configPath.UTF8String));
        Llm *llm = owner.get();
        if (!llm) {
            nativeLog.recordFailure(trace);
            *failure = @"手机 LLM 创建失败；请分享生成日志与 LLM 原始日志";
            return false;
        }
        if (needsLoad && !llm->load()) {
            nativeLog.recordFailure(trace);
            *failure = @"手机 LLM load() 失败；请分享生成日志与 LLM 原始日志";
            return false;
        }
        double loadSeconds = needsLoad ? secondsSince(started) : 0;
        if (headExperiment) {
            // Update seed BEFORE reset: patched reset recreates only the MNN
            // mixed sampler. Even a reused model starts the same request RNG.
            NSString *seedConfig=[NSString stringWithFormat:@"{\"cosyvoice_experiment_seed\":%d}",experimentSeed];
            if(!llm->set_config(seedConfig.UTF8String)){*failure=@"更新逐句采样seed失败";return false;}
        }
        if (reusable || headExperiment) llm->reset(); // NEW request/KV, never append sentences.
        if (metrics) { metrics[@"loadSeconds"]=@(loadSeconds); metrics[@"reused"]=@(!needsLoad); }
        trace.record([NSString stringWithFormat:@"LLM 加载完成 %.3f 秒；%@", loadSeconds, reusable ? @"复用路线：本句已reset KV（是否跨声学保留由实际调度决定）" : @"本句独立实例"]);
        std::string userContent = "You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呀。";
        userContent += text.UTF8String;
        std::string templated = llm->apply_chat_template(userContent);
        auto input = llm->tokenizer_encode(templated.empty() ? userContent : templated);
        for (int token : prompt) input.push_back(token + kSpeechOffset);
        const int outputLimit = residentProbe ? 2 : speechLimit + 1;
        trace.record([NSString stringWithFormat:@"LLM 输入 %zu token（含 87 个音色提示 token）；开始 prefill+decode，最多 %d 个输出 token%@",
                      input.size(), outputLimit, residentProbe ? @"；共驻探针，不要求完整句 EOS" : @""]);
        if (!sampledBudgetOK(trace)) {
            *failure = @"LLM 前向前可用内存不足";
            return false;
        }
        const auto *beforeContext = llm->getContext();
        const int64_t prefillBefore = beforeContext ? beforeContext->prefill_us : 0;
        const int64_t decodeBefore = beforeContext ? beforeContext->decode_us : 0;
        started = std::chrono::steady_clock::now();
        std::vector<int> output;
        try { output=llm->generate(input, outputLimit); }
        catch(const std::exception &error) {
            nativeLog.recordFailure(trace);
            *failure=[NSString stringWithFormat:@"LLM采样异常，已停止：%s",error.what()];return false;
        }
        if (!sampledBudgetOK(trace)) { *failure=@"LLM前向期间可用内存或20ms采样低于800 MB，停止"; return false; }
        const auto *context = llm->getContext();
        if(!context || context->status==MNN::Transformer::LlmStatus::INTERNAL_ERROR) {
            nativeLog.recordFailure(trace);
            *failure=@"LLM内部错误（含输出头形状/有限值检查失败），未作为正常EOS或完整音频";return false;
        }
        const int64_t prefillAfter = context ? context->prefill_us : 0;
        const int64_t decodeAfter = context ? context->decode_us : 0;
        double prefillSeconds = (prefillAfter >= prefillBefore ? prefillAfter-prefillBefore : prefillAfter)/1e6;
        double decodeSeconds = (decodeAfter >= decodeBefore ? decodeAfter-decodeBefore : decodeAfter)/1e6;
        if(metrics) {
            metrics[@"forwardSeconds"]=@(secondsSince(started));
            metrics[@"prefillSeconds"]=@(prefillSeconds);
            metrics[@"decodeSeconds"]=@(decodeSeconds);
            metrics[@"counterScope"]=@"per-request delta; reset-to-zero fallback";
        }
        bool eos = context && context->status == MNN::Transformer::LlmStatus::NORMAL_FINISHED;
        int invalidCount = 0;
        for (int token : output) {
            if (token == kSpeechEos) eos = true;
            else if (token >= kSpeechOffset && token <= kSpeechOffset + 6560) speech.push_back(token - kSpeechOffset);
            else ++invalidCount;
        }
        if(metrics) {
            metrics[@"eosValidated"]=@(eos);metrics[@"invalidTokens"]=@(invalidCount);
            metrics[@"speechTokens"]=@(speech.size());metrics[@"headMode"]=headExperiment?:@"default-full";
            if(headExperiment)metrics[@"seed"]=@(experimentSeed);
        }
        trace.record([NSString stringWithFormat:@"LLM 完成 %.3f 秒；prefill=%.3f 秒，decode=%.3f 秒；speech=%zu，非法=%d，EOS=%@，status=%d",
                      secondsSince(started), prefillSeconds,
                      decodeSeconds, speech.size(), invalidCount,
                      eos ? @"是" : @"否", context ? int(context->status) : -1]);
        if (speech.empty() || invalidCount || (!residentProbe && (!eos || speech.size() > speechLimit))) {
            *limitReached = !invalidCount && speech.size() >= speechLimit && (!eos || speech.size() > speechLimit);
            nativeLog.recordFailure(trace);
            *failure = residentProbe ? @"共驻LLM的两步输出不是有效语音token，未执行共驻Flow" : (!eos || speech.size() > speechLimit
                ? CosyVoiceTokenLimitMessage(speechLimit)
                : @"LLM 未生成有效的 speech token，未运行 Flow");
            return false;
        }
        if (residentProbe) {
            trace.record(@"LLM 权重与短 KV 仍存活；执行同一 Flow 会话的两步前向，二者不并行计算");
            if (!whileLlmAlive || !whileLlmAlive()) { *failure = @"共驻阶段 Flow 前向或内存门槛未通过"; return false; }
        } else {
        NSMutableString *csv = [NSMutableString string];
        for (size_t index = 0; index < speech.size(); ++index) [csv appendFormat:@"%@%d", index ? @"," : @"", speech[index]];
        [csv appendString:@"\n"];
        [csv writeToFile:documentPath(@"last-generated-speech-tokens.csv") atomically:YES encoding:NSUTF8StringEncoding error:nil];
        }
    }
    if (reusable) { trace.record(@"复用LLM仍存活；调用方决定后续声学调度；下次请求清KV并重新prefill"); return true; }
    if (residentProbe) { trace.record(@"共驻探针 LLM 已销毁；Flow 仍保持，未套用普通串行 2500 MB 回收门槛"); return true; }
    if (retainedFlow) {
        trace.record(@"整句LLM已正常EOS并销毁；仅保留真实speech token与常驻Flow，不保留跨句KV");
        if (!waitAvailable(CosyVoiceBudget::residentFlowBeforeRun,trace,@"LLM释放后（已有Flow，预留400 MB工作区＋800 MB底线）")) { *failure=@"LLM释放后余量不足1200 MB，停止常驻链路";return false; }
        return true;
    }
    trace.record(@"LLM 已销毁；仅保留 speech token，开始观察内存回收");
    auto started = std::chrono::steady_clock::now();
    int passed = 0;
    while (secondsSince(started) < 5.0) {
        passed = os_proc_available_memory() >= 2500000000ULL ? passed + 1 : 0;
        if (passed >= 2) {
            trace.record([NSString stringWithFormat:@"LLM→Flow 内存门槛通过；等待 %.3f 秒", secondsSince(started)]);
            return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(125));
    }
    trace.record(@"停止：LLM 释放后可用内存未恢复到 2500 MB");
    trace.memoryFailure = true;
    *failure = @"LLM 已运行，但内存回收未通过门槛；未加载 Flow";
    return false;
}
#include "CosyVoiceVerifiedAssets.inc"
struct Graph {
    std::unique_ptr<MNN::Interpreter, decltype(&MNN::Interpreter::destroy)> net{nullptr, &MNN::Interpreter::destroy};
    MNN::Session *session = nullptr;
    Trace *traceLog = nullptr;
    int cpuThreads;
    bool performance;
    bool flowW8 = false;
    NSDictionary *flowIdentity = nil;
    ~Graph() { if (session && net) net->releaseSession(session); }
    Graph(const Graph &) = delete;
    Graph &operator=(const Graph &) = delete;
    explicit Graph(int threads = 4, bool fast = false) : cpuThreads(threads), performance(fast) {}
    bool open(NSString *path, bool metal, Trace &trace, NSString *name, bool lowPrecision = false, bool bf16 = false, bool noTune = false, bool normalMemory = false, bool profile = false) {
        traceLog = &trace;
        auto started = std::chrono::steady_clock::now();
        net.reset(MNN::Interpreter::createFromFile(path.fileSystemRepresentation));
        trace.record([NSString stringWithFormat:@"%@ 图读取 %.3f 秒", name, secondsSince(started)]);
        if (!net) return false;
        if(profile)net->setSessionMode(MNN::Interpreter::Session_Debug);
        MNN::BackendConfig backend;
        backend.precision = lowPrecision ? MNN::BackendConfig::Precision_Low : MNN::BackendConfig::Precision_High;
        if (bf16) backend.precision = MNN::BackendConfig::Precision_Low_BF16;
        backend.memory = normalMemory ? MNN::BackendConfig::Memory_Normal : MNN::BackendConfig::Memory_Low;
        backend.power = performance ? MNN::BackendConfig::Power_High : MNN::BackendConfig::Power_Normal;
        MNN::ScheduleConfig config;
        config.type = metal ? MNN_FORWARD_METAL : MNN_FORWARD_CPU;
        config.backupType = MNN_FORWARD_CPU;
        config.numThread = metal ? 1 : cpuThreads;
        if (metal && noTune) config.mode = MNN_GPU_TUNING_NONE;
        config.backendConfig = &backend;
        started = std::chrono::steady_clock::now();
        session = net->createSession(config);
        trace.record([NSString stringWithFormat:@"%@ 会话创建 %.3f 秒", name, secondsSince(started)]);
        return session != nullptr;
    }
    bool shape(const char *name, const std::vector<int> &dimensions) {
        MNN::Tensor *tensor = net->getSessionInput(session, name);
        if (!tensor) return false;
        net->resizeTensor(tensor, dimensions);
        return true;
    }
    bool inputFloat(const char *name, const std::vector<float> &values) {
        if (!finite(values)) {
            traceLog->record([NSString stringWithFormat:@"输入 %s 含非有限值", name]);
            return false;
        }
        MNN::Tensor *input = net->getSessionInput(session, name);
        if (!input) return false;
        MNN::Tensor host(input, input->getDimensionType(), true);
        if (!host.host<float>() || host.usize() != values.size() * sizeof(float)) return false;
        std::memcpy(host.host<float>(), values.data(), values.size() * sizeof(float));
        if (input->copyFromHostTensor(&host)) return true;
        traceLog->record([NSString stringWithFormat:@"输入 %s copyFromHostTensor 失败；未直接写入后端内存", name]);
        return false;
    }
    bool inputInt(const char *name, const std::vector<int> &values) {
        MNN::Tensor *input = net->getSessionInput(session, name);
        if (!input) return false;
        MNN::Tensor host(input, input->getDimensionType(), true);
        if (!host.host<int>() || host.usize() != values.size() * sizeof(int)) return false;
        std::memcpy(host.host<int>(), values.data(), values.size() * sizeof(int));
        if (input->copyFromHostTensor(&host)) return true;
        traceLog->record([NSString stringWithFormat:@"输入 %s 整数复制失败", name]);
        return false;
    }
    bool output(const char *name, size_t count, std::vector<float> &values) {
        MNN::Tensor *tensor = net->getSessionOutput(session, name);
        if (!tensor) {
            traceLog->record([NSString stringWithFormat:@"输出 %s 不存在", name]);
            return false;
        }
        NSMutableString *dimensions = [NSMutableString string];
        for (int dimension : tensor->shape()) [dimensions appendFormat:@"%@%d", dimensions.length ? @"×" : @"", dimension];
        MNN::Tensor host(tensor, tensor->getDimensionType(), true);
        auto type = tensor->getType();
        traceLog->record([NSString stringWithFormat:@"输出 %s：shape=%@；type=%d/%d-bit；hostBytes=%zu，expectedBytes=%zu",
                          name, dimensions, int(type.code), int(type.bits), size_t(host.usize()), count * sizeof(float)]);
        if (!host.host<float>() || type.code != halide_type_float || type.bits != 32 || host.usize() != count * sizeof(float)) {
            traceLog->record(@"输出形状、类型或主机缓冲大小不匹配；未读取数据");
            return false;
        }
        if (!tensor->copyToHostTensor(&host)) {
            traceLog->record(@"输出 copyToHostTensor 失败；未直接读取后端内存");
            return false;
        }
        values.assign(host.host<float>(), host.host<float>() + count);
        size_t invalidCount = 0, firstInvalid = 0;
        for (size_t index = 0; index < values.size(); ++index) {
            if (!std::isfinite(values[index])) {
                if (invalidCount == 0) firstInvalid = index;
                ++invalidCount;
            }
        }
        if (invalidCount) {
            traceLog->record([NSString stringWithFormat:@"输出 %s 非有限值 %zu/%zu；首个位置 %zu；值=%g",
                              name, invalidCount, count, firstInvalid, double(values[firstInvalid])]);
            return false;
        }
        traceLog->record([NSString stringWithFormat:@"输出 %s 全部有限；峰值 %.6f", name, peak(values)]);
        return true;
    }
    bool run() {
        auto code = net->runSession(session);
        if (code != MNN::NO_ERROR) traceLog->record([NSString stringWithFormat:@"runSession 返回错误 %d", int(code)]);
        return code == MNN::NO_ERROR;
    }
    int mainBackend() const {
        // MNN requires 1 + config count ints, not a pointer to one scalar.
        // Every Graph here creates exactly one ScheduleConfig.
        int backends[2]={-1,-1};
        return net->getSessionInfo(session,MNN::Interpreter::BACKENDS,backends)?backends[0]:-1;
    }
    void releaseModel() { net->releaseModel(); }
};
#include "CosyVoiceHiFTBatch.inc"
#include "CosyVoiceMetalDiagnostics.inc"
#include "CosyVoiceMixedAttention.inc"
int reflected(int index, int length) {
    while (index < 0 || index >= length) index = index < 0 ? -index : 2 * length - 2 - index;
    return index;
}
uint32_t nextNoise(uint32_t &state) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}
bool writeWav(const std::vector<float> &samples, NSString *fileName = @"last-generated.wav") {
    NSMutableData *data = [NSMutableData dataWithCapacity:44 + samples.size() * 2];
    auto bytes = [&](const char *characters, size_t count) { [data appendBytes:characters length:count]; };
    auto le16 = [&](uint16_t value) { uint8_t b[2] = {uint8_t(value), uint8_t(value >> 8)}; [data appendBytes:b length:2]; };
    auto le32 = [&](uint32_t value) { uint8_t b[4] = {uint8_t(value), uint8_t(value >> 8), uint8_t(value >> 16), uint8_t(value >> 24)}; [data appendBytes:b length:4]; };
    bytes("RIFF", 4); le32(uint32_t(36 + samples.size() * 2)); bytes("WAVEfmt ", 8);
    le32(16); le16(1); le16(1); le32(kSampleRate); le32(kSampleRate * 2); le16(2); le16(16);
    bytes("data", 4); le32(uint32_t(samples.size() * 2));
    for (float sample : samples) le16(uint16_t(int16_t(std::lrint(std::clamp(sample, -0.99f, 0.99f) * 32767.0f))));
    return [data writeToFile:documentPath(fileName) atomically:YES];
}
}

@interface CosyVoiceGenerate ()
+ (NSDictionary<NSString *, id> *)runAtDirectory:(NSString *)directory text:(nullable NSString *)text lowPrecisionFlow:(BOOL)lowPrecision optimized:(BOOL)optimized cpuThreads:(NSInteger)threads;
+ (NSDictionary<NSString *, id> *)runAtDirectory:(NSString *)directory text:(nullable NSString *)text lowPrecisionFlow:(BOOL)lowPrecision optimized:(BOOL)optimized cpuThreads:(NSInteger)threads prepareHiFT:(BOOL)prepareHiFT;
+ (NSDictionary<NSString *, id> *)runAtDirectory:(NSString *)directory text:(nullable NSString *)text lowPrecisionFlow:(BOOL)lowPrecision optimized:(BOOL)optimized cpuThreads:(NSInteger)threads prepareHiFT:(BOOL)prepareHiFT resident:(Graph *)resident;
+ (NSDictionary<NSString *, id> *)runAtDirectory:(NSString *)directory text:(nullable NSString *)text lowPrecisionFlow:(BOOL)lowPrecision optimized:(BOOL)optimized cpuThreads:(NSInteger)threads prepareHiFT:(BOOL)prepareHiFT resident:(Graph *)resident lowPrecisionHiFT:(BOOL)lowHiFT;
+ (NSDictionary<NSString *, id> *)runAtDirectory:(NSString *)directory text:(nullable NSString *)text lowPrecisionFlow:(BOOL)lowPrecision optimized:(BOOL)optimized cpuThreads:(NSInteger)threads prepareHiFT:(BOOL)prepareHiFT resident:(Graph *)resident lowPrecisionHiFT:(BOOL)lowHiFT preparedSpeech:(const std::vector<int> *)prepared reuseLlm:(LlmOwner *)reusable;
+ (NSDictionary<NSString *, id> *)runAtDirectory:(NSString *)directory text:(nullable NSString *)text lowPrecisionFlow:(BOOL)lowPrecision optimized:(BOOL)optimized cpuThreads:(NSInteger)threads prepareHiFT:(BOOL)prepareHiFT resident:(Graph *)resident lowPrecisionHiFT:(BOOL)lowHiFT preparedSpeech:(const std::vector<int> *)prepared reuseLlm:(LlmOwner *)reusable flowW8:(BOOL)useW8;
+ (NSDictionary<NSString *, id> *)runAtDirectory:(NSString *)directory text:(nullable NSString *)text lowPrecisionFlow:(BOOL)lowPrecision optimized:(BOOL)optimized cpuThreads:(NSInteger)threads prepareHiFT:(BOOL)prepareHiFT resident:(Graph *)resident lowPrecisionHiFT:(BOOL)lowHiFT preparedSpeech:(const std::vector<int> *)prepared reuseLlm:(LlmOwner *)reusable flowW8:(BOOL)useW8 reuseHiFT:(HiFTBatchOwner *)reuseHiFT;
@end

@implementation CosyVoiceGenerate
+ (NSString *)generationLogPath { return documentPath(@"last-generation.log"); }
+ (NSString *)llmRuntimeLogPath { return documentPath(@"last-llm-runtime.log"); }
+ (NSString *)generatedWavPath { return documentPath(@"last-generated.wav"); }
+ (NSString *)hiftProbeLogPath { return documentPath(@"last-hift-precision.log"); }
+ (NSString *)residentProbeLogPath { return documentPath(@"last-resident-flow.log"); }
+ (NSString *)hiftReferenceWavPath { return documentPath(@"hift-high-reference.wav"); }
+ (NSString *)hiftCandidateWavPath { return documentPath(@"hift-low-candidate.wav"); }
+ (NSDictionary<NSString *, NSNumber *> *)memorySnapshot {
    return @{@"footprintMB": @(footprintBytes() / 1e6), @"availableMB": @(os_proc_available_memory() / 1e6)};
}

+ (NSDictionary<NSString *, id> *)runFixedTokenSampleAtDirectory:(NSString *)directory {
    return [self runFixedTokenSampleAtDirectory:directory lowPrecisionFlow:NO];
}

+ (NSDictionary<NSString *, id> *)runFixedTokenSampleAtDirectory:(NSString *)directory lowPrecisionFlow:(BOOL)lowPrecision {
    return [self runAtDirectory:directory text:nil lowPrecisionFlow:lowPrecision optimized:NO cpuThreads:4];
}

+ (NSDictionary<NSString *, id> *)generateText:(NSString *)text atDirectory:(NSString *)directory {
    return [self generateText:text atDirectory:directory optimized:NO cpuThreads:4];
}

+ (NSDictionary<NSString *, id> *)generateText:(NSString *)text atDirectory:(NSString *)directory optimized:(BOOL)optimized cpuThreads:(NSInteger)threads {
    return [self runAtDirectory:directory text:text lowPrecisionFlow:NO optimized:optimized cpuThreads:threads];
}
+ (NSDictionary<NSString *, id> *)generateText:(NSString *)text atDirectory:(NSString *)directory optimized:(BOOL)optimized cpuThreads:(NSInteger)threads lowPrecisionHiFT:(BOOL)low {
    return [self runAtDirectory:directory text:text lowPrecisionFlow:NO optimized:optimized cpuThreads:threads prepareHiFT:NO resident:nullptr lowPrecisionHiFT:low];
}
+ (NSDictionary<NSString *, id> *)generateText:(NSString *)text atDirectory:(NSString *)directory optimized:(BOOL)optimized cpuThreads:(NSInteger)threads lowPrecisionHiFT:(BOOL)low flowW8:(BOOL)useW8 {
    return [self runAtDirectory:directory text:text lowPrecisionFlow:NO optimized:optimized cpuThreads:threads prepareHiFT:NO resident:nullptr lowPrecisionHiFT:low preparedSpeech:nullptr reuseLlm:nullptr flowW8:useW8];
}

+ (NSDictionary<NSString *, id> *)runAtDirectory:(NSString *)directory text:(nullable NSString *)text lowPrecisionFlow:(BOOL)lowPrecision optimized:(BOOL)optimized cpuThreads:(NSInteger)threads {
    return [self runAtDirectory:directory text:text lowPrecisionFlow:lowPrecision optimized:optimized cpuThreads:threads prepareHiFT:NO];
}

+ (NSDictionary<NSString *, id> *)runAtDirectory:(NSString *)directory text:(nullable NSString *)text lowPrecisionFlow:(BOOL)lowPrecision optimized:(BOOL)optimized cpuThreads:(NSInteger)threads prepareHiFT:(BOOL)prepareHiFT {
    return [self runAtDirectory:directory text:text lowPrecisionFlow:lowPrecision optimized:optimized cpuThreads:threads prepareHiFT:prepareHiFT resident:nullptr];
}

+ (NSDictionary<NSString *, id> *)runAtDirectory:(NSString *)directory text:(nullable NSString *)text lowPrecisionFlow:(BOOL)lowPrecision optimized:(BOOL)optimized cpuThreads:(NSInteger)threads prepareHiFT:(BOOL)prepareHiFT resident:(Graph *)resident {
    return [self runAtDirectory:directory text:text lowPrecisionFlow:lowPrecision optimized:optimized cpuThreads:threads prepareHiFT:prepareHiFT resident:resident lowPrecisionHiFT:NO];
}

+ (NSDictionary<NSString *, id> *)runAtDirectory:(NSString *)directory text:(nullable NSString *)text lowPrecisionFlow:(BOOL)lowPrecision optimized:(BOOL)optimized cpuThreads:(NSInteger)threads prepareHiFT:(BOOL)prepareHiFT resident:(Graph *)resident lowPrecisionHiFT:(BOOL)lowHiFT {
    return [self runAtDirectory:directory text:text lowPrecisionFlow:lowPrecision optimized:optimized cpuThreads:threads prepareHiFT:prepareHiFT resident:resident lowPrecisionHiFT:lowHiFT preparedSpeech:nullptr reuseLlm:nullptr];
}

+ (NSDictionary<NSString *, id> *)runAtDirectory:(NSString *)directory text:(nullable NSString *)text lowPrecisionFlow:(BOOL)lowPrecision optimized:(BOOL)optimized cpuThreads:(NSInteger)threads prepareHiFT:(BOOL)prepareHiFT resident:(Graph *)resident lowPrecisionHiFT:(BOOL)lowHiFT preparedSpeech:(const std::vector<int> *)prepared reuseLlm:(LlmOwner *)reusable {
    return [self runAtDirectory:directory text:text lowPrecisionFlow:lowPrecision optimized:optimized cpuThreads:threads prepareHiFT:prepareHiFT resident:resident lowPrecisionHiFT:lowHiFT preparedSpeech:prepared reuseLlm:reusable flowW8:NO];
}

+ (NSDictionary<NSString *, id> *)runAtDirectory:(NSString *)directory text:(nullable NSString *)text lowPrecisionFlow:(BOOL)lowPrecision optimized:(BOOL)optimized cpuThreads:(NSInteger)threads prepareHiFT:(BOOL)prepareHiFT resident:(Graph *)resident lowPrecisionHiFT:(BOOL)lowHiFT preparedSpeech:(const std::vector<int> *)prepared reuseLlm:(LlmOwner *)reusable flowW8:(BOOL)requestedW8 {
    return [self runAtDirectory:directory text:text lowPrecisionFlow:lowPrecision optimized:optimized cpuThreads:threads prepareHiFT:prepareHiFT resident:resident lowPrecisionHiFT:lowHiFT preparedSpeech:prepared reuseLlm:reusable flowW8:requestedW8 reuseHiFT:nullptr];
}
+ (NSDictionary<NSString *, id> *)runAtDirectory:(NSString *)directory text:(nullable NSString *)text lowPrecisionFlow:(BOOL)lowPrecision optimized:(BOOL)optimized cpuThreads:(NSInteger)threads prepareHiFT:(BOOL)prepareHiFT resident:(Graph *)resident lowPrecisionHiFT:(BOOL)lowHiFT preparedSpeech:(const std::vector<int> *)prepared reuseLlm:(LlmOwner *)reusable flowW8:(BOOL)requestedW8 reuseHiFT:(HiFTBatchOwner *)reuseHiFT {
    @autoreleasepool {
        Trace trace(prepareHiFT ? @"last-hift-precision.log" : @"last-generation.log");
        std::unique_ptr<MemorySampler> sampling;
        sampling.reset(new MemorySampler()); trace.sampler=sampling.get();
        if (resident) resident->traceLog=&trace;
        HiFTTraceBinding hiftBinding{reuseHiFT};
        if(reuseHiFT && (!resident || !prepared || reusable || prepareHiFT)) return result(false,@"HiFT复用仅支持无LLM的有界小批声学",trace);
        const int cpuThreads = std::clamp(int(threads),2,4);
        const bool useW8=requestedW8||(resident&&resident->flowW8);
        NSDictionary *flowIdentity=resident?resident->flowIdentity:nil;
        const bool awq=[flowIdentity[@"recipe"] isEqual:@"cosyvoice3-flow-awq-ffn-w4-g32-clipped-v1"];
        NSString *flowLabel = useW8 ? @"Flow FFN W8/block64 CPU High/Memory Low" : (lowPrecision ? @"Flow CPU Low" : @"Flow CPU High");
        if(awq)flowLabel=@"Flow AWQ FFN W4/g32＋投影W8 CPU High/Memory Low";
        const bool textGeneration = text != nil;
        trace.record([NSString stringWithFormat:@"HiFT配置：F0=High，core=%@；Flow精度不随HiFT变化；20ms采样/最低余量800 MB",lowHiFT?@"Low":@"High"], true);
        trace.record([NSString stringWithFormat:@"开始：%@ → Conditioner CPU High → %@ → HiFT F0 High/core %@；串行生成 WAV",
                      prepareHiFT ? @"HiFT 精度对照准备：固定74个真实speech token" :
                      (prepared ? @"本小批已正常EOS的speech token（本次不重跑LLM）" : (textGeneration ? @"当前文字→手机 LLM" : @"固定 25 个 PC speech token（不运行手机文字 LLM）")), flowLabel,lowHiFT?@"Low":@"High"]);
        trace.record([NSString stringWithFormat:@"运行参数：%d CPU 线程；%@；模型精度和两步 Flow 不变",
                      cpuThreads, optimized ? @"优化候选（Flow 32 帧对齐、DSP 查表）" : @"v0.4.1 基线"]);
        const bool projection=[flowIdentity[@"recipe"] isEqual:@"cosyvoice3-flow-projection-w8-g64-v1"];
        if(projection)flowLabel=@"Flow FFN+Q/V/out W8 CPU High/Memory Low；K保留";
        if(useW8) {
            if(lowPrecision||reusable){return result(false,@"W8整链仅支持Flow High和阶段串行，未开放LLM共驻",trace);}
            if(!flowIdentity) {NSString *failure=nil;flowIdentity=verifyFlowW8(directory,trace,&failure);if(!flowIdentity)return result(false,failure?:@"W8身份检查失败",trace);}
            trace.record(awq?@"AWQ实验：44 FFN W4/g32、66投影W8/g64、44逆缩放；用户认可PC试听，但数值筛查未通过，不宣称真机性能/音质验收":(projection?@"显式投影W8整链候选：110个FFN/Q/V/out线性层；K保持原精度，无自动原版回退":@"Flow只量化44个FFN线性层，High精度/Memory Low动态整数；其余层不变，无自动原版回退"));
        }
        if(resident && !prepared)trace.record(@"完整常驻入口：真实当前文字，每句重新prefill；唯一384帧Flow复用，最多105 speech token/4.2秒；LLM释放后才运行HiFT");
        if(resident && prepared)trace.record(@"两句小批声学入口：LLM已经释放；仅完整EOS token进入唯一384帧Flow；F0/HiFT逐句运行");
        if (textGeneration) {
            text = [text stringByTrimmingCharactersInSet:NSCharacterSet.whitespaceAndNewlineCharacterSet];
            if (text.length == 0 || text.length > 40) {
                trace.record(@"停止：首版只接受 1–40 个 UTF-16 文字单元的短句");
                return result(false, @"请输入一条短句（首版最多 40 字符）；长句与流式尚待验证", trace);
            }
        }
        if (resident && !prepared && !waitAvailable(CosyVoiceBudget::residentLlmBeforeLoad,trace,@"常驻Flow下LLM加载前")) return result(false,@"常驻Flow下LLM加载前余量不足",trace);
        if (!resident && !prepared && !reusable && os_proc_available_memory() < 2800000000ULL) {
            trace.record(@"停止：初始可用内存低于 2800 MB");
            return memoryResult(@"初始可用内存不足；未加载模型", trace);
        }
        std::vector<int> tokens;
        NSString *csv = [NSString stringWithContentsOfFile:[directory stringByAppendingPathComponent:@"prompt-speech-tokens.csv"]
                                                   encoding:NSUTF8StringEncoding error:nil];
        for (NSString *item in [csv componentsSeparatedByCharactersInSet:[NSCharacterSet characterSetWithCharactersInString:@",\n\r "]]) {
            if (item.length) tokens.push_back(item.intValue);
        }
        if (tokens.size() != 87) {
            trace.record(@"停止：提示 speech token 数不是 87");
            return result(false, @"prompt-speech-tokens.csv 不符合当前固定样例", trace);
        }
        std::vector<int> speech(std::begin(kSpeechTokens), std::end(kSpeechTokens));
        if (prepareHiFT) speech.assign(std::begin(kProbeSpeechTokens), std::end(kProbeSpeechTokens));
        NSMutableDictionary *llmMetrics = [NSMutableDictionary dictionary];
        if (prepared) {
            speech = *prepared;
            if (!CosyVoiceBudget::fitsSpeech(int(speech.size())) ||
                !std::all_of(speech.begin(),speech.end(),[](int token){return token>=0 && token<=6560;}))
                return result(false,@"预生成speech token无效，未运行Flow",trace);
            trace.record([NSString stringWithFormat:@"使用本批已正常EOS的%zu个speech token；本次不重复运行LLM",speech.size()]);
        } else if (textGeneration) {
            speech.clear();
            NSString *failure = nil;
            bool limitReached = false;
            if (!generateSpeechTokens(directory, text, tokens, trace, speech, &failure, cpuThreads, &limitReached,false,{},resident!=nullptr,resident?kResidentSpeechTokens:kMaxSpeechTokens,reusable,llmMetrics)) {
                trace.record([@"停止：" stringByAppendingString:failure ?: @"LLM 失败"]);
                NSMutableDictionary *failed = [result(false, failure ?: @"LLM 失败", trace) mutableCopy];
                failed[@"retrySplit"] = @(limitReached);
                return failed;
            }
        }
        const int audioFrames = int(speech.size()) * 2;
        const int effectiveFrames = kPromptFrames + audioFrames;
        const int flowFrames = resident ? kResidentFlowFrames : (optimized ? ((effectiveFrames + 31) / 32) * 32 :
            (effectiveFrames <= 256 ? 256 : (effectiveFrames <= 384 ? 384 : 512)));
        if(effectiveFrames>flowFrames)return result(false,@"Flow有效长度超出常驻桶，未执行前向",trace);
        const int stftFrames = audioFrames * 120 + 1;
        const double audioSeconds = double(audioFrames * kHop) / kSampleRate;
        trace.record([NSString stringWithFormat:@"目标 %zu speech token；有效 Mel %d 帧；Flow 图 %d 帧；目标音频 %.2f 秒",
                      speech.size(), effectiveFrames, flowFrames, audioSeconds]);
        tokens.insert(tokens.end(), speech.begin(), speech.end());
        NSMutableDictionary *stageSeconds = [NSMutableDictionary dictionary];
        auto stageStarted = std::chrono::steady_clock::now();
        std::vector<float> mu;
        {
            Graph conditioner(cpuThreads, optimized);
            if (!conditioner.open([directory stringByAppendingPathComponent:@"flow-conditioner.fp32.mnn"], false, trace, @"Conditioner"))
                return result(false, @"Conditioner 会话失败", trace);
            if (!conditioner.shape("tokens", {1, int(tokens.size())})) return result(false, @"Conditioner 输入形状失败", trace);
            conditioner.net->resizeSession(conditioner.session);
            conditioner.releaseModel();
            if (!conditioner.inputInt("tokens", tokens) || !conditioner.run() ||
                !conditioner.output("mu", 80 * effectiveFrames, mu)) {
                trace.record(@"停止：Conditioner 前向或输出失败");
                return result(false, @"Conditioner 前向失败", trace);
            }
            trace.record([NSString stringWithFormat:@"Conditioner 完成：%d 帧；峰值 %.5f", effectiveFrames, peak(mu)]);
        }
        stageSeconds[@"conditioner"] = @(secondsSince(stageStarted));
        std::vector<float> noise, promptCond, spks;
        if (!readFloats(directory, @"rand-noise.bin", 80 * 15000, noise) ||
            !readFloats(directory, @"prompt-cond.bin", 80 * kPromptFrames, promptCond) ||
            !readFloats(directory, @"spks.bin", 80, spks))
            return result(false, @"缺少或损坏 Flow 固定条件", trace);
        std::vector<float> x(80 * flowFrames, 0), paddedMu(x.size(), 0), cond(x.size(), 0);
        for (int channel = 0; channel < 80; ++channel) {
            for (int frame = 0; frame < effectiveFrames; ++frame) {
                x[channel * flowFrames + frame] = noise[channel * 15000 + frame];
                paddedMu[channel * flowFrames + frame] = mu[channel * effectiveFrames + frame];
            }
            for (int frame = 0; frame < kPromptFrames; ++frame)
                cond[channel * flowFrames + frame] = promptCond[channel * kPromptFrames + frame];
        }
        noise.clear(); noise.shrink_to_fit(); mu.clear(); mu.shrink_to_fit();
        std::vector<float> mask(flowFrames, 0);
        std::fill(mask.begin(), mask.begin() + effectiveFrames, 1.0f);
        bool llmHeldForFlow = reusable && bool(*reusable);
        if (llmHeldForFlow && os_proc_available_memory() < CosyVoiceBudget::flowWithLlmBeforeResize) {
            trace.record(@"LLM共驻下Flow初始化规划不足2300 MB；先释放LLM，保留已生成token；本句不重跑LLM");
            reusable->reset(); llmHeldForFlow = false;
            llmMetrics[@"releasedBeforeFlowForBudget"] = @YES;
            if (!waitAvailable(2500000000ULL,trace,@"释放LLM后声学阶段")) return result(false,@"释放LLM后余量未恢复，未运行Flow",trace);
        }
        if (!resident && !llmHeldForFlow && os_proc_available_memory() < 2500000000ULL) {
            trace.record(@"停止：Flow 加载前内存门槛未通过");
            return memoryResult(@"Flow 加载前可用内存不足", trace);
        }
        std::vector<float> mel(80 * audioFrames);
        stageStarted = std::chrono::steady_clock::now();
        {
            std::unique_ptr<Graph> owned;
            if(!resident)owned.reset(new Graph(cpuThreads,optimized));
            Graph &flow=resident?*resident:*owned;
            if (!resident) {
            NSString *flowFile=useW8?flowIdentity[@"files"][@"candidateGraph"][@"file"]:@"flow.cfg-student-2step.batch1.fp16.mnn";
            flow.flowW8=useW8;flow.flowIdentity=flowIdentity;
            if (!flow.open([directory stringByAppendingPathComponent:flowFile], false, trace, flowLabel, lowPrecision))
                return result(false, @"Flow 会话失败", trace);
            if (!(flow.shape("x", {1, 80, flowFrames}) && flow.shape("mask", {1, 1, flowFrames}) &&
                  flow.shape("mu", {1, 80, flowFrames}) && flow.shape("spks", {1, 80}) &&
                  flow.shape("cond", {1, 80, flowFrames}) && flow.shape("t", {1})))
                return result(false, @"Flow 输入形状失败", trace);
            auto started = std::chrono::steady_clock::now();
            flow.net->resizeSession(flow.session);
            trace.record([NSString stringWithFormat:@"%@ %d 帧 resize %.3f 秒；尚未前向", flowLabel, flowFrames, secondsSince(started)]);
            flow.releaseModel();
            if (llmHeldForFlow && os_proc_available_memory() < 1000000000ULL) {
                if (!sampledBudgetOK(trace)) return result(false,@"Flow初始化时采样余量低于800 MB，停止候选",trace);
                trace.record(@"Flow初始化后LLM共驻余量不足1000 MB；释放LLM，保留同一Flow会话，不重跑token");
                reusable->reset(); llmHeldForFlow = false;
                llmMetrics[@"releasedAfterResizeForBudget"] = @YES;
                if (!waitAvailable(1200000000ULL,trace,@"释放LLM后Flow工作区")) return result(false,@"释放LLM后Flow工作区余量不足",trace);
            }
            if (os_proc_available_memory() < (llmHeldForFlow ? 1000000000ULL : 1200000000ULL)) {
                trace.record(@"安全停止：Flow resize后工作区规划余量不足；未前向");
                return memoryResult(@"Flow resize 后内存不足，已安全停止", trace);
            }
            } else {
                trace.record(@"复用本批唯一384帧Flow会话；读取/创建/resize=0；全部六个输入仍逐步重传");
            }
            const float midpoint = float(1.0 - std::cos(kPi / 4.0));
            for (int step = 0; step < 2; ++step) {
                std::vector<float> time = {step == 0 ? 0.0f : midpoint};
                trace.record([NSString stringWithFormat:@"%@ %d/2 开始；重新传入全部六个输入", flowLabel, step + 1]);
                if(resident&&!sampledBudgetOK(trace))return result(false,@"常驻Flow前向前余量或20ms采样不足800 MB",trace);
                auto started = std::chrono::steady_clock::now();
                double cpuStarted = processCPUSeconds();
                if (!(flow.inputFloat("x", x) && flow.inputFloat("mask", mask) &&
                      flow.inputFloat("mu", paddedMu) && flow.inputFloat("spks", spks) &&
                      flow.inputFloat("cond", cond) && flow.inputFloat("t", time))) {
                    trace.record(@"停止：Flow 输入或前向失败");
                    return result(false, @"Flow 前向失败", trace);
                }
                double inputSeconds = secondsSince(started);
                auto runStarted = std::chrono::steady_clock::now();
                if (!flow.run()) return result(false, @"Flow前向失败", trace);
                double runSeconds = secondsSince(runStarted);
                trace.record([NSString stringWithFormat:@"%@ %d/2 runSession 返回成功，%.3f 秒；开始读取输出",
                              flowLabel, step + 1, secondsSince(started)]);
                std::vector<float> velocity;
                auto readStarted = std::chrono::steady_clock::now();
                if (!flow.output("estimator_out", x.size(), velocity)) {
                    trace.record(@"停止：Flow 输出含非有限值或形状错误");
                    return result(false, @"Flow 输出无效", trace);
                }
                double readSeconds = secondsSince(readStarted);
                trace.record([NSString stringWithFormat:@"Flow分项：input=%.4f run=%.4f read=%.4f秒；阶段wall=%.4f，进程CPU=%.4f秒（多线程累计）；thermal=%ld",inputSeconds,runSeconds,readSeconds,secondsSince(started),processCPUSeconds()-cpuStarted,(long)NSProcessInfo.processInfo.thermalState]);
                float delta = step == 0 ? midpoint : 1.0f - midpoint;
                for (size_t index = 0; index < x.size(); ++index) x[index] += delta * velocity[index];
                if (!finite(x)) return result(false, @"Flow 更新后含非有限值", trace);
                if(resident&&!sampledBudgetOK(trace))return result(false,@"常驻Flow期间可用内存或20ms采样不足800 MB",trace);
                trace.record([NSString stringWithFormat:@"Flow %d/2 完成 %.3f 秒；latent 峰值 %.5f",
                              step + 1, secondsSince(started), peak(x)]);
                if (os_proc_available_memory() < 800000000ULL && step == 0) {
                    trace.record(@"安全停止：Flow 第一步后可用内存低于 800 MB");
                    return memoryResult(@"Flow 第一步后内存不足", trace);
                }
            }
            for (int channel = 0; channel < 80; ++channel)
                for (int frame = 0; frame < audioFrames; ++frame)
                    mel[channel * audioFrames + frame] = x[channel * flowFrames + kPromptFrames + frame];
        }
        trace.record([NSString stringWithFormat:@"%@；目标 Mel %d 帧，峰值 %.5f",resident?@"Flow保留供下一句；LLM已销毁，开始HiFT":@"Flow 已释放", audioFrames, peak(mel)]);
        stageSeconds[@"flowSetupTwoStepsRelease"] = @(secondsSince(stageStarted));
        stageStarted = std::chrono::steady_clock::now();
        if (os_proc_available_memory() < 1000000000ULL) {
            trace.record(@"安全停止：HiFT 前可用内存低于 1000 MB");
            return memoryResult(@"HiFT 前内存不足", trace);
        }
        std::vector<float> f0;
        {
            Graph fresh(cpuThreads, optimized);
            Graph *selected=reuseHiFT?reuseHiFT->prepare(false,audioFrames,false,cpuThreads,optimized,directory,trace):&fresh;
            if(!selected)return result(false,@"HiFT F0复用/resize失败",trace);
            Graph &graph=*selected;
            if (!reuseHiFT && !graph.open([directory stringByAppendingPathComponent:@"hift-f0.fp32.mnn"], false, trace, @"HiFT F0"))
                return result(false, @"HiFT F0 会话失败", trace);
            if (!reuseHiFT) {
                if (!graph.shape("mel", {1, 80, audioFrames})) return result(false, @"HiFT F0 形状失败", trace);
                graph.net->resizeSession(graph.session); graph.releaseModel();
            }
            if(!sampledBudgetOK(trace))return memoryResult(@"HiFT F0后余量低于800 MB",trace);
            if (!graph.inputFloat("mel", mel) || !graph.run() || !graph.output("f0", audioFrames, f0))
                return result(false, @"HiFT F0 前向失败", trace);
            trace.record([NSString stringWithFormat:@"HiFT F0 完成；峰值 %.5f", peak(f0)]);
        }
        stageSeconds[@"f0"] = @(secondsSince(stageStarted));
        stageStarted = std::chrono::steady_clock::now();
        std::vector<float> weights, bias;
        if (!readFloats(directory, @"source-linear-weight.bin", 9, weights) ||
            !readFloats(directory, @"source-linear-bias.bin", 1, bias))
            return result(false, @"HiFT source 参数缺失", trace);
        std::vector<float> source(audioFrames * kHop);
        uint32_t randomState = 1986;
        double phase[9] = {};
        for (int frame = 0; frame < audioFrames; ++frame) {
            bool voiced = f0[frame] > 10.0f;
            float sine[9];
            for (int harmonic = 0; harmonic < 9; ++harmonic) {
                phase[harmonic] += double(f0[frame]) * double(harmonic + 1) / kSampleRate * (2.0 * kPi * kHop);
                sine[harmonic] = voiced ? float(0.1 * std::sin(phase[harmonic])) : 0.0f;
            }
            float amplitude = voiced ? 0.003f : (0.1f / 3.0f);
            for (int sample = 0; sample < kHop; ++sample) {
                float sum = bias[0];
                for (int harmonic = 0; harmonic < 9; ++harmonic) {
                    float noiseValue = float(nextNoise(randomState) >> 8) / 16777216.0f;
                    sum += (noiseValue * amplitude + sine[harmonic]) * weights[harmonic];
                }
                source[frame * kHop + sample] = std::tanh(sum);
            }
        }
        std::vector<float> stft(18 * stftFrames);
        auto dspStarted = std::chrono::steady_clock::now();
        if (optimized) {
            stft = CosyVoiceDSP::stft(source, stftFrames);
        } else {
        for (int frame = 0; frame < stftFrames; ++frame) {
            for (int bin = 0; bin < 9; ++bin) {
                double real = 0, imag = 0;
                for (int index = 0; index < kFft; ++index) {
                    double window = 0.5 - 0.5 * std::cos(2.0 * kPi * index / kFft);
                    double sample = source[reflected(frame * 4 + index - 8, int(source.size()))] * window;
                    double angle = -2.0 * kPi * bin * index / kFft;
                    real += sample * std::cos(angle); imag += sample * std::sin(angle);
                }
                stft[bin * stftFrames + frame] = float(real);
                stft[(bin + 9) * stftFrames + frame] = float(imag);
            }
        }
        }
        source.clear(); source.shrink_to_fit();
        stageSeconds[@"sourceAndSTFT"] = @(secondsSince(stageStarted));
        trace.record([NSString stringWithFormat:@"HiFT source STFT 完成；峰值 %.5f", peak(stft)]);
        trace.record([NSString stringWithFormat:@"STFT DSP %.3f 秒", secondsSince(dspStarted)]);
        if (prepareHiFT) {
            trace.record(@"固定 Mel 与 F0/source-STFT 已准备；High/Low 将使用同一份输入，不运行两次 LLM/Flow");
            return @{@"success": @YES, @"frames": @(audioFrames), @"peakFootprintMB": @(trace.peakMB),
                     @"minAvailableMB": @(trace.minAvailableMB),
                     @"mel": [NSData dataWithBytes:mel.data() length:mel.size()*sizeof(float)],
                     @"stft": [NSData dataWithBytes:stft.data() length:stft.size()*sizeof(float)]};
        }
        if (os_proc_available_memory() < 1000000000ULL) return memoryResult(@"HiFT core 前内存不足", trace);
        std::vector<float> spectral;
        stageStarted = std::chrono::steady_clock::now();
        {
            Graph fresh(cpuThreads, optimized);
            Graph *selected=reuseHiFT?reuseHiFT->prepare(true,audioFrames,lowHiFT,cpuThreads,optimized,directory,trace):&fresh;
            if(!selected)return result(false,@"HiFT core复用/resize失败",trace);
            Graph &graph=*selected;
            if (!reuseHiFT && !graph.open([directory stringByAppendingPathComponent:@"hift-core.fp32.mnn"], false, trace, lowHiFT?@"HiFT core Low":@"HiFT core High", lowHiFT))
                return result(false, @"HiFT core 会话失败", trace);
            if (!reuseHiFT && !(graph.shape("mel", {1, 80, audioFrames}) &&
                  graph.shape("source_stft", {1, 18, stftFrames})))
                return result(false, @"HiFT core 输入形状失败", trace);
            if(!reuseHiFT){graph.net->resizeSession(graph.session); graph.releaseModel();}
            if (!sampledBudgetOK(trace)) return result(false, @"HiFT resize后当前或采样余量低于800 MB，停止", trace);
            if (!graph.inputFloat("mel", mel) || !graph.inputFloat("source_stft", stft) ||
                !graph.run() || !graph.output("spectral", 18 * stftFrames, spectral))
                return result(false, @"HiFT core 前向失败", trace);
            trace.record([NSString stringWithFormat:@"HiFT core 完成；频谱峰值 %.5f", peak(spectral)]);
        }
        stageSeconds[@"hiFTCoreSetupRunRelease"] = @(secondsSince(stageStarted));
        dspStarted = std::chrono::steady_clock::now();
        std::vector<float> pcm(audioFrames * kHop);
        if (optimized) {
            pcm = CosyVoiceDSP::istft(spectral, stftFrames, int(pcm.size()));
        } else {
        std::vector<double> accumulated((stftFrames - 1) * 4 + kFft, 0), envelope(accumulated.size(), 0);
        for (int frame = 0; frame < stftFrames; ++frame) {
            for (int index = 0; index < kFft; ++index) {
                double value = 0;
                for (int bin = 0; bin <= 8; ++bin) {
                    double magnitude = std::min(std::exp(std::clamp(double(spectral[bin * stftFrames + frame]), -80.0, 80.0)), 100.0);
                    double phaseAngle = std::sin(double(spectral[(bin + 9) * stftFrames + frame]));
                    double real = magnitude * std::cos(phaseAngle);
                    double imag = magnitude * std::sin(phaseAngle);
                    double angle = 2.0 * kPi * bin * index / kFft;
                    double contribution = real * std::cos(angle) - imag * std::sin(angle);
                    value += (bin == 0 || bin == 8) ? contribution : 2.0 * contribution;
                }
                double window = 0.5 - 0.5 * std::cos(2.0 * kPi * index / kFft);
                size_t position = size_t(frame * 4 + index);
                accumulated[position] += value / kFft * window;
                envelope[position] += window * window;
            }
        }
        for (size_t index = 0; index < pcm.size(); ++index) {
            size_t position = index + kFft / 2;
            double value = envelope[position] > 1e-11 ? accumulated[position] / envelope[position] : accumulated[position];
            pcm[index] = float(std::clamp(value, -0.99, 0.99));
        }
        }
        trace.record([NSString stringWithFormat:@"ISTFT DSP %.3f 秒", secondsSince(dspStarted)]);
        stageSeconds[@"istft"] = @(secondsSince(dspStarted));
        if(!sampledBudgetOK(trace))return result(false,@"HiFT期间可用内存或20ms采样不足800 MB，停止链路",trace);
        if (!finite(pcm)) return result(false, @"PCM 含非有限值", trace);
        double energy = 0;
        for (float value : pcm) energy += double(value) * value;
        double rms = std::sqrt(energy / pcm.size());
        if (rms < 0.001 || peak(pcm) > 1.001f || !writeWav(pcm)) {
            trace.record([NSString stringWithFormat:@"停止：PCM/WAV 异常；RMS %.6f，峰值 %.5f", rms, peak(pcm)]);
            return result(false, @"PCM 静音、越界或 WAV 写入失败", trace);
        }
        const double elapsed = secondsSince(trace.start);
        trace.record([NSString stringWithFormat:@"完成：%@→%@→%.2f 秒 WAV；总耗时 %.3f 秒；RTF %.3f；RMS %.5f；PCM 峰值 %.5f；整句串行，非流式。",
                      prepared ? @"本小批完整EOS token（LLM未重跑）" : (textGeneration ? @"当前文字→手机 LLM" : @"固定 PC speech token"), flowLabel, audioSeconds,
                      elapsed, elapsed / audioSeconds, rms, peak(pcm)]);
        NSMutableDictionary *completed = [result(true, [NSString stringWithFormat:@"%@生成 %.2f 秒 WAV；非流式",
                             textGeneration ? @"手机文字 LLM→Flow→HiFT 已" : @"固定 token 已", audioSeconds], trace) mutableCopy];
        completed[@"audioSeconds"] = @(audioSeconds);
        completed[@"generationSeconds"] = @(elapsed);
        completed[@"rtf"] = @(elapsed / audioSeconds);
        completed[@"residentFlow"] = @(resident!=nullptr);
        completed[@"hiftCorePrecision"] = lowHiFT ? @"low" : @"high";
        completed[@"speechTokens"] = @(speech.size());
        completed[@"eos"] = textGeneration ? @YES : (id)NSNull.null;
        completed[@"eosSource"] = prepared ? @"prepared-complete-LLM-output" : (textGeneration ? @"current-LLM-request" : @"fixed-prefix-fixture-not-verified");
        completed[@"llmMetrics"] = llmMetrics;
        completed[@"llmHeldForFlow"] = @(llmHeldForFlow);
        completed[@"flowQuantization"]=awq?@"AWQ FFN W4/g32 scaling+clipping; Q/V/out W8/g64; K preserved":(projection?@"FFN+Q/V/out W8/block64; K preserved":(useW8?@"FFN W8/block64":@"original FP16-stored"));
        if(awq){completed[@"pcNumericScreenPassed"]=@NO;completed[@"userListeningReviewed"]=@YES;completed[@"experimentalOnly"]=@YES;}
        completed[@"hiFTBatchReuse"]=@(reuseHiFT!=nullptr);
        if(reuseHiFT)completed[@"hiFTLifecycle"]=reuseHiFT->metrics();
        completed[@"melSHA256"]=dataSHA256([NSData dataWithBytes:mel.data() length:mel.size()*sizeof(float)]);
        completed[@"pcmSHA256"]=dataSHA256([NSData dataWithBytes:pcm.data() length:pcm.size()*sizeof(float)]);
        completed[@"flowPrecision"]=lowPrecision?@"Low":@"High";completed[@"flowMemory"]=@"Low";
        completed[@"flowSteps"]=@2;completed[@"llmHeadPruned"]=@NO;
        completed[@"speechTokenSHA256"]=dataSHA256([NSData dataWithBytes:speech.data() length:speech.size()*sizeof(int)]);
        if(flowIdentity)completed[@"flowIdentity"]=flowIdentity;
        completed[@"stageSeconds"] = stageSeconds;
        completed[@"stageSecondsScope"] = @"partial acoustic breakdown; total generationSeconds additionally includes LLM, input loading, waits and WAV writing; retained Flow setup is in outer call";
        return completed;
    }
}
#include "CosyVoiceOptimizationProbes.inc"
#include "CosyVoiceRouteExperiments.inc"
@end

#include "CosyVoiceResidentFlow.inc"
#include "CosyVoicePipeline.inc"
