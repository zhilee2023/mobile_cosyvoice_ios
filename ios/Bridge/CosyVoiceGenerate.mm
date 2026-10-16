#import "CosyVoiceGenerate.h"
#import <MNN/Interpreter.hpp>
#import <mach/mach.h>
#import <os/proc.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstring>
#include <iterator>
#include <memory>
#include <numeric>
#include <vector>

namespace {
constexpr int kPromptFrames = 174;
constexpr int kTargetTokens = 25;
constexpr int kEffectiveFrames = kPromptFrames + 2 * kTargetTokens;
constexpr int kFlowFrames = 256;
constexpr int kAudioFrames = 50;
constexpr int kSampleRate = 24000;
constexpr int kHop = 480;
constexpr int kFft = 16;
constexpr int kStftFrames = kAudioFrames * 120 + 1;
constexpr double kPi = 3.14159265358979323846;
// First 25 speech tokens generated on PC by the published MNN LLM for its
// fixed Chinese text. This is NOT on-device text-to-speech or live streaming.
constexpr int kSpeechTokens[kTargetTokens] = {
    29, 2, 247, 132, 234, 1694, 4115, 4509, 4592, 4604,
    2402, 2294, 4480, 4426, 5656, 5347, 2189, 2241, 4428, 4429,
    2216, 4430, 5132, 5666, 5089
};

double secondsSince(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
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
struct Trace {
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
        NSString *path = documentPath(@"last-generation.log");
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
NSDictionary<NSString *, id> *result(bool ok, NSString *message, Trace &trace) {
    return @{@"success": @(ok), @"message": message, @"peakFootprintMB": @(trace.peakMB),
             @"minAvailableMB": @(trace.minAvailableMB), @"logPath": documentPath(@"last-generation.log"),
             @"wavPath": documentPath(@"last-generated.wav")};
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
float peak(const std::vector<float> &values) {
    float value = 0;
    for (float sample : values) value = std::max(value, std::abs(sample));
    return value;
}
struct Graph {
    std::unique_ptr<MNN::Interpreter, decltype(&MNN::Interpreter::destroy)> net{nullptr, &MNN::Interpreter::destroy};
    MNN::Session *session = nullptr;
    Trace *traceLog = nullptr;
    ~Graph() { if (session && net) net->releaseSession(session); }
    Graph(const Graph &) = delete;
    Graph &operator=(const Graph &) = delete;
    Graph() = default;
    bool open(NSString *path, bool metal, Trace &trace, NSString *name, bool lowPrecision = false) {
        traceLog = &trace;
        auto started = std::chrono::steady_clock::now();
        net.reset(MNN::Interpreter::createFromFile(path.fileSystemRepresentation));
        trace.record([NSString stringWithFormat:@"%@ 图读取 %.3f 秒", name, secondsSince(started)]);
        if (!net) return false;
        MNN::BackendConfig backend;
        backend.precision = lowPrecision ? MNN::BackendConfig::Precision_Low : MNN::BackendConfig::Precision_High;
        backend.memory = MNN::BackendConfig::Memory_Low;
        MNN::ScheduleConfig config;
        config.type = metal ? MNN_FORWARD_METAL : MNN_FORWARD_CPU;
        config.backupType = MNN_FORWARD_CPU;
        config.numThread = metal ? 1 : 4;
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
    void releaseModel() { net->releaseModel(); }
};
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
bool writeWav(const std::vector<float> &samples) {
    NSMutableData *data = [NSMutableData dataWithCapacity:44 + samples.size() * 2];
    auto bytes = [&](const char *characters, size_t count) { [data appendBytes:characters length:count]; };
    auto le16 = [&](uint16_t value) { uint8_t b[2] = {uint8_t(value), uint8_t(value >> 8)}; [data appendBytes:b length:2]; };
    auto le32 = [&](uint32_t value) { uint8_t b[4] = {uint8_t(value), uint8_t(value >> 8), uint8_t(value >> 16), uint8_t(value >> 24)}; [data appendBytes:b length:4]; };
    bytes("RIFF", 4); le32(uint32_t(36 + samples.size() * 2)); bytes("WAVEfmt ", 8);
    le32(16); le16(1); le16(1); le32(kSampleRate); le32(kSampleRate * 2); le16(2); le16(16);
    bytes("data", 4); le32(uint32_t(samples.size() * 2));
    for (float sample : samples) le16(uint16_t(int16_t(std::lrint(std::clamp(sample, -0.99f, 0.99f) * 32767.0f))));
    return [data writeToFile:documentPath(@"last-generated.wav") atomically:YES];
}
}

@implementation CosyVoiceGenerate
+ (NSString *)generationLogPath { return documentPath(@"last-generation.log"); }
+ (NSString *)generatedWavPath { return documentPath(@"last-generated.wav"); }

+ (NSDictionary<NSString *, id> *)runFixedTokenSampleAtDirectory:(NSString *)directory {
    return [self runFixedTokenSampleAtDirectory:directory lowPrecisionFlow:NO];
}

+ (NSDictionary<NSString *, id> *)runFixedTokenSampleAtDirectory:(NSString *)directory lowPrecisionFlow:(BOOL)lowPrecision {
    @autoreleasepool {
        Trace trace;
        NSString *flowLabel = lowPrecision ? @"Flow CPU Low" : @"Flow CPU High";
        trace.record([NSString stringWithFormat:@"开始：固定 25 个真实 LLM speech token → Conditioner CPU High → %@ 256 → HiFT CPU High；不运行手机文字 LLM", flowLabel], true);
        if (os_proc_available_memory() < 2800000000ULL) {
            trace.record(@"停止：初始可用内存低于 2800 MB");
            return result(false, @"初始可用内存不足；未加载模型", trace);
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
        tokens.insert(tokens.end(), std::begin(kSpeechTokens), std::end(kSpeechTokens));
        std::vector<float> mu;
        {
            Graph conditioner;
            if (!conditioner.open([directory stringByAppendingPathComponent:@"flow-conditioner.fp32.mnn"], false, trace, @"Conditioner"))
                return result(false, @"Conditioner 会话失败", trace);
            if (!conditioner.shape("tokens", {1, int(tokens.size())})) return result(false, @"Conditioner 输入形状失败", trace);
            conditioner.net->resizeSession(conditioner.session);
            conditioner.releaseModel();
            if (!conditioner.inputInt("tokens", tokens) || !conditioner.run() ||
                !conditioner.output("mu", 80 * kEffectiveFrames, mu)) {
                trace.record(@"停止：Conditioner 前向或输出失败");
                return result(false, @"Conditioner 前向失败", trace);
            }
            trace.record([NSString stringWithFormat:@"Conditioner 完成：224 帧；峰值 %.5f", peak(mu)]);
        }
        std::vector<float> noise, promptCond, spks;
        if (!readFloats(directory, @"rand-noise.bin", 80 * 15000, noise) ||
            !readFloats(directory, @"prompt-cond.bin", 80 * kPromptFrames, promptCond) ||
            !readFloats(directory, @"spks.bin", 80, spks))
            return result(false, @"缺少或损坏 Flow 固定条件", trace);
        std::vector<float> x(80 * kFlowFrames, 0), paddedMu(x.size(), 0), cond(x.size(), 0);
        for (int channel = 0; channel < 80; ++channel) {
            for (int frame = 0; frame < kEffectiveFrames; ++frame) {
                x[channel * kFlowFrames + frame] = noise[channel * 15000 + frame];
                paddedMu[channel * kFlowFrames + frame] = mu[channel * kEffectiveFrames + frame];
            }
            for (int frame = 0; frame < kPromptFrames; ++frame)
                cond[channel * kFlowFrames + frame] = promptCond[channel * kPromptFrames + frame];
        }
        noise.clear(); noise.shrink_to_fit(); mu.clear(); mu.shrink_to_fit();
        std::vector<float> mask(kFlowFrames, 0);
        std::fill(mask.begin(), mask.begin() + kEffectiveFrames, 1.0f);
        if (os_proc_available_memory() < 2500000000ULL) {
            trace.record(@"停止：Flow 加载前内存门槛未通过");
            return result(false, @"Flow 加载前可用内存不足", trace);
        }
        std::vector<float> mel(80 * kAudioFrames);
        {
            Graph flow;
            if (!flow.open([directory stringByAppendingPathComponent:@"flow.cfg-student-2step.batch1.fp16.mnn"], false, trace, flowLabel, lowPrecision))
                return result(false, @"Flow 会话失败", trace);
            if (!(flow.shape("x", {1, 80, kFlowFrames}) && flow.shape("mask", {1, 1, kFlowFrames}) &&
                  flow.shape("mu", {1, 80, kFlowFrames}) && flow.shape("spks", {1, 80}) &&
                  flow.shape("cond", {1, 80, kFlowFrames}) && flow.shape("t", {1})))
                return result(false, @"Flow 输入形状失败", trace);
            auto started = std::chrono::steady_clock::now();
            flow.net->resizeSession(flow.session);
            trace.record([NSString stringWithFormat:@"%@ 256 resize %.3f 秒；尚未前向", flowLabel, secondsSince(started)]);
            flow.releaseModel();
            if (os_proc_available_memory() < 1200000000ULL) {
                trace.record(@"安全停止：Flow resize 后可用内存低于 1200 MB；未前向");
                return result(false, @"Flow resize 后内存不足，已安全停止", trace);
            }
            const float midpoint = float(1.0 - std::cos(kPi / 4.0));
            for (int step = 0; step < 2; ++step) {
                std::vector<float> time = {step == 0 ? 0.0f : midpoint};
                trace.record([NSString stringWithFormat:@"%@ %d/2 开始；重新传入全部六个输入", flowLabel, step + 1]);
                started = std::chrono::steady_clock::now();
                if (!(flow.inputFloat("x", x) && flow.inputFloat("mask", mask) &&
                      flow.inputFloat("mu", paddedMu) && flow.inputFloat("spks", spks) &&
                      flow.inputFloat("cond", cond) && flow.inputFloat("t", time) && flow.run())) {
                    trace.record(@"停止：Flow 输入或前向失败");
                    return result(false, @"Flow 前向失败", trace);
                }
                trace.record([NSString stringWithFormat:@"%@ %d/2 runSession 返回成功，%.3f 秒；开始读取输出",
                              flowLabel, step + 1, secondsSince(started)]);
                std::vector<float> velocity;
                if (!flow.output("estimator_out", x.size(), velocity)) {
                    trace.record(@"停止：Flow 输出含非有限值或形状错误");
                    return result(false, @"Flow 输出无效", trace);
                }
                float delta = step == 0 ? midpoint : 1.0f - midpoint;
                for (size_t index = 0; index < x.size(); ++index) x[index] += delta * velocity[index];
                if (!finite(x)) return result(false, @"Flow 更新后含非有限值", trace);
                trace.record([NSString stringWithFormat:@"Flow %d/2 完成 %.3f 秒；latent 峰值 %.5f",
                              step + 1, secondsSince(started), peak(x)]);
                if (os_proc_available_memory() < 800000000ULL && step == 0) {
                    trace.record(@"安全停止：Flow 第一步后可用内存低于 800 MB");
                    return result(false, @"Flow 第一步后内存不足", trace);
                }
            }
            for (int channel = 0; channel < 80; ++channel)
                for (int frame = 0; frame < kAudioFrames; ++frame)
                    mel[channel * kAudioFrames + frame] = x[channel * kFlowFrames + kPromptFrames + frame];
        }
        trace.record([NSString stringWithFormat:@"Flow 已释放；目标 Mel 50 帧，峰值 %.5f", peak(mel)]);
        if (os_proc_available_memory() < 1000000000ULL) {
            trace.record(@"安全停止：HiFT 前可用内存低于 1000 MB");
            return result(false, @"HiFT 前内存不足", trace);
        }
        std::vector<float> f0;
        {
            Graph graph;
            if (!graph.open([directory stringByAppendingPathComponent:@"hift-f0.fp32.mnn"], false, trace, @"HiFT F0"))
                return result(false, @"HiFT F0 会话失败", trace);
            if (!graph.shape("mel", {1, 80, kAudioFrames})) return result(false, @"HiFT F0 形状失败", trace);
            graph.net->resizeSession(graph.session); graph.releaseModel();
            if (!graph.inputFloat("mel", mel) || !graph.run() || !graph.output("f0", kAudioFrames, f0))
                return result(false, @"HiFT F0 前向失败", trace);
            trace.record([NSString stringWithFormat:@"HiFT F0 完成；峰值 %.5f", peak(f0)]);
        }
        std::vector<float> weights, bias;
        if (!readFloats(directory, @"source-linear-weight.bin", 9, weights) ||
            !readFloats(directory, @"source-linear-bias.bin", 1, bias))
            return result(false, @"HiFT source 参数缺失", trace);
        std::vector<float> source(kAudioFrames * kHop);
        uint32_t randomState = 1986;
        double phase[9] = {};
        for (int frame = 0; frame < kAudioFrames; ++frame) {
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
        std::vector<float> stft(18 * kStftFrames);
        for (int frame = 0; frame < kStftFrames; ++frame) {
            for (int bin = 0; bin < 9; ++bin) {
                double real = 0, imag = 0;
                for (int index = 0; index < kFft; ++index) {
                    double window = 0.5 - 0.5 * std::cos(2.0 * kPi * index / kFft);
                    double sample = source[reflected(frame * 4 + index - 8, int(source.size()))] * window;
                    double angle = -2.0 * kPi * bin * index / kFft;
                    real += sample * std::cos(angle); imag += sample * std::sin(angle);
                }
                stft[bin * kStftFrames + frame] = float(real);
                stft[(bin + 9) * kStftFrames + frame] = float(imag);
            }
        }
        source.clear(); source.shrink_to_fit();
        trace.record([NSString stringWithFormat:@"HiFT source STFT 完成；峰值 %.5f", peak(stft)]);
        if (os_proc_available_memory() < 1000000000ULL) return result(false, @"HiFT core 前内存不足", trace);
        std::vector<float> spectral;
        {
            Graph graph;
            if (!graph.open([directory stringByAppendingPathComponent:@"hift-core.fp32.mnn"], false, trace, @"HiFT core"))
                return result(false, @"HiFT core 会话失败", trace);
            if (!(graph.shape("mel", {1, 80, kAudioFrames}) &&
                  graph.shape("source_stft", {1, 18, kStftFrames})))
                return result(false, @"HiFT core 输入形状失败", trace);
            graph.net->resizeSession(graph.session); graph.releaseModel();
            if (!graph.inputFloat("mel", mel) || !graph.inputFloat("source_stft", stft) ||
                !graph.run() || !graph.output("spectral", 18 * kStftFrames, spectral))
                return result(false, @"HiFT core 前向失败", trace);
            trace.record([NSString stringWithFormat:@"HiFT core 完成；频谱峰值 %.5f", peak(spectral)]);
        }
        std::vector<double> accumulated((kStftFrames - 1) * 4 + kFft, 0), envelope(accumulated.size(), 0);
        for (int frame = 0; frame < kStftFrames; ++frame) {
            for (int index = 0; index < kFft; ++index) {
                double value = 0;
                for (int bin = 0; bin <= 8; ++bin) {
                    double magnitude = std::min(std::exp(std::clamp(double(spectral[bin * kStftFrames + frame]), -80.0, 80.0)), 100.0);
                    double phaseAngle = std::sin(double(spectral[(bin + 9) * kStftFrames + frame]));
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
        std::vector<float> pcm(kAudioFrames * kHop);
        for (size_t index = 0; index < pcm.size(); ++index) {
            size_t position = index + kFft / 2;
            double value = envelope[position] > 1e-11 ? accumulated[position] / envelope[position] : accumulated[position];
            pcm[index] = float(std::clamp(value, -0.99, 0.99));
        }
        if (!finite(pcm)) return result(false, @"PCM 含非有限值", trace);
        double energy = 0;
        for (float value : pcm) energy += double(value) * value;
        double rms = std::sqrt(energy / pcm.size());
        if (rms < 0.001 || peak(pcm) > 1.001f || !writeWav(pcm)) {
            trace.record([NSString stringWithFormat:@"停止：PCM/WAV 异常；RMS %.6f，峰值 %.5f", rms, peak(pcm)]);
            return result(false, @"PCM 静音、越界或 WAV 写入失败", trace);
        }
        trace.record([NSString stringWithFormat:@"完成：iPhone 固定 token→%@→1.00 秒 WAV；总耗时 %.3f 秒；RTF %.3f；RMS %.5f；PCM 峰值 %.5f。不是手机文字 LLM，也不是流式。",
                      flowLabel, secondsSince(trace.start), secondsSince(trace.start), rms, peak(pcm)]);
        return result(true, @"固定 speech token→1 秒语音已在 iPhone 完成；不是手机文字 LLM 或流式", trace);
    }
}
@end
