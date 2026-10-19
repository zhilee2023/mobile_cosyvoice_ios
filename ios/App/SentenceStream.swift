import AVFoundation
import Foundation

// The native owner serializes generate/close with std::mutex. It is passed
// to ONE inference queue, never used for parallel model computations.
extension CosyVoiceResidentFlow: @unchecked Sendable {}
extension CosyVoicePipeline: @unchecked Sendable {}

enum SentenceStreamError: LocalizedError {
    case failed(String)
    var errorDescription: String? { if case let .failed(message) = self { return message }; return nil }
}

@MainActor
final class SentenceStream: ObservableObject {
    @Published private(set) var isRunning = false
    @Published private(set) var status = "尚未开始句级生成"
    @Published private(set) var firstQueuedSeconds: Double?
    @Published private(set) var audioSeconds = 0.0
    @Published private(set) var computeSeconds = 0.0
    @Published private(set) var startupSeconds = 0.0
    @Published private(set) var generatedCount = 0
    @Published private(set) var plannedCount = 0
    @Published private(set) var peakMB = 0.0
    @Published private(set) var minAvailableMB = Double.greatestFiniteMagnitude
    @Published private(set) var hasCompletedAudio = false
    private var task: Task<Void, Never>?
    private var playbackTask: Task<Void, Error>?
    private let worker = DispatchQueue(label: "cosyvoice.sentence.inference", qos: .userInitiated)
    private var started = 0.0
    private var stopping = false
    private var player: ChunkPlayer?
    private var reportID: String?
    private var checkpoints: [[String: Any]] = []
    private var preparationSeconds = 0.0
    private var acousticCallSeconds = 0.0
    private var cleanupSeconds = 0.0
    static var documents: URL { FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0] }
    static var logURL: URL { documents.appendingPathComponent("last-sentence-stream.log") }
    static var audioURL: URL { documents.appendingPathComponent("last-sentence-stream.wav") }
    private var partialURL: URL { Self.documents.appendingPathComponent("last-sentence-stream.partial.wav") }
    private var diagnostics: URL { Self.documents.appendingPathComponent("last-sentence-diagnostics.log") }

    func start(text: String, directory: String, optimized: Bool, threads: Int, player: ChunkPlayer, resident: Bool = false, lowHiFT: Bool = true, experimentMode: String? = nil) {
        guard !isRunning else { return }
        let plan: [String]
        do { plan = try SentencePlanner.plan(text, limit: resident || experimentMode != nil ? 18 : SentencePlanner.defaultLimit) }
        catch { status = error.localizedDescription; return }
        player.stop()
        self.player = player
        isRunning = true; stopping = false; hasCompletedAudio = false
        audioSeconds = 0; computeSeconds = 0; startupSeconds = 0; generatedCount = 0; plannedCount = plan.count
        firstQueuedSeconds = nil; peakMB = 0; minAvailableMB = .greatestFiniteMagnitude
        checkpoints = []; preparationSeconds = 0; acousticCallSeconds = 0; cleanupSeconds = 0
        started = ProcessInfo.processInfo.systemUptime
        let routeLabel = experimentMode == "six-sentence-staged" ? "六句分阶段/唯一Flow复用" : (experimentMode == "llm-chain-reuse" ? "整链LLM复用/按需释放" : (experimentMode == "two-sentence-staged" ? "两句分阶段/唯一Flow复用" : (experimentMode == "three-sentence-staged" ? "三句分阶段/唯一Flow复用" : (resident ? "自适应常驻/串行回退" : "串行基线"))))
        reportID = DiagnosticReport.begin(kind: "sentence-generation", label: routeLabel,
            parameters: ["text": text, "optimized": optimized, "threads": threads, "hiFTLow": lowHiFT,
                         "adaptive": resident, "pipelineMode": experimentMode ?? "serial", "initialSegments": plan.count], sources: DiagnosticReport.sentenceSources)
        try? Data().write(to: Self.logURL, options: .atomic)
        try? Data().write(to: diagnostics, options: .atomic)
        record("开始句级流水：\(plan.count) 段；\(optimized ? "32 帧对齐/DSP 优化" : "v0.4.1 基线")；CPU \(threads) 线程；播放队列目标≤2秒、硬上限4秒；至多1句待播文件；不是 token 级流式")
        if experimentMode == "llm-chain-reuse" {
            record("LLM复用候选：每句清KV；尝试保留LLM穿过声学阶段，按规划释放；Flow逐句创建/销毁，最多160 speech token")
        } else if LongTextBenchmarks.batchLimit(experimentMode) > 0 {
            let batchLimit = LongTextBenchmarks.batchLimit(experimentMode)
            record("\(batchLimit)句分阶段候选：最多\(batchLimit)句完整token；释放LLM后复用唯一Flow384；每句最多105 speech token，无LLM/Flow共驻")
        } else {
            record(resident ? "自适应路线：尝试唯一Flow384；LLM前1800MB不足则释放Flow，本批改串行；每句重新prefill，无跨句KV；800MB底线不变" : "对照版：每句释放全部模型，每段最多160语音token/6.4秒")
        }
        record("HiFT core=\(lowHiFT ? "Low" : "High")，F0与Flow仍High；非token级流式")
        if let experimentMode { record("新整链路线：\(experimentMode)；先短文验收，不保证RTF降低/分钟级内存稳定；分阶段路线首句可能更晚") }
        task = Task { await run(plan: plan, directory: directory, optimized: optimized, threads: threads, player: player, resident: resident, lowHiFT: lowHiFT, experimentMode: experimentMode) }
    }

    func stop() {
        guard isRunning else { return }
        stopping = true
        task?.cancel(); playbackTask?.cancel(); player?.stop()
        status = "已停止播放；等待当前短句推理返回并释放，不再生成下一句。"
        record("用户停止：当前 native 调用不强杀，返回后结束；未完成部分不冒充完整音频")
    }

    private func record(_ message: String) {
        let memory = CosyVoiceGenerate.memorySnapshot()
        let footprint = memory["footprintMB"]?.doubleValue ?? 0
        let available = memory["availableMB"]?.doubleValue ?? 0
        peakMB = max(peakMB, footprint); minAvailableMB = min(minAvailableMB, available)
        let elapsed = ProcessInfo.processInfo.systemUptime - started
        let line = String(format: "%.3f\t%@\tfootprint=%.0f MB\tavailable=%.0f MB\n", elapsed, message, footprint, available)
        append(Data(line.utf8), to: Self.logURL)
    }

    private func append(_ data: Data, to url: URL) {
        if let file = try? FileHandle(forWritingTo: url) {
            defer { try? file.close() }
            try? file.seekToEnd(); try? file.write(contentsOf: data); try? file.synchronize()
        }
    }

    private func captureDiagnostics(sentence: Int, result: [String: Any]) {
        append(Data("\n===== 句 \(sentence) · \(result["message"] ?? "") =====\n".utf8), to: diagnostics)
        for name in [CosyVoiceGenerate.generationLogPath(), CosyVoiceGenerate.llmRuntimeLogPath()] {
            append(Data("\n--- \(URL(fileURLWithPath: name).lastPathComponent) ---\n".utf8), to: diagnostics)
            if let file = try? FileHandle(forReadingFrom: URL(fileURLWithPath: name)) {
                defer { try? file.close() }
                while let chunk = try? file.read(upToCount: 16_384), !chunk.isEmpty { append(chunk, to: diagnostics) }
            }
        }
    }

    private func infer(_ text: String, directory: String, optimized: Bool, threads: Int, session: CosyVoiceResidentFlow?, lowHiFT: Bool, pipeline: CosyVoicePipeline? = nil, staged: Bool = false) async -> [String: Any] {
        await withCheckedContinuation { continuation in
            worker.async {
                let result: [String: Any]
                if let pipeline {
                    result = staged ? pipeline.generatePrepared(lowPrecisionHiFT: lowHiFT) : pipeline.generateText(text, lowPrecisionHiFT: lowHiFT)
                } else {
                    result = session?.generateText(text, lowPrecisionHiFT: lowHiFT) ?? CosyVoiceGenerate.generateText(text, atDirectory: directory, optimized: optimized, cpuThreads: threads, lowPrecisionHiFT: lowHiFT)
                }
                continuation.resume(returning: result)
            }
        }
    }

    private func beginPipeline(_ directory: String, mode: String) async -> [String: Any] {
        await withCheckedContinuation { continuation in
            worker.async { continuation.resume(returning: CosyVoicePipeline.begin(atDirectory: directory, mode: mode)) }
        }
    }
    private func prepareBatch(_ texts: [String], pipeline: CosyVoicePipeline) async -> [String: Any] {
        await withCheckedContinuation { continuation in
            worker.async { continuation.resume(returning: pipeline.prepareTexts(texts)) }
        }
    }
    private func closePipeline(_ session: CosyVoicePipeline) async -> [String: Any] {
        let begin = ProcessInfo.processInfo.systemUptime
        let result: [String: Any] = await withCheckedContinuation { continuation in
            worker.async { continuation.resume(returning: session.close()) }
        }
        let elapsed = ProcessInfo.processInfo.systemUptime - begin
        computeSeconds += elapsed; cleanupSeconds += elapsed
        peakMB = max(peakMB, result["peakFootprintMB"] as? Double ?? 0)
        minAvailableMB = min(minAvailableMB, result["minAvailableMB"] as? Double ?? .greatestFiniteMagnitude)
        record("完整链路候选已关闭；模型及token队列回收，见统一报告中的生命周期")
        return result
    }

    private func beginResident(_ directory: String) async -> [String: Any] {
        await withCheckedContinuation { continuation in
            worker.async { continuation.resume(returning: CosyVoiceResidentFlow.begin(atDirectory: directory)) }
        }
    }

    private func closeResident(_ session: CosyVoiceResidentFlow) async {
        // Cleanup must still run when the enclosing Task has been cancelled.
        let output: [String: Any] = await withCheckedContinuation { continuation in
            worker.async { continuation.resume(returning: session.close()) }
        }
        peakMB = max(peakMB, output["peakFootprintMB"] as? Double ?? 0)
        minAvailableMB = min(minAvailableMB, output["minAvailableMB"] as? Double ?? .greatestFiniteMagnitude)
        record("本批唯一Flow已关闭；生命周期见 last-resident-lifecycle.log")
    }

    private func memoryGate() async throws {
        let begin = ProcessInfo.processInfo.systemUptime
        var passed = 0
        while ProcessInfo.processInfo.systemUptime - begin < 5 {
            try Task.checkCancellation()
            let available = CosyVoiceGenerate.memorySnapshot()["availableMB"]?.doubleValue ?? 0
            passed = available >= 2800 ? passed + 1 : 0
            if passed >= 2 { record(String(format: "句间内存门槛通过，等待 %.3f 秒", ProcessInfo.processInfo.systemUptime - begin)); return }
            try await Task.sleep(nanoseconds: 100_000_000)
        }
        throw SentenceStreamError.failed("句间可用内存未恢复到 2800 MB，已停止；未降低门槛。")
    }

    private func feed(_ url: URL, player: ChunkPlayer) async throws {
        let input = try AVAudioFile(forReading: url, commonFormat: .pcmFormatFloat32, interleaved: false)
        guard input.processingFormat.sampleRate == 24_000, input.processingFormat.channelCount == 1 else {
            throw SentenceStreamError.failed("句段 WAV 格式不是 24 kHz 单声道。")
        }
        var lastProgress = ProcessInfo.processInfo.systemUptime
        var previousQueue = player.queuedSeconds
        while input.framePosition < input.length {
            try Task.checkCancellation()
            if player.queuedSeconds + 0.2 > 2.0 {
                if player.queuedSeconds < previousQueue { lastProgress = ProcessInfo.processInfo.systemUptime }
                previousQueue = player.queuedSeconds
                if ProcessInfo.processInfo.systemUptime - lastProgress > 20 {
                    throw SentenceStreamError.failed("播放队列超过20秒未推进，可能被系统中断，已停止。")
                }
                try await Task.sleep(nanoseconds: 30_000_000)
                continue
            }
            guard let buffer = AVAudioPCMBuffer(pcmFormat: input.processingFormat, frameCapacity: 4800) else {
                throw SentenceStreamError.failed("无法分配200毫秒播放缓冲。")
            }
            try input.read(into: buffer, frameCount: 4800)
            guard buffer.frameLength > 0 else { break }
            try player.enqueueBuffer(buffer)
            lastProgress = ProcessInfo.processInfo.systemUptime
            previousQueue = player.queuedSeconds
            if firstQueuedSeconds == nil {
                firstQueuedSeconds = ProcessInfo.processInfo.systemUptime - started
                record(String(format: "首句首块已入播放队列 %.3f 秒（不是扬声器实测时间）", firstQueuedSeconds!))
            }
            await Task.yield()
        }
    }

    private func appendAudio(_ source: URL, writer: AVAudioFile) throws -> Double {
        let input = try AVAudioFile(forReading: source, commonFormat: .pcmFormatFloat32, interleaved: false)
        guard input.processingFormat.sampleRate == 24_000, input.processingFormat.channelCount == 1,
              input.length > 0, input.length <= 153_600 else { throw SentenceStreamError.failed("句段 WAV 长度或格式异常。") }
        while input.framePosition < input.length {
            guard let buffer = AVAudioPCMBuffer(pcmFormat: input.processingFormat, frameCapacity: 4800) else {
                throw SentenceStreamError.failed("无法分配合并音频缓冲。")
            }
            try input.read(into: buffer, frameCount: 4800)
            try writer.write(from: buffer)
        }
        return Double(input.length) / 24_000
    }

    private func run(plan: [String], directory: String, optimized: Bool, threads: Int, player: ChunkPlayer, resident: Bool, lowHiFT: Bool, experimentMode: String?) async {
        var writer: AVAudioFile?
        var residentSession: CosyVoiceResidentFlow?
        var pipelineSession: CosyVoicePipeline?
        var lifecycleResult: [String: Any] = [:]
        var attempts: [[String: Any]] = []
        var generationWallSeconds: Double?
        var batchRemaining = 0
        let stagedBatchLimit = LongTextBenchmarks.batchLimit(experimentMode)
        let staged = stagedBatchLimit > 0
        var completedPlayback = false
        // Only these two app-owned scratch files can be replaced/removed here.
        let slots = (0...1).map { Self.documents.appendingPathComponent("sentence-stream-slot-\($0).wav") }
        defer {
            writer = nil
            for url in slots { try? FileManager.default.removeItem(at: url) }
            playbackTask = nil; task = nil; isRunning = false
            DiagnosticReport.finish(reportID, state: completedPlayback ? "completed" : (stopping ? "cancelled" : "not-completed"),
                result: ["message": status, "generatedSegments": generatedCount, "plannedSegments": plannedCount,
                         "audioSeconds": audioSeconds, "computeSeconds": computeSeconds, "startupSeconds": startupSeconds,
                         "computeRTF": audioSeconds > 0 ? computeSeconds / audioSeconds : NSNull(),
                         "firstQueuedSeconds": firstQueuedSeconds.map { $0 as Any } ?? NSNull(),
                         "peakFootprintMB": peakMB, "minAvailableMB": minAvailableMB,
                         "completedAudio": hasCompletedAudio, "completedPlayback": completedPlayback,
                         "pipelineMode": experimentMode ?? "serial", "lifecycle": lifecycleResult,
                         "attempts": attempts,
                         "checkpoints": checkpoints, "batchLimit": stagedBatchLimit,
                         "preparationSeconds": preparationSeconds, "acousticCallSeconds": acousticCallSeconds,
                         "cleanupSeconds": cleanupSeconds,
                         "timingScope": "compute includes setup, LLM preparation, acoustic calls, failures and cleanup; wall also includes queue/file/report waits",
                         "generationWallSeconds": generationWallSeconds.map { $0 as Any } ?? NSNull(),
                         "generationWallRTF": generationWallSeconds.map { audioSeconds > 0 ? $0/audioSeconds as Any : NSNull() } ?? NSNull(),
                         "queueGaps": player.gapCount, "queueGapSeconds": player.gapSeconds, "tokenStreaming": false])
            reportID = nil
        }
        var pending = plan.map { (text: $0, depth: 0) }
        var at = 0
        do {
            if let experimentMode {
                let begin = ProcessInfo.processInfo.systemUptime
                let setup = await beginPipeline(directory, mode: experimentMode)
                startupSeconds = ProcessInfo.processInfo.systemUptime - begin
                computeSeconds += startupSeconds
                pipelineSession = setup["session"] as? CosyVoicePipeline
                peakMB = max(peakMB, setup["peakFootprintMB"] as? Double ?? 0)
                minAvailableMB = min(minAvailableMB, setup["minAvailableMB"] as? Double ?? .greatestFiniteMagnitude)
                guard setup["success"] as? Bool == true, pipelineSession != nil else {
                    throw SentenceStreamError.failed(setup["message"] as? String ?? "候选初始化失败")
                }
                record("本批候选调度器已就绪，模型按阶段加载；所有调用/等待/失败均计入计算RTF")
                try Task.checkCancellation()
            }
            if resident {
                let begin = ProcessInfo.processInfo.systemUptime
                let setup = await beginResident(directory)
                startupSeconds = ProcessInfo.processInfo.systemUptime - begin
                computeSeconds += startupSeconds
                residentSession = setup["session"] as? CosyVoiceResidentFlow
                peakMB = max(peakMB, setup["peakFootprintMB"] as? Double ?? 0)
                minAvailableMB = min(minAvailableMB, setup["minAvailableMB"] as? Double ?? .greatestFiniteMagnitude)
                guard setup["success"] as? Bool == true, residentSession != nil else {
                    throw SentenceStreamError.failed(setup["message"] as? String ?? "常驻Flow初始化失败。")
                }
                record(String(format: "常驻初始化 %.3f 秒，计入总计算RTF；仅实际常驻的句段读取/创建/resize=0，回退后重新加载", startupSeconds))
                try Task.checkCancellation()
            }
            writer = try AVAudioFile(forWriting: partialURL,
                settings: [AVFormatIDKey: kAudioFormatLinearPCM, AVSampleRateKey: 24_000,
                           AVNumberOfChannelsKey: 1, AVLinearPCMBitDepthKey: 16,
                           AVLinearPCMIsFloatKey: false, AVLinearPCMIsBigEndianKey: false],
                commonFormat: .pcmFormatFloat32, interleaved: false)
            while at < pending.count {
                try Task.checkCancellation()
                if !resident && pipelineSession == nil { try await memoryGate() }
                if staged, batchRemaining == 0, let pipelineSession {
                    let batch = Array(pending[at..<min(pending.count,at+stagedBatchLimit)].map(\.text))
                    let prepareBegin = ProcessInfo.processInfo.systemUptime
                    let prepared = await prepareBatch(batch, pipeline: pipelineSession)
                    let prepareElapsed = ProcessInfo.processInfo.systemUptime - prepareBegin
                    computeSeconds += prepareElapsed
                    preparationSeconds += prepareElapsed
                    attempts.append(["phase": "prepare-batch", "firstSegment": at+1, "batchSize": batch.count, "seconds": prepareElapsed, "result": prepared])
                    peakMB = max(peakMB, prepared["peakFootprintMB"] as? Double ?? 0)
                    minAvailableMB = min(minAvailableMB, prepared["minAvailableMB"] as? Double ?? .greatestFiniteMagnitude)
                    captureDiagnostics(sentence: at+1, result: prepared)
                    try Task.checkCancellation()
                    if prepared["success"] as? Bool != true {
                        let offset = prepared["failedTextIndex"] as? Int ?? 0
                        let index = at + offset
                        if prepared["retrySplit"] as? Bool == true, offset >= 0, offset < batch.count,
                           pending[index].depth < 3, pending[index].text.count > 1 {
                            let item = pending[index]
                            let pieces = try SentencePlanner.partition(item.text, limit: max(1,item.text.utf16.count/2))
                            guard pieces.count > 1, pending.count + pieces.count - 1 <= 128 else { throw SentenceStreamError.failed("句段不能继续拆分") }
                            pending.replaceSubrange(index...index, with: pieces.map { (text: $0, depth: item.depth+1) })
                            plannedCount = pending.count
                            record("分阶段准备超限；未播出此批，有限拆句后重新准备；失败及重算均计入RTF")
                            continue
                        }
                        throw SentenceStreamError.failed(prepared["message"] as? String ?? "批量token准备失败")
                    }
                    batchRemaining = prepared["preparedCount"] as? Int ?? 0
                    guard batchRemaining == batch.count else { throw SentenceStreamError.failed("预生成token队列数量错误") }
                    record("已准备\(batchRemaining)句完整EOS token，LLM已释放；开始唯一Flow声学阶段")
                }
                status = "生成第 \(at + 1)/\(pending.count) 段；已生成 \(String(format: "%.2f", audioSeconds)) 秒，上一句可同时播放。"
                record("句 \(at + 1)/\(pending.count) 开始，\(pending[at].text.utf16.count) UTF-16 单元")
                let begin = ProcessInfo.processInfo.systemUptime
                let output = await infer(pending[at].text, directory: directory, optimized: optimized, threads: threads, session: residentSession, lowHiFT: lowHiFT, pipeline: pipelineSession, staged: staged)
                let elapsed = ProcessInfo.processInfo.systemUptime - begin
                computeSeconds += elapsed
                acousticCallSeconds += elapsed
                attempts.append(["phase": "sentence", "segment": at+1, "seconds": elapsed, "result": output])
                peakMB = max(peakMB, output["peakFootprintMB"] as? Double ?? 0)
                minAvailableMB = min(minAvailableMB, output["minAvailableMB"] as? Double ?? .greatestFiniteMagnitude)
                captureDiagnostics(sentence: at + 1, result: output)
                if let mode = output["executionMode"] as? String {
                    record("本句实际路线：\(mode)；失败/等待/重试均已计入耗时")
                    record("本句LLM已保留=\(output["llmRetainedAfterRequest"] ?? "不适用")；小批Flow复用=\(output["flowReusedWithinBatch"] ?? output["flowReusedWithinPair"] ?? "不适用")；LLM规划释放=\(output["llmBudgetReleases"] ?? 0)")
                }
                if output["hadProtectedAttempt"] as? Bool == true {
                    record("本句曾触发内存保护后回退；即使重试生成成功，也不算全程内存门槛通过；最低余量未清零")
                }
                try Task.checkCancellation()
                if output["success"] as? Bool != true {
                    if staged { batchRemaining = 0 } // Native drops the remaining prepared queue on failure.
                    if output["retrySplit"] as? Bool == true, pending[at].depth < 3, pending[at].text.count > 1 {
                        let item = pending[at]
                        let pieces = try SentencePlanner.partition(item.text, limit: max(1, item.text.utf16.count / 2))
                        guard pieces.count > 1, pending.count + pieces.count - 1 <= 128 else {
                            throw SentenceStreamError.failed("句段无法继续安全拆分。")
                        }
                        pending.replaceSubrange(at...at, with: pieces.map { (text: $0, depth: item.depth + 1) })
                        plannedCount = pending.count
                        record("该段达到语音 token 上限，未截断；拆为 \(pieces.count) 段重做；失败尝试耗时仍计入 RTF")
                        continue
                    }
                    throw SentenceStreamError.failed(output["message"] as? String ?? "句段生成失败。")
                }
                // The native producer writes one fixed filename. Copy it to
                // alternating bounded slots BEFORE starting another inference.
                let slot = slots[at % 2]
                try? FileManager.default.removeItem(at: slot)
                try FileManager.default.copyItem(at: URL(fileURLWithPath: CosyVoiceGenerate.generatedWavPath()), to: slot)
                let duration = try appendAudio(slot, writer: writer!)
                audioSeconds += duration; generatedCount += 1
                record(String(format: "句 %d 完成，音频 %.2f 秒，推理 %.3f 秒，单句 RTF %.3f；累计计算 RTF %.3f", at + 1, duration, elapsed, elapsed / duration, computeSeconds / audioSeconds))
                if generatedCount % 3 == 0 || at + 1 == pending.count {
                    checkpoints.append(["segments": generatedCount, "audioSeconds": audioSeconds,
                        "computeSeconds": computeSeconds, "computeRTF": computeSeconds / audioSeconds,
                        "preparationSeconds": preparationSeconds, "acousticCallSeconds": acousticCallSeconds,
                        "queueGaps": player.gapCount, "thermalState": ProcessInfo.processInfo.thermalState.rawValue,
                        "peakFootprintMB": peakMB, "minAvailableMB": minAvailableMB,
                        "cleanupPending": true])
                    record(String(format: "长文检查点：%d句/%.2f秒，累计计算RTF %.3f；每3句记录温度、内存与断供，最终RTF另含关闭成本", generatedCount, audioSeconds, computeSeconds/audioSeconds))
                }
                // Await only the previous sentence's file feeder, not speaker
                // drain. This allows playback to overlap ONE native generation.
                if let previous = playbackTask { try await previous.value }
                try Task.checkCancellation()
                playbackTask = Task { try await self.feed(slot, player: player) }
                at += 1
                if staged { batchRemaining -= 1 }
            }
            if let session = pipelineSession {
                lifecycleResult = await closePipeline(session)
                pipelineSession = nil
            }
            if let session = residentSession {
                await closeResident(session)
                residentSession = nil
                record(String(format: "常驻稳态：句段计算 %.3f秒（含失败重试，不含初始化），稳态计算RTF %.3f；初始化 %.3f秒已计入总计算RTF", computeSeconds-startupSeconds, (computeSeconds-startupSeconds)/audioSeconds, startupSeconds))
            }
            writer = nil // Finalize WAV header before publishing completed audio.
            if FileManager.default.fileExists(atPath: Self.audioURL.path) {
                _ = try FileManager.default.replaceItemAt(Self.audioURL, withItemAt: partialURL)
            } else { try FileManager.default.moveItem(at: partialURL, to: Self.audioURL) }
            hasCompletedAudio = true
            generationWallSeconds = ProcessInfo.processInfo.systemUptime - started
            record(String(format: "全部生成完成：%d 段，音频 %.2f 秒；计算 %.3f 秒，计算 RTF %.3f；生成墙钟 %.3f 秒（含回收/队列等待），墙钟 RTF %.3f；首块入队 %.3f 秒；播放队列断供观测 %d 次/%.3f 秒；采样峰值 %.0f MB，最低可用 %.0f MB", generatedCount, audioSeconds, computeSeconds, computeSeconds / audioSeconds, ProcessInfo.processInfo.systemUptime - started, (ProcessInfo.processInfo.systemUptime - started) / audioSeconds, firstQueuedSeconds ?? -1, player.gapCount, player.gapSeconds, peakMB, minAvailableMB))
            if let last = playbackTask { try await last.value }
            status = "生成完成，正在播完队列…"
            let drainStart = ProcessInfo.processInfo.systemUptime
            while player.queuedSeconds > 0 {
                try Task.checkCancellation()
                if ProcessInfo.processInfo.systemUptime - drainStart > 20 { throw SentenceStreamError.failed("播放队列未及时播完。") }
                try await Task.sleep(nanoseconds: 50_000_000)
            }
            record(String(format: "播放完成；队列断供观测 %d 次/%.3f 秒；非 token 级流式", player.gapCount, player.gapSeconds))
            status = String(format: "完成 %d 段 / %.2f 秒音频；计算 RTF %.3f；首块入队 %.3f 秒；队列断供 %d 次。", generatedCount, audioSeconds, computeSeconds / audioSeconds, firstQueuedSeconds ?? -1, player.gapCount)
            completedPlayback = true
        } catch {
            playbackTask?.cancel(); player.stop()
            status = stopping ? "已停止；当前模型已返回，未继续生成。" : "停止：\(error.localizedDescription)"
            record(status + "；已生成部分仍保留为 partial 文件，不冒充整篇完成")
        }
        if let session = residentSession { await closeResident(session) }
        if let session = pipelineSession { lifecycleResult = await closePipeline(session) }
    }
}
