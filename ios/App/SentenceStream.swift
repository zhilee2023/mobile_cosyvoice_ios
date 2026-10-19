import AVFoundation
import Foundation

// The native owner serializes generate/close with std::mutex. It is passed
// to ONE inference queue, never used for parallel model computations.
extension CosyVoiceResidentFlow: @unchecked Sendable {}

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
    static var documents: URL { FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0] }
    static var logURL: URL { documents.appendingPathComponent("last-sentence-stream.log") }
    static var audioURL: URL { documents.appendingPathComponent("last-sentence-stream.wav") }
    private var partialURL: URL { Self.documents.appendingPathComponent("last-sentence-stream.partial.wav") }
    private var diagnostics: URL { Self.documents.appendingPathComponent("last-sentence-diagnostics.log") }

    func start(text: String, directory: String, optimized: Bool, threads: Int, player: ChunkPlayer, resident: Bool = false, lowHiFT: Bool = true) {
        guard !isRunning else { return }
        let plan: [String]
        do { plan = try SentencePlanner.plan(text, limit: resident ? 18 : SentencePlanner.defaultLimit) }
        catch { status = error.localizedDescription; return }
        player.stop()
        self.player = player
        isRunning = true; stopping = false; hasCompletedAudio = false
        audioSeconds = 0; computeSeconds = 0; startupSeconds = 0; generatedCount = 0; plannedCount = plan.count
        firstQueuedSeconds = nil; peakMB = 0; minAvailableMB = .greatestFiniteMagnitude
        started = ProcessInfo.processInfo.systemUptime
        try? Data().write(to: Self.logURL, options: .atomic)
        try? Data().write(to: diagnostics, options: .atomic)
        record("开始句级流水：\(plan.count) 段；\(optimized ? "32 帧对齐/DSP 优化" : "v0.4.1 基线")；CPU \(threads) 线程；播放队列目标≤2秒、硬上限4秒；至多1句待播文件；不是 token 级流式")
        record(resident ? "本批唯一Flow384常驻，固定4线程/DSP优化；每句重新prefill、释放LLM后运行HiFT；每段最多105语音token/4.2秒，不保留跨句KV" : "对照版：每句释放全部模型，每段最多160语音token/6.4秒")
        record("HiFT core=\(lowHiFT ? "Low" : "High")，F0与Flow仍High；非token级流式")
        task = Task { await run(plan: plan, directory: directory, optimized: optimized, threads: threads, player: player, resident: resident, lowHiFT: lowHiFT) }
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

    private func infer(_ text: String, directory: String, optimized: Bool, threads: Int, session: CosyVoiceResidentFlow?, lowHiFT: Bool) async -> [String: Any] {
        await withCheckedContinuation { continuation in
            worker.async {
                let result = session?.generateText(text, lowPrecisionHiFT: lowHiFT) ?? CosyVoiceGenerate.generateText(text, atDirectory: directory, optimized: optimized, cpuThreads: threads, lowPrecisionHiFT: lowHiFT)
                continuation.resume(returning: result)
            }
        }
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

    private func run(plan: [String], directory: String, optimized: Bool, threads: Int, player: ChunkPlayer, resident: Bool, lowHiFT: Bool) async {
        var writer: AVAudioFile?
        var residentSession: CosyVoiceResidentFlow?
        // Only these two app-owned scratch files can be replaced/removed here.
        let slots = (0...1).map { Self.documents.appendingPathComponent("sentence-stream-slot-\($0).wav") }
        defer {
            writer = nil
            for url in slots { try? FileManager.default.removeItem(at: url) }
            playbackTask = nil; task = nil; isRunning = false
        }
        var pending = plan.map { (text: $0, depth: 0) }
        var at = 0
        do {
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
                record(String(format: "常驻初始化 %.3f 秒，计入总计算RTF；后续句段读取/创建/resize=0", startupSeconds))
                try Task.checkCancellation()
            }
            writer = try AVAudioFile(forWriting: partialURL,
                settings: [AVFormatIDKey: kAudioFormatLinearPCM, AVSampleRateKey: 24_000,
                           AVNumberOfChannelsKey: 1, AVLinearPCMBitDepthKey: 16,
                           AVLinearPCMIsFloatKey: false, AVLinearPCMIsBigEndianKey: false],
                commonFormat: .pcmFormatFloat32, interleaved: false)
            while at < pending.count {
                try Task.checkCancellation()
                if !resident { try await memoryGate() }
                status = "生成第 \(at + 1)/\(pending.count) 段；已生成 \(String(format: "%.2f", audioSeconds)) 秒，上一句可同时播放。"
                record("句 \(at + 1)/\(pending.count) 开始，\(pending[at].text.utf16.count) UTF-16 单元")
                let begin = ProcessInfo.processInfo.systemUptime
                let output = await infer(pending[at].text, directory: directory, optimized: optimized, threads: threads, session: residentSession, lowHiFT: lowHiFT)
                let elapsed = ProcessInfo.processInfo.systemUptime - begin
                computeSeconds += elapsed
                peakMB = max(peakMB, output["peakFootprintMB"] as? Double ?? 0)
                minAvailableMB = min(minAvailableMB, output["minAvailableMB"] as? Double ?? .greatestFiniteMagnitude)
                captureDiagnostics(sentence: at + 1, result: output)
                try Task.checkCancellation()
                if output["success"] as? Bool != true {
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
                // Await only the previous sentence's file feeder, not speaker
                // drain. This allows playback to overlap ONE native generation.
                if let previous = playbackTask { try await previous.value }
                try Task.checkCancellation()
                playbackTask = Task { try await self.feed(slot, player: player) }
                at += 1
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
        } catch {
            playbackTask?.cancel(); player.stop()
            status = stopping ? "已停止；当前模型已返回，未继续生成。" : "停止：\(error.localizedDescription)"
            record(status + "；已生成部分仍保留为 partial 文件，不冒充整篇完成")
        }
        if let session = residentSession { await closeResident(session) }
    }
}
