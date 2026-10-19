import AVFoundation
import Foundation

enum ChunkPlayerError: LocalizedError {
    case tooManyQueued
    case invalidSamples
    var errorDescription: String? {
        switch self {
        case .tooManyQueued: return "播放队列超过 4 秒，停止继续缓存以保护内存。"
        case .invalidSamples: return "音频块为空或含无效数值。"
        }
    }
}

@MainActor
final class ChunkPlayer: ObservableObject {
    @Published private(set) var queuedSeconds = 0.0
    @Published private(set) var status = "未播放"
    @Published private(set) var gapCount = 0
    @Published private(set) var gapSeconds = 0.0

    private let engine = AVAudioEngine()
    private let node = AVAudioPlayerNode()
    private let format = AVAudioFormat(commonFormat: .pcmFormatFloat32,
                                       sampleRate: 24_000, channels: 1, interleaved: false)!
    private var queuedFrames = 0
    private var epoch = UUID()
    private var drainedAt: TimeInterval?
    private var hasScheduled = false

    init() {
        engine.attach(node)
        engine.connect(node, to: engine.mainMixerNode, format: format)
    }

    func enqueue(_ samples: [Float]) throws {
        guard !samples.isEmpty && samples.allSatisfy({ $0.isFinite }) else {
            throw ChunkPlayerError.invalidSamples
        }
        guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(samples.count)),
              let channel = buffer.floatChannelData?[0] else { throw ChunkPlayerError.invalidSamples }
        buffer.frameLength = AVAudioFrameCount(samples.count)
        samples.withUnsafeBufferPointer { channel.update(from: $0.baseAddress!, count: samples.count) }
        try enqueueBuffer(buffer)
    }

    // Buffer ownership passes to AVAudioPlayerNode until dataPlayedBack.
    // The file feeder creates a new <=200 ms buffer for every call.
    func enqueueBuffer(_ buffer: AVAudioPCMBuffer) throws {
        let sampleCount = Int(buffer.frameLength)
        guard sampleCount > 0, buffer.format.sampleRate == 24_000,
              buffer.format.channelCount == 1, let channel = buffer.floatChannelData?[0],
              UnsafeBufferPointer(start: channel, count: sampleCount).allSatisfy({ $0.isFinite }) else {
            throw ChunkPlayerError.invalidSamples
        }
        guard queuedFrames + sampleCount <= 24_000 * 4 else { throw ChunkPlayerError.tooManyQueued }
        if !engine.isRunning {
            try AVAudioSession.sharedInstance().setCategory(.playback, mode: .default)
            try AVAudioSession.sharedInstance().setActive(true)
            try engine.start()
        }
        if queuedFrames == 0, hasScheduled, let drainedAt {
            let gap = ProcessInfo.processInfo.systemUptime - drainedAt
            if gap > 0.04 { gapCount += 1; gapSeconds += gap }
        }
        hasScheduled = true
        drainedAt = nil
        queuedFrames += sampleCount
        queuedSeconds = Double(queuedFrames) / 24_000
        let scheduledEpoch = epoch
        node.scheduleBuffer(buffer, completionCallbackType: .dataPlayedBack) { [weak self] _ in
            Task { @MainActor [weak self] in
                guard let self, self.epoch == scheduledEpoch else { return }
                self.queuedFrames = max(0, self.queuedFrames - sampleCount)
                self.queuedSeconds = Double(self.queuedFrames) / 24_000
                if self.queuedFrames == 0 {
                    self.status = "队列已播完"
                    self.drainedAt = ProcessInfo.processInfo.systemUptime
                }
            }
        }
        if !node.isPlaying { node.play() }
        status = "分块播放中"
    }

    func stop(resetMetrics: Bool = true) {
        epoch = UUID() // Ignore completion callbacks from a stopped old run.
        node.stop()
        engine.stop()
        queuedFrames = 0
        queuedSeconds = 0
        hasScheduled = false
        drainedAt = nil
        if resetMetrics { gapCount = 0; gapSeconds = 0 }
        status = "已停止"
    }
}
