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

    private let engine = AVAudioEngine()
    private let node = AVAudioPlayerNode()
    private let format = AVAudioFormat(commonFormat: .pcmFormatFloat32,
                                       sampleRate: 24_000, channels: 1, interleaved: false)!
    private var queuedFrames = 0

    init() {
        engine.attach(node)
        engine.connect(node, to: engine.mainMixerNode, format: format)
    }

    func enqueue(_ samples: [Float]) throws {
        guard !samples.isEmpty && samples.allSatisfy({ $0.isFinite }) else {
            throw ChunkPlayerError.invalidSamples
        }
        guard queuedFrames + samples.count <= 24_000 * 4 else {
            throw ChunkPlayerError.tooManyQueued
        }
        if !engine.isRunning {
            try AVAudioSession.sharedInstance().setCategory(.playback, mode: .default)
            try AVAudioSession.sharedInstance().setActive(true)
            try engine.start()
        }
        guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(samples.count)),
              let channel = buffer.floatChannelData?[0] else {
            throw ChunkPlayerError.invalidSamples
        }
        buffer.frameLength = AVAudioFrameCount(samples.count)
        samples.withUnsafeBufferPointer { source in
            channel.update(from: source.baseAddress!, count: samples.count)
        }
        let sampleCount = samples.count
        queuedFrames += sampleCount
        queuedSeconds = Double(queuedFrames) / 24_000
        node.scheduleBuffer(buffer, completionCallbackType: .dataPlayedBack) { [weak self] _ in
            Task { @MainActor [weak self] in
                guard let self else { return }
                self.queuedFrames = max(0, self.queuedFrames - sampleCount)
                self.queuedSeconds = Double(self.queuedFrames) / 24_000
                if self.queuedFrames == 0 { self.status = "播放完成" }
            }
        }
        if !node.isPlaying { node.play() }
        status = "分块播放中"
    }

    func stop() {
        node.stop()
        engine.stop()
        queuedFrames = 0
        queuedSeconds = 0
        status = "已停止"
    }
}
