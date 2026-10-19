import Foundation

enum PlaybackMode: String, CaseIterable, Identifiable {
    case immediate, buffered, complete
    var id: String { rawValue }
    var title: String {
        switch self {
        case .immediate: return "首句即播（对照）"
        case .buffered: return "预加载三句再播"
        case .complete: return "全部生成后播放"
        }
    }
    func mayStart(readyFiles: Int, producerFinished: Bool) -> Bool {
        guard readyFiles > 0 else { return false }
        switch self {
        case .immediate: return true
        case .buffered: return readyFiles >= 3 || producerFinished
        case .complete: return producerFinished
        }
    }
}

enum PlaybackDiskBudget {
    static let maxFiles = 6 // Includes a file currently being fed to the player.
    static let maxBytes: Int64 = 8 * 1024 * 1024
    static func mayAppend(files: Int, bytes: Int64, newBytes: Int64) -> Bool {
        files >= 0 && files < maxFiles && bytes >= 0 && newBytes > 0 &&
        newBytes <= maxBytes && bytes <= maxBytes - newBytes
    }
}
