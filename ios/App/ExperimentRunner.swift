import Foundation
import Combine

@MainActor
final class ExperimentRunner: ObservableObject {
    static let routes: [(id: String, title: String)] = [
        ("flow-buckets", "① Flow：384 / 352 / 320帧"),
        ("flow-thread-4", "②a Flow High：仅4线程（冷启动测）"),
        ("flow-thread-2", "②b Flow High：仅2线程（冷启动测）"),
        ("flow-thread-3", "②c Flow High：仅3线程（冷启动测）"),
        ("llm-reuse", "③ LLM：独立 / 复用实例，逐句reset KV"),
        ("llm-threads", "④ LLM：4 / 2 / 3线程"),
        ("hift-threads", "⑤ HiFT Low：4 / 2 / 3线程"),
        ("flow-bf16", "⑥ Flow：High / BF16候选")
    ]
    static var summaryURL: URL { SentenceStream.documents.appendingPathComponent("last-experiment-summary.log") }
    @Published private(set) var isRunning = false
    @Published private(set) var status = "尚未开始新路线；旧结论不会自动沿用为新配置通过。"
    private var stopRequested = false
    private let worker = DispatchQueue(label: "cosyvoice.isolated.experiments", qos: .userInitiated)
    func start(routes: [String], directory: String) {
        guard !isRunning, !routes.isEmpty, routes.allSatisfy({ id in Self.routes.contains { $0.id == id } }) else { return }
        isRunning = true; stopRequested = false
        try? Data().write(to: Self.summaryURL, options: .atomic)
        Task {
            defer { isRunning = false }
            var completed = 0
            for (index, route) in routes.enumerated() {
                if stopRequested { break }
                let title = Self.routes.first { $0.id == route }!.title
                status = "\(index + 1)/\(routes.count)：\(title)；请保持前台。当前路线返回后才可停止。"
                record(status)
                let runID = DiagnosticReport.begin(kind: "module-experiment", label: title,
                    parameters: ["route": route, "wholeChainRTFMeasured": false],
                    sources: ["last-experiment-\(route).json", "last-experiment-\(route).log"])
                let result: [String: Any] = await withCheckedContinuation { continuation in
                    worker.async { continuation.resume(returning: CosyVoiceGenerate.runExperiment(route, atDirectory: directory)) }
                }
                let passed = result["success"] as? Bool == true
                let message = result["message"] as? String ?? "无结果"
                DiagnosticReport.finish(runID, state: passed ? "passed" : "not-passed", result: result)
                record(String(format: "%@：%@；%@；采样峰值%.0f MB，最低可用%.0f MB；耗时%.3f秒。", title, passed ? "本路线门槛通过" : "未全部通过/未运行，请看分组数据", message, result["peakFootprintMB"] as? Double ?? 0, result["minAvailableMB"] as? Double ?? 0, result["wallSeconds"] as? Double ?? 0))
                completed += 1
            }
            status = "已返回 \(completed)/\(routes.count) 条路线\(stopRequested ? "，按要求停止" : "")；返回不等于通过。只需分享统一测试报告；默认配置未自动更改。"
            record(status)
        }
    }
    func stop() { stopRequested = true; status = "当前路线返回后停止；不会强杀正在运行的模型。" }
    private func record(_ message: String) {
        if let file = try? FileHandle(forWritingTo: Self.summaryURL) {
            defer { try? file.close() }
            try? file.seekToEnd()
            try? file.write(contentsOf: Data((message + "\n").utf8)); try? file.synchronize()
        }
    }
}
