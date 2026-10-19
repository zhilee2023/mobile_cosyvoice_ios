import Foundation

@MainActor
enum DiagnosticReport {
    static let processID = UUID().uuidString
    static var lastError: String?
    static var archive: TestReportArchive {
        TestReportArchive(directory: FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0],
                          version: Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String ?? "unknown",
                          build: Bundle.main.object(forInfoDictionaryKey: "CFBundleVersion") as? String ?? "unknown", processID: processID)
    }
    static var url: URL { archive.url }
    static let generationSources = ["last-generation.log", "last-llm-runtime.log"]
    static let sentenceSources = ["last-sentence-stream.log", "last-sentence-diagnostics.log", "last-resident-lifecycle.log", "last-pipeline-lifecycle.log"] + generationSources
    static func prepare() {
        do {
            let routes = ["flow-buckets", "flow-threads", "flow-thread-4", "flow-thread-2", "flow-thread-3", "llm-reuse", "llm-threads", "hift-threads", "flow-bf16", "flow-metal-low-256", "flow-metal-low-384"]
            let oldFiles = sentenceSources + ["last-experiment-summary.log"] + routes.flatMap { ["last-experiment-\($0).json", "last-experiment-\($0).log"] }
            try archive.importLegacyOnce(sources: oldFiles)
            try archive.recoverInterrupted()
            lastError = nil
        } catch { lastError = error.localizedDescription }
    }
    static func begin(kind: String, label: String, parameters: [String: Any], sources: [String]) -> String? {
        do { return try archive.begin(kind: kind, label: label, parameters: parameters, sources: sources) }
        catch { lastError = error.localizedDescription; return nil }
    }
    static func finish(_ id: String?, state: String, result: [String: Any]) {
        guard let id else { return }
        do { try archive.finish(id: id, state: state, result: result) }
        catch { lastError = error.localizedDescription }
    }
}
