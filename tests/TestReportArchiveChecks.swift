import Foundation

@main
struct TestReportArchiveChecks {
    static func main() throws {
        func check(_ value: Bool) { precondition(value) }
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("cosyvoice-report-" + UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        var a = TestReportArchive(directory: root, version: "test", build: "15", processID: "launch-a")
        let log = root.appendingPathComponent("test.log")
        try Data("old".utf8).write(to: log)
        try FileManager.default.setAttributes([.modificationDate: Date(timeIntervalSince1970: 1)], ofItemAtPath: log.path)
        try a.importLegacyOnce(sources: ["test.log"])
        let first = try a.begin(kind: "module", label: "first", parameters: [:], sources: ["test.log", "../not-allowed"])
        try a.finish(id: first, state: "not-passed", result: ["success": false])
        let second = try a.begin(kind: "module", label: "second", parameters: [:], sources: ["test.log"])
        try Data("new-run-only".utf8).write(to: log)
        try a.finish(id: second, state: "passed", result: ["rows": [["finite": true]], "melL2": NSNull()])
        func runs(_ a: TestReportArchive) throws -> [[String: Any]] {
            let root = try JSONSerialization.jsonObject(with: Data(contentsOf: a.url)) as! [String: Any]
            return root["runs"] as! [[String: Any]]
        }
        let saved = try runs(a)
        precondition(saved.count == 3 && saved[0]["state"] as? String == "historical-files-not-correlated")
        let stale = saved[1]["sources"] as! [[String: Any]]
        precondition(stale[0]["state"] as? String == "stale-not-from-this-run" && stale[0]["text"] == nil)
        precondition(stale[1]["state"] as? String == "invalid-name")
        precondition((saved[2]["sources"] as! [[String: Any]])[0]["text"] as? String == "new-run-only")
        _ = try a.begin(kind: "module", label: "interrupted", parameters: [:], sources: ["test.log"])
        try a.recoverInterrupted() // Same process must NOT close an active test.
        check(try runs(a).last!["state"] as? String == "running")
        var b = TestReportArchive(directory: root, version: "test-next", build: "16", processID: "launch-b")
        try b.recoverInterrupted()
        check(try runs(b).last!["state"] as? String == "interrupted-not-completed")
        b.sourceLimit = 8
        let clipped = try b.begin(kind: "module", label: "clipped", parameters: [:], sources: ["test.log"])
        try Data(String(repeating: "歌词😀", count: 100).utf8).write(to: log)
        try b.finish(id: clipped, state: "passed", result: [:])
        check((try runs(b).last!["sources"] as! [[String: Any]])[0]["truncated"] as? Bool == true)
        a.maxRuns = 2
        let latest = try a.begin(kind: "module", label: "latest", parameters: [:], sources: [])
        try a.finish(id: latest, state: "passed", result: [:])
        check(try runs(a).count == 2)
        let prior = try Data(contentsOf: a.url)
        do { try a.finish(id: "does-not-exist", state: "passed", result: [:]); preconditionFailure("Mixed run IDs") } catch {}
        check(try Data(contentsOf: a.url) == prior)
        print("PASS: unified report history, run identity, stale-source rejection, interrupted recovery, UTF8 clipping and bounded retention")
    }
}
