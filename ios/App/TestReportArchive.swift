import Foundation

// Foundation-only, disk-backed history. Atomic saves; never delete source logs.
// Each run snapshots its own files before subsequent tests overwrite them.
struct TestReportArchive {
    let directory: URL
    let version: String
    let build: String
    let processID: String
    var maxRuns = 20
    var maxBytes = 8 * 1024 * 1024
    var sourceLimit = 256 * 1024
    var url: URL { directory.appendingPathComponent("cosyvoice-test-report.json") }

    private func read() throws -> [String: Any] {
        guard FileManager.default.fileExists(atPath: url.path) else {
            return ["schemaVersion": 1, "runs": [[String: Any]](), "prunedRuns": 0,
                    "scope": "Local tests; module timing is not whole-chain RTF. Historical runs retain their own version and IDs.",
                    "privacy": "Contains device/model diagnostics and possibly input text; no weights or audio."]
        }
        let data = try Data(contentsOf: url)
        guard let report = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              report["schemaVersion"] as? Int == 1, report["runs"] is [[String: Any]] else {
            throw NSError(domain: "TestReport", code: 1, userInfo: [NSLocalizedDescriptionKey: "统一报告无法解析，已保留原文件，未覆盖。"])
        }
        return report
    }
    private func save(_ report: [String: Any]) throws {
        var report = report
        var runs = report["runs"] as? [[String: Any]] ?? []
        var pruned = report["prunedRuns"] as? Int ?? 0
        while runs.count > maxRuns { runs.removeFirst(); pruned += 1 }
        report["updatedAt"] = ISO8601DateFormatter().string(from: Date())
        report["retention"] = "Latest \(maxRuns) runs, at most \(maxBytes) bytes; each source clipped at \(sourceLimit) bytes."
        while true {
            report["runs"] = runs; report["prunedRuns"] = pruned
            let data = try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
            if data.count <= maxBytes {
                try data.write(to: url, options: .atomic)
                return
            }
            guard runs.count > 1 else {
                throw NSError(domain: "TestReport", code: 2, userInfo: [NSLocalizedDescriptionKey: "单次报告超出大小限制，原报告仍保留。"])
            }
            runs.removeFirst(); pruned += 1
        }
    }
    @discardableResult
    func begin(kind: String, label: String, parameters: [String: Any], sources: [String]) throws -> String {
        var report = try read(), runs = report["runs"] as? [[String: Any]] ?? []
        let id = UUID().uuidString, now = Date()
        runs.append(["id": id, "kind": kind, "label": label, "parameters": parameters,
                     "processID": processID, "version": version, "build": build,
                     "os": ProcessInfo.processInfo.operatingSystemVersionString,
                     "startedAt": ISO8601DateFormatter().string(from: now), "startedAtEpoch": now.timeIntervalSince1970,
                     "startedUptime": ProcessInfo.processInfo.systemUptime,
                     "state": "running", "sourceNames": sources])
        report["runs"] = runs; try save(report)
        return id
    }
    private func snapshots(_ names: [String], after: Double?) -> [[String: Any]] {
        names.map { name in
            var item: [String: Any] = ["name": name]
            guard !name.isEmpty, !name.hasPrefix("."), !name.contains("/"), !name.contains("\\") else {
                item["state"] = "invalid-name"; return item
            }
            let source = directory.appendingPathComponent(name)
            do {
                let attributes = try FileManager.default.attributesOfItem(atPath: source.path)
                guard attributes[.type] as? FileAttributeType == .typeRegular,
                      let modified = attributes[.modificationDate] as? Date else {
                    item["state"] = "not-regular-file"; return item
                }
                item["modifiedAt"] = ISO8601DateFormatter().string(from: modified)
                if let after, modified.timeIntervalSince1970 < after {
                    item["state"] = "stale-not-from-this-run"; return item
                }
                let handle = try FileHandle(forReadingFrom: source)
                defer { try? handle.close() }
                let data = try handle.read(upToCount: sourceLimit + 1) ?? Data()
                item["text"] = String(decoding: data.prefix(sourceLimit), as: UTF8.self)
                item["truncated"] = data.count > sourceLimit
                item["fileBytes"] = attributes[.size]
                item["state"] = "captured"
            } catch { item["state"] = "missing-or-unreadable"; item["error"] = error.localizedDescription }
            return item
        }
    }
    func finish(id: String, state: String, result: [String: Any]) throws {
        var report = try read(), runs = report["runs"] as? [[String: Any]] ?? []
        guard let index = runs.firstIndex(where: { $0["id"] as? String == id }) else {
            throw NSError(domain: "TestReport", code: 3, userInfo: [NSLocalizedDescriptionKey: "找不到本次测试编号，未把结果归入其他测试。"])
        }
        let start = runs[index]["startedAtEpoch"] as? Double
        runs[index]["sources"] = snapshots(runs[index]["sourceNames"] as? [String] ?? [], after: start)
        runs[index]["state"] = state
        runs[index]["result"] = result
        runs[index]["finishedAt"] = ISO8601DateFormatter().string(from: Date())
        if runs[index]["processID"] as? String == processID,
           let uptime = runs[index]["startedUptime"] as? Double {
            runs[index]["elapsedWallSeconds"] = ProcessInfo.processInfo.systemUptime - uptime
        }
        report["runs"] = runs; try save(report)
    }
    func recoverInterrupted() throws {
        var report = try read(), runs = report["runs"] as? [[String: Any]] ?? []
        for index in runs.indices where runs[index]["state"] as? String == "running" && runs[index]["processID"] as? String != processID {
            runs[index]["state"] = "interrupted-not-completed"
            runs[index]["sources"] = snapshots(runs[index]["sourceNames"] as? [String] ?? [], after: runs[index]["startedAtEpoch"] as? Double)
            runs[index]["recoveredAt"] = ISO8601DateFormatter().string(from: Date())
        }
        report["runs"] = runs; try save(report)
    }
    func importLegacyOnce(sources: [String]) throws {
        guard !FileManager.default.fileExists(atPath: url.path) else { return }
        var legacy = self
        legacy.sourceLimit = min(sourceLimit, 32768)
        let captured = legacy.snapshots(sources, after: nil).filter { $0["state"] as? String == "captured" }
        var report = try read()
        if !captured.isEmpty {
            report["runs"] = [["id": UUID().uuidString, "kind": "legacy-import",
                "state": "historical-files-not-correlated", "version": "unknown-original-version",
                "sources": captured, "note": "Imported once. These files may come from DIFFERENT runs. Do not treat them as a new test."]]
        }
        try save(report)
    }
}
