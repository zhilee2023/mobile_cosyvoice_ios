import Foundation

private struct ModelManifest: Decodable {
    struct FileSpec: Decodable {
        let name: String
        let bytes: Int64
    }
    let files: [FileSpec]
}

@MainActor
final class ModelStore: ObservableObject {
    @Published private(set) var found = 0
    @Published private(set) var total = 0
    @Published private(set) var missing: [String] = []
    @Published private(set) var mismatched: [String] = []
    @Published private(set) var directory: URL
    @Published private(set) var status = "检查中"

    var ready: Bool { total > 0 && found == total && mismatched.isEmpty }

    init() {
        directory = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        refresh()
    }

    func refresh() {
        guard let url = Bundle.main.url(forResource: "model-manifest", withExtension: "json"),
              let data = try? Data(contentsOf: url),
              let manifest = try? JSONDecoder().decode(ModelManifest.self, from: data) else {
            status = "App 缺少 model-manifest.json"
            return
        }
        let documents = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let candidates = [
            documents,
            documents.appendingPathComponent("model", isDirectory: true),
            documents.appendingPathComponent("Models/CosyVoice3", isDirectory: true),
        ]
        let manager = FileManager.default
        directory = candidates.max { left, right in
            let leftCount = manifest.files.filter { manager.fileExists(atPath: left.appendingPathComponent($0.name).path) }.count
            let rightCount = manifest.files.filter { manager.fileExists(atPath: right.appendingPathComponent($0.name).path) }.count
            return leftCount < rightCount
        } ?? documents
        total = manifest.files.count
        missing = []
        mismatched = []
        for spec in manifest.files {
            let file = directory.appendingPathComponent(spec.name)
            guard let attributes = try? manager.attributesOfItem(atPath: file.path),
                  let size = attributes[.size] as? NSNumber else {
                missing.append(spec.name)
                continue
            }
            if size.int64Value != spec.bytes {
                mismatched.append(spec.name)
            }
        }
        found = total - missing.count - mismatched.count
        status = ready ? "\(total)/\(total) 文件大小匹配；尚未完成真机推理验证" : "已识别 \(found)/\(total)；缺失 \(missing.count)，大小不符 \(mismatched.count)"
    }
}
