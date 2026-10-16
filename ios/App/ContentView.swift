import SwiftUI

struct ContentView: View {
    @StateObject private var models = ModelStore()
    @StateObject private var player = ChunkPlayer()
    @State private var text = "你好，这是 CosyVoice 在 iPhone 上的端侧流式语音测试。"
    @State private var probeResult = "尚未运行 MNN 探针"
    @State private var probeRunning = false
    @State private var memoryProbeResult = "尚未测试大型模型内存"
    @State private var memoryProbeRunning = false
    @State private var demoRunning = false
    @State private var firstChunkMilliseconds: Int?
    @State private var errorMessage: String?

    var body: some View {
        NavigationStack {
            Form {
                Section("项目状态") {
                    Label("MNN / iPhone 15 · 端侧", systemImage: "iphone")
                    Text("完整 CosyVoice 推理尚未接通；当前可验证模型文件、MNN 会话和流式音频播放。")
                        .foregroundStyle(.secondary)
                }
                Section("模型") {
                    Text(models.status)
                    Text("查找位置：\(models.directory.path)")
                        .font(.caption).textSelection(.enabled)
                    Button("刷新模型文件") { models.refresh() }
                    if !models.missing.isEmpty {
                        Text("缺少：\(models.missing.prefix(3).joined(separator: "、"))\(models.missing.count > 3 ? "…" : "")")
                            .font(.caption).foregroundStyle(.secondary)
                    }
                }
                Section("输入文字") {
                    TextEditor(text: $text)
                        .frame(minHeight: 95)
                    Button("CosyVoice 流式生成（尚未接通）") {}
                        .disabled(true)
                    Text("只有 LLM 增量 token → Flow/HiFT 分块 → 播放全部接通后才启用。")
                        .font(.caption).foregroundStyle(.secondary)
                }
                Section("真机探针") {
                    Button("探测 Conditioner MNN 会话") { runProbe() }
                        .disabled(!models.ready || probeRunning || memoryProbeRunning)
                    Text(probeResult).font(.caption).textSelection(.enabled)
                }
                Section("大模型内存试验") {
                    Text("逐项运行，不同时常驻。只创建并调整会话形状，不运行前向；即使通过，也不能证明完整生成不会超内存。")
                        .font(.caption).foregroundStyle(.secondary)
                    Button("测 Flow · Metal High · 256 帧") { runMemoryProbe(flow: true) }
                        .disabled(!models.ready || probeRunning || memoryProbeRunning)
                    Button("测 HiFT core · CPU Low · 160 帧") { runMemoryProbe(flow: false) }
                        .disabled(!models.ready || probeRunning || memoryProbeRunning)
                    Text(memoryProbeResult).font(.caption).textSelection(.enabled)
                    let logPath = MNNProbe.memoryProbeLogPath()
                    if FileManager.default.fileExists(atPath: logPath) {
                        ShareLink("分享最近一次内存日志", item: URL(fileURLWithPath: logPath))
                    }
                    Text("若试验时 App 退出，重新打开后分享 last-memory-probe.log；日志保存在本 App 的文件共享区。")
                        .font(.caption).foregroundStyle(.secondary)
                }
                Section("分块播放") {
                    Button(demoRunning ? "播放自测中…" : "试听播放链路自测（非模型）") {
                        Task { await playDemoChunks() }
                    }
                    .disabled(demoRunning)
                    Button("停止播放") { player.stop() }
                    Text("\(player.status) · 队列 \(player.queuedSeconds, specifier: "%.2f") 秒")
                    if let firstChunkMilliseconds {
                        Text("自测首块入队：\(firstChunkMilliseconds) 毫秒；不等于模型首包延迟")
                            .font(.caption)
                    }
                }
                Section("文件导入说明") {
                    Text("在 Windows 解压约 1.4 GB 的模型 ZIP，把其中 17 个必需文件（包括 llm_config.json）通过 Apple Devices 的文件共享复制到 CosyVoiceMobile。不要导入 ZIP 本身。")
                        .font(.caption)
                }
            }
            .navigationTitle("CosyVoice Mobile")
            .alert("测试失败", isPresented: Binding(get: { errorMessage != nil }, set: { if !$0 { errorMessage = nil } })) {
                Button("确定", role: .cancel) { errorMessage = nil }
            } message: {
                Text(errorMessage ?? "未知错误")
            }
        }
    }

    private func runProbe() {
        let path = models.directory.path
        probeRunning = true
        probeResult = "正在创建 MNN CPU 会话…"
        DispatchQueue.global(qos: .userInitiated).async {
            let result = MNNProbe.probeConditioner(atDirectory: path)
            DispatchQueue.main.async {
                probeRunning = false
                if (result["success"] as? Bool) == true {
                    let load = result["loadSeconds"] as? Double ?? 0
                    let create = result["createSeconds"] as? Double ?? 0
                    let footprint = result["footprintMB"] as? Double ?? 0
                    let available = result["availableMB"] as? Double ?? 0
                    probeResult = String(format: "读取 %.3f s · 创建 %.3f s · footprint %.0f MB · available %.0f MB。尚未前向推理。", load, create, footprint, available)
                } else {
                    probeResult = result["message"] as? String ?? "MNN 探针失败"
                }
            }
        }
    }

    private func runMemoryProbe(flow: Bool) {
        let path = models.directory.path
        memoryProbeRunning = true
        memoryProbeResult = flow ? "正在创建 Flow Metal 会话…" : "正在创建 HiFT CPU 会话…"
        DispatchQueue.global(qos: .userInitiated).async {
            let result = flow
                ? MNNProbe.probeFlowMetalMemory(atDirectory: path)
                : MNNProbe.probeHiFTCPUMemory(atDirectory: path)
            DispatchQueue.main.async {
                memoryProbeRunning = false
                let peak = result["peakFootprintMB"] as? Double ?? 0
                let minimum = result["minAvailableMB"] as? Double ?? 0
                let message = result["message"] as? String ?? "内存试验未返回结果"
                memoryProbeResult = String(format: "%@；采样峰值 %.0f MB，最低可用 %.0f MB。详见日志。",
                                           message, peak, minimum)
            }
        }
    }

    @MainActor
    private func playDemoChunks() async {
        demoRunning = true
        defer { demoRunning = false }
        player.stop()
        let started = Date()
        for index in 0..<12 {
            if Task.isCancelled { break }
            let count = 2_880 // 120 ms at 24 kHz
            let samples: [Float] = (0..<count).map { frame in
                let time = Double(index * count + frame) / 24_000
                let envelope = min(1.0, min(Double(frame) / 240.0, Double(count - frame) / 240.0))
                return Float(0.08 * envelope * sin(2 * .pi * 440 * time))
            }
            do {
                try player.enqueue(samples)
            } catch {
                errorMessage = error.localizedDescription
                player.stop()
                break
            }
            if index == 0 {
                firstChunkMilliseconds = Int(Date().timeIntervalSince(started) * 1_000)
            }
            try? await Task.sleep(nanoseconds: 120_000_000)
        }
    }
}
