import SwiftUI

struct ContentView: View {
    @StateObject private var models = ModelStore()
    @StateObject private var player = ChunkPlayer()
    @State private var text = "你好，这是 CosyVoice 在 iPhone 上的端侧流式语音测试。"
    @State private var probeResult = "尚未运行 MNN 探针"
    @State private var probeRunning = false
    @State private var memoryProbeResult = "尚未测试大型模型内存"
    @State private var memoryProbeRunning = false
    @State private var generationRunning = false
    @State private var generationResult = "尚未运行手机模型生成"
    @State private var demoRunning = false
    @State private var firstChunkMilliseconds: Int?
    @State private var errorMessage: String?

    var body: some View {
        NavigationStack {
            Form {
                Section("项目状态") {
                    Label("MNN / iPhone 15 · 端侧", systemImage: "iphone")
                    Text("可测固定 speech token → 1 秒端侧语音；手机文字 LLM 和连续流式生成尚未接通。")
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
                        .disabled(!models.ready || probeRunning || memoryProbeRunning || generationRunning)
                    Text(probeResult).font(.caption).textSelection(.enabled)
                }
                Section("端侧生成试验 · 固定 speech token") {
                    Text("用 PC 文字模型预生成的 25 个 token，在本机运行 Conditioner → Flow CPU Low 256 帧 → HiFT，生成约 1 秒 WAV。不会使用上方输入文字；内存不足会停止。")
                        .font(.caption).foregroundStyle(.secondary)
                    Button(generationRunning ? "生成与记录中…" : "生成 1 秒固定样例 · Flow CPU Low") { runGeneration() }
                        .disabled(!models.ready || generationRunning || probeRunning || memoryProbeRunning)
                    Text(generationResult).font(.caption).textSelection(.enabled)
                    let generationLog = CosyVoiceGenerate.generationLogPath()
                    if FileManager.default.fileExists(atPath: generationLog) {
                        ShareLink("分享生成日志", item: URL(fileURLWithPath: generationLog))
                    }
                    let wavPath = CosyVoiceGenerate.generatedWavPath()
                    if FileManager.default.fileExists(atPath: wavPath) {
                        ShareLink("分享最近成功的生成 WAV", item: URL(fileURLWithPath: wavPath))
                    }
                    Text("若 App 退出，重开后分享 last-generation.log。旧 WAV 仅代表最近一次成功生成，不代表本次成功。")
                        .font(.caption).foregroundStyle(.secondary)
                }
                Section("大模型内存试验") {
                    Text("逐项运行；每项结束后分享日志，再重开 App 测下一项。只创建并调整会话形状，不运行前向；通过不代表生成安全。")
                        .font(.caption).foregroundStyle(.secondary)
                    Button("测 Flow · Metal High · 256 帧（基线）") { runMemoryProbe(mode: 0) }
                        .disabled(!models.ready || probeRunning || memoryProbeRunning || generationRunning)
                    Button("测 Flow · Metal NoTune · 256 帧") { runMemoryProbe(mode: 1) }
                        .disabled(!models.ready || probeRunning || memoryProbeRunning || generationRunning)
                    Button("测 Flow · Metal Low NoTune · 256 帧") { runMemoryProbe(mode: 4) }
                        .disabled(!models.ready || probeRunning || memoryProbeRunning || generationRunning)
                    Button("测 Flow · CPU Low · 256 帧") { runMemoryProbe(mode: 2) }
                        .disabled(!models.ready || probeRunning || memoryProbeRunning || generationRunning)
                    Button("测 HiFT core · CPU Low · 160 帧") { runMemoryProbe(mode: 3) }
                        .disabled(!models.ready || probeRunning || memoryProbeRunning || generationRunning)
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
                    .disabled(demoRunning || generationRunning)
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

    private func runGeneration() {
        let path = models.directory.path
        generationRunning = true
        generationResult = "正在依次运行模型；若 App 退出，重开后分享生成日志…"
        DispatchQueue.global(qos: .userInitiated).async {
            let result = CosyVoiceGenerate.runFixedTokenSample(atDirectory: path)
            DispatchQueue.main.async {
                generationRunning = false
                let message = result["message"] as? String ?? "生成未返回结果"
                let peak = result["peakFootprintMB"] as? Double ?? 0
                let available = result["minAvailableMB"] as? Double ?? 0
                generationResult = String(format: "%@；采样峰值 %.0f MB，最低可用 %.0f MB。", message, peak, available)
            }
        }
    }

    private func runMemoryProbe(mode: Int) {
        let path = models.directory.path
        memoryProbeRunning = true
        memoryProbeResult = "正在创建模型会话；请保持 App 在前台…"
        DispatchQueue.global(qos: .userInitiated).async {
            let result: [String: Any]
            switch mode {
            case 1: result = MNNProbe.probeFlowMetalNoTuneMemory(atDirectory: path)
            case 2: result = MNNProbe.probeFlowCPULowMemory(atDirectory: path)
            case 3: result = MNNProbe.probeHiFTCPUMemory(atDirectory: path)
            case 4: result = MNNProbe.probeFlowMetalLowNoTuneMemory(atDirectory: path)
            default: result = MNNProbe.probeFlowMetalMemory(atDirectory: path)
            }
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
