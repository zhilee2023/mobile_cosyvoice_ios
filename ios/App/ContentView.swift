import SwiftUI
import AVFoundation

struct ContentView: View {
    @StateObject private var models = ModelStore()
    @StateObject private var player = ChunkPlayer()
    @State private var text = "你好，欢迎来到离线语音测试。"
    @State private var wavPlayer: AVAudioPlayer?
    @State private var probeResult = "尚未运行 MNN 探针"
    @State private var probeRunning = false
    @State private var memoryProbeResult = "尚未测试大型模型内存"
    @State private var memoryProbeRunning = false
    @State private var singleGenerationRunning = false
    @State private var optimizationProbeRunning = false
    @State private var optimizationProbeResult = "两项新试验请分别冷启动后运行；不自动切换默认配置。"
    @State private var hiftPairReady = false
    @StateObject private var sentenceStream = SentenceStream()
    @State private var optimizedSentences = true
    @State private var sentenceThreads = 4
    @State private var lowHiFT = true
    @State private var generationResult = "尚未运行手机模型生成"
    @State private var demoRunning = false
    @State private var firstChunkMilliseconds: Int?
    @State private var errorMessage: String?
    private var generationRunning: Bool { singleGenerationRunning || sentenceStream.isRunning || optimizationProbeRunning }

    var body: some View {
        NavigationStack {
            Form {
                Section("项目状态") {
                    Label("MNN / iPhone 15 · 端侧", systemImage: "iphone")
                    Text("版本 \(Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String ?? "未知")")
                        .font(.caption).foregroundStyle(.secondary)
                    Text("短句全链路已在 iPhone 验证。新增按句生成即播：模型仍逐段串行，播放可与下一句生成重叠；不是 token 级流式。")
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
                    Text("\(text.utf16.count) UTF-16 单元 · 单句入口≤40；句级长文入口≤2000，按标点/长度拆段，每段最多160语音token。")
                        .font(.caption).foregroundStyle(.secondary)
                    Button(generationRunning ? "生成中…" : "用当前文字生成 · CPU") { runGeneration(useText: true) }
                        .disabled(!models.ready || generationRunning || probeRunning || memoryProbeRunning || text.isEmpty)
                    Text("手机 LLM 先生成整句 token，释放后运行 Flow 与 HiFT。已验证5.96秒音频，耗时7.834秒、RTF1.314。")
                        .font(.caption).foregroundStyle(.secondary)
                }
                Section("句级生成与 RTF 对照 · 实验") {
                    Toggle("HiFT core Low（已试听选择）", isOn: $lowHiFT)
                        .disabled(generationRunning)
                    Text("只切换声码器core；Flow仍High两步，F0仍High。关闭可回到High对照；不修改模型文件。")
                        .font(.caption).foregroundStyle(.secondary)
                    Toggle("优化候选：缩小 Flow 补零尺寸 / DSP 查表", isOn: $optimizedSentences)
                        .disabled(generationRunning)
                    Picker("CPU 线程对照", selection: $sentenceThreads) {
                        Text("4 线程（已验证基线）").tag(4)
                        Text("2 线程（速度待测）").tag(2)
                    }.disabled(generationRunning)
                    Button("填入三句测试文字") {
                        text = "你好，欢迎来到离线语音测试。我们正在测试按句生成和播放。希望每一句都清晰自然。"
                    }.disabled(generationRunning)
                    Button(sentenceStream.isRunning ? "句级生成与播放中…" : "按句生成，首句完成即播放") {
                        wavPlayer?.stop()
                        sentenceStream.start(text: text, directory: models.directory.path,
                                             optimized: optimizedSentences, threads: sentenceThreads, player: player, lowHiFT: lowHiFT)
                    }.disabled(!models.ready || generationRunning || probeRunning || memoryProbeRunning || demoRunning)
                    Button("按句生成 · 常驻Flow384 · 新实验") {
                        wavPlayer?.stop()
                        sentenceStream.start(text: text, directory: models.directory.path,
                                             optimized: true, threads: 4, player: player, resident: true, lowHiFT: lowHiFT)
                    }.disabled(!models.ready || generationRunning || probeRunning || memoryProbeRunning || demoRunning)
                    Button("停止句级生成与播放") { sentenceStream.stop() }
                        .disabled(!sentenceStream.isRunning)
                    Text(sentenceStream.status).font(.caption).textSelection(.enabled)
                    Text("队列 \(player.queuedSeconds, specifier: "%.2f") 秒 · 已生成 \(sentenceStream.audioSeconds, specifier: "%.2f") 秒 · 队列断供观测 \(player.gapCount) 次")
                        .font(.caption)
                    Text("每句释放模型；最多保留一条待播句段。停止后当前短句推理会先返回再释放。RTF>1时仍可能句间等待，分句本身不保证实时。关闭优化并选4线程可对照旧链路；2线程需另测。")
                        .font(.caption).foregroundStyle(.secondary)
                    Text("新常驻入口固定4线程/DSP优化，HiFT精度由Low开关控制：本批只保留一个384帧Flow，每段最多105语音token/4.2秒；每句重做prefill，LLM销毁后才跑HiFT。超长拆句重做、不截断。结束/失败/停止均关闭Flow；最低余量800 MB，不足就停止，不自动降低门槛。完整链路内存与RTF仍需真机验证。")
                        .font(.caption).foregroundStyle(.secondary)
                    if FileManager.default.fileExists(atPath: SentenceStream.logURL.path) {
                        ShareLink("分享句级汇总日志", item: SentenceStream.logURL).disabled(generationRunning)
                        ShareLink("分享各句模型诊断", item: SentenceStream.documents.appendingPathComponent("last-sentence-diagnostics.log"))
                            .disabled(generationRunning)
                    }
                    if FileManager.default.fileExists(atPath: SentenceStream.documents.appendingPathComponent("last-resident-lifecycle.log").path) {
                        ShareLink("分享常驻Flow生命周期", item: SentenceStream.documents.appendingPathComponent("last-resident-lifecycle.log"))
                            .disabled(generationRunning)
                    }
                    if FileManager.default.fileExists(atPath: SentenceStream.audioURL.path) {
                        ShareLink("分享最近完整长文 WAV", item: SentenceStream.audioURL).disabled(generationRunning)
                    }
                }
                Section("真机探针") {
                    Button("HiFT High / Low 同输入对照 · 2.96秒") { runOptimizationProbe(resident: false) }
                        .disabled(!models.ready || generationRunning || probeRunning || memoryProbeRunning || demoRunning)
                    Button("Flow常驻＋LLM前向 · 内存与复用探针") { runOptimizationProbe(resident: true) }
                        .disabled(!models.ready || generationRunning || probeRunning || memoryProbeRunning || demoRunning)
                    Text(optimizationProbeResult).font(.caption).textSelection(.enabled)
                    Text("HiFT只改变core精度，F0/Flow仍High。共驻试验只有一套384帧Flow，LLM只生成2个token，不运行HiFT或长文；约20ms采样内存，低于保护门槛就停止。每项做完分享日志，再冷启动测下一项。")
                        .font(.caption).foregroundStyle(.secondary)
                    if FileManager.default.fileExists(atPath: CosyVoiceGenerate.hiftProbeLogPath()) {
                        ShareLink("分享HiFT精度对照日志", item: URL(fileURLWithPath: CosyVoiceGenerate.hiftProbeLogPath()))
                            .disabled(generationRunning)
                    }
                    if FileManager.default.fileExists(atPath: CosyVoiceGenerate.residentProbeLogPath()) {
                        ShareLink("分享Flow/LLM共驻日志", item: URL(fileURLWithPath: CosyVoiceGenerate.residentProbeLogPath()))
                            .disabled(generationRunning)
                    }
                    if hiftPairReady {
                        Button("试听HiFT High参考") { playProbeWav(low: false) }.disabled(generationRunning)
                        Button("试听HiFT Low候选") { playProbeWav(low: true) }.disabled(generationRunning)
                        ShareLink("分享High参考WAV", item: URL(fileURLWithPath: CosyVoiceGenerate.hiftReferenceWavPath()))
                        ShareLink("分享Low候选WAV", item: URL(fileURLWithPath: CosyVoiceGenerate.hiftCandidateWavPath()))
                    }
                    Button("探测 Conditioner MNN 会话") { runProbe() }
                        .disabled(!models.ready || probeRunning || memoryProbeRunning || generationRunning)
                    Text(probeResult).font(.caption).textSelection(.enabled)
                }
                Section("端侧生成试验 · 固定 speech token") {
                    Text("用 PC 文字模型预生成的 25 个 token，在本机运行 Conditioner → Flow CPU High 256 帧 → HiFT，生成约 1 秒 WAV。不会使用上方输入文字；内存不足会停止。")
                        .font(.caption).foregroundStyle(.secondary)
                    Button(generationRunning ? "生成与记录中…" : "生成 1 秒固定样例 · Flow CPU High") { runGeneration() }
                        .disabled(!models.ready || generationRunning || probeRunning || memoryProbeRunning)
                    Button("复现 CPU Low · 详细输出诊断") { runGeneration(lowPrecision: true) }
                        .disabled(!models.ready || generationRunning || probeRunning || memoryProbeRunning)
                }
                Section("生成结果") {
                    Text(generationResult).font(.caption).textSelection(.enabled)
                    let generationLog = CosyVoiceGenerate.generationLogPath()
                    if FileManager.default.fileExists(atPath: generationLog) {
                        ShareLink("分享生成日志", item: URL(fileURLWithPath: generationLog))
                    }
                    let llmLog = CosyVoiceGenerate.llmRuntimeLogPath()
                    if FileManager.default.fileExists(atPath: llmLog) {
                        ShareLink("分享 LLM 原始日志", item: URL(fileURLWithPath: llmLog))
                            .disabled(generationRunning)
                    }
                    let wavPath = CosyVoiceGenerate.generatedWavPath()
                    if FileManager.default.fileExists(atPath: wavPath) {
                        Button("播放最近成功的生成 WAV") { playGeneratedWav() }
                            .disabled(generationRunning)
                        Button("停止试听") { wavPlayer?.stop() }
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
                    Button("停止播放") { player.stop() }.disabled(generationRunning)
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

    private func runOptimizationProbe(resident: Bool) {
        let path = models.directory.path
        wavPlayer?.stop(); player.stop()
        optimizationProbeRunning = true
        if !resident { hiftPairReady = false }
        optimizationProbeResult = resident ? "正在测唯一Flow与LLM共驻；若退出请重开后分享日志…" : "正在准备同一份Mel/F0输入并运行HiFT High/Low…"
        DispatchQueue.global(qos: .userInitiated).async {
            let result = resident ? CosyVoiceGenerate.probeResidentFlow(atDirectory: path)
                : CosyVoiceGenerate.probeHiFTPrecision(atDirectory: path)
            DispatchQueue.main.async {
                optimizationProbeRunning = false
                if !resident { hiftPairReady = result["success"] as? Bool == true }
                optimizationProbeResult = String(format: "%@；采样峰值%.0f MB，最低可用%.0f MB。",
                    result["message"] as? String ?? "探针未返回结果", result["peakFootprintMB"] as? Double ?? 0,
                    result["minAvailableMB"] as? Double ?? 0)
            }
        }
    }

    private func playProbeWav(low: Bool) {
        do {
            player.stop()
            try AVAudioSession.sharedInstance().setCategory(.playback, mode: .default)
            try AVAudioSession.sharedInstance().setActive(true)
            wavPlayer = try AVAudioPlayer(contentsOf: URL(fileURLWithPath: low ? CosyVoiceGenerate.hiftCandidateWavPath() : CosyVoiceGenerate.hiftReferenceWavPath()))
            if wavPlayer?.play() != true { errorMessage = "无法播放对照WAV" }
        } catch { errorMessage = error.localizedDescription }
    }

    private func runGeneration(lowPrecision: Bool = false, useText: Bool = false) {
        let path = models.directory.path
        let inputText = text
        let useLowHiFT = lowHiFT
        wavPlayer?.stop()
        player.stop()
        singleGenerationRunning = true
        generationResult = "正在依次运行模型；若 App 退出，重开后分享生成日志…"
        DispatchQueue.global(qos: .userInitiated).async {
            let result = useText
                ? CosyVoiceGenerate.generateText(inputText, atDirectory: path, optimized: true, cpuThreads: 4, lowPrecisionHiFT: useLowHiFT)
                : CosyVoiceGenerate.runFixedTokenSample(atDirectory: path, lowPrecisionFlow: lowPrecision)
            DispatchQueue.main.async {
                singleGenerationRunning = false
                let message = result["message"] as? String ?? "生成未返回结果"
                let peak = result["peakFootprintMB"] as? Double ?? 0
                let available = result["minAvailableMB"] as? Double ?? 0
                generationResult = String(format: "%@；采样峰值 %.0f MB，最低可用 %.0f MB。", message, peak, available)
            }
        }
    }

    private func playGeneratedWav() {
        do {
            player.stop()
            try AVAudioSession.sharedInstance().setCategory(.playback, mode: .default)
            try AVAudioSession.sharedInstance().setActive(true)
            wavPlayer = try AVAudioPlayer(contentsOf: URL(fileURLWithPath: CosyVoiceGenerate.generatedWavPath()))
            if wavPlayer?.play() != true { errorMessage = "无法播放生成的 WAV" }
        } catch {
            errorMessage = error.localizedDescription
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
