import SwiftUI
import AVFoundation

struct ContentView: View {
    @StateObject private var models = ModelStore()
    @StateObject private var player = ChunkPlayer()
    @StateObject private var sentences = SentenceStream()
    @StateObject private var experiments = ExperimentRunner()
    @State private var text = "你好，欢迎来到离线语音测试。我们正在测试按句生成和播放。希望每一句都清晰自然。"
    @State private var threads = 4
    @State private var lowHiFT = true
    @State private var singleRunning = false
    @State private var generationResult = "尚未生成"
    @State private var wavPlayer: AVAudioPlayer?
    @State private var errorMessage: String?
    @State private var reportStatus = ""
    private var busy: Bool { singleRunning || sentences.isRunning || experiments.isRunning }

    var body: some View {
        NavigationStack {
            Form {
                Section("端侧语音") {
                    Label("CosyVoice · MNN · iPhone", systemImage: "iphone")
                    Text("v\(Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String ?? "未知") · build \(Bundle.main.object(forInfoDictionaryKey: "CFBundleVersion") as? String ?? "未知")")
                        .font(.caption).foregroundStyle(.secondary)
                    Text("按句生成、首句完成即播放。每句仍先生成 token 再解码，不是句内 token 级流式。")
                        .font(.caption).foregroundStyle(.secondary)
                }
                Section("模型") {
                    Text(models.status)
                    Button("刷新文件") { models.refresh() }.disabled(busy)
                    if !models.missing.isEmpty {
                        Text("缺少：\(models.missing.prefix(3).joined(separator: "、"))")
                            .font(.caption).foregroundStyle(.secondary)
                    }
                }
                Section("输入与生成") {
                    TextEditor(text: $text).frame(minHeight: 110).disabled(busy)
                    Text("长文≤2000 UTF-16单元；自动拆句，不截断冒充读完。单句按钮≤40单元。")
                        .font(.caption).foregroundStyle(.secondary)
                    Toggle("HiFT core Low", isOn: $lowHiFT).disabled(busy)
                    Text("已试听选择Low；F0、Flow仍High，Flow两步。关闭可对照High。")
                        .font(.caption).foregroundStyle(.secondary)
                    Picker("串行 CPU 线程", selection: $threads) {
                        Text("2").tag(2); Text("3").tag(3); Text("4 · 基线").tag(4)
                    }.disabled(busy)
                    Button("按句生成与播放 · 串行基线") { startSentences(adaptive: false) }
                        .disabled(!models.ready || busy || text.isEmpty)
                    Button("按句生成 · 自适应常驻 / 串行回退") { startSentences(adaptive: true) }
                        .disabled(!models.ready || busy || text.isEmpty)
                    Text("自适应入口固定4线程：最多保留一套384帧Flow；LLM加载前余量不足1800 MB就释放Flow，本批剩余句改串行。800 MB底线不变。初始化、等待和失败重试均计入耗时，是否更快需实测。")
                        .font(.caption).foregroundStyle(.secondary)
                    Button("只生成当前短句 WAV") { generateSingle() }
                        .disabled(!models.ready || busy || text.isEmpty || text.utf16.count > 40)
                    Button("停止句级生成") { sentences.stop() }.disabled(!sentences.isRunning)
                    Text(sentences.status).font(.caption).textSelection(.enabled)
                    Text("已生成 \(sentences.audioSeconds, specifier: "%.2f") 秒 · 播放队列 \(player.queuedSeconds, specifier: "%.2f") 秒 · 断供观测 \(player.gapCount) 次")
                        .font(.caption)
                    Text("队列目标≤2秒、硬上限4秒；至多一条待播句段。停止会等待当前短句返回后释放，不强杀模型。")
                        .font(.caption).foregroundStyle(.secondary)
                    Text(generationResult).font(.caption).textSelection(.enabled)
                }
                Section("加速实验室 · 分模块测试") {
                    Text("固定同输入对照；每组只保留一个声学会话，记录首次 / 预热、偏差、20ms内存采样。结果不自动替换默认配置，也不是整链RTF。建议分别冷启动测试；连续运行若内存未恢复会记录未运行。")
                        .font(.caption).foregroundStyle(.secondary)
                    ForEach(ExperimentRunner.routes, id: \.id) { route in
                        Button(route.title) { startExperiments([route.id]) }
                            .disabled(!models.ready || busy)
                    }
                    Text("Flow线程请先单独测②a，再分别冷启动测②b／②c，避免上一组残留拦住下一组。统一报告会保留三组实际Mel供跨组核对；单组通过不等于线程间已一致。")
                        .font(.caption).foregroundStyle(.secondary)
                    Button("依次运行其他路线（不含Flow独立线程）") {
                        startExperiments(ExperimentRunner.routes.filter { !$0.id.hasPrefix("flow-thread-") }.map(\.id))
                    }
                        .disabled(!models.ready || busy)
                    Button("当前路线返回后停止") { experiments.stop() }
                        .disabled(!experiments.isRunning)
                    Text(experiments.status).font(.caption).textSelection(.enabled)
                    Text("BF16是隔离候选，不能保证支持、提速或无NaN；报告区分精度请求与主后端，不声称全图BF16。LLM复用只测实例复用＋逐句reset KV，不保留跨句KV，也未让所有模型同时常驻。")
                        .font(.caption).foregroundStyle(.secondary)
                    Text("W8：待补原始学生模型/导出链；当前没有可验收的Flow W8文件。不会把改文件名或FP16重新转换算成W8。")
                        .font(.caption).foregroundStyle(.secondary)
                }
                Section("统一测试报告 · 只发这一个") {
                    ShareLink("分享统一测试报告", item: DiagnosticReport.url).disabled(busy || !exists(DiagnosticReport.url))
                    Text(reportStatus).font(.caption).textSelection(.enabled)
                    Text("自动汇总模块数据、日志和完整生成记录，含版本/测试编号/状态；跨次测试保留最近20次、总计≤8 MB。旧文件明确标成历史，不冒充新测试；中断后重开可收集本轮已写日志。报告可能含输入文字，请注意隐私。不含模型或音频。")
                        .font(.caption).foregroundStyle(.secondary)
                }
                Section("播放与日志") {
                    if exists(URL(fileURLWithPath: CosyVoiceGenerate.generatedWavPath())) {
                        Button("试听最近成功短句") { play(URL(fileURLWithPath: CosyVoiceGenerate.generatedWavPath())) }.disabled(busy)
                        ShareLink("分享最近短句 WAV", item: URL(fileURLWithPath: CosyVoiceGenerate.generatedWavPath())).disabled(busy)
                    }
                    if exists(SentenceStream.audioURL) {
                        Button("试听最近完整长文") { play(SentenceStream.audioURL) }.disabled(busy)
                        ShareLink("分享最近完整长文", item: SentenceStream.audioURL).disabled(busy)
                    }
                    Button("停止试听") { wavPlayer?.stop(); if !busy { player.stop() } }
                    Text("旧WAV只代表最近成功记录。失败时partial保留，但不标成完整长文；分享汇总与各句诊断才能核对实际路线和RTF。")
                        .font(.caption).foregroundStyle(.secondary)
                }
                Section("已归档，不再占用测试按钮") {
                    Text("Conditioner加载、固定1秒样例、播放音、Metal内存/NoTune、Flow CPU Low NaN、HiFT High/Low、两token共驻探针已移除入口。原实现和日志保留，不再重复运行已有结论的项目。")
                        .font(.caption).foregroundStyle(.secondary)
                    Text("17个模型文件不变，ACE项目未改动。使用Windows文件共享导入解压后的文件，不导入ZIP。")
                        .font(.caption).foregroundStyle(.secondary)
                }
            }
            .navigationTitle("CosyVoice Mobile")
            .onAppear {
                DiagnosticReport.prepare()
                reportStatus = DiagnosticReport.lastError ?? "报告已就绪；测完不同模块后，只需分享同一个JSON。"
            }
            .onChange(of: busy) { _, running in
                if !running { reportStatus = DiagnosticReport.lastError ?? "结果已汇入统一报告，无需分别发日志和数据。" }
            }
            .alert("提示", isPresented: Binding(get: { errorMessage != nil }, set: { if !$0 { errorMessage = nil } })) {
                Button("确定", role: .cancel) { errorMessage = nil }
            } message: { Text(errorMessage ?? "未知错误") }
        }
    }
    private func exists(_ url: URL) -> Bool { FileManager.default.fileExists(atPath: url.path) }
    private func startSentences(adaptive: Bool) {
        wavPlayer?.stop()
        sentences.start(text: text, directory: models.directory.path, optimized: true,
                        threads: adaptive ? 4 : threads, player: player, resident: adaptive, lowHiFT: lowHiFT)
    }
    private func startExperiments(_ routes: [String]) {
        wavPlayer?.stop(); player.stop()
        experiments.start(routes: routes, directory: models.directory.path)
    }
    private func generateSingle() {
        let directory = models.directory.path, input = text, threadCount = threads, low = lowHiFT
        wavPlayer?.stop(); player.stop(); singleRunning = true
        generationResult = "正在生成；请保持前台，失败后分享日志。"
        let runID = DiagnosticReport.begin(kind: "single-generation", label: "当前短句生成",
            parameters: ["text": input, "threads": threadCount, "hiFTLow": low, "optimized": true], sources: DiagnosticReport.generationSources)
        DispatchQueue.global(qos: .userInitiated).async {
            let result = CosyVoiceGenerate.generateText(input, atDirectory: directory, optimized: true, cpuThreads: threadCount, lowPrecisionHiFT: low)
            DispatchQueue.main.async {
                DiagnosticReport.finish(runID, state: result["success"] as? Bool == true ? "completed" : "not-completed", result: result)
                singleRunning = false
                generationResult = String(format: "%@；峰值 %.0f MB，最低可用 %.0f MB。", result["message"] as? String ?? "未返回结果", result["peakFootprintMB"] as? Double ?? 0, result["minAvailableMB"] as? Double ?? 0)
            }
        }
    }
    private func play(_ url: URL) {
        do {
            player.stop()
            try AVAudioSession.sharedInstance().setCategory(.playback, mode: .default)
            try AVAudioSession.sharedInstance().setActive(true)
            wavPlayer = try AVAudioPlayer(contentsOf: url)
            if wavPlayer?.play() != true { errorMessage = "无法播放WAV" }
        } catch { errorMessage = error.localizedDescription }
    }
}
