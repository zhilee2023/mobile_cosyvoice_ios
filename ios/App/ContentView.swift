import SwiftUI
import AVFoundation

struct ContentView: View {
    @StateObject private var models = ModelStore()
    @StateObject private var player = ChunkPlayer()
    @StateObject private var sentences = SentenceStream()
    @StateObject private var experiments = ExperimentRunner()
    @State private var text = LongTextBenchmarks.fiveSentenceText
    private let threads = ExperimentCatalog.defaultThreads
    @State private var lowHiFT = true
    @State private var useW8Flow = true
    @State private var playbackMode: PlaybackMode = .buffered
    @State private var singleRunning = false
    @State private var generationResult = "尚未生成"
    @State private var wavPlayer: AVAudioPlayer?
    @State private var errorMessage: String?
    @State private var reportStatus = ""
    private var busy: Bool { singleRunning || sentences.isRunning || experiments.isRunning }
    private var isSingleSentence: Bool {
        guard !text.isEmpty, text.utf16.count <= 40 else { return false }
        return (try? SentencePlanner.plan(text, limit: 40).count) == 1
    }

    var body: some View {
        NavigationStack {
            Form {
                Section("端侧语音") {
                    Label("CosyVoice · MNN · iPhone", systemImage: "iphone")
                    Text("v\(Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String ?? "未知") · build \(Bundle.main.object(forInfoDictionaryKey: "CFBundleVersion") as? String ?? "未知")")
                        .font(.caption).foregroundStyle(.secondary)
                    Text("五句短测试，可先预加载三句再播，或全部生成后连续播放；不是句内 token 级流式。")
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
                    Button("载入五句测试文本") { text = LongTextBenchmarks.fiveSentenceText }.disabled(busy)
                    Picker("播放方式", selection: $playbackMode) {
                        ForEach(PlaybackMode.allCases) { mode in Text(mode.title).tag(mode) }
                    }.disabled(busy)
                    Text("本轮只测五句。预加载三句会延后首播，边播边生成后两句；全部生成后播放则不受后续推理供给影响。生成RTF不会因此自动下降。")
                        .font(.caption).foregroundStyle(.secondary)
                    Text("长文≤2000 UTF-16单元；自动拆句，不截断冒充读完。单句按钮≤40单元。")
                        .font(.caption).foregroundStyle(.secondary)
                    Toggle("HiFT core Low", isOn: $lowHiFT).disabled(busy)
                    Toggle("Flow FFN W8 · 低内存整数", isOn: $useW8Flow).disabled(busy)
                    Text("W8已完成五条真机Flow探针；本轮接入真实LLM→Conditioner→W8 Flow→HiFT。默认不裁剪输出头；关闭W8开关可用原版对照。缺少候选文件会明确停止，不偷偷回退。")
                        .font(.caption).foregroundStyle(.secondary)
                    Text("已试听选择Low；F0、Flow仍High，Flow两步。关闭可对照High。")
                        .font(.caption).foregroundStyle(.secondary)
                    Text("当前配置：LLM / Flow / HiFT 均4线程；Flow CPU High两步，F0 High。")
                        .font(.caption).foregroundStyle(.secondary)
                    Button("开始生成与播放") { startSentences() }
                        .disabled(!models.ready || busy || text.isEmpty)
                    Button("只生成当前短句 WAV") { generateSingle() }
                        .disabled(!models.ready || busy || !isSingleSentence)
                    Text("短句WAV仅接受一条句子；多句使用按句生成。超出speech token上限会提示拆短，不强制截断。")
                        .font(.caption).foregroundStyle(.secondary)
                    Button("停止句级生成") { sentences.stop() }.disabled(!sentences.isRunning)
                    Text(sentences.status).font(.caption).textSelection(.enabled)
                    Text(sentences.playbackStatus).font(.caption).foregroundStyle(.secondary)
                    Text("已生成 \(sentences.audioSeconds, specifier: "%.2f") 秒 · 播放队列 \(player.queuedSeconds, specifier: "%.2f") 秒 · 断供观测 \(player.gapCount) 次")
                        .font(.caption)
                    Text("预加载写磁盘：最多6条句段/8MiB；播放内存队列仍目标≤2秒、硬上限4秒。全部生成模式只保存合并WAV。停止会等待当前推理返回，不强杀模型。")
                        .font(.caption).foregroundStyle(.secondary)
                    Text(generationResult).font(.caption).textSelection(.enabled)
                }
                if !ExperimentRunner.routes.isEmpty {
                  Section("待验证实验 · 分模块测试") {
                    ForEach(ExperimentRunner.routes, id: \.id) { route in
                        Button(route.title) { startExperiments([route.id]) }
                            .disabled(!models.ready || busy)
                    }
                    Button("当前路线返回后停止") { experiments.stop() }
                        .disabled(!experiments.isRunning)
                    Text(experiments.status).font(.caption).textSelection(.enabled)
                    Text("④先做CPU热点并保存384帧参考，冷启动后再做⑤Metal High。复用已导入的W8模型和flow-w8-fixtures.f32.bin，不需新权重；每项只有一个Flow，无LLM/HiFT。")
                        .font(.caption).foregroundStyle(.secondary)
                    Text("CPU热点只收集标量计时，不读中间大张量。回调耗时包含诊断开销，不作为生产速度排名；普通两步另计。所有结果自动汇入同一份报告。")
                        .font(.caption).foregroundStyle(.secondary)
                    Text("Metal用High避免重复旧Low溢出路线，但内存仍可能不够。先验有限值、CPU偏差和模块速度，通过后才能考虑整链；不会自动切换默认CPU。")
                        .font(.caption).foregroundStyle(.secondary)
                  }
                }
                Section("新整链对照 · 待真机验收") {
                    ForEach(ExperimentCatalog.available.filter { $0.execution == .sentence }, id: \.id) { route in
                        Button(route.title) { startSentences(mode: route.id) }
                            .disabled(!models.ready || busy || text.isEmpty)
                        Text(route.conclusion).font(.caption).foregroundStyle(.secondary)
                    }
                    ForEach(ExperimentCatalog.planned, id: \.id) { route in
                        Text(route.title)
                        Text(route.conclusion).font(.caption).foregroundStyle(.secondary)
                    }
                    Text("生成仍用完整EOS、Flow High两步与HiFT Low；800MB底线不变。预加载只是播放策略，不开启全模型常驻。")
                        .font(.caption).foregroundStyle(.secondary)
                    Text("A/B/C自动用相同五句、全部生成后播放；A/B逐句独立LLM，C三句批内复用，每句仍清KV/重置seed。比较B和C才能判断复用收益。原模型、两个约208MB裁剪候选文件全部保留。")
                        .font(.caption).foregroundStyle(.secondary)
                }
                Section("统一测试报告 · 只发这一个") {
                    ShareLink("分享统一测试报告", item: DiagnosticReport.url).disabled(busy || !exists(DiagnosticReport.url))
                    Text(reportStatus).font(.caption).textSelection(.enabled)
                    Text("自动汇总模块数据、日志和完整生成记录，含版本/测试编号/状态；跨次测试保留最近20次、总计≤8 MB。旧文件明确标成历史，不冒充新测试；中断后重开可收集本轮已写日志。报告可能含输入文字，请注意隐私。不含模型或音频。")
                        .font(.caption).foregroundStyle(.secondary)
                }
                Section("播放与日志") {
                    ForEach([("end-to-end-original.wav", "原版整链"), ("end-to-end-flow-w8.wav", "Flow W8整链"), ("end-to-end-head-full.wav", "输出头配对A · 完整头"), ("end-to-end-head-reduced.wav", "输出头配对B · 裁剪头"), ("end-to-end-head-reuse.wav", "输出头配对C · 批内复用")], id: \.0) { item in
                        let url = SentenceStream.documents.appendingPathComponent(item.0)
                        if exists(url) {
                            Button("试听最近成功：\(item.1)") { play(url) }.disabled(busy)
                            ShareLink("分享音频：\(item.1)", item: url).disabled(busy)
                        }
                    }
                    if exists(URL(fileURLWithPath: CosyVoiceGenerate.generatedWavPath())) {
                        Button("试听最近成功短句") { play(URL(fileURLWithPath: CosyVoiceGenerate.generatedWavPath())) }.disabled(busy)
                        ShareLink("分享最近短句 WAV", item: URL(fileURLWithPath: CosyVoiceGenerate.generatedWavPath())).disabled(busy)
                    }
                    if exists(SentenceStream.audioURL) {
                        Button("试听最近完整长文") { play(SentenceStream.audioURL) }.disabled(busy)
                        ShareLink("分享最近完整长文", item: SentenceStream.audioURL).disabled(busy)
                    }
                    Button("停止试听") { wavPlayer?.stop(); if !busy { player.stop() } }
                    Text("旧WAV只代表最近成功记录。失败时partial保留，不标成完整长文；只需分享一个统一测试报告核对实际路线和RTF。")
                        .font(.caption).foregroundStyle(.secondary)
                }
                Section("已归档，不再占用测试按钮") {
                    DisclosureGroup("查看已有结论（无需重测）") {
                        ForEach(ExperimentCatalog.archived, id: \.id) { route in
                            Text(route.title).font(.subheadline)
                            Text(route.conclusion).font(.caption).foregroundStyle(.secondary)
                        }
                    }
                    Text("已确认的实验只留结论，不留运行按钮；后续只显示已实现、待验证的新路线。一次通过不自动代表音质或长文验收。历史报告与原实现保留，原有20次/8 MB保留策略不变。")
                        .font(.caption).foregroundStyle(.secondary)
                    Text("更早的加载/固定样例、Metal/NoTune、Flow Low、HiFT精度、短KV共驻探针继续归档。")
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
    private func startSentences(mode: String? = nil) {
        wavPlayer?.stop()
        let actualMode = mode ?? (useW8Flow ? "three-sentence-staged-w8" : "three-sentence-staged")
        let headPair = LongTextBenchmarks.headMode(actualMode) != nil
        sentences.start(text: headPair ? LongTextBenchmarks.fiveSentenceText : text, directory: models.directory.path, optimized: true,
                        threads: threads, player: player, resident: false, lowHiFT: lowHiFT,
                        experimentMode: actualMode, playbackMode: headPair ? .complete : playbackMode)
    }
    private func startExperiments(_ routes: [String]) {
        wavPlayer?.stop(); player.stop()
        experiments.start(routes: routes, directory: models.directory.path)
    }
    private func generateSingle() {
        let directory = models.directory.path, input = text, threadCount = threads, low = lowHiFT, w8 = useW8Flow
        wavPlayer?.stop(); player.stop(); singleRunning = true
        generationResult = "正在生成；请保持前台，失败后分享日志。"
        let runID = DiagnosticReport.begin(kind: "single-generation", label: "当前短句生成",
            parameters: ["text": input, "threads": threadCount, "hiFTLow": low, "flowW8": w8, "llmHeadPruned": false, "optimized": true], sources: DiagnosticReport.generationSources)
        DispatchQueue.global(qos: .userInitiated).async {
            let result = CosyVoiceGenerate.generateText(input, atDirectory: directory, optimized: true, cpuThreads: threadCount, lowPrecisionHiFT: low, flowW8: w8)
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
