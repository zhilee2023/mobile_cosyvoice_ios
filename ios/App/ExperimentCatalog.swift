import Foundation

/// Review decisions, not a cache of the last pass/fail flag. A finite short
/// forward must never automatically approve production or long-text residency.
enum ExperimentCatalog {
    enum State { case available, planned, archived }
    enum Execution { case module, sentence }
    struct Route {
        let id: String
        let title: String
        let state: State
        let conclusion: String
        var execution: Execution = .module
    }
    static let routes: [Route] = [
        .init(id: "flow-w8-reference", title: "① W8测试 · 原版五条参考", state: .available,
              conclusion: "先导入W8包三个文件。五条独立冻结输入，CPU High/Memory Normal；保存匹配的真机Mel参考。只测Flow，不运行LLM或HiFT。"),
        .init(id: "flow-w8-normal", title: "② FFN W8/block64 · 普通内存", state: .available,
              conclusion: "完成①后冷启动。只量化44个FFN线性层，High精度/Memory Normal；五条输入首次/预热两步与独立CPU参考对照。非整链RTF。"),
        .init(id: "flow-w8-low", title: "③ FFN W8/block64 · 低内存整数", state: .available,
              conclusion: "完成①后冷启动。仍是High精度，Memory Low动态整数路径；不是之前NaN的Precision Low。20ms内存采样底线800MB，无自动切换。"),
        .init(id: "flow-buckets", title: "Flow 帧数对照", state: .archived,
              conclusion: "32帧对齐已用于生成；缩小补零能提速，不能显著降低权重内存。"),
        .init(id: "flow-thread-4", title: "Flow 4线程", state: .archived,
              conclusion: "本轮预热两步1.810秒；作为当前基线。"),
        .init(id: "flow-thread-2", title: "Flow 2线程", state: .archived,
              conclusion: "2.219秒；Mel相对L2约2.73e-5，内存无明显收益。"),
        .init(id: "flow-thread-3", title: "Flow 3线程", state: .archived,
              conclusion: "2.385秒；Mel相对L2约2.74e-5，内存无明显收益。"),
        .init(id: "llm-reuse", title: "LLM 独立复用微基准", state: .archived,
              conclusion: "旧独立探针支持逐句清空KV后复用；不等于完整链路共驻已通过。"),
        .init(id: "llm-threads", title: "LLM 线程对照", state: .archived,
              conclusion: "已有独立探针未显示减线程的整体收益，当前保留4线程。"),
        .init(id: "hift-threads", title: "HiFT Low 线程对照", state: .archived,
              conclusion: "多轮4线程最快；最新预热约0.359秒。Low已试听选择。"),
        .init(id: "flow-bf16", title: "Flow 整图 BF16", state: .archived,
              conclusion: "重复在第一步出现NaN；不采用，不重复同一整图测试。"),
        .init(id: "adaptive-resident", title: "Flow384自适应常驻", state: .archived,
              conclusion: "现有整链经常首句前就释放Flow回退，额外初始化未换来收益。"),
        .init(id: "llm-chain-reuse", title: "整链 LLM 复用", state: .archived,
              conclusion: "v0.7.3真机三句完成，计算RTF 1.756；峰值2159 MB、首块入队5.48秒，未证明比串行更快。", execution: .sentence),
        .init(id: "two-sentence-staged", title: "两句分阶段／唯一 Flow", state: .archived,
              conclusion: "v0.7.3真机两次完成，计算RTF 1.744/1.668，峰值约1.61 GB；每三句仍创建两次Flow，首块约6.5秒。", execution: .sentence),
        .init(id: "three-sentence-staged", title: "CPU三句批量／当前播放策略", state: .available,
              conclusion: "build23已完成69.52秒、RTF1.789、最低余量1605MB；首句即播有23次断供。本轮固定五句，只比较播放缓冲策略。", execution: .sentence),
        .init(id: "six-sentence-staged", title: "六句一批／长文摊销对照", state: .archived,
              conclusion: "build23完成67.96秒、RTF1.835、首块12.28秒、21次断供；未证明比三句更快，连续热态测试非严格速度排名。保留实现，不再占测试按钮。", execution: .sentence),
        .init(id: "flow-metal-low-memory-256", title: "Metal Low 256帧会话内存", state: .archived,
              conclusion: "v0.7.5两次主后端Metal，峰值2062–2092MB、最低余量1129–1159MB；仅resize通过，尚无GPU前向结果。"),
        .init(id: "flow-metal-low-256", title: "GPU 256帧两步Flow", state: .archived,
              conclusion: "v0.7.6第一步17920/20480个值NaN，有效224帧均受影响；最低余量1115MB，属于数值失败。"),
        .init(id: "flow-metal-low-384", title: "GPU 384帧两步Flow", state: .archived,
              conclusion: "v0.7.6第一步25120/30720个值NaN，有效314帧均受影响；最低余量1106MB，未形成有效GPU测速。"),
        .init(id: "flow-metal-nan", title: "GPU 首个 NaN 算子定位", state: .archived,
              conclusion: "v0.7.7第三个Transformer块QK乘法有限输入→大量Inf，随后Softmax首次NaN；定位完成，最低余量1108MB。"),
        .init(id: "flow-metal-hybrid-256", title: "混合GPU · 256帧", state: .archived,
              conclusion: "build22多次有限且重复L2=0，但预热两步约1.90秒/目标1秒，峰值2086–2121MB；非有利长度，CPU正确性对照仍未验收，不重复测速。"),
        .init(id: "flow-cpu-reference-384", title: "① CPU独立参考 · 384帧", state: .available,
              conclusion: "冷启动测，保存同版本/输入SHA256/文件身份/目标Mel，之后冷启动运行②；不在GPU释放后再加载CPU。"),
        .init(id: "flow-metal-hybrid-384", title: "② GPU混合 · 384帧／对照CPU", state: .available,
              conclusion: "build22两步约2.63–2.75秒；此版只读取①的匹配CPU结果，无第二套模型；先确认数值与内存。"),
        .init(id: "flow-cpu-reference-512", title: "③ CPU长负载参考 · 512帧", state: .available,
              conclusion: "74个真实token重复两次，共148 token/5.92秒等效负载；是容量/吞吐测试，不是自然长句。冷启动测。"),
        .init(id: "flow-metal-hybrid-512", title: "④ GPU混合长负载 · 512帧", state: .available,
              conclusion: "先③，再冷启动测；FP32注意力保持原实现，首次/预热和Flow-only RTF分列；800MB底线不变，不保证能前向。")
    ]
    static var available: [Route] { routes.filter { $0.state == .available } }
    static var archived: [Route] { routes.filter { $0.state == .archived } }
    static var planned: [Route] { routes.filter { $0.state == .planned } }
    static let defaultThreads = 4
}
