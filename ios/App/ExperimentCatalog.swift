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
        .init(id: "three-sentence-staged", title: "三句分阶段／唯一 Flow（新对照）", state: .available,
              conclusion: "最新三句8.52秒音频，计算RTF1.518、首块7.95秒、最低余量1601MB，未观察到断供；更长文本仍待验收。", execution: .sentence),
        .init(id: "flow-metal-low-memory-256", title: "Metal Low 256帧会话内存", state: .archived,
              conclusion: "v0.7.5两次主后端Metal，峰值2062–2092MB、最低余量1129–1159MB；仅resize通过，尚无GPU前向结果。"),
        .init(id: "flow-metal-low-256", title: "GPU 256帧两步Flow", state: .archived,
              conclusion: "v0.7.6第一步17920/20480个值NaN，有效224帧均受影响；最低余量1115MB，属于数值失败。"),
        .init(id: "flow-metal-low-384", title: "GPU 384帧两步Flow", state: .archived,
              conclusion: "v0.7.6第一步25120/30720个值NaN，有效314帧均受影响；最低余量1106MB，未形成有效GPU测速。"),
        .init(id: "flow-metal-nan", title: "GPU 首个 NaN 算子定位", state: .archived,
              conclusion: "v0.7.7第三个Transformer块QK乘法有限输入→大量Inf，随后Softmax首次NaN；定位完成，最低余量1108MB。"),
        .init(id: "flow-metal-hybrid-256", title: "① GPU主干＋FP32注意力 · 256帧", state: .available,
              conclusion: "Metal Low主干，CPU FP32重算QK+mask+Softmax；两步与CPU High数值对照；先冷启动测本项。"),
        .init(id: "flow-metal-hybrid-384", title: "② GPU主干＋FP32注意力 · 384帧", state: .available,
              conclusion: "同一混合实现，固定70个token；256帧有限且内存通过后冷启动测；不是纯GPU或整链RTF。")
    ]
    static var available: [Route] { routes.filter { $0.state == .available } }
    static var archived: [Route] { routes.filter { $0.state == .archived } }
    static var planned: [Route] { routes.filter { $0.state == .planned } }
    static let defaultThreads = 4
}
