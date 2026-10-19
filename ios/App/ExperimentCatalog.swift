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
        .init(id: "llm-chain-reuse", title: "对照① 整链 LLM 复用", state: .available,
              conclusion: "每句清KV并重新prefill；声学阶段顺序运行。余量不足会释放LLM，报告记录真实复用次数。", execution: .sentence),
        .init(id: "two-sentence-staged", title: "对照② 两句分阶段／唯一 Flow", state: .available,
              conclusion: "先生成最多两句完整token，再释放LLM；小批内复用唯一Flow384，逐句出声。首句可能更晚。", execution: .sentence)
    ]
    static var available: [Route] { routes.filter { $0.state == .available } }
    static var archived: [Route] { routes.filter { $0.state == .archived } }
    static var planned: [Route] { routes.filter { $0.state == .planned } }
    static let defaultThreads = 4
}
