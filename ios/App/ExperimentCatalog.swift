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
        .init(id: "flow-w8-reference", title: "W8原版五条参考", state: .archived,
              conclusion: "build26五条/三轮数值通过，重复L2=0，预热两步均值2.18秒，峰值1584MB；本轮thermal=2，不当冷态排名。"),
        .init(id: "flow-w8-normal", title: "FFN W8/block64 普通内存", state: .archived,
              conclusion: "五条通过，平均Mel L2约0.23%；预热2.37秒、峰值1584MB，未显示内存/速度收益，不提升默认。"),
        .init(id: "flow-w8-low", title: "FFN W8/block64 低内存整数", state: .archived,
              conclusion: "五条通过，预热1.75秒，峰值1271MB；本轮Flow约少19.8%耗时/313MB内存，平均Mel L2约0.86%，整链与音质仍待验收。"),
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
        .init(id: "three-sentence-staged", title: "原版Flow · 当前文字整链对照", state: .archived,
              conclusion: "保留原包Flow和默认LLM；同文本重复采样可能得到不同token，严格声音差异需匹配token。", execution: .sentence),
        .init(id: "three-sentence-staged-w8", title: "Flow W8 · 当前文字完整生成", state: .archived,
              conclusion: "真实LLM→Conditioner→FFN W8 Flow High/Memory Low→HiFT；批内复用唯一Flow384，完整EOS，输出头未裁剪。", execution: .sentence),
        .init(id: "three-sentence-staged-w8-head-full", title: "输出头配对A · 完整头", state: .archived,
              conclusion: "五句370 token与裁剪头一致；RTF1.421对1.247，原配对同进程，结果不等于统计性冷态排名。保留实现和历史WAV。", execution: .sentence),
        .init(id: "three-sentence-staged-w8-head-reduced", title: "输出头配对B · 裁剪头独立加载", state: .archived,
              conclusion: "build28同版本与复用组五句370 token一致；RTF1.256对1.224，收益约2.6%，不是主要瓶颈。", execution: .sentence),
        .init(id: "three-sentence-staged-w8-head-reduced-reuse", title: "旧FFN W8整链五句", state: .archived,
              conclusion: "build30同进程配对RTF1.333，投影W8为1.123；370 token一致，非独立冷启动排名。已归档，不重复旧路线。", execution: .sentence),
        .init(id: "three-sentence-staged-projection-head-reuse", title: "投影W8／逐句HiFT", state: .archived,
              conclusion: "build30为RTF1.123，增加HiFT复用后1.081，Mel/PCM哈希相同；已归档，仅保留复用版本作本轮对照。", execution: .sentence),
        .init(id: "three-sentence-staged-projection-head-reuse-hift-reuse", title: "W8对照 · 固定五句", state: .available,
              conclusion: "投影W8＋HiFT批内复用；build30曾测RTF1.081、峰值1065MB，本轮须重测，不能直接归因跨版本差异。", execution: .sentence),
        .init(id: "three-sentence-staged-awq-head-reuse-hift-reuse", title: "AWQ W4 · 固定五句（实验）", state: .available,
              conclusion: "仅替换Flow为FFN AWQ W4/g32＋66投影W8；缩放/裁剪，K保留。PC试听认可但数值筛查未过，真机未验收；两步/4线程/裁剪头/HiFT复用不变。", execution: .sentence),
        .init(id: "flow-w8-profile", title: "Flow W8 · CPU热点", state: .archived,
              conclusion: "五条回调/普通Mel L2=0；Convolution约75%，五条top30中149/150为Q/K/V/输出投影；非整链RTF。"),
        .init(id: "flow-w8-metal-high", title: "Flow W8 · Metal High 384帧", state: .archived,
              conclusion: "build28冷启动约37MB，resize约2238MB、余量983MB，不达1100MB门槛，0次前向；不重复或放宽门槛。"),
        .init(id: "flow-projection-reference", title: "投影W8模块对照 · 原FFN", state: .archived,
              conclusion: "build29同进程五条完整配对，预热两步合计8.064秒，峰值1213MB；不是独立冷启动统计排名。"),
        .init(id: "flow-projection-candidate", title: "投影W8模块候选", state: .archived,
              conclusion: "build29五条通过，预热合计6.621秒（少17.9%）、峰值1005MB（少208MB）；最大相对手机Mel L2=1.52%，重复0。整链和听感待验收。"),
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
        .init(id: "flow-cpu-reference-384", title: "CPU独立参考 · 384帧", state: .archived,
              conclusion: "冷启动测，保存同版本/输入SHA256/文件身份/目标Mel，之后冷启动运行②；不在GPU释放后再加载CPU。"),
        .init(id: "flow-metal-hybrid-384", title: "GPU混合 · 384帧", state: .archived,
              conclusion: "build26 resize后峰值2295MB/余量926MB，未达1100MB规划门槛，未前向；不重复旧配置。"),
        .init(id: "flow-cpu-reference-512", title: "CPU长负载参考 · 512帧", state: .archived,
              conclusion: "74个真实token重复两次，共148 token/5.92秒等效负载；是容量/吞吐测试，不是自然长句。冷启动测。"),
        .init(id: "flow-metal-hybrid-512", title: "GPU混合长负载 · 512帧", state: .archived,
              conclusion: "build26 resize后峰值2307MB/余量914MB，内存保护停止，未前向；未证明整链可用。")
    ]
    static var available: [Route] { routes.filter { $0.state == .available } }
    static var archived: [Route] { routes.filter { $0.state == .archived } }
    static var planned: [Route] { routes.filter { $0.state == .planned } }
    static let defaultThreads = 4
}
