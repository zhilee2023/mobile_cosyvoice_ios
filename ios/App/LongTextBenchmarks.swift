import Foundation

enum LongTextBenchmarks {
    static let awqMode = "three-sentence-staged-awq-head-reuse-hift-reuse"
    static func usesAwq(_ mode: String?) -> Bool { mode == awqMode }
    static var fiveSentenceText: String { sentences.prefix(5).joined() }
    // Twelve different sentences, each within the EXISTING 18-UTF16 planner
    // limit. Double the passage for the long run; duration is measured, not set.
    static let sentences = [
        "清晨的阳光照进了安静的房间。",
        "我打开窗户，听见远处的鸟鸣。",
        "街边的小店已经准备好早餐。",
        "人们带着各自的计划开始一天。",
        "我们沿着河边慢慢向前走。",
        "微风吹过，树叶轻轻地摇动。",
        "桥下的流水映着蓝色天空。",
        "有人停下脚步，拍了一张照片。",
        "午后我们来到附近的图书馆。",
        "我找了一本书，坐在窗边阅读。",
        "傍晚的灯光照亮回家的路。",
        "愿每个平凡的日子都有新收获。"
    ]
    static func text(passages: Int) -> String {
        precondition(passages == 1 || passages == 2)
        return Array(repeating: sentences.joined(), count: passages).joined()
    }
    static func batchLimit(_ mode: String?) -> Int {
        if usesAwq(mode) { return 3 }
        switch mode {
        case "two-sentence-staged": return 2
        case "three-sentence-staged", "three-sentence-staged-w8", "three-sentence-staged-w8-head-full", "three-sentence-staged-w8-head-reduced", "three-sentence-staged-w8-head-reduced-reuse", "three-sentence-staged-projection-head-reuse", "three-sentence-staged-projection-head-reuse-hift-reuse": return 3
        case "six-sentence-staged": return 6
        default: return 0
        }
    }
    static func usesW8(_ mode: String?) -> Bool {
        usesAwq(mode) || ["three-sentence-staged-w8", "three-sentence-staged-w8-head-full", "three-sentence-staged-w8-head-reduced", "three-sentence-staged-w8-head-reduced-reuse", "three-sentence-staged-projection-head-reuse", "three-sentence-staged-projection-head-reuse-hift-reuse"].contains(mode ?? "")
    }
    static func usesProjection(_ mode: String?) -> Bool {
        ["three-sentence-staged-projection-head-reuse", "three-sentence-staged-projection-head-reuse-hift-reuse"].contains(mode ?? "")
    }
    static func reusesHiFT(_ mode: String?) -> Bool { usesAwq(mode) || mode == "three-sentence-staged-projection-head-reuse-hift-reuse" }
    static func flowRecipe(_ mode: String?) -> String {
        usesAwq(mode) ? "AWQ FFN W4/g32 scaling+clipping; Q/V/out W8/g64; K preserved" : (usesProjection(mode) ? "FFN+Q/V/out W8/block64; K preserved" : (usesW8(mode) ? "FFN W8/block64" : "original FP16-stored"))
    }
    static func headMode(_ mode: String?) -> String? {
        if usesAwq(mode) { return "reduced" }
        switch mode {
        case "three-sentence-staged-w8-head-full": return "full"
        case "three-sentence-staged-w8-head-reduced", "three-sentence-staged-w8-head-reduced-reuse", "three-sentence-staged-projection-head-reuse", "three-sentence-staged-projection-head-reuse-hift-reuse": return "reduced"
        default: return nil
        }
    }
    static func title(_ mode: String?) -> String? {
        if usesAwq(mode) { return "AWQ W4实验/裁剪头复用/HiFT批内复用" }
        switch mode {
        case "three-sentence-staged-w8": return "Flow W8完整生成/三句批量"
        case "three-sentence-staged-w8-head-full": return "输出头配对A/完整W8头/Flow W8"
        case "three-sentence-staged-w8-head-reduced": return "输出头配对B/裁剪W8头/Flow W8"
        case "three-sentence-staged-w8-head-reduced-reuse": return "输出头配对C/裁剪头批内复用/Flow W8"
        case "three-sentence-staged-projection-head-reuse": return "本轮B/投影W8/HiFT逐句创建"
        case "three-sentence-staged-projection-head-reuse-hift-reuse": return "本轮C/投影W8/HiFT批内复用"
        default: return nil
        }
    }
    static func audioArchiveName(_ mode: String?) -> String? {
        if usesAwq(mode) { return "end-to-end-awq.wav" }
        switch mode {
        case "three-sentence-staged": return "end-to-end-original.wav"
        case "three-sentence-staged-w8": return "end-to-end-flow-w8.wav"
        case "three-sentence-staged-w8-head-full": return "end-to-end-head-full.wav"
        case "three-sentence-staged-w8-head-reduced": return "end-to-end-head-reduced.wav"
        case "three-sentence-staged-w8-head-reduced-reuse": return "end-to-end-head-reuse.wav"
        case "three-sentence-staged-projection-head-reuse": return "end-to-end-projection.wav"
        case "three-sentence-staged-projection-head-reuse-hift-reuse": return "end-to-end-projection-hift-reuse.wav"
        default: return nil
        }
    }
}
