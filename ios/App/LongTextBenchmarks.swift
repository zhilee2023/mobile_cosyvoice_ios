import Foundation

enum LongTextBenchmarks {
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
        switch mode {
        case "two-sentence-staged": return 2
        case "three-sentence-staged", "three-sentence-staged-w8", "three-sentence-staged-w8-head-full", "three-sentence-staged-w8-head-reduced": return 3
        case "six-sentence-staged": return 6
        default: return 0
        }
    }
    static func usesW8(_ mode: String?) -> Bool {
        ["three-sentence-staged-w8", "three-sentence-staged-w8-head-full", "three-sentence-staged-w8-head-reduced"].contains(mode ?? "")
    }
    static func headMode(_ mode: String?) -> String? {
        switch mode {
        case "three-sentence-staged-w8-head-full": return "full"
        case "three-sentence-staged-w8-head-reduced": return "reduced"
        default: return nil
        }
    }
    static func title(_ mode: String?) -> String? {
        switch mode {
        case "three-sentence-staged-w8": return "Flow W8完整生成/三句批量"
        case "three-sentence-staged-w8-head-full": return "输出头配对A/完整W8头/Flow W8"
        case "three-sentence-staged-w8-head-reduced": return "输出头配对B/裁剪W8头/Flow W8"
        default: return nil
        }
    }
    static func audioArchiveName(_ mode: String?) -> String? {
        switch mode {
        case "three-sentence-staged": return "end-to-end-original.wav"
        case "three-sentence-staged-w8": return "end-to-end-flow-w8.wav"
        case "three-sentence-staged-w8-head-full": return "end-to-end-head-full.wav"
        case "three-sentence-staged-w8-head-reduced": return "end-to-end-head-reduced.wav"
        default: return nil
        }
    }
}
