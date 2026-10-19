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
        case "three-sentence-staged": return 3
        case "six-sentence-staged": return 6
        default: return 0
        }
    }
}
