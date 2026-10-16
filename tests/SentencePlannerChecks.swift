import Foundation

@main
struct SentencePlannerChecks {
    static func main() throws {
        func check(_ text: String) throws -> [String] {
            let parts = try SentencePlanner.plan(text)
            precondition(parts.allSatisfy { !$0.isEmpty && $0.utf16.count <= 24 })
            let normalize: (String) -> String = { String($0.filter { !$0.isWhitespace }) }
            precondition(normalize(parts.joined()) == normalize(text), "Dropped/repeated content")
            return parts
        }
        let chinese = try check("你好。今天好吗？我们继续测试！")
        precondition(chinese.count == 3)
        let newlines = try check("第一句\r\n第二句\n第三句")
        precondition(newlines.count == 3)
        let english = try check("Value 3.14 is valid. Next sentence.")
        precondition(english.count >= 2)
        let decimal = try check("Value 3.14 is valid.")
        precondition(decimal[0].contains("3.14"))
        _ = try check(String(repeating: "没有标点也不能丢字", count: 8))
        _ = try check("今天天气不错，我们一起出去走走，看看远方的风景，听听鸟儿唱歌。")
        _ = try check(String(repeating: "👩‍💻你好😀", count: 12))
        _ = try check(String(repeating: "字", count: 2000))
        do { _ = try SentencePlanner.plan(" \n\t"); preconditionFailure("Accepted empty") } catch SentencePlanError.empty {}
        do { _ = try SentencePlanner.plan(String(repeating: "字", count: 2001)); preconditionFailure("Accepted >2000") } catch SentencePlanError.tooLong {}
        do { _ = try SentencePlanner.plan(String(repeating: "字。", count: 129)); preconditionFailure("Accepted >128") } catch SentencePlanError.tooMany {}
        let retry = try SentencePlanner.partition("这是一句需要重新分成更短部分的文字。", limit: 10)
        precondition(retry.count > 1 && retry.joined() == "这是一句需要重新分成更短部分的文字。")
        print("PASS: sentence splitting preserves content, bounds UTF-16, handles emoji/newlines/decimals/retry limits")
    }
}
