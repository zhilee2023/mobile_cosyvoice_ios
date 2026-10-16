import Foundation

enum SentencePlanError: LocalizedError {
    case empty, tooLong, tooMany, oversizedCharacter
    var errorDescription: String? {
        switch self {
        case .empty: return "请输入需要朗读的文字。"
        case .tooLong: return "首轮长文测试最多 2000 个 UTF-16 单元。"
        case .tooMany: return "拆分超过 128 段，请减少文字。"
        case .oversizedCharacter: return "一个组合字符超过单段限制，无法安全拆分。"
        }
    }
}

enum SentencePlanner {
    static let defaultLimit = 24
    static func plan(_ text: String, limit: Int = defaultLimit) throws -> [String] {
        guard !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { throw SentencePlanError.empty }
        guard text.utf16.count <= 2000 else { throw SentencePlanError.tooLong }
        let chars = Array(text)
        let terminals: Set<Character> = ["。", "！", "？", "!", "?", "；", ";", "\n", "\r"]
        var units: [String] = [], current = ""
        for (index, char) in chars.enumerated() {
            current.append(char)
            // Preserve decimal numbers such as 3.14; sentence-final English
            // periods split only at whitespace/end, not inside URLs/initials.
            let period = char == "." && (index + 1 == chars.count || chars[index + 1].isWhitespace)
            let newline = char.unicodeScalars.allSatisfy { CharacterSet.newlines.contains($0) }
            if terminals.contains(char) || newline || period {
                units += try partition(current, limit: limit)
                current = ""
            }
        }
        units += try partition(current, limit: limit)
        guard !units.isEmpty else { throw SentencePlanError.empty }
        guard units.count <= 128 else { throw SentencePlanError.tooMany }
        return units
    }

    static func partition(_ text: String, limit: Int) throws -> [String] {
        precondition(limit > 0)
        var remaining = Array(text.trimmingCharacters(in: .whitespacesAndNewlines))
        let soft: Set<Character> = ["，", ",", "、", "：", ":", " ", "\t"]
        var result: [String] = []
        while !remaining.isEmpty {
            var count = 0, length = 0, boundary = 0
            for char in remaining {
                let size = String(char).utf16.count
                if length + size > limit { break }
                count += 1; length += size
                if (soft.contains(char) || char.isWhitespace) && length >= limit / 2 { boundary = count }
            }
            guard count > 0 else { throw SentencePlanError.oversizedCharacter }
            let end = count == remaining.count ? count : (boundary > 0 ? boundary : count)
            let part = String(remaining.prefix(end)).trimmingCharacters(in: .whitespacesAndNewlines)
            if !part.isEmpty { result.append(part) }
            remaining.removeFirst(end)
        }
        return result
    }
}
