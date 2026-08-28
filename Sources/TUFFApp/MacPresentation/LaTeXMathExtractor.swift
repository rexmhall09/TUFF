import Foundation

struct LaTeXFormula: Equatable, Sendable {
    let token: String
    let source: String
    let body: String
    let isDisplay: Bool
}

struct LaTeXExtraction: Equatable, Sendable {
    let markdownSource: String
    let formulas: [LaTeXFormula]
}

enum LaTeXMathExtractor {
    private struct Delimiter {
        let opening: String
        let closing: String
        let isDisplay: Bool
    }

    static func extract(from source: String) -> LaTeXExtraction {
        guard !source.isEmpty else {
            return LaTeXExtraction(markdownSource: source, formulas: [])
        }

        var output = ""
        var formulas: [LaTeXFormula] = []
        var index = source.startIndex
        var inlineCodeTicks: Int?
        var fencedCodeTicks: Int?

        while index < source.endIndex {
            if source[index] == "`" {
                let count = runLength(of: "`", at: index, in: source)
                let runEnd = source.index(index, offsetBy: count)
                output.append(contentsOf: source[index..<runEnd])

                if isFenceStart(at: index, in: source), count >= 3,
                   inlineCodeTicks == nil {
                    if fencedCodeTicks == nil {
                        fencedCodeTicks = count
                    } else if count >= fencedCodeTicks! {
                        fencedCodeTicks = nil
                    }
                } else if fencedCodeTicks == nil {
                    if inlineCodeTicks == nil {
                        inlineCodeTicks = count
                    } else if count == inlineCodeTicks {
                        inlineCodeTicks = nil
                    }
                }
                index = runEnd
                continue
            }

            if fencedCodeTicks == nil, inlineCodeTicks == nil,
               let delimiter = delimiter(at: index, in: source),
               let closingRange = closingRange(
                for: delimiter, after: index, in: source) {
                let bodyStart = source.index(index, offsetBy: delimiter.opening.count)
                let body = String(source[bodyStart..<closingRange.lowerBound])
                guard !body.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
                    output.append(source[index])
                    index = source.index(after: index)
                    continue
                }

                let token = uniqueToken(number: formulas.count, source: source)
                let end = closingRange.upperBound
                formulas.append(LaTeXFormula(
                    token: token,
                    source: String(source[index..<end]),
                    body: body,
                    isDisplay: delimiter.isDisplay))
                output.append(token)
                index = end
                continue
            }

            output.append(source[index])
            index = source.index(after: index)
        }

        return LaTeXExtraction(markdownSource: output, formulas: formulas)
    }

    private static func delimiter(at index: String.Index, in source: String) -> Delimiter? {
        guard !isEscaped(index, in: source) else { return nil }
        let suffix = source[index...]
        if suffix.hasPrefix("$$") {
            return Delimiter(opening: "$$", closing: "$$", isDisplay: true)
        }
        if suffix.hasPrefix(#"\["#) {
            return Delimiter(opening: #"\["#, closing: #"\]"#, isDisplay: true)
        }
        if suffix.hasPrefix(#"\("#) {
            return Delimiter(opening: #"\("#, closing: #"\)"#, isDisplay: false)
        }
        if suffix.hasPrefix("$") {
            let next = source.index(after: index)
            guard next < source.endIndex,
                  !source[next].isWhitespace,
                  source[next] != "$" else { return nil }
            return Delimiter(opening: "$", closing: "$", isDisplay: false)
        }
        return nil
    }

    private static func closingRange(
        for delimiter: Delimiter,
        after openingIndex: String.Index,
        in source: String
    ) -> Range<String.Index>? {
        var searchStart = source.index(
            openingIndex, offsetBy: delimiter.opening.count)
        while searchStart < source.endIndex,
              let range = source.range(
                of: delimiter.closing,
                range: searchStart..<source.endIndex) {
            if !isEscaped(range.lowerBound, in: source) {
                if delimiter.closing != "$" {
                    return range
                }
                let previous = source.index(before: range.lowerBound)
                if !source[previous].isWhitespace {
                    return range
                }
            }
            searchStart = range.upperBound
        }
        return nil
    }

    private static func isEscaped(_ index: String.Index, in source: String) -> Bool {
        var cursor = index
        var backslashes = 0
        while cursor > source.startIndex {
            let previous = source.index(before: cursor)
            guard source[previous] == "\\" else { break }
            backslashes += 1
            cursor = previous
        }
        return !backslashes.isMultiple(of: 2)
    }

    private static func runLength(
        of character: Character,
        at index: String.Index,
        in source: String
    ) -> Int {
        var cursor = index
        var count = 0
        while cursor < source.endIndex, source[cursor] == character {
            count += 1
            cursor = source.index(after: cursor)
        }
        return count
    }

    private static func isFenceStart(
        at index: String.Index,
        in source: String
    ) -> Bool {
        let lineStart = source[..<index].lastIndex(of: "\n")
            .map { source.index(after: $0) } ?? source.startIndex
        let prefix = source[lineStart..<index]
        return prefix.count <= 3 && prefix.allSatisfy { $0 == " " }
    }

    private static func uniqueToken(number: Int, source: String) -> String {
        var candidate = "TUFFMATHPLACEHOLDER\(number)END"
        while source.contains(candidate) {
            candidate.append("X")
        }
        return candidate
    }
}
