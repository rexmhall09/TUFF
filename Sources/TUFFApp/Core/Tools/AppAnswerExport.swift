import Foundation

/// An answer written to a file the person chose: the question, the answer as
/// the model wrote it, and the sources it was given, so the citations in the
/// saved text still resolve. Nothing is written without that explicit choice.
public enum AppAnswerExport {
    public static func markdown(prompt: String, response: String, modelName: String,
                                sources: [AppSource], date: Date = Date()) -> String {
        var lines = ["# \(title(prompt))", ""]
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withFullDate]
        lines.append("*\(modelName), \(formatter.string(from: date))*")
        lines += ["", "**Question**", "", prompt, "", "**Answer**", "", response]
        if !sources.isEmpty {
            lines += ["", "**Sources**", ""]
            for source in sources {
                switch source.kind {
                case .web:
                    lines.append("\(source.id). [\(source.title)](\(source.url ?? "")) (\(source.origin))")
                case .file:
                    let place = [source.filePath.map(AppFileNameText.safe), source.location].compactMap { $0 }
                        .joined(separator: ", ")
                    lines.append("\(source.id). \(source.title): \(place)")
                }
            }
        }
        return lines.joined(separator: "\n") + "\n"
    }

    /// A file name from the question: its first words, with characters that
    /// are awkward in file names removed.
    public static func suggestedFileName(prompt: String) -> String {
        let allowed = CharacterSet.alphanumerics.union(.whitespaces).union(CharacterSet(charactersIn: "-"))
        let cleaned = String(prompt.unicodeScalars.filter { allowed.contains($0) })
        let words = cleaned.split(whereSeparator: \.isWhitespace).prefix(8).joined(separator: " ")
        return (words.isEmpty ? "TUFF answer" : words) + ".md"
    }

    private static func title(_ prompt: String) -> String {
        let firstLine = prompt.split(separator: "\n").first.map(String.init) ?? "Answer"
        return firstLine.count > 80 ? String(firstLine.prefix(77)) + "..." : firstLine
    }
}
