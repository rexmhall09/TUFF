import Foundation
import TUFFEngine

/// How much of the context window the next message would use.
///
/// Deliberately an estimate, and labelled as one everywhere it is shown. The
/// real count comes from the model's tokenizer, which lives in the decode
/// service — a separate process that is not asked anything until a run starts.
/// Waiting for it would mean the figure only appeared after the moment it was
/// useful, so this is computed here from what the app already knows.
///
/// The estimate exists because the context limit was previously invisible until
/// it was hit: the renderer dropped the oldest turns to make the prompt fit and
/// nothing said so, and a document large enough to fill the window on its own
/// gave no warning before it was sent.
public struct AppContextUsage: Equatable, Sendable {
    /// Characters per token. The same divisor `ExtractedDocument` uses, so a
    /// file's reported size and the meter agree with each other.
    public static let charactersPerToken = 4

    /// Tokens the saved conversation is expected to contribute.
    public let historyTokens: Int
    /// Tokens the message being composed is expected to contribute, attached
    /// files and images included.
    public let draftTokens: Int
    /// The context the next run would actually use, which is the loaded
    /// session's until it is reloaded.
    public let maxTokens: Int

    public init(historyTokens: Int, draftTokens: Int, maxTokens: Int) {
        self.historyTokens = historyTokens
        self.draftTokens = draftTokens
        self.maxTokens = maxTokens
    }

    public var estimatedTokens: Int { historyTokens + draftTokens }

    public var fraction: Double {
        guard maxTokens > 0 else { return 0 }
        return min(1, max(0, Double(estimatedTokens) / Double(maxTokens)))
    }

    /// Close enough to the limit to be worth showing without being asked.
    public var isTight: Bool { fraction >= 0.75 }

    /// The estimate says the oldest turns will have to be dropped. It is an
    /// estimate, so this reads as a warning rather than a refusal — the run
    /// itself is what decides, and it drops turns rather than failing.
    public var willDropOldestTurns: Bool {
        estimatedTokens >= maxTokens && historyTokens > 0
    }

    /// The draft alone does not fit, which dropping history cannot fix.
    public var draftAloneOverflows: Bool { draftTokens >= maxTokens }

    public static func tokens(inText text: String) -> Int {
        guard !text.isEmpty else { return 0 }
        return max(1, text.count / charactersPerToken)
    }

    /// Images are counted at their worst case. The real cost depends on the
    /// picture's geometry, which is not known until it is preprocessed, and a
    /// meter that under-counted would promise room that is not there.
    public static func tokens(forImages count: Int, family: ModelFamily) -> Int {
        guard count > 0 else { return 0 }
        return count * VisionImageTokenBudget.maximumTokensPerImage(family: family)
    }

    /// Every message costs a little framing — role markers and turn
    /// delimiters — which is small per turn and not nothing over a long chat.
    public static let overheadTokensPerMessage = 8
}
