import Foundation

/// A bounded candidate block proposed by a speculative drafter.
public struct DraftProposal: Sendable, Equatable {
    public let tokenIDs: [Int32]

    public init(tokenIDs: [Int32]) {
        self.tokenIDs = tokenIDs
    }

    public var count: Int { tokenIDs.count }
    public var isEmpty: Bool { tokenIDs.isEmpty }
}

/// Produces candidate tokens without owning target-model KV state. The
/// generation loop passes committed/emitted history, so a drafter can be a
/// tiny model, a block-parallel model, or a cheap prompt-lookup strategy.
public protocol DraftTokenProducer: AnyObject, Sendable {
    func reset()

    func propose(history: [Int32],
                 maxTokens: Int,
                 startPosition: Int) async throws -> DraftProposal
}
