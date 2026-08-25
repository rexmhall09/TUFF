import Foundation

/// Shape and scaling contract for Gemma 4 per-layer embeddings (PLE).
public struct PerLayerEmbeddingPlan: Sendable, Equatable {
    public let numLayers: Int
    public let mainHiddenSize: Int
    public let perLayerHiddenSize: Int
    public let vocabSize: Int

    public init(config: ArchConfig) {
        precondition(config.hasPerLayerInputs, "architecture does not use PLE")
        self.numLayers = config.numLayers
        self.mainHiddenSize = config.hiddenSize
        self.perLayerHiddenSize = config.hiddenSizePerLayerInput
        self.vocabSize = config.vocabSizePerLayerInput
    }

    public var packedWidth: Int { numLayers * perLayerHiddenSize }
    public var tokenIdentityScale: Float { sqrt(Float(perLayerHiddenSize)) }
    public var contextProjectionScale: Float { 1 / sqrt(Float(mainHiddenSize)) }
    public var combinedScale: Float { 1 / sqrt(2) }

    public func packedRange(layer: Int) -> Range<Int> {
        precondition((0..<numLayers).contains(layer), "PLE layer out of range")
        let start = layer * perLayerHiddenSize
        return start..<(start + perLayerHiddenSize)
    }

    /// CPU oracle for the final PLE combination after the context projection's
    /// RMSNorm. Kept tiny so toy fixtures can validate the runtime contract.
    public func combine(tokenIdentity: [Float], contextAware: [Float]) -> [Float] {
        precondition(tokenIdentity.count == perLayerHiddenSize)
        precondition(contextAware.count == perLayerHiddenSize)
        return zip(tokenIdentity, contextAware).map {
            ($0 * tokenIdentityScale + $1) * combinedScale
        }
    }
}
