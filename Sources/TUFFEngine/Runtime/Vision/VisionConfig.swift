import Foundation

/// Family-owned vision architecture and processor contract. Gemma remains the
/// default so existing callers and fixtures keep their v1.0 behavior.
public struct VisionConfig: Sendable, Equatable {
    public let family: ModelFamily
    public let hiddenSize: Int
    public let intermediateSize: Int
    public let numLayers: Int
    public let numHeads: Int
    public let headDimension: Int
    public let patchSize: Int
    public let temporalPatchSize: Int
    public let patchDimension: Int
    public let maximumPatches: Int
    public let poolingKernel: Int
    public let textHiddenSize: Int
    public let positionEmbeddingSize: Int
    public let positionGridSide: Int
    public let rmsEpsilon: Float
    public let ropeTheta: Float
    public let attentionScale: Float
    public let minimumPixels: Int
    public let maximumPixels: Int

    public var maximumPooledTokens: Int {
        maximumPatches / (poolingKernel * poolingKernel)
    }

    public init(family: ModelFamily = .gemma4) {
        self.family = family
        hiddenSize = 1_152
        intermediateSize = 4_304
        numLayers = 27
        numHeads = 16
        headDimension = 72
        patchSize = 16
        rmsEpsilon = 1e-6
        switch family {
        case .gemma4:
            temporalPatchSize = 1
            patchDimension = 768
            maximumPatches = 2_520
            poolingKernel = 3
            textHiddenSize = 2_816
            positionEmbeddingSize = 10_240
            positionGridSide = 10_240
            ropeTheta = 100
            // Gemma normalizes Q/K inside the tower before attention. Keep the
            // existing unscaled score path exactly as v1.0 shipped it.
            attentionScale = 1
            minimumPixels = 0
            maximumPixels = maximumPatches * patchSize * patchSize
        case .qwen36:
            temporalPatchSize = 2
            patchDimension = 1_536
            maximumPatches = 65_536
            poolingKernel = 2
            textHiddenSize = 2_048
            positionEmbeddingSize = 2_304
            positionGridSide = 48
            ropeTheta = 10_000
            // Qwen's vision attention is standard scaled dot product
            // attention, with head dimension 1152 / 16 = 72.
            attentionScale = 1 / sqrt(Float(headDimension))
            minimumPixels = 65_536
            maximumPixels = 16_777_216
        case .gptOss:
            // GPT-OSS v2 checkpoints are text-only. Keep this initializer
            // fail-closed if a caller tries to construct a vision runtime.
            temporalPatchSize = 0
            patchDimension = 0
            maximumPatches = 0
            poolingKernel = 0
            textHiddenSize = 0
            positionEmbeddingSize = 0
            positionGridSide = 0
            ropeTheta = 0
            attentionScale = 0
            minimumPixels = 0
            maximumPixels = 0
            preconditionFailure("GPT-OSS has no vision companion")
        }
    }
}
