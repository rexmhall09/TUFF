import Metal

public enum MultimodalPrefillInputError: Error, Equatable {
    case tokenCountMismatch
    case invalidImageTokenRange
    case featureShapeMismatch
}

/// Three-axis positions consumed by Qwen's multimodal RoPE. Text tokens use
/// the same value on all axes; image tokens advance height and width inside the
/// merged patch grid. `ropeDelta` is retained for later generated text.
public struct MultimodalPositionIDs: Sendable, Equatable {
    public let temporal: [Int32]
    public let height: [Int32]
    public let width: [Int32]
    public let ropeDelta: Int32

    public var count: Int { temporal.count }

    public init(temporal: [Int32], height: [Int32], width: [Int32],
                ropeDelta: Int32) throws {
        guard temporal.count == height.count, height.count == width.count else {
            throw MultimodalPrefillInputError.tokenCountMismatch
        }
        self.temporal = temporal
        self.height = height
        self.width = width
        self.ropeDelta = ropeDelta
    }

    func slice(_ range: Range<Int>) throws -> MultimodalPositionIDs {
        try MultimodalPositionIDs(
            temporal: Array(temporal[range]),
            height: Array(height[range]),
            width: Array(width[range]),
            ropeDelta: ropeDelta)
    }

    func offsettingPositions(by offset: Int32, ropeDelta newDelta: Int32)
        throws -> MultimodalPositionIDs {
        try MultimodalPositionIDs(
            temporal: temporal.map { $0 + offset },
            height: height.map { $0 + offset },
            width: width.map { $0 + offset },
            ropeDelta: newDelta)
    }

    public static func qwen36(
        tokenCount: Int,
        imageSpans: [MultimodalImageSpan]
    ) throws -> MultimodalPositionIDs {
        var temporal = [Int32](repeating: 0, count: tokenCount)
        var height = temporal
        var width = temporal
        var token = 0
        var position: Int32 = 0

        func writeText(until end: Int) {
            while token < end {
                temporal[token] = position
                height[token] = position
                width[token] = position
                token += 1
                position += 1
            }
        }

        for span in imageSpans {
            writeText(until: span.tokenRange.lowerBound)
            let gridHeight = span.features.patchGridHeight / 2
            let gridWidth = span.features.patchGridWidth / 2
            guard gridHeight > 0, gridWidth > 0,
                  gridHeight * gridWidth == span.tokenRange.count else {
                throw MultimodalPrefillInputError.featureShapeMismatch
            }
            for y in 0..<gridHeight {
                for x in 0..<gridWidth {
                    temporal[token] = position
                    height[token] = position + Int32(y)
                    width[token] = position + Int32(x)
                    token += 1
                }
            }
            position += Int32(max(gridHeight, gridWidth))
        }
        writeText(until: tokenCount)
        return try MultimodalPositionIDs(
            temporal: temporal, height: height, width: width,
            ropeDelta: position - Int32(tokenCount))
    }
}

public struct MultimodalImageSpan: Sendable {
    public let tokenRange: Range<Int>
    public let features: VisionFeatures

    public init(tokenRange: Range<Int>, features: VisionFeatures) {
        self.tokenRange = tokenRange
        self.features = features
    }
}

public struct MultimodalPrefillInput: Sendable {
    public let effectiveTokenIDs: [Int32]
    public let embeddingTokenIDs: [Int32]
    public let imageSpans: [MultimodalImageSpan]
    public let family: ModelFamily
    public let positionIDs: MultimodalPositionIDs?
    public var usesBidirectionalImageAttention: Bool { family == .gemma4 }
    public var ropeDelta: Int32 { positionIDs?.ropeDelta ?? 0 }

    public var imageTokenRange: Range<Int> { imageSpans[0].tokenRange }
    public var imageFeatures: VisionFeatures { imageSpans[0].features }

    /// The same input with `tokens` placed in front of it, spans shifted to
    /// match. Used when a turn has to replay a boundary token the previous run
    /// left outside the KV: without this the multimodal input no longer lines
    /// up with the prefill suffix and the images were dropped instead.
    public func prepending(_ tokens: [Int32]) throws -> MultimodalPrefillInput {
        guard !tokens.isEmpty else { return self }
        let shift = tokens.count
        return try MultimodalPrefillInput(
            effectiveTokenIDs: tokens + effectiveTokenIDs,
            embeddingTokenIDs: tokens + embeddingTokenIDs,
            imageSpans: imageSpans.map {
                MultimodalImageSpan(
                    tokenRange: ($0.tokenRange.lowerBound + shift)
                        ..< ($0.tokenRange.upperBound + shift),
                    features: $0.features)
            },
            family: family)
    }

    /// The tail of a rendered prompt, for resuming on a cached KV prefix. Spans
    /// are rebased to the tail; a span that starts before `dropping` belongs to
    /// the cached prefix, whose features are already in the KV and are not
    /// available here, so it is rejected rather than re-injected.
    public func suffix(dropping cachedTokens: Int) throws -> MultimodalPrefillInput {
        guard cachedTokens >= 0, cachedTokens < effectiveTokenIDs.count else {
            throw MultimodalPrefillInputError.invalidImageTokenRange
        }
        let kept = imageSpans.filter { $0.tokenRange.lowerBound >= cachedTokens }
        guard kept.count == imageSpans.filter({
            $0.tokenRange.upperBound > cachedTokens
        }).count else {
            // A span straddling the boundary cannot be split.
            throw MultimodalPrefillInputError.invalidImageTokenRange
        }
        guard !kept.isEmpty else {
            // Every image is already inside the cached prefix, so the tail is
            // plain text and needs no multimodal input at all.
            throw MultimodalPrefillInputError.invalidImageTokenRange
        }
        return try MultimodalPrefillInput(
            effectiveTokenIDs: Array(effectiveTokenIDs.dropFirst(cachedTokens)),
            embeddingTokenIDs: Array(embeddingTokenIDs.dropFirst(cachedTokens)),
            imageSpans: kept.map {
                MultimodalImageSpan(
                    tokenRange: ($0.tokenRange.lowerBound - cachedTokens)
                        ..< ($0.tokenRange.upperBound - cachedTokens),
                    features: $0.features)
            },
            family: family,
            positionIDs: try positionIDs?.slice(
                cachedTokens..<effectiveTokenIDs.count))
    }

    public init(effectiveTokenIDs: [Int32],
                embeddingTokenIDs: [Int32],
                imageTokenRange: Range<Int>,
                imageFeatures: VisionFeatures,
                family: ModelFamily = .gemma4) throws {
        try self.init(
            effectiveTokenIDs: effectiveTokenIDs,
            embeddingTokenIDs: embeddingTokenIDs,
            imageSpans: [MultimodalImageSpan(
                tokenRange: imageTokenRange,
                features: imageFeatures)],
            family: family)
    }

    public init(effectiveTokenIDs: [Int32],
                embeddingTokenIDs: [Int32],
                imageSpans: [MultimodalImageSpan],
                family: ModelFamily = .gemma4,
                positionIDs explicitPositionIDs: MultimodalPositionIDs? = nil) throws {
        guard effectiveTokenIDs.count == embeddingTokenIDs.count else {
            throw MultimodalPrefillInputError.tokenCountMismatch
        }
        // The span count is self-limiting: spans are ordered, non-overlapping and
        // bounded by the prompt, so the prompt length caps how many can exist.
        guard !imageSpans.isEmpty else {
            throw MultimodalPrefillInputError.invalidImageTokenRange
        }
        var previousUpperBound = 0
        for span in imageSpans {
            let range = span.tokenRange
            guard !range.isEmpty,
                  range.lowerBound >= previousUpperBound,
                  range.upperBound <= embeddingTokenIDs.count,
                  range.count <= VisionConfig(family: family).maximumPooledTokens,
                  span.features.family == family else {
                throw MultimodalPrefillInputError.invalidImageTokenRange
            }
            let featureBytes = range.count
                * VisionConfig(family: family).textHiddenSize
                * MemoryLayout<Float16>.stride
            guard span.features.tokenCount == range.count,
                  span.features.hiddenSize == VisionConfig(family: family).textHiddenSize,
                  span.features.buffer.length >= featureBytes else {
                throw MultimodalPrefillInputError.featureShapeMismatch
            }
            previousUpperBound = range.upperBound
        }
        self.effectiveTokenIDs = effectiveTokenIDs
        self.embeddingTokenIDs = embeddingTokenIDs
        self.imageSpans = imageSpans
        self.family = family
        let positions = try explicitPositionIDs ?? (family == .qwen36
            ? MultimodalPositionIDs.qwen36(
                tokenCount: effectiveTokenIDs.count, imageSpans: imageSpans)
            : nil)
        guard positions?.count == nil || positions?.count == effectiveTokenIDs.count else {
            throw MultimodalPrefillInputError.tokenCountMismatch
        }
        self.positionIDs = positions
    }
}
