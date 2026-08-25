import Foundation

public struct Gemma4ImageGeometry: Sendable, Equatable {
    /// This initialiser is public and validates its inputs, so a degenerate
    /// aspect ratio must throw rather than trap on overflow.
    private static func checked(_ lhs: Int, _ rhs: Int) throws -> Int {
        let (value, overflow) = lhs.multipliedReportingOverflow(by: rhs)
        guard !overflow else {
            throw VisionImageError.invalidMetadata("source dimensions overflow")
        }
        return value
    }

    public let processedWidth: Int
    public let processedHeight: Int
    public let patchGridWidth: Int
    public let patchGridHeight: Int
    public let patchCount: Int
    public let softTokenCount: Int

    public init(
        sourceWidth: Int,
        sourceHeight: Int,
        config: VisionConfig = VisionConfig()
    ) throws {
        guard sourceWidth > 0, sourceHeight > 0 else {
            throw VisionImageError.invalidMetadata("source dimensions must be positive")
        }
        let (sourcePixels, sourceOverflow) = sourceWidth.multipliedReportingOverflow(
            by: sourceHeight)
        guard !sourceOverflow else {
            throw VisionImageError.invalidMetadata("source dimensions overflow")
        }
        if config.family == .qwen36 {
            self = try Self(
                qwenSourceWidth: sourceWidth,
                sourceHeight: sourceHeight,
                sourcePixels: sourcePixels,
                config: config)
            return
        }
        let patchPixels = config.patchSize * config.patchSize
        let targetPixels = config.maximumPatches * patchPixels
        let factor = sqrt(Double(targetPixels) / Double(sourcePixels))
        let sideMultiple = config.patchSize * config.poolingKernel
        var height = Int(floor(factor * Double(sourceHeight) / Double(sideMultiple)))
            * sideMultiple
        var width = Int(floor(factor * Double(sourceWidth) / Double(sideMultiple)))
            * sideMultiple
        let maximumSide = (config.maximumPatches
            / (config.poolingKernel * config.poolingKernel)) * sideMultiple
        if height == 0 && width == 0 {
            throw VisionImageError.invalidMetadata("aspect ratio produces zero target dimensions")
        } else if height == 0 {
            height = sideMultiple
            width = min(try Self.checked(sourceWidth / sourceHeight, sideMultiple), maximumSide)
        } else if width == 0 {
            width = sideMultiple
            height = min(try Self.checked(sourceHeight / sourceWidth, sideMultiple), maximumSide)
        }
        let (processedPixels, processedOverflow) = width.multipliedReportingOverflow(by: height)
        guard !processedOverflow, processedPixels <= targetPixels else {
            throw VisionImageError.invalidMetadata("target geometry exceeds patch budget")
        }

        let gridWidth = width / config.patchSize
        let gridHeight = height / config.patchSize
        let patches = gridWidth * gridHeight
        guard gridWidth > 0, gridHeight > 0,
              gridWidth.isMultiple(of: config.poolingKernel),
              gridHeight.isMultiple(of: config.poolingKernel),
              patches <= config.maximumPatches else {
            throw VisionImageError.invalidMetadata("target patch grid is invalid")
        }
        processedWidth = width
        processedHeight = height
        patchGridWidth = gridWidth
        patchGridHeight = gridHeight
        patchCount = patches
        softTokenCount = patches / (config.poolingKernel * config.poolingKernel)
    }

    /// Qwen's `smart_resize`, including nearest-even rounding and the exact
    /// min/max-pixel rules used by Transformers. The factor is patch size times
    /// spatial merge size, so every result can be merged without padding.
    private init(
        qwenSourceWidth sourceWidth: Int,
        sourceHeight: Int,
        sourcePixels: Int,
        config: VisionConfig
    ) throws {
        let factor = config.patchSize * config.poolingKernel
        guard max(sourceWidth, sourceHeight) / min(sourceWidth, sourceHeight) <= 200 else {
            throw VisionImageError.invalidMetadata("aspect ratio exceeds 200")
        }
        var height = max(factor, Int(
            (Double(sourceHeight) / Double(factor)).rounded(.toNearestOrEven)) * factor)
        var width = max(factor, Int(
            (Double(sourceWidth) / Double(factor)).rounded(.toNearestOrEven)) * factor)
        var pixels = try Self.checked(height, width)
        if pixels > config.maximumPixels {
            let beta = sqrt(Double(sourcePixels) / Double(config.maximumPixels))
            height = max(factor, Int(floor(
                Double(sourceHeight) / beta / Double(factor))) * factor)
            width = max(factor, Int(floor(
                Double(sourceWidth) / beta / Double(factor))) * factor)
        } else if pixels < config.minimumPixels {
            let beta = sqrt(Double(config.minimumPixels) / Double(sourcePixels))
            height = max(factor, Int(ceil(
                Double(sourceHeight) * beta / Double(factor))) * factor)
            width = max(factor, Int(ceil(
                Double(sourceWidth) * beta / Double(factor))) * factor)
        }
        pixels = try Self.checked(height, width)
        guard pixels <= config.maximumPixels else {
            throw VisionImageError.invalidMetadata("target geometry exceeds Qwen pixel budget")
        }
        let gridWidth = width / config.patchSize
        let gridHeight = height / config.patchSize
        let patches = try Self.checked(gridWidth, gridHeight)
        guard gridWidth.isMultiple(of: config.poolingKernel),
              gridHeight.isMultiple(of: config.poolingKernel),
              patches <= config.maximumPatches else {
            throw VisionImageError.invalidMetadata("target patch grid is invalid")
        }
        processedWidth = width
        processedHeight = height
        patchGridWidth = gridWidth
        patchGridHeight = gridHeight
        patchCount = patches
        softTokenCount = patches / (config.poolingKernel * config.poolingKernel)
    }

}
