import Testing
import Foundation
@testable import TUFFEngine

/// Opt-in capture for comparing the installed image tower with mlx-vlm using
/// exactly the same BF16 pixels. No model weights are bundled with the test.
@Suite struct Qwen38VisionParityTests {
    @Test(.enabled(if: ProcessInfo.processInfo.environment["TUFF_QWEN_VISION_PARITY_IMAGE"] != nil))
    func captureInstalledTowerFeatures() throws {
        let environment = ProcessInfo.processInfo.environment
        let input = try #require(environment["TUFF_QWEN_VISION_PARITY_IMAGE"])
        let root = try #require(environment["TUFF_QWEN_VISION_PARITY_MODEL"])
        let output = URL(fileURLWithPath: try #require(environment["TUFF_QWEN_VISION_PARITY_OUTPUT"]))
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
        let ctx = try MetalContext()
        let runtime = try VisionRuntime.open(textModelURL: URL(fileURLWithPath: root), context: ctx, environment: [:])
        let pixels = try VisionImagePreprocessor(device: ctx.device, config: runtime.config)
            .preprocess(fileURL: URL(fileURLWithPath: input))
        let features = try runtime.encodeImage(at: URL(fileURLWithPath: input))
        try Data(bytes: pixels.patchesBF16.contents(), count: pixels.geometry.patchCount * runtime.config.patchDimension * 2)
            .write(to: output.appendingPathComponent("patches.bf16"))
        try Data(bytes: features.buffer.contents(), count: features.tokenCount * features.hiddenSize * 2)
            .write(to: output.appendingPathComponent("features.f16"))
        let shape = ["height": pixels.geometry.patchGridHeight, "width": pixels.geometry.patchGridWidth,
                     "tokens": features.tokenCount, "hidden": features.hiddenSize]
        try JSONSerialization.data(withJSONObject: shape, options: [.sortedKeys]).write(to: output.appendingPathComponent("shape.json"))
        #expect(features.family == .qwen4Exp)
    }
}
