import Foundation
import Testing
import TUFFModelCatalog
@testable import TUFFRepackCore

/// Ornith's image packs come from Transformers-format checkpoints: other
/// tensor names, a channels-first patch embedding and, for 9B and 35B, a
/// different repository from the text model.
@Suite struct OrnithVisionSourceTests {
    @Test func transformersNamesMapToThePacksNames() {
        #expect(RepackPlanner.canonicalVisionName("model.visual.blocks.3.attn.qkv.weight")
                == "vision_tower.blocks.3.attn.qkv.weight")
        #expect(RepackPlanner.canonicalVisionName("vision_tower.merger.norm.weight")
                == "vision_tower.merger.norm.weight")
        #expect(RepackPlanner.canonicalVisionName("model.language_model.embed_tokens.weight")
                == "model.language_model.embed_tokens.weight")
    }

    /// `[out, channels, t, h, w]` becomes `[out, t, h, w, channels]`: every
    /// element lands where MLX's layout expects it.
    @Test func patchEmbeddingIsReorderedChannelsLast() throws {
        let shape: [UInt64] = [2, 3, 2, 2, 2]
        let count = shape.reduce(1, *)
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("reorder-\(UUID()).bin")
        defer { try? FileManager.default.removeItem(at: url) }
        let prefix = Data(repeating: 0xAB, count: 16)
        let values = (0..<UInt16(count)).map { $0 }
        try (prefix + values.withUnsafeBytes { Data($0) }).write(to: url)

        try VisionTensorReorder.channelsLastInPlace(path: url.path, offset: 16, sourceShape: shape)

        let data = try Data(contentsOf: url)
        #expect(data.prefix(16) == prefix)
        let result = data.dropFirst(16).withUnsafeBytes { Array($0.bindMemory(to: UInt16.self)) }
        for o in 0..<2 { for c in 0..<3 { for s in 0..<8 {
            #expect(result[(o * 8 + s) * 3 + c] == UInt16((o * 3 + c) * 8 + s))
        } } }
    }

    /// A separate image source is allowed only as the exact pair the catalog
    /// pins, never for another model and never for an unknown checkpoint.
    @Test func separateImageSourcesBindOnlyToTheirModel() {
        let ornith9 = "sha256:" + TUFFModelCatalog.ornith15_9B.source.sourceIndexSHA256
        let ornith35 = "sha256:" + TUFFModelCatalog.ornith15_35B_A3B.source.sourceIndexSHA256
        let pack9 = TUFFModelCatalog.ornith15_9B.addons[0].source.sourceIndexSHA256
        #expect(pack9 != TUFFModelCatalog.ornith15_9B.source.sourceIndexSHA256)
        #expect(TUFFModelCatalog.imagePackSourceMatches(textSourceSnapshotHash: ornith9,
                                                        visionSourceIndexSHA256: pack9))
        #expect(!TUFFModelCatalog.imagePackSourceMatches(textSourceSnapshotHash: ornith35,
                                                         visionSourceIndexSHA256: pack9))
        #expect(!TUFFModelCatalog.imagePackSourceMatches(textSourceSnapshotHash: ornith9,
                                                         visionSourceIndexSHA256: String(repeating: "0", count: 64)))
        // Same-checkpoint packs keep working as before.
        let qwen = TUFFModelCatalog.qwen36_35B_A3B.source.sourceIndexSHA256
        #expect(TUFFModelCatalog.imagePackSourceMatches(textSourceSnapshotHash: "sha256:" + qwen,
                                                        visionSourceIndexSHA256: qwen))
        #expect(SourceFingerprint.modelID(forIndexSha256: pack9, repoID: "ornith-ai/Ornith-1.5-9B")
                == "ornith-ai/Ornith-1.5-9B")
        #expect(SourceFingerprint.modelID(forIndexSha256: pack9, repoID: "someone/else") == nil)
    }
}
