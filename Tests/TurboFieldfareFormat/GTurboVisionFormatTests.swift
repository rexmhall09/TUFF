import Foundation
import Testing
@testable import TurboFieldfareFormat

private enum VisionFormatFixture {
    static let zeroSHA = String(repeating: "0", count: 64)

    static func manifest(tensors: [GTurboVisionTensorRegionV1]? = nil)
        -> GTurboVisionManifestV1 {
        let defaultTensors = [
            GTurboVisionTensorRegionV1(
                name: "vision_tower.patch_embedder.input_proj.weight",
                executionPosition: 0, offset: 0, size: 512,
                shape: [16, 16], dtype: .bf16),
            GTurboVisionTensorRegionV1(
                name: "embed_vision.embedding_projection.weight",
                executionPosition: 1, offset: GTurboVisionFormatV1.alignmentBytes,
                size: 256, shape: [8, 8], dtype: .u32,
                quantization: GTurboVisionQuantizationV1(
                    weightBits: 4, scheme: "affine", scaleType: "bf16",
                    biasType: "bf16", groupSize: 64)),
        ]
        return GTurboVisionManifestV1(
            modelID: "fixture/model", sourceRevision: "fixture-revision",
            sourceIndexSha256: zeroSHA, processorConfigSha256: zeroSHA,
            compatibleTextSourceSnapshotHash: "fixture-text-snapshot",
            compatibleTextManifestSha256: zeroSHA,
            files: [
                GTurboVisionFormatV1.weightsFile: GTurboManifestFileV1(
                    size: 2 * GTurboVisionFormatV1.alignmentBytes, sha256: zeroSHA),
                GTurboVisionFormatV1.processorFile: GTurboManifestFileV1(
                    size: 2, sha256: zeroSHA),
            ],
            tensors: tensors ?? defaultTensors)
    }

    static func receipt() -> GTurboVisionReceiptV1 {
        GTurboVisionReceiptV1(
            manifestSha256: zeroSHA,
            companionDirectoryPath: "/fixture/model.vision.gturbo",
            compatibleTextManifestSha256: zeroSHA,
            sourceRepoID: "fixture/model", sourceRevision: "fixture-revision",
            verificationTimestamp: "2026-08-05T00:00:00Z",
            toolVersion: "fixture",
            files: [
                GTurboVisionFormatV1.manifestFile: .init(size: 1, sha256: zeroSHA),
                GTurboVisionFormatV1.weightsFile: .init(
                    size: 2 * GTurboVisionFormatV1.alignmentBytes, sha256: zeroSHA),
                GTurboVisionFormatV1.processorFile: .init(size: 2, sha256: zeroSHA),
            ])
    }
}

@Suite struct GTurboVisionFormatCompatibilityTests {
    @Test func manifestAndReceiptRoundTripDeterministically() throws {
        let manifest = VisionFormatFixture.manifest()
        let manifestData = try GTurboVisionManifestCodec.encode(manifest)
        #expect(try GTurboVisionManifestCodec.decode(manifestData) == manifest)
        #expect(try GTurboVisionManifestCodec.encode(manifest) == manifestData)

        let receipt = VisionFormatFixture.receipt()
        let receiptData = try GTurboVisionReceiptCodec.encode(receipt)
        #expect(try GTurboVisionReceiptCodec.decode(receiptData) == receipt)
        #expect(try GTurboVisionReceiptCodec.encode(receipt) == receiptData)
    }

    @Test func rejectsUnknownVersion() {
        let original = VisionFormatFixture.manifest()
        let changed = GTurboVisionManifestV1(
            versionMajor: 2, modelID: original.modelID,
            sourceRevision: original.sourceRevision,
            sourceIndexSha256: original.sourceIndexSha256,
            processorConfigSha256: original.processorConfigSha256,
            compatibleTextSourceSnapshotHash: original.compatibleTextSourceSnapshotHash,
            compatibleTextManifestSha256: original.compatibleTextManifestSha256,
            files: original.files, tensors: original.tensors)
        #expect(throws: GTurboFormatError.self) {
            try GTurboVisionManifestCodec.encode(changed)
        }
    }

    @Test func rejectsReorderedRegions() {
        let original = VisionFormatFixture.manifest().tensors
        let changed = [
            GTurboVisionTensorRegionV1(
                name: original[0].name, executionPosition: 1,
                offset: original[0].offset, size: original[0].size,
                shape: original[0].shape, dtype: original[0].dtype),
            original[1],
        ]
        #expect(throws: GTurboFormatError.self) {
            try GTurboVisionManifestCodec.encode(
                VisionFormatFixture.manifest(tensors: changed))
        }
    }

    @Test func rejectsMisalignedOverlappingAndOutOfFileRegions() {
        let invalid: [GTurboVisionTensorRegionV1] = [
            .init(name: "a", executionPosition: 0, offset: 1,
                  size: 2, shape: [1], dtype: .bf16),
            .init(name: "b", executionPosition: 1,
                  offset: 2 * GTurboVisionFormatV1.alignmentBytes,
                  size: 2, shape: [1], dtype: .bf16),
        ]
        #expect(throws: GTurboFormatError.self) {
            try GTurboVisionManifestCodec.encode(
                VisionFormatFixture.manifest(tensors: invalid))
        }
    }

    @Test func rejectsInvalidStorageSizeAndQuantization() {
        let badSize = [GTurboVisionTensorRegionV1(
            name: "a", executionPosition: 0, offset: 0,
            size: 3, shape: [1], dtype: .bf16)]
        #expect(throws: GTurboFormatError.self) {
            try GTurboVisionManifestCodec.encode(
                VisionFormatFixture.manifest(tensors: badSize))
        }

        let badQuant = [GTurboVisionTensorRegionV1(
            name: "a", executionPosition: 0, offset: 0,
            size: 4, shape: [1], dtype: .u32)]
        #expect(throws: GTurboFormatError.self) {
            try GTurboVisionManifestCodec.encode(
                VisionFormatFixture.manifest(tensors: badQuant))
        }
    }

    @Test func rejectsUnexpectedPayloadOrReceiptFile() {
        let original = VisionFormatFixture.manifest()
        var files = original.files
        files["extra.bin"] = .init(size: 1, sha256: VisionFormatFixture.zeroSHA)
        let changed = GTurboVisionManifestV1(
            modelID: original.modelID, sourceRevision: original.sourceRevision,
            sourceIndexSha256: original.sourceIndexSha256,
            processorConfigSha256: original.processorConfigSha256,
            compatibleTextSourceSnapshotHash: original.compatibleTextSourceSnapshotHash,
            compatibleTextManifestSha256: original.compatibleTextManifestSha256,
            files: files, tensors: original.tensors)
        #expect(throws: GTurboFormatError.self) {
            try GTurboVisionManifestCodec.encode(changed)
        }

        let receipt = VisionFormatFixture.receipt()
        let missing = GTurboVisionReceiptV1(
            manifestSha256: receipt.manifestSha256,
            companionDirectoryPath: receipt.companionDirectoryPath,
            compatibleTextManifestSha256: receipt.compatibleTextManifestSha256,
            sourceRepoID: receipt.sourceRepoID,
            sourceRevision: receipt.sourceRevision,
            verificationTimestamp: receipt.verificationTimestamp,
            toolVersion: receipt.toolVersion,
            files: [GTurboVisionFormatV1.manifestFile:
                    receipt.files[GTurboVisionFormatV1.manifestFile]!])
        #expect(throws: GTurboFormatError.self) {
            try GTurboVisionReceiptCodec.encode(missing)
        }
    }

    @Test func qwenFamilyPinsCompleteArchitectureAndArtifactKind() throws {
        let manifest = qwenManifest()
        let data = try GTurboVisionManifestCodec.encode(manifest)
        let decoded = try GTurboVisionManifestCodec.decode(data)
        #expect(decoded.resolvedFamily == .qwen36)
        #expect(decoded.artifactKind == GTurboVisionFormatV1.qwen36ArtifactKind)
        #expect(decoded.tensors.count == 333)

        var corrupt = decoded.tensors
        let first = corrupt[0]
        corrupt[0] = GTurboVisionTensorRegionV1(
            name: first.name, executionPosition: first.executionPosition,
            offset: first.offset, size: 2, shape: [1], dtype: .bf16)
        let badShape = qwenManifest(tensors: corrupt)
        #expect(throws: GTurboFormatError.self) {
            try GTurboVisionManifestCodec.encode(badShape)
        }

        let wrongArtifact = GTurboVisionManifestV1(
            artifactKind: GTurboVisionFormatV1.artifactKind,
            family: .qwen36,
            modelID: manifest.modelID,
            sourceRevision: manifest.sourceRevision,
            sourceIndexSha256: manifest.sourceIndexSha256,
            processorConfigSha256: manifest.processorConfigSha256,
            compatibleTextSourceSnapshotHash: manifest.compatibleTextSourceSnapshotHash,
            compatibleTextManifestSha256: manifest.compatibleTextManifestSha256,
            files: manifest.files, tensors: manifest.tensors)
        #expect(throws: GTurboFormatError.self) {
            try GTurboVisionManifestCodec.encode(wrongArtifact)
        }
    }

    private func qwenManifest(
        tensors supplied: [GTurboVisionTensorRegionV1]? = nil
    ) -> GTurboVisionManifestV1 {
        var shapes: [(String, [UInt64])] = [
            ("vision_tower.patch_embed.proj.weight", [1_152, 2, 16, 16, 3]),
            ("vision_tower.patch_embed.proj.bias", [1_152]),
            ("vision_tower.pos_embed.weight", [2_304, 1_152]),
        ]
        for layer in 0..<27 {
            let p = "vision_tower.blocks.\(layer)."
            shapes += [
                (p + "attn.qkv.weight", [3_456, 1_152]),
                (p + "attn.qkv.bias", [3_456]),
                (p + "attn.proj.weight", [1_152, 1_152]),
                (p + "attn.proj.bias", [1_152]),
                (p + "mlp.linear_fc1.weight", [4_304, 1_152]),
                (p + "mlp.linear_fc1.bias", [4_304]),
                (p + "mlp.linear_fc2.weight", [1_152, 4_304]),
                (p + "mlp.linear_fc2.bias", [1_152]),
                (p + "norm1.weight", [1_152]),
                (p + "norm1.bias", [1_152]),
                (p + "norm2.weight", [1_152]),
                (p + "norm2.bias", [1_152]),
            ]
        }
        shapes += [
            ("vision_tower.merger.norm.weight", [1_152]),
            ("vision_tower.merger.norm.bias", [1_152]),
            ("vision_tower.merger.linear_fc1.weight", [4_608, 4_608]),
            ("vision_tower.merger.linear_fc1.bias", [4_608]),
            ("vision_tower.merger.linear_fc2.weight", [2_048, 4_608]),
            ("vision_tower.merger.linear_fc2.bias", [2_048]),
        ]
        var offset: UInt64 = 0
        let tensors = supplied ?? shapes.enumerated().map { index, item in
            offset = ((offset + GTurboVisionFormatV1.alignmentBytes - 1)
                / GTurboVisionFormatV1.alignmentBytes)
                * GTurboVisionFormatV1.alignmentBytes
            let elements = item.1.reduce(UInt64(1), *)
            defer { offset += elements * 2 }
            return GTurboVisionTensorRegionV1(
                name: item.0, executionPosition: index,
                offset: offset, size: elements * 2,
                shape: item.1, dtype: .bf16)
        }
        let size = tensors.map { $0.offset + $0.size }.max() ?? 0
        return GTurboVisionManifestV1(
            artifactKind: GTurboVisionFormatV1.qwen36ArtifactKind,
            family: .qwen36,
            modelID: "mlx-community/Qwen3.6-35B-A3B-4bit",
            sourceRevision: "fixture-revision",
            sourceIndexSha256: VisionFormatFixture.zeroSHA,
            processorConfigSha256: VisionFormatFixture.zeroSHA,
            compatibleTextSourceSnapshotHash: "qwen-text-snapshot",
            compatibleTextManifestSha256: VisionFormatFixture.zeroSHA,
            files: [
                GTurboVisionFormatV1.weightsFile: .init(
                    size: size, sha256: VisionFormatFixture.zeroSHA),
                GTurboVisionFormatV1.processorFile: .init(
                    size: 2, sha256: VisionFormatFixture.zeroSHA),
            ],
            tensors: tensors)
    }
}
