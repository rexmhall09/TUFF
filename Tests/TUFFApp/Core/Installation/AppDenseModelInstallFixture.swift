import Darwin
import Foundation
import TUFFEngine
@testable import TUFFAppCore

/// A complete dense `.gturbo` directory, shaped like a Gemma 4 E4B install.
///
/// Dense installs carry no `packed_experts/` directory at all: the format
/// rejects a dense manifest that references one. That is exactly why the probe
/// must not demand an expert layout, so this fixture deliberately omits it.
func makeCompleteDenseModelInstall(_ tag: String) throws -> URL {
    let descriptor = AppModelInstallDescriptor.gemma4E4B
    let directory = FileManager.default.temporaryDirectory
        .appendingPathComponent("tuff-dense-\(tag)-\(UUID().uuidString).gturbo")
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)

    let arch = ArchConfig.gemma4_E4B
    let files: [String: Any] = [
        "model_weights.bin": ["size": 0, "sha256": String(repeating: "0", count: 64)],
    ]
    let archObject: [String: Any] = [
        "hiddenSize": arch.hiddenSize,
        "ffnIntermediate": arch.intermediateSize,
        "moeIntermediateSize": 0,
        "numHeads": arch.numHeads,
        "numKVHeads": arch.numKVHeads,
        "numFullKVHeads": arch.numFullKVHeads,
        "headDim": arch.headDim,
        "fullHeadDim": arch.fullHeadDim,
        "vocabSize": arch.vocabSize,
        "slidingWindow": arch.slidingWindow,
        "finalLogitSoftcap": arch.finalLogitSoftcap,
        "ropeTheta": arch.ropeTheta,
        "fullRopeTheta": arch.fullRopeTheta,
        "partialRotaryFactor": arch.partialRotaryFactor,
        "numLayers": arch.numLayers,
        "numExperts": 0,
        "topKExperts": 0,
        "tieWordEmbeddings": arch.tieWordEmbeddings,
        "attentionKEqV": arch.attentionKEqV,
        "hiddenActivation": arch.hiddenActivation,
        "fullAttentionLayerMask": arch.fullAttentionLayerMask.map(Int.init),
        "feedForwardKind": "dense",
        "variant": arch.variant.rawValue,
        "hiddenSizePerLayerInput": arch.hiddenSizePerLayerInput,
        "vocabSizePerLayerInput": arch.vocabSizePerLayerInput,
        "numKVSharedLayers": arch.numKVSharedLayers,
    ]
    let manifest: [String: Any] = [
        "magic": "GTURBO",
        "versionMajor": 1,
        "versionMinor": 1,
        "flags": ["streamingPresent": false, "denseFFN": true],
        "modelID": "test/gemma-4-e4b",
        "sourceSnapshotHash": "sha256:" + descriptor.sourceIndexSHA256,
        "quant": [
            "embedding": denseQuantSlot(4),
            "attention": denseQuantSlot(4),
            "router": denseQuantSlot(8),
            "sharedExpert": denseQuantSlot(4),
            "routedExpert": denseQuantSlot(4),
        ],
        "arch": archObject,
        "files": files,
        "expertsPerLayer": 0,
        "numLayers": arch.numLayers,
        "expertStride": 0,
    ]
    let manifestData = try JSONSerialization.data(
        withJSONObject: manifest, options: [.sortedKeys])
    try manifestData.write(to: directory.appendingPathComponent("manifest.json"))
    let receipt: [String: Any] = [
        "schemaVersion": 1,
        "manifestSha256": Sha256Verifier.hashData(manifestData),
        "modelDirectoryPath": directory.standardizedFileURL.path,
        "verificationTimestamp": "2026-08-26T00:00:00Z",
        "toolVersion": "TUFFAppCoreTests",
        "files": [:],
    ]
    let receiptData = try JSONSerialization.data(withJSONObject: receipt, options: [.sortedKeys])
    try receiptData.write(to: directory.appendingPathComponent("verified-install.json"))
    return directory
}

private func denseQuantSlot(_ weightBits: Int) -> [String: Any] {
    [
        "weightBits": weightBits,
        "scheme": "affine",
        "scaleType": "bf16",
        "biasType": "bf16",
        "groupSize": Quantization.groupSize,
    ]
}
