import Foundation

package struct GTurboManifestFileV1: Codable, Equatable, Sendable {
    package let size: UInt64
    package let sha256: String

    package init(size: UInt64, sha256: String) {
        self.size = size
        self.sha256 = sha256
    }
}

package struct GTurboManifestArchV1: Codable, Equatable, Sendable {
    package let hiddenSize: Int
    package let ffnIntermediate: Int
    package let moeIntermediateSize: Int
    package let numHeads: Int
    package let numKVHeads: Int
    package let numFullKVHeads: Int
    package let headDim: Int
    package let fullHeadDim: Int
    package let vocabSize: Int
    package let slidingWindow: Int
    package let finalLogitSoftcap: Double
    package let ropeTheta: Double
    package let fullRopeTheta: Double
    package let partialRotaryFactor: Double
    package let numLayers: Int
    package let numExperts: Int
    package let topKExperts: Int
    package let tieWordEmbeddings: Bool
    package let attentionKEqV: Bool
    package let hiddenActivation: String
    package let fullAttentionLayerMask: [Int]
    package let hiddenSizePerLayerInput: Int?
    package let vocabSizePerLayerInput: Int?
    package let numKVSharedLayers: Int?
    /// First layer whose MLP is twice as wide, absent when none is.
    package let ffnDoubleWideFromLayer: Int?

    // Family extension fields. Gemma 4 manifests omit every one of them, so
    // they stay byte-identical to the pre-family format; a reader treats
    // absence as the Gemma 4 defaults.
    package let family: String?
    package let variant: String?
    package let feedForwardKind: String?
    package let attnOutputGate: Bool?
    package let attentionScale: Double?
    package let embeddingScaledBySqrtHidden: Bool?
    package let routerScaled: Bool?
    package let ffnSandwichNorms: Bool?
    package let sharedExpertGated: Bool?
    package let ropeNeoxSubdim: Bool?
    package let linearNumKHeads: Int?
    package let linearNumVHeads: Int?
    package let linearKeyHeadDim: Int?
    package let linearValueHeadDim: Int?
    package let linearConvKernelSize: Int?

    package init(hiddenSize: Int, ffnIntermediate: Int, moeIntermediateSize: Int,
                 numHeads: Int, numKVHeads: Int, numFullKVHeads: Int,
                 headDim: Int, fullHeadDim: Int, vocabSize: Int,
                 slidingWindow: Int, finalLogitSoftcap: Double,
                 ropeTheta: Double, fullRopeTheta: Double,
                 partialRotaryFactor: Double, numLayers: Int, numExperts: Int,
                 topKExperts: Int, tieWordEmbeddings: Bool, attentionKEqV: Bool,
                 hiddenActivation: String, fullAttentionLayerMask: [Int],
                 hiddenSizePerLayerInput: Int? = nil,
                 vocabSizePerLayerInput: Int? = nil,
                 numKVSharedLayers: Int? = nil,
                 ffnDoubleWideFromLayer: Int? = nil,
                 family: String? = nil,
                 variant: String? = nil,
                 feedForwardKind: String? = nil,
                 attnOutputGate: Bool? = nil,
                 attentionScale: Double? = nil,
                 embeddingScaledBySqrtHidden: Bool? = nil,
                 routerScaled: Bool? = nil,
                 ffnSandwichNorms: Bool? = nil,
                 sharedExpertGated: Bool? = nil,
                 ropeNeoxSubdim: Bool? = nil,
                 linearNumKHeads: Int? = nil,
                 linearNumVHeads: Int? = nil,
                 linearKeyHeadDim: Int? = nil,
                 linearValueHeadDim: Int? = nil,
                 linearConvKernelSize: Int? = nil) {
        self.hiddenSize = hiddenSize
        self.ffnIntermediate = ffnIntermediate
        self.moeIntermediateSize = moeIntermediateSize
        self.numHeads = numHeads
        self.numKVHeads = numKVHeads
        self.numFullKVHeads = numFullKVHeads
        self.headDim = headDim
        self.fullHeadDim = fullHeadDim
        self.vocabSize = vocabSize
        self.slidingWindow = slidingWindow
        self.finalLogitSoftcap = finalLogitSoftcap
        self.ropeTheta = ropeTheta
        self.fullRopeTheta = fullRopeTheta
        self.partialRotaryFactor = partialRotaryFactor
        self.numLayers = numLayers
        self.numExperts = numExperts
        self.topKExperts = topKExperts
        self.tieWordEmbeddings = tieWordEmbeddings
        self.attentionKEqV = attentionKEqV
        self.hiddenActivation = hiddenActivation
        self.fullAttentionLayerMask = fullAttentionLayerMask
        self.hiddenSizePerLayerInput = hiddenSizePerLayerInput
        self.vocabSizePerLayerInput = vocabSizePerLayerInput
        self.numKVSharedLayers = numKVSharedLayers
        self.ffnDoubleWideFromLayer = ffnDoubleWideFromLayer
        self.family = family
        self.variant = variant
        self.feedForwardKind = feedForwardKind
        self.attnOutputGate = attnOutputGate
        self.attentionScale = attentionScale
        self.embeddingScaledBySqrtHidden = embeddingScaledBySqrtHidden
        self.routerScaled = routerScaled
        self.ffnSandwichNorms = ffnSandwichNorms
        self.sharedExpertGated = sharedExpertGated
        self.ropeNeoxSubdim = ropeNeoxSubdim
        self.linearNumKHeads = linearNumKHeads
        self.linearNumVHeads = linearNumVHeads
        self.linearKeyHeadDim = linearKeyHeadDim
        self.linearValueHeadDim = linearValueHeadDim
        self.linearConvKernelSize = linearConvKernelSize
    }
}

package struct GTurboManifestQuantSlotV1: Codable, Equatable, Sendable {
    package let weightBits: Int
    package let scheme: String
    package let scaleType: String
    package let biasType: String
    package let groupSize: Int

    package init(weightBits: Int, scheme: String, scaleType: String,
                 biasType: String, groupSize: Int) {
        self.weightBits = weightBits
        self.scheme = scheme
        self.scaleType = scaleType
        self.biasType = biasType
        self.groupSize = groupSize
    }
}

package struct GTurboManifestQuantV1: Codable, Equatable, Sendable {
    package let embedding: GTurboManifestQuantSlotV1
    package let attention: GTurboManifestQuantSlotV1
    package let router: GTurboManifestQuantSlotV1
    package let sharedExpert: GTurboManifestQuantSlotV1
    package let routedExpert: GTurboManifestQuantSlotV1

    package init(embedding: GTurboManifestQuantSlotV1,
                 attention: GTurboManifestQuantSlotV1,
                 router: GTurboManifestQuantSlotV1,
                 sharedExpert: GTurboManifestQuantSlotV1,
                 routedExpert: GTurboManifestQuantSlotV1) {
        self.embedding = embedding
        self.attention = attention
        self.router = router
        self.sharedExpert = sharedExpert
        self.routedExpert = routedExpert
    }
}

package struct GTurboManifestV1: Codable, Equatable, Sendable {
    package let magic: String
    package let versionMajor: Int
    package let versionMinor: Int
    package let flags: [String: Bool]
    package let modelID: String
    package let sourceSnapshotHash: String?
    package let arch: GTurboManifestArchV1
    package let quant: GTurboManifestQuantV1?
    package let files: [String: GTurboManifestFileV1]
    package let expertsPerLayer: Int
    package let numLayers: Int
    package let expertStride: UInt64
    package let bitWidthOverridesHonored: Int?

    package init(magic: String = GTurboFormatV1.magic,
                 versionMajor: Int = GTurboFormatV1.versionMajor,
                 versionMinor: Int = GTurboFormatV1.versionMinor,
                 flags: [String: Bool], modelID: String,
                 sourceSnapshotHash: String?, arch: GTurboManifestArchV1,
                 quant: GTurboManifestQuantV1?,
                 files: [String: GTurboManifestFileV1],
                 expertsPerLayer: Int, numLayers: Int, expertStride: UInt64,
                 bitWidthOverridesHonored: Int?) {
        self.magic = magic
        self.versionMajor = versionMajor
        self.versionMinor = versionMinor
        self.flags = flags
        self.modelID = modelID
        self.sourceSnapshotHash = sourceSnapshotHash
        self.arch = arch
        self.quant = quant
        self.files = files
        self.expertsPerLayer = expertsPerLayer
        self.numLayers = numLayers
        self.expertStride = expertStride
        self.bitWidthOverridesHonored = bitWidthOverridesHonored
    }
}

package enum GTurboManifestCodec {
    package static func decode(_ data: Data) throws -> GTurboManifestV1 {
        let manifest = try decodeUnchecked(data)
        try validate(manifest)
        return manifest
    }

    package static func decodeUnchecked(_ data: Data) throws -> GTurboManifestV1 {
        let manifest: GTurboManifestV1
        do { manifest = try JSONDecoder().decode(GTurboManifestV1.self, from: data) }
        catch { throw GTurboFormatError.invalid(field: "manifest.json", reason: "\(error)") }
        return manifest
    }

    package static func encode(_ manifest: GTurboManifestV1) throws -> Data {
        try validate(manifest)
        let encoder = JSONEncoder()
        do {
            let object = try JSONSerialization.jsonObject(with: encoder.encode(manifest))
            return try JSONSerialization.data(
                withJSONObject: object,
                options: [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes])
        } catch {
            throw GTurboFormatError.invalid(field: "manifest.json", reason: "\(error)")
        }
    }

    package static func validate(_ manifest: GTurboManifestV1) throws {
        guard manifest.magic == GTurboFormatV1.magic else {
            throw GTurboFormatError.invalid(field: "manifest.magic", reason: "expected GTURBO")
        }
        guard manifest.versionMajor == GTurboFormatV1.versionMajor,
              manifest.versionMinor >= 0 else {
            throw GTurboFormatError.invalid(field: "manifest.version", reason: "unsupported version")
        }
        for flag in manifest.flags.keys where !GTurboFormatV1.knownFlags.contains(flag) {
            throw GTurboFormatError.invalid(field: "manifest.flags.\(flag)", reason: "unknown v1 flag")
        }
        for (flag, minimumMinor) in GTurboFormatV1.minimumMinorByFlag
        where manifest.flags[flag] == true && manifest.versionMinor < minimumMinor {
            throw GTurboFormatError.invalid(
                field: "manifest.flags.\(flag)",
                reason: "requires format 1.\(minimumMinor) or newer")
        }
        let feedForwardKind = manifest.arch.feedForwardKind ?? "moe"
        guard feedForwardKind == "moe" || feedForwardKind == "dense" else {
            throw GTurboFormatError.invalid(
                field: "manifest.arch.feedForwardKind", reason: "unknown feed-forward kind")
        }
        let isDense = feedForwardKind == "dense"
        guard (manifest.flags["denseFFN"] == true) == isDense else {
            throw GTurboFormatError.invalid(
                field: "manifest.flags.denseFFN",
                reason: "must match manifest.arch.feedForwardKind")
        }
        if isDense, manifest.flags["mxfp4Weights"] == true {
            throw GTurboFormatError.invalid(
                field: "manifest.flags.mxfp4Weights",
                reason: "MXFP4 is only valid for streamed routed experts")
        }
        guard !manifest.modelID.isEmpty,
              manifest.numLayers > 0 else {
            throw GTurboFormatError.invalid(field: "manifest", reason: "invalid dimensions or stride")
        }
        if isDense {
            guard manifest.expertsPerLayer == 0,
                  manifest.expertStride == 0,
                  manifest.arch.numExperts == 0,
                  manifest.arch.topKExperts == 0,
                  manifest.arch.moeIntermediateSize == 0,
                  manifest.flags["streamingPresent"] != true,
                  !manifest.files.keys.contains(where: {
                      $0.hasPrefix("packed_experts/")
                  }) else {
                throw GTurboFormatError.invalid(
                    field: "manifest.arch", reason: "dense FFN carries MoE metadata")
            }
        } else {
            guard manifest.expertsPerLayer > 0,
                  manifest.expertStride > 0,
                  manifest.expertStride % GTurboFormatV1.alignmentBytes == 0,
                  manifest.arch.numExperts == manifest.expertsPerLayer,
                  manifest.arch.moeIntermediateSize > 0,
                  manifest.arch.topKExperts > 0,
                  manifest.arch.topKExperts <= manifest.arch.numExperts else {
                throw GTurboFormatError.invalid(
                    field: "manifest.arch", reason: "invalid MoE streaming metadata")
            }
        }
        guard manifest.arch.numLayers == manifest.numLayers else {
            throw GTurboFormatError.invalid(
                field: "manifest.arch", reason: "layer count disagrees with manifest")
        }
        let arch = manifest.arch
        guard arch.hiddenSize > 0, arch.ffnIntermediate > 0,
              arch.numHeads > 0,
              arch.numKVHeads > 0, arch.numFullKVHeads > 0,
              arch.headDim > 0, arch.fullHeadDim > 0,
              arch.vocabSize > 0,
              // A sliding window is only meaningful when some layer uses it;
              // an all-full/linear graph (Qwen 3.6) declares 0.
              arch.slidingWindow > 0 || !arch.fullAttentionLayerMask.contains(0),
              arch.slidingWindow >= 0,
              arch.finalLogitSoftcap.isFinite,
              arch.ropeTheta.isFinite, arch.ropeTheta > 0,
              arch.fullRopeTheta.isFinite, arch.fullRopeTheta > 0,
              arch.partialRotaryFactor.isFinite,
              arch.partialRotaryFactor >= 0, arch.partialRotaryFactor <= 1,
              !arch.hiddenActivation.isEmpty,
              arch.fullAttentionLayerMask.count == arch.numLayers,
              // 0 = sliding-window attention, 1 = full attention,
              // 2 = linear (gated-DeltaNet) attention.
              arch.fullAttentionLayerMask.allSatisfy({ (0...2).contains($0) }) else {
            throw GTurboFormatError.invalid(
                field: "manifest.arch", reason: "invalid architecture values")
        }
        let perLayerHidden = arch.hiddenSizePerLayerInput ?? 0
        let perLayerVocab = arch.vocabSizePerLayerInput ?? 0
        let sharedKVLayers = arch.numKVSharedLayers ?? 0
        guard perLayerHidden >= 0, perLayerVocab >= 0,
              (perLayerHidden == 0) == (perLayerVocab == 0),
              sharedKVLayers >= 0, sharedKVLayers < arch.numLayers else {
            throw GTurboFormatError.invalid(
                field: "manifest.arch", reason: "invalid PLE or shared-KV values")
        }
        if let quant = manifest.quant {
            for (name, slot) in [
                ("embedding", quant.embedding),
                ("attention", quant.attention),
                ("router", quant.router),
                ("sharedExpert", quant.sharedExpert),
                ("routedExpert", quant.routedExpert),
            ] {
                guard slot.weightBits > 0, slot.weightBits <= 32,
                      slot.groupSize > 0,
                      !slot.scheme.isEmpty, !slot.scaleType.isEmpty,
                      !slot.biasType.isEmpty else {
                    throw GTurboFormatError.invalid(
                        field: "manifest.quant.\(name)", reason: "invalid quantization values")
                }
            }
            let routed = quant.routedExpert
            let routedIsMXFP4 = routed.scheme.lowercased() == "mxfp4"
            let hasMXFP4Flag = manifest.flags["mxfp4Weights"] == true
            guard routedIsMXFP4 == hasMXFP4Flag else {
                throw GTurboFormatError.invalid(
                    field: "manifest.flags.mxfp4Weights",
                    reason: "must match routed-expert quantization")
            }
            if routedIsMXFP4 {
                guard routed.weightBits == 4,
                      routed.groupSize == 32,
                      routed.scaleType.lowercased() == "ue8m0",
                      routed.biasType.lowercased() == "none" else {
                    throw GTurboFormatError.invalid(
                        field: "manifest.quant.routedExpert",
                        reason: "MXFP4 requires E2M1 blocks with UE8M0 scales")
                }
            }
        } else if manifest.flags["mxfp4Weights"] == true {
            throw GTurboFormatError.invalid(
                field: "manifest.quant", reason: "MXFP4 metadata is required")
        }
        let reservedFiles: Set<String> = ["manifest.json", "verified-install.json"]
        let filePaths = manifest.files.keys.sorted()
        var canonicalPaths: [String: String] = [:]
        for path in filePaths {
            try GTurboPathValidator.validateRelativePath(path, field: "manifest.files.\(path)")
            let key = GTurboPathValidator.appleFilesystemKey(path)
            guard canonicalPaths.updateValue(path, forKey: key) == nil else {
                throw GTurboFormatError.invalid(
                    field: "manifest.files.\(path)", reason: "filesystem-equivalent duplicate path")
            }
            guard key != "tokenizer",
                  !reservedFiles.contains(key),
                  !reservedFiles.contains(where: { key.hasPrefix("\($0)/") }) else {
                throw GTurboFormatError.invalid(
                    field: "manifest.files.\(path)", reason: "reserved artifact filename")
            }
            let entry = manifest.files[path]!
            guard entry.sha256.count == 64,
                  entry.sha256.unicodeScalars.allSatisfy({ scalar in
                      ("0"..."9").contains(Character(String(scalar)))
                          || ("a"..."f").contains(Character(String(scalar)))
                          || ("A"..."F").contains(Character(String(scalar)))
                  }) else {
                throw GTurboFormatError.invalid(
                    field: "manifest.files.\(path).sha256", reason: "expected 64 hexadecimal characters")
            }
        }
        for (key, path) in canonicalPaths {
            var components = key.split(separator: "/").map(String.init)
            while components.count > 1 {
                _ = components.removeLast()
                let ancestor = components.joined(separator: "/")
                if canonicalPaths[ancestor] != nil {
                    throw GTurboFormatError.invalid(
                        field: "manifest.files.\(path)",
                        reason: "file path collides with a directory prefix")
                }
            }
        }
    }

}
