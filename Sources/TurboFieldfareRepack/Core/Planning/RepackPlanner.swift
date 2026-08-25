import Foundation
import TurboFieldfareFormat

/// On-disk page alignment unit for `.gturbo` files. Fixed at 16 KB regardless
/// of host page size — the format is the contract, not the kernel.
enum Layout {
    static let pageBytes = GTurboFormatV1.alignmentBytes
}

// MARK: - Plan data types

struct ResidentEntry: Sendable {
    let name: String
    /// dtype byte for IndexEntry: 0 = U32, 1 = BF16, 2 = FP16, 3 = FP32.
    let dtype: UInt8
    /// Logical shape after dequant (max rank 4; trailing zeros).
    let logicalShape4: [UInt32]
    /// File offset where the (packed) weight bytes start.
    let fileOffset: UInt64
    /// Size in bytes of the weight bytes.
    let sizeBytes: UInt64
    /// Offset where BF16 scales start (0 if none).
    let scaleOffset: UInt64
    let scaleSize: UInt64
    /// Offset where BF16 biases start (0 if none).
    let biasOffset: UInt64
    let biasSize: UInt64
    /// Quantization spec (nil for unquantized scalars/norms).
    let quantSpec: QuantSpec?

    /// Source tensors that supply this entry's bytes.
    let sourceWeight: SourceTensor
    let sourceScales: SourceTensor?
    let sourceBiases: SourceTensor?
}

struct ResidentFilePlan: Sendable {
    let path: String
    let entries: [ResidentEntry]
    let stringTable: [UInt8]
    let stringTableOffsets: [UInt32]   // per-entry offsets into the table
    let indexSize: UInt64              // header + entries + table + padding
    let residentSize: UInt64           // tensor payload region
    var totalSize: UInt64 { indexSize + residentSize }
}

struct PerExpertTensorSlice: Sendable {
    let role: String                   // affine roles or GPT-OSS "mlp1"/"mlp2"
    let component: String              // weights | scales | bias/biases
    let dtype: UInt8                   // 0=U32, 1=BF16, 4=U8
    let logicalShape: [UInt64]         // per-expert logical shape
    let offsetInExpertBlob: UInt64     // within each expert blob
    let sizeInExpertBlob: UInt64
    /// For each expert e (0..<expertsPerLayer): source byte offset & size.
    let sourceOffsetPerExpert: UInt64  // stride per expert in source
    let sourceTensor: SourceTensor
    let bitsForWeights: Int?           // 4 for routed expert weight; nil for scales/biases
}

struct LayerFilePlan: Sendable {
    let layerIndex: Int
    let path: String
    let expertsPerLayer: Int
    let expertStride: UInt64
    let subTensors: [PerExpertTensorSlice]  // 9 entries: gate/up/down × {weights, scales, biases}
    var fileSize: UInt64 { UInt64(expertsPerLayer) * expertStride }

    func physicalRank(for logicalExpert: Int) -> Int {
        logicalExpert
    }

    init(layerIndex: Int,
                path: String,
                expertsPerLayer: Int,
                expertStride: UInt64,
                subTensors: [PerExpertTensorSlice]) {
        self.layerIndex = layerIndex
        self.path = path
        self.expertsPerLayer = expertsPerLayer
        self.expertStride = expertStride
        self.subTensors = subTensors
    }
}

struct RepackPlan: Sendable {
    let arch: ArchInfo
    let baseMode: String                  // "affine"
    let baseGroupSize: Int                // 64
    let bitsOverrideCount: Int
    let resident: ResidentFilePlan
    let layers: [LayerFilePlan]
    let matchedModelID: String?
    let excludedMultimodalTensorNames: [String]
}

// MARK: - Planner

struct VisionPackPlan: Sendable {
    struct Entry: Sendable {
        let source: SourceTensor
        let executionPosition: Int
        let fileOffset: UInt64
        let quantSpec: QuantSpec?
        /// Affine group size, pinned to 64 by `planVisionCompanion`.
        let groupSize: Int
    }

    let entries: [Entry]
    let weightsFileSize: UInt64
    let sourcePayloadBytes: UInt64
}

enum RepackPlanner {

    static func planVisionCompanion(
        meta: IndexLoader.SourceMetadata,
        shardHeaders: [Safetensors.Header]
    ) throws -> VisionPackPlan {
        guard meta.baseBits == 4, meta.baseGroupSize == 64,
              meta.baseMode.lowercased() == "affine" else {
            throw RepackError.configurationInvalid(
                detail: "vision companion requires MLX affine 4-bit group-64 source metadata")
        }

        guard let modelID = SourceFingerprint.modelID(
            forIndexSha256: meta.indexSha256Hex) else {
            throw RepackError.sourceFingerprintRejected(
                path: meta.indexPath, sha256: meta.indexSha256Hex)
        }
        let isQwen36 = modelID == SupportedModelSource.qwen36.modelID
        let tensors = shardHeaders.flatMap(\.tensors).filter {
            isMultimodalTensorName($0.name)
        }
        let expectedTensorCount = isQwen36 ? 333 : 358
        let expectedSourceBytes: UInt64 = isQwen36 ? 893_142_496 : 1_140_925_536
        guard tensors.count == expectedTensorCount else {
            throw RepackError.configurationInvalid(
                detail: "expected \(expectedTensorCount) vision tensors for \(modelID), found \(tensors.count)")
        }
        let sourceBytes = tensors.reduce(UInt64(0)) { $0 + $1.sizeBytes }
        guard sourceBytes == expectedSourceBytes else {
            throw RepackError.configurationInvalid(
                detail: "expected \(expectedSourceBytes) vision bytes for \(modelID), found \(sourceBytes)")
        }

        let ordered = tensors.sorted { visionExecutionKey($0.name) < visionExecutionKey($1.name) }
        var offset: UInt64 = 0
        var entries: [VisionPackPlan.Entry] = []
        entries.reserveCapacity(ordered.count)
        for (position, tensor) in ordered.enumerated() {
            offset = visionAlignUp(offset, to: GTurboVisionFormatV1.alignmentBytes)
            let quantSpec: QuantSpec?
            if tensor.dtype == .u32 {
                let spec = IndexLoader.quantSpec(forTensor: tensor.name, meta: meta)
                // Group size is not carried per tensor in this tree; the base
                // group was already pinned to 64 above.
                guard spec.bits == 4 else {
                    throw RepackError.configurationInvalid(
                        detail: "unsupported vision quantization for \(tensor.name)")
                }
                quantSpec = spec
            } else {
                guard tensor.dtype == .bf16 else {
                    throw RepackError.configurationInvalid(
                        detail: "unsupported vision dtype for \(tensor.name)")
                }
                quantSpec = nil
            }
            entries.append(.init(source: tensor,
                                 executionPosition: position,
                                 fileOffset: offset,
                                 quantSpec: quantSpec,
                                 groupSize: meta.baseGroupSize))
            offset += tensor.sizeBytes
        }
        return VisionPackPlan(entries: entries,
                              weightsFileSize: offset,
                              sourcePayloadBytes: sourceBytes)
    }

    private static func visionExecutionKey(_ name: String) -> String {
        if name.hasPrefix("vision_tower.patch_embed.") {
            return "0000/\(name)"
        }
        if name.hasPrefix("vision_tower.patch_embedder.") {
            return "0000/\(name)"
        }
        if let layer = layerIndex(in: name), name.hasPrefix("vision_tower.blocks.") {
            return String(format: "1000/%03d/%@", layer, name)
        }
        if let layer = layerIndex(in: name), name.hasPrefix("vision_tower.encoder.layers.") {
            return String(format: "1000/%03d/%@", layer, name)
        }
        if name.hasPrefix("vision_tower.merger.") || name == "vision_tower.pos_embed.weight" {
            return "3000/\(name)"
        }
        if name == "vision_tower.std_bias" || name == "vision_tower.std_scale" {
            return "2000/\(name)"
        }
        if name.hasPrefix("embed_vision.") {
            return "3000/\(name)"
        }
        return "9999/\(name)"
    }

    private static func visionAlignUp(_ value: UInt64, to alignment: UInt64) -> UInt64 {
        ((value + alignment - 1) / alignment) * alignment
    }

    /// Classify a tensor name. Routed-expert tensors split off the LM bucket.
    enum Bucket: Equatable {
        case lmResident
        case routedExpert(role: String, layer: Int)   // role = "gate"|"up"|"down"
        case excludedMultimodal
        case unknown
    }

    static func classify(_ name: String, numLayers: Int,
                         family: RepackModelFamily) -> Bucket {
        if family == .gptOss,
           (name == "lm_head.weight" || name.hasPrefix("model.")) {
            if name.contains(".mlp.experts."),
               let layer = layerIndex(in: name),
               layer >= 0 && layer < numLayers {
                return .routedExpert(
                    role: name.contains("gate_up_proj") ? "mlp1" : "mlp2",
                    layer: layer)
            }
            return .lmResident
        }
        if name.hasPrefix("language_model.") {
            // Routed expert?
            if let role = routedExpertRole(in: name, family: family),
               let layer = layerIndex(in: name),
               layer >= 0 && layer < numLayers {
                return .routedExpert(role: role, layer: layer)
            }
            return .lmResident
        }
        if isMultimodalTensorName(name) {
            return .excludedMultimodal
        }
        return .unknown
    }

    private static func routedExpertRole(in name: String,
                                         family: RepackModelFamily) -> String? {
        let routedContainer: String
        switch family {
        case .gemma4: routedContainer = ".experts.switch_glu."
        case .qwen36: routedContainer = ".mlp.switch_mlp."
        case .gptOss: routedContainer = ".mlp.experts."
        }
        guard name.contains(routedContainer) else { return nil }
        if name.contains(".gate_proj.") { return "gate" }
        if name.contains(".up_proj.")   { return "up" }
        if name.contains(".down_proj.") { return "down" }
        return nil
    }

    private static func layerIndex(in name: String) -> Int? {
        // matches either language/legacy vision "...layers.<N>..." or the
        // Qwen vision tower's "...blocks.<N>...".
        guard let r = name.range(of: ".layers.") ?? name.range(of: ".blocks.") else {
            return nil
        }
        let tail = name[r.upperBound...]
        guard let dot = tail.firstIndex(of: ".") else { return nil }
        return Int(tail[tail.startIndex..<dot])
    }

    /// Build the plan from parsed shard headers + source metadata.
    /// - throws: classification + companion + override count failures.
    static func plan(meta: IndexLoader.SourceMetadata,
                            arch: ArchInfo,
                            shardHeaders: [Safetensors.Header],
                            outputDir: String) throws -> RepackPlan {

        // Companion tensors may live in different shards, so resolve them
        // through one global registry.
        var registry: [String: SourceTensor] = [:]
        registry.reserveCapacity(meta.weightMap.count)
        for h in shardHeaders {
            for t in h.tensors { registry[t.name] = t }
        }

        // Source allowlisting owns exact fingerprint validation. Preserve the
        // declared override count for the output manifest audit.
        let bitsOverrideCount = meta.bitsOverrides.count

        if arch.family == .gptOss {
            return try planGPTOSS(
                meta: meta,
                arch: arch,
                registry: registry,
                outputDir: outputDir,
                bitsOverrideCount: bitsOverrideCount)
        }

        var lmResidentBases: [String] = []
        var excludedMultimodalNames: [String] = []
        var routedByLayerAndRole: [Int: [String: String]] = [:]
        for (name, _) in registry {
            if isMultimodalTensorName(name) {
                excludedMultimodalNames.append(name)
            }
            if name.hasSuffix(".scales") || name.hasSuffix(".biases") { continue }
            let b = classify(name, numLayers: arch.numLayers, family: arch.family)
            switch b {
            case .lmResident:                   lmResidentBases.append(name)
            case .routedExpert(let role, let layer):
                var byRole = routedByLayerAndRole[layer] ?? [:]
                if byRole[role] != nil {
                    throw RepackError.configurationInvalid(detail:
                        "two routed-expert tensors for layer \(layer) role \(role)")
                }
                byRole[role] = name
                routedByLayerAndRole[layer] = byRole
            case .excludedMultimodal:           continue
            case .unknown:                      throw RepackError.unknownTensorPrefix(name: name)
            }
        }

        // Sort deterministically. The LM order follows a fixed template.
        lmResidentBases.sort(by: lmResidentOrdering(family: arch.family))
        excludedMultimodalNames.sort()

        let residentPath = (outputDir as NSString).appendingPathComponent("model_weights.bin")
        let resident = try planResidentFile(path: residentPath,
                                            baseNames: lmResidentBases,
                                            registry: registry, meta: meta,
                                            family: arch.family)

        var layerPlans: [LayerFilePlan] = []
        if arch.feedForwardKind == .mixtureOfExperts {
            let layersDir = (outputDir as NSString).appendingPathComponent("packed_experts")
            layerPlans.reserveCapacity(arch.numLayers)
            for layer in 0..<arch.numLayers {
                let bundle = routedByLayerAndRole[layer] ?? [:]
                // Synthetic snapshots may legitimately have no routed experts.
                guard let gName = bundle["gate"], let uName = bundle["up"],
                      let dName = bundle["down"] else {
                    if bundle.isEmpty {
                        let name = "layer_\(String(format: "%02d", layer)).bin"
                        layerPlans.append(LayerFilePlan(
                            layerIndex: layer,
                            path: (layersDir as NSString).appendingPathComponent(name),
                            expertsPerLayer: 0,
                            expertStride: 0,
                            subTensors: []))
                        continue
                    }
                    throw RepackError.configurationInvalid(detail:
                        "layer \(layer) routed-expert bundle incomplete: \(bundle)")
                }
                let path = (layersDir as NSString)
                    .appendingPathComponent("layer_\(String(format: "%02d", layer)).bin")
                let lp = try planLayerFile(
                    path: path, layer: layer,
                    gateName: gName, upName: uName, downName: dName,
                    registry: registry, meta: meta, arch: arch)
                layerPlans.append(lp)
            }
        } else if !routedByLayerAndRole.isEmpty {
            throw RepackError.configurationInvalid(
                detail: "dense architecture contains routed-expert tensors")
        }

        let matched = SourceFingerprint.modelID(forIndexSha256: meta.indexSha256Hex)

        return RepackPlan(arch: arch,
                          baseMode: meta.baseMode,
                          baseGroupSize: meta.baseGroupSize,
                          bitsOverrideCount: bitsOverrideCount,
                          resident: resident,
                          layers: layerPlans,
                          matchedModelID: matched,
                          excludedMultimodalTensorNames: excludedMultimodalNames)
    }

    private static func isMultimodalTensorName(_ name: String) -> Bool {
        name.hasPrefix("vision_tower.") ||
            name.hasPrefix("embed_vision.") ||
            name.hasPrefix("audio_tower.") ||
            name.hasPrefix("embed_audio.") ||
            name.hasPrefix("embed_video.")
    }

    // MARK: - Resident planning

    private static func planResidentFile(path: String,
                                         baseNames: [String],
                                         registry: [String: SourceTensor],
                                         meta: IndexLoader.SourceMetadata,
                                         family: RepackModelFamily) throws
                                        -> ResidentFilePlan {
        let entryCount = baseNames.count

        var stringTable: [UInt8] = []
        var offsets: [UInt32] = []
        offsets.reserveCapacity(entryCount)
        for sourceName in baseNames {
            let n = canonicalResidentName(sourceName, family: family)
            offsets.append(UInt32(stringTable.count))
            stringTable.append(contentsOf: n.utf8)
        }

        // Index size includes the fixed header, fixed-width entries, and the
        // string table, padded to a 16 KB page boundary.
        let rawIdx = UInt64(GTurboBinary.indexHeaderBytes
            + entryCount * GTurboBinary.indexEntryBytes
            + stringTable.count)
        let indexSize = roundUpToPage(rawIdx)

        var fileCursor = indexSize
        var entries: [ResidentEntry] = []
        entries.reserveCapacity(entryCount)

        for sourceName in baseNames {
            let name = canonicalResidentName(sourceName, family: family)
            guard let weight = registry[sourceName] else {
                throw RepackError.missingTensor(name: sourceName)
            }
            let dtype = ietnyDtype(weight.dtype)
            let isQuantizedPacked = (weight.dtype == .u32)
                && sourceName.hasSuffix(".weight")

            if isQuantizedPacked {
                let base = String(sourceName.dropLast(".weight".count))
                guard let scales = registry[base + ".scales"] else {
                    throw RepackError.missingScalesCompanion(name: name)
                }
                guard let biases = registry[base + ".biases"] else {
                    throw RepackError.missingBiasesCompanion(name: name)
                }
                if scales.dtype != .bf16 || biases.dtype != .bf16 {
                    throw RepackError.dtypeMismatch(name: name,
                        detail: "expected BF16 scales/biases, got \(scales.dtype)/\(biases.dtype)")
                }
                let spec = IndexLoader.quantSpec(forTensor: sourceName, meta: meta)
                let logical = logicalShape(forPackedSource: weight.shape, bits: spec.bits)

                let wOff = fileCursor
                let wSize = weight.sizeBytes
                let sOff = wOff + wSize
                let sSize = scales.sizeBytes
                let bOff = sOff + sSize
                let bSize = biases.sizeBytes
                fileCursor = bOff + bSize

                entries.append(ResidentEntry(
                    name: name, dtype: GTurboFormatV1.DType.u32.rawValue,
                    logicalShape4: padTo4(logical),
                    fileOffset: wOff, sizeBytes: wSize,
                    scaleOffset: sOff, scaleSize: sSize,
                    biasOffset: bOff, biasSize: bSize,
                    quantSpec: spec,
                    sourceWeight: weight, sourceScales: scales, sourceBiases: biases))
            } else {
                // Unquantized (BF16 norm / scalar) — no companions.
                let off = fileCursor
                let size = weight.sizeBytes
                fileCursor = off + size

                entries.append(ResidentEntry(
                    name: name, dtype: dtype,
                    logicalShape4: padTo4(weight.shape),
                    fileOffset: off, sizeBytes: size,
                    scaleOffset: 0, scaleSize: 0,
                    biasOffset: 0, biasSize: 0,
                    quantSpec: nil,
                    sourceWeight: weight, sourceScales: nil, sourceBiases: nil))
            }
        }

        let residentSize = fileCursor - indexSize

        return ResidentFilePlan(path: path,
                                entries: entries,
                                stringTable: stringTable,
                                stringTableOffsets: offsets,
                                indexSize: indexSize,
                                residentSize: residentSize)
    }

    private static func canonicalResidentName(
        _ sourceName: String,
        family: RepackModelFamily
    ) -> String {
        guard family == .gptOss else { return sourceName }
        if sourceName == "lm_head.weight" {
            return "language_model.lm_head.weight"
        }
        return "language_model." + sourceName
    }

    // MARK: - GPT-OSS planning

    private static func planGPTOSS(
        meta: IndexLoader.SourceMetadata,
        arch: ArchInfo,
        registry: [String: SourceTensor],
        outputDir: String,
        bitsOverrideCount: Int
    ) throws -> RepackPlan {
        guard meta.baseMode.lowercased() == "mxfp4",
              meta.baseBits == 4,
              meta.baseGroupSize == 32,
              bitsOverrideCount == 0 else {
            throw RepackError.configurationInvalid(
                detail: "GPT-OSS requires MXFP4 group-32 source metadata without overrides")
        }

        let expertMarker = ".mlp.experts."
        var residentNames: [String] = []
        for name in registry.keys {
            if name.contains(expertMarker) { continue }
            guard name == "lm_head.weight" || name.hasPrefix("model.") else {
                throw RepackError.unknownTensorPrefix(name: name)
            }
            residentNames.append(name)
        }
        residentNames.sort(by: lmResidentOrdering(family: .gptOss))
        let residentPath = (outputDir as NSString)
            .appendingPathComponent("model_weights.bin")
        let resident = try planResidentFile(
            path: residentPath,
            baseNames: residentNames,
            registry: registry,
            meta: meta,
            family: .gptOss)
        guard resident.entries.allSatisfy({ $0.dtype == GTurboFormatV1.DType.bf16.rawValue }) else {
            throw RepackError.configurationInvalid(
                detail: "GPT-OSS resident tensors must be BF16")
        }

        let layersDir = (outputDir as NSString)
            .appendingPathComponent("packed_experts")
        var layers: [LayerFilePlan] = []
        layers.reserveCapacity(arch.numLayers)
        for layer in 0..<arch.numLayers {
            let prefix = "model.layers.\(layer).mlp.experts."
            let path = (layersDir as NSString).appendingPathComponent(
                "layer_\(String(format: "%02d", layer)).bin")
            layers.append(try planGPTOSSLayer(
                path: path,
                layer: layer,
                prefix: prefix,
                registry: registry,
                arch: arch))
        }

        return RepackPlan(
            arch: arch,
            baseMode: meta.baseMode,
            baseGroupSize: meta.baseGroupSize,
            bitsOverrideCount: bitsOverrideCount,
            resident: resident,
            layers: layers,
            matchedModelID: SourceFingerprint.modelID(
                forIndexSha256: meta.indexSha256Hex),
            excludedMultimodalTensorNames: [])
    }

    private static func planGPTOSSLayer(
        path: String,
        layer: Int,
        prefix: String,
        registry: [String: SourceTensor],
        arch: ArchInfo
    ) throws -> LayerFilePlan {
        let sources: [(role: String, component: String, suffix: String)] = [
            ("mlp1", "weights", "gate_up_proj_blocks"),
            ("mlp1", "scales", "gate_up_proj_scales"),
            ("mlp1", "bias", "gate_up_proj_bias"),
            ("mlp2", "weights", "down_proj_blocks"),
            ("mlp2", "scales", "down_proj_scales"),
            ("mlp2", "bias", "down_proj_bias"),
        ]
        var slices: [PerExpertTensorSlice] = []
        var cursor: UInt64 = 0
        for source in sources {
            let name = prefix + source.suffix
            guard let tensor = registry[name] else {
                throw RepackError.missingTensor(name: name)
            }
            let expectedDtype: SourceTensor.Dtype = source.component == "bias"
                ? .bf16 : .u8
            guard tensor.dtype == expectedDtype,
                  tensor.shape.first == UInt64(arch.numExperts) else {
                throw RepackError.shapeMismatch(
                    name: name,
                    detail: "expected \(expectedDtype) with leading \(arch.numExperts), got \(tensor.dtype) \(tensor.shape)")
            }
            let expectedShape: [UInt64]
            switch (source.role, source.component) {
            case ("mlp1", "weights"):
                expectedShape = [UInt64(arch.numExperts),
                                 UInt64(2 * arch.moeIntermediateSize),
                                 UInt64(arch.hiddenSize / 32), 16]
            case ("mlp1", "scales"):
                expectedShape = [UInt64(arch.numExperts),
                                 UInt64(2 * arch.moeIntermediateSize),
                                 UInt64(arch.hiddenSize / 32)]
            case ("mlp1", "bias"):
                expectedShape = [UInt64(arch.numExperts),
                                 UInt64(2 * arch.moeIntermediateSize)]
            case ("mlp2", "weights"):
                expectedShape = [UInt64(arch.numExperts),
                                 UInt64(arch.hiddenSize),
                                 UInt64(arch.moeIntermediateSize / 32), 16]
            case ("mlp2", "scales"):
                expectedShape = [UInt64(arch.numExperts),
                                 UInt64(arch.hiddenSize),
                                 UInt64(arch.moeIntermediateSize / 32)]
            default:
                expectedShape = [UInt64(arch.numExperts), UInt64(arch.hiddenSize)]
            }
            guard tensor.shape == expectedShape,
                  tensor.sizeBytes % UInt64(arch.numExperts) == 0 else {
                throw RepackError.shapeMismatch(
                    name: name,
                    detail: "expected shape \(expectedShape), got \(tensor.shape)")
            }
            let perExpert = tensor.sizeBytes / UInt64(arch.numExperts)
            let logicalShape: [UInt64]
            switch (source.role, source.component) {
            case ("mlp1", "weights"):
                logicalShape = [UInt64(2 * arch.moeIntermediateSize),
                                UInt64(arch.hiddenSize)]
            case ("mlp1", "scales"):
                logicalShape = [UInt64(2 * arch.moeIntermediateSize),
                                UInt64(arch.hiddenSize / 32)]
            case ("mlp1", "bias"):
                logicalShape = [UInt64(2 * arch.moeIntermediateSize)]
            case ("mlp2", "weights"):
                logicalShape = [UInt64(arch.hiddenSize),
                                UInt64(arch.moeIntermediateSize)]
            case ("mlp2", "scales"):
                logicalShape = [UInt64(arch.hiddenSize),
                                UInt64(arch.moeIntermediateSize / 32)]
            default:
                logicalShape = [UInt64(arch.hiddenSize)]
            }
            slices.append(PerExpertTensorSlice(
                role: source.role,
                component: source.component,
                dtype: tensor.dtype == .u8 ? SourceTensor.Dtype.u8.rawValue
                    : GTurboFormatV1.DType.bf16.rawValue,
                logicalShape: logicalShape,
                offsetInExpertBlob: cursor,
                sizeInExpertBlob: perExpert,
                sourceOffsetPerExpert: perExpert,
                sourceTensor: tensor,
                bitsForWeights: source.component == "weights" ? 4 : nil))
            cursor += perExpert
        }
        return LayerFilePlan(
            layerIndex: layer,
            path: path,
            expertsPerLayer: arch.numExperts,
            expertStride: roundUpToPage(cursor),
            subTensors: slices)
    }

    // MARK: - Layer planning

    private static func planLayerFile(path: String, layer: Int,
                                      gateName: String, upName: String, downName: String,
                                      registry: [String: SourceTensor],
                                      meta: IndexLoader.SourceMetadata,
                                      arch: ArchInfo) throws -> LayerFilePlan {
        let expertCount = arch.numExperts
        let roles: [(role: String, name: String)] = [
            ("gate", gateName), ("up", upName), ("down", downName)
        ]
        var subs: [PerExpertTensorSlice] = []
        subs.reserveCapacity(9)
        var blobCursor: UInt64 = 0

        for (role, name) in roles {
            guard let w = registry[name] else { throw RepackError.missingTensor(name: name) }
            if w.dtype != .u32 || w.shape.count != 3 || Int(w.shape[0]) != expertCount {
                throw RepackError.shapeMismatch(name: name,
                    detail: "expected U32 rank-3 with leading \(expertCount), got \(w.dtype) \(w.shape)")
            }
            let base = name.hasSuffix(".weight") ? String(name.dropLast(".weight".count)) : name
            guard let s = registry[base + ".scales"] else { throw RepackError.missingScalesCompanion(name: name) }
            guard let b = registry[base + ".biases"] else { throw RepackError.missingBiasesCompanion(name: name) }
            if s.dtype != .bf16 || b.dtype != .bf16 {
                throw RepackError.dtypeMismatch(name: name,
                    detail: "expected BF16 scales/biases, got \(s.dtype)/\(b.dtype)")
            }

            let perExpertWeightSize = w.sizeBytes / UInt64(expertCount)
            let perExpertScaleSize  = s.sizeBytes / UInt64(expertCount)
            let perExpertBiasSize   = b.sizeBytes / UInt64(expertCount)
            if perExpertWeightSize * UInt64(expertCount) != w.sizeBytes ||
               perExpertScaleSize  * UInt64(expertCount) != s.sizeBytes ||
               perExpertBiasSize   * UInt64(expertCount) != b.sizeBytes {
                throw RepackError.shapeMismatch(name: name,
                    detail: "source bytes not evenly divisible by \(expertCount) experts")
            }

            let spec = IndexLoader.quantSpec(forTensor: name, meta: meta)
            let perExpertSourceShape = Array(w.shape.dropFirst())
            let logicalPerExpert = logicalShape(forPackedSource: perExpertSourceShape, bits: spec.bits)
            let scalesLogical = Array(s.shape.dropFirst())
            let biasesLogical = Array(b.shape.dropFirst())

            let wSlice = PerExpertTensorSlice(
                role: role, component: "weights", dtype: GTurboFormatV1.DType.u32.rawValue,
                logicalShape: logicalPerExpert,
                offsetInExpertBlob: blobCursor, sizeInExpertBlob: perExpertWeightSize,
                sourceOffsetPerExpert: perExpertWeightSize, sourceTensor: w,
                bitsForWeights: spec.bits)
            blobCursor += perExpertWeightSize
            let sSlice = PerExpertTensorSlice(
                role: role, component: "scales", dtype: GTurboFormatV1.DType.bf16.rawValue,
                logicalShape: scalesLogical,
                offsetInExpertBlob: blobCursor, sizeInExpertBlob: perExpertScaleSize,
                sourceOffsetPerExpert: perExpertScaleSize, sourceTensor: s,
                bitsForWeights: nil)
            blobCursor += perExpertScaleSize
            let bSlice = PerExpertTensorSlice(
                role: role, component: "biases", dtype: GTurboFormatV1.DType.bf16.rawValue,
                logicalShape: biasesLogical,
                offsetInExpertBlob: blobCursor, sizeInExpertBlob: perExpertBiasSize,
                sourceOffsetPerExpert: perExpertBiasSize, sourceTensor: b,
                bitsForWeights: nil)
            blobCursor += perExpertBiasSize

            subs.append(wSlice); subs.append(sSlice); subs.append(bSlice)
        }

        let expertStride = roundUpToPage(blobCursor)
        return LayerFilePlan(layerIndex: layer, path: path,
                             expertsPerLayer: expertCount,
                             expertStride: expertStride,
                             subTensors: subs)
    }

    // MARK: - Helpers

    private static func ietnyDtype(_ d: SourceTensor.Dtype) -> UInt8 {
        switch d {
        case .u32: 0
        case .bf16: 1
        case .fp16: 2
        case .fp32: 3
        case .u8: 4
        }
    }

    private static func roundUpToPage(_ v: UInt64) -> UInt64 {
        let p = Layout.pageBytes
        return ((v + p - 1) / p) * p
    }

    private static func padTo4(_ s: [UInt64]) -> [UInt32] {
        var out: [UInt32] = []
        out.reserveCapacity(4)
        for v in s.prefix(4) { out.append(UInt32(v)) }
        while out.count < 4 { out.append(0) }
        return out
    }

    /// Logical shape of a packed quantized tensor whose source is `[D0,..,Dn-1, Dn/factor]`.
    private static func logicalShape(forPackedSource source: [UInt64], bits: Int) -> [UInt64] {
        let factor = UInt64(32 / bits)
        guard !source.isEmpty else { return source }
        var out = source
        out[out.count - 1] = source[source.count - 1] * factor
        return out
    }

    /// Stable order for the resident LM tensor list. Embedding first, then
    /// per-layer groups in layer index order, then the final norm (and, for
    /// families with an untied head, `lm_head` last).
    private static func lmResidentOrdering(family: RepackModelFamily)
        -> (String, String) -> Bool {
        // Compute a sort key per name; we order by (group rank, layer, slot rank, name).
        func key(_ sourceName: String) -> (Int, Int, Int, String) {
            let n = canonicalResidentName(sourceName, family: family)
            if n == "language_model.model.embed_tokens.weight" { return (0, 0, 0, n) }
            if n == "language_model.model.embed_tokens_per_layer.weight" { return (0, 0, 1, n) }
            if n == "language_model.model.norm.weight"          { return (3, 0, 0, n) }
            if n == "language_model.lm_head.weight"             { return (4, 0, 0, n) }
            if let li = layerIndex(in: n) {
                let slot: Int
                switch family {
                case .gemma4: slot = slotRank(in: n)
                case .qwen36: slot = qwenSlotRank(in: n)
                case .gptOss: slot = gptOssSlotRank(in: n)
                }
                return (1, li, slot, n)
            }
            return (2, 0, 0, n)
        }
        return { a, b in
            let ka = key(a), kb = key(b)
            if ka.0 != kb.0 { return ka.0 < kb.0 }
            if ka.1 != kb.1 { return ka.1 < kb.1 }
            if ka.2 != kb.2 { return ka.2 < kb.2 }
            return ka.3 < kb.3
        }
    }

    private static func gptOssSlotRank(in name: String) -> Int {
        if name.hasSuffix(".input_layernorm.weight") { return 0 }
        if name.contains(".self_attn.q_proj.weight") { return 1 }
        if name.contains(".self_attn.q_proj.bias") { return 2 }
        if name.contains(".self_attn.k_proj.weight") { return 3 }
        if name.contains(".self_attn.k_proj.bias") { return 4 }
        if name.contains(".self_attn.v_proj.weight") { return 5 }
        if name.contains(".self_attn.v_proj.bias") { return 6 }
        if name.contains(".self_attn.o_proj.weight") { return 7 }
        if name.contains(".self_attn.o_proj.bias") { return 8 }
        if name.hasSuffix(".self_attn.sinks") { return 9 }
        if name.hasSuffix(".post_attention_layernorm.weight") { return 10 }
        if name.contains(".mlp.router.weight") { return 11 }
        if name.contains(".mlp.router.bias") { return 12 }
        return 100
    }

    /// Within-layer slot order for the Qwen 3.6 family: full-attention
    /// projections/norms, then the gated-DeltaNet linear-attention bundle,
    /// then router, shared-expert gate and MLP, then the two layer norms.
    private static func qwenSlotRank(in n: String) -> Int {
        if n.contains(".self_attn.q_proj.weight")   { return 0 }
        if n.contains(".self_attn.k_proj.weight")   { return 1 }
        if n.contains(".self_attn.v_proj.weight")   { return 2 }
        if n.contains(".self_attn.o_proj.weight")   { return 3 }
        if n.contains(".self_attn.q_norm.weight")   { return 4 }
        if n.contains(".self_attn.k_norm.weight")   { return 5 }
        if n.contains(".linear_attn.in_proj_qkv.weight") { return 6 }
        if n.contains(".linear_attn.in_proj_z.weight")   { return 7 }
        if n.contains(".linear_attn.in_proj_a.weight")   { return 8 }
        if n.contains(".linear_attn.in_proj_b.weight")   { return 9 }
        if n.contains(".linear_attn.conv1d.weight")      { return 10 }
        if n.hasSuffix(".linear_attn.A_log")             { return 11 }
        if n.hasSuffix(".linear_attn.dt_bias")           { return 12 }
        if n.contains(".linear_attn.norm.weight")        { return 13 }
        if n.contains(".linear_attn.out_proj.weight")    { return 14 }
        if n.contains(".mlp.gate.weight")                { return 15 }
        if n.contains(".mlp.shared_expert_gate.weight")  { return 16 }
        if n.contains(".mlp.shared_expert.gate_proj.weight") { return 17 }
        if n.contains(".mlp.shared_expert.up_proj.weight")   { return 18 }
        if n.contains(".mlp.shared_expert.down_proj.weight") { return 19 }
        if n.hasSuffix(".input_layernorm.weight")        { return 20 }
        if n.hasSuffix(".post_attention_layernorm.weight") { return 21 }
        return 100
    }

    /// Within-layer slot order. Mirrors the per-layer description in the
    /// architecture doc.
    private static func slotRank(in n: String) -> Int {
        if n.contains(".self_attn.q_proj.weight") { return 0 }
        if n.contains(".self_attn.k_proj.weight") { return 1 }
        if n.contains(".self_attn.v_proj.weight") { return 2 }
        if n.contains(".self_attn.o_proj.weight") { return 3 }
        if n.contains(".self_attn.q_norm.weight") { return 4 }
        if n.contains(".self_attn.k_norm.weight") { return 5 }
        if n.contains(".router.proj.weight")      { return 6 }
        if n.contains(".router.scale")            { return 7 }
        if n.contains(".router.per_expert_scale") { return 8 }
        if n.contains(".mlp.gate_proj.weight")    { return 9 }
        if n.contains(".mlp.up_proj.weight")      { return 10 }
        if n.contains(".mlp.down_proj.weight")    { return 11 }
        if n.contains(".per_layer_input_gate.weight") { return 12 }
        if n.contains(".per_layer_projection.weight") { return 13 }
        if n.hasSuffix(".input_layernorm.weight") { return 14 }
        if n.hasSuffix(".post_attention_layernorm.weight") { return 15 }
        if n.hasSuffix(".pre_feedforward_layernorm.weight") { return 16 }
        if n.hasSuffix(".pre_feedforward_layernorm_2.weight") { return 17 }
        if n.hasSuffix(".post_feedforward_layernorm.weight") { return 18 }
        if n.hasSuffix(".post_feedforward_layernorm_1.weight") { return 19 }
        if n.hasSuffix(".post_feedforward_layernorm_2.weight") { return 20 }
        if n.hasSuffix(".post_per_layer_input_norm.weight") { return 21 }
        if n.hasSuffix(".layer_scalar")           { return 22 }
        return 100
    }
}
