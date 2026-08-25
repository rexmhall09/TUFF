import Foundation

/// Model family discriminator, mirrored into `manifest.json -> arch.family`
/// for non-Gemma families (Gemma manifests omit it — the format's original
/// architecture). Raw values match the runtime's `ModelFamily`.
enum RepackModelFamily: String, Sendable, Equatable {
    case gemma4 = "gemma4"
    case qwen36 = "qwen36"
    case gptOss = "gpt-oss"
}

enum RepackModelVariant: String, Sendable, Equatable {
    case gemma4_E4B = "gemma4-e4b"
    case gemma4_26B_A4B = "gemma4-26b-a4b"
    case qwen36_35B_A3B = "qwen36-35b-a3b"
    case gptOss_20B = "gpt-oss-20b"
}

enum RepackFeedForwardKind: String, Sendable, Equatable {
    case dense
    case mixtureOfExperts = "moe"
}

/// Architecture facts mirrored into `manifest.json -> arch`. Cross-checked by
/// the runtime loader at startup.
///
/// `fullAttentionLayerMask` values: 0 = sliding-window attention,
/// 1 = full attention, 2 = gated-DeltaNet linear attention.
struct ArchInfo: Sendable, Equatable {
    let hiddenSize: Int
    let intermediateSize: Int          // shared expert FFN
    let moeIntermediateSize: Int       // per-expert FFN
    let numHeads: Int
    let numKVHeads: Int
    let numFullKVHeads: Int
    let headDim: Int
    let fullHeadDim: Int
    let vocabSize: Int
    let slidingWindow: Int
    let finalLogitSoftcap: Double
    let ropeTheta: Double
    let fullRopeTheta: Double
    let partialRotaryFactor: Double
    let numLayers: Int
    let numExperts: Int
    let topKExperts: Int
    let tieWordEmbeddings: Bool
    let attentionKEqV: Bool
    /// 1 if `full_attention`, 0 if `sliding_attention`, 2 if `linear_attention`.
    let fullAttentionLayerMask: [UInt8]
    let hiddenActivation: String

    // Family-dependent extensions. Defaults describe Gemma 4 so the Gemma
    // path (and its manifest output) is unchanged, and so a caller that
    // predates the family split still builds the architecture it meant to.
    var family: RepackModelFamily = .gemma4
    var variant: RepackModelVariant = .gemma4_26B_A4B
    var feedForwardKind: RepackFeedForwardKind = .mixtureOfExperts
    var hiddenSizePerLayerInput: Int = 0
    var vocabSizePerLayerInput: Int = 0
    var numKVSharedLayers: Int = 0
    var attnOutputGate: Bool = false
    var attentionScale: Double = 1.0
    var embeddingScaledBySqrtHidden: Bool = true
    var routerScaled: Bool = true
    var ffnSandwichNorms: Bool = true
    var sharedExpertGated: Bool = false
    var ropeNeoxSubdim: Bool = false
    var linearNumKHeads: Int = 0
    var linearNumVHeads: Int = 0
    var linearKeyHeadDim: Int = 0
    var linearValueHeadDim: Int = 0
    var linearConvKernelSize: Int = 0

    static func load(configPath: String) throws -> ArchInfo {
        let data = try Data(contentsOf: URL(fileURLWithPath: configPath))
        guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw RepackError.configJsonInvalid(path: configPath, detail: "not a JSON object")
        }
        if (root["model_type"] as? String) == "gpt_oss" {
            return try loadGPTOSS20B(configPath: configPath, config: root)
        }
        guard let tc = root["text_config"] as? [String: Any] else {
            throw RepackError.configJsonInvalid(path: configPath, detail: "no text_config")
        }
        if (root["model_type"] as? String) == "qwen3_5_moe" {
            return try loadQwen36(configPath: configPath, tc: tc)
        }
        return try loadGemma4(configPath: configPath, tc: tc)
    }

    // MARK: - GPT-OSS

    private static func loadGPTOSS20B(
        configPath: String,
        config: [String: Any]
    ) throws -> ArchInfo {
        func i(_ key: String) throws -> Int {
            guard let value = (config[key] as? Int)
                    ?? (config[key] as? NSNumber)?.intValue else {
                throw RepackError.configJsonInvalid(
                    path: configPath, detail: "missing \(key)")
            }
            return value
        }
        func d(_ key: String) throws -> Double {
            guard let value = (config[key] as? Double)
                    ?? (config[key] as? NSNumber)?.doubleValue else {
                throw RepackError.configJsonInvalid(
                    path: configPath, detail: "missing \(key)")
            }
            return value
        }
        guard let layerTypes = config["layer_types"] as? [String] else {
            throw RepackError.configJsonInvalid(
                path: configPath, detail: "missing layer_types")
        }
        let mask: [UInt8] = try layerTypes.map { value in
            switch value {
            case "sliding_attention": return 0
            case "full_attention": return 1
            default:
                throw RepackError.configJsonInvalid(
                    path: configPath,
                    detail: "unknown layer_types entry \"\(value)\"")
            }
        }
        guard let quant = config["quantization_config"] as? [String: Any],
              (quant["quant_method"] as? String)?.lowercased() == "mxfp4" else {
            throw RepackError.configJsonInvalid(
                path: configPath,
                detail: "GPT-OSS requires quantization_config.quant_method=mxfp4")
        }
        let headDim = try i("head_dim")
        let experts = (config["num_local_experts"] as? Int)
            ?? (config["num_local_experts"] as? NSNumber)?.intValue
        let topK = (config["experts_per_token"] as? Int)
            ?? (config["experts_per_token"] as? NSNumber)?.intValue
            ?? (config["num_experts_per_tok"] as? Int)
            ?? (config["num_experts_per_tok"] as? NSNumber)?.intValue
        guard let experts, let topK else {
            throw RepackError.configJsonInvalid(
                path: configPath, detail: "missing GPT-OSS expert dimensions")
        }
        let arch = ArchInfo(
            hiddenSize: try i("hidden_size"),
            intermediateSize: try i("intermediate_size"),
            moeIntermediateSize: try i("intermediate_size"),
            numHeads: try i("num_attention_heads"),
            numKVHeads: try i("num_key_value_heads"),
            numFullKVHeads: try i("num_key_value_heads"),
            headDim: headDim,
            fullHeadDim: headDim,
            vocabSize: try i("vocab_size"),
            slidingWindow: try i("sliding_window"),
            finalLogitSoftcap: 0,
            ropeTheta: try d("rope_theta"),
            fullRopeTheta: try d("rope_theta"),
            partialRotaryFactor: 1,
            numLayers: try i("num_hidden_layers"),
            numExperts: experts,
            topKExperts: topK,
            tieWordEmbeddings: (config["tie_word_embeddings"] as? Bool) ?? false,
            attentionKEqV: false,
            fullAttentionLayerMask: mask,
            hiddenActivation: "swiglu_capped",
            family: .gptOss,
            variant: .gptOss_20B,
            feedForwardKind: .mixtureOfExperts,
            attnOutputGate: false,
            attentionScale: 1 / Double(headDim).squareRoot(),
            embeddingScaledBySqrtHidden: false,
            routerScaled: false,
            ffnSandwichNorms: false,
            sharedExpertGated: false,
            ropeNeoxSubdim: true)
        try crossCheckProductionGPTOSS20B(
            arch, config: config, configPath: configPath)
        return arch
    }

    private static func crossCheckProductionGPTOSS20B(
        _ arch: ArchInfo,
        config: [String: Any],
        configPath: String
    ) throws {
        guard arch.hiddenSize == 2_880, arch.numLayers == 24 else { return }
        let expectedMask = (0..<24).map { UInt8($0.isMultiple(of: 2) ? 0 : 1) }
        let rope = config["rope_scaling"] as? [String: Any]
        func number(_ object: Any?) -> Double? {
            (object as? Double) ?? (object as? NSNumber)?.doubleValue
        }
        guard arch.intermediateSize == 2_880,
              arch.moeIntermediateSize == 2_880,
              arch.numHeads == 64,
              arch.numKVHeads == 8,
              arch.headDim == 64,
              arch.vocabSize == 201_088,
              arch.slidingWindow == 128,
              arch.ropeTheta == 150_000,
              arch.numExperts == 32,
              arch.topKExperts == 4,
              arch.fullAttentionLayerMask == expectedMask,
              arch.attentionScale == 0.125,
              arch.hiddenActivation == "swiglu_capped",
              !arch.tieWordEmbeddings,
              config["attention_bias"] as? Bool == true,
              number(config["swiglu_limit"]) == 7,
              (config["initial_context_length"] as? Int) == 4_096,
              (config["max_position_embeddings"] as? Int) == 131_072,
              (rope?["rope_type"] as? String) == "yarn",
              number(rope?["factor"]) == 32,
              number(rope?["beta_fast"]) == 32,
              number(rope?["beta_slow"]) == 1,
              (rope?["original_max_position_embeddings"] as? Int) == 4_096 else {
            throw RepackError.configJsonInvalid(
                path: configPath,
                detail: "GPT-OSS config does not match the pinned 20B architecture baseline")
        }
    }

    // MARK: - Gemma 4

    private static func loadGemma4(configPath: String,
                                   tc: [String: Any]) throws -> ArchInfo {
        func i(_ k: String) throws -> Int {
            guard let n = (tc[k] as? Int) ?? (tc[k] as? NSNumber)?.intValue else {
                throw RepackError.configJsonInvalid(path: configPath, detail: "missing \(k)")
            }
            return n
        }
        func d(_ k: String) throws -> Double {
            guard let n = (tc[k] as? Double) ?? (tc[k] as? NSNumber)?.doubleValue else {
                throw RepackError.configJsonInvalid(path: configPath, detail: "missing \(k)")
            }
            return n
        }
        func optionalInt(_ k: String) -> Int? {
            (tc[k] as? Int) ?? (tc[k] as? NSNumber)?.intValue
        }
        let layerTypes = (tc["layer_types"] as? [String]) ?? []
        let mask = layerTypes.map { UInt8($0 == "full_attention" ? 1 : 0) }
        let rope = (tc["rope_parameters"] as? [String: Any]) ?? [:]
        let ropeFull = (rope["full_attention"] as? [String: Any]) ?? [:]
        let ropeSWA  = (rope["sliding_attention"] as? [String: Any]) ?? [:]
        let prf = (ropeFull["partial_rotary_factor"] as? Double)
            ?? (ropeFull["partial_rotary_factor"] as? NSNumber)?.doubleValue ?? 0.25
        let fullTheta = (ropeFull["rope_theta"] as? Double)
            ?? (ropeFull["rope_theta"] as? NSNumber)?.doubleValue ?? 1_000_000.0
        let swaTheta = (ropeSWA["rope_theta"] as? Double)
            ?? (ropeSWA["rope_theta"] as? NSNumber)?.doubleValue ?? 10_000.0
        let kEqV = (tc["attention_k_eq_v"] as? Bool) ?? false
        let tie = (tc["tie_word_embeddings"] as? Bool) ?? false
        let act = (tc["hidden_activation"] as? String) ?? "gelu_pytorch_tanh"
        let dense = (tc["enable_moe_block"] as? Bool) == false
            || optionalInt("num_experts") == nil
        let numKVHeads = try i("num_key_value_heads")
        let arch = ArchInfo(
            hiddenSize: try i("hidden_size"),
            intermediateSize: try i("intermediate_size"),
            moeIntermediateSize: dense ? 0 : try i("moe_intermediate_size"),
            numHeads: try i("num_attention_heads"),
            numKVHeads: numKVHeads,
            numFullKVHeads: optionalInt("num_global_key_value_heads") ?? numKVHeads,
            headDim: try i("head_dim"),
            fullHeadDim: try i("global_head_dim"),
            vocabSize: try i("vocab_size"),
            slidingWindow: try i("sliding_window"),
            finalLogitSoftcap: try d("final_logit_softcapping"),
            ropeTheta: swaTheta,
            fullRopeTheta: fullTheta,
            partialRotaryFactor: prf,
            numLayers: try i("num_hidden_layers"),
            numExperts: dense ? 0 : try i("num_experts"),
            topKExperts: dense ? 0 : try i("top_k_experts"),
            tieWordEmbeddings: tie,
            attentionKEqV: kEqV,
            fullAttentionLayerMask: mask,
            hiddenActivation: act,
            family: .gemma4,
            variant: dense ? .gemma4_E4B : .gemma4_26B_A4B,
            feedForwardKind: dense ? .dense : .mixtureOfExperts,
            hiddenSizePerLayerInput: optionalInt("hidden_size_per_layer_input") ?? 0,
            vocabSizePerLayerInput: optionalInt("vocab_size_per_layer_input") ?? 0,
            numKVSharedLayers: optionalInt("num_kv_shared_layers") ?? 0,
            attnOutputGate: false,
            attentionScale: 1.0,
            embeddingScaledBySqrtHidden: true,
            routerScaled: true,
            ffnSandwichNorms: true,
            sharedExpertGated: false,
            ropeNeoxSubdim: false,
            linearNumKHeads: 0,
            linearNumVHeads: 0,
            linearKeyHeadDim: 0,
            linearValueHeadDim: 0,
            linearConvKernelSize: 0)
        try crossCheckProductionGemma4(arch, configPath: configPath)
        return arch
    }

    private static func crossCheckProductionGemma4(_ arch: ArchInfo,
                                                    configPath: String) throws {
        guard arch.variant == .gemma4_E4B,
              arch.hiddenSize == 2_560,
              arch.numLayers == 42 else { return }
        var expectedMask = [UInt8](repeating: 0, count: 42)
        for layer in stride(from: 5, to: 42, by: 6) { expectedMask[layer] = 1 }
        guard arch.intermediateSize == 10_240,
              arch.moeIntermediateSize == 0,
              arch.numHeads == 8,
              arch.numKVHeads == 2,
              arch.numFullKVHeads == 2,
              arch.headDim == 256,
              arch.fullHeadDim == 512,
              arch.vocabSize == 262_144,
              arch.slidingWindow == 512,
              arch.numExperts == 0,
              arch.topKExperts == 0,
              arch.fullAttentionLayerMask == expectedMask,
              arch.hiddenSizePerLayerInput == 256,
              arch.vocabSizePerLayerInput == 262_144,
              arch.numKVSharedLayers == 18,
              arch.tieWordEmbeddings,
              !arch.attentionKEqV else {
            throw RepackError.configJsonInvalid(
                path: configPath,
                detail: "gemma4 dense config does not match the pinned E4B architecture baseline")
        }
    }

    // MARK: - Qwen 3.6 MoE (`model_type == "qwen3_5_moe"`)

    private static func loadQwen36(configPath: String,
                                   tc: [String: Any]) throws -> ArchInfo {
        func i(_ k: String) throws -> Int {
            guard let n = (tc[k] as? Int) ?? (tc[k] as? NSNumber)?.intValue else {
                throw RepackError.configJsonInvalid(path: configPath, detail: "missing \(k)")
            }
            return n
        }
        guard let layerTypes = tc["layer_types"] as? [String] else {
            throw RepackError.configJsonInvalid(path: configPath, detail: "missing layer_types")
        }
        var mask: [UInt8] = []
        mask.reserveCapacity(layerTypes.count)
        for t in layerTypes {
            switch t {
            case "linear_attention": mask.append(2)
            case "full_attention":   mask.append(1)
            default:
                throw RepackError.configJsonInvalid(
                    path: configPath, detail: "unknown layer_types entry \"\(t)\"")
            }
        }
        let rope = (tc["rope_parameters"] as? [String: Any]) ?? [:]
        guard let theta = (rope["rope_theta"] as? Double)
            ?? (rope["rope_theta"] as? NSNumber)?.doubleValue else {
            throw RepackError.configJsonInvalid(
                path: configPath, detail: "missing rope_parameters.rope_theta")
        }
        guard let prf = (rope["partial_rotary_factor"] as? Double)
            ?? (rope["partial_rotary_factor"] as? NSNumber)?.doubleValue else {
            throw RepackError.configJsonInvalid(
                path: configPath, detail: "missing rope_parameters.partial_rotary_factor")
        }
        let tie = (tc["tie_word_embeddings"] as? Bool) ?? false
        let gate = (tc["attn_output_gate"] as? Bool) ?? false
        let act = (tc["hidden_act"] as? String) ?? "silu"
        let headDim = try i("head_dim")

        let arch = ArchInfo(
            hiddenSize: try i("hidden_size"),
            intermediateSize: try i("shared_expert_intermediate_size"),
            moeIntermediateSize: try i("moe_intermediate_size"),
            numHeads: try i("num_attention_heads"),
            numKVHeads: try i("num_key_value_heads"),
            numFullKVHeads: try i("num_key_value_heads"),
            headDim: headDim,
            fullHeadDim: headDim,
            vocabSize: try i("vocab_size"),
            slidingWindow: 0,
            finalLogitSoftcap: 0.0,
            ropeTheta: theta,
            fullRopeTheta: theta,
            partialRotaryFactor: prf,
            numLayers: try i("num_hidden_layers"),
            numExperts: try i("num_experts"),
            topKExperts: try i("num_experts_per_tok"),
            tieWordEmbeddings: tie,
            attentionKEqV: false,
            fullAttentionLayerMask: mask,
            hiddenActivation: act,
            family: .qwen36,
            variant: .qwen36_35B_A3B,
            attnOutputGate: gate,
            attentionScale: 1.0 / Double(headDim).squareRoot(),
            embeddingScaledBySqrtHidden: false,
            routerScaled: false,
            ffnSandwichNorms: false,
            sharedExpertGated: true,
            ropeNeoxSubdim: true,
            linearNumKHeads: try i("linear_num_key_heads"),
            linearNumVHeads: try i("linear_num_value_heads"),
            linearKeyHeadDim: try i("linear_key_head_dim"),
            linearValueHeadDim: try i("linear_value_head_dim"),
            linearConvKernelSize: try i("linear_conv_kernel_dim"))
        try crossCheckProductionQwen36(arch, configPath: configPath)
        return arch
    }

    /// Production Qwen3.6-35B-A3B baseline (mirrors the runtime's
    /// `ArchConfig.qwen36_35B_A3B`; the repack target has no dependency on the
    /// runtime module). A config that matches the production shape
    /// (hidden 2048, 40 layers) must agree on every field; toy/synthetic
    /// configs are exempt.
    private static func crossCheckProductionQwen36(_ a: ArchInfo,
                                                   configPath: String) throws {
        guard a.hiddenSize == 2048, a.numLayers == 40 else { return }
        var expectedMask = [UInt8](repeating: 2, count: 40)
        for i in stride(from: 3, to: 40, by: 4) { expectedMask[i] = 1 }
        let expected = ArchInfo(
            hiddenSize: 2048,
            intermediateSize: 512,
            moeIntermediateSize: 512,
            numHeads: 16,
            numKVHeads: 2,
            numFullKVHeads: 2,
            headDim: 256,
            fullHeadDim: 256,
            vocabSize: 248_320,
            slidingWindow: 0,
            finalLogitSoftcap: 0.0,
            ropeTheta: 10_000_000.0,
            fullRopeTheta: 10_000_000.0,
            partialRotaryFactor: 0.25,
            numLayers: 40,
            numExperts: 256,
            topKExperts: 8,
            tieWordEmbeddings: false,
            attentionKEqV: false,
            fullAttentionLayerMask: expectedMask,
            hiddenActivation: "silu",
            family: .qwen36,
            variant: .qwen36_35B_A3B,
            attnOutputGate: true,
            attentionScale: 0.0625,
            embeddingScaledBySqrtHidden: false,
            routerScaled: false,
            ffnSandwichNorms: false,
            sharedExpertGated: true,
            ropeNeoxSubdim: true,
            linearNumKHeads: 16,
            linearNumVHeads: 32,
            linearKeyHeadDim: 128,
            linearValueHeadDim: 128,
            linearConvKernelSize: 4)
        guard a == expected else {
            throw RepackError.configJsonInvalid(
                path: configPath,
                detail: "qwen3_5_moe config does not match the pinned "
                    + "Qwen3.6-35B-A3B architecture baseline")
        }
    }
}
