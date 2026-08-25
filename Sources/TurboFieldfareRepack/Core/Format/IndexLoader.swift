import Foundation

/// Parses `model.safetensors.index.json` and `config.json -> quantization`.
enum IndexLoader {

    struct SourceMetadata {
        let indexPath: String
        let configPath: String
        let indexSha256Hex: String
        /// `tensor_name -> shard_filename`
        let weightMap: [String: String]
        /// Base bits / group_size / mode for any tensor not in the override table.
        let baseBits: Int
        let baseGroupSize: Int
        let baseMode: String
        /// Per-tensor overrides (keyed by tensor name **without** the trailing
        /// `.weight` — matches the way `config.json` writes them).
        let bitsOverrides: [String: QuantSpec]
        /// Resolved set of shard files referenced by the index, in
        /// encounter order. Order is stable enough for sequential I/O.
        let shardFilenames: [String]
    }

    static func load(snapshotDir: String) throws -> SourceMetadata {
        let indexPath  = (snapshotDir as NSString).appendingPathComponent("model.safetensors.index.json")
        let configPath = (snapshotDir as NSString).appendingPathComponent("config.json")

        let weightMap: [String: String]
        do {
            let data = try Data(contentsOf: URL(fileURLWithPath: indexPath))
            guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any],
                  let m = root["weight_map"] as? [String: String] else {
                throw RepackError.indexJsonInvalid(path: indexPath, detail: "no weight_map")
            }
            weightMap = m
        } catch let e as RepackError {
            throw e
        } catch {
            throw RepackError.indexJsonInvalid(path: indexPath, detail: "\(error)")
        }

        let indexSha = try Sha256Stream.hashFile(path: indexPath)

        var baseBits = 4
        var baseGroup = 64
        var baseMode = "affine"
        var overrides: [String: QuantSpec] = [:]
        do {
            let data = try Data(contentsOf: URL(fileURLWithPath: configPath))
            guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
                throw RepackError.configJsonInvalid(path: configPath, detail: "not a JSON object")
            }
            if let gptQuant = root["quantization_config"] as? [String: Any],
               (gptQuant["quant_method"] as? String)?.lowercased() == "mxfp4" {
                baseBits = 4
                baseGroup = 32
                baseMode = "mxfp4"
                return SourceMetadata(
                    indexPath: indexPath,
                    configPath: configPath,
                    indexSha256Hex: indexSha,
                    weightMap: weightMap,
                    baseBits: baseBits,
                    baseGroupSize: baseGroup,
                    baseMode: baseMode,
                    bitsOverrides: overrides,
                    shardFilenames: stableShardFilenames(weightMap))
            }
            guard let quant = root["quantization"] as? [String: Any] else {
                throw RepackError.configJsonInvalid(path: configPath, detail: "no quantization slot")
            }
            if let b = quant["bits"] as? Int      { baseBits  = b }
            if let g = quant["group_size"] as? Int { baseGroup = g }
            if let m = quant["mode"] as? String   { baseMode  = m }
            for (k, v) in quant where !(k == "bits" || k == "group_size" || k == "mode") {
                guard let entry = v as? [String: Any] else { continue }
                let bits = (entry["bits"] as? Int) ?? baseBits
                let g    = (entry["group_size"] as? Int) ?? baseGroup
                guard g == baseGroup else {
                    throw RepackError.configJsonInvalid(
                        path: configPath,
                        detail: "quantization override \(k) group_size \(g) != base \(baseGroup)")
                }
                overrides[k] = QuantSpec(bits: bits)
            }
        } catch let e as RepackError {
            throw e
        } catch {
            throw RepackError.configJsonInvalid(path: configPath, detail: "\(error)")
        }

        let shards = stableShardFilenames(weightMap)

        return SourceMetadata(indexPath: indexPath, configPath: configPath,
                              indexSha256Hex: indexSha,
                              weightMap: weightMap,
                              baseBits: baseBits, baseGroupSize: baseGroup,
                              baseMode: baseMode,
                              bitsOverrides: overrides,
                              shardFilenames: shards)
    }

    private static func stableShardFilenames(
        _ weightMap: [String: String]
    ) -> [String] {
        var seen = Set<String>()
        var shards: [String] = []
        for key in weightMap.keys.sorted() {
            let shard = weightMap[key]!
            if seen.insert(shard).inserted { shards.append(shard) }
        }
        return shards
    }

    /// Resolves the bits/group for one tensor name (with or without `.weight`).
    static func quantSpec(forTensor name: String,
                                 meta: SourceMetadata) -> QuantSpec {
        let stripped = name.hasSuffix(".weight")
            ? String(name.dropLast(".weight".count))
            : name
        if let o = meta.bitsOverrides[stripped] { return o }
        return QuantSpec(bits: meta.baseBits)
    }
}
