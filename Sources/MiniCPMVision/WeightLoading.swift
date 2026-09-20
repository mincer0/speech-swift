import Foundation
import MLX
import MLXNN

/// Strict loader for the converted MiniCPM visual bundle.
public enum MiniCPMVisionWeightLoader {
    public static let configurationFile = "config.json"
    public static let weightsFile = "model.safetensors"

    public static func load(
        model: MiniCPMVisionModel,
        from directory: URL,
        progressHandler: ((Double, String) -> Void)? = nil
    ) throws {
        let file = directory.appendingPathComponent(weightsFile)
        guard FileManager.default.fileExists(atPath: file.path) else {
            throw MiniCPMVisionError.missingModelFile(weightsFile)
        }
        progressHandler?(0.1, "Loading visual weights...")
        let raw = try MLX.loadArrays(url: file)
        let sanitized = sanitize(raw, model: model)

        let expected = Set(model.parameters().flattened().map(\.0))
        let received = Set(sanitized.keys)
        let missing = expected.subtracting(received).sorted()
        let extra = received.subtracting(expected).sorted()
        guard missing.isEmpty, extra.isEmpty else {
            var details: [String] = []
            if !missing.isEmpty {
                details.append("missing [\(missing.prefix(8).joined(separator: ", "))]")
            }
            if !extra.isEmpty {
                details.append("unexpected [\(extra.prefix(8).joined(separator: ", "))]")
            }
            throw MiniCPMVisionError.invalidWeights(details.joined(separator: "; "))
        }

        do {
            try model.update(
                parameters: ModuleParameters.unflattened(sanitized),
                verify: .all
            )
        } catch {
            throw MiniCPMVisionError.invalidWeights(error.localizedDescription)
        }
        model.train(false)
        eval(model)
        progressHandler?(1.0, "Visual weights loaded")
    }

    /// Canonicalize raw PyTorch/HuggingFace and mlx-vlm aliases. Canonical
    /// keys win when a bundle contains both an alias and the canonical form.
    public static func sanitize(
        _ weights: [String: MLXArray],
        model: MiniCPMVisionModel? = nil
    ) -> [String: MLXArray] {
        var output: [String: MLXArray] = [:]
        var packedQKV: [String: MLXArray] = [:]

        let ordered = weights.keys.sorted { lhs, rhs in
            let leftCanonical = lhs.hasPrefix("vision.")
                || lhs.hasPrefix("resampler.")
            let rightCanonical = rhs.hasPrefix("vision.")
                || rhs.hasPrefix("resampler.")
            if leftCanonical != rightCanonical { return leftCanonical }
            return lhs < rhs
        }

        for sourceKey in ordered {
            let value = weights[sourceKey]!
            var key = sourceKey
            if key.hasPrefix("vision_tower.") {
                key = "vision." + key.dropFirst("vision_tower.".count)
            } else if key.hasPrefix("vpm.") {
                key = "vision." + key.dropFirst("vpm.".count)
            }

            if key == "resampler.attn.in_proj_weight" {
                packedQKV["weight"] = value
                continue
            }
            if key == "resampler.attn.in_proj_bias" {
                packedQKV["bias"] = value
                continue
            }
            guard key.hasPrefix("vision.") || key.hasPrefix("resampler.") else {
                continue
            }

            var normalized = value
            if key.hasSuffix("embeddings.patch_embedding.weight"),
               normalized.ndim == 4
            {
                let patch = model?.configuration.patchSize ?? 14
                let channels = model?.configuration.visionChannels ?? 3
                // PyTorch OIHW → MLX OHWI. Converted bundles already use
                // [out,patch,patch,channels] and remain unchanged.
                let alreadyOHWI = normalized.dim(1) == patch
                    && normalized.dim(3) == channels
                if !alreadyOHWI {
                    normalized = normalized.transposed(0, 2, 3, 1)
                }
            }
            // `setdefault` semantics: aliases never overwrite canonical keys.
            if output[key] == nil { output[key] = normalized }
        }

        if let packed = packedQKV["weight"] {
            let parts = MLX.split(packed, parts: 3, axis: 0)
            let names = [
                "resampler.attn.q_proj.weight",
                "resampler.attn.k_proj.weight",
                "resampler.attn.v_proj.weight",
            ]
            for (name, part) in zip(names, parts) where output[name] == nil {
                output[name] = part
            }
        }
        if let packed = packedQKV["bias"] {
            let parts = MLX.split(packed, parts: 3, axis: 0)
            let names = [
                "resampler.attn.q_proj.bias",
                "resampler.attn.k_proj.bias",
                "resampler.attn.v_proj.bias",
            ]
            for (name, part) in zip(names, parts) where output[name] == nil {
                output[name] = part
            }
        }
        return output
    }
}

extension MiniCPMVisionModel {
    public static func fromDirectory(
        _ directory: URL,
        progressHandler: ((Double, String) -> Void)? = nil
    ) throws -> MiniCPMVisionModel {
        let configURL = directory.appendingPathComponent(
            MiniCPMVisionWeightLoader.configurationFile)
        guard FileManager.default.fileExists(atPath: configURL.path) else {
            throw MiniCPMVisionError.missingModelFile(
                MiniCPMVisionWeightLoader.configurationFile)
        }
        let configuration = try MiniCPMVisionConfiguration.fromBundleConfig(at: configURL)
        let model = MiniCPMVisionModel(configuration: configuration)
        try MiniCPMVisionWeightLoader.load(
            model: model,
            from: directory,
            progressHandler: progressHandler)
        return model
    }
}
