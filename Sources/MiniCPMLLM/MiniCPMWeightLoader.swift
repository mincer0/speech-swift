import Foundation
import MLX
import MLXNN
import MLXCommon

/// Strict loader for the Qwen-only bundle produced by
/// ``tools/convert_minicpm_to_mlx.py``.
public enum MiniCPMWeightLoader {
    public static func load(
        model: MiniCPMLanguageModel,
        from directory: URL,
        progressHandler: ((Double, String) -> Void)? = nil
    ) throws {
        progressHandler?(0.02, "Loading Qwen3-only safetensors...")
        let all = try CommonWeightLoader.loadAllSafetensors(from: directory)
        guard !all.isEmpty else {
            throw MiniCPMLLMError.invalidWeights("bundle contains no tensors")
        }

        var weights: [String: MLXArray] = [:]
        for (key, value) in all {
            guard key.hasPrefix("model.") || key.hasPrefix("lm_head.") else {
                throw MiniCPMLLMError.invalidWeights(
                    "non-Qwen tensor in bundle: \(key)")
            }
            weights[key.hasPrefix("model.") ? String(key.dropFirst(6)) : key] = value
        }

        let required = requiredKeys(for: model.config)
        let actual = Set(weights.keys)
        let missing = required.subtracting(actual).sorted()
        guard missing.isEmpty else {
            throw MiniCPMLLMError.invalidWeights(
                "missing tensors: \(missing.prefix(8).joined(separator: ", "))")
        }
        let unknown = actual.subtracting(required).sorted()
        guard unknown.isEmpty else {
            throw MiniCPMLLMError.invalidWeights(
                "unexpected Qwen tensors: \(unknown.prefix(8).joined(separator: ", "))")
        }

        progressHandler?(0.15, "Loading embeddings and output head...")
        try applyEmbeddingWeights(
            to: model.embedTokens,
            prefix: "embed_tokens",
            config: model.config,
            from: weights)
        CommonWeightLoader.applyRMSNormWeights(
            to: model.norm, prefix: "norm", from: weights)
        if let lmHead = model.lmHead {
            try applyLinearWeights(
                to: lmHead, prefix: "lm_head", config: model.config, from: weights)
        }

        let layers = model.layers
        for index in layers.indices {
            let prefix = "layers.\(index)"
            let layer = layers[index]
            CommonWeightLoader.applyRMSNormWeights(
                to: layer.inputLayerNorm,
                prefix: "\(prefix).input_layernorm",
                from: weights)
            CommonWeightLoader.applyRMSNormWeights(
                to: layer.postAttentionLayerNorm,
                prefix: "\(prefix).post_attention_layernorm",
                from: weights)

            try applyLinearWeights(
                to: layer.selfAttention.qProj,
                prefix: "\(prefix).self_attn.q_proj",
                config: model.config,
                from: weights)
            try applyLinearWeights(
                to: layer.selfAttention.kProj,
                prefix: "\(prefix).self_attn.k_proj",
                config: model.config,
                from: weights)
            try applyLinearWeights(
                to: layer.selfAttention.vProj,
                prefix: "\(prefix).self_attn.v_proj",
                config: model.config,
                from: weights)
            try applyLinearWeights(
                to: layer.selfAttention.oProj,
                prefix: "\(prefix).self_attn.o_proj",
                config: model.config,
                from: weights)
            CommonWeightLoader.applyRMSNormWeights(
                to: layer.selfAttention.qNorm,
                prefix: "\(prefix).self_attn.q_norm",
                from: weights)
            CommonWeightLoader.applyRMSNormWeights(
                to: layer.selfAttention.kNorm,
                prefix: "\(prefix).self_attn.k_norm",
                from: weights)

            try applyLinearWeights(
                to: layer.mlp.gateProj,
                prefix: "\(prefix).mlp.gate_proj",
                config: model.config,
                from: weights)
            try applyLinearWeights(
                to: layer.mlp.upProj,
                prefix: "\(prefix).mlp.up_proj",
                config: model.config,
                from: weights)
            try applyLinearWeights(
                to: layer.mlp.downProj,
                prefix: "\(prefix).mlp.down_proj",
                config: model.config,
                from: weights)
            progressHandler?(0.15 + 0.8 * Double(index + 1) / Double(max(layers.count, 1)), "Layer \(index + 1)/\(layers.count)")
        }
        eval(model)
        progressHandler?(1.0, "Qwen3 weights loaded")
    }

    private static func requiredKeys(for config: MiniCPMMLXConfig) -> Set<String> {
        var keys: Set<String> = ["embed_tokens.weight", "norm.weight"]
        if config.isQuantized {
            keys.formUnion(["embed_tokens.scales", "embed_tokens.biases"])
        }
        if !config.tieWordEmbeddings {
            keys.insert("lm_head.weight")
            if config.isQuantized {
                keys.formUnion(["lm_head.scales", "lm_head.biases"])
            }
        }
        for index in 0..<config.numHiddenLayers {
            let prefix = "layers.\(index)"
            keys.formUnion([
                "\(prefix).input_layernorm.weight",
                "\(prefix).post_attention_layernorm.weight",
                "\(prefix).self_attn.q_proj.weight",
                "\(prefix).self_attn.k_proj.weight",
                "\(prefix).self_attn.v_proj.weight",
                "\(prefix).self_attn.o_proj.weight",
                "\(prefix).self_attn.q_norm.weight",
                "\(prefix).self_attn.k_norm.weight",
                "\(prefix).mlp.gate_proj.weight",
                "\(prefix).mlp.up_proj.weight",
                "\(prefix).mlp.down_proj.weight"
            ])
            if config.isQuantized {
                for linear in [
                    "\(prefix).self_attn.q_proj",
                    "\(prefix).self_attn.k_proj",
                    "\(prefix).self_attn.v_proj",
                    "\(prefix).self_attn.o_proj",
                    "\(prefix).mlp.gate_proj",
                    "\(prefix).mlp.up_proj",
                    "\(prefix).mlp.down_proj"
                ] {
                    keys.formUnion(["\(linear).scales", "\(linear).biases"])
                }
            }
        }
        return keys
    }

    private static func applyEmbeddingWeights(
        to embedding: MiniCPMEmbedding,
        prefix: String,
        config: MiniCPMMLXConfig,
        from weights: [String: MLXArray]
    ) throws {
        if config.isQuantized {
            guard let quantized = embedding.quantized else {
                throw MiniCPMLLMError.invalidConfig(
                    "quantized config constructed a dense embedding")
            }
            try CommonWeightLoader.applyCheckedQuantizedEmbeddingWeights(
                to: quantized, prefix: prefix, from: weights)
        } else {
            guard let dense = embedding.dense else {
                throw MiniCPMLLMError.invalidConfig(
                    "dense config constructed a quantized embedding")
            }
            CommonWeightLoader.applyEmbeddingWeights(
                to: dense, prefix: prefix, from: weights)
        }
    }

    private static func applyLinearWeights(
        to linear: Linear,
        prefix: String,
        config: MiniCPMMLXConfig,
        from weights: [String: MLXArray]
    ) throws {
        if config.isQuantized {
            guard let quantized = linear as? QuantizedLinear else {
                throw MiniCPMLLMError.invalidConfig(
                    "quantized config constructed a dense linear layer for \(prefix)")
            }
            try CommonWeightLoader.applyCheckedQuantizedLinearWeights(
                to: quantized, prefix: prefix, from: weights)
        } else {
            guard !(linear is QuantizedLinear) else {
                throw MiniCPMLLMError.invalidConfig(
                    "dense config constructed a quantized linear layer for \(prefix)")
            }
            CommonWeightLoader.applyLinearWeights(
                to: linear, prefix: prefix, from: weights)
        }
    }
}
