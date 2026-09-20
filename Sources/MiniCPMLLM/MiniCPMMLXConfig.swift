import Foundation
import MLXCommon

/// The Qwen3 language-backbone portion of MiniCPM-o's duplex model.
///
/// The offline converter writes a compact Qwen3 ``config.json`` beside the
/// packed safetensors.  Keeping this config independent of the official
/// MiniCPMO config prevents the Swift runtime from ever constructing the
/// vision/audio/TTS modules merely to run a language-model turn.
public struct MiniCPMMLXConfig: Sendable, Equatable {
    public let hiddenSize: Int
    public let numHiddenLayers: Int
    public let numAttentionHeads: Int
    public let numKeyValueHeads: Int
    public let headDim: Int
    public let intermediateSize: Int
    public let vocabSize: Int
    public let maxPositionEmbeddings: Int
    public let ropeTheta: Float
    public let rmsNormEps: Float
    public let tieWordEmbeddings: Bool
    public let eosTokenId: Int
    public let quantGroupSize: Int
    public let quantBits: Int
    public let quantMode: String

    /// Whether this bundle stores MLX packed affine weights.  A dense BF16
    /// bundle uses the same group-size metadata for config compatibility but
    /// declares ``bits == 0`` and does not carry scales/biases tensors.
    public var isQuantized: Bool { quantBits > 0 }

    /// Load either a bundle directory or a direct ``config.json`` URL.  Swift
    /// cannot overload these two cases by argument label (both are `URL`), so
    /// keeping one entry point also avoids an ambiguous call from tests and
    /// clients that discover a bundle at runtime.
    public static func load(from url: URL) throws -> MiniCPMMLXConfig {
        let configURL: URL
        var isDirectory: ObjCBool = false
        if FileManager.default.fileExists(atPath: url.path, isDirectory: &isDirectory),
           isDirectory.boolValue {
            configURL = url.appendingPathComponent("config.json")
        } else {
            configURL = url
        }
        let data = try Data(contentsOf: configURL)
        guard let obj = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw MiniCPMLLMError.invalidConfig("config is not a JSON object")
        }
        func int(_ key: String, _ fallback: Int) -> Int {
            (obj[key] as? NSNumber)?.intValue ?? fallback
        }
        func float(_ key: String, _ fallback: Float) -> Float {
            (obj[key] as? NSNumber).map { $0.floatValue } ?? fallback
        }
        let hidden = int("hidden_size", 4096)
        let heads = int("num_attention_heads", 32)
        let headDim = int("head_dim", hidden / max(heads, 1))
        let eos: Int
        if let value = obj["eos_token_id"] as? NSNumber {
            eos = value.intValue
        } else if let values = obj["eos_token_id"] as? [NSNumber], let first = values.first {
            eos = first.intValue
        } else {
            eos = 151645
        }
        let quant = MiniCPMQuantization.resolve(searching: [obj])
        guard hidden > 0, heads > 0, headDim > 0,
              quant.groupSize > 0,
              quant.bits >= 0 else {
            throw MiniCPMLLMError.invalidConfig("invalid Qwen3 dimensions or quantization")
        }
        let mode: String
        if let nested = obj["quantization"] as? [String: Any], let value = nested["mode"] as? String {
            mode = value
        } else if let nested = obj["quantization_config"] as? [String: Any], let value = nested["mode"] as? String {
            mode = value
        } else {
            mode = "affine"
        }
        return MiniCPMMLXConfig(
            hiddenSize: hidden,
            numHiddenLayers: int("num_hidden_layers", 36),
            numAttentionHeads: heads,
            numKeyValueHeads: int("num_key_value_heads", 8),
            headDim: headDim,
            intermediateSize: int("intermediate_size", 12288),
            vocabSize: int("vocab_size", 151748),
            maxPositionEmbeddings: int("max_position_embeddings", 40960),
            ropeTheta: float("rope_theta", 1_000_000),
            rmsNormEps: float("rms_norm_eps", 1e-6),
            tieWordEmbeddings: (obj["tie_word_embeddings"] as? Bool) ?? false,
            eosTokenId: eos,
            quantGroupSize: quant.groupSize,
            quantBits: quant.bits,
            quantMode: mode
        )
    }
}

/// Local copy of the tiny quantization metadata resolver used by the Qwen
/// chat target.  MiniCPMLLM intentionally has no dependency on the chat/UI
/// target: the native duplex runtime may load just this language backbone.
private struct MiniCPMQuantization {
    let bits: Int
    let groupSize: Int

    static let defaultBits = 4
    static let defaultGroupSize = 64

    static func resolve(searching objects: [[String: Any]]) -> MiniCPMQuantization {
        func first<T>(_ read: ([String: Any]) -> T?) -> T? {
            for object in objects {
                if let value = read(object) { return value }
            }
            return nil
        }
        func number(_ key: String, in object: [String: Any]?) -> Int? {
            (object?[key] as? NSNumber)?.intValue
        }
        let nested = first {
            ($0["quantization"] as? [String: Any])
                ?? ($0["quantization_config"] as? [String: Any])
        }
        let label = first { $0["quantization"] as? String }
        let labelBits: Int? = label.flatMap { value in
            let digits = value.drop { !$0.isNumber }.prefix { $0.isNumber }
            guard let parsed = Int(digits), [2, 3, 4, 5, 6, 8].contains(parsed) else {
                return nil
            }
            return parsed
        }
        return MiniCPMQuantization(
            bits: first { number("quantization_bits", in: $0) }
                ?? first { number("bits", in: $0) }
                ?? number("bits", in: nested)
                ?? labelBits
                ?? defaultBits,
            groupSize: first { number("quantization_group_size", in: $0) }
                ?? number("group_size", in: nested)
                ?? defaultGroupSize)
    }
}

public enum MiniCPMLLMError: Error, LocalizedError, Sendable {
    case invalidConfig(String)
    case invalidWeights(String)
    case invalidState(String)
    case unsupported(String)

    public var errorDescription: String? {
        switch self {
        case .invalidConfig(let message): return "MiniCPM invalid config: \(message)"
        case .invalidWeights(let message): return "MiniCPM invalid weights: \(message)"
        case .invalidState(let message): return "MiniCPM invalid state: \(message)"
        case .unsupported(let message): return "MiniCPM unsupported: \(message)"
        }
    }
}
