import Foundation

public enum MiniCPMAudioError: Error, LocalizedError {
    case invalidConfiguration(String)
    case invalidFeatures(String)
    case missingModelFile(String)
    case cacheLimit(current: Int, incoming: Int, maximum: Int)

    public var errorDescription: String? {
        switch self {
        case .invalidConfiguration(let message):
            return "Invalid MiniCPM audio configuration: \(message)"
        case .invalidFeatures(let message):
            return "Invalid MiniCPM audio features: \(message)"
        case .missingModelFile(let file):
            return "Missing MiniCPM audio model file: \(file)"
        case .cacheLimit(let current, let incoming, let maximum):
            return "MiniCPM audio cache would grow from \(current) by \(incoming) positions beyond \(maximum)"
        }
    }
}

/// Geometry of MiniCPM-o 4.5's streaming Whisper audio encoder and projector.
public struct MiniCPMAudioConfiguration: Codable, Equatable, Sendable {
    public var hiddenSize: Int
    public var intermediateSize: Int
    public var attentionHeads: Int
    public var layers: Int
    public var melBins: Int
    public var maximumSourcePositions: Int
    public var layerNormEpsilon: Float
    public var poolStep: Int
    public var projectionSize: Int

    public init(
        hiddenSize: Int = 1_024,
        intermediateSize: Int = 4_096,
        attentionHeads: Int = 16,
        layers: Int = 24,
        melBins: Int = 80,
        maximumSourcePositions: Int = 1_500,
        layerNormEpsilon: Float = 1e-5,
        poolStep: Int = 5,
        projectionSize: Int = 4_096
    ) {
        self.hiddenSize = hiddenSize
        self.intermediateSize = intermediateSize
        self.attentionHeads = attentionHeads
        self.layers = layers
        self.melBins = melBins
        self.maximumSourcePositions = maximumSourcePositions
        self.layerNormEpsilon = layerNormEpsilon
        self.poolStep = poolStep
        self.projectionSize = projectionSize
    }

    public func validate() throws {
        guard hiddenSize > 0,
              intermediateSize > 0,
              attentionHeads > 0,
              hiddenSize.isMultiple(of: attentionHeads),
              layers > 0,
              melBins == 80,
              maximumSourcePositions > 0,
              layerNormEpsilon > 0,
              poolStep > 0,
              projectionSize > 0
        else {
            throw MiniCPMAudioError.invalidConfiguration(
                "unsupported encoder geometry"
            )
        }
    }

    /// Read the relevant fields directly from MiniCPM-o's original config.json.
    public static func fromMiniCPMConfig(at url: URL) throws -> Self {
        struct SourceAudio: Decodable {
            let hiddenSize: Int
            let intermediateSize: Int
            let attentionHeads: Int
            let layers: Int
            let melBins: Int
            let maximumSourcePositions: Int
            let layerNormEpsilon: Float?

            enum CodingKeys: String, CodingKey {
                case hiddenSize = "d_model"
                case intermediateSize = "encoder_ffn_dim"
                case attentionHeads = "encoder_attention_heads"
                case layers = "encoder_layers"
                case melBins = "num_mel_bins"
                case maximumSourcePositions = "max_source_positions"
                case layerNormEpsilon = "layer_norm_eps"
            }
        }
        struct SourceConfig: Decodable {
            let audio: SourceAudio
            let poolStep: Int
            let projectionSize: Int

            enum CodingKeys: String, CodingKey {
                case audio = "audio_config"
                case poolStep = "audio_pool_step"
                case projectionSize = "hidden_size"
            }
        }

        let source = try JSONDecoder().decode(
            SourceConfig.self,
            from: Data(contentsOf: url)
        )
        let result = Self(
            hiddenSize: source.audio.hiddenSize,
            intermediateSize: source.audio.intermediateSize,
            attentionHeads: source.audio.attentionHeads,
            layers: source.audio.layers,
            melBins: source.audio.melBins,
            maximumSourcePositions: source.audio.maximumSourcePositions,
            layerNormEpsilon: source.audio.layerNormEpsilon ?? 1e-5,
            poolStep: source.poolStep,
            projectionSize: source.projectionSize
        )
        try result.validate()
        return result
    }
}
