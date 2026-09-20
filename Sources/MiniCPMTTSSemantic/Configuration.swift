import Foundation

/// The part of MiniCPM-o's ``tts_config`` required by the semantic TTS path.
/// Unknown upstream fields are deliberately ignored so a newer MiniCPM bundle
/// remains loadable, while all values that affect tensor shapes or protocol
/// tokens are validated locally.
public struct MiniCPMTTSSemanticConfiguration: Codable, Equatable, Sendable {
    public var llmDim: Int
    public var hiddenSize: Int
    public var intermediateSize: Int
    public var numHiddenLayers: Int
    public var numAttentionHeads: Int
    public var numKeyValueHeads: Int
    public var numAudioTokens: Int
    public var numTextTokens: Int
    public var maxPositionEmbeddings: Int
    public var rmsNormEps: Float
    public var audioBOSTokenID: Int
    public var textEOSTokenID: Int
    public var normalizeProjectedHidden: Bool

    public init(
        llmDim: Int = 4096,
        hiddenSize: Int = 768,
        intermediateSize: Int = 3072,
        numHiddenLayers: Int = 20,
        numAttentionHeads: Int = 12,
        numKeyValueHeads: Int = 12,
        numAudioTokens: Int = 6562,
        numTextTokens: Int = 152_064,
        maxPositionEmbeddings: Int = 4096,
        rmsNormEps: Float = 1e-6,
        audioBOSTokenID: Int = 151_687,
        textEOSTokenID: Int = 151_692,
        normalizeProjectedHidden: Bool = true
    ) {
        self.llmDim = llmDim
        self.hiddenSize = hiddenSize
        self.intermediateSize = intermediateSize
        self.numHiddenLayers = numHiddenLayers
        self.numAttentionHeads = numAttentionHeads
        self.numKeyValueHeads = numKeyValueHeads
        self.numAudioTokens = numAudioTokens
        self.numTextTokens = numTextTokens
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.rmsNormEps = rmsNormEps
        self.audioBOSTokenID = audioBOSTokenID
        self.textEOSTokenID = textEOSTokenID
        self.normalizeProjectedHidden = normalizeProjectedHidden
    }

    private enum CodingKeys: String, CodingKey {
        case llmDim = "llm_dim"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case numAudioTokens = "num_audio_tokens"
        case numTextTokens = "num_text_tokens"
        case maxPositionEmbeddings = "max_position_embeddings"
        case rmsNormEps = "rms_norm_eps"
        case audioBOSTokenID = "audio_bos_token_id"
        case textEOSTokenID = "text_eos_token_id"
        case normalizeProjectedHidden = "normalize_projected_hidden"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.init(
            llmDim: try c.decodeIfPresent(Int.self, forKey: .llmDim) ?? 4096,
            hiddenSize: try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 768,
            intermediateSize: try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 3072,
            numHiddenLayers: try c.decodeIfPresent(Int.self, forKey: .numHiddenLayers) ?? 20,
            numAttentionHeads: try c.decodeIfPresent(Int.self, forKey: .numAttentionHeads) ?? 12,
            numKeyValueHeads: try c.decodeIfPresent(Int.self, forKey: .numKeyValueHeads) ?? 12,
            numAudioTokens: try c.decodeIfPresent(Int.self, forKey: .numAudioTokens) ?? 6562,
            numTextTokens: try c.decodeIfPresent(Int.self, forKey: .numTextTokens) ?? 152_064,
            maxPositionEmbeddings: try c.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 4096,
            rmsNormEps: try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6,
            audioBOSTokenID: try c.decodeIfPresent(Int.self, forKey: .audioBOSTokenID) ?? 151_687,
            textEOSTokenID: try c.decodeIfPresent(Int.self, forKey: .textEOSTokenID) ?? 151_692,
            normalizeProjectedHidden: try c.decodeIfPresent(Bool.self, forKey: .normalizeProjectedHidden) ?? true
        )
    }

    public func validate() throws {
        guard llmDim > 0,
              hiddenSize > 0,
              intermediateSize > 0,
              numHiddenLayers > 0,
              numAttentionHeads > 0,
              numKeyValueHeads > 0,
              numAttentionHeads.isMultiple(of: numKeyValueHeads),
              hiddenSize.isMultiple(of: numAttentionHeads),
              numAudioTokens > 1,
              numTextTokens > 0,
              maxPositionEmbeddings > 0,
              rmsNormEps > 0,
              audioBOSTokenID >= 0,
              audioBOSTokenID < numTextTokens,
              textEOSTokenID >= 0,
              textEOSTokenID < numTextTokens
        else {
            throw MiniCPMTTSSemanticError.invalidConfiguration(
                "unsupported TTS dimensions or token IDs")
        }
    }

    public static func fromBundle(at directory: URL) throws -> Self {
        let url = directory.appendingPathComponent("config.json")
        let data: Data
        do {
            data = try Data(contentsOf: url)
        } catch {
            throw MiniCPMTTSSemanticError.missingModelFile(url)
        }

        struct Bundle: Decodable {
            let ttsConfig: MiniCPMTTSSemanticConfiguration
            enum CodingKeys: String, CodingKey { case ttsConfig = "tts_config" }
        }
        let decoder = JSONDecoder()
        let value: MiniCPMTTSSemanticConfiguration
        do {
            // Converted bundles always carry ``tts_config``.  Accepting the
            // config object itself is useful for small parity fixtures.
            if let bundle = try? decoder.decode(Bundle.self, from: data) {
                value = bundle.ttsConfig
            } else {
                value = try decoder.decode(MiniCPMTTSSemanticConfiguration.self, from: data)
            }
        } catch {
            throw MiniCPMTTSSemanticError.invalidConfiguration(
                "cannot decode " + url.lastPathComponent + ": " + error.localizedDescription)
        }
        try value.validate()
        return value
    }
}

public enum MiniCPMTTSSemanticError: Error, LocalizedError {
    case invalidConfiguration(String)
    case invalidWeights(String)
    case missingModelFile(URL)
    case invalidInput(String)
    /// A cooperative cancellation requested at a token boundary.  The
    /// duplex engine catches this on its own generation thread and performs
    /// the KV/TTS rollback there; an async transport must never mutate those
    /// caches concurrently.
    case cancelled

    public var errorDescription: String? {
        switch self {
        case .invalidConfiguration(let message):
            return "Invalid MiniCPM TTS semantic configuration: " + message
        case .invalidWeights(let message):
            return "Invalid MiniCPM TTS semantic weights: " + message
        case .missingModelFile(let url):
            return "Missing MiniCPM TTS semantic model file: " + url.path
        case .invalidInput(let message):
            return "Invalid MiniCPM TTS semantic input: " + message
        case .cancelled:
            return "MiniCPM TTS semantic generation was cancelled"
        }
    }
}
