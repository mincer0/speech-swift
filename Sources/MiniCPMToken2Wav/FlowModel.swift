import MLX
import MLXNN

public final class MiniCPMToken2WavFlowCache {
    public let conformer: MiniCPMUpsampleConformerCache
    public let decoder: MiniCPMFlowMatchingCache
    public let promptInputPositions: Int
    public let promptOutputPositions: Int

    public init(
        conformer: MiniCPMUpsampleConformerCache,
        decoder: MiniCPMFlowMatchingCache,
        promptInputPositions: Int = 0,
        promptOutputPositions: Int = 0
    ) {
        self.conformer = conformer
        self.decoder = decoder
        self.promptInputPositions = promptInputPositions
        self.promptOutputPositions = promptOutputPositions
    }
}

public struct MiniCPMFlowOutput {
    public let mel: MLXArray
    public let cache: MiniCPMToken2WavFlowCache?

    public init(mel: MLXArray, cache: MiniCPMToken2WavFlowCache?) {
        self.mel = mel
        self.cache = cache
    }
}

/// MiniCPM-o 4.5's speech-token to mel flow model.
public final class MiniCPMToken2WavFlowModel: Module {
    public let configuration: MiniCPMFlowConfiguration
    @ModuleInfo(key: "input_embedding") public var inputEmbedding: Embedding
    @ModuleInfo(key: "spk_embed_affine_layer") public var speakerProjection: Linear
    @ModuleInfo public var encoder: MiniCPMUpsampleConformerEncoder
    @ModuleInfo(key: "encoder_proj") public var encoderProjection: Linear
    @ModuleInfo public var decoder: MiniCPMConditionalFlowMatching

    public init(configuration: MiniCPMFlowConfiguration = .init()) {
        self.configuration = configuration
        _inputEmbedding.wrappedValue = Embedding(
            embeddingCount: configuration.tokenVocabularySize,
            dimensions: configuration.tokenDimension)
        _speakerProjection.wrappedValue = Linear(
            configuration.speakerEmbeddingDimension,
            configuration.melDimension)
        _encoder.wrappedValue = MiniCPMUpsampleConformerEncoder(
            configuration: configuration)
        _encoderProjection.wrappedValue = Linear(
            configuration.tokenDimension,
            configuration.melDimension)
        _decoder.wrappedValue = MiniCPMConditionalFlowMatching(
            configuration: configuration)
        super.init()
    }

    public func projectedSpeaker(_ embedding: MLXArray) -> MLXArray {
        let normalized = embedding / sqrt(
            (embedding * embedding).sum(axis: 1, keepDims: true) + 1e-8)
        return speakerProjection(normalized)
    }

    /// Offline token-to-mel inference. Prompt tokens precede generated tokens;
    /// prompt mel occupies the same initial 50 Hz frames and is removed from
    /// the returned synthesis.
    public func generate(
        tokens: MLXArray,
        speakerEmbedding: MLXArray,
        promptTokens: MLXArray? = nil,
        promptMel: MLXArray? = nil
    ) -> MiniCPMFlowOutput {
        let allTokens = promptTokens.map { concatenated([$0, tokens], axis: 1) } ?? tokens
        let encoded = encoder(inputEmbedding(maximum(allTokens, MLXArray(Int32(0)))))
        let mu = encoderProjection(encoded).transposed(0, 2, 1)
        let promptFrames = promptMel?.dim(2) ?? 0
        var condition = MLXArray.zeros(like: mu)
        if let promptMel, promptFrames > 0 {
            let suffix = MLXArray.zeros([
                mu.dim(0), mu.dim(1), mu.dim(2) - promptFrames,
            ], dtype: mu.dtype)
            condition = concatenated([promptMel, suffix], axis: 2)
        }
        let mask = MLXArray.ones([mu.dim(0), 1, mu.dim(2)], dtype: mu.dtype)
        let decoded = decoder.generate(
            mu: mu,
            speakers: projectedSpeaker(speakerEmbedding),
            condition: condition,
            validMask: mask)
        let mel = promptFrames > 0 ? decoded.output[0..., 0..., promptFrames...] : decoded.output
        return MiniCPMFlowOutput(mel: mel, cache: nil)
    }

    /// Streaming inference. Non-final chunks must contain the three right
    /// lookahead tokens; the encoder consumes only the stable prefix.
    public func generateStreaming(
        tokens: MLXArray,
        speakerEmbedding: MLXArray,
        cache: MiniCPMToken2WavFlowCache? = nil,
        isFinal: Bool = false
    ) -> MiniCPMFlowOutput {
        let encoded = encoder.streaming(
            inputEmbedding(maximum(tokens, MLXArray(Int32(0)))),
            cache: cache?.conformer,
            isFinal: isFinal)
        let mu = encoderProjection(encoded.output).transposed(0, 2, 1)
        let condition = MLXArray.zeros(like: mu)
        let decoded = decoder.generate(
            mu: mu,
            speakers: projectedSpeaker(speakerEmbedding),
            condition: condition,
            cache: cache?.decoder)
        var nextConformer = encoded.cache
        var nextDecoder = decoded.cache
        if nextDecoder.length > configuration.maximumStreamingFrames {
            nextConformer = nextConformer.trimmed(
                promptInputPositions: cache?.promptInputPositions ?? 0,
                promptOutputPositions: cache?.promptOutputPositions ?? 0,
                maximumRecentOutputPositions: configuration.maximumStreamingFrames)
            nextDecoder = nextDecoder.trimmed(
                promptPositions: cache?.promptOutputPositions ?? 0,
                maximumRecentPositions: configuration.maximumStreamingFrames)
        }
        return MiniCPMFlowOutput(
            mel: decoded.output,
            cache: MiniCPMToken2WavFlowCache(
                conformer: nextConformer,
                decoder: nextDecoder,
                promptInputPositions: cache?.promptInputPositions ?? 0,
                promptOutputPositions: cache?.promptOutputPositions ?? 0))
    }

    /// Builds the persistent prompt caches used by duplex synthesis.
    public func prepareStreaming(
        promptTokens: MLXArray,
        promptMel: MLXArray,
        speakerEmbedding: MLXArray,
        silenceToken: Int32 = 4_218
    ) -> MiniCPMToken2WavFlowCache {
        let rightLookahead = MLXArray(
            [Int32](repeating: silenceToken, count: configuration.preLookahead))
            .reshaped([1, configuration.preLookahead])
        let tokens = concatenated([promptTokens, rightLookahead], axis: 1)
        let encoded = encoder.streaming(inputEmbedding(tokens))
        let mu = encoderProjection(encoded.output).transposed(0, 2, 1)
        precondition(
            mu.dim(2) == promptMel.dim(2),
            "prompt token and mel lengths must align at the 2x flow rate")
        let decoded = decoder.generate(
            mu: mu,
            speakers: projectedSpeaker(speakerEmbedding),
            condition: promptMel)
        eval(mu, decoded.output)
        return MiniCPMToken2WavFlowCache(
            conformer: encoded.cache,
            decoder: decoded.cache,
            promptInputPositions: promptTokens.dim(1),
            promptOutputPositions: promptMel.dim(2))
    }
}
