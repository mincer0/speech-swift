import Foundation
import MLX

/// The two text-generation surfaces exposed by the official MiniCPM-o demo.
/// Full-duplex listen/speak orchestration lives in MiniCPMDuplexRuntime; this
/// session is intentionally limited to turn-based and half-duplex chat.
public enum MiniCPMChatGenerationMode: String, Sendable, Equatable {
    case turnBased = "turn_based"
    case halfDuplex = "half_duplex"
}

public struct MiniCPMChatGenerationConfig: Sendable, Equatable {
    public var mode: MiniCPMChatGenerationMode
    public var maxNewTokens: Int
    public var sampling: MiniCPMSamplingConfig

    public init(
        mode: MiniCPMChatGenerationMode = .turnBased,
        maxNewTokens: Int = 256,
        sampling: MiniCPMSamplingConfig = .init()
    ) {
        self.mode = mode
        self.maxNewTokens = max(0, maxNewTokens)
        self.sampling = sampling
    }
}

public enum MiniCPMChatFinishReason: String, Sendable, Equatable {
    case turnEOS = "turn_eos"
    case endOfText = "end_of_text"
    case listen
    case chunkEOS = "chunk_eos"
    case chunkTTSEOS = "chunk_tts_eos"
    case length
}

public struct MiniCPMChatPrefillResult {
    public let plan: MiniCPMChatInputPlan
    public let cacheLength: Int
    public let logits: MLXArray

    init(plan: MiniCPMChatInputPlan, cacheLength: Int, logits: MLXArray) {
        self.plan = plan
        self.cacheLength = cacheLength
        self.logits = logits
    }
}

public struct MiniCPMChatGenerationStep {
    public let index: Int
    public let token: Int
    public let text: String
    public let isListen: Bool
    public let endOfTurn: Bool

    init(index: Int, token: Int, text: String, isListen: Bool, endOfTurn: Bool) {
        self.index = index
        self.token = token
        self.text = text
        self.isListen = isListen
        self.endOfTurn = endOfTurn
    }
}

/// One generated text token and the decoder hidden state produced when that
/// token is fed back through the language model.
///
/// The chat decoder samples a token from the logits for the previous position,
/// then feeds the sampled token to obtain the next logits.  Keeping the token
/// and that returned hidden state together prevents callers (notably semantic
/// TTS) from accidentally shifting the two sequences by one position.  A
/// terminator sampled by ``generate(config:)`` is deliberately not represented
/// here: it is recorded in ``MiniCPMChatGenerationStep`` and in the protocol
/// history, but it is never fed as a text condition.
public struct MiniCPMChatGenerationAlignedSpan {
    public let index: Int
    public let token: Int
    /// Hidden state for ``token``.  Generation feeds one token at a time, so
    /// this is normally shaped ``[1, 1, hiddenSize]``.
    public let hidden: MLXArray

    init(index: Int, token: Int, hidden: MLXArray) {
        self.index = index
        self.token = token
        self.hidden = hidden
    }
}

public struct MiniCPMChatGenerationResult {
    public let tokens: [Int]
    public let text: String
    public let steps: [MiniCPMChatGenerationStep]
    /// Explicit token/hidden alignment for semantic consumers.  The array is
    /// intentionally not filtered by token type: every non-terminator token
    /// sampled by the decoder has exactly one span, including protocol tokens
    /// that a caller may choose to handle separately.
    public let alignedSpans: [MiniCPMChatGenerationAlignedSpan]
    public let finishReason: MiniCPMChatFinishReason
    public let cacheLength: Int

    init(
        tokens: [Int],
        text: String,
        steps: [MiniCPMChatGenerationStep],
        alignedSpans: [MiniCPMChatGenerationAlignedSpan] = [],
        finishReason: MiniCPMChatFinishReason,
        cacheLength: Int
    ) {
        self.tokens = tokens
        self.text = text
        self.steps = steps
        self.alignedSpans = alignedSpans
        self.finishReason = finishReason
        self.cacheLength = cacheLength
    }

    /// Hidden states in the same order as ``tokens``.  This convenience does
    /// not alter or filter the explicit spans; it is only a projection for
    /// consumers that need to concatenate the sequence for a model call.
    public var hiddenStates: [MLXArray] { alignedSpans.map(\.hidden) }
}

/// Snapshot of a turn/half-duplex chat session.  Model weights remain shared
/// and immutable; only this session's KV/RNG and pending logits are captured.
public struct MiniCPMChatSessionSnapshot {
    public let language: MiniCPMSessionSnapshot
    public let pendingLogits: MLXArray?
    public let pendingPlan: MiniCPMChatInputPlan?

    init(
        language: MiniCPMSessionSnapshot,
        pendingLogits: MLXArray?,
        pendingPlan: MiniCPMChatInputPlan?
    ) {
        self.language = language
        self.pendingLogits = pendingLogits
        self.pendingPlan = pendingPlan
    }
}

public enum MiniCPMChatSessionError: Error, LocalizedError, Sendable, Equatable {
    case noPrefill
    case emptyEmbedding(String)
    case unsupportedBatch

    public var errorDescription: String? {
        switch self {
        case .noPrefill:
            return "MiniCPM chat generation requires a preceding prefill"
        case .emptyEmbedding(let key):
            return "MiniCPM external embedding is empty: \(key)"
        case .unsupportedBatch:
            return "MiniCPM native chat supports batch size one"
        }
    }
}

/// Independent turn/half-duplex session over a shared immutable
/// ``MiniCPMLanguageModel``.  The template builder keeps ordered media slots;
/// this class is the narrow point where a caller resolves each slot to an
/// MLX embedding and feeds it into the KV cache.
public final class MiniCPMChatSession: @unchecked Sendable {
    public let model: MiniCPMLanguageModel
    public let tokenizer: MiniCPMTokenizer
    public let templateBuilder: MiniCPMChatTemplateBuilder
    public private(set) var language: MiniCPMSession

    private var pendingLogits: MLXArray?
    private var pendingPlan: MiniCPMChatInputPlan?

    public init(
        model: MiniCPMLanguageModel,
        tokenizer: MiniCPMTokenizer,
        samplerSeed: UInt64? = nil
    ) {
        self.model = model
        self.tokenizer = tokenizer
        self.templateBuilder = MiniCPMChatTemplateBuilder(tokenizer: tokenizer)
        self.language = MiniCPMSession(model: model, samplerSeed: samplerSeed)
    }

    public var cacheLength: Int { language.cacheLength }
    public var hasPrefill: Bool { pendingLogits != nil }

    public func reset() {
        language.reset()
        pendingLogits = nil
        pendingPlan = nil
    }

    public func snapshot() -> MiniCPMChatSessionSnapshot {
        MiniCPMChatSessionSnapshot(
            language: language.snapshot(),
            pendingLogits: pendingLogits,
            pendingPlan: pendingPlan)
    }

    public func restore(_ snapshot: MiniCPMChatSessionSnapshot) {
        language.restore(snapshot.language)
        pendingLogits = snapshot.pendingLogits
        pendingPlan = snapshot.pendingPlan
    }

    /// Build and feed a complete turn.  ``embeddingProvider`` is called once
    /// for each media part and receives the original role/message/part index.
    @discardableResult
    public func prefill(
        messages: [MiniCPMChatMessage],
        options: MiniCPMChatTemplateOptions = .init(),
        embeddingProvider: (MiniCPMExternalEmbeddingSlot) throws -> MLXArray
    ) throws -> MiniCPMChatPrefillResult {
        let plan = try templateBuilder.build(messages: messages, options: options)
        return try prefill(plan: plan, embeddingProvider: embeddingProvider)
    }

    /// Feed an already-rendered plan without rebuilding the Jinja template.
    @discardableResult
    public func prefill(
        plan: MiniCPMChatInputPlan,
        embeddingProvider: (MiniCPMExternalEmbeddingSlot) throws -> MLXArray
    ) throws -> MiniCPMChatPrefillResult {
        var last: MiniCPMForwardOutput?
        for segment in plan.segments {
            switch segment {
            case .tokens(let ids):
                guard !ids.isEmpty else { continue }
                let input = MLXArray(ids.map(Int32.init)).expandedDimensions(axis: 0)
                last = language.feed(inputIds: input)
            case .embedding(let slot):
                let embedding = try embeddingProvider(slot)
                guard embedding.size > 0 else {
                    throw MiniCPMChatSessionError.emptyEmbedding(slot.media.key)
                }
                last = language.feed(embeddings: embedding)
            }
        }
        guard let last else { throw MiniCPMChatSessionError.noPrefill }
        pendingLogits = last.logits
        pendingPlan = plan
        return MiniCPMChatPrefillResult(
            plan: plan,
            cacheLength: language.cacheLength,
            logits: last.logits)
    }

    /// Build one half-duplex streaming segment.  The caller can snapshot before
    /// feeding a segment and restore if endpointing/speculation rejects it.
    @discardableResult
    public func prefillStreaming(
        message: MiniCPMChatMessage,
        context: MiniCPMStreamingPromptContext,
        separator: MiniCPMChatContentSeparator = .none,
        enableThinking: Bool = false,
        useTTSTemplate: Bool = true,
        embeddingProvider: (MiniCPMExternalEmbeddingSlot) throws -> MLXArray
    ) throws -> MiniCPMChatPrefillResult {
        let plan = try templateBuilder.buildStreaming(
            message: message,
            context: context,
            separator: separator,
            enableThinking: enableThinking,
            useTTSTemplate: useTTSTemplate)
        return try prefill(plan: plan, embeddingProvider: embeddingProvider)
    }

    /// Decode a non-streaming turn or one half-duplex response from the latest
    /// prefill.  In half-duplex mode listen/chunk markers end this call; in
    /// turn-based mode only ``eos``/``turn_eos`` terminate generation.
    public func generate(
        config: MiniCPMChatGenerationConfig = .init()
    ) throws -> MiniCPMChatGenerationResult {
        guard var logits = pendingLogits else {
            throw MiniCPMChatSessionError.noPrefill
        }
        let protocolIDs = MiniCPMProtocolTokenIds(
            listen: tokenizer.specialTokens.listen,
            chunkEOS: tokenizer.specialTokens.chunkEOS,
            chunkTTSEOS: tokenizer.specialTokens.chunkTTSEOS,
            turnEOS: tokenizer.specialTokens.turnEOS,
            speak: tokenizer.specialTokens.speak,
            unitEnd: tokenizer.specialTokens.unitEnd,
            ttsBOS: tokenizer.specialTokens.ttsBOS,
            allSpecialTokenIds: tokenizer.allSpecialTokenIds)
        let terminators: Set<Int> = config.mode == .halfDuplex
            ? [tokenizer.specialTokens.listen, tokenizer.specialTokens.chunkEOS,
               tokenizer.specialTokens.chunkTTSEOS, tokenizer.specialTokens.turnEOS,
               tokenizer.specialTokens.eos]
            : [tokenizer.specialTokens.turnEOS, tokenizer.specialTokens.eos]
        let forbidden: [Int] = config.mode == .halfDuplex
            ? tokenizer.badTokenIds + [tokenizer.specialTokens.ttsPad]
            : tokenizer.badTokenIds + [tokenizer.specialTokens.ttsPad,
                                       tokenizer.specialTokens.chunkEOS,
                                       tokenizer.specialTokens.chunkTTSEOS,
                                       tokenizer.specialTokens.listen]

        var generated: [Int] = []
        var steps: [MiniCPMChatGenerationStep] = []
        var alignedSpans: [MiniCPMChatGenerationAlignedSpan] = []
        var reason: MiniCPMChatFinishReason = .length
        for index in 0..<config.maxNewTokens {
            let tokenArray = language.sample(
                logits: logits,
                config: config.sampling,
                protocol: protocolIDs,
                forbidden: forbidden,
                previousTokens: language.generatedTokens)
            let token = tokenArray.item(Int.self)
            let isListen = token == tokenizer.specialTokens.listen
            let endOfTurn = token == tokenizer.specialTokens.turnEOS
            if terminators.contains(token) {
                reason = Self.finishReason(for: token, tokenizer: tokenizer)
                steps.append(MiniCPMChatGenerationStep(
                    index: index,
                    token: token,
                    text: "",
                    isListen: isListen,
                    endOfTurn: endOfTurn))
                language.recordProtocolToken(token, special: true)
                break
            }

            generated.append(token)
            let text = tokenizer.decode([token], skipSpecialTokens: true)
            steps.append(MiniCPMChatGenerationStep(
                index: index,
                token: token,
                text: text,
                isListen: isListen,
                endOfTurn: endOfTurn))
            language.recordProtocolToken(token, special: tokenizer.allSpecialTokenIds.contains(token))
            let output = language.feed(
                inputIds: MLXArray([Int32(token)]).expandedDimensions(axis: 0))
            // Capture the hidden returned by the same feed that advances the
            // KV cache for this token.  Terminators return above and therefore
            // cannot enter this array, keeping the alignment one-to-one with
            // ``generated``/``tokens``.
            alignedSpans.append(MiniCPMChatGenerationAlignedSpan(
                index: index,
                token: token,
                hidden: output.hidden))
            logits = output.logits
            pendingLogits = logits
        }

        return MiniCPMChatGenerationResult(
            tokens: generated,
            text: tokenizer.decode(generated, skipSpecialTokens: true),
            steps: steps,
            alignedSpans: alignedSpans,
            finishReason: reason,
            cacheLength: language.cacheLength)
    }

    public func generateHalfDuplex(
        config: MiniCPMChatGenerationConfig = .init(mode: .halfDuplex)
    ) throws -> MiniCPMChatGenerationResult {
        var effective = config
        effective.mode = .halfDuplex
        return try generate(config: effective)
    }

    private static func finishReason(
        for token: Int,
        tokenizer: MiniCPMTokenizer
    ) -> MiniCPMChatFinishReason {
        if token == tokenizer.specialTokens.listen { return .listen }
        if token == tokenizer.specialTokens.chunkEOS { return .chunkEOS }
        if token == tokenizer.specialTokens.chunkTTSEOS { return .chunkTTSEOS }
        if token == tokenizer.specialTokens.turnEOS { return .turnEOS }
        return .endOfText
    }
}

/// Explicit engine spelling for callers that treat the session as a backend.
public typealias MiniCPMChatEngine = MiniCPMChatSession
