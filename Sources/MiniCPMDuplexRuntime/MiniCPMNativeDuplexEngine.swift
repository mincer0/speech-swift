import Foundation
import AVFoundation
import MLX
import MiniCPMAudio
import MiniCPMLLM
import MiniCPMTTSSemantic
import MiniCPMToken2Wav
import MiniCPMVision

private extension MiniCPMDuplexConfig {
    func nativeSlidingWindowConfig(
        markerTokenIDs: [Int],
        specialTokenIDs: Set<Int>
    ) -> MiniCPMSlidingWindowConfig {
        MiniCPMSlidingWindowConfig(
            mode: slidingWindowMode,
            basicHighTokens: basicWindowHighTokens,
            basicLowTokens: basicWindowLowTokens,
            contextPreviousMaxTokens: contextPreviousMaxTokens,
            contextMaxUnits: contextMaxUnits,
            contextPreviousMarkerTokenIDs: markerTokenIDs,
            specialTokenIDs: specialTokenIDs)
    }
}

public enum MiniCPMNativeDuplexEngineError: Error, LocalizedError, Sendable {
    case notPrepared
    case pendingUnit
    case noGenerationLogits
    case unsupported(String)
    case invalidInput(String)
    case audioUnavailable
    case visionUnavailable
    case rollbackFailed(String)

    public var errorDescription: String? {
        switch self {
        case .notPrepared:
            return "MiniCPM native duplex engine is not prepared"
        case .pendingUnit:
            return "MiniCPM native duplex engine has a unit awaiting finalize or rollback"
        case .noGenerationLogits:
            return "MiniCPM native duplex prefill produced no generation logits"
        case .unsupported(let message):
            return "MiniCPM native duplex unsupported: \(message)"
        case .invalidInput(let message):
            return "MiniCPM native duplex invalid input: \(message)"
        case .audioUnavailable:
            return "MiniCPM native audio encoder could not produce a complete chunk"
        case .visionUnavailable:
            return "MiniCPM native vision input requires a loaded vision bundle"
        case .rollbackFailed(let message):
            return "MiniCPM native duplex rollback failed: \(message)"
        }
    }
}

/// Bounds the official duplex policy's otherwise-unlimited sequence of
/// leading `<|listen|>` decisions after real user speech.  Startup silence is
/// deliberately unarmed, and the guard stands down as soon as a spoken model
/// unit begins so multi-chunk answers remain entirely model-driven.
struct MiniCPMResponseStarvationGuard: Sendable, Equatable {
    enum Intervention: Sendable, Equatable {
        case none
        case scaleListen(Float)
        case forbidListen
    }

    let scaleAfterListenUnits: Int
    let forbidAfterListenUnits: Int
    let rescueListenScale: Float
    private(set) var hasPendingUserSpeech = false
    private(set) var consecutiveListenUnits = 0

    init(
        // Each committed listen unit carries one second of silence. Three
        // units preserve the model's normal response timing; six cap the
        // user-visible post-speech wait before a response decision is due.
        scaleAfterListenUnits: Int = 3,
        forbidAfterListenUnits: Int = 6,
        rescueListenScale: Float = 0.5
    ) {
        let scaleLimit = max(1, scaleAfterListenUnits)
        self.scaleAfterListenUnits = scaleLimit
        self.forbidAfterListenUnits = max(scaleLimit, forbidAfterListenUnits)
        self.rescueListenScale = max(0, min(1, rescueListenScale))
    }

    mutating func observeSpeechActivity(_ active: Bool) {
        guard active else { return }
        hasPendingUserSpeech = true
        consecutiveListenUnits = 0
    }

    func intervention(currentTurnEnded: Bool) -> Intervention {
        guard hasPendingUserSpeech, currentTurnEnded else { return .none }
        if consecutiveListenUnits >= forbidAfterListenUnits {
            return .forbidListen
        }
        if consecutiveListenUnits >= scaleAfterListenUnits {
            return .scaleListen(rescueListenScale)
        }
        return .none
    }

    mutating func commit(isListen: Bool, endOfTurn: Bool) {
        guard hasPendingUserSpeech else { return }
        if endOfTurn || !isListen {
            reset()
        } else {
            consecutiveListenUnits += 1
        }
    }

    mutating func reset() {
        hasPendingUserSpeech = false
        consecutiveListenUnits = 0
    }
}

/// Detects the quantized decoder's degenerate phrase loop: the same phrase
/// or sentence repeating across consecutive spoken units instead of the turn
/// coming to an end.  The detection is ported from the reference duplex
/// coordinator (a short non-trivial phrase repeated four times at the tail)
/// and extended with a sentence-length rule, because one-second units slice
/// a loop at arbitrary points.  Once tripped, the next generation closes the
/// turn with the existing forced-listen path instead of letting the loop
/// consume the whole response budget.
struct MiniCPMResponseRepeatGuard: Sendable, Equatable {
    /// Enough tail context for the 3×24-character sentence rule plus margin.
    private static let detectionWindow = 600

    private(set) var responseText = ""
    private(set) var trippedPhrase: String?
    private(set) var unitCount = 0
    /// Matches the upstream streaming-answer safety cap: a response that
    /// still has not ended after this many one-second units is treated as a
    /// generation loop even when no repeated phrase matched.
    static let maxUnitsPerResponse = 120

    var tripped: Bool { trippedPhrase != nil }

    mutating func append(text: String) {
        guard !text.isEmpty else { return }
        unitCount += 1
        responseText = String((responseText + text).suffix(Self.detectionWindow))
        if trippedPhrase == nil, let phrase = Self.repeatedPhraseSuffix(in: responseText) {
            trippedPhrase = phrase
        }
        if unitCount >= Self.maxUnitsPerResponse {
            trippedPhrase = trippedPhrase ?? "(unit cap)"
        }
    }

    mutating func reset() {
        responseText = ""
        trippedPhrase = nil
        unitCount = 0
    }

    /// Return the repeated phrase when the normalized text ends with the
    /// same non-trivial phrase repeated consecutively: four times for short
    /// phrases (single-character runs such as 哈哈 or 谢谢 are excluded —
    /// they are common in laughter and acknowledgements), three times for
    /// sentence-length phrases.
    static func repeatedPhraseSuffix(in text: String) -> String? {
        let normalized = Array(text.filter { $0.isLetter || $0.isNumber })
        let count = normalized.count
        guard count >= 8 else { return nil }
        for phraseLength in 2...min(24, count / 3) {
            let repetitions = phraseLength <= 12 ? 4 : 3
            let span = phraseLength * repetitions
            guard count >= span else { continue }
            let suffix = Array(normalized[(count - span)...])
            let phrase = Array(suffix[0..<phraseLength])
            guard Set(phrase).count > 1 else { continue }
            var isLoop = true
            for repetition in 1..<repetitions where isLoop {
                for offset in 0..<phraseLength
                where suffix[repetition * phraseLength + offset] != phrase[offset] {
                    isLoop = false
                    break
                }
            }
            if isLoop { return String(phrase) }
        }
        return nil
    }
}

/// Concrete MLX MiniCPM-o duplex engine. Model weights are shared through
/// MiniCPMNativeModels, while every instance owns all mutable KV/cache state.
public final class MiniCPMNativeDuplexEngine: MiniCPMDuplexEngine {
    public let models: MiniCPMNativeModels

    private let imageProcessor = MiniCPMImageProcessor()
    private var languageSession: MiniCPMSession
    private var audioSession: MiniCPMAudioStreamingSession
    private var ttsSession: MiniCPMTTSSession?
    private var token2wav: MiniCPMToken2WavPipeline?

    private var mode: MiniCPMDuplexMode = .fullDuplex
    private var prepared = false
    private var prefetched = false
    private var interrupted = false
    /// Runtime-owned monotonic cancellation probe. The transport only flips
    /// a lock-protected epoch; this closure is polled on the engine thread so
    /// MLX/KV/TTS state is never mutated concurrently.
    private var interruptionProbe: (@Sendable () -> Bool)?
    private var pendingFinalize = false
    private var pendingLogits: MLXArray?
    private var pendingTerminator: Int?
    private var pendingInputType = "audio"
    private var pendingForceListenOverride = false
    private var pendingSampledListenSpeakDecision = false
    private var responseStarvationGuard = MiniCPMResponseStarvationGuard()
    /// Speech-loop detector over the currently open response.  Tripping it
    /// forces the next generation to close the turn (see generate()).
    private var responseRepeatGuard = MiniCPMResponseRepeatGuard()
    private var pendingGeneratedTokens: [Int] = []
    private var pendingGeneratedText = ""
    private var pendingIsListen = false
    private var pendingEndOfTurn = false
    private var audioHistory = MiniCPMDuplexAudioHistory()
    /// Exact encoder/frontend state at the start of the pending language unit.
    /// Restoring this checkpoint avoids replaying all committed PCM after a
    /// barge-in or generation error.
    private var pendingAudioSessionSnapshot: MiniCPMAudioStreamingSession.Snapshot?
    /// PCM admitted by the transport but not yet consumed by the streaming
    /// frontend.  The upstream coordinator pads the first request to
    /// ``FIRST_CHUNK_MS`` (16,560 samples), asks the native frontend for its
    /// effective 16,480-sample window, and retains the 80-sample remainder.
    /// Keeping this queue at the caller boundary avoids making the audio core
    /// guess transport packet boundaries.
    private var streamingAudioBuffer: [Float] = []
    private var streamingAudioInputStarted = false
    private var pendingStreamingAudioBufferSnapshot: [Float]?
    private var pendingStreamingAudioInputStartedSnapshot: Bool?
    private var slidingWindowConfig = MiniCPMSlidingWindowConfig()
    private var protocolState = MiniCPMDuplexProtocolState()
    private var pendingTTSnapshot: MiniCPMTTSSessionSnapshot?
    private var pendingTerminatorFed = false
    /// Conversation checkpoint taken before the first input of the current
    /// user turn.  A per-unit rollback is not sufficient for barge-in: live
    /// audio can span several already-committed listen units before the model
    /// starts speaking.  Keeping the checkpoint at turn scope lets cancel
    /// remove the whole interrupted question, not just its last response
    /// fragment.
    private struct ActiveTurnSnapshot {
        let language: MiniCPMSessionSnapshot
        let audio: MiniCPMAudioStreamingSession.Snapshot
        let tts: MiniCPMTTSSessionSnapshot?
        let audioHistory: MiniCPMDuplexAudioHistory
        let streamingAudioBuffer: [Float]
        let streamingAudioInputStarted: Bool
        let protocolState: MiniCPMDuplexProtocolState
    }
    private var activeTurnSnapshot: ActiveTurnSnapshot?
    private var diagnosticGenerationEpoch: UInt64?
    private var diagnosticGenerationStartedAtNS: UInt64?
    private var diagnosticGenerationExitedAtNS: UInt64?
    private var diagnosticCancelReturnedAtNS: UInt64?
    private let stageTimingEnabled: Bool
    /// Qwen3-backbone reasoning markers. The duplex sampler must never emit
    /// them: they are plain (non-special) tokens in this vocabulary, so
    /// `skipSpecialTokens` does not hide them, and a sampled `<think>` both
    /// corrupts the spoken text and leaks garbage into the semantic-TTS
    /// condition ("unknown speech" artifacts).
    private var reasoningForbiddenTokenIDs: [Int] = []
    private var latestAudioEncoderMS: Double?
    private var latestAudioBridgeMS: Double?
    private var latestLLMPrefillMS: Double?
    private var latestLLMDecodeMS: Double?
    private var latestSemanticTTSMS: Double?
    private var latestToken2WavMS: Double?
    private var latestFlowMS: Double?
    private var latestHiFTMS: Double?
    private var latestContinuationOnly = false

    public init(models: MiniCPMNativeModels, stageTimingEnabled: Bool? = nil) {
        self.models = models
        self.reasoningForbiddenTokenIDs = ["<think>", "</think>"].compactMap {
            models.tokenizer.tokenId($0)
        }
        self.stageTimingEnabled = stageTimingEnabled
            ?? Self.stageTimingRequestedFromEnvironment()
        languageSession = MiniCPMSession(model: models.llm)
        audioSession = MiniCPMAudioStreamingSession(model: models.audio)
        ttsSession = models.ttsSemantic?.makeSession()
        token2wav = try? models.makeToken2WavPipeline()
        token2wav?.setStageTimingEnabled(self.stageTimingEnabled)
    }

    /// Emit one opt-in raw-logit JSONL record.  The caller resolves the gate
    /// once per generation; this method is never reached in normal runs.
    private func emitLogitTrace(
        _ trace: MiniCPMLogitTrace,
        unit: Int,
        generate: Int,
        step: Int
    ) {
        var textCache: [Int: String] = [:]
        func tokenText(_ tokenID: Int) -> String {
            if let cached = textCache[tokenID] { return cached }
            let text = models.tokenizer.decode([tokenID], skipSpecialTokens: false)
            textCache[tokenID] = text
            return text
        }
        func entryObject(_ entry: MiniCPMLogitTraceEntry?) -> Any {
            guard let entry else { return NSNull() }
            let rawLogit: Any = entry.logit.isFinite ? entry.logit : NSNull()
            return [
                "token_id": entry.tokenID,
                "logit": rawLogit,
                "rank": entry.rank,
                "text": tokenText(entry.tokenID),
            ] as [String: Any]
        }
        let sampled: [String: Any] = [
            "token_id": trace.sampledToken,
            "text": tokenText(trace.sampledToken),
        ]
        let object: [String: Any] = [
            "unit": unit,
            "generate": generate,
            "step": step,
            "top_k": trace.topK.map { entryObject($0) },
            "listen": entryObject(trace.listen),
            "chunk_eos": entryObject(trace.chunkEOS),
            "chunk_tts_eos": entryObject(trace.chunkTTSEOS),
            "turn_eos": entryObject(trace.turnEOS),
            "speak": entryObject(trace.speak),
            "tts_bos": entryObject(trace.ttsBOS),
            "sampled_token": sampled,
        ]
        guard JSONSerialization.isValidJSONObject(object),
              let data = try? JSONSerialization.data(
                  withJSONObject: object, options: [.sortedKeys]) else {
            return
        }
        var line = data
        line.append(0x0A)
        FileHandle.standardError.write(line)
    }

    /// Protocol-only diagnostics; no model tensors are exposed.
    public var protocolTrace: [MiniCPMDuplexProtocolTraceEvent] {
        protocolState.trace
    }

    public var currentTurnEnded: Bool { protocolState.currentTurnEnded }

    public func setGenerationEpoch(_ epoch: UInt64) {
        diagnosticGenerationEpoch = epoch
    }

    /// Snapshot only scalar state.  This is intentionally callable after a
    /// serialized interrupt/rollback, never from the transport cancellation
    /// thread, so the reported KV/TTS values describe one coherent boundary.
    /// Visible-text budget for one duplex unit.
    ///
    /// Audio is pinned at 25 speech codes per unit (25 codes/s x 40 ms = 1.00 s),
    /// and natural Mandarin is ~5-6 characters per second, so ~6 characters is
    /// the largest amount of text one unit can actually speak. Raising this
    /// (the historical value was 28) lets a dense unit overrun the 26-code TTS
    /// cap and the utterance gets truncated mid-syllable.
    private static let maxVisibleCharactersPerUnit = 6

    /// Floor for one duplex unit, the counterpart of `maxVisibleCharactersPerUnit`.
    ///
    /// **Currently disabled (0).** Measuring the per-unit stage timings showed
    /// that suppressing the natural `chunk_eos` probe did NOT raise the text per
    /// unit at all (still 1-5 characters) but did push the model past its
    /// trained stop point, producing duplicated and garbled continuations
    /// ("但她她说", "我还以为是什那") plus audio that no longer matched the
    /// text. The natural probe is the model's real cadence; the block-internal
    /// silence it exposes has to be fixed on the audio side instead (let the
    /// TTS emit only as many codes as the text needs).
    private static let minVisibleCharactersPerUnit = 0

    public func diagnosticSnapshot() -> MiniCPMDuplexEngineDiagnosticSnapshot? {
        let window = languageSession.windowStats()
        let windowConfig = window["config"] as? [String: Any]
        let token2WavStats = token2wav?.contextStatistics()
        let ttsContextPresent: Bool = if let ttsSession {
            ttsSession.turnActive
                || ttsSession.cacheOffset > 0
                || !ttsSession.committedTokens.isEmpty
                || ttsSession.pendingToken != nil
        } else {
            false
        }
        return MiniCPMDuplexEngineDiagnosticSnapshot(
            generationEpoch: diagnosticGenerationEpoch,
            unitIndex: protocolState.unitIndex,
            currentTurnEnded: protocolState.currentTurnEnded,
            committedUnitCount: languageSession.units.count,
            kvCacheLength: languageSession.cacheLength,
            slidingWindowMode: windowConfig?["sliding_window_mode"] as? String,
            slidingWindowHighWatermark: windowConfig?["basic_window_high_tokens"] as? Int,
            slidingWindowLowWatermark: windowConfig?["basic_window_low_tokens"] as? Int,
            pendingTokenCount: pendingGeneratedTokens.count,
            pendingTerminator: pendingTerminator,
            prefetched: prefetched,
            needsFinalize: pendingFinalize,
            ttsContextPresent: ttsContextPresent,
            ttsPendingToken: ttsSession?.pendingToken,
            ttsCommittedTokenCount: ttsSession?.committedTokens.count ?? 0,
            ttsCacheLength: ttsSession?.cacheOffset ?? 0,
            token2WavPrepared: token2wav?.isPrepared ?? false,
            token2WavPendingTokenCount: token2wav?.pendingTokenCount,
            token2WavDecoderPositions: token2WavStats?.decoderPositions,
            continuationOnly: latestContinuationOnly,
            audioEncoderMS: latestAudioEncoderMS,
            audioBridgeMS: latestAudioBridgeMS,
            llmPrefillMS: latestLLMPrefillMS,
            llmDecodeMS: latestLLMDecodeMS,
            semanticTTSMS: latestSemanticTTSMS,
            token2WavMS: latestToken2WavMS,
            flowMS: latestFlowMS,
            hiftMS: latestHiFTMS,
            generationStartedAtNS: diagnosticGenerationStartedAtNS,
            generationExitedAtNS: diagnosticGenerationExitedAtNS,
            cancelReturnedAtNS: diagnosticCancelReturnedAtNS)
    }

    public func setInterruptionProbe(_ probe: (@Sendable () -> Bool)?) {
        interruptionProbe = probe
    }

    public func prepare(
        mode: MiniCPMDuplexMode,
        config: MiniCPMDuplexConfig
    ) throws -> String {
        interruptionProbe = nil
        guard !pendingFinalize && !prefetched else {
            throw MiniCPMNativeDuplexEngineError.pendingUnit
        }
        self.mode = mode
        let markerIDs = models.tokenizer.encode(
            config.contextPreviousMarker, addSpecialTokens: false)
        slidingWindowConfig = config.nativeSlidingWindowConfig(
            markerTokenIDs: markerIDs,
            specialTokenIDs: models.tokenizer.allSpecialTokenIds)
        clearPendingState()
        resetUnitStageTimings()
        responseStarvationGuard.reset()
        responseRepeatGuard.reset()
        // `prepare` starts a new conversation on this engine.  Audio chunks
        // committed by a previous conversation must not be replayed into the
        // new audio frontend (and must not affect a later rollback).
        audioHistory.reset()
        pendingAudioSessionSnapshot = nil
        streamingAudioBuffer.removeAll(keepingCapacity: true)
        streamingAudioInputStarted = false
        pendingStreamingAudioBufferSnapshot = nil
        pendingStreamingAudioInputStartedSnapshot = nil
        protocolState.reset()
        activeTurnSnapshot = nil
        pendingTTSnapshot = nil
        pendingTerminatorFed = false
        languageSession.reset()
        languageSession.configureSlidingWindow(slidingWindowConfig)
        audioSession.reset()
        ttsSession?.reset()
        if let token2wav, token2wav.isPrepared {
            token2wav.resetForNewTurn()
        }

        if config.generateAudio {
            guard let token2wav else {
                throw MiniCPMNativeDuplexEngineError.unsupported(
                    "audio generation requested but Token2Wav weights are not loaded")
            }
            if !token2wav.isPrepared, let reference = config.ttsReferenceAudio,
               !reference.isEmpty {
                _ = try models.prepareVoicePrompt(
                    pcm: reference,
                    sampleRate: config.ttsReferenceAudioSampleRate,
                    pipeline: token2wav)
            }
            guard token2wav.isPrepared else {
                throw MiniCPMNativeDuplexEngineError.unsupported(
                    "audio generation requested without a fixed or dynamic voice prompt")
            }
        }

        // The official duplex.prepare sequence is intentionally split around
        // the LLM reference embedding:
        //   [system prefix, <|audio_start|>] → [audio embedding] →
        //   [<|audio_end|>, <|im_end|>]
        // Feeding the complete textual prompt first shifts the embedding
        // outside the placeholder and changes every subsequent RoPE position.
        // With no LLM reference, omit both audio markers; an empty placeholder
        // is not equivalent to a missing system voice context.
        let llmReference = config.llmReferenceAudio
        let hasLLMReference = !(llmReference?.isEmpty ?? true)
        let promptPrefix = "<|im_start|>system\n\(config.systemPrompt)"
            + (hasLLMReference ? "\n<|audio_start|>" : "")
        let promptSuffix = hasLLMReference
            ? "<|audio_end|><|im_end|>"
            : "<|im_end|>"
        languageSession.beginSystemPrompt()
        let prefixIDs = models.tokenizer.encode(
            promptPrefix, addSpecialTokens: false)
        if !prefixIDs.isEmpty {
            _ = languageSession.feed(inputIds: MLXArray(prefixIDs.map(Int32.init)))
        }
        if hasLLMReference, let reference = llmReference {
            // A reference clip is system conditioning, not live-stream audio:
            // use the offline frontend path so it cannot pollute the next
            // turn's streaming mel/KV cache.
            let normalized = try Self.resampleReferenceAudio(
                reference,
                from: config.llmReferenceAudioSampleRate)
            let encoded = try audioSession.encodeReferenceAudio(normalized)
            _ = languageSession.feed(embeddings: encoded.output.embeddings)
        }
        languageSession.markSystemPromptSuffix()
        let suffixIDs = models.tokenizer.encode(
            promptSuffix, addSpecialTokens: false)
        if !suffixIDs.isEmpty {
            _ = languageSession.feed(inputIds: MLXArray(suffixIDs.map(Int32.init)))
        }
        languageSession.finishSystemPrompt()
        let prompt = hasLLMReference
            ? promptPrefix + "[audio embedding]" + promptSuffix
            : promptPrefix + promptSuffix
        prepared = true
        interrupted = false
        return prompt
    }

    public func prefill(
        mode: MiniCPMDuplexMode,
        input: MiniCPMDuplexInput
    ) throws -> MiniCPMDuplexPrefillResult {
        // ``interrupted`` is a one-turn state flag set by the previous
        // cancel. A subsequent prefill starts a fresh unit and must clear it;
        // only the live epoch probe can abort this prefill before it begins.
        guard !(interruptionProbe?() ?? false) else {
            interrupted = true
            return MiniCPMDuplexPrefillResult(accepted: false, mode: mode, input: input)
        }
        interrupted = false
        guard prepared else {
            throw MiniCPMNativeDuplexEngineError.notPrepared
        }
        guard !pendingFinalize && !prefetched else {
            throw MiniCPMNativeDuplexEngineError.pendingUnit
        }
        if let legacy = input.frames, !legacy.isEmpty {
            throw MiniCPMNativeDuplexEngineError.unsupported(
                "frames:[[Float]] is ambiguous; send encoded frameData instead")
        }
        if let messages = input.messages, !messages.isEmpty {
            // Role/template expansion is intentionally not approximated by
            // flattening ordered multimodal parts into this streaming API.
            // Until the native chat-template implementation is available,
            // reject it explicitly instead of returning a plausible but
            // semantically wrong answer.
            throw MiniCPMNativeDuplexEngineError.unsupported(
                "ordered role-bearing messages require the MiniCPM chat template path")
        }

        if let voice = input.voicePCM, !voice.isEmpty {
            let pipeline = try requireToken2WavPipeline()
            _ = try models.prepareVoicePrompt(
                pcm: voice,
                sampleRate: input.voiceSampleRate ?? 16_000,
                pipeline: pipeline)
        }

        let hasAudio = !(input.audio?.isEmpty ?? true)
        let hasFrames = !(input.frameData?.isEmpty ?? true)
        let hasText = !(input.text?.isEmpty ?? true)
        let continuationRequested = input.continuationOnly
            || Self.parseBoolean(input.metadata["continuation_tick"]) == true
            || Self.parseBoolean(input.metadata["continuationOnly"]) == true
        let continuationOnly = Self.shouldUseContinuationOnly(
            requested: continuationRequested,
            currentTurnEnded: protocolState.currentTurnEnded,
            audio: input.audio,
            hasFrames: hasFrames,
            hasText: hasText)
        guard hasAudio || hasFrames || hasText else {
            throw MiniCPMNativeDuplexEngineError.invalidInput(
                "prefill requires audio, frameData, or text")
        }
        resetUnitStageTimings(continuationOnly: continuationOnly)

        if activeTurnSnapshot == nil {
            activeTurnSnapshot = ActiveTurnSnapshot(
                language: languageSession.snapshot(),
                audio: audioSession.snapshot(),
                tts: ttsSession?.snapshot(),
                audioHistory: audioHistory,
                streamingAudioBuffer: streamingAudioBuffer,
                streamingAudioInputStarted: streamingAudioInputStarted,
                protocolState: protocolState)
        }

        _ = languageSession.beginUnit()
        _ = protocolState.beginUnit()
        pendingTTSnapshot = ttsSession?.snapshot()
        pendingAudioSessionSnapshot = audioSession.snapshot()
        pendingStreamingAudioBufferSnapshot = streamingAudioBuffer
        pendingStreamingAudioInputStartedSnapshot = streamingAudioInputStarted
        audioHistory.beginUnit()
        pendingTerminatorFed = false
        // A short first audio packet can be buffered by the mel frontend and
        // return before an accepted language unit exists. Preserve a forced
        // listen/barge-in request across that retry instead of letting the
        // next call's pending-state clear erase it.
        let inheritedForceListen = pendingForceListenOverride
        let inheritedSampledDecision = pendingSampledListenSpeakDecision
        let requestedForceListen = input.forceListen
            ?? Self.parseBoolean(input.metadata["force_listen"])
            ?? Self.parseBoolean(input.metadata["forceListen"])
            ?? false
        let requestedSampledDecision = input.allowSampledListenSpeakDecision
            ?? Self.parseBoolean(input.metadata["allow_sampled_listen_speak_decision"])
            ?? Self.parseBoolean(input.metadata["allowSampledListenSpeakDecision"])
            ?? false
        let requestedSpeechActivity = Self.parseBoolean(input.metadata["speech_active"])
            ?? Self.parseBoolean(input.metadata["speechActive"])
            ?? false
        clearPendingState()
        pendingForceListenOverride = inheritedForceListen || requestedForceListen
        pendingSampledListenSpeakDecision = inheritedSampledDecision || requestedSampledDecision
        responseStarvationGuard.observeSpeechActivity(requestedSpeechActivity)
        pendingInputType = hasAudio && hasFrames ? "omni"
            : hasFrames ? "vision"
            : hasAudio ? "audio" : "text"

        do {
            let unitTokenStart = DispatchTime.now().uptimeNanoseconds
            var last = feedToken(models.tokenizer.specialTokens.unit)
            if stageTimingEnabled {
                latestLLMPrefillMS = elapsedMS(since: unitTokenStart)
            }

            if hasFrames {
                guard !interruptionRequested() else {
                    throw MiniCPMNativeDuplexEngineError.invalidInput("prefill interrupted")
                }
                guard let vision = models.vision else {
                    throw MiniCPMNativeDuplexEngineError.visionUnavailable
                }
                let frameCount = input.frameData?.count ?? 0
                // An explicitly supplied empty list carries the same meaning
                // as an omitted value. Do not let it fall through to the
                // per-frame branch below: indexing `[]` would otherwise trap
                // before the caller receives a structured input error.
                let requestedMaxSlices = input.maxSliceNums
                    ?? Self.parseIntList(input.metadata["max_slice_nums"])
                let maxSliceValues = Self.normalizedMaxSliceNums(requestedMaxSlices)
                try Self.validateMaxSliceNums(maxSliceValues, frameCount: frameCount)
                for (frameIndex, data) in (input.frameData ?? []).enumerated() {
                    guard !interruptionRequested() else {
                        throw MiniCPMNativeDuplexEngineError.invalidInput("prefill interrupted")
                    }
                    let image = try imageProcessor.image(from: data)
                    let maxSlices: Int? = if let maxSliceValues {
                        maxSliceValues.count == 1
                            ? maxSliceValues[0]
                            : maxSliceValues[frameIndex]
                    } else {
                        nil
                    }
                    let batch = try imageProcessor.process(
                        [image], maxSliceNums: maxSlices)
                    let groups = try vision.encode(batch)
                    for group in groups {
                        guard !interruptionRequested() else {
                            throw MiniCPMNativeDuplexEngineError.invalidInput("prefill interrupted")
                        }
                        last = try feedRequiredToken(
                            models.tokenizer.specialTokens.image,
                            after: last)
                        last = feedVisionEmbedding(group.overview)
                        last = try feedRequiredToken(
                            models.tokenizer.specialTokens.imageEnd,
                            after: last)
                        for slice in group.slices {
                            guard !interruptionRequested() else {
                                throw MiniCPMNativeDuplexEngineError.invalidInput("prefill interrupted")
                            }
                            last = try feedRequiredToken(
                                models.tokenizer.specialTokens.slice,
                                after: last)
                            last = feedVisionEmbedding(slice)
                            last = try feedRequiredToken(
                                models.tokenizer.specialTokens.sliceEnd,
                                after: last)
                        }
                    }
                }
            }

            if hasAudio, !continuationOnly, let samples = input.audio {
                guard !interruptionRequested() else {
                    throw MiniCPMNativeDuplexEngineError.invalidInput("prefill interrupted")
                }
                audioHistory.accept(samples, producedEmbedding: false)
                // ``StreamingMelProcessor`` intentionally owns only model
                // frontend state.  Match the pinned coordinator's transport
                // contract here: a first 1,000 ms packet is left-padded to
                // 1,035 ms (16,560 samples), then only the effective 1,030 ms
                // (16,480 samples) window is sent to the frontend.  The 80
                // sample tail remains queued for the next regular 16,000
                // sample window.  This lets a normal 16,000-sample browser
                // packet produce an accepted first unit instead of waiting for
                // another packet.
                streamingAudioBuffer.append(contentsOf: samples)
                if !streamingAudioInputStarted {
                    streamingAudioInputStarted = true
                    let required = Self.firstStreamingChunkRequiredSamples
                    if streamingAudioBuffer.count < required {
                        streamingAudioBuffer.insert(
                            contentsOf: repeatElement(
                                0,
                                count: required - streamingAudioBuffer.count),
                            at: 0)
                    }
                }
                let needed = audioSession.melProcessor.nextChunkSampleCount
                guard streamingAudioBuffer.count >= needed else {
                    _ = languageSession.rollbackUnit()
                    _ = protocolState.rollbackUnit()
                    pendingTTSnapshot = nil
                    pendingAudioSessionSnapshot = nil
                    // Keep this raw chunk in the frontend tail: the next
                    // accepted chunk completes the same convolution window.
                    audioHistory.preserveTailAfterRejectedUnit()
                    // `rollbackUnit()` restores language/protocol/audio state
                    // but intentionally does not know about this per-input
                    // latch. Keep it until a real accepted unit is generated
                    // (or an explicit rollback/reset clears all pending
                    // state).
                    pendingForceListenOverride = inheritedForceListen || requestedForceListen
                    pendingSampledListenSpeakDecision =
                        inheritedSampledDecision || requestedSampledDecision
                    return MiniCPMDuplexPrefillResult(
                        accepted: false, mode: mode, input: input)
                }
                let chunk = Array(streamingAudioBuffer.prefix(needed))
                let audioEncoderStart = DispatchTime.now().uptimeNanoseconds
                let audioEncoding = try audioSession.acceptAudio(chunk)
                if stageTimingEnabled, let audioEncoding {
                    eval(audioEncoding.output.encoderStates)
                    latestAudioEncoderMS = elapsedMS(since: audioEncoderStart)
                    let audioBridgeStart = DispatchTime.now().uptimeNanoseconds
                    eval(audioEncoding.output.embeddings)
                    latestAudioBridgeMS = elapsedMS(since: audioBridgeStart)
                }
                guard let encoded = audioEncoding else {
                    _ = languageSession.rollbackUnit()
                    _ = protocolState.rollbackUnit()
                    pendingTTSnapshot = nil
                    if let checkpoint = pendingAudioSessionSnapshot {
                        audioSession.restore(checkpoint)
                    }
                    pendingAudioSessionSnapshot = nil
                    audioHistory.preserveTailAfterRejectedUnit()
                    pendingForceListenOverride = inheritedForceListen || requestedForceListen
                    pendingSampledListenSpeakDecision =
                        inheritedSampledDecision || requestedSampledDecision
                    return MiniCPMDuplexPrefillResult(
                        accepted: false, mode: mode, input: input)
                }
                // The first request is 16,480 samples even though its padded
                // transport buffer is 16,560. Subsequent requests are exactly
                // whatever the frontend reports (normally 16,000).
                streamingAudioBuffer.removeFirst(
                    min(needed, streamingAudioBuffer.count))
                let llmAudioPrefillStart = DispatchTime.now().uptimeNanoseconds
                last = languageSession.feed(embeddings: encoded.output.embeddings)
                if stageTimingEnabled {
                    latestLLMPrefillMS = (latestLLMPrefillMS ?? 0)
                        + elapsedMS(since: llmAudioPrefillStart)
                }
            }

            if hasText, let text = input.text {
                guard !interruptionRequested() else {
                    throw MiniCPMNativeDuplexEngineError.invalidInput("prefill interrupted")
                }
                let ids = models.tokenizer.encode(text, addSpecialTokens: false)
                if !ids.isEmpty {
                    let llmTextPrefillStart = DispatchTime.now().uptimeNanoseconds
                    last = languageSession.feed(
                        inputIds: MLXArray(ids.map(Int32.init)))
                    if stageTimingEnabled {
                        latestLLMPrefillMS = (latestLLMPrefillMS ?? 0)
                            + elapsedMS(since: llmTextPrefillStart)
                    }
                }
            }
            pendingLogits = last.logits
            prefetched = true
            interrupted = false
            return MiniCPMDuplexPrefillResult(
                accepted: true, mode: mode, input: input)
        } catch {
            _ = languageSession.rollbackUnit()
            _ = protocolState.rollbackUnit()
            pendingTTSnapshot = nil
            _ = audioHistory.rollbackUnit()
            if let snapshot = pendingStreamingAudioBufferSnapshot {
                streamingAudioBuffer = snapshot
            }
            if let snapshot = pendingStreamingAudioInputStartedSnapshot {
                streamingAudioInputStarted = snapshot
            }
            if let checkpoint = pendingAudioSessionSnapshot {
                audioSession.restore(checkpoint)
            }
            pendingAudioSessionSnapshot = nil
            pendingStreamingAudioBufferSnapshot = nil
            pendingStreamingAudioInputStartedSnapshot = nil
            throw error
        }
    }

    public func generate(
        mode: MiniCPMDuplexMode,
        config: MiniCPMDuplexConfig
    ) throws -> MiniCPMDuplexOutput {
        guard prepared else {
            throw MiniCPMNativeDuplexEngineError.notPrepared
        }
        guard prefetched else {
            throw MiniCPMNativeDuplexEngineError.noGenerationLogits
        }
        if interruptionRequested() { return interruptionOutput() }
        guard var logits = pendingLogits else {
            throw MiniCPMNativeDuplexEngineError.noGenerationLogits
        }

        diagnosticGenerationStartedAtNS = DispatchTime.now().uptimeNanoseconds
        diagnosticGenerationExitedAtNS = nil
        defer {
            diagnosticGenerationExitedAtNS = DispatchTime.now().uptimeNanoseconds
        }
        let llmDecodeStart = DispatchTime.now().uptimeNanoseconds

        let ids = models.tokenizer.specialTokens
        let protocolTokenIDs = MiniCPMDuplexProtocolTokenIDs(ids)
        let allowsListen = mode == .fullDuplex || mode == .omni
        var protocolConfig = config
        if !allowsListen { protocolConfig.forceListenCount = 0 }
        let sampling = MiniCPMSamplingConfig(
            mode: .sampling,
            temperature: config.temperature,
            topK: config.topK,
            topP: config.topP,
            repetitionPenalty: config.repetitionPenalty,
            repetitionWindowSize: config.repetitionWindowSize,
            lengthPenalty: config.lengthPenalty,
            listenTopK: allowsListen ? config.listenTopK : nil,
            listenProbabilityScale: allowsListen ? config.listenProbabilityScale : 0)
        let logitTraceTopK = MiniCPMSampler.logitTraceTopK()

        // A decoder phrase loop must close the turn the same way a client
        // forced listen does: feed turn_eos, sample <|listen|>, and release
        // the TTS/Token2Wav turn state.  Consuming the trip here makes the
        // guard re-armable for a later loop within the same session.
        let repeatForcedListen = responseRepeatGuard.tripped
        if repeatForcedListen { responseRepeatGuard.reset() }
        let forceListenOverride = pendingForceListenOverride || repeatForcedListen
        let forceListenGeneration = protocolState.beginGeneration(
            config: protocolConfig,
            forceListenOverride: forceListenOverride)
        if forceListenOverride,
           protocolState.closeTurnForForcedListen(turnEOS: ids.turnEOS) {
            // Barge-in must close an already speaking turn in KV before
            // forcing <|listen|>. Otherwise the next unit inherits an open
            // TTS turn and can emit fragments or repeat audio.
            let closed = languageSession.feed(
                inputIds: MLXArray([Int32(ids.turnEOS)]))
            languageSession.recordProtocolToken(ids.turnEOS, special: true)
            logits = closed.logits
            ttsSession?.finishTurn()
            if let token2wav, token2wav.isPrepared {
                token2wav.interruptAndReset()
            }
        }
        var generated: [Int] = []
        var hidden: [MLXArray] = []
        var isListen = false
        var endOfTurn = false
        var terminator: Int?
        let limit = max(0, config.maxNewTokens)
        if limit > 0 {
            for index in 0..<limit {
                if interruptionRequested() { return interruptionOutput() }
                let step: MiniCPMDuplexProtocolStep
                if index == limit - 1, config.lsMode.lowercased() == "explicit" {
                    step = protocolState.finishAtLimit(
                        index: index, tokens: protocolTokenIDs, config: protocolConfig)
                } else {
                    // Pinned unified upstream initializes tts_pad in the
                    // decoder's forbidden list. Its later "suppress after
                    // previous chunk" branch is unreachable because that
                    // token is never removed. Keep the actual source behavior
                    // here instead of following the stale branch comment.
                    var forbidden = Self.forbiddenTokenIDs(
                        badTokenIds: models.tokenizer.badTokenIds,
                        ttsPadID: ids.ttsPad)
                    for special in [ids.chunkEOS]
                        where !forbidden.contains(special) {
                        forbidden.append(special)
                    }
                    if !allowsListen { forbidden.append(ids.listen) }
                    for marker in reasoningForbiddenTokenIDs
                        where !forbidden.contains(marker) {
                        forbidden.append(marker)
                    }
                    var stepSampling = sampling
                    if index == 0, allowsListen, !forceListenGeneration {
                        switch responseStarvationGuard.intervention(
                            currentTurnEnded: protocolState.currentTurnEnded) {
                        case .none:
                            break
                        case .scaleListen(let scale):
                            stepSampling.listenProbabilityScale = min(
                                stepSampling.listenProbabilityScale, scale)
                        case .forbidListen:
                            if !forbidden.contains(ids.listen) {
                                forbidden.append(ids.listen)
                            }
                        }
                    }
                    if index == 0,
                       Self.shouldUseGreedyListenSpeakDecision(
                           currentTurnEnded: protocolState.currentTurnEnded,
                           configuredGreedy: config.greedyListenSpeakDecision,
                           allowSampledDecision: pendingSampledListenSpeakDecision) {
                        stepSampling.mode = .greedy
                    }
                    // Keep the text budget and the audio budget in step.
                    //
                    // The audio side is pinned at 25 speech codes per duplex unit
                    // (`MiniCPMToken2WavPipelineConfiguration.chunkTokenCount`),
                    // i.e. exactly 1.00 s of waveform for ~6 characters of
                    // Mandarin. The official natural `chunk_eos` probe, however,
                    // returns the *unmasked* argmax and happily ends a unit after
                    // 1-3 visible characters. The surplus then renders as silence
                    // inside the block (measured 0.3-0.98 s per block), which is
                    // audible as broken-up, choppy speech. Hold the probe back
                    // until this unit has collected enough visible text; the
                    // masked sample below still applies the full protocol policy,
                    // and `turn_eos` / barge-in paths are untouched.
                    stepSampling.suppressNaturalChunkEOS =
                        models.tokenizer.decode(generated, skipSpecialTokens: true).count
                            < Self.minVisibleCharactersPerUnit
                    let tokenArray = languageSession.sample(
                        logits: logits,
                        config: stepSampling,
                        protocol: MiniCPMProtocolTokenIds(
                            listen: ids.listen,
                            chunkEOS: ids.chunkEOS,
                            chunkTTSEOS: ids.chunkTTSEOS,
                            turnEOS: ids.turnEOS,
                            speak: ids.speak,
                            unitEnd: ids.unitEnd,
                            ttsBOS: ids.ttsBOS,
                            allSpecialTokenIds: models.tokenizer.allSpecialTokenIds),
                        forbidden: forbidden,
                        previousTokens: languageSession.generatedTokens + protocolState.generatedIDs)
                    let sampled = tokenArray.item(Int.self)
                    if let logitTraceTopK, !forceListenGeneration {
                        let trace = MiniCPMSampler.logitTrace(
                            logits: logits,
                            protocol: MiniCPMProtocolTokenIds(
                                listen: ids.listen,
                                chunkEOS: ids.chunkEOS,
                                chunkTTSEOS: ids.chunkTTSEOS,
                                turnEOS: ids.turnEOS,
                                speak: ids.speak,
                                unitEnd: ids.unitEnd,
                                ttsBOS: ids.ttsBOS,
                                allSpecialTokenIds: models.tokenizer.allSpecialTokenIds),
                            topK: logitTraceTopK,
                            sampledToken: sampled)
                        emitLogitTrace(
                            trace,
                            unit: protocolState.unitIndex,
                            generate: max(protocolState.generateCount - 1, 0),
                            step: index)
                    }
                    let isChunkTerminator = sampled == ids.listen
                        || sampled == ids.chunkEOS
                        || sampled == ids.chunkTTSEOS
                    // Per-unit visible-text budget. The audio side is pinned at
                    // 25 speech codes per duplex unit, i.e. 25 codes/s x 40 ms =
                    // exactly 1.00 s of waveform, and natural Mandarin runs at
                    // roughly 5-6 characters per second. A unit therefore must
                    // not accumulate more than ~6 visible characters: with the
                    // previous bound of 28 a dense unit produced 7-8 characters
                    // against a 26-code cap, so the tail of the utterance was
                    // truncated - audible as swallowed syllables. Official
                    // MiniCPM-o binds the same ratio from the other side
                    // (`generate_chunk_size = 10` text tokens per 25-code chunk
                    // in `modeling_minicpmo.py`). Release the candidate token and
                    // defer a chunk_eos instead of feeding it to KV.
                    let reachesCharacterLimit = index != 0
                        && !isChunkTerminator
                        && models.tokenizer.decode(
                            generated + [sampled], skipSpecialTokens: true).count
                            >= Self.maxVisibleCharactersPerUnit
                    if reachesCharacterLimit {
                        step = protocolState.finishAtCharacterLimit(
                            index: index,
                            tokens: protocolTokenIDs,
                            config: protocolConfig)
                    } else {
                        step = protocolState.consume(
                            sampledToken: sampled,
                            index: index,
                            tokens: protocolTokenIDs,
                            config: protocolConfig,
                            allowListen: allowsListen)
                    }
                }

                if step.isListen { isListen = true }
                if step.endOfTurn { endOfTurn = true }
                if step.collectForTTS {
                    generated.append(step.token)
                }
                if step.shouldFeed {
                    if interruptionRequested() { return interruptionOutput() }
                    let output = languageSession.feed(
                        inputIds: MLXArray([Int32(step.token)]))
                    if step.collectForTTS { hidden.append(output.hidden) }
                    logits = output.logits
                }
                if step.shouldStop {
                    terminator = config.lsMode.lowercased() == "explicit"
                        ? step.token : nil
                    pendingTerminatorFed = step.shouldFeed
                    break
                }
                if interruptionRequested() { return interruptionOutput() }
            }
        }
        if isListen {
            hidden.removeAll(keepingCapacity: true)
            responseRepeatGuard.reset()
        }

        let text = generated.isEmpty
            ? ""
            : models.tokenizer.decode(generated, skipSpecialTokens: true)
        if !isListen && !text.isEmpty {
            responseRepeatGuard.append(text: text)
        }
        if stageTimingEnabled {
            latestLLMDecodeMS = elapsedMS(since: llmDecodeStart)
        }
        var waveform: [Float]?
        // Gate the vocoder on speakable text: a unit whose collected tokens
        // are all protocol markers (e.g. a bare <|speak|> stutter) would hand
        // the semantic TTS a condition with no speech-start content and come
        // out as a full unit of unrecognizable noise.
        if !isListen && !text.isEmpty && config.generateAudio,
           let semantic = models.ttsSemantic,
           let ttsSession,
           let token2wav,
           token2wav.isPrepared {
            if interruptionRequested() { return interruptionOutput() }
            let semanticStart = DispatchTime.now().uptimeNanoseconds
            let hiddenSequence = concatenated(hidden, axis: 1)
            let tokenSequence = MLXArray(generated.map(Int32.init))
            let condition = try semantic.buildCondition(
                hidden: hiddenSequence,
                tokenIDs: tokenSequence,
                addAudioBOS: true,
                addTextEOS: false)
            pendingTTSnapshot = ttsSession.snapshot()
            do {
                let tts = try semantic.generateStreamingChunk(
                    inputsEmbeds: condition,
                    session: ttsSession,
                    endOfTurn: endOfTurn,
                    temperature: config.ttsTemperature,
                    repetitionPenalty: config.ttsRepetitionPenalty,
                    eosToken: semantic.configuration.numAudioTokens - 1,
                    // `false` matches the official duplex loop, which never
                    // uses `force_no_stop`. Mid-turn EOS is already prevented
                    // upstream by `generateStreamingChunk` computing
                    // `minNewTokens = (isFirst || endOfTurn) ? 0 : 26`, and a
                    // first chunk is *allowed* to stop early ("allow decoding
                    // <1s audio") without ending the turn - see the note in
                    // `MiniCPMTTSSemantic.generateStreamingChunk`.
                    forceNoStop: false,
                    maxNewToken: 26,
                    resetAfterEnd: false,
                    shouldCancel: interruptionProbe)
                if stageTimingEnabled {
                    latestSemanticTTSMS = elapsedMS(since: semanticStart)
                }
                if interruptionRequested() { return interruptionOutput() }
                let token2WavStart = DispatchTime.now().uptimeNanoseconds
                let chunks = try token2wav.append(
                    tts.committedTokens.map(Int32.init),
                    // Match MiniCPMODuplex: only the first semantic chunk forces
                    // a short-window flush. Middle chunks use the normal 25+3
                    // look-ahead cadence; isFinal drains the remaining window.
                    // `isFinalChunk` is now exactly `endOfTurn` (see the
                    // semantic layer), so the teardown below fires only at a
                    // real turn boundary and stays in sync with the
                    // finishTurn() contract.
                    forceFlush: tts.isFirstChunk,
                    isFinal: tts.isFinalChunk,
                    shouldCancel: interruptionProbe)
                if interruptionRequested() { return interruptionOutput() }
                if !chunks.isEmpty {
                    waveform = chunks.flatMap { samples($0.waveform) }
                }
                if stageTimingEnabled {
                    latestToken2WavMS = elapsedMS(since: token2WavStart)
                    latestFlowMS = token2wav.lastStageTiming?.flowMS
                    latestHiFTMS = token2wav.lastStageTiming?.hiftMS
                }
                if tts.isFinalChunk {
                    // The wrapper intentionally leaves the session alive until
                    // Token2Wav has consumed its 25 committed codes.  Re-open
                    // the immutable prompt cache after the final waveform flush
                    // so the next duplex chunk starts a fresh turn.
                    // Required contract: `generateStreamingChunk` refuses a new
                    // chunk while a final one is pending, which is why the
                    // decoder must be stopped from ending the turn itself
                    // mid-sentence (see `forceNoStop` above).
                    ttsSession.finishTurn()
                    token2wav.resetForNewTurn()
                }
            } catch MiniCPMTTSSemanticError.cancelled {
                return interruptionOutput()
            } catch MiniCPMToken2WavError.cancelled {
                return interruptionOutput()
            }
        }
        if endOfTurn {
            // A turn can end before any text/TTS code is emitted.  Keep the
            // semantic and waveform stages in the same idle state anyway.
            ttsSession?.finishTurn()
            if let token2wav, token2wav.isPrepared {
                token2wav.resetForNewTurn()
            }
            responseRepeatGuard.reset()
        }

        var audioPaddingSamples: Int?
        // A listen boundary always returns a real one-second silence payload;
        // this keeps downstream playback and capture clocks aligned. Listen
        // markers are not sent to the speaker, so they have no trim metadata.
        if isListen {
            waveform = [Float](repeating: 0, count: 24_000)
        } else if !endOfTurn, let currentWaveform = waveform,
                  currentWaveform.count < 24_000 {
            let padding = 24_000 - currentWaveform.count
            audioPaddingSamples = padding
            waveform = [Float](repeating: 0, count: padding)
                + currentWaveform
        }

        pendingFinalize = true
        prefetched = false
        pendingLogits = nil
        pendingTerminator = terminator
        pendingGeneratedTokens = generated
        pendingGeneratedText = text
        pendingIsListen = isListen
        pendingEndOfTurn = endOfTurn
        if interruptionRequested() { return interruptionOutput() }
        return MiniCPMDuplexOutput(
            text: text,
            audio: waveform,
            audioPaddingSamples: audioPaddingSamples,
            isListen: isListen,
            endOfTurn: endOfTurn,
            needsFinalize: true,
            finishReason: isListen ? "listen" : (endOfTurn ? "turn_end" : "chunk"))
    }

    public func finalize() throws {
        guard pendingFinalize else { return }
        let completedTurn = pendingEndOfTurn
        if let pendingTerminator {
            // Unified MiniCPM defers the chunk terminator and feeds it in the
            // same forward pass as `</unit>`.  Do not run a separate logits
            // step for the terminator during generation.
            _ = languageSession.feed(inputIds: MLXArray([
                Int32(pendingTerminator),
                Int32(models.tokenizer.specialTokens.unitEnd),
            ]))
            languageSession.recordProtocolToken(pendingTerminator, special: true)
        } else {
            _ = languageSession.feed(
                inputIds: MLXArray([Int32(models.tokenizer.specialTokens.unitEnd)]))
        }
        languageSession.recordProtocolToken(
            models.tokenizer.specialTokens.unitEnd, special: true)
        languageSession.commitUnit(
            generatedTokens: pendingGeneratedTokens,
            generatedText: pendingGeneratedText,
            isListen: pendingIsListen,
            inputType: pendingInputType)
        protocolState.commitUnit(unitEndID: models.tokenizer.specialTokens.unitEnd)
        audioHistory.commitUnit()
        pendingAudioSessionSnapshot = nil
        pendingStreamingAudioBufferSnapshot = nil
        pendingStreamingAudioInputStartedSnapshot = nil
        // Match the fixed upstream order: commit first, then enforce the
        // configured basic/context policy. A rejected generation never reaches
        // this point, so rollback cannot evict any previously accepted unit.
        _ = languageSession.enforceSlidingWindow()
        pendingTTSnapshot = nil
        pendingTerminatorFed = false
        responseStarvationGuard.commit(
            isListen: pendingIsListen,
            endOfTurn: pendingEndOfTurn)
        clearPendingState()
        if completedTurn {
            // The turn checkpoint is no longer needed once the answer's
            // canonical end has been accepted.  Releasing it here also avoids
            // pinning the pre-turn KV cache across later conversation turns.
            activeTurnSnapshot = nil
        }
    }

    public func rollback() throws {
        guard pendingFinalize || prefetched else { return }
        _ = languageSession.rollbackUnit()
        _ = protocolState.rollbackUnit()
        if let pendingTTSnapshot, let ttsSession {
            try? ttsSession.rollback(to: pendingTTSnapshot)
        }
        pendingTTSnapshot = nil
        _ = audioHistory.rollbackUnit()
        if let snapshot = pendingStreamingAudioBufferSnapshot {
            streamingAudioBuffer = snapshot
        }
        if let snapshot = pendingStreamingAudioInputStartedSnapshot {
            streamingAudioInputStarted = snapshot
        }
        if let checkpoint = pendingAudioSessionSnapshot {
            audioSession.restore(checkpoint)
        }
        pendingAudioSessionSnapshot = nil
        pendingStreamingAudioBufferSnapshot = nil
        pendingStreamingAudioInputStartedSnapshot = nil
        clearPendingState()
        // Keep the turn checkpoint: a transport rollback rejected only this
        // unit, while the user turn is still live and may receive more audio.
        if let token2wav, token2wav.isPrepared {
            token2wav.interruptAndReset()
        }
        // The discarded unit never reached the transport, so its text must
        // not count toward the speech-loop detector.
        responseRepeatGuard.reset()
        pendingTerminatorFed = false
    }

    public func interrupt() {
        defer {
            diagnosticCancelReturnedAtNS = DispatchTime.now().uptimeNanoseconds
        }
        interrupted = true
        repairInterruptedTurn()
    }

    public func reset() {
        interruptionProbe = nil
        token2wav?.releaseContext()
        token2wav = try? models.makeToken2WavPipeline()
        token2wav?.setStageTimingEnabled(stageTimingEnabled)
        languageSession.reset()
        audioSession.reset()
        ttsSession?.reset()
        protocolState.reset()
        pendingTTSnapshot = nil
        pendingTerminatorFed = false
        audioHistory.reset()
        pendingAudioSessionSnapshot = nil
        streamingAudioBuffer.removeAll(keepingCapacity: true)
        streamingAudioInputStarted = false
        pendingStreamingAudioBufferSnapshot = nil
        pendingStreamingAudioInputStartedSnapshot = nil
        activeTurnSnapshot = nil
        responseStarvationGuard.reset()
        responseRepeatGuard.reset()
        clearPendingState()
        prepared = false
        interrupted = false
        diagnosticGenerationEpoch = nil
        diagnosticGenerationStartedAtNS = nil
        diagnosticGenerationExitedAtNS = nil
        diagnosticCancelReturnedAtNS = nil
        resetUnitStageTimings()
    }

    public func setSlidingWindow(tokens: Int?) {
        let normalized = tokens.map { max(0, $0) }
        slidingWindowConfig = MiniCPMSlidingWindowConfig(
            mode: normalized == nil ? "off" : "basic",
            basicHighTokens: normalized ?? slidingWindowConfig.basicHighTokens,
            basicLowTokens: normalized ?? slidingWindowConfig.basicLowTokens,
            contextPreviousMaxTokens: slidingWindowConfig.contextPreviousMaxTokens,
            contextMaxUnits: slidingWindowConfig.contextMaxUnits,
            contextPreviousMarkerTokenIDs: slidingWindowConfig.contextPreviousMarkerTokenIDs,
            specialTokenIDs: slidingWindowConfig.specialTokenIDs)
        languageSession.configureSlidingWindow(slidingWindowConfig)
    }

    public func setSlidingWindow(config: MiniCPMDuplexConfig) {
        let markerIDs = models.tokenizer.encode(
            config.contextPreviousMarker, addSpecialTokens: false)
        slidingWindowConfig = config.nativeSlidingWindowConfig(
            markerTokenIDs: markerIDs,
            specialTokenIDs: models.tokenizer.allSpecialTokenIds)
        languageSession.configureSlidingWindow(slidingWindowConfig)
    }

    private func clearPendingState() {
        prefetched = false
        pendingFinalize = false
        pendingLogits = nil
        pendingTerminator = nil
        pendingGeneratedTokens.removeAll(keepingCapacity: true)
        pendingGeneratedText = ""
        pendingIsListen = false
        pendingEndOfTurn = false
        pendingForceListenOverride = false
        pendingSampledListenSpeakDecision = false
    }

    private func resetUnitStageTimings(continuationOnly: Bool = false) {
        latestContinuationOnly = continuationOnly
        latestAudioEncoderMS = nil
        latestAudioBridgeMS = nil
        latestLLMPrefillMS = nil
        latestLLMDecodeMS = nil
        latestSemanticTTSMS = nil
        latestToken2WavMS = nil
        latestFlowMS = nil
        latestHiFTMS = nil
    }

    private func elapsedMS(since start: UInt64) -> Double {
        Double(DispatchTime.now().uptimeNanoseconds - start) / 1_000_000.0
    }

    private static func stageTimingRequestedFromEnvironment() -> Bool {
        guard let value = ProcessInfo.processInfo.environment[
            "MINICPM_STAGE_TIMING"
        ]?.lowercased() else {
            return false
        }
        return value == "1" || value == "true" || value == "yes"
    }

    private func interruptionRequested() -> Bool {
        interrupted || (interruptionProbe?() ?? false)
    }

    /// Abort from the engine's own thread.  Rollback touches MLX KV, audio
    /// frontend snapshots, semantic TTS and Token2Wav state, so it must remain
    /// serialized with the synchronous generation loop rather than being
    /// invoked by the network task.
    private func interruptionOutput() -> MiniCPMDuplexOutput {
        interrupted = true
        // The generation loop can observe the nonisolated epoch before the
        // actor receives the transport's explicit control frame. Repair here,
        // on the same serialized engine thread, so the first stale return
        // cannot erase the information needed to close a provisional spoken
        // unit.
        repairInterruptedTurn()
        return MiniCPMDuplexOutput(
            isListen: true, needsFinalize: false, finishReason: "interrupt")
    }

    /// Roll back the provisional unit and close the interrupted language turn
    /// as one serialized operation.  The important distinction is between
    /// the protocol state before and after rollback: a spoken unit can be
    /// provisional while the last committed unit is a listen unit, leaving
    /// `currentTurnEnded == true` after rollback even though the language KV
    /// still needs a canonical `turn_eos + </unit>` boundary.
    private func repairInterruptedTurn() {
        responseStarvationGuard.reset()
        responseRepeatGuard.reset()
        if let snapshot = activeTurnSnapshot {
            restoreActiveTurn(snapshot)
            activeTurnSnapshot = nil
            // The next accepted input is a one-unit model fence.  It is set
            // after restoring pending state because clearPendingState()
            // intentionally clears per-input overrides.
            pendingForceListenOverride = true
            return
        }

        let hadPendingUnit = pendingFinalize || prefetched
        let hadOpenTurnBeforeRollback = !protocolState.currentTurnEnded
        let hadPendingSpokenOutput = pendingFinalize
            && !pendingIsListen
            && !pendingEndOfTurn
        let needsTurnRepair = hadOpenTurnBeforeRollback || hadPendingSpokenOutput

        // Capture the decision above before rollback clears the provisional
        // protocol/session metadata. `rollback()` is idempotent when the
        // generation already repaired itself on an epoch probe.
        if hadPendingUnit { try? rollback() }

        var repaired = 0
        if needsTurnRepair {
            let ids = models.tokenizer.specialTokens
            repaired = languageSession.closeInterruptedTurn(
                turnEosId: ids.turnEOS,
                unitEndId: ids.unitEnd)
            if repaired > 0 {
                _ = protocolState.closeInterruptedTurn(
                    turnEOS: ids.turnEOS,
                    unitEnd: ids.unitEnd,
                    force: true)
                ttsSession?.finishTurn()
                if let token2wav, token2wav.isPrepared {
                    token2wav.interruptAndReset()
                }
            }
        }

        // The next input gets one listen-only unit as a transport/model fence.
        // `prefill` preserves this latch if its first audio window is short.
        // Keep it even when there was no language unit to repair; that is the
        // safe boundary for cancel-before-first-token and repeated cancel.
        pendingForceListenOverride = true
        _ = repaired
    }

    /// Restore the state immediately before the current user turn.  This is
    /// deliberately one serialized operation on the engine thread: language
    /// KV, audio frontend cache, protocol history and TTS all describe the
    /// same turn boundary and must never be repaired independently.
    private func restoreActiveTurn(_ snapshot: ActiveTurnSnapshot) {
        languageSession.restore(snapshot.language)
        audioSession.restore(snapshot.audio)
        audioHistory = snapshot.audioHistory
        streamingAudioBuffer = snapshot.streamingAudioBuffer
        streamingAudioInputStarted = snapshot.streamingAudioInputStarted
        protocolState = snapshot.protocolState
        if let ttsSnapshot = snapshot.tts {
            try? ttsSession?.rollback(to: ttsSnapshot)
        } else {
            ttsSession?.reset()
        }
        if let token2wav, token2wav.isPrepared {
            token2wav.interruptAndReset()
        }
        pendingAudioSessionSnapshot = nil
        pendingStreamingAudioBufferSnapshot = nil
        pendingStreamingAudioInputStartedSnapshot = nil
        pendingTTSnapshot = nil
        pendingTerminatorFed = false
        clearPendingState()
    }

    private static let firstStreamingChunkRequiredSamples =
        1_035 * 16_000 / 1_000

    private static func parseBoolean(_ value: String?) -> Bool? {
        guard let value else { return nil }
        switch value.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() {
        case "1", "true", "yes", "on": return true
        case "0", "false", "no", "off": return false
        default: return nil
        }
    }

    static func shouldUseGreedyListenSpeakDecision(
        currentTurnEnded: Bool,
        configuredGreedy: Bool,
        allowSampledDecision: Bool
    ) -> Bool {
        currentTurnEnded && configuredGreedy && !allowSampledDecision
    }

    static func shouldUseContinuationOnly(
        requested: Bool,
        currentTurnEnded: Bool,
        audio: [Float]?,
        hasFrames: Bool,
        hasText: Bool
    ) -> Bool {
        // Official MiniCPM duplex semantics: EVERY prefill unit carries real
        // audio embeddings - the model hears the continuous time axis, which
        // is what keeps unit-boundary text coherent. The former zero-silence
        // skip (continuation shortcut) starved the model of that context and
        // caused boundary garbling; with the bucketed KV cache the encoder
        // costs ~16ms per chunk, so the shortcut is retired outright.
        _ = requested
        _ = currentTurnEnded
        _ = audio
        _ = hasFrames
        _ = hasText
        return false
    }

    private static func parseIntList(_ value: String?) -> [Int]? {
        guard let value else { return nil }
        let parsed = value
            .split(separator: ",")
            .compactMap { Int($0.trimmingCharacters(in: .whitespacesAndNewlines)) }
        return parsed.isEmpty ? nil : parsed
    }

    /// Match the fixed unified MiniCPM decoder: ``tts_pad`` is part of the
    /// initial forbidden set, not merely a one-chunk suppression marker. The
    /// upstream implementation's later conditional append/remove is dead
    /// because this token is never removed from that set.
    static func forbiddenTokenIDs(
        badTokenIds: [Int],
        ttsPadID: Int
    ) -> [Int] {
        var forbidden = badTokenIds
        if ttsPadID >= 0, !forbidden.contains(ttsPadID) {
            forbidden.append(ttsPadID)
        }
        return forbidden
    }

    /// Normalize the optional vision slicing override at the runtime
    /// boundary. An explicitly supplied empty array is equivalent to an
    /// omitted override; keeping that invariant here prevents the per-frame
    /// loop from indexing an empty list.
    static func normalizedMaxSliceNums(_ values: [Int]?) -> [Int]? {
        guard let values, !values.isEmpty else { return nil }
        return values
    }

    /// Validate the official ``max_slice_nums`` shape before decoding any
    /// frame. A scalar applies to every frame; otherwise the list must have
    /// exactly one positive value per frame.
    static func validateMaxSliceNums(
        _ values: [Int]?,
        frameCount: Int
    ) throws {
        guard frameCount >= 0 else {
            throw MiniCPMNativeDuplexEngineError.invalidInput(
                "frame count must not be negative")
        }
        guard let values else { return }
        guard values.allSatisfy({ $0 > 0 }) else {
            throw MiniCPMNativeDuplexEngineError.invalidInput(
                "maxSliceNums values must be positive")
        }
        guard values.count == 1 || values.count == frameCount else {
            throw MiniCPMNativeDuplexEngineError.invalidInput(
                "maxSliceNums must contain one value or one value per frame")
        }
    }

    private func feedToken(_ token: Int) -> MiniCPMForwardOutput {
        languageSession.feed(inputIds: MLXArray([Int32(token)]))
    }

    private func feedRequiredToken(
        _ token: Int?,
        after previous: MiniCPMForwardOutput
    ) throws -> MiniCPMForwardOutput {
        guard let token, token >= 0 else {
            throw MiniCPMNativeDuplexEngineError.unsupported(
                "loaded tokenizer has no required visual protocol token")
        }
        _ = previous
        return feedToken(token)
    }

    private func feedVisionEmbedding(
        _ embedding: MLXArray
    ) -> MiniCPMForwardOutput {
        let value = embedding.ndim == 3
            ? embedding
            : embedding.expandedDimensions(axis: 0)
        return languageSession.feed(embeddings: value)
    }

    private func requireToken2WavPipeline() throws -> MiniCPMToken2WavPipeline {
        guard let token2wav else {
            throw MiniCPMNativeDuplexEngineError.unsupported(
                "Token2Wav weights are not loaded")
        }
        return token2wav
    }

    private func samples(_ waveform: MLXArray) -> [Float] {
        eval(waveform)
        return waveform.asType(.float32).asArray(Float.self)
    }

    /// MiniCPM's Whisper frontend is fixed at 16 kHz.  Keep resampling at the
    /// runtime boundary so callers may pass the 44.1/48 kHz PCM commonly
    /// produced by AVAudioEngine without silently changing the model's time
    /// axis.  AVAudioConverter is drained to end-of-stream to retain the SRC
    /// filter tail; the result is normalized to the exact rational frame count
    /// expected by the frontend.
    private static func resampleReferenceAudio(
        _ samples: [Float],
        from sampleRate: Int
    ) throws -> [Float] {
        guard sampleRate > 0 else {
            throw MiniCPMNativeDuplexEngineError.invalidInput(
                "LLM reference sample rate must be positive")
        }
        guard !samples.isEmpty else { return samples }
        let targetRate = 16_000
        guard sampleRate != targetRate else { return samples }
        guard samples.allSatisfy(\.isFinite) else {
            throw MiniCPMNativeDuplexEngineError.invalidInput(
                "LLM reference audio contains NaN or infinity")
        }
        guard let sourceFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: Double(sampleRate),
            channels: 1,
            interleaved: false),
            let targetFormat = AVAudioFormat(
                commonFormat: .pcmFormatFloat32,
                sampleRate: Double(targetRate),
                channels: 1,
                interleaved: false),
            let converter = AVAudioConverter(
                from: sourceFormat, to: targetFormat),
            let source = AVAudioPCMBuffer(
                pcmFormat: sourceFormat,
                frameCapacity: AVAudioFrameCount(samples.count))
        else {
            throw MiniCPMNativeDuplexEngineError.unsupported(
                "cannot create \(sampleRate) Hz → 16 kHz reference resampler")
        }
        source.frameLength = AVAudioFrameCount(samples.count)
        samples.withUnsafeBufferPointer { pointer in
            source.floatChannelData![0].update(
                from: pointer.baseAddress!, count: samples.count)
        }
        let ratio = Double(targetRate) / Double(sampleRate)
        let capacity = AVAudioFrameCount(
            ceil(Double(samples.count) * ratio)) + 4096
        guard let target = AVAudioPCMBuffer(
            pcmFormat: targetFormat, frameCapacity: capacity) else {
            throw MiniCPMNativeDuplexEngineError.unsupported(
                "cannot allocate reference resampler output")
        }
        var fed = false
        var result: [Float] = []
        result.reserveCapacity(Int(capacity))
        while true {
            target.frameLength = 0
            var conversionError: NSError?
            let status = converter.convert(to: target, error: &conversionError) {
                _, outStatus in
                if fed {
                    outStatus.pointee = .endOfStream
                    return nil
                }
                fed = true
                outStatus.pointee = .haveData
                return source
            }
            if let conversionError {
                throw MiniCPMNativeDuplexEngineError.unsupported(
                    "reference resampling failed: \(conversionError.localizedDescription)")
            }
            if target.frameLength > 0, let channel = target.floatChannelData?[0] {
                result.append(contentsOf: UnsafeBufferPointer(
                    start: channel, count: Int(target.frameLength)))
            }
            if status == .error {
                throw MiniCPMNativeDuplexEngineError.unsupported(
                    "reference resampling failed")
            }
            if status == .endOfStream { break }
            guard target.frameLength > 0 else {
                throw MiniCPMNativeDuplexEngineError.unsupported(
                    "reference resampler made no progress")
            }
        }
        let expected = max(1, Int((Double(samples.count) * ratio).rounded()))
        if result.count > expected {
            result.removeLast(result.count - expected)
        } else if result.count < expected {
            result.append(contentsOf: repeatElement(0, count: expected - result.count))
        }
        return result
    }
}
