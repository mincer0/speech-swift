import Foundation
import CoreGraphics
import MLX
import MiniCPMAudio
import MiniCPMLLM
import MiniCPMTTSSemantic
import MiniCPMToken2Wav
import MiniCPMVision

/// Media owned by a turn/half-duplex request.  The chat-template layer keeps
/// only a stable key for each value; this enum is the native side of that
/// indirection and deliberately has no JSON/backend dependencies.
public enum MiniCPMNativeChatMedia: @unchecked Sendable {
    case audio(samples: [Float], sampleRate: Int)
    case image(Data)
    /// A video remains one ordered content part.  Frames are concatenated
    /// into one embedding slot, so a video cannot be reordered with adjacent
    /// text/audio by a legacy "all frames first" path.  An MP4 audio track is
    /// retained when the transport decoder supplied one and is encoded in the
    /// same slot after the visual sequence.
    case video(frames: [Data], audio: [Float]?, sampleRate: Int?)
}

public struct MiniCPMNativeChatInput: @unchecked Sendable {
    public let messages: [MiniCPMChatMessage]
    public let media: [String: MiniCPMNativeChatMedia]
    public let templateOptions: MiniCPMChatTemplateOptions
    public let streamingContext: MiniCPMStreamingPromptContext?
    /// Use the streaming audio frontend for a sequence of short chunks.  A
    /// complete utterance can still use ``streamingContext`` (to retain the
    /// role boundary) while encoding its accumulated waveform offline.
    public let streamingAudio: Bool
    /// A complete messages array normally contains the entire conversation
    /// and therefore starts from a clean session.  Streaming segments set
    /// this to false after the first accepted segment.
    public let resetSession: Bool

    public init(
        messages: [MiniCPMChatMessage],
        media: [String: MiniCPMNativeChatMedia] = [:],
        templateOptions: MiniCPMChatTemplateOptions = .init(),
        streamingContext: MiniCPMStreamingPromptContext? = nil,
        streamingAudio: Bool = false,
        resetSession: Bool = false
    ) {
        self.messages = messages
        self.media = media
        self.templateOptions = templateOptions
        self.streamingContext = streamingContext
        self.streamingAudio = streamingAudio
        self.resetSession = resetSession
    }
}

public struct MiniCPMNativeChatPrefillResult: Sendable, Equatable {
    public let accepted: Bool
    public let needsMoreAudio: Bool
    public let cacheLength: Int

    public init(accepted: Bool, needsMoreAudio: Bool = false, cacheLength: Int = 0) {
        self.accepted = accepted
        self.needsMoreAudio = needsMoreAudio
        self.cacheLength = cacheLength
    }
}

public struct MiniCPMNativeChatOutput: Sendable, Equatable {
    public let text: String
    /// Optional 24 kHz Float32 PCM synthesized from the generated text. Chat
    /// callers receive text and audio in one transactional output so a failed
    /// transport write can roll both language and waveform state back.
    public let audio: [Float]?
    public let isListen: Bool
    public let endOfTurn: Bool
    public let reason: String
    /// Chat generation is transactional just like duplex generation: the
    /// caller must acknowledge the wire events with ``finalize()``.  If the
    /// transport write fails, ``stop()`` restores the prefill snapshot.
    public let needsFinalize: Bool

    public init(
        text: String,
        audio: [Float]? = nil,
        isListen: Bool,
        endOfTurn: Bool,
        reason: String,
        needsFinalize: Bool = true
    ) {
        self.text = text
        self.audio = audio
        self.isListen = isListen
        self.endOfTurn = endOfTurn
        self.reason = reason
        self.needsFinalize = needsFinalize
    }
}

public enum MiniCPMNativeChatEngineError: Error, LocalizedError, Sendable, Equatable {
    case notPrepared
    case pendingResponse
    case noPrefill
    case invalidInput(String)
    case unsupported(String)
    case mediaMissing(String)
    case audioUnavailable(String)
    case visionUnavailable
    case rollbackFailed(String)

    public var errorDescription: String? {
        switch self {
        case .notPrepared:
            return "MiniCPM native chat engine is not prepared"
        case .pendingResponse:
            return "MiniCPM native chat engine has a response awaiting finalize or stop"
        case .noPrefill:
            return "MiniCPM native chat generation requires a preceding prefill"
        case .invalidInput(let message):
            return "MiniCPM native chat invalid input: \(message)"
        case .unsupported(let message):
            return "MiniCPM native chat unsupported: \(message)"
        case .mediaMissing(let key):
            return "MiniCPM native chat media slot is missing: \(key)"
        case .audioUnavailable(let message):
            return "MiniCPM native chat audio encoding failed: \(message)"
        case .visionUnavailable:
            return "MiniCPM native chat vision input requires a loaded vision bundle"
        case .rollbackFailed(let message):
            return "MiniCPM native chat rollback failed: \(message)"
        }
    }
}

/// Production turn/half-duplex decoder over the shared native MiniCPM model
/// weights.  Every instance owns a ``MiniCPMChatSession`` and media/frontend
/// caches; only immutable model modules are shared by ``MiniCPMNativeModels``.
public final class MiniCPMNativeChatEngine: @unchecked Sendable {
    public let models: MiniCPMNativeModels
    public let session: MiniCPMChatSession

    private let imageProcessor = MiniCPMImageProcessor()
    private var audioSession: MiniCPMAudioStreamingSession
    private var mode: MiniCPMChatGenerationMode = .turnBased
    private var generationConfig = MiniCPMChatGenerationConfig()
    private var prepared = false
    private var pending = false
    private var interrupted = false
    private var prefillSnapshot: MiniCPMChatSessionSnapshot?
    private var audioSnapshot: MiniCPMAudioStreamingSession.Snapshot?
    private var ttsSession: MiniCPMTTSSession?
    private var token2wav: MiniCPMToken2WavPipeline?
    private var ttsPrompt: MiniCPMToken2WavPrompt?
    private var ttsVoicePCM: [Float]?
    private var ttsVoiceSampleRate = 16_000
    private var ttsEnabled = false
    private var ttsTemperature: Float = 0.8
    private var ttsRepetitionPenalty: Float = 1.05
    private var ttsRecords: [MiniCPMChatTTSAppendRecord] = []
    private var prefillTTSSnapshot: MiniCPMTTSSessionSnapshot?
    private var prefillTTSRecords: [MiniCPMChatTTSAppendRecord] = []
    private var pendingTTSFinal = false

    private struct MiniCPMChatTTSAppendRecord {
        let tokens: [Int32]
        let forceFlush: Bool
        let isFinal: Bool
    }

    public init(models: MiniCPMNativeModels, samplerSeed: UInt64? = nil) {
        self.models = models
        self.session = MiniCPMChatSession(
            model: models.llm,
            tokenizer: models.tokenizer,
            samplerSeed: samplerSeed)
        self.audioSession = MiniCPMAudioStreamingSession(model: models.audio)
    }

    public var cacheLength: Int { session.cacheLength }
    public var isPrepared: Bool { prepared }

    public func prepare(
        mode: MiniCPMChatGenerationMode,
        maxNewTokens: Int = 256,
        sampling: MiniCPMSamplingConfig = .init(),
        ttsEnabled: Bool = false,
        ttsTemperature: Float = 0.8,
        ttsRepetitionPenalty: Float = 1.05,
        ttsVoicePCM: [Float]? = nil,
        ttsVoiceSampleRate: Int = 16_000
    ) throws {
        self.mode = mode
        generationConfig = MiniCPMChatGenerationConfig(
            mode: mode,
            maxNewTokens: maxNewTokens,
            sampling: sampling)
        session.reset()
        audioSession.reset()
        self.ttsEnabled = ttsEnabled
        self.ttsTemperature = ttsTemperature
        self.ttsRepetitionPenalty = ttsRepetitionPenalty
        self.ttsVoicePCM = ttsVoicePCM
        self.ttsVoiceSampleRate = ttsVoiceSampleRate
        self.ttsSession = nil
        self.token2wav = nil
        self.ttsPrompt = nil
        self.ttsRecords.removeAll(keepingCapacity: true)
        self.prefillTTSSnapshot = nil
        self.prefillTTSRecords.removeAll(keepingCapacity: true)
        self.pendingTTSFinal = false
        if ttsEnabled {
            try prepareTTS()
        }
        prefillSnapshot = nil
        audioSnapshot = nil
        pending = false
        interrupted = false
        prepared = true
    }

    /// Replace the per-session Token2Wav voice prompt between turns. A prompt
    /// is prepared lazily from raw PCM when the converted prompt encoder is
    /// available; fixed prompts loaded with the model remain untouched.
    public func setTTSVoicePrompt(
        pcm: [Float]?,
        sampleRate: Int = 16_000
    ) throws {
        guard prepared else { throw MiniCPMNativeChatEngineError.notPrepared }
        guard !pending else { throw MiniCPMNativeChatEngineError.pendingResponse }
        if ttsVoicePCM == pcm, ttsVoiceSampleRate == sampleRate {
            return
        }
        ttsVoicePCM = pcm
        ttsVoiceSampleRate = sampleRate
        guard ttsEnabled else { return }
        try resetTTSState(keepPrompt: false)
        try prepareTTS()
    }

    /// Update per-request sampling controls before prefill.  The backend
    /// protocol carries generation options on each turn (rather than only in
    /// ``session.init``); keeping this setter outside ``generate`` makes it
    /// impossible to change a sampler while a response transaction is open.
    public func setGeneration(
        maxNewTokens: Int,
        sampling: MiniCPMSamplingConfig
    ) throws {
        guard prepared else { throw MiniCPMNativeChatEngineError.notPrepared }
        guard !pending else { throw MiniCPMNativeChatEngineError.pendingResponse }
        generationConfig = MiniCPMChatGenerationConfig(
            mode: mode,
            maxNewTokens: maxNewTokens,
            sampling: sampling)
    }

    /// Feed one complete role-bearing turn or one streaming segment.  Media
    /// is resolved by slot key only after the template has rendered, so the
    /// source order of text/audio/image/video is retained exactly.
    public func prefill(_ input: MiniCPMNativeChatInput) throws -> MiniCPMNativeChatPrefillResult {
        guard prepared else { throw MiniCPMNativeChatEngineError.notPrepared }
        guard !pending else { throw MiniCPMNativeChatEngineError.pendingResponse }
        guard !input.messages.isEmpty else {
            throw MiniCPMNativeChatEngineError.invalidInput("messages must not be empty")
        }

        let before = session.snapshot()
        let beforeAudio = audioSession.snapshot()
        let beforeTTS = ttsSession?.snapshot()
        let beforeTTSRecords = ttsRecords
        if input.resetSession {
            session.reset()
            audioSession.reset()
            if ttsEnabled {
                try resetTTSState(keepPrompt: true)
            }
        }
        interrupted = false

        do {
            let plan: MiniCPMChatInputPlan
            if let context = input.streamingContext {
                guard input.messages.count == 1 else {
                    throw MiniCPMNativeChatEngineError.unsupported(
                        "streaming prefill accepts exactly one role-bearing message")
                }
                plan = try session.templateBuilder.buildStreaming(
                    message: input.messages[0],
                    context: context,
                    separator: input.templateOptions.contentSeparator,
                    enableThinking: input.templateOptions.enableThinking,
                    useTTSTemplate: input.templateOptions.useTTSTemplate)
            } else {
                plan = try session.templateBuilder.build(
                    messages: input.messages,
                    options: input.templateOptions)
            }

            // Resolve every slot before touching the language KV cache.  A
            // short streaming audio chunk therefore returns ``accepted=false``
            // without leaving a half-rendered prompt in the session.
            var embeddings: [String: MLXArray] = [:]
            embeddings.reserveCapacity(plan.slots.count)
            for slot in plan.slots {
                guard let media = input.media[slot.media.key] else {
                    throw MiniCPMNativeChatEngineError.mediaMissing(slot.media.key)
                }
                if embeddings[slot.media.key] == nil {
                    guard let value = try encode(media, streaming: input.streamingAudio) else {
                        // The streaming mel frontend has not reached one
                        // complete pooled chunk yet. Keep its bounded tail and
                        // let the caller append more audio.
                        return MiniCPMNativeChatPrefillResult(
                            accepted: false,
                            needsMoreAudio: true,
                            cacheLength: session.cacheLength)
                    }
                    embeddings[slot.media.key] = value
                }
            }

            _ = try session.prefill(plan: plan) { slot in
                guard let embedding = embeddings[slot.media.key] else {
                    throw MiniCPMNativeChatEngineError.mediaMissing(slot.media.key)
                }
                return embedding
            }
            pending = true
            prefillSnapshot = before
            audioSnapshot = beforeAudio
            prefillTTSSnapshot = beforeTTS
            prefillTTSRecords = beforeTTSRecords
            return MiniCPMNativeChatPrefillResult(
                accepted: true,
                cacheLength: session.cacheLength)
        } catch {
            session.restore(before)
            audioSession.restore(beforeAudio)
            if ttsEnabled {
                try? restoreTTSState(snapshot: beforeTTS, records: beforeTTSRecords)
            }
            throw Self.wrap(error)
        }
    }

    public func generate() throws -> MiniCPMNativeChatOutput {
        guard prepared else { throw MiniCPMNativeChatEngineError.notPrepared }
        guard pending else { throw MiniCPMNativeChatEngineError.noPrefill }
        if interrupted {
            stop()
            return MiniCPMNativeChatOutput(
                text: "",
                audio: nil,
                isListen: true,
                endOfTurn: false,
                reason: "interrupt",
                needsFinalize: false)
        }
        do {
            let result = try session.generate(config: generationConfig)
            let isListen: Bool
            switch result.finishReason {
            case .listen, .chunkEOS, .chunkTTSEOS:
                isListen = mode == .halfDuplex
            default:
                isListen = false
            }
            let endOfTurn = result.finishReason == .turnEOS
                || result.finishReason == .endOfText
            let waveform = try generateTTS(
                result: result,
                endOfTurn: endOfTurn)
            return MiniCPMNativeChatOutput(
                text: result.text,
                audio: waveform,
                isListen: isListen,
                endOfTurn: endOfTurn,
                reason: result.finishReason.rawValue,
                needsFinalize: true)
        } catch {
            throw Self.wrap(error)
        }
    }

    /// Commit the current generated response.  ChatSession has already
    /// advanced its KV/RNG state; clearing the rollback snapshot makes the
    /// transaction durable for the next turn.
    public func finalize() throws {
        pending = false
        prefillSnapshot = nil
        audioSnapshot = nil
        prefillTTSSnapshot = nil
        prefillTTSRecords.removeAll(keepingCapacity: true)
        if pendingTTSFinal {
            // Semantic and Token2Wav final calls have already flushed all
            // waveform chunks. Re-open the immutable voice prompt for the
            // next half-duplex segment while dropping old turn history.
            ttsRecords.removeAll(keepingCapacity: true)
            if ttsEnabled {
                try resetTTSState(keepPrompt: true)
            }
        }
        pendingTTSFinal = false
    }

    public func interrupt() {
        interrupted = true
    }

    public func clearInterrupt() {
        interrupted = false
    }

    /// Restore the exact state captured before the latest prefill.  This is
    /// used when the transport cannot write the generated response.
    public func stop() {
        guard let snapshot = prefillSnapshot else {
            pending = false
            audioSnapshot = nil
            prefillTTSSnapshot = nil
            prefillTTSRecords.removeAll(keepingCapacity: true)
            return
        }
        session.restore(snapshot)
        if let audioSnapshot { audioSession.restore(audioSnapshot) }
        if ttsEnabled {
            try? restoreTTSState(
                snapshot: prefillTTSSnapshot,
                records: prefillTTSRecords)
        }
        pending = false
        prefillSnapshot = nil
        audioSnapshot = nil
        prefillTTSSnapshot = nil
        prefillTTSRecords.removeAll(keepingCapacity: true)
        pendingTTSFinal = false
    }

    public func reset() {
        session.reset()
        audioSession.reset()
        ttsSession?.reset()
        ttsRecords.removeAll(keepingCapacity: true)
        prefillTTSRecords.removeAll(keepingCapacity: true)
        prefillTTSSnapshot = nil
        pendingTTSFinal = false
        if ttsEnabled {
            try? resetTTSState(keepPrompt: true)
        }
        pending = false
        prefillSnapshot = nil
        audioSnapshot = nil
        interrupted = false
    }

    public func close() {
        reset()
        token2wav?.releaseContext()
        token2wav = nil
        ttsSession = nil
        prepared = false
    }

    // MARK: Chat semantic TTS

    private func prepareTTS() throws {
        guard let semantic = models.ttsSemantic else {
            throw MiniCPMNativeChatEngineError.audioUnavailable(
                "semantic TTS weights are not loaded")
        }
        guard let pipeline = try models.makeToken2WavPipeline() else {
            throw MiniCPMNativeChatEngineError.audioUnavailable(
                "Token2Wav weights are not loaded")
        }
        var preparedPipeline = pipeline
        if !preparedPipeline.isPrepared {
            guard let voice = ttsVoicePCM, !voice.isEmpty else {
                throw MiniCPMNativeChatEngineError.audioUnavailable(
                    "a voice prompt is required for Chat TTS")
            }
            do {
                let profile = try models.prepareVoicePrompt(
                    pcm: voice,
                    sampleRate: ttsVoiceSampleRate,
                    pipeline: preparedPipeline)
                ttsPrompt = profile.prompt
            } catch {
                throw MiniCPMNativeChatEngineError.audioUnavailable(
                    "voice prompt preparation failed: \(error.localizedDescription)")
            }
        }
        token2wav = preparedPipeline
        ttsSession = semantic.makeSession()
        ttsRecords.removeAll(keepingCapacity: true)
        pendingTTSFinal = false
    }

    /// Rebuild Token2Wav from the immutable voice prompt and committed append
    /// log. The pipeline intentionally exposes no mutable-cache snapshot; a
    /// short replay keeps transport rollback exact without sharing state
    /// between WebSocket sessions or replaying stale final-turn tokens.
    private func rebuildToken2Wav(
        records: [MiniCPMChatTTSAppendRecord]
    ) throws {
        guard ttsEnabled else { return }
        guard let fresh = try models.makeToken2WavPipeline() else {
            throw MiniCPMNativeChatEngineError.audioUnavailable(
                "Token2Wav weights are not loaded")
        }
        let pipeline = fresh
        if !pipeline.isPrepared {
            guard let prompt = ttsPrompt else {
                throw MiniCPMNativeChatEngineError.audioUnavailable(
                    "voice prompt is not prepared")
            }
            try pipeline.prepare(prompt: prompt)
        }
        for record in records {
            _ = try pipeline.append(
                record.tokens,
                forceFlush: record.forceFlush,
                isFinal: record.isFinal)
        }
        token2wav = pipeline
    }

    private func resetTTSState(keepPrompt: Bool) throws {
        ttsSession?.reset()
        ttsRecords.removeAll(keepingCapacity: true)
        pendingTTSFinal = false
        if !ttsEnabled { return }
        if !keepPrompt { ttsPrompt = nil }
        try rebuildToken2Wav(records: [])
    }

    private func restoreTTSState(
        snapshot: MiniCPMTTSSessionSnapshot?,
        records: [MiniCPMChatTTSAppendRecord]
    ) throws {
        guard ttsEnabled else { return }
        if let snapshot, let ttsSession {
            try ttsSession.rollback(to: snapshot)
        } else {
            ttsSession?.reset()
        }
        ttsRecords = records
        pendingTTSFinal = false
        try rebuildToken2Wav(records: records)
    }

    private func generateTTS(
        result: MiniCPMChatGenerationResult,
        endOfTurn: Bool
    ) throws -> [Float]? {
        guard ttsEnabled else { return nil }
        guard let semantic = models.ttsSemantic,
              let ttsSession,
              let token2wav else {
            throw MiniCPMNativeChatEngineError.audioUnavailable(
                "Chat TTS is enabled but its session is not prepared")
        }

        // The first generated language token is the protocol probe (usually
        // <|speak|>), not visible semantic text. Drop it explicitly at this
        // caller boundary; MiniCPMChatSession still exposes the full aligned
        // sequence for auditing and non-TTS consumers.
        let visibleSpans = result.alignedSpans.dropFirst()
        guard !visibleSpans.isEmpty else {
            if endOfTurn {
                _ = try token2wav.append([], forceFlush: true, isFinal: true)
                ttsSession.finishTurn()
                pendingTTSFinal = true
            }
            return nil
        }
        let hidden = concatenated(visibleSpans.map(\.hidden), axis: 1)
        let tokenIDs = MLXArray(visibleSpans.map { Int32($0.token) })
        let condition = try semantic.buildCondition(
            hidden: hidden,
            tokenIDs: tokenIDs,
            addAudioBOS: true,
            // Chat turn/half-final conditions include text EOS. Duplex units
            // deliberately use false and are implemented in the other engine.
            addTextEOS: mode == .turnBased || endOfTurn)

        let chunks: [MiniCPMToken2WavChunk]
        let appendTokens: [Int32]
        let forceFlush: Bool
        let final: Bool
        if mode == .halfDuplex {
            let semanticChunk = try semantic.generateStreamingChunk(
                inputsEmbeds: condition,
                session: ttsSession,
                endOfTurn: endOfTurn,
                temperature: ttsTemperature,
                repetitionPenalty: ttsRepetitionPenalty,
                eosToken: semantic.configuration.numAudioTokens - 1,
                forceNoStop: false,
                maxNewToken: 26,
                resetAfterEnd: false)
            appendTokens = semanticChunk.committedTokens.map(Int32.init)
            forceFlush = semanticChunk.isFirstChunk || semanticChunk.isFinalChunk
            final = semanticChunk.isFinalChunk
        } else {
            let full = try semantic.generate(
                inputsEmbeds: condition,
                session: ttsSession,
                temperature: ttsTemperature,
                repetitionPenalty: ttsRepetitionPenalty,
                eosToken: semantic.configuration.numAudioTokens - 1,
                forceNoStop: false,
                maxNewToken: 26,
                resetAfterGeneration: false)
            appendTokens = full.tokens.map(Int32.init)
            forceFlush = true
            final = true
        }

        chunks = try token2wav.append(
            appendTokens,
            forceFlush: forceFlush,
            isFinal: final)
        if !appendTokens.isEmpty, !final {
            ttsRecords.append(MiniCPMChatTTSAppendRecord(
                tokens: appendTokens,
                forceFlush: forceFlush,
                isFinal: false))
        }
        if final {
            // Token2Wav has consumed/flushed the committed codes before the
            // semantic KV is released. Keep both pending until finalize so a
            // failed wire write can restore the prefill snapshot.
            ttsSession.finishTurn()
            pendingTTSFinal = true
        }
        return chunks.flatMap { chunk in
            eval(chunk.waveform)
            return chunk.waveform.asType(.float32).asArray(Float.self)
        }
    }

    private func encode(
        _ media: MiniCPMNativeChatMedia,
        streaming: Bool
    ) throws -> MLXArray? {
        switch media {
        case .audio(let samples, let sampleRate):
            let normalized = try Self.normalizedAudio(samples, sampleRate: sampleRate)
            if streaming {
                guard let output = try audioSession.acceptAudio(normalized) else {
                    return nil
                }
                return output.output.embeddings
            }
            return try audioSession.encodeOfflineAudio(normalized).output.embeddings

        case .image(let data):
            return try encodeImage(data)

        case .video(let frames, let audio, let sampleRate):
            guard !frames.isEmpty else {
                throw MiniCPMNativeChatEngineError.invalidInput("video has no decodable frames")
            }
            var visual: [MLXArray] = []
            visual.reserveCapacity(frames.count)
            for frame in frames {
                visual.append(try encodeImage(frame))
            }
            // Keep video and optional audio in one batch-major slot. Vision
            // groups are [T,H], while the audio encoder returns [1,T,H].
            // Normalizing both to [1,T,H] avoids an accidental rank mismatch
            // when an MP4 carries an audio track as well as frames.
            let visualBatch = visual.map { value in
                value.ndim == 3 ? value : value.expandedDimensions(axis: 0)
            }
            var value = visualBatch.count == 1
                ? visualBatch[0]
                : concatenated(visualBatch, axis: 1)
            if let audio, !audio.isEmpty {
                let normalized = try Self.normalizedAudio(
                    audio,
                    sampleRate: sampleRate ?? 16_000)
                let encodedAudio = try audioSession
                    .encodeOfflineAudio(normalized)
                    .output.embeddings
                let audioEmbedding = encodedAudio.ndim == 3
                    ? encodedAudio
                    : encodedAudio.expandedDimensions(axis: 0)
                // A video remains one slot in the plan.  Keeping its visual
                // and audio embeddings together avoids reordering either
                // channel relative to neighboring content parts.
                value = concatenated([value, audioEmbedding], axis: 1)
            }
            return value
        }
    }

    private func encodeImage(_ data: Data) throws -> MLXArray {
        guard let vision = models.vision else {
            throw MiniCPMNativeChatEngineError.visionUnavailable
        }
        let image: CGImage
        do {
            image = try imageProcessor.image(from: data)
        } catch {
            throw MiniCPMNativeChatEngineError.invalidInput(
                "image decoding failed: \(error.localizedDescription)")
        }
        do {
            // ChatTemplateBuilder owns only the outer image markers.  The
            // default chat path uses one overview (max_slice_nums=1), which
            // is the canonical MiniCPM image embedding shape.  HD slices are
            // rejected by the JSON adapter rather than silently dropped.
            let batch = try imageProcessor.process([image], maxSliceNums: 1)
            let groups = try vision.encode(batch)
            guard let overview = groups.first?.overview else {
                throw MiniCPMNativeChatEngineError.invalidInput(
                    "vision encoder returned no image embedding")
            }
            return overview
        } catch let error as MiniCPMNativeChatEngineError {
            throw error
        } catch {
            throw MiniCPMNativeChatEngineError.invalidInput(
                "vision encoding failed: \(error.localizedDescription)")
        }
    }

    private static func normalizedAudio(
        _ samples: [Float],
        sampleRate: Int
    ) throws -> [Float] {
        guard !samples.isEmpty else {
            throw MiniCPMNativeChatEngineError.invalidInput("audio content is empty")
        }
        guard sampleRate > 0 else {
            throw MiniCPMNativeChatEngineError.invalidInput(
                "audio sample rate must be positive")
        }
        guard samples.allSatisfy(\.isFinite) else {
            throw MiniCPMNativeChatEngineError.invalidInput(
                "audio contains NaN or infinity")
        }
        let target = 16_000
        guard sampleRate != target else { return samples }
        guard samples.count > 1 else { return [samples[0]] }
        let outputCount = max(1, Int((Double(samples.count) * Double(target) / Double(sampleRate)).rounded()))
        return (0..<outputCount).map { index in
            let source = Double(index) * Double(sampleRate) / Double(target)
            let lower = min(samples.count - 1, Int(source.rounded(.down)))
            let upper = min(samples.count - 1, lower + 1)
            let fraction = Float(source - Double(lower))
            return samples[lower] + (samples[upper] - samples[lower]) * fraction
        }
    }

    private static func wrap(_ error: Error) -> MiniCPMNativeChatEngineError {
        if let typed = error as? MiniCPMNativeChatEngineError { return typed }
        if let typed = error as? MiniCPMChatTemplateError {
            return .invalidInput(typed.localizedDescription)
        }
        if let typed = error as? MiniCPMChatSessionError {
            return .invalidInput(typed.localizedDescription)
        }
        return .invalidInput(error.localizedDescription)
    }
}
