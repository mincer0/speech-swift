import Foundation
import MiniCPMDuplexRuntime

/// Bridges the framework-neutral ``MiniCPMDuplexRuntime`` actor to the
/// backend WebSocket protocol.  A host supplies a runtime whose
/// ``engineFactory`` wraps the actual MLX decoder/audio/TTS components; this
/// adapter owns only JSON-to-typed-input conversion and wire audio encoding.
public final class MiniCPMDuplexRuntimeEngineAdapter: MiniCPMDemoBackendEngine, @unchecked Sendable {
    private let runtime: MiniCPMDuplexRuntime
    /// Turn/half-duplex must use a chat-session implementation.  Keeping the
    /// dependency optional makes the MLX server boot for full audio/omni
    /// sessions while failing closed (rather than flattening chat messages)
    /// until the MLX Chat builder is available.
    private let chatEngine: (any MiniCPMDemoChatEngine)?
    private let sessionID: String
    private let defaultVoicePCM: [Float]?
    private let defaultVoiceSampleRate: Int?
    private var sessionVoicePCM: [Float]?
    private var sessionVoiceSampleRate: Int?
    private var didInjectDefaultVoice = false
    private var mode: MiniCPMDuplexMode = .fullDuplex
    private var config = MiniCPMDuplexConfig()

    public init(
        runtime: MiniCPMDuplexRuntime,
        sessionID: String = "backend",
        defaultVoicePCM: [Float]? = nil,
        defaultVoiceSampleRate: Int? = nil,
        chatEngine: (any MiniCPMDemoChatEngine)? = nil
    ) {
        self.runtime = runtime
        self.chatEngine = chatEngine
        self.sessionID = sessionID
        self.defaultVoicePCM = defaultVoicePCM
        self.defaultVoiceSampleRate = defaultVoiceSampleRate
        self.sessionVoicePCM = defaultVoicePCM
        self.sessionVoiceSampleRate = defaultVoiceSampleRate
    }

    public func initialize(mode: MiniCPMBackendMode, parameters: [String: MiniCPMJSONValue]) async throws {
        self.mode = Self.runtimeMode(mode)
        if mode == .turnBased || mode == .halfDuplex {
            guard let chatEngine else {
                throw MiniCPMDemoBackendError.engine(
                    "\(mode.rawValue) requires an injected MiniCPMChatEngine; "
                    + "the duplex runtime is not a chat/half-duplex fallback")
            }
            try await chatEngine.initialize(mode: mode, parameters: parameters)
            return
        }
        config = Self.configuration(parameters)
        let voice = Self.mergedVoice(parameters)
        let refAudio = Self.samples(
            voice["ref_audio"]
                ?? voice["ref_audio_base64"]
                ?? voice["audio"]
                ?? parameters["ref_audio"]
                ?? parameters["ref_audio_base64"])
        let ttsAudio = Self.samples(
            voice["tts_ref_audio"]
                ?? voice["tts_ref_audio_base64"]
                ?? parameters["tts_ref_audio"]
                ?? parameters["tts_ref_audio_base64"])
        let refRate = Int(
            voice["ref_sample_rate"]?.numberValue
                ?? voice["sample_rate"]?.numberValue
                ?? 16_000)
        let ttsRate = Int(
            voice["tts_ref_sample_rate"]?.numberValue
                ?? voice["tts_sample_rate"]?.numberValue
                ?? Double(refRate))
        // Production's configured voice is the same default used by
        // DuplexView.prepare: unless an explicit LLM reference is supplied,
        // fan it out to both the language system prompt and Token2Wav. A
        // dedicated TTS reference below may still override only that channel.
        let llmReference = refAudio ?? defaultVoicePCM
        let llmReferenceRate = refAudio == nil
            ? (defaultVoiceSampleRate ?? refRate)
            : refRate
        config.llmReferenceAudio = llmReference
        config.llmReferenceAudioSampleRate = llmReferenceRate
        // Token2Wav consumes only the TTS channel. `ref_audio` remains an LLM
        // conditioning channel whenever a dedicated TTS reference or server
        // default is available; reusing it in those cases could overwrite a
        // fixed Token2Wav prompt. A default voice supplied by the production
        // loader is nevertheless a valid TTS reference and must be prepared
        // before runtime.prepare: waiting until the first prefill is too late
        // because NativeEngine.prepare validates that Token2Wav already has a
        // prompt.
        // Resolve the Token2Wav speaker prompt independently from the LLM
        // reference.  An explicit TTS channel wins; otherwise the server's
        // configured voice is the stable production default.  The legacy
        // `ref_audio` compatibility fallback is used only when neither of
        // those TTS-specific sources is available.
        let ttsReference: [Float]?
        let ttsReferenceRate: Int
        if let ttsAudio {
            ttsReference = ttsAudio
            ttsReferenceRate = ttsRate
        } else if let defaultVoicePCM {
            ttsReference = defaultVoicePCM
            ttsReferenceRate = defaultVoiceSampleRate ?? 16_000
        } else {
            ttsReference = refAudio
            ttsReferenceRate = refAudio == nil ? ttsRate : refRate
        }
        config.ttsReferenceAudio = ttsReference
        config.ttsReferenceAudioSampleRate = ttsReferenceRate
        sessionVoicePCM = ttsReference
        sessionVoiceSampleRate = ttsReference == nil ? nil : ttsReferenceRate
        // The reference is now part of prepare(); injecting the same samples
        // on the first live input would call prepareVoicePrompt twice and can
        // overwrite/fail a fixed Token2Wav prompt.
        didInjectDefaultVoice = ttsReference != nil
        _ = try await runtime.prepare(sessionID: sessionID, mode: self.mode, config: config)
    }

    public func prefill(mode: MiniCPMBackendMode, input: [String: MiniCPMJSONValue]) async throws {
        if mode == .turnBased || mode == .halfDuplex {
            guard let chatEngine else {
                throw MiniCPMDemoBackendError.engine(
                    "\(mode.rawValue) requires an injected MiniCPMChatEngine")
            }
            try await chatEngine.prefill(mode: mode, input: input)
            return
        }
        _ = try await prefillResult(mode: mode, input: input)
    }

    public func prefillResult(mode: MiniCPMBackendMode, input: [String: MiniCPMJSONValue]) async throws -> MiniCPMDemoPrefillResult {
        if mode == .turnBased || mode == .halfDuplex {
            guard let chatEngine else {
                throw MiniCPMDemoBackendError.engine(
                    "\(mode.rawValue) requires an injected MiniCPMChatEngine")
            }
            return try await chatEngine.prefillResult(mode: mode, input: input)
        }
        let configuredVoice = sessionVoicePCM
        let injectVoice = !didInjectDefaultVoice && configuredVoice != nil
            && Self.samples(input["voice"] ?? input["voice_reference"] ?? input["tts_ref_audio"]) == nil
        let typed = try Self.input(
            input,
            defaultVoicePCM: injectVoice ? configuredVoice : nil,
            defaultVoiceSampleRate: injectVoice ? sessionVoiceSampleRate : nil)
        if injectVoice { didInjectDefaultVoice = true }
        let result = try await runtime.prefill(typed, sessionID: sessionID)
        return MiniCPMDemoPrefillResult(
            accepted: result.accepted,
            needsMoreAudio: !result.accepted && (typed.audio?.isEmpty == false))
    }

    public func generate(mode: MiniCPMBackendMode, input: [String: MiniCPMJSONValue]) async throws -> [MiniCPMDemoOutput] {
        if mode == .turnBased || mode == .halfDuplex {
            guard let chatEngine else {
                throw MiniCPMDemoBackendError.engine(
                    "\(mode.rawValue) requires an injected MiniCPMChatEngine")
            }
            return try await chatEngine.generate(mode: mode, input: input)
        }
        let output = try await runtime.generate(sessionID: sessionID)
        let audio = output.audio.map(Self.encodeFloat32)
        var metrics = await diagnosticSnapshot()
        if let padding = output.audioPaddingSamples {
            metrics["audio_padding_samples"] = .number(Double(padding))
        }
        if output.isListen {
            return [MiniCPMDemoOutput(
                kind: "listen",
                audio: audio,
                done: output.endOfTurn,
                needsFinalize: output.needsFinalize,
                reason: output.finishReason,
                metrics: metrics)]
        }
        return [MiniCPMDemoOutput(
            kind: "text",
            text: output.text,
            audio: audio,
            done: output.endOfTurn || !output.needsFinalize,
            needsFinalize: output.needsFinalize,
            reason: output.finishReason,
            metrics: metrics)]
    }

    public func finalize() async throws {
        if mode == .turnBased || mode == .halfDuplex {
            if let chatEngine { try await chatEngine.finalize() }
            return
        }
        try await runtime.finalize(sessionID: sessionID)
    }

    public func interrupt() async {
        if mode == .turnBased || mode == .halfDuplex {
            if let chatEngine { await chatEngine.interrupt() }
            return
        }
        // This synchronous signal is intentionally sent before awaiting the
        // actor-isolated repair path.  The runtime may currently be inside a
        // blocking MLX generate call, so an actor-only interrupt would not be
        // scheduled until the stale audio/text had already been produced.
        runtime.requestInterrupt(sessionID: sessionID)
        try? await runtime.interrupt(sessionID: sessionID)
    }

    public func clearInterrupt() async {
        if mode == .turnBased || mode == .halfDuplex {
            if let chatEngine { await chatEngine.clearInterrupt() }
        }
    }

    public func reset() async {
        if mode == .turnBased || mode == .halfDuplex {
            if let chatEngine { await chatEngine.reset() }
            return
        }
        runtime.requestInterrupt(sessionID: sessionID)
        try? await runtime.reset(sessionID: sessionID)
    }

    public func stop() async {
        if mode == .turnBased || mode == .halfDuplex {
            if let chatEngine { await chatEngine.stop() }
            return
        }
        // ``stop`` is used by backend error/close paths before ``close``.
        // Signal the nonisolated epoch first; otherwise a runtime actor that
        // is currently blocked in synchronous MLX generation cannot enter the
        // rollback call below until the stale turn has completed.
        runtime.requestInterrupt(sessionID: sessionID)
        try? await runtime.rollback(sessionID: sessionID)
    }

    public func close() async {
        if mode == .turnBased || mode == .halfDuplex {
            if let chatEngine { await chatEngine.close() }
            return
        }
        runtime.requestInterrupt(sessionID: sessionID)
        try? await runtime.close(sessionID: sessionID)
    }

    public func diagnosticSnapshot() async -> [String: MiniCPMJSONValue] {
        guard mode == .fullDuplex || mode == .omni else { return [:] }
        guard let snapshot = try? await runtime.diagnostics(sessionID: sessionID) else {
            return [:]
        }
        return Self.diagnosticValues(snapshot)
    }

    private static func diagnosticValues(
        _ snapshot: MiniCPMDuplexRuntimeDiagnosticSnapshot
    ) -> [String: MiniCPMJSONValue] {
        var values: [String: MiniCPMJSONValue] = [
            "session_id": .string(snapshot.sessionID),
            "epoch": .number(Double(snapshot.generationEpoch)),
            "stale_output_discard_count": .number(
                Double(snapshot.staleOutputDiscardCount)),
        ]
        if let requested = snapshot.cancelRequestedAtNS {
            values["cancel_requested_at_ns"] = .number(Double(requested))
        }
        if let returned = snapshot.cancelReturnedAtNS {
            values["cancel_returned_at_ns"] = .number(Double(returned))
        }
        guard let engine = snapshot.engine else { return values }
        values["engine_epoch"] = engine.generationEpoch.map {
            .number(Double($0))
        } ?? .null
        values["unit_index"] = .number(Double(engine.unitIndex))
        values["current_turn_ended"] = .bool(engine.currentTurnEnded)
        values["committed_unit_count"] = .number(Double(engine.committedUnitCount))
        values["kv_cache_length"] = engine.kvCacheLength.map {
            .number(Double($0))
        } ?? .null
        values["sliding_window_mode"] = engine.slidingWindowMode.map {
            .string($0)
        } ?? .null
        values["sliding_window_high_watermark"] = engine.slidingWindowHighWatermark.map {
            .number(Double($0))
        } ?? .null
        values["sliding_window_low_watermark"] = engine.slidingWindowLowWatermark.map {
            .number(Double($0))
        } ?? .null
        values["pending_token_count"] = .number(Double(engine.pendingTokenCount))
        values["pending_terminator"] = engine.pendingTerminator.map {
            .number(Double($0))
        } ?? .null
        values["prefetched"] = .bool(engine.prefetched)
        values["needs_finalize"] = .bool(engine.needsFinalize)
        values["tts_context_present"] = .bool(engine.ttsContextPresent)
        values["tts_pending_token"] = engine.ttsPendingToken.map {
            .number(Double($0))
        } ?? .null
        values["tts_committed_token_count"] = .number(
            Double(engine.ttsCommittedTokenCount))
        values["tts_cache_length"] = .number(Double(engine.ttsCacheLength))
        values["token2wav_prepared"] = .bool(engine.token2WavPrepared)
        values["token2wav_pending_token_count"] = engine.token2WavPendingTokenCount.map {
            .number(Double($0))
        } ?? .null
        values["token2wav_decoder_positions"] = engine.token2WavDecoderPositions.map {
            .number(Double($0))
        } ?? .null
        values["continuation_only"] = .bool(engine.continuationOnly)
        if let value = engine.audioEncoderMS {
            values["audio_encoder_ms"] = .number(value)
        }
        if let value = engine.audioBridgeMS {
            values["audio_bridge_ms"] = .number(value)
        }
        if let value = engine.llmPrefillMS {
            values["llm_prefill_ms"] = .number(value)
        }
        if let value = engine.llmDecodeMS {
            values["llm_decode_ms"] = .number(value)
        }
        if let value = engine.semanticTTSMS {
            values["semantic_tts_ms"] = .number(value)
        }
        if let value = engine.token2WavMS {
            values["token2wav_ms"] = .number(value)
        }
        if let value = engine.flowMS {
            values["flow_ms"] = .number(value)
        }
        if let value = engine.hiftMS {
            values["hift_ms"] = .number(value)
        }
        if let started = engine.generationStartedAtNS {
            values["generation_started_at_ns"] = .number(Double(started))
        }
        if let exited = engine.generationExitedAtNS {
            values["generation_exited_at_ns"] = .number(Double(exited))
        }
        if let returned = engine.cancelReturnedAtNS {
            values["engine_cancel_returned_at_ns"] = .number(Double(returned))
        }
        return values
    }

    private static func runtimeMode(_ mode: MiniCPMBackendMode) -> MiniCPMDuplexMode {
        switch mode {
        case .turnBased: return .turnBased
        case .halfDuplex: return .halfDuplex
        case .fullDuplex: return .fullDuplex
        case .omni: return .omni
        }
    }

    private static func mergedVoice(_ parameters: [String: MiniCPMJSONValue]) -> [String: MiniCPMJSONValue] {
        var merged = parameters["defaults"]?.objectValue ?? [:]
        if let voice = parameters["voice"]?.objectValue {
            merged.merge(voice) { _, newer in newer }
        }
        return merged
    }

    private static func configuration(_ parameters: [String: MiniCPMJSONValue]) -> MiniCPMDuplexConfig {
        let config = parameters["config"]?.objectValue
            ?? parameters["sampling"]?.objectValue
            ?? parameters["duplex"]?.objectValue
            ?? parameters
        let maxTokens = Int(
            config["max_new_speak_tokens_per_chunk"]?.numberValue
                ?? config["max_new_tokens"]?.numberValue
                ?? config["maxNewTokens"]?.numberValue
                ?? 20)
        let temperature = Float(config["temperature"]?.numberValue ?? 0.7)
        let topK = Int(config["top_k"]?.numberValue ?? config["topK"]?.numberValue ?? 20)
        let topP = Float(config["top_p"]?.numberValue ?? config["topP"]?.numberValue ?? 0.8)
        let window = config["sliding_window_tokens"]?.numberValue.map(Int.init)
        let windowMode = config["sliding_window_mode"]?.stringValue
            ?? config["slidingWindowMode"]?.stringValue
            ?? "off"
        let highWindow = Int(
            config["basic_window_high_tokens"]?.numberValue
                ?? config["basicWindowHighTokens"]?.numberValue
                ?? 8_000)
        let lowWindow = Int(
            config["basic_window_low_tokens"]?.numberValue
                ?? config["basicWindowLowTokens"]?.numberValue
                ?? 6_000)
        let contextPreviousMax = Int(
            config["context_previous_max_tokens"]?.numberValue
                ?? config["contextPreviousMaxTokens"]?.numberValue
                ?? 500)
        let contextMaxUnits = Int(
            config["context_max_units"]?.numberValue
                ?? config["contextMaxUnits"]?.numberValue
                ?? 24)
        let contextMarker = config["context_previous_marker"]?.stringValue
            ?? config["contextPreviousMarker"]?.stringValue
            ?? "\n\nprevious: "
        let lsMode = config["ls_mode"]?.stringValue
            ?? config["lsMode"]?.stringValue
            ?? "explicit"
        let forceListenCount = Int(
            config["force_listen_count"]?.numberValue
                ?? config["forceListenCount"]?.numberValue
                ?? 0)
        let listenScale = Float(
            config["listen_prob_scale"]?.numberValue
                ?? config["listen_probability_scale"]?.numberValue
                ?? config["listenProbabilityScale"]?.numberValue
                ?? 1.0)
        let listenTopK = (config["listen_top_k"] ?? config["listenTopK"])?.numberValue.map(Int.init)
        let greedyListenSpeakDecision = config["greedy_listen_speak_decision"]?.boolValue
            ?? config["greedyListenSpeakDecision"]?.boolValue
            ?? false
        let repetitionPenalty = Float(
            config["repetition_penalty"]?.numberValue
                ?? config["repetitionPenalty"]?.numberValue
                ?? 1.05)
        let repetitionWindow = Int(
            config["text_repetition_window_size"]?.numberValue
                ?? config["repetition_window_size"]?.numberValue
                ?? config["repetitionWindowSize"]?.numberValue
                ?? 512)
        let textPenalty = Float(
            config["text_repetition_penalty"]?.numberValue
                ?? config["repetition_penalty"]?.numberValue
                ?? 1.05)
        // The official duplex default is 1.1. The browser talks over the
        // model constantly, and at 1.1 the quantized model yields mid-sentence
        // every couple of units, which fragments answers, resets the
        // semantic-TTS turn, and leaves dangling fragments that the vocoder
        // then under-speaks (audible holes). Hold the turn open harder.
        let lengthPenalty = Float(config["length_penalty"]?.numberValue ?? 1.2)
        let ttsTemperature = Float(config["tts_temperature"]?.numberValue ?? 0.8)
        let ttsRepetitionPenalty = Float(config["tts_repetition_penalty"]?.numberValue ?? 1.05)
        let generateAudio = config["generate_audio"]?.boolValue
            ?? config["generateAudio"]?.boolValue
            ?? true
        let systemPrompt = config["system_prompt"]?.stringValue
            ?? config["systemPrompt"]?.stringValue
            ?? "Streaming Omni Conversation."
        return MiniCPMDuplexConfig(
            maxNewTokens: maxTokens,
            temperature: temperature,
            topK: topK,
            topP: topP,
            slidingWindowTokens: window,
            lsMode: lsMode,
            forceListenCount: forceListenCount,
            listenProbabilityScale: listenScale,
            listenTopK: listenTopK,
            repetitionPenalty: textPenalty == 1.05 ? repetitionPenalty : textPenalty,
            repetitionWindowSize: repetitionWindow,
            generateAudio: generateAudio,
            systemPrompt: systemPrompt,
            referenceAudio: nil,
            lengthPenalty: lengthPenalty,
            ttsTemperature: ttsTemperature,
            ttsRepetitionPenalty: ttsRepetitionPenalty,
            slidingWindowMode: windowMode,
            basicWindowHighTokens: highWindow,
            basicWindowLowTokens: lowWindow,
            contextPreviousMaxTokens: contextPreviousMax,
            contextMaxUnits: contextMaxUnits,
            contextPreviousMarker: contextMarker,
            greedyListenSpeakDecision: greedyListenSpeakDecision)
    }

    private static func input(
        _ object: [String: MiniCPMJSONValue],
        defaultVoicePCM: [Float]? = nil,
        defaultVoiceSampleRate: Int? = nil
    ) throws -> MiniCPMDuplexInput {
        var audio = samples(object["audio"] ?? object["audio_waveform"])
        var text = object["text"]?.stringValue
            ?? object["content"]?.stringValue
        var frameData = try frameBytes(
            object["video_frames"]
                ?? object["frame_base64_list"]
                ?? object["frame_list"]
                ?? object["frame_data"]
                ?? object["frames"])
        let stackFrames = object["stack_frames"]?.boolValue
            ?? object["stackFrames"]?.boolValue
            ?? true
        let maxFrames = Int(
            object["max_frames"]?.numberValue
                ?? object["maxFrames"]?.numberValue
                ?? 32)
        let messageParts = try parseMessageContent(
            object["messages"], stackFrames: stackFrames, maxFrames: maxFrames)
        if let messages = messageParts.messages, !messages.isEmpty {
            // Keep client-managed history as a typed, role-bearing sequence.
            // Do not flatten it into independent audio/text/frame fields: the
            // native chat-template path can then preserve content order (or
            // reject explicitly if that path is unavailable).
            audio = nil
            text = nil
            frameData = nil
        } else {
            if audio == nil { audio = messageParts.audio }
            if text == nil { text = messageParts.text }
            if frameData == nil { frameData = messageParts.frames }
        }

        if audio == nil {
            audio = samples(object["audio_base64"] ?? object["audio_data"])
        }

        var voicePCM = samples(
            object["voice"]
                ?? object["voice_reference"]
                ?? object["tts_ref_audio"]) ?? defaultVoicePCM
        let voiceRate = Int(
            object["voice_sample_rate"]?.numberValue
                ?? object["sample_rate"]?.numberValue
                ?? Double(defaultVoiceSampleRate ?? 16_000))
        var metadata: [String: String] = [:]
        var maxSliceNums: [Int]?
        for key in [
            "streaming", "omni_mode", "use_tts_template", "enable_thinking", "generate_audio",
            "speech_active", "allow_sampled_listen_speak_decision",
            "continuation_tick", "continuationOnly",
        ] {
            if let value = object[key]?.boolValue { metadata[key] = value ? "true" : "false" }
        }
        if let value = object["omni"]?.boolValue { metadata["omni_mode"] = value ? "true" : "false" }
        if let hints = object["hints"]?.objectValue {
            if let value = hints["force_listen"]?.boolValue { metadata["force_listen"] = value ? "true" : "false" }
            if let value = hints["max_slice_nums"]?.numberValue { metadata["max_slice_nums"] = String(Int(value)) }
        }
        metadata["stack_frames"] = stackFrames ? "true" : "false"
        metadata["max_frames"] = String(max(1, maxFrames))
        let hintedSlices = object["hints"]?.objectValue?["max_slice_nums"]
            ?? object["hints"]?.objectValue?["maxSliceNums"]
        if let maxSlices = object["max_slice_nums"] ?? object["maxSliceNums"] ?? hintedSlices {
            switch maxSlices {
            case .number(let number):
                guard number.isFinite, number > 0, number.rounded() == number else {
                    throw MiniCPMDemoBackendError.invalidMessage(
                        "max_slice_nums must be a positive integer or an integer array")
                }
                maxSliceNums = [Int(number)]
                metadata["max_slice_nums"] = String(Int(number))
            case .array(let values):
                let parsed = values.map { $0.numberValue }
                guard parsed.allSatisfy({
                    guard let value = $0 else { return false }
                    return value.isFinite && value > 0 && value.rounded() == value
                }) else {
                    throw MiniCPMDemoBackendError.invalidMessage(
                        "max_slice_nums array must contain positive integers")
                }
                if let frameData, !frameData.isEmpty, parsed.count != frameData.count {
                    throw MiniCPMDemoBackendError.invalidMessage(
                        "max_slice_nums array length must equal the number of frames")
                }
                let frameCount = frameData?.count ?? messageParts.frames?.count ?? 0
                if frameCount > 0, parsed.count != frameCount {
                    throw MiniCPMDemoBackendError.invalidMessage(
                        "max_slice_nums array length must equal the number of frames")
                }
                maxSliceNums = parsed.compactMap { $0.map(Int.init) }
                metadata["max_slice_nums_array"] = parsed
                    .compactMap { $0.map { String(Int($0)) } }
                    .joined(separator: ",")
            default:
                throw MiniCPMDemoBackendError.invalidMessage(
                    "max_slice_nums must be a positive integer or an integer array")
            }
        }
        if let audioRate = messageParts.audioSampleRate {
            metadata["audio_sample_rate"] = String(audioRate)
        }
        if let messages = messageParts.messages, !messages.isEmpty {
            metadata["ordered_messages"] = "true"
            metadata["message_count"] = String(messages.count)
        }
        if let generation = object["generation"]?.objectValue {
            for key in ["max_new_tokens", "length_penalty", "temperature", "top_k", "top_p"] {
                if let value = generation[key]?.numberValue { metadata["generation.\(key)"] = String(value) }
            }
        }
        if let tts = object["tts"]?.objectValue {
            if let enabled = tts["enabled"]?.boolValue { metadata["tts.enabled"] = enabled ? "true" : "false" }
            if let enabled = tts["enabled"]?.boolValue, enabled { metadata["generate_audio"] = "true" }
            if let reference = samples(tts["ref_audio_data"] ?? tts["ref_audio"]), voicePCM == nil {
                // Preserve the actual reference for the native Token2Wav
                // adapter; the `voice` field remains the canonical path.
                voicePCM = reference
                metadata["tts.ref_audio_samples"] = String(reference.count)
            }
        }
        return MiniCPMDuplexInput(
            audio: audio,
            frameData: frameData,
            voicePCM: voicePCM,
            voiceSampleRate: voicePCM == nil ? nil : voiceRate,
            text: text,
            metadata: metadata,
            messages: messageParts.messages,
            forceListen: object["force_listen"]?.boolValue
                ?? object["forceListen"]?.boolValue,
            maxSliceNums: maxSliceNums,
            allowSampledListenSpeakDecision:
                object["allow_sampled_listen_speak_decision"]?.boolValue
                    ?? object["allowSampledListenSpeakDecision"]?.boolValue
                    ?? object["speech_active"]?.boolValue,
            continuationOnly: object["continuation_tick"]?.boolValue
                ?? object["continuationOnly"]?.boolValue
                ?? false)
    }

    private static func parseMessageContent(
        _ value: MiniCPMJSONValue?,
        stackFrames: Bool,
        maxFrames: Int
    ) throws -> (
        text: String?,
        audio: [Float]?,
        frames: [Data]?,
        audioSampleRate: Int?,
        messages: [MiniCPMDuplexMessage]?
    ) {
        guard case .array(let messages) = value else {
            return (nil, nil, nil, nil, nil)
        }
        var texts: [String] = []
        var audioParts: [[Float]] = []
        var frames: [Data] = []
        var audioSampleRate: Int?
        var ordered: [MiniCPMDuplexMessage] = []
        for (messageIndex, message) in messages.enumerated() {
            guard case .object(let object) = message else {
                throw MiniCPMDemoBackendError.invalidMessage(
                    "messages[\(messageIndex)] must be an object")
            }
            let role = object["role"]?.stringValue ?? "user"
            var orderedParts: [MiniCPMDuplexContentPart] = []
            let content = object["content"]
            if let string = content?.stringValue, !string.isEmpty {
                texts.append(string)
                orderedParts.append(.text(string))
                ordered.append(MiniCPMDuplexMessage(role: role, parts: orderedParts))
                continue
            }
            guard case .array(let parts) = content else {
                throw MiniCPMDemoBackendError.invalidMessage(
                    "messages[\(messageIndex)].content must be a string or array")
            }
            for (partIndex, part) in parts.enumerated() {
                guard case .object(let payload) = part else {
                    throw MiniCPMDemoBackendError.invalidMessage(
                        "messages[\(messageIndex)].content[\(partIndex)] must be an object")
                }
                switch payload["type"]?.stringValue?.lowercased() {
                case "text":
                    guard let value = payload["text"]?.stringValue, !value.isEmpty else {
                        throw MiniCPMDemoBackendError.invalidMessage(
                            "messages[\(messageIndex)].content[\(partIndex)] text is empty")
                    }
                    texts.append(value)
                    orderedParts.append(.text(value))
                case "audio":
                    guard let value = try samplesStrict(
                        payload["data"] ?? payload["audio"] ?? payload["base64"],
                        path: "messages.audio") else {
                        throw MiniCPMDemoBackendError.invalidMessage(
                            "messages.audio must contain Float32 PCM samples")
                    }
                    audioParts.append(value)
                    orderedParts.append(.audio(
                        samples: value,
                        sampleRate: Int(payload["sample_rate"]?.numberValue
                            ?? payload["sampleRate"]?.numberValue
                            ?? 16_000)))
                    if let rate = payload["sample_rate"]?.numberValue
                        ?? payload["sampleRate"]?.numberValue {
                        audioSampleRate = Int(rate)
                    }
                case "image":
                    let data = try encodedFrame(
                        payload["data"] ?? payload["base64"] ?? payload["image"],
                        path: "messages.image")
                    guard isJPEG(data) else {
                        throw MiniCPMDemoBackendError.invalidMessage(
                            "messages.image must be a single JPEG frame")
                    }
                    frames.append(data)
                    orderedParts.append(.image(data))
                case "video":
                    let data = try encodedFrame(
                        payload["data"] ?? payload["base64"] ?? payload["video"],
                        path: "messages.video")
                    let stackCount = Int(
                        payload["stack_frames"]?.numberValue
                            ?? payload["stackFrames"]?.numberValue
                            ?? 1)
                    guard stackCount > 0 else {
                        throw MiniCPMDemoBackendError.invalidMessage(
                            "messages.video stack_frames must be a positive integer")
                    }
                    let media = try MiniCPMDemoMediaDecoder.decodeVideo(
                        data,
                        stackFrames: stackCount > 1,
                        maxFrames: maxFrames)
                    frames.append(contentsOf: media.frames)
                    // Keep the original video container as one ordered part.
                    // The future chat-template builder owns frame/audio
                    // interleaving; flattening extracted frames here changes
                    // the source order and is explicitly forbidden.
                    orderedParts.append(.video(data))
                    if let mediaAudio = media.audio {
                        audioParts.append(mediaAudio)
                        audioSampleRate = media.audioSampleRate
                    }
                case "frame":
                    let data = try encodedFrame(
                        payload["data"] ?? payload["base64"] ?? payload["frame"],
                        path: "messages.frame")
                    guard isJPEG(data) else {
                        throw MiniCPMDemoBackendError.invalidMessage(
                            "messages.frame must be a single JPEG frame")
                    }
                    frames.append(data)
                    orderedParts.append(.image(data))
                default:
                    // Unknown content types must not be silently flattened;
                    // doing so changes chat-template semantics and hides a
                    // client/server schema mismatch.
                    throw MiniCPMDemoBackendError.invalidMessage(
                        "messages[\(messageIndex)].content[\(partIndex)] has unsupported type")
                }
            }
            if !orderedParts.isEmpty {
                ordered.append(MiniCPMDuplexMessage(role: role, parts: orderedParts))
            }
        }
        let audio = audioParts.flatMap { $0 }
        return (
            texts.isEmpty ? nil : texts.joined(),
            audio.isEmpty ? nil : audio,
            frames.isEmpty ? nil : frames,
            audioSampleRate,
            ordered.isEmpty ? nil : ordered)
    }

    private static func frameBytes(_ value: MiniCPMJSONValue?) throws -> [Data]? {
        guard let value else { return nil }
        switch value {
        case .string(let encoded):
            guard let data = Data(base64Encoded: encoded), !data.isEmpty else {
                throw MiniCPMDemoBackendError.invalidMessage("frame data is not valid base64")
            }
            guard isJPEG(data) else {
                throw MiniCPMDemoBackendError.invalidMessage("frame data must be JPEG; video containers require type=video")
            }
            return [data]
        case .array(let values):
            // A single image payload is represented as a flat byte array;
            // multiple frames are represented as an array of such arrays.
            if let one = byteData(values) {
                guard isJPEG(one) else {
                    throw MiniCPMDemoBackendError.invalidMessage("frame data must be JPEG")
                }
                return [one]
            }
            var many: [Data] = []
            for item in values {
                guard case .array(let bytes) = item else { return nil }
                guard let data = byteData(bytes), isJPEG(data) else {
                    throw MiniCPMDemoBackendError.invalidMessage("frame data must be JPEG")
                }
                many.append(data)
            }
            return many.isEmpty ? nil : many
        case .object(let object):
            return try frameBytes(
                object["data"] ?? object["base64"] ?? object["image"] ?? object["frame"])
        default:
            return nil
        }
    }

    private static func encodedFrame(
        _ value: MiniCPMJSONValue?,
        path: String
    ) throws -> Data {
        guard let value else {
            throw MiniCPMDemoBackendError.invalidMessage("(path) requires data")
        }
        switch value {
        case .string(let encoded):
            guard let data = Data(base64Encoded: encoded), !data.isEmpty else {
                throw MiniCPMDemoBackendError.invalidMessage("(path) is not valid base64")
            }
            return data
        case .array(let values):
            guard let data = byteData(values) else {
                throw MiniCPMDemoBackendError.invalidMessage("(path) must contain encoded bytes")
            }
            return data
        case .object(let object):
            return try encodedFrame(
                object["data"] ?? object["base64"] ?? object["image"] ?? object["video"],
                path: path)
        default:
            throw MiniCPMDemoBackendError.invalidMessage("(path) must be a base64 string")
        }
    }

    private static func isJPEG(_ data: Data) -> Bool {
        data.count >= 4 && data[data.startIndex] == 0xff
            && data[data.startIndex + 1] == 0xd8
            && data[data.index(data.endIndex, offsetBy: -2)] == 0xff
            && data[data.index(data.endIndex, offsetBy: -1)] == 0xd9
    }

    private static func byteData(_ values: [MiniCPMJSONValue]) -> Data? {
        var data = Data(capacity: values.count)
        for value in values {
            guard let number = value.numberValue,
                  number.isFinite,
                  number >= 0,
                  number <= 255,
                  number.rounded() == number else {
                return nil
            }
            data.append(UInt8(number))
        }
        return data.isEmpty ? nil : data
    }

    private static func samples(_ value: MiniCPMJSONValue?) -> [Float]? {
        guard let value else { return nil }
        switch value {
        case .array(let values):
            let numbers = values.compactMap { $0.numberValue.map(Float.init) }
            return numbers.isEmpty ? nil : numbers
        case .object(let object):
            return samples(object["data"] ?? object["pcm"] ?? object["audio"])
        default:
            return nil
        }
    }

    private static func samplesStrict(
        _ value: MiniCPMJSONValue?,
        path: String
    ) throws -> [Float]? {
        guard let value else { return nil }
        switch value {
        case .array(let values):
            let numbers = values.map { $0.numberValue }
            guard numbers.allSatisfy({ $0?.isFinite == true }) else {
                throw MiniCPMDemoBackendError.invalidMessage("(path) contains NaN or infinity")
            }
            let converted = numbers.compactMap { $0.map(Float.init) }
            guard converted.count == numbers.count,
                  converted.allSatisfy(\.isFinite) else {
                throw MiniCPMDemoBackendError.invalidMessage("(path) contains values outside Float32 range")
            }
            return converted
        case .string:
            let normalized = try MiniCPMDemoWireDecoder.normalizeObjectStrict([
                "audio": value
            ])
            return samples(normalized["audio"])
        default:
            return nil
        }
    }

    private static func encodeFloat32(_ samples: [Float]) -> String {
        var data = Data(capacity: samples.count * MemoryLayout<Float>.size)
        for sample in samples {
            var bits = sample.bitPattern.littleEndian
            withUnsafeBytes(of: &bits) { data.append(contentsOf: $0) }
        }
        return data.base64EncodedString()
    }
}
