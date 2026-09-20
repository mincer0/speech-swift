import Foundation
import MiniCPMDuplexRuntime
import MiniCPMLLM

/// Backend/protocol adapter for ``MiniCPMNativeChatEngine``.  Keeping JSON
/// decoding here (rather than in the MLX runtime target) avoids a target cycle
/// while giving production and protocol tests one authoritative ordered-media
/// conversion path.
public final class MiniCPMNativeChatEngineAdapter: MiniCPMDemoChatEngine, @unchecked Sendable {
    public let native: MiniCPMNativeChatEngine

    private var mode: MiniCPMBackendMode = .turnBased
    private var generation = MiniCPMChatGenerationConfig()
    private var templateOptions = MiniCPMChatTemplateOptions()
    private var prepared = false

    // Half-duplex PCM chunks are buffered until the client supplies an
    // explicit utterance_end/is_final marker.  This is a bounded fallback for
    // transports that do not ship a Silero VAD stream; it prevents each 16k
    // packet from being mislabeled as a complete user turn.
    private var halfAudioBuffer: [Float] = []
    private var halfAudioSampleRate = 16_000
    private var halfConversationStarted = false
    private var halfAwaitingNextUser = false
    private var halfSpeechSeen = false
    private var halfSilentChunks = 0
    private var pendingHalfResponse = false
    private var lastOutput: MiniCPMNativeChatOutput?
    private let maxHalfUtteranceSamples = 16_000 * 30
    private let vadFactory: MiniCPMStreamingVADFactory?
    private var halfVAD: (any MiniCPMStreamingVADProvider)?
    private var allowEnergyVADFallback = false
    private let defaultVoicePCM: [Float]?
    private let defaultVoiceSampleRate: Int

    public init(
        models: MiniCPMNativeModels,
        samplerSeed: UInt64? = nil,
        vadFactory: MiniCPMStreamingVADFactory? = nil,
        allowEnergyVADFallback: Bool = false,
        defaultVoicePCM: [Float]? = nil,
        defaultVoiceSampleRate: Int = 16_000
    ) {
        native = MiniCPMNativeChatEngine(models: models, samplerSeed: samplerSeed)
        self.vadFactory = vadFactory
        self.allowEnergyVADFallback = allowEnergyVADFallback
        // Providers are session-scoped. Do not invoke a throwing factory from
        // init (which cannot report its failure); initialize(.halfDuplex)
        // creates the provider and propagates model-load errors to session.init.
        self.halfVAD = nil
        self.defaultVoicePCM = defaultVoicePCM
        self.defaultVoiceSampleRate = defaultVoiceSampleRate
    }

    public func initialize(
        mode: MiniCPMBackendMode,
        parameters: [String: MiniCPMJSONValue]
    ) async throws {
        guard mode == .turnBased || mode == .halfDuplex else {
            throw MiniCPMDemoBackendError.engine(
                "native chat engine cannot serve \(mode.rawValue); full/omni use the duplex runtime")
        }
        let values = Self.mergedParameters(parameters)
        self.mode = mode
        if let requestedFallback = Self.bool(values, [
            "allow_energy_vad_fallback", "allowEnergyVADFallback",
            "vad.allow_energy_fallback", "vad.allowEnergyFallback"
        ]) {
            allowEnergyVADFallback = requestedFallback
        }
        // Recreate rather than merely reset so each initialize/session gets a
        // provider with no leaked model state. Full/turn paths never call it;
        // the optional provider is only consumed by direct half-duplex PCM.
        halfVAD = nil
        if mode == .halfDuplex {
            halfVAD = try vadFactory?()
        }
        halfVAD?.reset()
        let maxTokens = max(0, Self.int(values, [
            "max_new_tokens", "maxNewTokens", "max_new_speak_tokens_per_chunk"
        ]) ?? (mode == .halfDuplex ? 128 : 256))
        let sampling = MiniCPMSamplingConfig(
            mode: Self.bool(values, ["do_sample", "sampling.do_sample"]) == false
                || Self.string(values, ["sampling.mode", "sampling_mode"])?.lowercased() == "greedy"
                ? .greedy : .sampling,
            temperature: Self.float(values, ["temperature", "sampling.temperature"]) ?? 0.7,
            topK: Self.int(values, ["top_k", "topK", "sampling.top_k", "sampling.topK"]) ?? 20,
            topP: Self.float(values, ["top_p", "topP", "sampling.top_p", "sampling.topP"]) ?? 0.8,
            repetitionPenalty: Self.float(values, [
                "repetition_penalty", "repetitionPenalty", "sampling.repetition_penalty"
            ]) ?? 1.05,
            repetitionWindowSize: Self.int(values, [
                "repetition_window_size", "text_repetition_window_size"
            ]) ?? 512,
            lengthPenalty: Self.float(values, ["length_penalty", "sampling.length_penalty"]) ?? 1.1,
            listenTopK: Self.int(values, ["listen_top_k", "listenTopK"]),
            listenProbabilityScale: Self.float(values, [
                "listen_prob_scale", "listen_probability_scale", "listenProbabilityScale"
            ]) ?? 1.0)
        generation = MiniCPMChatGenerationConfig(
            mode: mode == .halfDuplex ? .halfDuplex : .turnBased,
            maxNewTokens: maxTokens,
            sampling: sampling)
        templateOptions = MiniCPMChatTemplateOptions(
            addGenerationPrompt: true,
            useTTSTemplate: Self.bool(values, [
                "use_tts_template", "useTTSTemplate",
                "tts.use_template", "tts.useTTSTemplate"
            ]) ?? Self.bool(values, [
                "tts.enabled", "tts_enabled", "generate_audio", "generateAudio"
            ]) ?? (mode == .halfDuplex),
            enableThinking: Self.bool(values, ["enable_thinking", "enableThinking"]) ?? false,
            contentSeparator: mode == .halfDuplex ? .none : .newline,
            enableTTSTemplateForAudio: false)
        let ttsEnabled = Self.bool(values, [
            "tts.enabled", "tts_enabled", "generate_audio", "generateAudio"
        ]) ?? templateOptions.useTTSTemplate
        let ttsTemperature = Self.float(values, [
            "tts.temperature", "ttsTemperature", "tts_temperature"
        ]) ?? 0.8
        let ttsRepetitionPenalty = Self.float(values, [
            "tts.repetition_penalty", "ttsRepetitionPenalty", "tts_repetition_penalty"
        ]) ?? 1.05
        let initialVoice = Self.samples(values, keys: [
            "tts.ref_audio", "tts.ref_audio_data", "tts_ref_audio",
            "voice", "voice_reference"
        ]) ?? defaultVoicePCM
        let initialVoiceRate = Self.int(values, [
            "tts.ref_sample_rate", "tts_sample_rate", "ttsRefSampleRate",
            "sample_rate", "sampleRate"
        ]) ?? defaultVoiceSampleRate
        try native.prepare(
            mode: generation.mode,
            maxNewTokens: generation.maxNewTokens,
            sampling: generation.sampling,
            ttsEnabled: ttsEnabled,
            ttsTemperature: ttsTemperature,
            ttsRepetitionPenalty: ttsRepetitionPenalty,
            ttsVoicePCM: initialVoice,
            ttsVoiceSampleRate: initialVoiceRate)
        halfAudioBuffer.removeAll(keepingCapacity: true)
        halfAudioSampleRate = 16_000
        halfConversationStarted = false
        halfAwaitingNextUser = false
        halfSpeechSeen = false
        halfSilentChunks = 0
        pendingHalfResponse = false
        lastOutput = nil
        prepared = true
    }

    public func prefill(
        mode: MiniCPMBackendMode,
        input: [String: MiniCPMJSONValue]
    ) async throws {
        _ = try await prefillResult(mode: mode, input: input)
    }

    public func prefillResult(
        mode: MiniCPMBackendMode,
        input: [String: MiniCPMJSONValue]
    ) async throws -> MiniCPMDemoPrefillResult {
        guard prepared, mode == self.mode else {
            throw MiniCPMDemoBackendError.invalidState("native chat engine is not initialized for \(mode.rawValue)")
        }
        let values = Self.mergedParameters(input)
        if generation.mode == .turnBased || generation.mode == .halfDuplex,
           let voice = Self.samples(values, keys: [
               "tts.ref_audio", "tts.ref_audio_data", "tts_ref_audio",
               "voice", "voice_reference"
           ]) {
            let rate = Self.int(values, [
                "tts.ref_sample_rate", "tts_sample_rate", "ttsRefSampleRate",
                "sample_rate", "sampleRate"
            ]) ?? defaultVoiceSampleRate
            try native.setTTSVoicePrompt(pcm: voice, sampleRate: rate)
        }
        if Self.hasGenerationOverride(values) {
            let maxTokens = Self.int(values, [
                "max_new_tokens", "maxNewTokens", "max_new_speak_tokens_per_chunk"
            ]) ?? generation.maxNewTokens
            let updatedSampling = Self.sampling(values, fallback: generation.sampling)
            try native.setGeneration(maxNewTokens: maxTokens, sampling: updatedSampling)
            generation = MiniCPMChatGenerationConfig(
                mode: generation.mode,
                maxNewTokens: maxTokens,
                sampling: updatedSampling)
        }
        if mode == .halfDuplex,
           Self.hasDirectAudio(input),
           !Self.isMessagesArray(input["messages"]) {
            return try prefillHalfAudioChunk(input)
        }

        let parsed: ParsedMessages
        if Self.isMessagesArray(input["messages"]) {
            parsed = try Self.parseMessages(
                input["messages"],
                maxVideoFrames: Self.int(input, ["max_frames", "maxFrames"]) ?? 32)
        } else {
            parsed = try Self.parseDirectInput(
                input,
                maxVideoFrames: Self.int(input, ["max_frames", "maxFrames"]) ?? 32)
        }
        let isStreaming = mode == .halfDuplex
            && (Self.bool(values, ["streaming", "stream_input", "chunk_input"]) ?? false)
        let context: MiniCPMStreamingPromptContext?
        if isStreaming {
            guard parsed.messages.count == 1 else {
                throw MiniCPMDemoBackendError.invalidMessage(
                    "half-duplex streaming messages must contain one role-bearing message")
            }
            context = halfConversationStarted
                ? .nextUser(completedResponse: halfAwaitingNextUser)
                : .firstUser
        } else {
            context = nil
        }
        let options = MiniCPMChatTemplateOptions(
            addGenerationPrompt: true,
            useTTSTemplate: templateOptions.useTTSTemplate,
            enableThinking: Self.bool(values, ["enable_thinking", "enableThinking"]) ?? templateOptions.enableThinking,
            contentSeparator: context == nil ? templateOptions.contentSeparator : .none,
            enableTTSTemplateForAudio: false)
        let result = try native.prefill(MiniCPMNativeChatInput(
            messages: parsed.messages,
            media: parsed.media,
            templateOptions: options,
            streamingContext: context,
            streamingAudio: isStreaming,
            // `messages` is a complete conversation in turn mode. Streaming
            // segments are appended to the existing ChatSession KV state.
            resetSession: context == nil))
        if result.accepted {
            pendingHalfResponse = mode == .halfDuplex
        }
        return MiniCPMDemoPrefillResult(
            accepted: result.accepted,
            needsMoreAudio: result.needsMoreAudio)
    }

    public func generate(
        mode: MiniCPMBackendMode,
        input: [String: MiniCPMJSONValue]
    ) async throws -> [MiniCPMDemoOutput] {
        guard prepared, mode == self.mode else {
            throw MiniCPMDemoBackendError.invalidState("native chat engine is not initialized for \(mode.rawValue)")
        }
        let output = try native.generate()
        lastOutput = output
        let wireReason = output.endOfTurn ? "end_of_turn" : output.reason
        return [MiniCPMDemoOutput(
            kind: output.isListen ? "listen" : "text",
            text: output.isListen ? nil : output.text,
            audio: output.audio.map(Self.encodeFloat32),
            format: "pcm_f32",
            sampleRate: 24_000,
            done: output.endOfTurn,
            needsFinalize: output.needsFinalize,
            reason: wireReason)]
    }

    public func finalize() async throws {
        try native.finalize()
        if pendingHalfResponse {
            halfConversationStarted = true
            if lastOutput?.endOfTurn == true {
                halfAwaitingNextUser = true
            } else if lastOutput?.isListen == true {
                halfAwaitingNextUser = false
            }
        }
        pendingHalfResponse = false
        lastOutput = nil
    }

    public func interrupt() async {
        native.interrupt()
    }

    public func clearInterrupt() async {
        native.clearInterrupt()
    }

    public func stop() async {
        native.stop()
        pendingHalfResponse = false
        lastOutput = nil
    }

    public func reset() async {
        native.reset()
        halfVAD?.reset()
        halfAudioBuffer.removeAll(keepingCapacity: true)
        halfConversationStarted = false
        halfAwaitingNextUser = false
        halfSpeechSeen = false
        halfSilentChunks = 0
        pendingHalfResponse = false
        lastOutput = nil
    }

    public func close() async {
        native.close()
        halfVAD?.reset()
        halfAudioBuffer.removeAll(keepingCapacity: true)
        halfSpeechSeen = false
        halfSilentChunks = 0
        prepared = false
    }

    // MARK: Half-duplex explicit EOU fallback

    private func prefillHalfAudioChunk(
        _ input: [String: MiniCPMJSONValue]
    ) throws -> MiniCPMDemoPrefillResult {
        guard let chunk = Self.samples(input["audio"] ?? input["audio_waveform"])
                ?? Self.samples(input["pcm"])
        else {
            throw MiniCPMDemoBackendError.invalidMessage(
                "half-duplex audio must be a finite Float32 sample array")
        }
        guard !chunk.isEmpty else {
            throw MiniCPMDemoBackendError.invalidMessage("half-duplex audio chunk is empty")
        }
        guard chunk.allSatisfy(\.isFinite) else {
            throw MiniCPMDemoBackendError.invalidMessage("half-duplex audio contains NaN or infinity")
        }
        halfAudioBuffer.append(contentsOf: chunk)
        guard halfAudioBuffer.count <= maxHalfUtteranceSamples else {
            halfAudioBuffer.removeAll(keepingCapacity: true)
            throw MiniCPMDemoBackendError.engine(
                "half-duplex utterance exceeds \(maxHalfUtteranceSamples / 16_000)s; send utterance_end before the limit")
        }
        halfAudioSampleRate = Self.int(input, ["sample_rate", "sampleRate", "audio_sample_rate"])
            ?? halfAudioSampleRate
        let vadEvents = halfVAD?.process(
            samples: chunk,
            sampleRate: halfAudioSampleRate) ?? .init()
        if vadEvents.speechStarted { halfSpeechSeen = true }
        if vadEvents.speechEnded { halfSilentChunks = 0 }

        // Energy is an opt-in compatibility fallback only. Production should
        // inject a StreamingVADProvider (CoreML/ANE preferred) or send an
        // explicit utterance_end marker; an absent provider must not silently
        // masquerade as a real VAD decision.
        var energyEnd = false
        if allowEnergyVADFallback {
            let rms = sqrt(chunk.reduce(0.0) { $0 + Double($1 * $1) } / Double(chunk.count))
            if rms >= 0.012 {
                halfSpeechSeen = true
                halfSilentChunks = 0
            } else if halfSpeechSeen {
                halfSilentChunks += 1
            }
            energyEnd = halfSpeechSeen && halfSilentChunks >= 8
        }
        let isFinal = Self.bool(input, [
            "is_final", "isFinal", "utterance_end", "utteranceEnd", "audio_end", "end_of_turn"
        ]) ?? (vadEvents.speechEnded || energyEnd)
        guard isFinal else {
            // Without an explicit boundary or injected VAD, fail closed: no
            // ChatSession generation starts while the utterance is unknown.
            return MiniCPMDemoPrefillResult(accepted: false, needsMoreAudio: true)
        }

        if Self.bool(input, [
            "is_final", "isFinal", "utterance_end", "utteranceEnd", "audio_end", "end_of_turn"
        ]) == true {
            let flushed = halfVAD?.flush(sampleRate: halfAudioSampleRate) ?? .init()
            if flushed.speechStarted { halfSpeechSeen = true }
        }

        let key = "half_audio_\(UUID().uuidString)"
        let message = MiniCPMChatMessage(
            role: .user,
            parts: [.media(.audio(
                key,
                sampleRate: halfAudioSampleRate,
                sampleCount: halfAudioBuffer.count))])
        let context: MiniCPMStreamingPromptContext = halfConversationStarted
            ? .nextUser(completedResponse: halfAwaitingNextUser)
            : .firstUser
        let result: MiniCPMNativeChatPrefillResult
        do {
            result = try native.prefill(MiniCPMNativeChatInput(
                messages: [message],
                media: [key: .audio(
                    samples: halfAudioBuffer,
                    sampleRate: halfAudioSampleRate)],
                templateOptions: MiniCPMChatTemplateOptions(
                    addGenerationPrompt: false,
                    useTTSTemplate: false,
                    enableThinking: templateOptions.enableThinking,
                    contentSeparator: .none,
                    enableTTSTemplateForAudio: false),
                streamingContext: context,
                streamingAudio: false,
                resetSession: false))
        } catch {
            throw error
        }
        if result.accepted {
            halfAudioBuffer.removeAll(keepingCapacity: true)
            halfSpeechSeen = false
            halfSilentChunks = 0
            halfVAD?.reset()
        }
        pendingHalfResponse = result.accepted
        return MiniCPMDemoPrefillResult(
            accepted: result.accepted,
            needsMoreAudio: result.needsMoreAudio)
    }

    // MARK: Ordered wire conversion

    struct ParsedMessages {
        var messages: [MiniCPMChatMessage]
        var media: [String: MiniCPMNativeChatMedia]
    }

    static func parseMessages(
        _ value: MiniCPMJSONValue?,
        maxVideoFrames: Int
    ) throws -> ParsedMessages {
        guard case .array(let values) = value else {
            throw MiniCPMDemoBackendError.invalidMessage(
                "turn/half input requires an ordered messages array")
        }
        guard !values.isEmpty else {
            throw MiniCPMDemoBackendError.invalidMessage("messages must not be empty")
        }
        var messages: [MiniCPMChatMessage] = []
        var media: [String: MiniCPMNativeChatMedia] = [:]
        for (messageIndex, value) in values.enumerated() {
            guard case .object(let object) = value else {
                throw MiniCPMDemoBackendError.invalidMessage(
                    "messages[\(messageIndex)] must be an object")
            }
            let roleValue = object["role"]?.stringValue ?? "user"
            guard let role = MiniCPMChatRole(rawValue: roleValue.lowercased()) else {
                throw MiniCPMDemoBackendError.invalidMessage(
                    "messages[\(messageIndex)] has unsupported role: \(roleValue)")
            }
            let reasoning = object["reasoning_content"]?.stringValue
                ?? object["reasoningContent"]?.stringValue
            let content = object["content"]
            var parts: [MiniCPMChatContentPart] = []
            if let text = content?.stringValue {
                guard !text.isEmpty else {
                    throw MiniCPMDemoBackendError.invalidMessage(
                        "messages[\(messageIndex)].content text is empty")
                }
                parts = [.text(text)]
            } else if case .array(let contentParts) = content {
                for (partIndex, value) in contentParts.enumerated() {
                    guard case .object(let payload) = value else {
                        throw MiniCPMDemoBackendError.invalidMessage(
                            "messages[\(messageIndex)].content[\(partIndex)] must be an object")
                    }
                    let type = payload["type"]?.stringValue?.lowercased() ?? ""
                    switch type {
                    case "text":
                        guard let text = payload["text"]?.stringValue, !text.isEmpty else {
                            throw MiniCPMDemoBackendError.invalidMessage(
                                "messages[\(messageIndex)].content[\(partIndex)] text is empty")
                        }
                        parts.append(.text(text))
                    case "audio":
                        guard let samples = samples(payload["data"] ?? payload["audio"] ?? payload["base64"]),
                              !samples.isEmpty else {
                            throw MiniCPMDemoBackendError.invalidMessage(
                                "messages[\(messageIndex)].content[\(partIndex)] audio is invalid")
                        }
                        let sampleRate = int(payload, ["sample_rate", "sampleRate"]) ?? 16_000
                        let key = "m\(messageIndex)_p\(partIndex)"
                        let mediaValue = MiniCPMNativeChatMedia.audio(
                            samples: samples,
                            sampleRate: sampleRate)
                        media[key] = mediaValue
                        parts.append(.media(.audio(
                            key,
                            sampleRate: sampleRate,
                            sampleCount: samples.count)))
                    case "image", "frame":
                        let data = try encodedData(
                            payload["data"] ?? payload["base64"] ?? payload["image"] ?? payload["frame"],
                            path: "messages[\(messageIndex)].content[\(partIndex)]")
                        let key = "m\(messageIndex)_p\(partIndex)"
                        media[key] = .image(data)
                        parts.append(.media(.image(key)))
                    case "video", "mp4":
                        let data = try encodedData(
                            payload["data"] ?? payload["base64"] ?? payload["video"] ?? payload["mp4"],
                            path: "messages[\(messageIndex)].content[\(partIndex)]")
                        let stack = int(payload, ["stack_frames", "stackFrames"]) ?? 1
                        guard stack > 0 else {
                            throw MiniCPMDemoBackendError.invalidMessage(
                                "video stack_frames must be positive")
                        }
                        let decoded: MiniCPMDemoMediaPayload
                        do {
                            decoded = try MiniCPMDemoMediaDecoder.decodeVideo(
                                data,
                                stackFrames: stack > 1,
                                maxFrames: max(1, maxVideoFrames))
                        } catch {
                            throw MiniCPMDemoBackendError.invalidMessage(
                                "video/MP4 decode failed: \(error.localizedDescription)")
                        }
                        guard !decoded.frames.isEmpty else {
                            throw MiniCPMDemoBackendError.invalidMessage(
                                "video/MP4 contains no decodable frames")
                        }
                        let key = "m\(messageIndex)_p\(partIndex)"
                        media[key] = .video(
                            frames: decoded.frames,
                            audio: decoded.audio,
                            sampleRate: decoded.audioSampleRate)
                        parts.append(.media(.video(key)))
                    default:
                        throw MiniCPMDemoBackendError.invalidMessage(
                            "messages[\(messageIndex)].content[\(partIndex)] has unsupported type: \(type)")
                    }
                }
            } else {
                throw MiniCPMDemoBackendError.invalidMessage(
                    "messages[\(messageIndex)].content must be a string or array")
            }
            guard !parts.isEmpty else {
                throw MiniCPMDemoBackendError.invalidMessage(
                    "messages[\(messageIndex)].content is empty")
            }
            messages.append(MiniCPMChatMessage(
                role: role,
                parts: parts,
                reasoningContent: reasoning))
        }
        return ParsedMessages(messages: messages, media: media)
    }

    static func parseDirectInput(
        _ input: [String: MiniCPMJSONValue],
        maxVideoFrames: Int
    ) throws -> ParsedMessages {
        var parts: [MiniCPMChatContentPart] = []
        var media: [String: MiniCPMNativeChatMedia] = [:]
        if let text = input["text"]?.stringValue ?? input["content"]?.stringValue {
            guard !text.isEmpty else {
                throw MiniCPMDemoBackendError.invalidMessage("text content is empty")
            }
            parts.append(.text(text))
        }
        if let samples = samples(input["audio"] ?? input["audio_waveform"] ?? input["pcm"]) {
            let rate = int(input, ["sample_rate", "sampleRate", "audio_sample_rate"]) ?? 16_000
            let key = "direct_audio"
            media[key] = .audio(samples: samples, sampleRate: rate)
            parts.append(.media(.audio(key, sampleRate: rate, sampleCount: samples.count)))
        }
        if let imageValue = input["image"] ?? input["frame"] ?? input["frame_data"] {
            let data = try encodedData(imageValue, path: "image")
            let key = "direct_image"
            media[key] = .image(data)
            parts.append(.media(.image(key)))
        }
        if let videoValue = input["video"] ?? input["mp4"] ?? input["video_data"] {
            let data = try encodedData(videoValue, path: "video")
            let decoded: MiniCPMDemoMediaPayload
            do {
                decoded = try MiniCPMDemoMediaDecoder.decodeVideo(
                    data,
                    stackFrames: true,
                    maxFrames: max(1, maxVideoFrames))
            } catch {
                throw MiniCPMDemoBackendError.invalidMessage(
                    "video/MP4 decode failed: \(error.localizedDescription)")
            }
            guard !decoded.frames.isEmpty else {
                throw MiniCPMDemoBackendError.invalidMessage("video/MP4 contains no decodable frames")
            }
            let key = "direct_video"
            media[key] = .video(
                frames: decoded.frames,
                audio: decoded.audio,
                sampleRate: decoded.audioSampleRate)
            parts.append(.media(.video(key)))
        }
        guard !parts.isEmpty else {
            throw MiniCPMDemoBackendError.invalidMessage(
                "turn/half input requires messages, text, audio, image, or video")
        }
        return ParsedMessages(
            messages: [MiniCPMChatMessage(role: .user, parts: parts)],
            media: media)
    }

    private static func hasDirectAudio(_ input: [String: MiniCPMJSONValue]) -> Bool {
        (input["audio"] ?? input["audio_waveform"] ?? input["pcm"]) != nil
    }

    private static func isMessagesArray(_ value: MiniCPMJSONValue?) -> Bool {
        guard let value else { return false }
        if case .array = value { return true }
        return false
    }

    private static func mergedParameters(
        _ object: [String: MiniCPMJSONValue]
    ) -> [String: MiniCPMJSONValue] {
        var values = object
        for key in ["config", "generation", "sampling", "chat", "tts", "defaults"] {
            guard let nested = object[key]?.objectValue else { continue }
            for (name, value) in nested { values[name] = value }
            for (name, value) in nested {
                values["\(key).\(name)"] = value
            }
        }
        return values
    }

    private static func string(
        _ values: [String: MiniCPMJSONValue],
        _ keys: [String]
    ) -> String? {
        keys.lazy.compactMap { values[$0]?.stringValue }.first
    }

    private static func bool(
        _ values: [String: MiniCPMJSONValue],
        _ keys: [String]
    ) -> Bool? {
        for key in keys {
            if let value = values[key]?.boolValue { return value }
            if let value = values[key]?.stringValue {
                switch value.lowercased() {
                case "1", "true", "yes", "on": return true
                case "0", "false", "no", "off": return false
                default: break
                }
            }
        }
        return nil
    }

    private static func int(
        _ values: [String: MiniCPMJSONValue],
        _ keys: [String]
    ) -> Int? {
        keys.lazy.compactMap { values[$0]?.numberValue.map(Int.init) }.first
    }

    private static func float(
        _ values: [String: MiniCPMJSONValue],
        _ keys: [String]
    ) -> Float? {
        keys.lazy.compactMap { values[$0]?.numberValue.map(Float.init) }.first
    }

    private static func hasGenerationOverride(
        _ values: [String: MiniCPMJSONValue]
    ) -> Bool {
        [
            "max_new_tokens", "maxNewTokens", "max_new_speak_tokens_per_chunk",
            "temperature", "top_k", "topK", "top_p", "topP", "do_sample",
            "repetition_penalty", "repetitionPenalty", "length_penalty",
            "listen_top_k", "listenTopK", "listen_prob_scale",
            "listen_probability_scale", "listenProbabilityScale"
        ].contains { values[$0] != nil }
    }

    private static func sampling(
        _ values: [String: MiniCPMJSONValue],
        fallback: MiniCPMSamplingConfig
    ) -> MiniCPMSamplingConfig {
        MiniCPMSamplingConfig(
            mode: bool(values, ["do_sample", "sampling.do_sample"]) == false
                || string(values, ["sampling.mode", "sampling_mode"])?.lowercased() == "greedy"
                ? .greedy : fallback.mode,
            temperature: float(values, ["temperature", "sampling.temperature"]) ?? fallback.temperature,
            topK: int(values, ["top_k", "topK", "sampling.top_k", "sampling.topK"]) ?? fallback.topK,
            topP: float(values, ["top_p", "topP", "sampling.top_p", "sampling.topP"]) ?? fallback.topP,
            repetitionPenalty: float(values, [
                "repetition_penalty", "repetitionPenalty", "sampling.repetition_penalty"
            ]) ?? fallback.repetitionPenalty,
            repetitionWindowSize: int(values, [
                "repetition_window_size", "text_repetition_window_size"
            ]) ?? fallback.repetitionWindowSize,
            lengthPenalty: float(values, ["length_penalty", "sampling.length_penalty"]) ?? fallback.lengthPenalty,
            listenTopK: int(values, ["listen_top_k", "listenTopK"]) ?? fallback.listenTopK,
            listenProbabilityScale: float(values, [
                "listen_prob_scale", "listen_probability_scale", "listenProbabilityScale"
            ]) ?? fallback.listenProbabilityScale)
    }

    private static func samples(_ value: MiniCPMJSONValue?) -> [Float]? {
        guard let value else { return nil }
        switch value {
        case .array(let values):
            let result = values.compactMap { $0.numberValue.map(Float.init) }
            guard result.count == values.count, !result.isEmpty,
                  result.allSatisfy(\.isFinite) else { return nil }
            return result
        case .object(let object):
            return samples(object["data"] ?? object["pcm"] ?? object["audio"])
        case .string(let encoded):
            guard let data = Data(base64Encoded: encoded), data.count.isMultiple(of: 4), !data.isEmpty else {
                return nil
            }
            let result = stride(from: 0, to: data.count, by: 4).map { offset -> Float in
                let bits = data[offset..<offset + 4].withUnsafeBytes {
                    $0.loadUnaligned(as: UInt32.self)
                }
                return Float(bitPattern: UInt32(littleEndian: bits))
            }
            return result.allSatisfy(\.isFinite) ? result : nil
        default:
            return nil
        }
    }

    private static func samples(
        _ values: [String: MiniCPMJSONValue],
        keys: [String]
    ) -> [Float]? {
        for key in keys {
            if let result = samples(values[key]) { return result }
        }
        return nil
    }

    private static func encodeFloat32(_ samples: [Float]) -> String {
        var data = Data(capacity: samples.count * MemoryLayout<Float>.stride)
        for sample in samples {
            var bits = sample.bitPattern.littleEndian
            withUnsafeBytes(of: &bits) { data.append(contentsOf: $0) }
        }
        return data.base64EncodedString()
    }

    private static func encodedData(
        _ value: MiniCPMJSONValue?,
        path: String
    ) throws -> Data {
        guard let value else {
            throw MiniCPMDemoBackendError.invalidMessage("\(path) requires data")
        }
        switch value {
        case .string(let encoded):
            guard let data = Data(base64Encoded: encoded), !data.isEmpty else {
                throw MiniCPMDemoBackendError.invalidMessage("\(path) is not valid base64")
            }
            return data
        case .array(let values):
            var data = Data(capacity: values.count)
            for value in values {
                guard let number = value.numberValue,
                      number.isFinite,
                      number >= 0,
                      number <= 255,
                      number.rounded() == number else {
                    throw MiniCPMDemoBackendError.invalidMessage("\(path) must contain encoded byte values")
                }
                data.append(UInt8(number))
            }
            guard !data.isEmpty else {
                throw MiniCPMDemoBackendError.invalidMessage("\(path) is empty")
            }
            return data
        case .object(let object):
            return try encodedData(
                object["data"] ?? object["base64"] ?? object["image"] ?? object["video"] ?? object["mp4"],
                path: path)
        default:
            throw MiniCPMDemoBackendError.invalidMessage("\(path) must contain encoded bytes")
        }
    }
}
