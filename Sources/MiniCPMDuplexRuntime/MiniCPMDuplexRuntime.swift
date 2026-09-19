import Foundation
import MLX
import MiniCPMAudio
import MiniCPMLLM
import MiniCPMTTSSemantic
import MiniCPMToken2Wav
import MiniCPMVision

/// The three execution surfaces exposed by MiniCPM-o's Demo.  The runtime
/// keeps them in one actor so a half-duplex request can never accidentally
/// consume the full-duplex listen/speak session.
public enum MiniCPMDuplexMode: String, Codable, Sendable {
    case turnBased = "turn_based"
    case halfDuplex = "half_duplex"
    case fullDuplex = "full_duplex"
    case omni = "omni"
}

public struct MiniCPMDuplexConfig: Sendable, Equatable {
    public var maxNewTokens: Int
    public var temperature: Float
    public var topK: Int
    public var topP: Float
    /// Legacy token-cap alias retained for source compatibility. It is no
    /// longer divided by a guessed per-unit constant; when supplied it maps
    /// to an explicit basic high/low watermark.
    @available(*, deprecated, message: "Use slidingWindowMode/basicWindowHighTokens/basicWindowLowTokens")
    public var slidingWindowTokens: Int?
    /// Official DuplexWindowConfig mode: ``off``, ``basic`` or ``context``.
    public var slidingWindowMode: String
    public var basicWindowHighTokens: Int
    public var basicWindowLowTokens: Int
    public var contextPreviousMaxTokens: Int
    public var contextMaxUnits: Int
    public var contextPreviousMarker: String
    /// Keep the official MiniCPM streaming protocol's explicit chunk
    /// terminator behavior by default.
    public var lsMode: String
    public var forceListenCount: Int
    public var listenProbabilityScale: Float
    public var listenTopK: Int?
    /// Make only the leading listen/speak decision deterministic while the
    /// model is between turns. A per-input recent-speech override may restore
    /// the official sampled decision without changing text generation.
    public var greedyListenSpeakDecision: Bool
    public var repetitionPenalty: Float
    public var repetitionWindowSize: Int
    /// Penalty applied to protocol terminators after the natural chunk-EOS
    /// probe (official MiniCPM default: 1.1).
    public var lengthPenalty: Float
    /// Semantic TTS has an independent sampler in the official duplex
    /// runtime; do not inherit the text-head temperature by accident.
    public var ttsTemperature: Float
    public var ttsRepetitionPenalty: Float
    public var generateAudio: Bool
    public var systemPrompt: String
    /// Optional audio embedded into the LLM system prompt.  This is the
    /// official ``voice.ref_audio`` channel and is deliberately separate
    /// from the Token2Wav voice prompt: callers may use one speaker reference
    /// for understanding and another for generated speech.
    public var llmReferenceAudio: [Float]?
    public var llmReferenceAudioSampleRate: Int
    /// Optional audio used only to prepare the Token2Wav/CosyVoice prompt.
    /// It must never be fed through the live MiniCPM audio encoder.
    public var ttsReferenceAudio: [Float]?
    public var ttsReferenceAudioSampleRate: Int
    /// Legacy single-reference field.  When supplied, it initializes both
    /// channels unless an explicit channel value is provided.  New callers
    /// should prefer ``llmReferenceAudio`` and ``ttsReferenceAudio``.
    public var referenceAudio: [Float]?
    public var referenceAudioSampleRate: Int

    public init(
        maxNewTokens: Int = 20,
        temperature: Float = 0.7,
        topK: Int = 20,
        topP: Float = 0.8,
        slidingWindowTokens: Int? = nil,
        lsMode: String = "explicit",
        // Match the official MiniCPM-o 4.5 duplex default. Startup silence is
        // gated by the transport instead of adding three seconds of latency.
        forceListenCount: Int = 0,
        listenProbabilityScale: Float = 1.0,
        listenTopK: Int? = nil,
        repetitionPenalty: Float = 1.05,
        repetitionWindowSize: Int = 512,
        generateAudio: Bool = true,
        systemPrompt: String = "Streaming Omni Conversation.",
        referenceAudio: [Float]? = nil,
        referenceAudioSampleRate: Int = 16_000,
        // New sampler/TTS controls are appended after the original
        // positional arguments.  Keeping every legacy label in its former
        // slot means clients that used the pre-duplex positional initializer
        // continue to compile; named callers are unaffected.
        lengthPenalty: Float = 1.1,
        ttsTemperature: Float = 0.8,
        ttsRepetitionPenalty: Float = 1.05,
        llmReferenceAudio: [Float]? = nil,
        llmReferenceAudioSampleRate: Int? = nil,
        ttsReferenceAudio: [Float]? = nil,
        ttsReferenceAudioSampleRate: Int? = nil,
        slidingWindowMode: String = "off",
        basicWindowHighTokens: Int = 8_000,
        basicWindowLowTokens: Int = 6_000,
        contextPreviousMaxTokens: Int = 500,
        contextMaxUnits: Int = 24,
        contextPreviousMarker: String = "\n\nprevious: ",
        greedyListenSpeakDecision: Bool = false
    ) {
        self.maxNewTokens = max(0, maxNewTokens)
        self.temperature = temperature
        self.topK = topK
        self.topP = topP
        let normalizedLegacyWindow = slidingWindowTokens.map { max(0, $0) }
        self.slidingWindowTokens = normalizedLegacyWindow
        let normalizedMode = slidingWindowMode.lowercased()
        // A legacy cap enables basic mode explicitly. This preserves old
        // callers while eliminating the former `/256` unit approximation.
        self.slidingWindowMode = normalizedLegacyWindow != nil && normalizedMode == "off"
            ? "basic" : (["off", "basic", "context"].contains(normalizedMode) ? normalizedMode : "off")
        let high = normalizedLegacyWindow ?? max(0, basicWindowHighTokens)
        self.basicWindowHighTokens = high
        self.basicWindowLowTokens = max(0, min(normalizedLegacyWindow ?? basicWindowLowTokens, high))
        self.contextPreviousMaxTokens = max(0, contextPreviousMaxTokens)
        self.contextMaxUnits = max(0, contextMaxUnits)
        self.contextPreviousMarker = contextPreviousMarker
        self.lsMode = lsMode
        self.forceListenCount = max(0, forceListenCount)
        self.listenProbabilityScale = max(0, listenProbabilityScale)
        self.listenTopK = listenTopK.map { max(0, $0) }
        self.greedyListenSpeakDecision = greedyListenSpeakDecision
        self.repetitionPenalty = repetitionPenalty
        self.repetitionWindowSize = max(0, repetitionWindowSize)
        self.lengthPenalty = lengthPenalty
        self.ttsTemperature = ttsTemperature
        self.ttsRepetitionPenalty = ttsRepetitionPenalty
        self.generateAudio = generateAudio
        self.systemPrompt = systemPrompt
        // Preserve source compatibility with the original one-reference
        // initializer while making the two official voice channels explicit.
        self.llmReferenceAudio = llmReferenceAudio ?? referenceAudio
        self.llmReferenceAudioSampleRate =
            llmReferenceAudioSampleRate ?? referenceAudioSampleRate
        self.ttsReferenceAudio = ttsReferenceAudio ?? referenceAudio
        self.ttsReferenceAudioSampleRate =
            ttsReferenceAudioSampleRate ?? referenceAudioSampleRate
        self.referenceAudio = referenceAudio
        self.referenceAudioSampleRate = referenceAudioSampleRate
    }
}

/// One ordered multimodal item from MiniCPM-o's chat template. Keeping the
/// item tagged prevents callers from accidentally reordering audio and text
/// by flattening them into independent legacy fields.
public enum MiniCPMDuplexContentPart: Sendable, Equatable {
    case text(String)
    case audio(samples: [Float], sampleRate: Int)
    case image(Data)
    case video(Data)

    public var isEmpty: Bool {
        switch self {
        case .text(let value): return value.isEmpty
        case .audio(let samples, _): return samples.isEmpty
        case .image(let data), .video(let data): return data.isEmpty
        }
    }
}

/// Role-bearing content in source order. The role remains a string to carry
/// developer/tool roles introduced by newer official templates.
public struct MiniCPMDuplexMessage: Sendable, Equatable {
    public var role: String
    public var parts: [MiniCPMDuplexContentPart]

    public init(role: String, parts: [MiniCPMDuplexContentPart]) {
        self.role = role
        self.parts = parts
    }

    public var isEmpty: Bool {
        role.isEmpty || parts.isEmpty || parts.allSatisfy(\.isEmpty)
    }
}

public struct MiniCPMDuplexInput: Sendable, Equatable {
    public var audio: [Float]?
    /// Encoded image/video frames.  Each element is decoded by the native
    /// MiniCPM image processor; images are deliberately not represented as
    /// ``[[Float]]`` because that loses format and geometry metadata.
    public var frameData: [Data]?
    /// Optional reference/voice PCM.  This is distinct from ``audio`` (the
    /// live user chunk) and is consumed by Token2Wav prompt preparation.
    public var voicePCM: [Float]?
    public var voiceSampleRate: Int?
    /// Legacy packed-frame field retained for source compatibility. Native
    /// engines reject it instead of guessing whether rows are RGB pixels.
    public var frames: [[Float]]?
    public var text: String?
    public var metadata: [String: String]
    /// Optional per-frame vision slice limits. A single value may be supplied
    /// for all frames; an array preserves the official per-frame form.
    public var maxSliceNums: [Int]?
    /// Per-input protocol override. This affects only the next generation;
    /// the configured ``MiniCPMDuplexConfig.forceListenCount`` is untouched.
    public var forceListen: Bool?
    /// Allow the next leading listen/speak decision to use official sampling.
    /// Browser clients set this only while recent user speech is present, so
    /// long silence remains deterministic without delaying sentence endings.
    public var allowSampledListenSpeakDecision: Bool?
    /// Advance an already-open spoken turn without re-encoding a transport
    /// silence packet. Engines must validate that the supplied audio is all
    /// zero and that the model is currently speaking before honoring it.
    public var continuationOnly: Bool
    /// Ordered, role-bearing multimodal content. Legacy fields remain for
    /// direct streaming chunks; chat-template callers should populate this
    /// field so the runtime can preserve source order or fail closed.
    public var messages: [MiniCPMDuplexMessage]?

    public init(
        audio: [Float]? = nil,
        frameData: [Data]? = nil,
        voicePCM: [Float]? = nil,
        voiceSampleRate: Int? = nil,
        frames: [[Float]]? = nil,
        text: String? = nil,
        metadata: [String: String] = [:],
        messages: [MiniCPMDuplexMessage]? = nil,
        forceListen: Bool? = nil,
        maxSliceNums: [Int]? = nil,
        allowSampledListenSpeakDecision: Bool? = nil,
        continuationOnly: Bool = false
    ) {
        self.audio = audio
        self.frameData = frameData
        self.voicePCM = voicePCM
        self.voiceSampleRate = voiceSampleRate
        self.frames = frames
        self.text = text
        self.metadata = metadata
        self.messages = messages
        self.forceListen = forceListen
        self.maxSliceNums = maxSliceNums
        self.allowSampledListenSpeakDecision = allowSampledListenSpeakDecision
        self.continuationOnly = continuationOnly
    }

    public var isEmpty: Bool {
        (audio?.isEmpty ?? true)
            && (frameData?.isEmpty ?? true)
            && (voicePCM?.isEmpty ?? true)
            && (frames?.isEmpty ?? true)
            && (text?.isEmpty ?? true)
            && (messages?.isEmpty ?? true)
    }
}

public struct MiniCPMDuplexPrefillResult: Sendable, Equatable {
    public var accepted: Bool
    public var mode: MiniCPMDuplexMode
    public var input: MiniCPMDuplexInput

    public init(accepted: Bool = true, mode: MiniCPMDuplexMode, input: MiniCPMDuplexInput) {
        self.accepted = accepted
        self.mode = mode
        self.input = input
    }
}

public struct MiniCPMDuplexOutput: Sendable, Equatable {
    public var text: String
    public var audio: [Float]?
    /// Number of leading samples added only to keep a non-final streaming
    /// packet on MiniCPM's one-second transport timeline. The browser may
    /// remove this explicitly marked padding before playback; nil means the
    /// waveform has no transport padding.
    public var audioPaddingSamples: Int?
    public var isListen: Bool
    public var endOfTurn: Bool
    /// True until ``finalize`` accepts the generated unit or ``rollback``
    /// discards it.  This mirrors the Python coordinator's deferred marker.
    public var needsFinalize: Bool
    public var finishReason: String?

    public init(
        text: String = "",
        audio: [Float]? = nil,
        audioPaddingSamples: Int? = nil,
        isListen: Bool = false,
        endOfTurn: Bool = false,
        needsFinalize: Bool = true,
        finishReason: String? = nil
    ) {
        self.text = text
        self.audio = audio
        self.audioPaddingSamples = audioPaddingSamples
        self.isListen = isListen
        self.endOfTurn = endOfTurn
        self.needsFinalize = needsFinalize
        self.finishReason = finishReason
    }
}

public enum MiniCPMDuplexRuntimeError: Error, LocalizedError, Sendable, Equatable {
    case unknownSession(String)
    case invalidState(String)
    case emptyInput

    public var errorDescription: String? {
        switch self {
        case .unknownSession(let id): return "unknown MiniCPM duplex session: \(id)"
        case .invalidState(let message): return message
        case .emptyInput: return "MiniCPM duplex input is empty"
        }
    }
}

/// Model boundary owned by one actor session.  A concrete implementation can
/// wrap ``MiniCPMSession`` plus audio/vision/TTS adapters; keeping this
/// protocol framework-neutral lets the state machine be tested without a
/// checkpoint or Metal device.
public protocol MiniCPMDuplexEngine: AnyObject {
    func prepare(mode: MiniCPMDuplexMode, config: MiniCPMDuplexConfig) throws -> String
    func prefill(mode: MiniCPMDuplexMode, input: MiniCPMDuplexInput) throws -> MiniCPMDuplexPrefillResult
    func generate(mode: MiniCPMDuplexMode, config: MiniCPMDuplexConfig) throws -> MiniCPMDuplexOutput
    func finalize() throws
    func rollback() throws
    func interrupt()
    func reset()
    func setSlidingWindow(tokens: Int?)
    func setSlidingWindow(config: MiniCPMDuplexConfig)
    /// Install a synchronous, lock-free-at-call-site cancellation probe for
    /// long-running token/audio generation.  The runtime actor itself cannot
    /// be re-entered while ``generate`` is executing, so transports signal
    /// cancellation through a separate registry and the engine polls this
    /// closure at safe model-step boundaries.
    func setInterruptionProbe(_ probe: (@Sendable () -> Bool)?)
    /// Attach the transport/runtime generation ticket to engine diagnostics.
    /// Engines that do not expose model diagnostics may keep the default no-op.
    func setGenerationEpoch(_ epoch: UInt64)
    /// Return a structured snapshot of mutable model/turn state.  The
    /// snapshot is deliberately value-only so it can cross the backend
    /// adapter without retaining MLX arrays or model objects.
    func diagnosticSnapshot() -> MiniCPMDuplexEngineDiagnosticSnapshot?
}

public extension MiniCPMDuplexEngine {
    func setSlidingWindow(tokens: Int?) { _ = tokens }
    func setSlidingWindow(config: MiniCPMDuplexConfig) {
        setSlidingWindow(tokens: config.slidingWindowTokens)
    }

    func setInterruptionProbe(_ probe: (@Sendable () -> Bool)?) {
        _ = probe
    }

    func setGenerationEpoch(_ epoch: UInt64) {
        _ = epoch
    }

    func diagnosticSnapshot() -> MiniCPMDuplexEngineDiagnosticSnapshot? {
        nil
    }
}

/// Value-only state captured around a native duplex generation/cancel.
/// Keeping this contract in the runtime target lets the WebSocket layer emit
/// auditable traces without importing MLX or serializing implementation
/// objects.  Optional fields are intentionally nullable for deterministic
/// test engines and future backends that do not own those caches.
public struct MiniCPMDuplexEngineDiagnosticSnapshot: Sendable, Equatable {
    public let generationEpoch: UInt64?
    public let unitIndex: Int
    public let currentTurnEnded: Bool
    public let committedUnitCount: Int
    public let kvCacheLength: Int?
    public let slidingWindowMode: String?
    public let slidingWindowHighWatermark: Int?
    public let slidingWindowLowWatermark: Int?
    public let pendingTokenCount: Int
    public let pendingTerminator: Int?
    public let prefetched: Bool
    public let needsFinalize: Bool
    public let ttsContextPresent: Bool
    public let ttsPendingToken: Int?
    public let ttsCommittedTokenCount: Int
    public let ttsCacheLength: Int
    public let token2WavPrepared: Bool
    public let token2WavPendingTokenCount: Int?
    public let token2WavDecoderPositions: Int?
    public let continuationOnly: Bool
    public let audioEncoderMS: Double?
    public let audioBridgeMS: Double?
    public let llmPrefillMS: Double?
    public let llmDecodeMS: Double?
    public let semanticTTSMS: Double?
    public let token2WavMS: Double?
    public let flowMS: Double?
    public let hiftMS: Double?
    public let generationStartedAtNS: UInt64?
    public let generationExitedAtNS: UInt64?
    public let cancelReturnedAtNS: UInt64?

    public init(
        generationEpoch: UInt64? = nil,
        unitIndex: Int = 0,
        currentTurnEnded: Bool = true,
        committedUnitCount: Int = 0,
        kvCacheLength: Int? = nil,
        slidingWindowMode: String? = nil,
        slidingWindowHighWatermark: Int? = nil,
        slidingWindowLowWatermark: Int? = nil,
        pendingTokenCount: Int = 0,
        pendingTerminator: Int? = nil,
        prefetched: Bool = false,
        needsFinalize: Bool = false,
        ttsContextPresent: Bool = false,
        ttsPendingToken: Int? = nil,
        ttsCommittedTokenCount: Int = 0,
        ttsCacheLength: Int = 0,
        token2WavPrepared: Bool = false,
        token2WavPendingTokenCount: Int? = nil,
        token2WavDecoderPositions: Int? = nil,
        continuationOnly: Bool = false,
        audioEncoderMS: Double? = nil,
        audioBridgeMS: Double? = nil,
        llmPrefillMS: Double? = nil,
        llmDecodeMS: Double? = nil,
        semanticTTSMS: Double? = nil,
        token2WavMS: Double? = nil,
        flowMS: Double? = nil,
        hiftMS: Double? = nil,
        generationStartedAtNS: UInt64? = nil,
        generationExitedAtNS: UInt64? = nil,
        cancelReturnedAtNS: UInt64? = nil
    ) {
        self.generationEpoch = generationEpoch
        self.unitIndex = unitIndex
        self.currentTurnEnded = currentTurnEnded
        self.committedUnitCount = committedUnitCount
        self.kvCacheLength = kvCacheLength
        self.slidingWindowMode = slidingWindowMode
        self.slidingWindowHighWatermark = slidingWindowHighWatermark
        self.slidingWindowLowWatermark = slidingWindowLowWatermark
        self.pendingTokenCount = max(0, pendingTokenCount)
        self.pendingTerminator = pendingTerminator
        self.prefetched = prefetched
        self.needsFinalize = needsFinalize
        self.ttsContextPresent = ttsContextPresent
        self.ttsPendingToken = ttsPendingToken
        self.ttsCommittedTokenCount = max(0, ttsCommittedTokenCount)
        self.ttsCacheLength = max(0, ttsCacheLength)
        self.token2WavPrepared = token2WavPrepared
        self.token2WavPendingTokenCount = token2WavPendingTokenCount
        self.token2WavDecoderPositions = token2WavDecoderPositions
        self.continuationOnly = continuationOnly
        self.audioEncoderMS = audioEncoderMS
        self.audioBridgeMS = audioBridgeMS
        self.llmPrefillMS = llmPrefillMS
        self.llmDecodeMS = llmDecodeMS
        self.semanticTTSMS = semanticTTSMS
        self.token2WavMS = token2WavMS
        self.flowMS = flowMS
        self.hiftMS = hiftMS
        self.generationStartedAtNS = generationStartedAtNS
        self.generationExitedAtNS = generationExitedAtNS
        self.cancelReturnedAtNS = cancelReturnedAtNS
    }
}

/// Runtime-level trace snapshot.  It combines the actor/transport epoch
/// bookkeeping with the native engine's KV/TTS snapshot and is the payload
/// used by the Demo backend's structured interrupt trace.
public struct MiniCPMDuplexRuntimeDiagnosticSnapshot: Sendable, Equatable {
    public let sessionID: String
    public let generationEpoch: UInt64
    public let staleOutputDiscardCount: UInt64
    public let cancelRequestedAtNS: UInt64?
    public let cancelReturnedAtNS: UInt64?
    public let engine: MiniCPMDuplexEngineDiagnosticSnapshot?

    public init(
        sessionID: String,
        generationEpoch: UInt64,
        staleOutputDiscardCount: UInt64 = 0,
        cancelRequestedAtNS: UInt64? = nil,
        cancelReturnedAtNS: UInt64? = nil,
        engine: MiniCPMDuplexEngineDiagnosticSnapshot? = nil
    ) {
        self.sessionID = sessionID
        self.generationEpoch = generationEpoch
        self.staleOutputDiscardCount = staleOutputDiscardCount
        self.cancelRequestedAtNS = cancelRequestedAtNS
        self.cancelReturnedAtNS = cancelReturnedAtNS
        self.engine = engine
    }
}

/// A deterministic implementation useful for wiring smoke tests and as the
/// safe default before MLX model components are injected.
public final class MiniCPMNoopDuplexEngine: MiniCPMDuplexEngine {
    private var prepared = false
    private var prefetched = false
    private var needsFinalize = false

    public init() {}

    public func prepare(mode: MiniCPMDuplexMode, config: MiniCPMDuplexConfig) throws -> String {
        prepared = true
        prefetched = false
        needsFinalize = false
        return "<|im_start|>system\nStreaming Omni Conversation.\n<|audio_start|><|audio_end|><|im_end|>"
    }

    public func prefill(mode: MiniCPMDuplexMode, input: MiniCPMDuplexInput) throws -> MiniCPMDuplexPrefillResult {
        guard prepared else { throw MiniCPMDuplexRuntimeError.invalidState("prepare must be called first") }
        guard !input.isEmpty else { throw MiniCPMDuplexRuntimeError.emptyInput }
        guard !needsFinalize else { throw MiniCPMDuplexRuntimeError.invalidState("previous unit needs finalize") }
        prefetched = true
        return MiniCPMDuplexPrefillResult(mode: mode, input: input)
    }

    public func generate(mode: MiniCPMDuplexMode, config: MiniCPMDuplexConfig) throws -> MiniCPMDuplexOutput {
        guard prefetched else { throw MiniCPMDuplexRuntimeError.invalidState("prefill must be called first") }
        needsFinalize = true
        if mode == .fullDuplex || mode == .omni {
            return MiniCPMDuplexOutput(isListen: true, endOfTurn: false, needsFinalize: true, finishReason: "listen")
        }
        return MiniCPMDuplexOutput(text: "", isListen: false, endOfTurn: true, needsFinalize: true, finishReason: "stop")
    }

    public func finalize() throws {
        guard needsFinalize else { return }
        needsFinalize = false
        prefetched = false
    }

    public func rollback() throws {
        needsFinalize = false
        prefetched = false
    }

    public func interrupt() {
        needsFinalize = false
        prefetched = false
    }

    public func reset() {
        prepared = false
        prefetched = false
        needsFinalize = false
    }
}

/// Synchronous interruption state shared by the transport and the runtime
/// actor.  ``MiniCPMDuplexRuntime.generate`` intentionally remains a
/// synchronous engine call, so an actor-isolated ``interrupt`` method cannot
/// run until a long MLX token/TTS loop has returned.  A monotonic per-session
/// epoch lets the receive task request cancellation immediately without
/// touching any mutable model/KV state from another thread.
private final class MiniCPMDuplexInterruptionRegistry: @unchecked Sendable {
    private let lock = NSLock()
    private var epochs: [String: UInt64] = [:]
    private var requestedAtNS: [String: UInt64] = [:]

    @discardableResult
    func register(sessionID: String) -> UInt64 {
        lock.lock()
        defer { lock.unlock() }
        if let epoch = epochs[sessionID] { return epoch }
        epochs[sessionID] = 0
        return 0
    }

    func remove(sessionID: String) {
        lock.lock()
        epochs.removeValue(forKey: sessionID)
        requestedAtNS.removeValue(forKey: sessionID)
        lock.unlock()
    }

    func current(sessionID: String) -> UInt64? {
        lock.lock()
        defer { lock.unlock() }
        return epochs[sessionID]
    }

    /// Advance the session epoch.  A generation that captured the prior
    /// epoch will observe a mismatch; a later generation captures the new
    /// value and therefore does not consume an old cancel request.
    @discardableResult
    func requestInterrupt(sessionID: String) -> UInt64? {
        lock.lock()
        defer { lock.unlock() }
        guard let current = epochs[sessionID] else { return nil }
        let next = current &+ 1
        epochs[sessionID] = next
        requestedAtNS[sessionID] = DispatchTime.now().uptimeNanoseconds
        return next
    }

    func lastRequestAtNS(sessionID: String) -> UInt64? {
        lock.lock()
        defer { lock.unlock() }
        return requestedAtNS[sessionID]
    }

    func isCurrent(sessionID: String, epoch: UInt64) -> Bool {
        lock.lock()
        defer { lock.unlock() }
        return epochs[sessionID] == epoch
    }
}

public actor MiniCPMDuplexRuntime {
    public struct SessionSnapshot: Sendable, Equatable {
        public let id: String
        public let mode: MiniCPMDuplexMode
        public let prepared: Bool
        public let prefetched: Bool
        public let needsFinalize: Bool
        public let interrupted: Bool
        public let slidingWindowTokens: Int?
        public let slidingWindowMode: String
        public let basicWindowHighTokens: Int
        public let basicWindowLowTokens: Int
        public let contextPreviousMaxTokens: Int
        public let contextMaxUnits: Int
        public let contextPreviousMarker: String

        init(
            id: String,
            mode: MiniCPMDuplexMode,
            prepared: Bool,
            prefetched: Bool,
            needsFinalize: Bool,
            interrupted: Bool,
            config: MiniCPMDuplexConfig
        ) {
            self.id = id
            self.mode = mode
            self.prepared = prepared
            self.prefetched = prefetched
            self.needsFinalize = needsFinalize
            self.interrupted = interrupted
            self.slidingWindowTokens = config.slidingWindowTokens
            self.slidingWindowMode = config.slidingWindowMode
            self.basicWindowHighTokens = config.basicWindowHighTokens
            self.basicWindowLowTokens = config.basicWindowLowTokens
            self.contextPreviousMaxTokens = config.contextPreviousMaxTokens
            self.contextMaxUnits = config.contextMaxUnits
            self.contextPreviousMarker = config.contextPreviousMarker
        }
    }

    private final class Session {
        let id: String
        let engine: any MiniCPMDuplexEngine
        var mode: MiniCPMDuplexMode
        var config: MiniCPMDuplexConfig
        var prepared = false
        var prefetched = false
        var needsFinalize = false
        var interrupted = false
        /// Epoch captured by the current prefill/generation transaction.
        /// ``requestInterrupt`` advances the shared registry without waiting
        /// for this actor; generate compares this ticket before and after the
        /// blocking engine call.
        var generationEpoch: UInt64 = 0
        var staleOutputDiscardCount: UInt64 = 0
        var lastCancelRequestedAtNS: UInt64?
        var lastCancelReturnedAtNS: UInt64?

        init(id: String, engine: any MiniCPMDuplexEngine, mode: MiniCPMDuplexMode, config: MiniCPMDuplexConfig) {
            self.id = id
            self.engine = engine
            self.mode = mode
            self.config = config
        }
    }

    public typealias EngineFactory = () -> any MiniCPMDuplexEngine

    private let engineFactory: EngineFactory
    /// Deliberately nonisolated: transport cancellation must be able to set
    /// the epoch while this actor is inside a synchronous MLX generate call.
    private nonisolated let interruptionRegistry: MiniCPMDuplexInterruptionRegistry
    private var sessions: [String: Session] = [:]

    public init(engineFactory: @escaping EngineFactory = { MiniCPMNoopDuplexEngine() }) {
        self.engineFactory = engineFactory
        self.interruptionRegistry = MiniCPMDuplexInterruptionRegistry()
    }

    /// Request a cancellation without awaiting the actor.  This method only
    /// changes a lock-protected epoch; rollback/turn repair remains inside
    /// the actor and the engine's synchronous execution path.
    @discardableResult
    public nonisolated func requestInterrupt(sessionID: String = "default") -> UInt64? {
        interruptionRegistry.requestInterrupt(sessionID: sessionID)
    }

    public func sessionIDs() -> [String] {
        sessions.keys.sorted()
    }

    public func snapshot(sessionID: String) throws -> SessionSnapshot {
        let session = try requireSession(sessionID)
        return SessionSnapshot(
            id: session.id,
            mode: session.mode,
            prepared: session.prepared,
            prefetched: session.prefetched,
            needsFinalize: session.needsFinalize,
            interrupted: session.interrupted,
            config: session.config)
    }

    @discardableResult
    public func prepare(
        sessionID: String = "default",
        mode: MiniCPMDuplexMode = .fullDuplex,
        config: MiniCPMDuplexConfig = .init()
    ) throws -> String {
        let session = getOrCreate(sessionID, mode: mode, config: config)
        guard !session.needsFinalize && !session.prefetched else {
            throw MiniCPMDuplexRuntimeError.invalidState(
                "session \(sessionID) has a pending unit; finalize or rollback first")
        }
        session.mode = mode
        session.config = config
        session.generationEpoch = interruptionRegistry.current(sessionID: sessionID)
            ?? interruptionRegistry.register(sessionID: sessionID)
        session.engine.setGenerationEpoch(session.generationEpoch)
        session.engine.setInterruptionProbe(nil)
        let prompt = try session.engine.prepare(mode: mode, config: config)
        session.prepared = true
        session.prefetched = false
        session.interrupted = false
        session.engine.setSlidingWindow(config: config)
        return prompt
    }

    @discardableResult
    public func prefill(
        _ input: MiniCPMDuplexInput,
        sessionID: String = "default"
    ) throws -> MiniCPMDuplexPrefillResult {
        let session = try requireSession(sessionID)
        guard session.prepared else { throw MiniCPMDuplexRuntimeError.invalidState("prepare must be called first") }
        guard !session.needsFinalize else { throw MiniCPMDuplexRuntimeError.invalidState("previous unit needs finalize") }
        guard !input.isEmpty else { throw MiniCPMDuplexRuntimeError.emptyInput }
        let generationEpoch = interruptionRegistry.current(sessionID: sessionID)
            ?? interruptionRegistry.register(sessionID: sessionID)
        guard interruptionRegistry.isCurrent(sessionID: sessionID, epoch: generationEpoch) else {
            // A cancel can arrive after the backend dequeues this input but
            // before the synchronous MLX prefill starts.  This is a normal
            // barge-in boundary, not a backend failure: repair the engine on
            // the actor and let the transport drop this input quietly.
            session.engine.interrupt()
            session.interrupted = true
            session.prefetched = false
            session.needsFinalize = false
            session.generationEpoch = interruptionRegistry.current(sessionID: sessionID)
                ?? generationEpoch
            return MiniCPMDuplexPrefillResult(
                accepted: false, mode: session.mode, input: input)
        }
        session.generationEpoch = generationEpoch
        session.engine.setGenerationEpoch(generationEpoch)
        session.interrupted = false
        session.engine.setInterruptionProbe { [registry = interruptionRegistry, sessionID, generationEpoch] in
            !registry.isCurrent(sessionID: sessionID, epoch: generationEpoch)
        }
        defer { session.engine.setInterruptionProbe(nil) }
        let result: MiniCPMDuplexPrefillResult
        do {
            result = try session.engine.prefill(mode: session.mode, input: input)
        } catch {
            if !interruptionRegistry.isCurrent(sessionID: sessionID, epoch: generationEpoch) {
                // Native engines roll back their KV/audio/TTS checkpoint when
                // their interruption probe trips and may report that boundary
                // as a synchronous error.  Treat it exactly like an accepted
                // false prefill so MiniCPMDemoBackend remains active for the
                // next microphone chunk instead of entering backend_error.
                session.engine.interrupt()
                session.interrupted = true
                session.prefetched = false
                session.needsFinalize = false
                session.generationEpoch = interruptionRegistry.current(sessionID: sessionID)
                    ?? generationEpoch
                return MiniCPMDuplexPrefillResult(
                    accepted: false, mode: session.mode, input: input)
            }
            throw error
        }
        guard interruptionRegistry.isCurrent(sessionID: sessionID, epoch: generationEpoch) else {
            session.engine.interrupt()
            session.interrupted = true
            session.prefetched = false
            session.needsFinalize = false
            session.generationEpoch = interruptionRegistry.current(sessionID: sessionID)
                ?? generationEpoch
            return MiniCPMDuplexPrefillResult(accepted: false, mode: session.mode, input: input)
        }
        session.prefetched = result.accepted
        return result
    }

    public func generate(sessionID: String = "default") throws -> MiniCPMDuplexOutput {
        let session = try requireSession(sessionID)
        guard session.prepared, session.prefetched else {
            throw MiniCPMDuplexRuntimeError.invalidState("prefill must be called before generate")
        }
        let generationEpoch = session.generationEpoch
        guard interruptionRegistry.isCurrent(sessionID: sessionID, epoch: generationEpoch) else {
            session.engine.interrupt()
            session.interrupted = true
            session.prefetched = false
            session.needsFinalize = false
            return MiniCPMDuplexOutput(isListen: true, needsFinalize: false, finishReason: "interrupt")
        }
        if session.interrupted {
            return MiniCPMDuplexOutput(isListen: true, needsFinalize: false, finishReason: "interrupt")
        }
        session.engine.setGenerationEpoch(generationEpoch)
        session.engine.setInterruptionProbe { [registry = interruptionRegistry, sessionID, generationEpoch] in
            !registry.isCurrent(sessionID: sessionID, epoch: generationEpoch)
        }
        defer { session.engine.setInterruptionProbe(nil) }
        let output: MiniCPMDuplexOutput
        do {
            output = try session.engine.generate(mode: session.mode, config: session.config)
        } catch {
            if !interruptionRegistry.isCurrent(sessionID: sessionID, epoch: generationEpoch) {
                session.staleOutputDiscardCount &+= 1
                session.engine.interrupt()
                session.interrupted = true
                session.prefetched = false
                session.needsFinalize = false
                return MiniCPMDuplexOutput(isListen: true, needsFinalize: false, finishReason: "interrupt")
            }
            throw error
        }
        let outputIsCurrent = interruptionRegistry.isCurrent(
            sessionID: sessionID, epoch: generationEpoch)
        if !outputIsCurrent {
            session.staleOutputDiscardCount &+= 1
        }
        let wasInterrupted = output.finishReason == "interrupt" || !outputIsCurrent
        if wasInterrupted {
            // Native engines that poll the probe already roll back their
            // model state before returning.  If a legacy engine ignored the
            // probe but the epoch changed, perform the serialized repair now;
            // never mutate KV/TTS state from the transport thread.
            if output.finishReason != "interrupt" {
                session.engine.interrupt()
            }
            session.interrupted = true
            session.prefetched = false
            session.needsFinalize = false
            return MiniCPMDuplexOutput(isListen: true, needsFinalize: false, finishReason: "interrupt")
        }
        // A listen marker is a complete duplex unit.  Commit its </unit>
        // immediately so the next microphone chunk can enter the session;
        // only spoken output keeps the deferred-finalize transaction open.
        if output.isListen && output.needsFinalize {
            try session.engine.finalize()
            session.prefetched = false
            session.needsFinalize = false
        } else {
            session.needsFinalize = output.needsFinalize
        }
        return MiniCPMDuplexOutput(
            text: output.text,
            audio: output.audio,
            isListen: output.isListen,
            endOfTurn: output.endOfTurn,
            needsFinalize: output.needsFinalize && !output.isListen,
            finishReason: output.finishReason)
    }

    public func finalize(sessionID: String = "default") throws {
        let session = try requireSession(sessionID)
        guard session.needsFinalize else { return }
        try session.engine.finalize()
        session.needsFinalize = false
        session.prefetched = false
    }

    public func rollback(sessionID: String = "default") throws {
        let session = try requireSession(sessionID)
        guard session.prefetched || session.needsFinalize else { return }
        try session.engine.rollback()
        session.needsFinalize = false
        session.prefetched = false
    }

    public func interrupt(sessionID: String = "default") async throws {
        let session = try requireSession(sessionID)
        // This defer is deliberately installed before any repair work.  The
        // timestamp describes the actor-level interrupt method's actual exit,
        // after engine rollback and epoch bookkeeping have completed, rather
        // than the earlier point at which the engine interrupt was requested.
        defer {
            session.lastCancelReturnedAtNS = DispatchTime.now().uptimeNanoseconds
        }
        session.engine.interrupt()
        let now = DispatchTime.now().uptimeNanoseconds
        session.lastCancelRequestedAtNS =
            interruptionRegistry.lastRequestAtNS(sessionID: sessionID) ?? now
        session.interrupted = true
        if session.needsFinalize || session.prefetched {
            try session.engine.rollback()
        }
        session.needsFinalize = false
        session.prefetched = false
        session.generationEpoch = interruptionRegistry.current(sessionID: sessionID)
            ?? session.generationEpoch
        session.engine.setGenerationEpoch(session.generationEpoch)
    }

    /// Return a value-only snapshot suitable for a structured backend trace.
    /// This method is actor-isolated so the engine's MLX/KV state is sampled
    /// only after any serialized interrupt/rollback repair has completed.
    public func diagnostics(
        sessionID: String = "default"
    ) throws -> MiniCPMDuplexRuntimeDiagnosticSnapshot {
        let session = try requireSession(sessionID)
        let epoch = interruptionRegistry.current(sessionID: sessionID)
            ?? session.generationEpoch
        let engine = session.engine.diagnosticSnapshot()
        return MiniCPMDuplexRuntimeDiagnosticSnapshot(
            sessionID: sessionID,
            generationEpoch: epoch,
            staleOutputDiscardCount: session.staleOutputDiscardCount,
            cancelRequestedAtNS: session.lastCancelRequestedAtNS
                ?? interruptionRegistry.lastRequestAtNS(sessionID: sessionID),
            cancelReturnedAtNS: session.lastCancelReturnedAtNS,
            engine: engine)
    }

    public func setSlidingWindow(tokens: Int?, sessionID: String = "default") throws {
        let session = try requireSession(sessionID)
        let normalized = tokens.map { max(0, $0) }
        session.config.slidingWindowTokens = normalized
        if let normalized {
            session.config.slidingWindowMode = "basic"
            session.config.basicWindowHighTokens = normalized
            session.config.basicWindowLowTokens = normalized
        } else {
            session.config.slidingWindowMode = "off"
        }
        session.engine.setSlidingWindow(config: session.config)
    }

    public func reset(sessionID: String = "default") throws {
        let session = try requireSession(sessionID)
        session.engine.reset()
        session.prepared = false
        session.prefetched = false
        session.needsFinalize = false
        session.interrupted = false
        session.generationEpoch = interruptionRegistry.current(sessionID: sessionID)
            ?? session.generationEpoch
        session.staleOutputDiscardCount = 0
        session.lastCancelRequestedAtNS = nil
        session.lastCancelReturnedAtNS = nil
        session.engine.setGenerationEpoch(session.generationEpoch)
    }

    public func close(sessionID: String = "default") throws {
        let session = try requireSession(sessionID)
        do {
            if session.needsFinalize || session.prefetched {
                try session.engine.rollback()
            }
        } catch {
            // Even when rollback reports an audio/KV replay error, release
            // the per-session mutable state before removing the actor entry.
            // Shared immutable model weights remain owned by the factory.
            session.engine.reset()
            sessions.removeValue(forKey: sessionID)
            interruptionRegistry.remove(sessionID: sessionID)
            throw error
        }
        // `reset` is the engine-level release boundary: it drops KV, audio
        // frontend tails, semantic-TTS state, and Token2Wav prompt caches.
        // Invoke it exactly once before discarding the session object.
        session.engine.reset()
        sessions.removeValue(forKey: sessionID)
        interruptionRegistry.remove(sessionID: sessionID)
    }

    private func getOrCreate(_ id: String, mode: MiniCPMDuplexMode, config: MiniCPMDuplexConfig) -> Session {
        if let session = sessions[id] { return session }
        interruptionRegistry.register(sessionID: id)
        let session = Session(id: id, engine: engineFactory(), mode: mode, config: config)
        sessions[id] = session
        return session
    }

    private func requireSession(_ id: String) throws -> Session {
        guard let session = sessions[id] else {
            throw MiniCPMDuplexRuntimeError.unknownSession(id)
        }
        return session
    }
}
