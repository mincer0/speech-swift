import Foundation

/// The Swift side of the MLX bridge.  Heavy model implementations conform to
/// this protocol; the protocol itself intentionally knows nothing about MLX or
/// Torch and can therefore be exercised with a deterministic test engine.
public protocol MiniCPMDemoBackendEngine: Sendable {
    func initialize(mode: MiniCPMBackendMode, parameters: [String: MiniCPMJSONValue]) async throws
    func prefill(mode: MiniCPMBackendMode, input: [String: MiniCPMJSONValue]) async throws
    /// Result-aware prefill used by streaming audio engines.  A short first
    /// PCM chunk may be buffered without being generation-ready (for example
    /// MiniCPM's 16,480-sample mel warm-up); callers must wait for more input
    /// instead of invoking `generate` on an empty-logits state.
    func prefillResult(mode: MiniCPMBackendMode, input: [String: MiniCPMJSONValue]) async throws -> MiniCPMDemoPrefillResult
    func generate(mode: MiniCPMBackendMode, input: [String: MiniCPMJSONValue]) async throws -> [MiniCPMDemoOutput]
    /// Accept the generated unit.  MiniCPM's runtime intentionally defers
    /// this boundary so an interrupt can still roll back a half-spoken unit;
    /// the dispatcher calls it only after a real `done`/end-of-turn marker.
    func finalize() async throws
    func interrupt() async
    func clearInterrupt() async
    func stop() async
    func reset() async
    func close() async
    /// Value-only diagnostic payload emitted with interrupt/state events.
    /// Engines without native runtime state may return an empty dictionary.
    func diagnosticSnapshot() async -> [String: MiniCPMJSONValue]
}

/// Dependency boundary for turn-based and half-duplex implementations.
/// The official Demo uses a chat/half-duplex view with a different prefill
/// and generation contract than the continuous audio/omni duplex runtime.
/// A future MLX ChatSession can conform here without flattening ordered
/// messages into the duplex input shape.
public protocol MiniCPMDemoChatEngine: MiniCPMDemoBackendEngine {}

/// Framework-neutral VAD result consumed by the half-duplex chat adapter.
/// Concrete production code can wrap ``SpeechVAD.StreamingVADProcessor`` (or
/// a CoreML-backed equivalent) without making this backend target depend on
/// the model/VAD package.  A provider is created per backend/session through
/// ``MiniCPMStreamingVADFactory`` and must retain its own streaming state.
public struct MiniCPMStreamingVADEvents: Sendable, Equatable {
    public let speechStarted: Bool
    public let speechEnded: Bool

    public init(speechStarted: Bool = false, speechEnded: Bool = false) {
        self.speechStarted = speechStarted
        self.speechEnded = speechEnded
    }
}

public protocol MiniCPMStreamingVADProvider: AnyObject {
    /// Process an arbitrary PCM packet. Providers may resample when the
    /// transport sample rate is not 16 kHz, or return no events until enough
    /// samples have accumulated for their model.
    func process(samples: [Float], sampleRate: Int) -> MiniCPMStreamingVADEvents

    /// Flush the provider at an explicit transport boundary. This is useful
    /// for providers whose state machine emits ``speechEnded`` only after a
    /// final padded frame.
    func flush(sampleRate: Int) -> MiniCPMStreamingVADEvents

    func reset()
}

/// Factory boundary intentionally lives in MiniCPMDemoBackend so production
/// can inject a SpeechVAD/CoreML implementation from the executable target
/// without introducing a target cycle here. Creation is throwing so a model
/// load failure reaches ``session.init`` instead of degrading into a provider
/// that silently waits forever for an end-of-utterance event.
public typealias MiniCPMStreamingVADFactory =
    @Sendable () throws -> any MiniCPMStreamingVADProvider

/// Short integration spellings for executables that already use the
/// ``StreamingVADProcessor`` name in their audio stack.
public typealias StreamingVADProvider = MiniCPMStreamingVADProvider
public typealias StreamingVADProviderFactory = MiniCPMStreamingVADFactory

public enum MiniCPMDemoSchedulerError: Error, LocalizedError, Sendable {
    case cancelled
    case timeout
    case queueFull
    case unknownTicket(String)

    public var errorDescription: String? {
        switch self {
        case .cancelled: return "MiniCPM inference queue ticket was cancelled"
        case .timeout: return "MiniCPM inference queue timed out"
        case .queueFull: return "MiniCPM inference queue is full"
        case .unknownTicket(let ticket): return "unknown MiniCPM queue ticket: \(ticket)"
        }
    }
}

public struct MiniCPMDemoInferenceLease: Sendable, Hashable, Equatable {
    public let ticketID: String
    public let sessionID: String

    public init(ticketID: String, sessionID: String) {
        self.ticketID = ticketID
        self.sessionID = sessionID
    }
}

public struct MiniCPMDemoQueueTicket: Sendable, Hashable, Equatable {
    public let id: String
    public let sessionID: String
    public let requestType: String
    public let enqueuedAt: Date

    public init(
        id: String,
        sessionID: String,
        requestType: String = "unknown",
        enqueuedAt: Date = Date()
    ) {
        self.id = id
        self.sessionID = sessionID
        self.requestType = requestType
        self.enqueuedAt = enqueuedAt
    }
}

public struct MiniCPMDemoQueueStatus: Sendable, Equatable {
    public let ticketID: String
    public let sessionID: String
    public let position: Int
    public let queueLength: Int
    public let estimatedWaitMS: Double
    public let requestType: String
    public let enqueuedAt: Date

    public init(
        ticketID: String,
        sessionID: String,
        position: Int,
        queueLength: Int,
        estimatedWaitMS: Double,
        requestType: String = "unknown",
        enqueuedAt: Date = Date()
    ) {
        self.ticketID = ticketID
        self.sessionID = sessionID
        self.position = position
        self.queueLength = queueLength
        self.estimatedWaitMS = estimatedWaitMS
        self.requestType = requestType
        self.enqueuedAt = enqueuedAt
    }
}

public enum MiniCPMDemoQueueAdmission: Sendable, Equatable {
    case granted(MiniCPMDemoInferenceLease, MiniCPMDemoQueueStatus)
    case queued(MiniCPMDemoQueueTicket, MiniCPMDemoQueueStatus)
}

/// Process-wide FIFO scheduler for model leases.  MLX/Metal weights may be
/// shared by multiple backend factories, but active generation is serialized
/// by default so two browser sessions cannot concurrently allocate KV/TTS
/// state or exceed unified memory.  The lease is held for a session lifetime,
/// giving full/omni streams affinity to one model worker; callers may choose a
/// higher limit only when their model factory is known to be safely shareable.
public actor MiniCPMDemoInferenceScheduler {
    private struct Entry {
        var ticket: MiniCPMDemoQueueTicket
        var continuation: CheckedContinuation<MiniCPMDemoInferenceLease, Error>?
        var readyLease: MiniCPMDemoInferenceLease?
    }

    public let maxConcurrent: Int
    public let maxQueueSize: Int
    public private(set) var serviceEstimateMS: Double
    private var active: [String: MiniCPMDemoInferenceLease] = [:]
    private var entries: [String: Entry] = [:]
    private var fifo: [String] = []
    private var counter: UInt64 = 0

    public init(
        maxConcurrent: Int = 1,
        maxQueueSize: Int = 1_000,
        serviceEstimateMS: Double = 1_000
    ) {
        self.maxConcurrent = max(1, maxConcurrent)
        self.maxQueueSize = max(0, maxQueueSize)
        self.serviceEstimateMS = max(0, serviceEstimateMS)
    }

    public func enqueue(sessionID: String, requestType: String = "unknown") throws -> MiniCPMDemoQueueAdmission {
        if let existing = active.values.first(where: { $0.sessionID == sessionID }) {
            let status = MiniCPMDemoQueueStatus(
                ticketID: existing.ticketID,
                sessionID: sessionID,
                position: 0,
                queueLength: fifo.count,
                estimatedWaitMS: 0)
            return .granted(existing, status)
        }
        if let pending = entries.values.first(where: { $0.ticket.sessionID == sessionID }) {
            return .queued(pending.ticket, status(for: pending.ticket.id))
        }
        let ticket = MiniCPMDemoQueueTicket(
            id: "ticket_\(String(format: "%08llu", nextCounter()))",
            sessionID: sessionID,
            requestType: requestType)
        if active.count < maxConcurrent {
            let lease = MiniCPMDemoInferenceLease(ticketID: ticket.id, sessionID: sessionID)
            active[ticket.id] = lease
            return .granted(lease, MiniCPMDemoQueueStatus(
                ticketID: ticket.id,
                sessionID: sessionID,
                position: 0,
                queueLength: fifo.count,
                estimatedWaitMS: 0))
        }
        guard fifo.count < maxQueueSize else {
            throw MiniCPMDemoSchedulerError.queueFull
        }
        let entry = Entry(ticket: ticket, continuation: nil, readyLease: nil)
        entries[ticket.id] = entry
        fifo.append(ticket.id)
        return .queued(ticket, status(for: ticket.id))
    }

    public func wait(_ ticket: MiniCPMDemoQueueTicket, timeoutMS: Int? = nil) async throws -> MiniCPMDemoInferenceLease {
        guard let entry = entries[ticket.id], entry.ticket == ticket else {
            throw MiniCPMDemoSchedulerError.unknownTicket(ticket.id)
        }
        if let ready = entry.readyLease {
            entries.removeValue(forKey: ticket.id)
            return ready
        }
        // A socket can disappear while its connection task is waiting here.
        // Tie task cancellation to the queue entry so a cancelled waiter does
        // not leave a continuation (and therefore a session) retained until
        // the timeout task eventually fires.
        return try await withTaskCancellationHandler(operation: {
            try await withCheckedThrowingContinuation { continuation in
                // Do not capture/mutate the local `entry` across the
                // continuation's @Sendable closure.  Swift 6 diagnoses that
                // pattern as a concurrent mutation; the actor dictionary is
                // the authoritative state and is already isolated here.
                guard var current = entries[ticket.id], current.ticket == ticket else {
                    continuation.resume(throwing: MiniCPMDemoSchedulerError.unknownTicket(ticket.id))
                    return
                }
                current.continuation = continuation
                entries[ticket.id] = current
                if let timeoutMS, timeoutMS > 0 {
                    Task {
                        do {
                            try await Task.sleep(nanoseconds: UInt64(timeoutMS) * 1_000_000)
                            self.timeout(ticket.id)
                        } catch {
                            // Cancellation is expected when a lease is
                            // granted or the WebSocket closes first.
                        }
                    }
                }
            }
        }, onCancel: {
            Task { await self.cancel(ticketID: ticket.id) }
        })
    }

    public func release(_ lease: MiniCPMDemoInferenceLease) {
        guard active.removeValue(forKey: lease.ticketID) != nil else { return }
        pump()
    }

    /// Force-release a session lease during HTTP control-channel close. The
    /// WebSocket owner normally calls ``release(_:)`` in its defer path; this
    /// method covers the case where a client closes over the unary endpoint
    /// before the socket loop observes EOF.
    public func release(sessionID: String) {
        let leases = active.values.filter { $0.sessionID == sessionID }
        for lease in leases { _ = active.removeValue(forKey: lease.ticketID) }
        cancel(sessionID: sessionID)
        pump()
    }

    public func cancel(ticketID: String) {
        guard let entry = entries.removeValue(forKey: ticketID) else { return }
        fifo.removeAll { $0 == ticketID }
        if entry.readyLease != nil {
            active.removeValue(forKey: ticketID)
        }
        entry.continuation?.resume(throwing: MiniCPMDemoSchedulerError.cancelled)
        pump()
    }

    public func cancel(sessionID: String) {
        let ticketIDs = entries.values
            .filter { $0.ticket.sessionID == sessionID }
            .map { $0.ticket.id }
        for ticketID in ticketIDs { cancel(ticketID: ticketID) }
    }

    public func status(ticketID: String) throws -> MiniCPMDemoQueueStatus {
        guard let entry = entries[ticketID] else {
            if let lease = active[ticketID] {
                return MiniCPMDemoQueueStatus(ticketID: ticketID, sessionID: lease.sessionID, position: 0, queueLength: fifo.count, estimatedWaitMS: 0)
            }
            throw MiniCPMDemoSchedulerError.unknownTicket(ticketID)
        }
        return status(for: entry.ticket.id)
    }

    public func snapshot() -> (active: Int, queued: Int, maxConcurrent: Int) {
        (active.count, fifo.count, maxConcurrent)
    }

    /// Update the queue's coarse service-time estimate after the persistent
    /// ETA store learns a better baseline/EMA.  Queue positions are derived
    /// from this value at read time, so no stale ticket rows need rewriting.
    public func updateServiceEstimateMS(_ value: Double) {
        serviceEstimateMS = max(0, value.isFinite ? value : 0)
    }

    public func queuedStatuses() -> [MiniCPMDemoQueueStatus] {
        fifo.compactMap { ticketID in
            guard entries[ticketID] != nil else { return nil }
            return status(for: ticketID)
        }
    }

    private func timeout(_ ticketID: String) {
        guard let entry = entries[ticketID], entry.continuation != nil else { return }
        entries.removeValue(forKey: ticketID)
        fifo.removeAll { $0 == ticketID }
        entry.continuation?.resume(throwing: MiniCPMDemoSchedulerError.timeout)
        pump()
    }

    private func pump() {
        while active.count < maxConcurrent, !fifo.isEmpty {
            let ticketID = fifo.removeFirst()
            guard var entry = entries[ticketID] else { continue }
            let lease = MiniCPMDemoInferenceLease(ticketID: ticketID, sessionID: entry.ticket.sessionID)
            active[ticketID] = lease
            if let continuation = entry.continuation {
                entries.removeValue(forKey: ticketID)
                continuation.resume(returning: lease)
            } else {
                entry.readyLease = lease
                entries[ticketID] = entry
            }
        }
    }

    private func status(for ticketID: String) -> MiniCPMDemoQueueStatus {
        guard let entry = entries[ticketID] else {
            let lease = active[ticketID]
            return MiniCPMDemoQueueStatus(ticketID: ticketID, sessionID: lease?.sessionID ?? "", position: 0, queueLength: fifo.count, estimatedWaitMS: 0)
        }
        let index = fifo.firstIndex(of: ticketID)
        let position = index.map { $0 + 1 } ?? 0
        return MiniCPMDemoQueueStatus(
            ticketID: ticketID,
            sessionID: entry.ticket.sessionID,
            position: position,
            queueLength: fifo.count,
            estimatedWaitMS: Double(max(0, position - 1)) * serviceEstimateMS,
            requestType: entry.ticket.requestType,
            enqueuedAt: entry.ticket.enqueuedAt)
    }

    private func nextCounter() -> UInt64 {
        counter += 1
        return counter
    }
}

public extension MiniCPMDemoBackendEngine {
    func prefillResult(mode: MiniCPMBackendMode, input: [String: MiniCPMJSONValue]) async throws -> MiniCPMDemoPrefillResult {
        try await prefill(mode: mode, input: input)
        return MiniCPMDemoPrefillResult()
    }
    func finalize() async throws {}
    func interrupt() async {}
    func clearInterrupt() async {}
    func stop() async {}
    func reset() async {}
    func close() async {}
    func diagnosticSnapshot() async -> [String: MiniCPMJSONValue] { [:] }
}

public struct MiniCPMDemoPrefillResult: Sendable, Equatable {
    public var accepted: Bool
    public var needsMoreAudio: Bool

    public init(accepted: Bool = true, needsMoreAudio: Bool = false) {
        self.accepted = accepted
        self.needsMoreAudio = needsMoreAudio
    }
}

/// A no-op engine is useful for protocol smoke tests and keeps the server
/// bootable before a real MLX model is injected.  It emits one listen event in
/// duplex modes and echoes the last text field in turn mode.
public struct MiniCPMDemoNoopEngine: MiniCPMDemoBackendEngine {
    public init() {}

    public func initialize(mode: MiniCPMBackendMode, parameters: [String: MiniCPMJSONValue]) async throws {}
    public func prefill(mode: MiniCPMBackendMode, input: [String: MiniCPMJSONValue]) async throws {}

    public func generate(mode: MiniCPMBackendMode, input: [String: MiniCPMJSONValue]) async throws -> [MiniCPMDemoOutput] {
        if mode == .turnBased || mode == .halfDuplex {
            let text = input["text"]?.stringValue ?? ""
            return [MiniCPMDemoOutput(kind: "text", text: text.isEmpty ? "" : text, done: true, reason: "turn_end")]
        }
        return [MiniCPMDemoOutput(kind: "listen", done: false, reason: "awaiting_engine")]
    }
}

/// Actor that owns one active MiniCPM Demo session.  It serializes WebSocket
/// messages and model calls, preventing overlapping generate calls from
/// corrupting decoder/TTS state.
public actor MiniCPMDemoBackend {
    public enum State: String, Sendable {
        case new
        case initialized
        case active
        case paused
        case closed
        case error
    }

    private let engine: any MiniCPMDemoBackendEngine
    private var state: State = .new
    private var mode: MiniCPMBackendMode = .fullDuplex
    private var sessionID: String?
    private var eventCounter: UInt64 = 0
    private var responseCounter: UInt64 = 0
    /// The official backend keeps one response open while a spoken turn is
    /// streamed in multiple one-second units.  It is cleared only when the
    /// model emits a listen/end-of-turn marker (or a turn-based response.done).
    private var activeResponseID: String?
    private var inputCounter: UInt64 = 0
    private var pendingFinalize = false
    /// Keep the normalized init envelope so reset can rebuild the runtime
    /// with the same mode/configuration instead of leaving it half-cleared.
    private var lastInitializeParameters: [String: MiniCPMJSONValue] = [:]
    private var lastMetrics: [String: MiniCPMJSONValue] = ["backend": .string("swift_mlx")]
    private let recorder: (any MiniCPMDemoEventRecorder)?
    /// Audio seconds already sent (as deltas) into the currently open spoken
    /// response, and the wall-clock timestamp of its first audio delta.
    /// A browser plays at 1x realtime, so `sent − elapsed` approximates the
    /// client's buffered audio — the quantity a barge-in can never exceed and
    /// the cushion that absorbs generation spikes.  Unlike counting synthetic
    /// units against client packets, this bound holds no matter how the units
    /// were produced.
    private var responseAudioSentSeconds: Double = 0
    private var responseFirstAudioAtNS: UInt64?
    /// Consecutive driver-generated units without an intervening real input.
    /// Secondary backstop for responses that emit no (or tiny) audio, where
    /// the playback-based bound alone would never engage.
    private var syntheticStreak = 0
    /// Playback-cushion bound in seconds: large enough to absorb per-unit
    /// generation spikes (semantic TTS / Token2Wav bursts, the first-chunk
    /// force flush) and to keep the player buffered, small enough that
    /// interrupting a response discards at most a few seconds of audio.
    static let maxBufferedAudioSeconds: Double = 3.0
    static let maxConsecutiveSyntheticUnits = 4

    public init(
        engine: any MiniCPMDemoBackendEngine = MiniCPMDemoNoopEngine(),
        recorder: (any MiniCPMDemoEventRecorder)? = nil
    ) {
        self.engine = engine
        self.recorder = recorder
    }

    public func health() -> [String: MiniCPMJSONValue] {
        [
            // The process is healthy before a WebSocket session is opened;
            // `worker_status` retains the precise lifecycle state.
            "status": .string(state == .new || state == .active || state == .initialized || state == .paused ? "ready" : state.rawValue),
            "backend": .string("swift_mlx"),
            "worker_status": .string(state.rawValue),
            "active_session_id": sessionID.map(MiniCPMJSONValue.string) ?? .null,
        ]
    }

    public func handle(_ request: MiniCPMDemoRequest) async -> [MiniCPMDemoEvent] {
        do {
            switch request.type {
            case "session.init":
                return try await initialize(request.payloadObject)
            case "input.append":
                return try await append(request.input?.objectValue ?? request.payloadObject, messageID: request.messageID)
            case "session.close", "close":
                return await close(reason: request.reason ?? "client_closed")
            case "control", "session.control":
                return await control(request.payloadObject)
            default:
                return [errorEvent(code: "invalid_input", message: "unsupported message type: \(request.type)")]
            }
        } catch {
            state = .error
            activeResponseID = nil
            responseAudioSentSeconds = 0
            responseFirstAudioAtNS = nil
            syntheticStreak = 0
            pendingFinalize = false
            await engine.stop()
            return [errorEvent(code: "backend_error", message: error.localizedDescription)]
        }
    }

    public func initialize(_ parameters: [String: MiniCPMJSONValue]) async throws -> [MiniCPMDemoEvent] {
        guard state == .new || state == .closed || state == .error else {
            throw MiniCPMDemoBackendError.invalidState("backend is already \(state.rawValue)")
        }
        let parameters = MiniCPMDemoWireDecoder.normalizeObject(parameters)
        let modeValue = parameters["mode"]?.stringValue
        mode = try MiniCPMBackendMode(rawOrAlias: modeValue)
        lastInitializeParameters = parameters
        pendingFinalize = false
        activeResponseID = nil
        responseAudioSentSeconds = 0
        responseFirstAudioAtNS = nil
            syntheticStreak = 0
        sessionID = "sess_\(UUID().uuidString.replacingOccurrences(of: "-", with: "").prefix(12))"
        state = .initialized
        try await engine.initialize(mode: mode, parameters: parameters)
        let id = sessionID
        let initialized = emit("backend.initialized", payload: [
            "backend": .string("swift_mlx"),
            "mode": .string(mode.rawValue),
            "state": .string(state.rawValue),
            "capabilities": .object(capabilities(for: mode)),
            "defaults": parameters["defaults"] ?? .object([:]),
        ])
        let created = emit("session.created", payload: ["mode": .string(mode.rawValue), "metrics": .object(lastMetrics)])
        state = .active
        let active = emit("backend.state", payload: ["state": .string(state.rawValue), "capabilities": .object(capabilities(for: mode))])
        // Keep the local `id` alive through the event construction in case an
        // injected engine changes actor state while initializing.
        _ = id
        return [initialized, created, active]
    }

    /// Commit the generated unit after its complete event batch has been
    /// written to the transport.  Keeping this separate from `handle` lets a
    /// disconnect/write failure roll the runtime unit back instead of
    /// accepting audio that never reached the client.
    public func commitPendingResponse() async throws {
        guard pendingFinalize else { return }
        try await engine.finalize()
        pendingFinalize = false
    }

    /// Discard a generated unit when the transport cannot write its events.
    public func rollbackPendingResponse() async {
        guard pendingFinalize else { return }
        await engine.stop()
        pendingFinalize = false
    }

    private func append(_ input: [String: MiniCPMJSONValue], messageID: String?) async throws -> [MiniCPMDemoEvent] {
        guard state == .initialized || state == .active else {
            throw MiniCPMDemoBackendError.invalidState("backend is \(state.rawValue)")
        }
        guard !pendingFinalize else {
            throw MiniCPMDemoBackendError.invalidState("previous response is awaiting transport acceptance")
        }
        inputCounter += 1
        var decodedInput = MiniCPMDemoWireDecoder.normalizeObject(input)
        let inputID = messageID
            ?? decodedInput["input_id"]?.stringValue
            ?? "in_\(String(format: "%08llu", inputCounter))"
        // A real client packet ends any synthetic streak.
        syntheticStreak = 0
        // Browser clients only mark zero-filled silence with
        // `continuation_tick` while their own model-state heuristic says
        // "speaking", so plenty of silent packets arrive unmarked and each
        // would otherwise pay a full 350-440ms audio-encoder prefill.  The
        // engine itself still validates every precondition (turn open, no
        // frames/text, all-zero samples), so deciding it here purely saves
        // that work; nothing about the client flag is trusted.
        if isDuplexMode, Self.isAllZeroAudioInput(decodedInput) {
            decodedInput["continuation_tick"] = .bool(true)
        }
        let requestedResponseID = decodedInput["response_id"]?.stringValue
        let wallClockStart = DispatchTime.now().uptimeNanoseconds
        var events: [MiniCPMDemoEvent] = [emit("input.accepted", payload: ["input_id": .string(inputID)], inputID: inputID)]
        do {
            // Every mode has an explicit prefill/generate boundary.  In
            // particular, half duplex never falls back to the full-duplex
            // listen/speak coordinator.
            let prefill = try await engine.prefillResult(mode: mode, input: decodedInput)
            let prefillMS = elapsedMS(since: wallClockStart)
            lastMetrics["prefill_ms"] = .number(prefillMS)
            events.append(emit("input.committed", payload: ["input_id": .string(inputID), "metrics": .object(lastMetrics)], inputID: inputID))
            guard prefill.accepted else {
                // Streaming audio frontends may consume a short chunk only
                // to fill their convolution/mel context.  It is not a model
                // turn and must never reach generate/no-generation-logits.
                if isDuplexMode {
                    events.append(emit("response.output.delta", payload: [
                        "kind": .string("listen"),
                        "reason": .string(prefill.needsMoreAudio ? "awaiting_audio" : "prefill_buffered"),
                    ], inputID: inputID))
                }
                return events
            }
            let outputs = try await engine.generate(mode: mode, input: decodedInput)
            // A streaming/full-duplex chunk without `done` is intentionally
            // left open.  Fabricating `response.done` here used to turn the
            // first acknowledgement (e.g. “好的”) into an end-of-turn and
            // prevented the model from continuing to speak.
            events.append(contentsOf: try emitOutputs(
                outputs,
                inputID: inputID,
                requestedResponseID: requestedResponseID,
                wallClockStart: wallClockStart))
            return events
        } catch {
            state = .error
            activeResponseID = nil
            responseAudioSentSeconds = 0
            responseFirstAudioAtNS = nil
            syntheticStreak = 0
            pendingFinalize = false
            await engine.stop()
            events.append(errorEvent(code: "engine_error", message: error.localizedDescription, inputID: inputID))
            return events
        }
    }

    /// Convert engine outputs into wire events, tracking the open response
    /// envelope exactly like the pinned official backend: one response stays
    /// open across multiple spoken one-second units and is closed by a listen
    /// marker (full duplex) or a turn-based `response.done`.
    private func emitOutputs(
        _ outputs: [MiniCPMDemoOutput],
        inputID: String,
        requestedResponseID: String?,
        wallClockStart: UInt64
    ) throws -> [MiniCPMDemoEvent] {
        var events: [MiniCPMDemoEvent] = []
        var responseID = activeResponseID ?? requestedResponseID
        var textParts: [String] = []
        var audioParts: [EncodedAudioPart] = []
        for output in outputs {
            guard output.kind == "listen" || output.kind == "text" || output.kind == "audio" else {
                throw MiniCPMDemoBackendError.engine(
                    "engine returned unsupported output kind: \(output.kind)")
            }
            if output.kind == "listen" {
                let metrics = mergedMetrics(output.metrics, wallClockStart: wallClockStart)
                // The duplex engine already produced a one-second silence
                // payload for a listen boundary ("this keeps downstream
                // playback and capture clocks aligned"). Emit it: dropping it
                // here leaves a one-second hole in the client's playback
                // timeline, which is exactly what the Ahead/buffer indicator
                // reports as a dry queue even when the model is legitimately
                // listening. `silent` lets a client that prefers the old
                // behaviour skip the block without guessing from the samples.
                if let audio = output.audio {
                    audioParts.append(EncodedAudioPart(
                        format: output.format,
                        sampleRate: output.sampleRate,
                        data: audio))
                    events.append(emit("response.output.delta", payload: [
                        "kind": .string("audio"),
                        "audio": .string(audio),
                        "format": .string(output.format),
                        "sample_rate": .number(Double(output.sampleRate)),
                        "silent": .bool(true),
                        "metrics": .object(metrics),
                    ], responseID: responseID, inputID: inputID))
                }
                events.append(emit("response.output.delta", payload: ["kind": .string("listen"), "reason": .string(output.reason ?? "model_listen"), "metrics": .object(metrics)], responseID: responseID, inputID: inputID))
                if output.needsFinalize { pendingFinalize = true }
                // A listen marker closes the currently spoken response.
                responseID = nil
                activeResponseID = nil
                responseAudioSentSeconds = 0
                responseFirstAudioAtNS = nil
            syntheticStreak = 0
                continue
            }
            if responseID == nil {
                responseID = "resp_\(String(format: "%08llu", nextResponse()))"
                activeResponseID = responseID
                responseAudioSentSeconds = 0
                responseFirstAudioAtNS = nil
            syntheticStreak = 0
                events.append(emit("response.started", payload: ["response_id": .string(responseID!)], responseID: responseID, inputID: inputID))
            }
            if let text = output.text, !text.isEmpty {
                textParts.append(text)
                events.append(emit("response.output.delta", payload: ["kind": .string("text"), "text": .string(text), "metrics": .object(mergedMetrics(output.metrics, wallClockStart: wallClockStart))], responseID: responseID, inputID: inputID))
            }
            if let audio = output.audio {
                audioParts.append(EncodedAudioPart(
                    format: output.format,
                    sampleRate: output.sampleRate,
                    data: audio))
                if responseFirstAudioAtNS == nil {
                    responseFirstAudioAtNS = DispatchTime.now().uptimeNanoseconds
                }
                responseAudioSentSeconds += Double(Self.base64SampleCount(audio))
                    / Double(max(1, output.sampleRate))
                events.append(emit("response.output.delta", payload: ["kind": .string("audio"), "audio": .string(audio), "format": .string(output.format), "sample_rate": .number(Double(output.sampleRate)), "metrics": .object(mergedMetrics(output.metrics, wallClockStart: wallClockStart))], responseID: responseID, inputID: inputID))
            }
            if output.needsFinalize { pendingFinalize = true }
            if output.done, !isDuplexMode {
                events.append(emit("response.done", payload: ["text": .string(textParts.joined()), "audio": combinedAudio(audioParts).map(MiniCPMJSONValue.string) ?? .null, "reason": .string(output.reason ?? "turn_end"), "metrics": .object(mergedMetrics(output.metrics, wallClockStart: wallClockStart))], responseID: responseID, inputID: inputID))
                pendingFinalize = pendingFinalize || output.needsFinalize
                activeResponseID = nil
                responseAudioSentSeconds = 0
                responseFirstAudioAtNS = nil
            syntheticStreak = 0
                responseID = nil
            } else if output.done, isDuplexMode,
                      output.reason == "turn_end" || output.reason == "end_of_turn" {
                // Full duplex uses the listen marker as its turn boundary;
                // response.done is a turn-based-only event.
                if let responseID {
                    events.append(emit("response.output.delta", payload: ["kind": .string("listen"), "reason": .string("turn_end")], responseID: responseID, inputID: inputID))
                }
                activeResponseID = nil
                responseAudioSentSeconds = 0
                responseFirstAudioAtNS = nil
            syntheticStreak = 0
                responseID = nil
            }
        }
        return events
    }

    /// Generate the next duplex unit without waiting for the transport's next
    /// `input.append`.  The one-input-one-unit worker loop otherwise ties the
    /// output audio rate to the client's packet cadence: a client that streams
    /// one-second packets (or waits for per-packet acknowledgements) makes a
    /// fully model-driven answer arrive slower than realtime, which renders as
    /// choppy playback.  While a spoken response is open and no real packet is
    /// queued, the transport worker calls this method to advance the turn with
    /// a zero-filled continuation tick — the native engine handles that without
    /// re-running the audio encoder, so output streams at model speed and the
    /// client builds a playback buffer instead of running dry.  Returns nil
    /// when there is nothing to continue (no open spoken response, inactive
    /// state, cap reached) or the engine rejected/ended the turn.
    func continueResponse() async -> [MiniCPMDemoEvent]? {
        guard state == .active, isDuplexMode, activeResponseID != nil, !pendingFinalize else {
            return nil
        }
        if let firstAt = responseFirstAudioAtNS {
            // The browser plays at 1x realtime starting from the first audio
            // delta's arrival, so anything sent beyond elapsed wall time is
            // un-played client buffer.  Stop driving once that cushion is
            // full; real packets (1s audio per 1s wall) keep it topped up
            // while absorbing generation spikes.
            let elapsedSeconds = Double(DispatchTime.now().uptimeNanoseconds &- firstAt) / 1_000_000_000.0
            let buffered = responseAudioSentSeconds - elapsedSeconds
            guard buffered < Self.maxBufferedAudioSeconds else { return nil }
        }
        guard syntheticStreak < Self.maxConsecutiveSyntheticUnits else {
            return nil
        }
        inputCounter += 1
        let inputID = "cont_\(String(format: "%08llu", inputCounter))"
        let syntheticInput = MiniCPMDemoWireDecoder.normalizeObject([
            "input_id": .string(inputID),
            "audio": .string(Self.encodeFloat32([Float](repeating: 0, count: 16_000))),
            "format": .string("pcm_f32le"),
            "sample_rate": .number(16_000),
            "continuation_tick": .bool(true),
            "speech_active": .bool(false),
        ])
        let wallClockStart = DispatchTime.now().uptimeNanoseconds
        do {
            let prefill = try await engine.prefillResult(mode: mode, input: syntheticInput)
            guard prefill.accepted else { return nil }
            let outputs = try await engine.generate(mode: mode, input: syntheticInput)
            syntheticStreak += 1
            return try emitOutputs(
                outputs,
                inputID: inputID,
                requestedResponseID: activeResponseID,
                wallClockStart: wallClockStart)
        } catch {
            state = .error
            activeResponseID = nil
            responseAudioSentSeconds = 0
            responseFirstAudioAtNS = nil
            syntheticStreak = 0
            pendingFinalize = false
            await engine.stop()
            return [errorEvent(code: "engine_error", message: error.localizedDescription, inputID: inputID)]
        }
    }

    /// Detect a silent Float32 PCM packet so the backend can offer the
    /// engine a continuation tick without trusting the client's own flag.
    /// Only plain `pcm_f32le` payloads are inspected; anything else (WAV
    /// containers, 16-bit PCM, lists) stays on the regular encode path.
    static func isAllZeroAudioInput(_ input: [String: MiniCPMJSONValue]) -> Bool {
        guard input["text"] == nil, input["video_frames"] == nil,
              input["frame_base64_list"] == nil, input["messages"] == nil else {
            return false
        }
        let format = input["format"]?.stringValue?.lowercased()
        if let format, !format.isEmpty, !format.contains("f32") {
            return false
        }
        // `normalizeObject` already decodes base64 PCM into a numeric array
        // at the wire boundary, so both representations can reach here.
        switch input["audio"] ?? input["audio_waveform"] {
        case .array(let values):
            guard !values.isEmpty else { return false }
            for value in values {
                guard let number = value.numberValue, number == 0, number.isFinite else {
                    return false
                }
            }
            return true
        case .string(let encoded):
            guard !encoded.isEmpty,
                  encoded.count <= 8 * 1024 * 1024,
                  let data = Data(base64Encoded: encoded),
                  !data.isEmpty,
                  data.count.isMultiple(of: 4) else {
                return false
            }
            return data.withUnsafeBytes { raw in
                let samples = raw.bindMemory(to: Float.self)
                for sample in samples where sample != 0 || !sample.isFinite {
                    return false
                }
                return true
            }
        default:
            return false
        }
    }

    /// Approximate decoded Float32 sample count of a base64 PCM payload.
    static func base64SampleCount(_ encoded: String) -> Int {
        let padding = encoded.hasSuffix("==") ? 2 : (encoded.hasSuffix("=") ? 1 : 0)
        return max(0, encoded.count / 4 * 3 - padding) / MemoryLayout<Float>.stride
    }

    private static func encodeFloat32(_ samples: [Float]) -> String {
        var data = Data(capacity: samples.count * MemoryLayout<Float>.stride)
        for sample in samples {
            var bits = sample.bitPattern.littleEndian
            withUnsafeBytes(of: &bits) { data.append(contentsOf: $0) }
        }
        return data.base64EncodedString()
    }

    private func control(_ payload: [String: MiniCPMJSONValue]) async -> [MiniCPMDemoEvent] {
        let command = payload["type"]?.stringValue ?? payload["command"]?.stringValue ?? payload["action"]?.stringValue ?? ""
        switch command {
        case "response.cancel", "legacy.interrupt", "backend.interrupt", "interrupt", "break":
            await engine.interrupt()
            pendingFinalize = false
            activeResponseID = nil
            responseAudioSentSeconds = 0
            responseFirstAudioAtNS = nil
            syntheticStreak = 0
            var payload: [String: MiniCPMJSONValue] = [
                "state": .string(state.rawValue),
                "reason": .string("interrupt"),
            ]
            let trace = await engine.diagnosticSnapshot()
            if !trace.isEmpty {
                // Keep both names for old clients (metrics) and new trace
                // tooling. The object is scalar-only and safe to persist.
                payload["trace"] = .object(trace)
                payload["metrics"] = .object(trace)
            }
            return [emit("backend.state", payload: payload)]
        case "backend.clear_interrupt", "session.clear_interrupt", "clear_interrupt":
            await engine.clearInterrupt()
            return [emit("backend.state", payload: ["state": .string(state.rawValue), "reason": .string("interrupt_cleared")])]
        case "backend.reset", "session.reset", "reset":
            do {
                await engine.reset()
                // `reset` clears KV/TTS state in the native engine.  It does
                // not imply that the model is initialized again, so replay
                // the exact session.init envelope before accepting input.
                try await engine.initialize(mode: mode, parameters: lastInitializeParameters)
                pendingFinalize = false
                activeResponseID = nil
                responseAudioSentSeconds = 0
                responseFirstAudioAtNS = nil
            syntheticStreak = 0
                state = .active
                return [emit("backend.state", payload: ["state": .string(state.rawValue), "reason": .string("reset")])]
            } catch {
                state = .error
                pendingFinalize = false
                activeResponseID = nil
                responseAudioSentSeconds = 0
                responseFirstAudioAtNS = nil
            syntheticStreak = 0
                await engine.stop()
                return [errorEvent(code: "backend_error", message: "reset failed: \(error.localizedDescription)")]
            }
        case "backend.pause", "session.pause", "pause":
            if state == .active { state = .paused }
            return [emit("backend.state", payload: ["state": .string(state.rawValue), "reason": .string("paused")])]
        case "backend.resume", "session.resume", "resume":
            if state == .paused { state = .active }
            return [emit("backend.state", payload: ["state": .string(state.rawValue), "reason": .string("resumed")])]
        default:
            return [errorEvent(code: "unsupported", message: "unknown backend control: \(command)")]
        }
    }

    private func close(reason: String) async -> [MiniCPMDemoEvent] {
        guard state != .closed else { return [] }
        await engine.stop()
        pendingFinalize = false
        await engine.close()
        activeResponseID = nil
        responseAudioSentSeconds = 0
        responseFirstAudioAtNS = nil
            syntheticStreak = 0
        state = .closed
        let events = [
            emit("backend.closed", payload: ["reason": .string(reason)]),
            emit("session.closed", payload: ["reason": .string(reason)]),
        ]
        recorder?.finish()
        return events
    }

    private func nextResponse() -> UInt64 {
        responseCounter += 1
        return responseCounter
    }

    private func emit(_ type: String, payload: [String: MiniCPMJSONValue], responseID: String? = nil, inputID: String? = nil) -> MiniCPMDemoEvent {
        eventCounter += 1
        let event = MiniCPMDemoEvent(eventID: "evt_\(String(format: "%08llu", eventCounter))", type: type, payload: payload, sessionID: sessionID, responseID: responseID, inputID: inputID)
        recorder?.record(event)
        return event
    }

    private func errorEvent(code: String, message: String, inputID: String? = nil) -> MiniCPMDemoEvent {
        emit("error", payload: ["code": .string(code), "message": .string(message), "scope": .string("backend")], inputID: inputID)
    }

    private func capabilities(for mode: MiniCPMBackendMode) -> [String: MiniCPMJSONValue] {
        let duplex = isDuplexMode
        let capabilities: [String: MiniCPMJSONValue] = [
            "mode": .string(mode.rawValue),
            "text_input": .bool(true),
            "audio_input": .bool(true),
            "image_input": .bool(true),
            // Audio full-duplex is not a video protocol.  Only the omni mode
            // advertises video frames; claiming video support for `audio`
            // made clients send frame payloads to the wrong frontend.
            "video_input": .bool(mode == .omni),
            "text_output": .bool(true),
            "audio_output": .bool(true),
            "streaming_output": .bool(true),
            "multi_turn": .bool(true),
            "interrupt": .bool(duplex),
            "rollback": .bool(!duplex),
            "autonomous_listen_speak": .bool(duplex),
            "tts_reference": .bool(true),
            "metrics": .bool(true),
        ]
        return capabilities
    }

    private var isDuplexMode: Bool {
        mode == .fullDuplex || mode == .omni
    }

    private struct EncodedAudioPart {
        var format: String
        var sampleRate: Int
        var data: String
    }

    /// Join Float32 PCM chunks for a turn-based `response.done` payload.
    /// Streaming deltas stay untouched; this only prevents the non-streaming
    /// aggregate from silently dropping every chunk except the last one.
    private func combinedAudio(_ parts: [EncodedAudioPart]) -> String? {
        guard !parts.isEmpty else { return nil }
        guard parts.allSatisfy({ $0.format.lowercased().contains("f32") }) else {
            return parts.last?.data
        }
        var bytes = Data()
        for part in parts {
            guard let data = Data(base64Encoded: part.data), data.count.isMultiple(of: 4) else {
                return parts.last?.data
            }
            bytes.append(data)
        }
        return bytes.isEmpty ? nil : bytes.base64EncodedString()
    }

    private func elapsedMS(since start: UInt64) -> Double {
        Double(DispatchTime.now().uptimeNanoseconds - start) / 1_000_000.0
    }

    private func mergedMetrics(
        _ output: [String: MiniCPMJSONValue],
        wallClockStart: UInt64
    ) -> [String: MiniCPMJSONValue] {
        var metrics = lastMetrics
        // Stage timings and transport padding describe exactly one generated
        // unit. Carrying an omitted value forward makes a continuation look
        // as if it re-ran the audio encoder; carrying padding forward can also
        // trim valid samples from the next audio chunk.
        for key in [
            "audio_encoder_ms", "audio_bridge_ms", "llm_prefill_ms",
            "llm_decode_ms", "semantic_tts_ms", "token2wav_ms",
            "flow_ms", "hift_ms", "audio_padding_samples",
            "continuation_only",
        ] {
            metrics.removeValue(forKey: key)
        }
        metrics.merge(output) { _, newer in newer }
        metrics["wall_clock_ms"] = .number(elapsedMS(since: wallClockStart))
        lastMetrics = metrics
        return metrics
    }
}
