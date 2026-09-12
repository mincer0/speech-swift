import Foundation
import Hummingbird
import HummingbirdWebSocket
import NIOCore

/// Registry used by the official backend-client control channel.  WebSocket
/// carries only init/push/events; `POST /sessions/:id/close` calls this actor
/// and waits for the backend to release its runtime state before returning.
public actor MiniCPMDemoBackendRegistry {
    private var sessions: [String: MiniCPMDemoBackend] = [:]
    private var schedulerSessionIDs: [String: String] = [:]
    private let scheduler: MiniCPMDemoInferenceScheduler?

    public init(scheduler: MiniCPMDemoInferenceScheduler? = nil) {
        self.scheduler = scheduler
    }

    public func register(sessionID: String, backend: MiniCPMDemoBackend, schedulerSessionID: String? = nil) {
        sessions[sessionID] = backend
        if let schedulerSessionID { schedulerSessionIDs[sessionID] = schedulerSessionID }
    }

    @discardableResult
    public func close(sessionID: String, reason: String) async -> Bool {
        guard let backend = sessions.removeValue(forKey: sessionID) else { return false }
        let schedulerID = schedulerSessionIDs.removeValue(forKey: sessionID) ?? sessionID
        await scheduler?.release(sessionID: schedulerID)
        _ = await backend.handle(MiniCPMDemoRequest(type: "session.close", reason: reason))
        return true
    }

    public func forget(sessionID: String) async {
        sessions.removeValue(forKey: sessionID)
        let schedulerID = schedulerSessionIDs.removeValue(forKey: sessionID) ?? sessionID
        await scheduler?.release(sessionID: schedulerID)
    }

    public func count() -> Int { sessions.count }
}

/// Serve the MiniCPM Demo ``/backend`` wire protocol over a Hummingbird
/// WebSocket.  The route owner can mount this handler alongside the existing
/// OpenAI Realtime endpoint without coupling either protocol's session state.
public func handleMiniCPMDemoBackendWS(
    inbound: WebSocketInboundStream,
    outbound: WebSocketOutboundWriter,
    backend: MiniCPMDemoBackend,
    registry: MiniCPMDemoBackendRegistry? = nil,
    scheduler: MiniCPMDemoInferenceScheduler? = nil,
    strictWire: Bool = true,
    maxMessageSize: Int = 128 * 1024 * 1024,
    queueTimeoutMS: Int? = 120_000,
    recordingStore: MiniCPMDemoSessionStore? = nil,
    // Queue lifecycle events belong to the public Gateway transport.  The
    // pinned `/backend` worker protocol is deliberately only init/push/events
    // (close is HTTP unary), so keep this off by default for direct worker
    // clients.  Callers embedding a combined Gateway may opt in explicitly.
    emitQueueEvents: Bool = false
) async throws {
    try await handleMiniCPMDemoWS(
        inbound: inbound,
        outbound: outbound,
        backend: backend,
        registry: registry,
        scheduler: scheduler,
        strictWire: strictWire,
        maxMessageSize: maxMessageSize,
        queueTimeoutMS: queueTimeoutMS,
        recordingStore: recordingStore,
        emitQueueEvents: emitQueueEvents,
        realtimeMode: nil,
        allowRealtimeControls: false,
        allowSessionClose: false)
}

/// Serve the public Gateway/OpenAI-Realtime compatible transport.  The
/// upstream Gateway maps `chat|video|audio` to `turn_based|omni|full_duplex`
/// and accepts a client `session.close`; control messages are transport-level
/// commands and never enter the model context.  The fixed `/backend` worker
/// protocol above remains stricter (close is still HTTP-unary there).
public func handleMiniCPMDemoRealtimeWS(
    inbound: WebSocketInboundStream,
    outbound: WebSocketOutboundWriter,
    backend: MiniCPMDemoBackend,
    mode: MiniCPMRealtimeMode = .video,
    registry: MiniCPMDemoBackendRegistry? = nil,
    scheduler: MiniCPMDemoInferenceScheduler? = nil,
    strictWire: Bool = true,
    maxMessageSize: Int = 128 * 1024 * 1024,
    queueTimeoutMS: Int? = 120_000,
    emitQueueEvents: Bool = true,
    recordingStore: MiniCPMDemoSessionStore? = nil
) async throws {
    try await handleMiniCPMDemoWS(
        inbound: inbound,
        outbound: outbound,
        backend: backend,
        registry: registry,
        scheduler: scheduler,
        strictWire: strictWire,
        maxMessageSize: maxMessageSize,
        queueTimeoutMS: queueTimeoutMS,
        recordingStore: recordingStore,
        emitQueueEvents: emitQueueEvents,
        realtimeMode: mode,
        allowRealtimeControls: true,
        allowSessionClose: true)
}

/// The WebSocket writer is deliberately owned by one task.  Calling
/// `WebSocketOutboundWriter.write` from both the receive loop and a generation
/// task can interleave frames when the NIO write suspends, which is especially
/// visible during barge-in.  Producers enqueue a frame and await its write
/// acknowledgement; only the writer task touches the underlying channel.
///
/// The channel is bounded and tracks every acknowledgement continuation.  A
/// failed write/finish resumes all pending producers, avoiding the old leak
/// where a buffered frame could leave a generation task suspended forever.
private struct MiniCPMDemoOutboundMessage: Sendable {
    let id: UInt64
    let text: String
}

private final class MiniCPMDemoOutboundChannel: @unchecked Sendable {
    private static let capacity = 128
    let stream: AsyncStream<MiniCPMDemoOutboundMessage>
    private let continuation: AsyncStream<MiniCPMDemoOutboundMessage>.Continuation
    private let lock = NSLock()
    private var nextID: UInt64 = 0
    private var pending: [UInt64: CheckedContinuation<Void, Error>] = [:]
    private var terminalError: Error?
    private var finished = false

    init() {
        var continuation: AsyncStream<MiniCPMDemoOutboundMessage>.Continuation!
        self.stream = AsyncStream(bufferingPolicy: .bufferingOldest(Self.capacity)) {
            continuation = $0
        }
        self.continuation = continuation
    }

    func send(_ text: String) async throws {
        try await withCheckedThrowingContinuation { (completion: CheckedContinuation<Void, Error>) in
            lock.lock()
            if finished {
                let error = terminalError ?? CancellationError()
                lock.unlock()
                completion.resume(throwing: error)
                return
            }
            nextID &+= 1
            let id = nextID
            pending[id] = completion
            let result = continuation.yield(.init(id: id, text: text))
            switch result {
            case .enqueued:
                lock.unlock()
            case .dropped(let dropped):
                let droppedCompletion = pending.removeValue(forKey: dropped.id)
                let currentCompletion = pending.removeValue(forKey: id)
                lock.unlock()
                let error = MiniCPMDemoBackendError.engine("outbound queue is full")
                droppedCompletion?.resume(throwing: error)
                currentCompletion?.resume(throwing: error)
            case .terminated:
                let currentCompletion = pending.removeValue(forKey: id)
                let error = terminalError ?? CancellationError()
                lock.unlock()
                currentCompletion?.resume(throwing: error)
            @unknown default:
                let currentCompletion = pending.removeValue(forKey: id)
                lock.unlock()
                currentCompletion?.resume(throwing: CancellationError())
            }
        }
    }

    func complete(id: UInt64, error: Error? = nil) {
        lock.lock()
        let completion = pending.removeValue(forKey: id)
        lock.unlock()
        if let error {
            completion?.resume(throwing: error)
        } else {
            completion?.resume()
        }
    }

    func finish(error: Error? = nil) {
        lock.lock()
        guard !finished else {
            lock.unlock()
            return
        }
        finished = true
        terminalError = error
        let completions = pending.values
        pending.removeAll(keepingCapacity: false)
        lock.unlock()
        continuation.finish()
        let failure = error ?? CancellationError()
        for completion in completions {
            completion.resume(throwing: failure)
        }
    }
}

private actor MiniCPMDemoGenerationEpoch {
    private var value: UInt64 = 0
    private var terminated = false

    func current() -> UInt64 { value }

    @discardableResult
    func invalidate() -> UInt64 {
        value &+= 1
        return value
    }

    func close() {
        terminated = true
        value &+= 1
    }

    func isCurrent(_ candidate: UInt64) -> Bool {
        !terminated && candidate == value
    }
}

struct MiniCPMDemoQueuedInput: Sendable {
    let request: MiniCPMDemoRequest
    let epoch: UInt64
    /// The encoded, canonical request size charged against the transport
    /// backlog.  Charging the normalized envelope (rather than the original
    /// base64 text) keeps the bound meaningful after audio decoding expands a
    /// packet into a Float32 array.
    let encodedBytes: Int
}

enum MiniCPMDemoInputQueueError: Error, LocalizedError, Sendable, Equatable {
    case full(itemCount: Int, byteCount: Int, itemLimit: Int, byteLimit: Int)
    case itemTooLarge(size: Int, byteLimit: Int)
    case closed

    var errorDescription: String? {
        switch self {
        case let .full(itemCount, byteCount, itemLimit, byteLimit):
            return "input queue is full (items=\(itemCount)/\(itemLimit), bytes=\(byteCount)/\(byteLimit))"
        case let .itemTooLarge(size, byteLimit):
            // A client that sends one oversized video/audio frame receives a
            // structured transport error.  The 1009 hint maps to the
            // WebSocket message-too-big close code used by Gateway clients.
            return "input queue item is too large (bytes=\(size), limit=\(byteLimit), close_code=1009)"
        case .closed:
            return "input queue is closed"
        }
    }
}

/// A non-blocking, bounded FIFO for realtime input packets.  AsyncStream's
/// buffering policy only limits item count and cannot account for the decoded
/// Float32/audio or frame bytes retained by each request.  This wrapper keeps
/// both budgets under one short lock and releases the byte charge as soon as
/// the generation worker dequeues an item.
final class MiniCPMDemoInputQueue: @unchecked Sendable {
    static let defaultItemCapacity = 8
    static let defaultByteCapacity = 32 * 1024 * 1024

    let itemCapacity: Int
    let byteCapacity: Int
    let stream: AsyncStream<MiniCPMDemoQueuedInput>
    private let continuation: AsyncStream<MiniCPMDemoQueuedInput>.Continuation
    private let lock = NSLock()
    private var itemCount = 0
    private var byteCount = 0
    private var finished = false

    init(
        itemCapacity: Int = MiniCPMDemoInputQueue.defaultItemCapacity,
        byteCapacity: Int = MiniCPMDemoInputQueue.defaultByteCapacity
    ) {
        self.itemCapacity = max(1, itemCapacity)
        self.byteCapacity = max(1, byteCapacity)
        var continuation: AsyncStream<MiniCPMDemoQueuedInput>.Continuation!
        self.stream = AsyncStream(bufferingPolicy: .bufferingOldest(self.itemCapacity)) {
            continuation = $0
        }
        self.continuation = continuation
    }

    var queuedItemCount: Int {
        lock.lock()
        defer { lock.unlock() }
        return itemCount
    }

    var queuedByteCount: Int {
        lock.lock()
        defer { lock.unlock() }
        return byteCount
    }

    func enqueue(_ item: MiniCPMDemoQueuedInput) throws {
        let size = max(0, item.encodedBytes)
        lock.lock()
        guard !finished else {
            lock.unlock()
            throw MiniCPMDemoInputQueueError.closed
        }
        guard size <= byteCapacity else {
            lock.unlock()
            throw MiniCPMDemoInputQueueError.itemTooLarge(size: size, byteLimit: byteCapacity)
        }
        guard itemCount < itemCapacity, byteCount <= byteCapacity - size else {
            let error = MiniCPMDemoInputQueueError.full(
                itemCount: itemCount,
                byteCount: byteCount,
                itemLimit: itemCapacity,
                byteLimit: byteCapacity)
            lock.unlock()
            throw error
        }

        // Keep the lock through yield + accounting.  A consumer can be
        // resumed synchronously by AsyncStream, but it cannot release this
        // item until it reacquires the same lock, so the counters stay exact.
        let result = continuation.yield(item)
        switch result {
        case .enqueued:
            itemCount += 1
            byteCount += size
            lock.unlock()
        case .dropped:
            // The stream should never drop because the explicit item budget is
            // at least as strict as its buffering policy.  Treat an unexpected
            // drop as a hard error rather than silently losing client audio.
            lock.unlock()
            throw MiniCPMDemoInputQueueError.full(
                itemCount: itemCount,
                byteCount: byteCount,
                itemLimit: itemCapacity,
                byteLimit: byteCapacity)
        case .terminated:
            finished = true
            lock.unlock()
            throw MiniCPMDemoInputQueueError.closed
        @unknown default:
            lock.unlock()
            throw MiniCPMDemoInputQueueError.closed
        }
    }

    /// Called exactly once by the worker after it receives an item from
    /// ``stream``.  The item remains owned by the worker/model, but no longer
    /// counts against the receive backlog budget.
    func release(_ item: MiniCPMDemoQueuedInput) {
        let size = max(0, item.encodedBytes)
        lock.lock()
        if itemCount > 0 { itemCount -= 1 }
        byteCount = max(0, byteCount - size)
        lock.unlock()
    }

    func finish() {
        lock.lock()
        guard !finished else {
            lock.unlock()
            return
        }
        finished = true
        lock.unlock()
        continuation.finish()
    }
}

/// Recording is created only after session.init has produced a stable ID, but
/// the reader may receive frames before that.  This actor keeps the pending
/// canonical frames and makes recorder access safe from the generation task.
private actor MiniCPMDemoRecorderBox {
    private static let maxPendingFrames = 128
    private static let maxPendingBytes = 4 * 1024 * 1024
    private var recorder: MiniCPMDemoWireRecordingSession?
    private var pendingInbound: [MiniCPMJSONValue] = []
    private var pendingInboundBytes = 0

    func attach(_ recorder: MiniCPMDemoWireRecordingSession) async {
        self.recorder = recorder
        for frame in pendingInbound {
            await recorder.record(direction: .up, frame: frame)
        }
        pendingInbound.removeAll(keepingCapacity: true)
        pendingInboundBytes = 0
    }

    func recordInbound(_ frame: MiniCPMJSONValue) async {
        if let recorder {
            await recorder.record(direction: .up, frame: frame)
        } else {
            let frameBytes = (try? MiniCPMDemoPersistence.encoder().encode(frame).count) ?? 0
            guard pendingInbound.count < Self.maxPendingFrames,
                  pendingInboundBytes + frameBytes <= Self.maxPendingBytes else {
                // Recording is diagnostics-only; cap pre-init buffering so a
                // client cannot retain arbitrary media before session.init.
                return
            }
            pendingInbound.append(frame)
            pendingInboundBytes += frameBytes
        }
    }

    func recordOutbound(_ frame: MiniCPMJSONValue) async {
        await recorder?.record(direction: .down, frame: frame)
    }

    func close(reason: String) async {
        await recorder?.close(reason: reason)
    }
}

private func handleMiniCPMDemoWS(
    inbound: WebSocketInboundStream,
    outbound: WebSocketOutboundWriter,
    backend: MiniCPMDemoBackend,
    registry: MiniCPMDemoBackendRegistry?,
    scheduler: MiniCPMDemoInferenceScheduler?,
    strictWire: Bool,
    maxMessageSize: Int,
    queueTimeoutMS: Int?,
    recordingStore: MiniCPMDemoSessionStore?,
    emitQueueEvents: Bool,
    realtimeMode: MiniCPMRealtimeMode?,
    allowRealtimeControls: Bool,
    allowSessionClose: Bool
) async throws {
    let decoder = JSONDecoder()
    let outputChannel = MiniCPMDemoOutboundChannel()
    // This task is the sole owner of the NIO writer.  It is explicitly
    // awaited during teardown, so a disconnect cannot leave an unstructured
    // writer task retaining the WebSocket channel.
    let outputTask = Task<Void, Never> {
        for await message in outputChannel.stream {
            do {
                try await outbound.write(.text(message.text))
                outputChannel.complete(id: message.id)
            } catch {
                outputChannel.complete(id: message.id, error: error)
                outputChannel.finish(error: error)
                break
            }
        }
    }
    var registeredSessionID: String?
    var didInit = false
    let schedulerSessionID = "conn_\(UUID().uuidString.replacingOccurrences(of: "-", with: "").prefix(12))"
    var lease: MiniCPMDemoInferenceLease?
    // Public Gateway clients wait for queue_done before sending session.init;
    // the admission block below emits it before the receive loop. The strict
    // `/backend` channel disables queue events and processes its first init
    // in-place after a silent lease acquisition.
    // Keep the ticket while `wait` is suspended. If the transport task is
    // cancelled before a lease is granted, the defer path must remove this
    // queued entry as well; otherwise it retains a continuation until the
    // global queue timeout and makes `/queue` report a phantom session.
    var pendingTicket: MiniCPMDemoQueueTicket?
    var forceListenNext = false
    let recorderBox = MiniCPMDemoRecorderBox()
    let generationEpoch = MiniCPMDemoGenerationEpoch()
    // Audio/video chunks are FIFO but bounded by both packet count and the
    // encoded canonical request bytes.  If generation falls behind, reject
    // the new frame explicitly instead of retaining unbounded PCM payloads.
    let inputQueue = MiniCPMDemoInputQueue()
    var terminalReason = "transport_closed"

    // Acquire the process-wide model lease before consuming session.init. The
    // public Gateway additionally exposes queue events at this boundary;
    // strict `/backend` clients wait silently when emitQueueEvents is false.
    // Doing admission before reading init mirrors the pinned Gateway → worker
    // ordering and prevents an optimistic init from being processed twice.
    if let scheduler {
        do {
            let admission = try await scheduler.enqueue(sessionID: schedulerSessionID)
            switch admission {
            case .granted(let granted, let status):
                lease = granted
                if emitQueueEvents {
                    try await writeMiniCPMDemoQueueEvent(
                        outbound: outputChannel, type: "session.queue_done",
                        ticketID: status.ticketID, position: 0,
                        queueLength: status.queueLength,
                        estimatedWaitMS: 0, sessionID: schedulerSessionID)
                }
            case .queued(let ticket, let status):
                pendingTicket = ticket
                if emitQueueEvents {
                    try await writeMiniCPMDemoQueueEvent(
                        outbound: outputChannel, type: "session.queued",
                        ticketID: ticket.id, position: status.position,
                        queueLength: status.queueLength,
                        estimatedWaitMS: status.estimatedWaitMS,
                        sessionID: schedulerSessionID)
                    try await writeMiniCPMDemoQueueEvent(
                        outbound: outputChannel, type: "session.queue_update",
                        ticketID: ticket.id, position: status.position,
                        queueLength: status.queueLength,
                        estimatedWaitMS: status.estimatedWaitMS,
                        sessionID: schedulerSessionID)
                }
                lease = try await scheduler.wait(ticket, timeoutMS: queueTimeoutMS)
                pendingTicket = nil
                if emitQueueEvents {
                    try await writeMiniCPMDemoQueueEvent(
                        outbound: outputChannel, type: "session.queue_done",
                        ticketID: ticket.id, position: 0,
                        queueLength: 0, estimatedWaitMS: 0,
                        sessionID: schedulerSessionID)
                }
            }
        } catch {
            if allowRealtimeControls {
                try? await writeMiniCPMDemoRealtimeError(
                    outbound: outputChannel, sessionID: nil,
                    message: error.localizedDescription)
            } else {
                try? await writeMiniCPMDemoError(
                    outbound: outputChannel, message: error.localizedDescription)
            }
            if let pendingTicket {
                await scheduler.cancel(ticketID: pendingTicket.id)
            }
            if let lease { await scheduler.release(lease) }
            outputChannel.finish()
            _ = await outputTask.value
            return
        }
    }

    // One FIFO generation worker is enough to preserve model ordering while
    // the receive loop remains free to service interrupt/reset/close frames.
    let generationTask: Task<Void, Error> = Task {
        // Drive the open spoken response with synthetic continuation ticks
        // while no real packet is queued.  Without this, one engine unit is
        // only ever produced per client packet, so the output audio rate is
        // capped by the client's packet cadence (choppy playback whenever the
        // model can speak faster than packets arrive).  Real queued inputs
        // always take priority: the loop yields back as soon as the queue is
        // non-empty so live audio is prefilled in order.
        func driveContinuation(for queued: MiniCPMDemoQueuedInput) async throws {
            while await generationEpoch.isCurrent(queued.epoch) {
                if inputQueue.queuedItemCount > 0 { return }
                guard let events = await backend.continueResponse() else { return }
                if let error = events.first(where: { $0.type == "error" }) {
                    let message = error.payload["message"]?.stringValue ?? "backend error"
                    if allowRealtimeControls {
                        try await writeMiniCPMDemoRealtimeError(
                            outbound: outputChannel,
                            sessionID: registeredSessionID,
                            message: message)
                    } else {
                        try await writeMiniCPMDemoError(outbound: outputChannel, message: message)
                    }
                    throw MiniCPMDemoBackendError.engine(message)
                }
                try await emitMiniCPMDemoEvents(
                    events,
                    outbound: outputChannel,
                    recorder: recorderBox,
                    allowRealtimeControls: allowRealtimeControls,
                    epoch: queued.epoch,
                    generationEpoch: generationEpoch)
                if await generationEpoch.isCurrent(queued.epoch) {
                    try await backend.commitPendingResponse()
                } else {
                    await backend.rollbackPendingResponse()
                    return
                }
            }
        }
        do {
            for await queued in inputQueue.stream {
                inputQueue.release(queued)
                try Task.checkCancellation()
                guard await generationEpoch.isCurrent(queued.epoch) else { continue }
                let events = await backend.handle(queued.request)
                guard await generationEpoch.isCurrent(queued.epoch) else {
                    await backend.rollbackPendingResponse()
                    continue
                }
                if let error = events.first(where: { $0.type == "error" }) {
                    let message = error.payload["message"]?.stringValue ?? "backend error"
                    if allowRealtimeControls {
                        try await writeMiniCPMDemoRealtimeError(
                            outbound: outputChannel,
                            sessionID: registeredSessionID,
                            message: message)
                    } else {
                        try await writeMiniCPMDemoError(outbound: outputChannel, message: message)
                    }
                    throw MiniCPMDemoBackendError.engine(message)
                }
                try await emitMiniCPMDemoEvents(
                    events,
                    outbound: outputChannel,
                    recorder: recorderBox,
                    allowRealtimeControls: allowRealtimeControls,
                    epoch: queued.epoch,
                    generationEpoch: generationEpoch)
                if await generationEpoch.isCurrent(queued.epoch) {
                    try await backend.commitPendingResponse()
                } else {
                    await backend.rollbackPendingResponse()
                    continue
                }
                try await driveContinuation(for: queued)
            }
        } catch is CancellationError {
            // Normal transport teardown cancels the worker after interrupting
            // the engine.  Do not turn that expected path into a wire error.
        } catch {
            // The receive loop runs independently from generation so controls
            // can interrupt an in-flight model call.  If generation itself
            // fails, close the FIFO immediately: the next client input then
            // tears down the transport instead of accumulating behind a dead
            // worker and returning backpressure forever.
            inputQueue.finish()
            throw error
        }
    }

    var backendClosed = false
    do {
        for try await message in inbound.messages(maxSize: maxMessageSize) {
            guard case .text(let text) = message else {
                throw MiniCPMDemoBackendError.invalidMessage("binary WebSocket frames are not accepted")
            }
            guard let data = text.data(using: .utf8) else {
                throw MiniCPMDemoBackendError.invalidMessage("message is not UTF-8")
            }
            var request = try decoder.decode(MiniCPMDemoRequest.self, from: data)
            var forceListenControlEvent: Bool?
            var consumedForceListen = false
            if let realtimeMode, allowRealtimeControls {
                let translation = try translateRealtimeRequest(
                    request, defaultMode: realtimeMode.backendMode,
                    didInit: didInit, forceListenNext: forceListenNext,
                    allowSessionClose: allowSessionClose)
                request = translation.request
                forceListenNext = translation.forceListenNext
                forceListenControlEvent = translation.forceListenControlEvent
                consumedForceListen = translation.consumedForceListen
            }
            guard request.type == "session.init"
                || request.type == "input.append"
                || (allowRealtimeControls && request.type == "control")
                || (allowSessionClose && request.type == "session.close")
                || forceListenControlEvent != nil else {
                throw MiniCPMDemoBackendError.invalidMessage(
                    allowRealtimeControls
                        ? "unsupported realtime message type: \(request.type)"
                        : "unsupported backend WebSocket message type: \(request.type); use HTTP close")
            }
            guard request.type == "session.close" || didInit == (request.type != "session.init") else {
                throw MiniCPMDemoBackendError.invalidMessage(
                    didInit ? "session.init may only be the first WebSocket message" : "first WebSocket message must be session.init")
            }
            guard request.type != "session.close" || didInit else {
                throw MiniCPMDemoBackendError.invalidState("session.close requires an initialized realtime session")
            }
            if let forceListenControlEvent {
                guard didInit else {
                    throw MiniCPMDemoBackendError.invalidState("force_listen requires an initialized realtime session")
                }
                try await writeMiniCPMDemoControlEvent(
                    outbound: outputChannel, recorder: recorderBox,
                    sessionID: registeredSessionID, enabled: forceListenControlEvent)
                continue
            }
            if strictWire {
                switch request.type {
                case "session.init":
                    guard let payload = request.payload?.objectValue ?? request.input?.objectValue else {
                        throw MiniCPMDemoBackendError.invalidMessage("session.init payload must be a JSON object")
                    }
                    request.payload = .object(try MiniCPMDemoWireDecoder.normalizeObjectStrict(payload))
                    request.input = nil
                case "input.append":
                    guard let input = request.input?.objectValue ?? request.payload?.objectValue else {
                        throw MiniCPMDemoBackendError.invalidMessage("input.append input must be a JSON object")
                    }
                    request.input = .object(try MiniCPMDemoWireDecoder.normalizeObjectStrict(input))
                    request.payload = nil
                default: break
                }
            }

            // Record the translated/strict canonical envelope, not an alias
            // frame that the backend will never actually see.
            let canonicalFrame = try MiniCPMDemoWireRecordingSupport.requestValue(request)
            await recorderBox.recordInbound(canonicalFrame)

            if request.type == "session.init" {
                didInit = true
                let events = await backend.handle(request)
                if let sessionID = events.first(where: { $0.type == "session.created" })?.sessionID {
                    registeredSessionID = sessionID
                    if let recordingStore {
                        let mode = request.payloadObject["mode"]?.stringValue
                            ?? realtimeMode?.backendMode.rawValue ?? "full_duplex"
                        let metadata = MiniCPMDemoSessionMetadata(
                            sessionID: sessionID, mode: mode,
                            appType: realtimeMode?.rawValue ?? mode,
                            worker: ["id": .string("local_mlx")],
                            source: ["transport": .string(realtimeMode == nil ? "backend" : "realtime")])
                        if let recorder = try? await MiniCPMDemoWireRecordingSession(
                            store: recordingStore, metadata: metadata) {
                            await recorderBox.attach(recorder)
                        }
                    }
                    if let registry {
                        await registry.register(sessionID: sessionID, backend: backend,
                                                schedulerSessionID: schedulerSessionID)
                    }
                }
                if let error = events.first(where: { $0.type == "error" }) {
                    throw MiniCPMDemoBackendError.engine(error.payload["message"]?.stringValue ?? "backend error")
                }
                try await emitMiniCPMDemoEvents(
                    events, outbound: outputChannel, recorder: recorderBox,
                    allowRealtimeControls: allowRealtimeControls)
                continue
            }

            if request.type == "input.append" {
                let epoch = await generationEpoch.current()
                let encodedBytes = try MiniCPMDemoWireRecordingSupport
                    .requestEncodedByteCount(request)
                do {
                    try inputQueue.enqueue(.init(
                        request: request, epoch: epoch, encodedBytes: encodedBytes))
                } catch let error as MiniCPMDemoInputQueueError {
                    if allowRealtimeControls,
                       case .full = error {
                        try await writeMiniCPMDemoRealtimeBackpressure(
                            outbound: outputChannel,
                            recorder: recorderBox,
                            sessionID: registeredSessionID,
                            inputID: request.messageID
                                ?? request.payloadObject["input_id"]?.stringValue,
                            queuedItems: inputQueue.queuedItemCount,
                            retryAfterMS: 350)
                        continue
                    }
                    throw MiniCPMDemoBackendError.invalidMessage(error.localizedDescription)
                }
                // A force-listen latch is consumed by the first queued input.
                // The native prefill itself remains responsible for deciding
                // whether that input needs another audio window.
                if consumedForceListen { forceListenNext = false }
                continue
            }

            if request.type == "control" {
                let command = request.payloadObject["type"]?.stringValue
                    ?? request.payloadObject["command"]?.stringValue
                    ?? request.payloadObject["action"]?.stringValue ?? ""
                if ["response.cancel", "legacy.interrupt", "backend.interrupt",
                    "interrupt", "break", "reset", "session.reset",
                    "backend.reset"].contains(command) {
                    await generationEpoch.invalidate()
                }
                let events = await backend.handle(request)
                if let error = events.first(where: { $0.type == "error" }) {
                    throw MiniCPMDemoBackendError.engine(error.payload["message"]?.stringValue ?? "backend error")
                }
                try await emitMiniCPMDemoEvents(
                    events, outbound: outputChannel, recorder: recorderBox,
                    allowRealtimeControls: allowRealtimeControls)
                continue
            }

            if request.type == "session.close" {
                await generationEpoch.close()
                inputQueue.finish()
                let events = await backend.handle(request)
                try await emitMiniCPMDemoEvents(
                    events, outbound: outputChannel, recorder: recorderBox,
                    allowRealtimeControls: allowRealtimeControls)
                terminalReason = request.reason ?? "client_closed"
                backendClosed = true
                break
            }
        }
    } catch {
        terminalReason = "backend_error"
        await generationEpoch.close()
        await backend.rollbackPendingResponse()
        if allowRealtimeControls {
            try? await writeMiniCPMDemoRealtimeError(
                outbound: outputChannel, sessionID: registeredSessionID,
                message: error.localizedDescription)
        } else {
            try? await writeMiniCPMDemoError(outbound: outputChannel, message: error.localizedDescription)
        }
    }

    // Structured teardown: interrupt/close the engine first so a blocked
    // generate observes cancellation, then await the FIFO worker before
    // releasing scheduler/registry/recording ownership.
    if !backendClosed, didInit {
        await generationEpoch.close()
        inputQueue.finish()
        _ = await backend.handle(MiniCPMDemoRequest(
            type: "control",
            payload: .object(["type": .string("response.cancel")])))
        _ = await backend.handle(MiniCPMDemoRequest(type: "session.close", reason: terminalReason))
        backendClosed = true
    }
    generationTask.cancel()
    _ = try? await generationTask.value
    await recorderBox.close(reason: terminalReason)
    if let lease, let scheduler { await scheduler.release(lease) }
    if let pendingTicket, let scheduler { await scheduler.cancel(ticketID: pendingTicket.id) }
    if let registeredSessionID, let registry { await registry.forget(sessionID: registeredSessionID) }
    outputChannel.finish()
    _ = await outputTask.value
}

private func isOfficialMiniCPMDemoEvent(_ type: String) -> Bool {
    switch type {
    case "session.created", "input.committed", "response.output.delta", "response.done", "session.closed":
        return true
    default:
        // `input.committed` is the public completion acknowledgement for an
        // input that legitimately produces no model delta.  The earlier
        // `input.accepted`, backend initialization, response.started and
        // backend.state events remain internal diagnostics.
        return false
    }
}

struct MiniCPMDemoRealtimeTranslation {
    var request: MiniCPMDemoRequest
    var forceListenNext: Bool
    var forceListenControlEvent: Bool?
    /// True when the current input consumed a previously latched
    /// force-listen request. The server clears or preserves that latch only
    /// after observing the backend result.
    var consumedForceListen: Bool
}

/// Translate the small set of public Realtime aliases into the backend
/// envelope.  `force_listen` is deliberately represented as a one-input
/// latch: the native engine exposes this as an input override, not a mutable
/// model setting.  The returned control event is emitted only after the latch
/// is actually updated, so an acknowledgement cannot claim unsupported work.
func translateRealtimeRequest(
    _ original: MiniCPMDemoRequest,
    defaultMode: MiniCPMBackendMode,
    didInit: Bool,
    forceListenNext: Bool,
    allowSessionClose: Bool
) throws -> MiniCPMDemoRealtimeTranslation {
    var request = original
    var nextForceListen = forceListenNext
    var controlEvent: Bool?
    var consumedForceListen = false

    switch original.type {
    case "session.init":
        var payload = original.payloadObject
        if payload["mode"] == nil {
            payload["mode"] = .string(defaultMode.rawValue)
        }
        request.payload = .object(payload)
        request.input = nil

    case "input.append", "input_audio_buffer.append", "audio_chunk":
        var input = original.input?.objectValue ?? original.payloadObject
        if input["audio"] == nil {
            input["audio"] = input["audio_base64"] ?? input["audio_data"]
        }
        if input["video_frames"] == nil {
            input["video_frames"] = input["frame_base64_list"]
        }
        consumedForceListen = nextForceListen
        if nextForceListen, input["force_listen"] == nil {
            input["force_listen"] = .bool(true)
        }
        // Do not clear the latch yet. The backend may only buffer a short
        // audio window; the post-handle path keeps it latched until a real
        // generate call is reached.
        request = MiniCPMDemoRequest(
            type: "input.append",
            input: .object(input),
            messageID: original.messageID,
            reason: original.reason)

    case "response.cancel", "session.interrupt", "interrupt", "break":
        request = MiniCPMDemoRequest(
            type: "control",
            payload: .object(["type": .string("response.cancel")]),
            messageID: original.messageID,
            reason: original.reason)

    case "session.pause", "pause":
        request = MiniCPMDemoRequest(
            type: "control",
            payload: .object(["type": .string("pause")]),
            messageID: original.messageID,
            reason: original.reason)

    case "session.resume", "resume":
        request = MiniCPMDemoRequest(
            type: "control",
            payload: .object(["type": .string("resume")]),
            messageID: original.messageID,
            reason: original.reason)

    case "session.force_listen", "force_listen", "force-listen":
        guard didInit else {
            throw MiniCPMDemoBackendError.invalidState(
                "force_listen requires an initialized realtime session")
        }
        let payload = original.payloadObject
        let enabled = payload["enabled"]?.boolValue
            ?? payload["force_listen"]?.boolValue
            ?? payload["forceListen"]?.boolValue
            ?? true
        nextForceListen = enabled
        controlEvent = enabled
        // No backend call is needed: this is now a concrete next-input latch.
        request = MiniCPMDemoRequest(type: "__realtime_force_listen__")

    case "session.close", "close":
        guard allowSessionClose else {
            throw MiniCPMDemoBackendError.invalidMessage(
                "session.close uses the HTTP control channel for /backend")
        }
        request = MiniCPMDemoRequest(
            type: "session.close",
            reason: original.reason
                ?? original.payloadObject["reason"]?.stringValue)

    case "control", "session.control":
        // Keep the backend's own command aliases intact. Unknown commands are
        // rejected by MiniCPMDemoBackend instead of being acknowledged here.
        break

    default:
        break
    }

    return MiniCPMDemoRealtimeTranslation(
        request: request,
        forceListenNext: nextForceListen,
        forceListenControlEvent: controlEvent,
        consumedForceListen: consumedForceListen)
}

func isAwaitingAudioEvent(_ event: MiniCPMDemoEvent) -> Bool {
    guard event.type == "response.output.delta",
          event.kind == "listen" else { return false }
    let reason = event.reason?.lowercased()
        ?? event.payload["reason"]?.stringValue?.lowercased()
        ?? ""
    return reason == "awaiting_audio" || reason == "prefill_buffered"
}

private func emitMiniCPMDemoEvents(
    _ events: [MiniCPMDemoEvent],
    outbound: MiniCPMDemoOutboundChannel,
    recorder: MiniCPMDemoRecorderBox,
    allowRealtimeControls: Bool,
    epoch: UInt64? = nil,
    generationEpoch: MiniCPMDemoGenerationEpoch? = nil
) async throws {
    for event in events {
        if let epoch, let generationEpoch,
           !(await generationEpoch.isCurrent(epoch)) {
            // Barge-in/reset invalidates the entire transaction.  Do not
            // record or write any later text/audio from that old response.
            return
        }
        if allowRealtimeControls, event.type == "backend.state" {
            try await writeMiniCPMDemoStateEvent(outbound: outbound, recorder: recorder, event: event)
            continue
        }
        guard isOfficialMiniCPMDemoEvent(event.type) else { continue }
        let encoded = try encodeOfficialMiniCPMDemoEvent(event)
        try await outbound.send(encoded)
        if let value = try? MiniCPMDemoWireRecordingSupport.jsonValue(encoded) {
            await recorder.recordOutbound(value)
        }
    }
}

private func writeMiniCPMDemoControlEvent(
    outbound: MiniCPMDemoOutboundChannel,
    recorder: MiniCPMDemoRecorderBox,
    sessionID: String?,
    enabled: Bool
) async throws {
    var event: [String: MiniCPMJSONValue] = [
        "type": .string("session.state"),
        "state": .string("active"),
        "control": .string("force_listen"),
        "force_listen": .bool(enabled),
        "scope": .string("next_input"),
        "server_send_ts": .number(Date().timeIntervalSince1970),
    ]
    if let sessionID { event["session_id"] = .string(sessionID) }
    let value = MiniCPMJSONValue.object(event)
    try await outbound.send(try JSONEncoder().encode(value).asString())
    await recorder.recordOutbound(value)
}

private func writeMiniCPMDemoStateEvent(
    outbound: MiniCPMDemoOutboundChannel,
    recorder: MiniCPMDemoRecorderBox,
    event: MiniCPMDemoEvent
) async throws {
    var object: [String: MiniCPMJSONValue] = [
        "type": .string("session.state"),
        "state": event.payload["state"] ?? .string("active"),
        "reason": event.payload["reason"] ?? .null,
        "server_send_ts": .number(Date().timeIntervalSince1970),
    ]
    if let sessionID = event.sessionID { object["session_id"] = .string(sessionID) }
    if let metrics = event.payload["metrics"] { object["metrics"] = metrics }
    let value = MiniCPMJSONValue.object(object)
    try await outbound.send(try JSONEncoder().encode(value).asString())
    await recorder.recordOutbound(value)
}

private func writeMiniCPMDemoRealtimeError(
    outbound: MiniCPMDemoOutboundChannel,
    sessionID: String?,
    message: String
) async throws {
    var error: [String: MiniCPMJSONValue] = [
        "code": .string("session_failed"),
        "message": .string(message),
        "type": .string("server_error"),
    ]
    var event: [String: MiniCPMJSONValue] = [
        "type": .string("error"),
        "error": .object(error),
        "server_send_ts": .number(Date().timeIntervalSince1970),
    ]
    if let sessionID {
        event["session_id"] = .string(sessionID)
        error["session_id"] = .string(sessionID)
    }
    // Keep the nested error object complete after adding the optional session
    // identity (the assignment above intentionally avoids duplicating fields
    // in the top-level event).
    event["error"] = .object(error)
    try await outbound.send(try JSONEncoder().encode(MiniCPMJSONValue.object(event)).asString())
    if let sessionID {
        let closed: MiniCPMJSONValue = .object([
            "type": .string("session.closed"),
            "session_id": .string(sessionID),
            "reason": .string("backend_error"),
            "diagnostic": .object(["message": .string(message)]),
            "server_send_ts": .number(Date().timeIntervalSince1970),
        ])
        try await outbound.send(try JSONEncoder().encode(closed).asString())
    }
}

private func writeMiniCPMDemoRealtimeBackpressure(
    outbound: MiniCPMDemoOutboundChannel,
    recorder: MiniCPMDemoRecorderBox,
    sessionID: String?,
    inputID: String?,
    queuedItems: Int,
    retryAfterMS: Int
) async throws {
    var event: [String: MiniCPMJSONValue] = [
        "type": .string("response.backpressure"),
        "queued_items": .number(Double(queuedItems)),
        "retry_after_ms": .number(Double(retryAfterMS)),
        "server_send_ts": .number(Date().timeIntervalSince1970),
    ]
    if let sessionID { event["session_id"] = .string(sessionID) }
    if let inputID { event["input_id"] = .string(inputID) }
    let value = MiniCPMJSONValue.object(event)
    try await outbound.send(try JSONEncoder().encode(value).asString())
    await recorder.recordOutbound(value)
}

private func encodeOfficialMiniCPMDemoEvent(_ event: MiniCPMDemoEvent) throws -> String {
    var object: [String: MiniCPMJSONValue] = ["type": .string(event.type)]
    if let sessionID = event.sessionID { object["session_id"] = .string(sessionID) }
    if let responseID = event.responseID { object["response_id"] = .string(responseID) }
    if let inputID = event.inputID { object["input_id"] = .string(inputID) }
    if let kind = event.kind { object["kind"] = .string(kind) }
    if let text = event.text { object["text"] = .string(text) }
    if let audio = event.audio { object["audio"] = .string(audio) }
    if let format = event.format { object["format"] = .string(format) }
    if let sampleRate = event.sampleRate { object["sample_rate"] = .number(Double(sampleRate)) }
    if let reason = event.reason { object["reason"] = .string(reason) }
    if let metrics = event.metrics { object["metrics"] = .object(metrics) }
    if let state = event.state { object["state"] = .string(state) }
    if let error = event.error { object["error"] = error }
    // Lifecycle fields (not present in MiniCPMDemoEvent's flattened surface)
    // remain available from the payload for official clients.
    if event.type == "session.created", let mode = event.payload["mode"] {
        object["mode"] = mode
    }
    if event.type == "session.closed", let reason = event.payload["reason"] {
        object["reason"] = reason
    }
    object["server_send_ts"] = .number(Date().timeIntervalSince1970)
    let data = try JSONEncoder().encode(MiniCPMJSONValue.object(object))
    guard let value = String(data: data, encoding: .utf8) else {
        throw MiniCPMDemoBackendError.engine("failed to encode backend event")
    }
    return value
}

private func writeMiniCPMDemoError(outbound: MiniCPMDemoOutboundChannel, message: String) async throws {
    // The pinned protocol has no recoverable `error` event.  Keep a compact
    // diagnostic for legacy clients, but never emit it for a valid stream
    // event (official workers treat unknown events as fatal anyway).
    let event = MiniCPMJSONValue.object([
        "type": .string("session.closed"),
        "reason": .string("invalid_message"),
        "diagnostic": .object(["message": .string(message)]),
        "server_send_ts": .number(Date().timeIntervalSince1970),
    ])
    try await outbound.send(try JSONEncoder().encode(event).asString())
}

private func writeMiniCPMDemoQueueEvent(
    outbound: MiniCPMDemoOutboundChannel,
    type: String,
    ticketID: String,
    position: Int,
    queueLength: Int,
    estimatedWaitMS: Double,
    sessionID: String
) async throws {
    let event = MiniCPMJSONValue.object([
        "type": .string(type),
        "ticket_id": .string(ticketID),
        "position": .number(Double(position)),
        "queue_length": .number(Double(queueLength)),
        "estimated_wait_s": .number(estimatedWaitMS / 1_000.0),
        "session_id": .string(sessionID),
        "server_send_ts": .number(Date().timeIntervalSince1970),
    ])
    try await outbound.send(try JSONEncoder().encode(event).asString())
}

private extension Data {
    func asString() throws -> String {
        guard let value = String(data: self, encoding: .utf8) else {
            throw MiniCPMDemoBackendError.engine("failed to encode backend event")
        }
        return value
    }
}

private enum MiniCPMDemoWireRecordingSupport {
    static func requestValue(_ request: MiniCPMDemoRequest) throws -> MiniCPMJSONValue {
        try requestValueAndEncodedByteCount(request).value
    }

    static func requestEncodedByteCount(_ request: MiniCPMDemoRequest) throws -> Int {
        try requestValueAndEncodedByteCount(request).encodedBytes
    }

    static func requestValueAndEncodedByteCount(
        _ request: MiniCPMDemoRequest
    ) throws -> (value: MiniCPMJSONValue, encodedBytes: Int) {
        let data = try MiniCPMDemoPersistence.encoder().encode(request)
        return (
            try MiniCPMDemoPersistence.decoder().decode(MiniCPMJSONValue.self, from: data),
            data.count)
    }

    static func jsonValue(_ string: String) throws -> MiniCPMJSONValue {
        try JSONDecoder().decode(MiniCPMJSONValue.self, from: Data(string.utf8))
    }
}
