import Foundation
import Hummingbird
import HummingbirdCore
import HummingbirdWebSocket
import NIOCore

/// Metadata for the single-process Mac deployment.  The Gateway protocol
/// still exposes worker/queue observability, but this process deliberately
/// reports one logical `local_mlx` worker instead of pretending that
/// `maxConcurrent` represents several worker processes.
public struct MiniCPMDemoLogicalWorker: Sendable, Equatable {
    public let id: String
    public let status: String
    public let capacity: Int
    public let activeSessions: Int
    public let queueLength: Int

    public init(
        id: String = "local_mlx",
        status: String,
        capacity: Int,
        activeSessions: Int,
        queueLength: Int
    ) {
        self.id = id
        self.status = status
        self.capacity = capacity
        self.activeSessions = activeSessions
        self.queueLength = queueLength
    }
}

/// Register the control-plane routes that have equivalents in the upstream
/// Gateway.  The endpoint is intentionally an explicit registration helper:
/// the standalone executable may choose its own health route and can mount
/// these on the same Router without coupling the backend actor to Hummingbird
/// application construction.
public func registerMiniCPMDemoRealtimeRoutes(
    _ router: Router<BasicRequestContext>,
    scheduler: MiniCPMDemoInferenceScheduler,
    registry: MiniCPMDemoBackendRegistry,
    workerID: String = "local_mlx"
) {
    router.get("/status") { _, _ in
        let snapshot = await scheduler.snapshot()
        let activeSessions = await registry.count()
        let worker = MiniCPMDemoLogicalWorker(
            id: workerID,
            status: snapshot.active > 0 ? "busy" : "idle",
            capacity: snapshot.maxConcurrent,
            activeSessions: activeSessions,
            queueLength: snapshot.queued)
        let payload = MiniCPMDemoRealtimeRoutes.statusPayload(
            worker: worker,
            scheduler: snapshot,
            maxQueueSize: await scheduler.maxQueueSize,
            activeSessions: activeSessions)
        return try MiniCPMDemoRealtimeRoutes.jsonResponse(payload)
    }

    router.get("/workers") { _, _ in
        let snapshot = await scheduler.snapshot()
        let activeSessions = await registry.count()
        let worker = MiniCPMDemoLogicalWorker(
            id: workerID,
            status: snapshot.active > 0 ? "busy" : "idle",
            capacity: snapshot.maxConcurrent,
            activeSessions: activeSessions,
            queueLength: snapshot.queued)
        let row = MiniCPMDemoRealtimeRoutes.workerPayload(worker)
        let payload = MiniCPMJSONValue.object([
            "total": .number(1),
            "workers": .array([row]),
        ])
        return try MiniCPMDemoRealtimeRoutes.jsonResponse(payload)
    }

    // The upstream endpoint is used by queue UIs to distinguish an assigned
    // ticket from one that has disappeared.  Active tickets remain visible as
    // position=0 through the scheduler actor.
    router.get("/api/queue/:ticket_id") { _, context in
        let ticketID = try context.parameters.require("ticket_id")
        do {
            let status = try await scheduler.status(ticketID: ticketID)
            let payload = MiniCPMJSONValue.object([
                "found": .bool(true),
                "ticket": .object([
                    "ticket_id": .string(status.ticketID),
                    "session_id": .string(status.sessionID),
                    "request_type": .string(status.requestType),
                    "position": .number(Double(status.position)),
                    "queue_length": .number(Double(status.queueLength)),
                    "estimated_wait_s": .number(status.estimatedWaitMS / 1_000.0),
                    "enqueued_at": .string(ISO8601DateFormatter().string(from: status.enqueuedAt)),
                ]),
            ])
            return try MiniCPMDemoRealtimeRoutes.jsonResponse(payload)
        } catch {
            let payload = MiniCPMJSONValue.object([
                "found": .bool(false),
                "message": .string("ticket not in queue (may have been assigned or cancelled)"),
            ])
            return try MiniCPMDemoRealtimeRoutes.jsonResponse(payload)
        }
    }

    // `/cache` is a lightweight compatibility alias. It reports process-level
    // state only; mutable per-session KV lengths remain backend metrics.
    router.get("/cache") { _, _ in
        let snapshot = await scheduler.snapshot()
        let activeSessions = await registry.count()
        let payload = MiniCPMJSONValue.object([
            "workers": .array([
                .object([
                    "worker_id": .string(workerID),
                    "status": .string(snapshot.active > 0 ? "busy" : "idle"),
                    "active_sessions": .number(Double(activeSessions)),
                    "queue_length": .number(Double(snapshot.queued)),
                    "capacity": .number(Double(snapshot.maxConcurrent)),
                ])
            ])
        ])
        return try MiniCPMDemoRealtimeRoutes.jsonResponse(payload)
    }
}

/// Build the combined HTTP/1 WebSocket channel used by the standalone server.
/// `/backend` keeps the strict worker protocol; `/v1/realtime` selects the
/// public `chat|video|audio` mode from the query string.  An unknown mode is
/// rejected before upgrade rather than silently falling back to video.
public func makeMiniCPMDemoWebSocketServer(
    configuration: WebSocketServerConfiguration = .init(maxFrameSize: 128 * 1024 * 1024),
    makeBackend: @escaping @Sendable () -> MiniCPMDemoBackend,
    registry: MiniCPMDemoBackendRegistry? = nil,
    scheduler: MiniCPMDemoInferenceScheduler? = nil,
    queueTimeoutMS: Int? = 120_000,
    emitQueueEvents: Bool = true,
    recordingStore: MiniCPMDemoSessionStore? = nil
) -> HTTPServerBuilder {
    .http1WebSocketUpgrade(configuration: configuration) { request, _, _ in
        let requestURI = URI(request.path ?? "/")
        let path = requestURI.path
        guard path == "/backend" || path == "/v1/realtime" else {
            return .dontUpgrade
        }

        let realtimeMode: MiniCPMRealtimeMode?
        if path == "/v1/realtime" {
            let rawMode = requestURI.queryParameters["mode"].map { String($0) }
            guard let parsed = try? MiniCPMRealtimeMode(rawOrAlias: rawMode) else {
                return .dontUpgrade
            }
            realtimeMode = parsed
        } else {
            realtimeMode = nil
        }

        let connectionBackend = makeBackend()
        return .upgrade([:]) { inbound, outbound, _ in
            if let realtimeMode {
                try await handleMiniCPMDemoRealtimeWS(
                    inbound: inbound,
                    outbound: outbound,
                    backend: connectionBackend,
                    mode: realtimeMode,
                    registry: registry,
                    scheduler: scheduler,
                    queueTimeoutMS: queueTimeoutMS,
                    emitQueueEvents: emitQueueEvents,
                    recordingStore: recordingStore)
            } else {
                try await handleMiniCPMDemoBackendWS(
                    inbound: inbound,
                    outbound: outbound,
                    backend: connectionBackend,
                    registry: registry,
                    scheduler: scheduler,
                    queueTimeoutMS: queueTimeoutMS,
                    // Queue events are a Gateway/public concern.  `/backend`
                    // is the strict worker channel even when both upgrades
                    // share one process; do not leak scheduler events into
                    // the worker protocol.
                    recordingStore: recordingStore,
                    emitQueueEvents: false)
            }
        }
    }
}

private enum MiniCPMDemoRealtimeRoutes {
    static func statusPayload(
        worker: MiniCPMDemoLogicalWorker,
        scheduler: (active: Int, queued: Int, maxConcurrent: Int),
        maxQueueSize: Int,
        activeSessions: Int
    ) -> MiniCPMJSONValue {
        let workerPayload = self.workerPayload(worker)
        return .object([
            "gateway_healthy": .bool(true),
            "total_workers": .number(1),
            "idle_workers": .number(worker.status == "idle" ? 1 : 0),
            "busy_workers": .number(worker.status == "busy" ? 1 : 0),
            // The local process is a shared multimodal worker; it is not
            // truthful to classify it as a separate duplex worker process.
            "duplex_workers": .number(0),
            "loading_workers": .number(0),
            "error_workers": .number(0),
            "offline_workers": .number(0),
            "queue_length": .number(Double(scheduler.queued)),
            "max_queue_size": .number(Double(maxQueueSize)),
            "active_sessions": .number(Double(activeSessions)),
            "logical_worker": workerPayload,
            "running_tasks": .array(activeSessions > 0 ? [workerPayload] : []),
        ])
    }

    static func workerPayload(_ worker: MiniCPMDemoLogicalWorker) -> MiniCPMJSONValue {
        .object([
            "worker_id": .string(worker.id),
            "id": .string(worker.id),
            "status": .string(worker.status),
            "backend": .string("swift_mlx"),
            "capacity": .number(Double(worker.capacity)),
            "active_sessions": .number(Double(worker.activeSessions)),
            "queue_length": .number(Double(worker.queueLength)),
        ])
    }

    static func jsonResponse(_ payload: MiniCPMJSONValue) throws -> Response {
        let data = try JSONEncoder().encode(payload)
        return Response(
            status: .ok,
            headers: [.contentType: "application/json"],
            body: .init(byteBuffer: .init(data: data)))
    }
}
