import Foundation
import Hummingbird
import HummingbirdCore
import NIOCore

/// Legacy `/queue` alias retained for the upstream admin page. The canonical
/// endpoint is `/api/queue`.
public func miniCPMDemoQueueResponse(
    scheduler: MiniCPMDemoInferenceScheduler
) async throws -> Response {
    try await MiniCPMDemoGatewayHTTP.queueResponse(scheduler: scheduler)
}

/// Register the persistent data/API surface of the pinned MiniCPM-o Gateway.
/// The handlers deliberately return the upstream JSON field names rather than
/// Swift implementation names; the vendored browser pages can therefore run
/// unchanged against the local MLX process.
public func registerMiniCPMDemoGatewayDataRoutes(
    _ router: Router<BasicRequestContext>,
    services: MiniCPMDemoServiceContainer,
    scheduler: MiniCPMDemoInferenceScheduler,
    registry: MiniCPMDemoBackendRegistry
) {
    router.post("/api/_debug/record_trace") { request, _ in
        do {
            let object = try await MiniCPMDemoGatewayHTTP.objectBody(request, maxBytes: 2 * 1024 * 1024)
            await MiniCPMDemoGatewayHTTP.appendDebugTrace(
                object: object,
                root: services.layout.root)
            return try MiniCPMDemoGatewayHTTP.json(.object(["ok": .bool(true)]))
        } catch {
            // The upstream endpoint is deliberately best-effort: a recording
            // diagnostic must never prevent the live session from continuing.
            return try MiniCPMDemoGatewayHTTP.json(.object(["ok": .bool(false)]))
        }
    }

    router.get("/api/frontend_defaults") { _, _ in
        try await MiniCPMDemoGatewayHTTP.json(
            try MiniCPMDemoGatewayHTTP.encodeValue(await services.config.frontendDefaults()))
    }

    router.get("/api/presets") { _, _ in
        guard let presets = services.presets else {
            return try MiniCPMDemoGatewayHTTP.json(.object([:]))
        }
        let listed = try await presets.list()
        var result: [String: MiniCPMJSONValue] = [:]
        for (mode, rows) in listed {
            result[mode] = .array(rows.map { MiniCPMDemoGatewayHTTP.presetValue($0) })
        }
        return try MiniCPMDemoGatewayHTTP.json(.object(result))
    }

    router.get("/api/presets/:mode/:preset_id/audio") { _, context in
        guard let presets = services.presets else {
            throw HTTPError(.notFound, message: "Preset not found")
        }
        let mode = try context.parameters.require("mode")
        let presetID = try context.parameters.require("preset_id")
        let audio = try await presets.audio(mode: mode, presetID: presetID)
        var systemAudio: [MiniCPMJSONValue] = []
        var refAudio: MiniCPMJSONValue?
        let listed = try await presets.list()
        if let preset = listed[mode]?.first(where: { $0.id == presetID }) {
            if case .array(let content)? = preset.values["system_content"] {
                for item in content where item["type"]?.stringValue == "audio" {
                    if let path = item["path"]?.stringValue,
                       let loaded = audio.first(where: { $0.path == path }) {
                        systemAudio.append(MiniCPMDemoGatewayHTTP.audioValue(loaded))
                    }
                }
            }
            if case .object(let ref)? = preset.values["ref_audio"],
               let path = ref["path"]?.stringValue,
               let loaded = audio.first(where: { $0.path == path }) {
                refAudio = MiniCPMDemoGatewayHTTP.audioValue(loaded)
            }
        }
        var payload: [String: MiniCPMJSONValue] = [
            "system_content_audio": .array(systemAudio)
        ]
        if let refAudio { payload["ref_audio"] = refAudio }
        return try MiniCPMDemoGatewayHTTP.json(.object(payload))
    }

    router.get("/api/default_ref_audio") { _, _ in
        do {
            let downloaded = try await services.assets.defaultReferenceAudio()
            let normalized = try MiniCPMDemoAudioNormalizer.normalize(
                data: downloaded.data,
                mimeType: downloaded.record.mimeType,
                fileExtension: URL(fileURLWithPath: downloaded.record.filename).pathExtension)
            let samples = normalized.samples.count
            let value: MiniCPMJSONValue = .object([
                "name": .string(downloaded.record.name),
                "duration": .number(normalized.durationSeconds),
                "sample_rate": .number(Double(normalized.sampleRate)),
                "samples": .number(Double(samples)),
                "base64": .string(normalized.float32Data.base64EncodedString()),
            ])
            return try MiniCPMDemoGatewayHTTP.json(value)
        } catch {
            return try MiniCPMDemoGatewayHTTP.errorResponse(error)
        }
    }

    router.get("/api/assets/ref_audio") { _, _ in
        let records = await services.assets.list(kind: .referenceAudio)
        let value: MiniCPMJSONValue = .object([
            "total": .number(Double(records.count)),
            "ref_audios": .array(records.map(MiniCPMDemoGatewayHTTP.assetValue)),
        ])
        return try MiniCPMDemoGatewayHTTP.json(value)
    }

    router.post("/api/assets/ref_audio") { request, _ in
        do {
            let object = try await MiniCPMDemoGatewayHTTP.objectBody(request, maxBytes: 256 * 1024 * 1024)
            guard let name = object["name"]?.stringValue,
                  let encoded = object["audio_base64"]?.stringValue,
                  let data = Data(base64Encoded: encoded) else {
                throw MiniCPMDemoServiceError.invalidMedia("name and audio_base64 are required")
            }
            let record = try await services.assets.uploadReferenceAudio(name: name, data: data)
            return try MiniCPMDemoGatewayHTTP.json(.object([
                "success": .bool(true),
                "id": .string(record.id),
                "name": .string(record.name),
                "message": .string("Uploaded successfully"),
            ]))
        } catch {
            return try MiniCPMDemoGatewayHTTP.errorResponse(error)
        }
    }

    router.post("/api/assets/ref_audio/external") { request, _ in
        do {
            let object = try await MiniCPMDemoGatewayHTTP.objectBody(request, maxBytes: 8 * 1024 * 1024)
            guard let name = object["name"]?.stringValue,
                  let path = object["path"]?.stringValue else {
                throw MiniCPMDemoServiceError.invalidMedia("name and path are required")
            }
            let file = URL(fileURLWithPath: path).standardizedFileURL
            guard let upstreamRoot = services.upstreamRoot,
                  MiniCPMDemoGatewayHTTP.isConfined(file, under: upstreamRoot),
                  FileManager.default.fileExists(atPath: file.path) else {
                throw MiniCPMDemoServiceError.invalidMedia("external audio path must be inside upstream root")
            }
            let record = try await services.assets.uploadReferenceAudio(
                name: name,
                data: Data(contentsOf: file),
                mimeType: MiniCPMDemoGatewayHTTP.mimeType(for: file.pathExtension),
                sampleRate: nil,
                durationMS: nil)
            return try MiniCPMDemoGatewayHTTP.json(.object([
                "success": .bool(true), "id": .string(record.id),
                "name": .string(record.name), "message": .string("Registered successfully")
            ]))
        } catch {
            return try MiniCPMDemoGatewayHTTP.errorResponse(error)
        }
    }

    router.get("/api/assets/ref_audio/:ref_id") { request, context in
        let id = try context.parameters.require("ref_id")
        do {
            let record = try await services.assets.get(id)
            let file = try await services.assets.fileURL(id)
            return try MiniCPMDemoGatewayHTTP.binary(
                file: file,
                mimeType: record.mimeType,
                filename: record.filename,
                headOnly: request.method == .head)
        } catch { return try MiniCPMDemoGatewayHTTP.errorResponse(error) }
    }
    router.head("/api/assets/ref_audio/:ref_id") { request, context in
        let id = try context.parameters.require("ref_id")
        do {
            let record = try await services.assets.get(id)
            return try MiniCPMDemoGatewayHTTP.binary(
                data: Data(), mimeType: record.mimeType, filename: record.filename, headOnly: true,
                contentLength: Int(record.sizeBytes))
        } catch { return try MiniCPMDemoGatewayHTTP.errorResponse(error) }
    }
    router.delete("/api/assets/ref_audio/:ref_id") { _, context in
        let id = try context.parameters.require("ref_id")
        do {
            let removed = try await services.assets.delete(id)
            guard removed else { throw MiniCPMDemoServiceError.notFound(id) }
            return try MiniCPMDemoGatewayHTTP.json(.object([
                "success": .bool(true), "id": .string(id), "message": .string("Deleted")
            ]))
        } catch { return try MiniCPMDemoGatewayHTTP.errorResponse(error) }
    }

    router.get("/api/queue") { _, _ in
        try await MiniCPMDemoGatewayHTTP.queueResponse(scheduler: scheduler)
    }
    router.delete("/api/queue/:ticket_id") { _, context in
        let ticketID = try context.parameters.require("ticket_id")
        do {
            _ = try await scheduler.status(ticketID: ticketID)
            await scheduler.cancel(ticketID: ticketID)
            return try MiniCPMDemoGatewayHTTP.json(.object(["success": .bool(true)]))
        } catch {
            return try MiniCPMDemoGatewayHTTP.json(.object(["success": .bool(false)]))
        }
    }

    router.get("/api/config/eta") { _, _ in
        try MiniCPMDemoGatewayHTTP.etaResponse(await services.config.etaStatus())
    }
    router.put("/api/config/eta") { request, _ in
        do {
            let object = try await MiniCPMDemoGatewayHTTP.objectBody(request, maxBytes: 64 * 1024)
            let current = await services.config.etaStatus().config
            let updated = MiniCPMDemoGatewayHTTP.etaConfig(from: object, current: current)
            let status = try await services.config.updateETA(updated)
            await scheduler.updateServiceEstimateMS(updated.chatSeconds * 1_000)
            return try MiniCPMDemoGatewayHTTP.etaResponse(status)
        } catch { return try MiniCPMDemoGatewayHTTP.errorResponse(error) }
    }

    // `/cache` is registered by `registerMiniCPMDemoRealtimeRoutes`, which
    // runs first in the standalone server. Keep this compatibility alias in
    // one route owner; registering it here as well would add a second GET
    // responder to the same Hummingbird endpoint and trap at startup.

    router.get("/sessions") { _, _ in
        let rows = (try? await services.sessions.list(includeActive: true)) ?? []
        let active = rows.filter(\.active)
        let completed = rows.filter { !$0.active }
        let activePayload = MiniCPMDemoGatewayHTTP.activeSessionValues(active)
        return try MiniCPMDemoGatewayHTTP.json(.object([
            // Keep the upstream admin contract (`sessions` means active
            // worker sessions) while exposing durable completed rows for the
            // local admin/viewer without a second round trip.
            "total": .number(Double(active.count)),
            "sessions": .array(activePayload),
            "active": .array(activePayload),
            "completed": .array(completed.map(MiniCPMDemoGatewayHTTP.sessionSummaryValue)),
            "registry_active": .number(Double(await registry.count())),
        ]))
    }
    router.get("/api/sessions") { _, _ in
        do {
            let rows = try await services.sessions.list(includeActive: true)
            let active = rows.filter(\.active)
            let completed = rows.filter { !$0.active }
            return try MiniCPMDemoGatewayHTTP.json(.object([
                "total": .number(Double(rows.count)),
                "sessions": .array(rows.map(MiniCPMDemoGatewayHTTP.sessionSummaryValue)),
                "active": .array(MiniCPMDemoGatewayHTTP.activeSessionValues(active)),
                "completed": .array(completed.map(MiniCPMDemoGatewayHTTP.sessionSummaryValue)),
            ]))
        } catch { return try MiniCPMDemoGatewayHTTP.errorResponse(error) }
    }
    router.get("/api/sessions/:session_id") { _, context in
        let id = try context.parameters.require("session_id")
        do { return try MiniCPMDemoGatewayHTTP.json(try await MiniCPMDemoGatewayHTTP.encodeValue(await services.sessions.metadata(sessionID: id))) }
        catch { return try MiniCPMDemoGatewayHTTP.errorResponse(error) }
    }
    router.get("/api/sessions/:session_id/recording") { _, context in
        let id = try context.parameters.require("session_id")
        do {
            let recording = try await services.sessions.recording(sessionID: id)
            let metadata = try await services.sessions.metadata(sessionID: id)
            let encodedEvents = try recording.events.map { try MiniCPMDemoGatewayHTTP.encodeValue($0) }
            return try MiniCPMDemoGatewayHTTP.json(.object([
                "session_id": .string(id),
                "meta": try MiniCPMDemoGatewayHTTP.encodeValue(metadata),
                "events": .array(encodedEvents),
            ]))
        } catch { return try MiniCPMDemoGatewayHTTP.errorResponse(error) }
    }
    router.get("/api/sessions/:session_id/download") { request, context in
        let id = try context.parameters.require("session_id")
        do {
            let archive = try await MiniCPMDemoGatewayHTTP.sessionZipFile(services: services, sessionID: id)
            return try MiniCPMDemoGatewayHTTP.binary(
                file: archive.url, mimeType: "application/zip", filename: "\(id).zip",
                headOnly: request.method == .head, removeAfter: true)
        } catch { return try MiniCPMDemoGatewayHTTP.errorResponse(error) }
    }
    router.head("/api/sessions/:session_id/download") { request, context in
        let id = try context.parameters.require("session_id")
        do {
            let archive = try await MiniCPMDemoGatewayHTTP.sessionZipFile(services: services, sessionID: id)
            return try MiniCPMDemoGatewayHTTP.binary(
                file: archive.url, mimeType: "application/zip", filename: "\(id).zip",
                headOnly: true, removeAfter: true)
        } catch { return try MiniCPMDemoGatewayHTTP.errorResponse(error) }
    }
    router.get("/api/sessions/:session_id/comment") { _, context in
        let id = try context.parameters.require("session_id")
        do {
            let comment = try await services.shares.comment(sessionID: id)
            return try MiniCPMDemoGatewayHTTP.json(.object(["comment": .string(comment.text)]))
        } catch { return try MiniCPMDemoGatewayHTTP.errorResponse(error) }
    }
    router.post("/api/sessions/:session_id/comment") { request, context in
        let id = try context.parameters.require("session_id")
        do {
            let object = try await MiniCPMDemoGatewayHTTP.objectBody(request, maxBytes: 64 * 1024)
            let text = object["comment"]?.stringValue ?? ""
            let comment = try await services.shares.setComment(sessionID: id, text: text)
            return try MiniCPMDemoGatewayHTTP.json(.object([
                "status": .string("ok"), "len": .number(Double(comment.text.count))
            ]))
        } catch { return try MiniCPMDemoGatewayHTTP.errorResponse(error) }
    }
    router.post("/api/sessions/:session_id/upload-recording") { request, context in
        let id = try context.parameters.require("session_id")
        do {
            let upload = try await MiniCPMDemoGatewayHTTP.multipartFile(request, maxBytes: 200 * 1024 * 1024)
            let stored = try await services.sessions.storeFrontendReplay(
                sessionID: id, data: upload.data,
                fileExtension: upload.extension,
                mimeType: upload.mimeType,
                role: "frontend_replay")
            return try MiniCPMDemoGatewayHTTP.json(.object([
                "status": .string("ok"),
                "path": .string(stored.relativePath),
                "size_bytes": .number(Double(stored.sizeBytes)),
            ]))
        } catch { return try MiniCPMDemoGatewayHTTP.errorResponse(error) }
    }
    router.get("/api/sessions/:session_id/assets/**") { request, context in
        let id = try context.parameters.require("session_id")
        let path = context.parameters.getCatchAll().map(String.init).joined(separator: "/")
        do {
            let file = try MiniCPMDemoGatewayHTTP.sessionAssetURL(
                services: services, sessionID: id, relativePath: path)
            return try MiniCPMDemoGatewayHTTP.binary(file: file, mimeType: MiniCPMDemoGatewayHTTP.mimeType(for: file.pathExtension), filename: file.lastPathComponent, headOnly: request.method == .head)
        } catch { return try MiniCPMDemoGatewayHTTP.errorResponse(error) }
    }
    router.head("/api/sessions/:session_id/assets/**") { _, context in
        let id = try context.parameters.require("session_id")
        let path = context.parameters.getCatchAll().map(String.init).joined(separator: "/")
        do {
            let file = try MiniCPMDemoGatewayHTTP.sessionAssetURL(services: services, sessionID: id, relativePath: path)
            let attrs = try FileManager.default.attributesOfItem(atPath: file.path)
            let size = (attrs[.size] as? NSNumber)?.intValue ?? 0
            return try MiniCPMDemoGatewayHTTP.binary(data: Data(), mimeType: MiniCPMDemoGatewayHTTP.mimeType(for: file.pathExtension), filename: file.lastPathComponent, headOnly: true, contentLength: size)
        } catch { return try MiniCPMDemoGatewayHTTP.errorResponse(error) }
    }

    router.get("/api/apps") { _, _ in
        let apps = await services.admin.listApps(includeDisabled: false)
        return try MiniCPMDemoGatewayHTTP.json(.object([
            "apps": .array(apps.map { .object([
                "app_id": .string($0.appID), "name": .string($0.name), "route": .string($0.route)
            ]) })
        ]))
    }
    router.get("/api/admin/apps") { _, _ in
        let apps = await services.admin.listApps(includeDisabled: true)
        return try MiniCPMDemoGatewayHTTP.json(.object([
            "apps": .array(try apps.map { try MiniCPMDemoGatewayHTTP.encodeValue($0) })
        ]))
    }
    router.put("/api/admin/apps/:app_id") { request, context in
        let appID = try context.parameters.require("app_id")
        do {
            let object = try await MiniCPMDemoGatewayHTTP.objectBody(request, maxBytes: 64 * 1024)
            guard let enabled = object["enabled"]?.boolValue else {
                throw MiniCPMDemoServiceError.invalidIdentifier("enabled")
            }
            let app = try await services.admin.setAppEnabled(appID: appID, enabled: enabled)
            return try MiniCPMDemoGatewayHTTP.json(.object([
                "success": .bool(true), "app": try MiniCPMDemoGatewayHTTP.encodeValue(app)
            ]))
        } catch { return try MiniCPMDemoGatewayHTTP.errorResponse(error) }
    }
}

private enum MiniCPMDemoGatewayHTTP {
    private static let debugTraceLock = NSLock()
    private static let debugTraceMaxBytes = 5 * 1024 * 1024

    struct UploadPart: Sendable {
        let data: Data
        let mimeType: String
        let `extension`: String
    }

    static func json(_ value: MiniCPMJSONValue, status: HTTPResponse.Status = .ok) throws -> Response {
        let data = try JSONEncoder().encode(value)
        return Response(status: status, headers: [.contentType: "application/json"], body: .init(byteBuffer: .init(data: data)))
    }

    static func appendDebugTrace(
        object: [String: MiniCPMJSONValue],
        root: URL
    ) async {
        let directory = root.appendingPathComponent(".run-logs", isDirectory: true)
        let path = directory.appendingPathComponent("mobile-record-trace.jsonl")
        let rolled = directory.appendingPathComponent("mobile-record-trace.jsonl.1")
        do {
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            var record = object
            record["ts"] = .string(ISO8601DateFormatter().string(from: Date()))
            let line = try MiniCPMDemoPersistence.encoder().encode(MiniCPMJSONValue.object(record)) + Data([0x0A])
            try withDebugTraceLock {
                if let attrs = try? FileManager.default.attributesOfItem(atPath: path.path),
                   let size = (attrs[.size] as? NSNumber)?.intValue,
                   size > debugTraceMaxBytes {
                    try? FileManager.default.removeItem(at: rolled)
                    try? FileManager.default.moveItem(at: path, to: rolled)
                }
                if !FileManager.default.createFile(atPath: path.path, contents: nil) {
                    guard FileManager.default.fileExists(atPath: path.path) else { return }
                }
                let handle = try FileHandle(forWritingTo: path)
                defer { try? handle.close() }
                try handle.seekToEnd()
                try handle.write(contentsOf: line)
            }
        } catch {
            // Intentionally ignored, matching the Python gateway's fail-safe
            // logging path.
        }
    }

    /// `NSLock.lock()` is annotated unavailable from async contexts in Swift
    /// 6.  Keep the critical section in a synchronous helper so the entire
    /// rotate-and-append sequence remains atomic without blocking an async
    /// task at the call site.
    private static func withDebugTraceLock<T>(_ body: () throws -> T) rethrows -> T {
        debugTraceLock.lock()
        defer { debugTraceLock.unlock() }
        return try body()
    }

    static func errorResponse(_ error: Error) throws -> Response {
        let status: HTTPResponse.Status
        switch error {
        case MiniCPMDemoServiceError.notFound: status = .notFound
        case MiniCPMDemoServiceError.limitExceeded: status = .init(code: 413, reasonPhrase: "Payload Too Large")
        case MiniCPMDemoServiceError.alreadyExists, MiniCPMDemoServiceError.closed: status = .conflict
        case MiniCPMDemoServiceError.persistence: status = .internalServerError
        default: status = .badRequest
        }
        return try json(.object(["detail": .string(error.localizedDescription)]), status: status)
    }

    static func objectBody(_ request: Request, maxBytes: Int) async throws -> [String: MiniCPMJSONValue] {
        let body = try await request.body.collect(upTo: maxBytes)
        guard body.readableBytes > 0 else { return [:] }
        let data = Data(buffer: body)
        let value = try JSONDecoder().decode(MiniCPMJSONValue.self, from: data)
        guard let object = value.objectValue else {
            throw MiniCPMDemoServiceError.invalidIdentifier("JSON object required")
        }
        return object
    }

    static func encodeValue<T: Encodable>(_ value: T) throws -> MiniCPMJSONValue {
        let data = try MiniCPMDemoPersistence.encoder().encode(value)
        return try MiniCPMDemoPersistence.decoder().decode(MiniCPMJSONValue.self, from: data)
    }

    static func presetValue(_ preset: MiniCPMDemoPreset) -> MiniCPMJSONValue {
        .object(preset.values.merging(["id": .string(preset.id)]) { current, _ in current })
    }

    static func assetValue(_ asset: MiniCPMDemoAssetRecord) -> MiniCPMJSONValue {
        .object([
            "id": .string(asset.id), "name": .string(asset.name), "filename": .string(asset.filename),
            "source_type": .string(asset.kind == .referenceAudio ? "upload" : "upload"),
            "kind": .string(asset.kind.rawValue), "mime_type": .string(asset.mimeType),
            "size_bytes": .number(Double(asset.sizeBytes)),
            "sample_rate": asset.sampleRate.map { .number(Double($0)) } ?? .null,
            "duration_ms": asset.durationMS.map { .number(Double($0)) } ?? .null,
            "created_at": .string(ISO8601DateFormatter().string(from: asset.createdAt)),
        ])
    }

    static func sessionSummaryValue(_ summary: MiniCPMDemoSessionSummary) -> MiniCPMJSONValue {
        .object([
            "session_id": .string(summary.metadata.sessionID), "mode": .string(summary.metadata.mode),
            "app_type": .string(summary.metadata.appType), "active": .bool(summary.active),
            "size_bytes": .number(Double(summary.sizeBytes)),
            "created_at": .string(ISO8601DateFormatter().string(from: summary.metadata.createdAt)),
        ])
    }

    static func activeSessionValues(_ summaries: [MiniCPMDemoSessionSummary]) -> [MiniCPMJSONValue] {
        summaries.map { summary in
            let created = ISO8601DateFormatter().string(from: summary.metadata.createdAt)
            return .object([
                "ticket_id": .string(summary.metadata.sessionID),
                "worker_id": .string(summary.metadata.worker["id"]?.stringValue ?? "local_mlx"),
                "messages_hash": summary.metadata.config["messages_hash"] ?? .string(""),
                "last_active": .string(created),
                "session_id": .string(summary.metadata.sessionID),
                "mode": .string(summary.metadata.mode),
                "active": .bool(true),
            ])
        }
    }

    static func audioValue(_ audio: MiniCPMDemoPresetAudio) -> MiniCPMJSONValue {
        .object([
            "data": .string(audio.data.base64EncodedString()), "name": .string(audio.name),
            "path": .string(audio.path), "duration": .number(audio.durationSeconds),
            "sample_rate": .number(Double(audio.sampleRate)),
            "samples": .number(Double(audio.data.count / MemoryLayout<Float>.size)),
        ])
    }

    static func etaResponse(_ status: MiniCPMDemoETAStatus) throws -> Response {
        let config = status.config
        func ema(_ key: String) -> MiniCPMJSONValue { status.ema[key].map { .number($0) } ?? .null }
        func samples(_ key: String) -> MiniCPMJSONValue { .number(Double(status.samples[key] ?? 0)) }
        return try json(.object([
            "config": .object([
                "eta_chat_s": .number(config.chatSeconds), "eta_half_duplex_s": .number(config.halfDuplexSeconds),
                "eta_audio_duplex_s": .number(config.audioDuplexSeconds), "eta_omni_duplex_s": .number(config.omniDuplexSeconds),
                "ema_alpha": .number(config.emaAlpha),
            ]),
            "ema_alpha": .number(config.emaAlpha),
            "ema_chat_s": ema("chat"), "ema_half_duplex_s": ema("half_duplex_audio"),
            "ema_audio_duplex_s": ema("audio_duplex"), "ema_omni_duplex_s": ema("omni_duplex"),
            "ema_chat_samples": samples("chat"), "ema_half_duplex_samples": samples("half_duplex_audio"),
            "ema_audio_duplex_samples": samples("audio_duplex"), "ema_omni_duplex_samples": samples("omni_duplex"),
        ]))
    }

    static func etaConfig(from object: [String: MiniCPMJSONValue], current: MiniCPMDemoETAConfig) -> MiniCPMDemoETAConfig {
        func number(_ key: String, _ fallback: Double) -> Double { object[key]?.numberValue ?? fallback }
        return MiniCPMDemoETAConfig(
            chatSeconds: number("eta_chat_s", current.chatSeconds),
            halfDuplexSeconds: number("eta_half_duplex_s", current.halfDuplexSeconds),
            audioDuplexSeconds: number("eta_audio_duplex_s", current.audioDuplexSeconds),
            omniDuplexSeconds: number("eta_omni_duplex_s", current.omniDuplexSeconds),
            emaAlpha: number("ema_alpha", current.emaAlpha),
            emaMinimumSamples: Int(number("ema_min_samples", Double(current.emaMinimumSamples))))
    }

    static func queueResponse(scheduler: MiniCPMDemoInferenceScheduler) async throws -> Response {
        let statuses = await scheduler.queuedStatuses()
        let snapshot = await scheduler.snapshot()
        let now = Date()
        let items: [MiniCPMJSONValue] = statuses.map { status in
            .object([
                "ticket_id": .string(status.ticketID), "request_type": .string(status.requestType),
                "position": .number(Double(status.position)), "estimated_wait_s": .number(status.estimatedWaitMS / 1_000),
                "enqueued_at": .string(ISO8601DateFormatter().string(from: status.enqueuedAt)),
                "wait_elapsed_s": .number(max(0, now.timeIntervalSince(status.enqueuedAt))),
            ])
        }
        return try json(.object([
            "queue_length": .number(Double(snapshot.queued)),
            "max_queue_size": .number(Double(await scheduler.maxQueueSize)),
            "items": .array(items), "running_tasks": .array([]),
        ]))
    }

    static func binary(
        data: Data,
        mimeType: String,
        filename: String,
        headOnly: Bool,
        contentLength: Int? = nil
    ) throws -> Response {
        var headers: HTTPFields = [.contentType: mimeType]
        headers[.contentDisposition] = "attachment; filename=\"\(filename.replacingOccurrences(of: "\"", with: ""))\""
        let length = contentLength ?? data.count
        headers[.contentLength] = String(length)
        return Response(status: .ok, headers: headers, body: headOnly ? .init() : .init(byteBuffer: .init(data: data)))
    }

    /// Stream a file without materialising it in a `Data` value.  Temporary
    /// archives pass `removeAfter` so their lifetime extends until the HTTP
    /// writer has finished (and not merely until the route returns).
    static func binary(
        file: URL,
        mimeType: String,
        filename: String,
        headOnly: Bool,
        removeAfter: Bool = false
    ) throws -> Response {
        guard FileManager.default.fileExists(atPath: file.path),
              (try? file.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile) == true else {
            throw MiniCPMDemoServiceError.notFound(filename)
        }
        let attrs = try FileManager.default.attributesOfItem(atPath: file.path)
        let size = (attrs[.size] as? NSNumber)?.intValue ?? 0
        var headers: HTTPFields = [.contentType: mimeType]
        headers[.contentDisposition] = "attachment; filename=\"\(filename.replacingOccurrences(of: "\"", with: ""))\""
        headers[.contentLength] = String(size)
        if headOnly {
            if removeAfter { try? FileManager.default.removeItem(at: file) }
            return Response(status: .ok, headers: headers, body: .init())
        }
        let body = ResponseBody(contentLength: size) { writer in
            let handle = try FileHandle(forReadingFrom: file)
            defer {
                try? handle.close()
                if removeAfter { try? FileManager.default.removeItem(at: file) }
            }
            while let chunk = try handle.read(upToCount: 64 * 1024), !chunk.isEmpty {
                try await writer.write(ByteBuffer(data: chunk))
            }
            try await writer.finish(nil)
        }
        return Response(status: .ok, headers: headers, body: body)
    }

    static func mimeType(for extension: String) -> String {
        switch `extension`.lowercased() {
        case "wav": return "audio/wav"
        case "webm": return "video/webm"
        case "mp4": return "video/mp4"
        case "mp3": return "audio/mpeg"
        case "jpg", "jpeg": return "image/jpeg"
        case "png": return "image/png"
        case "json": return "application/json"
        case "html": return "text/html; charset=utf-8"
        case "css": return "text/css; charset=utf-8"
        case "js", "mjs": return "text/javascript; charset=utf-8"
        default: return "application/octet-stream"
        }
    }

    static func isConfined(_ file: URL, under root: URL) -> Bool {
        let base = root.resolvingSymlinksInPath().standardizedFileURL.path
        let resolved = file.resolvingSymlinksInPath().standardizedFileURL.path
        return resolved == base || resolved.hasPrefix(base + "/")
    }

    static func sessionAssetURL(services: MiniCPMDemoServiceContainer, sessionID: String, relativePath: String) throws -> URL {
        guard !relativePath.isEmpty, !relativePath.hasPrefix("/"), !relativePath.split(separator: "/").contains("..") else {
            throw MiniCPMDemoServiceError.invalidIdentifier("asset path")
        }
        let session = try services.layout.sessionURL(sessionID)
        let file = session.appendingPathComponent(relativePath)
        guard isConfined(file, under: session), FileManager.default.fileExists(atPath: file.path) else {
            throw MiniCPMDemoServiceError.notFound(relativePath)
        }
        return file
    }

    static func multipartFile(_ request: Request, maxBytes: Int) async throws -> UploadPart {
        let contentType = request.headers[.contentType] ?? ""
        if let declared = request.headers[.contentLength].flatMap(Int.init), declared > maxBytes {
            throw MiniCPMDemoServiceError.limitExceeded("multipart body bytes")
        }
        // Do not use `collect(upTo:)` here: browsers can upload long WebM
        // recordings and that API allocates one contiguous buffer before any
        // validation.  Spool bounded chunks to a private temporary file first.
        let temporary = FileManager.default.temporaryDirectory
            .appendingPathComponent("minicpm-upload-\(UUID().uuidString)", isDirectory: false)
        var total = 0
        do {
            guard FileManager.default.createFile(atPath: temporary.path, contents: nil) else {
                throw MiniCPMDemoServiceError.persistence("cannot create upload spool")
            }
            let handle = try FileHandle(forWritingTo: temporary)
            defer { try? handle.close() }
            for try await buffer in request.body {
                let chunk = Data(buffer: buffer)
                guard total <= maxBytes - chunk.count else {
                    throw MiniCPMDemoServiceError.limitExceeded("multipart body bytes")
                }
                try handle.write(contentsOf: chunk)
                total += chunk.count
            }
            try handle.synchronize()
        } catch {
            try? FileManager.default.removeItem(at: temporary)
            throw error
        }
        defer { try? FileManager.default.removeItem(at: temporary) }
        let data = try Data(contentsOf: temporary)
        if !contentType.lowercased().hasPrefix("multipart/form-data") {
            let mime = contentType.split(separator: ";", maxSplits: 1).first.map(String.init) ?? "application/octet-stream"
            return UploadPart(data: data, mimeType: mime, extension: extensionForMime(mime))
        }
        guard let boundary = contentType.split(separator: ";").map(String.init).first(where: { $0.trimmingCharacters(in: .whitespaces).hasPrefix("boundary=") })?.split(separator: "=", maxSplits: 1).last.map(String.init) else {
            throw MiniCPMDemoServiceError.invalidMedia("multipart boundary missing")
        }
        let marker = Data(("--" + boundary.trimmingCharacters(in: CharacterSet(charactersIn: "\""))).utf8)
        for chunkSub in split(data: data, marker: marker) {
            var chunk = chunkSub
            while chunk.first == 0x0d || chunk.first == 0x0a { chunk.removeFirst() }
            guard let headerEnd = chunk.range(of: Data([0x0d, 0x0a, 0x0d, 0x0a])) else { continue }
            let headers = String(decoding: chunk[..<headerEnd.lowerBound], as: UTF8.self)
            guard headers.contains("name=\"file\"") else { continue }
            var payload = Data(chunk[headerEnd.upperBound...])
            while payload.last == 0x0d || payload.last == 0x0a { payload.removeLast() }
            let mime = headers.split(separator: "\n").first(where: { $0.lowercased().hasPrefix("content-type:") })?.split(separator: ":", maxSplits: 1).last.map { $0.trimmingCharacters(in: .whitespacesAndNewlines).split(separator: ";", maxSplits: 1).first.map(String.init) ?? "application/octet-stream" } ?? "application/octet-stream"
            let filename = headers.split(separator: "\n").first(where: { $0.contains("filename=") })?.split(separator: "filename=", maxSplits: 1).last.map { $0.trimmingCharacters(in: CharacterSet(charactersIn: "\";\r")) }
            return UploadPart(data: payload, mimeType: mime, extension: filename.map { URL(fileURLWithPath: String($0)).pathExtension }.flatMap({ $0.isEmpty ? nil : $0 }) ?? extensionForMime(mime))
        }
        throw MiniCPMDemoServiceError.invalidMedia("multipart file field missing")
    }

    private static func split(data: Data, marker: Data) -> [Data] {
        let bytes = Array(data)
        let needle = Array(marker)
        guard !needle.isEmpty else { return [data] }
        var result: [Data] = []
        var start = 0
        var index = 0
        while index + needle.count <= bytes.count {
            if Array(bytes[index..<(index + needle.count)]) == needle {
                result.append(Data(bytes[start..<index]))
                index += needle.count
                start = index
            } else {
                index += 1
            }
        }
        result.append(Data(bytes[start..<bytes.count]))
        return result
    }

    private static func extensionForMime(_ mime: String) -> String {
        switch mime.lowercased() {
        case "audio/wav": return "wav"
        case "audio/webm": return "webm"
        case "video/webm": return "webm"
        case "video/mp4": return "mp4"
        default: return "bin"
        }
    }

    struct SessionArchive: Sendable {
        let url: URL
        let size: Int
    }

    /// Build a classic ZIP in a temporary file.  The first pass computes each
    /// file's CRC/size, allowing the second pass to copy 64-KiB chunks directly
    /// to the archive without retaining a complete session in memory.
    static func sessionZipFile(services: MiniCPMDemoServiceContainer, sessionID: String) async throws -> SessionArchive {
        let session = try services.layout.sessionURL(sessionID)
        let canonicalSession = session.resolvingSymlinksInPath().standardizedFileURL
        guard FileManager.default.fileExists(atPath: canonicalSession.path),
              (try? canonicalSession.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) == true else {
            throw MiniCPMDemoServiceError.notFound(sessionID)
        }
        let files = (FileManager.default.enumerator(at: session, includingPropertiesForKeys: [.isRegularFileKey], options: [.skipsHiddenFiles])?.compactMap { $0 as? URL }.filter { (try? $0.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile) == true } ?? []).sorted { $0.path < $1.path }
        guard files.count < 10_000 else { throw MiniCPMDemoServiceError.limitExceeded("zip file count") }
        struct Entry {
            let file: URL
            let name: String
            let crc: UInt32
            let size: UInt32
            var offset: UInt32 = 0
        }
        var entries: [Entry] = []
        entries.reserveCapacity(files.count)
        let basePath = canonicalSession.path
        let basePrefix = basePath.hasSuffix("/") ? basePath : basePath + "/"
        for file in files {
            let canonicalFile = file.resolvingSymlinksInPath().standardizedFileURL
            let canonicalPath = canonicalFile.path
            guard canonicalPath.hasPrefix(basePrefix) else {
                throw MiniCPMDemoServiceError.invalidIdentifier("session archive path escapes session")
            }
            let relativePath = String(canonicalPath.dropFirst(basePrefix.count))
            guard !relativePath.isEmpty,
                  !relativePath.hasPrefix("/"),
                  !relativePath.split(separator: "/").contains("..") else {
                throw MiniCPMDemoServiceError.invalidIdentifier("session archive relative path")
            }
            let attrs = try FileManager.default.attributesOfItem(atPath: canonicalPath)
            let rawSize = (attrs[.size] as? NSNumber)?.int64Value ?? 0
            guard rawSize >= 0, rawSize <= Int64(UInt32.max) else { throw MiniCPMDemoServiceError.limitExceeded("zip file size") }
            let name = "\(sessionID)/\(relativePath)"
            guard name.utf8.count <= Int(UInt16.max) else { throw MiniCPMDemoServiceError.limitExceeded("zip entry name") }
            entries.append(Entry(file: canonicalFile, name: name, crc: try crc32(canonicalFile), size: UInt32(rawSize)))
        }

        let archiveURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("minicpm-session-\(UUID().uuidString).zip")
        guard FileManager.default.createFile(atPath: archiveURL.path, contents: nil) else {
            throw MiniCPMDemoServiceError.persistence("cannot create zip file")
        }
        do {
            let handle = try FileHandle(forWritingTo: archiveURL)
            defer { try? handle.close() }
            var offset: UInt64 = 0
            for index in entries.indices {
                let nameData = Data(entries[index].name.utf8)
                entries[index].offset = try Self.checkedUInt32(offset, label: "zip offset")
                var header = Data()
                header.appendLE(UInt32(0x04034b50)); header.appendLE(UInt16(20)); header.appendLE(UInt16(0)); header.appendLE(UInt16(0)); header.appendLE(UInt16(0)); header.appendLE(UInt16(0)); header.appendLE(entries[index].crc); header.appendLE(entries[index].size); header.appendLE(entries[index].size); header.appendLE(UInt16(nameData.count)); header.appendLE(UInt16(0)); header.append(nameData)
                try handle.write(contentsOf: header)
                offset += UInt64(header.count)
                try copyFile(entries[index].file, to: handle)
                offset += UInt64(entries[index].size)
            }
            let centralOffset = try Self.checkedUInt32(offset, label: "zip central offset")
            var central = Data()
            for entry in entries {
                let nameData = Data(entry.name.utf8)
                central.appendLE(UInt32(0x02014b50)); central.appendLE(UInt16(20)); central.appendLE(UInt16(20)); central.appendLE(UInt16(0)); central.appendLE(UInt16(0)); central.appendLE(UInt16(0)); central.appendLE(UInt16(0)); central.appendLE(entry.crc); central.appendLE(entry.size); central.appendLE(entry.size); central.appendLE(UInt16(nameData.count)); central.appendLE(UInt16(0)); central.appendLE(UInt16(0)); central.appendLE(UInt16(0)); central.appendLE(UInt16(0)); central.appendLE(UInt32(0)); central.appendLE(entry.offset); central.append(nameData)
            }
            guard entries.count <= Int(UInt16.max), central.count <= Int(UInt32.max) else { throw MiniCPMDemoServiceError.limitExceeded("zip central directory") }
            try handle.write(contentsOf: central)
            offset += UInt64(central.count)
            var end = Data()
            end.appendLE(UInt32(0x06054b50)); end.appendLE(UInt16(0)); end.appendLE(UInt16(0)); end.appendLE(UInt16(entries.count)); end.appendLE(UInt16(entries.count)); end.appendLE(UInt32(central.count)); end.appendLE(centralOffset); end.appendLE(UInt16(0))
            try handle.write(contentsOf: end)
            offset += UInt64(end.count)
            try handle.synchronize()
            return SessionArchive(url: archiveURL, size: try Self.checkedInt(offset, label: "zip size"))
        } catch {
            try? FileManager.default.removeItem(at: archiveURL)
            throw error
        }
    }

    private static func checkedUInt32(_ value: UInt64, label: String) throws -> UInt32 {
        guard value <= UInt64(UInt32.max) else { throw MiniCPMDemoServiceError.limitExceeded(label) }
        return UInt32(value)
    }

    private static func checkedInt(_ value: UInt64, label: String) throws -> Int {
        guard value <= UInt64(Int.max) else { throw MiniCPMDemoServiceError.limitExceeded(label) }
        return Int(value)
    }

    private static func crc32(_ file: URL) throws -> UInt32 {
        var crc: UInt32 = 0xffffffff
        let handle = try FileHandle(forReadingFrom: file)
        defer { try? handle.close() }
        while let chunk = try handle.read(upToCount: 64 * 1024), !chunk.isEmpty {
            for byte in chunk {
                crc ^= UInt32(byte)
                for _ in 0..<8 { crc = (crc >> 1) ^ ((crc & 1) == 1 ? 0xedb88320 : 0) }
            }
        }
        return crc ^ 0xffffffff
    }

    private static func copyFile(_ file: URL, to handle: FileHandle) throws {
        let source = try FileHandle(forReadingFrom: file)
        defer { try? source.close() }
        while let chunk = try source.read(upToCount: 64 * 1024), !chunk.isEmpty {
            try handle.write(contentsOf: chunk)
        }
    }
}

private extension Data {
    mutating func appendLE<T: FixedWidthInteger>(_ value: T) {
        var little = value.littleEndian
        Swift.withUnsafeBytes(of: &little) { append(contentsOf: $0) }
    }
}
