import Foundation

/// Persistent storage limits shared by the Demo service layer.  The limits
/// are checked before every write; they are not advisory UI settings.
public struct MiniCPMDemoStorageLimits: Codable, Sendable, Equatable {
    public var maxTotalBytes: Int64
    public var maxSessionBytes: Int64
    public var maxAssetBytes: Int64
    public var maxCommentCharacters: Int
    public var maxEventBytes: Int64

    public init(
        maxTotalBytes: Int64 = 50 * 1024 * 1024 * 1024,
        maxSessionBytes: Int64 = 2 * 1024 * 1024 * 1024,
        maxAssetBytes: Int64 = 200 * 1024 * 1024,
        maxCommentCharacters: Int = 2_000,
        maxEventBytes: Int64 = 8 * 1024 * 1024
    ) {
        self.maxTotalBytes = max(0, maxTotalBytes)
        self.maxSessionBytes = max(0, maxSessionBytes)
        self.maxAssetBytes = max(0, maxAssetBytes)
        self.maxCommentCharacters = max(0, maxCommentCharacters)
        self.maxEventBytes = max(0, maxEventBytes)
    }
}

public enum MiniCPMDemoServiceError: Error, LocalizedError, Sendable, Equatable {
    case invalidIdentifier(String)
    case invalidName(String)
    case notFound(String)
    case alreadyExists(String)
    case closed(String)
    case limitExceeded(String)
    case invalidMedia(String)
    case invalidComment(String)
    case expired(String)
    case revoked(String)
    case persistence(String)

    public var errorDescription: String? {
        switch self {
        case .invalidIdentifier(let value): return "invalid identifier: \(value)"
        case .invalidName(let value): return "invalid name: \(value)"
        case .notFound(let value): return "not found: \(value)"
        case .alreadyExists(let value): return "already exists: \(value)"
        case .closed(let value): return "closed: \(value)"
        case .limitExceeded(let value): return "storage limit exceeded: \(value)"
        case .invalidMedia(let value): return "invalid media: \(value)"
        case .invalidComment(let value): return "invalid comment: \(value)"
        case .expired(let value): return "expired: \(value)"
        case .revoked(let value): return "revoked: \(value)"
        case .persistence(let value): return "persistence error: \(value)"
        }
    }
}

/// On-disk layout shared by sessions, recordings, assets, comments and share
/// tokens.  Every service receives this value instead of constructing paths
/// from request strings, keeping traversal checks in one place.
public struct MiniCPMDemoStorageLayout: Sendable, Equatable {
    public let root: URL
    public let sessions: URL
    public let assets: URL
    public let admin: URL

    public init(root: URL) throws {
        let normalized = root.standardizedFileURL
        self.root = normalized
        self.sessions = normalized.appendingPathComponent("sessions", isDirectory: true)
        self.assets = normalized.appendingPathComponent("assets", isDirectory: true)
        self.admin = normalized.appendingPathComponent("admin", isDirectory: true)
        do {
            try FileManager.default.createDirectory(
                at: self.sessions, withIntermediateDirectories: true)
            try FileManager.default.createDirectory(
                at: self.assets, withIntermediateDirectories: true)
            try FileManager.default.createDirectory(
                at: self.admin, withIntermediateDirectories: true)
        } catch {
            throw MiniCPMDemoServiceError.persistence(
                "cannot create storage root \(normalized.path): \(error.localizedDescription)")
        }
    }

    public func sessionURL(_ sessionID: String) throws -> URL {
        try Self.safeComponent(sessionID, label: "session_id")
        let url = sessions.appendingPathComponent(sessionID, isDirectory: true)
        return try confined(url, under: sessions)
    }

    public func sessionCommentURL(_ sessionID: String) throws -> URL {
        try sessionURL(sessionID).appendingPathComponent("comment.txt")
    }

    public func recordingURL(_ sessionID: String) throws -> URL {
        let url = try sessionURL(sessionID).appendingPathComponent("recording", isDirectory: true)
        return try confined(url, under: sessions)
    }

    public static func safeComponent(_ value: String, label: String) throws {
        guard !value.isEmpty, value.count <= 160,
              value != ".", value != "..",
              value.unicodeScalars.allSatisfy({ scalar in
                  scalar.isASCII && (scalar.value == 45 || scalar.value == 95
                      || scalar.value == 46
                      || (scalar.value >= 48 && scalar.value <= 57)
                      || (scalar.value >= 65 && scalar.value <= 90)
                      || (scalar.value >= 97 && scalar.value <= 122))
              }), !value.contains("..") else {
            throw MiniCPMDemoServiceError.invalidIdentifier("\(label)=\(value)")
        }
    }

    fileprivate func confined(_ url: URL, under base: URL) throws -> URL {
        let resolvedBase = base.resolvingSymlinksInPath().standardizedFileURL.path
        let resolved = url.resolvingSymlinksInPath().standardizedFileURL.path
        guard resolved == resolvedBase || resolved.hasPrefix(resolvedBase + "/") else {
            throw MiniCPMDemoServiceError.invalidIdentifier("path escapes storage root")
        }
        return url
    }
}

public struct MiniCPMDemoSessionMetadata: Codable, Sendable, Equatable {
    public var sessionID: String
    public var mode: String
    public var appType: String
    public var createdAt: Date
    public var endedAt: Date?
    public var durationSeconds: Double?
    public var closeReason: String?
    public var worker: [String: MiniCPMJSONValue]
    public var identity: [String: MiniCPMJSONValue]
    public var source: [String: MiniCPMJSONValue]
    public var config: [String: MiniCPMJSONValue]

    enum CodingKeys: String, CodingKey {
        case sessionID = "session_id"
        case mode, appType = "app_type"
        case createdAt = "created_at"
        case endedAt = "ended_at"
        case durationSeconds = "duration_s"
        case closeReason = "close_reason"
        case worker, identity, source, config
    }

    public init(
        sessionID: String,
        mode: String,
        appType: String? = nil,
        createdAt: Date = Date(),
        endedAt: Date? = nil,
        durationSeconds: Double? = nil,
        closeReason: String? = nil,
        worker: [String: MiniCPMJSONValue] = [:],
        identity: [String: MiniCPMJSONValue] = [:],
        source: [String: MiniCPMJSONValue] = [:],
        config: [String: MiniCPMJSONValue] = [:]
    ) {
        self.sessionID = sessionID
        self.mode = mode
        self.appType = appType ?? mode
        self.createdAt = createdAt
        self.endedAt = endedAt
        self.durationSeconds = durationSeconds
        self.closeReason = closeReason
        self.worker = worker
        self.identity = identity
        self.source = source
        self.config = config
    }

    // The pinned Demo writes epoch seconds in meta.json and the session
    // viewer multiplies those values by 1000.  Decode the older Swift
    // ISO-8601 representation as well so existing sessions remain readable.
    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.sessionID = try container.decode(String.self, forKey: .sessionID)
        self.mode = try container.decode(String.self, forKey: .mode)
        self.appType = try container.decodeIfPresent(String.self, forKey: .appType) ?? self.mode
        self.createdAt = try Self.decodeDate(container, forKey: .createdAt)
        self.endedAt = try Self.decodeOptionalDate(container, forKey: .endedAt)
        self.durationSeconds = try container.decodeIfPresent(Double.self, forKey: .durationSeconds)
        self.closeReason = try container.decodeIfPresent(String.self, forKey: .closeReason)
        self.worker = try container.decodeIfPresent([String: MiniCPMJSONValue].self, forKey: .worker) ?? [:]
        self.identity = try container.decodeIfPresent([String: MiniCPMJSONValue].self, forKey: .identity) ?? [:]
        self.source = try container.decodeIfPresent([String: MiniCPMJSONValue].self, forKey: .source) ?? [:]
        self.config = try container.decodeIfPresent([String: MiniCPMJSONValue].self, forKey: .config) ?? [:]
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(sessionID, forKey: .sessionID)
        try container.encode(mode, forKey: .mode)
        try container.encode(appType, forKey: .appType)
        try container.encode(createdAt.timeIntervalSince1970, forKey: .createdAt)
        try container.encodeIfPresent(endedAt?.timeIntervalSince1970, forKey: .endedAt)
        try container.encodeIfPresent(durationSeconds, forKey: .durationSeconds)
        try container.encodeIfPresent(closeReason, forKey: .closeReason)
        try container.encode(worker, forKey: .worker)
        try container.encode(identity, forKey: .identity)
        try container.encode(source, forKey: .source)
        try container.encode(config, forKey: .config)
    }

    private static func decodeDate(
        _ container: KeyedDecodingContainer<CodingKeys>,
        forKey key: CodingKeys
    ) throws -> Date {
        if let seconds = try? container.decode(Double.self, forKey: key) {
            return Date(timeIntervalSince1970: seconds)
        }
        if let text = try? container.decode(String.self, forKey: key),
           let date = ISO8601DateFormatter().date(from: text) {
            return date
        }
        throw DecodingError.dataCorruptedError(
            forKey: key, in: container, debugDescription: "expected epoch seconds or ISO-8601 date")
    }

    private static func decodeOptionalDate(
        _ container: KeyedDecodingContainer<CodingKeys>,
        forKey key: CodingKeys
    ) throws -> Date? {
        guard container.contains(key) else { return nil }
        guard try container.decodeNil(forKey: key) == false else { return nil }
        return try decodeDate(container, forKey: key)
    }
}

public struct MiniCPMDemoSessionSummary: Codable, Sendable, Equatable {
    public let metadata: MiniCPMDemoSessionMetadata
    public let active: Bool
    public let sizeBytes: Int64

    public init(metadata: MiniCPMDemoSessionMetadata, active: Bool, sizeBytes: Int64) {
        self.metadata = metadata
        self.active = active
        self.sizeBytes = sizeBytes
    }
}

public struct MiniCPMDemoStorageStats: Codable, Sendable, Equatable {
    public let totalBytes: Int64
    public let sessionCount: Int
    public let activeSessionCount: Int
    public let assetCount: Int
    public let maxTotalBytes: Int64

    public init(
        totalBytes: Int64,
        sessionCount: Int,
        activeSessionCount: Int,
        assetCount: Int,
        maxTotalBytes: Int64
    ) {
        self.totalBytes = totalBytes
        self.sessionCount = sessionCount
        self.activeSessionCount = activeSessionCount
        self.assetCount = assetCount
        self.maxTotalBytes = maxTotalBytes
    }
}

enum MiniCPMDemoPersistence {
    static func encoder() -> JSONEncoder {
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = [.sortedKeys]
        return encoder
    }

    static func decoder() -> JSONDecoder {
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return decoder
    }

    static func write<T: Encodable>(_ value: T, to url: URL) throws {
        let data: Data
        do {
            data = try encoder().encode(value)
        } catch {
            throw MiniCPMDemoServiceError.persistence("encode \(url.lastPathComponent): \(error.localizedDescription)")
        }
        do {
            try data.write(to: url, options: [.atomic])
        } catch {
            throw MiniCPMDemoServiceError.persistence("write \(url.path): \(error.localizedDescription)")
        }
    }

    static func read<T: Decodable>(_ type: T.Type, from url: URL) throws -> T {
        do {
            return try decoder().decode(type, from: Data(contentsOf: url))
        } catch {
            throw MiniCPMDemoServiceError.persistence("read \(url.path): \(error.localizedDescription)")
        }
    }

    static func directorySize(_ url: URL) -> Int64 {
        guard let enumerator = FileManager.default.enumerator(
            at: url, includingPropertiesForKeys: [.isRegularFileKey, .fileSizeKey],
            options: [.skipsHiddenFiles]) else { return 0 }
        var total: Int64 = 0
        for case let item as URL in enumerator {
            guard (try? item.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile) == true,
                  let size = try? item.resourceValues(forKeys: [.fileSizeKey]).fileSize else { continue }
            total += Int64(size)
        }
        return total
    }

    static func ensureDirectory(_ url: URL) throws {
        do { try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true) }
        catch { throw MiniCPMDemoServiceError.persistence("mkdir \(url.path): \(error.localizedDescription)") }
    }
}

public struct MiniCPMDemoSessionHandle: Sendable, Equatable {
    public let sessionID: String
    public let metadata: MiniCPMDemoSessionMetadata

    public init(sessionID: String, metadata: MiniCPMDemoSessionMetadata) {
        self.sessionID = sessionID
        self.metadata = metadata
    }
}

/// Durable session registry.  It owns only metadata and recorder handles;
/// recording bytes are written by ``MiniCPMDemoRecordingService`` directly to
/// disk, so a process restart does not lose completed events.
public actor MiniCPMDemoSessionStore {
    public let layout: MiniCPMDemoStorageLayout
    public let limits: MiniCPMDemoStorageLimits
    private var active: [String: MiniCPMDemoRecordingService] = [:]
    private var closing: Set<String> = []

    public init(
        root: URL,
        limits: MiniCPMDemoStorageLimits = .init()
    ) throws {
        self.layout = try MiniCPMDemoStorageLayout(root: root)
        self.limits = limits
    }

    public func create(_ metadata: MiniCPMDemoSessionMetadata) throws -> MiniCPMDemoSessionHandle {
        try MiniCPMDemoStorageLayout.safeComponent(metadata.sessionID, label: "session_id")
        let directory = try layout.sessionURL(metadata.sessionID)
        guard !FileManager.default.fileExists(atPath: directory.path) else {
            throw MiniCPMDemoServiceError.alreadyExists(metadata.sessionID)
        }
        let current = MiniCPMDemoPersistence.directorySize(layout.root)
        guard current < limits.maxTotalBytes else {
            throw MiniCPMDemoServiceError.limitExceeded("max_total_bytes")
        }
        try MiniCPMDemoPersistence.ensureDirectory(directory)
        do {
            try MiniCPMDemoPersistence.write(metadata, to: directory.appendingPathComponent("meta.json"))
            let recorder = try MiniCPMDemoRecordingService(
                sessionID: metadata.sessionID,
                directory: directory,
                limits: limits)
            active[metadata.sessionID] = recorder
            return MiniCPMDemoSessionHandle(sessionID: metadata.sessionID, metadata: metadata)
        } catch {
            try? FileManager.default.removeItem(at: directory)
            throw error
        }
    }

    public func metadata(sessionID: String) throws -> MiniCPMDemoSessionMetadata {
        let path = try layout.sessionURL(sessionID).appendingPathComponent("meta.json")
        guard FileManager.default.fileExists(atPath: path.path) else {
            throw MiniCPMDemoServiceError.notFound(sessionID)
        }
        return try MiniCPMDemoPersistence.read(MiniCPMDemoSessionMetadata.self, from: path)
    }

    public func list(includeActive: Bool = true) throws -> [MiniCPMDemoSessionSummary] {
        let entries = try FileManager.default.contentsOfDirectory(
            at: layout.sessions, includingPropertiesForKeys: [.isDirectoryKey], options: [.skipsHiddenFiles])
        return entries.compactMap { directory in
            guard (try? directory.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) == true else { return nil }
            let id = directory.lastPathComponent
            guard let metadata = try? self.metadata(sessionID: id) else { return nil }
            let isActive = active[id] != nil && !closing.contains(id)
            guard includeActive || !isActive else { return nil }
            return MiniCPMDemoSessionSummary(
                metadata: metadata,
                active: isActive,
                sizeBytes: MiniCPMDemoPersistence.directorySize(directory))
        }.sorted { $0.metadata.createdAt > $1.metadata.createdAt }
    }

    public func appendEvent(
        sessionID: String,
        direction: MiniCPMDemoRecordingDirection,
        frame: MiniCPMJSONValue
    ) async throws -> MiniCPMDemoRecordedEvent {
        let recorder = try activeRecorder(sessionID)
        return try await recorder.append(direction: direction, frame: frame)
    }

    public func appendEvent(
        sessionID: String,
        direction: MiniCPMDemoRecordingDirection,
        event: MiniCPMDemoEvent,
        timestamp: Date = Date()
    ) async throws -> MiniCPMDemoRecordedEvent {
        let recorder = try activeRecorder(sessionID)
        return try await recorder.append(direction: direction, event: event, timestamp: timestamp)
    }

    public func storeRecordingBlob(
        sessionID: String,
        data: Data,
        fileExtension: String,
        mimeType: String,
        role: String? = nil,
        sampleRate: Int = MiniCPMDemoAudioNormalizer.targetSampleRate
    ) async throws -> MiniCPMDemoStoredBlob {
        let recorder: MiniCPMDemoRecordingService
        if let activeRecorder = active[sessionID], !closing.contains(sessionID) {
            recorder = activeRecorder
        } else {
            // Frontend replay upload happens after the WebSocket has closed in
            // the official Demo.  Re-open only the durable recorder for an
            // existing session; event append/generation still requires the
            // active actor-owned recorder above.
            _ = try metadata(sessionID: sessionID)
            recorder = try MiniCPMDemoRecordingService(
                sessionID: sessionID,
                directory: try layout.sessionURL(sessionID),
                limits: limits)
        }
        return try await recorder.storeBlob(
            data: data,
            fileExtension: fileExtension,
            mimeType: mimeType,
            role: role,
            sampleRate: sampleRate)
    }

    /// Store the browser's final replay at the session root, matching the
    /// upstream `frontend_replay.{webm,mp4}` convention.  It is intentionally
    /// separate from protocol blobs: the session viewer discovers this file
    /// by its stable name rather than by scanning `blob/`.
    public func storeFrontendReplay(
        sessionID: String,
        data: Data,
        fileExtension: String,
        mimeType: String,
        role: String? = nil
    ) throws -> MiniCPMDemoStoredBlob {
        _ = try metadata(sessionID: sessionID)
        guard !data.isEmpty else { throw MiniCPMDemoServiceError.invalidMedia("empty replay") }
        guard Int64(data.count) <= limits.maxAssetBytes else {
            throw MiniCPMDemoServiceError.limitExceeded("max_asset_bytes")
        }
        let normalizedMime = mimeType.lowercased().split(separator: ";", maxSplits: 1).first.map(String.init) ?? ""
        let allowed: [String: (String, String)] = [
            "video/webm": ("webm", "video/webm"),
            "video/mp4": ("mp4", "video/mp4"),
            "audio/wav": ("wav", "audio/wav"),
            "audio/webm": ("webm", "audio/webm"),
        ]
        guard let canonical = allowed[normalizedMime] else {
            throw MiniCPMDemoServiceError.invalidMedia("unsupported replay MIME type")
        }
        let ext = try MiniCPMDemoRecordingService.normalizedFileExtension(canonical.0)
        let directory = try layout.sessionURL(sessionID)
        let path = directory.appendingPathComponent("frontend_replay.\(ext)")
        let current = MiniCPMDemoPersistence.directorySize(layout.root)
        let existing = (try? FileManager.default.attributesOfItem(atPath: path.path))?[.size] as? NSNumber
        let existingBytes = Int64(existing?.intValue ?? 0)
        guard current - existingBytes <= limits.maxTotalBytes - Int64(data.count) else {
            throw MiniCPMDemoServiceError.limitExceeded("max_total_bytes")
        }
        try data.write(to: path, options: [.atomic])
        return MiniCPMDemoStoredBlob(
            relativePath: "frontend_replay.\(ext)",
            mimeType: canonical.1,
            sizeBytes: Int64(data.count),
            role: role ?? "frontend_replay")
    }

    public func recording(sessionID: String) async throws -> MiniCPMDemoRecordingSnapshot {
        _ = try metadata(sessionID: sessionID)
        let directory = try layout.sessionURL(sessionID)
        let stream = directory.appendingPathComponent("stream.jsonl")
        // Do not let MiniCPMDemoRecordingService create a missing stream here:
        // a deleted/corrupt recording is a 404 in the upstream gateway, not a
        // successful empty recording that hides data loss.
        guard FileManager.default.fileExists(atPath: stream.path) else {
            throw MiniCPMDemoServiceError.notFound("recording (sessionID)")
        }
        let recorder = try MiniCPMDemoRecordingService(
            sessionID: sessionID,
            directory: directory,
            limits: limits)
        return try await recorder.snapshot()
    }

    @discardableResult
    public func close(sessionID: String, reason: String = "client_closed") async throws -> MiniCPMDemoSessionMetadata {
        guard active[sessionID] != nil else {
            let current = try metadata(sessionID: sessionID)
            return current
        }
        guard !closing.contains(sessionID) else { throw MiniCPMDemoServiceError.closed(sessionID) }
        closing.insert(sessionID)
        defer { closing.remove(sessionID) }
        if let recorder = active[sessionID] { try await recorder.close() }
        var current = try metadata(sessionID: sessionID)
        let ended = Date()
        current.endedAt = ended
        current.durationSeconds = max(0, ended.timeIntervalSince(current.createdAt))
        current.closeReason = reason
        try MiniCPMDemoPersistence.write(
            current, to: try layout.sessionURL(sessionID).appendingPathComponent("meta.json"))
        active.removeValue(forKey: sessionID)
        return current
    }

    public func delete(sessionID: String) throws {
        guard active[sessionID] == nil else { throw MiniCPMDemoServiceError.closed("active session \(sessionID)") }
        let directory = try layout.sessionURL(sessionID)
        guard FileManager.default.fileExists(atPath: directory.path) else { throw MiniCPMDemoServiceError.notFound(sessionID) }
        try FileManager.default.removeItem(at: directory)
    }

    public func cleanup(
        now: Date = Date(),
        retention: TimeInterval? = nil,
        maxTotalBytes: Int64? = nil
    ) throws -> MiniCPMDemoCleanupReport {
        var summaries = try list(includeActive: true)
        let cutoff = retention.map { now.addingTimeInterval(-$0) }
        var deleted: [String] = []
        for summary in summaries where !summary.active {
            if let cutoff, summary.metadata.createdAt < cutoff {
                try delete(sessionID: summary.metadata.sessionID)
                deleted.append(summary.metadata.sessionID)
            }
        }
        summaries = try list(includeActive: true)
        var total = summaries.reduce(Int64(0)) { $0 + $1.sizeBytes }
        let limit = maxTotalBytes ?? limits.maxTotalBytes
        if limit >= 0 {
            for summary in summaries.sorted(by: { $0.metadata.createdAt < $1.metadata.createdAt })
                where !summary.active && total > limit {
                try delete(sessionID: summary.metadata.sessionID)
                deleted.append(summary.metadata.sessionID)
                total -= summary.sizeBytes
            }
        }
        return MiniCPMDemoCleanupReport(
            scanned: summaries.count + deleted.count,
            deletedSessionIDs: deleted,
            remainingBytes: total)
    }

    public func storageStats(assetCount: Int = 0) throws -> MiniCPMDemoStorageStats {
        let sessions = try list(includeActive: true)
        return MiniCPMDemoStorageStats(
            totalBytes: MiniCPMDemoPersistence.directorySize(layout.root),
            sessionCount: sessions.count,
            activeSessionCount: sessions.filter(\.active).count,
            assetCount: assetCount,
            maxTotalBytes: limits.maxTotalBytes)
    }

    private func activeRecorder(_ sessionID: String) throws -> MiniCPMDemoRecordingService {
        guard let recorder = active[sessionID], !closing.contains(sessionID) else {
            throw MiniCPMDemoServiceError.closed(sessionID)
        }
        return recorder
    }
}

public struct MiniCPMDemoCleanupReport: Codable, Sendable, Equatable {
    public let scanned: Int
    public let deletedSessionIDs: [String]
    public let remainingBytes: Int64

    public init(scanned: Int, deletedSessionIDs: [String], remainingBytes: Int64) {
        self.scanned = scanned
        self.deletedSessionIDs = deletedSessionIDs
        self.remainingBytes = remainingBytes
    }
}
