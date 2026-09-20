import Foundation

public enum MiniCPMDemoRecordingDirection: String, Codable, Sendable {
    case up
    case down
    case internalEvent = "internal"
}

public struct MiniCPMDemoRecordedEvent: Codable, Sendable, Equatable {
    public let sequence: Int64
    public let timestamp: Double
    public let direction: MiniCPMDemoRecordingDirection
    public let frame: MiniCPMJSONValue

    /// Compatibility accessor for callers that used the pre-parity Swift
    /// schema. On disk the canonical field is `ts` (epoch seconds).
    public var timestampMS: Int64 { Int64((timestamp * 1_000).rounded()) }

    enum CodingKeys: String, CodingKey {
        case sequence = "seq"
        case timestamp = "ts"
        case legacyTimestampMS = "ts_ms"
        case direction = "dir"
        case frame
    }

    public init(
        sequence: Int64,
        timestampMS: Int64,
        direction: MiniCPMDemoRecordingDirection,
        frame: MiniCPMJSONValue
    ) {
        self.sequence = sequence
        self.timestamp = Double(timestampMS) / 1_000
        self.direction = direction
        self.frame = frame
    }

    public init(
        sequence: Int64,
        timestamp: Double,
        direction: MiniCPMDemoRecordingDirection,
        frame: MiniCPMJSONValue
    ) {
        self.sequence = sequence
        self.timestamp = timestamp
        self.direction = direction
        self.frame = frame
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.sequence = try container.decode(Int64.self, forKey: .sequence)
        if let seconds = try? container.decode(Double.self, forKey: .timestamp) {
            self.timestamp = seconds
        } else if let milliseconds = try? container.decode(Int64.self, forKey: .legacyTimestampMS) {
            self.timestamp = Double(milliseconds) / 1_000
        } else {
            throw DecodingError.keyNotFound(
                CodingKeys.timestamp,
                .init(codingPath: container.codingPath, debugDescription: "recording timestamp missing"))
        }
        self.direction = try container.decode(MiniCPMDemoRecordingDirection.self, forKey: .direction)
        self.frame = try container.decode(MiniCPMJSONValue.self, forKey: .frame)
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(sequence, forKey: .sequence)
        try container.encode(timestamp, forKey: .timestamp)
        try container.encode(direction, forKey: .direction)
        try container.encode(frame, forKey: .frame)
    }
}

public struct MiniCPMDemoStoredBlob: Codable, Sendable, Equatable {
    public let relativePath: String
    public let mimeType: String
    public let sizeBytes: Int64
    public let role: String?

    public init(relativePath: String, mimeType: String, sizeBytes: Int64, role: String? = nil) {
        self.relativePath = relativePath
        self.mimeType = mimeType
        self.sizeBytes = sizeBytes
        self.role = role
    }
}

public struct MiniCPMDemoRecordingSnapshot: Codable, Sendable, Equatable {
    public let sessionID: String
    public let events: [MiniCPMDemoRecordedEvent]
    public let blobs: [MiniCPMDemoStoredBlob]
    public let sizeBytes: Int64

    public init(
        sessionID: String,
        events: [MiniCPMDemoRecordedEvent],
        blobs: [MiniCPMDemoStoredBlob],
        sizeBytes: Int64
    ) {
        self.sessionID = sessionID
        self.events = events
        self.blobs = blobs
        self.sizeBytes = sizeBytes
    }
}

/// Append-only on-disk recorder for the official backend stream.  It mirrors
/// `data/sessions/<id>/stream.jsonl` from the Python Demo while exposing blob
/// storage separately so large PCM/JPEG payloads never accumulate in memory.
public actor MiniCPMDemoRecordingService {
    public let sessionID: String
    public let directory: URL
    public let limits: MiniCPMDemoStorageLimits

    private let streamURL: URL
    private let blobsURL: URL
    private var nextSequence: Int64 = 0
    private var closed = false

    public init(
        sessionID: String,
        directory: URL,
        limits: MiniCPMDemoStorageLimits = .init()
    ) throws {
        try MiniCPMDemoStorageLayout.safeComponent(sessionID, label: "session_id")
        self.sessionID = sessionID
        self.directory = directory.standardizedFileURL
        self.limits = limits
        self.streamURL = self.directory.appendingPathComponent("stream.jsonl")
        self.blobsURL = self.directory.appendingPathComponent("blob", isDirectory: true)
        try MiniCPMDemoPersistence.ensureDirectory(self.directory)
        try MiniCPMDemoPersistence.ensureDirectory(self.blobsURL)
        if !FileManager.default.fileExists(atPath: streamURL.path) {
            guard FileManager.default.createFile(atPath: streamURL.path, contents: nil) else {
                throw MiniCPMDemoServiceError.persistence("cannot create stream.jsonl")
            }
        }
        if FileManager.default.fileExists(atPath: streamURL.path) {
            let existing = try Self.readEvents(from: streamURL)
            self.nextSequence = (existing.last?.sequence ?? -1) + 1
        }
    }

    public func append(
        direction: MiniCPMDemoRecordingDirection,
        frame: MiniCPMJSONValue,
        timestamp: Date = Date()
    ) throws -> MiniCPMDemoRecordedEvent {
        try ensureOpen()
        let event = MiniCPMDemoRecordedEvent(
            sequence: nextSequence,
            timestamp: timestamp.timeIntervalSince1970,
            direction: direction,
            frame: frame)
        let data: Data
        do {
            data = try MiniCPMDemoPersistence.encoder().encode(event) + Data([0x0A])
        } catch {
            throw MiniCPMDemoServiceError.persistence("encode recording event: \(error.localizedDescription)")
        }
        guard Int64(data.count) <= limits.maxEventBytes else {
            throw MiniCPMDemoServiceError.limitExceeded("max_event_bytes")
        }
        try ensureWriteCapacity(additional: Int64(data.count))
        do {
            let handle = try FileHandle(forWritingTo: streamURL)
            defer { try? handle.close() }
            try handle.seekToEnd()
            try handle.write(contentsOf: data)
            try handle.synchronize()
        } catch {
            throw MiniCPMDemoServiceError.persistence("append stream.jsonl: \(error.localizedDescription)")
        }
        nextSequence += 1
        return event
    }

    public func append(
        direction: MiniCPMDemoRecordingDirection,
        event: MiniCPMDemoEvent,
        timestamp: Date = Date()
    ) throws -> MiniCPMDemoRecordedEvent {
        let value: MiniCPMJSONValue
        do {
            let data = try MiniCPMDemoPersistence.encoder().encode(event)
            value = try MiniCPMDemoPersistence.decoder().decode(MiniCPMJSONValue.self, from: data)
        } catch {
            throw MiniCPMDemoServiceError.persistence("encode protocol event: \(error.localizedDescription)")
        }
        return try append(direction: direction, frame: value, timestamp: timestamp)
    }

    /// Store PCM, JPEG, video or any other already validated media payload.
    /// The generated filename is never derived from a client path and is
    /// checked for existence before writing, preventing accidental overwrite.
    public func storeBlob(
        data: Data,
        fileExtension: String,
        mimeType: String,
        role: String? = nil,
        sampleRate: Int = MiniCPMDemoAudioNormalizer.targetSampleRate
    ) throws -> MiniCPMDemoStoredBlob {
        try ensureOpen()
        guard !data.isEmpty else { throw MiniCPMDemoServiceError.invalidMedia("empty blob") }
        guard Int64(data.count) <= limits.maxAssetBytes else {
            throw MiniCPMDemoServiceError.limitExceeded("max_asset_bytes")
        }
        let ext = try Self.normalizedFileExtension(fileExtension)
        var storedData = data
        let normalizedMime = mimeType.lowercased().split(separator: ";", maxSplits: 1).first.map(String.init)
        if ext == "wav" || normalizedMime == "audio/wav" {
            guard ext == "wav", normalizedMime == "audio/wav" else {
                throw MiniCPMDemoServiceError.invalidMedia("WAV extension and MIME type disagree")
            }
            if !Self.isRIFFWAV(data) {
                // Raw PCM in the wire protocol is Float32.  Only wrap it if
                // every sample is finite; arbitrary bytes must never be
                // silently relabelled as a WAV file.
                guard let samples = MiniCPMDemoAudioNormalizer.samples(fromRawFloat32: data) else {
                    throw MiniCPMDemoServiceError.invalidMedia("invalid WAV or raw Float32 payload")
                }
                storedData = MiniCPMDemoAudioNormalizer.wavData(samples: samples, sampleRate: sampleRate)
            }
        } else if let normalizedMime {
            guard Self.hasValidSignature(data, mimeType: normalizedMime, fileExtension: ext) else {
                throw MiniCPMDemoServiceError.invalidMedia("media signature does not match (normalizedMime)")
            }
        }
        var target: URL
        repeat {
            target = blobsURL.appendingPathComponent("blob_\(UUID().uuidString.lowercased()).\(ext)")
        } while FileManager.default.fileExists(atPath: target.path)
        try ensureWriteCapacity(additional: Int64(storedData.count))
        guard FileManager.default.createFile(atPath: target.path, contents: storedData) else {
            throw MiniCPMDemoServiceError.persistence("cannot create recording blob")
        }
        let relative = "blob/\(target.lastPathComponent)"
        return MiniCPMDemoStoredBlob(
            relativePath: relative, mimeType: mimeType, sizeBytes: Int64(storedData.count), role: role)
    }

    public func close() throws {
        closed = true
    }

    public func isClosed() -> Bool { closed }

    public func snapshot() throws -> MiniCPMDemoRecordingSnapshot {
        let events = try Self.readEvents(from: streamURL)
        let blobs = try Self.readBlobs(from: blobsURL)
        return MiniCPMDemoRecordingSnapshot(
            sessionID: sessionID,
            events: events,
            blobs: blobs,
            sizeBytes: MiniCPMDemoPersistence.directorySize(directory))
    }

    public static func readEvents(from streamURL: URL) throws -> [MiniCPMDemoRecordedEvent] {
        guard FileManager.default.fileExists(atPath: streamURL.path) else { return [] }
        let data: Data
        do { data = try Data(contentsOf: streamURL) }
        catch { throw MiniCPMDemoServiceError.persistence("read stream.jsonl: \(error.localizedDescription)") }
        var events: [MiniCPMDemoRecordedEvent] = []
        for line in String(decoding: data, as: UTF8.self).split(separator: "\n") {
            guard let lineData = String(line).data(using: .utf8),
                  let event = try? MiniCPMDemoPersistence.decoder().decode(
                    MiniCPMDemoRecordedEvent.self, from: lineData) else {
                continue
            }
            events.append(event)
        }
        return events
    }

    private static func isRIFFWAV(_ data: Data) -> Bool {
        guard data.count >= 12 else { return false }
        return data.prefix(4).elementsEqual(Data("RIFF".utf8))
            && data.subdata(in: 8..<12).elementsEqual(Data("WAVE".utf8))
    }

    private static func hasValidSignature(_ data: Data, mimeType: String, fileExtension: String) -> Bool {
        switch mimeType {
        case "image/jpeg":
            return fileExtension == "jpg" || fileExtension == "jpeg"
                ? data.count >= 3 && data.prefix(3).elementsEqual(Data([0xff, 0xd8, 0xff])) : false
        case "image/png":
            return fileExtension == "png"
                && data.count >= 8
                && data.prefix(8).elementsEqual(Data([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]))
        case "video/webm", "audio/webm":
            return fileExtension == "webm"
                && data.count >= 4 && data.prefix(4).elementsEqual(Data([0x1a, 0x45, 0xdf, 0xa3]))
        case "video/mp4":
            guard fileExtension == "mp4", data.count >= 8 else { return false }
            return data.subdata(in: 4..<8).elementsEqual(Data("ftyp".utf8))
        case "audio/mpeg":
            guard fileExtension == "mp3", data.count >= 2 else { return false }
            return data.prefix(3).elementsEqual(Data("ID3".utf8))
                || (data[0] == 0xff && (data[1] & 0xe0) == 0xe0)
        default:
            // Unknown application/octet-stream payloads are intentionally
            // left unconstrained; the extension/MIME still comes from the
            // trusted protocol field rather than a client filename.
            return true
        }
    }

    private static func readBlobs(from url: URL) throws -> [MiniCPMDemoStoredBlob] {
        guard FileManager.default.fileExists(atPath: url.path) else { return [] }
        let files = try FileManager.default.contentsOfDirectory(
            at: url, includingPropertiesForKeys: [.isRegularFileKey, .fileSizeKey], options: [.skipsHiddenFiles])
        return files.compactMap { file in
            guard (try? file.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile) == true,
                  let size = try? file.resourceValues(forKeys: [.fileSizeKey]).fileSize else { return nil }
            return MiniCPMDemoStoredBlob(
                relativePath: "blob/\(file.lastPathComponent)",
                mimeType: Self.mimeType(for: file.pathExtension),
                sizeBytes: Int64(size))
        }.sorted { $0.relativePath < $1.relativePath }
    }

    private func ensureOpen() throws {
        guard !closed else { throw MiniCPMDemoServiceError.closed(sessionID) }
    }

    private func ensureWriteCapacity(additional: Int64) throws {
        let sessionBytes = MiniCPMDemoPersistence.directorySize(directory)
        guard additional >= 0, sessionBytes <= limits.maxSessionBytes - additional else {
            throw MiniCPMDemoServiceError.limitExceeded("max_session_bytes")
        }
        let root = directory.deletingLastPathComponent().deletingLastPathComponent()
        let total = MiniCPMDemoPersistence.directorySize(root)
        guard total <= limits.maxTotalBytes - additional else {
            throw MiniCPMDemoServiceError.limitExceeded("max_total_bytes")
        }
    }

    static func normalizedFileExtension(_ value: String) throws -> String {
        let ext = value.trimmingCharacters(in: CharacterSet(charactersIn: ". ")).lowercased()
        guard !ext.isEmpty, ext.count <= 8,
              ext.unicodeScalars.allSatisfy({ $0.isASCII && ($0.value == 45 || $0.value == 95
                  || ($0.value >= 48 && $0.value <= 57)
                  || ($0.value >= 97 && $0.value <= 122)) }) else {
            throw MiniCPMDemoServiceError.invalidMedia("unsafe extension")
        }
        return ext
    }

    private static func mimeType(for ext: String) -> String {
        switch ext.lowercased() {
        case "wav": return "audio/wav"
        case "jpg", "jpeg": return "image/jpeg"
        case "png": return "image/png"
        case "webm": return "video/webm"
        case "mp4": return "video/mp4"
        case "mp3": return "audio/mpeg"
        default: return "application/octet-stream"
        }
    }
}

/// Side-channel recorder used by Gateway WebSockets.  It stores canonical
/// protocol frames through ``MiniCPMDemoSessionStore`` and externalizes large
/// base64/PCM/video fields to bounded blob files before JSONL persistence.
/// Recording failures are fail-safe: the live transport can continue while
/// the session metadata still records its final close reason.
public actor MiniCPMDemoWireRecordingSession {
    public let sessionID: String
    private let store: MiniCPMDemoSessionStore
    private var enabled = true
    private var closed = false

    public init(
        store: MiniCPMDemoSessionStore,
        metadata: MiniCPMDemoSessionMetadata
    ) async throws {
        self.store = store
        self.sessionID = metadata.sessionID
        _ = try await store.create(metadata)
    }

    public func record(direction: MiniCPMDemoRecordingDirection, frame: MiniCPMJSONValue) async {
        guard enabled, !closed else { return }
        do {
            let canonical = try await externalize(
                frame, sampleRate: direction == .down ? 24_000 : 16_000)
            _ = try await store.appendEvent(sessionID: sessionID, direction: direction, frame: canonical)
        } catch {
            // Recorder I/O is deliberately not allowed to tear down a model
            // stream. Disable subsequent writes after a capacity/disk error.
            enabled = false
        }
    }

    public func close(reason: String) async {
        guard !closed else { return }
        closed = true
        _ = try? await store.close(sessionID: sessionID, reason: reason)
    }

    private func externalize(
        _ value: MiniCPMJSONValue,
        key: String? = nil,
        format: String? = nil,
        sampleRate: Int = MiniCPMDemoAudioNormalizer.targetSampleRate
    ) async throws -> MiniCPMJSONValue {
        switch value {
        case .object(let object):
            let objectFormat = object["format"]?.stringValue ?? format
            var result: [String: MiniCPMJSONValue] = [:]
            for (childKey, childValue) in object {
                result[childKey] = try await externalize(
                    childValue, key: childKey, format: objectFormat, sampleRate: sampleRate)
            }
            return .object(result)
        case .array(let values):
            if isImageKey(key), values.count > 2_048, floatData(values) != nil {
                // Numeric arrays are PCM-like values, not encoded JPEG/PNG
                // bytes.  Refuse to create a fake `.jpg` blob that downstream
                // viewers would later fail to decode.
                throw MiniCPMDemoServiceError.invalidMedia("numeric image/frame array is not encoded image data")
            }
            guard isMediaKey(key), values.count > 2_048,
                  let data = floatData(values) else {
                return .array(try await values.asyncMap {
                    try await externalize($0, key: key, format: format, sampleRate: sampleRate)
                })
            }
            let blob = try await store.storeRecordingBlob(
                sessionID: sessionID,
                data: data,
                fileExtension: extensionFor(key: key, format: format),
                mimeType: mimeTypeFor(key: key, format: format),
                role: key,
                sampleRate: sampleRate)
            return .string("@\(blob.relativePath)")
        case .string(let string):
            guard isMediaKey(key), let data = Data(base64Encoded: string), data.count > 4_096 else {
                return value
            }
            let blob = try await store.storeRecordingBlob(
                sessionID: sessionID,
                data: data,
                fileExtension: extensionFor(key: key, format: format),
                mimeType: mimeTypeFor(key: key, format: format),
                role: key,
                sampleRate: sampleRate)
            return .string("@\(blob.relativePath)")
        default:
            return value
        }
    }

    private func isMediaKey(_ key: String?) -> Bool {
        guard let key = key?.lowercased() else { return false }
        return key.contains("audio") || key.contains("pcm") || key.contains("image")
            || key.contains("frame") || key.contains("video") || key == "data"
    }

    private func isImageKey(_ key: String?) -> Bool {
        guard let key = key?.lowercased() else { return false }
        return key.contains("image") || key.contains("frame")
    }

    private func floatData(_ values: [MiniCPMJSONValue]) -> Data? {
        var data = Data(capacity: values.count * 4)
        for value in values {
            guard let number = value.numberValue, number.isFinite else { return nil }
            var sample = Float(number).bitPattern.littleEndian
            withUnsafeBytes(of: &sample) { data.append(contentsOf: $0) }
        }
        return data
    }

    private func extensionFor(key: String?, format: String?) -> String {
        let lower = (format ?? "").lowercased()
        // RealtimeSession sends raw Float32 audio under `audio` without a
        // format field.  It is still a WAV recording after storeBlob wraps
        // the raw payload, so keep the extension aligned with the MIME type.
        if lower.contains("wav") || lower.contains("pcm") { return "wav" }
        if lower.contains("webm") { return "webm" }
        if lower.contains("mp3") || lower.contains("mpeg") { return "mp3" }
        if !lower.isEmpty { return "bin" }
        if key?.lowercased().contains("audio") == true
            || key?.lowercased().contains("pcm") == true { return "wav" }
        if key?.lowercased().contains("image") == true || key?.lowercased().contains("frame") == true { return "jpg" }
        return "bin"
    }

    private func mimeTypeFor(key: String?, format: String?) -> String {
        let lower = (format ?? "").lowercased()
        if lower.contains("wav") || lower.contains("pcm") { return "audio/wav" }
        if lower.contains("webm") {
            return key?.lowercased().contains("audio") == true ? "audio/webm" : "video/webm"
        }
        if lower.contains("mp3") || lower.contains("mpeg") { return "audio/mpeg" }
        if !lower.isEmpty { return "application/octet-stream" }
        if key?.lowercased().contains("audio") == true
            || key?.lowercased().contains("pcm") == true { return "audio/wav" }
        if key?.lowercased().contains("image") == true || key?.lowercased().contains("frame") == true { return "image/jpeg" }
        if key?.lowercased().contains("video") == true { return "video/webm" }
        return "application/octet-stream"
    }
}

private extension Array {
    func asyncMap<T>(_ transform: (Element) async throws -> T) async throws -> [T] {
        var result: [T] = []
        result.reserveCapacity(count)
        for element in self { result.append(try await transform(element)) }
        return result
    }
}
