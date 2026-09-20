import Foundation

public enum MiniCPMBackendMode: String, Codable, Sendable {
    case turnBased = "turn_based"
    case halfDuplex = "half_duplex"
    case fullDuplex = "full_duplex"
    /// Omni/video full-duplex is kept as a distinct protocol mode.  It shares
    /// the duplex scheduler with audio, but the runtime can use the value to
    /// select the vision/TDM input path and to report the actual capability in
    /// ``session.created`` instead of silently relabelling video as audio.
    case omni = "omni"

    public init(rawOrAlias value: String?) throws {
        let normalized = (value ?? "full_duplex").lowercased().replacingOccurrences(of: "-", with: "_")
        switch normalized {
        case "turn_based", "turn", "chat": self = .turnBased
        case "half_duplex", "half", "streaming": self = .halfDuplex
        case "full_duplex", "full", "duplex", "audio": self = .fullDuplex
        case "omni", "video": self = .omni
        default: throw MiniCPMDemoBackendError.unsupportedMode(value ?? "")
        }
    }
}

/// Public mode names used by the Gateway/OpenAI-Realtime compatible endpoint.
/// They intentionally remain separate from ``MiniCPMBackendMode``: `video`
/// and `audio` are public transport capabilities, while the backend receives
/// `omni` and `full_duplex` respectively.  Keeping this conversion explicit
/// prevents a video connection from being silently relabelled as audio.
public enum MiniCPMRealtimeMode: String, Codable, Sendable, Equatable {
    case chat
    /// Explicit half-duplex compatibility extension.  The pinned Gateway
    /// exposes only chat/video/audio, but the legacy half-duplex UI uses a
    /// separate mode.  Keeping it as a transport mode lets a real
    /// HalfCoordinator be injected later without pretending that full-duplex
    /// listen/speak events are half-duplex VAD events.
    case half
    case video
    case audio

    public init(rawOrAlias value: String?) throws {
        let normalized = (value ?? "video")
            .lowercased()
            .replacingOccurrences(of: "-", with: "_")
        switch normalized {
        case "chat", "turn", "turn_based": self = .chat
        case "half", "half_duplex", "streaming": self = .half
        case "video", "omni", "omni_duplex": self = .video
        case "audio", "full", "full_duplex", "audio_duplex": self = .audio
        default:
            throw MiniCPMDemoBackendError.unsupportedMode(value ?? "")
        }
    }

    public var backendMode: MiniCPMBackendMode {
        switch self {
        case .chat: return .turnBased
        case .half: return .halfDuplex
        case .video: return .omni
        case .audio: return .fullDuplex
        }
    }
}

public enum MiniCPMDemoBackendError: Error, LocalizedError, Sendable {
    case invalidMessage(String)
    case invalidState(String)
    case unsupportedMode(String)
    case engine(String)

    public var errorDescription: String? {
        switch self {
        case .invalidMessage(let message): return message
        case .invalidState(let message): return message
        case .unsupportedMode(let mode): return "unsupported MiniCPM Demo mode: \(mode)"
        case .engine(let message): return message
        }
    }
}

/// Raw client envelope.  `payload` and `input` remain dynamic to preserve the
/// Python Demo's public API when new fields are added.
public struct MiniCPMDemoRequest: Codable, Sendable {
    public var type: String
    public var payload: MiniCPMJSONValue?
    public var input: MiniCPMJSONValue?
    public var messageID: String?
    public var reason: String?

    enum CodingKeys: String, CodingKey {
        case type
        case payload
        case input
        case messageID = "message_id"
        case reason
    }

    public init(type: String, payload: MiniCPMJSONValue? = nil, input: MiniCPMJSONValue? = nil, messageID: String? = nil, reason: String? = nil) {
        self.type = type
        self.payload = payload
        self.input = input
        self.messageID = messageID
        self.reason = reason
    }

    public var payloadObject: [String: MiniCPMJSONValue] {
        if let object = payload?.objectValue { return object }
        if let object = input?.objectValue { return object }
        return [:]
    }
}

public struct MiniCPMDemoEvent: Codable, Sendable, Equatable {
    public var version: String = "backend.class.v1"
    public var eventID: String
    public var type: String
    public var timestampMS: Int64
    public var payload: [String: MiniCPMJSONValue]
    public var sessionID: String?
    public var responseID: String?
    public var inputID: String?

    // Realtime clients also expect these common fields at the top level.
    public var kind: String?
    public var text: String?
    public var audio: String?
    public var format: String?
    public var sampleRate: Int?
    public var reason: String?
    public var metrics: [String: MiniCPMJSONValue]?
    public var state: String?
    public var error: MiniCPMJSONValue?

    enum CodingKeys: String, CodingKey {
        case version
        case eventID = "event_id"
        case type
        case timestampMS = "timestamp_ms"
        case payload
        case sessionID = "session_id"
        case responseID = "response_id"
        case inputID = "input_id"
        case kind, text, audio, format
        case sampleRate = "sample_rate"
        case reason, metrics, state, error
    }

    public init(eventID: String, type: String, timestampMS: Int64 = Int64(Date().timeIntervalSince1970 * 1000), payload: [String: MiniCPMJSONValue] = [:], sessionID: String? = nil, responseID: String? = nil, inputID: String? = nil) {
        self.eventID = eventID
        self.type = type
        self.timestampMS = timestampMS
        self.payload = payload
        self.sessionID = sessionID
        self.responseID = responseID
        self.inputID = inputID
        self.kind = payload["kind"]?.stringValue
        self.text = payload["text"]?.stringValue
        self.audio = payload["audio"]?.stringValue
        self.format = payload["format"]?.stringValue
        self.sampleRate = payload["sample_rate"]?.numberValue.map(Int.init)
        self.reason = payload["reason"]?.stringValue
        self.metrics = payload["metrics"]?.objectValue
        self.state = payload["state"]?.stringValue
        self.error = payload["error"]
    }
}

public struct MiniCPMDemoOutput: Sendable, Equatable {
    public var kind: String
    public var text: String?
    /// Audio is already base64 PCM at the protocol boundary.
    public var audio: String?
    public var format: String = "pcm_f32"
    public var sampleRate: Int = 24_000
    public var done: Bool = false
    /// Internal transaction marker.  A streaming spoken unit can span many
    /// chunks without a wire-level response.done; the transport still has to
    /// commit each successfully written unit before accepting the next one.
    public var needsFinalize: Bool = true
    public var reason: String?
    public var metrics: [String: MiniCPMJSONValue] = [:]

    public init(kind: String, text: String? = nil, audio: String? = nil, format: String = "pcm_f32", sampleRate: Int = 24_000, done: Bool = false, needsFinalize: Bool = true, reason: String? = nil, metrics: [String: MiniCPMJSONValue] = [:]) {
        self.kind = kind
        self.text = text
        self.audio = audio
        self.format = format
        self.sampleRate = sampleRate
        self.done = done
        self.needsFinalize = needsFinalize
        self.reason = reason
        self.metrics = metrics
    }
}

/// Optional event sink used by the native server to produce a deterministic
/// session trace.  Recording is deliberately orthogonal to inference: a
/// recorder receives the already-normalized protocol event and must never
/// inspect or mutate model state.
public protocol MiniCPMDemoEventRecorder: Sendable {
    func record(_ event: MiniCPMDemoEvent)
    func finish()
}

/// Small JSONL recorder suitable for local diagnostics and E2E harnesses.
/// It serializes each event as one line and uses a lock so the sink remains
/// safe when a transport/defer task closes a session while an event is being
/// emitted.  It intentionally records protocol events, not raw microphone
/// bytes; callers that need media traces can add their own input/output sink
/// at the transport boundary.
public final class MiniCPMDemoJSONLRecorder: MiniCPMDemoEventRecorder, @unchecked Sendable {
    private let handle: FileHandle
    private let lock = NSLock()
    private var finished = false

    public init(url: URL, createIntermediateDirectories: Bool = true) throws {
        if createIntermediateDirectories {
            try FileManager.default.createDirectory(
                at: url.deletingLastPathComponent(),
                withIntermediateDirectories: true)
        }
        if !FileManager.default.fileExists(atPath: url.path) {
            guard FileManager.default.createFile(atPath: url.path, contents: nil) else {
                throw MiniCPMDemoBackendError.engine("cannot create recording file: \(url.path)")
            }
        }
        self.handle = try FileHandle(forWritingTo: url)
        try self.handle.seekToEnd()
    }

    deinit { try? handle.close() }

    public func record(_ event: MiniCPMDemoEvent) {
        lock.lock()
        defer { lock.unlock() }
        guard !finished else { return }
        do {
            var data = try JSONEncoder().encode(event)
            data.append(0x0a)
            try handle.write(contentsOf: data)
        } catch {
            // A diagnostics sink must not take down a live inference session.
            // The transport still exposes the event; callers can inspect the
            // file/OS error separately if they need strict recording.
        }
    }

    public func finish() {
        lock.lock()
        defer { lock.unlock() }
        guard !finished else { return }
        finished = true
        try? handle.synchronize()
        try? handle.close()
    }
}

/// Decode binary fields at the transport boundary.  Model adapters therefore
/// receive PCM/sample or byte arrays and never mistake a base64 payload for a
/// filesystem path.  Unrecognised strings (including real paths) are kept
/// unchanged for backwards compatibility.
public enum MiniCPMDemoWireDecoder {
    public static func normalizeObject(_ object: [String: MiniCPMJSONValue]) -> [String: MiniCPMJSONValue] {
        object.reduce(into: [String: MiniCPMJSONValue]()) { result, entry in
            let (key, value) = entry
            result[key] = normalize(value, key: key)
        }
    }

    /// Decode the pinned `/backend` audio schema.  The official transport is
    /// deliberately narrower than the legacy/UI helpers below: audio is a
    /// base64 encoded, little-endian IEEE-754 Float32 stream.  In particular,
    /// WAV containers and signed PCM16 are not wire-compatible and must not be
    /// guessed here.  Keeping this as a throwing entry point lets the WebSocket
    /// handler fail before a malformed request reaches the model/frontend.
    public static func normalizeObjectStrict(
        _ object: [String: MiniCPMJSONValue]
    ) throws -> [String: MiniCPMJSONValue] {
        var normalized: [String: MiniCPMJSONValue] = [:]
        normalized.reserveCapacity(object.count)
        for (key, value) in object {
            normalized[key] = try normalizeStrict(value, key: key, path: key)
        }
        return normalized
    }

    public static func normalize(_ value: MiniCPMJSONValue, key: String? = nil) -> MiniCPMJSONValue {
        switch value {
        case .object(let object):
            // Demo clients commonly wrap a voice/audio payload as
            // `{data|base64|audio_base64: "...", format: ...}`.  Looking
            // only at the nested key (`data`) would leave the base64 string
            // opaque and the native audio encoder would receive a path-like
            // value.  Decode the wrapper at the boundary, just like the
            // Python server's `_decode_audio` does.
            if isAudioKey(key),
               let encoded = object["data"]?.stringValue
                    ?? object["base64"]?.stringValue
                    ?? object["audio_base64"]?.stringValue
                    ?? object["pcm"]?.stringValue,
               let pcm = decodePCM(encoded, format: object["format"]?.stringValue) {
                return .array(pcm.map { .number(Double($0)) })
            }
            return .object(normalizeObject(object))
        case .array(let values):
            return .array(values.map { normalize($0, key: key) })
        case .string(let string):
            guard let key = key?.lowercased() else { return value }
            if isAudioKey(key) {
                if let pcm = decodePCM(string) { return .array(pcm.map { .number(Double($0)) }) }
            }
            if key.contains("frame") || key.contains("image") {
                if let bytes = Data(base64Encoded: string), !bytes.isEmpty {
                    return .array(bytes.map { .number(Double($0)) })
                }
            }
            return value
        default:
            return value
        }
    }

    private static func normalizeStrict(
        _ value: MiniCPMJSONValue,
        key: String?,
        path: String
    ) throws -> MiniCPMJSONValue {
        switch value {
        case .object(let object):
            // Content items carry their encoding in `data`, rather than in a
            // field named `audio`.  Decode by type before recursively walking
            // the object so this path cannot silently leave a base64 string
            // for the native adapter to mistake for a filesystem path.
            if object["type"]?.stringValue?.lowercased() == "audio" {
                guard let encoded = object["data"]?.stringValue
                    ?? object["audio"]?.stringValue
                    ?? object["base64"]?.stringValue else {
                    throw MiniCPMDemoBackendError.invalidMessage(
                        "audio content requires a base64 data string")
                }
                let pcm = try decodeStrictFloat32(encoded, format: object["format"]?.stringValue, path: path)
                var result = object
                result["data"] = .array(pcm.map { .number(Double($0)) })
                // Do not recursively reinterpret arbitrary metadata fields.
                // They remain available to the chat/template layer unchanged.
                return .object(result)
            }

            if isAudioKey(key),
               let encodedKey = ["data", "base64", "audio_base64", "pcm"].first(where: {
                   object[$0]?.stringValue != nil
               }),
               let encoded = object[encodedKey]?.stringValue {
                let pcm = try decodeStrictFloat32(encoded, format: object["format"]?.stringValue, path: path)
                var result = object
                result[encodedKey] = .array(pcm.map { .number(Double($0)) })
                // Keep sample-rate/format metadata alongside the decoded
                // samples; callers such as voice.ref_audio need both.
                return .object(result)
            }

            var result: [String: MiniCPMJSONValue] = [:]
            result.reserveCapacity(object.count)
            for (childKey, childValue) in object {
                result[childKey] = try normalizeStrict(
                    childValue,
                    key: childKey,
                    path: "nested audio field")
            }
            return .object(result)
        case .array(let values):
            var normalized: [MiniCPMJSONValue] = []
            normalized.reserveCapacity(values.count)
            for child in values {
                normalized.append(try normalizeStrict(
                    child,
                    key: key,
                    path: "nested audio array"))
            }
            return .array(normalized)
        case .string(let string):
            guard isAudioKey(key) else { return value }
            let pcm = try decodeStrictFloat32(string, format: nil, path: path)
            return .array(pcm.map { .number(Double($0)) })
        default:
            return value
        }
    }

    private static func isAudioKey(_ key: String?) -> Bool {
        guard let key else { return false }
        let normalized = key.lowercased()
        return normalized.contains("audio")
            || normalized.contains("voice")
            || normalized.contains("pcm")
            || normalized == "ref"
            || normalized == "reference"
    }

    private static func decodePCM(_ string: String, format: String? = nil) -> [Float]? {
        guard let data = Data(base64Encoded: string), !data.isEmpty else { return nil }
        // Python's wire decoder accepts WAV PCM16 as well as raw Float32.
        if data.count >= 44, data.prefix(4) == Data("RIFF".utf8), data[8..<12] == Data("WAVE".utf8) {
            return decodeWAVPCM16(data)
        }
        let format = format?.lowercased() ?? ""
        if format.contains("s16") || format.contains("pcm16") || format.contains("int16") {
            return data.count.isMultiple(of: 2) ? decodePCM16(data[...]) : nil
        }
        if format.contains("f32") || format.contains("float32") {
            return data.count.isMultiple(of: 4) ? decodeFloat32(data[...]) : nil
        }
        if data.count % 4 == 0 {
            return decodeFloat32(data[...])
        }
        guard data.count % 2 == 0 else { return nil }
        return decodePCM16(data[...])
    }

    private static func decodeStrictFloat32(
        _ encoded: String,
        format: String?,
        path: String
    ) throws -> [Float] {
        guard let data = Data(base64Encoded: encoded) else {
            throw MiniCPMDemoBackendError.invalidMessage(
                "audio data is not valid base64")
        }
        guard !data.isEmpty else {
            throw MiniCPMDemoBackendError.invalidMessage(
                "audio data must not be empty")
        }
        if data.count >= 12,
           data.prefix(4) == Data("RIFF".utf8),
           data[8..<12] == Data("WAVE".utf8) {
            throw MiniCPMDemoBackendError.invalidMessage(
                "WAV containers are not accepted; send raw little-endian Float32 PCM")
        }
        if let format {
            let normalized = format.lowercased()
            if normalized.contains("s16") || normalized.contains("pcm16")
                || normalized.contains("int16") || normalized.contains("wav") {
                throw MiniCPMDemoBackendError.invalidMessage(
                    "audio format is not accepted; send raw little-endian Float32 PCM")
            }
            if !normalized.isEmpty,
               !normalized.contains("f32"),
               !normalized.contains("float32"),
               !normalized.contains("float") {
                throw MiniCPMDemoBackendError.invalidMessage(
                    "unsupported audio format; expected Float32 PCM")
            }
        }
        guard data.count.isMultiple(of: 4) else {
            throw MiniCPMDemoBackendError.invalidMessage(
                "Float32 PCM byte length must be divisible by 4")
        }
        let samples = decodeFloat32(data[...])
        guard !samples.isEmpty, samples.allSatisfy(\.isFinite) else {
            throw MiniCPMDemoBackendError.invalidMessage(
                "Float32 PCM contains NaN or infinity")
        }
        return samples
    }

    private static func decodeWAVPCM16(_ data: Data) -> [Float]? {
        // Walk RIFF chunks instead of assuming a fixed 44-byte header; LIST,
        // JUNK and extensible fmt chunks are legal in recorded references.
        var offset = 12
        var channels = 1
        var bitsPerSample = 16
        var payload: Data.SubSequence?
        while offset + 8 <= data.count {
            let id = String(data: data[offset..<(offset + 4)], encoding: .ascii)
            let size = data[(offset + 4)..<(offset + 8)].withUnsafeBytes {
                $0.loadUnaligned(as: UInt32.self)
            }
            let start = offset + 8
            let end = start + Int(size)
            guard end >= start, end <= data.count else { return nil }
            if id == "fmt ", end - start >= 16 {
                let fmt = data[start..<(start + 2)].withUnsafeBytes {
                    $0.loadUnaligned(as: UInt16.self)
                }
                channels = Int(data[(start + 2)..<(start + 4)].withUnsafeBytes {
                    $0.loadUnaligned(as: UInt16.self)
                })
                bitsPerSample = Int(data[(start + 14)..<(start + 16)].withUnsafeBytes {
                    $0.loadUnaligned(as: UInt16.self)
                })
                guard fmt == 1, channels > 0 else { return nil }
            } else if id == "data" {
                payload = data[start..<end]
                break
            }
            // RIFF chunks are word aligned.
            offset = end + (end.isMultiple(of: 2) ? 0 : 1)
        }
        guard bitsPerSample == 16, let payload, payload.count >= 2 else { return nil }
        let interleaved = decodePCM16(payload)
        guard channels > 1 else { return interleaved }
        let frames = interleaved.count / channels
        guard frames > 0 else { return [] }
        return (0..<frames).map { frame in
            let start = frame * channels
            return interleaved[start..<(start + channels)].reduce(0, +) / Float(channels)
        }
    }

    private static func decodeFloat32(_ data: Data.SubSequence) -> [Float] {
        guard data.count >= 4 else { return [] }
        return stride(from: data.startIndex, to: data.endIndex - 3, by: 4).map { index in
            let bits = UInt32(data[index])
                | UInt32(data[index + 1]) << 8
                | UInt32(data[index + 2]) << 16
                | UInt32(data[index + 3]) << 24
            return Float(bitPattern: bits)
        }
    }

    private static func decodePCM16(_ data: Data.SubSequence) -> [Float] {
        guard data.count >= 2 else { return [] }
        return stride(from: data.startIndex, to: data.endIndex - 1, by: 2).map { index in
            let bits = UInt16(data[index]) | UInt16(data[index + 1]) << 8
            return Float(Int16(bitPattern: bits)) / 32768.0
        }
    }
}
