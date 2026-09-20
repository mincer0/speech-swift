import Foundation

/// A small, lossless JSON value used at the MiniCPM Demo wire boundary.
///
/// The Python backend deliberately keeps model options opaque.  Using an
/// enum instead of `[String: Any]` gives the Swift actor a Sendable payload
/// while still accepting arbitrary generation/voice/message fields.
public enum MiniCPMJSONValue: Codable, Equatable, Sendable {
    case null
    case bool(Bool)
    case number(Double)
    case string(String)
    case array([MiniCPMJSONValue])
    case object([String: MiniCPMJSONValue])

    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if container.decodeNil() {
            self = .null
        } else if let value = try? container.decode(Bool.self) {
            self = .bool(value)
        } else if let value = try? container.decode(Double.self) {
            self = .number(value)
        } else if let value = try? container.decode(String.self) {
            self = .string(value)
        } else if let value = try? container.decode([MiniCPMJSONValue].self) {
            self = .array(value)
        } else if let value = try? container.decode([String: MiniCPMJSONValue].self) {
            self = .object(value)
        } else {
            throw DecodingError.dataCorruptedError(in: container, debugDescription: "unsupported JSON value")
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .null:
            try container.encodeNil()
        case .bool(let value):
            try container.encode(value)
        case .number(let value):
            try container.encode(value)
        case .string(let value):
            try container.encode(value)
        case .array(let value):
            try container.encode(value)
        case .object(let value):
            try container.encode(value)
        }
    }

    public var stringValue: String? {
        guard case .string(let value) = self else { return nil }
        return value
    }

    public var boolValue: Bool? {
        guard case .bool(let value) = self else { return nil }
        return value
    }

    public var numberValue: Double? {
        guard case .number(let value) = self else { return nil }
        return value
    }

    public var objectValue: [String: MiniCPMJSONValue]? {
        guard case .object(let value) = self else { return nil }
        return value
    }

    public subscript(key: String) -> MiniCPMJSONValue? {
        guard case .object(let value) = self else { return nil }
        return value[key]
    }
}

public extension MiniCPMJSONValue {
    init(_ value: String) { self = .string(value) }
    init(_ value: Bool) { self = .bool(value) }
    init(_ value: Int) { self = .number(Double(value)) }
    init(_ value: Double) { self = .number(value) }
}

