import Foundation
import CryptoKit
import MLX

/// Loader for the opt-in real-model LLM golden exported by
/// ``scripts/export_minicpm_o_llm_golden.py``.  The fixture is not committed
/// into SwiftPM resources because it contains model-specific outputs and can
/// be regenerated after an intentional converter/model revision.  Tests look
/// at ``MINICPM_LLM_GOLDEN_DIR`` first, then at a bundled ``llm-golden``
/// directory when one is supplied by a downstream package.
struct MiniCPMLLMGoldenFixture {
    enum FixtureError: Error, CustomStringConvertible {
        case missing(String)
        case invalid(String)

        var description: String {
            switch self {
            case .missing(let value): return "missing golden fixture: \(value)"
            case .invalid(let value): return "invalid golden fixture: \(value)"
            }
        }
    }

    let root: URL
    let manifest: [String: Any]

    private static var verifiedModelURLs: [String: URL] = [:]

    static func locate() throws -> MiniCPMLLMGoldenFixture? {
        let fileManager = FileManager.default
        let candidates: [URL] = {
            if let value = ProcessInfo.processInfo.environment["MINICPM_LLM_GOLDEN_DIR"],
               !value.isEmpty {
                return [URL(fileURLWithPath: value, isDirectory: true)]
            }
            guard let bundled = Bundle.module.resourceURL else { return [] }
            return [bundled.appendingPathComponent("llm-golden", isDirectory: true)]
        }()
        for candidate in candidates where fileManager.fileExists(
            atPath: candidate.appendingPathComponent("manifest.json").path) {
            let data = try Data(contentsOf: candidate.appendingPathComponent("manifest.json"))
            let object = try JSONSerialization.jsonObject(with: data)
            guard let manifest = object as? [String: Any] else {
                throw FixtureError.invalid("manifest root is not an object")
            }
            return MiniCPMLLMGoldenFixture(root: candidate, manifest: manifest)
        }
        return nil
    }

    var schema: String {
        manifest["schema"] as? String ?? ""
    }

    /// Resolve the model directory and verify the manifest's provenance.
    ///
    /// An explicitly supplied ``MINICPM_LLM_GOLDEN_MODEL`` is an assertion,
    /// not a hint: empty values, non-directories, and any provenance mismatch
    /// throw ``FixtureError`` so a configured golden test cannot silently turn
    /// into an XCTest skip.  Discovery from the manifest remains best-effort;
    /// an absent or stale discovered model returns ``nil`` and lets the
    /// opt-in tests skip when no model was explicitly requested.
    func modelURL(
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) throws -> URL? {
        if let value = environment["MINICPM_LLM_GOLDEN_MODEL"] {
            guard !value.isEmpty else {
                throw FixtureError.invalid("MINICPM_LLM_GOLDEN_MODEL is empty")
            }
            return try validatedModelURL(
                URL(fileURLWithPath: value, isDirectory: true), explicit: true)
        }
        guard let model = manifest["model"] as? [String: Any],
              let relative = model["repo_relative_path"] as? String,
              !relative.isEmpty,
              !relative.hasPrefix("/"),
              !relative.split(separator: "/").contains(where: { $0 == ".." }) else { return nil }

        // The manifest contains an explicit path relative to the repository,
        // never a guessed fixture sibling.  Walk ancestors so this works for
        // both ``test-output/...`` and a SwiftPM resource bundle.  Every
        // candidate is checked against the model provenance hashes below.
        var ancestor = root
        while true {
            let candidate = ancestor.appendingPathComponent(relative, isDirectory: true)
            if let validated = try validatedModelURL(candidate, explicit: false) { return validated }
            // Foundation can return an empty URL after deleting the last
            // component of "/" on some macOS releases.  Stop at the root
            // component instead of walking that empty URL forever.
            guard ancestor.pathComponents.count > 1 else { break }
            let parent = ancestor.deletingLastPathComponent()
            if parent.path == ancestor.path { break }
            ancestor = parent
        }
        return nil
    }

    private func validatedModelURL(_ candidate: URL, explicit: Bool) throws -> URL? {
        let manager = FileManager.default
        var isDirectory: ObjCBool = false
        guard manager.fileExists(atPath: candidate.path, isDirectory: &isDirectory),
              isDirectory.boolValue else {
            if explicit {
                throw FixtureError.invalid(
                    "explicit model is not a directory: \(candidate.path)")
            }
            return nil
        }
        guard let model = manifest["model"] as? [String: Any] else {
            if explicit {
                throw FixtureError.invalid("manifest has no model provenance")
            }
            return nil
        }
        guard let expectedTree = model["tree_sha256"] as? String,
              expectedTree.count == 64,
              let expectedConfig = model["config_sha256"] as? String,
              expectedConfig.count == 64 else {
            if explicit {
                throw FixtureError.invalid("model provenance hashes are incomplete")
            }
            return nil
        }

        let cacheKey = "\(candidate.standardizedFileURL.path)#\(expectedTree)#\(expectedConfig)"
        if !explicit, let cached = Self.verifiedModelURLs[cacheKey] {
            return cached
        }
        do {
            let config = candidate.appendingPathComponent("config.json")
            guard manager.fileExists(atPath: config.path) else {
                if explicit {
                    throw FixtureError.invalid(
                        "explicit model is missing config.json: \(candidate.path)")
                }
                return nil
            }
            let actualConfig = try Self.sha256File(config)
            guard actualConfig == expectedConfig else {
                if explicit {
                    throw FixtureError.invalid(
                        "model config sha256 mismatch: expected \(expectedConfig), got \(actualConfig)")
                }
                return nil
            }
            let actualTree = try Self.sha256Tree(candidate)
            guard actualTree == expectedTree else {
                if explicit {
                    throw FixtureError.invalid(
                        "model tree sha256 mismatch: expected \(expectedTree), got \(actualTree)")
                }
                return nil
            }
        } catch let error as FixtureError {
            throw error
        } catch {
            if explicit {
                throw FixtureError.invalid(
                    "cannot verify explicit model provenance: \(error.localizedDescription)")
            }
            return nil
        }
        if !explicit {
            Self.verifiedModelURLs[cacheKey] = candidate
        }
        return candidate
    }

    func object(at path: [String]) throws -> [String: Any] {
        var current: Any = manifest
        for key in path {
            guard let object = current as? [String: Any], let next = object[key] else {
                throw FixtureError.missing(path.joined(separator: "."))
            }
            current = next
        }
        guard let object = current as? [String: Any] else {
            throw FixtureError.invalid("\(path.joined(separator: ".")) is not an object")
        }
        return object
    }

    func value(at path: [String]) throws -> Any {
        var current: Any = manifest
        for key in path {
            guard let object = current as? [String: Any], let next = object[key] else {
                throw FixtureError.missing(path.joined(separator: "."))
            }
            current = next
        }
        return current
    }

    /// Read a JSON integer while keeping the call sites small and producing
    /// a useful fixture error instead of a failed cast buried in a test.
    func intValue(_ object: [String: Any], _ key: String) throws -> Int {
        guard let value = object[key] as? NSNumber else {
            throw FixtureError.invalid("\(key) is not an integer")
        }
        return value.intValue
    }

    func floatValue(_ object: [String: Any], _ key: String) throws -> Float {
        guard let value = object[key] as? NSNumber else {
            throw FixtureError.invalid("\(key) is not a number")
        }
        return value.floatValue
    }

    func optionalIntValue(_ object: [String: Any], _ key: String) throws -> Int? {
        guard let value = object[key], !(value is NSNull) else { return nil }
        guard let number = value as? NSNumber else {
            throw FixtureError.invalid("\(key) is not an integer or null")
        }
        return number.intValue
    }

    func intValues(_ object: [String: Any], _ key: String) throws -> [Int] {
        guard let values = object[key] as? [NSNumber] else {
            throw FixtureError.invalid("\(key) is not an integer array")
        }
        return values.map(\.intValue)
    }

    func stringValue(_ object: [String: Any], _ key: String) throws -> String {
        guard let value = object[key] as? String else {
            throw FixtureError.invalid("\(key) is not a string")
        }
        return value
    }

    func scenario(_ name: String) throws -> [String: Any] {
        try object(at: ["scenarios", name])
    }

    func arraySpec(_ name: String) throws -> [String: Any] {
        try object(at: ["arrays", name])
    }

    func arrayPath(_ name: String) throws -> URL {
        try validatedArraySpec(name, expectedDType: nil).path
    }

    func arrayShape(_ name: String) throws -> [Int] {
        try validatedArraySpec(name, expectedDType: nil).shape
    }

    func floatArray(_ name: String) throws -> [Float] {
        let data = try validatedArrayData(name, expectedDType: "float32-le")
        var values: [Float] = []
        values.reserveCapacity(data.count / MemoryLayout<UInt32>.size)
        for offset in stride(from: 0, to: data.count, by: MemoryLayout<UInt32>.size) {
            let raw = data.withUnsafeBytes {
                $0.loadUnaligned(fromByteOffset: offset, as: UInt32.self)
            }
            values.append(Float(bitPattern: UInt32(littleEndian: raw)))
        }
        return values
    }

    func intArray(_ name: String) throws -> [Int] {
        let data = try validatedArrayData(name, expectedDType: "int32-le")
        var values: [Int] = []
        values.reserveCapacity(data.count / MemoryLayout<Int32>.size)
        for offset in stride(from: 0, to: data.count, by: MemoryLayout<Int32>.size) {
            let raw = data.withUnsafeBytes {
                $0.loadUnaligned(fromByteOffset: offset, as: Int32.self)
            }
            values.append(Int(Int32(littleEndian: raw)))
        }
        return values
    }

    private struct ArraySpec {
        let path: URL
        let shape: [Int]
        let dtype: String
        let count: Int
        let sha256: String
    }

    private func validatedArraySpec(
        _ name: String,
        expectedDType: String?
    ) throws -> ArraySpec {
        let object = try arraySpec(name)
        guard let relative = object["path"] as? String,
              !relative.isEmpty,
              !relative.hasPrefix("/"),
              !relative.split(separator: "/").contains(where: { $0 == ".." }),
              let values = object["shape"] as? [NSNumber],
              let dtype = object["dtype"] as? String,
              let countNumber = object["count"] as? NSNumber,
              let digest = object["sha256"] as? String else {
            throw FixtureError.invalid("array \(name) metadata is incomplete")
        }
        let shape = values.map(\.intValue)
        guard shape.allSatisfy({ $0 >= 0 }) else {
            throw FixtureError.invalid("array \(name) has a negative shape")
        }
        var shapeCount = 1
        for value in shape {
            let product = shapeCount.multipliedReportingOverflow(by: value)
            guard !product.overflow else {
                throw FixtureError.invalid("array \(name) shape overflows Int")
            }
            shapeCount = product.partialValue
        }
        let count = countNumber.intValue
        guard shapeCount == count else {
            throw FixtureError.invalid("array \(name) shape/count mismatch")
        }
        if let expectedDType, dtype != expectedDType {
            throw FixtureError.invalid(
                "array \(name) dtype \(dtype), expected \(expectedDType)")
        }
        guard digest.count == 64 else {
            throw FixtureError.invalid("array \(name) has invalid sha256")
        }
        let path = root.appendingPathComponent(relative)
        return ArraySpec(path: path, shape: shape, dtype: dtype, count: count, sha256: digest)
    }

    private func validatedArrayData(_ name: String, expectedDType: String) throws -> Data {
        let spec = try validatedArraySpec(name, expectedDType: expectedDType)
        let data = try Data(contentsOf: spec.path, options: [.mappedIfSafe])
        let elementSize: Int
        switch spec.dtype {
        case "float32-le", "int32-le": elementSize = MemoryLayout<UInt32>.size
        default: throw FixtureError.invalid("array \(name) has unsupported dtype \(spec.dtype)")
        }
        let byteCount = spec.count.multipliedReportingOverflow(by: elementSize)
        guard !byteCount.overflow, data.count == byteCount.partialValue else {
            throw FixtureError.invalid("array \(name) byte count/shape mismatch")
        }
        let actual = Self.sha256(data)
        guard actual == spec.sha256 else {
            throw FixtureError.invalid("array \(name) sha256 mismatch")
        }
        return data
    }

    /// Validate every binary artifact referenced by the manifest.  Golden
    /// tests call this once before loading a model so a truncated or swapped
    /// fixture cannot silently produce a misleading numerical failure later.
    func validateAllArrays() throws {
        guard let arrays = manifest["arrays"] as? [String: Any] else {
            throw FixtureError.invalid("manifest has no arrays")
        }
        for name in arrays.keys.sorted() {
            let spec = try arraySpec(name)
            guard let dtype = spec["dtype"] as? String else {
                throw FixtureError.invalid("array \(name) has no dtype")
            }
            _ = try validatedArrayData(name, expectedDType: dtype)
        }
    }

    func jsonFile(_ relativePath: String) throws -> Any {
        let data = try Data(contentsOf: root.appendingPathComponent(relativePath))
        return try JSONSerialization.jsonObject(with: data)
    }

    func output(_ object: [String: Any]) throws -> GoldenOutput {
        guard let logits = object["logits_selected"] as? String,
              let hidden = object["hidden"] as? String,
              let top20 = object["top20"] as? String,
              let top20Values = object["top20_values"] as? String,
              let selected = object["selected_indices"] as? [NSNumber] else {
            throw FixtureError.invalid("output record is incomplete")
        }
        return GoldenOutput(
            logitsSelected: logits,
            hidden: hidden,
            top20: top20,
            top20Values: top20Values,
            selectedIndices: selected.map(\.intValue))
    }

    struct GoldenOutput {
        let logitsSelected: String
        let hidden: String
        let top20: String
        let top20Values: String
        let selectedIndices: [Int]
    }

    private static func sha256(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    private static func sha256File(_ url: URL) throws -> String {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        var digest = SHA256()
        while let chunk = try handle.read(upToCount: 8 * 1024 * 1024), !chunk.isEmpty {
            digest.update(data: chunk)
        }
        return digest.finalize().map { String(format: "%02x", $0) }.joined()
    }

    private static func sha256Tree(_ root: URL) throws -> String {
        let manager = FileManager.default
        let relativePaths = (manager.subpaths(atPath: root.path) ?? []).filter { relative in
            var isDirectory: ObjCBool = false
            let url = root.appendingPathComponent(relative)
            return manager.fileExists(atPath: url.path, isDirectory: &isDirectory)
                && !isDirectory.boolValue
        }.sorted()
        var digest = SHA256()
        for relative in relativePaths {
            let relativeData = Data(relative.utf8)
            var length = UInt64(relativeData.count).littleEndian
            withUnsafeBytes(of: &length) { digest.update(data: Data($0)) }
            digest.update(data: relativeData)
            guard let fileDigest = Data(hexString: try sha256File(
                root.appendingPathComponent(relative))) else {
                throw FixtureError.invalid(
                    "invalid SHA-256 digest for model file \(relative)")
            }
            digest.update(data: fileDigest)
        }
        return digest.finalize().map { String(format: "%02x", $0) }.joined()
    }
}

private extension Data {
    init?(hexString: String) {
        guard hexString.count.isMultiple(of: 2) else { return nil }
        self.init(capacity: hexString.count / 2)
        var index = hexString.startIndex
        while index < hexString.endIndex {
            let next = hexString.index(index, offsetBy: 2)
            guard let byte = UInt8(hexString[index..<next], radix: 16) else { return nil }
            append(byte)
            index = next
        }
    }
}
