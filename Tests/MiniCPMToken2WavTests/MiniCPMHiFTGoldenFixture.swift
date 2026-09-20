import CryptoKit
import Foundation
import MLX

/// Fail-closed loader for the deterministic official PyTorch HiFT oracle.
struct MiniCPMHiFTGoldenFixture {
    enum FixtureError: Error, CustomStringConvertible {
        case missing(String)
        case invalid(String)

        var description: String {
            switch self {
            case .missing(let value): return "missing MiniCPM HiFT golden: \(value)"
            case .invalid(let value): return "invalid MiniCPM HiFT golden: \(value)"
            }
        }
    }

    static let schema = "minicpm-o-hift-golden/v1"
    static let upstreamRevision =
        "ba7fa9cc6ad63c894f1bd5e5afac28466953519d"
    static let upstreamRepository = "OpenBMB/MiniCPM-o-Demo"

    let root: URL
    let manifest: [String: Any]
    let arrays: [String: MLXArray]

    static func locate() throws -> MiniCPMHiFTGoldenFixture? {
        let manager = FileManager.default
        guard let configured = ProcessInfo.processInfo.environment[
            "MINICPM_HIFT_GOLDEN_DIR"], !configured.isEmpty else {
            return nil
        }
        let root = URL(fileURLWithPath: configured, isDirectory: true)
        let manifestURL: URL? = {
            let canonical = root.appendingPathComponent("manifest.json")
            if manager.fileExists(atPath: canonical.path) { return canonical }
            return (try? manager.contentsOfDirectory(
                at: root, includingPropertiesForKeys: nil,
                options: [.skipsHiddenFiles]))?.first {
                $0.lastPathComponent.hasSuffix(".manifest.json")
            }
        }()
        guard let manifestURL else {
            throw FixtureError.missing(
                "manifest.json or *.manifest.json in MINICPM_HIFT_GOLDEN_DIR=\(configured)")
        }
        let object = try JSONSerialization.jsonObject(
            with: Data(contentsOf: manifestURL))
        guard let manifest = object as? [String: Any] else {
            throw FixtureError.invalid("manifest root is not an object")
        }
        guard let artifact = manifest["artifact"] as? [String: Any],
              let relative = artifact["path"] as? String,
              isSafeRelativePath(relative) else {
            throw FixtureError.invalid("artifact path is missing or absolute")
        }
        let artifactURL = root.appendingPathComponent(relative)
        guard manager.fileExists(atPath: artifactURL.path) else {
            throw FixtureError.missing("artifact \(relative)")
        }
        return MiniCPMHiFTGoldenFixture(
            root: root,
            manifest: manifest,
            arrays: try MLX.loadArrays(url: artifactURL))
    }

    static func isSafeRelativePath(_ value: String) -> Bool {
        !value.isEmpty
            && !value.hasPrefix("/")
            && !value.split(separator: "/").contains(where: { $0 == ".." })
    }

    func tensor(_ name: String) throws -> MLXArray {
        guard let value = arrays[name] else {
            throw FixtureError.missing("tensor \(name)")
        }
        return value
    }

    func tensorSpec(_ name: String) throws -> [String: Any] {
        guard let artifact = manifest["artifact"] as? [String: Any],
              let tensors = artifact["tensors"] as? [String: Any],
              let object = tensors[name] as? [String: Any] else {
            throw FixtureError.missing("manifest tensor \(name)")
        }
        return object
    }

    func validateAll() throws {
        guard manifest["schema"] as? String == Self.schema,
              manifest["oracle_backend"] as? String == "official_pytorch" else {
            throw FixtureError.invalid("unsupported schema or oracle backend")
        }
        try validateNoAbsolutePathStrings(manifest)
        guard let upstream = manifest["upstream"] as? [String: Any],
              upstream["repository"] as? String == Self.upstreamRepository,
              upstream["revision_kind"] as? String == "git_commit",
              upstream["revision"] as? String == Self.upstreamRevision,
              validDigest(upstream["tree_sha256"]) else {
            throw FixtureError.invalid("official upstream provenance is incomplete")
        }
        guard let source = manifest["source"] as? [String: Any],
              validDigest(source["model_tree_sha256"]),
              validDigest(source["config_sha256"]),
              validDigest(source["index_sha256"]),
              validDigest(source["token2wav_tree_sha256"]) else {
            throw FixtureError.invalid("model provenance hashes are incomplete")
        }
        guard let shards = source["shards"] as? [[String: Any]], !shards.isEmpty else {
            throw FixtureError.invalid("model shard provenance is missing")
        }
        for shard in shards {
            guard let path = shard["path"] as? String,
                  Self.isSafeRelativePath(path),
                  shard["size"] is NSNumber,
                  validDigest(shard["sha256"]) else {
                throw FixtureError.invalid("invalid model shard provenance")
            }
        }
        guard let token2wavFiles = source["token2wav_files"] as? [[String: Any]],
              !token2wavFiles.isEmpty else {
            throw FixtureError.invalid("token2wav input provenance is missing")
        }
        for file in token2wavFiles {
            guard let path = file["path"] as? String,
                  Self.isSafeRelativePath(path),
                  file["size"] is NSNumber,
                  validDigest(file["sha256"]) else {
                throw FixtureError.invalid("invalid token2wav input provenance")
            }
        }
        guard let converter = manifest["converter"] as? [String: Any],
              let converterPath = converter["path"] as? String,
              Self.isSafeRelativePath(converterPath),
              validDigest(converter["sha256"]) else {
            throw FixtureError.invalid("converter provenance is incomplete")
        }
        guard let runtime = manifest["runtime"] as? [String: Any],
              runtime["output_dtype"] as? String == "float32",
              runtime["seed"] is NSNumber,
              runtime["random_algorithm"] as? String
                == "torch_cpu_default_generator" else {
            throw FixtureError.invalid("deterministic runtime metadata is incomplete")
        }

        guard let artifact = manifest["artifact"] as? [String: Any],
              let artifactPath = artifact["path"] as? String,
              Self.isSafeRelativePath(artifactPath),
              validDigest(artifact["sha256"]),
              let tensorMap = artifact["tensors"] as? [String: Any],
              let count = artifact["tensor_count"] as? NSNumber,
              count.intValue == tensorMap.count else {
            throw FixtureError.invalid("artifact metadata is incomplete")
        }
        let artifactURL = root.appendingPathComponent(artifactPath)
        guard Self.sha256File(artifactURL) == artifact["sha256"] as? String else {
            throw FixtureError.invalid("artifact sha256 mismatch")
        }
        guard Set(arrays.keys) == Set(tensorMap.keys) else {
            throw FixtureError.invalid("manifest/artifact tensor names differ")
        }
        for name in tensorMap.keys.sorted() {
            guard let array = arrays[name] else {
                throw FixtureError.missing("tensor \(name)")
            }
            let spec = try tensorSpec(name)
            try validateTensor(array, spec: spec, name: name)
        }
    }

    private func validateTensor(
        _ value: MLXArray, spec: [String: Any], name: String
    ) throws {
        guard let shape = spec["shape"] as? [NSNumber],
              let dtype = spec["dtype"] as? String,
              let numel = spec["numel"] as? NSNumber,
              let byteLength = spec["byte_length"] as? NSNumber,
              validDigest(spec["sha256"]),
              spec["finite"] as? Bool == true else {
            throw FixtureError.invalid("tensor \(name) metadata is incomplete")
        }
        guard shape.map(\.intValue) == value.shape,
              numel.intValue == value.size,
              dtype == dtypeName(value.dtype) else {
            throw FixtureError.invalid("tensor \(name) shape/dtype mismatch")
        }
        eval(value)
        let bytes = try rawBytes(value, dtype: dtype)
        guard bytes.count == byteLength.intValue,
              Self.sha256(bytes) == spec["sha256"] as? String else {
            throw FixtureError.invalid("tensor \(name) byte/hash mismatch")
        }
        let values = value.asType(.float32).asArray(Float.self)
        guard values.allSatisfy(\.isFinite) else {
            throw FixtureError.invalid("tensor \(name) contains non-finite values")
        }
    }

    private func rawBytes(_ value: MLXArray, dtype: String) throws -> Data {
        switch dtype {
        case "float32":
            var data = Data(capacity: value.size * MemoryLayout<Float>.size)
            for item in value.asType(.float32).asArray(Float.self) {
                var bits = item.bitPattern.littleEndian
                withUnsafeBytes(of: &bits) { data.append(contentsOf: $0) }
            }
            return data
        case "int32":
            var data = Data(capacity: value.size * MemoryLayout<Int32>.size)
            for item in value.asType(.int32).asArray(Int32.self) {
                var bits = UInt32(bitPattern: item).littleEndian
                withUnsafeBytes(of: &bits) { data.append(contentsOf: $0) }
            }
            return data
        default:
            throw FixtureError.invalid("unsupported tensor dtype \(dtype)")
        }
    }

    private func dtypeName(_ dtype: DType) -> String {
        if dtype == .float32 { return "float32" }
        if dtype == .int32 { return "int32" }
        return String(describing: dtype)
    }

    private func validDigest(_ value: Any?) -> Bool {
        guard let value = value as? String else { return false }
        return value.count == 64 && value.allSatisfy {
            ($0 >= "0" && $0 <= "9") || ($0 >= "a" && $0 <= "f")
        }
    }

    private func validateNoAbsolutePathStrings(_ value: Any) throws {
        switch value {
        case let string as String where string.hasPrefix("/"):
            throw FixtureError.invalid("manifest contains an absolute path")
        case let object as [String: Any]:
            for child in object.values { try validateNoAbsolutePathStrings(child) }
        case let array as [Any]:
            for child in array { try validateNoAbsolutePathStrings(child) }
        default:
            break
        }
    }

    private static func sha256(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    private static func sha256File(_ url: URL) -> String {
        guard let handle = try? FileHandle(forReadingFrom: url) else { return "" }
        defer { try? handle.close() }
        var digest = SHA256()
        while true {
            let chunk: Data?
            do {
                chunk = try handle.read(upToCount: 8 * 1024 * 1024)
            } catch {
                return ""
            }
            guard let chunk, !chunk.isEmpty else { break }
            digest.update(data: chunk)
        }
        return digest.finalize().map { String(format: "%02x", $0) }.joined()
    }
}
