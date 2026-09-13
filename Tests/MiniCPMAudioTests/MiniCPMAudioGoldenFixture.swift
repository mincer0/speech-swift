import CryptoKit
import Foundation
import MLX

/// Loader and fail-closed validator for the opt-in MiniCPM audio golden.
///
/// ``export_minicpm_o_audio_golden.py`` writes one safetensors artifact and a
/// side-car ``*.manifest.json``.  The manifest never contains a local model
/// path; the model is supplied explicitly with ``MINICPM_AUDIO_MLX_PATH``.
struct MiniCPMAudioGoldenFixture {
    enum FixtureError: Error, CustomStringConvertible {
        case missing(String)
        case invalid(String)

        var description: String {
            switch self {
            case .missing(let value): return "missing MiniCPM audio golden: \(value)"
            case .invalid(let value): return "invalid MiniCPM audio golden: \(value)"
            }
        }
    }

    let root: URL
    let manifest: [String: Any]
    let arrays: [String: MLXArray]

    static func locate() throws -> MiniCPMAudioGoldenFixture? {
        let manager = FileManager.default
        var candidates: [URL] = []
        let explicitlyConfigured = ProcessInfo.processInfo.environment["MINICPM_AUDIO_GOLDEN_DIR"]
        if let value = explicitlyConfigured, !value.isEmpty {
            candidates.append(URL(fileURLWithPath: value, isDirectory: true))
        }

        for candidate in candidates {
            guard let manifestURL = manifestURL(in: candidate),
                  manager.fileExists(atPath: manifestURL.path) else { continue }
            let data = try Data(contentsOf: manifestURL)
            guard let manifest = try JSONSerialization.jsonObject(with: data)
                    as? [String: Any] else {
                throw FixtureError.invalid("manifest root is not an object")
            }
            guard let artifact = manifest["artifact"] as? [String: Any],
                  let relative = artifact["path"] as? String,
                  Self.isSafeRelativePath(relative) else {
                throw FixtureError.invalid("artifact path is missing or absolute")
            }
            let artifactURL = candidate.appendingPathComponent(relative)
            guard manager.fileExists(atPath: artifactURL.path) else {
                throw FixtureError.missing("artifact \(relative)")
            }
            let arrays = try MLX.loadArrays(url: artifactURL)
            return MiniCPMAudioGoldenFixture(
                root: candidate,
                manifest: manifest,
                arrays: arrays)
        }
        if let explicitlyConfigured, !explicitlyConfigured.isEmpty {
            throw FixtureError.missing(
                "manifest.json in MINICPM_AUDIO_GOLDEN_DIR=\(explicitlyConfigured)")
        }
        return nil
    }

    private static func manifestURL(in directory: URL) -> URL? {
        let canonical = directory.appendingPathComponent("manifest.json")
        if FileManager.default.fileExists(atPath: canonical.path) { return canonical }
        guard let files = try? FileManager.default.contentsOfDirectory(
            at: directory,
            includingPropertiesForKeys: nil,
            options: [.skipsHiddenFiles]) else { return nil }
        return files.first { $0.lastPathComponent.hasSuffix(".manifest.json") }
    }

    static func isSafeRelativePath(_ value: String) -> Bool {
        !value.isEmpty
            && !value.hasPrefix("/")
            && !value.split(separator: "/").contains(where: { $0 == ".." })
    }

    var schema: String { manifest["schema"] as? String ?? "" }

    var modelURL: URL? {
        guard let value = ProcessInfo.processInfo.environment["MINICPM_AUDIO_MLX_PATH"],
              !value.isEmpty else { return nil }
        let url = URL(fileURLWithPath: value, isDirectory: true)
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: url.path, isDirectory: &isDirectory),
              isDirectory.boolValue else { return nil }
        return url
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

    /// Validate the manifest contract and every tensor's shape/hash/statistics
    /// before a real model is loaded.  This catches stale/swapped fixtures
    /// instead of turning them into an opaque numerical parity failure.
    func validateAllArrays() throws {
        guard schema == "minicpm-o-audio-golden/v2",
              let artifact = manifest["artifact"] as? [String: Any],
              let tensors = artifact["tensors"] as? [String: Any] else {
            throw FixtureError.invalid("unsupported schema or missing artifact tensors")
        }
        guard let count = artifact["tensor_count"] as? NSNumber,
              count.intValue == tensors.count else {
            throw FixtureError.invalid("tensor_count mismatch")
        }
        guard let metadata = manifest["runtime"] as? [String: Any],
              (metadata["sample_rate"] as? NSNumber)?.intValue == 16_000,
              (metadata["mel_bins"] as? NSNumber)?.intValue == 80 else {
            throw FixtureError.invalid("runtime frontend metadata is incomplete")
        }
        for name in tensors.keys.sorted() {
            let expected = try tensorSpec(name)
            let actual = try tensor(name)
            guard let shape = expected["shape"] as? [NSNumber],
                  let dtype = expected["dtype"] as? String,
                  let numel = expected["numel"] as? NSNumber,
                  let digest = expected["sha256"] as? String,
                  let finite = expected["finite"] as? Bool else {
                throw FixtureError.invalid("tensor \(name) metadata is incomplete")
            }
            guard shape.map(\.intValue) == actual.shape,
                  numel.intValue == actual.size,
                  dtype == "float32",
                  digest.count == 64,
                  finite else {
                throw FixtureError.invalid("tensor \(name) shape/dtype contract mismatch")
            }
            eval(actual)
            let values = actual.asType(.float32).asArray(Float.self)
            guard values.allSatisfy(\.isFinite) else {
                throw FixtureError.invalid("tensor \(name) contains non-finite values")
            }
            let bytes = Self.float32Bytes(values)
            guard Self.sha256(bytes) == digest else {
                throw FixtureError.invalid("tensor \(name) sha256 mismatch")
            }
            let maxAbs = values.map { abs($0) }.max() ?? 0
            // Accumulate in Double so the manifest's PyTorch float32 reduction
            // is not compared against a sequential Float sum.  The latter is
            // order-sensitive and drifts by ~1.8e-6 for
            // context_cache_0_layer_02_keys, just beyond the 1e-6 contract
            // tolerance, even though the safetensors bytes/hash match.
            let meanAbs = values.isEmpty
                ? 0
                : values.reduce(0.0) { $0 + Double(abs($1)) } / Double(values.count)
            let expectedMax = try floatValue(expected, "max_abs")
            let expectedMean = try floatValue(expected, "mean_abs")
            guard abs(maxAbs - expectedMax) <= 1e-5,
                  abs(meanAbs - Double(expectedMean)) <= 1e-6 else {
                throw FixtureError.invalid("tensor \(name) statistics mismatch")
            }
        }
    }

    static func float32Bytes(_ values: [Float]) -> Data {
        var bytes = Data(capacity: values.count * MemoryLayout<Float>.size)
        for value in values {
            var bits = value.bitPattern.littleEndian
            withUnsafeBytes(of: &bits) { bytes.append(contentsOf: $0) }
        }
        return bytes
    }

    static func sha256(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }
}
