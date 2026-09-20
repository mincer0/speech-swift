import CryptoKit
import Foundation
import MLX
import XCTest

@testable import MiniCPMToken2Wav

/// Cross-language checks for the prompt frontends.  The fixture is opt-in:
/// ordinary unit tests do not need a 500-frame CAM++ model or a converted
/// S3Tokenizer sidecar, while CI can enable each expensive model independently.
final class MiniCPMPromptParityTests: XCTestCase {
    private let schema = "minicpm-o-prompt-golden/v3"
    private let upstreamRepository = "OpenBMB/MiniCPM-o-Demo"
    private let s3ONNXHash =
        "d43342aa12163a80bf07bffb94c9de2e120a8df2f9917cd2f642e7f4219c6f71"
    private let camONNXHash =
        "a6ac6a63997761ae2997373e2ee1c47040854b4b759ea41ec48e4e42df0f4d73"
    private let canonicalDependency = "minicpmo-utils"
    private let canonicalDependencyVersion = "1.0.6"
    private let canonicalToken2WavHash =
        "0598c25ad9883f75d6d5ca1e9276c03adbfa0357a29442164198ea6931a06efb"
    private let canonicalDemoCommit =
        "ba7fa9cc6ad63c894f1bd5e5afac28466953519d"

    private func requireFixture() throws -> (URL, [String: MLXArray], [[String: Any]], [String: Any]) {
        guard let directory = ProcessInfo.processInfo.environment["MINICPM_PROMPT_GOLDEN_DIR"],
              !directory.isEmpty else {
            throw XCTSkip("set MINICPM_PROMPT_GOLDEN_DIR to run prompt frontend parity")
        }
        let root = URL(fileURLWithPath: directory, isDirectory: true)
        let fixtureURL = root.appendingPathComponent("minicpm-prompt-golden.safetensors")
        let metadataURL = root.appendingPathComponent("minicpm-prompt-golden.json")
        guard FileManager.default.fileExists(atPath: fixtureURL.path),
              FileManager.default.fileExists(atPath: metadataURL.path) else {
            throw FixtureError.invalid(
                "prompt golden fixture is incomplete in configured directory")
        }
        let arrays = try MLX.loadArrays(url: fixtureURL)
        let object = try JSONSerialization.jsonObject(
            with: Data(contentsOf: metadataURL))
        guard let metadata = object as? [String: Any],
              let cases = metadata["cases"] as? [[String: Any]] else {
            throw FixtureError.invalid("prompt golden metadata has no cases")
        }
        try validateFixture(root: root, artifactURL: fixtureURL,
                            arrays: arrays, metadata: metadata, cases: cases)
        return (root, arrays, cases, metadata)
    }

    private enum FixtureError: Error, CustomStringConvertible {
        case invalid(String)

        var description: String {
            switch self {
            case .invalid(let value): return "invalid MiniCPM prompt golden: \(value)"
            }
        }
    }

    private func validateFixture(
        root: URL,
        artifactURL: URL,
        arrays: [String: MLXArray],
        metadata: [String: Any],
        cases: [[String: Any]]
    ) throws {
        guard metadata["schema"] as? String == schema,
              metadata["format"] as? NSNumber == 3,
              metadata["oracle_backend"] as? String
                == "official_onnx_and_pytorch_frontends" else {
            throw FixtureError.invalid("unsupported prompt golden schema")
        }
        try validateNoAbsolutePathStrings(metadata)
        guard let upstream = metadata["upstream"] as? [String: Any],
              upstream["repository"] as? String == upstreamRepository,
              upstream["revision_kind"] as? String == "git_commit",
              upstream["revision"] as? String == canonicalDemoCommit,
              validDigest(upstream["tree_sha256"]) else {
            throw FixtureError.invalid("official Demo provenance is incomplete")
        }
        guard let source = metadata["source"] as? [String: Any],
              source["canonical_prompt_oracle"] as? String
                == "stepaudio2.Token2wav._prepare_prompt",
              source["canonical_dependency"] as? String == canonicalDependency,
              source["canonical_dependency_version"] as? String
                == canonicalDependencyVersion,
              source["canonical_dependency_version_expected"] as? String
                == canonicalDependencyVersion,
              source["canonical_dependency_version_actual"] as? String
                == canonicalDependencyVersion,
              source["canonical_token2wav_wrapper_sha256"] as? String
                == canonicalToken2WavHash,
              source["canonical_token2wav_wrapper_sha256_expected"] as? String
                == canonicalToken2WavHash,
              source["canonical_token2wav_wrapper_sha256_actual"] as? String
                == canonicalToken2WavHash,
              source["canonical_provenance_status"] as? String == "ok",
              !bool(source["canonical_provenance_override"]),
              source["canonical_demo_commit"] as? String == canonicalDemoCommit,
              source["canonical_upstream_tree_sha256"] as? String
                == upstream["tree_sha256"] as? String,
              source["expected_s3_tokenizer_onnx_sha256"] as? String == s3ONNXHash,
              source["expected_campplus_onnx_sha256"] as? String == camONNXHash,
              source["s3_tokenizer_onnx_sha256"] as? String == s3ONNXHash,
              source["campplus_onnx_sha256"] as? String == camONNXHash,
              !bool(source["onnx_hash_override"]) else {
            throw FixtureError.invalid(
                "canonical dependency, ONNX, or Demo provenance mismatch")
        }
        guard let artifact = metadata["artifact"] as? [String: Any],
              let artifactPath = artifact["path"] as? String,
              Self.isSafeRelativePath(artifactPath),
              artifactPath == artifactURL.lastPathComponent,
              validDigest(artifact["sha256"]),
              let count = artifact["tensor_count"] as? NSNumber,
              let tensorMap = artifact["tensors"] as? [String: Any],
              count.intValue == tensorMap.count,
              Set(tensorMap.keys) == Set(arrays.keys) else {
            throw FixtureError.invalid("prompt artifact metadata is incomplete")
        }
        guard Self.sha256File(artifactURL) == artifact["sha256"] as? String else {
            throw FixtureError.invalid("prompt artifact sha256 mismatch")
        }
        for name in tensorMap.keys.sorted() {
            guard let value = arrays[name],
                  let spec = tensorMap[name] as? [String: Any] else {
                throw FixtureError.invalid("tensor \(name) is missing")
            }
            try validateTensor(value, spec: spec, name: name)
        }
        try validateCases(cases, arrays: arrays, metadata: metadata)
        _ = root
    }

    private func validateCases(
        _ cases: [[String: Any]],
        arrays: [String: MLXArray],
        metadata: [String: Any]
    ) throws {
        guard !cases.isEmpty else { throw FixtureError.invalid("no prompt cases") }
        let source = try requireObject(metadata, key: "source")
        let extensionEnabled = bool(source["s3_long_extension_enabled"])
        let extensionBackend = source["s3_long_backend"] as? String
        var names = Set<String>()
        for definition in cases {
            guard let name = definition["name"] as? String,
                  !name.isEmpty,
                  !name.contains("/") else {
                throw FixtureError.invalid("case name is invalid")
            }
            guard names.insert(name).inserted else {
                throw FixtureError.invalid("duplicate case \(name)")
            }
            let duration = try finiteNumber(definition["duration_seconds"],
                                            label: "\(name) duration")
            guard duration > 0 else {
                throw FixtureError.invalid("\(name) duration must be positive")
            }
            let isLong = duration > 30.0
            let segmented = bool(definition["s3_long_segmented"])
            if isLong {
                guard extensionEnabled,
                      extensionBackend
                        == "external_s3tokenizer_long_compatible_non_demo_extension",
                      segmented,
                      definition["s3_long_backend"] as? String
                        == extensionBackend else {
                    throw FixtureError.invalid(
                        "long case \(name) is not explicitly marked as a non-Demo extension")
                }
            } else {
                guard !segmented,
                      definition["s3_tokens_direct_key"] as? String != nil else {
                    throw FixtureError.invalid(
                        "canonical case \(name) has long-extension metadata")
                }
            }

            let pcmKey = try requiredKey(definition, "pcm_key", name: name)
            let s3MelKey = try requiredKey(definition, "s3_mel_key", name: name)
            let s3MelLengthKey = try requiredKey(
                definition, "s3_mel_length_key", name: name)
            let tokenKey = try requiredKey(definition, "s3_token_key", name: name)
            let tokenLengthKey = try requiredKey(
                definition, "s3_token_length_key", name: name)
            let mergedTokenKey = try requiredKey(
                definition, "s3_tokens_merged_key", name: name)
            let flowMelKey = try requiredKey(definition, "flow_mel_key", name: name)
            let flowPromptKey = try requiredKey(
                definition, "flow_prompt_mel_key", name: name)
            let rawCamKey = try requiredKey(
                definition, "cam_raw_fbank_key", name: name)
            let fixedCamKey = try requiredKey(
                definition, "cam_fixed_fbank_key", name: name)
            let embeddingKey = try requiredKey(
                definition, "cam_embedding_key", name: name)
            for key in [pcmKey, s3MelKey, s3MelLengthKey, tokenKey,
                        tokenLengthKey, mergedTokenKey, flowMelKey,
                        flowPromptKey, rawCamKey, fixedCamKey, embeddingKey] {
                guard arrays[key] != nil else {
                    throw FixtureError.invalid("\(name) references missing tensor \(key)")
                }
            }
            let pcm = arrays[pcmKey]!
            let s3Mel = arrays[s3MelKey]!
            let s3MelLength = arrays[s3MelLengthKey]!
            let tokens = arrays[tokenKey]!
            let tokenLength = arrays[tokenLengthKey]!
            let mergedTokens = arrays[mergedTokenKey]!
            let flowMel = arrays[flowMelKey]!
            let flowPrompt = arrays[flowPromptKey]!
            let rawCam = arrays[rawCamKey]!
            let fixedCam = arrays[fixedCamKey]!
            let embedding = arrays[embeddingKey]!
            guard pcm.ndim == 1,
                  int(definition["sample_count"]) == pcm.size,
                  s3Mel.shape == [1, 128, s3Mel.dim(2)],
                  s3MelLength.shape == [1], s3MelLength.dtype == .int32,
                  tokens.ndim == 2, tokens.dim(0) == 1,
                  tokenLength.shape == [1], tokenLength.dtype == .int32,
                  mergedTokens.shape == tokens.shape,
                  flowMel.ndim == 3, flowMel.dim(0) == 1,
                  flowPrompt.ndim == 3, flowPrompt.dim(0) == 1,
                  rawCam.ndim == 3, rawCam.dim(0) == 1,
                  fixedCam.shape == [1, 500, 80],
                  embedding.ndim == 2, embedding.dim(0) == 1 else {
                throw FixtureError.invalid("\(name) tensor shape/dtype contract failed")
            }
            let s3MelLengthValue = s3MelLength.asType(.int32).asArray(Int32.self)[0]
            let tokenLengthValue = tokenLength.asType(.int32).asArray(Int32.self)[0]
            guard Int(s3MelLengthValue) == s3Mel.dim(2),
                  Int(tokenLengthValue) == tokens.dim(1),
                  int(definition["s3_mel_length"]) == s3Mel.dim(2),
                  int(definition["s3_token_length"]) == tokens.dim(1),
                  flowPrompt.dim(2) == tokens.dim(1) * 2,
                  int(definition["flow_prompt_mel_length"]) == flowPrompt.dim(2),
                  fixedCam.shape == (definition["cam_fixed_shape"] as? [NSNumber])?.map(\.intValue),
                  int(definition["cam_raw_frame_count"]) == rawCam.dim(1) else {
                throw FixtureError.invalid("\(name) tensor length metadata mismatch")
            }
            if isLong {
                guard let segments = definition["s3_segments"] as? [[String: Any]],
                      !segments.isEmpty else {
                    throw FixtureError.invalid("long case \(name) has no segments")
                }
                for segment in segments {
                    let melKey = try requiredKey(segment, "s3_mel_key", name: name)
                    let segmentTokenKey = try requiredKey(
                        segment, "s3_token_key", name: name)
                    guard arrays[melKey] != nil, arrays[segmentTokenKey] != nil,
                          int(segment["sample_start"]) >= 0,
                          int(segment["sample_end"]) <= pcm.size,
                          int(segment["sample_end"]) > int(segment["sample_start"]) else {
                        throw FixtureError.invalid("\(name) has invalid long segment")
                    }
                }
            }
        }
    }

    private func requiredKey(
        _ object: [String: Any], _ key: String, name: String
    ) throws -> String {
        guard let value = object[key] as? String, !value.isEmpty else {
            throw FixtureError.invalid("\(name) missing \(key)")
        }
        return value
    }

    private func requireObject(_ object: [String: Any], key: String) throws -> [String: Any] {
        guard let value = object[key] as? [String: Any] else {
            throw FixtureError.invalid("missing object \(key)")
        }
        return value
    }

    private func finiteNumber(_ value: Any?, label: String) throws -> Double {
        guard let number = value as? NSNumber,
              number.doubleValue.isFinite else {
            throw FixtureError.invalid("\(label) is not finite")
        }
        return number.doubleValue
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
            throw FixtureError.invalid("metadata contains an absolute path")
        case let object as [String: Any]:
            for child in object.values { try validateNoAbsolutePathStrings(child) }
        case let array as [Any]:
            for child in array { try validateNoAbsolutePathStrings(child) }
        default:
            break
        }
    }

    private static func isSafeRelativePath(_ value: String) -> Bool {
        !value.isEmpty
            && !value.hasPrefix("/")
            && !value.split(separator: "/").contains(where: { $0 == ".." })
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

    private func assertClose(
        _ actual: MLXArray,
        _ expected: MLXArray,
        label: String,
        tolerance: Float = 2e-3
    ) {
        XCTAssertEqual(actual.shape, expected.shape, "\(label) shape")
        let lhs = actual.asType(.float32).asArray(Float.self)
        let rhs = expected.asType(.float32).asArray(Float.self)
        XCTAssertEqual(lhs.count, rhs.count, "\(label) element count")
        guard lhs.count == rhs.count, !lhs.isEmpty else { return }
        XCTAssertTrue(lhs.allSatisfy(\.isFinite), "\(label) actual contains non-finite values")
        XCTAssertTrue(rhs.allSatisfy(\.isFinite), "\(label) golden contains non-finite values")
        var maxAbs: Float = 0
        var sumAbs: Float = 0
        var dot: Double = 0
        var lhsNorm: Double = 0
        var rhsNorm: Double = 0
        for (left, right) in zip(lhs, rhs) {
            let delta = abs(left - right)
            maxAbs = max(maxAbs, delta)
            sumAbs += delta
            dot += Double(left) * Double(right)
            lhsNorm += Double(left) * Double(left)
            rhsNorm += Double(right) * Double(right)
        }
        let meanAbs = sumAbs / Float(lhs.count)
        let cosine: Float
        if lhsNorm == 0, rhsNorm == 0 {
            cosine = 1
        } else {
            cosine = Float(dot / max(sqrt(lhsNorm * rhsNorm), Double.leastNonzeroMagnitude))
        }
        print("[PromptParity] \(label) shape=\(actual.shape) finite=true maxAbs=\(maxAbs) meanAbs=\(meanAbs) cosine=\(cosine)")
        XCTAssertLessThanOrEqual(maxAbs, tolerance, "\(label) maxAbs=\(maxAbs)")
        XCTAssertGreaterThan(cosine, 0.999, "\(label) cosine=\(cosine)")
    }

    private func int(_ value: Any?, default fallback: Int = 0) -> Int {
        (value as? NSNumber)?.intValue ?? fallback
    }

    private func bool(_ value: Any?, default fallback: Bool = false) -> Bool {
        (value as? NSNumber)?.boolValue ?? fallback
    }

    private func alignFlowMel(_ mel: MLXArray, targetFrames: Int) -> MLXArray {
        precondition(mel.ndim == 3 && mel.dim(0) == 1)
        let channels = mel.dim(1)
        let sourceFrames = mel.dim(2)
        let source = mel.asType(.float32).asArray(Float.self)
        var output = [Float](repeating: 0, count: channels * targetFrames)
        let copied = min(sourceFrames, targetFrames)
        for channel in 0..<channels {
            for frame in 0..<copied {
                output[channel * targetFrames + frame] =
                    source[channel * sourceFrames + frame]
            }
            if copied < targetFrames {
                let fill = copied > 0
                    ? source[channel * sourceFrames + copied - 1] : 0
                for frame in copied..<targetFrames {
                    output[channel * targetFrames + frame] = fill
                }
            }
        }
        return MLXArray(output, [1, channels, targetFrames])
    }

    private func mergeS3Segments(_ segments: [[Int32]], overlap: Int = 4, tokenRate: Int = 25) -> [Int32] {
        guard !segments.isEmpty else { return [] }
        let overlapTokens = (overlap / 2) * tokenRate
        var merged: [Int32] = []
        for (index, segment) in segments.enumerated() {
            let start = index == 0 ? 0 : min(overlapTokens, segment.count)
            let end = index == segments.count - 1
                ? segment.count
                : max(start, segment.count - overlapTokens)
            merged.append(contentsOf: segment[start..<end])
        }
        return merged
    }

    func testPromptMetadataIntegerFieldsUseNSNumberParser() throws {
        let object = try JSONSerialization.jsonObject(
            with: Data(#"{"s3_mel_length": 37, "s3_token_length": 19}"#.utf8))
        let definition = try XCTUnwrap(object as? [String: Any])

        XCTAssertEqual(int(definition["s3_mel_length"]), 37)
        XCTAssertEqual(int(definition["s3_token_length"]), 19)
    }

    func testS3AndCamPlusPlusPromptFrontendsMatchGolden() throws {
        let (root, arrays, cases, metadata) = try requireFixture()
        let s3Extractor = MiniCPMS3TokenizerMelExtractor()
        let flowExtractor = MiniCPMFlowPromptMelExtractor()
        let camExtractor = MiniCPMCamPlusPlusMelExtractor()

        let environment = ProcessInfo.processInfo.environment
        let speechEncoder: MiniCPMNativePromptEncoder?
        if let path = environment["MINICPM_SPEECH_TOKENIZER_WEIGHTS"] {
            speechEncoder = try MiniCPMNativePromptEncoder(
                speechTokenizerWeightsURL: URL(fileURLWithPath: path),
                camPlusPlusModelURL: nil)
        } else {
            speechEncoder = nil
        }
        let speakerEncoder: MiniCPMNativePromptEncoder?
        if let path = environment["MINICPM_CAMPPLUS_COREML"] {
            speakerEncoder = try MiniCPMNativePromptEncoder(
                speechTokenizerWeightsURL: nil,
                camPlusPlusModelURL: URL(fileURLWithPath: path))
        } else {
            speakerEncoder = nil
        }

        for definition in cases {
            let name = try XCTUnwrap(definition["name"] as? String)
            let pcmKey = try XCTUnwrap(definition["pcm_key"] as? String)
            let pcmArray = try XCTUnwrap(arrays[pcmKey])
            let pcm = pcmArray.asType(.float32).asArray(Float.self)

            let s3Mel = s3Extractor.extract(pcm)
            let expectedS3Mel = try XCTUnwrap(
                arrays[try XCTUnwrap(definition["s3_mel_key"] as? String)])
            assertClose(s3Mel, expectedS3Mel, label: "\(name).s3_mel")
            XCTAssertEqual(
                s3Mel.dim(2),
                int(definition["s3_mel_length"]),
                "\(name) S3 mel length")

            // S3Tokenizer inference is gated separately from the signal
            // frontend so a fixture remains useful before its safetensors
            // conversion has completed. Canonical cases use the pinned
            // stepaudio2.Token2wav prompt path; only an explicitly labelled
            // external extension may exercise overlapped long segments.
            if let speechEncoder {
                let isLong = bool(definition["s3_long_segmented"])
                if isLong {
                    let source = try XCTUnwrap(metadata["source"] as? [String: Any])
                    guard bool(source["s3_long_extension_enabled"]),
                          source["s3_long_backend"] as? String
                            == "external_s3tokenizer_long_compatible_non_demo_extension",
                          definition["s3_long_backend"] as? String
                            == "external_s3tokenizer_long_compatible_non_demo_extension"
                    else {
                        throw MiniCPMToken2WavPromptPreparationError.invalidTokens(
                            "long prompt case is not explicitly marked as the non-Demo extension")
                    }
                }
                if isLong, let segmentDefinitions = definition["s3_segments"] as? [[String: Any]] {
                    var segmentTokens: [[Int32]] = []
                    for segment in segmentDefinitions {
                        let start = int(segment["sample_start"])
                        let end = int(segment["sample_end"])
                        XCTAssertGreaterThanOrEqual(start, 0)
                        XCTAssertLessThanOrEqual(end, pcm.count)
                        let segmentPCM = Array(pcm[start..<end])
                        let segmentMel = s3Extractor.extract(segmentPCM)
                        let expectedMel = try XCTUnwrap(
                            arrays[try XCTUnwrap(segment["s3_mel_key"] as? String)])
                        assertClose(segmentMel, expectedMel, label: "\(name).segment\(int(segment["index"])).s3_mel")
                        let tokens = try speechEncoder.encodeSpeechTokens(tokenizerMel: segmentMel)
                        let expectedTokens = try XCTUnwrap(
                            arrays[try XCTUnwrap(segment["s3_token_key"] as? String)])
                        XCTAssertEqual(tokens.asType(.int32).asArray(Int32.self),
                                       expectedTokens.asType(.int32).asArray(Int32.self),
                                       "\(name).segment\(int(segment["index"])).s3_tokens")
                        segmentTokens.append(Array(tokens.asType(.int32).asArray(Int32.self)))
                    }
                    let merged = mergeS3Segments(segmentTokens)
                    let expectedMerged = try XCTUnwrap(
                        arrays[try XCTUnwrap(definition["s3_tokens_merged_key"] as? String)])
                    XCTAssertEqual(merged,
                                   expectedMerged.asType(.int32).asArray(Int32.self),
                                   "\(name).s3_tokens_merged")
                    XCTAssertEqual(merged.count, int(definition["s3_token_length"]),
                                   "\(name) merged S3 token length")
                } else {
                    let tokens = try speechEncoder.encodeSpeechTokens(tokenizerMel: s3Mel)
                    let expectedTokens = try XCTUnwrap(
                        arrays[try XCTUnwrap(definition["s3_token_key"] as? String)])
                    XCTAssertEqual(tokens.asType(.int32).asArray(Int32.self),
                                   expectedTokens.asType(.int32).asArray(Int32.self),
                                   "\(name).s3_tokens")
                    XCTAssertEqual(
                        tokens.dim(1),
                        int(definition["s3_token_length"]),
                        "\(name) S3 token length")
                }
            }

            let flowAudio = MiniCPMToken2WavPromptPreparer.resampleAudio(
                pcm, from: int(definition["sample_rate"], default: 16_000), to: 24_000)
            let flowMel = flowExtractor.extract(flowAudio)
            let expectedFlowMel = try XCTUnwrap(
                arrays[try XCTUnwrap(definition["flow_mel_key"] as? String)])
            assertClose(flowMel, expectedFlowMel, label: "\(name).flow_mel", tolerance: 3e-3)
            let alignedFlow = alignFlowMel(
                flowMel,
                targetFrames: int(definition["s3_token_length"]) * 2)
            let expectedAlignedFlow = try XCTUnwrap(
                arrays[try XCTUnwrap(definition["flow_prompt_mel_key"] as? String)])
            assertClose(alignedFlow, expectedAlignedFlow,
                        label: "\(name).flow_prompt_mel", tolerance: 3e-3)
            XCTAssertEqual(
                alignedFlow.dim(2),
                int(definition["s3_token_length"]) * 2,
                "\(name) Flow prompt must be 2x speech-token count")

            let cam = camExtractor.extractWithDiagnostics(pcm, targetFrames: 500)
            let expectedCam = try XCTUnwrap(
                arrays[try XCTUnwrap(definition["cam_fixed_fbank_key"] as? String)])
            assertClose(cam.mel, expectedCam, label: "\(name).cam_fixed_fbank", tolerance: 3e-3)
            XCTAssertEqual(
                cam.rawFrameCount,
                int(definition["cam_raw_frame_count"]),
                "\(name) CAM++ raw frame count")
            XCTAssertEqual(
                cam.adaptation.rawValue,
                definition["cam_adaptation"] as? String,
                "\(name) CAM++ adaptation")

            if let speakerEncoder {
                let embedding = try speakerEncoder.encodeSpeakerEmbedding(speakerMel: cam.mel)
                let expectedEmbedding = try XCTUnwrap(
                    arrays[try XCTUnwrap(definition["cam_embedding_key"] as? String)])
                assertClose(
                    embedding,
                    expectedEmbedding,
                    label: "\(name).cam_embedding",
                    tolerance: 5e-3)
            }
        }
        _ = root // Keep the fixture root available for debugger inspection.
    }
}
