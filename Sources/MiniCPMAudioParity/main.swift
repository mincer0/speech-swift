import Foundation
import MiniCPMAudio
import MLX

private struct Metrics: Encodable {
    let name: String
    let shape: [Int]
    let actualDType: String
    let expectedDType: String
    let actualMaxAbsolute: Float
    let actualMeanAbsolute: Float
    let expectedMaxAbsolute: Float
    let expectedMeanAbsolute: Float
    let maxAbsoluteError: Float
    let meanAbsoluteError: Float
    let maxRelativeError: Float
    let relativeL2Error: Float
    let cosineSimilarity: Float
    let worstIndex: [Int]
    let actualAtWorst: Float
    let expectedAtWorst: Float
}

private struct Report: Encodable {
    let format: String
    let modelBundle: String
    let goldenArtifact: String
    let attentionVariant: String
    let geluVariant: String
    let metrics: [Metrics]
}

private func dtypeName(_ value: MLXArray) -> String {
    String(describing: value.dtype)
}

private func multidimensionalIndex(_ flatIndex: Int, shape: [Int]) -> [Int] {
    guard !shape.isEmpty else { return [] }
    var remainder = flatIndex
    var result = Array(repeating: 0, count: shape.count)
    for index in stride(from: shape.count - 1, through: 0, by: -1) {
        let dimension = shape[index]
        guard dimension > 0 else { return [] }
        result[index] = remainder % dimension
        remainder /= dimension
    }
    return result
}

private func compare(
    name: String,
    actual: MLXArray,
    expected: MLXArray
) throws -> Metrics {
    guard actual.shape == expected.shape else {
        throw MiniCPMAudioError.invalidFeatures(
            "\(name) shape mismatch: MLX \(actual.shape), PyTorch \(expected.shape)"
        )
    }
    let lhs = actual.asType(.float32)
    let rhs = expected.asType(.float32)
    eval(lhs, rhs)
    let actualValues = lhs.asArray(Float.self)
    let expectedValues = rhs.asArray(Float.self)
    guard actualValues.count == expectedValues.count else {
        throw MiniCPMAudioError.invalidFeatures(
            "\(name) element count mismatch: MLX \(actualValues.count), PyTorch \(expectedValues.count)"
        )
    }

    var maxAbsoluteError = 0.0
    var maxRelativeError = 0.0
    var meanAbsoluteError = 0.0
    var sumSquaredError = 0.0
    var sumSquaredExpected = 0.0
    var dot = 0.0
    var actualNormSquared = 0.0
    var expectedNormSquared = 0.0
    var worstFlatIndex = 0
    for index in actualValues.indices {
        let left = Double(actualValues[index])
        let right = Double(expectedValues[index])
        let difference = abs(left - right)
        let relative = difference / max(abs(right), 1e-12)
        if difference > maxAbsoluteError {
            maxAbsoluteError = difference
            worstFlatIndex = index
        }
        maxRelativeError = max(maxRelativeError, relative)
        meanAbsoluteError += difference
        sumSquaredError += difference * difference
        sumSquaredExpected += right * right
        dot += left * right
        actualNormSquared += left * left
        expectedNormSquared += right * right
    }
    let count = Double(actualValues.count)
    meanAbsoluteError = count == 0 ? 0 : meanAbsoluteError / count
    let cosine: Double
    if actualNormSquared > 0 && expectedNormSquared > 0 {
        cosine = dot / (sqrt(actualNormSquared) * sqrt(expectedNormSquared))
    } else {
        cosine = dot == 0 ? 1 : 0
    }
    return Metrics(
        name: name,
        shape: actual.shape,
        actualDType: dtypeName(actual),
        expectedDType: dtypeName(expected),
        actualMaxAbsolute: actualValues.map { abs($0) }.max() ?? 0,
        actualMeanAbsolute: count == 0
            ? 0
            : Float(actualValues.reduce(0.0) { $0 + Double(abs($1)) } / count),
        expectedMaxAbsolute: expectedValues.map { abs($0) }.max() ?? 0,
        expectedMeanAbsolute: count == 0
            ? 0
            : Float(expectedValues.reduce(0.0) { $0 + Double(abs($1)) } / count),
        maxAbsoluteError: Float(maxAbsoluteError),
        meanAbsoluteError: Float(meanAbsoluteError),
        maxRelativeError: Float(maxRelativeError),
        relativeL2Error: sumSquaredExpected > 0
            ? Float(sqrt(sumSquaredError / sumSquaredExpected))
            : (sumSquaredError == 0 ? 0 : .infinity),
        cosineSimilarity: Float(cosine),
        worstIndex: multidimensionalIndex(worstFlatIndex, shape: actual.shape),
        actualAtWorst: actualValues.isEmpty ? 0 : actualValues[worstFlatIndex],
        expectedAtWorst: expectedValues.isEmpty ? 0 : expectedValues[worstFlatIndex]
    )
}

private func appendCacheMetrics(
    prefix: String,
    cache: MiniCPMAudioCache?,
    golden: [String: MLXArray],
    actualTensors: inout [String: MLXArray]
) throws -> [Metrics] {
    guard let cache else {
        throw MiniCPMAudioError.invalidFeatures(
            "\(prefix) output did not contain an encoder cache")
    }
    var metrics: [Metrics] = []
    for (index, layer) in cache.layers.enumerated() {
        let suffix = String(format: "layer_%02d", index)
        let keyName = "\(prefix)_\(suffix)_keys"
        let valueName = "\(prefix)_\(suffix)_values"
        guard let expectedKeys = golden[keyName],
              let expectedValues = golden[valueName] else {
            throw MiniCPMAudioError.invalidFeatures(
                "golden file is missing \(keyName) or \(valueName)")
        }
        actualTensors[keyName] = layer.keys
        actualTensors[valueName] = layer.values
        metrics.append(try compare(
            name: keyName, actual: layer.keys, expected: expectedKeys))
        metrics.append(try compare(
            name: valueName, actual: layer.values, expected: expectedValues))
    }
    return metrics
}

@main
enum MiniCPMAudioParityCommand {
    static func main() throws {
        let arguments = CommandLine.arguments
        guard arguments.count >= 3 else {
            FileHandle.standardError.write(Data(
                (
                    "usage: minicpm-audio-parity <audio-bundle-dir> "
                    + "<golden.safetensors> [--output-dir dir] [--soak rounds]\n"
                ).utf8
            ))
            throw MiniCPMAudioError.invalidFeatures("invalid parity command arguments")
        }
        let bundle = URL(fileURLWithPath: arguments[1], isDirectory: true)
        let goldenURL = URL(fileURLWithPath: arguments[2])
        var outputDirectory: URL?
        var soakRounds: Int?
        var cursor = 3
        while cursor < arguments.count {
            switch arguments[cursor] {
            case "--output-dir":
                guard cursor + 1 < arguments.count,
                      outputDirectory == nil else {
                    throw MiniCPMAudioError.invalidFeatures(
                        "--output-dir requires exactly one directory")
                }
                outputDirectory = URL(
                    fileURLWithPath: arguments[cursor + 1], isDirectory: true)
                cursor += 2
            case "--soak":
                guard cursor + 1 < arguments.count,
                      soakRounds == nil,
                      let rounds = Int(arguments[cursor + 1]),
                      rounds > 0 else {
                    throw MiniCPMAudioError.invalidFeatures(
                        "--soak requires one positive integer")
                }
                soakRounds = rounds
                cursor += 2
            default:
                throw MiniCPMAudioError.invalidFeatures(
                    "unknown parity argument \(arguments[cursor])")
            }
        }
        let golden = try MLX.loadArrays(url: goldenURL)
        guard let features = golden["features"],
              let expectedStates = golden["encoder_states"],
              let expectedEmbeddings = golden["embeddings"]
        else {
            throw MiniCPMAudioError.invalidFeatures(
                "golden file must contain features, encoder_states, and embeddings"
            )
        }
        let model = try MiniCPMAudioModel.fromDirectory(bundle)
        let output = try model.encode(features: features, useCache: false)
        eval(output.encoderStates, output.embeddings)
        var actualTensors: [String: MLXArray] = [
            "encoder_states": output.encoderStates,
            "embeddings": output.embeddings,
        ]
        var metrics = try [
            compare(
                name: "encoder_states",
                actual: output.encoderStates,
                expected: expectedStates
            ),
            compare(
                name: "embeddings",
                actual: output.embeddings,
                expected: expectedEmbeddings
            ),
        ]
        if let waveform = golden["waveform"] {
            eval(waveform)
            let samples = waveform.asType(.float32).asArray(Float.self)
            let extracted = try MiniCPMWhisperFeatureExtractor().extract(
                samples,
                normalization: .dynamic(rangeDB: 8)
            )
            let nativeFeatures = MLXArray(
                extracted.data,
                [1, extracted.timeFrames, extracted.melBins]
            )
            actualTensors["log_mel_features"] = nativeFeatures
            metrics.append(try compare(
                name: "log_mel_features",
                actual: nativeFeatures,
                expected: features
            ))
            // The extractor intentionally returns host Float32.  Measure the
            // explicit model-boundary cast separately so a BF16 golden does
            // not make the raw PCM→mel error look larger than it is.
            let modelInputFeatures = nativeFeatures.asType(model.featureDType)
            actualTensors["model_input_features"] = modelInputFeatures
            metrics.append(try compare(
                name: "model_input_features",
                actual: modelInputFeatures,
                expected: features
            ))
            let endToEnd = try model.encode(
                features: modelInputFeatures,
                useCache: false
            )
            eval(endToEnd.embeddings)
            actualTensors["end_to_end_embeddings"] = endToEnd.embeddings
            metrics.append(try compare(
                name: "end_to_end_embeddings",
                actual: endToEnd.embeddings,
                expected: expectedEmbeddings
            ))
        }
        if let firstFeatures = golden["stream_features_0"],
           let firstStates = golden["stream_encoder_states_0"],
           let firstEmbeddings = golden["stream_embeddings_0"],
           let secondFeatures = golden["stream_features_1"],
           let secondStates = golden["stream_encoder_states_1"],
           let secondEmbeddings = golden["stream_embeddings_1"] {
            let first = try model.encode(features: firstFeatures, useCache: true)
            guard first.cache?.length == 50 else {
                throw MiniCPMAudioError.invalidFeatures(
                    "first streaming cache must contain 50 positions"
                )
            }
            let second = try model.encode(
                features: secondFeatures,
                cache: first.cache,
                useCache: true
            )
            guard second.cache?.length == 100 else {
                throw MiniCPMAudioError.invalidFeatures(
                    "second streaming cache must contain 100 positions"
                )
            }
            eval(
                first.encoderStates, first.embeddings,
                second.encoderStates, second.embeddings
            )
            actualTensors["stream_encoder_states_0"] = first.encoderStates
            actualTensors["stream_embeddings_0"] = first.embeddings
            actualTensors["stream_encoder_states_1"] = second.encoderStates
            actualTensors["stream_embeddings_1"] = second.embeddings
            metrics.append(try compare(
                name: "stream_encoder_states_0",
                actual: first.encoderStates,
                expected: firstStates
            ))
            metrics.append(try compare(
                name: "stream_embeddings_0",
                actual: first.embeddings,
                expected: firstEmbeddings
            ))
            metrics.append(try compare(
                name: "stream_encoder_states_1",
                actual: second.encoderStates,
                expected: secondStates
            ))
            metrics.append(try compare(
                name: "stream_embeddings_1",
                actual: second.embeddings,
                expected: secondEmbeddings
            ))
            metrics.append(contentsOf: try appendCacheMetrics(
                prefix: "stream_cache_0",
                cache: first.cache,
                golden: golden,
                actualTensors: &actualTensors))
            metrics.append(contentsOf: try appendCacheMetrics(
                prefix: "stream_cache_1",
                cache: second.cache,
                golden: golden,
                actualTensors: &actualTensors))
        }
        if let firstFeatures = golden["context_stream_features_0"],
           let firstStates = golden["context_stream_encoder_states_0"],
           let firstEmbeddings = golden["context_stream_embeddings_0"],
           let secondFeatures = golden["context_stream_features_1"],
           let secondStates = golden["context_stream_encoder_states_1"],
           let secondEmbeddings = golden["context_stream_embeddings_1"] {
            let session = MiniCPMAudioStreamingSession(model: model)
            let first = try session.encodeFeatures(
                firstFeatures, prefixExtraFrames: 0, suffixExtraFrames: 2)
            guard session.cache?.length == 49 else {
                throw MiniCPMAudioError.invalidFeatures(
                    "first context streaming cache must contain 49 positions")
            }
            let second = try session.encodeFeatures(
                secondFeatures, prefixExtraFrames: 2, suffixExtraFrames: 2)
            guard session.cache?.length == 97 else {
                throw MiniCPMAudioError.invalidFeatures(
                    "second context streaming cache must contain 97 positions")
            }
            eval(
                first.output.encoderStates, first.output.embeddings,
                second.output.encoderStates, second.output.embeddings)
            actualTensors["context_stream_encoder_states_0"]
                = first.output.encoderStates
            actualTensors["context_stream_embeddings_0"]
                = first.output.embeddings
            actualTensors["context_stream_encoder_states_1"]
                = second.output.encoderStates
            actualTensors["context_stream_embeddings_1"]
                = second.output.embeddings
            metrics.append(try compare(
                name: "context_stream_encoder_states_0",
                actual: first.output.encoderStates,
                expected: firstStates))
            metrics.append(try compare(
                name: "context_stream_embeddings_0",
                actual: first.output.embeddings,
                expected: firstEmbeddings))
            metrics.append(try compare(
                name: "context_stream_encoder_states_1",
                actual: second.output.encoderStates,
                expected: secondStates))
            metrics.append(try compare(
                name: "context_stream_embeddings_1",
                actual: second.output.embeddings,
                expected: secondEmbeddings))
            metrics.append(contentsOf: try appendCacheMetrics(
                prefix: "context_cache_0",
                cache: first.output.cache,
                golden: golden,
                actualTensors: &actualTensors))
            metrics.append(contentsOf: try appendCacheMetrics(
                prefix: "context_cache_1",
                cache: second.output.cache,
                golden: golden,
                actualTensors: &actualTensors))
        }
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        FileHandle.standardOutput.write(try encoder.encode(metrics))
        FileHandle.standardOutput.write(Data("\n".utf8))

        guard metrics.allSatisfy({ $0.cosineSimilarity >= 0.999 }) else {
            throw MiniCPMAudioError.invalidFeatures(
                "MLX/PyTorch cosine similarity is below 0.999"
            )
        }

        if let rounds = soakRounds {
            let soakFeatures = golden["stream_features_1"] ?? features
            let session = MiniCPMAudioStreamingSession(model: model)
            let started = ContinuousClock.now
            for _ in 0..<rounds {
                let result = try session.encodeFeatures(soakFeatures)
                eval(result.output.embeddings)
                let peak = MLX.abs(result.output.embeddings)
                    .max().item(Float.self)
                guard peak.isFinite else {
                    throw MiniCPMAudioError.invalidFeatures(
                        "non-finite embedding during streaming soak"
                    )
                }
            }
            let elapsed = ContinuousClock.now - started
            FileHandle.standardError.write(Data(
                "soak rounds=\(rounds) resets=\(session.cacheResetCount) elapsed=\(elapsed)\n".utf8
            ))
        }

        if let outputDirectory {
            try FileManager.default.createDirectory(
                at: outputDirectory,
                withIntermediateDirectories: true)
            let outputURL = outputDirectory.appendingPathComponent(
                "swift_output.safetensors")
            try MLX.save(
                arrays: actualTensors,
                metadata: [
                    "format": "minicpm-o-audio-swift-parity/v1",
                    "model_bundle": bundle.path,
                    "golden_artifact": goldenURL.path,
                    "attention_variant": ProcessInfo.processInfo.environment[
                        "MINICPM_AUDIO_ATTENTION_VARIANT"
                    ] ?? "default_mpsgraph",
                    "gelu_variant": ProcessInfo.processInfo.environment[
                        "MINICPM_AUDIO_GELU_VARIANT"
                    ] ?? "default_mpsgraph",
                ],
                url: outputURL)
            let report = Report(
                format: "minicpm-o-audio-swift-parity/v1",
                modelBundle: bundle.path,
                goldenArtifact: goldenURL.path,
                attentionVariant: ProcessInfo.processInfo.environment[
                    "MINICPM_AUDIO_ATTENTION_VARIANT"
                ] ?? "default_mpsgraph",
                geluVariant: ProcessInfo.processInfo.environment[
                    "MINICPM_AUDIO_GELU_VARIANT"
                ] ?? "default_mpsgraph",
                metrics: metrics)
            let reportURL = outputDirectory.appendingPathComponent("error.json")
            try encoder.encode(report).write(to: reportURL)
        }
    }
}
