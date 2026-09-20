import Foundation
import MiniCPMToken2Wav
import MLX

private struct Metric: Encodable {
    let name: String
    let shape: [Int]
    let maximumAbsoluteError: Float
    let meanAbsoluteError: Float
    let cosineSimilarity: Float
}

private func compare(_ name: String, _ actual: MLXArray, _ expected: MLXArray) throws -> Metric {
    guard actual.shape == expected.shape else {
        throw MiniCPMToken2WavError.invalidWeights(
            "\(name) shape mismatch: MLX \(actual.shape), PyTorch \(expected.shape)")
    }
    let lhs = actual.asType(.float32)
    let rhs = expected.asType(.float32)
    let difference = MLX.abs(lhs - rhs)
    let numerator = sum(lhs * rhs)
    let denominator = sqrt(sum(lhs * lhs) * sum(rhs * rhs))
    eval(difference, numerator, denominator)
    return Metric(
        name: name,
        shape: lhs.shape,
        maximumAbsoluteError: difference.max().item(Float.self),
        meanAbsoluteError: difference.mean().item(Float.self),
        cosineSimilarity: (numerator / denominator).item(Float.self))
}

@main
enum MiniCPMHiFTParityCommand {
    static func main() throws {
        guard CommandLine.arguments.count == 3 else {
            FileHandle.standardError.write(Data(
                "usage: minicpm-hift-parity <hift-bundle-dir> <golden.safetensors>\n".utf8))
            throw MiniCPMToken2WavError.invalidWeights("invalid parity arguments")
        }
        let bundle = URL(fileURLWithPath: CommandLine.arguments[1], isDirectory: true)
        let golden = try MLX.loadArrays(
            url: URL(fileURLWithPath: CommandLine.arguments[2]))
        guard let mel = golden["mel"], let source = golden["source"] else {
            throw MiniCPMToken2WavError.invalidWeights(
                "golden fixture must include mel and source")
        }

        let model = try MiniCPMHiFTGenerator.fromDirectory(bundle)
        model.train(false)
        let output = model.generate(mel: mel, sourceOverride: source)
        eval(output.waveform, output.f0)

        var actual: [String: MLXArray] = output.intermediates
        actual["f0"] = output.f0
        actual["source"] = output.source
        actual["waveform"] = output.waveform
        if let referenceF0 = golden["f0"], let sourceNoise = golden["source_noise"] {
            actual["generated_source"] = model.generateSource(
                f0: referenceF0,
                standardNoise: sourceNoise,
                initialPhase: golden["source_initial_phase"])
        }
        let names = [
            "f0", "source", "generated_source", "source_stft", "conv_pre",
            "ups.0", "stages.0", "ups.1", "stages.1",
            "ups.2", "stages.2", "logits", "waveform",
        ]
        let metrics = try names.map { name -> Metric in
            let expectedName = name == "generated_source" ? "source" : name
            guard let actualValue = actual[name], let expectedValue = golden[expectedName] else {
                throw MiniCPMToken2WavError.invalidWeights("missing parity tensor \(name)")
            }
            return try compare(name, actualValue, expectedValue)
        }

        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        FileHandle.standardOutput.write(try encoder.encode(metrics))
        FileHandle.standardOutput.write(Data("\n".utf8))
        guard metrics.allSatisfy({ $0.cosineSimilarity >= 0.999 }) else {
            throw MiniCPMToken2WavError.invalidWeights(
                "MLX/PyTorch HiFT cosine similarity is below 0.999")
        }
    }
}
