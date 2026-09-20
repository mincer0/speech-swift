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
        name: name, shape: actual.shape,
        maximumAbsoluteError: difference.max().item(Float.self),
        meanAbsoluteError: difference.mean().item(Float.self),
        cosineSimilarity: (numerator / denominator).item(Float.self))
}

@main
enum MiniCPMFlowParityCommand {
    static func main() throws {
        guard CommandLine.arguments.count == 3 else {
            throw MiniCPMToken2WavError.invalidWeights(
                "usage: minicpm-flow-parity <bundle-dir> <golden.safetensors>")
        }
        let model = try MiniCPMToken2WavFlowModel.fromDirectory(
            URL(fileURLWithPath: CommandLine.arguments[1], isDirectory: true))
        model.train(false)
        let golden = try MLX.loadArrays(
            url: URL(fileURLWithPath: CommandLine.arguments[2]))
        guard let tokens = golden["tokens"],
              let expectedEmbedding = golden["token_embeddings"],
              let expectedEncoder = golden["encoder_output"],
              let x = golden["dit_x"],
              let mu = golden["dit_mu"],
              let speakers = golden["dit_speakers"],
              let condition = golden["dit_condition"],
              let time = golden["dit_time"],
              let mask = golden["dit_mask"],
              let expectedDiT = golden["dit_output"],
              let cfmMu = golden["cfm_mu"],
              let cfmSpeaker = golden["cfm_speaker"],
              let cfmCondition = golden["cfm_condition"],
              let cfmMask = golden["cfm_mask"],
              let cfmNoise = golden["cfm_noise"],
              let expectedCFM = golden["cfm_output_2_steps"]
        else {
            throw MiniCPMToken2WavError.invalidWeights("incomplete flow golden fixture")
        }

        let embedding = model.inputEmbedding(tokens)
        let encoderOutput = model.encoder(embedding)
        let dit = model.decoder.estimator(
            x: x, mu: mu, time: time, speakers: speakers,
            condition: condition, validMask: mask)
        var metrics = try [
            compare("token_embeddings", embedding, expectedEmbedding),
            compare("encoder_output", encoderOutput, expectedEncoder),
            compare("dit_output", dit.output, expectedDiT),
        ]

        model.decoder.fixedNoise = cfmNoise
        let cfm = model.decoder.generate(
            mu: cfmMu,
            speakers: cfmSpeaker,
            condition: cfmCondition,
            validMask: cfmMask,
            odeSteps: 2)
        metrics.append(try compare("cfm_output_2_steps", cfm.output, expectedCFM))

        if let streamTokens1 = golden["stream_tokens_1"],
           let streamTokens2 = golden["stream_tokens_2"],
           let expectedStream1 = golden["stream_encoder_1"],
           let expectedStream2 = golden["stream_encoder_2"],
           let expectedFirstK1 = golden["stream_first_k_1"],
           let expectedFirstV1 = golden["stream_first_v_1"],
           let expectedSecondK1 = golden["stream_second_k_1"],
           let expectedSecondV1 = golden["stream_second_v_1"],
           let expectedFirstK2 = golden["stream_first_k_2"],
           let expectedFirstV2 = golden["stream_first_v_2"],
           let expectedSecondK2 = golden["stream_second_k_2"],
           let expectedSecondV2 = golden["stream_second_v_2"] {
            let first = model.encoder.streaming(model.inputEmbedding(streamTokens1))
            let second = model.encoder.streaming(
                model.inputEmbedding(streamTokens2), cache: first.cache)
            metrics += try [
                compare("stream_encoder_1", first.output, expectedStream1),
                compare("stream_first_k_1", first.cache.firstAttention[0].keys, expectedFirstK1),
                compare("stream_first_v_1", first.cache.firstAttention[0].values, expectedFirstV1),
                compare("stream_second_k_1", first.cache.secondAttention[0].keys, expectedSecondK1),
                compare("stream_second_v_1", first.cache.secondAttention[0].values, expectedSecondV1),
                compare("stream_encoder_2", second.output, expectedStream2),
                compare("stream_first_k_2", second.cache.firstAttention[0].keys, expectedFirstK2),
                compare("stream_first_v_2", second.cache.firstAttention[0].values, expectedFirstV2),
                compare("stream_second_k_2", second.cache.secondAttention[0].keys, expectedSecondK2),
                compare("stream_second_v_2", second.cache.secondAttention[0].values, expectedSecondV2),
            ]
        }

        if let streamX1 = golden["dit_stream_x_1"],
           let streamMu1 = golden["dit_stream_mu_1"],
           let streamCondition1 = golden["dit_stream_condition_1"],
           let streamX2 = golden["dit_stream_x_2"],
           let streamMu2 = golden["dit_stream_mu_2"],
           let streamCondition2 = golden["dit_stream_condition_2"],
           let streamSpeakers = golden["dit_stream_speakers"],
           let streamTime = golden["dit_stream_time"],
           let expectedOutput1 = golden["dit_stream_output_1"],
           let expectedOutput2 = golden["dit_stream_output_2"],
           let expectedK1 = golden["dit_stream_k_1"],
           let expectedV1 = golden["dit_stream_v_1"],
           let expectedK2 = golden["dit_stream_k_2"],
           let expectedV2 = golden["dit_stream_v_2"],
           let expectedCNNFirst2 = golden["dit_stream_cnn_first_2"],
           let expectedCNNSecond2 = golden["dit_stream_cnn_second_2"] {
            let first = model.decoder.estimator(
                x: streamX1, mu: streamMu1, time: streamTime,
                speakers: streamSpeakers, condition: streamCondition1)
            let second = model.decoder.estimator(
                x: streamX2, mu: streamMu2, time: streamTime,
                speakers: streamSpeakers, condition: streamCondition2,
                cache: first.cache)
            metrics += try [
                compare("dit_stream_output_1", first.output, expectedOutput1),
                compare("dit_stream_k_1", first.cache.layers[0].attention.keys, expectedK1),
                compare("dit_stream_v_1", first.cache.layers[0].attention.values, expectedV1),
                compare("dit_stream_output_2", second.output, expectedOutput2),
                compare("dit_stream_k_2", second.cache.layers[0].attention.keys, expectedK2),
                compare("dit_stream_v_2", second.cache.layers[0].attention.values, expectedV2),
                compare("dit_stream_cnn_first_2", second.cache.layers[0].convolution.first, expectedCNNFirst2),
                compare("dit_stream_cnn_second_2", second.cache.layers[0].convolution.second, expectedCNNSecond2),
            ]
        }

        let jsonEncoder = JSONEncoder()
        jsonEncoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        FileHandle.standardOutput.write(try jsonEncoder.encode(metrics))
        FileHandle.standardOutput.write(Data("\n".utf8))
        guard metrics.allSatisfy({ $0.cosineSimilarity >= 0.999 }) else {
            throw MiniCPMToken2WavError.invalidWeights(
                "MLX/PyTorch flow cosine similarity is below 0.999")
        }
    }
}
