import Foundation
import XCTest
import MLX
@testable import MiniCPMLLM

/// Regression gates for the dense (BF16) MiniCPM language-backbone bundle.
///
/// The repository fixture is quantized, so these tests create a tiny dense
/// bundle in a temporary directory.  Keeping the checkpoint synthetic makes
/// the loader contract and the forward oracle independent of a downloaded
/// model and keeps this suite fast enough to run on every change.
final class MiniCPMDenseLoadingTests: XCTestCase {
    private let configJSON = #"""
    {
      "model_type": "qwen3",
      "hidden_size": 4,
      "num_hidden_layers": 1,
      "intermediate_size": 4,
      "num_attention_heads": 1,
      "num_key_value_heads": 1,
      "head_dim": 4,
      "rms_norm_eps": 1e-6,
      "vocab_size": 5,
      "max_position_embeddings": 32,
      "rope_theta": 10000,
      "tie_word_embeddings": false,
      "quantization": {
        "bits": 0,
        "group_size": 4,
        "mode": "none"
      }
    }
    """#

    // Main-thread Swift test execution must confirm this BF16 oracle; this
    // delegated task intentionally did not start a Swift test process.
    private let expectedHidden: [Float] = [
        2, 0, 0, 0,
    ]
    private let expectedLogits: [Float] = [
        2, -2, 0, 0, 0,
    ]

    func testBitsZeroConfigSelectsDensePath() throws {
        let directory = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let configURL = directory.appendingPathComponent("config.json")
        try Data(configJSON.utf8).write(to: configURL)

        let config = try MiniCPMMLXConfig.load(from: configURL)
        XCTAssertEqual(config.quantBits, 0)
        XCTAssertFalse(config.isQuantized)
        XCTAssertEqual(config.quantGroupSize, 4)
        XCTAssertEqual(config.hiddenSize, 4)
        XCTAssertEqual(config.vocabSize, 5)
    }

    func testDenseSafetensorsLoadIsStrictAndScaleFree() throws {
        let validDirectory = try makeDenseBundle()
        defer { try? FileManager.default.removeItem(at: validDirectory) }
        let config = try MiniCPMMLXConfig.load(from: validDirectory)
        let model = MiniCPMLanguageModel(config: config)

        XCTAssertNotNil(model.embedTokens.dense)
        XCTAssertNotNil(model.lmHead)
        try MiniCPMWeightLoader.load(model: model, from: validDirectory)

        let checkpoint = try MLX.loadArrays(
            url: validDirectory.appendingPathComponent("model.safetensors"))
        XCTAssertEqual(checkpoint["model.embed_tokens.weight"]?.dtype, .bfloat16)
        XCTAssertEqual(checkpoint["lm_head.weight"]?.dtype, .bfloat16)
        XCTAssertFalse(checkpoint.keys.contains { key in
            key.hasSuffix(".scales") || key.hasSuffix(".biases")
        })
        let parameterKeys = model.parameters().flattened().map(\.0)
        XCTAssertFalse(parameterKeys.contains { key in
            key.hasSuffix(".scales") || key.hasSuffix(".biases")
        })

        let missingDirectory = try makeDenseBundle(omitting: "model.embed_tokens.weight")
        defer { try? FileManager.default.removeItem(at: missingDirectory) }
        let missingModel = MiniCPMLanguageModel(config: config)
        XCTAssertThrowsError(try MiniCPMWeightLoader.load(
            model: missingModel, from: missingDirectory)) { error in
            guard case MiniCPMLLMError.invalidWeights(let message) = error else {
                return XCTFail("expected invalidWeights, got \(error)")
            }
            XCTAssertTrue(message.contains("missing tensors"))
            XCTAssertTrue(message.contains("embed_tokens.weight"))
        }

        let extraDirectory = try makeDenseBundle(
            adding: ("model.embed_tokens.scales", MLXArray.zeros([5, 1], dtype: .bfloat16)))
        defer { try? FileManager.default.removeItem(at: extraDirectory) }
        let extraModel = MiniCPMLanguageModel(config: config)
        XCTAssertThrowsError(try MiniCPMWeightLoader.load(
            model: extraModel, from: extraDirectory)) { error in
            guard case MiniCPMLLMError.invalidWeights(let message) = error else {
                return XCTFail("expected invalidWeights, got \(error)")
            }
            XCTAssertTrue(message.contains("unexpected Qwen tensors"))
            XCTAssertTrue(message.contains("embed_tokens.scales"))
        }
    }

    func testDenseForwardHasDeterministicLogits() throws {
        let directory = try makeDenseBundle()
        defer { try? FileManager.default.removeItem(at: directory) }
        let config = try MiniCPMMLXConfig.load(from: directory)
        let model = MiniCPMLanguageModel(config: config)
        try MiniCPMWeightLoader.load(model: model, from: directory)

        let inputIds = MLXArray([Int32(1), Int32(3)]).expandedDimensions(axis: 0)
        let output = model.forward(inputIds: inputIds, state: model.initialState())
        eval(output.hidden, output.logits)
        let hidden = output.hidden[0, 1].asType(.float32).asArray(Float.self)
        let logits = output.logits[0, 1].asType(.float32).asArray(Float.self)

        XCTAssertEqual(hidden.count, expectedHidden.count)
        XCTAssertEqual(logits.count, expectedLogits.count)
        for (actual, expected) in zip(hidden, expectedHidden) {
            XCTAssertEqual(actual, expected, accuracy: 1e-3)
        }
        for (actual, expected) in zip(logits, expectedLogits) {
            XCTAssertEqual(actual, expected, accuracy: 1e-3)
        }
    }

    private func makeTemporaryDirectory() throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("MiniCPMDenseLoadingTests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: false)
        return directory
    }

    private func makeDenseBundle(
        omitting omittedKey: String? = nil,
        adding extra: (String, MLXArray)? = nil
    ) throws -> URL {
        let directory = try makeTemporaryDirectory()
        try Data(configJSON.utf8).write(to: directory.appendingPathComponent("config.json"))
        var weights = denseWeights()
        if let omittedKey { weights.removeValue(forKey: omittedKey) }
        if let extra { weights[extra.0] = extra.1 }
        try MLX.save(
            arrays: weights,
            metadata: ["format": "mlx"],
            url: directory.appendingPathComponent("model.safetensors"))
        return directory
    }

    private func denseWeights() -> [String: MLXArray] {
        func tensor(_ values: [Float], _ shape: [Int]) -> MLXArray {
            MLXArray(values, shape).asType(.bfloat16)
        }
        return [
            "model.embed_tokens.weight": tensor([
                0.00, 1.00, 0.00, 0.00,
                0.00, 0.00, 1.00, 0.00,
                0.00, 0.00, 0.00, 1.00,
                1.00, 0.00, 0.00, 0.00,
                0.00, 0.00, 0.00, 0.00,
            ], [5, 4]),
            "model.norm.weight": tensor([1, 1, 1, 1], [4]),
            "lm_head.weight": tensor([
                1.00, 0.00, 0.00, 0.00,
                -1.00, 0.00, 0.00, 0.00,
                0.00, 1.00, 0.00, 0.00,
                0.00, 0.00, 1.00, 0.00,
                0.00, 0.00, 0.00, 1.00,
            ], [5, 4]),
            "model.layers.0.input_layernorm.weight": tensor([1, 1, 1, 1], [4]),
            "model.layers.0.post_attention_layernorm.weight": tensor([1, 1, 1, 1], [4]),
            "model.layers.0.self_attn.q_proj.weight": tensor([Float](repeating: 0, count: 16), [4, 4]),
            "model.layers.0.self_attn.k_proj.weight": tensor([Float](repeating: 0, count: 16), [4, 4]),
            "model.layers.0.self_attn.v_proj.weight": tensor([Float](repeating: 0, count: 16), [4, 4]),
            "model.layers.0.self_attn.o_proj.weight": tensor([Float](repeating: 0, count: 16), [4, 4]),
            "model.layers.0.self_attn.q_norm.weight": tensor([1, 1, 1, 1], [4]),
            "model.layers.0.self_attn.k_norm.weight": tensor([1, 1, 1, 1], [4]),
            "model.layers.0.mlp.gate_proj.weight": tensor([Float](repeating: 0, count: 16), [4, 4]),
            "model.layers.0.mlp.up_proj.weight": tensor([Float](repeating: 0, count: 16), [4, 4]),
            "model.layers.0.mlp.down_proj.weight": tensor([Float](repeating: 0, count: 16), [4, 4]),
        ]
    }
}
