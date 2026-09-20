import Foundation
import XCTest
import MLX
import MLXNN
@testable import MiniCPMLLM

/// Bit-level regression coverage for the dense MiniCPM BF16 RMSNorm path.
/// The fixture is one head/token from the real layer-16 q_norm trace; it is
/// intentionally small enough to run without loading the model bundle.
final class MiniCPMExactBF16RMSNormTests: XCTestCase {
    private let dimensions = 128
    private let epsilon: Float = 1e-6

    func testFusedRMSNormReproducesKnownSingleMismatch() throws {
        let fixture = try loadFixture()
        let fused = RMSNorm(dimensions: dimensions, eps: epsilon)
        fused.update(parameters: ModuleParameters.unflattened([
            "weight": fixture.weight,
        ]))

        let actual = fused(fixture.input)
        eval(actual)
        let values = actual.asType(.float32).asArray(Float.self)
        let reference = fixture.reference.asType(.float32).asArray(Float.self)
        XCTAssertEqual(values.count, dimensions)
        XCTAssertEqual(reference.count, dimensions)

        let mismatches = zip(values, reference).filter { $0 != $1 }
        XCTAssertEqual(
            mismatches.count, 1,
            "fused RMSNorm should reproduce the documented one-value layer-16 mismatch")
        XCTAssertEqual(values[20], -0.384765625)
        XCTAssertEqual(reference[20], -0.38671875)
    }

    func testExactBF16RMSNormFixtureIsBitExactAndKeepsWeightKey() throws {
        let fixture = try loadFixture()
        let exact = MiniCPMExactBF16RMSNorm(dimensions: dimensions, eps: epsilon)
        exact.update(parameters: ModuleParameters.unflattened([
            "weight": fixture.weight,
        ]))

        XCTAssertEqual(exact.parameters().flattened().map(\.0), ["weight"])
        let actual = exact(fixture.input)
        eval(actual)
        XCTAssertEqual(actual.dtype, .bfloat16)
        XCTAssertEqual(actual.shape, fixture.input.shape)
        XCTAssertEqual(
            actual.asType(.float32).asArray(Float.self),
            fixture.reference.asType(.float32).asArray(Float.self))

        // A second call exercises the same operation graph without changing
        // the parameter key or BF16 result bytes.
        let repeated = exact(fixture.input)
        eval(repeated)
        XCTAssertEqual(
            repeated.asType(.float32).asArray(Float.self),
            fixture.reference.asType(.float32).asArray(Float.self))
    }

    func testBroadcastLeadingDimensionsRemainBitExact() throws {
        let fixture = try loadFixture()
        let exact = MiniCPMExactBF16RMSNorm(dimensions: dimensions, eps: epsilon)
        exact.update(parameters: ModuleParameters.unflattened([
            "weight": fixture.weight,
        ]))
        let input = MLX.broadcast(fixture.input, to: [2, 3, dimensions])
        let reference = MLX.broadcast(fixture.reference, to: [2, 3, dimensions])

        let actual = exact(input)
        eval(actual, reference)
        XCTAssertEqual(actual.shape, [2, 3, dimensions])
        XCTAssertEqual(actual.dtype, .bfloat16)
        XCTAssertEqual(
            actual.asType(.float32).asArray(Float.self), reference.asType(.float32).asArray(Float.self))
    }

    func testFloat32InputUsesInheritedFusedFallback() throws {
        let fixture = try loadFixture()
        let exact = MiniCPMExactBF16RMSNorm(dimensions: dimensions, eps: epsilon)
        let fused = RMSNorm(dimensions: dimensions, eps: epsilon)
        let weight = fixture.weight.asType(.float32)
        exact.update(parameters: ModuleParameters.unflattened(["weight": weight]))
        fused.update(parameters: ModuleParameters.unflattened(["weight": weight]))

        let input = fixture.input.asType(.float32)
        let exactOutput = exact(input)
        let fusedOutput = fused(input)
        eval(exactOutput, fusedOutput)
        XCTAssertEqual(exactOutput.dtype, .float32)
        XCTAssertEqual(
            exactOutput.asArray(Float.self), fusedOutput.asArray(Float.self))
    }

    func testFloat32WeightUsesInheritedMixedPrecisionFallback() throws {
        let fixture = try loadFixture()
        let exact = MiniCPMExactBF16RMSNorm(dimensions: dimensions, eps: epsilon)
        let fused = RMSNorm(dimensions: dimensions, eps: epsilon)
        let weight = fixture.weight.asType(.float32)
        exact.update(parameters: ModuleParameters.unflattened(["weight": weight]))
        fused.update(parameters: ModuleParameters.unflattened(["weight": weight]))

        let exactOutput = exact(fixture.input)
        let fusedOutput = fused(fixture.input)
        eval(exactOutput, fusedOutput)
        XCTAssertEqual(exactOutput.dtype, fusedOutput.dtype)
        XCTAssertEqual(
            exactOutput.asType(.float32).asArray(Float.self),
            fusedOutput.asType(.float32).asArray(Float.self))
    }

    func testValidationRejectsMalformedLastDimensionAndWeightRank() {
        let input = MLXArray.zeros([1, 2, dimensions], dtype: .bfloat16)
        XCTAssertEqual(
            MiniCPMExactBF16RMSNorm.validationError(
                input: input,
                weight: MLXArray.zeros([dimensions / 2], dtype: .bfloat16)),
            "MiniCPMExactBF16RMSNorm input width does not match weight")
        XCTAssertEqual(
            MiniCPMExactBF16RMSNorm.validationError(
                input: input,
                weight: MLXArray.zeros([1, dimensions], dtype: .bfloat16)),
            "MiniCPMExactBF16RMSNorm requires a rank-1 weight")
    }

    func testLayer24Head6MatchesPyTorchMPS() {
        let values: [Float] = [
            1.796875, 0.90625, 0.29296875, 1.7109375, 2.625, -2.921875, 1.4296875,
            -2.40625, -1.390625, -0.47265625, -1.3125, 2.09375, -2.015625,
            -4.6875, 0.080078125, -0.734375, -0.71484375, 0.6171875, -1.3984375,
            1.7578125, 0.2890625, -2.640625, -2.84375, 0.0395507812, -0.275390625,
            1.3515625, -0.337890625, -0.6640625, 0.478515625, -0.9375, 0.147460938,
            -1.34375, 0.127929688, 0.29296875, -0.6015625, -0.384765625,
            -0.75390625, -0.139648438, 2.109375, -0.55859375, 0.58203125, -3.40625,
            1.53125, -1.3671875, -1.0546875, -2.15625, 0.357421875, -0.3125,
            -2.421875, -0.8828125, -1.90625, 1.1953125, -1.0078125, -2.90625,
            0.333984375, 0.7109375, -4.84375, 0.93359375, 1.859375, 4.46875,
            -3.125, -0.74609375, 0.76953125, 0.38671875, -3.578125, -1.828125,
            -3.890625, -5.125, -0.5625, -1.5, 2.375, -0.3359375, 0.147460938,
            -0.90625, -1.3125, 0.400390625, -2.28125, -0.1728515625, -0.67578125,
            -0.609375, -1.6328125, -2.640625, -0.5625, -1.890625, 0.341796875,
            0.0329589844, 3.046875, 0.68359375, 0.478515625, 1.5859375, 1.59375,
            1.0859375, -0.443359375, 3.5625, -0.0125732422, -1.453125, 1.5234375,
            0.337890625, 2.1875, -0.1962890625, 2.296875, 0.140625, -1.4453125,
            0.51171875, 0.609375, 0.80859375, 0.33203125, -0.7890625, -0.546875,
            -0.82421875, -1.0625, -1.84375, -0.3515625, -2.703125, 0.130859375,
            0.291015625, 1.3828125, 5.625, -0.181640625, -0.322265625, -0.2197265625,
            -5.75, -1.25, -2.46875, -1.1875, 1.515625, 1.859375, -0.2421875,
        ]
        let exact = MiniCPMExactBF16RMSNorm(dimensions: 128, eps: 1e-6)
        exact.update(parameters: ModuleParameters.unflattened([
            "weight": MLXArray.ones([128], dtype: .bfloat16),
        ]))
        let output = exact(MLXArray(values, [1, 1, 128]).asType(.bfloat16))
        eval(output)
        let actual = output.asType(.float32).asArray(Float.self)
        let expected: [Int: Float] = [
            25: 0.734375,
            26: -0.18359375,
            78: -0.3671875,
            97: 0.18359375,
            113: -1.46875,
        ]
        XCTAssertEqual(output.dtype, .bfloat16)
        for (index, value) in expected {
            XCTAssertEqual(actual[index], value, "index \(index)")
        }
    }

    /// Source: cached sequence [151643,151645]→198, layer2/head26 q_norm.
    /// The input and expected value are from the official PyTorch MPS trace.
    func testCachedLayer2Head26QNormBoundaryMatchesPyTorchMPS() {
        let values: [Float] = [
            -0.0830078125, -0.1044921875, -0.07177734375, 0.03540039062,
            0.0283203125, -0.08544921875, -0.1318359375, -0.091796875,
            -0.1474609375, 0.04907226562, 0.169921875, -0.002380371094,
            -0.02490234375, 0.03857421875, 0.03271484375, -0.04833984375,
            0.004669189453, 0.04858398438, -0.04736328125, 0.376953125,
            -0.408203125, -0.09423828125, 0.04736328125, 0.02172851562,
            -0.03637695312, 0.0791015625, -0.01251220703, 0.1088867188,
            -0.003860473633, 0.02502441406, 0.009948730469, 0.02844238281,
            0.01153564453, -0.06982421875, 0.2578125, -0.02026367188,
            0.003707885742, 0.03833007812, 0.0004177093506, 0.05395507812,
            0.01293945312, 0.01904296875, 0.314453125, 0.008361816406,
            0.08642578125, 0.004577636719, -0.029296875, -0.04248046875,
            0.05249023438, 0.03930664062, -0.01721191406, -0.005737304688,
            0.02746582031, -0.004089355469, -0.003890991211, 0.002944946289,
            -0.05322265625, -0.004913330078, -0.03686523438, -0.0029296875,
            0.002258300781, -0.03955078125, 0.03491210938, 0.02209472656,
            0.02685546875, -0.01831054688, 0.05883789062, -0.0537109375,
            -0.01141357422, 0.0005340576172, 0.033203125, 0.1328125,
            -0.009460449219, 0.008605957031, -0.01220703125, -0.04150390625,
            0.0107421875, 0.01385498047, 0.01965332031, 0.1079101562,
            -0.01318359375, -0.02978515625, -0.01696777344, 0.05932617188,
            -0.1196289062, 0.251953125, 0.1186523438, -0.06201171875,
            -3.838539124e-05, -0.03979492188, -4.434585571e-05, -0.0205078125,
            0.004547119141, -0.0751953125, 0.01635742188, -0.02893066406,
            -0.01916503906, 0.06494140625, -0.1875, -0.0234375,
            0.0008850097656, 0.01165771484, -0.02478027344, -0.03125,
            -0.02697753906, -0.03833007812, 0.1494140625, -0.1162109375,
            -0.02966308594, 0.03979492188, -0.01977539062, 0.06030273438,
            0.2158203125, -0.0703125, -0.05151367188, -0.001968383789,
            0.03979492188, 0.01519775391, 0.0341796875, -0.003311157227,
            0.009338378906, -0.002029418945, 0.01599121094, 0.04248046875,
            -0.02001953125, 0.00634765625, -0.01904296875, 0.01312255859,
        ]
        var weightValues = Array(repeating: Float(1), count: dimensions)
        weightValues[32] = 1.5234375

        let exact = MiniCPMExactBF16RMSNorm(dimensions: dimensions, eps: epsilon)
        exact.update(parameters: ModuleParameters.unflattened([
            "weight": MLXArray(weightValues, [dimensions]).asType(.bfloat16),
        ]))

        let input = MLXArray(values, [1, 1, dimensions]).asType(.bfloat16)
        let output = exact(input)
        eval(output)

        XCTAssertEqual(output.dtype, .bfloat16)
        XCTAssertEqual(output.shape, [1, 1, dimensions])
        XCTAssertEqual(
            output.asType(.float32).asArray(Float.self)[32],
            0.2041015625)
    }

    func testMultipleInstancesAlternateRepeatedAndMaterializeNonContiguousInput() {
        let values: [Float] = Array(repeating: 1, count: dimensions)
        let input = MLXArray(values, [1, 1, dimensions]).asType(.bfloat16)
        let nonContiguous = MLX.broadcast(input, to: [2, 3, dimensions]).transposed(1, 0, 2)
        let weight = MLXArray.ones([dimensions], dtype: .bfloat16)
        let first = MiniCPMExactBF16RMSNorm(dimensions: dimensions, eps: epsilon)
        let second = MiniCPMExactBF16RMSNorm(dimensions: dimensions, eps: epsilon)
        first.update(parameters: ModuleParameters.unflattened(["weight": weight]))
        second.update(parameters: ModuleParameters.unflattened(["weight": weight]))

        var previousFirst: [Float]?
        var previousSecond: [Float]?
        for _ in 0..<4 {
            let firstOutput = first(input)
            let secondOutput = second(nonContiguous)
            eval(firstOutput, secondOutput)
            XCTAssertEqual(firstOutput.shape, [1, 1, dimensions])
            XCTAssertEqual(secondOutput.shape, [3, 2, dimensions])
            XCTAssertEqual(firstOutput.dtype, .bfloat16)
            XCTAssertEqual(secondOutput.dtype, .bfloat16)
            let firstValues = firstOutput.asType(.float32).asArray(Float.self)
            let secondValues = secondOutput.asType(.float32).asArray(Float.self)
            XCTAssertEqual(firstValues, Array(repeating: 1, count: dimensions))
            XCTAssertEqual(secondValues, Array(repeating: 1, count: 6 * dimensions))
            XCTAssertEqual(firstValues, previousFirst ?? firstValues)
            XCTAssertEqual(secondValues, previousSecond ?? secondValues)
            previousFirst = firstValues
            previousSecond = secondValues
        }
    }

    func testDenseAndQuantizedNormRouting() throws {
        let dense = MiniCPMLanguageModel(config: try loadConfig(bits: 0))
        let denseLayer = try XCTUnwrap(dense.layers.first)
        XCTAssertTrue(denseLayer.inputLayerNorm is MiniCPMExactBF16RMSNorm)
        XCTAssertTrue(denseLayer.postAttentionLayerNorm is MiniCPMExactBF16RMSNorm)
        XCTAssertTrue(denseLayer.selfAttention.qNorm is MiniCPMExactBF16RMSNorm)
        XCTAssertTrue(denseLayer.selfAttention.kNorm is MiniCPMExactBF16RMSNorm)
        XCTAssertTrue(dense.norm is MiniCPMExactBF16RMSNorm)

        let quantized = MiniCPMLanguageModel(config: try loadConfig(bits: 8))
        let quantizedLayer = try XCTUnwrap(quantized.layers.first)
        XCTAssertFalse(quantizedLayer.inputLayerNorm is MiniCPMExactBF16RMSNorm)
        XCTAssertFalse(quantizedLayer.postAttentionLayerNorm is MiniCPMExactBF16RMSNorm)
        XCTAssertFalse(quantizedLayer.selfAttention.qNorm is MiniCPMExactBF16RMSNorm)
        XCTAssertFalse(quantizedLayer.selfAttention.kNorm is MiniCPMExactBF16RMSNorm)
        XCTAssertFalse(quantized.norm is MiniCPMExactBF16RMSNorm)
    }

    private struct Fixture {
        let input: MLXArray
        let weight: MLXArray
        let reference: MLXArray
    }

    private func loadFixture() throws -> Fixture {
        let directory = URL(fileURLWithPath: fixtureDirectory, isDirectory: true)
        let shape = [1, 1, dimensions]
        return Fixture(
            input: try readFloat32(directory.appendingPathComponent("input.f32"), shape: shape)
                .asType(.bfloat16),
            weight: try readFloat32(
                directory.appendingPathComponent("weight.f32"), shape: [dimensions])
                .asType(.bfloat16),
            reference: try readFloat32(
                directory.appendingPathComponent("output.f32"), shape: shape)
                .asType(.bfloat16))
    }

    private func readFloat32(_ url: URL, shape: [Int]) throws -> MLXArray {
        let data = try Data(contentsOf: url)
        let expectedBytes = shape.reduce(1, *) * MemoryLayout<Float>.stride
        guard data.count == expectedBytes else {
            throw NSError(
                domain: "MiniCPMExactBF16RMSNormFixture",
                code: 1,
                userInfo: [NSLocalizedDescriptionKey:
                    "fixture tensor \(url.lastPathComponent) has \(data.count) bytes; "
                        + "expected \(expectedBytes)"])
        }
        let values = data.withUnsafeBytes { raw in
            Array(raw.bindMemory(to: Float.self))
        }
        return MLXArray(values, shape)
    }

    private func loadConfig(bits: Int) throws -> MiniCPMMLXConfig {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("MiniCPMExactBF16RMSNorm-\(UUID().uuidString).json")
        defer { try? FileManager.default.removeItem(at: url) }
        let json = #"""
        {
          "model_type": "qwen3",
          "hidden_size": 32,
          "num_hidden_layers": 1,
          "intermediate_size": 32,
          "num_attention_heads": 4,
          "num_key_value_heads": 2,
          "head_dim": 8,
          "rms_norm_eps": 1e-6,
          "vocab_size": 16,
          "max_position_embeddings": 32,
          "rope_theta": 10000,
          "tie_word_embeddings": false,
          "quantization": {
            "bits": \#(bits),
            "group_size": 32,
            "mode": "affine"
          }
        }
        """#
        try Data(json.utf8).write(to: url)
        return try MiniCPMMLXConfig.load(from: url)
    }

    private var fixtureDirectory: String {
        if let bundled = Bundle.module.url(
            forResource: "ExactBF16RMSNorm",
            withExtension: nil,
            subdirectory: "Fixtures") {
            return bundled.path
        }
        return URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .appendingPathComponent("Fixtures/ExactBF16RMSNorm", isDirectory: true)
            .path
    }
}
