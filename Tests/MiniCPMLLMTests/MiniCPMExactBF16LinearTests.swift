import Foundation
import Metal
import XCTest
import MLX
import MLXNN
import MLXCommon
@testable import MiniCPMLLM

/// Model-free coverage for MiniCPM's dense Linear factory and MPSGraph bridge.
final class MiniCPMExactBF16LinearTests: XCTestCase {
    func testBF16MPSGraphOutputCanBeConsumedByMLX() throws {
        guard MTLCreateSystemDefaultDevice() != nil else {
            throw XCTSkip("Metal GPU device is unavailable")
        }

        let weight = MLXArray([
            Float(1), 2, 3,
            -1, 0.5, 2,
        ], [2, 3]).asType(.bfloat16)
        let linear = MiniCPMExactBF16Linear(weight: weight, bias: nil)
        XCTAssertEqual(linear.parameters().flattened().map(\.0), ["weight"])
        let input = MLXArray([
            Float(1), 2, -1,
            2, -1, 0.5,
        ], [2, 3]).asType(.bfloat16)

        let output = linear(input)
        eval(output)
        XCTAssertEqual(output.shape, [2, 2])
        XCTAssertEqual(output.dtype, .bfloat16)
        let outputValues = output.asType(.float32).asArray(Float.self)
        XCTAssertTrue(outputValues.allSatisfy(\.isFinite))
        XCTAssertFalse(outputValues.allSatisfy { $0 == 0 })

        // This is an actual MLX operation on the MPSGraph-written backing
        // buffer, rather than a host-side read of the graph result.
        let consumed = output + MLXArray.zeros([2, 2], dtype: .bfloat16)
        eval(consumed)
        XCTAssertEqual(consumed.shape, [2, 2])
        XCTAssertEqual(consumed.dtype, .bfloat16)
        let consumedValues = consumed.asType(.float32).asArray(Float.self)
        XCTAssertTrue(consumedValues.allSatisfy(\.isFinite))
        XCTAssertEqual(consumedValues, outputValues)

        // A second shape-identical call exercises the cached graph plan.
        let repeated = linear(input)
        eval(repeated)
        XCTAssertEqual(repeated.dtype, .bfloat16)
        XCTAssertEqual(repeated.asType(.float32).asArray(Float.self), outputValues)
    }

    func testBF16NonContiguousInputIsMaterializedForMPSGraph() throws {
        guard MTLCreateSystemDefaultDevice() != nil else {
            throw XCTSkip("Metal GPU device is unavailable")
        }

        let weight = MLXArray([
            Float(1), 2, 3,
            -1, 0.5, 2,
        ], [2, 3]).asType(.bfloat16)
        let linear = MiniCPMExactBF16Linear(weight: weight, bias: nil)

        // Transposing the last two axes creates a legal BF16 view with the
        // same trailing width required by Linear, but non-row-contiguous
        // strides.  The implementation must materialize this view in MLX
        // before handing it to MPSGraph's no-copy TensorData bridge.
        let base = MLXArray([
            Float(1), 2, 3, 4, 5, 6,
            -1, -2, -3, -4, -5, -6,
        ], [2, 3, 2]).asType(.bfloat16)
        let input = base.transposed(0, 2, 1)
        eval(input)
        XCTAssertEqual(input.shape, [2, 2, 3])
        XCTAssertNotEqual(input.asData(access: .noCopy).strides, [6, 3, 1])

        let output = linear(input)
        let expected = matmul(input, weight.transposed())
        eval(output, expected)
        XCTAssertEqual(output.shape, [2, 2, 2])
        XCTAssertEqual(output.dtype, .bfloat16)
        XCTAssertEqual(
            output.asType(.float32).asArray(Float.self),
            expected.asType(.float32).asArray(Float.self))

        // Verify that the MPSGraph-written backing storage remains a normal
        // MLX value after the bridge, not merely a host-side readback.
        let consumed = output + MLXArray.zeros(output.shape, dtype: .bfloat16)
        eval(consumed)
        XCTAssertEqual(
            consumed.asType(.float32).asArray(Float.self),
            output.asType(.float32).asArray(Float.self))
    }

    func testBF16SingletonAxesWithNonStandardStridesRemainNoCopyCompatible() throws {
        guard MTLCreateSystemDefaultDevice() != nil else {
            throw XCTSkip("Metal GPU device is unavailable")
        }

        let weight = MLXArray([
            Float(1), 2, 3,
            -1, 0.5, 2,
        ], [2, 3]).asType(.bfloat16)
        let linear = MiniCPMExactBF16Linear(weight: weight, bias: nil)

        // asStrided deliberately gives the two singleton axes arbitrary
        // strides. They never participate in address calculation, while
        // the trailing H axis remains contiguous and suitable for MPSGraph.
        let base = MLXArray([Float(1), 2, 3], [3]).asType(.bfloat16)
        let input = asStrided(base, [1, 1, 3], strides: [97, 41, 1])
        eval(input)
        XCTAssertEqual(input.shape, [1, 1, 3])
        XCTAssertEqual(input.asData(access: .noCopy).strides, [97, 41, 1])
        XCTAssertEqual(input.asType(.float32).asArray(Float.self), [1, 2, 3])

        let output = linear(input)
        let expected = matmul(input, weight.transposed())
        eval(output, expected)
        XCTAssertEqual(output.shape, [1, 1, 2])
        XCTAssertEqual(output.dtype, .bfloat16)
        XCTAssertEqual(
            output.asType(.float32).asArray(Float.self),
            expected.asType(.float32).asArray(Float.self))

        let consumed = output + MLXArray.zeros(output.shape, dtype: .bfloat16)
        eval(consumed)
        XCTAssertEqual(
            consumed.asType(.float32).asArray(Float.self),
            output.asType(.float32).asArray(Float.self))
    }

    func testSharedQueueT1T4AlternationAcrossLinearInstances() throws {
        guard MTLCreateSystemDefaultDevice() != nil else {
            throw XCTSkip("Metal GPU device is unavailable")
        }

        // Integer-valued BF16 operands keep the expected sums exactly
        // representable while still exercising distinct plans for T=1 and
        // T=4 on two independent Linear instances.
        let matrixElementCount = 8 * 16
        let weightAValues: [Float] = Array(0 ..< matrixElementCount).map { index in
            let value = (index * 7) % 5 - 2
            return Float(value)
        }
        let weightBValues: [Float] = Array(0 ..< matrixElementCount).map { index in
            let value = (index * 11 + 1) % 7 - 3
            return Float(value)
        }
        let inputT1Values: [Float] = Array(0 ..< 16).map { index in
            let value = (index * 5 + 2) % 7 - 3
            return Float(value)
        }
        let inputT4Values: [Float] = Array(0 ..< 64).map { index in
            let value = (index * 3 + 1) % 5 - 2
            return Float(value)
        }
        let weightA = MLXArray(weightAValues, [8, 16]).asType(.bfloat16)
        let weightB = MLXArray(weightBValues, [8, 16]).asType(.bfloat16)
        let inputT1 = MLXArray(inputT1Values, [1, 16]).asType(.bfloat16)
        let inputT4 = MLXArray(inputT4Values, [4, 16]).asType(.bfloat16)
        let linearA = MiniCPMExactBF16Linear(weight: weightA, bias: nil)
        let linearB = MiniCPMExactBF16Linear(weight: weightB, bias: nil)

        func values(_ output: MLXArray) -> [Float] {
            eval(output)
            return output.asType(.float32).asArray(Float.self)
        }

        let expectedA1 = matmul(inputT1, weightA.transposed())
        let expectedA4 = matmul(inputT4, weightA.transposed())
        let expectedB1 = matmul(inputT1, weightB.transposed())
        let expectedB4 = matmul(inputT4, weightB.transposed())
        let firstA1 = linearA(inputT1)
        let firstB4 = linearB(inputT4)
        let firstA4 = linearA(inputT4)
        let firstB1 = linearB(inputT1)
        XCTAssertEqual(values(firstA1), values(expectedA1))
        XCTAssertEqual(values(firstA4), values(expectedA4))
        XCTAssertEqual(values(firstB1), values(expectedB1))
        XCTAssertEqual(values(firstB4), values(expectedB4))

        // Alternate both shapes and instances repeatedly.  The shared queue
        // lock should keep every result bit-stable after plan creation.
        let baselineA1 = values(firstA1)
        let baselineA4 = values(firstA4)
        let baselineB1 = values(firstB1)
        let baselineB4 = values(firstB4)
        for _ in 0 ..< 4 {
            XCTAssertEqual(values(linearA(inputT1)), baselineA1)
            XCTAssertEqual(values(linearB(inputT4)), baselineB4)
            XCTAssertEqual(values(linearA(inputT4)), baselineA4)
            XCTAssertEqual(values(linearB(inputT1)), baselineB1)
        }
    }

    func testFloat32InputUsesLinearFallback() {
        let weight = MLXArray([
            Float(1), 2, 3,
            -1, 0.5, 2,
        ], [2, 3])
        let linear = MiniCPMExactBF16Linear(weight: weight, bias: nil)
        let input = MLXArray([
            Float(1), 2, -1,
            2, -1, 0.5,
        ], [2, 3])

        let output = linear(input)
        let expected = matmul(input, weight.transposed())
        eval(output, expected)
        XCTAssertEqual(output.shape, [2, 2])
        XCTAssertEqual(output.dtype, .float32)
        XCTAssertEqual(output.asArray(Float.self), expected.asArray(Float.self))
    }

    func testEightBitMiniCPMFactoryStillUsesQuantizedLinear() throws {
        let configURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("MiniCPMExactBF16LinearTests-\(UUID().uuidString).json")
        defer { try? FileManager.default.removeItem(at: configURL) }
        let configJSON = #"""
        {
          "model_type": "qwen3",
          "hidden_size": 32,
          "num_hidden_layers": 1,
          "intermediate_size": 32,
          "num_attention_heads": 1,
          "num_key_value_heads": 1,
          "head_dim": 32,
          "rms_norm_eps": 1e-6,
          "vocab_size": 32,
          "max_position_embeddings": 32,
          "rope_theta": 10000,
          "tie_word_embeddings": false,
          "quantization": {
            "bits": 8,
            "group_size": 32,
            "mode": "affine"
          }
        }
        """#
        try Data(configJSON.utf8).write(to: configURL)

        let config = try MiniCPMMLXConfig.load(from: configURL)
        let model = MiniCPMLanguageModel(config: config)
        let layer = try XCTUnwrap(model.layers.first)
        XCTAssertTrue(layer.selfAttention.qProj is QuantizedLinear)
        XCTAssertTrue(layer.selfAttention.kProj is QuantizedLinear)
        XCTAssertTrue(layer.selfAttention.vProj is QuantizedLinear)
        XCTAssertTrue(layer.selfAttention.oProj is QuantizedLinear)
        XCTAssertTrue(layer.mlp.gateProj is QuantizedLinear)
        XCTAssertTrue(layer.mlp.upProj is QuantizedLinear)
        XCTAssertTrue(layer.mlp.downProj is QuantizedLinear)
        XCTAssertTrue(model.lmHead is QuantizedLinear)
    }
}
