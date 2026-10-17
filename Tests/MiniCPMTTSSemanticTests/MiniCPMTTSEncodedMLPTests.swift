import XCTest
import MLX
import MLXNN
import MiniCPMLLM

@testable import MiniCPMTTSSemantic

final class MiniCPMTTSEncodedMLPTests: XCTestCase {
    private func array(shape: [Int], phase: Int, divisor: Float) -> MLXArray {
        let count = shape.reduce(1, *)
        let values = (0 ..< count).map { index in
            Float((index * 17 + phase) % 97 - 48) / divisor
        }
        return MLXArray(values, shape).asType(.bfloat16)
    }

    private func projections(hidden: Int, intermediate: Int) -> (
        gate: MiniCPMExactBF16Linear, up: MiniCPMExactBF16Linear,
        down: MiniCPMExactBF16Linear
    ) {
        let divisor = Float(hidden).squareRoot() * 24
        return (
            MiniCPMExactBF16Linear(weight: array(
                shape: [intermediate, hidden], phase: 7, divisor: divisor)),
            MiniCPMExactBF16Linear(weight: array(
                shape: [intermediate, hidden], phase: 31, divisor: divisor)),
            MiniCPMExactBF16Linear(weight: array(
                shape: [hidden, intermediate], phase: 53,
                divisor: Float(intermediate).squareRoot() * 24))
        )
    }

    private func reference(
        _ input: MLXArray, gate: Linear, up: Linear, down: Linear
    ) -> MLXArray {
        let projected = MiniCPMTTSBatchedBF16Projection.apply(
            input, projections: [gate, up], environment: [:])
        return down(MiniCPMExactBF16SiLU.apply(projected[0]) * projected[1])
    }

    private func assertBitwiseEqual(
        _ actual: MLXArray, _ expected: MLXArray,
        file: StaticString = #filePath, line: UInt = #line
    ) {
        XCTAssertEqual(actual.shape, expected.shape, file: file, line: line)
        XCTAssertEqual(actual.dtype, .bfloat16, file: file, line: line)
        XCTAssertEqual(expected.dtype, .bfloat16, file: file, line: line)
        XCTAssertEqual(
            actual.asData(access: .copy).data, expected.asData(access: .copy).data,
            file: file, line: line)
        // Exercise an ordinary MLX consumer of the externally written
        // output, after the encoded path has completed its private queue.
        let consumed = actual + MLXArray.zeros(actual.shape, dtype: .bfloat16)
        let expectedConsumed = expected + MLXArray.zeros(expected.shape, dtype: .bfloat16)
        XCTAssertEqual(
            consumed.asData(access: .copy).data, expectedConsumed.asData(access: .copy).data,
            file: file, line: line)
    }

    private func requireMetalSupport() throws {
        guard MiniCPMTTSEncodedBF16MLP.isAvailable else {
            throw XCTSkip("Encoded BF16 MLP Metal pipeline is unavailable")
        }
    }

    func testDecodeAndPrefillMatchOriginalAtDecoderWidths() throws {
        try requireMetalSupport()
        for (hidden, intermediate) in [(16, 32), (64, 256), (768, 3072)] {
            let layers = projections(hidden: hidden, intermediate: intermediate)
            for tokens in [1, 3] {
                let input = array(shape: [1, tokens, hidden], phase: 3, divisor: 24)
                let expected = reference(
                    input, gate: layers.gate, up: layers.up, down: layers.down)
                let actual = try XCTUnwrap(MiniCPMTTSEncodedBF16MLP.apply(
                    input: input, gate: layers.gate, up: layers.up, down: layers.down))
                assertBitwiseEqual(actual, expected)
            }
        }
    }

    func testNoncontiguousInputMatchesOriginalAtDecoderWidths() throws {
        try requireMetalSupport()
        let layers = projections(hidden: 768, intermediate: 3072)
        let input = array(shape: [2, 3, 768], phase: 11, divisor: 24).transposed(1, 0, 2)
        eval(input)
        XCTAssertNotEqual(input.asData(access: .noCopy).strides, [1536, 768, 1])
        let expected = reference(input, gate: layers.gate, up: layers.up, down: layers.down)
        let actual = try XCTUnwrap(MiniCPMTTSEncodedBF16MLP.apply(
            input: input, gate: layers.gate, up: layers.up, down: layers.down))
        assertBitwiseEqual(actual, expected)
    }

    func testParameterUpdatesRefreshEncodedFeedsAndPreserveReturnedOutputs() throws {
        try requireMetalSupport()
        let layers = projections(hidden: 64, intermediate: 256)
        let input = array(shape: [1, 2, 64], phase: 13, divisor: 24)
        let original = try XCTUnwrap(MiniCPMTTSEncodedBF16MLP.apply(
            input: input, gate: layers.gate, up: layers.up, down: layers.down))
        let originalBytes = original.asData(access: .copy).data
        try layers.gate.update(parameters: ModuleParameters.unflattened([
            "weight": array(shape: [256, 64], phase: 61, divisor: 192),
        ]), verify: .all)
        try layers.up.update(parameters: ModuleParameters.unflattened([
            "weight": array(shape: [256, 64], phase: 83, divisor: 192),
        ]), verify: .all)
        try layers.down.update(parameters: ModuleParameters.unflattened([
            "weight": array(shape: [64, 256], phase: 23, divisor: 384),
        ]), verify: .all)
        let expected = reference(input, gate: layers.gate, up: layers.up, down: layers.down)
        let updated = try XCTUnwrap(MiniCPMTTSEncodedBF16MLP.apply(
            input: input, gate: layers.gate, up: layers.up, down: layers.down))
        assertBitwiseEqual(updated, expected)
        XCTAssertNotEqual(updated.asData(access: .copy).data, originalBytes)
        XCTAssertEqual(original.asData(access: .copy).data, originalBytes)

        // Alternate shapes after updating: cached graph plans must use the
        // newly prepared weight feeds for every input shape.
        let nextInput = array(shape: [1, 1, 64], phase: 41, divisor: 24)
        let next = try XCTUnwrap(MiniCPMTTSEncodedBF16MLP.apply(
            input: nextInput, gate: layers.gate, up: layers.up, down: layers.down))
        assertBitwiseEqual(next, reference(
            nextInput, gate: layers.gate, up: layers.up, down: layers.down))
    }

    func testUnsupportedDtypesAndProjectionImplementationsReturnNil() {
        let layers = projections(hidden: 16, intermediate: 32)
        let input = array(shape: [1, 1, 16], phase: 3, divisor: 24)
        XCTAssertNil(MiniCPMTTSEncodedBF16MLP.apply(
            input: input.asType(.float32), gate: layers.gate, up: layers.up, down: layers.down))
        XCTAssertNil(MiniCPMTTSEncodedBF16MLP.apply(
            input: input, gate: Linear(weight: layers.gate.weight),
            up: layers.up, down: layers.down))
        let biased = MiniCPMExactBF16Linear(
            weight: layers.gate.weight, bias: MLXArray.zeros([32], dtype: .bfloat16))
        XCTAssertNil(MiniCPMTTSEncodedBF16MLP.apply(
            input: input, gate: biased, up: layers.up, down: layers.down))
        let wrongDown = MiniCPMExactBF16Linear(
            weight: array(shape: [16, 48], phase: 3, divisor: 24))
        XCTAssertNil(MiniCPMTTSEncodedBF16MLP.apply(
            input: input, gate: layers.gate, up: layers.up, down: wrongDown))
    }
}
