import XCTest
import MLX
import MLXNN
import MiniCPMLLM

@testable import MiniCPMTTSSemantic

final class MiniCPMTTSEncodedBlockTests: XCTestCase {
    private struct Layers {
        let output: MiniCPMExactBF16Linear
        let norm: RMSNorm
        let gate: MiniCPMExactBF16Linear
        let up: MiniCPMExactBF16Linear
        let down: MiniCPMExactBF16Linear
    }

    private func array(shape: [Int], phase: Int, divisor: Float) -> MLXArray {
        let values = (0 ..< shape.reduce(1, *)).map { index in
            Float((index * 17 + phase) % 97 - 48) / divisor
        }
        return MLXArray(values, shape).asType(.bfloat16)
    }

    private func layers(hidden: Int, intermediate: Int) -> Layers {
        let divisor = Float(hidden).squareRoot() * 24
        let norm = RMSNorm(dimensions: hidden, eps: 1e-6)
        let normValues = (0 ..< hidden).map { Float(0.75) + Float(($0 * 7) % 37) / 64 }
        norm.update(parameters: ModuleParameters.unflattened([
            "weight": MLXArray(normValues, [hidden]).asType(.bfloat16),
        ]))
        return Layers(
            output: MiniCPMExactBF16Linear(weight: array(
                shape: [hidden, hidden], phase: 11, divisor: divisor)), norm: norm,
            gate: MiniCPMExactBF16Linear(weight: array(
                shape: [intermediate, hidden], phase: 7, divisor: divisor)),
            up: MiniCPMExactBF16Linear(weight: array(
                shape: [intermediate, hidden], phase: 31, divisor: divisor)),
            down: MiniCPMExactBF16Linear(weight: array(
                shape: [hidden, intermediate], phase: 53,
                divisor: Float(intermediate).squareRoot() * 24)))
    }

    private func reference(
        residual: MLXArray, attended: MLXArray, layers: Layers
    ) -> MLXArray {
        let postAttention = residual + layers.output(attended)
        let normalized = layers.norm(postAttention)
        let projected = MiniCPMTTSBatchedBF16Projection.apply(
            normalized, projections: [layers.gate, layers.up], environment: [:])
        let mlp = layers.down(MiniCPMExactBF16SiLU.apply(projected[0]) * projected[1])
        return postAttention + mlp
    }

    private func apply(
        residual: MLXArray, attended: MLXArray, layers: Layers
    ) -> MLXArray? {
        MiniCPMTTSEncodedBF16Block.apply(
            residual: residual, attended: attended, outputProjection: layers.output,
            norm: layers.norm, gate: layers.gate, up: layers.up, down: layers.down,
            epsilon: layers.norm.eps)
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
        let consumed = actual + MLXArray.zeros(actual.shape, dtype: .bfloat16)
        let referenceConsumed = expected + MLXArray.zeros(expected.shape, dtype: .bfloat16)
        XCTAssertEqual(
            consumed.asData(access: .copy).data, referenceConsumed.asData(access: .copy).data,
            file: file, line: line)
    }

    private func requireMetalSupport() throws {
        guard MiniCPMTTSEncodedBF16Block.isAvailable else {
            throw XCTSkip("Encoded BF16 block Metal pipelines are unavailable")
        }
    }

    func testDecodeAndPrefillMatchOriginalIncludingPinnedRMSNorm() throws {
        try requireMetalSupport()
        // Width 127 covers the partial-read branch; 768/3072 is the pinned
        // TTS decoder width and uses 192 threads for each RMSNorm row.
        for (hidden, intermediate) in [(16, 32), (127, 512), (768, 3072)] {
            let layers = layers(hidden: hidden, intermediate: intermediate)
            for tokens in [1, 3] {
                let residual = array(shape: [1, tokens, hidden], phase: 3, divisor: 24)
                let attended = array(shape: [1, tokens, hidden], phase: 29, divisor: 32)
                let expected = reference(residual: residual, attended: attended, layers: layers)
                let actual = try XCTUnwrap(apply(
                    residual: residual, attended: attended, layers: layers))
                assertBitwiseEqual(actual, expected)
            }
        }
    }

    func testNoncontiguousResidualAndAttentionMatchOriginalAtDecoderWidth() throws {
        try requireMetalSupport()
        let layers = layers(hidden: 768, intermediate: 3072)
        let residual = array(shape: [2, 3, 768], phase: 11, divisor: 24).transposed(1, 0, 2)
        let attended = array(shape: [2, 3, 768], phase: 43, divisor: 32).transposed(1, 0, 2)
        eval(residual, attended)
        XCTAssertNotEqual(residual.asData(access: .noCopy).strides, [1536, 768, 1])
        XCTAssertNotEqual(attended.asData(access: .noCopy).strides, [1536, 768, 1])
        let expected = reference(residual: residual, attended: attended, layers: layers)
        let actual = try XCTUnwrap(apply(residual: residual, attended: attended, layers: layers))
        assertBitwiseEqual(actual, expected)
    }

    func testModuleUpdatesRefreshAllWeightFeedsAndKeepReturnedOutputAlive() throws {
        try requireMetalSupport()
        let layers = layers(hidden: 64, intermediate: 256)
        let residual = array(shape: [1, 2, 64], phase: 13, divisor: 24)
        let attended = array(shape: [1, 2, 64], phase: 41, divisor: 32)
        let original = try XCTUnwrap(apply(
            residual: residual, attended: attended, layers: layers))
        let originalBytes = original.asData(access: .copy).data
        try layers.output.update(parameters: ModuleParameters.unflattened([
            "weight": array(shape: [64, 64], phase: 37, divisor: 192),
        ]), verify: .all)
        try layers.norm.update(parameters: ModuleParameters.unflattened([
            "weight": MLXArray((0 ..< 64).map { Float(0.5) + Float($0 % 19) / 32 }, [64])
                .asType(.bfloat16),
        ]), verify: .all)
        try layers.gate.update(parameters: ModuleParameters.unflattened([
            "weight": array(shape: [256, 64], phase: 61, divisor: 192),
        ]), verify: .all)
        try layers.up.update(parameters: ModuleParameters.unflattened([
            "weight": array(shape: [256, 64], phase: 83, divisor: 192),
        ]), verify: .all)
        try layers.down.update(parameters: ModuleParameters.unflattened([
            "weight": array(shape: [64, 256], phase: 23, divisor: 384),
        ]), verify: .all)
        let updated = try XCTUnwrap(apply(residual: residual, attended: attended, layers: layers))
        assertBitwiseEqual(updated, reference(
            residual: residual, attended: attended, layers: layers))
        XCTAssertNotEqual(updated.asData(access: .copy).data, originalBytes)
        XCTAssertEqual(original.asData(access: .copy).data, originalBytes)

        let nextResidual = array(shape: [1, 1, 64], phase: 47, divisor: 24)
        let nextAttended = array(shape: [1, 1, 64], phase: 73, divisor: 32)
        let next = try XCTUnwrap(apply(
            residual: nextResidual, attended: nextAttended, layers: layers))
        assertBitwiseEqual(next, reference(
            residual: nextResidual, attended: nextAttended, layers: layers))
    }

    func testUnsupportedNormDtypesShapesAndProjectionImplementationsReturnNil() {
        let layers = layers(hidden: 16, intermediate: 32)
        let input = array(shape: [1, 1, 16], phase: 3, divisor: 24)
        func rejected(
            residual: MLXArray? = nil, attended: MLXArray? = nil,
            output: Linear? = nil, norm: RMSNorm? = nil, epsilon: Float? = nil,
            gate: Linear? = nil, up: Linear? = nil, down: Linear? = nil
        ) -> MLXArray? {
            MiniCPMTTSEncodedBF16Block.apply(
                residual: residual ?? input, attended: attended ?? input,
                outputProjection: output ?? layers.output, norm: norm ?? layers.norm,
                gate: gate ?? layers.gate, up: up ?? layers.up, down: down ?? layers.down,
                epsilon: epsilon ?? layers.norm.eps)
        }
        XCTAssertNil(rejected(residual: input.asType(.float32)))
        XCTAssertNil(rejected(attended: input.asType(.float32)))
        XCTAssertNil(rejected(attended: array(shape: [1, 2, 16], phase: 7, divisor: 24)))
        XCTAssertNil(rejected(residual: MLXArray.zeros([1, 1, 4097], dtype: .bfloat16),
                              attended: MLXArray.zeros([1, 1, 4097], dtype: .bfloat16)))
        XCTAssertNil(rejected(norm: RMSNorm(dimensions: 16, eps: 1e-6)))
        let exactNorm = MiniCPMExactBF16RMSNorm(dimensions: 16, eps: 1e-6)
        exactNorm.update(parameters: ModuleParameters.unflattened([
            "weight": MLXArray.ones([16], dtype: .bfloat16),
        ]))
        XCTAssertNil(rejected(norm: exactNorm))
        XCTAssertNil(rejected(epsilon: -1))
        XCTAssertNil(rejected(epsilon: Float.nan))
        XCTAssertNil(rejected(epsilon: 1e-5))
        XCTAssertNil(rejected(output: Linear(weight: layers.output.weight)))
        XCTAssertNil(rejected(gate: Linear(weight: layers.gate.weight)))
        let biased = MiniCPMExactBF16Linear(
            weight: layers.up.weight, bias: MLXArray.zeros([32], dtype: .bfloat16))
        XCTAssertNil(rejected(up: biased))
        XCTAssertNil(rejected(down: MiniCPMExactBF16Linear(weight: array(
            shape: [16, 48], phase: 3, divisor: 24))))
    }
}
