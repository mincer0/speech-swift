import XCTest
import Metal
import MLX
import MLXNN
import MiniCPMLLM

@testable import MiniCPMTTSSemantic

final class MiniCPMTTSBatchedProjectionTests: XCTestCase {
    private func weight(rows: Int, columns: Int, phase: Float, dtype: DType) -> MLXArray {
        MLXArray((0 ..< rows * columns).map {
            Float(($0 * 17) % 43 - 21) / 13 + phase
        }, [rows, columns]).asType(dtype)
    }

    private func assertIdentical(_ actual: [MLXArray], _ expected: [MLXArray]) {
        XCTAssertEqual(actual.count, expected.count)
        for (lhs, rhs) in zip(actual, expected) {
            XCTAssertEqual(lhs.shape, rhs.shape)
            XCTAssertEqual(lhs.dtype, rhs.dtype)
            XCTAssertEqual(lhs.asType(.float32).asArray(Float.self),
                           rhs.asType(.float32).asArray(Float.self))
        }
    }

    func testThreeBF16ProjectionsMatchIndependentExactMatmuls() {
        let projections: [Linear] = [32, 16, 16].enumerated().map { index, rows in
            MiniCPMExactBF16Linear(weight: weight(
                rows: rows, columns: 16, phase: Float(index) / 7, dtype: .bfloat16))
        }
        for tokens in [1, 3] {
            let input = weight(rows: tokens, columns: 16, phase: 0.2, dtype: .bfloat16)
                .expandedDimensions(axis: 0)
            XCTAssertTrue(MiniCPMTTSBatchedBF16Projection.canBatch(
                input: input, projections: projections, environment: [:]))
            let expected = projections.map { $0(input) }
            let actual = MiniCPMTTSBatchedBF16Projection.apply(
                input, projections: projections, environment: [:])
            assertIdentical(actual, expected)
        }
    }

    func testTwoBF16ProjectionsMaterializeNoncontiguousInput() {
        let projections: [Linear] = (0 ..< 2).map {
            MiniCPMExactBF16Linear(weight: weight(
                rows: 32, columns: 16, phase: Float($0) / 5, dtype: .bfloat16))
        }
        let input = weight(rows: 6, columns: 16, phase: 0, dtype: .bfloat16)
            .reshaped(2, 3, 16).transposed(1, 0, 2)
        let expected = projections.map { $0(input) }
        let actual = MiniCPMTTSBatchedBF16Projection.apply(
            input, projections: projections, environment: [:])
        assertIdentical(actual, expected)
    }

    func testFloat32UsesOriginalLinearFallback() {
        let projections: [Linear] = (0 ..< 2).map {
            MiniCPMExactBF16Linear(weight: weight(
                rows: 8, columns: 16, phase: Float($0) / 5, dtype: .float32))
        }
        let input = weight(rows: 3, columns: 16, phase: 0, dtype: .float32)
        XCTAssertFalse(MiniCPMTTSBatchedBF16Projection.canBatch(
            input: input, projections: projections, environment: [:]))
        assertIdentical(MiniCPMTTSBatchedBF16Projection.apply(
            input, projections: projections, environment: [:]), projections.map { $0(input) })
    }

    func testDisabledBatchingUsesIndependentExactMatmuls() {
        let projections: [Linear] = (0 ..< 2).map {
            MiniCPMExactBF16Linear(weight: weight(
                rows: 8, columns: 16, phase: Float($0), dtype: .bfloat16))
        }
        let input = weight(rows: 1, columns: 16, phase: 0, dtype: .bfloat16)
        let disabled = ["MINICPM_TTS_BATCHED_PROJECTIONS": "0"]
        XCTAssertFalse(MiniCPMTTSBatchedBF16Projection.canBatch(
            input: input, projections: projections, environment: disabled))
        assertIdentical(MiniCPMTTSBatchedBF16Projection.apply(
            input, projections: projections, environment: disabled), projections.map { $0(input) })
    }

    func testMixedLinearImplementationsRetainOriginalPath() {
        let projections: [Linear] = [
            MiniCPMExactBF16Linear(weight: weight(
                rows: 8, columns: 16, phase: 0, dtype: .bfloat16)),
            Linear(weight: weight(rows: 8, columns: 16, phase: 1, dtype: .bfloat16)),
        ]
        let input = weight(rows: 1, columns: 16, phase: 0, dtype: .bfloat16)
        XCTAssertFalse(MiniCPMTTSBatchedBF16Projection.canBatch(
            input: input, projections: projections, environment: [:]))
        assertIdentical(MiniCPMTTSBatchedBF16Projection.apply(
            input, projections: projections, environment: [:]), projections.map { $0(input) })
    }

    func testGroupedAndIndependentCallsSharePreparedWeightsAfterUpdate() throws {
        let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
        let projections = (0 ..< 2).map {
            MiniCPMExactBF16Linear(weight: weight(
                rows: 8, columns: 16, phase: Float($0) / 5, dtype: .bfloat16))
        }
        let input = weight(rows: 1, columns: 16, phase: 0.2, dtype: .bfloat16)
        let before = projections.map { $0.preparedBF16Weight(device: device) }
        let original = MiniCPMTTSBatchedBF16Projection.apply(
            input, projections: projections, environment: [:])
        assertIdentical(original, projections.map { $0(input) })
        for index in projections.indices {
            XCTAssertTrue(before[index] === projections[index].preparedBF16Weight(device: device))
        }

        let replacement = weight(rows: 8, columns: 16, phase: 2, dtype: .bfloat16)
        try projections[0].update(
            parameters: ModuleParameters.unflattened(["weight": replacement]), verify: .all)
        let after = projections.map { $0.preparedBF16Weight(device: device) }
        XCTAssertFalse(before[0] === after[0])
        XCTAssertTrue(before[1] === after[1])
        let updated = MiniCPMTTSBatchedBF16Projection.apply(
            input, projections: projections, environment: [:])
        assertIdentical(updated, projections.map { $0(input) })
        XCTAssertNotEqual(
            updated[0].asType(.float32).asArray(Float.self),
            original[0].asType(.float32).asArray(Float.self))
        assertIdentical([updated[1]], [original[1]])
        for index in projections.indices {
            XCTAssertTrue(after[index] === projections[index].preparedBF16Weight(device: device))
        }
    }
}
