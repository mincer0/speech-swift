import XCTest
import MLX
import MLXNN
@testable import MiniCPMLLM

/// Model-free parity coverage for the dense MiniCPM BF16 SiLU cast order.
final class MiniCPMExactBF16SiLUTests: XCTestCase {
    func testBF16SiLUUsesFloat32SigmoidAndProductBeforeCast() {
        let input = MLXArray([
            Float(-8), -4, -2, -1, -0.5, -0.25, 0,
            0.25, 0.5, 1, 2, 4, 8,
        ], [1, 1, 13]).asType(.bfloat16)
        let output = MiniCPMExactBF16SiLU.apply(input)
        let inputFloat = input.asType(.float32)
        let expected = (inputFloat * sigmoid(inputFloat)).asType(.bfloat16)
        eval(output, expected)

        XCTAssertEqual(output.dtype, .bfloat16)
        XCTAssertEqual(
            output.asType(.float32).asArray(Float.self),
            expected.asType(.float32).asArray(Float.self))
        XCTAssertEqual(
            output.asType(.float32).asArray(Float.self),
            [
                -0.002685546875, -0.07177734375, -0.23828125,
                -0.26953125, -0.1884765625, -0.109375, 0,
                0.140625, 0.310546875, 0.73046875, 1.7578125,
                3.921875, 8,
            ])

        // Keep at least one input in the fixture that distinguishes this
        // cast order from MLX's fused BF16 implementation.
        let fused = silu(input)
        eval(fused)
        XCTAssertNotEqual(
            fused.asType(.float32).asArray(Float.self),
            output.asType(.float32).asArray(Float.self))
    }

    func testNonBF16InputKeepsItsDtype() {
        let input = MLXArray([Float(-1), 0, 1])
        let output = MiniCPMExactBF16SiLU.apply(input)
        eval(output)
        XCTAssertEqual(output.dtype, .float32)
        let values = output.asArray(Float.self)
        XCTAssertEqual(values.count, 3)
        XCTAssertEqual(values[0], -0.26894143, accuracy: 1e-5)
        XCTAssertEqual(values[1], 0, accuracy: 1e-5)
        XCTAssertEqual(values[2], 0.7310586, accuracy: 1e-5)
    }
}
