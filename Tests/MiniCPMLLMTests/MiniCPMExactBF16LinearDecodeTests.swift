import Foundation
import Metal
import XCTest
import MLX
@testable import MiniCPMLLM

/// Diagnostic-only comparison of PyTorch MPS BF16 Linear at T=1/T>1 against
/// the existing MPSGraph GEMM bridge.  This deliberately uses a tiny weight
/// slice (8 x 4096), not a model checkpoint, and does not alter production
/// routing.
final class MiniCPMExactBF16LinearDecodeTests: XCTestCase {
    private struct Metrics {
        let maxAbs: Double
        let meanAbs: Double
        let neq: Int
    }

    func testPyTorchMPSLinearT1AndPrefillDiagnostics() throws {
        guard MTLCreateSystemDefaultDevice() != nil else {
            throw XCTSkip("Metal GPU device is unavailable")
        }
        let directory = try fixtureDirectory()
        let weightShape = [8, 4096]
        let inputT1Shape = [1, 4096]
        let inputT2Shape = [2, 4096]
        let outputT1Shape = [1, 8]
        let outputT2Shape = [2, 8]
        let weight = try readFloat32(directory.appendingPathComponent("weight.f32"), count: weightShape)
            .asBF16(shape: weightShape)
        let inputT1 = try readFloat32(directory.appendingPathComponent("input_t1.f32"), count: inputT1Shape)
            .asBF16(shape: inputT1Shape)
        let inputT2 = try readFloat32(directory.appendingPathComponent("input_t2.f32"), count: inputT2Shape)
            .asBF16(shape: inputT2Shape)
        let referenceT1 = try readFloat32(
            directory.appendingPathComponent("output_t1.f32"), count: outputT1Shape)
        let referenceT2 = try readFloat32(
            directory.appendingPathComponent("output_t2.f32"), count: outputT2Shape)

        let linear = MiniCPMExactBF16Linear(weight: weight, bias: nil)
        let actualT1 = linear(inputT1)
        let actualT2 = linear(inputT2)
        eval(actualT1, actualT2)
        XCTAssertEqual(actualT1.shape, outputT1Shape)
        XCTAssertEqual(actualT2.shape, outputT2Shape)
        XCTAssertEqual(actualT1.dtype, .bfloat16)
        XCTAssertEqual(actualT2.dtype, .bfloat16)

        let valuesT1 = actualT1.asType(.float32).asArray(Float.self)
        let valuesT2 = actualT2.asType(.float32).asArray(Float.self)
        let metricsT1 = metrics(valuesT1, against: referenceT1)
        let metricsT2 = metrics(valuesT2, against: referenceT2)
        report("MPSGraph GEMM T1", metricsT1)
        report("MPSGraph GEMM T2", metricsT2)

        // This test is diagnostic rather than a tolerance gate: retaining the
        // exact vectors lets a future implementation compare any decode-only
        // replacement against the same PyTorch MPS bytes.
        XCTAssertTrue(valuesT1.allSatisfy(\.isFinite))
        XCTAssertTrue(valuesT2.allSatisfy(\.isFinite))
    }

    private func fixtureDirectory() throws -> URL {
        if let bundled = Bundle.module.url(
            forResource: "ExactBF16LinearDecode",
            withExtension: nil,
            subdirectory: "Fixtures") {
            return bundled
        }
        let source = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .appendingPathComponent("Fixtures/ExactBF16LinearDecode", isDirectory: true)
        guard FileManager.default.fileExists(atPath: source.path) else {
            throw XCTSkip("linear decode fixture is unavailable: \(source.path)")
        }
        return source
    }

    private func readFloat32(_ url: URL, count shape: [Int]) throws -> [Float] {
        let data: Data
        do {
            data = try Data(contentsOf: url)
        } catch {
            throw XCTSkip("missing linear decode fixture tensor \(url.path): \(error)")
        }
        let expectedBytes = shape.reduce(1, *) * MemoryLayout<Float>.stride
        guard data.count == expectedBytes else {
            throw XCTSkip(
                "fixture tensor \(url.lastPathComponent) has \(data.count) bytes; expected \(expectedBytes)")
        }
        return data.withUnsafeBytes { raw in
            Array(raw.bindMemory(to: Float.self))
        }
    }

    private func metrics(_ actual: [Float], against expected: [Float]) -> Metrics {
        guard actual.count == expected.count, !actual.isEmpty else {
            return Metrics(maxAbs: .infinity, meanAbs: .infinity, neq: -1)
        }
        var maxAbs = 0.0
        var sumAbs = 0.0
        var neq = 0
        for (lhs, rhs) in zip(actual, expected) {
            let delta = abs(Double(lhs) - Double(rhs))
            maxAbs = max(maxAbs, delta)
            sumAbs += delta
            if lhs.bitPattern != rhs.bitPattern { neq += 1 }
        }
        return Metrics(maxAbs: maxAbs, meanAbs: sumAbs / Double(actual.count), neq: neq)
    }

    private func report(_ label: String, _ metrics: Metrics) {
        print(
            "MINICPM_LINEAR_DECODE \(label) max_abs=\(metrics.maxAbs) "
                + "mean_abs=\(metrics.meanAbs) neq=\(metrics.neq)")
    }
}

private extension Array where Element == Float {
    func asBF16(shape: [Int]) -> MLXArray {
        MLXArray(self, shape).asType(.bfloat16)
    }
}
