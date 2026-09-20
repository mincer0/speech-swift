import Foundation
import MLX
import XCTest
@testable import MiniCPMToken2Wav

/// Real-weight HiFT parity.  Every major stage is compared with the official
/// PyTorch oracle; checking only the final waveform would hide transposed
/// convolution, source-STFT, or phase/noise mismatches.
final class MiniCPMHiFTGoldenTests: XCTestCase {
    private func loadFixture() throws -> MiniCPMHiFTGoldenFixture {
        guard let fixture = try MiniCPMHiFTGoldenFixture.locate() else {
            throw XCTSkip(
                "set MINICPM_HIFT_GOLDEN_DIR to run MiniCPM HiFT golden tests")
        }
        try fixture.validateAll()
        return fixture
    }

    private func loadModel() throws -> MiniCPMHiFTGenerator {
        guard let value = ProcessInfo.processInfo.environment[
            "MINICPM_HIFT_MLX_PATH"], !value.isEmpty else {
            throw XCTSkip("set MINICPM_HIFT_MLX_PATH to the converted HiFT bundle")
        }
        let url = URL(fileURLWithPath: value, isDirectory: true)
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(
            atPath: url.path, isDirectory: &isDirectory), isDirectory.boolValue else {
            throw MiniCPMHiFTGoldenFixture.FixtureError.missing(
                "valid model directory in MINICPM_HIFT_MLX_PATH=\(value)")
        }
        let model = try MiniCPMHiFTGenerator.fromDirectory(url)
        model.train(false)
        return model
    }

    private func assertStage(
        _ actual: MLXArray,
        expectedName: String,
        fixture: MiniCPMHiFTGoldenFixture,
        label: String,
        maxError: Float = 0.08,
        meanError: Float = 0.015,
        cosineFloor: Float = 0.999
    ) throws {
        let expected = try fixture.tensor(expectedName)
        XCTAssertEqual(actual.shape, expected.shape, "\(label) shape")
        guard actual.shape == expected.shape else { return }
        let expectedDType = (try fixture.tensorSpec(expectedName))["dtype"] as? String
        if expectedDType == "float32" {
            XCTAssertEqual(actual.dtype, .float32, "\(label) dtype")
        }
        eval(actual, expected)
        let lhs = actual.asType(.float32)
        let rhs = expected.asType(.float32)
        let lhsValues = lhs.asArray(Float.self)
        let rhsValues = rhs.asArray(Float.self)
        XCTAssertTrue(lhsValues.allSatisfy(\.isFinite), "\(label) non-finite")
        XCTAssertTrue(rhsValues.allSatisfy(\.isFinite), "\(label) golden non-finite")
        let difference = MLX.abs(lhs - rhs)
        eval(difference)
        XCTAssertLessThanOrEqual(
            difference.max().item(Float.self), maxError, "\(label) max error")
        XCTAssertLessThanOrEqual(
            difference.mean().item(Float.self), meanError, "\(label) mean error")
        var dot: Double = 0
        var leftNorm: Double = 0
        var rightNorm: Double = 0
        for (left, right) in zip(lhsValues, rhsValues) {
            let x = Double(left)
            let y = Double(right)
            dot += x * y
            leftNorm += x * x
            rightNorm += y * y
        }
        let cosine = leftNorm > 0 && rightNorm > 0
            ? Float(dot / (sqrt(leftNorm) * sqrt(rightNorm))) : (dot == 0 ? 1 : 0)
        XCTAssertGreaterThanOrEqual(cosine, cosineFloor, "\(label) cosine")
    }

    func testManifestAndArtifactAreAuditable() throws {
        let fixture = try loadFixture()
        XCTAssertEqual(fixture.manifest["schema"] as? String,
                       MiniCPMHiFTGoldenFixture.schema)
        let runtime = try XCTUnwrap(fixture.manifest["runtime"] as? [String: Any])
        XCTAssertEqual(runtime["output_dtype"] as? String, "float32")
        XCTAssertEqual(runtime["random_algorithm"] as? String,
                       "torch_cpu_default_generator")
    }

    func testAllHiFTStagesAndDeterministicSource() throws {
        let fixture = try loadFixture()
        let model = try loadModel()
        let mel = try fixture.tensor("mel")
        let source = try fixture.tensor("source")
        let output = model.generate(
            mel: mel,
            sourceOverride: source)
        eval(output.waveform, output.f0, output.source)

        try assertStage(
            output.f0, expectedName: "f0", fixture: fixture,
            label: "F0 predictor", maxError: 0.1, meanError: 0.02)
        try assertStage(
            output.source, expectedName: "source", fixture: fixture,
            label: "source override")
        for index in 0 ..< 3 {
            try assertStage(
                output.intermediates["ups.\(index)"]!,
                expectedName: "ups.\(index)", fixture: fixture,
                label: "upsample stage \(index)")
            try assertStage(
                output.intermediates["stages.\(index)"]!,
                expectedName: "stages.\(index)", fixture: fixture,
                label: "residual stage \(index)")
        }
        try assertStage(
            output.intermediates["source_stft"]!, expectedName: "source_stft",
            fixture: fixture, label: "source STFT", maxError: 0.15, meanError: 0.025)
        try assertStage(
            output.intermediates["conv_pre"]!, expectedName: "conv_pre",
            fixture: fixture, label: "conv_pre")
        try assertStage(
            output.intermediates["logits"]!, expectedName: "logits",
            fixture: fixture, label: "ISTFT logits")
        try assertStage(
            output.waveform, expectedName: "waveform", fixture: fixture,
            label: "waveform", maxError: 0.02, meanError: 0.005)

        let generatedSource = model.generateSource(
            f0: try fixture.tensor("f0"),
            standardNoise: try fixture.tensor("source_noise"),
            initialPhase: try fixture.tensor("source_initial_phase"))
        try assertStage(
            generatedSource, expectedName: "source", fixture: fixture,
            label: "deterministic SineGen2 source")
    }
}
