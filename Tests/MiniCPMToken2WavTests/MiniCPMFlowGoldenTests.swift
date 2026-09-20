import Foundation
import MLX
import XCTest
@testable import MiniCPMToken2Wav

/// Real-weight Flow parity.  This is opt-in because loading the converted
/// bundle is expensive, but a configured fixture is never silently skipped.
final class MiniCPMFlowGoldenTests: XCTestCase {
    private func loadFixture() throws -> MiniCPMFlowGoldenFixture {
        guard let fixture = try MiniCPMFlowGoldenFixture.locate() else {
            throw XCTSkip(
                "set MINICPM_FLOW_GOLDEN_DIR to run MiniCPM Flow golden tests")
        }
        try fixture.validateAll()
        return fixture
    }

    private func loadModel() throws -> MiniCPMToken2WavFlowModel {
        guard let value = ProcessInfo.processInfo.environment[
            "MINICPM_FLOW_MLX_PATH"], !value.isEmpty else {
            throw XCTSkip("set MINICPM_FLOW_MLX_PATH to the converted Flow bundle")
        }
        let url = URL(fileURLWithPath: value, isDirectory: true)
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(
            atPath: url.path, isDirectory: &isDirectory), isDirectory.boolValue else {
            throw MiniCPMFlowGoldenFixture.FixtureError.missing(
                "valid model directory in MINICPM_FLOW_MLX_PATH=\(value)")
        }
        let model = try MiniCPMToken2WavFlowModel.fromDirectory(url)
        model.train(false)
        return model
    }

    private func cosine(_ lhs: [Float], _ rhs: [Float]) -> Float {
        guard lhs.count == rhs.count, !lhs.isEmpty else { return 0 }
        var dot: Double = 0
        var leftNorm: Double = 0
        var rightNorm: Double = 0
        for (left, right) in zip(lhs, rhs) {
            let x = Double(left)
            let y = Double(right)
            dot += x * y
            leftNorm += x * x
            rightNorm += y * y
        }
        guard leftNorm > 0, rightNorm > 0 else { return dot == 0 ? 1 : 0 }
        return Float(dot / (sqrt(leftNorm) * sqrt(rightNorm)))
    }

    private func assertStage(
        _ actual: MLXArray,
        expectedName: String,
        fixture: MiniCPMFlowGoldenFixture,
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
        XCTAssertGreaterThanOrEqual(
            cosine(lhsValues, rhsValues), cosineFloor, "\(label) cosine")
    }

    func testManifestAndArtifactAreAuditable() throws {
        let fixture = try loadFixture()
        XCTAssertEqual(fixture.manifest["schema"] as? String,
                       MiniCPMFlowGoldenFixture.schema)
        let runtime = try XCTUnwrap(fixture.manifest["runtime"] as? [String: Any])
        XCTAssertEqual(runtime["output_dtype"] as? String, "float32")
        XCTAssertEqual(runtime["token_dtype"] as? String, "int32")
        XCTAssertEqual(runtime["mask_dtype"] as? String, "float32")
        XCTAssertEqual(runtime["random_algorithm"] as? String,
                       "torch_cpu_default_generator")
    }

    func testOfflineEmbeddingEncoderAndDiTStages() throws {
        let fixture = try loadFixture()
        let model = try loadModel()
        let tokens = try fixture.tensor("tokens")
        let embedding = model.inputEmbedding(tokens)
        try assertStage(
            embedding, expectedName: "token_embeddings", fixture: fixture,
            label: "input embedding")

        let encoded = model.encoder(embedding)
        try assertStage(
            encoded, expectedName: "encoder_output", fixture: fixture,
            label: "Conformer encoder")

        let dit = model.decoder.estimator(
            x: try fixture.tensor("dit_x"),
            mu: try fixture.tensor("dit_mu"),
            time: try fixture.tensor("dit_time"),
            speakers: try fixture.tensor("dit_speakers"),
            condition: try fixture.tensor("dit_condition"),
            validMask: try fixture.tensor("dit_mask"))
        try assertStage(
            dit.output, expectedName: "dit_output", fixture: fixture,
            label: "DiT estimator")
    }

    func testConditionalFlowMatchingStageUsesCapturedNoise() throws {
        let fixture = try loadFixture()
        let model = try loadModel()
        model.decoder.fixedNoise = try fixture.tensor("cfm_noise")
        let output = model.decoder.generate(
            mu: try fixture.tensor("cfm_mu"),
            speakers: try fixture.tensor("cfm_speaker"),
            condition: try fixture.tensor("cfm_condition"),
            validMask: try fixture.tensor("cfm_mask"),
            odeSteps: 2)
        try assertStage(
            output.output, expectedName: "cfm_output_2_steps", fixture: fixture,
            label: "two-step CFM")
    }

    func testStreamingEncoderAndDiTCacheStages() throws {
        let fixture = try loadFixture()
        let model = try loadModel()
        let first = model.encoder.streaming(
            model.inputEmbedding(try fixture.tensor("stream_tokens_1")))
        let second = model.encoder.streaming(
            model.inputEmbedding(try fixture.tensor("stream_tokens_2")),
            cache: first.cache)
        try assertStage(
            first.output, expectedName: "stream_encoder_1", fixture: fixture,
            label: "stream encoder 1")
        try assertStage(
            second.output, expectedName: "stream_encoder_2", fixture: fixture,
            label: "stream encoder 2")
        try assertStage(
            first.cache.firstAttention[0].keys,
            expectedName: "stream_first_k_1", fixture: fixture,
            label: "stream first attention keys 1")
        try assertStage(
            first.cache.firstAttention[0].values,
            expectedName: "stream_first_v_1", fixture: fixture,
            label: "stream first attention values 1")
        try assertStage(
            first.cache.secondAttention[0].keys,
            expectedName: "stream_second_k_1", fixture: fixture,
            label: "stream second attention keys 1")
        try assertStage(
            first.cache.secondAttention[0].values,
            expectedName: "stream_second_v_1", fixture: fixture,
            label: "stream second attention values 1")
        try assertStage(
            second.cache.firstAttention[0].keys,
            expectedName: "stream_first_k_2", fixture: fixture,
            label: "stream first attention keys 2")
        try assertStage(
            second.cache.firstAttention[0].values,
            expectedName: "stream_first_v_2", fixture: fixture,
            label: "stream first attention values 2")
        try assertStage(
            second.cache.secondAttention[0].keys,
            expectedName: "stream_second_k_2", fixture: fixture,
            label: "stream second attention keys 2")
        try assertStage(
            second.cache.secondAttention[0].values,
            expectedName: "stream_second_v_2", fixture: fixture,
            label: "stream second attention values 2")

        let firstDiT = model.decoder.estimator(
            x: try fixture.tensor("dit_stream_x_1"),
            mu: try fixture.tensor("dit_stream_mu_1"),
            time: try fixture.tensor("dit_stream_time"),
            speakers: try fixture.tensor("dit_stream_speakers"),
            condition: try fixture.tensor("dit_stream_condition_1"))
        let secondDiT = model.decoder.estimator(
            x: try fixture.tensor("dit_stream_x_2"),
            mu: try fixture.tensor("dit_stream_mu_2"),
            time: try fixture.tensor("dit_stream_time"),
            speakers: try fixture.tensor("dit_stream_speakers"),
            condition: try fixture.tensor("dit_stream_condition_2"),
            cache: firstDiT.cache)
        try assertStage(
            firstDiT.output, expectedName: "dit_stream_output_1", fixture: fixture,
            label: "stream DiT output 1")
        try assertStage(
            secondDiT.output, expectedName: "dit_stream_output_2", fixture: fixture,
            label: "stream DiT output 2")
        try assertStage(
            secondDiT.cache.layers[0].convolution.first,
            expectedName: "dit_stream_cnn_first_2", fixture: fixture,
            label: "stream DiT convolution first")
        try assertStage(
            secondDiT.cache.layers[0].convolution.second,
            expectedName: "dit_stream_cnn_second_2", fixture: fixture,
            label: "stream DiT convolution second")
    }
}
