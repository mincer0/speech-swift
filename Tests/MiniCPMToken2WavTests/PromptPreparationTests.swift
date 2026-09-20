import MLX
@testable import MiniCPMToken2Wav
import XCTest

private struct FixturePromptEncoder: MiniCPMToken2WavPromptEncoder {
    let tokenCount: Int

    func encodeSpeechTokens(tokenizerMel: MLXArray) throws -> MLXArray {
        XCTAssertEqual(tokenizerMel.ndim, 3)
        XCTAssertEqual(tokenizerMel.dim(1), 128)
        XCTAssertGreaterThan(tokenizerMel.dim(2), 0)
        return MLXArray((0..<tokenCount).map { Int32(100 + $0) })
            .expandedDimensions(axis: 0)
    }

    func encodeSpeakerEmbedding(speakerMel: MLXArray) throws -> MLXArray {
        XCTAssertEqual(speakerMel.shape, [1, 500, 80])
        return MLXArray((0..<192).map { Float($0) / 192 }).expandedDimensions(axis: 0)
    }
}

final class MiniCPMToken2WavPromptPreparationTests: XCTestCase {
    func testCamPlusPlusMelMatchesKaldiFbankShapeAndPrefix() throws {
        let pcm = (0..<16_000).map {
            Float(sin(Double($0) * 0.017)) * 0.2
        }
        let mel = MiniCPMCamPlusPlusMelExtractor().extract(pcm)
        XCTAssertEqual(mel.shape, [1, 500, 80])
        let values = mel.asArray(Float.self)
        // The first 98 rows are the snip-edge Kaldi fbank rows; subsequent
        // rows tile the short reference to satisfy CAM++'s fixed 500-frame
        // CoreML shape.  These values come from torchaudio 2.8's
        // compliance.kaldi.fbank followed by per-bin CMN.
        let expected: [Float] = [
            -0.03937161, -0.00545299, 0.00453401, 0.00490761,
            0.03364182, 0.03115940, 0.02146626, 0.06922054,
            0.18071938, 0.22106552, 0.02828789, 0.08611870,
            0.21548748, -0.06551743, -0.00000095, -0.00866127,
        ]
        for (actual, reference) in zip(values, expected) {
            XCTAssertEqual(actual, reference, accuracy: 2e-3)
        }
    }

    func testCamPlusPlusMelReportsShortClipTiling() throws {
        let extractor = MiniCPMCamPlusPlusMelExtractor()
        let result = extractor.extractWithDiagnostics(
            [Float](repeating: 0, count: 16_000), targetFrames: 500)
        XCTAssertEqual(result.mel.shape, [1, 500, 80])
        XCTAssertEqual(result.rawFrameCount, 98)
        XCTAssertEqual(result.adaptation, .tiledShort)
    }

    func testCamPlusPlusMelReportsLongClipCenterCrop() throws {
        let extractor = MiniCPMCamPlusPlusMelExtractor()
        // 598 snip-edge frames at 16 kHz: 1 + (96_000 - 400) / 160.
        let result = extractor.extractWithDiagnostics(
            [Float](repeating: 0, count: 96_000), targetFrames: 500)
        XCTAssertEqual(result.mel.shape, [1, 500, 80])
        XCTAssertEqual(result.rawFrameCount, 598)
        XCTAssertEqual(result.adaptation, .centerCroppedLong)
    }

    func testCamPlusPlusMelReportsExactFrameCount() throws {
        let extractor = MiniCPMCamPlusPlusMelExtractor()
        // Exactly 500 snip-edge frames, with no tiling or crop.
        let result = extractor.extractWithDiagnostics(
            [Float](repeating: 0, count: 80_240), targetFrames: 500)
        XCTAssertEqual(result.rawFrameCount, 500)
        XCTAssertEqual(result.adaptation, .exact)
    }

    func testNativeMelsHaveExpectedRatesAndAlignment() throws {
        let sampleCount = 16_000
        let pcm = (0..<sampleCount).map { i in
            Float(sin(Double(i) * 2 * Double.pi * 220 / 16_000)) * 0.1
        }
        let preparer = MiniCPMToken2WavPromptPreparer()
        let profile = try preparer.prepare(
            pcm: pcm,
            sampleRate: 16_000,
            encoder: FixturePromptEncoder(tokenCount: 25))

        XCTAssertEqual(profile.prompt.tokens.shape, [1, 25])
        XCTAssertEqual(profile.prompt.mel.shape, [1, 80, 50])
        XCTAssertEqual(profile.prompt.speakerEmbedding.shape, [1, 192])
        XCTAssertEqual(profile.promptMelFrameCount, 50)
        XCTAssertEqual(profile.tokenizerMelFrameCount, 100)
        XCTAssertEqual(profile.speakerMelFrameCount, 500)
        XCTAssertEqual(profile.speakerRawMelFrameCount, 98)
        XCTAssertEqual(profile.speakerAdaptation, .tiledShort)
        XCTAssertTrue(profile.prompt.mel.asArray(Float.self).allSatisfy(\.isFinite))
    }

    func testPromptPreparationResamplesNon16kReferenceDeterministically() throws {
        // 8 kHz input is converted to each MiniCPM frontend rate by the same
        // torchaudio-compatible sinc path.  The 1-second duration should
        // remain 98 raw CAM++ frames after conversion to 16 kHz.
        let pcm = (0..<8_000).map {
            Float(sin(Double($0) * 2 * Double.pi * 220 / 8_000)) * 0.1
        }
        let profile = try MiniCPMToken2WavPromptPreparer().prepare(
            pcm: pcm,
            sampleRate: 8_000,
            encoder: FixturePromptEncoder(tokenCount: 25))
        XCTAssertEqual(profile.tokenizerMelFrameCount, 100)
        XCTAssertEqual(profile.speakerRawMelFrameCount, 98)
        XCTAssertEqual(profile.speakerAdaptation, .tiledShort)
    }

    func testPromptTokensAndEmbeddingCanBeInjectedWithoutModel() throws {
        let pcm = [Float](repeating: 0, count: 8_000)
        let tokens = MLXArray([Int32(1), 2, 3, 4, 5]).expandedDimensions(axis: 0)
        let speaker = MLXArray([Float](repeating: 0.25, count: 192)).expandedDimensions(axis: 0)
        let profile = try MiniCPMToken2WavPromptPreparer().prepare(
            pcm: pcm,
            sampleRate: 16_000,
            promptTokens: tokens,
            speakerEmbedding: speaker)

        XCTAssertEqual(profile.prompt.tokens.shape, [1, 5])
        XCTAssertEqual(profile.prompt.mel.shape, [1, 80, 10])
        XCTAssertEqual(profile.prompt.speakerEmbedding.shape, [1, 192])
    }

    func testDynamicVoiceAndFixedPromptProduceEquivalentArrays() throws {
        // A dynamic reference voice first runs the native encoders, while a
        // fixed prompt reuses their three resulting arrays.  Both paths must
        // feed exactly the same Token2Wav prompt; otherwise changing voice at
        // runtime can silently alter the flow conditioning or alignment.
        let pcm = (0 ..< 16_000).map {
            Float(sin(Double($0) * 0.013)) * 0.15
        }
        let preparer = MiniCPMToken2WavPromptPreparer()
        let dynamic = try preparer.prepare(
            pcm: pcm,
            sampleRate: 16_000,
            encoder: FixturePromptEncoder(tokenCount: 21))
        let fixed = try preparer.prepare(
            pcm: pcm,
            sampleRate: 16_000,
            promptTokens: dynamic.prompt.tokens,
            speakerEmbedding: dynamic.prompt.speakerEmbedding)

        eval(
            dynamic.prompt.tokens,
            dynamic.prompt.mel,
            dynamic.prompt.speakerEmbedding,
            fixed.prompt.tokens,
            fixed.prompt.mel,
            fixed.prompt.speakerEmbedding)
        XCTAssertEqual(
            dynamic.prompt.tokens.asType(.int32).asArray(Int32.self),
            fixed.prompt.tokens.asType(.int32).asArray(Int32.self))
        XCTAssertEqual(
            dynamic.prompt.mel.asType(.float32).asArray(Float.self),
            fixed.prompt.mel.asType(.float32).asArray(Float.self))
        XCTAssertEqual(
            dynamic.prompt.speakerEmbedding.asType(.float32).asArray(Float.self),
            fixed.prompt.speakerEmbedding.asType(.float32).asArray(Float.self))
        XCTAssertEqual(dynamic.promptMelFrameCount, fixed.promptMelFrameCount)
        XCTAssertEqual(dynamic.speakerMelFrameCount, fixed.speakerMelFrameCount)
    }

    func testInvalidPCMIsRejectedBeforeFeatureExtraction() {
        let preparer = MiniCPMToken2WavPromptPreparer()
        XCTAssertThrowsError(try preparer.prepare(
            pcm: [0, .nan], sampleRate: 16_000,
            promptTokens: MLXArray([Int32(1)]).expandedDimensions(axis: 0),
            speakerEmbedding: MLXArray([Float](repeating: 0, count: 192)).expandedDimensions(axis: 0))) { error in
            guard case MiniCPMToken2WavPromptPreparationError.invalidPCM = error else {
                return XCTFail("unexpected error: \(error)")
            }
        }
    }

    func testMissingModelOutputsAreExplicit() {
        let preparer = MiniCPMToken2WavPromptPreparer()
        XCTAssertThrowsError(try preparer.prepare(
            pcm: [Float](repeating: 0, count: 4_000), sampleRate: 16_000)) { error in
            guard case MiniCPMToken2WavPromptPreparationError.missingSpeechTokenizer = error else {
                return XCTFail("unexpected error: \(error)")
            }
        }
    }

    func testNativePromptDiscoveryDoesNotSilentlyUseONNX() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(
            at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let assets = root.appendingPathComponent("assets/token2wav", isDirectory: true)
        try FileManager.default.createDirectory(
            at: assets, withIntermediateDirectories: true)
        try Data().write(
            to: assets.appendingPathComponent("speech_tokenizer_v2_25hz.onnx"))
        try Data().write(to: assets.appendingPathComponent("campplus.onnx"))

        XCTAssertThrowsError(try MiniCPMNativePromptEncoder.discover(in: root)) { error in
            guard case MiniCPMToken2WavPromptPreparationError.missingModelAsset = error else {
                return XCTFail("unexpected discovery error: \(error)")
            }
        }
    }

    func testNativePromptDiscoveryInterpolatesMissingCamPlusPlusPath() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(
            at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        // A placeholder safetensors file is enough to move discovery to the
        // CAM++ branch; the encoder is not instantiated after the failure.
        try Data().write(to: root.appendingPathComponent("speech_tokenizer.safetensors"))
        let assets = root.appendingPathComponent("assets/token2wav", isDirectory: true)
        try FileManager.default.createDirectory(
            at: assets, withIntermediateDirectories: true)
        let onnx = assets.appendingPathComponent("campplus.onnx")
        try Data().write(to: onnx)

        XCTAssertThrowsError(try MiniCPMNativePromptEncoder.discover(in: root)) { error in
            guard case MiniCPMToken2WavPromptPreparationError.missingModelAsset = error else {
                return XCTFail("unexpected discovery error: \(error)")
            }
            let description = error.localizedDescription
            XCTAssertTrue(description.contains(onnx.path))
            XCTAssertTrue(description.contains(root.path))
            XCTAssertFalse(description.contains("(onnx.path)"))
            XCTAssertFalse(description.contains("(modelDirectory.path)"))
        }
    }

    func testNativePromptAssetDiscoveryAcceptsCompiledCamDirectory() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(
            at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        try Data().write(to: root.appendingPathComponent("speech_tokenizer.safetensors"))
        let cam = root.appendingPathComponent("MiniCPM-CamPlusPlus.mlmodelc", isDirectory: true)
        try FileManager.default.createDirectory(at: cam, withIntermediateDirectories: true)

        let assets = try MiniCPMNativePromptEncoder.discoverAssets(in: root)
        XCTAssertEqual(assets.speechTokenizerWeights.path,
                       root.appendingPathComponent("speech_tokenizer.safetensors").path)
        XCTAssertEqual(assets.camPlusPlusModel.path, cam.path)
        var isDirectory: ObjCBool = false
        XCTAssertTrue(FileManager.default.fileExists(
            atPath: assets.camPlusPlusModel.path, isDirectory: &isDirectory))
        XCTAssertTrue(isDirectory.boolValue)
    }
}
