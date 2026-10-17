import MLX
import MLXNN
@testable import MiniCPMAudio
import XCTest

final class MiniCPMAudioConfigurationTests: XCTestCase {
    func testStreamingKeyGeometryUsesOfficialLengthsUnlessExplicitlyOptedIn() {
        XCTAssertFalse(MiniCPMEncoderAttention.keyBucketEnabled(environment: [:]))
        XCTAssertFalse(MiniCPMEncoderAttention.keyBucketEnabled(environment: [
            "MINICPM_AUDIO_FAST": "1", "MINICPM_AUDIO_LINEAR": "1",
        ]))
        XCTAssertFalse(MiniCPMEncoderAttention.keyBucketEnabled(environment: [
            "MINICPM_AUDIO_KV_BUCKET": "0",
        ]))
        XCTAssertTrue(MiniCPMEncoderAttention.keyBucketEnabled(environment: [
            "MINICPM_AUDIO_KV_BUCKET": "1",
        ]))
        for realLength in [50, 100, 150, 256, 300, 1_450] {
            XCTAssertEqual(MiniCPMEncoderAttention.attentionKeyLength(
                realLength: realLength, bucketEnabled: false), realLength)
        }
        XCTAssertEqual(MiniCPMEncoderAttention.attentionKeyLength(
            realLength: 50, bucketEnabled: true), 256)
        XCTAssertEqual(MiniCPMEncoderAttention.attentionKeyLength(
            realLength: 300, bucketEnabled: true), 512)
    }

    func testReadsOriginalMiniCPMConfigShape() throws {
        let json = """
        {
          "audio_pool_step": 5,
          "hidden_size": 4096,
          "audio_config": {
            "d_model": 1024,
            "encoder_ffn_dim": 4096,
            "encoder_attention_heads": 16,
            "encoder_layers": 24,
            "num_mel_bins": 80,
            "max_source_positions": 1500
          }
        }
        """
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
            .appendingPathExtension("json")
        try Data(json.utf8).write(to: url)
        defer { try? FileManager.default.removeItem(at: url) }

        let configuration = try MiniCPMAudioConfiguration.fromMiniCPMConfig(at: url)
        XCTAssertEqual(configuration.hiddenSize, 1_024)
        XCTAssertEqual(configuration.intermediateSize, 4_096)
        XCTAssertEqual(configuration.attentionHeads, 16)
        XCTAssertEqual(configuration.layers, 24)
        XCTAssertEqual(configuration.poolStep, 5)
        XCTAssertEqual(configuration.projectionSize, 4_096)
        XCTAssertEqual(configuration.layerNormEpsilon, 1e-5)
    }
}

final class MiniCPMWhisperFeatureExtractorTests: XCTestCase {
    func testFloat32FFTMatchesTorchAtBF16EncoderBoundary() throws {
        // Independent torch 2.8 CPU stft/hann/mel oracle, fixed floor -10.
        let pcm = (0..<560).map { Float(($0 * 37) % 255 - 127) / 16384 }
        let features = try MiniCPMWhisperFeatureExtractor().extract(
            pcm, normalization: .fixed(floorDB: -10))
        let expectedBits: [UInt16] = [
            0xbe2f, 0xbdd9, 0xbd9c, 0xbd3d, 0x3b5b, 0xbd28, 0xbe2d, 0xbea1, 0xbeb3, 0xbe85, 0xbe49, 0xbe16,
            0xbe4f, 0xbf16, 0xbf09, 0xbea6, 0xbe82, 0xbe5b, 0xbe5b, 0xbea6, 0xbf23, 0xbed4, 0xbe95, 0xbe74,
            0xbe6a, 0xbe74, 0xbee5, 0xbf0e, 0xbea3, 0xbe63, 0xbe63, 0xbe85, 0xbee3, 0xbeb3, 0xbe48, 0xbe50,
            0xbe81, 0xbeef, 0xbe2e, 0xbe1a, 0xbe4c, 0xbe87, 0xbd8b, 0xbd93, 0xbe2d, 0x3db9, 0x3e3b, 0x3e9a,
            0x3e8e, 0x3e2e, 0xbe0b, 0xbd6d, 0xbe8b, 0xbe47, 0xbea0, 0xbe9c, 0xbeb7, 0xbebc, 0xbed4, 0xbec5,
            0xbedd, 0xbeb8, 0xbeba, 0xbe7d, 0xbdbb, 0x3dd9, 0x3de7, 0xbe0d, 0xbe7b, 0xbe9d, 0xbeb0, 0xbec0,
            0xbec5, 0xbebc, 0xbe9d, 0xbdb4, 0x3d71, 0xbdbb, 0xbe99, 0xbeb8, 0xbefe, 0xbeef, 0xbefe, 0xbf06,
            0xbeb6, 0xbd32, 0xbc2b, 0xbdb5, 0xbeb6, 0xbf1c, 0xbf51, 0xbed7, 0xbe43, 0xbe02, 0xbe88, 0xbf1d,
            0xbf29, 0xbf14, 0xbed7, 0xbe74, 0xbe64, 0xbe9e, 0xbf06, 0xbf18, 0xbf2b, 0xbeec, 0xbe86, 0xbe7c,
            0xbebb, 0xbf34, 0xbf40, 0xbedf, 0xbe8e, 0xbe84, 0xbf06, 0xbf17, 0xbece, 0xbe6d, 0xbe91, 0xbf09,
            0xbed3, 0xbe28, 0xbe66, 0xbf05, 0xbde6, 0xbb2f, 0xbec1, 0x3e85, 0x3ebb, 0x3d6b, 0xbe0a, 0xbdff,
            0xbee1, 0xbe51, 0xbee2, 0xbe95, 0xbec6, 0xbeaa, 0xbec8, 0xbea7, 0xbecb, 0xbe8e, 0xbea3, 0xbe59,
            0xbd72, 0x3cb6, 0x3e20, 0xbe1e, 0xbe7f, 0xbe9c, 0xbeab, 0xbeb5, 0xbeb4, 0xbea4, 0xbe7c, 0xbd6c,
            0x3cf5, 0xbd24, 0xbe85, 0xbeab, 0xbef6, 0xbed2, 0xbee7, 0xbee6, 0xbe62, 0xbcdb, 0xbc9c, 0xbe05,
            0xbede, 0xbee0, 0xbedb, 0xbe9c, 0xbe19, 0xbe30, 0xbedc, 0xbec9, 0xbee5, 0xbee1, 0xbe94, 0xbe77,
            0xbe9a, 0xbec5, 0xbec2, 0xbee6, 0xbed7, 0xbeab, 0xbe91, 0xbecf, 0xbea4, 0xbed7, 0xbee2, 0xbeb7,
            0xbec0, 0xbe93, 0xbeb5, 0xbeeb, 0xbebf, 0xbe9c, 0xbe80, 0xbed6, 0xbed1, 0xbe4a, 0xbe4b, 0xbed8,
            0xbe0d, 0xbb49, 0xbe97, 0x3e86, 0x3eba, 0x3d35, 0xbde9, 0xbe1a, 0xbeaa, 0xbe66, 0xbecb, 0xbe99,
            0xbec4, 0xbeb0, 0xbebf, 0xbeb3, 0xbeb9, 0xbe99, 0xbea2, 0xbe53, 0xbd86, 0x3cfb, 0x3e1f, 0xbe1c,
            0xbe82, 0xbe9e, 0xbeb0, 0xbeb8, 0xbeb5, 0xbea5, 0xbe7f, 0xbd75, 0x3d07, 0xbd2e, 0xbe87, 0xbead,
        ]
        XCTAssertEqual(features.timeFrames, 3)
        let actual = MLXArray(features.data).asType(.bfloat16).asType(.float32)
            .asArray(Float.self)
        let expected = expectedBits.map { Float(bitPattern: UInt32($0) << 16) }
        XCTAssertEqual(actual, expected)
    }

    func testOneSecondProducesOneHundredFrames() throws {
        let audio = (0..<16_000).map { sample in
            Float(sin(2 * Double.pi * 440 * Double(sample) / 16_000)) * 0.1
        }
        let features = try MiniCPMWhisperFeatureExtractor().extract(audio)
        XCTAssertEqual(features.timeFrames, 100)
        XCTAssertEqual(features.melBins, 80)
        XCTAssertEqual(features.data.count, 8_000)
        XCTAssertTrue(features.data.allSatisfy(\.isFinite))
    }

    func testStreamingFirstChunkIncludesTwoRightContextFrames() throws {
        let processor = MiniCPMStreamingMelProcessor()
        let audio = [Float](repeating: 0, count: processor.nextChunkSampleCount)
        let features = try XCTUnwrap(processor.accept(audio))
        XCTAssertEqual(features.timeFrames, 102)
        XCTAssertEqual(features.prefixExtraFrames, 0)
        XCTAssertEqual(features.suffixExtraFrames, 2)
        XCTAssertEqual(processor.nextChunkSampleCount, 16_000)
    }

    func testFirstChunkRequestRemainsRemainderWhenInputArrivesInPackets() throws {
        let processor = MiniCPMStreamingMelProcessor()
        XCTAssertEqual(processor.nextChunkSampleCount, 16_480)
        XCTAssertNil(try processor.accept([Float](repeating: 0, count: 1_000)))
        XCTAssertEqual(processor.nextChunkSampleCount, 15_480)
        XCTAssertNil(try processor.accept([Float](repeating: 0, count: 15_479)))
        XCTAssertEqual(processor.nextChunkSampleCount, 1)
        XCTAssertNotNil(try processor.accept([0]))
        XCTAssertFalse(processor.nextChunkSampleCount == 1)
    }

    func testTwoRegularChunksMatchFullExtractionAtEachEmission() throws {
        let first = (0..<16_480).map { index in
            Float(sin(Double(index) * 0.017)) * 0.2
        }
        let second = (0..<16_000).map { index in
            Float(cos(Double(index) * 0.013)) * 0.15
        }
        let processor = MiniCPMStreamingMelProcessor()
        let extractor = MiniCPMWhisperFeatureExtractor()
        let firstOutput = try XCTUnwrap(processor.accept(first))
        // The upstream streaming processor uses the fixed -10 dB floor for
        // buffers shorter than five seconds; compare against the same offline
        // normalization mode rather than the extractor's dynamic default.
        let firstFull = try extractor.extract(
            first, normalization: .fixed(floorDB: -10))
        XCTAssertEqual(firstOutput.timeFrames, 102)
        let firstExpected = firstFull.frames(0..<102).data
        XCTAssertEqual(firstOutput.data.count, firstExpected.count)
        XCTAssertLessThan(
            zip(firstOutput.data, firstExpected)
                .map { abs($0 - $1) }.max() ?? .infinity,
            1e-5)

        let secondOutput = try XCTUnwrap(processor.accept(second))
        let combined = first + second
        let secondFull = try extractor.extract(
            combined, normalization: .fixed(floorDB: -10))
        XCTAssertEqual(secondOutput.timeFrames, 104)
        XCTAssertEqual(secondOutput.prefixExtraFrames, 2)
        XCTAssertEqual(secondOutput.suffixExtraFrames, 2)
        let secondExpected = secondFull.frames(98..<202).data
        XCTAssertEqual(secondOutput.data.count, secondExpected.count)
        XCTAssertLessThan(
            zip(secondOutput.data, secondExpected)
                .map { abs($0 - $1) }.max() ?? .infinity,
            1e-5)
        // The normalization window retains PCM until the official 30/10 s
        // slide, so an early loud event expires with the reference window.
        XCTAssertEqual(processor.bufferedSampleCount, 32_480)
        XCTAssertEqual(processor.bufferStartSampleIndex, 0)
    }

    func testFinalFlushReturnsReflectedTailAndSnapshotReplaysExactly() throws {
        let first = (0..<16_480).map { Float(sin(Double($0) * 0.011)) * 0.2 }
        let second = (0..<4_321).map { Float(cos(Double($0) * 0.021)) * 0.1 }
        let processor = MiniCPMStreamingMelProcessor()
        _ = try XCTUnwrap(processor.accept(first))
        let checkpoint = processor.snapshot()
        let expectedTail = try XCTUnwrap(processor.accept(second, isFinal: true))
        XCTAssertEqual(expectedTail.prefixExtraFrames, 2)
        XCTAssertEqual(expectedTail.suffixExtraFrames, 0)
        let full = try MiniCPMWhisperFeatureExtractor().extract(
            first + second, normalization: .fixed(floorDB: -10))
        let expected = full.frames(98..<full.timeFrames).data
        XCTAssertLessThan(zip(expectedTail.data, expected)
            .map { abs($0 - $1) }.max() ?? .infinity, 1e-5)

        processor.restoreSnapshot(checkpoint)
        let replayedTail = try XCTUnwrap(processor.accept(second, isFinal: true))
        XCTAssertEqual(replayedTail.timeFrames, expectedTail.timeFrames)
        XCTAssertLessThan(
            zip(replayedTail.data, expectedTail.data)
                .map { abs($0 - $1) }.max() ?? .infinity,
            1e-5)
        XCTAssertEqual(processor.emittedFrameCount, checkpoint.lastEmittedFrame
            + expectedTail.timeFrames - expectedTail.prefixExtraFrames)
        let flushedCheckpoint = processor.snapshot()
        processor.reset()
        processor.restore(flushedCheckpoint)
        XCTAssertNil(try processor.accept([], isFinal: true))
    }

    func testLongSessionKeepsBoundedPCMStorage() throws {
        let processor = MiniCPMStreamingMelProcessor()
        let chunk = (0..<16_000).map { Float(sin(Double($0) * 0.007)) * 0.05 }
        _ = try processor.accept([Float](repeating: 0, count: 16_480))
        for _ in 0..<120 {
            _ = try processor.accept(chunk)
            XCTAssertLessThan(processor.bufferedSampleCount, 480_000)
        }
        XCTAssertGreaterThan(processor.totalSampleCount, 1_900_000)
        XCTAssertGreaterThan(processor.emittedFrameCount, 11_000)
    }
}

final class MiniCPMAudioCacheTests: XCTestCase {
    func testDefaultStreamingAttentionCacheAndMaskHaveNoPaddedPositions() throws {
        guard !MiniCPMEncoderAttention.keyBucketEnabled() else {
            throw XCTSkip("unset MINICPM_AUDIO_KV_BUCKET to test official default geometry")
        }
        let configuration = MiniCPMAudioConfiguration(
            hiddenSize: 8, intermediateSize: 16, attentionHeads: 2,
            layers: 1, poolStep: 1, projectionSize: 8)
        let attention = MiniCPMEncoderAttention(configuration)
        let input = MLXArray.zeros([1, 4, 8])
        var trace: [String: MLXArray] = [:]
        let (_, first) = attention(input, cache: nil, useCache: true,
                                   tracePrefix: "", trace: { trace[$0] = $1 })
        XCTAssertEqual(try XCTUnwrap(trace["attention_scores"]).shape, [1, 2, 4, 4])
        XCTAssertEqual(first?.keys.shape, [1, 2, 4, 4])
        XCTAssertEqual(first?.realLength, 4)

        trace.removeAll()
        let (_, second) = attention(input, cache: first, useCache: true,
                                    tracePrefix: "", trace: { trace[$0] = $1 })
        let scores = try XCTUnwrap(trace["attention_scores"])
        let probabilities = try XCTUnwrap(trace["attention_probabilities"])
        eval(scores, probabilities)
        XCTAssertEqual(scores.shape, [1, 2, 4, 8])
        XCTAssertEqual(second?.keys.shape, [1, 2, 8, 4])
        XCTAssertEqual(second?.realLength, 8)
        // Zero inputs yield zero K (the official projection has no bias), so
        // every real key has equal mass. A masked padded block would introduce
        // extra zero-probability positions and a different key dimension.
        XCTAssertEqual(scores.max().item(Float.self), 0)
        XCTAssertEqual(scores.min().item(Float.self), 0)
        XCTAssertEqual(probabilities.min().item(Float.self), 0.125, accuracy: 1e-6)
        XCTAssertEqual(probabilities.max().item(Float.self), 0.125, accuracy: 1e-6)
    }

    func testCacheGrowsByPostConvolutionPositions() throws {
        let configuration = MiniCPMAudioConfiguration(
            hiddenSize: 8,
            intermediateSize: 32,
            attentionHeads: 2,
            layers: 2,
            melBins: 80,
            maximumSourcePositions: 32,
            poolStep: 2,
            projectionSize: 16
        )
        let model = MiniCPMAudioModel(configuration: configuration)
        model.train(false)
        let features = MLXArray.zeros([1, 8, 80])
        let first = try model.encode(features: features)
        let second = try model.encode(features: features, cache: first.cache)
        eval(first.embeddings, second.embeddings)

        XCTAssertEqual(first.encoderStates.shape, [1, 4, 8])
        XCTAssertEqual(first.embeddings.shape, [1, 2, 16])
        XCTAssertEqual(first.cache?.length, 4)
        XCTAssertEqual(second.cache?.length, 8)
        XCTAssertEqual(second.cache?.layers.count, 2)
    }

    func testStreamingSessionReleasesCacheAtPositionLimit() throws {
        let configuration = MiniCPMAudioConfiguration(
            hiddenSize: 8,
            intermediateSize: 32,
            attentionHeads: 2,
            layers: 1,
            melBins: 80,
            maximumSourcePositions: 8,
            poolStep: 2,
            projectionSize: 16
        )
        let session = MiniCPMAudioStreamingSession(
            model: MiniCPMAudioModel(configuration: configuration)
        )
        let features = MLXArray.zeros([1, 8, 80]) // four encoder positions
        let first = try session.encodeFeatures(features)
        let second = try session.encodeFeatures(features)
        eval(first.output.embeddings, second.output.embeddings)

        XCTAssertFalse(first.didResetCache)
        XCTAssertTrue(second.didResetCache)
        XCTAssertEqual(session.cacheResetCount, 1)
        XCTAssertEqual(session.cache?.length, 4)
    }

    func testContextCacheResetUsesOfficialPreCropGreaterOrEqualBoundary() throws {
        let configuration = MiniCPMAudioConfiguration(
            hiddenSize: 8, intermediateSize: 32, attentionHeads: 2,
            layers: 1, maximumSourcePositions: 10, poolStep: 1,
            projectionSize: 16)
        let session = MiniCPMAudioStreamingSession(
            model: MiniCPMAudioModel(configuration: configuration))
        // Four CNN positions minus one right-context position = three core
        // positions. Before the third call, 6 + 4 == 10 triggers the official
        // reset; incorrectly checking 6 + 3 would keep the stale cache.
        let features = MLXArray.zeros([1, 8, 80])
        let first = try session.encodeFeatures(features, suffixExtraFrames: 2)
        let second = try session.encodeFeatures(features, suffixExtraFrames: 2)
        let third = try session.encodeFeatures(features, suffixExtraFrames: 2)
        eval(first.output.embeddings, second.output.embeddings, third.output.embeddings)
        XCTAssertFalse(first.didResetCache)
        XCTAssertFalse(second.didResetCache)
        XCTAssertEqual(second.output.cache?.length, 6)
        XCTAssertTrue(third.didResetCache)
        XCTAssertEqual(third.output.cache?.length, 3)
        XCTAssertEqual(session.cacheResetCount, 1)
    }

    func testStreamingSessionSnapshotRestoresAcrossCacheReset() throws {
        let configuration = MiniCPMAudioConfiguration(
            hiddenSize: 8,
            intermediateSize: 32,
            attentionHeads: 2,
            layers: 1,
            melBins: 80,
            maximumSourcePositions: 8,
            poolStep: 2,
            projectionSize: 16)
        let session = MiniCPMAudioStreamingSession(
            model: MiniCPMAudioModel(configuration: configuration))
        let features = MLXArray.zeros([1, 8, 80])
        _ = try session.encodeFeatures(features)
        let checkpoint = session.snapshot()

        let advanced = try session.encodeFeatures(features)
        eval(advanced.output.embeddings)
        XCTAssertTrue(advanced.didResetCache)
        XCTAssertEqual(session.cacheResetCount, 1)

        // Restore the exact pre-reset cache/frontend boundary. Repeating the
        // same input must take the same reset path and produce identical
        // embeddings without replaying any earlier feature chunk.
        session.restore(checkpoint)
        XCTAssertEqual(session.cache?.length, 4)
        XCTAssertEqual(session.cacheResetCount, 0)
        let replayed = try session.encodeFeatures(features)
        eval(replayed.output.embeddings)
        XCTAssertTrue(replayed.didResetCache)
        XCTAssertEqual(replayed.output.embeddings.shape,
                       advanced.output.embeddings.shape)
        let drift = zip(
            replayed.output.embeddings.asType(.float32).asArray(Float.self),
            advanced.output.embeddings.asType(.float32).asArray(Float.self))
            .map { abs($0 - $1) }.max() ?? 0
        XCTAssertLessThan(drift, 1e-5)
    }

    func testStreamingSessionSnapshotRestoresShortFirstWindow() throws {
        let configuration = MiniCPMAudioConfiguration(
            hiddenSize: 8,
            intermediateSize: 32,
            attentionHeads: 2,
            layers: 1,
            melBins: 80,
            maximumSourcePositions: 32,
            poolStep: 2,
            projectionSize: 16)
        let session = MiniCPMAudioStreamingSession(
            model: MiniCPMAudioModel(configuration: configuration))
        let checkpoint = session.snapshot()
        let packet = [Float](repeating: 0, count: 1_000)
        XCTAssertNil(try session.acceptAudio(packet))
        XCTAssertEqual(session.melProcessor.totalSampleCount, 1_000)

        session.restore(checkpoint)
        XCTAssertEqual(session.melProcessor.totalSampleCount, 0)
        XCTAssertNil(try session.acceptAudio(packet))
        XCTAssertEqual(session.melProcessor.totalSampleCount, 1_000)
        XCTAssertEqual(
            session.melProcessor.nextChunkSampleCount,
            MiniCPMStreamingMelProcessor.firstChunkSamples - 1_000)
    }

    func testStreamingSessionAccountsForRemovedCNNContext() throws {
        let configuration = MiniCPMAudioConfiguration(
            hiddenSize: 8,
            intermediateSize: 32,
            attentionHeads: 2,
            layers: 1,
            melBins: 80,
            maximumSourcePositions: 32,
            poolStep: 1,
            projectionSize: 16)
        let session = MiniCPMAudioStreamingSession(
            model: MiniCPMAudioModel(configuration: configuration))
        let features = MLXArray.zeros([1, 10, 80])
        let first = try session.encodeFeatures(
            features, prefixExtraFrames: 0, suffixExtraFrames: 2)
        let second = try session.encodeFeatures(
            features, prefixExtraFrames: 2, suffixExtraFrames: 2)
        eval(first.output.embeddings, second.output.embeddings)

        XCTAssertEqual(first.output.encoderStates.dim(1), 4)
        XCTAssertEqual(second.output.encoderStates.dim(1), 3)
        XCTAssertEqual(session.cache?.length, 7)
    }

    func testOfflineReferenceDoesNotPolluteStreamingCache() throws {
        let configuration = MiniCPMAudioConfiguration(
            hiddenSize: 8,
            intermediateSize: 32,
            attentionHeads: 2,
            layers: 1,
            melBins: 80,
            maximumSourcePositions: 128,
            poolStep: 1,
            projectionSize: 16)
        let session = MiniCPMAudioStreamingSession(
            model: MiniCPMAudioModel(configuration: configuration))

        _ = try session.encodeFeatures(MLXArray.zeros([1, 8, 80]))
        let beforeReference = try XCTUnwrap(session.cache?.length)

        // The offline path has its own mel extractor and uses no encoder KV;
        // a 100 ms clip is enough to exercise the complete frontend while
        // keeping this test independent of downloaded model weights.
        let reference = try session.encodeReferenceAudio(
            [Float](repeating: 0, count: 1_600))
        eval(reference.output.embeddings)
        XCTAssertFalse(reference.didResetCache)
        XCTAssertTrue(reference.output.embeddings.asType(.float32)
            .asArray(Float.self).allSatisfy(\.isFinite))
        XCTAssertEqual(session.cache?.length, beforeReference)

        _ = try session.encodeFeatures(MLXArray.zeros([1, 8, 80]))
        XCTAssertEqual(session.cache?.length, beforeReference + 4)
    }

    func testAcceptAudioUsesStreamingMelSession() throws {
        let configuration = MiniCPMAudioConfiguration(
            hiddenSize: 8,
            intermediateSize: 32,
            attentionHeads: 2,
            layers: 1,
            melBins: 80,
            maximumSourcePositions: 128,
            poolStep: 5,
            projectionSize: 16)
        let session = MiniCPMAudioStreamingSession(
            model: MiniCPMAudioModel(configuration: configuration))
        let first = try XCTUnwrap(session.acceptAudio(
            [Float](repeating: 0, count: MiniCPMStreamingMelProcessor.firstChunkSamples)))
        let second = try XCTUnwrap(session.acceptAudio(
            [Float](repeating: 0, count: MiniCPMStreamingMelProcessor.regularChunkSamples)))
        eval(first.output.embeddings, second.output.embeddings)

        // The PCM path must use the official context metadata: first chunk
        // 0/2, then 2/2.  Each two-frame side removes one post-convolution
        // position, and the model applies that crop exactly once.
        XCTAssertEqual(first.output.encoderStates.shape, [1, 50, 8])
        XCTAssertEqual(first.output.embeddings.shape, [1, 10, 16])
        XCTAssertEqual(first.output.cache?.length, 50)
        XCTAssertEqual(second.output.encoderStates.shape, [1, 50, 8])
        XCTAssertEqual(second.output.embeddings.shape, [1, 10, 16])
        XCTAssertEqual(session.cache?.length, 100)
    }
}

final class E2EMiniCPMAudioConvolutionLaunchTests: XCTestCase {
    private func array(_ shape: [Int], phase: Int, dtype: DType) -> MLXArray {
        let values = (0..<shape.reduce(1, *)).map { index in
            Float((index * 17 + phase) % 97 - 48) / 113
        }
        return MLXArray(values, shape).asType(dtype)
    }

    private func assertLaunchesMatch(
        convolutionIndex: Int,
        dtype: DType,
        file: StaticString = #filePath,
        line: UInt = #line
    ) throws {
        // Exercise clamped groups and a partially filled final group, both
        // padding edges, and more than one batch plane for each stride.
        for channels in [8, 33] {
            let configuration = MiniCPMAudioConfiguration(
                hiddenSize: channels, intermediateSize: channels * 2,
                attentionHeads: 1, layers: 1, poolStep: 1,
                projectionSize: channels)
            let model = MiniCPMAudioModel(configuration: configuration)
            let layer = convolutionIndex == 1 ? model.encoder.conv1 : model.encoder.conv2
            try layer.update(parameters: ModuleParameters.unflattened([
                "weight": array(layer.weight.shape, phase: 7, dtype: dtype),
                "bias": array([channels], phase: 31, dtype: dtype),
            ]), verify: .all)
            let inputChannels = convolutionIndex == 1 ? configuration.melBins : channels
            for length in [5, 6] {
                for batch in [1, 2] {
                    let input = array([batch, length, inputChannels], phase: 53, dtype: dtype)
                    let variants = model.diagnosticConvolutionLaunchVariants(
                        input: input, convolutionIndex: convolutionIndex)
                    eval(variants.scalar, variants.grouped)
                    let outputLength = convolutionIndex == 1 ? length : (length + 1) / 2
                    let context = "conv\(convolutionIndex), \(dtype), B=\(batch), L=\(length), C=\(channels)"
                    XCTAssertEqual(variants.grouped.shape, [batch, outputLength, channels],
                                   context, file: file, line: line)
                    XCTAssertEqual(variants.scalar.shape, variants.grouped.shape,
                                   context, file: file, line: line)
                    XCTAssertEqual(variants.scalar.dtype, dtype, context, file: file, line: line)
                    XCTAssertEqual(variants.grouped.dtype, dtype, context, file: file, line: line)
                    XCTAssertEqual(
                        variants.grouped.asData(access: .copy).data,
                        variants.scalar.asData(access: .copy).data,
                        context, file: file, line: line)
                }
            }
        }
    }

    func testBF16Conv1GroupedLaunchMatchesScalarBitwise() throws {
        try assertLaunchesMatch(convolutionIndex: 1, dtype: .bfloat16)
    }

    func testBF16Conv2GroupedLaunchMatchesScalarBitwise() throws {
        try assertLaunchesMatch(convolutionIndex: 2, dtype: .bfloat16)
    }

    func testFloat32Conv1GroupedLaunchMatchesScalarBitwise() throws {
        try assertLaunchesMatch(convolutionIndex: 1, dtype: .float32)
    }

    func testFloat32Conv2GroupedLaunchMatchesScalarBitwise() throws {
        try assertLaunchesMatch(convolutionIndex: 2, dtype: .float32)
    }
}

final class E2EMiniCPMAudioTests: XCTestCase {
    func testStageTimingBenchmark() throws {
        guard let path = ProcessInfo.processInfo.environment["MINICPM_AUDIO_MLX_PATH"] else {
            throw XCTSkip("set MINICPM_AUDIO_MLX_PATH to the converted audio bundle")
        }
        let loadStart = Date()
        let model = try MiniCPMAudioModel.fromDirectory(
            URL(fileURLWithPath: path, isDirectory: true))
        print("[bench] model load: \(Date().timeIntervalSince(loadStart) * 1000) ms")

        let sampleRate = 16_000
        let audio = (0..<sampleRate).map { Float($0 % 100) / 100.0 * 0.02 }
        let extractor = MiniCPMWhisperFeatureExtractor()
        let session = MiniCPMAudioStreamingSession(model: model)

        // warm-up: first calls pay Metal kernel compilation.
        for _ in 0..<2 {
            _ = try session.acceptAudio(audio)
            _ = try extractor.extract(audio)
        }
        session.reset()

        var melMS: [Double] = []
        var acceptMS: [Double] = []
        for _ in 0..<8 {
            let t0 = Date()
            let features = try extractor.extract(audio)
            let t1 = Date()
            var last: MiniCPMStreamingAudioEncoding?
            for chunkStart in stride(from: 0, to: audio.count, by: sampleRate) {
                last = try session.acceptAudio(Array(audio[chunkStart..<min(chunkStart + sampleRate, audio.count)]))
            }
            if let last { eval(last.output.embeddings) }
            let t2 = Date()
            melMS.append(t1.timeIntervalSince(t0) * 1000)
            acceptMS.append(t2.timeIntervalSince(t1) * 1000)
            session.reset()
        }
        func stats(_ values: [Double]) -> String {
            let sorted = values.sorted()
            return String(format: "min=%.1f med=%.1f max=%.1f ms",
                          sorted.first ?? -1, sorted[sorted.count / 2], sorted.last ?? -1)
        }
        print("[bench] mel extract 1s audio: \(stats(melMS))")
        print("[bench] acceptAudio (encode path, per 1s): \(stats(acceptMS))")

        // long-lived session: does the streaming KV growth slow chunks down?
        let longSession = MiniCPMAudioStreamingSession(model: model)
        var perChunk: [Double] = []
        for index in 0..<20 {
            let t0 = Date()
            _ = try longSession.acceptAudio(audio)
            perChunk.append(Date().timeIntervalSince(t0) * 1000)
        }
        print("[bench] long session per-chunk ms: \(perChunk.map { Int($0) })")

        // memory-pressure probe: a large resident allocation, then re-time
        let ballast = MLXArray.zeros([64, 16_000_000], dtype: .bfloat16)
        eval(ballast)
        print("[bench] ballast bytes: \(ballast.nbytes / 1_000_000) MB")
        var pressuredMS: [Double] = []
        let pressuredSession = MiniCPMAudioStreamingSession(model: model)
        for _ in 0..<6 {
            let t0 = Date()
            _ = try pressuredSession.acceptAudio(audio)
            pressuredMS.append(Date().timeIntervalSince(t0) * 1000)
        }
        print("[bench] acceptAudio under 2GB ballast: \(stats(pressuredMS))")

        // raw encoder forward on a precomputed feature block
        let features = try extractor.extract(audio)
        let array = MLXArray(features.data, [1, features.timeFrames, features.melBins])
        _ = try model.encode(features: array, useCache: false)
        var encodeMS: [Double] = []
        for _ in 0..<8 {
            let t0 = Date()
            let out = try model.encode(features: array, useCache: false)
            eval(out.embeddings)
            encodeMS.append(Date().timeIntervalSince(t0) * 1000)
        }
        print("[bench] model.encode [1,\(features.timeFrames),80]: \(stats(encodeMS))")

        // shape scaling: is the cost per-call overhead or real compute?
        for frames in [10, 25, 50, 100] {
            let block = MLXArray.zeros([1, frames, 80])
            _ = try model.encode(features: block, useCache: false)
            var times: [Double] = []
            for _ in 0..<4 {
                let t0 = Date()
                let out = try model.encode(features: block, useCache: false)
                eval(out.embeddings)
                times.append(Date().timeIntervalSince(t0) * 1000)
            }
            print("[bench] encode frames=\(frames): \(stats(times))")
        }
        // repeated tiny forward: pure per-call overhead floor
        let tiny = MLXArray.zeros([1, 16, 80])
        _ = try model.encode(features: tiny, useCache: false)
        var tinyMS: [Double] = []
        var buildOnlyMS: [Double] = []
        for _ in 0..<8 {
            let t0 = Date()
            let out = try model.encode(features: tiny, useCache: false)
            let t1 = Date()
            eval(out.embeddings)
            let t2 = Date()
            tinyMS.append(t2.timeIntervalSince(t0) * 1000)
            buildOnlyMS.append(t1.timeIntervalSince(t0) * 1000)
        }
        print("[bench] encode frames=16 total: \(stats(tinyMS))  graph-build-only: \(stats(buildOnlyMS))")
        // large Metal buffer cache, then re-time the tiny forward
        MLX.Memory.cacheLimit = 2 * 1024 * 1024 * 1024
        var cachedMS: [Double] = []
        for _ in 0..<8 {
            let t0 = Date()
            let out = try model.encode(features: tiny, useCache: false)
            eval(out.embeddings)
            cachedMS.append(Date().timeIntervalSince(t0) * 1000)
        }
        print("[bench] encode frames=16 with 2GB buffer cache: \(stats(cachedMS))")
        MLX.Memory.cacheLimit = MLX.Memory.cacheLimit
    }

    func testRealWeightsProduceFiniteProjectedEmbeddings() throws {
        guard let path = ProcessInfo.processInfo.environment["MINICPM_AUDIO_MLX_PATH"] else {
            throw XCTSkip("set MINICPM_AUDIO_MLX_PATH to the converted audio bundle")
        }
        let model = try MiniCPMAudioModel.fromDirectory(
            URL(fileURLWithPath: path, isDirectory: true))
        let features = MLXArray.zeros([1, 100, 80])
        let output = try model.encode(features: features, useCache: false)
        eval(output.embeddings)

        XCTAssertEqual(output.embeddings.shape, [1, 10, 4_096])
        XCTAssertTrue(output.embeddings.asType(.float32)
            .asArray(Float.self).allSatisfy(\.isFinite))
    }
}
