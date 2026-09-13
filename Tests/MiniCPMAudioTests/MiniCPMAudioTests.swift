import MLX
@testable import MiniCPMAudio
import XCTest

final class MiniCPMAudioConfigurationTests: XCTestCase {
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

    func testStreamingFirstChunkEmitsExactlyOneHundredStableFrames() throws {
        let processor = MiniCPMStreamingMelProcessor()
        let audio = [Float](repeating: 0, count: processor.nextChunkSampleCount)
        let features = try XCTUnwrap(processor.accept(audio))
        XCTAssertEqual(features.timeFrames, 100)
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
        XCTAssertEqual(firstOutput.timeFrames, 100)
        let firstExpected = firstFull.frames(0..<100).data
        XCTAssertEqual(firstOutput.data.count, firstExpected.count)
        XCTAssertLessThan(
            zip(firstOutput.data, firstExpected)
                .map { abs($0 - $1) }.max() ?? .infinity,
            1e-5)

        let secondOutput = try XCTUnwrap(processor.accept(second))
        let combined = first + second
        let secondFull = try extractor.extract(
            combined, normalization: .fixed(floorDB: -10))
        XCTAssertEqual(secondOutput.timeFrames, 100)
        let secondExpected = secondFull.frames(100..<200).data
        XCTAssertEqual(secondOutput.data.count, secondExpected.count)
        XCTAssertLessThan(
            zip(secondOutput.data, secondExpected)
                .map { abs($0 - $1) }.max() ?? .infinity,
            1e-5)
        // The old implementation retained the complete waveform.  The new
        // ring only needs the 200-sample left context plus the next lookahead.
        XCTAssertLessThan(processor.bufferedSampleCount, 2_000)
        XCTAssertEqual(processor.bufferStartSampleIndex, 31_800)
    }

    func testFinalFlushReturnsReflectedTailAndSnapshotReplaysExactly() throws {
        let first = (0..<16_480).map { Float(sin(Double($0) * 0.011)) * 0.2 }
        let second = (0..<4_321).map { Float(cos(Double($0) * 0.021)) * 0.1 }
        let processor = MiniCPMStreamingMelProcessor()
        _ = try XCTUnwrap(processor.accept(first))
        let checkpoint = processor.snapshot()
        let expectedTail = try XCTUnwrap(processor.accept(second, isFinal: true))

        processor.restoreSnapshot(checkpoint)
        let replayedTail = try XCTUnwrap(processor.accept(second, isFinal: true))
        XCTAssertEqual(replayedTail.timeFrames, expectedTail.timeFrames)
        XCTAssertLessThan(
            zip(replayedTail.data, expectedTail.data)
                .map { abs($0 - $1) }.max() ?? .infinity,
            1e-5)
        XCTAssertEqual(processor.emittedFrameCount, checkpoint.lastEmittedFrame
            + expectedTail.timeFrames)
    }

    func testLongSessionKeepsBoundedPCMStorage() throws {
        let processor = MiniCPMStreamingMelProcessor()
        let chunk = (0..<16_000).map { Float(sin(Double($0) * 0.007)) * 0.05 }
        _ = try processor.accept([Float](repeating: 0, count: 16_480))
        for _ in 0..<120 {
            _ = try processor.accept(chunk)
            XCTAssertLessThanOrEqual(processor.bufferedSampleCount, 33_000)
        }
        XCTAssertGreaterThan(processor.totalSampleCount, 1_900_000)
        XCTAssertGreaterThan(processor.emittedFrameCount, 11_000)
    }
}

final class MiniCPMAudioCacheTests: XCTestCase {
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
            poolStep: 2,
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
        XCTAssertEqual(first.output.encoderStates.shape, [1, 49, 8])
        XCTAssertEqual(first.output.embeddings.shape, [1, 24, 16])
        XCTAssertEqual(first.output.cache?.length, 49)
        XCTAssertEqual(second.output.encoderStates.shape, [1, 48, 8])
        XCTAssertEqual(second.output.embeddings.shape, [1, 24, 16])
        XCTAssertEqual(session.cache?.length, 97)
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
