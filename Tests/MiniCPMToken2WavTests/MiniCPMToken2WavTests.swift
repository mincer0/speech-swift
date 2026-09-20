import Foundation
import MLX
@testable import MiniCPMToken2Wav
import XCTest

final class MiniCPMHiFTConfigurationTests: XCTestCase {
    func testMiniCPMSourceUpsampleScale() {
        XCTAssertEqual(MiniCPMHiFTConfiguration().sourceUpsampleScale, 480)
        var custom = MiniCPMHiFTConfiguration()
        custom.upsampleRates = [2, 2]
        custom.fftHopLength = 3
        XCTAssertEqual(custom.sourceUpsampleScale, 12)
    }

    func testLinearUpsampleUsesAlignCornersFalseCoordinates() {
        let input = MLXArray([Float(0), 10], [1, 2, 1])
        let output = miniCPMLinearUpsample(input, scale: 2)
        eval(output)
        XCTAssertEqual(output.shape, [1, 4, 1])
        let values = output.asArray(Float.self)
        XCTAssertEqual(values[0], 0, accuracy: 1e-6)
        XCTAssertEqual(values[1], 2.5, accuracy: 1e-6)
        XCTAssertEqual(values[2], 7.5, accuracy: 1e-6)
        XCTAssertEqual(values[3], 10, accuracy: 1e-6)
    }

    func testLinearResizeUsesHalfPixelCoordinatesWhenDownsampling() {
        let input = MLXArray([Float(0), 10, 20, 30], [1, 4, 1])
        let output = miniCPMLinearResize(input, outputLength: 2)
        eval(output)
        XCTAssertEqual(output.shape, [1, 2, 1])
        let values = output.asArray(Float.self)
        XCTAssertEqual(values[0], 5, accuracy: 1e-6)
        XCTAssertEqual(values[1], 25, accuracy: 1e-6)
    }

    func testSTFTAndISTFTRoundTripShape() {
        let input = MLXArray((0 ..< 96).map { Float(sin(Double($0) * 0.2)) }, [1, 96])
        let spectrum = miniCPMSTFT(input, fftSize: 16, hopLength: 4)
        let magnitude = sqrt(spectrum.real * spectrum.real + spectrum.imaginary * spectrum.imaginary)
        let phase = atan2(spectrum.imaginary, spectrum.real)
        let output = miniCPMISTFT(
            magnitude: magnitude, phase: phase, fftSize: 16, hopLength: 4)
        eval(output)
        XCTAssertEqual(output.shape, input.shape)
        XCTAssertLessThan(MLX.abs(output - input).max().item(Float.self), 1e-4)
    }
}

final class MiniCPMSpeechTokenBufferTests: XCTestCase {
    func testTokenBufferCheckpointRestoresAfterCancelledWindow() {
        var buffer = MiniCPMSpeechTokenBuffer(
            chunkTokenCount: 25,
            lookaheadTokenCount: 3,
            minimumForceFlushTokenCount: 5,
            initialTokens: [4_218, 4_218, 4_218])
        let checkpoint = buffer

        // The pipeline owns this value-type checkpoint around Flow/HiFT
        // synthesis.  Simulate a job that has already consumed a stable
        // window, then restore the exact pending look-ahead state.
        _ = buffer.append(
            (0 ..< 28).map(Int32.init), forceFlush: false, isFinal: false)
        XCTAssertNotEqual(buffer.tokens, checkpoint.tokens)
        buffer = checkpoint
        XCTAssertEqual(buffer.tokens, checkpoint.tokens)

        let resumed = buffer.append(
            (100 ..< 128).map(Int32.init), forceFlush: false, isFinal: false)
        XCTAssertEqual(resumed.count, 1)
        XCTAssertEqual(resumed[0].tokens,
                       checkpoint.tokens + (100 ..< 125).map(Int32.init))
        XCTAssertEqual(buffer.tokens, [122, 123, 124, 125, 126, 127])
    }

    func testRegularChunkKeepsThreeLookaheadTokens() {
        var buffer = MiniCPMSpeechTokenBuffer(
            chunkTokenCount: 25,
            lookaheadTokenCount: 3,
            minimumForceFlushTokenCount: 5,
            initialTokens: [4_218, 4_218, 4_218])
        let generated = (0 ..< 25).map(Int32.init)
        let jobs = buffer.append(generated, forceFlush: false, isFinal: false)

        XCTAssertEqual(jobs.count, 1)
        XCTAssertEqual(jobs[0].tokens.count, 28)
        XCTAssertEqual(jobs[0].stableTokenCount, 25)
        XCTAssertFalse(jobs[0].isFinal)
        XCTAssertEqual(buffer.tokens, [22, 23, 24])
    }

    func testForceFlushProcessesShortStablePrefix() {
        var buffer = MiniCPMSpeechTokenBuffer(
            chunkTokenCount: 25,
            lookaheadTokenCount: 3,
            minimumForceFlushTokenCount: 5,
            initialTokens: [4_218, 4_218, 4_218])
        let jobs = buffer.append(
            [10, 11, 12, 13, 14], forceFlush: true, isFinal: false)

        XCTAssertEqual(jobs, [MiniCPMSpeechTokenJob(
            tokens: [4_218, 4_218, 4_218, 10, 11, 12, 13, 14],
            isFinal: false,
            stableTokenCount: 5)])
        XCTAssertEqual(buffer.tokens, [12, 13, 14])
    }

    func testFinalChunkConsumesRemainingLookaheadExactlyOnce() {
        var buffer = MiniCPMSpeechTokenBuffer(
            chunkTokenCount: 25,
            lookaheadTokenCount: 3,
            minimumForceFlushTokenCount: 5)
        let first = buffer.append(
            (0 ..< 28).map(Int32.init), forceFlush: false, isFinal: false)
        let final = buffer.append([28, 29], forceFlush: false, isFinal: true)

        XCTAssertEqual(first[0].tokens, (0 ..< 28).map(Int32.init))
        XCTAssertEqual(final, [MiniCPMSpeechTokenJob(
            tokens: [25, 26, 27, 28, 29],
            isFinal: true,
            stableTokenCount: 5)])
        XCTAssertTrue(buffer.tokens.isEmpty)
    }

    func testLongStoryProducesConsecutiveWindowsBeforeFinalFlush() {
        var buffer = MiniCPMSpeechTokenBuffer(
            chunkTokenCount: 25,
            lookaheadTokenCount: 3,
            minimumForceFlushTokenCount: 5,
            initialTokens: [4_218, 4_218, 4_218])

        // 60 semantic codes must yield two regular 25+3 windows.  The
        // remainder is emitted only by the explicit final flush; it must not
        // make the stream stop after the first sentence/window.
        let jobs = buffer.append(
            (0 ..< 60).map(Int32.init), forceFlush: false, isFinal: true)

        XCTAssertEqual(jobs.count, 3)
        XCTAssertEqual(jobs[0].tokens, [4_218, 4_218, 4_218] + (0 ..< 25).map(Int32.init))
        XCTAssertEqual(jobs[0].stableTokenCount, 25)
        // The first 3 lookahead codes are retained by the first window, so
        // the second window starts at generated code 22 rather than 25.
        XCTAssertEqual(jobs[1].tokens, (22 ..< 50).map(Int32.init))
        XCTAssertEqual(jobs[1].stableTokenCount, 25)
        XCTAssertEqual(jobs[2].tokens, (47 ..< 60).map(Int32.init))
        XCTAssertEqual(jobs[2].stableTokenCount, 13)
        XCTAssertTrue(jobs[2].isFinal)
        XCTAssertTrue(buffer.tokens.isEmpty)
    }
}

final class MiniCPMHiFTCancellationTests: XCTestCase {
    private final class CancellationProbe: @unchecked Sendable {
        private let lock = NSLock()
        private var count = 0
        private let cancelAt: Int

        init(cancelAt: Int) { self.cancelAt = cancelAt }

        func shouldCancel() -> Bool {
            lock.lock()
            defer { lock.unlock() }
            count += 1
            return count >= cancelAt
        }
    }

    private func tinyHiFT() -> MiniCPMHiFTGenerator {
        var configuration = MiniCPMHiFTConfiguration()
        // Keep the trained channel width expected by F0Predictor, but remove
        // upsampling/residual stages so a no-weight construction remains a
        // small deterministic cancellation probe.
        configuration.upsampleRates = []
        configuration.upsampleKernelSizes = []
        configuration.residualKernelSizes = []
        configuration.residualDilations = []
        configuration.sourceResidualKernelSizes = []
        configuration.sourceResidualDilations = []
        return MiniCPMHiFTGenerator(configuration: configuration)
    }

    func testHiFTCancellationRestoresOverlapAndResumes() throws {
        let session = MiniCPMHiFTStreamingSession(
            model: tinyHiFT(),
            melCacheFrames: 1)
        let mel = MLXArray.zeros([1, 80, 4])
        let first = try session.synthesize(mel: mel, isFinal: false)
        eval(first.waveform)
        XCTAssertFalse(first.waveform.asArray(Float.self).isEmpty)
        let melFrames = session.cachedMelFrames
        let sourceSamples = session.cachedSourceSampleCount
        let speechSamples = session.cachedSpeechSampleCount

        // synthesize() checks after model.generate; cancelAt=2 therefore
        // exercises Snapshot.restore after a generated chunk, not merely the
        // cheap entry guard.
        let probe = CancellationProbe(cancelAt: 2)
        XCTAssertThrowsError(try session.synthesize(
            mel: mel,
            isFinal: false,
            shouldCancel: { probe.shouldCancel() })) { error in
            guard case MiniCPMToken2WavError.cancelled = error else {
                return XCTFail("expected HiFT cancellation, got \(error)")
            }
        }
        XCTAssertEqual(session.cachedMelFrames, melFrames)
        XCTAssertEqual(session.cachedSourceSampleCount, sourceSamples)
        XCTAssertEqual(session.cachedSpeechSampleCount, speechSamples)

        // A subsequent chunk must be accepted from the restored overlap
        // state; stale generated speech from the cancelled call is not kept.
        let resumed = try session.synthesize(mel: mel, isFinal: false)
        eval(resumed.waveform)
        let samples = resumed.waveform.asArray(Float.self)
        XCTAssertFalse(samples.isEmpty)
        XCTAssertTrue(samples.allSatisfy(\.isFinite))
    }
}

final class MiniCPMStreamingCacheTrimmingTests: XCTestCase {
    private func attention(length: Int) -> MiniCPMConformerAttentionCache {
        let values = MLXArray((0 ..< length).map(Float.init))
            .reshaped([1, 1, length, 1])
        return MiniCPMConformerAttentionCache(keys: values, values: values)
    }

    func testConformerTrimKeepsPromptAndNewestGeneratedFrames() {
        let cache = MiniCPMUpsampleConformerCache(
            firstAttention: [attention(length: 20)],
            secondAttention: [attention(length: 40)],
            preLookaheadConvolution: MLXArray.zeros([1, 1, 2]),
            upsampleConvolution: MLXArray.zeros([1, 1, 4]))
        let trimmed = cache.trimmed(
            promptInputPositions: 4,
            promptOutputPositions: 8,
            maximumRecentOutputPositions: 6)

        XCTAssertEqual(trimmed.inputPositions, 7)
        XCTAssertEqual(trimmed.outputPositions, 14)
        XCTAssertEqual(
            trimmed.firstAttention[0].keys.asArray(Float.self),
            [0, 1, 2, 3, 17, 18, 19])
        XCTAssertEqual(
            trimmed.secondAttention[0].keys.asArray(Float.self),
            [0, 1, 2, 3, 4, 5, 6, 7, 34, 35, 36, 37, 38, 39])
    }

    func testDiTTrimRespectsNewestFirstCacheOrder() {
        let values = MLXArray((0 ..< 20).map(Float.init))
            .reshaped([1, 1, 20, 1])
        let layer = MiniCPMDiTLayerCache(
            attention: MiniCPMDiTAttentionCache(keys: values, values: values),
            convolution: MiniCPMDiTConvolutionCache(
                first: MLXArray.zeros([1, 1, 2]),
                second: MLXArray.zeros([1, 1, 2])))
        let trimmed = MiniCPMDiTCache(layers: [layer]).trimmed(
            promptPositions: 4, maximumRecentPositions: 6)

        XCTAssertEqual(trimmed.length, 10)
        XCTAssertEqual(
            trimmed.layers[0].attention.keys.asArray(Float.self),
            [0, 1, 2, 3, 4, 5, 16, 17, 18, 19])
    }

    func testFlowTrimPreservesAbsoluteNoiseFrameOffset() {
        let values = MLXArray((0 ..< 20).map(Float.init))
            .reshaped([1, 1, 20, 1])
        let layer = MiniCPMDiTLayerCache(
            attention: MiniCPMDiTAttentionCache(keys: values, values: values),
            convolution: MiniCPMDiTConvolutionCache(
                first: MLXArray.zeros([1, 1, 2]),
                second: MLXArray.zeros([1, 1, 2])))
        let original = MiniCPMFlowMatchingCache(
            odeStepCaches: [MiniCPMDiTCache(layers: [layer])],
            totalFrames: 137)
        let trimmed = original.trimmed(
            promptPositions: 4, maximumRecentPositions: 6)

        XCTAssertEqual(trimmed.length, 10)
        // Noise is indexed by the absolute stream position, not by the
        // post-trim cache length; otherwise long stories repeat the same
        // random frame after every sliding-window trim.
        XCTAssertEqual(trimmed.totalFrames, 137)
    }
}

final class E2EMiniCPMToken2WavTests: XCTestCase {
    private final class CancellationProbe: @unchecked Sendable {
        private let lock = NSLock()
        private var callCount = 0
        private let cancelAt: Int

        init(cancelAt: Int) {
            self.cancelAt = cancelAt
        }

        var calls: Int {
            lock.lock()
            defer { lock.unlock() }
            return callCount
        }

        func shouldCancel() -> Bool {
            lock.lock()
            defer { lock.unlock() }
            callCount += 1
            return callCount >= cancelAt
        }
    }

    private struct RealFixture {
        let pipeline: MiniCPMToken2WavPipeline
        let generatedTokens: [Int32]
    }

    private func loadRealFixture() throws -> RealFixture {
        let environment = ProcessInfo.processInfo.environment
        guard let modelPath = environment["MINICPM_TOKEN2WAV_MLX_PATH"],
              let fixturePath = environment["MINICPM_TOKEN2WAV_FIXTURE"] else {
            throw XCTSkip(
                "set MINICPM_TOKEN2WAV_MLX_PATH and MINICPM_TOKEN2WAV_FIXTURE")
        }
        let arrays = try MLX.loadArrays(
            url: URL(fileURLWithPath: fixturePath))
        guard let promptTokens = arrays["prompt_tokens"],
              let promptMel = arrays["prompt_mel"],
              let speaker = arrays["speaker_embedding"],
              let generated = arrays["generated_tokens"] else {
            XCTFail("incomplete token2wav fixture")
            throw MiniCPMToken2WavError.invalidInput(
                "incomplete token2wav fixture")
        }
        var flowConfiguration = MiniCPMFlowConfiguration()
        // Two steps are the pinned P3-B smoke setting.  Flow/HiFT parity has
        // a separate ten-step production configuration gate.
        flowConfiguration.odeSteps = 2
        let pipeline = try MiniCPMToken2WavPipeline(
            modelDirectory: URL(fileURLWithPath: modelPath, isDirectory: true),
            flowConfiguration: flowConfiguration)
        try pipeline.prepare(prompt: MiniCPMToken2WavPrompt(
            tokens: promptTokens,
            mel: promptMel,
            speakerEmbedding: speaker))
        return RealFixture(
            pipeline: pipeline,
            generatedTokens: generated.asType(.int32)
                .reshaped([-1]).asArray(Int32.self))
    }

    private func stream(
        _ pipeline: MiniCPMToken2WavPipeline,
        tokens: [Int32]
    ) throws -> [MiniCPMToken2WavChunk] {
        let feedPattern = [7, 13, 5, 19, 11]
        var offset = 0
        var feedIndex = 0
        var chunks: [MiniCPMToken2WavChunk] = []
        while offset < tokens.count {
            let count = Swift.min(
                feedPattern[feedIndex % feedPattern.count],
                tokens.count - offset)
            let end = offset + count
            chunks.append(contentsOf: try pipeline.append(
                Array(tokens[offset ..< end]),
                forceFlush: feedIndex % 3 == 2 && end != tokens.count,
                isFinal: end == tokens.count))
            offset = end
            feedIndex += 1
        }
        return chunks
    }

    private func samples(_ chunk: MiniCPMToken2WavChunk) -> [Float] {
        eval(chunk.waveform)
        return chunk.waveform.asType(.float32).asArray(Float.self)
    }

    private func samples(_ chunks: [MiniCPMToken2WavChunk]) -> [Float] {
        chunks.flatMap(samples)
    }

    private func assertStatisticsEqual(
        _ lhs: MiniCPMToken2WavContextStatistics,
        _ rhs: MiniCPMToken2WavContextStatistics,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        XCTAssertEqual(lhs.promptInputPositions, rhs.promptInputPositions,
                       file: file, line: line)
        XCTAssertEqual(lhs.promptOutputPositions, rhs.promptOutputPositions,
                       file: file, line: line)
        XCTAssertEqual(lhs.conformerInputPositions, rhs.conformerInputPositions,
                       file: file, line: line)
        XCTAssertEqual(lhs.conformerOutputPositions, rhs.conformerOutputPositions,
                       file: file, line: line)
        XCTAssertEqual(lhs.decoderPositions, rhs.decoderPositions,
                       file: file, line: line)
        XCTAssertEqual(lhs.totalGeneratedAndPromptFrames,
                       rhs.totalGeneratedAndPromptFrames,
                       file: file, line: line)
        XCTAssertEqual(lhs.pendingTokens, rhs.pendingTokens,
                       file: file, line: line)
        XCTAssertEqual(lhs.vocoderMelFrames, rhs.vocoderMelFrames,
                       file: file, line: line)
        XCTAssertEqual(lhs.vocoderSourceSamples, rhs.vocoderSourceSamples,
                       file: file, line: line)
        XCTAssertEqual(lhs.vocoderSpeechSamples, rhs.vocoderSpeechSamples,
                       file: file, line: line)
    }

    private func assertWaveformsEqual(
        _ lhs: [Float],
        _ rhs: [Float],
        accuracy: Float = 1e-4,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        XCTAssertEqual(lhs.count, rhs.count, file: file, line: line)
        guard lhs.count == rhs.count else { return }
        let maximum = zip(lhs, rhs).map { Swift.abs($0 - $1) }.max() ?? 0
        XCTAssertLessThanOrEqual(maximum, accuracy, file: file, line: line)
    }

    func testRealPipelineEmitsFiniteWaveform() throws {
        let fixture = try loadRealFixture()
        let chunks = try fixture.pipeline.append(
            fixture.generatedTokens, isFinal: true)
        let samples = chunks.flatMap { $0.waveform.asArray(Float.self) }

        XCTAssertFalse(samples.isEmpty)
        XCTAssertTrue(samples.allSatisfy(\.isFinite))
        XCTAssertGreaterThan(samples.map { Swift.abs($0) }.max() ?? 0, 0)
        XCTAssertFalse(fixture.pipeline.isTurnOpen)
    }

    func testRealStreamingFlushSeamAndResetLifecycle() throws {
        let fixture = try loadRealFixture()
        let pipeline = fixture.pipeline
        let tokens = fixture.generatedTokens
        XCTAssertEqual(tokens.count, 80, "pinned smoke fixture token count")

        let prepared = try XCTUnwrap(pipeline.contextStatistics())
        XCTAssertEqual(prepared.pendingTokens, 3)
        XCTAssertEqual(prepared.vocoderMelFrames, 0)
        XCTAssertEqual(prepared.vocoderSourceSamples, 0)
        XCTAssertEqual(prepared.vocoderSpeechSamples, 0)
        let promptStatistics = prepared

        let chunks = try stream(pipeline, tokens: tokens)
        XCTAssertEqual(chunks.count, 4)
        XCTAssertEqual(chunks.map(\.inputTokenCount), [28, 28, 15, 21])
        XCTAssertEqual(chunks.map(\.stableTokenCount), [25, 25, 12, 21])
        XCTAssertEqual(chunks.map(\.melFrameCount), [50, 50, 24, 42])
        XCTAssertEqual(chunks.map(\.isFinal), [false, false, false, true])
        XCTAssertEqual(chunks.dropLast().filter(\.isFinal).count, 0)

        let chunkSamples = chunks.map(samples)
        XCTAssertEqual(chunkSamples.map(\.count), [24_000, 24_000, 11_520, 24_000])
        let waveform = chunkSamples.flatMap { $0 }
        XCTAssertEqual(waveform.count, 83_520)
        XCTAssertTrue(waveform.allSatisfy(\.isFinite))
        XCTAssertGreaterThan(waveform.map { Swift.abs($0) }.max() ?? 0, 0)

        // Every emitted boundary is after the held 160 ms overlap.  A large
        // one-sample jump here indicates a missing crossfade or duplicated
        // tail; the fixed real fixture stays far below this conservative
        // bound.
        var boundary = 0
        for chunk in chunkSamples.dropLast() {
            boundary += chunk.count
            XCTAssertLessThan(
                Swift.abs(waveform[boundary] - waveform[boundary - 1]),
                0.5,
                "streaming seam at sample \(boundary)")
        }

        let finished = try XCTUnwrap(pipeline.contextStatistics())
        XCTAssertFalse(pipeline.isTurnOpen)
        XCTAssertEqual(finished.pendingTokens, 0)
        XCTAssertEqual(finished.vocoderMelFrames, 0)
        XCTAssertEqual(finished.vocoderSourceSamples, 0)
        XCTAssertEqual(finished.vocoderSpeechSamples, 0)
        XCTAssertEqual(finished.decoderPositions, promptStatistics.decoderPositions)
        XCTAssertEqual(
            finished.conformerOutputPositions,
            promptStatistics.conformerOutputPositions)
        XCTAssertThrowsError(try pipeline.append([], isFinal: true)) { error in
            guard case MiniCPMToken2WavError.turnClosed = error else {
                return XCTFail("expected a closed turn after final flush, got \(error)")
            }
        }

        // Resetting a completed turn must restore the immutable prompt cache,
        // the three-token right lookahead prefix, and empty HiFT overlap.
        pipeline.resetForNewTurn()
        XCTAssertTrue(pipeline.isTurnOpen)
        let resetStatistics = try XCTUnwrap(pipeline.contextStatistics())
        XCTAssertEqual(resetStatistics.pendingTokens, 3)
        XCTAssertEqual(resetStatistics.vocoderMelFrames, 0)
        XCTAssertEqual(resetStatistics.vocoderSourceSamples, 0)
        XCTAssertEqual(resetStatistics.vocoderSpeechSamples, 0)
        XCTAssertEqual(resetStatistics.decoderPositions, promptStatistics.decoderPositions)
        XCTAssertEqual(
            resetStatistics.conformerOutputPositions,
            promptStatistics.conformerOutputPositions)

        let replay = try stream(pipeline, tokens: tokens)
        assertWaveformsEqual(waveform, samples(replay))
        XCTAssertEqual(replay.map(\.inputTokenCount), chunks.map(\.inputTokenCount))
        XCTAssertEqual(replay.map(\.stableTokenCount), chunks.map(\.stableTokenCount))
        XCTAssertEqual(replay.map(\.isFinal), chunks.map(\.isFinal))
    }

    func testRealForceFlushShortWindowAndInterruptCancellationRollback() throws {
        let fixture = try loadRealFixture()
        let pipeline = fixture.pipeline
        let tokens = fixture.generatedTokens

        // A semantic boundary may arrive with only five committed codes.  The
        // real Flow/HiFT path must synthesize that short stable prefix while
        // retaining exactly the three-token lookahead.
        let short = try pipeline.append(
            Array(tokens.prefix(5)), forceFlush: true, isFinal: false)
        XCTAssertEqual(short.count, 1)
        XCTAssertEqual(short[0].inputTokenCount, 8)
        XCTAssertEqual(short[0].stableTokenCount, 5)
        XCTAssertFalse(short[0].isFinal)
        XCTAssertEqual(pipeline.pendingTokenCount, 3)
        XCTAssertTrue(samples(short).allSatisfy(\.isFinite))

        // Final flush consumes the retained lookahead once, even when there
        // are no newly decoded semantic codes in this call.
        let final = try pipeline.append([], isFinal: true)
        XCTAssertEqual(final.count, 1)
        XCTAssertTrue(final[0].isFinal)
        XCTAssertEqual(final[0].inputTokenCount, 3)
        XCTAssertEqual(final[0].stableTokenCount, 3)
        XCTAssertEqual(pipeline.pendingTokenCount, 0)
        XCTAssertFalse(pipeline.isTurnOpen)
        XCTAssertTrue(samples(final).allSatisfy(\.isFinite))

        pipeline.resetForNewTurn()
        let baseline = try pipeline.append(
            Array(tokens.prefix(25)), isFinal: false)
        let baselineSamples = samples(baseline)
        XCTAssertEqual(baseline.count, 1)
        XCTAssertEqual(pipeline.pendingTokenCount, 3)

        // Interrupt first, then cancel after Flow and HiFT have had a chance
        // to run.  The retry must be bit-stable with the pre-interrupt window;
        // otherwise stale Flow KV, source overlap, or token-buffer state
        // crossed the epoch boundary.
        pipeline.interruptAndReset()
        let beforeCancel = try XCTUnwrap(pipeline.contextStatistics())
        XCTAssertEqual(beforeCancel.pendingTokens, 3)
        XCTAssertEqual(beforeCancel.vocoderMelFrames, 0)
        XCTAssertEqual(beforeCancel.vocoderSourceSamples, 0)
        XCTAssertEqual(beforeCancel.vocoderSpeechSamples, 0)

        let probe = CancellationProbe(cancelAt: 6)
        XCTAssertThrowsError(try pipeline.append(
            Array(tokens.prefix(25)),
            isFinal: false,
            shouldCancel: { probe.shouldCancel() })) { error in
            guard case MiniCPMToken2WavError.cancelled = error else {
                return XCTFail("expected Token2Wav cancellation, got \(error)")
            }
        }
        XCTAssertGreaterThanOrEqual(probe.calls, 6)
        let afterCancel = try XCTUnwrap(pipeline.contextStatistics())
        assertStatisticsEqual(beforeCancel, afterCancel)
        XCTAssertTrue(pipeline.isTurnOpen)

        let retry = try pipeline.append(
            Array(tokens.prefix(25)), isFinal: false)
        assertWaveformsEqual(baselineSamples, samples(retry))
        XCTAssertEqual(retry[0].inputTokenCount, baseline[0].inputTokenCount)
        XCTAssertEqual(retry[0].stableTokenCount, baseline[0].stableTokenCount)

        // No stale chunk remains queued after an interrupt.  A fresh turn can
        // be closed independently and the next append is rejected rather
        // than accidentally continuing the cancelled epoch.
        pipeline.interruptAndReset()
        let replacement = try pipeline.append(
            Array(tokens.suffix(5)), forceFlush: true, isFinal: true)
        XCTAssertFalse(replacement.isEmpty)
        XCTAssertTrue(replacement.last?.isFinal == true)
        XCTAssertFalse(pipeline.isTurnOpen)
        XCTAssertThrowsError(try pipeline.append([tokens[0]])) { error in
            guard case MiniCPMToken2WavError.turnClosed = error else {
                return XCTFail("expected replacement turn to be closed, got \(error)")
            }
        }
    }

    func testRealWindowedBridgeCarriesFlowAndHiFTState() throws {
        let fixture = try loadRealFixture()
        let pipeline = fixture.pipeline
        let tokens = fixture.generatedTokens
        let windows: [([Int32], Bool)] = [
            ([Int32](repeating: 4_218, count: 3) + Array(tokens[0 ..< 25]), false),
            (Array(tokens[22 ..< 50]), false),
            (Array(tokens[47 ..< 62]), false),
            (Array(tokens[59 ..< 80]), true),
        ]

        let chunks = try windows.map { window, isFinal in
            try pipeline.synthesizeWindow(window, isFinal: isFinal)
        }
        XCTAssertEqual(chunks.map(\.inputTokenCount), [28, 28, 15, 21])
        XCTAssertEqual(chunks.map(\.stableTokenCount), [25, 25, 12, 21])
        XCTAssertEqual(chunks.map(\.melFrameCount), [50, 50, 24, 42])
        XCTAssertEqual(chunks.map(\.isFinal), [false, false, false, true])
        XCTAssertEqual(chunks.map { samples($0).count },
                       [24_000, 24_000, 11_520, 24_000])
        XCTAssertTrue(samples(chunks).allSatisfy(\.isFinite))
        XCTAssertFalse(pipeline.isTurnOpen)

        // synthesizeWindow deliberately leaves token-window ownership with
        // the caller; its Flow/HiFT caches are nevertheless reset after the
        // final window and the next interrupt starts from the prompt cache.
        let statistics = try XCTUnwrap(pipeline.contextStatistics())
        XCTAssertEqual(statistics.decoderPositions, 302)
        XCTAssertEqual(statistics.conformerOutputPositions, 302)
        XCTAssertEqual(statistics.vocoderMelFrames, 0)
        XCTAssertEqual(statistics.vocoderSourceSamples, 0)
        XCTAssertEqual(statistics.vocoderSpeechSamples, 0)
        pipeline.interruptAndReset()
        XCTAssertTrue(pipeline.isTurnOpen)
        let reset = try XCTUnwrap(pipeline.contextStatistics())
        XCTAssertEqual(reset.decoderPositions, 302)
        XCTAssertEqual(reset.conformerOutputPositions, 302)
        XCTAssertEqual(reset.vocoderMelFrames, 0)
        XCTAssertEqual(reset.vocoderSourceSamples, 0)
        XCTAssertEqual(reset.vocoderSpeechSamples, 0)
    }
}
