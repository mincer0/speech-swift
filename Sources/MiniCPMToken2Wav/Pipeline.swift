import Foundation
import MLX

public struct MiniCPMToken2WavPipelineConfiguration: Sendable {
    public var chunkTokenCount = 25
    public var initialSilenceTokenCount = 3
    public var silenceToken: Int32 = 4_218
    public var minimumForceFlushTokenCount = 5
    public var clearMetalCacheAtEndOfTurn = true

    public init() {}
}

public struct MiniCPMToken2WavPrompt {
    public let tokens: MLXArray
    /// Prompt mel in either [batch, 80, frames] or [batch, frames, 80].
    public let mel: MLXArray
    public let speakerEmbedding: MLXArray

    public init(tokens: MLXArray, mel: MLXArray, speakerEmbedding: MLXArray) {
        self.tokens = tokens
        self.mel = mel
        self.speakerEmbedding = speakerEmbedding
    }
}

public struct MiniCPMToken2WavChunk {
    public let waveform: MLXArray
    public let melFrameCount: Int
    public let inputTokenCount: Int
    public let stableTokenCount: Int
    public let isFinal: Bool

    public init(
        waveform: MLXArray,
        melFrameCount: Int,
        inputTokenCount: Int,
        stableTokenCount: Int,
        isFinal: Bool
    ) {
        self.waveform = waveform
        self.melFrameCount = melFrameCount
        self.inputTokenCount = inputTokenCount
        self.stableTokenCount = stableTokenCount
        self.isFinal = isFinal
    }
}

public struct MiniCPMToken2WavContextStatistics: Sendable, Encodable {
    public let promptInputPositions: Int
    public let promptOutputPositions: Int
    public let conformerInputPositions: Int
    public let conformerOutputPositions: Int
    public let decoderPositions: Int
    public let totalGeneratedAndPromptFrames: Int
    public let pendingTokens: Int
    public let vocoderMelFrames: Int
    public let vocoderSourceSamples: Int
    public let vocoderSpeechSamples: Int
}

struct MiniCPMSpeechTokenJob: Equatable {
    let tokens: [Int32]
    let isFinal: Bool
    let stableTokenCount: Int
}

/// Pure token-window state machine shared by the native pipeline and tests.
struct MiniCPMSpeechTokenBuffer {
    let chunkTokenCount: Int
    let lookaheadTokenCount: Int
    let minimumForceFlushTokenCount: Int
    private(set) var tokens: [Int32]

    init(
        chunkTokenCount: Int,
        lookaheadTokenCount: Int,
        minimumForceFlushTokenCount: Int,
        initialTokens: [Int32] = []
    ) {
        precondition(chunkTokenCount > 0)
        precondition(lookaheadTokenCount >= 0)
        precondition(minimumForceFlushTokenCount > 0)
        self.chunkTokenCount = chunkTokenCount
        self.lookaheadTokenCount = lookaheadTokenCount
        self.minimumForceFlushTokenCount = minimumForceFlushTokenCount
        tokens = initialTokens
    }

    mutating func append(
        _ newTokens: [Int32],
        forceFlush: Bool,
        isFinal: Bool
    ) -> [MiniCPMSpeechTokenJob] {
        tokens.append(contentsOf: newTokens)
        var jobs: [MiniCPMSpeechTokenJob] = []
        let fullWindow = chunkTokenCount + lookaheadTokenCount

        if forceFlush {
            let minimumWindow = lookaheadTokenCount + minimumForceFlushTokenCount
            while tokens.count >= minimumWindow {
                let inputCount = Swift.min(fullWindow, tokens.count)
                let stableCount = Swift.min(
                    chunkTokenCount, inputCount - lookaheadTokenCount)
                jobs.append(MiniCPMSpeechTokenJob(
                    tokens: Array(tokens.prefix(inputCount)),
                    isFinal: false,
                    stableTokenCount: stableCount))
                tokens.removeFirst(stableCount)
            }
        } else {
            while tokens.count >= fullWindow {
                jobs.append(MiniCPMSpeechTokenJob(
                    tokens: Array(tokens.prefix(fullWindow)),
                    isFinal: false,
                    stableTokenCount: chunkTokenCount))
                tokens.removeFirst(chunkTokenCount)
            }
        }

        if isFinal, !tokens.isEmpty {
            jobs.append(MiniCPMSpeechTokenJob(
                tokens: tokens,
                isFinal: true,
                stableTokenCount: tokens.count))
            tokens.removeAll(keepingCapacity: true)
        }
        return jobs
    }
}

public struct MiniCPMToken2WavStageTiming: Sendable, Equatable {
    public let flowMS: Double
    public let hiftMS: Double

    public init(flowMS: Double, hiftMS: Double) {
        self.flowMS = max(0, flowMS)
        self.hiftMS = max(0, hiftMS)
    }

    fileprivate func adding(_ other: Self) -> Self {
        Self(flowMS: flowMS + other.flowMS, hiftMS: hiftMS + other.hiftMS)
    }
}

/// Stateful MiniCPM-o 4.5 speech-token to 24 kHz waveform pipeline.
///
/// The class is intentionally single-threaded. A caller should serialize
/// `append` calls (an actor is a natural owner in an AVAudioEngine runtime).
public final class MiniCPMToken2WavPipeline {
    public let flow: MiniCPMToken2WavFlowModel
    public let vocoder: MiniCPMHiFTStreamingSession
    public let configuration: MiniCPMToken2WavPipelineConfiguration

    private var baseFlowCache: MiniCPMToken2WavFlowCache?
    private var flowCache: MiniCPMToken2WavFlowCache?
    private var speakerEmbedding: MLXArray?
    private var tokenBuffer: MiniCPMSpeechTokenBuffer
    private var turnOpen = false
    private var stageTimingEnabled = false
    public private(set) var lastStageTiming: MiniCPMToken2WavStageTiming?

    public init(
        flow: MiniCPMToken2WavFlowModel,
        hift: MiniCPMHiFTGenerator,
        configuration: MiniCPMToken2WavPipelineConfiguration = .init()
    ) {
        precondition(
            configuration.initialSilenceTokenCount
                == flow.configuration.preLookahead,
            "initial silence prefix must match the Flow pre-lookahead")
        self.flow = flow
        vocoder = MiniCPMHiFTStreamingSession(model: hift)
        self.configuration = configuration
        tokenBuffer = MiniCPMSpeechTokenBuffer(
            chunkTokenCount: configuration.chunkTokenCount,
            lookaheadTokenCount: flow.configuration.preLookahead,
            minimumForceFlushTokenCount:
                configuration.minimumForceFlushTokenCount)
    }

    public convenience init(
        modelDirectory: URL,
        flowConfiguration: MiniCPMFlowConfiguration = .init(),
        pipelineConfiguration: MiniCPMToken2WavPipelineConfiguration = .init()
    ) throws {
        let flow = try MiniCPMToken2WavFlowModel.fromDirectory(
            modelDirectory, configuration: flowConfiguration)
        let hift = try MiniCPMHiFTGenerator.fromDirectory(modelDirectory)
        flow.train(false)
        hift.train(false)
        self.init(
            flow: flow,
            hift: hift,
            configuration: pipelineConfiguration)
    }

    public var isPrepared: Bool { baseFlowCache != nil && speakerEmbedding != nil }
    public var isTurnOpen: Bool { turnOpen }
    public var pendingTokenCount: Int { tokenBuffer.tokens.count }

    /// Enables synchronization at the Flow/HiFT boundary for one-shot
    /// profiling. Leave disabled in production so MLX can preserve its normal
    /// lazy scheduling across both stages.
    public func setStageTimingEnabled(_ enabled: Bool) {
        stageTimingEnabled = enabled
        lastStageTiming = nil
    }

    /// Extract and install a voice prompt directly from raw mono PCM.  This
    /// keeps the runtime-facing API independent of the old Python/Torch
    /// exporter.  A converted S3Tokenizer-v2/CAM++ implementation can be
    /// passed through `encoder`; during the migration callers may inject the
    /// two model outputs explicitly while the mel front-end remains native.
    @discardableResult
    public func preparePromptAudio(
        pcm: [Float],
        sampleRate: Int,
        encoder: (any MiniCPMToken2WavPromptEncoder)? = nil,
        promptTokens: MLXArray? = nil,
        speakerEmbedding: MLXArray? = nil
    ) throws -> MiniCPMToken2WavPromptProfile {
        let profile = try MiniCPMToken2WavPromptPreparer().prepare(
            pcm: pcm,
            sampleRate: sampleRate,
            encoder: encoder,
            promptTokens: promptTokens,
            speakerEmbedding: speakerEmbedding)
        try prepare(prompt: profile.prompt)
        return profile
    }

    public func prepare(prompt: MiniCPMToken2WavPrompt) throws {
        guard prompt.tokens.ndim == 2,
              prompt.tokens.dim(0) == 1,
              prompt.tokens.dim(1) > 0 else {
            throw MiniCPMToken2WavError.invalidInput(
                "prompt tokens must have shape [1, tokenCount]")
        }
        guard prompt.mel.ndim == 3, prompt.mel.dim(0) == 1 else {
            throw MiniCPMToken2WavError.invalidInput(
                "prompt mel must have rank 3 and batch size 1")
        }
        let mel: MLXArray
        if prompt.mel.dim(1) == flow.configuration.melDimension {
            mel = prompt.mel
        } else if prompt.mel.dim(2) == flow.configuration.melDimension {
            mel = prompt.mel.transposed(0, 2, 1)
        } else {
            throw MiniCPMToken2WavError.invalidInput(
                "prompt mel must contain 80 mel channels")
        }
        guard mel.dim(2) == prompt.tokens.dim(1) * flow.configuration.upsampleStride else {
            throw MiniCPMToken2WavError.invalidInput(
                "prompt mel frames must equal prompt token count times 2")
        }
        guard prompt.speakerEmbedding.ndim == 2,
              prompt.speakerEmbedding.shape == [
                  1, flow.configuration.speakerEmbeddingDimension,
              ] else {
            throw MiniCPMToken2WavError.invalidInput(
                "speaker embedding must have shape [1, 192]")
        }
        try validate(tokens: prompt.tokens)

        let prepared = flow.prepareStreaming(
            promptTokens: prompt.tokens.asType(.int32),
            promptMel: mel.asType(.float32),
            speakerEmbedding: prompt.speakerEmbedding.asType(.float32),
            silenceToken: configuration.silenceToken)
        eval(
            prepared.conformer.firstAttention[0].keys,
            prepared.conformer.secondAttention[0].keys,
            prepared.decoder.odeStepCaches[0].layers[0].attention.keys)
        baseFlowCache = prepared
        speakerEmbedding = prompt.speakerEmbedding.asType(.float32)
        resetForNewTurn()
    }

    public func resetForNewTurn() {
        flowCache = baseFlowCache
        vocoder.reset()
        tokenBuffer = MiniCPMSpeechTokenBuffer(
            chunkTokenCount: configuration.chunkTokenCount,
            lookaheadTokenCount: flow.configuration.preLookahead,
            minimumForceFlushTokenCount:
                configuration.minimumForceFlushTokenCount,
            initialTokens: [Int32](
                repeating: configuration.silenceToken,
                count: configuration.initialSilenceTokenCount))
        turnOpen = isPrepared
    }

    /// Drops the current generated context and begins a fresh prompt-based turn.
    public func interruptAndReset() {
        resetForNewTurn()
        Memory.clearCache()
    }

    /// Releases both generated and persistent prompt state.
    public func releaseContext() {
        baseFlowCache = nil
        flowCache = nil
        speakerEmbedding = nil
        vocoder.reset()
        tokenBuffer = MiniCPMSpeechTokenBuffer(
            chunkTokenCount: configuration.chunkTokenCount,
            lookaheadTokenCount: flow.configuration.preLookahead,
            minimumForceFlushTokenCount:
                configuration.minimumForceFlushTokenCount)
        turnOpen = false
        Memory.clearCache()
    }

    /// Adds newly decoded speech tokens and returns every waveform chunk that
    /// became stable. Set `forceFlush` at semantic chunk boundaries and
    /// `isFinal` when the spoken turn ends.
    public func append(
        _ tokens: [Int32],
        forceFlush: Bool = false,
        isFinal: Bool = false,
        shouldCancel: (@Sendable () -> Bool)? = nil
    ) throws -> [MiniCPMToken2WavChunk] {
        lastStageTiming = nil
        guard let speakerEmbedding, var currentCache = flowCache else {
            throw MiniCPMToken2WavError.pipelineNotPrepared
        }
        guard turnOpen else { throw MiniCPMToken2WavError.turnClosed }
        try validate(tokens: tokens)
        try checkCancellation(shouldCancel)

        // append mutates three independent streaming states. Keep a single
        // transaction checkpoint so cancellation after a partial Flow/HiFT
        // job cannot leave a half-consumed token window behind.
        let tokenBufferSnapshot = tokenBuffer
        let flowCacheSnapshot = flowCache
        let vocoderSnapshot = vocoder.snapshot()
        do {
            let jobs = tokenBuffer.append(
                tokens, forceFlush: forceFlush, isFinal: isFinal)
            var chunks: [MiniCPMToken2WavChunk] = []
            var timing = MiniCPMToken2WavStageTiming(flowMS: 0, hiftMS: 0)
            chunks.reserveCapacity(jobs.count)
            for job in jobs {
                try checkCancellation(shouldCancel)
                let result = try synthesize(
                    job: job,
                    speakerEmbedding: speakerEmbedding,
                    cache: currentCache,
                    shouldCancel: shouldCancel)
                currentCache = result.cache
                chunks.append(result.chunk)
                if let stage = result.timing {
                    timing = timing.adding(stage)
                }
            }
            try checkCancellation(shouldCancel)
            flowCache = currentCache

            if isFinal {
                finishTurn()
            }
            if stageTimingEnabled { lastStageTiming = timing }
            return chunks
        } catch MiniCPMToken2WavError.cancelled {
            tokenBuffer = tokenBufferSnapshot
            flowCache = flowCacheSnapshot
            vocoder.restore(vocoderSnapshot)
            throw MiniCPMToken2WavError.cancelled
        }
    }

    /// Processes one already-windowed upstream chunk. Use this bridge API
    /// when an external controller owns the 25+3 token buffer; this pipeline
    /// then owns only Flow/HiFT and their caches.
    public func synthesizeWindow(
        _ tokens: [Int32],
        isFinal: Bool = false,
        shouldCancel: (@Sendable () -> Bool)? = nil
    ) throws -> MiniCPMToken2WavChunk {
        lastStageTiming = nil
        guard let speakerEmbedding, let currentCache = flowCache else {
            throw MiniCPMToken2WavError.pipelineNotPrepared
        }
        guard turnOpen else { throw MiniCPMToken2WavError.turnClosed }
        guard !tokens.isEmpty else {
            throw MiniCPMToken2WavError.invalidInput(
                "a streaming token window cannot be empty")
        }
        let lookahead = flow.configuration.preLookahead
        guard isFinal || tokens.count > lookahead else {
            throw MiniCPMToken2WavError.invalidInput(
                "a non-final token window must contain stable tokens plus lookahead")
        }
        try validate(tokens: tokens)
        try checkCancellation(shouldCancel)
        let flowCacheSnapshot = flowCache
        let vocoderSnapshot = vocoder.snapshot()
        do {
            let job = MiniCPMSpeechTokenJob(
                tokens: tokens,
                isFinal: isFinal,
                stableTokenCount: isFinal ? tokens.count : tokens.count - lookahead)
            let result = try synthesize(
                job: job,
                speakerEmbedding: speakerEmbedding,
                cache: currentCache,
                shouldCancel: shouldCancel)
            try checkCancellation(shouldCancel)
            flowCache = result.cache
            if isFinal { finishTurn() }
            lastStageTiming = result.timing
            return result.chunk
        } catch MiniCPMToken2WavError.cancelled {
            flowCache = flowCacheSnapshot
            vocoder.restore(vocoderSnapshot)
            throw MiniCPMToken2WavError.cancelled
        }
    }

    public func contextStatistics() -> MiniCPMToken2WavContextStatistics? {
        guard let cache = flowCache else { return nil }
        return MiniCPMToken2WavContextStatistics(
            promptInputPositions: cache.promptInputPositions,
            promptOutputPositions: cache.promptOutputPositions,
            conformerInputPositions: cache.conformer.inputPositions,
            conformerOutputPositions: cache.conformer.outputPositions,
            decoderPositions: cache.decoder.length,
            totalGeneratedAndPromptFrames: cache.decoder.totalFrames,
            pendingTokens: tokenBuffer.tokens.count,
            vocoderMelFrames: vocoder.cachedMelFrames,
            vocoderSourceSamples: vocoder.cachedSourceSampleCount,
            vocoderSpeechSamples: vocoder.cachedSpeechSampleCount)
    }

    private func validate(tokens: [Int32]) throws {
        if let invalid = tokens.first(where: {
            $0 < 0 || Int($0) >= flow.configuration.tokenVocabularySize
        }) {
            throw MiniCPMToken2WavError.invalidInput(
                "speech token \(invalid) is outside 0..<\(flow.configuration.tokenVocabularySize)")
        }
    }

    private func validate(tokens: MLXArray) throws {
        let minimumToken = tokens.min().item(Int32.self)
        let maximumToken = tokens.max().item(Int32.self)
        if minimumToken < 0 || Int(maximumToken) >= flow.configuration.tokenVocabularySize {
            throw MiniCPMToken2WavError.invalidInput(
                "prompt speech tokens are outside 0..<\(flow.configuration.tokenVocabularySize)")
        }
    }

    private func synthesize(
        job: MiniCPMSpeechTokenJob,
        speakerEmbedding: MLXArray,
        cache: MiniCPMToken2WavFlowCache,
        shouldCancel: (@Sendable () -> Bool)? = nil
    ) throws -> (
        chunk: MiniCPMToken2WavChunk,
        cache: MiniCPMToken2WavFlowCache,
        timing: MiniCPMToken2WavStageTiming?
    ) {
        try checkCancellation(shouldCancel)
        let tokenArray = MLXArray(job.tokens).reshaped([1, job.tokens.count])
        let flowStart = DispatchTime.now().uptimeNanoseconds
        let flowOutput = flow.generateStreaming(
            tokens: tokenArray,
            speakerEmbedding: speakerEmbedding,
            cache: cache,
            isFinal: job.isFinal)
        let flowMS: Double
        if stageTimingEnabled {
            eval(flowOutput.mel)
            flowMS = elapsedMS(since: flowStart)
        } else {
            flowMS = 0
        }
        try checkCancellation(shouldCancel)
        guard let nextCache = flowOutput.cache else {
            throw MiniCPMToken2WavError.invalidInput(
                "streaming flow did not return a cache")
        }
        let hiftStart = DispatchTime.now().uptimeNanoseconds
        let vocoderOutput = try vocoder.synthesize(
            mel: flowOutput.mel,
            isFinal: job.isFinal,
            shouldCancel: shouldCancel)
        let hiftMS: Double
        if stageTimingEnabled {
            eval(vocoderOutput.waveform)
            hiftMS = elapsedMS(since: hiftStart)
        } else {
            eval(flowOutput.mel, vocoderOutput.waveform)
            hiftMS = 0
        }
        try checkCancellation(shouldCancel)
        return (
            MiniCPMToken2WavChunk(
                waveform: vocoderOutput.waveform,
                melFrameCount: flowOutput.mel.dim(2),
                inputTokenCount: job.tokens.count,
                stableTokenCount: job.stableTokenCount,
                isFinal: job.isFinal),
            nextCache,
            stageTimingEnabled
                ? MiniCPMToken2WavStageTiming(flowMS: flowMS, hiftMS: hiftMS)
                : nil)
    }

    private func checkCancellation(_ shouldCancel: (@Sendable () -> Bool)?) throws {
        if shouldCancel?() == true {
            throw MiniCPMToken2WavError.cancelled
        }
    }

    private func elapsedMS(since start: UInt64) -> Double {
        Double(DispatchTime.now().uptimeNanoseconds - start) / 1_000_000.0
    }

    private func finishTurn() {
        // Returned arrays are materialized before this point, so generated
        // KV/CNN and HiFT overlap state can be released immediately.
        flowCache = baseFlowCache
        vocoder.reset()
        turnOpen = false
        if configuration.clearMetalCacheAtEndOfTurn {
            Memory.clearCache()
        }
    }
}
