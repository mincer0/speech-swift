import Foundation
import MLX

public struct MiniCPMStreamingVocoderOutput {
    public let waveform: MLXArray
    public let fullWaveform: MLXArray

    public init(waveform: MLXArray, fullWaveform: MLXArray) {
        self.waveform = waveform
        self.fullWaveform = fullWaveform
    }
}

/// Stateful 160 ms overlap/crossfade wrapper used by MiniCPM's token2wav path.
public final class MiniCPMHiFTStreamingSession {
    /// Rollback checkpoint for one streaming vocoder job. MLX arrays are
    /// immutable through this wrapper, so retaining the references avoids a
    /// full overlap-buffer copy.
    public struct Snapshot {
        fileprivate let melCache: MLXArray?
        fileprivate let sourceCache: MLXArray?
        fileprivate let speechCache: MLXArray?

        fileprivate init(
            melCache: MLXArray?,
            sourceCache: MLXArray?,
            speechCache: MLXArray?
        ) {
            self.melCache = melCache
            self.sourceCache = sourceCache
            self.speechCache = speechCache
        }
    }

    public let model: MiniCPMHiFTGenerator
    public let melCacheFrames: Int
    public let sourceCacheSamples: Int
    private var melCache: MLXArray?
    private var sourceCache: MLXArray?
    private var speechCache: MLXArray?

    public init(model: MiniCPMHiFTGenerator, melCacheFrames: Int = 8) {
        self.model = model
        self.melCacheFrames = melCacheFrames
        // One mel frame expands by the model's actual hop/upsample product.
        // The bundled CosyVoice2 configuration is 8*5*3*4 = 480 samples,
        // but deriving this from the loaded config keeps overlap/reset
        // correct for parity fixtures and smaller test configurations.
        sourceCacheSamples = melCacheFrames * model.configuration.sourceUpsampleScale
    }

    public func reset() {
        melCache = nil
        sourceCache = nil
        speechCache = nil
    }

    public func snapshot() -> Snapshot {
        Snapshot(
            melCache: melCache,
            sourceCache: sourceCache,
            speechCache: speechCache)
    }

    public func restore(_ snapshot: Snapshot) {
        melCache = snapshot.melCache
        sourceCache = snapshot.sourceCache
        speechCache = snapshot.speechCache
    }

    public var cachedMelFrames: Int { melCache?.dim(2) ?? 0 }
    public var cachedSourceSampleCount: Int { sourceCache?.dim(2) ?? 0 }
    public var cachedSpeechSampleCount: Int { speechCache?.dim(1) ?? 0 }

    private func crossfade(_ speech: MLXArray, previous: MLXArray) -> MLXArray {
        let overlap = Swift.min(sourceCacheSamples, Swift.min(speech.dim(1), previous.dim(1)))
        guard overlap > 0 else { return speech }
        let windowLength = overlap * 2
        let hamming = (0 ..< windowLength).map { index in
            Float(0.54 - 0.46 * Foundation.cos(
                2 * Double.pi * Double(index) / Double(windowLength - 1)))
        }
        let fadeIn = MLXArray(Array(hamming[0 ..< overlap])).reshaped([1, overlap])
        let fadeOut = MLXArray(Array(hamming[overlap...])).reshaped([1, overlap])
        let blended = speech[0..., 0 ..< overlap] * fadeIn
            + previous[0..., (previous.dim(1) - overlap)...] * fadeOut
        return concatenated([blended, speech[0..., overlap...]], axis: 1)
    }

    public func synthesize(
        mel chunk: MLXArray,
        isFinal: Bool = false,
        shouldCancel: (@Sendable () -> Bool)? = nil
    ) throws -> MiniCPMStreamingVocoderOutput {
        if shouldCancel?() == true { throw MiniCPMToken2WavError.cancelled }
        let checkpoint = snapshot()
        let mel = melCache.map { concatenated([$0, chunk], axis: 2) } ?? chunk
        let generated = model.generate(
            mel: mel, cachedSourcePrefix: sourceCache)
        if shouldCancel?() == true {
            restore(checkpoint)
            throw MiniCPMToken2WavError.cancelled
        }
        var speech = generated.waveform
        let isFirst = speechCache == nil
        if let speechCache {
            speech = crossfade(speech, previous: speechCache)
        }

        melCache = mel[0..., 0..., Swift.max(0, mel.dim(2) - melCacheFrames)...]
        sourceCache = generated.source[
            0..., 0..., Swift.max(0, generated.source.dim(2) - sourceCacheSamples)...]
        speechCache = speech[
            0..., Swift.max(0, speech.dim(1) - sourceCacheSamples)...]

        var emitted = speech
        if !isFinal {
            // Match the upstream short-chunk contract as well as the normal
            // 1-second path: always hold one overlap window for the next
            // chunk.  If a tiny force-flush produces fewer than 3,840
            // samples, the stable prefix is empty and the first emission is
            // exactly one overlap-sized silence block (rather than leaking
            // an un-crossfaded partial waveform).
            let stableLength = Swift.max(0, speech.dim(1) - sourceCacheSamples)
            let stable = speech[0..., 0 ..< stableLength]
            emitted = stable
            if isFirst {
                emitted = concatenated([
                    MLXArray.zeros([speech.dim(0), sourceCacheSamples], dtype: speech.dtype),
                    stable,
                ], axis: 1)
            }
        }
        if shouldCancel?() == true {
            restore(checkpoint)
            throw MiniCPMToken2WavError.cancelled
        }
        return MiniCPMStreamingVocoderOutput(
            waveform: emitted, fullWaveform: speech)
    }
}
