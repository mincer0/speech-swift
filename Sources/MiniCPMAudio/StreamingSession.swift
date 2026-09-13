import MLX

public struct MiniCPMStreamingAudioEncoding {
    public let output: MiniCPMAudioEncoding
    public let didResetCache: Bool

    public init(output: MiniCPMAudioEncoding, didResetCache: Bool) {
        self.output = output
        self.didResetCache = didResetCache
    }
}

/// Owns MiniCPM's mel and Whisper KV state for one duplex conversation.
public final class MiniCPMAudioStreamingSession {
    /// Exact mutable state needed to resume a streaming encoder without
    /// replaying the conversation waveform.  `MiniCPMAudioCache` is immutable
    /// (its key/value arrays are `let`), so retaining the reference is a cheap
    /// snapshot; the mel processor snapshot only copies its bounded overlap
    /// ring and normalisation scalars.
    public struct Snapshot {
        public let cache: MiniCPMAudioCache?
        public let melProcessor: MiniCPMStreamingMelProcessor.Snapshot
        public let cacheResetCount: Int

        public init(
            cache: MiniCPMAudioCache?,
            melProcessor: MiniCPMStreamingMelProcessor.Snapshot,
            cacheResetCount: Int
        ) {
            self.cache = cache
            self.melProcessor = melProcessor
            self.cacheResetCount = cacheResetCount
        }
    }

    public let model: MiniCPMAudioModel
    public let melProcessor: MiniCPMStreamingMelProcessor
    public private(set) var cache: MiniCPMAudioCache?
    public private(set) var cacheResetCount = 0

    public init(
        model: MiniCPMAudioModel,
        melProcessor: MiniCPMStreamingMelProcessor = .init()
    ) {
        self.model = model
        self.melProcessor = melProcessor
    }

    /// Capture the streaming frontend and encoder boundary.  This is
    /// intentionally a metadata/reference operation: no committed PCM or KV
    /// tensor is copied.
    public func snapshot() -> Snapshot {
        Snapshot(
            cache: cache,
            melProcessor: melProcessor.snapshot(),
            cacheResetCount: cacheResetCount)
    }

    /// Restore a checkpoint produced by ``snapshot()``.  The restored cache is
    /// the exact immutable object used at the checkpoint, so a rollback does
    /// not need to rebuild the encoder by re-encoding old audio.
    public func restore(_ snapshot: Snapshot) {
        cache = snapshot.cache
        melProcessor.restore(snapshot.melProcessor)
        cacheResetCount = snapshot.cacheResetCount
    }

    public func reset() {
        cache = nil
        cacheResetCount = 0
        melProcessor.reset()
    }

    public func releaseEncoderContext() {
        cache = nil
        cacheResetCount += 1
        Memory.clearCache()
    }

    public func acceptAudio(
        _ samples: [Float],
        isFinal: Bool = false
    ) throws -> MiniCPMStreamingAudioEncoding? {
        let isFirstChunk = melProcessor.emittedFrameCount == 0
        guard let features = try melProcessor.accept(samples, isFinal: isFinal) else {
            return nil
        }
        // The upstream duplex path marks every emitted mel chunk as carrying
        // two frames of CNN context on the right.  On the first chunk there
        // is no left context to discard; each later chunk also carries two
        // redundant frames on the left.  Keep this decision at the PCM entry
        // point because callers of ``encodeFeatures`` may intentionally pass
        // ordinary (non-context) chunks with its 0/0 defaults.
        //
        // ``MiniCPMAudioModel`` performs the actual post-convolution slice
        // exactly once.  The streaming wrapper only supplies the metadata;
        // its cache-length accounting below uses the same geometry without
        // slicing the feature tensor a second time.
        let prefixExtraFrames = isFirstChunk ? 0 : 2
        let array = MLXArray(
            features.data,
            [1, features.timeFrames, features.melBins]
        )
        return try encodeFeatures(
            array,
            prefixExtraFrames: prefixExtraFrames,
            suffixExtraFrames: 2
        )
    }

    /// Encode a complete, independent waveform without touching this
    /// session's streaming mel/KV state.
    ///
    /// MiniCPM uses the same 16 kHz, 80-bin Whisper frontend for both paths,
    /// but a reference/prompt clip must not become part of the live turn's
    /// left-context cache.  This method therefore constructs a fresh
    /// extractor and calls the encoder without a cache.  The returned wrapper
    /// deliberately has `didResetCache == false`; the live session remains
    /// byte-for-byte unchanged and can continue accepting audio afterwards.
    public func encodeOfflineAudio(
        _ samples: [Float]
    ) throws -> MiniCPMStreamingAudioEncoding {
        guard !samples.isEmpty else {
            throw MiniCPMAudioError.invalidFeatures(
                "offline audio must not be empty")
        }
        let extractor = MiniCPMWhisperFeatureExtractor()
        let features = try extractor.extract(samples)
        let array = MLXArray(
            features.data,
            [1, features.timeFrames, features.melBins]
        )
        let output = try model.encode(
            features: array,
            cache: nil,
            useCache: false
        )
        return MiniCPMStreamingAudioEncoding(output: output, didResetCache: false)
    }

    /// Alias used by duplex callers when the independent clip is a voice or
    /// system prompt.  Keeping this as a separate named entry point makes the
    /// cache-isolation contract explicit at call sites.
    public func encodeReferenceAudio(
        _ samples: [Float]
    ) throws -> MiniCPMStreamingAudioEncoding {
        try encodeOfflineAudio(samples)
    }

    public func encodeFeatures(
        _ features: MLXArray,
        prefixExtraFrames: Int = 0,
        suffixExtraFrames: Int = 0
    ) throws -> MiniCPMStreamingAudioEncoding {
        let prefixRemove = prefixExtraFrames > 0
            ? (prefixExtraFrames + 1) / 2 : 0
        let suffixRemove = suffixExtraFrames > 0
            ? (suffixExtraFrames + 1) / 2 : 0
        let incomingPositions = (features.dim(1) - 1) / 2 + 1
            - prefixRemove - suffixRemove
        guard incomingPositions > 0 else {
            throw MiniCPMAudioError.invalidFeatures(
                "CNN context removal consumed the entire streaming chunk")
        }
        var didReset = false
        if let cache,
           cache.length + incomingPositions
                >= model.configuration.maximumSourcePositions {
            self.cache = nil
            cacheResetCount += 1
            didReset = true
            Memory.clearCache()
        }
        let output = try model.encode(
            features: features,
            cache: cache,
            useCache: true,
            prefixExtraFrames: prefixExtraFrames,
            suffixExtraFrames: suffixExtraFrames
        )
        cache = output.cache
        return MiniCPMStreamingAudioEncoding(
            output: output,
            didResetCache: didReset
        )
    }
}
