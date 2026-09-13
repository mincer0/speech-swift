import Accelerate
import Foundation

public struct MiniCPMLogMelFeatures: Equatable, Sendable {
    /// Row-major `[timeFrames, melBins]`, ready to construct an MLX NLC tensor.
    public let data: [Float]
    public let timeFrames: Int
    public let melBins: Int

    public init(data: [Float], timeFrames: Int, melBins: Int) {
        self.data = data
        self.timeFrames = timeFrames
        self.melBins = melBins
    }

    public func frames(_ range: Range<Int>) -> MiniCPMLogMelFeatures {
        precondition(range.lowerBound >= 0 && range.upperBound <= timeFrames)
        let start = range.lowerBound * melBins
        let end = range.upperBound * melBins
        return MiniCPMLogMelFeatures(
            data: Array(data[start..<end]),
            timeFrames: range.count,
            melBins: melBins
        )
    }
}

public enum MiniCPMLogMelNormalization: Equatable, Sendable {
    case dynamic(rangeDB: Float = 8)
    case fixed(floorDB: Float = -10)
}

/// Exact 400-point Whisper frontend used by MiniCPM-o 4.5.
///
/// Accelerate does not expose a real DFT setup for length 400, so this uses a
/// batched matrix DFT. It matches `torch.stft(center: true)`, drops the final
/// frame, applies the Slaney-normalized 80-bin filterbank, then MiniCPM's
/// configurable log floor.
public final class MiniCPMWhisperFeatureExtractor: @unchecked Sendable {
    public static let sampleRate = 16_000
    public static let fftSize = 400
    public static let hopLength = 160
    public static let melBins = 80

    // The pinned Transformers frontend promotes the waveform and Hann window
    // to Float64 before calculating the reference spectrogram.  Keep the
    // matrix DFT and mel reduction at that precision as well.  Returning
    // Float remains intentional: MLX casts the feature tensor to the encoder
    // checkpoint dtype at the model boundary.
    private let hannWindow: [Double]
    private let dftRealBasis: [Double]
    private let dftImaginaryBasis: [Double]
    private let melFilterbank: [Double]

    public init() {
        hannWindow = (0..<Self.fftSize).map { index in
            0.5 * (1 - cos(2 * Double.pi * Double(index) / Double(Self.fftSize)))
        }
        let frequencyBins = Self.fftSize / 2 + 1
        var real = [Double](repeating: 0, count: Self.fftSize * frequencyBins)
        var imaginary = real
        for sample in 0..<Self.fftSize {
            for bin in 0..<frequencyBins {
                let angle = -2 * Double.pi * Double(sample * bin) / Double(Self.fftSize)
                real[sample * frequencyBins + bin] = cos(angle)
                imaginary[sample * frequencyBins + bin] = sin(angle)
            }
        }
        dftRealBasis = real
        dftImaginaryBasis = imaginary
        melFilterbank = Self.makeMelFilterbank()
    }

    public func extract(
        _ audio: [Float],
        normalization: MiniCPMLogMelNormalization = .dynamic()
    ) throws -> MiniCPMLogMelFeatures {
        guard audio.count >= Self.fftSize else {
            throw MiniCPMAudioError.invalidFeatures(
                "at least \(Self.fftSize) waveform samples are required"
            )
        }

        let pad = Self.fftSize / 2
        var centered = [Double](repeating: 0, count: audio.count + 2 * pad)
        for index in 0..<pad {
            centered[index] = Double(audio[pad - index])
        }
        for index in audio.indices {
            centered[pad + index] = Double(audio[index])
        }
        for index in 0..<pad {
            centered[pad + audio.count + index] =
                Double(audio[audio.count - 2 - index])
        }

        let stftFrames = (centered.count - Self.fftSize) / Self.hopLength + 1
        let outputFrames = stftFrames - 1
        let frequencyBins = Self.fftSize / 2 + 1
        var windowed = [Double](repeating: 0, count: stftFrames * Self.fftSize)
        centered.withUnsafeBufferPointer { source in
            windowed.withUnsafeMutableBufferPointer { destination in
                for frame in 0..<stftFrames {
                    vDSP_vmulD(
                        source.baseAddress! + frame * Self.hopLength,
                        1,
                        hannWindow,
                        1,
                        destination.baseAddress! + frame * Self.fftSize,
                        1,
                        vDSP_Length(Self.fftSize)
                    )
                }
            }
        }

        let spectrumCount = stftFrames * frequencyBins
        var real = [Double](repeating: 0, count: spectrumCount)
        var imaginary = [Double](repeating: 0, count: spectrumCount)
        vDSP_mmulD(
            windowed, 1, dftRealBasis, 1, &real, 1,
            vDSP_Length(stftFrames), vDSP_Length(frequencyBins),
            vDSP_Length(Self.fftSize)
        )
        vDSP_mmulD(
            windowed, 1, dftImaginaryBasis, 1, &imaginary, 1,
            vDSP_Length(stftFrames), vDSP_Length(frequencyBins),
            vDSP_Length(Self.fftSize)
        )
        var realSquared = [Double](repeating: 0, count: spectrumCount)
        var imaginarySquared = [Double](repeating: 0, count: spectrumCount)
        vDSP_vsqD(real, 1, &realSquared, 1, vDSP_Length(spectrumCount))
        vDSP_vsqD(
            imaginary, 1, &imaginarySquared, 1, vDSP_Length(spectrumCount))
        var power = [Double](repeating: 0, count: spectrumCount)
        vDSP_vaddD(
            realSquared, 1, imaginarySquared, 1, &power, 1,
            vDSP_Length(spectrumCount)
        )

        var mel = [Double](repeating: 0, count: stftFrames * Self.melBins)
        vDSP_mmulD(
            power, 1, melFilterbank, 1, &mel, 1,
            vDSP_Length(stftFrames), vDSP_Length(Self.melBins),
            vDSP_Length(frequencyBins)
        )

        // Dropping the final frame means the first outputFrames rows are the
        // entire result because the matrix above is frame-major.
        let count = outputFrames * Self.melBins
        for index in 0..<count {
            mel[index] = log10(max(mel[index], 1e-10))
        }
        let floor: Double
        switch normalization {
        case .dynamic(let rangeDB):
            let peak = mel.prefix(count).max() ?? -.infinity
            floor = peak - Double(rangeDB)
        case .fixed(let floorDB):
            floor = Double(floorDB)
        }
        for index in 0..<count {
            mel[index] = (max(mel[index], floor) + 4) / 4
        }
        return MiniCPMLogMelFeatures(
            data: mel.prefix(count).map(Float.init),
            timeFrames: outputFrames,
            melBins: Self.melBins
        )
    }

    /// Extract the unnormalised log-mel rows for a set of already-windowed
    /// STFT frames.  The streaming frontend uses this to calculate only new
    /// frames after the old PCM has been retired from its bounded ring.  The
    /// matrix dimensions and Accelerate operations intentionally mirror
    /// ``extract`` above; callers apply the global fixed/dynamic clipping and
    /// Whisper scale after the rows have been selected.
    ///
    /// `windowed` is row-major `[frame, fftSize]` and must already include the
    /// centered/reflected waveform context for every requested frame.
    fileprivate func rawLogMelFrames(
        _ windowed: [Float],
        frameCount: Int
    ) -> [Float] {
        guard frameCount > 0 else { return [] }
        precondition(windowed.count == frameCount * Self.fftSize)

        let frequencyBins = Self.fftSize / 2 + 1
        var tapered = [Double](repeating: 0, count: windowed.count)
        let doubleWindowed = windowed.map(Double.init)
        doubleWindowed.withUnsafeBufferPointer { source in
            tapered.withUnsafeMutableBufferPointer { destination in
                for frame in 0..<frameCount {
                    vDSP_vmulD(
                        source.baseAddress! + frame * Self.fftSize,
                        1,
                        hannWindow,
                        1,
                        destination.baseAddress! + frame * Self.fftSize,
                        1,
                        vDSP_Length(Self.fftSize)
                    )
                }
            }
        }

        let spectrumCount = frameCount * frequencyBins
        var real = [Double](repeating: 0, count: spectrumCount)
        var imaginary = [Double](repeating: 0, count: spectrumCount)
        vDSP_mmulD(
            tapered, 1, dftRealBasis, 1, &real, 1,
            vDSP_Length(frameCount), vDSP_Length(frequencyBins),
            vDSP_Length(Self.fftSize)
        )
        vDSP_mmulD(
            tapered, 1, dftImaginaryBasis, 1, &imaginary, 1, 
            vDSP_Length(frameCount), vDSP_Length(frequencyBins),
            vDSP_Length(Self.fftSize)
        )

        var realSquared = [Double](repeating: 0, count: spectrumCount)
        var imaginarySquared = [Double](repeating: 0, count: spectrumCount)
        vDSP_vsqD(real, 1, &realSquared, 1, vDSP_Length(spectrumCount))
        vDSP_vsqD(
            imaginary, 1, &imaginarySquared, 1, vDSP_Length(spectrumCount))
        var power = [Double](repeating: 0, count: spectrumCount)
        vDSP_vaddD(
            realSquared, 1, imaginarySquared, 1, &power, 1,
            vDSP_Length(spectrumCount)
        )

        var mel = [Double](repeating: 0, count: frameCount * Self.melBins)
        vDSP_mmulD(
            power, 1, melFilterbank, 1, &mel, 1,
            vDSP_Length(frameCount), vDSP_Length(Self.melBins),
            vDSP_Length(frequencyBins)
        )
        for index in mel.indices {
            mel[index] = log10(max(mel[index], 1e-10))
        }
        return mel.map(Float.init)
    }

    private static func makeMelFilterbank() -> [Double] {
        let frequencyBins = fftSize / 2 + 1
        let minimumLogHertz = 1_000.0
        let minimumLogMel = 15.0
        let logStep = 27.0 / log(6.4)

        func hertzToMel(_ hertz: Double) -> Double {
            hertz < minimumLogHertz
                ? 3 * hertz / 200
                : minimumLogMel + log(hertz / minimumLogHertz) * logStep
        }
        func melToHertz(_ mel: Double) -> Double {
            mel < minimumLogMel
                ? 200 * mel / 3
                : minimumLogHertz * exp((mel - minimumLogMel) / logStep)
        }

        let minimum = hertzToMel(0)
        let maximum = hertzToMel(Double(sampleRate) / 2)
        let points = (0..<(melBins + 2)).map { index -> Double in
            melToHertz(
                minimum + Double(index) * (maximum - minimum) / Double(melBins + 1)
            )
        }
        let differences = zip(points, points.dropFirst()).map { $1 - $0 }
        var filters = [Double](repeating: 0, count: frequencyBins * melBins)
        for bin in 0..<frequencyBins {
            let hertz = Double(bin) * Double(sampleRate) / Double(fftSize)
            for mel in 0..<melBins {
                let rising = (hertz - points[mel]) / differences[mel]
                let falling = (points[mel + 2] - hertz) / differences[mel + 1]
                let triangle = max(0, min(rising, falling))
                let slaney = 2 / (points[mel + 2] - points[mel])
                filters[bin * melBins + mel] = triangle * slaney
            }
        }
        return filters
    }
}

/// Stateful implementation of MiniCPM's `StreamingMelProcessorExact`.
public final class MiniCPMStreamingMelProcessor {
    public static let regularChunkSamples = 16_000
    /// Upstream requests 1035 ms, then aligns down to 1030 ms / 16,480 samples.
    public static let firstChunkSamples = 16_480

    private let extractor: MiniCPMWhisperFeatureExtractor

    // The upstream implementation is deliberately exact but keeps the whole
    // conversation waveform and recomputes the complete STFT after every
    // chunk.  That is fine for a short fixture, but turns a long full-duplex
    // call into O(n²) work and unbounded memory.  We retain only the 200-sample
    // left context needed by the next centered frame plus the not-yet-stable
    // right context.  `bufferStartSample` maps this compact ring to the
    // original stream coordinate system.
    private var buffer: [Float] = []
    private var bufferStartSample = 0
    private var totalSamplesProcessed = 0
    private var lastEmittedFrame = 0
    private var scannedStableFrames = 0
    private var stableRawMaximum = -Float.infinity
    private var isFirstChunk = true

    /// A copyable checkpoint for barge-in/replay.  The PCM retained here is
    /// bounded to the same overlap ring as the live processor; no complete
    /// conversation waveform is duplicated.
    public struct Snapshot: Equatable, Sendable {
        public let buffer: [Float]
        public let bufferStartSample: Int
        public let totalSamplesProcessed: Int
        public let lastEmittedFrame: Int
        public let scannedStableFrames: Int
        public let stableRawMaximum: Float
        public let isFirstChunk: Bool

        public init(
            buffer: [Float],
            bufferStartSample: Int,
            totalSamplesProcessed: Int,
            lastEmittedFrame: Int,
            scannedStableFrames: Int,
            stableRawMaximum: Float,
            isFirstChunk: Bool
        ) {
            self.buffer = buffer
            self.bufferStartSample = bufferStartSample
            self.totalSamplesProcessed = totalSamplesProcessed
            self.lastEmittedFrame = lastEmittedFrame
            self.scannedStableFrames = scannedStableFrames
            self.stableRawMaximum = stableRawMaximum
            self.isFirstChunk = isFirstChunk
        }
    }

    public init(extractor: MiniCPMWhisperFeatureExtractor = .init()) {
        self.extractor = extractor
    }

    /// Number of samples needed to complete the current frontend input
    /// window.  A caller may feed smaller packets; unlike the old
    /// `buffer.isEmpty` check this continues to request the remainder of the
    /// 16,480-sample first window until it is actually complete.
    public var nextChunkSampleCount: Int {
        if isFirstChunk {
            return max(1, Self.firstChunkSamples - totalSamplesProcessed)
        }
        let consumed = max(0, totalSamplesProcessed - Self.firstChunkSamples)
        let remainder = Self.regularChunkSamples
            - (consumed % Self.regularChunkSamples)
        return max(1, remainder)
    }

    /// Number of PCM samples retained in the bounded overlap ring.
    public var bufferedSampleCount: Int { buffer.count }
    public var totalSampleCount: Int { totalSamplesProcessed }
    public var emittedFrameCount: Int { lastEmittedFrame }
    public var bufferStartSampleIndex: Int { bufferStartSample }

    public func snapshot() -> Snapshot {
        Snapshot(
            buffer: buffer,
            bufferStartSample: bufferStartSample,
            totalSamplesProcessed: totalSamplesProcessed,
            lastEmittedFrame: lastEmittedFrame,
            scannedStableFrames: scannedStableFrames,
            stableRawMaximum: stableRawMaximum,
            isFirstChunk: isFirstChunk
        )
    }

    public func restore(_ snapshot: Snapshot) {
        precondition(snapshot.bufferStartSample >= 0)
        precondition(snapshot.totalSamplesProcessed >= snapshot.bufferStartSample)
        precondition(snapshot.lastEmittedFrame >= 0)
        precondition(snapshot.scannedStableFrames >= snapshot.lastEmittedFrame)
        precondition(snapshot.buffer.count
            == snapshot.totalSamplesProcessed - snapshot.bufferStartSample)
        buffer = snapshot.buffer
        bufferStartSample = snapshot.bufferStartSample
        totalSamplesProcessed = snapshot.totalSamplesProcessed
        lastEmittedFrame = snapshot.lastEmittedFrame
        scannedStableFrames = snapshot.scannedStableFrames
        stableRawMaximum = snapshot.stableRawMaximum
        isFirstChunk = snapshot.isFirstChunk
    }

    /// Alias matching the Python processor's naming and the duplex rollback
    /// call sites.  Restoring a snapshot is deterministic because all
    /// normalization state is included in the checkpoint.
    public func restoreSnapshot(_ snapshot: Snapshot) { restore(snapshot) }
    public func rollback(to snapshot: Snapshot) { restore(snapshot) }

    public func reset() {
        buffer.removeAll(keepingCapacity: true)
        bufferStartSample = 0
        totalSamplesProcessed = 0
        lastEmittedFrame = 0
        scannedStableFrames = 0
        stableRawMaximum = -Float.infinity
        isFirstChunk = true
    }

    /// Append PCM and emit stable 100-frame chunks. The two-frame look-ahead
    /// used to make the centered STFT stable remains buffered automatically.
    public func accept(_ samples: [Float], isFinal: Bool = false) throws -> MiniCPMLogMelFeatures? {
        guard samples.allSatisfy(\.isFinite) else {
            throw MiniCPMAudioError.invalidFeatures(
                "PCM contains NaN or infinity")
        }
        if !samples.isEmpty {
            buffer.append(contentsOf: samples)
            totalSamplesProcessed += samples.count
        }
        if totalSamplesProcessed >= Self.firstChunkSamples {
            isFirstChunk = false
        }

        let fullFrames = totalSamplesProcessed
            / MiniCPMWhisperFeatureExtractor.hopLength
        let stableFrames: Int
        if totalSamplesProcessed < MiniCPMWhisperFeatureExtractor.fftSize / 2 {
            stableFrames = 0
        } else {
            stableFrames = min(
                fullFrames,
                (totalSamplesProcessed
                    - MiniCPMWhisperFeatureExtractor.fftSize / 2)
                    / MiniCPMWhisperFeatureExtractor.hopLength + 1
            )
        }

        // Track the maximum over every stable row.  Dynamic MiniCPM
        // normalisation is global over the accumulated stream, so retaining
        // this scalar is sufficient after old PCM has been retired.
        if stableFrames > scannedStableFrames {
            let start = scannedStableFrames
            let count = stableFrames - start
            let raw = rawRows(startFrame: start, frameCount: count)
            if let maximum = raw.max() {
                stableRawMaximum = max(stableRawMaximum, maximum)
            }
            scannedStableFrames = stableFrames
        }

        let currentMaximum = max(
            stableRawMaximum,
            trailingRawMaximum(fullFrames: fullFrames, stableFrames: stableFrames)
        )
        let targetEnd = lastEmittedFrame + 100
        let end: Int
        if isFinal {
            // A final call is a true flush: return all rows not emitted yet,
            // including the one centered frame that still needs reflected
            // right padding.  This avoids silently dropping a short tail.
            end = fullFrames
        } else if isFirstChunk && totalSamplesProcessed < Self.firstChunkSamples {
            // The protocol's first window is deliberately 1,030 ms (16,480
            // samples), even though centered STFT math would already make
            // 100 rows stable at 16,479 samples.  Keep accepting/buffering
            // the short packet so all callers observe the same first-window
            // boundary as the upstream processor.
            return nil
        } else if stableFrames >= targetEnd {
            end = targetEnd
        } else {
            return nil
        }
        guard end > lastEmittedFrame else { return nil }

        let raw = rawRows(
            startFrame: lastEmittedFrame,
            frameCount: end - lastEmittedFrame)
        let normalization: MiniCPMLogMelNormalization =
            totalSamplesProcessed < 5 * MiniCPMWhisperFeatureExtractor.sampleRate
            ? .fixed(floorDB: -10)
            : .dynamic(rangeDB: 8)
        let result = normalize(
            raw,
            mode: normalization,
            dynamicMaximum: currentMaximum
        )
        lastEmittedFrame = end
        retireConsumedPCM()
        return result
    }

    private func sample(at index: Int) -> Float {
        // Reflect padding exactly as torch.stft(center: true, pad_mode=
        // "reflect") for the two boundaries.  Non-final rows never ask for
        // future samples past totalSamplesProcessed because they are emitted
        // only after the right look-ahead is stable.
        var reflected = index
        if totalSamplesProcessed <= 1 { return 0 }
        while reflected < 0 || reflected >= totalSamplesProcessed {
            if reflected < 0 {
                reflected = -reflected
            } else {
                reflected = 2 * totalSamplesProcessed - 2 - reflected
            }
        }
        let local = reflected - bufferStartSample
        guard local >= 0 && local < buffer.count else {
            // A requested frame must be inside the overlap ring.  Keep this
            // as a deterministic failure rather than silently introducing
            // zeroes that would corrupt parity.
            preconditionFailure(
                "MiniCPM streaming ring dropped required PCM context")
        }
        return buffer[local]
    }

    private func rawRows(startFrame: Int, frameCount: Int) -> [Float] {
        guard frameCount > 0 else { return [] }
        let pad = MiniCPMWhisperFeatureExtractor.fftSize / 2
        let hop = MiniCPMWhisperFeatureExtractor.hopLength
        var windows = [Float](repeating: 0,
                              count: frameCount * MiniCPMWhisperFeatureExtractor.fftSize)
        for frame in 0..<frameCount {
            let center = (startFrame + frame) * hop
            let first = center - pad
            for sampleIndex in 0..<MiniCPMWhisperFeatureExtractor.fftSize {
                windows[frame * MiniCPMWhisperFeatureExtractor.fftSize + sampleIndex]
                    = self.sample(at: first + sampleIndex)
            }
        }
        return extractor.rawLogMelFrames(windows, frameCount: frameCount)
    }

    private func trailingRawMaximum(fullFrames: Int, stableFrames: Int) -> Float {
        guard fullFrames > stableFrames else { return -Float.infinity }
        let raw = rawRows(
            startFrame: stableFrames,
            frameCount: fullFrames - stableFrames)
        return raw.max() ?? -Float.infinity
    }

    private func normalize(
        _ raw: [Float],
        mode: MiniCPMLogMelNormalization,
        dynamicMaximum: Float
    ) -> MiniCPMLogMelFeatures {
        var values = raw
        guard !values.isEmpty else {
            return MiniCPMLogMelFeatures(
                data: [], timeFrames: 0,
                melBins: MiniCPMWhisperFeatureExtractor.melBins)
        }
        let floor: Float
        switch mode {
        case .fixed(let floorDB):
            floor = floorDB
        case .dynamic(let rangeDB):
            floor = dynamicMaximum - rangeDB
        }
        var maximum = Float.greatestFiniteMagnitude
        var mutableFloor = floor
        vDSP_vclip(
            values, 1, &mutableFloor, &maximum, &values, 1,
            vDSP_Length(values.count))
        var scale: Float = 0.25
        var offset: Float = 1
        vDSP_vsmsa(
            values, 1, &scale, &offset, &values, 1,
            vDSP_Length(values.count))
        return MiniCPMLogMelFeatures(
            data: values,
            timeFrames: values.count / MiniCPMWhisperFeatureExtractor.melBins,
            melBins: MiniCPMWhisperFeatureExtractor.melBins)
    }

    private func retireConsumedPCM() {
        let pad = MiniCPMWhisperFeatureExtractor.fftSize / 2
        let hop = MiniCPMWhisperFeatureExtractor.hopLength
        let requiredStart = max(0, lastEmittedFrame * hop - pad)
        let drop = requiredStart - bufferStartSample
        guard drop > 0 else { return }
        precondition(drop <= buffer.count)
        buffer.removeFirst(drop)
        bufferStartSample = requiredStart
    }
}
