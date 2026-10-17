import Accelerate
import Foundation
#if canImport(MLX)
import MLX
#endif

public struct MiniCPMLogMelFeatures: Equatable, Sendable {
    /// Row-major `[timeFrames, melBins]`, ready to construct an MLX NLC tensor.
    public let data: [Float]
    public let timeFrames: Int
    public let melBins: Int
    /// Extra mel rows supplied around the core of a streaming chunk.
    /// Offline features have no CNN context to remove.
    public let prefixExtraFrames: Int
    public let suffixExtraFrames: Int

    public init(
        data: [Float], timeFrames: Int, melBins: Int,
        prefixExtraFrames: Int = 0, suffixExtraFrames: Int = 0
    ) {
        self.data = data
        self.timeFrames = timeFrames
        self.melBins = melBins
        self.prefixExtraFrames = prefixExtraFrames
        self.suffixExtraFrames = suffixExtraFrames
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
/// Uses the CPU Float32 pocketFFT route to match `torch.stft(center: true)`, drops the final
/// frame, applies the Slaney-normalized 80-bin filterbank, then MiniCPM's
/// configurable log floor.
public final class MiniCPMWhisperFeatureExtractor: @unchecked Sendable {
    public static let sampleRate = 16_000
    public static let fftSize = 400
    public static let hopLength = 160
    public static let melBins = 80

    // Transformers uses torch's Float32 frontend when torch is installed.
    // Its NumPy-only Float64 fallback is not the official MPS input path.
    // Retain the matrix fallback only for standalone geometry/DSP probes
    // compiled without MLX; model inference always uses CPU pocketFFT.
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
        var centered = [Float](repeating: 0, count: audio.count + 2 * pad)
        for index in 0..<pad {
            centered[index] = audio[pad - index]
        }
        centered.replaceSubrange(pad..<(pad + audio.count), with: audio)
        for index in 0..<pad {
            centered[pad + audio.count + index] = audio[audio.count - 2 - index]
        }

        let outputFrames = (centered.count - Self.fftSize) / Self.hopLength
        var frames = [Float](repeating: 0, count: outputFrames * Self.fftSize)
        for frame in 0..<outputFrames {
            let source = frame * Self.hopLength
            frames.replaceSubrange(
                (frame * Self.fftSize)..<((frame + 1) * Self.fftSize),
                with: centered[source..<(source + Self.fftSize)])
        }
        var mel = rawLogMelFrames(frames, frameCount: outputFrames)
        let floor: Float
        switch normalization {
        case .dynamic(let rangeDB):
            floor = (mel.max() ?? -.infinity) - rangeDB
        case .fixed(let floorDB):
            floor = floorDB
        }
        for index in mel.indices {
            mel[index] = (max(mel[index], floor) + 4) / 4
        }
        return MiniCPMLogMelFeatures(
            data: mel, timeFrames: outputFrames, melBins: Self.melBins)
    }

    /// Extract the unnormalised log-mel rows for a set of already-windowed
    /// STFT frames.  The streaming frontend uses this to calculate only new
    /// frames after the old PCM has been retired from its bounded ring.  The
    /// Float32 FFT, complex magnitude and mel reduction match torch's CPU
    /// frontend; callers apply the global fixed/dynamic clipping and
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

#if canImport(MLX)
        let frequencyBins = Self.fftSize / 2 + 1
        var tapered = windowed
        for index in tapered.indices {
            tapered[index] *= Self.torchHannWindow[index % Self.fftSize]
        }
        // GPU FFT uses a different reduction order. Explicitly select the
        // existing MLX CPU pocketFFT implementation used by torch/scipy.
        let spectrum = MLXFFT.rfft(
            MLXArray(tapered, [frameCount, Self.fftSize]), stream: .cpu)
        let real = spectrum.realPart(stream: .cpu).asArray(Float.self)
        let imaginary = spectrum.imaginaryPart(stream: .cpu).asArray(Float.self)
        var power = [Float](repeating: 0, count: real.count)
        for index in power.indices {
            // torch computes abs(complex64) before squaring. Summing re²+im²
            // directly has a different Float32 rounding boundary.
            let magnitude = hypotf(real[index], imaginary[index])
            power[index] = magnitude * magnitude
        }
        var mel = [Float](repeating: 0, count: frameCount * Self.melBins)
        vDSP_mmul(
            power, 1, melFilterbank.map(Float.init), 1, &mel, 1,
            vDSP_Length(frameCount), vDSP_Length(Self.melBins),
            vDSP_Length(frequencyBins))
        for index in mel.indices {
            mel[index] = log10f(max(mel[index], 1e-10))
        }
        return mel
#else
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
#endif
    }

    // Fixed model geometry: torch 2.8 CPU hann_window(400, periodic: true).
    // Store Float32 bits because platform cosf/vector kernels can differ by
    // one ULP, enough to change the BF16 input after the log-mel transform.
    private static let torchHannWindow: [Float] = ([
        0x00000000, 0x38816000, 0x39815c00, 0x3a118200, 0x3a815200, 0x3aca0700, 0x3b116d00, 0x3b45e300,
        0x3b813140, 0x3ba373c0, 0x3bc9b6c0, 0x3bf3f800, 0x3c111a60, 0x3c2a34e0, 0x3c454a40, 0x3c625880,
        0x3c80aef0, 0x3c912c40, 0x3ca2a310, 0x3cb51260, 0x3cc878f0, 0x3cdcd5a0, 0x3cf22700, 0x3d0435e8,
        0x3d0fd158, 0x3d1be510, 0x3d287040, 0x3d357230, 0x3d42ea08, 0x3d50d6f0, 0x3d5f37f8, 0x3d6e0c48,
        0x3d7d52f0, 0x3d868578, 0x3d8e99a4, 0x3d96e580, 0x3d9f6880, 0x3da82220, 0x3db111cc, 0x3dba36f8,
        0x3dc3910c, 0x3dcd1f78, 0x3dd6e19c, 0x3de0d6dc, 0x3deafe94, 0x3df55824, 0x3dffe2e4, 0x3e054f14,
        0x3e0ac4a2, 0x3e1051c0, 0x3e15f61a, 0x3e1bb152, 0x3e21830c, 0x3e276ae8, 0x3e2d688a, 0x3e337b8c,
        0x3e39a392, 0x3e3fe034, 0x3e46310e, 0x3e4c95b8, 0x3e530dd0, 0x3e5998ea, 0x3e60369c, 0x3e66e67c,
        0x3e6da81e, 0x3e747b14, 0x3e7b5eee, 0x3e8129a0, 0x3e84abcc, 0x3e8835c2, 0x3e8bc748, 0x3e8f6027,
        0x3e930022, 0x3e96a700, 0x3e9a5486, 0x3e9e0876, 0x3ea1c297, 0x3ea582ac, 0x3ea94879, 0x3ead13c0,
        0x3eb0e444, 0x3eb4b9c6, 0x3eb8940a, 0x3ebc72d2, 0x3ec055dd, 0x3ec43cee, 0x3ec827c5, 0x3ecc1623,
        0x3ed007c8, 0x3ed3fc76, 0x3ed7f3eb, 0x3edbede7, 0x3edfea2b, 0x3ee3e876, 0x3ee7e887, 0x3eebea1d,
        0x3eefecf8, 0x3ef3f0d5, 0x3ef7f577, 0x3efbfa9c, 0x3f000000, 0x3f0202b3, 0x3f040545, 0x3f060796,
        0x3f080986, 0x3f0a0af3, 0x3f0c0bbe, 0x3f0e0bc7, 0x3f100aec, 0x3f12090e, 0x3f14060c, 0x3f1601c7,
        0x3f17fc1d, 0x3f19f4f0, 0x3f1bec1f, 0x3f1de18b, 0x3f1fd513, 0x3f21c699, 0x3f23b5fb, 0x3f25a31e,
        0x3f278ddf, 0x3f297621, 0x3f2b5bc4, 0x3f2d3eaa, 0x3f2f1eb5, 0x3f30fbc6, 0x3f32d5bf, 0x3f34ac82,
        0x3f367ff0, 0x3f384fee, 0x3f3a1c5d, 0x3f3be521, 0x3f3daa1c, 0x3f3f6b32, 0x3f412846, 0x3f42e13c,
        0x3f4495fa, 0x3f464662, 0x3f47f25a, 0x3f4999c7, 0x3f4b3c8e, 0x3f4cda93, 0x3f4e73be, 0x3f5007f5,
        0x3f51971e, 0x3f53211e, 0x3f54a5e0, 0x3f562548, 0x3f579f3e, 0x3f5913ad, 0x3f5a827a, 0x3f5beb90,
        0x3f5d4ed8, 0x3f5eac3c, 0x3f6003a4, 0x3f6154fc, 0x3f62a02e, 0x3f63e525, 0x3f6523cd, 0x3f665c12,
        0x3f678ddf, 0x3f68b922, 0x3f69ddc7, 0x3f6afbbc, 0x3f6c12f0, 0x3f6d2350, 0x3f6e2ccc, 0x3f6f2f52,
        0x3f702ad2, 0x3f711f3c, 0x3f720c81, 0x3f72f292, 0x3f73d160, 0x3f74a8de, 0x3f7578fc, 0x3f7641b0,
        0x3f7702eb, 0x3f77bca2, 0x3f786ec8, 0x3f791954, 0x3f79bc39, 0x3f7a576e, 0x3f7aeae8, 0x3f7b769e,
        0x3f7bfa89, 0x3f7c769e, 0x3f7cead8, 0x3f7d572c, 0x3f7dbb97, 0x3f7e1810, 0x3f7e6c92, 0x3f7eb918,
        0x3f7efd9e, 0x3f7f3a1d, 0x3f7f6e93, 0x3f7f9afc, 0x3f7fbf57, 0x3f7fdba0, 0x3f7fefd4, 0x3f7ffbf5,
        0x3f800000, 0x3f7ffbf5, 0x3f7fefd4, 0x3f7fdba0, 0x3f7fbf57, 0x3f7f9afc, 0x3f7f6e93, 0x3f7f3a1c,
        0x3f7efd9d, 0x3f7eb918, 0x3f7e6c92, 0x3f7e1810, 0x3f7dbb96, 0x3f7d572c, 0x3f7cead7, 0x3f7c769e,
        0x3f7bfa88, 0x3f7b769e, 0x3f7aeae7, 0x3f7a576c, 0x3f79bc38, 0x3f791952, 0x3f786ec8, 0x3f77bca0,
        0x3f7702ea, 0x3f7641ae, 0x3f7578fb, 0x3f74a8dc, 0x3f73d15e, 0x3f72f290, 0x3f720c80, 0x3f711f3a,
        0x3f702ad0, 0x3f6f2f50, 0x3f6e2cca, 0x3f6d2350, 0x3f6c12f0, 0x3f6afbbc, 0x3f69ddc6, 0x3f68b921,
        0x3f678dde, 0x3f665c10, 0x3f6523cc, 0x3f63e524, 0x3f62a02d, 0x3f6154fb, 0x3f6003a3, 0x3f5eac3a,
        0x3f5d4ed7, 0x3f5beb8f, 0x3f5a8278, 0x3f5913aa, 0x3f579f3c, 0x3f562545, 0x3f54a5dd, 0x3f53211c,
        0x3f51971a, 0x3f5007f2, 0x3f4e73bc, 0x3f4cda90, 0x3f4b3c8a, 0x3f4999c4, 0x3f47f258, 0x3f464660,
        0x3f4495f7, 0x3f42e13a, 0x3f412843, 0x3f3f6b2e, 0x3f3daa19, 0x3f3be51e, 0x3f3a1c5a, 0x3f384feb,
        0x3f367fed, 0x3f34ac7e, 0x3f32d5bc, 0x3f30fbc3, 0x3f2f1eb2, 0x3f2d3ea7, 0x3f2b5bc1, 0x3f29761e,
        0x3f278ddc, 0x3f25a31a, 0x3f23b5f8, 0x3f21c694, 0x3f1fd50f, 0x3f1de186, 0x3f1bec1b, 0x3f19f4ec,
        0x3f17fc19, 0x3f1601c2, 0x3f140608, 0x3f12090a, 0x3f100ae8, 0x3f0e0bc2, 0x3f0c0bba, 0x3f0a0aef,
        0x3f080981, 0x3f060792, 0x3f040541, 0x3f0202b2, 0x3f000000, 0x3efbfa9b, 0x3ef7f576, 0x3ef3f0d4,
        0x3eefecf5, 0x3eebea1a, 0x3ee7e884, 0x3ee3e873, 0x3edfea28, 0x3edbede4, 0x3ed7f3e8, 0x3ed3fc73,
        0x3ed007c6, 0x3ecc1620, 0x3ec827c2, 0x3ec43ceb, 0x3ec055da, 0x3ebc72cf, 0x3eb89408, 0x3eb4b9c4,
        0x3eb0e441, 0x3ead13bd, 0x3ea94876, 0x3ea582aa, 0x3ea1c294, 0x3e9e0872, 0x3e9a5482, 0x3e96a6fc,
        0x3e93001e, 0x3e8f6022, 0x3e8bc744, 0x3e8835be, 0x3e84abc8, 0x3e81299c, 0x3e7b5ee6, 0x3e747b0a,
        0x3e6da816, 0x3e66e674, 0x3e603694, 0x3e5998e2, 0x3e530dc8, 0x3e4c95b0, 0x3e463104, 0x3e3fe02a,
        0x3e39a388, 0x3e337b84, 0x3e2d6880, 0x3e276ae0, 0x3e218302, 0x3e1bb14a, 0x3e15f612, 0x3e1051b8,
        0x3e0ac498, 0x3e054f0a, 0x3dffe2d4, 0x3df55814, 0x3deafe84, 0x3de0d6cc, 0x3dd6e18c, 0x3dcd1f68,
        0x3dc390fc, 0x3dba36e8, 0x3db111bc, 0x3da82210, 0x3d9f6870, 0x3d96e570, 0x3d8e9994, 0x3d868568,
        0x3d7d52d0, 0x3d6e0c30, 0x3d5f37e0, 0x3d50d6d0, 0x3d42e9f0, 0x3d357218, 0x3d287028, 0x3d1be4f0,
        0x3d0fd140, 0x3d0435d0, 0x3cf226d0, 0x3cdcd570, 0x3cc878d0, 0x3cb51230, 0x3ca2a2f0, 0x3c912c10,
        0x3c80aed0, 0x3c625860, 0x3c454a20, 0x3c2a34e0, 0x3c111a40, 0x3bf3f800, 0x3bc9b6c0, 0x3ba373c0,
        0x3b813140, 0x3b45e300, 0x3b116d00, 0x3aca0700, 0x3a815200, 0x3a118200, 0x39815c00, 0x38816000,
    ] as [UInt32]).map { Float(bitPattern: $0) }

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

    // Match the upstream 30-second normalization window, sliding by ten
    // seconds. Keeping its PCM costs less than 2 MB and lets us recompute the
    // peak with the new reflected left boundary after a slide. Between slides
    // only newly stable STFT rows are computed. The retained window also
    // contains the two mel rows and 200 PCM samples needed for CNN/STFT context.
    private static let slideTriggerSamples = 30 * MiniCPMWhisperFeatureExtractor.sampleRate
    private static let slideStrideSamples = 10 * MiniCPMWhisperFeatureExtractor.sampleRate
    public static let cnnContextFrames = 2
    private var buffer: [Float] = []
    private var bufferStartSample = 0
    private var totalSamplesProcessed = 0
    private var lastEmittedFrame = 0
    private var scannedStableFrames = 0
    private var stableRawMaximum = -Float.infinity
    private var isFirstChunk = true

    /// A copyable checkpoint for barge-in/replay. The bounded normalization
    /// window and its peak are restored together, including any CNN overlap.
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

    /// Number of PCM samples retained in the bounded normalization window.
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
        precondition(snapshot.scannedStableFrames >= snapshot.bufferStartSample
            / MiniCPMWhisperFeatureExtractor.hopLength)
        precondition(snapshot.scannedStableFrames <= snapshot.totalSamplesProcessed
            / MiniCPMWhisperFeatureExtractor.hopLength)
        precondition(snapshot.lastEmittedFrame <= snapshot.totalSamplesProcessed
            / MiniCPMWhisperFeatureExtractor.hopLength)
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

    /// Emit 100 core mel rows plus two CNN-context rows on each available
    /// side: 102 rows initially and 104 thereafter. A final flush retains its
    /// left context but has no redundant right rows to discard.
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
        slideNormalizationWindowIfNeeded()

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

        // The normalization peak covers the current upstream PCM window.
        // Sliding resets this scan so reflected boundary rows and the removal
        // of a loud historical event both affect the new peak.
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
        let coreStart = lastEmittedFrame
        let targetEnd = coreStart + 100
        let coreEnd: Int
        let emitEnd: Int
        if isFinal {
            // A final call is a true flush: return all rows not emitted yet,
            // including the one centered frame that still needs reflected
            // right padding.  This avoids silently dropping a short tail.
            coreEnd = fullFrames
            emitEnd = fullFrames
        } else if isFirstChunk && totalSamplesProcessed < Self.firstChunkSamples {
            // The protocol's first window is deliberately 1,030 ms (16,480
            // samples), even though centered STFT math would already make
            // 100 rows stable at 16,479 samples.  Keep accepting/buffering
            // the short packet so all callers observe the same first-window
            // boundary as the upstream processor.
            return nil
        } else if stableFrames >= targetEnd + Self.cnnContextFrames {
            coreEnd = targetEnd
            emitEnd = targetEnd + Self.cnnContextFrames
        } else {
            return nil
        }
        guard coreEnd > coreStart else { return nil }
        let emitStart = max(bufferStartSample / MiniCPMWhisperFeatureExtractor.hopLength,
                            max(0, coreStart - Self.cnnContextFrames))
        let raw = rawRows(
            startFrame: emitStart,
            frameCount: emitEnd - emitStart)
        let normalization: MiniCPMLogMelNormalization =
            buffer.count < 5 * MiniCPMWhisperFeatureExtractor.sampleRate
            ? .fixed(floorDB: -10)
            : .dynamic(rangeDB: 8)
        let result = normalize(
            raw,
            mode: normalization,
            dynamicMaximum: currentMaximum,
            prefixExtraFrames: coreStart - emitStart,
            suffixExtraFrames: emitEnd - coreEnd
        )
        lastEmittedFrame = coreEnd
        return result
    }

    private func sample(at index: Int) -> Float {
        // Reflect padding exactly as torch.stft(center: true, pad_mode=
        // "reflect") for the two boundaries.  Non-final rows never ask for
        // future samples past totalSamplesProcessed because they are emitted
        // only after the right look-ahead is stable.
        var reflected = index
        if buffer.count <= 1 { return 0 }
        while reflected < bufferStartSample || reflected >= totalSamplesProcessed {
            if reflected < bufferStartSample {
                reflected = 2 * bufferStartSample - reflected
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
        dynamicMaximum: Float,
        prefixExtraFrames: Int,
        suffixExtraFrames: Int
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
            melBins: MiniCPMWhisperFeatureExtractor.melBins,
            prefixExtraFrames: prefixExtraFrames,
            suffixExtraFrames: suffixExtraFrames)
    }

    private func slideNormalizationWindowIfNeeded() {
        let pad = MiniCPMWhisperFeatureExtractor.fftSize / 2
        let hop = MiniCPMWhisperFeatureExtractor.hopLength
        guard buffer.count >= Self.slideTriggerSamples else { return }
        // Protect un-emitted PCM when callers supply more than one chunk in a
        // packet. Normal one-second calls slide by exactly the official ten
        // seconds; buffered packets can drain before those samples retire.
        let requiredStart = max(0,
            (lastEmittedFrame - Self.cnnContextFrames) * hop - pad)
        let safeDrop = max(0, requiredStart - bufferStartSample)
        let drop = min(Self.slideStrideSamples, (safeDrop / hop) * hop)
        guard drop > 0 else { return }
        buffer.removeFirst(drop)
        bufferStartSample += drop
        scannedStableFrames = bufferStartSample / hop
        stableRawMaximum = -Float.infinity
    }
}
