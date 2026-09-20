import Accelerate
import Foundation
import MLX

/// Errors raised while constructing a MiniCPM-o Token2Wav voice prompt from
/// raw PCM.  The prompt fixture used by the original Python bridge contains
/// three arrays (`prompt_tokens`, `prompt_mel`, and `speaker_embedding`).  This
/// file owns the deterministic signal-processing part of producing those
/// arrays; model-specific token/speaker encoders are injected through
/// `MiniCPMToken2WavPromptEncoder`.
public enum MiniCPMToken2WavPromptPreparationError: Error, LocalizedError {
    case invalidSampleRate(Int)
    case invalidPCM(String)
    case missingSpeechTokenizer
    case missingSpeakerEncoder
    case missingModelAsset(String)
    case invalidTokens(String)
    case invalidSpeakerEmbedding(String)
    case emptyMel(String)

    public var errorDescription: String? {
        switch self {
        case .invalidSampleRate(let rate):
            return "Reference PCM has an invalid sample rate: \(rate)"
        case .invalidPCM(let reason):
            return "Reference PCM is invalid: \(reason)"
        case .missingSpeechTokenizer:
            return "No native speech-tokenizer was supplied; provide prompt tokens or an encoder"
        case .missingSpeakerEncoder:
            return "No native CAM++ speaker encoder was supplied; provide a speaker embedding or an encoder"
        case .missingModelAsset(let reason):
            return "MiniCPM prompt model asset is unavailable: \(reason)"
        case .invalidTokens(let reason):
            return "Reference speech tokens are invalid: \(reason)"
        case .invalidSpeakerEmbedding(let reason):
            return "Reference speaker embedding is invalid: \(reason)"
        case .emptyMel(let name):
            return "Reference audio produced no \(name) mel frames"
        }
    }
}

/// Optional model side of native prompt preparation.
///
/// The protocol deliberately accepts MLX arrays rather than Swift audio
/// buffers.  A future six-layer S3Tokenizer-v2 MLX module and a CAM++ MLX or
/// CoreML adapter can therefore be plugged in without changing this protocol
/// or the JSONL helper.  Until those weights are converted, callers may inject
/// `promptTokens` and `speakerEmbedding` directly (for example from an
/// existing safetensors fixture).
public protocol MiniCPMToken2WavPromptEncoder {
    func encodeSpeechTokens(tokenizerMel: MLXArray) throws -> MLXArray
    func encodeSpeakerEmbedding(speakerMel: MLXArray) throws -> MLXArray
}

/// How the variable-length CAM++ fbank was adapted to MiniCPM's fixed
/// `[1, 500, 80]` CoreML input.  Keeping this in the profile is useful when a
/// prompt looks wrong: a short clip is tiled, whereas a long clip is center
/// cropped.  A zero-frame clip is represented by an all-zero padded tensor.
public enum MiniCPMSpeakerMelAdaptation: String, Sendable, Equatable {
    case exact
    case tiledShort
    case centerCroppedLong
    case emptyPadded
}

/// Result of CAM++ feature extraction before/after its fixed-shape adapter.
/// `mel` is always the tensor that should be passed to CAM++.
public struct MiniCPMCamPlusPlusMelExtraction {
    public let mel: MLXArray
    public let rawFrameCount: Int
    public let adaptation: MiniCPMSpeakerMelAdaptation

    public init(
        mel: MLXArray,
        rawFrameCount: Int,
        adaptation: MiniCPMSpeakerMelAdaptation
    ) {
        self.mel = mel
        self.rawFrameCount = rawFrameCount
        self.adaptation = adaptation
    }
}

/// Native prompt arrays plus useful frame-count diagnostics.
public struct MiniCPMToken2WavPromptProfile {
    public let prompt: MiniCPMToken2WavPrompt
    /// The exact fixed-shape [1,500,80] tensor sent to CAM++.  Exposing this
    /// only as a profile diagnostic lets parity tooling compare the native
    /// Kaldi frontend against the original ONNX input without recomputing a
    /// second fbank implementation.
    public let speakerMel: MLXArray
    public let tokenizerMelFrameCount: Int
    public let promptTokenCount: Int
    public let promptMelFrameCount: Int
    /// Number of frames produced by the Kaldi fbank before the fixed 500-frame
    /// CAM++ adapter.  This is intentionally separate from
    /// `speakerMelFrameCount`, which remains 500 for compatibility.
    public let speakerRawMelFrameCount: Int
    public let speakerMelFrameCount: Int
    public let speakerAdaptation: MiniCPMSpeakerMelAdaptation

    public init(
        prompt: MiniCPMToken2WavPrompt,
        speakerMel: MLXArray,
        tokenizerMelFrameCount: Int,
        promptTokenCount: Int,
        promptMelFrameCount: Int,
        speakerMelFrameCount: Int,
        speakerRawMelFrameCount: Int? = nil,
        speakerAdaptation: MiniCPMSpeakerMelAdaptation = .exact
    ) {
        self.prompt = prompt
        self.speakerMel = speakerMel
        self.tokenizerMelFrameCount = tokenizerMelFrameCount
        self.promptTokenCount = promptTokenCount
        self.promptMelFrameCount = promptMelFrameCount
        self.speakerMelFrameCount = speakerMelFrameCount
        // The optional default keeps source compatibility for callers that
        // construct profiles manually while native preparation always passes
        // the true raw frame count.
        self.speakerRawMelFrameCount = speakerRawMelFrameCount ?? speakerMelFrameCount
        self.speakerAdaptation = speakerAdaptation
    }
}

/// Configuration for the MiniCPM-o reference front-end.
public struct MiniCPMToken2WavPromptPreparationConfiguration: Sendable {
    public var tokenizerSampleRate = 16_000
    public var flowSampleRate = 24_000
    public var speakerSampleRate = 16_000
    public var speakerFrameCount = 500
    public var melChannels = 80
    public var melFramesPerToken = 2

    public init() {}
}

/// Converts one mono PCM clip to the exact three-array Token2Wav prompt
/// profile.  STFT/mel extraction is implemented with Accelerate and is usable
/// without Python, Torch, ONNX Runtime, or a CoreML model.
public final class MiniCPMToken2WavPromptPreparer {
    public let configuration: MiniCPMToken2WavPromptPreparationConfiguration

    private let tokenizerMelExtractor = MiniCPMS3TokenizerMelExtractor()
    private let flowMelExtractor = MiniCPMFlowPromptMelExtractor()
    private let speakerMelExtractor = MiniCPMCamPlusPlusMelExtractor()

    public init(
        configuration: MiniCPMToken2WavPromptPreparationConfiguration = .init()
    ) {
        self.configuration = configuration
    }

    /// Resample reference PCM with the same sinc/Hann kernel used by prompt
    /// preparation.  Exposing this small deterministic helper lets parity
    /// tooling inspect the standalone 24 kHz Flow frontend without creating
    /// a fake speech-tokenizer or speaker encoder.
    public static func resampleAudio(
        _ input: [Float], from sourceRate: Int, to targetRate: Int
    ) -> [Float] {
        MiniCPMTorchaudioResampler.resample(
            input, from: sourceRate, to: targetRate)
    }

    /// Prepare a prompt from mono Float32 PCM.  `promptTokens` and
    /// `speakerEmbedding` may be supplied by a converted native encoder; when
    /// omitted, the injected encoder is called on the native mel features.
    /// Supplying neither is intentionally an error rather than a silent fake:
    /// it prevents a Torch-generated fixture from being accidentally replaced
    /// by invalid zeros in production.
    public func prepare(
        pcm: [Float],
        sampleRate: Int,
        encoder: (any MiniCPMToken2WavPromptEncoder)? = nil,
        promptTokens: MLXArray? = nil,
        speakerEmbedding: MLXArray? = nil
    ) throws -> MiniCPMToken2WavPromptProfile {
        try validatePCM(pcm, sampleRate: sampleRate)

        let tokenizerAudio = resample(
            pcm, from: sampleRate, to: configuration.tokenizerSampleRate)
        let flowAudio = resample(
            pcm, from: sampleRate, to: configuration.flowSampleRate)
        let speakerAudio = resample(
            pcm, from: sampleRate, to: configuration.speakerSampleRate)

        let tokenizerMel = tokenizerMelExtractor.extract(tokenizerAudio)
        guard tokenizerMel.ndim == 3, tokenizerMel.dim(2) > 0 else {
            throw MiniCPMToken2WavPromptPreparationError.emptyMel("tokenizer")
        }

        let tokens: MLXArray
        if let promptTokens {
            tokens = promptTokens
        } else if let encoder {
            tokens = try encoder.encodeSpeechTokens(tokenizerMel: tokenizerMel)
        } else {
            throw MiniCPMToken2WavPromptPreparationError.missingSpeechTokenizer
        }
        try validateTokens(tokens)

        let promptTokenCount = tokens.dim(1)
        let targetFlowFrames = promptTokenCount * configuration.melFramesPerToken
        let flowMel = flowMelExtractor.extract(flowAudio)
        guard flowMel.ndim == 3, flowMel.dim(2) > 0 else {
            throw MiniCPMToken2WavPromptPreparationError.emptyMel("flow")
        }
        let alignedFlowMel = alignFlowMel(
            flowMel, targetFrames: targetFlowFrames)

        let speakerExtraction = speakerMelExtractor.extractWithDiagnostics(
            speakerAudio, targetFrames: configuration.speakerFrameCount)
        let speakerMel = speakerExtraction.mel
        guard speakerMel.ndim == 3, speakerMel.dim(1) == configuration.speakerFrameCount,
              speakerMel.dim(2) == configuration.melChannels else {
            throw MiniCPMToken2WavPromptPreparationError.emptyMel("speaker")
        }

        let embedding: MLXArray
        if let speakerEmbedding {
            embedding = speakerEmbedding
        } else if let encoder {
            embedding = try encoder.encodeSpeakerEmbedding(speakerMel: speakerMel)
        } else {
            throw MiniCPMToken2WavPromptPreparationError.missingSpeakerEncoder
        }
        try validateSpeakerEmbedding(embedding)

        let prompt = MiniCPMToken2WavPrompt(
            tokens: tokens.asType(.int32),
            mel: alignedFlowMel.asType(.float32),
            speakerEmbedding: embedding.asType(.float32))
        return MiniCPMToken2WavPromptProfile(
            prompt: prompt,
            speakerMel: speakerMel,
            tokenizerMelFrameCount: tokenizerMel.dim(2),
            promptTokenCount: promptTokenCount,
            promptMelFrameCount: alignedFlowMel.dim(2),
            speakerMelFrameCount: speakerMel.dim(1),
            speakerRawMelFrameCount: speakerExtraction.rawFrameCount,
            speakerAdaptation: speakerExtraction.adaptation)
    }

    private func validatePCM(_ pcm: [Float], sampleRate: Int) throws {
        guard sampleRate > 0 else {
            throw MiniCPMToken2WavPromptPreparationError.invalidSampleRate(sampleRate)
        }
        guard !pcm.isEmpty else {
            throw MiniCPMToken2WavPromptPreparationError.invalidPCM("audio is empty")
        }
        guard pcm.allSatisfy({ $0.isFinite }) else {
            throw MiniCPMToken2WavPromptPreparationError.invalidPCM(
                "audio contains NaN or infinity")
        }
    }

    private func validateTokens(_ tokens: MLXArray) throws {
        guard tokens.ndim == 2, tokens.dim(0) == 1, tokens.dim(1) > 0 else {
            throw MiniCPMToken2WavPromptPreparationError.invalidTokens(
                "expected shape [1, tokenCount]")
        }
        guard tokens.dtype == .int32 || tokens.dtype == .int64 || tokens.dtype == .int16 else {
            throw MiniCPMToken2WavPromptPreparationError.invalidTokens(
                "expected an integer MLX array")
        }
    }

    private func validateSpeakerEmbedding(_ embedding: MLXArray) throws {
        guard embedding.ndim == 2,
              embedding.dim(0) == 1,
              embedding.dim(1) == 192 else {
            throw MiniCPMToken2WavPromptPreparationError.invalidSpeakerEmbedding(
                "expected shape [1, 192]")
        }
        let values = embedding.asType(.float32).asArray(Float.self)
        guard values.allSatisfy({ $0.isFinite }) else {
            throw MiniCPMToken2WavPromptPreparationError.invalidSpeakerEmbedding(
                "contains NaN or infinity")
        }
    }

    private func alignFlowMel(_ mel: MLXArray, targetFrames: Int) -> MLXArray {
        let channels = mel.dim(1)
        let sourceFrames = mel.dim(2)
        let source = mel.asType(.float32).asArray(Float.self)
        var output = [Float](repeating: 0, count: channels * targetFrames)
        let copiedFrames = Swift.min(sourceFrames, targetFrames)
        for channel in 0..<channels {
            for frame in 0..<copiedFrames {
                output[channel * targetFrames + frame] =
                    source[channel * sourceFrames + frame]
            }
            if copiedFrames < targetFrames {
                let fill: Float = copiedFrames > 0
                    ? source[channel * sourceFrames + copiedFrames - 1] : 0
                for frame in copiedFrames..<targetFrames {
                    output[channel * targetFrames + frame] = fill
                }
            }
        }
        return MLXArray(output, [1, channels, targetFrames])
    }

    /// Band-limited resampler matching torchaudio.functional.resample's
    /// default `sinc_interp_hann` configuration.  MiniCPM's official prompt
    /// path calls torchaudio for all three rates (16 kHz tokenizer/CAM++ and
    /// 24 kHz flow), so linear interpolation is not parity-safe for a
    /// non-16-kHz reference clip.
    private func resample(_ input: [Float], from sourceRate: Int, to targetRate: Int)
        -> [Float]
    {
        Self.resampleAudio(input, from: sourceRate, to: targetRate)
    }
}

/// Small scalar implementation of torchaudio's polyphase
/// `sinc_interp_hann` resampler.  The prompt reference is short (normally a
/// few seconds), so keeping this deterministic implementation local avoids a
/// dependency on a second audio framework while preserving the official
/// MiniCPM sample-rate conversion semantics.
private enum MiniCPMTorchaudioResampler {
    static func resample(
        _ input: [Float],
        from sourceRate: Int,
        to targetRate: Int,
        lowpassFilterWidth: Int = 6,
        rolloff: Float = 0.99
    ) -> [Float] {
        guard sourceRate > 0, targetRate > 0, !input.isEmpty else {
            return input
        }
        guard sourceRate != targetRate else { return input }

        let divisor = gcd(sourceRate, targetRate)
        let original = sourceRate / divisor
        let target = targetRate / divisor
        let baseFrequency = Float(min(original, target)) * rolloff
        let width = Int(
            ceil(Float(lowpassFilterWidth * original) / baseFrequency))
        let kernelLength = 2 * width + original
        let scale = baseFrequency / Float(original)
        let pi = Float.pi

        // Build the same phase-major kernel as torchaudio's
        // `_get_sinc_resample_kernel`: [target, 1, kernelLength].
        var kernel = [Float](repeating: 0, count: target * kernelLength)
        for phase in 0..<target {
            let phaseOffset = -Float(phase) / Float(target)
            for tap in 0..<kernelLength {
                let index = Float(-width + tap) / Float(original)
                var time = (phaseOffset + index) * baseFrequency
                time = max(-Float(lowpassFilterWidth),
                           min(Float(lowpassFilterWidth), time))
                let window = cos(time * pi / Float(lowpassFilterWidth) / 2)
                let angle = time * pi
                let sinc = angle == 0 ? Float(1) : sin(angle) / angle
                kernel[phase * kernelLength + tap] =
                    sinc * window * window * scale
            }
        }

        // `conv1d` in torchaudio pads `(width, width + original)`, strides by
        // `original`, then interleaves all `target` phases.  Writing that
        // convolution explicitly keeps the boundary zero-padding and tap
        // accumulation order clear and avoids introducing a second MLX graph
        // into prompt construction.
        let outputCount = Int(
            ceil(Double(target) * Double(input.count) / Double(original)))
        guard outputCount > 0 else { return [] }
        let convolutionSteps = input.count / original + 1
        var output = [Float](repeating: 0, count: convolutionSteps * target)
        for step in 0..<convolutionSteps {
            let baseInputIndex = step * original - width
            for phase in 0..<target {
                let kernelBase = phase * kernelLength
                var value: Float = 0
                for tap in 0..<kernelLength {
                    let inputIndex = baseInputIndex + tap
                    guard inputIndex >= 0, inputIndex < input.count else {
                        continue
                    }
                    value += input[inputIndex] * kernel[kernelBase + tap]
                }
                output[step * target + phase] = value
            }
        }
        if output.count > outputCount {
            output.removeLast(output.count - outputCount)
        }
        return output
    }

    private static func gcd(_ lhs: Int, _ rhs: Int) -> Int {
        var a = abs(lhs)
        var b = abs(rhs)
        while b != 0 {
            let remainder = a % b
            a = b
            b = remainder
        }
        return max(a, 1)
    }
}

/// S3Tokenizer-v2's 128-bin Whisper-style front-end.  The ONNX tokenizer
/// consumes `[1, 128, T]` at 100 Hz and emits 25 Hz codes after two stride-2
/// convolutions; its graph is intentionally kept separate from the flow mel.
public final class MiniCPMS3TokenizerMelExtractor {
    public let sampleRate = 16_000
    public let nFFT = 400
    public let hopLength = 160
    public let nMels = 128

    private let realDFT: MiniCPMRealDFT
    private var window: [Float]
    private var filterbank: [Float]

    public init() {
        let fftLength = 400
        let sampleRate = 16_000
        let melCount = 128
        window = (0..<fftLength).map {
            0.5 * (1 - cos(2 * Float.pi * Float($0) / Float(fftLength)))
        }
        realDFT = MiniCPMRealDFT(size: fftLength)
        filterbank = MiniCPMSTFTFilterbank.make(
            sampleRate: sampleRate, paddedFFT: fftLength,
            nMels: melCount, fMin: 0, fMax: 8_000, scale: .slaney)
    }

    public func extract(_ audio: [Float]) -> MLXArray {
        let framesAll = MiniCPMSTFTKernel.frames(
            audioCount: audio.count, nFFT: nFFT, hop: hopLength,
            centerPad: nFFT / 2)
        let frames = max(framesAll - 1, 0) // s3tokenizer drops stft[..., :-1]
        let power = MiniCPMSTFTKernel.power(
            audio: audio, nFFT: nFFT, hop: hopLength,
            centerPad: nFFT / 2, realDFT: realDFT,
            window: window, frameCount: frames)
        var mel = MiniCPMSTFTKernel.applyFilterbank(
            power: power, frameCount: frames, bins: nFFT / 2 + 1,
            channels: nMels, filterbank: filterbank)
        let count = mel.count
        if count > 0 {
            var n = Int32(count)
            var lo: Float = 1e-10
            var hi = Float.greatestFiniteMagnitude
            vDSP_vclip(mel, 1, &lo, &hi, &mel, 1, vDSP_Length(count))
            vvlog10f(&mel, mel, &n)
            var maxValue: Float = -.infinity
            vDSP_maxv(mel, 1, &maxValue, vDSP_Length(count))
            var minValue = maxValue - 8
            vDSP_vclip(mel, 1, &minValue, &hi, &mel, 1, vDSP_Length(count))
            var add: Float = 4
            var div: Float = 4
            vDSP_vsadd(mel, 1, &add, &mel, 1, vDSP_Length(count))
            vDSP_vsdiv(mel, 1, &div, &mel, 1, vDSP_Length(count))
        }
        // Kernel layout is [frame, mel]. MLX tokenizer expects [1, mel, frame].
        var channelsFirst = [Float](repeating: 0, count: nMels * frames)
        if frames > 0 {
            vDSP_mtrans(mel, 1, &channelsFirst, 1,
                        vDSP_Length(nMels), vDSP_Length(frames))
        }
        return MLXArray(channelsFirst, [1, nMels, frames])
    }
}

/// CosyVoice/MiniCPM Token2Wav prompt mel front-end at 24 kHz and 50 Hz.
public final class MiniCPMFlowPromptMelExtractor {
    public let sampleRate = 24_000
    public let nFFT = 1_920
    public let hopLength = 480
    public let nMels = 80

    private var dftSetup: vDSP_DFT_Setup
    private var window: [Float]
    private var filterbank: [Float]

    public init() {
        let fftLength = 1_920
        let sampleRate = 24_000
        let melCount = 80
        window = (0..<fftLength).map {
            0.5 * (1 - cos(2 * Float.pi * Float($0) / Float(fftLength)))
        }
        guard let setup = vDSP_DFT_zop_CreateSetup(
            nil, vDSP_Length(fftLength), .FORWARD) else {
            fatalError("unable to create flow prompt FFT setup")
        }
        dftSetup = setup
        filterbank = MiniCPMSTFTFilterbank.make(
            sampleRate: sampleRate, paddedFFT: fftLength,
            nMels: melCount, fMin: 0, fMax: Float(sampleRate / 2), scale: .slaney)
    }

    deinit { vDSP_DFT_DestroySetup(dftSetup) }

    public func extract(_ audio: [Float]) -> MLXArray {
        // ``torch.stft(center=True)`` pads by n_fft / 2 on both sides.  The
        // hop-sized padding used by the old streaming path drops the final
        // prompt frame (for example 12 instead of 13 frames for a 250 ms
        // clip), which also shifts every Flow-prompt alignment downstream.
        let pad = nFFT / 2
        let frameCount = MiniCPMSTFTKernel.frames(
            audioCount: audio.count, nFFT: nFFT, hop: hopLength, centerPad: pad)
        let magnitude = MiniCPMSTFTKernel.magnitude(
            audio: audio, nFFT: nFFT, hop: hopLength, centerPad: pad,
            dftSetup: dftSetup, window: window,
            frameCount: frameCount)
        var mel = MiniCPMSTFTKernel.applyFilterbank(
            power: magnitude, frameCount: frameCount,
            bins: nFFT / 2 + 1, channels: nMels, filterbank: filterbank)
        let count = mel.count
        if count > 0 {
            var n = Int32(count)
            var lo: Float = 1e-5
            var hi = Float.greatestFiniteMagnitude
            vDSP_vclip(mel, 1, &lo, &hi, &mel, 1, vDSP_Length(count))
            vvlogf(&mel, mel, &n)
        }
        var channelsFirst = [Float](repeating: 0, count: nMels * frameCount)
        if frameCount > 0 {
            vDSP_mtrans(mel, 1, &channelsFirst, 1,
                        vDSP_Length(nMels), vDSP_Length(frameCount))
        }
        return MLXArray(channelsFirst, [1, nMels, frameCount])
    }
}

/// CAM++ Kaldi-compatible 80-bin fbank front-end.  The resulting tensor is
/// `[1, targetFrames, 80]`, ready for a CoreML/MLX CAM++ adapter.
public final class MiniCPMCamPlusPlusMelExtractor {
    public let sampleRate = 16_000
    public let nFFT = 400
    public let hopLength = 160
    public let nMels = 80

    private let realDFT: MiniCPMRealDFT
    private var window: [Float]
    private var filterbank: [Float]

    public init() {
        let fftLength = 400
        let sampleRate = 16_000
        let melCount = 80
        window = (0..<fftLength).map {
            let h = 0.5 - 0.5 * cos(2 * Float.pi * Float($0) / Float(fftLength - 1))
            return pow(h, 0.85)
        }
        // Match Kaldi's round-to-power-of-two FFT, while retaining the
        // 400-sample analysis window above.
        realDFT = MiniCPMRealDFT(size: 512)
        // 3D-Speaker's campplus.onnx is fed by
        // ``torchaudio.compliance.kaldi.fbank``.  That is not the HTK/Slaney
        // frontend used by the Whisper and CosyVoice paths above: Kaldi uses
        // a 20 Hz lower cutoff, natural-mel (1127*ln(1+f/700)) triangular
        // banks without area normalisation, rounds the 400-point window to a
        // 512-point FFT by default, discards the Nyquist column, and floors
        // each power bin at Float.ulpOfOne before taking ln.
        filterbank = Self.makeKaldiFilterbank(
            sampleRate: sampleRate, fftSize: 512, melBins: melCount,
            lowFrequency: 20, highFrequency: Float(sampleRate / 2))
    }

    public func extract(_ audio: [Float], targetFrames: Int = 500) -> MLXArray {
        extractWithDiagnostics(audio, targetFrames: targetFrames).mel
    }

    /// Extract and report the variable-length fbank count and fixed-shape
    /// adaptation.  The plain `extract` API remains for existing callers.
    public func extractWithDiagnostics(
        _ audio: [Float], targetFrames: Int = 500
    ) -> MiniCPMCamPlusPlusMelExtraction {
        guard !audio.isEmpty, targetFrames > 0 else {
            return MiniCPMCamPlusPlusMelExtraction(
                mel: MLXArray.zeros([1, max(targetFrames, 0), nMels]),
                rawFrameCount: 0,
                adaptation: .emptyPadded)
        }
        // Kaldi's default ``snip_edges=true`` emits only complete 25 ms
        // windows.  In particular, a one-second clip has 98 frames rather
        // than the 101 centered/reflect-padded frames used by Whisper.
        let rawFrames = audio.count >= nFFT
            ? 1 + (audio.count - nFFT) / hopLength : 0
        guard rawFrames > 0 else {
            return MiniCPMCamPlusPlusMelExtraction(
                mel: MLXArray.zeros([1, targetFrames, nMels]),
                rawFrameCount: 0,
                adaptation: .emptyPadded)
        }

        // torchaudio's Kaldi implementation keeps a 400-sample analysis
        // window but zero-pads it to the next power of two (512) before the
        // real FFT.  The resulting mel bank therefore has 257 columns,
        // including a zero-weight Nyquist column.
        let paddedFFT = 512
        let paddedBins = paddedFFT / 2 + 1
        var power = [Float](repeating: 0, count: rawFrames * paddedBins)
        var frame = [Float](repeating: 0, count: paddedFFT)
        var real = [Float](repeating: 0, count: paddedBins)
        var imaginary = [Float](repeating: 0, count: paddedBins)
        for index in 0..<rawFrames {
            let start = index * hopLength
            var mean: Float = 0
            for sample in 0..<nFFT { mean += audio[start + sample] }
            mean /= Float(nFFT)
            var previous: Float = 0
            for sample in 0..<nFFT {
                let centered = audio[start + sample] - mean
                // Kaldi pads the first sample by replication before
                // pre-emphasis, so its first value is 0.03 * centered.
                let emphasized = centered - 0.97 * (sample == 0 ? centered : previous)
                frame[sample] = emphasized * window[sample]
                previous = centered
            }
            // The frame array is reused for every analysis window; clear the
            // zero-padding tail explicitly so no previous frame can leak into
            // the 512-point FFT.
            for sample in nFFT..<paddedFFT { frame[sample] = 0 }
            realDFT.power(frame, real: &real, imaginary: &imaginary)
            let base = index * paddedBins
            for bin in 0..<paddedBins {
                power[base + bin] = real[bin] * real[bin] + imaginary[bin] * imaginary[bin]
            }
        }

        var mel = MiniCPMSTFTKernel.applyFilterbank(
            power: power, frameCount: rawFrames,
            bins: paddedBins, channels: nMels, filterbank: filterbank)
        let count = mel.count
        if count > 0 {
            var n = Int32(count)
            // torchaudio.compliance.kaldi.fbank uses the dtype epsilon
            // (`std::numeric_limits<float>::epsilon()`) as the per-bin
            // spectral floor.  This is deliberately larger than the
            // 1e-10 floor used by the Whisper/flow front-ends: quiet CAM++
            // bands must become log(epsilon), otherwise their values drift
            // by several dB and the subsequent CMN no longer matches Kaldi.
            var lo: Float = Float.ulpOfOne
            var hi = Float.greatestFiniteMagnitude
            vDSP_vclip(mel, 1, &lo, &hi, &mel, 1, vDSP_Length(count))
            vvlogf(&mel, mel, &n)
            // CMN over time, matching kaldi.fbank(...); feat - feat.mean(0).
            var means = [Float](repeating: 0, count: nMels)
            for frame in 0..<rawFrames {
                for channel in 0..<nMels { means[channel] += mel[frame * nMels + channel] }
            }
            let inv = 1 / Float(max(rawFrames, 1))
            for channel in 0..<nMels { means[channel] *= inv }
            for frame in 0..<rawFrames {
                for channel in 0..<nMels { mel[frame * nMels + channel] -= means[channel] }
            }
        }

        // CAM++ consumes a fixed [1, 500, 80] input. Short clips tile rather
        // than zero-pad; long clips use the center 500 frames.
        var output = [Float](repeating: 0, count: targetFrames * nMels)
        if rawFrames > 0 {
            let offset = rawFrames >= targetFrames ? (rawFrames - targetFrames) / 2 : 0
            for frame in 0..<targetFrames {
                let source = rawFrames >= targetFrames
                    ? offset + frame : frame % rawFrames
                for channel in 0..<nMels {
                    output[frame * nMels + channel] = mel[source * nMels + channel]
                }
            }
        }
        let adaptation: MiniCPMSpeakerMelAdaptation
        if rawFrames == targetFrames {
            adaptation = .exact
        } else if rawFrames < targetFrames {
            adaptation = .tiledShort
        } else {
            adaptation = .centerCroppedLong
        }
        return MiniCPMCamPlusPlusMelExtraction(
            mel: MLXArray(output, [1, targetFrames, nMels]),
            rawFrameCount: rawFrames,
            adaptation: adaptation)
    }

    private static func makeKaldiFilterbank(
        sampleRate: Int,
        fftSize: Int,
        melBins: Int,
        lowFrequency: Float,
        highFrequency: Float
    ) -> [Float] {
        let bins = fftSize / 2 + 1
        let fftBinWidth = Float(sampleRate) / Float(fftSize)
        let melScale: (Float) -> Float = { frequency in
            1127 * log(1 + frequency / 700)
        }
        let lowMel = melScale(lowFrequency)
        let highMel = melScale(highFrequency)
        let delta = (highMel - lowMel) / Float(melBins + 1)
        var bank = [Float](repeating: 0, count: melBins * bins)
        for mel in 0..<melBins {
            let left = lowMel + Float(mel) * delta
            let center = lowMel + Float(mel + 1) * delta
            let right = lowMel + Float(mel + 2) * delta
            for bin in 0..<bins {
                // Kaldi's get_mel_banks computes only num_fft_bins = N/2
                // columns and pads the Nyquist column with zero.
                guard bin < fftSize / 2 else { continue }
                let value = melScale(Float(bin) * fftBinWidth)
                let rising = (value - left) / (center - left)
                let falling = (right - value) / (right - center)
                bank[mel * bins + bin] = max(0, min(rising, falling))
            }
        }
        return bank
    }
}

private enum MiniCPMMelScale { case slaney, htk }

private enum MiniCPMSTFTFilterbank {
    static func make(
        sampleRate: Int, paddedFFT: Int, nMels: Int,
        fMin: Float, fMax: Float, scale: MiniCPMMelScale
    ) -> [Float] {
        let bins = paddedFFT / 2 + 1
        let hzToMel: (Float) -> Float = { hz in
            switch scale {
            case .htk: return 2595 * log10(1 + hz / 700)
            case .slaney:
                let fSp: Float = 200 / 3
                let minLogHz: Float = 1000
                let minLogMel = minLogHz / fSp
                let logStep: Float = log(6.4) / 27
                return hz < minLogHz ? hz / fSp : minLogMel + log(hz / minLogHz) / logStep
            }
        }
        let melToHz: (Float) -> Float = { mel in
            switch scale {
            case .htk: return 700 * (pow(10, mel / 2595) - 1)
            case .slaney:
                let fSp: Float = 200 / 3
                let minLogMel: Float = 15
                let logStep: Float = log(6.4) / 27
                return mel < minLogMel ? fSp * mel : 1000 * exp(logStep * (mel - minLogMel))
            }
        }
        let lo = hzToMel(fMin)
        let hi = hzToMel(fMax)
        let points = (0..<(nMels + 2)).map { i in
            melToHz(lo + Float(i) / Float(nMels + 1) * (hi - lo))
        }
        var output = [Float](repeating: 0, count: nMels * bins)
        for mel in 0..<nMels {
            let left = points[mel]
            let center = points[mel + 1]
            let right = points[mel + 2]
            let norm: Float = 2 / max(right - left, 1e-12)
            for bin in 0..<bins {
                let hz = Float(bin) * Float(sampleRate) / Float(paddedFFT)
                var value: Float = 0
                if hz >= left && hz <= center {
                    value = (hz - left) / max(center - left, 1e-12)
                } else if hz > center && hz <= right {
                    value = (right - hz) / max(right - center, 1e-12)
                }
                output[mel * bins + bin] = value * norm
            }
        }
        return output
    }
}

/// Real-input DFT fallback for Apple's vDSP, whose optimized DFT setup does
/// not support N=400 on current macOS.  The tokenizer/CAM++ front-ends use it
/// to preserve Torch's native FFT bins without relying on platform FFT size
/// heuristics.  Twiddles are precomputed once; each dot product is vectorized by
/// Accelerate and is fast enough for short reference clips.
private struct MiniCPMRealDFT {
    let size: Int
    let bins: Int
    let cosine: [Float]
    let negativeSine: [Float]

    init(size: Int) {
        self.size = size
        bins = size / 2 + 1
        var cosines = [Float](repeating: 0, count: bins * size)
        var sines = [Float](repeating: 0, count: bins * size)
        for bin in 0..<bins {
            for sample in 0..<size {
                let angle = 2 * Float.pi * Float(bin * sample) / Float(size)
                cosines[bin * size + sample] = cos(angle)
                sines[bin * size + sample] = -sin(angle)
            }
        }
        cosine = cosines
        negativeSine = sines
    }

    func power(_ frame: [Float], real: inout [Float], imaginary: inout [Float]) {
        for bin in 0..<bins {
            var re: Float = 0
            var im: Float = 0
            frame.withUnsafeBufferPointer { frameBuffer in
                cosine.withUnsafeBufferPointer { cosineBuffer in
                    negativeSine.withUnsafeBufferPointer { sineBuffer in
                        vDSP_dotpr(
                            frameBuffer.baseAddress!, 1,
                            cosineBuffer.baseAddress! + bin * size, 1,
                            &re, vDSP_Length(size))
                        vDSP_dotpr(
                            frameBuffer.baseAddress!, 1,
                            sineBuffer.baseAddress! + bin * size, 1,
                            &im, vDSP_Length(size))
                    }
                }
            }
            real[bin] = re
            imaginary[bin] = im
        }
    }
}

private enum MiniCPMSTFTKernel {
    static func frames(audioCount: Int, nFFT: Int, hop: Int, centerPad: Int) -> Int {
        guard audioCount > 0 else { return 0 }
        let count = centerPad * 2 + audioCount
        return count >= nFFT ? (count - nFFT) / hop + 1 : 0
    }

    static func paddedAudio(_ audio: [Float], pad: Int) -> [Float] {
        guard !audio.isEmpty else { return [] }
        var output = [Float](repeating: 0, count: audio.count + 2 * pad)
        for i in 0..<pad { output[i] = audio[min(pad - i, audio.count - 1)] }
        output.replaceSubrange(pad..<(pad + audio.count), with: audio)
        for i in 0..<pad {
            let source = max(audio.count - 2 - i, 0)
            output[pad + audio.count + i] = audio[source]
        }
        return output
    }

    static func power(
        audio: [Float], nFFT: Int, hop: Int, centerPad: Int,
        realDFT: MiniCPMRealDFT, window: [Float], frameCount: Int
    ) -> [Float] {
        let padded = paddedAudio(audio, pad: centerPad)
        let bins = nFFT / 2 + 1
        var windowed = [Float](repeating: 0, count: nFFT)
        var real = [Float](repeating: 0, count: bins)
        var imaginary = [Float](repeating: 0, count: bins)
        var output = [Float](repeating: 0, count: frameCount * bins)
        for t in 0..<frameCount {
            let start = t * hop
            if start + nFFT > padded.count { break }
            padded.withUnsafeBufferPointer { ptr in
                vDSP_vmul(ptr.baseAddress! + start, 1, window, 1,
                          &windowed, 1, vDSP_Length(nFFT))
            }
            realDFT.power(windowed, real: &real, imaginary: &imaginary)
            let base = t * bins
            for bin in 0..<bins {
                output[base + bin] = real[bin] * real[bin] + imaginary[bin] * imaginary[bin]
            }
        }
        return output
    }

    static func power(
        audio: [Float], nFFT: Int, hop: Int, centerPad: Int,
        dftSetup: vDSP_DFT_Setup, window: [Float], frameCount: Int
    ) -> [Float] {
        let padded = paddedAudio(audio, pad: centerPad)
        let bins = nFFT / 2 + 1
        var realInput = [Float](repeating: 0, count: nFFT)
        var imagInput = [Float](repeating: 0, count: nFFT)
        var realOutput = [Float](repeating: 0, count: nFFT)
        var imagOutput = [Float](repeating: 0, count: nFFT)
        var output = [Float](repeating: 0, count: frameCount * bins)
        for t in 0..<frameCount {
            let start = t * hop
            if start + nFFT > padded.count { break }
            padded.withUnsafeBufferPointer { ptr in
                vDSP_vmul(ptr.baseAddress! + start, 1, window, 1,
                          &realInput, 1, vDSP_Length(nFFT))
            }
            vDSP_vclr(&imagInput, 1, vDSP_Length(nFFT))
            vDSP_DFT_Execute(
                dftSetup, realInput, imagInput, &realOutput, &imagOutput)
            let base = t * bins
            for k in 0..<bins {
                output[base + k] =
                    realOutput[k] * realOutput[k] + imagOutput[k] * imagOutput[k]
            }
        }
        return output
    }

    static func magnitude(
        audio: [Float], nFFT: Int, hop: Int, centerPad: Int,
        dftSetup: vDSP_DFT_Setup, window: [Float], frameCount: Int
    ) -> [Float] {
        let powerValues = power(
            audio: audio, nFFT: nFFT, hop: hop, centerPad: centerPad,
            dftSetup: dftSetup, window: window,
            frameCount: frameCount)
        // `power` stores |FFT|² with the vDSP scale correction. Convert to
        // magnitude for the flow front-end (which uses magnitude, not power).
        return powerValues.map { sqrt(max($0, 0) + 1e-9) }
    }

    static func applyFilterbank(
        power: [Float], frameCount: Int, bins: Int,
        channels: Int, filterbank: [Float]
    ) -> [Float] {
        guard frameCount > 0 else { return [] }
        var transposed = [Float](repeating: 0, count: bins * channels)
        vDSP_mtrans(filterbank, 1, &transposed, 1,
                    vDSP_Length(bins), vDSP_Length(channels))
        var output = [Float](repeating: 0, count: frameCount * channels)
        vDSP_mmul(power, 1, transposed, 1, &output, 1,
                  vDSP_Length(frameCount), vDSP_Length(channels), vDSP_Length(bins))
        return output
    }
}
