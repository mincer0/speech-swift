import Foundation
import MLX
import MLXNN

public struct MiniCPMHiFTOutput {
    public let waveform: MLXArray
    public let source: MLXArray
    public let f0: MLXArray
    public let intermediates: [String: MLXArray]

    public init(
        waveform: MLXArray,
        source: MLXArray,
        f0: MLXArray,
        intermediates: [String: MLXArray]
    ) {
        self.waveform = waveform
        self.source = source
        self.f0 = f0
        self.intermediates = intermediates
    }
}

/// Linear interpolation with PyTorch's `align_corners=False` coordinates.
/// Input and output are NLC; only the time axis is resized.
///
/// Keeping this as an explicit index/weight operation (instead of relying on
/// a framework resize primitive) matters for HiFT parity: both the initial
/// sample-rate F0 downsample and the cumulative-phase upsample use the exact
/// half-pixel coordinates chosen by `torch.nn.functional.interpolate`.
func miniCPMLinearResize(_ input: MLXArray, outputLength: Int) -> MLXArray {
    precondition(input.ndim == 3, "linear resize expects an NLC tensor")
    precondition(input.dim(1) > 0 && outputLength > 0)
    let inputLength = input.dim(1)
    guard inputLength != outputLength else { return input }
    var leftIndices = [Int32](repeating: 0, count: outputLength)
    var rightIndices = [Int32](repeating: 0, count: outputLength)
    var rightWeights = [Float](repeating: 0, count: outputLength)

    for outputIndex in 0 ..< outputLength {
        let coordinate =
            (Float(outputIndex) + 0.5) * Float(inputLength) / Float(outputLength) - 0.5
        let unclampedLeft = Int(floor(coordinate))
        let left = Swift.max(0, Swift.min(inputLength - 1, unclampedLeft))
        let right = Swift.max(0, Swift.min(inputLength - 1, unclampedLeft + 1))
        leftIndices[outputIndex] = Int32(left)
        rightIndices[outputIndex] = Int32(right)
        rightWeights[outputIndex] = right == left ? 0 : coordinate - Float(unclampedLeft)
    }

    let left = input.take(MLXArray(leftIndices), axis: 1)
    let right = input.take(MLXArray(rightIndices), axis: 1)
    let weight = MLXArray(rightWeights).reshaped([1, outputLength, 1])
    return left * (1 - weight) + right * weight
}

func miniCPMLinearUpsample(_ input: MLXArray, scale: Int) -> MLXArray {
    precondition(scale > 0)
    return miniCPMLinearResize(input, outputLength: input.dim(1) * scale)
}

final class MiniCPMSineGenerator2 {
    let configuration: MiniCPMHiFTConfiguration

    init(configuration: MiniCPMHiFTConfiguration) {
        self.configuration = configuration
    }

    /// MiniCPM/CosyVoice2 SineGen2. Input F0 is at the mel/frame rate.
    /// An optional standard-normal noise tensor and initial harmonic phase
    /// make cross-runtime parity exact.
    func callAsFunction(
        _ f0: MLXArray,
        noise: MLXArray? = nil,
        initialPhase: MLXArray? = nil
    ) -> MLXArray {
        precondition(f0.ndim == 2 && f0.dim(1) > 0, "SineGen2 expects [B, frames] F0")
        let harmonics = configuration.harmonicCount + 1
        let scale = configuration.sourceUpsampleScale
        let batch = f0.dim(0)
        let frameCount = f0.dim(1)
        let sampleCount = frameCount * scale
        let multipliers = MLXArray((1 ... harmonics).map(Float.init))
            .reshaped([1, 1, harmonics])

        // HiFT first nearest-upsamples the frame-rate F0 to the vocoder sample
        // rate.  SineGen2 then linearly downsamples the *phase increments*,
        // accumulates at frame rate, and linearly upsamples the cumulative
        // phase.  Do not collapse this to a frame staircase: the two linear
        // interpolation boundaries are observable in the trained source.
        let f0Sample = MLX.repeated(
            f0.asType(.float32).expandedDimensions(axis: 2),
            count: scale,
            axis: 1)
        let radians = f0Sample * multipliers / Float(configuration.sampleRate)
        var wrapped = radians - floor(radians)

        // PyTorch's SineGen2 adds one random [0, 1) cycle to the first sample
        // of each overtone and pins the fundamental's initial phase to zero.
        // Accepting that tensor from the caller lets the exporter replay the
        // exact PyTorch draw; the default keeps native inference stochastic.
        let providedPhase: MLXArray
        if let initialPhase {
            precondition(
                (initialPhase.ndim == 2 && initialPhase.shape == [batch, harmonics])
                    || (initialPhase.ndim == 3
                        && initialPhase.shape == [batch, 1, harmonics]),
                "initial phase must have shape [B, H] or [B, 1, H]")
            providedPhase = initialPhase.ndim == 2
                ? initialPhase.expandedDimensions(axis: 1)
                : initialPhase
        } else {
            providedPhase = MLXRandom.uniform(
                0 ..< 1, [batch, 1, harmonics], key: MLXRandom.key(1))
        }
        let fundamentalMask = MLXArray(
            ([Float](repeating: 1, count: harmonics).enumerated().map {
                $0.offset == 0 ? 0 : $0.element
            })
        ).reshaped([1, 1, harmonics])
        let phaseOffset = providedPhase.asType(.float32) * fundamentalMask
        let firstRadians = wrapped[0..., 0 ..< 1, 0...] + phaseOffset
        if sampleCount == 1 {
            wrapped = firstRadians
        } else {
            wrapped = concatenated([
                firstRadians,
                wrapped[0..., 1..., 0...],
            ], axis: 1)
        }

        let frameRadians = miniCPMLinearResize(wrapped, outputLength: frameCount)
        let framePhase = cumsum(frameRadians, axis: 1) * Float(2 * Float.pi)
        let phase = miniCPMLinearResize(
            framePhase * Float(scale), outputLength: sampleCount)

        // Voicing is derived from the nearest-upsampled F0, exactly as in the
        // reference implementation.  It is therefore constant within each
        // frame block but still has sample-rate shape for noise mixing.
        let uv = MLX.where(
            f0Sample .> configuration.voicedThreshold,
            MLXArray(Float(1)), MLXArray(Float(0)))
        let noiseAmplitude = uv * configuration.noiseStandardDeviation
            + (1 - uv) * (configuration.sineAmplitude / 3)
        let standardNoise = noise ?? MLXRandom.normal(
            [batch, sampleCount, harmonics],
            key: MLXRandom.key(0))
        return sin(phase) * configuration.sineAmplitude * uv
            + standardNoise * noiseAmplitude
    }
}

final class MiniCPMSourceModuleHnNSF2: Module {
    let sineGenerator: MiniCPMSineGenerator2
    @ModuleInfo(key: "l_linear") var linear: Linear

    init(configuration: MiniCPMHiFTConfiguration) {
        sineGenerator = MiniCPMSineGenerator2(configuration: configuration)
        _linear.wrappedValue = Linear(configuration.harmonicCount + 1, 1)
        super.init()
    }

    /// Returns NCL source excitation [B, 1, samples].
    func callAsFunction(
        _ f0: MLXArray,
        noise: MLXArray? = nil,
        initialPhase: MLXArray? = nil
    ) -> MLXArray {
        tanh(linear(sineGenerator(
            f0, noise: noise, initialPhase: initialPhase))).transposed(0, 2, 1)
    }
}

final class MiniCPMF0Predictor: Module {
    @ModuleInfo var condnet: [MiniCPMConv1d]
    @ModuleInfo var classifier: Linear

    init(inputChannels: Int, conditionChannels: Int = 512) {
        var layers: [MiniCPMConv1d] = []
        for index in 0 ..< 5 {
            layers.append(MiniCPMConv1d(
                inputChannels: index == 0 ? inputChannels : conditionChannels,
                outputChannels: conditionChannels,
                kernelSize: 3,
                padding: 1))
        }
        _condnet.wrappedValue = layers
        _classifier.wrappedValue = Linear(conditionChannels, 1)
        super.init()
    }

    func callAsFunction(_ mel: MLXArray) -> MLXArray {
        var value = mel
        for convolution in condnet {
            value = miniCPMELU(convolution(value))
        }
        value = classifier(value.transposed(0, 2, 1))
        return abs(value.squeezed(axis: 2))
    }
}

private func periodicHannWindow(_ size: Int) -> [Float] {
    (0 ..< size).map {
        Float(0.5 * (1 - cos(2 * Double.pi * Double($0) / Double(size))))
    }
}

func miniCPMSTFT(
    _ signal: MLXArray, fftSize: Int, hopLength: Int
) -> (real: MLXArray, imaginary: MLXArray) {
    let batch = signal.dim(0)
    let signalLength = signal.dim(1)
    let bins = fftSize / 2 + 1
    let window = MLXArray(periodicHannWindow(fftSize))

    var realMatrix = [Float](repeating: 0, count: bins * fftSize)
    var imaginaryMatrix = [Float](repeating: 0, count: bins * fftSize)
    for frequency in 0 ..< bins {
        for sample in 0 ..< fftSize {
            let angle = 2 * Double.pi * Double(frequency * sample) / Double(fftSize)
            realMatrix[frequency * fftSize + sample] = Float(cos(angle))
            imaginaryMatrix[frequency * fftSize + sample] = Float(-sin(angle))
        }
    }

    var padded = signal
    let centerPadding = fftSize / 2
    if centerPadding > 0 {
        let leftIndices = MLXArray((1 ... centerPadding).reversed().map(Int32.init))
        let rightIndices = MLXArray(
            ((signalLength - 1 - centerPadding) ..< (signalLength - 1))
                .reversed().map(Int32.init))
        padded = concatenated([
            signal.take(leftIndices, axis: 1), signal,
            signal.take(rightIndices, axis: 1),
        ], axis: 1)
    }
    if padded.dim(1) < fftSize {
        padded = concatenated([
            padded, MLXArray.zeros([batch, fftSize - padded.dim(1)]),
        ], axis: 1)
    }

    let frameCount = Swift.max((padded.dim(1) - fftSize) / hopLength + 1, 1)
    let frames = (0 ..< frameCount).map { index in
        padded[0..., (index * hopLength) ..< (index * hopLength + fftSize)]
    }
    let windowed = stacked(frames, axis: 1) * window
    let realDFT = MLXArray(realMatrix).reshaped([bins, fftSize])
    let imaginaryDFT = MLXArray(imaginaryMatrix).reshaped([bins, fftSize])
    let real = matmul(windowed, realDFT.transposed()).transposed(0, 2, 1)
    let imaginary = matmul(windowed, imaginaryDFT.transposed()).transposed(0, 2, 1)
    return (real, imaginary)
}

func miniCPMISTFT(
    magnitude: MLXArray,
    phase: MLXArray,
    fftSize: Int,
    hopLength: Int
) -> MLXArray {
    let batch = magnitude.dim(0)
    let bins = fftSize / 2 + 1
    let frameCount = magnitude.dim(2)
    let clippedMagnitude = minimum(
        magnitude.transposed(0, 2, 1), MLXArray(Float(100)))
    let phaseNLC = phase.transposed(0, 2, 1)
    let real = clippedMagnitude * cos(phaseNLC)
    let imaginary = clippedMagnitude * sin(phaseNLC)

    let mirrorIndices = MLXArray((1 ... (bins - 2)).reversed().map(Int32.init))
    let fullReal = concatenated([real, real.take(mirrorIndices, axis: 2)], axis: 2)
    let fullImaginary = concatenated([
        imaginary, -imaginary.take(mirrorIndices, axis: 2),
    ], axis: 2)

    var inverseCosine = [Float](repeating: 0, count: fftSize * fftSize)
    var inverseSine = [Float](repeating: 0, count: fftSize * fftSize)
    for sample in 0 ..< fftSize {
        for frequency in 0 ..< fftSize {
            let angle = 2 * Double.pi * Double(sample * frequency) / Double(fftSize)
            inverseCosine[sample * fftSize + frequency] = Float(cos(angle) / Double(fftSize))
            inverseSine[sample * fftSize + frequency] = Float(sin(angle) / Double(fftSize))
        }
    }
    let cosineMatrix = MLXArray(inverseCosine).reshaped([fftSize, fftSize])
    let sineMatrix = MLXArray(inverseSine).reshaped([fftSize, fftSize])
    let timeDomain = matmul(fullReal, cosineMatrix.transposed())
        - matmul(fullImaginary, sineMatrix.transposed())

    let hann = periodicHannWindow(fftSize)
    let windowed = timeDomain * MLXArray(hann)
    let segmentsPerFrame = fftSize / hopLength
    let outputLength = (frameCount + segmentsPerFrame - 1) * hopLength
    let segments = windowed.reshaped([
        batch, frameCount, segmentsPerFrame, hopLength,
    ])
    var output = MLXArray.zeros([batch, outputLength])
    for segmentIndex in 0 ..< segmentsPerFrame {
        let segment = segments[0..., 0..., segmentIndex, 0...]
            .reshaped([batch, frameCount * hopLength])
        let leftPadding = segmentIndex * hopLength
        let rightPadding = outputLength - leftPadding - frameCount * hopLength
        var pieces: [MLXArray] = []
        if leftPadding > 0 { pieces.append(MLXArray.zeros([batch, leftPadding])) }
        pieces.append(segment)
        if rightPadding > 0 { pieces.append(MLXArray.zeros([batch, rightPadding])) }
        output = output + concatenated(pieces, axis: 1)
    }

    var normalization = [Float](repeating: 0, count: outputLength)
    for frame in 0 ..< frameCount {
        for sample in 0 ..< fftSize {
            normalization[frame * hopLength + sample] += hann[sample] * hann[sample]
        }
    }
    for index in normalization.indices where normalization[index] < 1e-8 {
        normalization[index] = 1e-8
    }
    output = output / MLXArray(normalization).reshaped([1, outputLength])
    let edge = fftSize / 2
    return output[0..., edge ..< (outputLength - edge)]
}

/// Native MLX Swift port of MiniCPM-o 4.5's bundled CosyVoice2 HiFTGenerator.
public final class MiniCPMHiFTGenerator: Module {
    public let configuration: MiniCPMHiFTConfiguration

    @ModuleInfo(key: "m_source") var source: MiniCPMSourceModuleHnNSF2
    @ModuleInfo(key: "f0_predictor") var f0Predictor: MiniCPMF0Predictor
    @ModuleInfo(key: "conv_pre") var convolutionPre: MiniCPMConv1d
    @ModuleInfo var ups: [MiniCPMConvTranspose1d]
    @ModuleInfo(key: "source_downs") var sourceDowns: [MiniCPMConv1d]
    @ModuleInfo(key: "source_resblocks") var sourceResidualBlocks: [MiniCPMHiFTResidualBlock]
    @ModuleInfo var resblocks: [MiniCPMHiFTResidualBlock]
    @ModuleInfo(key: "conv_post") var convolutionPost: MiniCPMConv1d

    public init(configuration: MiniCPMHiFTConfiguration = .init()) {
        self.configuration = configuration
        _source.wrappedValue = MiniCPMSourceModuleHnNSF2(configuration: configuration)
        _f0Predictor.wrappedValue = MiniCPMF0Predictor(
            inputChannels: configuration.inputChannels)
        _convolutionPre.wrappedValue = MiniCPMConv1d(
            inputChannels: configuration.inputChannels,
            outputChannels: configuration.baseChannels,
            kernelSize: 7,
            padding: 3)

        func channels(_ stage: Int) -> Int {
            configuration.baseChannels / (1 << stage)
        }
        var upsampleLayers: [MiniCPMConvTranspose1d] = []
        for index in configuration.upsampleRates.indices {
            let stride = configuration.upsampleRates[index]
            let kernel = configuration.upsampleKernelSizes[index]
            upsampleLayers.append(MiniCPMConvTranspose1d(
                inputChannels: channels(index),
                outputChannels: channels(index + 1),
                kernelSize: kernel,
                stride: stride,
                padding: (kernel - stride) / 2))
        }
        _ups.wrappedValue = upsampleLayers

        let reversedRates = [1] + Array(configuration.upsampleRates.reversed().dropLast())
        var cumulative = 1
        var sourceStrides: [Int] = []
        for rate in reversedRates {
            cumulative *= rate
            sourceStrides.append(cumulative)
        }
        sourceStrides.reverse()
        let sourceChannels = configuration.fftSize + 2
        var sourceDownsampleLayers: [MiniCPMConv1d] = []
        var sourceBlocks: [MiniCPMHiFTResidualBlock] = []
        for index in configuration.upsampleRates.indices {
            let stride = sourceStrides[index]
            sourceDownsampleLayers.append(MiniCPMConv1d(
                inputChannels: sourceChannels,
                outputChannels: channels(index + 1),
                kernelSize: stride == 1 ? 1 : stride * 2,
                stride: stride,
                padding: stride == 1 ? 0 : stride / 2))
            sourceBlocks.append(MiniCPMHiFTResidualBlock(
                channels: channels(index + 1),
                kernelSize: configuration.sourceResidualKernelSizes[index],
                dilations: configuration.sourceResidualDilations[index]))
        }
        _sourceDowns.wrappedValue = sourceDownsampleLayers
        _sourceResidualBlocks.wrappedValue = sourceBlocks

        var residualBlocks: [MiniCPMHiFTResidualBlock] = []
        for stage in configuration.upsampleRates.indices {
            for kernelIndex in configuration.residualKernelSizes.indices {
                residualBlocks.append(MiniCPMHiFTResidualBlock(
                    channels: channels(stage + 1),
                    kernelSize: configuration.residualKernelSizes[kernelIndex],
                    dilations: configuration.residualDilations[kernelIndex]))
            }
        }
        _resblocks.wrappedValue = residualBlocks
        _convolutionPost.wrappedValue = MiniCPMConv1d(
            inputChannels: channels(configuration.upsampleRates.count),
            outputChannels: sourceChannels,
            kernelSize: 7,
            padding: 3)
        super.init()
    }

    public func callAsFunction(_ mel: MLXArray) -> MLXArray {
        generate(mel: mel).waveform
    }

    /// Runs the CosyVoice2 SineGen2/NSF branch independently. The optional
    /// standard-normal noise tensor makes validation and resumed streaming
    /// deterministic without changing the trained source projection.
    public func generateSource(
        f0: MLXArray,
        standardNoise: MLXArray? = nil,
        initialPhase: MLXArray? = nil
    ) -> MLXArray {
        source(f0, noise: standardNoise, initialPhase: initialPhase)
    }

    /// Generates audio and optionally consumes a reference source/noise tensor.
    /// `sourceOverride` is useful for deterministic PyTorch/MLX parity.
    public func generate(
        mel: MLXArray,
        sourceOverride: MLXArray? = nil,
        sourceNoise: MLXArray? = nil,
        sourceInitialPhase: MLXArray? = nil,
        cachedSourcePrefix: MLXArray? = nil
    ) -> MiniCPMHiFTOutput {
        let melNCL = mel.dim(mel.ndim - 1) == configuration.inputChannels
            ? mel.transposed(0, 2, 1) : mel
        let f0 = f0Predictor(melNCL)
        var excitation = sourceOverride ?? source(
            f0, noise: sourceNoise, initialPhase: sourceInitialPhase)
        if let cachedSourcePrefix, cachedSourcePrefix.dim(2) > 0 {
            let prefixLength = Swift.min(cachedSourcePrefix.dim(2), excitation.dim(2))
            excitation = concatenated([
                cachedSourcePrefix[0..., 0..., 0 ..< prefixLength],
                excitation[0..., 0..., prefixLength...],
            ], axis: 2)
        }
        let sourceSpectrum = miniCPMSTFT(
            excitation.squeezed(axis: 1),
            fftSize: configuration.fftSize,
            hopLength: configuration.fftHopLength)
        let sourceSTFT = concatenated([
            sourceSpectrum.real, sourceSpectrum.imaginary,
        ], axis: 1)

        var intermediates: [String: MLXArray] = ["source_stft": sourceSTFT]
        var value = convolutionPre(melNCL)
        intermediates["conv_pre"] = value
        let kernelCount = configuration.residualKernelSizes.count

        for stage in configuration.upsampleRates.indices {
            value = miniCPMLeakyReLU(value, slope: configuration.leakyReLUSlope)
            value = ups[stage](value)
            intermediates["ups.\(stage)"] = value
            if stage == configuration.upsampleRates.count - 1 {
                value = concatenated([value[0..., 0..., 1 ..< 2], value], axis: 2)
            }

            var sourceValue = sourceDowns[stage](sourceSTFT)
            sourceValue = sourceResidualBlocks[stage](sourceValue)
            value = value + sourceValue

            let offset = stage * kernelCount
            var fused = resblocks[offset](value)
            for kernel in 1 ..< kernelCount {
                fused = fused + resblocks[offset + kernel](value)
            }
            value = fused / Float(kernelCount)
            intermediates["stages.\(stage)"] = value
        }

        // Upstream calls F.leaky_relu(x) without the configured slope here;
        // PyTorch's default is 0.01 (the per-stage activations use 0.1).
        value = miniCPMLeakyReLU(value, slope: 0.01)
        let logits = convolutionPost(value)
        intermediates["logits"] = logits
        let bins = configuration.fftSize / 2 + 1
        let magnitude = exp(logits[0..., 0 ..< bins, 0...])
        let phase = sin(logits[0..., bins ..< (configuration.fftSize + 2), 0...])
        let waveform = clip(miniCPMISTFT(
            magnitude: magnitude,
            phase: phase,
            fftSize: configuration.fftSize,
            hopLength: configuration.fftHopLength),
            min: -configuration.audioLimit,
            max: configuration.audioLimit)
        return MiniCPMHiFTOutput(
            waveform: waveform,
            source: excitation,
            f0: f0,
            intermediates: intermediates)
    }
}
