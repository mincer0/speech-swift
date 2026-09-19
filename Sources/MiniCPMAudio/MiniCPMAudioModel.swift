import Foundation
import MLX
import MLXFast
import MLXCommon
import MLXNN

public final class MiniCPMAudioLayerCache {
    public let keys: MLXArray
    public let values: MLXArray
    /// Real (unpadded) key positions. Equal to `keys.dim(2)` when the cache
    /// carries no bucket padding.
    public let realLength: Int

    public convenience init(keys: MLXArray, values: MLXArray) {
        self.init(keys: keys, values: values, realLength: keys.dim(2))
    }

    public init(keys: MLXArray, values: MLXArray, realLength: Int) {
        self.keys = keys
        self.values = values
        self.realLength = realLength
    }

    public var length: Int { keys.dim(2) }
}

public final class MiniCPMAudioCache {
    public let layers: [MiniCPMAudioLayerCache]

    public init(layers: [MiniCPMAudioLayerCache]) {
        self.layers = layers
    }

    /// Real (unpadded) key positions across the streaming cache.
    public var length: Int { layers.first?.realLength ?? 0 }
}

public struct MiniCPMAudioEncoding {
    /// Pooled embeddings ready to replace MiniCPM's audio placeholder tokens.
    /// Shape: `[batch, audioTokens, projectionSize]`.
    public let embeddings: MLXArray
    /// Unpooled final Whisper states, useful for parity diagnostics.
    public let encoderStates: MLXArray
    public let cache: MiniCPMAudioCache?

    public init(
        embeddings: MLXArray,
        encoderStates: MLXArray,
        cache: MiniCPMAudioCache?
    ) {
        self.embeddings = embeddings
        self.encoderStates = encoderStates
        self.cache = cache
    }
}

final class MiniCPMEncoderAttention: Module {
    let heads: Int
    let headDimension: Int
    let scale: Float

    /// Streaming KV caches pad the key length to multiples of this bucket so
    /// attention tensor shapes repeat within a session and the compiled
    /// graph plans / Metal kernels are reused instead of re-planned per
    /// chunk (~400ms per new shape). Must match the golden oracle's
    /// KV_BUCKET_POSITIONS.
    static let keyBucketLength = 256


    @ModuleInfo(key: "q_proj") var queryProjection: Linear
    @ModuleInfo(key: "k_proj") var keyProjection: Linear
    @ModuleInfo(key: "v_proj") var valueProjection: Linear
    @ModuleInfo(key: "out_proj") var outputProjection: Linear

    init(_ config: MiniCPMAudioConfiguration) {
        heads = config.attentionHeads
        headDimension = config.hiddenSize / config.attentionHeads
        scale = 1 / sqrt(Float(headDimension))
        _queryProjection.wrappedValue = Linear(
            config.hiddenSize, config.hiddenSize, bias: true)
        _keyProjection.wrappedValue = Linear(
            config.hiddenSize, config.hiddenSize, bias: false)
        _valueProjection.wrappedValue = Linear(
            config.hiddenSize, config.hiddenSize, bias: true)
        _outputProjection.wrappedValue = Linear(
            config.hiddenSize, config.hiddenSize, bias: true)
        super.init()
    }

    func callAsFunction(
        _ input: MLXArray,
        cache: MiniCPMAudioLayerCache?,
        useCache: Bool,
        tracePrefix: String? = nil,
        trace: ((String, MLXArray) -> Void)? = nil
    ) -> (MLXArray, MiniCPMAudioLayerCache?) {
        let batch = input.dim(0)
        let length = input.dim(1)
        // Match transformers' eager WhisperAttention order exactly: scale the
        // BF16 q projection before the BF16 QK matmul, rather than scaling the
        // accumulated scores afterwards.  The MPS eager path keeps q/k/v in
        // BF16 and materializes each transposed head view before matmul.
        let query = (audioLinear(queryProjection, input) * MLXArray(scale).asType(input.dtype))
            .reshaped(batch, length, heads, headDimension)
            .transposed(0, 2, 1, 3)
            .contiguous()
        if let tracePrefix { trace?(tracePrefix + "query", query) }
        let currentKeys = audioLinear(keyProjection, input)
            .reshaped(batch, length, heads, headDimension)
            .transposed(0, 2, 1, 3)
            .contiguous()
        if let tracePrefix { trace?(tracePrefix + "key", currentKeys) }
        let currentValues = audioLinear(valueProjection, input)
            .reshaped(batch, length, heads, headDimension)
            .transposed(0, 2, 1, 3)
            .contiguous()
        if let tracePrefix { trace?(tracePrefix + "value", currentValues) }

        var keys = currentKeys
        var values = currentValues
        var totalRealLength = length
        var additiveMask: MLXArray?
        if let cache {
            keys = concatenated([cache.keys, currentKeys], axis: 2)
            values = concatenated([cache.values, currentValues], axis: 2)
            totalRealLength = cache.realLength + length
        }
        if useCache {
            // Static-shape bucket layout: keys are laid out
            // [zeros | real_past | current] with the total padded up to the
            // next multiple of keyBucketLength, and the additive mask
            // exposes only the real positions (the real tail). Padded rows
            // are zero and receive -10000; exp(-10000 - max) underflows to
            // exactly +0 in the softmax, so the layout is mathematically
            // transparent while attention shapes stay stable within a bucket
            // (one plan re-build per ~16s of audio instead of per chunk).
            let pastReal = totalRealLength - length
            let bucket = ((totalRealLength + Self.keyBucketLength - 1)
                / Self.keyBucketLength) * Self.keyBucketLength
            let headPad = bucket - totalRealLength
            if headPad >= 0 {
                if let cache {
                    // [zeros(headPad) | real_past | current]
                    let headKeys = MLXArray.zeros(
                        [keys.dim(0), keys.dim(1), headPad, keys.dim(3)],
                        dtype: keys.dtype)
                    let headValues = MLXArray.zeros(
                        [values.dim(0), values.dim(1), headPad, values.dim(3)],
                        dtype: values.dtype)
                    keys = concatenated([headKeys, cache.keys, currentKeys], axis: 2)
                    values = concatenated([headValues, cache.values, currentValues], axis: 2)
                } else {
                    // First chunk: [zeros(headPad) | current]
                    let headKeys = MLXArray.zeros(
                        [keys.dim(0), keys.dim(1), headPad, keys.dim(3)],
                        dtype: keys.dtype)
                    let headValues = MLXArray.zeros(
                        [values.dim(0), values.dim(1), headPad, values.dim(3)],
                        dtype: values.dtype)
                    keys = concatenated([headKeys, currentKeys], axis: 2)
                    values = concatenated([headValues, currentValues], axis: 2)
                }
                // Mask: -10000 for the head zero block, 0 for the real tail.
                var rowMask = MLXArray(
                    [Float](repeating: -10_000.0,
                            count: keys.dim(0) * length * headPad))
                    .reshaped(keys.dim(0), 1, length, headPad)
                    .asType(keys.dtype)
                rowMask = concatenated([
                    rowMask,
                    MLXArray.zeros(
                        [keys.dim(0), 1, length, pastReal + length],
                        dtype: keys.dtype),
                ], axis: 3)
                additiveMask = tiled(rowMask, repetitions: [1, heads, 1, 1])
                    .contiguous()
            }
        }

        // MiniCPM passes an all-zero additive mask for streaming calls: every
        // query sees the complete prior cache and current chunk.  Offline
        // encoding omits the mask.  The production MPSGraph path keeps the
        // complete QK -> optional neutral-mask -> softmax -> PV chain in one
        // BF16 graph so the reduction boundaries match the official PyTorch
        // MPS path.
        // The pinned MiniCPM audio bundle is BF16 and its official PyTorch
        // MPS reference keeps the complete attention chain on the BF16
        // MPSGraph boundary.  That boundary is not interchangeable with
        // MLX's separate matmul/FP32-softmax path: a one-ULP difference is
        // amplified by the recurrent 24-layer encoder (notably layer 7).
        // Make the parity-tested graph path the production default.  The
        // native/MLX choice remains an explicit diagnostic escape hatch for
        // operator experiments and non-parity comparisons.
        let attentionVariant = ProcessInfo.processInfo.environment[
            "MINICPM_AUDIO_ATTENTION_VARIANT"
        ]?.lowercased()
        let useMPSGraphAttention = query.dtype == .bfloat16
            && attentionVariant != "native"
            && attentionVariant != "mlx"

        let scores: MLXArray
        let probabilities: MLXArray?
        let contextHeads: MLXArray
        if useMPSGraphAttention {
            let result = MiniCPMAudioBF16Attention.apply(
                query: query,
                key: keys,
                value: values,
                additiveMask: additiveMask,
                captureIntermediates: tracePrefix != nil
            )
            scores = result.scores
            probabilities = result.probabilities
            contextHeads = result.context
        } else {
            var mlxScores = matmul(
                query, keys.transposed(0, 1, 3, 2).contiguous())
            if let additiveMask {
                // Bucketed streaming mask (0 real / -10000 padded tail).
                // Elementwise, so reduction boundaries are identical to the
                // pinned neutral-mask add.
                mlxScores = mlxScores + additiveMask
            }
            scores = mlxScores
            if scores.dtype == .bfloat16 {
                // This branch is the explicit native/MLX diagnostic fallback;
                // the parity-tested complete graph is selected above for
                // BF16 production inference.
                let variant = ProcessInfo.processInfo.environment[
                    "MINICPM_AUDIO_SOFTMAX_VARIANT"
                ]?.lowercased()
                if variant == "mpsgraph" {
                    probabilities = MiniCPMAudioBF16Softmax.apply(
                        scores, axis: -1)
                } else {
                    probabilities = softmax(scores.asType(.float32), axis: -1)
                        .asType(scores.dtype)
                }
            } else {
                // Lightweight/unit-test configurations may intentionally use
                // Float32 weights. Keep those paths on the ordinary MLX op.
                probabilities = softmax(scores, axis: -1, precise: true)
            }
            contextHeads = matmul(
                probabilities!, values.contiguous())
        }
        if let tracePrefix {
            trace?(tracePrefix + "attention_scores", scores)
            if let probabilities {
                trace?(tracePrefix + "attention_probabilities", probabilities)
            }
        }
        if let tracePrefix {
            trace?(tracePrefix + "attention_context_heads", contextHeads)
        }
        let attended = contextHeads
            .transposed(0, 2, 1, 3)
            .contiguous()
            .reshaped(batch, length, heads * headDimension)
        if let tracePrefix {
            trace?(tracePrefix + "attention_context_merged", attended)
        }
        let next: MiniCPMAudioLayerCache?
        if useCache {
            // Head-padded layout: the real positions (real_past + current)
            // are the LAST totalRealLength entries of the bucketed tensors.
            // Slicing from the front would store the zero head block and
            // drop the current chunk's keys.
            let cacheKeys = keys.dim(2) > totalRealLength
                ? keys[0..., 0..., (keys.dim(2) - totalRealLength)..., 0...]
                    .contiguous()
                : keys
            let cacheValues = values.dim(2) > totalRealLength
                ? values[0..., 0..., (values.dim(2) - totalRealLength)..., 0...]
                    .contiguous()
                : values
            next = MiniCPMAudioLayerCache(
                keys: cacheKeys, values: cacheValues,
                realLength: totalRealLength)
        } else {
            next = nil
        }
        let output = audioLinear(outputProjection, attended)
        if let tracePrefix {
            trace?(tracePrefix + "attention_output", output)
        }
        return (output, next)
    }
}

final class MiniCPMEncoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttention: MiniCPMEncoderAttention
    @ModuleInfo(key: "self_attn_layer_norm") var selfAttentionLayerNorm: LayerNorm
    @ModuleInfo(key: "fc1") var feedForward1: Linear
    @ModuleInfo(key: "fc2") var feedForward2: Linear
    @ModuleInfo(key: "final_layer_norm") var finalLayerNorm: LayerNorm

    init(_ config: MiniCPMAudioConfiguration) {
        _selfAttention.wrappedValue = MiniCPMEncoderAttention(config)
        _selfAttentionLayerNorm.wrappedValue = LayerNorm(
            dimensions: config.hiddenSize,
            eps: config.layerNormEpsilon
        )
        _feedForward1.wrappedValue = Linear(
            config.hiddenSize, config.intermediateSize, bias: true)
        _feedForward2.wrappedValue = Linear(
            config.intermediateSize, config.hiddenSize, bias: true)
        _finalLayerNorm.wrappedValue = LayerNorm(
            dimensions: config.hiddenSize,
            eps: config.layerNormEpsilon
        )
        super.init()
    }

    func callAsFunction(
        _ input: MLXArray,
        cache: MiniCPMAudioLayerCache?,
        useCache: Bool,
        layerIndex: Int,
        trace: ((String, MLXArray) -> Void)? = nil
    ) -> (MLXArray, MiniCPMAudioLayerCache?) {
        let prefix = String(format: "layer_%02d_", layerIndex)
        let preAttentionNorm = audioLayerNorm(
            selfAttentionLayerNorm, input)
        trace?(prefix + "pre_attention_norm", preAttentionNorm)
        let (attention, nextCache) = selfAttention(
            preAttentionNorm,
            cache: cache,
            useCache: useCache,
            tracePrefix: prefix,
            trace: trace
        )
        trace?(prefix + "attention_output", attention)
        // MPS binary addition promotes the BF16 operands through its common
        // floating type before writing the BF16 destination.  Spell that
        // boundary explicitly instead of relying on the backend's elementwise
        // BF16 kernel; this also avoids a recurrent one-ULP residual drift.
        var hidden = audioResidual(input, attention)
        trace?(prefix + "post_attention_residual", hidden)
        let preFeedForwardNorm = audioLayerNorm(finalLayerNorm, hidden)
        trace?(prefix + "pre_ffn_norm", preFeedForwardNorm)
        let feedForwardProjection = audioLinear(
            feedForward1, preFeedForwardNorm)
        trace?(prefix + "fc1_output", feedForwardProjection)
        let feedForwardActivation = audioGELU(feedForwardProjection)
        trace?(prefix + "ffn_activation", feedForwardActivation)
        let feedForwardOutput = audioLinear(
            feedForward2, feedForwardActivation)
        trace?(prefix + "ffn_output", feedForwardOutput)
        hidden = audioResidual(hidden, feedForwardOutput)
        trace?(prefix + "layer_output", hidden)
        return (hidden, nextCache)
    }
}

final class MiniCPMWhisperEncoder: Module {
    let config: MiniCPMAudioConfiguration

    @ModuleInfo var conv1: Conv1d
    @ModuleInfo var conv2: Conv1d
    @ModuleInfo(key: "embed_positions") var positionEmbedding: Embedding
    @ModuleInfo var layers: [MiniCPMEncoderLayer]
    @ModuleInfo(key: "layer_norm") var layerNorm: LayerNorm

    init(_ config: MiniCPMAudioConfiguration) {
        self.config = config
        _conv1.wrappedValue = Conv1d(
            inputChannels: config.melBins,
            outputChannels: config.hiddenSize,
            kernelSize: 3,
            padding: 1
        )
        _conv2.wrappedValue = Conv1d(
            inputChannels: config.hiddenSize,
            outputChannels: config.hiddenSize,
            kernelSize: 3,
            stride: 2,
            padding: 1
        )
        _positionEmbedding.wrappedValue = Embedding(
            embeddingCount: config.maximumSourcePositions,
            dimensions: config.hiddenSize
        )
        _layers.wrappedValue = (0..<config.layers).map { _ in
            MiniCPMEncoderLayer(config)
        }
        _layerNorm.wrappedValue = LayerNorm(
            dimensions: config.hiddenSize,
            eps: config.layerNormEpsilon
        )
        super.init()
    }

    func callAsFunction(
        _ features: MLXArray,
        cache: MiniCPMAudioCache?,
        useCache: Bool,
        prefixExtraFrames: Int,
        suffixExtraFrames: Int,
        trace: ((String, MLXArray) -> Void)? = nil
    ) throws -> (MLXArray, MiniCPMAudioCache?) {
        var hidden = features.asType(conv1.weight.dtype)
        // Match the MPS Conv1d boundary: accumulate the complete window in
        // FP32, add bias as a separate operation, then round once to BF16.
        // Keeping the bias outside the accumulator matters for this model's
        // high-gain layer-7 FFN and is not equivalent to seeding a matmul with
        // the bias term.
        let conv1Output = audioConv1dMPS(conv1, hidden)
        trace?("conv1_output", conv1Output.transposed(0, 2, 1).contiguous())
        hidden = audioGELU(conv1Output)
        trace?("conv1_gelu", hidden.transposed(0, 2, 1).contiguous())
        let conv2Output = audioConv1dMPS(conv2, hidden)
        trace?("conv2_output", conv2Output.transposed(0, 2, 1).contiguous())
        hidden = audioGELU(conv2Output)
        trace?("conv2_gelu", hidden.transposed(0, 2, 1).contiguous())

        // This is the single tensor crop for duplex CNN context.  The
        // streaming session repeats the length arithmetic only to decide
        // whether its KV cache must be reset; it never slices `features`.
        let prefixRemove = prefixExtraFrames > 0
            ? (prefixExtraFrames + 1) / 2 : 0
        let suffixRemove = suffixExtraFrames > 0
            ? (suffixExtraFrames + 1) / 2 : 0
        if prefixRemove > 0 || suffixRemove > 0 {
            let end = hidden.dim(1) - suffixRemove
            guard end > prefixRemove else {
                throw MiniCPMAudioError.invalidFeatures(
                    "CNN context removal consumed the entire chunk"
                )
            }
            hidden = hidden[0..., prefixRemove..<end, 0...]
        }

        let pastLength = cache?.length ?? 0
        let currentLength = hidden.dim(1)
        guard pastLength + currentLength <= config.maximumSourcePositions else {
            throw MiniCPMAudioError.cacheLimit(
                current: pastLength,
                incoming: currentLength,
                maximum: config.maximumSourcePositions
            )
        }
        guard cache == nil || cache!.layers.count == layers.count else {
            throw MiniCPMAudioError.invalidFeatures(
                "KV cache layer count does not match the encoder"
            )
        }

        let positions = positionEmbedding.weight[
            pastLength..<(pastLength + currentLength), 0...
        ].expandedDimensions(axis: 0).asType(hidden.dtype)
        trace?("position_embeddings", positions)
        hidden = hidden + positions
        trace?("position_added", hidden)

        var nextLayers: [MiniCPMAudioLayerCache] = []
        if useCache { nextLayers.reserveCapacity(layers.count) }
        for index in layers.indices {
            let prior = cache?.layers[index]
            let (output, next) = layers[index](
                hidden,
                cache: prior,
                useCache: useCache,
                layerIndex: index,
                trace: trace
            )
            hidden = output
            if let next { nextLayers.append(next) }
        }
        hidden = audioLayerNorm(layerNorm, hidden)
        trace?("encoder_output", hidden)
        return (
            hidden,
            useCache ? MiniCPMAudioCache(layers: nextLayers) : nil
        )
    }
}

final class MiniCPMAudioProjection: Module {
    @ModuleInfo var linear1: Linear
    @ModuleInfo var linear2: Linear

    init(_ config: MiniCPMAudioConfiguration) {
        _linear1.wrappedValue = Linear(
            config.hiddenSize, config.projectionSize, bias: true)
        _linear2.wrappedValue = Linear(
            config.projectionSize, config.projectionSize, bias: true)
        super.init()
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        audioLinear(linear2, relu(audioLinear(linear1, input)))
    }
}

/// Keep all audio projections on the parity-tested MPS-compatible boundary.
/// MLXNN's fused ``addMM`` path is retained as the fallback for non-standard
/// ranks, while the rank-3 audio path uses the explicit MLX Metal reduction
/// below so q/k/v and FFN projections share PyTorch MPS's FP32 accumulation.
@inline(__always)
private func audioLinear(_ layer: Linear, _ input: MLXArray) -> MLXArray {
    audioLinearMPS(layer, input)
}

/// Average-pool an NLC tensor with the PyTorch MPS accumulation boundary.
/// MiniCPM uses a non-overlapping pool (`stride == kernel`), so no padding or
/// partial-window policy is needed here.
@inline(__always)
private func audioAveragePool1d(
    _ input: MLXArray,
    kernelSize: Int
) -> MLXArray {
    precondition(input.ndim == 3 && kernelSize > 0)
    let batch = input.dim(0)
    let length = input.dim(1)
    let channels = input.dim(2)
    // Match AvgPool1d's floor output geometry: extra tail positions that do
    // not fill a complete window are intentionally ignored (context-mode
    // chunks are commonly 48/49 positions rather than a multiple of five).
    precondition(length >= kernelSize)
    let outputLength = (length - kernelSize) / kernelSize + 1
    let windows = asStrided(
        input,
        [batch, outputLength, kernelSize, channels],
        strides: [length * channels, kernelSize * channels, channels, 1])
    return mean(windows.asType(.float32), axes: [2]).asType(input.dtype)
}

/// Diagnostic/production candidate for the eager PyTorch MPS Linear boundary.
///
/// MLX's general matmul reduction is numerically stable, but its tiling can
/// round BF16 inputs differently from ``torch.nn.functional.linear`` on MPS.
/// This kernel deliberately accumulates one output element in FP32, adds the
/// bias after the reduction, and performs one destination-dtype store.  Keep
/// The grid is grouped along sequence positions, so SIMD lanes share the
/// output row's weight address while each output's FMA order stays unchanged.
@inline(__always)
private func audioLinearMPS(_ layer: Linear, _ input: MLXArray) -> MLXArray {
    guard input.ndim == 3 else { return layer(input) }
    let contiguousInput = input.contiguous()
    let contiguousWeight = layer.weight.contiguous()
    let outputChannels = contiguousWeight.dim(0)
    let inputChannels = contiguousWeight.dim(1)
    precondition(contiguousInput.dim(2) == inputChannels)
    let bias = (layer.bias ?? MLXArray.zeros(
        [outputChannels], dtype: contiguousInput.dtype)).contiguous()
    return MiniCPMAudioMetalKernels.linear(
        [
            contiguousInput,
            contiguousWeight,
            bias,
            Int32(contiguousInput.dim(1)),
            Int32(inputChannels),
            Int32(outputChannels),
        ],
        template: [("T", contiguousInput.dtype)],
        grid: (outputChannels, contiguousInput.dim(1), contiguousInput.dim(0)),
        threadGroup: (1, min(32, contiguousInput.dim(1)), 1),
        outputShapes: [[contiguousInput.dim(0), contiguousInput.dim(1), outputChannels]],
        outputDTypes: [contiguousInput.dtype]
    )[0]
}

@inline(__always)
private func audioResidual(_ lhs: MLXArray, _ rhs: MLXArray) -> MLXArray {
    (lhs.asType(.float32) + rhs.asType(.float32)).asType(lhs.dtype)
}

/// PyTorch MPS accumulates the BF16 Whisper convolutions in FP32 and rounds
/// once after the bias add.  MLX's dtype-preserving Conv1d kernel exposes its
/// intermediate BF16 rounding on this model; those first-window errors are
/// later amplified by a high-gain FFN channel in encoder layer 7.
@inline(__always)
private func audioConv1d(_ layer: Conv1d, _ input: MLXArray) -> MLXArray {
    let outputDType = input.dtype
    var output = MLX.conv1d(
        input.asType(.float32),
        layer.weight.asType(.float32),
        stride: layer.stride,
        padding: layer.padding,
        dilation: layer.dilation,
        groups: layer.groups
    )
    if let bias = layer.bias {
        output = output + bias.asType(.float32)
    }
    return output.asType(outputDType)
}

/// Conv1d boundary used by production MiniCPM audio inference.
///
/// ``MLX.conv1d``/matmul are excellent general kernels, but their reduction
/// tiling and fused bias path do not reproduce the pinned PyTorch MPS result
/// for this BF16 Whisper front-end.  This small MLX Metal kernel follows the
/// reference order explicitly: temporal window, input-channel reduction in
/// FP32, separate bias add, and one BF16 store.
@inline(__always)
private func audioConv1dMPS(_ layer: Conv1d, _ input: MLXArray) -> MLXArray {
    guard let bias = layer.bias else {
        return audioConv1d(layer, input)
    }
    let contiguousInput = input.contiguous()
    let contiguousWeight = layer.weight.contiguous()
    let contiguousBias = bias.contiguous()
    let batch = contiguousInput.dim(0)
    let inputLength = contiguousInput.dim(1)
    let inputChannels = contiguousInput.dim(2)
    let outputChannels = contiguousWeight.dim(0)
    let kernelSize = contiguousWeight.dim(1)
    let outputLength = (inputLength + 2 * layer.padding
        - layer.dilation * (kernelSize - 1) - 1) / layer.stride + 1
    precondition(outputLength > 0)

    return MiniCPMAudioMetalKernels.conv1d(
        [
            contiguousInput,
            contiguousWeight,
            contiguousBias,
            Int32(inputLength),
            Int32(outputLength),
            Int32(inputChannels),
            Int32(outputChannels),
            Int32(kernelSize),
            Int32(layer.stride),
            Int32(layer.padding),
        ],
        template: [("T", contiguousInput.dtype)],
        grid: (outputChannels, outputLength, batch),
        threadGroup: (1, 1, 1),
        outputShapes: [[batch, outputLength, outputChannels]],
        outputDTypes: [contiguousInput.dtype]
    )[0]
}

/// PyTorch MPS builds exact GELU as `x * normcdf(x)`.  Its MPSGraph constants
/// inherit the BF16 input dtype, including the `1 / sqrt(2)` multiplier, while
/// the elementwise kernel itself retains FP32-like arithmetic and rounds once
/// at the output.  Preserving that quantized constant is important: using the
/// full-precision value introduces one-ULP activation drift which is amplified
/// by a high-gain channel in encoder layer 7.
@inline(__always)
private func audioGELU(_ input: MLXArray) -> MLXArray {
    if input.dtype == .bfloat16 {
        // The official PyTorch MPS encoder evaluates exact GELU as one BF16
        // MPSGraph expression.  The MLX/Metal approximation is numerically
        // close in isolation, but a single BF16 ULP at layer 0 is amplified
        // by the high-gain FFN in layer 7.  Use the parity-tested graph by
        // default; native/metal and eager remain opt-in diagnostics.
        switch ProcessInfo.processInfo.environment[
            "MINICPM_AUDIO_GELU_VARIANT"
        ]?.lowercased() {
        case "native", "metal":
            return audioGELUMPS(input)
        case "eager":
            return audioGELUEagerFP32(input)
        default:
            return MiniCPMAudioBF16GELU.apply(input)
        }
    }
    return audioGELUMPS(input)
}

/// Candidate implementation of the MPSGraph ``gelu(approximate: none)``
/// graph used by the official PyTorch MPS backend.  Keeping the whole
/// expression in one Metal kernel is important: the graph evaluates the
/// intermediate erf expression in the device's native arithmetic and rounds
/// only at the destination, whereas spelling the same formula as separate
/// MLX operations can introduce an extra dtype/reduction boundary.
@inline(__always)
private func audioGELUMPS(_ input: MLXArray) -> MLXArray {
    let contiguous = input.contiguous()
    return MiniCPMAudioMetalKernels.gelu(
        [contiguous, Int32(contiguous.size)],
        template: [("T", contiguous.dtype)],
        grid: (contiguous.size, 1, 1),
        threadGroup: (1, 1, 1),
        outputShapes: [contiguous.shape],
        outputDTypes: [contiguous.dtype]
    )[0]
}

/// Diagnostic-only exact-erf candidate.  MLX's public `erf` kernel evaluates
/// the expression in FP32 when given an FP32 input; unlike the fused Metal
/// probe above this lets us measure whether the MPS reference used a more
/// accurate transcendental boundary before the final BF16 store.
@inline(__always)
private func audioGELUEagerFP32(_ input: MLXArray) -> MLXArray {
    let value = input.asType(.float32)
    let scale = MLXArray(Float(0.7071067811865475))
    let cdf = (erf(value * scale) + MLXArray(Float(1))) * MLXArray(Float(0.5))
    return (cdf * value).asType(input.dtype)
}

/// MLXFast implementation of the BF16 MPS LayerNorm kernel used by PyTorch.
///
/// PyTorch's MPS backend does not use MPSGraph for this operation.  It reads
/// four BF16 values per thread, reduces FP32 partial sums with SIMD groups,
/// computes `sumSq / N - mean * mean`, uses precise rsqrt, and stores the
/// affine result once as BF16.  Keeping the same reduction topology avoids a
/// recurrent one-ULP drift that otherwise becomes visible in layer 7's FFN.
private enum MiniCPMAudioMetalKernels {
    /// Sequential FP32 affine projection used to pin the MPS Linear boundary.
    static let linear = MLXFast.metalKernel(
        name: "minicpm_audio_linear_mps_fp32_reduce",
        inputNames: ["x", "weight", "bias", "sequence_length", "input_channels", "output_channels"],
        outputNames: ["out"],
        source: """
            uint outputChannel = thread_position_in_grid.x;
            uint sequence = thread_position_in_grid.y;
            uint batch = thread_position_in_grid.z;
            uint sequenceLength = uint(sequence_length);
            uint inputChannels = uint(input_channels);
            uint outputChannels = uint(output_channels);
            if (outputChannel >= outputChannels || sequence >= sequenceLength) {
                return;
            }

            uint inputOffset = (batch * sequenceLength + sequence) * inputChannels;
            uint weightOffset = outputChannel * inputChannels;
            float accumulator = 0.0f;
            for (uint channel = 0; channel < inputChannels; channel++) {
                accumulator = metal::fma(
                    float(x[inputOffset + channel]),
                    float(weight[weightOffset + channel]),
                    accumulator);
            }
            uint outputOffset = (batch * sequenceLength + sequence) * outputChannels
                + outputChannel;
            accumulator += float(bias[outputChannel]);
            out[outputOffset] = T(accumulator);
        """,
        header: """
            #include <metal_stdlib>
            using namespace metal;
        """
    )

    /// Fused exact-GELU candidate mirroring the MPSGraph norm-CDF expression:
    /// ``((erf(x * sqrt(1/2)) + 1) * 0.5) * x``.
    static let gelu = MLXFast.metalKernel(
        name: "minicpm_audio_gelu_mps_graph",
        inputNames: ["x", "element_count"],
        outputNames: ["out"],
        source: """
            uint index = thread_position_in_grid.x;
            uint count = uint(element_count);
            if (index >= count) {
                return;
            }

            float value = float(x[index]);
            // MPSGraph creates these constants with the input tensor's dtype
            // before evaluating the graph.  Cast the scalar through T to keep
            // the BF16 constant boundary visible to the compiler.
            float sqrt1_2 = float(T(0.7071067811865475244f));
            float cdf = minicpm_erf(value * sqrt1_2);
            cdf = (cdf + 1.0f) * 0.5f;
            out[index] = T(cdf * value);
        """,
        header: """
            #include <metal_stdlib>
            using namespace metal;

            // MLX's custom-kernel include paths are not exposed to kernels
            // created through MLXFast.  Keep the same faithfully-rounded
            // erf approximation inline so this fused boundary remains
            // self-contained and deterministic.
            inline float minicpm_expm1f_scaled(float a, float b) {
                float f, j, r, s, t, u, v, x, y;
                int i;
                j = metal::fma(1.442695f, a, 12582912.0f);
                j = j - 12582912.0f;
                i = int(j);
                f = metal::fma(j, -6.93145752e-1f, a);
                s = f * f;
                if (a == 0.0f) {
                    s = a;
                }
                r = 1.97350979e-4f;
                r = metal::fma(r, f, 1.39309070e-3f);
                r = metal::fma(r, f, 8.33343994e-3f);
                r = metal::fma(r, f, 4.16668020e-2f);
                r = metal::fma(r, f, 1.66666716e-1f);
                r = metal::fma(r, f, 4.99999970e-1f);
                u = (j == 1.0f) ? (f + 0.5f) : f;
                v = metal::fma(r, s, u);
                s = 0.5f * b;
                t = metal::ldexp(s, i);
                y = t - s;
                x = (t - y) - s;
                r = metal::fma(v, t, x) + y;
                r = r + r;
                if (j == 0.0f) {
                    r = v;
                }
                if (j == 1.0f) {
                    r = v + v;
                }
                return r;
            }

            inline float minicpm_expm1f(float a) {
                float r = minicpm_expm1f_scaled(a, 1.0f);
                if (metal::abs(a - 1.0f) > 88.0f) {
                    r = metal::pow(2.0f, a);
                    r = metal::fma(r, r, -1.0f);
                }
                return r;
            }

            inline float minicpm_erf(float a) {
                float r, s, t, u;
                t = metal::abs(a);
                s = a * a;
                if (t > 0.927734375f) {
                    r = metal::fma(-1.72853470e-5f, t, 3.83197126e-4f);
                    u = metal::fma(-3.88396438e-3f, t, 2.42546219e-2f);
                    r = metal::fma(r, s, u);
                    r = metal::fma(r, t, -1.06777877e-1f);
                    r = metal::fma(r, t, -6.34846687e-1f);
                    r = metal::fma(r, t, -1.28717512e-1f);
                    r = metal::fma(r, t, -t);
                    r = -minicpm_expm1f(r);
                    r = metal::copysign(r, a);
                } else {
                    r = -5.96761703e-4f;
                    r = metal::fma(r, s, 4.99119423e-3f);
                    r = metal::fma(r, s, -2.67681349e-2f);
                    r = metal::fma(r, s, 1.12819925e-1f);
                    r = metal::fma(r, s, -3.76125336e-1f);
                    r = metal::fma(r, s, 1.28379166e-1f);
                    r = metal::fma(r, a, a);
                }
                return r;
            }
        """
    )

    /// Sequential FP32 convolution used to reproduce PyTorch MPS Conv1d's
    /// numerical boundary.  The bias is deliberately added *after* the full
    /// reduction, matching ``aten::convolution``'s separate bias add.
    static let conv1d = MLXFast.metalKernel(
        name: "minicpm_audio_conv1d_mps_fp32_reduce",
        inputNames: [
            "x", "weight", "bias", "input_length", "output_length",
            "input_channels", "output_channels", "kernel_size", "stride",
            "padding",
        ],
        outputNames: ["out"],
        source: """
            uint outputChannel = thread_position_in_grid.x;
            uint outputTime = thread_position_in_grid.y;
            uint batch = thread_position_in_grid.z;
            uint inputLength = uint(input_length);
            uint outputLength = uint(output_length);
            uint inputChannels = uint(input_channels);
            uint outputChannels = uint(output_channels);
            uint kernelSize = uint(kernel_size);
            uint strideValue = uint(stride);
            int paddingValue = int(padding);

            if (outputChannel >= outputChannels || outputTime >= outputLength) {
                return;
            }

            float accumulator = 0.0f;
            for (uint kernelIndex = 0; kernelIndex < kernelSize; kernelIndex++) {
                int inputTime = int(outputTime * strideValue + kernelIndex)
                    - paddingValue;
                if (inputTime < 0 || uint(inputTime) >= inputLength) {
                    continue;
                }
                for (uint channel = 0; channel < inputChannels; channel++) {
                    uint inputOffset = (batch * inputLength
                        + uint(inputTime)) * inputChannels + channel;
                    uint weightOffset = (outputChannel * kernelSize
                        + kernelIndex) * inputChannels + channel;
                    accumulator = metal::fma(
                        float(x[inputOffset]),
                        float(weight[weightOffset]),
                        accumulator);
                }
            }

            // Keep this outside the reduction.  Seeding ``accumulator`` with
            // bias changes the FP32 rounding path and does not match MPS.
            accumulator += float(bias[outputChannel]);
            uint outputOffset = (batch * outputLength + outputTime)
                * outputChannels + outputChannel;
            out[outputOffset] = T(accumulator);
        """,
        header: """
            #include <metal_stdlib>
            using namespace metal;
        """
    )

    static let layerNorm = MLXFast.metalKernel(
        name: "minicpm_audio_layer_norm_mps_bf16",
        inputNames: ["x", "weight", "bias", "axis_size", "epsilon"],
        outputNames: ["out"],
        source: """
            uint row = threadgroup_position_in_grid.x;
            uint tid = thread_position_in_threadgroup.x;
            uint lane = thread_index_in_simdgroup;
            uint simdgroup = simdgroup_index_in_threadgroup;
            uint axis = uint(axis_size);
            uint base = tid * 4;
            uint rowOffset = row * axis;
            const device T* rowInput = x + rowOffset + base;

            float partialSum = 0.0f;
            float partialSumSq = 0.0f;
            if (base + 4 <= axis) {
                float4 values = float4(
                    rowInput[0], rowInput[1], rowInput[2], rowInput[3]);
                partialSum = values.x + values.y + values.z + values.w;
                partialSumSq = dot(values, values);
            } else {
                uint remaining = axis > base ? axis - base : 0;
                if (remaining >= 3) {
                    float3 values = float3(rowInput[0], rowInput[1], rowInput[2]);
                    partialSum = values.x + values.y + values.z;
                    partialSumSq = dot(values, values);
                } else if (remaining >= 2) {
                    float2 values = float2(rowInput[0], rowInput[1]);
                    partialSum = values.x + values.y;
                    partialSumSq = dot(values, values);
                } else if (remaining >= 1) {
                    float value = float(rowInput[0]);
                    partialSum = value;
                    partialSumSq = metal::fma(value, value, 0.0f);
                }
            }

            threadgroup float partialSums[32];
            threadgroup float partialSumsSq[32];
            threadgroup float sharedMean[1];
            threadgroup float sharedInverseStd[1];

            if (simdgroup == 0) {
                partialSums[lane] = 0.0f;
                partialSumsSq[lane] = 0.0f;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            float groupSum = simd_sum(partialSum);
            float groupSumSq = simd_sum(partialSumSq);
            if (lane == 0) {
                partialSums[simdgroup] = groupSum;
                partialSumsSq[simdgroup] = groupSumSq;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (simdgroup == 0) {
                float total = simd_sum(partialSums[lane]);
                float totalSq = simd_sum(partialSumsSq[lane]);
                if (lane == 0) {
                    float meanValue = total / float(axis);
                    float variance = totalSq / float(axis)
                        - meanValue * meanValue;
                    variance = variance < 1e-6f ? 0.0f : variance;
                    sharedMean[0] = meanValue;
                    sharedInverseStd[0] = metal::precise::rsqrt(
                        variance + epsilon);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            float meanValue = sharedMean[0];
            float inverseStd = sharedInverseStd[0];
            device T* rowOutput = out + rowOffset + base;
            for (uint index = 0; index < 4; index++) {
                uint channel = base + index;
                if (channel < axis) {
                    float normalized = (float(rowInput[index]) - meanValue)
                        * inverseStd;
                    normalized *= float(weight[channel]);
                    normalized += float(bias[channel]);
                    rowOutput[index] = T(normalized);
                }
            }
        """,
        header: """
            #include <metal_stdlib>
            #include <metal_simdgroup>
            using namespace metal;
        """
    )
}

@inline(__always)
private func audioLayerNormEager(_ layer: LayerNorm, _ input: MLXArray) -> MLXArray {
    let inputFloat = input.asType(.float32)
    let meanValue = mean(inputFloat, axes: [-1], keepDims: true)
    let centered = inputFloat - meanValue
    // MPS native_layer_norm computes FP32 moments for BF16 input, applies the
    // BF16 affine parameters in that FP32 path, and rounds only the result.
    // Keeping the second moment as a separate reduction matches that boundary
    // more closely than MLXFast's dtype-preserving fused kernel.
    let variance = maximum(
        mean(inputFloat * inputFloat, axes: [-1], keepDims: true)
            - meanValue * meanValue,
        MLXArray(Float(0)))
    let normalized = centered * rsqrt(variance + MLXArray(layer.eps))
    let weight = (layer.weight ?? MLXArray.ones([input.dim(-1)])).asType(.float32)
    var output = normalized * weight
    if let bias = layer.bias {
        output = output + bias.asType(.float32)
    }
    return output.asType(input.dtype)
}

@inline(__always)
private func audioLayerNorm(_ layer: LayerNorm, _ input: MLXArray) -> MLXArray {
    // Keep the parity-tested MPS kernel as the production default, but expose
    // an opt-in process switch for recurrent-drift diagnostics.  The switch is
    // intentionally read at call time so a test can run several fresh model
    // processes with different candidates without changing the default path.
    switch ProcessInfo.processInfo.environment["MINICPM_AUDIO_LAYERNORM_VARIANT"] {
    case "eager":
        return audioLayerNormEager(layer, input)
    case "native":
        return layer(input)
    default:
        break
    }
    // All MiniCPM audio LayerNorms are BF16 with affine parameters.  Keep a
    // general eager fallback for alternate bundles/configurations and for
    // non-Metal platforms; production M2 inference takes the exact kernel.
    guard input.dtype == .bfloat16,
          let weight = layer.weight,
          let bias = layer.bias,
          input.dim(-1) <= 4096
    else {
        return audioLayerNormEager(layer, input)
    }

    let axis = input.dim(-1)
    let threads = min((axis + 3) / 4, 1024)
    let rows = input.size / axis
    return MiniCPMAudioMetalKernels.layerNorm(
        [input.contiguous(), weight.asType(input.dtype), bias.asType(input.dtype),
         Int32(axis), layer.eps],
        template: [("T", input.dtype)],
        grid: (rows * threads, 1, 1),
        threadGroup: (threads, 1, 1),
        outputShapes: [input.shape],
        outputDTypes: [input.dtype]
    )[0]
}

/// Native MLX Swift implementation of MiniCPM-o 4.5's audio encoder path.
public final class MiniCPMAudioModel: Module {
    public let configuration: MiniCPMAudioConfiguration

    /// Dtype used at the Whisper convolution boundary.  The host frontend
    /// emits Float32, while the converted MiniCPM audio bundle may require a
    /// lower-precision model input (the pinned bundle is BF16).  Exposing the
    /// boundary makes parity tools measure the same tensor the encoder sees.
    public var featureDType: DType { encoder.conv1.weight.dtype }

    @ModuleInfo var encoder: MiniCPMWhisperEncoder
    @ModuleInfo var projection: MiniCPMAudioProjection
    private let pool: AvgPool1d

    public init(configuration: MiniCPMAudioConfiguration = .init()) {
        self.configuration = configuration
        _encoder.wrappedValue = MiniCPMWhisperEncoder(configuration)
        _projection.wrappedValue = MiniCPMAudioProjection(configuration)
        pool = AvgPool1d(
            kernelSize: configuration.poolStep,
            stride: configuration.poolStep
        )
        super.init()
    }

    /// Encode Whisper log-mel features in time-major layout `[B, T, 80]`.
    public func encode(
        features: MLXArray,
        cache: MiniCPMAudioCache? = nil,
        useCache: Bool = true,
        prefixExtraFrames: Int = 0,
        suffixExtraFrames: Int = 0
    ) throws -> MiniCPMAudioEncoding {
        guard features.ndim == 3,
              features.dim(0) == 1,
              features.dim(1) > 0,
              features.dim(2) == configuration.melBins
        else {
            throw MiniCPMAudioError.invalidFeatures(
                "expected [1, time, \(configuration.melBins)]"
            )
        }
        let (states, nextCache) = try encoder(
            features,
            cache: cache,
            useCache: useCache,
            prefixExtraFrames: prefixExtraFrames,
            suffixExtraFrames: suffixExtraFrames,
            trace: nil
        )
        guard states.dim(1) >= configuration.poolStep else {
            throw MiniCPMAudioError.invalidFeatures(
                "chunk is shorter than one pooled audio token"
            )
        }
        // PyTorch's BF16 MPS avg_pool1d accumulates each window in FP32 and
        // rounds once at the output boundary. MLXNN's generic AvgPool1d
        // reduces the BF16 window directly, which leaves a one-ULP
        // discrepancy in streamed audio tokens.
        let embeddings = audioAveragePool1d(
            projection(states), kernelSize: configuration.poolStep)
        return MiniCPMAudioEncoding(
            embeddings: embeddings,
            encoderStates: states,
            cache: nextCache
        )
    }

    /// Internal first-chunk operator trace used by the pinned PyTorch parity
    /// fixture.  It deliberately bypasses projection/pooling so a mismatch can
    /// be attributed to the earliest encoder boundary instead of its final
    /// accumulated output.  Production inference always calls ``encode``.
    func diagnosticEncoderTrace(
        features: MLXArray,
        useCache: Bool = true
    ) throws -> [String: MLXArray] {
        var tensors: [String: MLXArray] = [:]
        _ = try encoder(
            features,
            cache: nil,
            useCache: useCache,
            prefixExtraFrames: 0,
            suffixExtraFrames: 0
        ) { name, value in
            tensors[name] = value
        }
        return tensors
    }

    /// Run only the transformer stack from an already prepared
    /// `position_added` tensor.  This opt-in diagnostic separates front-end
    /// convolution drift from recurrent layer/operator drift.
    func diagnosticLayerStackTrace(
        positionAdded: MLXArray,
        useCache: Bool = true
    ) -> [String: MLXArray] {
        var tensors: [String: MLXArray] = ["position_added": positionAdded]
        var hidden = positionAdded
        var nextLayers: [MiniCPMAudioLayerCache] = []
        if useCache { nextLayers.reserveCapacity(encoder.layers.count) }
        for index in encoder.layers.indices {
            let (output, next) = encoder.layers[index](
                hidden,
                cache: nil,
                useCache: useCache,
                layerIndex: index
            ) { name, value in
                tensors[name] = value
            }
            hidden = output
            if let next { nextLayers.append(next) }
        }
        hidden = audioLayerNorm(encoder.layerNorm, hidden)
        tensors["encoder_output"] = hidden
        return tensors
    }

    /// Run one transformer layer against an independently supplied oracle
    /// input. The normal stack trace feeds each Swift layer's output into the
    /// next layer, so a small early rounding difference can obscure the first
    /// operator that diverges. This opt-in probe keeps the layer input fixed
    /// to the official PyTorch boundary and returns all intermediate stages.
    func diagnosticIndependentLayerTrace(
        input: MLXArray,
        layerIndex: Int,
        useCache: Bool = true
    ) -> [String: MLXArray] {
        precondition(encoder.layers.indices.contains(layerIndex))
        var tensors: [String: MLXArray] = [:]
        let (output, _) = encoder.layers[layerIndex](
            input,
            cache: nil,
            useCache: useCache,
            layerIndex: layerIndex
        ) { name, value in
            tensors[name] = value
        }
        tensors[String(format: "layer_%02d_independent_output", layerIndex)] = output
        return tensors
    }

    /// Compare candidate pure-MLX LayerNorm operation boundaries against a
    /// pinned PyTorch intermediate.  This is intentionally internal and only
    /// exercised by the opt-in operator-trace test.
    func diagnosticLayerNormVariants(
        input: MLXArray,
        layerIndex: Int,
        feedForward: Bool
    ) -> [String: MLXArray] {
        precondition(encoder.layers.indices.contains(layerIndex))
        let layer = encoder.layers[layerIndex]
        let normalization = feedForward
            ? layer.finalLayerNorm
            : layer.selfAttentionLayerNorm
        let inputFloat = input.asType(.float32)
        let meanValue = mean(inputFloat, axes: [-1], keepDims: true)
        let centered = inputFloat - meanValue
        let centeredVariance = mean(
            centered * centered, axes: [-1], keepDims: true)
        let operatorVariance = variance(
            inputFloat, axes: [-1], keepDims: true)
        let momentVariance = maximum(
            mean(inputFloat * inputFloat, axes: [-1], keepDims: true)
                - meanValue * meanValue,
            MLXArray(Float(0)))

        func affine(
            _ normalized: MLXArray,
            roundBeforeAffine: Bool
        ) -> MLXArray {
            if roundBeforeAffine {
                var output = normalized.asType(input.dtype)
                if let weight = normalization.weight {
                    output = output * weight.asType(input.dtype)
                }
                if let bias = normalization.bias {
                    output = output + bias.asType(input.dtype)
                }
                return output
            }
            var output = normalized
            if let weight = normalization.weight {
                output = output * weight.asType(.float32)
            }
            if let bias = normalization.bias {
                output = output + bias.asType(.float32)
            }
            return output.asType(input.dtype)
        }

        var outputs = [
            "fast": normalization(input),
            // Exercise the exact production Metal path on an oracle input so
            // a recurrent drift can be separated from the kernel's own error.
            "mps_kernel": audioLayerNorm(normalization, input),
        ]
        for (name, value) in [
            ("centered", centeredVariance),
            ("variance_op", operatorVariance),
            ("moments", momentVariance),
        ] {
            let denominator = sqrt(value + MLXArray(normalization.eps))
            let reciprocal = MLXArray(Float(1)) / denominator
            let normalized = centered * reciprocal
            outputs["\(name)_sqrt_float_affine"] = affine(
                normalized, roundBeforeAffine: false)
            outputs["\(name)_sqrt_bf16_affine"] = affine(
                normalized, roundBeforeAffine: true)
            let normalizedRSqrt = centered
                * rsqrt(value + MLXArray(normalization.eps))
            outputs["\(name)_rsqrt_float_affine"] = affine(
                normalizedRSqrt, roundBeforeAffine: false)
            outputs["\(name)_rsqrt_bf16_affine"] = affine(
                normalizedRSqrt, roundBeforeAffine: true)
        }
        return outputs
    }

    func diagnosticAttention(
        input: MLXArray,
        layerIndex: Int
    ) -> MLXArray {
        precondition(encoder.layers.indices.contains(layerIndex))
        return encoder.layers[layerIndex].selfAttention(
            input, cache: nil, useCache: true).0
    }

    func diagnosticAttentionTrace(
        input: MLXArray,
        layerIndex: Int
    ) -> [String: MLXArray] {
        precondition(encoder.layers.indices.contains(layerIndex))
        var tensors: [String: MLXArray] = [:]
        _ = encoder.layers[layerIndex].selfAttention(
            input,
            cache: nil,
            useCache: true,
            tracePrefix: ""
        ) { name, value in
            tensors[name] = value
        }
        return tensors
    }

    func diagnosticFeedForwardVariants(
        input: MLXArray,
        layerIndex: Int
    ) -> [String: MLXArray] {
        precondition(encoder.layers.indices.contains(layerIndex))
        let layer = encoder.layers[layerIndex]
        let projected = audioLinear(layer.feedForward1, input)
        let nativeActivation = gelu(projected)
        let floatActivation = audioGELU(projected)
        let eagerFP32Activation = audioGELUEagerFP32(projected)
        return [
            "native_activation": nativeActivation,
            "native_output": audioLinear(
                layer.feedForward2, nativeActivation),
            "float_activation": floatActivation,
            "float_output": audioLinear(
                layer.feedForward2, floatActivation),
            "eager_fp32_activation": eagerFP32Activation,
            "eager_fp32_output": audioLinear(
                layer.feedForward2, eagerFP32Activation),
        ]
    }

    /// Compare the MLX average-pooling reduction boundary with explicit
    /// BF16/FP32 window reductions.  The encoder states and projection can be
    /// exactly aligned while `AvgPool1d` still differs on MPS because its
    /// BF16 mean reduction uses a different accumulation boundary from
    /// PyTorch's `avg_pool1d` kernel.  Keep these candidates internal until
    /// the streaming golden selects one.
    func diagnosticProjectionPoolingVariants(
        states: MLXArray
    ) -> [String: MLXArray] {
        let projected = projection(states)
        let batch = projected.dim(0)
        let length = projected.dim(1)
        let channels = projected.dim(2)
        let step = configuration.poolStep
        precondition(length >= step)
        let outputLength = (length - step) / step + 1
        let windows = asStrided(
            projected,
            [batch, outputLength, step, channels],
            strides: [length * channels, step * channels, channels, 1])
        let windowsFloat = windows.asType(.float32)
        return [
            "native": pool(projected),
            "bf16_window_mean": mean(windows, axes: [2]),
            "fp32_window_mean": mean(windowsFloat, axes: [2])
                .asType(projected.dtype),
            "fp32_window_sum_div": (
                sum(windowsFloat, axes: [2]) / MLXArray(Float(step))
            ).asType(projected.dtype),
        ]
    }

    /// Compare MLXNN's fused ``addMM`` Linear with explicit BF16 and FP32
    /// matmul/bias boundaries. MPS uses ``MPSNDArrayMatrixMultiplication``
    /// for these weights; keeping the candidates here lets parity tests decide
    /// which accumulation boundary is least divergent without changing the
    /// production path prematurely.
    func diagnosticFeedForwardLinearVariants(
        input: MLXArray,
        activation: MLXArray,
        layerIndex: Int
    ) -> [String: MLXArray] {
        precondition(encoder.layers.indices.contains(layerIndex))
        let layer = encoder.layers[layerIndex]

        func explicit(_ weight: MLXArray, _ bias: MLXArray?, _ value: MLXArray)
            -> MLXArray {
            var output = matmul(value, weight.transposed())
            if let bias { output = output + bias }
            return output
        }

        func floatAccum(_ weight: MLXArray, _ bias: MLXArray?, _ value: MLXArray)
            -> MLXArray {
            var output = matmul(
                value.asType(.float32), weight.asType(.float32).transposed())
            if let bias { output = output + bias.asType(.float32) }
            return output.asType(value.dtype)
        }

        return [
            "fc1_native": audioLinear(layer.feedForward1, input),
            "fc1_explicit_bf16": explicit(
                layer.feedForward1.weight, layer.feedForward1.bias, input),
            "fc1_float_accum": floatAccum(
                layer.feedForward1.weight, layer.feedForward1.bias, input),
            "fc2_native": audioLinear(layer.feedForward2, activation),
            "fc2_explicit_bf16": explicit(
                layer.feedForward2.weight, layer.feedForward2.bias, activation),
            "fc2_float_accum": floatAccum(
                layer.feedForward2.weight, layer.feedForward2.bias, activation),
        ]
    }

    /// Compare the attention projection boundaries against the pinned eager
    /// PyTorch trace.  The Q/K/V tensors are the first place where a tiny
    /// Linear/addMM rounding difference can alter every subsequent score, so
    /// keep the explicit candidates available without changing production
    /// execution until the golden trace selects one.
    func diagnosticAttentionLinearVariants(
        input: MLXArray,
        layerIndex: Int
    ) -> [String: MLXArray] {
        precondition(encoder.layers.indices.contains(layerIndex))
        let attention = encoder.layers[layerIndex].selfAttention

        func explicit(_ weight: MLXArray, _ bias: MLXArray?, _ value: MLXArray)
            -> MLXArray {
            var output = matmul(value, weight.transposed())
            if let bias { output = output + bias }
            return output
        }

        func floatAccum(_ weight: MLXArray, _ bias: MLXArray?, _ value: MLXArray)
            -> MLXArray {
            var output = matmul(
                value.asType(.float32), weight.asType(.float32).transposed())
            if let bias { output = output + bias.asType(.float32) }
            return output.asType(value.dtype)
        }

        func scaled(_ value: MLXArray, scalarType: DType) -> MLXArray {
            let factor = MLXArray(attention.scale).asType(scalarType)
            return value * factor
        }

        let qNative = audioLinear(attention.queryProjection, input)
        let kNative = audioLinear(attention.keyProjection, input)
        let vNative = audioLinear(attention.valueProjection, input)
        let qExplicit = explicit(
            attention.queryProjection.weight,
            attention.queryProjection.bias,
            input)
        let kExplicit = explicit(
            attention.keyProjection.weight,
            attention.keyProjection.bias,
            input)
        let vExplicit = explicit(
            attention.valueProjection.weight,
            attention.valueProjection.bias,
            input)
        let qFloat = floatAccum(
            attention.queryProjection.weight,
            attention.queryProjection.bias,
            input)
        let kFloat = floatAccum(
            attention.keyProjection.weight,
            attention.keyProjection.bias,
            input)
        let vFloat = floatAccum(
            attention.valueProjection.weight,
            attention.valueProjection.bias,
            input)
        let qMPS = audioLinearMPS(attention.queryProjection, input)
        let kMPS = audioLinearMPS(attention.keyProjection, input)
        let vMPS = audioLinearMPS(attention.valueProjection, input)

        return [
            "q_native_scaled_bf16": scaled(qNative, scalarType: input.dtype),
            "q_native_scaled_float": scaled(qNative, scalarType: .float32)
                .asType(input.dtype),
            "q_explicit_scaled_bf16": scaled(qExplicit, scalarType: input.dtype),
            "q_explicit_scaled_float": scaled(qExplicit, scalarType: .float32)
                .asType(input.dtype),
            "q_float_scaled_bf16": scaled(qFloat, scalarType: input.dtype),
            "q_float_scaled_float": scaled(qFloat, scalarType: .float32)
                .asType(input.dtype),
            "q_mps_scaled_bf16": scaled(qMPS, scalarType: input.dtype),
            "q_mps_scaled_float": scaled(qMPS, scalarType: .float32)
                .asType(input.dtype),
            "k_native": kNative,
            "k_explicit": kExplicit,
            "k_float": kFloat,
            "k_mps": kMPS,
            "v_native": vNative,
            "v_explicit": vExplicit,
            "v_float": vFloat,
            "v_mps": vMPS,
        ]
    }

    /// Compare convolution/GELU operation boundaries against the pinned
    /// PyTorch first-window trace.  The candidates remain useful for guarding
    /// the production convolution boundary against future MLX kernel changes.
    func diagnosticConvolutionVariants(
        features: MLXArray,
        referenceConv1GELU: MLXArray? = nil
    ) -> [String: MLXArray] {
        let input = features.asType(encoder.conv1.weight.dtype)
        let conv1FP32 = audioConv1d(encoder.conv1, input)
        let conv1BF16 = encoder.conv1(input)
        let conv1Im2ColBF16 = audioConv1dIm2Col(
            encoder.conv1, input, accumulateInFloat: false)
        let conv1Im2ColFP32 = audioConv1dIm2Col(
            encoder.conv1, input, accumulateInFloat: true)
        let conv1GELUInput = referenceConv1GELU.map {
            $0.transposed(0, 2, 1).contiguous()
        } ?? audioGELU(conv1FP32)

        let conv2FP32 = audioConv1d(encoder.conv2, conv1GELUInput)
        let conv2BF16 = encoder.conv2(conv1GELUInput)
        let conv2Im2ColBF16 = audioConv1dIm2Col(
            encoder.conv2, conv1GELUInput, accumulateInFloat: false)
        let conv2Im2ColFP32 = audioConv1dIm2Col(
            encoder.conv2, conv1GELUInput, accumulateInFloat: true)

        func ncl(_ value: MLXArray) -> MLXArray {
            value.transposed(0, 2, 1).contiguous()
        }
        return [
            "conv1_fp32": ncl(conv1FP32),
            "conv1_bf16": ncl(conv1BF16),
            "conv1_im2col_bf16": ncl(conv1Im2ColBF16),
            "conv1_im2col_fp32": ncl(conv1Im2ColFP32),
            "conv1_gelu_native": ncl(gelu(conv1FP32)),
            "conv1_gelu_fp32": ncl(audioGELU(conv1FP32)),
            "conv2_fp32": ncl(conv2FP32),
            "conv2_bf16": ncl(conv2BF16),
            "conv2_im2col_bf16": ncl(conv2Im2ColBF16),
            "conv2_im2col_fp32": ncl(conv2Im2ColFP32),
            "conv2_gelu_native": ncl(gelu(conv2FP32)),
            "conv2_gelu_fp32": ncl(audioGELU(conv2FP32)),
        ]
    }
}

/// Explicit NLC im2col + GEMM path for the Whisper convolutions.
/// The model uses groups == 1 and kernel dilation == 1, so each temporal
/// window can be flattened and multiplied by the flattened [out, K*in]
/// weight matrix.  Conv1 uses this path in production because it matches the
/// official MPS accumulation boundary; the diagnostic also exercises it for
/// Conv2 while evaluating alternatives.
private func audioConv1dIm2Col(
    _ layer: Conv1d,
    _ input: MLXArray,
    accumulateInFloat: Bool
) -> MLXArray {
    precondition(layer.groups == 1 && layer.dilation == 1)
    let batch = input.dim(0)
    let length = input.dim(1)
    let channels = input.dim(2)
    let kernel = layer.weight.dim(1)
    let padding = layer.padding
    let stride = layer.stride
    let outputLength = (length + 2 * padding - kernel) / stride + 1
    precondition(outputLength > 0)

    let padded: MLXArray
    if padding > 0 {
        let zeros = MLXArray.zeros(
            [batch, padding, channels], dtype: input.dtype)
        padded = concatenated([zeros, input, zeros], axis: 1)
    } else {
        padded = input
    }
    var windows: [MLXArray] = []
    windows.reserveCapacity(outputLength)
    for index in 0..<outputLength {
        let start = index * stride
        windows.append(padded[0..., start..<(start + kernel), 0...])
    }
    let unfolded = stacked(windows, axis: 1)
        .reshaped(batch, outputLength, kernel * channels)
    let weight = layer.weight
        .reshaped(layer.weight.dim(0), kernel * channels)
        .transposed()
    let matrixInput = accumulateInFloat ? unfolded.asType(.float32) : unfolded
    let matrixWeight = accumulateInFloat ? weight.asType(.float32) : weight
    var output = matmul(matrixInput, matrixWeight)
    if let bias = layer.bias {
        output = output + (accumulateInFloat ? bias.asType(.float32) : bias)
    }
    return output.asType(input.dtype)
}
