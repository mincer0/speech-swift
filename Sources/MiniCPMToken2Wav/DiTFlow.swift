import Foundation
import MLX
import MLXFast
import MLXNN

public final class MiniCPMDiTAttentionCache {
    public let keys: MLXArray
    public let values: MLXArray

    public init(keys: MLXArray, values: MLXArray) {
        self.keys = keys
        self.values = values
    }

    public var length: Int { keys.dim(2) }
}

public final class MiniCPMDiTConvolutionCache {
    public let first: MLXArray
    public let second: MLXArray

    public init(first: MLXArray, second: MLXArray) {
        self.first = first
        self.second = second
    }
}

public final class MiniCPMDiTLayerCache {
    public let attention: MiniCPMDiTAttentionCache
    public let convolution: MiniCPMDiTConvolutionCache

    public init(
        attention: MiniCPMDiTAttentionCache,
        convolution: MiniCPMDiTConvolutionCache
    ) {
        self.attention = attention
        self.convolution = convolution
    }
}

public final class MiniCPMDiTCache {
    public let layers: [MiniCPMDiTLayerCache]

    public init(layers: [MiniCPMDiTLayerCache]) {
        self.layers = layers
    }

    public var length: Int { layers.first?.attention.length ?? 0 }

    public func trimmed(
        promptPositions: Int = 0,
        maximumRecentPositions: Int
    ) -> MiniCPMDiTCache {
        guard length > promptPositions + maximumRecentPositions else { return self }
        return MiniCPMDiTCache(layers: layers.map { layer in
            // DiT cache order is newest-first.
            let promptStart = layer.attention.length - promptPositions
            let attention = MiniCPMDiTAttentionCache(
                keys: concatenated([
                    layer.attention.keys[
                        0..., 0..., 0 ..< maximumRecentPositions, 0...],
                    layer.attention.keys[0..., 0..., promptStart..., 0...],
                ], axis: 2),
                values: concatenated([
                    layer.attention.values[
                        0..., 0..., 0 ..< maximumRecentPositions, 0...],
                    layer.attention.values[0..., 0..., promptStart..., 0...],
                ], axis: 2))
            return MiniCPMDiTLayerCache(
                attention: attention, convolution: layer.convolution)
        })
    }
}

final class MiniCPMTimestepEmbedder: Module {
    let frequencyDimension: Int
    @ModuleInfo var linear1: Linear
    @ModuleInfo var linear2: Linear

    init(hiddenDimension: Int, frequencyDimension: Int = 256) {
        self.frequencyDimension = frequencyDimension
        _linear1.wrappedValue = Linear(frequencyDimension, hiddenDimension)
        _linear2.wrappedValue = Linear(hiddenDimension, hiddenDimension)
        super.init()
    }

    func callAsFunction(_ time: MLXArray) -> MLXArray {
        let half = frequencyDimension / 2
        let frequencies = exp(
            MLXArray(0 ..< Int32(half)).asType(.float32)
                * Float(-Foundation.log(10_000) / Double(half)))
        let arguments = time.asType(.float32).expandedDimensions(axis: 1)
            * 1_000 * frequencies.expandedDimensions(axis: 0)
        let embedding = concatenated([cos(arguments), sin(arguments)], axis: 1)
        return linear2(silu(linear1(embedding)))
    }
}

final class MiniCPMDiTAttention: Module {
    let heads: Int
    let headDimension: Int
    let scale: Float
    @ModuleInfo(key: "to_q") var query: Linear
    @ModuleInfo(key: "to_k") var key: Linear
    @ModuleInfo(key: "to_v") var value: Linear
    @ModuleInfo(key: "q_norm") var queryNorm: LayerNorm
    @ModuleInfo(key: "k_norm") var keyNorm: LayerNorm
    @ModuleInfo var proj: Linear

    init(configuration: MiniCPMFlowConfiguration) {
        heads = configuration.decoderHeads
        headDimension = configuration.decoderHeadDimension
        scale = 1 / Float(Foundation.sqrt(Double(headDimension)))
        let inner = heads * headDimension
        _query.wrappedValue = Linear(configuration.decoderHiddenDimension, inner)
        _key.wrappedValue = Linear(configuration.decoderHiddenDimension, inner)
        _value.wrappedValue = Linear(configuration.decoderHiddenDimension, inner)
        _queryNorm.wrappedValue = LayerNorm(dimensions: headDimension)
        _keyNorm.wrappedValue = LayerNorm(dimensions: headDimension)
        _proj.wrappedValue = Linear(inner, configuration.decoderHiddenDimension)
        super.init()
    }

    func callAsFunction(
        _ input: MLXArray,
        validMask: MLXArray?,
        cache: MiniCPMDiTAttentionCache?
    ) -> (MLXArray, MiniCPMDiTAttentionCache) {
        let batch = input.dim(0)
        let queryLength = input.dim(1)
        func splitHeads(_ value: MLXArray) -> MLXArray {
            value.reshaped([batch, queryLength, heads, headDimension])
                .transposed(0, 2, 1, 3)
        }
        let queries = queryNorm(splitHeads(query(input)))
        var keys = keyNorm(splitHeads(key(input)))
        var values = splitHeads(value(input))
        if let cache {
            // MiniCPM's streaming DiT intentionally stores newest chunks
            // first (`cat([current, cache])`). Attention is permutation
            // invariant here, but matching this order matters when trimming.
            keys = concatenated([keys, cache.keys], axis: 2)
            values = concatenated([values, cache.values], axis: 2)
        }
        let nextCache = MiniCPMDiTAttentionCache(keys: keys, values: values)
        let attentionMask: MLXArray? = validMask.map {
            MLX.where(
                ($0 .== 0).expandedDimensions(axis: 1),
                MLXArray(Float(-1e9)), MLXArray(Float(0))).asType(input.dtype)
        }
        let attended = MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: keys,
            values: values,
            scale: scale,
            mask: attentionMask)
            .transposed(0, 2, 1, 3)
            .reshaped([batch, queryLength, heads * headDimension])
        return (proj(attended), nextCache)
    }
}

final class MiniCPMDiTMLP: Module {
    @ModuleInfo var fc1: Linear
    @ModuleInfo var fc2: Linear

    init(configuration: MiniCPMFlowConfiguration) {
        let hidden = configuration.decoderHiddenDimension
        _fc1.wrappedValue = Linear(hidden, hidden * configuration.decoderMLPRatio)
        _fc2.wrappedValue = Linear(hidden * configuration.decoderMLPRatio, hidden)
        super.init()
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        fc2(geluApproximate(fc1(input)))
    }
}

private func miniCPMMish(_ input: MLXArray) -> MLXArray {
    input * tanh(log1p(exp(input)))
}

final class MiniCPMDiTCausalConvBlock: Module {
    let channels: Int
    @ModuleInfo var firstConv: MiniCPMConv1d
    @ModuleInfo var normalization: LayerNorm
    @ModuleInfo var secondConv: MiniCPMConv1d

    init(channels: Int) {
        self.channels = channels
        _firstConv.wrappedValue = MiniCPMConv1d(
            inputChannels: channels, outputChannels: channels, kernelSize: 3)
        _normalization.wrappedValue = LayerNorm(dimensions: channels)
        _secondConv.wrappedValue = MiniCPMConv1d(
            inputChannels: channels, outputChannels: channels, kernelSize: 3)
        super.init()
    }

    func callAsFunction(
        _ input: MLXArray,
        cache: MiniCPMDiTConvolutionCache?
    ) -> (MLXArray, MiniCPMDiTConvolutionCache) {
        let ncl = input.transposed(0, 2, 1)
        let zeros = MLXArray.zeros(
            [input.dim(0), channels, 2], dtype: input.dtype)
        let firstCache = cache?.first ?? zeros
        let firstCombined = concatenated([firstCache, ncl], axis: 2)
        var value = firstConv(firstCombined).transposed(0, 2, 1)
        value = miniCPMMish(normalization(value))
        let intermediateNCL = value.transposed(0, 2, 1)
        let secondCache = cache?.second ?? zeros
        let secondCombined = concatenated([secondCache, intermediateNCL], axis: 2)
        value = secondConv(secondCombined).transposed(0, 2, 1)
        return (value, MiniCPMDiTConvolutionCache(
            first: firstCombined[0..., 0..., (firstCombined.dim(2) - 2)...],
            second: secondCombined[0..., 0..., (secondCombined.dim(2) - 2)...]))
    }
}

final class MiniCPMDiTBlock: Module {
    let hiddenDimension: Int
    @ModuleInfo var norm1: LayerNorm
    @ModuleInfo var attn: MiniCPMDiTAttention
    @ModuleInfo var norm2: LayerNorm
    @ModuleInfo var mlp: MiniCPMDiTMLP
    @ModuleInfo var norm3: LayerNorm
    @ModuleInfo var conv: MiniCPMDiTCausalConvBlock
    @ModuleInfo(key: "adaLN_modulation") var adaptiveModulation: Linear

    init(configuration: MiniCPMFlowConfiguration) {
        hiddenDimension = configuration.decoderHiddenDimension
        _norm1.wrappedValue = LayerNorm(
            dimensions: hiddenDimension, eps: 1e-6, affine: false)
        _attn.wrappedValue = MiniCPMDiTAttention(configuration: configuration)
        _norm2.wrappedValue = LayerNorm(
            dimensions: hiddenDimension, eps: 1e-6, affine: false)
        _mlp.wrappedValue = MiniCPMDiTMLP(configuration: configuration)
        _norm3.wrappedValue = LayerNorm(
            dimensions: hiddenDimension, eps: 1e-6, affine: false)
        _conv.wrappedValue = MiniCPMDiTCausalConvBlock(channels: hiddenDimension)
        _adaptiveModulation.wrappedValue = Linear(hiddenDimension, hiddenDimension * 9)
        super.init()
    }

    func callAsFunction(
        _ input: MLXArray,
        time: MLXArray,
        validMask: MLXArray?,
        cache: MiniCPMDiTLayerCache?
    ) -> (MLXArray, MiniCPMDiTLayerCache) {
        let chunks = split(adaptiveModulation(silu(time)), parts: 9, axis: -1)
        func modulate(_ input: MLXArray, shift: MLXArray, scale: MLXArray) -> MLXArray {
            input * (1 + scale) + shift
        }
        let attentionInput = modulate(norm1(input), shift: chunks[0], scale: chunks[1])
        let attentionResult = attn(
            attentionInput, validMask: validMask, cache: cache?.attention)
        var value = input + chunks[2] * attentionResult.0

        let convolutionInput = modulate(norm3(value), shift: chunks[6], scale: chunks[7])
        let convolutionResult = conv(convolutionInput, cache: cache?.convolution)
        value = value + chunks[8] * convolutionResult.0

        let mlpInput = modulate(norm2(value), shift: chunks[3], scale: chunks[4])
        value = value + chunks[5] * mlp(mlpInput)
        return (value, MiniCPMDiTLayerCache(
            attention: attentionResult.1,
            convolution: convolutionResult.1))
    }
}

final class MiniCPMDiTFinalLayer: Module {
    @ModuleInfo(key: "adaLN_modulation") var adaptiveModulation: Linear
    @ModuleInfo(key: "norm_final") var normalization: LayerNorm
    @ModuleInfo var linear: Linear

    init(configuration: MiniCPMFlowConfiguration) {
        let hidden = configuration.decoderHiddenDimension
        _adaptiveModulation.wrappedValue = Linear(hidden, hidden * 2)
        _normalization.wrappedValue = LayerNorm(
            dimensions: hidden, eps: 1e-6, affine: false)
        _linear.wrappedValue = Linear(hidden, configuration.melDimension)
        super.init()
    }

    func callAsFunction(_ input: MLXArray, time: MLXArray) -> MLXArray {
        let chunks = split(adaptiveModulation(silu(time)), parts: 2, axis: -1)
        return linear(normalization(input) * (1 + chunks[1]) + chunks[0])
    }
}

public final class MiniCPMDiT: Module {
    public let configuration: MiniCPMFlowConfiguration
    @ModuleInfo(key: "t_embedder") var timeEmbedder: MiniCPMTimestepEmbedder
    @ModuleInfo(key: "in_proj") var inputProjection: Linear
    @ModuleInfo var blocks: [MiniCPMDiTBlock]
    @ModuleInfo(key: "final_layer") var finalLayer: MiniCPMDiTFinalLayer

    public init(configuration: MiniCPMFlowConfiguration = .init()) {
        self.configuration = configuration
        _timeEmbedder.wrappedValue = MiniCPMTimestepEmbedder(
            hiddenDimension: configuration.decoderHiddenDimension)
        _inputProjection.wrappedValue = Linear(
            configuration.decoderInputChannels,
            configuration.decoderHiddenDimension)
        _blocks.wrappedValue = (0 ..< configuration.decoderBlocks).map { _ in
            MiniCPMDiTBlock(configuration: configuration)
        }
        _finalLayer.wrappedValue = MiniCPMDiTFinalLayer(configuration: configuration)
        super.init()
    }

    public func callAsFunction(
        x: MLXArray,
        mu: MLXArray,
        time: MLXArray,
        speakers: MLXArray,
        condition: MLXArray,
        validMask: MLXArray? = nil,
        cache: MiniCPMDiTCache? = nil
    ) -> (output: MLXArray, cache: MiniCPMDiTCache) {
        let length = x.dim(2)
        let speakerFrames = MLX.repeated(
            speakers.expandedDimensions(axis: 2), count: length, axis: 2)
        let combined = concatenated([x, mu, speakerFrames, condition], axis: 1)
        var value = inputProjection(combined.transposed(0, 2, 1))
        let timeEmbedding = timeEmbedder(time).expandedDimensions(axis: 1)
        var nextLayers: [MiniCPMDiTLayerCache] = []
        for index in blocks.indices {
            let result = blocks[index](
                value,
                time: timeEmbedding,
                validMask: validMask,
                cache: cache.map { $0.layers[index] })
            value = result.0
            nextLayers.append(result.1)
        }
        value = finalLayer(value, time: timeEmbedding).transposed(0, 2, 1)
        return (value, MiniCPMDiTCache(layers: nextLayers))
    }
}

public final class MiniCPMFlowMatchingCache {
    public let odeStepCaches: [MiniCPMDiTCache]
    public let totalFrames: Int

    public init(odeStepCaches: [MiniCPMDiTCache], totalFrames: Int? = nil) {
        self.odeStepCaches = odeStepCaches
        self.totalFrames = totalFrames ?? (odeStepCaches.first?.length ?? 0)
    }

    public var length: Int { odeStepCaches.first?.length ?? 0 }

    public func trimmed(
        promptPositions: Int = 0,
        maximumRecentPositions: Int
    ) -> MiniCPMFlowMatchingCache {
        MiniCPMFlowMatchingCache(
            odeStepCaches: odeStepCaches.map {
                $0.trimmed(
                    promptPositions: promptPositions,
                    maximumRecentPositions: maximumRecentPositions)
            },
            totalFrames: totalFrames)
    }
}

public final class MiniCPMConditionalFlowMatching: Module {
    public let configuration: MiniCPMFlowConfiguration
    @ModuleInfo(key: "estimator") public var estimator: MiniCPMDiT
    public var fixedNoise: MLXArray?

    public init(configuration: MiniCPMFlowConfiguration = .init()) {
        self.configuration = configuration
        _estimator.wrappedValue = MiniCPMDiT(configuration: configuration)
        super.init()
    }

    private func noise(batch: Int, offset: Int, length: Int) -> MLXArray {
        if let fixedNoise,
           fixedNoise.dim(0) == batch,
           fixedNoise.dim(2) >= offset + length {
            return fixedNoise[0..., 0..., offset ..< (offset + length)].asType(.float32)
        }
        return MLXRandom.normal(
            [batch, configuration.melDimension, length], key: MLXRandom.key(UInt64(offset)))
    }

    public func generate(
        mu: MLXArray,
        speakers: MLXArray,
        condition: MLXArray,
        validMask: MLXArray? = nil,
        cache: MiniCPMFlowMatchingCache? = nil,
        odeSteps: Int? = nil
    ) -> (output: MLXArray, cache: MiniCPMFlowMatchingCache) {
        let batch = mu.dim(0)
        let length = mu.dim(2)
        let offset = cache?.totalFrames ?? 0
        var value = noise(batch: batch, offset: offset, length: length)
        let muBatch = concatenated([mu, MLXArray.zeros(like: mu)], axis: 0)
        let speakerBatch = concatenated([
            speakers, MLXArray.zeros(like: speakers),
        ], axis: 0)
        let conditionBatch = concatenated([
            condition, MLXArray.zeros(like: condition),
        ], axis: 0)
        let maskBatch = validMask.map { concatenated([$0, $0], axis: 0) }
        var nextStepCaches: [MiniCPMDiTCache] = []
        let stepCount = odeSteps ?? configuration.odeSteps
        let schedule = (0 ... stepCount).map { index -> Float in
            let time = Float(index) / Float(stepCount)
            return 1 - Foundation.cos(time * 0.5 * Float.pi)
        }
        for step in 0 ..< stepCount {
            let currentTime = schedule[step]
            let delta = schedule[step + 1] - currentTime
            let xBatch = concatenated([value, value], axis: 0)
            let time = MLXArray([Float](repeating: currentTime, count: batch * 2))
            let result = estimator(
                x: xBatch,
                mu: muBatch,
                time: time,
                speakers: speakerBatch,
                condition: conditionBatch,
                validMask: maskBatch,
                cache: cache.map { $0.odeStepCaches[step] })
            let conditioned = result.output[0 ..< batch].asType(.float32)
            let unconditioned = result.output[batch...].asType(.float32)
            let velocity = (1 + configuration.classifierFreeGuidance) * conditioned
                - configuration.classifierFreeGuidance * unconditioned
            value = value + delta * velocity
            eval(value)
            nextStepCaches.append(result.cache)
        }
        return (value, MiniCPMFlowMatchingCache(
            odeStepCaches: nextStepCaches,
            totalFrames: offset + length))
    }
}
