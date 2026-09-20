import Foundation
import MLX
import MLXNN

public final class MiniCPMConformerAttentionCache {
    public let keys: MLXArray
    public let values: MLXArray

    public init(keys: MLXArray, values: MLXArray) {
        self.keys = keys
        self.values = values
    }

    public var length: Int { keys.dim(2) }
}

public final class MiniCPMUpsampleConformerCache {
    public let firstAttention: [MiniCPMConformerAttentionCache]
    public let secondAttention: [MiniCPMConformerAttentionCache]
    public let preLookaheadConvolution: MLXArray
    public let upsampleConvolution: MLXArray

    public init(
        firstAttention: [MiniCPMConformerAttentionCache],
        secondAttention: [MiniCPMConformerAttentionCache],
        preLookaheadConvolution: MLXArray,
        upsampleConvolution: MLXArray
    ) {
        self.firstAttention = firstAttention
        self.secondAttention = secondAttention
        self.preLookaheadConvolution = preLookaheadConvolution
        self.upsampleConvolution = upsampleConvolution
    }

    public var inputPositions: Int { firstAttention.first?.length ?? 0 }
    public var outputPositions: Int { secondAttention.first?.length ?? 0 }

    public func trimmed(
        promptInputPositions: Int = 0,
        promptOutputPositions: Int = 0,
        maximumRecentOutputPositions: Int
    ) -> MiniCPMUpsampleConformerCache {
        let maximumRecentInputPositions = maximumRecentOutputPositions / 2
        func trim(
            _ caches: [MiniCPMConformerAttentionCache],
            prompt: Int,
            recent: Int
        ) -> [MiniCPMConformerAttentionCache] {
            caches.map { cache in
                guard cache.length > prompt + recent else { return cache }
                let recentStart = cache.length - recent
                let keys = concatenated([
                    cache.keys[0..., 0..., 0 ..< prompt, 0...],
                    cache.keys[0..., 0..., recentStart..., 0...],
                ], axis: 2)
                let values = concatenated([
                    cache.values[0..., 0..., 0 ..< prompt, 0...],
                    cache.values[0..., 0..., recentStart..., 0...],
                ], axis: 2)
                return MiniCPMConformerAttentionCache(
                    keys: keys, values: values)
            }
        }
        return MiniCPMUpsampleConformerCache(
            firstAttention: trim(
                firstAttention, prompt: promptInputPositions,
                recent: maximumRecentInputPositions),
            secondAttention: trim(
                secondAttention, prompt: promptOutputPositions,
                recent: maximumRecentOutputPositions),
            preLookaheadConvolution: preLookaheadConvolution,
            upsampleConvolution: upsampleConvolution)
    }
}

final class MiniCPMLinearEmbedding: Module {
    @ModuleInfo var linear: Linear
    @ModuleInfo var normalization: LayerNorm
    let scale: Float

    init(inputDimension: Int, outputDimension: Int) {
        _linear.wrappedValue = Linear(inputDimension, outputDimension)
        _normalization.wrappedValue = LayerNorm(
            dimensions: outputDimension, eps: 1e-5)
        scale = Float(Foundation.sqrt(Double(outputDimension)))
        super.init()
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        normalization(linear(input)) * scale
    }
}

func miniCPMRelativePositionEncoding(size: Int, dimension: Int) -> MLXArray {
    var positive = [Float](repeating: 0, count: size * dimension)
    var negative = [Float](repeating: 0, count: size * dimension)
    for position in 0 ..< size {
        for feature in stride(from: 0, to: dimension, by: 2) {
            let divisor = Foundation.exp(
                Double(feature) * -(Foundation.log(10_000) / Double(dimension)))
            let angle = Double(position) * divisor
            positive[position * dimension + feature] = Float(Foundation.sin(angle))
            positive[position * dimension + feature + 1] = Float(Foundation.cos(angle))
            negative[position * dimension + feature] = Float(Foundation.sin(-angle))
            negative[position * dimension + feature + 1] = Float(Foundation.cos(-angle))
        }
    }
    let positiveArray = MLXArray(positive, [size, dimension])
    let positiveIndices = MLXArray((0 ..< size).reversed().map(Int32.init))
    let reversedPositive = positiveArray.take(positiveIndices, axis: 0)
    if size == 1 { return reversedPositive.expandedDimensions(axis: 0) }
    let negativeArray = MLXArray(negative, [size, dimension])[1...]
    return concatenated([reversedPositive, negativeArray], axis: 0)
        .expandedDimensions(axis: 0)
}

final class MiniCPMRelativeAttention: Module {
    let heads: Int
    let headDimension: Int
    let scale: Float

    @ParameterInfo(key: "pos_bias_u") var positionBiasU: MLXArray
    @ParameterInfo(key: "pos_bias_v") var positionBiasV: MLXArray
    @ModuleInfo(key: "linear_q") var query: Linear
    @ModuleInfo(key: "linear_k") var key: Linear
    @ModuleInfo(key: "linear_v") var value: Linear
    @ModuleInfo(key: "linear_out") var output: Linear
    @ModuleInfo(key: "linear_pos") var position: Linear

    init(dimension: Int, heads: Int) {
        self.heads = heads
        headDimension = dimension / heads
        scale = 1 / Float(Foundation.sqrt(Double(headDimension)))
        _positionBiasU.wrappedValue = MLXArray.zeros([heads, headDimension])
        _positionBiasV.wrappedValue = MLXArray.zeros([heads, headDimension])
        _query.wrappedValue = Linear(dimension, dimension)
        _key.wrappedValue = Linear(dimension, dimension)
        _value.wrappedValue = Linear(dimension, dimension)
        _output.wrappedValue = Linear(dimension, dimension)
        _position.wrappedValue = Linear(dimension, dimension, bias: false)
        super.init()
    }

    private func relativeShift(_ input: MLXArray, keyLength: Int) -> MLXArray {
        let batch = input.dim(0)
        let headCount = input.dim(1)
        let queryLength = input.dim(2)
        let positionLength = input.dim(3)
        let zeros = MLXArray.zeros(
            [batch, headCount, queryLength, 1], dtype: input.dtype)
        let padded = concatenated([zeros, input], axis: 3)
            .reshaped([batch, headCount, positionLength + 1, queryLength])
        let shifted = padded[0..., 0..., 1..., 0...]
            .reshaped([batch, headCount, queryLength, positionLength])
        return shifted[0..., 0..., 0..., 0 ..< keyLength]
    }

    func callAsFunction(
        _ input: MLXArray,
        positionEmbedding: MLXArray,
        validMask: MLXArray? = nil,
        cache: MiniCPMConformerAttentionCache? = nil
    ) -> (MLXArray, MiniCPMConformerAttentionCache) {
        let batch = input.dim(0)
        let queryLength = input.dim(1)
        func splitHeads(_ tensor: MLXArray) -> MLXArray {
            tensor.reshaped([batch, -1, heads, headDimension])
                .transposed(0, 2, 1, 3)
        }
        let queries = splitHeads(query(input))
        var keys = splitHeads(key(input))
        var values = splitHeads(value(input))
        if let cache {
            keys = concatenated([cache.keys, keys], axis: 2)
            values = concatenated([cache.values, values], axis: 2)
        }
        let nextCache = MiniCPMConformerAttentionCache(keys: keys, values: values)
        let keyLength = keys.dim(2)
        let queryTransposed = queries.transposed(0, 2, 1, 3)
        let biasedU = (queryTransposed + positionBiasU).transposed(0, 2, 1, 3)
        let biasedV = (queryTransposed + positionBiasV).transposed(0, 2, 1, 3)
        let contentScores = matmul(biasedU, keys.transposed(0, 1, 3, 2))

        let positionBatch = positionEmbedding.dim(0)
        let projectedPosition = position(positionEmbedding)
            .reshaped([positionBatch, -1, heads, headDimension])
            .transposed(0, 2, 1, 3)
        var positionScores = matmul(
            biasedV, projectedPosition.transposed(0, 1, 3, 2))
        if positionScores.shape != contentScores.shape {
            positionScores = relativeShift(positionScores, keyLength: keyLength)
        }
        var scores = (contentScores + positionScores) * scale
        if let validMask {
            let invalid = (validMask .== 0).expandedDimensions(axis: 1)
            scores = MLX.where(invalid, MLXArray(Float(-1e9)), scores)
        }
        let probabilities = softmax(scores, axis: -1)
        let attended = matmul(probabilities, values)
            .transposed(0, 2, 1, 3)
            .reshaped([batch, queryLength, heads * headDimension])
        return (output(attended), nextCache)
    }
}

final class MiniCPMConformerFeedForward: Module {
    @ModuleInfo(key: "w_1") var input: Linear
    @ModuleInfo(key: "w_2") var output: Linear

    init(dimension: Int, hiddenDimension: Int) {
        _input.wrappedValue = Linear(dimension, hiddenDimension)
        _output.wrappedValue = Linear(hiddenDimension, dimension)
        super.init()
    }

    func callAsFunction(_ value: MLXArray) -> MLXArray {
        output(silu(input(value)))
    }
}

final class MiniCPMConformerBlock: Module {
    @ModuleInfo(key: "self_attn") var attention: MiniCPMRelativeAttention
    @ModuleInfo(key: "feed_forward") var feedForward: MiniCPMConformerFeedForward
    @ModuleInfo(key: "norm_ff") var feedForwardNorm: LayerNorm
    @ModuleInfo(key: "norm_mha") var attentionNorm: LayerNorm

    init(configuration: MiniCPMFlowConfiguration) {
        _attention.wrappedValue = MiniCPMRelativeAttention(
            dimension: configuration.tokenDimension,
            heads: configuration.encoderHeads)
        _feedForward.wrappedValue = MiniCPMConformerFeedForward(
            dimension: configuration.tokenDimension,
            hiddenDimension: configuration.encoderFeedForwardDimension)
        _feedForwardNorm.wrappedValue = LayerNorm(
            dimensions: configuration.tokenDimension, eps: 1e-12)
        _attentionNorm.wrappedValue = LayerNorm(
            dimensions: configuration.tokenDimension, eps: 1e-12)
        super.init()
    }

    func callAsFunction(
        _ input: MLXArray,
        positionEmbedding: MLXArray,
        validMask: MLXArray?,
        cache: MiniCPMConformerAttentionCache?
    ) -> (MLXArray, MiniCPMConformerAttentionCache) {
        let (attentionOutput, nextCache) = attention(
            attentionNorm(input),
            positionEmbedding: positionEmbedding,
            validMask: validMask,
            cache: cache)
        let attended = input + attentionOutput
        return (attended + feedForward(feedForwardNorm(attended)), nextCache)
    }
}

final class MiniCPMPreLookahead: Module {
    let length: Int
    @ModuleInfo var conv1: MiniCPMConv1d
    @ModuleInfo var conv2: MiniCPMConv1d

    init(channels: Int, length: Int) {
        self.length = length
        _conv1.wrappedValue = MiniCPMConv1d(
            inputChannels: channels, outputChannels: channels,
            kernelSize: length + 1)
        _conv2.wrappedValue = MiniCPMConv1d(
            inputChannels: channels, outputChannels: channels,
            kernelSize: 3)
        super.init()
    }

    func offline(_ input: MLXArray) -> MLXArray {
        let ncl = input.transposed(0, 2, 1)
        let right = MLXArray.zeros(
            [ncl.dim(0), ncl.dim(1), length], dtype: ncl.dtype)
        var value = conv1(concatenated([ncl, right], axis: 2))
        value = miniCPMLeakyReLU(value, slope: 0.01)
        let left = MLXArray.zeros([value.dim(0), value.dim(1), 2], dtype: value.dtype)
        value = conv2(concatenated([left, value], axis: 2))
        return value.transposed(0, 2, 1) + input
    }

    func streaming(
        _ input: MLXArray,
        cache: MLXArray?
    ) -> (output: MLXArray, cache: MLXArray) {
        var value = conv1(input.transposed(0, 2, 1))
        value = miniCPMLeakyReLU(value, slope: 0.01)
        let nextCache = value[0..., 0..., (value.dim(2) - 2)...]
        let previous = cache ?? MLXArray.zeros(
            [value.dim(0), value.dim(1), 2], dtype: value.dtype)
        value = conv2(concatenated([previous, value], axis: 2))
        let residual = input[0..., 0 ..< value.dim(2), 0...]
        return (value.transposed(0, 2, 1) + residual, nextCache)
    }
}

final class MiniCPMConformerUpsample: Module {
    let stride: Int
    @ModuleInfo var conv: MiniCPMConv1d

    init(channels: Int, stride: Int) {
        self.stride = stride
        _conv.wrappedValue = MiniCPMConv1d(
            inputChannels: channels, outputChannels: channels,
            kernelSize: stride * 2 + 1)
        super.init()
    }

    private func repeatFrames(_ input: MLXArray) -> MLXArray {
        MLX.repeated(input.expandedDimensions(axis: 3), count: stride, axis: 3)
            .reshaped([input.dim(0), input.dim(1), input.dim(2) * stride])
    }

    func offline(_ input: MLXArray) -> MLXArray {
        let upsampled = repeatFrames(input.transposed(0, 2, 1))
        let left = MLXArray.zeros(
            [upsampled.dim(0), upsampled.dim(1), stride * 2], dtype: upsampled.dtype)
        return conv(concatenated([left, upsampled], axis: 2)).transposed(0, 2, 1)
    }

    func streaming(
        _ input: MLXArray,
        cache: MLXArray?
    ) -> (output: MLXArray, cache: MLXArray) {
        let upsampled = repeatFrames(input.transposed(0, 2, 1))
        let previous = cache ?? MLXArray.zeros(
            [upsampled.dim(0), upsampled.dim(1), stride * 2], dtype: upsampled.dtype)
        let combined = concatenated([previous, upsampled], axis: 2)
        let nextCache = combined[0..., 0..., (combined.dim(2) - stride * 2)...]
        return (conv(combined).transposed(0, 2, 1), nextCache)
    }
}

public final class MiniCPMUpsampleConformerEncoder: Module {
    public let configuration: MiniCPMFlowConfiguration
    @ModuleInfo var embed: MiniCPMLinearEmbedding
    @ModuleInfo(key: "pre_lookahead_layer") var preLookahead: MiniCPMPreLookahead
    @ModuleInfo var encoders: [MiniCPMConformerBlock]
    @ModuleInfo(key: "up_layer") var upsample: MiniCPMConformerUpsample
    @ModuleInfo(key: "up_embed") var upsampleEmbed: MiniCPMLinearEmbedding
    @ModuleInfo(key: "up_encoders") var upsampleEncoders: [MiniCPMConformerBlock]
    @ModuleInfo(key: "after_norm") var afterNorm: LayerNorm

    public init(configuration: MiniCPMFlowConfiguration = .init()) {
        self.configuration = configuration
        _embed.wrappedValue = MiniCPMLinearEmbedding(
            inputDimension: configuration.tokenDimension,
            outputDimension: configuration.tokenDimension)
        _preLookahead.wrappedValue = MiniCPMPreLookahead(
            channels: configuration.tokenDimension,
            length: configuration.preLookahead)
        _encoders.wrappedValue = (0 ..< configuration.encoderBlocks).map { _ in
            MiniCPMConformerBlock(configuration: configuration)
        }
        _upsample.wrappedValue = MiniCPMConformerUpsample(
            channels: configuration.tokenDimension,
            stride: configuration.upsampleStride)
        _upsampleEmbed.wrappedValue = MiniCPMLinearEmbedding(
            inputDimension: configuration.tokenDimension,
            outputDimension: configuration.tokenDimension)
        _upsampleEncoders.wrappedValue = (0 ..< configuration.encoderUpsampleBlocks).map { _ in
            MiniCPMConformerBlock(configuration: configuration)
        }
        _afterNorm.wrappedValue = LayerNorm(
            dimensions: configuration.tokenDimension, eps: 1e-5)
        super.init()
    }

    public func callAsFunction(_ input: MLXArray, validMask: MLXArray? = nil) -> MLXArray {
        var value = embed(input)
        value = preLookahead.offline(value)
        let firstPosition = miniCPMRelativePositionEncoding(
            size: value.dim(1), dimension: configuration.tokenDimension)
            .asType(value.dtype)
        for encoder in encoders {
            (value, _) = encoder(
                value, positionEmbedding: firstPosition,
                validMask: validMask, cache: nil)
        }
        value = upsample.offline(value)
        value = upsampleEmbed(value)
        let secondPosition = miniCPMRelativePositionEncoding(
            size: value.dim(1), dimension: configuration.tokenDimension)
            .asType(value.dtype)
        for encoder in upsampleEncoders {
            (value, _) = encoder(
                value, positionEmbedding: secondPosition,
                validMask: nil, cache: nil)
        }
        return afterNorm(value)
    }

    public func streaming(
        _ input: MLXArray,
        cache: MiniCPMUpsampleConformerCache? = nil,
        isFinal: Bool = false
    ) -> (output: MLXArray, cache: MiniCPMUpsampleConformerCache) {
        var value = embed(input)
        if isFinal {
            value = concatenated([
                value,
                MLXArray.zeros([
                    value.dim(0), configuration.preLookahead, value.dim(2),
                ], dtype: value.dtype),
            ], axis: 1)
        }
        let lookahead = preLookahead.streaming(
            value, cache: cache?.preLookaheadConvolution)
        value = lookahead.output
        let firstOffset = cache?.firstAttention.first?.length ?? 0
        let firstPosition = miniCPMRelativePositionEncoding(
            size: firstOffset + value.dim(1),
            dimension: configuration.tokenDimension).asType(value.dtype)
        var firstCaches: [MiniCPMConformerAttentionCache] = []
        for index in encoders.indices {
            let result = encoders[index](
                value,
                positionEmbedding: firstPosition,
                validMask: nil,
                cache: cache.map { $0.firstAttention[index] })
            value = result.0
            firstCaches.append(result.1)
        }
        let upsampled = upsample.streaming(
            value, cache: cache?.upsampleConvolution)
        value = upsampleEmbed(upsampled.output)
        let secondOffset = cache?.secondAttention.first?.length ?? 0
        let secondPosition = miniCPMRelativePositionEncoding(
            size: secondOffset + value.dim(1),
            dimension: configuration.tokenDimension).asType(value.dtype)
        var secondCaches: [MiniCPMConformerAttentionCache] = []
        for index in upsampleEncoders.indices {
            let result = upsampleEncoders[index](
                value,
                positionEmbedding: secondPosition,
                validMask: nil,
                cache: cache.map { $0.secondAttention[index] })
            value = result.0
            secondCaches.append(result.1)
        }
        let nextCache = MiniCPMUpsampleConformerCache(
            firstAttention: firstCaches,
            secondAttention: secondCaches,
            preLookaheadConvolution: lookahead.cache,
            upsampleConvolution: upsampled.cache)
        return (afterNorm(value), nextCache)
    }
}
