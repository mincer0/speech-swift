import Foundation
import MLX
import MLXCommon
import MLXFast
import MLXNN

// MARK: - Public packed/grouped values

/// A target patch grid for one packed image (height, width).
public struct MiniCPMTargetSize: Equatable, Sendable {
    public let height: Int
    public let width: Int

    public init(height: Int, width: Int) {
        self.height = height
        self.width = width
    }

    public var patchCount: Int { height * width }

    public var array: MLXArray {
        MLXArray([Int32(height), Int32(width)])
    }
}

/// One source image and its optional HD tiles after image preprocessing.
///
/// `overview` is always first. `slices` contains only HD tiles; keeping the
/// distinction here prevents a caller from accidentally wrapping all visual
/// features in one `<image>` marker.
public struct MiniCPMImageGroup {
    public let overview: MiniCPMPackedImage
    public let slices: [MiniCPMPackedImage]

    public init(overview: MiniCPMPackedImage, slices: [MiniCPMPackedImage] = []) {
        self.overview = overview
        self.slices = slices
    }

    public var all: [MiniCPMPackedImage] { [overview] + slices }
}

/// A packed image tensor ready for the SigLIP tower.
///
/// `pixels` is `[C, patchSize, patchCount * patchSize]`, the exact layout
/// produced by MiniCPM's `torch.nn.functional.unfold` helper. It is not a
/// conventional `[C,H,W]` image tensor.
public struct MiniCPMPackedImage {
    public let pixels: MLXArray
    public let targetSize: MiniCPMTargetSize

    public init(pixels: MLXArray, targetSize: MiniCPMTargetSize) {
        self.pixels = pixels
        self.targetSize = targetSize
    }
}

/// Grouped output from the visual encoder. Each tensor has shape `[64, 4096]`
/// for the default MiniCPM-o 4.5 configuration.
public struct MiniCPMVisionGroup {
    public let overview: MLXArray
    public let slices: [MLXArray]

    public init(overview: MLXArray, slices: [MLXArray] = []) {
        self.overview = overview
        self.slices = slices
    }

    public var all: [MLXArray] { [overview] + slices }
}

// MARK: - SigLIP tower

final class MiniCPMSiglipAttention: Module {
    private let heads: Int
    private let headDimension: Int
    private let scale: Float

    @ModuleInfo(key: "q_proj") var queryProjection: Linear
    @ModuleInfo(key: "k_proj") var keyProjection: Linear
    @ModuleInfo(key: "v_proj") var valueProjection: Linear
    @ModuleInfo(key: "out_proj") var outputProjection: Linear

    init(configuration: MiniCPMVisionConfiguration) {
        heads = configuration.visionHeads
        headDimension = configuration.visionHiddenSize / configuration.visionHeads
        scale = 1 / sqrt(Float(headDimension))
        _queryProjection.wrappedValue = Linear(
            configuration.visionHiddenSize,
            configuration.visionHiddenSize,
            bias: true
        )
        _keyProjection.wrappedValue = Linear(
            configuration.visionHiddenSize,
            configuration.visionHiddenSize,
            bias: true
        )
        _valueProjection.wrappedValue = Linear(
            configuration.visionHiddenSize,
            configuration.visionHiddenSize,
            bias: true
        )
        _outputProjection.wrappedValue = Linear(
            configuration.visionHiddenSize,
            configuration.visionHiddenSize,
            bias: true
        )
        super.init()
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        let query = queryProjection(input)
        let key = keyProjection(input)
        let value = valueProjection(input)
        return outputProjection(SDPA.multiHead(
            q: query,
            k: key,
            v: value,
            numHeads: heads,
            headDim: headDimension,
            scale: scale
        ))
    }
}

final class MiniCPMSiglipMLP: Module {
    @ModuleInfo(key: "fc1") var first: Linear
    @ModuleInfo(key: "fc2") var second: Linear

    init(configuration: MiniCPMVisionConfiguration) {
        _first.wrappedValue = Linear(
            configuration.visionHiddenSize,
            configuration.visionIntermediateSize,
            bias: true
        )
        _second.wrappedValue = Linear(
            configuration.visionIntermediateSize,
            configuration.visionHiddenSize,
            bias: true
        )
        super.init()
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        // MiniCPM's SigLIP config uses ``gelu_pytorch_tanh`` rather than the
        // exact-erf GELU.  MLX's tanh approximation is the same 0.044715
        // formulation used by PyTorch's implementation.
        second(geluApproximate(first(input)))
    }
}

final class MiniCPMSiglipLayer: Module {
    @ModuleInfo(key: "layer_norm1") var firstNorm: LayerNorm
    @ModuleInfo(key: "layer_norm2") var secondNorm: LayerNorm
    @ModuleInfo(key: "self_attn") var attention: MiniCPMSiglipAttention
    @ModuleInfo var mlp: MiniCPMSiglipMLP

    init(configuration: MiniCPMVisionConfiguration) {
        _firstNorm.wrappedValue = LayerNorm(
            dimensions: configuration.visionHiddenSize,
            eps: configuration.layerNormEpsilon
        )
        _secondNorm.wrappedValue = LayerNorm(
            dimensions: configuration.visionHiddenSize,
            eps: configuration.layerNormEpsilon
        )
        _attention.wrappedValue = MiniCPMSiglipAttention(configuration: configuration)
        _mlp.wrappedValue = MiniCPMSiglipMLP(configuration: configuration)
        super.init()
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        let attended = input + attention(firstNorm(input))
        return attended + mlp(secondNorm(attended))
    }
}

final class MiniCPMSiglipEmbeddings: Module {
    private let configuration: MiniCPMVisionConfiguration
    private let patchesPerSide: Int

    @ModuleInfo(key: "patch_embedding") var patchEmbedding: Conv2d
    @ModuleInfo(key: "position_embedding") var positionEmbedding: Embedding

    init(configuration: MiniCPMVisionConfiguration) {
        self.configuration = configuration
        patchesPerSide = configuration.imageSize / configuration.patchSize
        _patchEmbedding.wrappedValue = Conv2d(
            inputChannels: configuration.visionChannels,
            outputChannels: configuration.visionHiddenSize,
            kernelSize: .init((configuration.patchSize, configuration.patchSize)),
            stride: .init((configuration.patchSize, configuration.patchSize)),
            padding: 0,
            bias: true
        )
        _positionEmbedding.wrappedValue = Embedding(
            embeddingCount: patchesPerSide * patchesPerSide,
            dimensions: configuration.visionHiddenSize
        )
        super.init()
    }

    /// Packed input `[B,C,P,N*P]` is transposed to MLX Conv2d's NHWC layout.
    func callAsFunction(_ pixels: MLXArray, targetSizes: MLXArray) throws -> MLXArray {
        guard pixels.ndim == 4,
              pixels.dim(1) == configuration.visionChannels
        else {
            throw MiniCPMVisionError.invalidInput(
                "packed pixels must have shape [B,3,patchSize,patchCount*patchSize]"
            )
        }
        guard targetSizes.ndim == 2,
              targetSizes.dim(0) == pixels.dim(0),
              targetSizes.dim(1) == 2
        else {
            throw MiniCPMVisionError.invalidInput(
                "target sizes must have shape [B,2]"
            )
        }

        let batch = pixels.dim(0)
        let nhwc = pixels.transposed(0, 2, 3, 1)
        let convolved = patchEmbedding(nhwc)
        let sequenceLength = convolved.dim(1) * convolved.dim(2)
        var hidden = convolved.reshaped([batch, sequenceLength, configuration.visionHiddenSize])

        // Build the fractional SigLIP buckets on the CPU. Target grids are
        // tiny (at most 70×70), and this avoids a device-side integer loop in
        // every image call while preserving NumPy searchsorted(..., side=right).
        eval(targetSizes)
        let target = targetSizes.asType(.int32).asArray(Int32.self)
        var ids = [Int32](repeating: 0, count: batch * sequenceLength)
        for b in 0..<batch {
            let height = max(1, Int(target[b * 2]))
            let width = max(1, Int(target[b * 2 + 1]))
            guard height * width <= sequenceLength else {
                throw MiniCPMVisionError.invalidInput(
                    "target grid \(height)x\(width) exceeds packed sequence length \(sequenceLength)"
                )
            }
            guard height <= patchesPerSide, width <= patchesPerSide else {
                throw MiniCPMVisionError.invalidInput(
                    "target grid \(height)x\(width) exceeds positional grid \(patchesPerSide)x\(patchesPerSide)"
                )
            }
            for row in 0..<height {
                let rowBucket = Self.bucket(row, count: height, side: patchesPerSide)
                for column in 0..<width {
                    let columnBucket = Self.bucket(column, count: width, side: patchesPerSide)
                    ids[b * sequenceLength + row * width + column] = Int32(
                        rowBucket * patchesPerSide + columnBucket
                    )
                }
            }
        }
        let positionIDs = MLXArray(ids, [batch, sequenceLength])
        hidden = hidden + positionEmbedding(positionIDs)
        return hidden
    }

    /// NumPy `searchsorted(boundaries, frac, side="right")` for the regular
    /// boundaries `1/side, ..., (side-1)/side`.
    private static func bucket(_ index: Int, count: Int, side: Int) -> Int {
        guard count > 0 else { return 0 }
        let fraction = Float(index) / Float(count)
        var bucket = 0
        while bucket < side - 1,
              fraction >= Float(bucket + 1) / Float(side)
        {
            bucket += 1
        }
        return bucket
    }
}

/// Explicit ``encoder.layers`` container matching the HuggingFace SigLIP
/// parameter tree.  A flat array on the tower would produce ``vision.layers``
/// and fail strict loading of the converted ``vision.encoder.layers.*`` keys.
final class MiniCPMSiglipEncoder: Module {
    @ModuleInfo(key: "layers") var layers: [MiniCPMSiglipLayer]

    init(configuration: MiniCPMVisionConfiguration) {
        _layers.wrappedValue = (0..<configuration.visionLayers).map { _ in
            MiniCPMSiglipLayer(configuration: configuration)
        }
        super.init()
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        var hidden = input
        for layer in layers {
            hidden = layer(hidden)
        }
        return hidden
    }
}

final class MiniCPMSiglipTower: Module {
    let configuration: MiniCPMVisionConfiguration

    @ModuleInfo var embeddings: MiniCPMSiglipEmbeddings
    @ModuleInfo var encoder: MiniCPMSiglipEncoder
    @ModuleInfo(key: "post_layernorm") var postLayerNorm: LayerNorm

    init(configuration: MiniCPMVisionConfiguration) {
        self.configuration = configuration
        _embeddings.wrappedValue = MiniCPMSiglipEmbeddings(configuration: configuration)
        _encoder.wrappedValue = MiniCPMSiglipEncoder(configuration: configuration)
        _postLayerNorm.wrappedValue = LayerNorm(
            dimensions: configuration.visionHiddenSize,
            eps: configuration.layerNormEpsilon
        )
        super.init()
    }

    func callAsFunction(_ pixels: MLXArray, targetSizes: MLXArray) throws -> MLXArray {
        var hidden = try embeddings(pixels, targetSizes: targetSizes)
        hidden = encoder(hidden)
        return postLayerNorm(hidden)
    }
}

// MARK: - MiniCPM resampler

final class MiniCPMCrossAttention: Module {
    private let heads: Int
    private let headDimension: Int
    private let scale: Float

    @ModuleInfo(key: "q_proj") var queryProjection: Linear
    @ModuleInfo(key: "k_proj") var keyProjection: Linear
    @ModuleInfo(key: "v_proj") var valueProjection: Linear
    @ModuleInfo(key: "out_proj") var outputProjection: Linear

    init(dimensions: Int, heads: Int) {
        self.heads = heads
        headDimension = dimensions / heads
        scale = 1 / sqrt(Float(headDimension))
        _queryProjection.wrappedValue = Linear(dimensions, dimensions, bias: true)
        _keyProjection.wrappedValue = Linear(dimensions, dimensions, bias: true)
        _valueProjection.wrappedValue = Linear(dimensions, dimensions, bias: true)
        _outputProjection.wrappedValue = Linear(dimensions, dimensions, bias: true)
        super.init()
    }

    func callAsFunction(
        queries: MLXArray,
        keys: MLXArray,
        values: MLXArray,
        keyPaddingMask: MLXArray? = nil
    ) -> MLXArray {
        let q = queryProjection(queries)
            .reshaped([queries.dim(0), queries.dim(1), heads, headDimension])
            .transposed(0, 2, 1, 3)
        let k = keyProjection(keys)
            .reshaped([keys.dim(0), keys.dim(1), heads, headDimension])
            .transposed(0, 2, 1, 3)
        let v = valueProjection(values)
            .reshaped([values.dim(0), values.dim(1), heads, headDimension])
            .transposed(0, 2, 1, 3)
        var mask: MLXArray? = nil
        if let keyPaddingMask {
            // The Python implementation treats true as padding and false as
            // valid. MLX's additive mask is broadcast over query and heads.
            let expanded = keyPaddingMask.expandedDimensions(axis: 1)
                .expandedDimensions(axis: 1)
            mask = MLX.where(
                expanded,
                MLXArray(Float(-1e9)).asType(q.dtype),
                MLXArray(Float(0)).asType(q.dtype)
            )
        }
        let attended = MLXFast.scaledDotProductAttention(
            queries: q,
            keys: k,
            values: v,
            scale: scale,
            mask: mask
        )
        return outputProjection(
            attended.transposed(0, 2, 1, 3)
                .reshaped([queries.dim(0), queries.dim(1), heads * headDimension])
        )
    }
}

final class MiniCPMResampler: Module {
    let configuration: MiniCPMVisionConfiguration

    @ParameterInfo(key: "query") var query: MLXArray
    @ParameterInfo(key: "proj") var projection: MLXArray
    @ModuleInfo(key: "kv_proj") var keyValueProjection: Linear
    @ModuleInfo(key: "attn") var attention: MiniCPMCrossAttention
    @ModuleInfo(key: "ln_q") var queryNorm: LayerNorm
    @ModuleInfo(key: "ln_kv") var keyValueNorm: LayerNorm
    @ModuleInfo(key: "ln_post") var postNorm: LayerNorm

    init(configuration: MiniCPMVisionConfiguration) {
        self.configuration = configuration
        _query.wrappedValue = MLXArray.zeros([configuration.queryNum, configuration.outputDim])
        _projection.wrappedValue = MLXArray.zeros([configuration.outputDim, configuration.outputDim])
        _keyValueProjection.wrappedValue = Linear(
            configuration.visionHiddenSize,
            configuration.outputDim,
            bias: false
        )
        _attention.wrappedValue = MiniCPMCrossAttention(
            dimensions: configuration.outputDim,
            heads: configuration.resamplerHeads
        )
        _queryNorm.wrappedValue = LayerNorm(
            dimensions: configuration.outputDim,
            eps: 1e-6
        )
        _keyValueNorm.wrappedValue = LayerNorm(
            dimensions: configuration.outputDim,
            eps: 1e-6
        )
        _postNorm.wrappedValue = LayerNorm(
            dimensions: configuration.outputDim,
            eps: 1e-6
        )
        super.init()
    }

    func callAsFunction(_ input: MLXArray, targetSizes: MLXArray) throws -> MLXArray {
        guard input.ndim == 3,
              targetSizes.ndim == 2,
              input.dim(0) == targetSizes.dim(0),
              targetSizes.dim(1) == 2
        else {
            throw MiniCPMVisionError.invalidInput(
                "resampler expects input [B,L,D] and target sizes [B,2]"
            )
        }
        let batch = input.dim(0)
        eval(targetSizes)
        let target = targetSizes.asType(.int32).asArray(Int32.self)
        var lengths = [Int](repeating: 0, count: batch)
        var maxLength = 0
        var positionHeight = configuration.maxGridHeight
        var positionWidth = configuration.maxGridWidth
        for b in 0..<batch {
            let height = Int(target[b * 2])
            let width = Int(target[b * 2 + 1])
            guard height > 0, width > 0 else {
                throw MiniCPMVisionError.invalidInput("target size must be positive")
            }
            let length = height * width
            guard length <= input.dim(1) else {
                throw MiniCPMVisionError.invalidInput(
                    "target grid \(height)x\(width) exceeds vision sequence \(input.dim(1))"
                )
            }
            lengths[b] = length
            maxLength = max(maxLength, length)
            // The reference Resampler grows its cached 2-D sinusoidal table
            // when a target grid exceeds the initial 70x70 window.  Keep the
            // bundle defaults as the minimum, but do not reject a valid
            // packed image merely because its aspect ratio needs a wider
            // positional cache.
            positionHeight = max(positionHeight, height)
            positionWidth = max(positionWidth, width)
        }

        var positionValues = [Float](repeating: 0, count: batch * maxLength * configuration.outputDim)
        var padding = [Bool](repeating: true, count: batch * maxLength)
        let positionTable = Self.makePositionTable(
            height: positionHeight,
            width: positionWidth,
            dimensions: configuration.outputDim
        )
        for b in 0..<batch {
            let height = Int(target[b * 2])
            let width = Int(target[b * 2 + 1])
            let length = lengths[b]
            for row in 0..<height {
                for column in 0..<width {
                    let source = (row * positionWidth + column)
                        * configuration.outputDim
                    let destination = (b * maxLength + row * width + column)
                        * configuration.outputDim
                    positionValues.replaceSubrange(
                        destination..<(destination + configuration.outputDim),
                        with: positionTable[source..<(source + configuration.outputDim)]
                    )
                    padding[b * maxLength + row * width + column] = false
                }
            }
            // Keep the compiler aware that lengths is intentionally checked
            // above even when maxLength is shared across a batch.
            _ = length
        }
        let position = MLXArray(
            positionValues,
            [batch, maxLength, configuration.outputDim]
        ).asType(input.dtype)
        let keyPadding = MLXArray(padding, [batch, maxLength])

        var values = keyValueProjection(input)
        values = keyValueNorm(values[0..., 0..<maxLength, 0...])
        var queries = queryNorm(query)
        queries = broadcast(queries.expandedDimensions(axis: 0), to: [batch, configuration.queryNum, configuration.outputDim])
        let keys = values + position
        let attended = attention(
            queries: queries,
            keys: keys,
            values: values,
            keyPaddingMask: keyPadding
        )
        return matmul(postNorm(attended), projection)
    }

    static func makePositionTable(
        height: Int,
        width: Int,
        dimensions: Int
    ) -> [Float] {
        let axis = dimensions / 2
        let half = axis / 2
        let omega = (0..<half).map {
            // The upstream helper calls get_1d_sincos_pos_embed with
            // embed_dim = dimensions / 2.  That helper creates
            // ``embed_dim / 2`` frequencies and divides the exponent by
            // ``embed_dim / 2`` as well.  In this 2-D layout that is
            // ``dimensions / 4`` (`half`), not the full per-axis width
            // (`axis`).  For D=8 the frequencies are [1, 0.01].
            pow(10_000.0, -Float($0) / Float(half))
        }
        var result = [Float](repeating: 0, count: height * width * dimensions)
        for row in 0..<height {
            for column in 0..<width {
                let base = (row * width + column) * dimensions
                for i in 0..<half {
                    result[base + i] = sin(Float(column) * omega[i])
                    result[base + half + i] = cos(Float(column) * omega[i])
                    result[base + axis + i] = sin(Float(row) * omega[i])
                    result[base + axis + half + i] = cos(Float(row) * omega[i])
                }
            }
        }
        return result
    }
}

// MARK: - Public model

/// Native MLX Swift implementation of MiniCPM-o 4.5's visual path.
public final class MiniCPMVisionModel: Module {
    public let configuration: MiniCPMVisionConfiguration

    @ModuleInfo var vision: MiniCPMSiglipTower
    @ModuleInfo var resampler: MiniCPMResampler

    public init(configuration: MiniCPMVisionConfiguration = .init()) {
        self.configuration = configuration
        _vision.wrappedValue = MiniCPMSiglipTower(configuration: configuration)
        _resampler.wrappedValue = MiniCPMResampler(configuration: configuration)
        super.init()
    }

    /// Encode already-packed tensors. Input can be `[C,P,N*P]` or batched
    /// `[B,C,P,N*P]`; output is `[B,queryNum,outputDim]`.
    public func encode(
        pixelValues: MLXArray,
        targetSizes: MLXArray
    ) throws -> MLXArray {
        let batched: MLXArray
        if pixelValues.ndim == 3 {
            batched = pixelValues.expandedDimensions(axis: 0)
        } else {
            batched = pixelValues
        }
        let sizes: MLXArray
        if targetSizes.ndim == 1 {
            sizes = targetSizes.expandedDimensions(axis: 0)
        } else {
            sizes = targetSizes
        }
        let computePixels = batched.asType(
            configuration.computePrecision == .float32 ? .float32 : .bfloat16
        )
        let hidden = try vision(computePixels, targetSizes: sizes)
        return try resampler(hidden, targetSizes: sizes)
    }

    /// Encode one or more preprocessed image groups while preserving source vs
    /// HD slice boundaries. Every group is evaluated independently so images
    /// with different target grids never require fake sequence padding.
    public func encode(_ groups: [MiniCPMImageGroup]) throws -> [MiniCPMVisionGroup] {
        var result: [MiniCPMVisionGroup] = []
        result.reserveCapacity(groups.count)
        for group in groups {
            let outputs = try group.all.map { packed in
                try encode(
                    pixelValues: packed.pixels,
                    targetSizes: packed.targetSize.array
                )[0]
            }
            guard let overview = outputs.first else {
                throw MiniCPMVisionError.invalidInput("image group has no overview")
            }
            result.append(MiniCPMVisionGroup(
                overview: overview,
                slices: Array(outputs.dropFirst())
            ))
        }
        return result
    }

    public func encode(_ batch: MiniCPMImageBatch) throws -> [MiniCPMVisionGroup] {
        try encode(batch.groups)
    }
}

/// A processor output batch. The processor itself lives in `Processor.swift`
/// so callers that only have packed arrays need no image framework imports.
public struct MiniCPMImageBatch {
    public let groups: [MiniCPMImageGroup]

    public init(groups: [MiniCPMImageGroup]) {
        self.groups = groups
    }
}
