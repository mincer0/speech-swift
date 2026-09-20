import Foundation
import MLX
import MLXFast
import MLXNN
import MLXCommon

/// MiniCPM keeps packed quantized projections on the established MLX path,
/// while dense BF16 projections use the exact MPSGraph implementation.  The
/// environment switch is intentionally diagnostic-only: production dense
/// models use MPSGraph unless MINICPM_MLX_BF16_LINEAR=1 is explicitly set.
private func makeMiniCPMLinear(
    _ inputDimensions: Int,
    _ outputDimensions: Int,
    bias: Bool,
    groupSize: Int,
    bits: Int
) -> Linear {
    if bits > 0 {
        return QuantizedLinear(
            inputDimensions,
            outputDimensions,
            bias: bias,
            groupSize: groupSize,
            bits: bits)
    }
    if ProcessInfo.processInfo.environment["MINICPM_MLX_BF16_LINEAR"] == "1" {
        return Linear(inputDimensions, outputDimensions, bias: bias)
    }
    return MiniCPMExactBF16Linear(inputDimensions, outputDimensions, bias: bias)
}

private func makeMiniCPMRMSNorm(
    _ dimensions: Int,
    eps: Float,
    bits: Int
) -> RMSNorm {
    if bits == 0 {
        return MiniCPMExactBF16RMSNorm(dimensions: dimensions, eps: eps)
    }
    return RMSNorm(dimensions: dimensions, eps: eps)
}

public struct MiniCPMForwardOutput {
    public let logits: MLXArray
    public let hidden: MLXArray
    public let state: MiniCPMInferenceState

    public init(logits: MLXArray, hidden: MLXArray, state: MiniCPMInferenceState) {
        self.logits = logits
        self.hidden = hidden
        self.state = state
    }
}

/// A value-semantic view over one layer's KV storage.
///
/// The old MiniCPM implementation represented a cache as a pair of arrays and
/// appended every block with `concatenated([old, new], axis: 2)`.  That is
/// quadratic for token-at-a-time decoding: at position `n`, every layer copies
/// all `n` previous keys and values before it can attend to the new token.
///
/// `MiniCPMKVCache` separates the *storage* from the logical view.  Storage is
/// a shared, geometrically grown buffer; a view only contains the live count and
/// absolute base position.  Appending into spare capacity writes after the
/// current logical end and returns another view, so taking a session snapshot
/// or rolling one back only moves those integer boundaries.  A reallocation is
/// done at geometric growth points, rather than once per token.  Importantly,
/// no write ever touches entries below the caller's current `count`; old views
/// therefore remain valid while a speculative unit is decoded.
public struct MiniCPMKVCache {
    fileprivate final class Storage {
        var keys: MLXArray
        var values: MLXArray
        var growthCount: Int

        init(keys: MLXArray, values: MLXArray, growthCount: Int = 0) {
            self.keys = keys
            self.values = values
            self.growthCount = growthCount
        }
    }

    fileprivate let storage: Storage
    /// Number of valid entries along the token axis.
    public let count: Int
    /// Absolute position represented by entry zero.  MiniCPM currently never
    /// trims a cache in place, but retaining the base makes RoPE bookkeeping
    /// correct if a bounded cache is added later.
    public let base: Int

    fileprivate init(storage: Storage, count: Int, base: Int) {
        self.storage = storage
        self.count = count
        self.base = base
    }

    /// Build a cache view around an existing (legacy) key/value pair.  This is
    /// used by the public state initializer so older callers can still create a
    /// state from arrays without paying a migration cost at the call site.
    public init(keys: MLXArray, values: MLXArray, base: Int = 0) {
        precondition(keys.ndim == 4 && values.ndim == 4,
                     "MiniCPM KV arrays must have shape [B,H,T,D]")
        precondition(keys.shape == values.shape,
                     "MiniCPM key/value cache shapes must match")
        self.storage = Storage(keys: keys, values: values)
        self.count = keys.dim(2)
        self.base = base
    }

    /// Create an empty cache with a modest decode runway.  Keeping a real
    /// (non-zero) token axis avoids backend-specific issues with zero-shaped
    /// MLX arrays while still making the first append allocation cheap.
    fileprivate init(
        batch: Int,
        heads: Int,
        headDim: Int,
        dtype: DType,
        base: Int = 0
    ) {
        let initialCapacity = 512
        self.storage = Storage(
            keys: MLXArray.zeros([batch, heads, initialCapacity, headDim], dtype: dtype),
            values: MLXArray.zeros([batch, heads, initialCapacity, headDim], dtype: dtype))
        self.count = 0
        self.base = base
    }

    /// Number of geometric reallocations performed by this cache's storage.
    /// Exposed for long-context regression tests and runtime diagnostics.
    public var growthCount: Int { storage.growthCount }

    /// Allocated token capacity.  Only `0 ..< count` is valid for attention.
    public var capacity: Int { storage.keys.dim(2) }

    /// The valid key/value view consumed by SDPA.  Slicing excludes spare
    /// capacity and is a view operation, not a copy.
    public func window() -> (MLXArray, MLXArray) {
        guard count < capacity else { return (storage.keys, storage.values) }
        return (
            storage.keys[0..., 0..., 0 ..< count, 0...],
            storage.values[0..., 0..., 0 ..< count, 0...])
    }

    /// Copy a logical range into independent storage.  A plain MLX slice is
    /// only a view; retaining it while the source cache is later appended can
    /// therefore expose writes through a session snapshot.  Context-window
    /// rebuilds use this helper for both the fixed prefix and the retained
    /// unit suffix before doing any forward pass or RoPE surgery.
    fileprivate func copying(
        start: Int,
        end: Int,
        base: Int? = nil
    ) -> MiniCPMKVCache? {
        let lower = max(0, min(start, count))
        let upper = max(lower, min(end, count))
        let length = upper - lower
        let newBase = base ?? self.base + lower
        if length == 0 {
            return MiniCPMKVCache(
                batch: storage.keys.dim(0),
                heads: storage.keys.dim(1),
                headDim: storage.keys.dim(3),
                dtype: storage.keys.dtype,
                base: newBase)
        }

        let keys = MLXArray.zeros(
            [storage.keys.dim(0), storage.keys.dim(1), length, storage.keys.dim(3)],
            dtype: storage.keys.dtype)
        let values = MLXArray.zeros(
            [storage.values.dim(0), storage.values.dim(1), length, storage.values.dim(3)],
            dtype: storage.values.dtype)
        keys[0..., 0..., 0 ..< length, 0...] =
            storage.keys[0..., 0..., lower ..< upper, 0...]
        values[0..., 0..., 0 ..< length, 0...] =
            storage.values[0..., 0..., lower ..< upper, 0...]
        return MiniCPMKVCache(
            storage: Storage(keys: keys, values: values),
            count: length,
            base: newBase)
    }

    /// Append post-RoPE keys and values and return the new logical view.
    ///
    /// Existing storage is reused while there is room.  At capacity, a new
    /// buffer is allocated at a geometric size and the live prefix is copied
    /// once.  The old storage remains untouched, which is what makes snapshots
    /// safe even though normal appends use in-place slice assignment.
    public func appending(keys newKeys: MLXArray, values newValues: MLXArray) -> MiniCPMKVCache {
        precondition(newKeys.ndim == 4 && newValues.ndim == 4,
                     "MiniCPM KV arrays must have shape [B,H,T,D]")
        precondition(newKeys.shape == newValues.shape,
                     "MiniCPM key/value cache shapes must match")
        let tokenCount = newKeys.dim(2)
        guard tokenCount > 0 else { return self }

        // The first allocation leaves a small decode runway after a prefill;
        // subsequent allocations double capacity.  This avoids the otherwise
        // inevitable copy on the first one-token decode after a large prompt.
        if count + tokenCount <= capacity {
            storage.keys[0..., 0..., count ..< (count + tokenCount), 0...] = newKeys
            storage.values[0..., 0..., count ..< (count + tokenCount), 0...] = newValues
            return MiniCPMKVCache(storage: storage, count: count + tokenCount, base: base)
        }

        let required = count + tokenCount
        let spare = 512
        let nextCapacity: Int
        if capacity == 0 {
            nextCapacity = max(required, required + spare)
        } else {
            nextCapacity = max(required, max(capacity * 2, capacity + spare))
        }
        let grownKeys = MLXArray.zeros(
            [newKeys.dim(0), newKeys.dim(1), nextCapacity, newKeys.dim(3)],
            dtype: newKeys.dtype)
        let grownValues = MLXArray.zeros(
            [newValues.dim(0), newValues.dim(1), nextCapacity, newValues.dim(3)],
            dtype: newValues.dtype)
        if count > 0 {
            grownKeys[0..., 0..., 0 ..< count, 0...] =
                storage.keys[0..., 0..., 0 ..< count, 0...]
            grownValues[0..., 0..., 0 ..< count, 0...] =
                storage.values[0..., 0..., 0 ..< count, 0...]
        }
        grownKeys[0..., 0..., count ..< required, 0...] = newKeys
        grownValues[0..., 0..., count ..< required, 0...] = newValues
        let grownStorage = Storage(
            keys: grownKeys,
            values: grownValues,
            growthCount: storage.growthCount + 1)
        return MiniCPMKVCache(storage: grownStorage, count: required, base: base)
    }

    /// Remove a contiguous token range while preserving the storage prefix and
    /// suffix.  Unlike replaying the retained embeddings, this surgery keeps
    /// every value vector (and therefore every retained hidden-history value)
    /// bit-for-bit unchanged.  Keys in the suffix are the only tensors that
    /// need work: their RoPE coordinates are shifted left by the removed
    /// length before the compacted cache is exposed.
    ///
    /// The returned cache owns fresh backing arrays.  This is intentional:
    /// callers commonly hold a speculative snapshot of the old state, and an
    /// in-place write would make that snapshot observe a later window event.
    func dropping(
        preserve: Int,
        length: Int,
        ropeTheta: Float
    ) -> MiniCPMKVCache? {
        guard length > 0, count > 0 else { return nil }
        let prefixLength = max(0, min(preserve, count))
        let available = count - prefixLength
        guard length <= available else { return nil }

        let suffixLength = count - prefixLength - length
        let keepLength = count - length
        let newKeys = MLXArray.zeros(storage.keys.shape, dtype: storage.keys.dtype)
        let newValues = MLXArray.zeros(storage.values.shape, dtype: storage.values.dtype)

        if prefixLength > 0 {
            newKeys[0..., 0..., 0 ..< prefixLength, 0...] =
                storage.keys[0..., 0..., 0 ..< prefixLength, 0...]
            newValues[0..., 0..., 0 ..< prefixLength, 0...] =
                storage.values[0..., 0..., 0 ..< prefixLength, 0...]
        }

        if suffixLength > 0 {
            let suffixKeys = storage.keys[
                0..., 0..., (prefixLength + length) ..< count, 0...]
            let suffixValues = storage.values[
                0..., 0..., (prefixLength + length) ..< count, 0...]
            let reindexedKeys = MiniCPMCacheRoPE.realign(
                suffixKeys,
                oldStart: base + prefixLength + length,
                newStart: base + prefixLength,
                ropeTheta: ropeTheta)
            newKeys[0..., 0..., prefixLength ..< keepLength, 0...] = reindexedKeys
            newValues[0..., 0..., prefixLength ..< keepLength, 0...] = suffixValues
        }

        return MiniCPMKVCache(
            storage: Storage(
                keys: newKeys,
                values: newValues,
                growthCount: storage.growthCount),
            count: keepLength,
            base: base)
    }
}

/// Value-semantic per-session inference state.  `kvCaches` remains a legacy
/// array-of-tuples view for source compatibility and diagnostics.  The model
/// itself uses `cacheViews`, whose logical boundaries can be advanced without
/// concatenating the full history on every token.
public struct MiniCPMInferenceState {
    public var kvCaches: [(MLXArray, MLXArray)?]
    public var position: Int
    fileprivate var cacheViews: [MiniCPMKVCache?]

    public init(kvCaches: [(MLXArray, MLXArray)?], position: Int) {
        self.kvCaches = kvCaches
        self.position = position
        self.cacheViews = kvCaches.map { pair in
            guard let pair else { return nil }
            return MiniCPMKVCache(keys: pair.0, values: pair.1)
        }
    }

    fileprivate init(cacheViews: [MiniCPMKVCache?], position: Int) {
        self.cacheViews = cacheViews
        self.position = position
        self.kvCaches = cacheViews.map { $0?.window() }
    }

    public static func initial(config: MiniCPMMLXConfig) -> MiniCPMInferenceState {
        MiniCPMInferenceState(
            kvCaches: [(MLXArray, MLXArray)?](repeating: nil, count: config.numHiddenLayers),
            position: 0)
    }

    public var sequenceLength: Int {
        cacheViews.first.flatMap { $0?.count }
            ?? kvCaches.first.flatMap { $0?.0.dim(2) }
            ?? position
    }

    /// Total number of geometric reallocations across layers.  A stable value
    /// here is useful for catching accidental reintroduction of per-token
    /// concatenation in long-context tests.
    public var cacheGrowthCount: Int {
        cacheViews.reduce(0) { $0 + ($1?.growthCount ?? 0) }
    }

    /// Allocated KV token capacity by layer (`0` for an empty layer).
    public var cacheCapacities: [Int] {
        cacheViews.map { $0?.capacity ?? 0 }
    }
}

/// Qwen3's official rotary operation for the dense BF16 bundle.
///
/// The PyTorch module materializes ``inv_freq`` as BF16, promotes it to FP32
/// for the position matmul, computes trigonometric values in FP32, then casts
/// cos/sin to the input dtype before applying split-half rotation.  Keeping
/// this sequence in one helper lets the production attention path and the
/// tiny parity regression exercise exactly the same operation.
enum MiniCPMExactBF16RoPE {
    static func inverseFrequency(headDim: Int, ropeTheta: Float) -> MLXArray {
        let half = headDim / 2
        return MLXArray((0..<half).map {
            Float(pow(Double(ropeTheta), -2 * Double($0) / Double(headDim)))
        }, [1, half]).asType(.bfloat16)
    }

    static func apply(
        _ input: MLXArray,
        inverseFrequency: MLXArray,
        offset: Int
    ) -> MLXArray {
        let headDim = input.dim(-1)
        let half = headDim / 2
        let length = input.dim(2)
        let positions = MLXArray((0..<length).map { Float(offset + $0) }, [length, 1])
        let freqs = matmul(positions, inverseFrequency.asType(.float32))
        let cosBase = concatenated([cos(freqs), cos(freqs)], axis: -1).asType(input.dtype)
        let sinBase = concatenated([sin(freqs), sin(freqs)], axis: -1).asType(input.dtype)
        let cosBroadcast = cosBase.expandedDimensions(axis: 0).expandedDimensions(axis: 0)
        let sinBroadcast = sinBase.expandedDimensions(axis: 0).expandedDimensions(axis: 0)
        let first = input[0..., 0..., 0..., 0 ..< half]
        let second = input[0..., 0..., 0..., half ..< headDim]
        let rotateHalf = concatenated([-second, first], axis: -1)
        return input * cosBroadcast + rotateHalf * sinBroadcast
    }
}

/// Float32 RoPE surgery used when a basic sliding window removes a middle
/// segment.  The pinned PyTorch utility computes inverse frequencies directly
/// in float32 for reindexing (unlike the dense forward path, which stores the
/// model's ``inv_freq`` as BF16), computes old/new cosines and sines in
/// float32, then casts those trigonometric tensors back to the key dtype.
enum MiniCPMCacheRoPE {
    static func realign(
        _ keys: MLXArray,
        oldStart: Int,
        newStart: Int,
        ropeTheta: Float
    ) -> MLXArray {
        let length = keys.dim(2)
        guard length > 0 else { return keys }

        let headDim = keys.dim(3)
        let half = headDim / 2
        let inverseFrequency = MLXArray((0 ..< half).map { index in
            Float(pow(Double(ropeTheta), -Double(index) / Double(half)))
        }, [1, half])

        func cosSin(start: Int) -> (MLXArray, MLXArray) {
            let positions = MLXArray(
                (0 ..< length).map { Float(start + $0) }, [length, 1])
            let angles = matmul(positions, inverseFrequency)
            let cosines = concatenated([cos(angles), cos(angles)], axis: -1)
                .asType(keys.dtype)
                .expandedDimensions(axis: 0)
                .expandedDimensions(axis: 0)
            let sines = concatenated([sin(angles), sin(angles)], axis: -1)
                .asType(keys.dtype)
                .expandedDimensions(axis: 0)
                .expandedDimensions(axis: 0)
            return (cosines, sines)
        }

        func rotateHalf(_ input: MLXArray) -> MLXArray {
            let first = input[0..., 0..., 0..., 0 ..< half]
            let second = input[0..., 0..., 0..., half ..< headDim]
            return concatenated([-second, first], axis: -1)
        }

        let (oldCos, oldSin) = cosSin(start: oldStart)
        let unrotated = oldCos * keys - oldSin * rotateHalf(keys)
        let (newCos, newSin) = cosSin(start: newStart)
        return newCos * unrotated + newSin * rotateHalf(unrotated)
    }
}

final class MiniCPMAttention: Module {
    let numQHeads: Int
    let numKVHeads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm
    let rope: RoPE
    /// The official Qwen3 PyTorch module stores inverse frequencies in BF16,
    /// promotes them to FP32 for the position matmul, and casts cos/sin back
    /// to the input dtype.  MLX's fused RoPE computes frequencies directly
    /// from FP32 values, which is observably different for this BF16 bundle.
    /// Dense BF16 uses the exact path by default; MINICPM_FUSED_ROPE=1 is a
    /// diagnostic opt-out. Quantized bundles retain the fused implementation.
    private let useExactBF16RoPE: Bool
    private let exactBF16InverseFrequency: MLXArray
    /// Dense BF16 multi-token attention uses the exact PyTorch/MPSGraph SDPA
    /// sequence by default, including calls with a cached prefix. Cached
    /// one-token decode remains on MLX's fused path, whose native mode has
    /// the backend semantics required for the incremental cache. Packed
    /// bundles also remain on the fused path; MINICPM_MLX_BF16_SDPA=1 is a
    /// diagnostic opt-out.
    private let useExactBF16SDPA: Bool

    // Internal diagnostic visibility for the model-independent routing tests;
    // the call-site still applies cache/length/dtype guards on every request.
    var exactBF16SDPAEnabled: Bool { useExactBF16SDPA }

    /// Dense BF16 multi-token calls use the exact PyTorch/MPSGraph SDPA
    /// sequence even when a cached prefix is present.  A cached one-token
    /// decode remains on MLX's fused path; packed/quantized and explicit
    /// fused-opt-out configurations likewise stay fused.
    func shouldUseExactBF16SDPA(
        cache _: MiniCPMKVCache?,
        length: Int,
        dtype: DType
    ) -> Bool {
        useExactBF16SDPA && length > 1 && dtype == .bfloat16
    }

    init(_ config: MiniCPMMLXConfig) {
        numQHeads = config.numAttentionHeads
        numKVHeads = config.numKeyValueHeads
        headDim = config.headDim
        scale = 1 / sqrt(Float(config.headDim))
        let qDim = config.numAttentionHeads * config.headDim
        let kvDim = config.numKeyValueHeads * config.headDim
        _qProj = ModuleInfo(wrappedValue: makeMiniCPMLinear(
            config.hiddenSize, qDim, bias: false,
            groupSize: config.quantGroupSize, bits: config.quantBits))
        _kProj = ModuleInfo(wrappedValue: makeMiniCPMLinear(
            config.hiddenSize, kvDim, bias: false,
            groupSize: config.quantGroupSize, bits: config.quantBits))
        _vProj = ModuleInfo(wrappedValue: makeMiniCPMLinear(
            config.hiddenSize, kvDim, bias: false,
            groupSize: config.quantGroupSize, bits: config.quantBits))
        _oProj = ModuleInfo(wrappedValue: makeMiniCPMLinear(
            qDim, config.hiddenSize, bias: false,
            groupSize: config.quantGroupSize, bits: config.quantBits))
        _qNorm = ModuleInfo(wrappedValue: makeMiniCPMRMSNorm(
            config.headDim, eps: config.rmsNormEps, bits: config.quantBits))
        _kNorm = ModuleInfo(wrappedValue: makeMiniCPMRMSNorm(
            config.headDim, eps: config.rmsNormEps, bits: config.quantBits))
        rope = RoPE(
            dimensions: config.headDim,
            traditional: false,
            base: config.ropeTheta)
        useExactBF16RoPE = config.quantBits == 0
            && ProcessInfo.processInfo.environment["MINICPM_FUSED_ROPE"] != "1"
        exactBF16InverseFrequency = MiniCPMExactBF16RoPE.inverseFrequency(
            headDim: config.headDim, ropeTheta: config.ropeTheta)
        useExactBF16SDPA = config.quantBits == 0
            && ProcessInfo.processInfo.environment["MINICPM_MLX_BF16_SDPA"] != "1"
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        cache: MiniCPMKVCache?,
        offset: Int
    ) -> (MLXArray, MiniCPMKVCache) {
        let batch = x.dim(0)
        let length = x.dim(1)
        var q = qNorm(qProj(x).reshaped(batch, length, numQHeads, headDim))
        var k = kNorm(kProj(x).reshaped(batch, length, numKVHeads, headDim))
        var v = vProj(x).reshaped(batch, length, numKVHeads, headDim)
        q = q.transposed(0, 2, 1, 3)
        k = k.transposed(0, 2, 1, 3)
        v = v.transposed(0, 2, 1, 3)

        // `base + count` is equivalent to the old tuple length while the
        // cache starts at zero.  Keeping the absolute base here means a future
        // bounded/trimmed cache does not accidentally reset RoPE positions.
        let ropeOffset = cache.map { $0.base + $0.count } ?? offset
        if useExactBF16RoPE {
            q = MiniCPMExactBF16RoPE.apply(
                q, inverseFrequency: exactBF16InverseFrequency, offset: ropeOffset)
            k = MiniCPMExactBF16RoPE.apply(
                k, inverseFrequency: exactBF16InverseFrequency, offset: ropeOffset)
        } else {
            q = rope(q, offset: ropeOffset)
            k = rope(k, offset: ropeOffset)
        }

        let nextCache: MiniCPMKVCache
        if let cache {
            nextCache = cache.appending(keys: k, values: v)
        } else {
            nextCache = MiniCPMKVCache(
                batch: k.dim(0), heads: k.dim(1), headDim: k.dim(3),
                dtype: k.dtype, base: offset)
                .appending(keys: k, values: v)
        }
        let (keys, values) = nextCache.window()

        // PyTorch's dense BF16 path uses a specific MPSGraph sequence whose
        // accumulation order differs from MLX's fused kernel.  Context-window
        // rebuilds feed multiple tokens against a copied prefix cache, so the
        // exact route must cover cached multi-token calls as well.  Cached
        // one-token decode retains the fused route, whose native single-token
        // mode handles the full prefix without an explicit mask.  Float32
        // fixtures and packed quantized bundles likewise remain fused.
        let canUseExactBF16SDPA = shouldUseExactBF16SDPA(
            cache: cache, length: length, dtype: q.dtype)
            && k.dtype == .bfloat16
            && v.dtype == .bfloat16
        let output: MLXArray
        if canUseExactBF16SDPA {
            // PyTorch's cached multi-token sdpa_general_mps path receives a
            // full additive mask for the rectangular Q/K geometry.  A
            // right-aligned mask lets the first new query see the cached
            // prefix plus itself while preserving causal order across the
            // appended segment.  Square prefill remains on the helper's
            // original causal-select graph.
            let additiveMask: MLXArray? = keys.dim(2) > length
                ? MiniCPMExactBF16SDPA.rightAlignedAdditiveMask(
                    batch: batch,
                    queryHeads: numQHeads,
                    queryTokens: length,
                    keyTokens: keys.dim(2))
                : nil
            let exactHeads = MiniCPMExactBF16SDPA.attend(
                query: q,
                key: keys,
                value: values,
                scale: scale,
                additiveMask: additiveMask)
            output = exactHeads.reshaped(batch, length, numQHeads * headDim)
        } else {
            // A single-token decode can attend the entire prefix without a
            // mask.  The fused path handles that case and all quantized calls,
            // preserving GQA behavior and existing cache semantics.
            let mask: MLXFast.ScaledDotProductAttentionMaskMode = length <= 1
                ? .none
                : .causal
            output = SDPA.attendAndMerge(
                qHeads: q,
                kHeads: keys,
                vHeads: values,
                scale: scale,
                mask: mask)
        }
        return (oProj(output), nextCache)
    }

}

final class MiniCPMMLP: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear
    private let useExactBF16SiLU: Bool

    init(_ config: MiniCPMMLXConfig) {
        useExactBF16SiLU = config.quantBits == 0
        _gateProj = ModuleInfo(wrappedValue: makeMiniCPMLinear(
            config.hiddenSize, config.intermediateSize, bias: false,
            groupSize: config.quantGroupSize, bits: config.quantBits))
        _upProj = ModuleInfo(wrappedValue: makeMiniCPMLinear(
            config.hiddenSize, config.intermediateSize, bias: false,
            groupSize: config.quantGroupSize, bits: config.quantBits))
        _downProj = ModuleInfo(wrappedValue: makeMiniCPMLinear(
            config.intermediateSize, config.hiddenSize, bias: false,
            groupSize: config.quantGroupSize, bits: config.quantBits))
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let gate = gateProj(x)
        // The dense BF16 checkpoint was exported from PyTorch, whose SiLU
        // path computes sigmoid/product in FP32 before storing BF16.  MLX's
        // fused BF16 silu uses a different precision order.  Keep the exact
        // helper limited to dense bundles; packed 8-bit layers retain the
        // established fused MLX path.
        let activatedGate = useExactBF16SiLU ? MiniCPMExactBF16SiLU.apply(gate) : silu(gate)
        return downProj(activatedGate * upProj(x))
    }
}

final class MiniCPMTransformerLayer: Module {
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm
    @ModuleInfo(key: "self_attn") var selfAttention: MiniCPMAttention
    @ModuleInfo var mlp: MiniCPMMLP

    init(_ config: MiniCPMMLXConfig) {
        _inputLayerNorm = ModuleInfo(wrappedValue: makeMiniCPMRMSNorm(
            config.hiddenSize, eps: config.rmsNormEps, bits: config.quantBits))
        _postAttentionLayerNorm = ModuleInfo(wrappedValue: makeMiniCPMRMSNorm(
            config.hiddenSize, eps: config.rmsNormEps, bits: config.quantBits))
        _selfAttention = ModuleInfo(wrappedValue: MiniCPMAttention(config))
        _mlp = ModuleInfo(wrappedValue: MiniCPMMLP(config))
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        cache: MiniCPMKVCache?,
        offset: Int
    ) -> (MLXArray, MiniCPMKVCache) {
        let (attention, newCache) = selfAttention(
            inputLayerNorm(x), cache: cache, offset: offset)
        let residual = x + attention
        return (residual + mlp(postAttentionLayerNorm(residual)), newCache)
    }
}

/// Embedding layer that selects the representation matching the bundle
/// precision.  ``PreQuantizedEmbedding`` is a custom MLX module rather than
/// an ``Embedding`` subclass, so keeping both behind one stable wrapper lets
/// the rest of the language model expose the same API for dense and packed
/// bundles.
final class MiniCPMEmbedding: Module {
    @ModuleInfo(key: "quantized") var quantized: PreQuantizedEmbedding?
    @ModuleInfo(key: "dense") var dense: Embedding?

    init(_ config: MiniCPMMLXConfig) {
        if config.isQuantized {
            _quantized = ModuleInfo(wrappedValue: PreQuantizedEmbedding(
                embeddingCount: config.vocabSize,
                dimensions: config.hiddenSize,
                groupSize: config.quantGroupSize,
                bits: config.quantBits))
            _dense = ModuleInfo(wrappedValue: nil)
        } else {
            _quantized = ModuleInfo(wrappedValue: nil)
            _dense = ModuleInfo(wrappedValue: Embedding(
                embeddingCount: config.vocabSize,
                dimensions: config.hiddenSize))
        }
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        if let quantized { return quantized(x) }
        guard let dense else { fatalError("MiniCPM embedding is not initialized") }
        return dense(x)
    }

    func asLinear(_ x: MLXArray) -> MLXArray {
        if let quantized { return quantized.asLinear(x) }
        guard let dense else { fatalError("MiniCPM embedding is not initialized") }
        return dense.asLinear(x)
    }
}

/// Pure MLX Qwen3 language backbone used by the MiniCPM duplex coordinator.
/// It deliberately has no audio encoder, vision tower or TTS module; those
/// components can feed projected embeddings into ``forward(inputEmbeddings:)``.
public final class MiniCPMLanguageModel: Module {
    public let config: MiniCPMMLXConfig

    @ModuleInfo(key: "embed_tokens") var embedTokens: MiniCPMEmbedding
    @ModuleInfo var layers: [MiniCPMTransformerLayer]
    @ModuleInfo var norm: RMSNorm
    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public init(config: MiniCPMMLXConfig) {
        self.config = config
        _embedTokens = ModuleInfo(wrappedValue: MiniCPMEmbedding(config))
        _layers = ModuleInfo(wrappedValue: (0..<config.numHiddenLayers).map { _ in
            MiniCPMTransformerLayer(config)
        })
        _norm = ModuleInfo(wrappedValue: makeMiniCPMRMSNorm(
            config.hiddenSize, eps: config.rmsNormEps, bits: config.quantBits))
        _lmHead = ModuleInfo(wrappedValue: config.tieWordEmbeddings ? nil : makeMiniCPMLinear(
            config.hiddenSize,
            config.vocabSize,
            bias: false,
            groupSize: config.quantGroupSize,
            bits: config.quantBits))
        super.init()
    }

    public func initialState() -> MiniCPMInferenceState {
        MiniCPMInferenceState.initial(config: config)
    }

    /// Return a state view with a suffix removed from every transformer KV
    /// layer.  This is the low-level counterpart of the Python MLX
    /// ``KVCache.trim`` path and is intentionally value-semantic: the old
    /// state keeps its logical count for speculative snapshots while the new
    /// state reuses the same geometrically-grown storage.
    ///
    /// Session callers should normally use ``MiniCPMSession.rollbackUnit`` so
    /// unit metadata is trimmed together with the cache.  The explicit state
    /// helper exists for parity fixtures and cache diagnostics.
    public func trimmedState(
        _ state: MiniCPMInferenceState,
        to targetLength: Int
    ) -> MiniCPMInferenceState {
        let current = state.sequenceLength
        let target = max(0, min(Int(targetLength), current))
        guard target < current else { return state }
        let views = state.cacheViews.map { cache -> MiniCPMKVCache? in
            guard let cache else { return nil }
            return MiniCPMKVCache(
                storage: cache.storage,
                count: min(target, cache.count),
                base: cache.base)
        }
        return MiniCPMInferenceState(
            cacheViews: views,
            position: min(target, state.position))
    }

    /// Drop a basic-window segment from the middle of every layer's cache.
    /// Prefix and suffix values are copied unchanged; suffix keys are
    /// re-encoded from their old RoPE positions to the compacted positions.
    /// The old state and its backing storage remain untouched so snapshots are
    /// safe to restore after a speculative window operation.
    public func droppingState(
        _ state: MiniCPMInferenceState,
        preserveLength: Int,
        droppingLength: Int
    ) -> MiniCPMInferenceState? {
        guard droppingLength > 0 else { return nil }
        guard !state.cacheViews.isEmpty,
              state.cacheViews.allSatisfy({ $0 != nil }) else {
            return nil
        }
        let current = state.sequenceLength
        guard current > 0,
              preserveLength >= 0,
              preserveLength + droppingLength <= current else {
            return nil
        }

        var views: [MiniCPMKVCache?] = []
        views.reserveCapacity(state.cacheViews.count)
        for cache in state.cacheViews {
            guard let cache else {
                views.append(nil)
                continue
            }
            guard cache.count == current,
                  let compacted = cache.dropping(
                    preserve: preserveLength,
                    length: droppingLength,
                    ropeTheta: config.ropeTheta) else {
                return nil
            }
            views.append(compacted)
        }

        return MiniCPMInferenceState(
            cacheViews: views,
            position: current - droppingLength)
    }

    /// Rebuild the context-sliding layout without replaying the retained
    /// units.  The pinned MiniCPM utility transforms
    /// ``[prefix][oldPrevious][suffix][oldUnits]`` into
    /// ``[prefix][newPrevious][suffix][remainingUnits]``.  Only the new
    /// previous/suffix segment is forwarded; the remaining unit cache is
    /// copied from the old tail and its keys are RoPE-reindexed to the new
    /// physical position.  Every copied range owns fresh storage so a caller
    /// holding the old state (or a session snapshot) observes no writes from
    /// the rebuild.
    public func rebuildingContextState(
        _ state: MiniCPMInferenceState,
        prefixLength: Int,
        segmentEmbeddings: MLXArray?,
        retainedLength: Int
    ) -> MiniCPMInferenceState? {
        let current = state.sequenceLength
        let prefix = max(0, prefixLength)
        let retained = max(0, retainedLength)
        guard prefix <= current,
              retained <= current - prefix,
              state.cacheViews.count == layers.count else {
            return nil
        }

        let oldCaches = state.cacheViews
        if current > 0 {
            guard oldCaches.count == layers.count,
                  oldCaches.allSatisfy({ $0?.count == current }) else {
                return nil
            }
        }

        // Copy the old tail before forwarding anything.  The prefix copy is
        // also independent: ``MiniCPMKVCache.appending`` writes into spare
        // capacity and must never overwrite the source state's suffix.
        let retainedStart = current - retained
        var retainedCaches: [MiniCPMKVCache?] = []
        retainedCaches.reserveCapacity(oldCaches.count)
        for cache in oldCaches {
            guard let cache else {
                retainedCaches.append(nil)
                continue
            }
            guard let tail = cache.copying(
                start: retainedStart,
                end: current,
                base: retainedStart) else {
                return nil
            }
            retainedCaches.append(tail)
        }

        // Use an independent prefix cache as the only past state for the
        // forward.  With an empty old cache there is no prefix to copy; the
        // normal uncached path still constructs all layer caches correctly.
        var prefixCaches: [MiniCPMKVCache?] = []
        prefixCaches.reserveCapacity(oldCaches.count)
        for cache in oldCaches {
            guard let cache else {
                prefixCaches.append(nil)
                continue
            }
            guard let prefixCache = cache.copying(
                start: 0,
                end: prefix,
                base: 0) else {
                return nil
            }
            prefixCaches.append(prefixCache)
        }

        var rebuilt = MiniCPMInferenceState(
            cacheViews: prefixCaches,
            position: prefix)
        if let segmentEmbeddings {
            let segmentLength = segmentEmbeddings.ndim >= 2
                ? segmentEmbeddings.dim(segmentEmbeddings.ndim - 2)
                : 0
            if segmentLength > 0 {
                rebuilt = forward(
                    inputEmbeddings: segmentEmbeddings,
                    state: rebuilt).state
            }
        }

        let newSystemLength = rebuilt.sequenceLength
        guard newSystemLength >= prefix else { return nil }

        var finalCaches: [MiniCPMKVCache?] = []
        finalCaches.reserveCapacity(oldCaches.count)
        for index in oldCaches.indices {
            guard let head = rebuilt.cacheViews[index] else {
                // This is only valid for a genuinely empty uncached state;
                // a retained tail cannot be attached without a layer cache.
                guard retainedCaches[index] == nil || retained == 0 else {
                    return nil
                }
                finalCaches.append(nil)
                continue
            }

            var final = head
            if retained > 0 {
                guard let tail = retainedCaches[index] else { return nil }
                let (_, tailValues) = tail.window()
                let (tailKeys, _) = tail.window()
                let reindexedKeys = MiniCPMCacheRoPE.realign(
                    tailKeys,
                    oldStart: retainedStart,
                    newStart: newSystemLength,
                    ropeTheta: config.ropeTheta)
                final = final.appending(keys: reindexedKeys, values: tailValues)
            }
            finalCaches.append(final)
        }

        let newLength = newSystemLength + retained
        return MiniCPMInferenceState(
            cacheViews: finalCaches,
            position: newLength)
    }

    public func embed(_ tokenIds: MLXArray) -> MLXArray {
        embedTokens(tokenIds)
    }

    public func forward(
        inputIds: MLXArray,
        state: MiniCPMInferenceState
    ) -> MiniCPMForwardOutput {
        // mlx_lm's duplex feed path casts both token and externally supplied
        // embeddings to BF16 before entering Qwen3.  Keeping that cast on the
        // token-id convenience overload avoids a subtle parity split where
        // `embed()` alone returns the dequantized FP32 values.
        forward(inputEmbeddings: embed(inputIds).asType(.bfloat16), state: state)
    }

    /// Feed precomputed audio/vision/text embeddings without fabricating token
    /// IDs.  Accepts `[T,H]` or `[B,T,H]` and always returns `[B,T,V]` logits.
    public func forward(
        inputEmbeddings: MLXArray,
        state: MiniCPMInferenceState
    ) -> MiniCPMForwardOutput {
        // ``MLXStreamDecoder.feed`` normalizes every input (including token
        // embeddings returned by ``embed_tokens``) to BF16 before invoking
        // Qwen3.  Keep the Swift embedding path identical: callers can pass
        // FP32 audio/projector output, but the backbone must see the same
        // BF16 values as the Python reference.  Without this cast the
        // quantized linear kernels consume FP32 on a different Metal path and
        // the resulting logits drift by ~1e-3 even though layer weights are
        // identical.
        let inputEmbeddings = inputEmbeddings.asType(.bfloat16)
        let embeddings: MLXArray
        if inputEmbeddings.ndim == 2 {
            embeddings = inputEmbeddings.expandedDimensions(axis: 0)
        } else if inputEmbeddings.ndim == 3 {
            embeddings = inputEmbeddings
        } else {
            fatalError("MiniCPM embeddings must have shape [T,H] or [B,T,H]")
        }
        guard embeddings.dim(2) == config.hiddenSize else {
            fatalError("MiniCPM embedding hidden size does not match config")
        }
        let length = embeddings.dim(1)
        var hidden = embeddings
        var cacheViews = state.cacheViews
        for (index, layer) in layers.enumerated() {
            let (nextHidden, nextCache) = layer(
                hidden,
                cache: cacheViews[index],
                offset: state.position)
            hidden = nextHidden
            cacheViews[index] = nextCache
        }
        hidden = norm(hidden)
        let logits = lmHead?(hidden) ?? embedTokens.asLinear(hidden)
        let nextState = MiniCPMInferenceState(
            cacheViews: cacheViews,
            position: state.position + length)
        return MiniCPMForwardOutput(logits: logits, hidden: hidden, state: nextState)
    }
}
