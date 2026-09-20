import Foundation
import MLX
import MLXFast
import MLXNN
import MLXRandom
import MiniCPMLLM

/// KV state for one MiniCPM audio-token decoder layer.
public final class MiniCPMTTSLayerCache {
    /// Backing storage is geometrically grown and intentionally contains
    /// spare entries.  Attention must use ``keys``/``values`` below, which
    /// expose only the logical prefix.  Keeping the capacity out of the
    /// attention key axis is important: the causal mask is built from the
    /// logical length, not from the allocation size.
    private var storedKeys: MLXArray?
    private var storedValues: MLXArray?

    /// Logical number of cached positions.  This is separate from backing
    /// capacity so one-token decode does not rebuild the whole history.
    public private(set) var count: Int = 0

    /// Number of times the backing allocation had to grow.  Exposed for
    /// diagnostics and regression tests; normal inference does not inspect it.
    public private(set) var capacityGrowthCount: Int = 0

    /// Current backing capacity along the sequence axis.
    public var capacity: Int { storedKeys?.dim(2) ?? 0 }

    /// Logical K/V views consumed by attention.  They never expose the spare
    /// capacity region, which contains uninitialised/zero padding.
    public var keys: MLXArray? {
        guard let storedKeys, count > 0 else { return nil }
        if count == storedKeys.dim(2) { return storedKeys }
        return storedKeys[0..., 0..., 0 ..< count, 0...]
    }

    public var values: MLXArray? {
        guard let storedValues, count > 0 else { return nil }
        if count == storedValues.dim(2) { return storedValues }
        return storedValues[0..., 0..., 0 ..< count, 0...]
    }

    /// A snapshot keeps references to the current backing arrays.  Slice
    /// assignment is mutable in MLX, so the next write must detach first;
    /// otherwise rollback would observe the post-snapshot writes.  This flag
    /// is deliberately conservative and is cleared after the first detached
    /// write.  Multiple snapshots taken before a write can all share the same
    /// backing storage safely.
    private var hasOutstandingSnapshot = false

    private static let minimumCapacity = 32

    public init() {}

    public var offset: Int { count }

    @discardableResult
    public func update(keys newKeys: MLXArray, values newValues: MLXArray) -> (MLXArray, MLXArray) {
        let tokenCount = newKeys.dim(2)
        guard tokenCount > 0 else {
            // Empty prefill is legal for a few duplex edge cases.  It must
            // not allocate a zero-sized cache or alter the logical offset.
            return (self.keys ?? newKeys, self.values ?? newValues)
        }

        precondition(newValues.dim(2) == tokenCount,
                     "MiniCPM TTS K/V sequence lengths differ")
        let required = count + tokenCount
        prepareForMutation(required: required, keys: newKeys, values: newValues)

        guard let storedKeys, let storedValues else {
            preconditionFailure("cache storage was not allocated")
        }
        let end = count + tokenCount
        storedKeys[0..., 0..., count ..< end, 0...] = newKeys
        storedValues[0..., 0..., count ..< end, 0...] = newValues
        count = end

        // Return only the live prefix.  A view is sufficient here and avoids
        // re-materialising the entire history on every token.
        return (
            count == storedKeys.dim(2)
                ? storedKeys
                : storedKeys[0..., 0..., 0 ..< count, 0...],
            count == storedValues.dim(2)
                ? storedValues
                : storedValues[0..., 0..., 0 ..< count, 0...])
    }

    public func reset() {
        storedKeys = nil
        storedValues = nil
        count = 0
        capacityGrowthCount = 0
        hasOutstandingSnapshot = false
    }

    /// Ensure there is room for ``required`` entries.  Growth is geometric,
    /// so the number of full-history copies is O(log T) rather than one copy
    /// per generated token.  If a snapshot is live, detach even when there is
    /// spare room to preserve its immutable view for rollback.
    private func prepareForMutation(
        required: Int,
        keys newKeys: MLXArray,
        values newValues: MLXArray
    ) {
        let oldCapacity = capacity
        let needsGrowth = storedKeys == nil || required > oldCapacity
        guard needsGrowth || hasOutstandingSnapshot else { return }

        let targetCapacity = needsGrowth
            ? Self.roundedCapacity(required)
            : oldCapacity
        let newShape = [newKeys.dim(0), newKeys.dim(1), targetCapacity, newKeys.dim(3)]
        let newKeyStorage = MLXArray.zeros(newShape, dtype: newKeys.dtype)
        let newValueStorage = MLXArray.zeros(
            [newValues.dim(0), newValues.dim(1), targetCapacity, newValues.dim(3)],
            dtype: newValues.dtype)

        if let oldKeys = storedKeys, let oldValues = storedValues, count > 0 {
            newKeyStorage[0..., 0..., 0 ..< count, 0...] =
                oldKeys[0..., 0..., 0 ..< count, 0...]
            newValueStorage[0..., 0..., 0 ..< count, 0...] =
                oldValues[0..., 0..., 0 ..< count, 0...]
        }
        storedKeys = newKeyStorage
        storedValues = newValueStorage
        hasOutstandingSnapshot = false
        if needsGrowth { capacityGrowthCount += 1 }
    }

    private static func roundedCapacity(_ required: Int) -> Int {
        var result = minimumCapacity
        while result < required {
            result *= 2
        }
        return result
    }

    fileprivate struct State {
        let keys: MLXArray?
        let values: MLXArray?
        let count: Int
        let capacityGrowthCount: Int
    }

    fileprivate func snapshotState() -> State {
        hasOutstandingSnapshot = true
        return State(
            keys: storedKeys,
            values: storedValues,
            count: count,
            capacityGrowthCount: capacityGrowthCount)
    }

    fileprivate func restoreState(_ state: State) {
        storedKeys = state.keys
        storedValues = state.values
        count = state.count
        capacityGrowthCount = state.capacityGrowthCount
        // Keep detaching after restore.  The caller may reuse the same
        // snapshot for another interrupt, and that snapshot must remain
        // stable across every replay.
        hasOutstandingSnapshot = true
    }
}

/// An immutable checkpoint of one semantic-TTS stream.
///
/// Each layer cache uses geometric backing storage and copy-on-write when a
/// snapshot is live.  A snapshot therefore retains the existing storage
/// without an eager full-history copy, while an interrupted/rejected unit can
/// still be rolled back exactly after mutable MLX slice writes.
public struct MiniCPMTTSSessionSnapshot {
    public let cacheOffset: Int
    public let pendingToken: Int?
    public let committedTokens: [Int]
    public let lastCommittedTokens: [Int]
    public let textStartPosition: Int
    public let chunkIndex: Int
    public let turnActive: Bool

    fileprivate let layerStates: [MiniCPMTTSLayerCache.State]

    fileprivate init(
        layerStates: [MiniCPMTTSLayerCache.State],
        pendingToken: Int?,
        committedTokens: [Int],
        lastCommittedTokens: [Int],
        textStartPosition: Int,
        chunkIndex: Int,
        turnActive: Bool
    ) {
        self.layerStates = layerStates
        self.cacheOffset = layerStates.first?.count ?? 0
        self.pendingToken = pendingToken
        self.committedTokens = committedTokens
        self.lastCommittedTokens = lastCommittedTokens
        self.textStartPosition = textStartPosition
        self.chunkIndex = chunkIndex
        self.turnActive = turnActive
    }
}

/// Per-conversation state.  It is intentionally not stored on the model, so
/// multiple duplex sessions can share one loaded TTS model without sharing
/// KV history or a pending audio code.
public final class MiniCPMTTSSession {
    public let layerCaches: [MiniCPMTTSLayerCache]
    public private(set) var pendingToken: Int?
    public private(set) var committedTokens: [Int] = []
    public private(set) var lastCommittedTokens: [Int] = []
    /// Position at which the next semantic condition starts for the protocol
    /// streaming path.  The upstream MiniCPM-o contract advances by the
    /// condition length plus the 25 codes returned from a normal 26-step
    /// generation.  The sampled look-ahead code is retained internally for
    /// the next KV update, but is not part of this text-position accounting.
    public private(set) var textStartPosition: Int = 0
    /// Number of accepted streaming chunks in the current turn.
    public private(set) var chunkIndex: Int = 0
    public private(set) var turnActive: Bool = false

    public init(layerCount: Int) {
        precondition(layerCount > 0, "MiniCPM TTS decoder must have at least one layer")
        layerCaches = (0..<layerCount).map { _ in MiniCPMTTSLayerCache() }
    }

    public var cacheOffset: Int { layerCaches.first?.offset ?? 0 }

    /// Lower-level ``generateChunk`` exposes the newest sampled code as a
    /// pending look-ahead token.  The protocol ``generateStreamingChunk``
    /// facade keeps this slot between chunks so the next KV update consumes
    /// it, while still hiding it from the Token2Wav-facing result.
    public func clearPendingToken() { pendingToken = nil }

    public func reset() {
        layerCaches.forEach { $0.reset() }
        pendingToken = nil
        committedTokens.removeAll(keepingCapacity: true)
        lastCommittedTokens.removeAll(keepingCapacity: true)
        textStartPosition = 0
        chunkIndex = 0
        turnActive = false
    }

    /// End a TTS turn after its final token2wav flush.  This is intentionally
    /// equivalent to ``reset`` and is named separately so duplex code can make
    /// the EOS lifecycle explicit.
    public func finishTurn() { reset() }

    /// Capture KV, pending-code and stream-position state before an async
    /// duplex unit.  Call ``rollback(to:)`` if the unit is stale/interrupted.
    public func snapshot() -> MiniCPMTTSSessionSnapshot {
        MiniCPMTTSSessionSnapshot(
            layerStates: layerCaches.map { $0.snapshotState() },
            pendingToken: pendingToken,
            committedTokens: committedTokens,
            lastCommittedTokens: lastCommittedTokens,
            textStartPosition: textStartPosition,
            chunkIndex: chunkIndex,
            turnActive: turnActive)
    }

    /// Restore a previously captured checkpoint.  The snapshot is rejected if
    /// it belongs to another decoder depth, preventing accidental cross-session
    /// KV aliasing.
    public func restore(_ snapshot: MiniCPMTTSSessionSnapshot) throws {
        guard snapshot.layerStates.count == layerCaches.count else {
            throw MiniCPMTTSSemanticError.invalidInput(
                "session snapshot belongs to another decoder")
        }
        for (cache, state) in zip(layerCaches, snapshot.layerStates) {
            cache.restoreState(state)
        }
        pendingToken = snapshot.pendingToken
        committedTokens = snapshot.committedTokens
        lastCommittedTokens = snapshot.lastCommittedTokens
        textStartPosition = snapshot.textStartPosition
        chunkIndex = snapshot.chunkIndex
        turnActive = snapshot.turnActive
    }

    /// Transaction-friendly spelling for ``restore``.
    public func rollback(to snapshot: MiniCPMTTSSessionSnapshot) throws {
        try restore(snapshot)
    }

    fileprivate func setPendingToken(_ token: Int?) { pendingToken = token }
    fileprivate func recordCommitted(_ tokens: [Int]) {
        lastCommittedTokens = tokens
        committedTokens.append(contentsOf: tokens)
    }

    fileprivate func recordStreamingProgress(
        textStartPosition: Int,
        chunkIndex: Int,
        turnActive: Bool
    ) {
        self.textStartPosition = textStartPosition
        self.chunkIndex = chunkIndex
        self.turnActive = turnActive
    }
}

public struct MiniCPMTTSGenerationResult {
    /// Codes safe to send to Token2Wav.  The latest sampled code is excluded
    /// because it has not yet been fed through the decoder cache.
    public let committedTokens: [Int]
    /// The sampled-but-not-fed code for the next chunk, or nil after EOS.
    public let pendingToken: Int?
    public let sampledTokenCount: Int
    public let cacheOffset: Int
    /// Whether the sampled sequence ended because ``eosToken`` was selected.
    /// EOS itself is never part of ``committedTokens`` or ``pendingToken``.
    public let eosEmitted: Bool

    public init(
        committedTokens: [Int],
        pendingToken: Int?,
        sampledTokenCount: Int,
        cacheOffset: Int,
        eosEmitted: Bool = false
    ) {
        self.committedTokens = committedTokens
        self.pendingToken = pendingToken
        self.sampledTokenCount = sampledTokenCount
        self.cacheOffset = cacheOffset
        self.eosEmitted = eosEmitted
    }
}

/// Result returned by the non-streaming/full-sequence TTS facade.  Unlike a
/// protocol streaming chunk, ``tokens`` contains the complete sampled audio
/// code sequence, including the final look-ahead token when no EOS was
/// emitted.  This mirrors MiniCPM-o's ``tts.generate`` contract while still
/// using the same session-local KV implementation as duplex generation.
public struct MiniCPMTTSFullGenerationResult {
    public let tokens: [Int]
    public let sampledTokenCount: Int
    public let eosEmitted: Bool
    /// Logical KV length before an optional ``finishTurn`` reset.  The final
    /// pending token is intentionally not in this cache length.
    public let cacheOffset: Int
    public let sessionWasReset: Bool

    public var newTokens: [Int] { tokens }
}

/// Result of an interleaved (multi-condition) semantic TTS pass.  Each entry
/// in ``chunkTokens`` corresponds to one condition; ``tokens`` is their
/// concatenation in generation order.
public struct MiniCPMTTSInterleavedGenerationResult {
    public let tokens: [Int]
    public let chunkTokens: [[Int]]
    public let sampledTokenCount: Int
    public let eosEmitted: Bool
    public let cacheOffset: Int
    public let sessionWasReset: Bool

    public var newTokens: [Int] { tokens }
}

/// Result of one protocol-level semantic TTS chunk.
///
/// A normal 26-step look-ahead chunk exposes 25 codes to Token2Wav and hides
/// the newest sampled code.  That look-ahead code remains in the session and
/// is consumed before the next condition, matching the official decoder's KV
/// behavior.  ``textStartPosition`` is the value used for this condition;
/// ``nextTextStartPosition`` is the value after the returned 25-code prefix
/// has been accounted for.  This matches MiniCPM-o's Python
/// ``generate_chunk`` return value.
public struct MiniCPMTTSStreamingChunkResult {
    public let chunkIndex: Int
    public let isFirstChunk: Bool
    public let isFinalChunk: Bool
    public let endedByEOS: Bool
    public let textStartPosition: Int
    public let nextTextStartPosition: Int
    public let conditionTokenCount: Int
    public let committedTokens: [Int]
    public let pendingToken: Int?
    public let sampledTokenCount: Int
    /// Decoder KV offset before an end-of-turn reset, useful for diagnostics.
    public let cacheOffset: Int
    /// True when this result caused the session to return to an idle state.
    public let sessionWasReset: Bool

    public var committedTokenCount: Int { committedTokens.count }
    public var hasPendingToken: Bool { pendingToken != nil }

    fileprivate init(
        chunkIndex: Int,
        isFirstChunk: Bool,
        isFinalChunk: Bool,
        endedByEOS: Bool,
        textStartPosition: Int,
        nextTextStartPosition: Int,
        conditionTokenCount: Int,
        generation: MiniCPMTTSGenerationResult,
        sessionWasReset: Bool
    ) {
        self.chunkIndex = chunkIndex
        self.isFirstChunk = isFirstChunk
        self.isFinalChunk = isFinalChunk
        self.endedByEOS = endedByEOS
        self.textStartPosition = textStartPosition
        self.nextTextStartPosition = nextTextStartPosition
        self.conditionTokenCount = conditionTokenCount
        self.committedTokens = generation.committedTokens
        self.pendingToken = generation.pendingToken
        self.sampledTokenCount = generation.sampledTokenCount
        self.cacheOffset = generation.cacheOffset
        self.sessionWasReset = sessionWasReset
    }
}

@inline(__always)
private func makeMiniCPMTTSLinear(
    _ inputDimensions: Int,
    _ outputDimensions: Int,
    bias: Bool = false
) -> Linear {
    // Keep the production path on MLX.  The MPSGraph BF16 implementation is a
    // parity diagnostic only: on the pinned CPU and MPS PyTorch oracles it did
    // not improve the first-logits gate and changed stable-prefix tokens.  It
    // may still be selected explicitly while investigating backend arithmetic,
    // but an unproved diagnostic kernel must never become the model default.
    let backend = ProcessInfo.processInfo.environment[
        "MINICPM_TTS_BF16_LINEAR"]?.lowercased()
    let exactKernels = ProcessInfo.processInfo.environment[
        "MINICPM_TTS_EXACT_KERNELS"]?.lowercased()
    if !bias, backend == "mpsgraph" || exactKernels == "mpsgraph" {
        return MiniCPMExactBF16Linear(
            inputDimensions,
            outputDimensions,
            bias: false)
    }
    return Linear(inputDimensions, outputDimensions, bias: bias)
}

@inline(__always)
private func makeMiniCPMTTSRMSNorm(_ dimensions: Int, eps: Float) -> RMSNorm {
    let exact = ProcessInfo.processInfo.environment[
        "MINICPM_TTS_EXACT_KERNELS"]?.lowercased() == "mpsgraph"
    if exact {
        return MiniCPMExactBF16RMSNorm(dimensions: dimensions, eps: eps)
    }
    return RMSNorm(dimensions: dimensions, eps: eps)
}

final class MiniCPMTTSProjector: Module {
    @ModuleInfo(key: "linear1") var linear1: Linear
    @ModuleInfo(key: "linear2") var linear2: Linear

    init(inputDimensions: Int, hiddenSize: Int) {
        _linear1 = ModuleInfo(wrappedValue: Linear(inputDimensions, hiddenSize, bias: true))
        _linear2 = ModuleInfo(wrappedValue: Linear(hiddenSize, hiddenSize, bias: true))
        super.init()
    }

    func callAsFunction(_ hidden: MLXArray) -> MLXArray {
        linear2(relu(linear1(hidden)))
    }
}

/// HuggingFace Llama's default rotary embedding used by the pinned MiniCPM-o
/// TTS checkpoint.
///
/// ``MLXNN.RoPE(traditional: false)`` is interleaved, while Transformers'
/// ``rotate_half`` is split-half.  The distinction is invisible at position
/// zero for some inputs but produces a large attention drift as soon as the
/// sequence contains non-zero positions.  Keep the frequency/triangular
/// arithmetic explicit here so the Swift path follows the official order:
/// float32 frequencies and trigonometry, cast cos/sin to the input dtype, then
/// split-half rotation.
struct MiniCPMTTSHFStyleRoPE {
    let dimensions: Int
    let base: Float
    /// Transformers initializes the default Llama inverse frequencies in
    /// float32 (the model input is cast only after cos/sin are computed).
    let inverseFrequency: MLXArray

    init(dimensions: Int, base: Float = 10_000) {
        precondition(dimensions > 0 && dimensions.isMultiple(of: 2))
        self.dimensions = dimensions
        self.base = base
        let half = dimensions / 2
        self.inverseFrequency = MLXArray((0..<half).map { index in
            Float(pow(Double(base), -2 * Double(index) / Double(dimensions)))
        }, [1, half])
    }

    func callAsFunction(_ input: MLXArray, offset: Int = 0) -> MLXArray {
        precondition(input.ndim == 4)
        precondition(input.dim(-1) == dimensions)
        let length = input.dim(2)
        guard length > 0 else { return input }

        let positions = MLXArray(
            (0..<length).map { Float(offset + $0) }, [length, 1])
        let frequencies = matmul(
            positions,
            inverseFrequency.asType(.float32))
        let cosine = concatenated([cos(frequencies), cos(frequencies)], axis: -1)
            .asType(input.dtype)
            .expandedDimensions(axis: 0)
            .expandedDimensions(axis: 0)
        let sine = concatenated([sin(frequencies), sin(frequencies)], axis: -1)
            .asType(input.dtype)
            .expandedDimensions(axis: 0)
            .expandedDimensions(axis: 0)

        let half = dimensions / 2
        let first = input[0..., 0..., 0..., 0 ..< half]
        let second = input[0..., 0..., 0..., half ..< dimensions]
        let rotateHalf = concatenated([-second, first], axis: -1)
        return input * cosine + rotateHalf * sine
    }
}

final class MiniCPMTTSAudioAttention: Module {
    let numHeads: Int
    let numKVHeads: Int
    let headDimension: Int
    let scale: Float
    let rope: MiniCPMTTSHFStyleRoPE
    private let useExactBF16SDPA: Bool

    @ModuleInfo(key: "q_proj") var query: Linear
    @ModuleInfo(key: "k_proj") var key: Linear
    @ModuleInfo(key: "v_proj") var value: Linear
    @ModuleInfo(key: "o_proj") var output: Linear

    init(_ configuration: MiniCPMTTSSemanticConfiguration) {
        numHeads = configuration.numAttentionHeads
        numKVHeads = configuration.numKeyValueHeads
        headDimension = configuration.hiddenSize / configuration.numAttentionHeads
        scale = 1 / sqrt(Float(headDimension))
        useExactBF16SDPA = ProcessInfo.processInfo.environment[
            "MINICPM_TTS_EXACT_KERNELS"]?.lowercased() == "mpsgraph"
        let hidden = configuration.hiddenSize
        _query = ModuleInfo(wrappedValue: makeMiniCPMTTSLinear(
            hidden, numHeads * headDimension))
        _key = ModuleInfo(wrappedValue: makeMiniCPMTTSLinear(
            hidden, numKVHeads * headDimension))
        _value = ModuleInfo(wrappedValue: makeMiniCPMTTSLinear(
            hidden, numKVHeads * headDimension))
        _output = ModuleInfo(wrappedValue: makeMiniCPMTTSLinear(
            numHeads * headDimension, hidden))
        rope = MiniCPMTTSHFStyleRoPE(
            dimensions: headDimension,
            base: 10_000)
        super.init()
    }

    func callAsFunction(
        _ input: MLXArray,
        cache: MiniCPMTTSLayerCache,
        offset: Int,
        mask: MLXFast.ScaledDotProductAttentionMaskMode
    ) -> MLXArray {
        let batch = input.dim(0)
        let sequence = input.dim(1)
        var q = query(input).reshaped(batch, sequence, numHeads, headDimension)
        var k = key(input).reshaped(batch, sequence, numKVHeads, headDimension)
        let v = value(input).reshaped(batch, sequence, numKVHeads, headDimension)
        q = q.transposed(0, 2, 1, 3)
        k = k.transposed(0, 2, 1, 3)
        let values = v.transposed(0, 2, 1, 3)

        let ropeOffset = cache.offset > 0 ? cache.offset : offset
        q = rope(q, offset: ropeOffset)
        k = rope(k, offset: ropeOffset)
        let (keys, cachedValues) = cache.update(keys: k, values: values)
        let attended: MLXArray
        if useExactBF16SDPA,
           q.dtype == .bfloat16,
           keys.dtype == .bfloat16,
           cachedValues.dtype == .bfloat16,
           sequence > 1
        {
            let additiveMask: MLXArray? = keys.dim(2) > sequence
                ? MiniCPMExactBF16SDPA.rightAlignedAdditiveMask(
                    batch: batch,
                    queryHeads: numHeads,
                    queryTokens: sequence,
                    keyTokens: keys.dim(2))
                : nil
            let exact = MiniCPMExactBF16SDPA.attend(
                query: q,
                key: keys,
                value: cachedValues,
                scale: scale,
                additiveMask: additiveMask)
            attended = exact
                .reshaped(batch, sequence, numHeads * headDimension)
        } else {
            attended = MLXFast.scaledDotProductAttention(
                queries: q,
                keys: keys,
                values: cachedValues,
                scale: scale,
                mask: mask)
                .transposed(0, 2, 1, 3)
                .reshaped(batch, sequence, numHeads * headDimension)
        }
        return output(attended)
    }
}

final class MiniCPMTTSAudioLayer: Module {
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm
    @ModuleInfo(key: "self_attn") var attention: MiniCPMTTSAudioAttention
    @ModuleInfo var mlp: MiniCPMTTSMLP

    init(_ configuration: MiniCPMTTSSemanticConfiguration) {
        _inputLayerNorm = ModuleInfo(wrappedValue: makeMiniCPMTTSRMSNorm(
            configuration.hiddenSize,
            eps: configuration.rmsNormEps))
        _postAttentionLayerNorm = ModuleInfo(wrappedValue: makeMiniCPMTTSRMSNorm(
            configuration.hiddenSize,
            eps: configuration.rmsNormEps))
        _attention = ModuleInfo(wrappedValue: MiniCPMTTSAudioAttention(configuration))
        _mlp = ModuleInfo(wrappedValue: MiniCPMTTSMLP(configuration))
        super.init()
    }

    func callAsFunction(
        _ input: MLXArray,
        cache: MiniCPMTTSLayerCache,
        offset: Int,
        mask: MLXFast.ScaledDotProductAttentionMaskMode
    ) -> MLXArray {
        let attended = attention(
            inputLayerNorm(input),
            cache: cache,
            offset: offset,
            mask: mask)
        let residual = input + attended
        return residual + mlp(postAttentionLayerNorm(residual))
    }
}

final class MiniCPMTTSMLP: Module {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "up_proj") var up: Linear
    @ModuleInfo(key: "down_proj") var down: Linear
    private let useExactBF16SiLU: Bool

    init(_ configuration: MiniCPMTTSSemanticConfiguration) {
        useExactBF16SiLU = ProcessInfo.processInfo.environment[
            "MINICPM_TTS_EXACT_KERNELS"]?.lowercased() == "mpsgraph"
        _gate = ModuleInfo(wrappedValue: makeMiniCPMTTSLinear(
            configuration.hiddenSize,
            configuration.intermediateSize))
        _up = ModuleInfo(wrappedValue: makeMiniCPMTTSLinear(
            configuration.hiddenSize,
            configuration.intermediateSize))
        _down = ModuleInfo(wrappedValue: makeMiniCPMTTSLinear(
            configuration.intermediateSize,
            configuration.hiddenSize))
        super.init()
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        let gateValue = gate(input)
        let activated = useExactBF16SiLU
            ? MiniCPMExactBF16SiLU.apply(gateValue)
            : silu(gateValue)
        return down(activated * up(input))
    }
}

final class MiniCPMTTSAudioDecoder: Module {
    @ModuleInfo(key: "layers") var layers: [MiniCPMTTSAudioLayer]
    @ModuleInfo var norm: RMSNorm

    let configuration: MiniCPMTTSSemanticConfiguration

    init(_ configuration: MiniCPMTTSSemanticConfiguration) {
        self.configuration = configuration
        _layers = ModuleInfo(wrappedValue: (0..<configuration.numHiddenLayers).map { _ in
            MiniCPMTTSAudioLayer(configuration)
        })
        _norm = ModuleInfo(wrappedValue: makeMiniCPMTTSRMSNorm(
            configuration.hiddenSize,
            eps: configuration.rmsNormEps))
        super.init()
    }

    func callAsFunction(
        _ embeddings: MLXArray,
        caches: [MiniCPMTTSLayerCache],
        offset: Int
    ) -> MLXArray {
        precondition(caches.count == layers.count)
        let batch = embeddings.dim(0)
        let sequence = embeddings.dim(1)
        let keyLength = (caches.first?.offset ?? 0) + sequence
        let mask: MLXFast.ScaledDotProductAttentionMaskMode
        if ProcessInfo.processInfo.environment["MINICPM_TTS_MASK_VARIANT"] == "causal"
            && sequence > 1
            && (caches.first?.offset ?? 0) == 0
        {
            // Diagnostic/compatibility path: the pinned PyTorch decoder calls
            // SDPA with ``is_causal`` for a first full-sequence prefill.  MLX
            // can construct the same triangular mask internally, avoiding a
            // BF16 cast of ``-finfo.max`` in an explicit additive array.
            mask = .causal
        } else if sequence == 1 && (caches.first?.offset ?? 0) > 0 {
            mask = .none
        } else {
            let past = keyLength - sequence
            // MLX's triangular mask is one for allowed entries.  Convert it
            // into the additive mask expected by SDPA, matching LlamaModel.
            let causal = MLXArray.tri(sequence, m: keyLength, k: past, type: Float.self) - 1
            let additive = causal * Float.greatestFiniteMagnitude
            mask = .array(additive.reshaped(batch, 1, sequence, keyLength).asType(embeddings.dtype))
        }

        var hidden = embeddings
        for (index, layer) in layers.enumerated() {
            hidden = layer(hidden, cache: caches[index], offset: offset, mask: mask)
        }
        return norm(hidden)
    }
}

/// Pure-MLX MiniCPM-o semantic TTS condition + audio-token decoder.
///
/// The language model remains owned by the duplex runtime.  It passes its
/// hidden states to ``buildCondition`` and then carries a session-local
/// ``MiniCPMTTSSession`` through ``generateChunk``.  This keeps model weights
/// immutable/shared while KV and pending audio-code state remain isolated per
/// conversation.
public final class MiniCPMTTSSemantic: Module {
    public let configuration: MiniCPMTTSSemanticConfiguration

    @ModuleInfo(key: "emb_text") public var textEmbedding: Embedding
    @ModuleInfo(key: "projector_semantic") var semanticProjector: MiniCPMTTSProjector
    @ModuleInfo(key: "emb_code") var audioEmbedding: Embedding
    @ModuleInfo(key: "model") var decoder: MiniCPMTTSAudioDecoder
    @ModuleInfo(key: "head_code") var audioHead: Linear

    public init(configuration: MiniCPMTTSSemanticConfiguration = .init()) {
        self.configuration = configuration
        _textEmbedding = ModuleInfo(wrappedValue: Embedding(
            embeddingCount: configuration.numTextTokens,
            dimensions: configuration.hiddenSize))
        _semanticProjector = ModuleInfo(wrappedValue: MiniCPMTTSProjector(
            inputDimensions: configuration.llmDim,
            hiddenSize: configuration.hiddenSize))
        _audioEmbedding = ModuleInfo(wrappedValue: Embedding(
            embeddingCount: configuration.numAudioTokens,
            dimensions: configuration.hiddenSize))
        _decoder = ModuleInfo(wrappedValue: MiniCPMTTSAudioDecoder(configuration))
        _audioHead = ModuleInfo(wrappedValue: makeMiniCPMTTSLinear(
            configuration.hiddenSize,
            configuration.numAudioTokens))
        super.init()
    }

    public func makeSession() -> MiniCPMTTSSession {
        MiniCPMTTSSession(layerCount: configuration.numHiddenLayers)
    }

    /// Load ``models/MiniCPM-o-4_5-tts-mlx`` with strict key checking.
    public static func fromDirectory(_ directory: URL) throws -> MiniCPMTTSSemantic {
        let configuration = try MiniCPMTTSSemanticConfiguration.fromBundle(at: directory)
        let weightsURL = directory.appendingPathComponent("model.safetensors")
        guard FileManager.default.fileExists(atPath: weightsURL.path) else {
            throw MiniCPMTTSSemanticError.missingModelFile(weightsURL)
        }
        let model = MiniCPMTTSSemantic(configuration: configuration)
        let raw: [String: MLXArray]
        do {
            raw = try MLX.loadArrays(url: weightsURL)
        } catch {
            throw MiniCPMTTSSemanticError.invalidWeights(error.localizedDescription)
        }
        var mapped: [String: MLXArray] = [:]
        mapped.reserveCapacity(raw.count)
        for (key, value) in raw {
            guard key.hasPrefix("tts.") else {
                throw MiniCPMTTSSemanticError.invalidWeights(
                    "unexpected non-TTS tensor (key)")
            }
            let stripped = String(key.dropFirst(4))
            if key == "tts.head_code.0.parametrizations.weight.original0"
                || key == "tts.head_code.0.parametrizations.weight.original1"
            {
                continue
            }
            guard stripped != "model.embed_tokens.weight",
                  !stripped.hasPrefix("projector_spk.")
            else { continue }

            // The Python checkpoint follows the original MiniCPM module
            // names (``model.layers.*.self_attn.q_proj``), while MLX Swift
            // uses the names of the Swift properties.  Keep this translation
            // explicit instead of relying on a permissive loader: it makes a
            // renamed or omitted tensor fail the strict key check below.
            let swiftKey: String?
            switch stripped {
            case "emb_text.weight":
                swiftKey = "textEmbedding.weight"
            case "emb_code.0.weight":
                swiftKey = "audioEmbedding.weight"
            case "projector_semantic.linear1.weight":
                swiftKey = "semanticProjector.linear1.weight"
            case "projector_semantic.linear1.bias":
                swiftKey = "semanticProjector.linear1.bias"
            case "projector_semantic.linear2.weight":
                swiftKey = "semanticProjector.linear2.weight"
            case "projector_semantic.linear2.bias":
                swiftKey = "semanticProjector.linear2.bias"
            case "model.norm.weight":
                swiftKey = "decoder.norm.weight"
            default:
                swiftKey = Self.swiftDecoderWeightKey(from: stripped)
            }
            guard let swiftKey else {
                throw MiniCPMTTSSemanticError.invalidWeights(
                    "unrecognized TTS tensor (\(key))")
            }
            mapped[swiftKey] = value
        }

        guard let magnitude = raw["tts.head_code.0.parametrizations.weight.original0"],
              let direction = raw["tts.head_code.0.parametrizations.weight.original1"]
        else {
            throw MiniCPMTTSSemanticError.invalidWeights("missing weight-normalized audio head")
        }
        let direction32 = direction.asType(.float32)
        let magnitude32 = magnitude.asType(.float32)
        let norm = MLX.sqrt((direction32 * direction32).sum(axis: -1, keepDims: true))
        mapped["audioHead.weight"] = (
            direction32 * magnitude32 / MLX.maximum(norm, MLXArray(Float(1e-12)))
        ).asType(model.textEmbedding.weight.dtype)

        let expected = Set(model.parameters().flattened().map(\.0))
        let actual = Set(mapped.keys)
        let missing = expected.subtracting(actual).sorted()
        let unknown = actual.subtracting(expected).sorted()
        guard missing.isEmpty, unknown.isEmpty else {
            throw MiniCPMTTSSemanticError.invalidWeights(
                "strict key mismatch; missing=\(missing.prefix(8)), unexpected=\(unknown.prefix(8))")
        }
        do {
            try model.update(
                parameters: ModuleParameters.unflattened(mapped),
                verify: .all)
        } catch {
            throw MiniCPMTTSSemanticError.invalidWeights(error.localizedDescription)
        }
        model.train(false)
        eval(model)
        return model
    }

    /// Translate one ``tts.model.layers.*`` checkpoint key to the names
    /// emitted by MLX Swift's ``Module.parameters()`` tree.
    private static func swiftDecoderWeightKey(from stripped: String) -> String? {
        let pieces = stripped.split(separator: ".", omittingEmptySubsequences: true)
        guard pieces.count >= 5,
              pieces[0] == "model",
              pieces[1] == "layers",
              pieces[2].allSatisfy(\.isNumber)
        else { return nil }
        let layer = pieces[2]
        let suffix = pieces.dropFirst(3).joined(separator: ".")
        switch suffix {
        case "input_layernorm.weight":
            return "decoder.layers.\(layer).inputLayerNorm.weight"
        case "post_attention_layernorm.weight":
            return "decoder.layers.\(layer).postAttentionLayerNorm.weight"
        case "self_attn.q_proj.weight":
            return "decoder.layers.\(layer).attention.query.weight"
        case "self_attn.k_proj.weight":
            return "decoder.layers.\(layer).attention.key.weight"
        case "self_attn.v_proj.weight":
            return "decoder.layers.\(layer).attention.value.weight"
        case "self_attn.o_proj.weight":
            return "decoder.layers.\(layer).attention.output.weight"
        case "mlp.gate_proj.weight":
            return "decoder.layers.\(layer).mlp.gate.weight"
        case "mlp.up_proj.weight":
            return "decoder.layers.\(layer).mlp.up.weight"
        case "mlp.down_proj.weight":
            return "decoder.layers.\(layer).mlp.down.weight"
        default:
            return nil
        }
    }

    /// Build the [text embedding + normalized projected hidden + audio BOS]
    /// sequence consumed by the 20-layer audio decoder.
    ///
    /// ``hidden``/``tokenIDs`` must already be aligned to the *semantic text
    /// span*.  In the official streaming loop the first sampled token is a
    /// probe for the next-step logits and is deliberately omitted before this
    /// helper is called; `<|speak|>`/chunk markers are likewise protocol
    /// bookkeeping, not an implicit extra condition here.  Turn-based final
    /// callers can request ``addTextEOS`` before the always-appended audio
    /// BOS, while duplex units leave it false.
    public func buildCondition(
        hidden: MLXArray,
        tokenIDs: MLXArray,
        addAudioBOS: Bool = true,
        addTextEOS: Bool = false
    ) throws -> MLXArray {
        let hiddenSequence: MLXArray
        if hidden.ndim == 2 {
            hiddenSequence = hidden.expandedDimensions(axis: 0)
        } else if hidden.ndim == 3, hidden.dim(0) == 1 {
            hiddenSequence = hidden
        } else {
            throw MiniCPMTTSSemanticError.invalidInput("hidden must be [T,D] or [1,T,D]")
        }
        guard hiddenSequence.dim(2) == configuration.llmDim else {
            throw MiniCPMTTSSemanticError.invalidInput("hidden width does not equal llm_dim")
        }
        let ids = tokenIDs.reshaped(-1).asType(.int32)
        guard ids.dim(0) == hiddenSequence.dim(1) else {
            throw MiniCPMTTSSemanticError.invalidInput("hidden/token sequence lengths differ")
        }
        // ``MiniCPMODuplex._convert_results_to_tts_input([])`` is a valid
        // first-turn condition: there are no LLM text tokens, only the audio
        // BOS embedding.  Avoid reductions (min/max/projector) over an empty
        // sequence and preserve optional text-EOS/BOS ordering explicitly.
        if ids.dim(0) == 0 {
            let dtype = textEmbedding.weight.dtype
            var empty = MLXArray.zeros([1, 0, configuration.hiddenSize], dtype: dtype)
            if addTextEOS {
                let eos = textEmbedding(MLXArray([Int32(configuration.textEOSTokenID)]))
                    .expandedDimensions(axis: 0)
                empty = concatenated([empty, eos.asType(dtype)], axis: 1)
            }
            if addAudioBOS {
                let bos = textEmbedding(MLXArray([Int32(configuration.audioBOSTokenID)]))
                    .expandedDimensions(axis: 0)
                empty = concatenated([empty, bos.asType(dtype)], axis: 1)
            }
            return empty
        }
        let minID = ids.min().item(Int.self)
        let maxID = ids.max().item(Int.self)
        guard minID >= 0, maxID < configuration.numTextTokens else {
            throw MiniCPMTTSSemanticError.invalidInput("text token outside embedding vocabulary")
        }

        let dtype = textEmbedding.weight.dtype
        let text = textEmbedding(ids).asType(dtype)
        var projected = semanticProjector(hiddenSequence.asType(dtype))
        if configuration.normalizeProjectedHidden {
            let projected32 = projected.asType(.float32)
            let denominator = MLX.sqrt(
                MLX.sum(projected32 * projected32, axis: -1, keepDims: true))
            projected = (
                projected32 / MLX.maximum(denominator, MLXArray(Float(1e-12)))
            ).asType(dtype)
        }
        var condition = text.expandedDimensions(axis: 0) + projected
        if addTextEOS {
            let eos = textEmbedding(MLXArray([Int32(configuration.textEOSTokenID)]))
                .expandedDimensions(axis: 0)
            condition = concatenated([condition, eos.asType(dtype)], axis: 1)
        }
        if addAudioBOS {
            let bos = textEmbedding(MLXArray([Int32(configuration.audioBOSTokenID)]))
                .expandedDimensions(axis: 0)
            condition = concatenated([condition, bos.asType(dtype)], axis: 1)
        }
        return condition
    }

    public func audioBOSEmbedding() -> MLXArray {
        textEmbedding(MLXArray([Int32(configuration.audioBOSTokenID)]))
            .expandedDimensions(axis: 0)
    }

    /// Generate one streaming chunk.  ``session`` owns all mutable KV and
    /// pending-token state; the model itself remains safe to share.
    @discardableResult
    public func generateChunk(
        inputsEmbeds: MLXArray,
        session: MiniCPMTTSSession,
        temperature: Float = 0.7,
        repetitionPenalty: Float = 1.0,
        eosToken: Int? = nil,
        forceNoStop: Bool = false,
        maxNewToken: Int = 500,
        minNewTokens: Int = 0,
        shouldCancel: (@Sendable () -> Bool)? = nil
    ) throws -> MiniCPMTTSGenerationResult {
        if shouldCancel?() == true { throw MiniCPMTTSSemanticError.cancelled }
        guard session.layerCaches.count == decoder.layers.count else {
            throw MiniCPMTTSSemanticError.invalidInput("session belongs to another decoder")
        }
        guard maxNewToken >= 0, minNewTokens >= 0 else {
            throw MiniCPMTTSSemanticError.invalidInput("negative generation length")
        }
        let eos = eosToken ?? configuration.numAudioTokens - 1
        guard eos >= 0, eos < configuration.numAudioTokens else {
            throw MiniCPMTTSSemanticError.invalidInput("EOS outside audio vocabulary")
        }

        var embeds: MLXArray
        if inputsEmbeds.ndim == 2 {
            embeds = inputsEmbeds.expandedDimensions(axis: 0)
        } else if inputsEmbeds.ndim == 3, inputsEmbeds.dim(0) == 1 {
            embeds = inputsEmbeds
        } else {
            throw MiniCPMTTSSemanticError.invalidInput("embeddings must be [T,H] or [1,T,H]")
        }
        guard embeds.dim(2) == configuration.hiddenSize else {
            throw MiniCPMTTSSemanticError.invalidInput("embedding width does not equal TTS hidden size")
        }

        // The previous chunk's newest code is deliberately not part of the
        // committed stream.  Feed it now, before the next semantic condition.
        if let pending = session.pendingToken {
            let pendingEmbedding = audioEmbedding(
                MLXArray([Int32(pending)])).expandedDimensions(axis: 0)
            embeds = concatenated([pendingEmbedding, embeds], axis: 1)
            session.clearPendingToken()
        }

        var generated: [Int] = []
        generated.reserveCapacity(maxNewToken)
        let cacheOffset = session.cacheOffset
        for step in 0..<maxNewToken {
            if shouldCancel?() == true { throw MiniCPMTTSSemanticError.cancelled }
            let current: MLXArray
            if step == 0 {
                current = embeds
            } else {
                current = audioEmbedding(
                    MLXArray([Int32(generated[generated.count - 1])]))
                    .expandedDimensions(axis: 0)
            }
            let hidden = decoder(
                current,
                caches: session.layerCaches,
                offset: cacheOffset)
            var logits = audioHead(hidden[0, hidden.dim(1) - 1]).asType(.float32)
            if repetitionPenalty != 1, !generated.isEmpty {
                logits = applyRepetitionPenalty(
                    logits,
                    generated: generated,
                    penalty: repetitionPenalty)
            }
            if forceNoStop || step < minNewTokens {
                let ids = MLXArray(0..<Int32(configuration.numAudioTokens))
                logits = MLX.where(
                    ids .== MLXArray(Int32(eos)),
                    MLXArray(-Float.greatestFiniteMagnitude),
                    logits)
            }
            let token = sample(logits: logits, temperature: temperature)
            if shouldCancel?() == true { throw MiniCPMTTSSemanticError.cancelled }
            generated.append(token)
            if token == eos { break }
        }

        let pending: Int?
        let committed: [Int]
        let eosEmitted = generated.last == eos
        if let newest = generated.last, newest != eos {
            pending = newest
            committed = Array(generated.dropLast())
        } else {
            pending = nil
            committed = generated.isEmpty ? [] : Array(generated.dropLast())
        }
        session.setPendingToken(pending)
        session.recordCommitted(committed)
        return MiniCPMTTSGenerationResult(
            committedTokens: committed,
            pendingToken: pending,
            sampledTokenCount: generated.count,
            cacheOffset: session.cacheOffset,
            eosEmitted: eosEmitted)
    }

    /// Non-streaming/full-sequence semantic TTS entry point.
    ///
    /// The official MiniCPM chat/half-duplex path calls ``tts.generate``
    /// once with a complete text condition.  The native implementation keeps
    /// one numerical kernel for that path and duplex chunks: a chunk is
    /// generated with the same KV cache, then its pending look-ahead code is
    /// folded back into the returned full sequence.  A private session is
    /// created by default and released after generation, so a regular Chat
    /// caller never has to manage KV state.  Supplying a session is useful for
    /// an engine that wants to retain or inspect the cache; set
    /// ``resetAfterGeneration`` to false in that case.
    public func generate(
        inputsEmbeds: MLXArray,
        session: MiniCPMTTSSession? = nil,
        temperature: Float = 0.7,
        repetitionPenalty: Float = 1.0,
        eosToken: Int? = nil,
        forceNoStop: Bool = false,
        maxNewToken: Int = 2048,
        minNewTokens: Int = 0,
        resetAfterGeneration: Bool = true,
        shouldCancel: (@Sendable () -> Bool)? = nil
    ) throws -> MiniCPMTTSFullGenerationResult {
        let ownedSession = session ?? makeSession()
        let generation = try generateChunk(
            inputsEmbeds: inputsEmbeds,
            session: ownedSession,
            temperature: temperature,
            repetitionPenalty: repetitionPenalty,
            eosToken: eosToken,
            forceNoStop: forceNoStop,
            maxNewToken: maxNewToken,
            minNewTokens: minNewTokens,
            shouldCancel: shouldCancel)

        // ``generateChunk`` holds one normal token pending (the look-ahead
        // code) and drops EOS entirely.  Full-sequence callers need the
        // former, but never the latter.
        var tokens = generation.committedTokens
        if let pending = generation.pendingToken {
            tokens.append(pending)
        }
        let cacheOffset = generation.cacheOffset
        let shouldReset = resetAfterGeneration
        if shouldReset {
            ownedSession.finishTurn()
        }
        return MiniCPMTTSFullGenerationResult(
            tokens: tokens,
            sampledTokenCount: generation.sampledTokenCount,
            eosEmitted: generation.eosEmitted,
            cacheOffset: cacheOffset,
            sessionWasReset: shouldReset)
    }

    /// Source-compatible spelling for callers ported directly from the
    /// upstream Python ``tts.generate`` entry point.
    public func generate(
        inputs_embeds: MLXArray,
        session: MiniCPMTTSSession? = nil,
        temperature: Float = 0.7,
        repetition_penalty: Float = 1.0,
        eos_token: Int? = nil,
        force_no_stop: Bool = false,
        max_new_token: Int = 2048,
        min_new_token: Int = 0,
        reset_after_generation: Bool = true,
        should_cancel: (@Sendable () -> Bool)? = nil
    ) throws -> MiniCPMTTSFullGenerationResult {
        try generate(
            inputsEmbeds: inputs_embeds,
            session: session,
            temperature: temperature,
            repetitionPenalty: repetition_penalty,
            eosToken: eos_token,
            forceNoStop: force_no_stop,
            maxNewToken: max_new_token,
            minNewTokens: min_new_token,
            resetAfterGeneration: reset_after_generation,
            shouldCancel: should_cancel)
    }

    /// Interleaved semantic TTS facade used by MiniCPM's chat path.
    ///
    /// ``conditions`` are fed in order, carrying the same pending-code and
    /// KV state between calls.  Optional speaker embeddings are prepended to
    /// the first condition (the official path uses an empty tensor when no
    /// speaker prompt is present).  Every condition contributes its complete
    /// sampled code sequence to the result; the final look-ahead code of one
    /// condition is fed before the next condition, preserving the streaming
    /// protocol without a second decoder implementation.
    public func interleavedGenerate(
        speakerEmbeds: MLXArray? = nil,
        conditions: [MLXArray],
        session: MiniCPMTTSSession? = nil,
        temperature: Float = 0.7,
        repetitionPenalty: Float = 1.0,
        eosToken: Int? = nil,
        forceNoStop: Bool = false,
        maxNewToken: Int = 500,
        minNewTokens: Int = 0,
        resetAfterGeneration: Bool = true
    ) throws -> MiniCPMTTSInterleavedGenerationResult {
        let ownedSession = session ?? makeSession()
        guard !conditions.isEmpty else {
            let shouldReset = resetAfterGeneration
            if shouldReset { ownedSession.finishTurn() }
            return MiniCPMTTSInterleavedGenerationResult(
                tokens: [],
                chunkTokens: [],
                sampledTokenCount: 0,
                eosEmitted: false,
                cacheOffset: ownedSession.cacheOffset,
                sessionWasReset: shouldReset)
        }

        var allTokens: [Int] = []
        var perChunk: [[Int]] = []
        var sampledCount = 0
        var endedByEOS = false
        perChunk.reserveCapacity(conditions.count)

        for (index, rawCondition) in conditions.enumerated() {
            var condition = rawCondition
            if index == 0, let speakerEmbeds, speakerEmbeds.size > 0 {
                condition = concatenateCondition(prefix: speakerEmbeds, condition: condition)
            }
            let generation = try generateChunk(
                inputsEmbeds: condition,
                session: ownedSession,
                temperature: temperature,
                repetitionPenalty: repetitionPenalty,
                eosToken: eosToken,
                forceNoStop: forceNoStop,
                maxNewToken: maxNewToken,
                minNewTokens: minNewTokens)
            var chunk = generation.committedTokens
            if let pending = generation.pendingToken {
                chunk.append(pending)
            }
            perChunk.append(chunk)
            allTokens.append(contentsOf: chunk)
            sampledCount += generation.sampledTokenCount
            endedByEOS = endedByEOS || generation.eosEmitted

            // The generic facade intentionally carries its look-ahead code
            // into the next condition, so account for every sampled code in
            // this explicitly interleaved API.  The protocol streaming facade
            // below follows the stricter upstream contract and hides it from
            // the Token2Wav-facing result while retaining it for the next KV
            // update.
            let conditionCount = condition.ndim == 2 ? condition.dim(0) : condition.dim(1)
            let nextTextStart = ownedSession.textStartPosition
                + conditionCount
                + generation.sampledTokenCount
            ownedSession.recordStreamingProgress(
                textStartPosition: nextTextStart,
                chunkIndex: index + 1,
                turnActive: !endedByEOS)

            if generation.eosEmitted { break }
        }

        let cacheOffset = ownedSession.cacheOffset
        let shouldReset = resetAfterGeneration
        if shouldReset { ownedSession.finishTurn() }
        return MiniCPMTTSInterleavedGenerationResult(
            tokens: allTokens,
            chunkTokens: perChunk,
            sampledTokenCount: sampledCount,
            eosEmitted: endedByEOS,
            cacheOffset: cacheOffset,
            sessionWasReset: shouldReset)
    }

    /// Source-compatible spelling for callers ported directly from the
    /// upstream Python ``interleaved_generate`` method.
    public func interleaved_generate(
        spk_embeds: MLXArray? = nil,
        conditions: [MLXArray],
        session: MiniCPMTTSSession? = nil,
        temperature: Float = 0.7,
        repetition_penalty: Float = 1.0,
        eos_token: Int? = nil,
        force_no_stop: Bool = false,
        max_new_token: Int = 500,
        min_new_tokens: Int = 0,
        reset_after_generation: Bool = true
    ) throws -> MiniCPMTTSInterleavedGenerationResult {
        try interleavedGenerate(
            speakerEmbeds: spk_embeds,
            conditions: conditions,
            session: session,
            temperature: temperature,
            repetitionPenalty: repetition_penalty,
            eosToken: eos_token,
            forceNoStop: force_no_stop,
            maxNewToken: max_new_token,
            minNewTokens: min_new_tokens,
            resetAfterGeneration: reset_after_generation)
    }

    private func concatenateCondition(prefix: MLXArray, condition: MLXArray) -> MLXArray {
        let lhs: MLXArray
        if prefix.ndim == 2 {
            lhs = prefix.expandedDimensions(axis: 0)
        } else {
            precondition(prefix.ndim == 3 && prefix.dim(0) == 1,
                         "speaker embeddings must be [T,H] or [1,T,H]")
            lhs = prefix
        }
        let rhs: MLXArray
        if condition.ndim == 2 {
            rhs = condition.expandedDimensions(axis: 0)
        } else {
            precondition(condition.ndim == 3 && condition.dim(0) == 1,
                         "condition must be [T,H] or [1,T,H]")
            rhs = condition
        }
        precondition(lhs.dim(2) == rhs.dim(2),
                     "speaker and condition embedding widths differ")
        return concatenated([lhs, rhs], axis: 1)
    }

    /// Generate one protocol-level streaming chunk with MiniCPM-o's
    /// 26/25 look-ahead contract.
    ///
    /// The first and final chunks allow EOS immediately.  A middle chunk
    /// masks EOS for all 26 steps, returning the 25-code prefix while hiding
    /// the newest look-ahead code, exactly as the upstream Python helper does.
    /// Token2Wav owns the exposed-code buffering; the semantic session still
    /// carries the hidden look-ahead code between protocol chunks so its KV
    /// history stays aligned with the official implementation.
    /// ``textStartPosition`` advances by condition length + committed count,
    /// matching the official ``text_start_pos`` bookkeeping.  A final chunk or
    /// sampled EOS is reported without clearing the session: the caller must
    /// append/flush ``committedTokens`` to Token2Wav first and then call
    /// ``session.finishTurn()``.  This ordering keeps the semantic KV alive
    /// until waveform synthesis has consumed the final codes.
    public func generateStreamingChunk(
        inputsEmbeds: MLXArray,
        session: MiniCPMTTSSession,
        endOfTurn: Bool = false,
        temperature: Float = 0.7,
        repetitionPenalty: Float = 1.0,
        eosToken: Int? = nil,
        forceNoStop: Bool = false,
        maxNewToken: Int = 26,
        resetAfterEnd: Bool = false,
        shouldCancel: (@Sendable () -> Bool)? = nil
    ) throws -> MiniCPMTTSStreamingChunkResult {
        if shouldCancel?() == true { throw MiniCPMTTSSemanticError.cancelled }
        guard maxNewToken > 0 else {
            throw MiniCPMTTSSemanticError.invalidInput(
                "streaming chunk must generate at least one token")
        }
        // A final chunk is deliberately not reset inside this method: the
        // caller must hand its committed codes to Token2Wav first and only
        // then call ``finishTurn()``. Reject a second chunk while that
        // hand-off is pending. Without this guard an interrupted duplex turn
        // could accidentally reuse the old pending code/KV and the next
        // sentence would start by replaying the previous audio token.
        if session.chunkIndex > 0, !session.turnActive {
            throw MiniCPMTTSSemanticError.invalidInput(
                "streaming turn already ended; flush Token2Wav and call finishTurn()")
        }

        let conditionTokenCount: Int
        if inputsEmbeds.ndim == 2 {
            conditionTokenCount = inputsEmbeds.dim(0)
        } else if inputsEmbeds.ndim == 3, inputsEmbeds.dim(0) == 1 {
            conditionTokenCount = inputsEmbeds.dim(1)
        } else {
            throw MiniCPMTTSSemanticError.invalidInput(
                "embeddings must be [T,H] or [1,T,H]")
        }

        let isFirst = session.chunkIndex == 0
        let chunkIndex = session.chunkIndex
        // Only a real turn boundary may cut a chunk short.
        //
        // The historical `isFirst` exemption ("allow decoding <1s audio") let
        // the first chunk of every turn stop after 5-13 codes instead of 25.
        // In the duplex path that short chunk then rendered as a whole second
        // of near-silence (measured RMS < 0.01 over the entire 1.00 s block,
        // e.g. "天气非" / "好的，" / "嗯，那"), so the unit's text was never
        // spoken at all - the most audible form of swallowed characters.
        // The explicit forceNoStop flag remains available for deterministic
        // warm-up/oracle tests and takes precedence over this EOS policy.
        let minNewTokens = endOfTurn ? 0 : maxNewToken
        let generation = try generateChunk(
            inputsEmbeds: inputsEmbeds,
            session: session,
            temperature: temperature,
            repetitionPenalty: repetitionPenalty,
            eosToken: eosToken,
            forceNoStop: forceNoStop,
            maxNewToken: maxNewToken,
            minNewTokens: minNewTokens,
            shouldCancel: shouldCancel)

        // ``generateChunk`` samples one look-ahead code beyond the 25 codes
        // committed to Token2Wav.  The official ``generate_chunk`` omits that
        // newest code from its returned tensor, but its KV history is
        // continued by feeding the code at the beginning of the next call.
        // Preserve that session state while hiding the code from the protocol
        // result and advancing text position by the committed prefix only.
        let streamingGeneration: MiniCPMTTSGenerationResult
        if generation.pendingToken == nil {
            streamingGeneration = generation
        } else {
            streamingGeneration = MiniCPMTTSGenerationResult(
                committedTokens: generation.committedTokens,
                pendingToken: nil,
                sampledTokenCount: generation.sampledTokenCount,
                cacheOffset: generation.cacheOffset,
                eosEmitted: generation.eosEmitted)
        }

        let textStart = session.textStartPosition
        let nextTextStart = textStart
            + conditionTokenCount
            + streamingGeneration.committedTokens.count
        let endedByEOS = generation.eosEmitted
        // Official MiniCPM-o keeps `endedByEOS` purely as information. Its
        // duplex loop (`modeling_minicpmo.py`, `_generate_chunk` /
        // `_generate_waveform_from_tokens`) tears the TTS session and Token2Wav
        // down **only** on `end_of_turn`; a decoder that happens to stop on its
        // own mid-turn is not a turn boundary there - it keeps `old_kv` and
        // merely advances `tts_text_start_pos`. Treating `endedByEOS` as final
        // closed the streaming turn early, so the following unit hit the
        // "streaming turn already ended" guard, and the teardown discarded the
        // vocoder's 160 ms crossfade overlap plus the 3 look-ahead codes -
        // audible as swallowed syllables at every such boundary.
        let finalChunk = endOfTurn
        // Keep the default false: duplex must flush token2wav before freeing
        // this session's KV.  Tests and simple callers may opt into an
        // immediate reset when no waveform stage is present.
        let shouldReset = resetAfterEnd && finalChunk
        session.recordStreamingProgress(
            textStartPosition: nextTextStart,
            chunkIndex: chunkIndex + 1,
            turnActive: !finalChunk)

        let result = MiniCPMTTSStreamingChunkResult(
            chunkIndex: chunkIndex,
            isFirstChunk: isFirst,
            isFinalChunk: finalChunk,
            endedByEOS: endedByEOS,
            textStartPosition: textStart,
            nextTextStartPosition: nextTextStart,
            conditionTokenCount: conditionTokenCount,
            generation: streamingGeneration,
            sessionWasReset: shouldReset)
        if shouldReset {
            // Opt-in only; the normal duplex call sequence invokes
            // session.finishTurn() after token2wav has consumed the result.
            session.finishTurn()
        }
        return result
    }

    private func applyRepetitionPenalty(
        _ logits: MLXArray,
        generated: [Int],
        penalty: Float,
        window: Int = 16
    ) -> MLXArray {
        guard penalty > 0 else { return logits }
        let recent = Array(generated.suffix(window))
        guard !recent.isEmpty else { return logits }
        let ids = MLXArray(recent.map(Int32.init)).expandedDimensions(axis: 1)
        let vocab = MLXArray(0..<Int32(configuration.numAudioTokens)).expandedDimensions(axis: 0)
        let frequencies = (ids .== vocab).asType(.float32).sum(axis: 0)
        let alpha = MLX.exp(MLX.log(MLXArray(penalty)) * frequencies)
        return MLX.where(logits .< MLXArray(Float(0)), logits * alpha, logits / alpha)
    }

    private func sample(logits: MLXArray, temperature: Float) -> Int {
        let selected: MLXArray
        if temperature <= 0 {
            selected = argMax(logits, axis: 0)
        } else {
            let scaled = logits / MLXArray(temperature)
            let gumbel = MLXRandom.gumbel(scaled.shape)
            selected = argMax(scaled + gumbel, axis: 0)
        }
        return selected.item(Int.self)
    }
}
