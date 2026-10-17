import Foundation
import MLX

public struct MiniCPMProtocolTokenIds: Sendable, Equatable {
    public let listen: Int
    public let chunkEOS: Int
    public let chunkTTSEOS: Int
    public let turnEOS: Int
    public let speak: Int
    public let unitEnd: Int
    public let ttsBOS: Int?
    /// Additional protocol markers known by the tokenizer.  They are not
    /// automatically forbidden as output (some markers are valid decisions
    /// in a duplex turn). Chat excludes these from repetition history;
    /// officialDuplex follows the caller's complete generated history.
    public let allSpecialTokenIds: Set<Int>

    public init(
        listen: Int,
        chunkEOS: Int,
        chunkTTSEOS: Int,
        turnEOS: Int,
        speak: Int,
        unitEnd: Int,
        ttsBOS: Int? = nil,
        allSpecialTokenIds: Set<Int> = []
    ) {
        self.listen = listen
        self.chunkEOS = chunkEOS
        self.chunkTTSEOS = chunkTTSEOS
        self.turnEOS = turnEOS
        self.speak = speak
        self.unitEnd = unitEnd
        self.ttsBOS = ttsBOS
        self.allSpecialTokenIds = allSpecialTokenIds
    }
}

/// One raw-logit entry used by the opt-in duplex sampler diagnostic.  Ranks
/// are one-based and use a deterministic token-ID tie break after sorting by
/// descending raw logit.
public struct MiniCPMLogitTraceEntry: Sendable, Equatable {
    public let tokenID: Int
    public let logit: Float
    public let rank: Int

    public init(tokenID: Int, logit: Float, rank: Int) {
        self.tokenID = tokenID
        self.logit = logit
        self.rank = rank
    }
}

/// Pure, model-free snapshot of the raw logits consumed by one sampler step.
/// The optional protocol entries are nil only when a protocol ID is outside
/// the supplied vocabulary; real MiniCPM token tables should always provide
/// all six markers.
public struct MiniCPMLogitTrace: Sendable, Equatable {
    public let topK: [MiniCPMLogitTraceEntry]
    public let listen: MiniCPMLogitTraceEntry?
    public let chunkEOS: MiniCPMLogitTraceEntry?
    public let chunkTTSEOS: MiniCPMLogitTraceEntry?
    public let turnEOS: MiniCPMLogitTraceEntry?
    public let speak: MiniCPMLogitTraceEntry?
    public let ttsBOS: MiniCPMLogitTraceEntry?
    public let sampledToken: Int

    public init(
        topK: [MiniCPMLogitTraceEntry],
        listen: MiniCPMLogitTraceEntry?,
        chunkEOS: MiniCPMLogitTraceEntry?,
        chunkTTSEOS: MiniCPMLogitTraceEntry?,
        turnEOS: MiniCPMLogitTraceEntry?,
        speak: MiniCPMLogitTraceEntry?,
        ttsBOS: MiniCPMLogitTraceEntry?,
        sampledToken: Int
    ) {
        self.topK = topK
        self.listen = listen
        self.chunkEOS = chunkEOS
        self.chunkTTSEOS = chunkTTSEOS
        self.turnEOS = turnEOS
        self.speak = speak
        self.ttsBOS = ttsBOS
        self.sampledToken = sampledToken
    }
}

public struct MiniCPMSamplingConfig: Sendable, Equatable {
    public var mode: Mode
    public var temperature: Float
    public var topK: Int
    public var topP: Float
    public var repetitionPenalty: Float
    public var repetitionWindowSize: Int
    /// Chat uses Hugging Face's sign-aware text penalty. Duplex uses the
    /// checkpoint's StreamDecoder rule, including protocol tokens in history.
    public var repetitionPolicy: RepetitionPolicy
    /// Chat/pinned-runtime turn_eos shaping after the natural chunk_eos probe.
    /// The checkpoint's officialDuplex policy has no length penalty and
    /// ignores this value.
    public var lengthPenalty: Float
    public var listenTopK: Int?
    public var listenProbabilityScale: Float
    /// Optional multiplicative adjustment in probability space for the
    /// protocol listen token. This is kept separate from
    /// ``listenProbabilityScale`` because multiplying a negative logit by a
    /// value below one increases it. A value of 0 forbids listen; values in
    /// (0, 1) add ``log(value)`` to the listen logit.
    public var listenLogProbabilityScale: Float
    /// Optionally skip the natural chunk_eos early return. The final sample
    /// still obeys the caller's forbidden masks. False follows the checkpoint's
    /// natural-probe behavior without imposing a minimum text length.
    public var suppressNaturalChunkEOS: Bool

    public enum Mode: Sendable, Equatable { case sampling, greedy }
    public enum RepetitionPolicy: String, Sendable, Equatable {
        case huggingFace = "hugging_face"
        case officialDuplex = "official_duplex"
    }

    public init(
        mode: Mode = .sampling,
        temperature: Float = 0.7,
        topK: Int = 20,
        topP: Float = 0.8,
        repetitionPenalty: Float = 1.05,
        repetitionWindowSize: Int = 512,
        repetitionPolicy: RepetitionPolicy = .huggingFace,
        lengthPenalty: Float = 1.1,
        listenTopK: Int? = nil,
        listenProbabilityScale: Float = 1.0,
        suppressNaturalChunkEOS: Bool = false,
        listenLogProbabilityScale: Float = 1.0
    ) {
        self.mode = mode
        self.temperature = temperature
        self.topK = topK
        self.topP = topP
        self.repetitionPenalty = repetitionPenalty
        self.repetitionWindowSize = repetitionWindowSize
        self.repetitionPolicy = repetitionPolicy
        self.lengthPenalty = lengthPenalty
        self.listenTopK = listenTopK
        self.listenProbabilityScale = listenProbabilityScale
        self.listenLogProbabilityScale = max(0, listenLogProbabilityScale)
        self.suppressNaturalChunkEOS = suppressNaturalChunkEOS
    }
}

/// The random state used by a MiniCPM decode session.
///
/// MLX's categorical operators do not expose a per-session generator.  A
/// small host-side SplitMix64 generator gives each duplex session an
/// independent stream while keeping the state value-semantic: copying it is a
/// cheap snapshot, and restoring it makes a speculative/rolled-back decode
/// consume exactly the same draws as the original path.
public struct MiniCPMSamplerState: Sendable, Equatable {
    private var state: UInt64
    /// The official decoder records filtered samples, including protocol
    /// markers. Natural chunkEOS and listenTopK early returns are unrecorded.
    public private(set) var sampledTokens: [Int] = []

    /// Clear session history while preserving the random generator position.
    public mutating func clearTokenHistory() {
        sampledTokens.removeAll(keepingCapacity: true)
    }

    fileprivate mutating func recordSampledToken(_ token: Int) {
        sampledTokens.append(token)
    }

    /// Create a state from a caller supplied seed.  With no seed, the seed is
    /// drawn from the system generator, so production sessions never all use
    /// the same quantile (the old implementation's implicit `0.5`).
    public init(seed: UInt64? = nil) {
        if let seed {
            state = seed
        } else {
            var generator = SystemRandomNumberGenerator()
            state = generator.next()
        }
    }

    /// A readable state value is useful for diagnostics and persistence.  The
    /// PRNG remains encapsulated so callers cannot accidentally skip the
    /// SplitMix mixing step when advancing it.
    public var rawState: UInt64 { state }

    /// Return a value in `[0, 1)` and advance this session's stream.
    public mutating func nextUniform() -> Float {
        let bits = nextUInt64() >> 40       // 24 random bits, exactly representable as Float
        return Float(bits) / 16_777_216.0
    }

    /// Value-semantic snapshot/restore helpers make the intended rollback API
    /// explicit without exposing the mutable storage.
    public func snapshot() -> MiniCPMSamplerState { self }

    public mutating func restore(_ snapshot: MiniCPMSamplerState) {
        self = snapshot
    }

    private mutating func nextUInt64() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var value = state
        value = (value ^ (value >> 30)) &* 0xBF58_476D_1CE4_E5B9
        value = (value ^ (value >> 27)) &* 0x94D0_49BB_1331_11EB
        return value ^ (value >> 31)
    }
}

/// MiniCPM's protocol-aware sampler.  The natural ``chunk_eos`` probe is
/// deliberately performed before suppression, repetition, listen shaping,
/// top-k and top-p, matching the official StreamDecoder ordering.
public enum MiniCPMSampler {
    /// Parse the strict opt-in environment gate.  Values outside 1...20,
    /// including empty/whitespace/non-decimal strings, disable diagnostics.
    public static func logitTraceTopK(from text: String?) -> Int? {
        guard let text,
              !text.isEmpty,
              text.unicodeScalars.allSatisfy({ CharacterSet.decimalDigits.contains($0) }),
              let value = Int(text),
              (1 ... 20).contains(value) else {
            return nil
        }
        return value
    }

    /// Read the diagnostic gate once per generation.  Keeping the environment
    /// lookup outside ``sample`` means the ordinary sampler has no extra work
    /// or output when the gate is absent.
    public static func logitTraceTopK(
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> Int? {
        logitTraceTopK(from: environment["MINICPM_DUPLEX_LOGIT_TRACE_TOPK"])
    }

    /// Compute deterministic raw-logit top-k and protocol-token ranks without
    /// touching MLX or any random state.  Ties are ordered by token ID so the
    /// JSONL output is stable across runs/backends.
    public static func logitTrace(
        logits: [Float],
        `protocol`: MiniCPMProtocolTokenIds,
        topK: Int,
        sampledToken: Int
    ) -> MiniCPMLogitTrace {
        let order = logits.indices.sorted { lhs, rhs in
            let left = logits[lhs]
            let right = logits[rhs]
            if left.isNaN || right.isNaN {
                if left.isNaN != right.isNaN { return !left.isNaN }
                return lhs < rhs
            }
            if left == right { return lhs < rhs }
            return left > right
        }
        let limit = min(max(topK, 0), order.count)
        let entries = order.prefix(limit).enumerated().map { offset, index in
            MiniCPMLogitTraceEntry(
                tokenID: index, logit: logits[index], rank: offset + 1)
        }
        func entry(for tokenID: Int) -> MiniCPMLogitTraceEntry? {
            guard tokenID >= 0, tokenID < logits.count,
                  let rank = order.firstIndex(of: tokenID) else {
                return nil
            }
            return MiniCPMLogitTraceEntry(
                tokenID: tokenID, logit: logits[tokenID], rank: rank + 1)
        }
        return MiniCPMLogitTrace(
            topK: entries,
            listen: entry(for: `protocol`.listen),
            chunkEOS: entry(for: `protocol`.chunkEOS),
            chunkTTSEOS: entry(for: `protocol`.chunkTTSEOS),
            turnEOS: entry(for: `protocol`.turnEOS),
            speak: entry(for: `protocol`.speak),
            ttsBOS: `protocol`.ttsBOS.flatMap(entry),
            sampledToken: sampledToken)
    }

    /// MLX bridge used only by the gated duplex diagnostic.  The conversion
    /// is deliberately outside the ordinary sampler path so no synchronization
    /// is added when diagnostics are disabled.
    public static func logitTrace(
        logits: MLXArray,
        `protocol`: MiniCPMProtocolTokenIds,
        topK: Int,
        sampledToken: Int
    ) -> MiniCPMLogitTrace {
        let row = lastRow(logits).asType(.float32)
        eval(row)
        return logitTrace(
            logits: row.asArray(Float.self),
            protocol: `protocol`,
            topK: topK,
            sampledToken: sampledToken)
    }

    /// Sample using the per-session generator.  This is the production entry
    /// point; unlike the legacy fixed-quantile helper below, it consumes the
    /// same number of draws as the official decoder: one for the natural
    /// `chunk_eos` probe, and a second one for a filtered sample unless the
    /// natural probe or listenTopK rule returned early.
    public static func sample(
        logits: MLXArray,
        config: MiniCPMSamplingConfig,
        `protocol`: MiniCPMProtocolTokenIds,
        forbidden: [Int] = [],
        previousTokens: [Int] = [],
        state: inout MiniCPMSamplerState
    ) -> MLXArray {
        let row = lastRow(logits).asType(.float32)
        let naturalDraw = config.mode == .greedy ? 0 : state.nextUniform()
        let natural = choose(
            row, mode: config.mode, temperature: 1, topK: 0, topP: 1,
            uniform: naturalDraw)
        if shouldReturnNaturalChunkEOS(natural, config: config, protocol: `protocol`) {
            return natural
        }

        let result = sampleAfterNaturalProbe(
            row: row,
            config: config,
            protocol: `protocol`,
            forbidden: forbidden,
            previousTokens: previousTokens,
            uniform: { config.mode == .greedy ? 0 : state.nextUniform() })
        if result.recordHistory { state.recordSampledToken(result.token.item(Int.self)) }
        return result.token
    }

    /// Sample with a caller-provided draw.  Supplying a draw is useful for
    /// parity tests and deterministic replay; production code should pass a
    /// `MiniCPMSamplerState` instead.
    public static func sample(
        logits: MLXArray,
        config: MiniCPMSamplingConfig,
        `protocol`: MiniCPMProtocolTokenIds,
        forbidden: [Int] = [],
        previousTokens: [Int] = [],
        uniform: Float? = nil
    ) -> MLXArray {
        return sample(
            logits: logits,
            config: config,
            protocol: `protocol`,
            forbidden: forbidden,
            previousTokens: previousTokens,
            uniforms: uniform.map { [$0] } ?? [])
    }

    /// Deterministic compatibility entry point accepting the two official
    /// draws explicitly: ``uniforms[0]`` probes the natural ``chunk_eos``
    /// decision and ``uniforms[1]`` drives the filtered categorical sample.
    /// A one-element array intentionally reuses the first draw for backwards
    /// compatibility with callers of the old ``uniform:`` overload.
    public static func sample(
        logits: MLXArray,
        config: MiniCPMSamplingConfig,
        `protocol`: MiniCPMProtocolTokenIds,
        forbidden: [Int] = [],
        previousTokens: [Int] = [],
        uniforms: [Float]
    ) -> MLXArray {
        let row = lastRow(logits).asType(.float32)
        let naturalDraw = uniforms.first ?? Float.random(in: 0 ..< 1)
        let natural = choose(
            row,
            mode: config.mode,
            temperature: 1,
            topK: 0,
            topP: 1,
            uniform: naturalDraw)
        // The checkpoint's natural probe runs before the forbidden masks.
        // A caller can explicitly suppress this early return; the final
        // categorical sample still applies the caller's masks and policy.
        if shouldReturnNaturalChunkEOS(natural, config: config, protocol: `protocol`) {
            return natural
        }

        let finalDraw = uniforms.count > 1 ? uniforms[1] : naturalDraw
        return sampleAfterNaturalProbe(
            row: row,
            config: config,
            protocol: `protocol`,
            forbidden: forbidden,
            previousTokens: previousTokens,
            uniform: { finalDraw }).token
    }

    private static func shouldReturnNaturalChunkEOS(
        _ token: MLXArray,
        config: MiniCPMSamplingConfig,
        `protocol`: MiniCPMProtocolTokenIds
    ) -> Bool {
        !config.suppressNaturalChunkEOS && token.item(Int.self) == `protocol`.chunkEOS
    }

    private static func sampleAfterNaturalProbe(
        row: MLXArray,
        config: MiniCPMSamplingConfig,
        `protocol`: MiniCPMProtocolTokenIds,
        forbidden: [Int],
        previousTokens: [Int],
        uniform: () -> Float
    ) -> (token: MLXArray, recordHistory: Bool) {

        // MLXArray is a reference type, and asType returns self when the dtype
        // already matches. A distinct handle keeps indexed score updates from
        // changing caller-owned logits or a later natural probe/diagnostic.
        var scores = row.reshaped(row.shape)
        let blocked = forbidden.filter { $0 >= 0 && $0 < scores.dim(0) }
        if !blocked.isEmpty {
            scores[MLXArray(blocked.map(Int32.init))] = MLXArray(-Float.infinity)
        }

        // Python's generated_tokens[-0:] means the complete history. Preserve
        // that checkpoint convention only for officialDuplex; chat uses zero
        // to disable its repetition window.
        let history = config.repetitionPolicy == .officialDuplex
            && config.repetitionWindowSize == 0
            ? previousTokens[...]
            : previousTokens.suffix(max(config.repetitionWindowSize, 0))
        // Chat excludes protocol markers. The official duplex decoder starts
        // with special_token_ids=[], so all supplied generated history counts.
        var protocolSpecials: Set<Int> = [
            `protocol`.listen,
            `protocol`.chunkEOS,
            `protocol`.chunkTTSEOS,
            `protocol`.turnEOS,
            `protocol`.speak,
            `protocol`.unitEnd,
        ]
        if let ttsBOS = `protocol`.ttsBOS { protocolSpecials.insert(ttsBOS) }
        protocolSpecials.formUnion(`protocol`.allSpecialTokenIds)
        let unique = Set(history).filter {
            $0 >= 0 && $0 < scores.dim(0)
                && (config.repetitionPolicy == .officialDuplex || !protocolSpecials.contains($0))
        }
        if config.repetitionPenalty != 1, !unique.isEmpty {
            let ids = MLXArray(unique.map(Int32.init))
            let values = scores[ids]
            let penalty = config.repetitionPenalty
            let penalized: MLXArray
            switch config.repetitionPolicy {
            case .huggingFace:
                penalized = MLX.where(values .< 0, values * penalty, values / penalty)
            case .officialDuplex:
                // StreamDecoder applies this rule regardless of logit sign.
                penalized = penalty > 1 ? values / penalty : values * (1 / penalty)
            }
            // Forbidden sentinels must stay masked when encouraging repetition.
            scores[ids] = MLX.where(isFinite(values), penalized, values)
        }

        // The chat/pinned runtime policy shapes turn_eos. The checkpoint's
        // MPS StreamDecoder has no length penalty, so officialDuplex skips it.
        // Keep masked values untouched when applying the chat policy.
        if config.repetitionPolicy == .huggingFace,
           config.lengthPenalty != 1,
           config.lengthPenalty.isFinite,
           config.lengthPenalty > 0 {
            let token = `protocol`.turnEOS
            if token >= 0, token < scores.dim(0) {
                let current = scores[token]
                if isFinite(current).item(Bool.self) {
                    scores[token] = (current .> 0).item(Bool.self)
                        ? current / config.lengthPenalty
                        : current * config.lengthPenalty
                }
            }
        }
        if config.listenProbabilityScale != 1,
           `protocol`.listen >= 0,
           `protocol`.listen < scores.dim(0) {
            let current = scores[`protocol`.listen]
            // Keep a masked listen logit masked.  Multiplying `-inf` by zero
            // would otherwise create NaN and make the entire categorical
            // distribution invalid.
            if isFinite(current).item(Bool.self) {
                scores[`protocol`.listen] = current * config.listenProbabilityScale
            }
        }
        if config.listenLogProbabilityScale != 1,
           `protocol`.listen >= 0,
           `protocol`.listen < scores.dim(0) {
            let current = scores[`protocol`.listen]
            if isFinite(current).item(Bool.self) {
                scores[`protocol`.listen] = config.listenLogProbabilityScale > 0
                    ? current + Foundation.log(config.listenLogProbabilityScale)
                    : MLXArray(-Float.infinity)
            }
        }
        if let listenTopK = config.listenTopK,
           listenTopK > 0,
           `protocol`.listen >= 0,
           `protocol`.listen < scores.dim(0) {
            let rank = (scores .> scores[`protocol`.listen]).asType(.int32).sum().item(Int.self)
            if rank < listenTopK {
                return (MLXArray(Int32(`protocol`.listen)), false)
            }
        }

        return (choose(
            scores,
            mode: config.mode,
            temperature: max(config.temperature, 0.000001),
            topK: config.topK,
            topP: config.topP,
            uniform: uniform()), true)
    }

    private static func lastRow(_ logits: MLXArray) -> MLXArray {
        if logits.ndim == 3 { return logits[0, logits.dim(1) - 1] }
        if logits.ndim == 2 { return logits[logits.dim(0) - 1] }
        return logits
    }

    private static func choose(
        _ raw: MLXArray,
        mode: MiniCPMSamplingConfig.Mode,
        temperature: Float,
        topK: Int,
        topP: Float,
        uniform: Float
    ) -> MLXArray {
        if mode == .greedy { return argMax(raw, axis: 0) }
        var scores = raw / temperature
        let vocab = scores.dim(0)
        let k = topK > 0 && topK < vocab ? topK : vocab
        let order = argSort(scores, axis: 0)[.stride(by: -1)]
        if k < vocab {
            // torch.topk defines a score threshold; every token tied with the
            // kth score survives, even when that leaves more than k tokens.
            let threshold = take(scores, order[k - 1], axis: 0)
            scores = which(scores .< threshold, MLXArray(-Float.infinity), scores)
        }
        let ranked = take(scores, order, axis: 0)
        let probabilities = softmax(ranked, axis: 0, precise: true)
        let kept: MLXArray
        if topP > 0 && topP < 1 {
            kept = which(
                // Official top-p filtering keeps the first token whose
                // cumulative mass reaches/exceeds the threshold.  The
                // exclusive prefix comparison is therefore `<=`, including
                // the boundary token when the preceding mass is exactly
                // `topP`.
                cumsum(probabilities, axis: 0, inclusive: false) .<= topP,
                probabilities,
                MLXArray(Float(0)))
        } else {
            kept = probabilities
        }
        let mass = kept.sum()
        let target = which(mass .> 0, mass, MLXArray(Float(1))) * min(max(uniform, 0), 0.999999)
        let rank = (cumsum(kept, axis: 0, inclusive: true) .< target)
            .asType(.int32).sum()
        let safeRank = minimum(rank, MLXArray(Int32(order.dim(0) - 1)))
        return take(order, safeRank, axis: 0)
    }
}
