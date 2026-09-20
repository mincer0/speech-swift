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
    /// in a duplex turn), but are excluded from repetition history.
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
    /// Penalty applied to protocol terminators after the natural `chunk_eos`
    /// probe.  MiniCPM's official chunk generator uses this to make a higher
    /// value mean "keep generating a little longer".  It is intentionally
    /// separate from `repetitionPenalty` because it only touches terminators.
    public var lengthPenalty: Float
    public var listenTopK: Int?
    public var listenProbabilityScale: Float
    /// Hold back the official natural `chunk_eos` probe. The probe returns the
    /// raw (unmasked) argmax, so it can end a duplex unit after a single visible
    /// character even though the audio side is pinned at 25 codes / 1.00 s per
    /// unit. Callers set this while a unit has not yet collected enough visible
    /// text, which keeps the text and audio budgets aligned.
    public var suppressNaturalChunkEOS: Bool

    public enum Mode: Sendable, Equatable { case sampling, greedy }

    public init(
        mode: Mode = .sampling,
        temperature: Float = 0.7,
        topK: Int = 20,
        topP: Float = 0.8,
        repetitionPenalty: Float = 1.05,
        repetitionWindowSize: Int = 512,
        lengthPenalty: Float = 1.1,
        listenTopK: Int? = nil,
        listenProbabilityScale: Float = 1.0,
        suppressNaturalChunkEOS: Bool = false
    ) {
        self.mode = mode
        self.temperature = temperature
        self.topK = topK
        self.topP = topP
        self.repetitionPenalty = repetitionPenalty
        self.repetitionWindowSize = repetitionWindowSize
        self.lengthPenalty = lengthPenalty
        self.listenTopK = listenTopK
        self.listenProbabilityScale = listenProbabilityScale
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
    /// `chunk_eos` probe, and a second one for the filtered sample when the
    /// probe did not terminate the chunk.
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
        if natural.item(Int.self) == `protocol`.chunkEOS { return natural }

        let finalDraw = config.mode == .greedy ? 0 : state.nextUniform()
        return sampleAfterNaturalProbe(
            row: row,
            config: config,
            protocol: `protocol`,
            forbidden: forbidden,
            previousTokens: previousTokens,
            uniform: finalDraw)
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
        // The natural `chunk_eos` probe is the official decoder's early-exit
        // path and ignores both the forbidden list and the character bounds.
        // A duplex unit may therefore end after one or two visible characters
        // even though the token2wav stage is pinned at 25 codes (1.00 s) per
        // unit - the surplus renders as silence inside the block, which is
        // audible as choppy, broken-up speech. `suppressNaturalChunkEOS` lets
        // the caller hold the probe back until the unit has collected enough
        // visible text; the masked categorical sample below still applies the
        // normal protocol policy.
        if !config.suppressNaturalChunkEOS, natural.item(Int.self) == `protocol`.chunkEOS {
            return natural
        }

        let finalDraw = uniforms.count > 1 ? uniforms[1] : naturalDraw
        return sampleAfterNaturalProbe(
            row: row,
            config: config,
            protocol: `protocol`,
            forbidden: forbidden,
            previousTokens: previousTokens,
            uniform: finalDraw)
    }

    private static func sampleAfterNaturalProbe(
        row: MLXArray,
        config: MiniCPMSamplingConfig,
        `protocol`: MiniCPMProtocolTokenIds,
        forbidden: [Int],
        previousTokens: [Int],
        uniform: Float
    ) -> MLXArray {

        var scores = row
        let blocked = forbidden.filter { $0 >= 0 && $0 < scores.dim(0) }
        if !blocked.isEmpty {
            scores[MLXArray(blocked.map(Int32.init))] = MLXArray(-Float.greatestFiniteMagnitude)
        }

        let history = previousTokens.suffix(max(config.repetitionWindowSize, 0))
        // Protocol markers are bookkeeping, not text.  Applying repetition
        // pressure to them makes a previous `<|listen|>` or `<|turn_eos|>`
        // alter the next conversational decision and is unlike the official
        // decoder, which keeps special tokens in a separate history.
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
            $0 >= 0 && $0 < scores.dim(0) && !protocolSpecials.contains($0)
        }
        if config.repetitionPenalty != 1, !unique.isEmpty {
            let ids = MLXArray(unique.map(Int32.init))
            let values = scores[ids]
            let penalty = config.repetitionPenalty
            // Match HuggingFace `RepetitionPenaltyLogitsProcessor`: a token that
            // already appeared has to be pushed *away* from zero, on whichever
            // side of zero it sits.
            //     logit <  0  -> multiply  (further negative)
            //     logit >= 0  -> divide    (closer to zero)
            // The previous unconditional `values / penalty` moved a negative
            // logit *toward* zero, i.e. it rewarded repetition of exactly the
            // tokens the model had just produced. With ~150k negative logits in
            // the tail, that lifted junk pieces - and the character just
            // emitted - into the top-k/top-p window. That is the source of the
            // duplicated characters ("苹果果", "准备备着") and stray tokens ("_").
            let penalized = MLX.where(values .< 0, values * penalty, values / penalty)
            // Keep the non-finite mask sentinels untouched: scaling them can
            // turn a finite mask into NaN on some MLX backends.
            scores[ids] = MLX.where(isFinite(values), penalized, values)
        }

        // The official decoder applies the length penalty only to
        // ``turn_eos`` after the natural ``chunk_eos`` probe.  In particular,
        // ``chunk_tts_eos`` is a distinct chunk boundary and must retain its
        // model logit.  Keep masked values untouched: multiplying a
        // non-finite sentinel can otherwise turn a finite mask into NaN on
        // some MLX backends.
        if config.lengthPenalty != 1,
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
            // distribution invalid; the official decoder skips non-finite
            // values for the same reason.
            if isFinite(current).item(Bool.self) {
                scores[`protocol`.listen] = current * config.listenProbabilityScale
            }
        }
        if let listenTopK = config.listenTopK,
           listenTopK > 0,
           `protocol`.listen >= 0,
           `protocol`.listen < scores.dim(0) {
            let rank = (scores .> scores[`protocol`.listen]).asType(.int32).sum().item(Int.self)
            if rank < listenTopK {
                return MLXArray(Int32(`protocol`.listen))
            }
        }

        return choose(
            scores,
            mode: config.mode,
            temperature: max(config.temperature, 0.000001),
            topK: config.topK,
            topP: config.topP,
            uniform: uniform)
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
        var order = argSort(scores, axis: 0)[.stride(by: -1)]
        if k < vocab { order = order[0 ..< k] }
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
