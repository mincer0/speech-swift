import Foundation
import MiniCPMAudio
import MiniCPMLLM

/// The protocol subset needed by the duplex state machine. Keeping a public
/// initializer here lets deterministic tests use synthetic IDs without
/// exposing MiniCPMLLM's tokenizer-internal token table constructor.
public struct MiniCPMDuplexProtocolTokenIDs: Sendable, Equatable {
    public let eos: Int
    public let unit: Int
    public let unitEnd: Int
    public let listen: Int
    public let speak: Int
    public let ttsBOS: Int
    public let chunkEOS: Int
    public let chunkTTSEOS: Int
    public let turnEOS: Int

    public init(
        eos: Int,
        unit: Int = -1,
        unitEnd: Int = -1,
        listen: Int,
        speak: Int,
        ttsBOS: Int,
        chunkEOS: Int,
        chunkTTSEOS: Int,
        turnEOS: Int
    ) {
        self.eos = eos
        self.unit = unit
        self.unitEnd = unitEnd
        self.listen = listen
        self.speak = speak
        self.ttsBOS = ttsBOS
        self.chunkEOS = chunkEOS
        self.chunkTTSEOS = chunkTTSEOS
        self.turnEOS = turnEOS
    }

    public init(_ ids: MiniCPMSpecialTokenIDs) {
        self.init(
            eos: ids.eos,
            unit: ids.unit,
            unitEnd: ids.unitEnd,
            listen: ids.listen,
            speak: ids.speak,
            ttsBOS: ids.ttsBOS,
            chunkEOS: ids.chunkEOS,
            chunkTTSEOS: ids.chunkTTSEOS,
            turnEOS: ids.turnEOS)
    }
}

/// A small, framework-free representation of one MiniCPM-o streaming
/// generation step.  Keeping this state machine separate from MLX makes the
/// protocol auditable without loading the 8-bit language model and prevents
/// the control markers from becoming accidental model-specific behavior.
public struct MiniCPMDuplexProtocolStep: Sendable, Equatable {
    public let index: Int
    public let token: Int
    public let shouldFeed: Bool
    public let shouldStop: Bool
    public let isListen: Bool
    public let endOfTurn: Bool
    public let collectForTTS: Bool
    public let forced: Bool
    public let coercedListen: Bool

    public init(
        index: Int,
        token: Int,
        shouldFeed: Bool,
        shouldStop: Bool,
        isListen: Bool,
        endOfTurn: Bool,
        collectForTTS: Bool,
        forced: Bool = false,
        coercedListen: Bool = false
    ) {
        self.index = index
        self.token = token
        self.shouldFeed = shouldFeed
        self.shouldStop = shouldStop
        self.isListen = isListen
        self.endOfTurn = endOfTurn
        self.collectForTTS = collectForTTS
        self.forced = forced
        self.coercedListen = coercedListen
    }
}

/// Stable trace item used by deterministic protocol tests and production
/// diagnostics.  No MLX array or model object is retained here.
public struct MiniCPMDuplexProtocolTraceEvent: Sendable, Equatable {
    public let name: String
    public let index: Int?
    public let token: Int?

    public init(name: String, index: Int? = nil, token: Int? = nil) {
        self.name = name
        self.index = index
        self.token = token
    }
}

public struct MiniCPMDuplexProtocolSnapshot: Sendable, Equatable {
    public let currentTurnEnded: Bool
    public let totalIDs: [Int]
    public let responseIDs: [Int]
    public let generatedIDs: [Int]
    public let generateCount: Int
    public let unitIndex: Int
    public let committedUnitTerminator: Int?
}

/// The non-neural part of MiniCPM-o's TDM (turn decision model) protocol.
///
/// The official decoder's important invariants are represented explicitly:
///
/// * ``turn_eos`` is a normal fed token and does not stop the current chunk;
/// * chunk terminators are deferred and fed together with ``</unit>`` during
///   finalize, while the final loop slot is still reserved for explicit
///   ``chunk_eos``;
/// * the first generated token is not sent to semantic TTS;
/// * ``listen`` is forbidden while a spoken turn is open, and is coerced to
///   ``tts_bos`` unless a caller deliberately forces a listen;
/// * accepting a generated unit and rolling it back are separate operations.
public struct MiniCPMDuplexProtocolState: Sendable {
    public private(set) var currentTurnEnded = true
    public private(set) var totalIDs: [Int] = []
    public private(set) var responseIDs: [Int] = []
    public private(set) var generatedIDs: [Int] = []
    public private(set) var generateCount = 0
    public private(set) var unitIndex = 0
    public private(set) var trace: [MiniCPMDuplexProtocolTraceEvent] = []

    private var pendingSnapshot: MiniCPMDuplexProtocolSnapshot?
    private var forceListen = false
    /// Terminator deferred by the currently generated unit.  The upstream
    /// finalizer feeds this marker together with `</unit>`; retaining it here
    /// lets an interrupt replace both markers in the protocol history when a
    /// provisional spoken unit is rolled back.
    private var pendingUnitTerminator: Int?
    /// Terminator used by the last committed unit.  This is separate from
    /// `totalIDs.last` because the latter is always `</unit>` after commit.
    private var committedUnitTerminator: Int?

    public init() {}

    public mutating func reset() {
        currentTurnEnded = true
        totalIDs.removeAll(keepingCapacity: true)
        responseIDs.removeAll(keepingCapacity: true)
        generatedIDs.removeAll(keepingCapacity: true)
        generateCount = 0
        unitIndex = 0
        trace.removeAll(keepingCapacity: true)
        pendingSnapshot = nil
        forceListen = false
        pendingUnitTerminator = nil
        committedUnitTerminator = nil
    }

    public mutating func beginUnit() -> MiniCPMDuplexProtocolSnapshot {
        let snapshot = snapshot()
        pendingSnapshot = snapshot
        generatedIDs.removeAll(keepingCapacity: true)
        pendingUnitTerminator = nil
        trace.append(.init(name: "unit_start", index: unitIndex))
        return snapshot
    }

    /// Begin one `generate` call.  `forceListenOverride` affects this call
    /// only; it never mutates the configured force-listen counter.
    public mutating func beginGeneration(
        config: MiniCPMDuplexConfig,
        forceListenOverride: Bool = false
    ) -> Bool {
        forceListen = forceListenOverride
            || generateCount < max(0, config.forceListenCount)
        generateCount += 1
        trace.append(.init(name: forceListen ? "force_listen" : "generate", index: generateCount - 1))
        return forceListen
    }

    /// Consume one sampled token.  The caller must feed the returned token
    /// through the LLM exactly when ``shouldFeed`` is true.
    public mutating func consume(
        sampledToken: Int,
        index: Int,
        tokens: MiniCPMDuplexProtocolTokenIDs,
        config: MiniCPMDuplexConfig,
        forced: Bool = false,
        allowListen: Bool = true
    ) -> MiniCPMDuplexProtocolStep {
        let originallySampled = sampledToken
        var token = forceListen ? tokens.listen : sampledToken
        var coerced = false

        // EOS from the language head is an alias for the protocol turn EOS.
        // Keeping one canonical marker makes the following iteration able to
        // continue until chunk_eos, as in the official decoder.
        if token == tokens.eos {
            token = tokens.turnEOS
        }

        // A model listen while an answer is open must not yield a second
        // listen boundary.  Explicit force-listen is the sole exception.
        if !forceListen && token == tokens.listen && (!currentTurnEnded || !allowListen) {
            token = tokens.ttsBOS
            coerced = true
            trace.append(.init(name: "coerce_listen", index: index, token: originallySampled))
        }

        totalIDs.append(token)
        let isListen = token == tokens.listen
        let isChunkTerminator = token == tokens.listen
            || token == tokens.chunkEOS
            || token == tokens.chunkTTSEOS

        if isChunkTerminator {
            // MiniCPM's unified duplex decoder defers chunk terminators until
            // finalize_unit(), where it feeds `[terminator, </unit>]` in one
            // forward pass. This both matches the official KV sequence and
            // avoids an unnecessary logits evaluation for a token that cannot
            // affect the returned TTS chunk.
            let shouldFeed = false
            pendingUnitTerminator = token
            trace.append(.init(
                name: isListen ? "listen" : "chunk_terminator",
                index: index,
                token: token))
            if forced { trace.append(.init(name: "chunk_eos_forced", index: index, token: token)) }
            forceListen = false
            return MiniCPMDuplexProtocolStep(
                index: index,
                token: token,
                shouldFeed: shouldFeed,
                shouldStop: true,
                isListen: isListen,
                endOfTurn: false,
                collectForTTS: false,
                forced: forced,
                coercedListen: coerced)
        }

        // `turn_eos` is deliberately handled here, not as a terminator.  It
        // enters KV and is followed by the normal chunk terminator/flush.
        currentTurnEnded = false
        if token != tokens.speak {
            responseIDs.append(token)
        }
        let endOfTurn = token == tokens.turnEOS
        if endOfTurn { currentTurnEnded = true }

        // The first generated token is generally <|speak|> or <|tts_bos|> and
        // is a protocol priming token, not semantic text for TTS. The priming
        // markers MUST stay in the collected sequence: the semantic-TTS
        // condition consumes them as its speech-start marker (P3-A parity).
        let collectForTTS = index != 0
        if collectForTTS { generatedIDs.append(token) }
        trace.append(.init(name: "token", index: index, token: token))
        forceListen = false
        return MiniCPMDuplexProtocolStep(
            index: index,
            token: token,
            shouldFeed: true,
            shouldStop: false,
            isListen: false,
            endOfTurn: endOfTurn,
            collectForTTS: collectForTTS,
            forced: forced,
            coercedListen: coerced)
    }

    /// Reserve the final generation slot for the explicit chunk terminator.
    public mutating func finishAtLimit(
        index: Int,
        tokens: MiniCPMDuplexProtocolTokenIDs,
        config: MiniCPMDuplexConfig
    ) -> MiniCPMDuplexProtocolStep {
        consume(
            sampledToken: tokens.chunkEOS,
            index: index,
            tokens: tokens,
            config: config,
            forced: true)
    }

    /// Reject a sampled token that would exceed the upstream visible-text
    /// chunk bound. The rejected token has already consumed its sampler draw,
    /// but never enters KV or the generated response history.
    public mutating func finishAtCharacterLimit(
        index: Int,
        tokens: MiniCPMDuplexProtocolTokenIDs,
        config: MiniCPMDuplexConfig
    ) -> MiniCPMDuplexProtocolStep {
        trace.append(.init(name: "character_limit", index: index))
        return finishAtLimit(index: index, tokens: tokens, config: config)
    }

    /// Close a committed spoken unit after a barge-in.  This mirrors the
    /// official decoder's `turn_eos` + `</unit>` repair and deliberately does
    /// nothing while an asynchronous unit is still pending. `force` is used
    /// when the interrupted spoken unit itself was provisional: rollback can
    /// restore `currentTurnEnded == true` even though the previous committed
    /// unit still ends in a listen/chunk terminator that must be replaced.
    public mutating func closeInterruptedTurn(turnEOS: Int, unitEnd: Int) {
        closeInterruptedTurn(turnEOS: turnEOS, unitEnd: unitEnd, force: false)
    }

    @discardableResult
    public mutating func closeInterruptedTurn(
        turnEOS: Int,
        unitEnd: Int,
        force: Bool
    ) -> Bool {
        guard pendingSnapshot == nil else { return false }
        guard !currentTurnEnded || force else { return false }

        // A repeated cancel must not append another marker pair.  This also
        // makes the transport-side interrupt and the engine-side probe repair
        // safely idempotent when both observe the same epoch transition.
        if totalIDs.last == unitEnd,
           totalIDs.dropLast().last == turnEOS {
            currentTurnEnded = true
            committedUnitTerminator = nil
            return false
        }

        if totalIDs.last == unitEnd {
            totalIDs.removeLast()
            if let committedUnitTerminator,
               totalIDs.last == committedUnitTerminator {
                totalIDs.removeLast()
            }
        }
        totalIDs.append(contentsOf: [turnEOS, unitEnd])
        if responseIDs.last != turnEOS { responseIDs.append(turnEOS) }
        currentTurnEnded = true
        committedUnitTerminator = nil
        trace.append(.init(name: "interrupt_turn_eos", index: unitIndex, token: turnEOS))
        trace.append(.init(name: "unit_end", index: unitIndex, token: unitEnd))
        return true
    }

    /// Per-input force-listen override used by barge-in handling. The
    /// upstream decoder closes an already speaking turn with a fed
    /// ``turn_eos`` before it samples ``listen``; this marker is therefore
    /// outside the generated unit and must not be treated as a TTS token.
    @discardableResult
    public mutating func closeTurnForForcedListen(turnEOS: Int) -> Bool {
        guard !currentTurnEnded else { return false }
        totalIDs.append(turnEOS)
        currentTurnEnded = true
        trace.append(.init(name: "force_listen_turn_eos", index: unitIndex, token: turnEOS))
        return true
    }

    public func snapshot() -> MiniCPMDuplexProtocolSnapshot {
        MiniCPMDuplexProtocolSnapshot(
            currentTurnEnded: currentTurnEnded,
            totalIDs: totalIDs,
            responseIDs: responseIDs,
            generatedIDs: generatedIDs,
            generateCount: generateCount,
            unitIndex: unitIndex,
            committedUnitTerminator: committedUnitTerminator)
    }

    public mutating func commitUnit(unitEndID: Int) {
        totalIDs.append(unitEndID)
        trace.append(.init(name: "unit_end", index: unitIndex, token: unitEndID))
        unitIndex += 1
        pendingSnapshot = nil
        committedUnitTerminator = pendingUnitTerminator
        pendingUnitTerminator = nil
        generatedIDs.removeAll(keepingCapacity: true)
    }

    @discardableResult
    public mutating func rollbackUnit() -> Bool {
        guard let pendingSnapshot else { return false }
        currentTurnEnded = pendingSnapshot.currentTurnEnded
        totalIDs = pendingSnapshot.totalIDs
        responseIDs = pendingSnapshot.responseIDs
        generatedIDs = pendingSnapshot.generatedIDs
        generateCount = pendingSnapshot.generateCount
        unitIndex = pendingSnapshot.unitIndex
        committedUnitTerminator = pendingSnapshot.committedUnitTerminator
        self.pendingSnapshot = nil
        forceListen = false
        pendingUnitTerminator = nil
        trace.append(.init(name: "unit_rollback", index: unitIndex))
        return true
    }

    public var hasPendingUnit: Bool { pendingSnapshot != nil }
}

/// Transactional raw-audio tail used by the streaming convolution frontend.
///
/// The encoder session itself is checkpointed by
/// ``MiniCPMAudioStreamingSession.snapshot()``.  Accepted audio therefore
/// must not be retained here for replay: doing so made every long conversation
/// keep all PCM forever and forced rollback to re-encode the whole call.  This
/// type now keeps only the short, not-yet-embedded frontend tail.  The bound is
/// deliberately a little larger than the first 1,030 ms window so a packetized
/// first chunk and its completion remain diagnosable without becoming a second
/// copy of the conversation waveform.
public struct MiniCPMDuplexAudioHistory: Sendable, Equatable {
    public static let maximumTailSamples =
        MiniCPMStreamingMelProcessor.firstChunkSamples
        + MiniCPMStreamingMelProcessor.regularChunkSamples

    /// Deprecated compatibility surface. Accepted PCM is no longer retained;
    /// this collection remains empty so older diagnostics can compile without
    /// accidentally reintroducing an unbounded history.
    public private(set) var committedChunks: [[Float]] = []
    public private(set) var tailChunks: [[Float]] = []

    private var pendingTail: [[Float]]?

    public init() {}

    public mutating func reset() {
        committedChunks.removeAll(keepingCapacity: true)
        tailChunks.removeAll(keepingCapacity: true)
        pendingTail = nil
    }

    public mutating func beginUnit() {
        pendingTail = tailChunks
    }

    /// Record a chunk as soon as the frontend consumes it. ``producedEmbedding``
    /// is diagnostic only; both complete and incomplete chunks stay in the
    /// tail until the unit is accepted.
    public mutating func accept(_ samples: [Float], producedEmbedding: Bool) {
        _ = producedEmbedding
        tailChunks.append(samples)
        trimTailToBound()
    }

    /// Keep a rejected/incomplete frontend unit's newly buffered PCM as the
    /// rollback baseline for the next unit.  The audio frontend can consume
    /// the first 16,000-sample packet and still return no embedding because it
    /// is waiting for the remaining 480 samples.  That packet is not a
    /// language unit yet, but dropping it here would make replay after a
    /// later barge-in differ from the uninterrupted stream.
    public mutating func preserveTailAfterRejectedUnit() {
        pendingTail = tailChunks
    }

    public mutating func commitUnit() {
        // The audio session checkpoint is the authoritative accepted state;
        // retaining committed PCM here would duplicate it and make memory
        // grow with conversation length.
        committedChunks.removeAll(keepingCapacity: true)
        tailChunks.removeAll(keepingCapacity: true)
        pendingTail = nil
    }

    @discardableResult
    public mutating func rollbackUnit() -> Bool {
        guard let pendingTail else { return false }
        tailChunks = pendingTail
        self.pendingTail = nil
        return true
    }

    /// Compatibility diagnostic. Runtime rollback no longer consumes this;
    /// only the bounded unembedded tail is exposed.
    public var replayChunks: [[Float]] { tailChunks }

    private mutating func trimTailToBound() {
        let limit = Self.maximumTailSamples
        guard limit > 0 else {
            tailChunks.removeAll(keepingCapacity: true)
            return
        }
        var total = tailChunks.reduce(0) { $0 + $1.count }
        guard total > limit else { return }

        // Drop complete old packets first. If the newest packet itself is
        // larger than the bound, retain its suffix rather than silently
        // making the tail unbounded.
        while total > limit, !tailChunks.isEmpty {
            let excess = total - limit
            if excess >= tailChunks[0].count {
                total -= tailChunks[0].count
                tailChunks.removeFirst()
            } else {
                tailChunks[0].removeFirst(excess)
                total = limit
            }
        }
    }
}
