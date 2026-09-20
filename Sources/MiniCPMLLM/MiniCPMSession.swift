import Foundation
import MLX

public struct MiniCPMUnitRecord {
    /// Stable order id used by the duplex sliding-window bookkeeping.
    public var unitID: Int
    /// ``audio``/``vision``/``omni``/``text``; retained for diagnostics and
    /// parity with the upstream ``StreamDecoder`` unit history.
    public var inputType: String
    public var embeddings: [MLXArray]
    public var generatedTokens: [Int]
    public var generatedText: String
    public var isListen: Bool

    public init(
        unitID: Int = 0,
        inputType: String = "audio",
        embeddings: [MLXArray] = [],
        generatedTokens: [Int] = [],
        generatedText: String = "",
        isListen: Bool = false
    ) {
        self.unitID = unitID
        self.inputType = inputType
        self.embeddings = embeddings
        self.generatedTokens = generatedTokens
        self.generatedText = generatedText
        self.isListen = isListen
    }

    /// Number of decoder positions occupied by this unit. Stored embeddings
    /// are normalized to ``[1, T, H]`` by ``MiniCPMSession.feed``.
    public var length: Int {
        embeddings.reduce(0) { partial, value in
            guard value.ndim >= 2 else { return partial + 1 }
            return partial + value.dim(value.ndim - 2)
        }
    }
}

/// Exact counterpart of the official ``DuplexWindowConfig``. The legacy
/// runtime used ``slidingWindowTokens / 256`` as a unit-count approximation;
/// keeping watermarks and context limits explicit avoids that lossy mapping.
public struct MiniCPMSlidingWindowConfig: Sendable, Equatable {
    public var mode: String
    public var basicHighTokens: Int
    public var basicLowTokens: Int
    public var contextPreviousMaxTokens: Int
    public var contextMaxUnits: Int
    public var contextPreviousMarkerTokenIDs: [Int]
    public var specialTokenIDs: Set<Int>

    public init(
        mode: String = "off",
        basicHighTokens: Int = 8_000,
        basicLowTokens: Int = 6_000,
        contextPreviousMaxTokens: Int = 500,
        contextMaxUnits: Int = 24,
        contextPreviousMarkerTokenIDs: [Int] = [],
        specialTokenIDs: Set<Int> = []
    ) {
        let normalizedMode = mode.lowercased()
        self.mode = ["off", "basic", "context"].contains(normalizedMode)
            ? normalizedMode : "off"
        self.basicHighTokens = max(0, basicHighTokens)
        self.basicLowTokens = max(0, min(basicLowTokens, self.basicHighTokens))
        self.contextPreviousMaxTokens = max(0, contextPreviousMaxTokens)
        self.contextMaxUnits = max(0, contextMaxUnits)
        self.contextPreviousMarkerTokenIDs = contextPreviousMarkerTokenIDs
        self.specialTokenIDs = specialTokenIDs
    }

    public var enabled: Bool { mode != "off" }
}

/// A value snapshot of all state that can affect a streaming decode turn.
/// Arrays are immutable MLX handles; retaining them is substantially cheaper
/// than cloning the full KV cache and still gives exact rollback semantics.
public struct MiniCPMSessionSnapshot {
    public let inference: MiniCPMInferenceState
    public let units: [MiniCPMUnitRecord]
    public let systemEmbeddings: [MLXArray]
    public let systemPrefixEmbeddings: [MLXArray]
    public let systemSuffixEmbeddings: [MLXArray]
    public let systemExtraEmbeddings: [MLXArray]
    public let contextPreviousEmbeddings: MLXArray?
    public let contextPreviousTokenIDs: [Int]
    public let contextPreviousText: String
    public let activeEmbeddings: [MLXArray]
    public let protocolTokens: [Int]
    public let generatedTokens: [Int]
    public let generatedSpecialTokens: [Int]
    public let nextUnitID: Int
    public let windowConfig: MiniCPMSlidingWindowConfig
    /// RNG position is part of a decode snapshot.  Restoring only the KV
    /// cache while leaving the sampler advanced changes every later token.
    public let samplerState: MiniCPMSamplerState

    init(
        inference: MiniCPMInferenceState,
        units: [MiniCPMUnitRecord],
        systemEmbeddings: [MLXArray],
        systemPrefixEmbeddings: [MLXArray],
        systemSuffixEmbeddings: [MLXArray],
        systemExtraEmbeddings: [MLXArray],
        contextPreviousEmbeddings: MLXArray?,
        contextPreviousTokenIDs: [Int],
        contextPreviousText: String,
        activeEmbeddings: [MLXArray],
        protocolTokens: [Int],
        generatedTokens: [Int],
        generatedSpecialTokens: [Int],
        samplerState: MiniCPMSamplerState,
        nextUnitID: Int,
        windowConfig: MiniCPMSlidingWindowConfig
    ) {
        self.inference = inference
        self.units = units
        self.systemEmbeddings = systemEmbeddings
        self.systemPrefixEmbeddings = systemPrefixEmbeddings
        self.systemSuffixEmbeddings = systemSuffixEmbeddings
        self.systemExtraEmbeddings = systemExtraEmbeddings
        self.contextPreviousEmbeddings = contextPreviousEmbeddings
        self.contextPreviousTokenIDs = contextPreviousTokenIDs
        self.contextPreviousText = contextPreviousText
        self.activeEmbeddings = activeEmbeddings
        self.protocolTokens = protocolTokens
        self.generatedTokens = generatedTokens
        self.generatedSpecialTokens = generatedSpecialTokens
        self.samplerState = samplerState
        self.nextUnitID = nextUnitID
        self.windowConfig = windowConfig
    }
}

/// Session coordinator around the stateless MLX backbone.  It owns no audio
/// frontend or TTS implementation; those components call ``feed(embeddings:)``
/// and can therefore share the official MiniCPM duplex protocol unchanged.
public final class MiniCPMSession {
    public let model: MiniCPMLanguageModel
    public private(set) var state: MiniCPMInferenceState
    public private(set) var protocolTokens: [Int] = []
    public private(set) var generatedTokens: [Int] = []
    public private(set) var generatedSpecialTokens: [Int] = []
    /// Per-session stochastic state.  The property is settable so the
    /// duplex coordinator can hand an externally persisted state back to the
    /// session; `snapshot()`/`restore(_:)` also carry it automatically.
    public var samplerState: MiniCPMSamplerState

    public private(set) var units: [MiniCPMUnitRecord] = []
    private var systemEmbeddings: [MLXArray] = []
    /// Context sliding keeps the original system prefix and suffix separate:
    /// ``[prefix][previous marker + text][suffix][units...]``. The aggregate
    /// ``systemEmbeddings`` remains for source compatibility/diagnostics.
    private var systemPrefixEmbeddings: [MLXArray] = []
    private var systemSuffixEmbeddings: [MLXArray] = []
    private var systemExtraEmbeddings: [MLXArray] = []
    private var contextPreviousEmbeddings: MLXArray?
    private var contextPreviousTokenIDs: [Int] = []
    private var contextPreviousText: String = ""
    private var nextUnitID: Int = 0
    private var windowConfig = MiniCPMSlidingWindowConfig()
    private enum SystemPromptPhase { case none, prefix, suffix, closed }
    private var systemPromptPhase: SystemPromptPhase = .none
    private var activeEmbeddings: [MLXArray] = []
    private var pendingSnapshot: MiniCPMSessionSnapshot?

    public init(model: MiniCPMLanguageModel, samplerSeed: UInt64? = nil) {
        self.model = model
        self.state = model.initialState()
        self.samplerState = MiniCPMSamplerState(seed: samplerSeed)
    }

    public var cacheLength: Int { state.sequenceLength }
    public var hasPendingUnit: Bool { pendingSnapshot != nil }

    public func reset() {
        state = model.initialState()
        protocolTokens.removeAll(keepingCapacity: true)
        generatedTokens.removeAll(keepingCapacity: true)
        generatedSpecialTokens.removeAll(keepingCapacity: true)
        units.removeAll(keepingCapacity: true)
        systemEmbeddings.removeAll(keepingCapacity: true)
        systemPrefixEmbeddings.removeAll(keepingCapacity: true)
        systemSuffixEmbeddings.removeAll(keepingCapacity: true)
        systemExtraEmbeddings.removeAll(keepingCapacity: true)
        contextPreviousEmbeddings = nil
        contextPreviousTokenIDs.removeAll(keepingCapacity: true)
        contextPreviousText = ""
        nextUnitID = 0
        windowConfig = MiniCPMSlidingWindowConfig()
        systemPromptPhase = .none
        activeEmbeddings.removeAll(keepingCapacity: true)
        pendingSnapshot = nil
    }

    public func snapshot() -> MiniCPMSessionSnapshot {
        MiniCPMSessionSnapshot(
            inference: state,
            units: units,
            systemEmbeddings: systemEmbeddings,
            systemPrefixEmbeddings: systemPrefixEmbeddings,
            systemSuffixEmbeddings: systemSuffixEmbeddings,
            systemExtraEmbeddings: systemExtraEmbeddings,
            contextPreviousEmbeddings: contextPreviousEmbeddings,
            contextPreviousTokenIDs: contextPreviousTokenIDs,
            contextPreviousText: contextPreviousText,
            activeEmbeddings: activeEmbeddings,
            protocolTokens: protocolTokens,
            generatedTokens: generatedTokens,
            generatedSpecialTokens: generatedSpecialTokens,
            samplerState: samplerState,
            nextUnitID: nextUnitID,
            windowConfig: windowConfig)
    }

    public func restore(_ snapshot: MiniCPMSessionSnapshot) {
        state = snapshot.inference
        units = snapshot.units
        systemEmbeddings = snapshot.systemEmbeddings
        systemPrefixEmbeddings = snapshot.systemPrefixEmbeddings
        systemSuffixEmbeddings = snapshot.systemSuffixEmbeddings
        systemExtraEmbeddings = snapshot.systemExtraEmbeddings
        contextPreviousEmbeddings = snapshot.contextPreviousEmbeddings
        contextPreviousTokenIDs = snapshot.contextPreviousTokenIDs
        contextPreviousText = snapshot.contextPreviousText
        activeEmbeddings = snapshot.activeEmbeddings
        protocolTokens = snapshot.protocolTokens
        generatedTokens = snapshot.generatedTokens
        generatedSpecialTokens = snapshot.generatedSpecialTokens
        samplerState = snapshot.samplerState
        nextUnitID = snapshot.nextUnitID
        windowConfig = snapshot.windowConfig
        systemPromptPhase = systemPrefixEmbeddings.isEmpty && systemSuffixEmbeddings.isEmpty
            ? .none : .closed
        pendingSnapshot = nil
    }

    // MARK: Official system-prompt boundaries

    /// Start capturing the fixed prefix of the official system prompt. The
    /// caller feeds prefix text and optional reference-audio embeddings after
    /// this call, then switches to the suffix with
    /// ``markSystemPromptSuffix()``.
    public func beginSystemPrompt() {
        systemEmbeddings.removeAll(keepingCapacity: true)
        systemPrefixEmbeddings.removeAll(keepingCapacity: true)
        systemSuffixEmbeddings.removeAll(keepingCapacity: true)
        systemExtraEmbeddings.removeAll(keepingCapacity: true)
        contextPreviousEmbeddings = nil
        contextPreviousTokenIDs.removeAll(keepingCapacity: true)
        contextPreviousText = ""
        systemPromptPhase = .prefix
    }

    /// Mark the boundary between the fixed prefix (including reference audio)
    /// and the suffix which follows the dynamic context marker in context
    /// sliding mode.
    public func markSystemPromptSuffix() {
        guard systemPromptPhase == .prefix else { return }
        systemPrefixEmbeddings = systemEmbeddings
        systemEmbeddings.removeAll(keepingCapacity: true)
        systemPromptPhase = .suffix
    }

    /// Finish system-prompt capture. The aggregate is retained for callers
    /// that only inspect ``systemEmbeddings``; replay always uses the explicit
    /// prefix/suffix/extra partitions to protect the prompt correctly.
    public func finishSystemPrompt() {
        guard systemPromptPhase == .suffix || systemPromptPhase == .prefix else { return }
        if systemPromptPhase == .suffix {
            systemSuffixEmbeddings = systemEmbeddings
        } else {
            systemPrefixEmbeddings = systemEmbeddings
        }
        systemEmbeddings = systemPrefixEmbeddings + systemSuffixEmbeddings
        systemPromptPhase = .closed
    }

    /// Configure the exact upstream window policy. Configuration is captured
    /// in unit snapshots, so a rollback never observes a partially changed
    /// policy or context state.
    public func configureSlidingWindow(_ config: MiniCPMSlidingWindowConfig) {
        windowConfig = config
        if config.mode != "context" {
            contextPreviousEmbeddings = nil
            contextPreviousTokenIDs.removeAll(keepingCapacity: true)
            contextPreviousText = ""
        }
    }

    public var slidingWindowConfiguration: MiniCPMSlidingWindowConfig { windowConfig }

    public var previousContext: (text: String, tokenIDs: [Int]) {
        (contextPreviousText, contextPreviousTokenIDs)
    }

    @discardableResult
    public func beginUnit() -> MiniCPMSessionSnapshot {
        let snapshot = self.snapshot()
        pendingSnapshot = snapshot
        activeEmbeddings.removeAll(keepingCapacity: true)
        return snapshot
    }

    public func commitUnit(
        generatedTokens: [Int] = [],
        generatedText: String = "",
        isListen: Bool = false,
        inputType: String = "audio"
    ) {
        guard pendingSnapshot != nil else { return }
        units.append(MiniCPMUnitRecord(
            unitID: nextUnitID,
            inputType: inputType,
            embeddings: activeEmbeddings,
            generatedTokens: generatedTokens,
            generatedText: generatedText,
            isListen: isListen))
        nextUnitID += 1
        self.generatedTokens.append(contentsOf: generatedTokens)
        pendingSnapshot = nil
        activeEmbeddings.removeAll(keepingCapacity: true)
    }

    @discardableResult
    public func rollbackUnit() -> Int {
        guard let snapshot = pendingSnapshot else { return 0 }
        let removed = max(0, state.position - snapshot.inference.position)
        restore(snapshot)
        return removed
    }

    @discardableResult
    public func feed(inputIds: MLXArray) -> MiniCPMForwardOutput {
        let embeddings = model.embed(inputIds)
        return feed(embeddings: embeddings)
    }

    @discardableResult
    public func feed(embeddings: MLXArray) -> MiniCPMForwardOutput {
        let output = model.forward(inputEmbeddings: embeddings, state: state)
        state = output.state
        let replay = normalizeForReplay(embeddings)
        if pendingSnapshot != nil {
            activeEmbeddings.append(replay)
        } else {
            switch systemPromptPhase {
            case .prefix:
                systemEmbeddings.append(replay)
            case .suffix:
                systemEmbeddings.append(replay)
            case .closed:
                systemExtraEmbeddings.append(replay)
            case .none:
                systemEmbeddings.append(replay)
            }
        }
        return output
    }

    public func recordProtocolToken(_ token: Int, special: Bool = false) {
        protocolTokens.append(token)
        if special {
            generatedSpecialTokens.append(token)
        } else {
            generatedTokens.append(token)
        }
    }

    /// Draw one protocol token while advancing this session's RNG.  Keeping
    /// this tiny adapter beside the KV/session snapshot makes it difficult for
    /// a caller to accidentally use a global/fixed random source.  The token
    /// is deliberately not recorded here: callers may need to inspect or
    /// route protocol markers before committing them to a unit.
    @discardableResult
    public func sample(
        logits: MLXArray,
        config: MiniCPMSamplingConfig,
        `protocol`: MiniCPMProtocolTokenIds,
        forbidden: [Int] = [],
        previousTokens: [Int]? = nil
    ) -> MLXArray {
        MiniCPMSampler.sample(
            logits: logits,
            config: config,
            protocol: `protocol`,
            forbidden: forbidden,
            previousTokens: previousTokens ?? generatedTokens,
            state: &samplerState)
    }

    /// Enforce the selected window policy. Basic mode performs cache surgery
    /// in place of replay: retained values stay untouched and only suffix keys
    /// are RoPE-reindexed after a middle segment is removed. Context mode
    /// retains its existing replay implementation for now.
    @discardableResult
    public func enforceSlidingWindow() -> Bool {
        guard windowConfig.enabled, pendingSnapshot == nil else { return false }
        switch windowConfig.mode {
        case "basic":
            return enforceBasicWindow()
        case "context":
            return enforceContextWindow()
        default:
            return false
        }
    }

    /// Compatibility entry point for callers that used the old unit-count
    /// approximation. It now configures an explicit basic high/low watermark
    /// instead of dividing token counts by a guessed 256-token unit size.
    @discardableResult
    public func enforceSlidingWindow(maxUnits: Int) -> Bool {
        guard maxUnits >= 0 else { return false }
        let old = windowConfig
        windowConfig = MiniCPMSlidingWindowConfig(
            mode: "context",
            basicHighTokens: old.basicHighTokens,
            basicLowTokens: old.basicLowTokens,
            contextPreviousMaxTokens: old.contextPreviousMaxTokens,
            contextMaxUnits: maxUnits,
            contextPreviousMarkerTokenIDs: old.contextPreviousMarkerTokenIDs,
            specialTokenIDs: old.specialTokenIDs)
        return enforceSlidingWindow()
    }

    public func windowStats() -> [String: Any] {
        let preserveLength: Int
        if windowConfig.mode == "basic" {
            preserveLength = basicSystemPreserveLength
        } else {
            let hasExplicitLayout = !systemPrefixEmbeddings.isEmpty
                || !systemSuffixEmbeddings.isEmpty
            let prefixParts = hasExplicitLayout ? systemPrefixEmbeddings : systemEmbeddings
            let suffixParts = hasExplicitLayout ? systemSuffixEmbeddings : []
            preserveLength = prefixParts.reduce(0, { $0 + Self.embeddingLength($1) })
                + suffixParts.reduce(0, { $0 + Self.embeddingLength($1) })
                + systemExtraEmbeddings.reduce(0, { $0 + Self.embeddingLength($1) })
                + (contextPreviousEmbeddings.map(Self.embeddingLength) ?? 0)
        }
        return [
            "cache_length": cacheLength,
            "unit_count": units.count,
            "unit_lengths": units.map(\.length),
            "unit_total_length": units.reduce(0) { $0 + $1.length },
            "system_preserve_length": preserveLength,
            "window_enabled": windowConfig.enabled,
            "config": [
                "sliding_window_mode": windowConfig.mode,
                "basic_window_high_tokens": windowConfig.basicHighTokens,
                "basic_window_low_tokens": windowConfig.basicLowTokens,
                "context_previous_max_tokens": windowConfig.contextPreviousMaxTokens,
                "context_max_units": windowConfig.contextMaxUnits,
            ],
            "previous_token_count": contextPreviousTokenIDs.count,
            "previous_text": contextPreviousText,
        ]
    }

    private func enforceBasicWindow() -> Bool {
        guard cacheLength > windowConfig.basicHighTokens else { return false }
        var dropped = false
        let preserveLength = basicSystemPreserveLength
        while cacheLength > windowConfig.basicLowTokens {
            guard let index = units.firstIndex(where: { $0.inputType != "system" }) else { break }
            let unit = units[index]
            let unitLength = unit.length
            if unitLength <= 0 {
                units.remove(at: index)
                dropped = true
                rebuildGeneratedTokenHistory()
                continue
            }

            // The official StreamDecoder removes [preserve, preserve+len),
            // preserving every V vector and hidden-history value after it.
            // Only the suffix K tensors are reindexed to their compressed
            // RoPE positions. If metadata cannot describe the cache safely,
            // leave the unit and state untouched rather than replaying it.
            guard let compacted = model.droppingState(
                state,
                preserveLength: preserveLength,
                droppingLength: unitLength) else {
                break
            }
            state = compacted
            units.remove(at: index)
            dropped = true
            rebuildGeneratedTokenHistory()
        }
        return dropped
    }

    private func enforceContextWindow() -> Bool {
        guard units.count > windowConfig.contextMaxUnits else { return false }
        var dropped = false
        while units.count > windowConfig.contextMaxUnits {
            guard let index = units.firstIndex(where: { $0.inputType != "system" }) else { break }

            // Keep the complete pre-eviction metadata around until the new
            // cache layout has been built.  The model helper copies old cache
            // ranges before forwarding, so a failed rebuild can safely leave
            // both the state and its snapshot untouched.
            let oldUnits = units
            let oldPreviousEmbeddings = contextPreviousEmbeddings
            let oldPreviousTokenIDs = contextPreviousTokenIDs
            let oldPreviousText = contextPreviousText
            let removed = units.remove(at: index)
            appendPreviousContext(from: removed)
            let retainedLength = units.reduce(0) { $0 + $1.length }
            guard rebuildContextCache(retainedLength: retainedLength) else {
                units = oldUnits
                contextPreviousEmbeddings = oldPreviousEmbeddings
                contextPreviousTokenIDs = oldPreviousTokenIDs
                contextPreviousText = oldPreviousText
                break
            }
            dropped = true
            rebuildGeneratedTokenHistory()
        }
        return dropped
    }

    private func appendPreviousContext(from unit: MiniCPMUnitRecord) {
        // The official extractor skips listen units unconditionally.  A
        // listen record should therefore never leak generated metadata into
        // the previous-text segment, even if a caller accidentally attaches
        // non-empty token/text fields to it.
        guard !unit.isListen else { return }
        let content = unit.generatedTokens.filter { !windowConfig.specialTokenIDs.contains($0) }
        guard !content.isEmpty || !unit.generatedText.isEmpty else {
            // Upstream still rebuilds after dropping a listen/empty unit; the
            // caller handles the replay, but no marker should be injected.
            return
        }

        let marker = windowConfig.contextPreviousMarkerTokenIDs
        if contextPreviousTokenIDs.isEmpty {
            contextPreviousTokenIDs = marker + content
        } else {
            contextPreviousTokenIDs.append(contentsOf: content)
        }
        contextPreviousText += unit.generatedText

        let markerLength = marker.count
        let maxContent = windowConfig.contextPreviousMaxTokens
        let contentStart = min(markerLength, contextPreviousTokenIDs.count)
        var contentTokens = Array(contextPreviousTokenIDs.dropFirst(contentStart))
        if contentTokens.count > maxContent {
            contentTokens.removeFirst(contentTokens.count - maxContent)
        }
        contextPreviousTokenIDs = marker + contentTokens
        // The text is diagnostic only; token IDs are the authoritative replay
        // representation. Keep a bounded suffix to avoid retaining dropped
        // transcripts indefinitely when no tokenizer decoder is available.
        if contextPreviousText.count > maxContent * 4 {
            contextPreviousText = String(contextPreviousText.suffix(maxContent * 4))
        }
        // The token IDs are authoritative.  ``rebuildContextCache`` embeds
        // them immediately before the official prefix/new-previous/suffix
        // forward and stores the resulting array for snapshots and stats.
        contextPreviousEmbeddings = nil
    }

    /// Apply the official context-window cache surgery after one or more
    /// units have been removed.  The previous token IDs are re-embedded once,
    /// then concatenated with the fixed suffix (and any post-template system
    /// extras) for a single BF16 forward using only the copied prefix cache.
    /// Remaining unit embeddings are deliberately not replayed: their old KV
    /// tail is copied and RoPE-reindexed by the model helper.
    private func rebuildContextCache(retainedLength: Int) -> Bool {
        let hasExplicitLayout = !systemPrefixEmbeddings.isEmpty
            || !systemSuffixEmbeddings.isEmpty
        let prefixParts = hasExplicitLayout ? systemPrefixEmbeddings : systemEmbeddings
        let suffixParts = hasExplicitLayout ? systemSuffixEmbeddings : []

        let prefixLength = prefixParts.reduce(0) {
            $0 + Self.embeddingLength($1)
        }
        let previousEmbeddings: MLXArray?
        if contextPreviousTokenIDs.isEmpty {
            previousEmbeddings = nil
        } else {
            let ids = MLXArray(contextPreviousTokenIDs.map(Int32.init))
            previousEmbeddings = model.embed(ids).asType(.bfloat16)
        }

        var segmentParts: [MLXArray] = []
        if let previousEmbeddings {
            segmentParts.append(previousEmbeddings)
        }
        segmentParts.append(contentsOf: suffixParts)
        segmentParts.append(contentsOf: systemExtraEmbeddings)
        let segment = Self.joinEmbeddings(segmentParts)

        guard let rebuilt = model.rebuildingContextState(
            state,
            prefixLength: prefixLength,
            segmentEmbeddings: segment,
            retainedLength: retainedLength) else {
            return false
        }
        state = rebuilt
        contextPreviousEmbeddings = previousEmbeddings
        return true
    }

    private func rebuildGeneratedTokenHistory() {
        generatedTokens = units.flatMap(\.generatedTokens)
    }

    private static func joinEmbeddings(_ values: [MLXArray]) -> MLXArray? {
        let normalized = values.compactMap { value -> MLXArray? in
            guard value.ndim >= 2 else { return nil }
            let result = value.ndim == 3
                ? value
                : value.expandedDimensions(axis: 0)
            guard result.dim(1) > 0 else { return nil }
            return result.asType(.bfloat16)
        }
        guard !normalized.isEmpty else { return nil }
        if normalized.count == 1 { return normalized[0] }
        return concatenated(normalized, axis: 1)
    }

    /// Length of the fixed system prompt in basic mode. The explicit
    /// prefix/suffix partitions belong to context mode; aggregate
    /// ``systemEmbeddings`` is authoritative for basic sliding.
    private var basicSystemPreserveLength: Int {
        let aggregate = systemEmbeddings.reduce(0) {
            $0 + Self.embeddingLength($1)
        } + systemExtraEmbeddings.reduce(0) {
            $0 + Self.embeddingLength($1)
        }
        if aggregate > 0 || (systemPrefixEmbeddings.isEmpty && systemSuffixEmbeddings.isEmpty) {
            return aggregate
        }
        return systemPrefixEmbeddings.reduce(0) { $0 + Self.embeddingLength($1) }
            + systemSuffixEmbeddings.reduce(0) { $0 + Self.embeddingLength($1) }
            + systemExtraEmbeddings.reduce(0) { $0 + Self.embeddingLength($1) }
    }

    /// Close an accepted response when the user interrupts.  The last
    /// ``</unit>`` embedding is removed, then canonical turn markers are fed
    /// through the same model/cache path.
    @discardableResult
    public func closeInterruptedTurn(turnEosId: Int, unitEndId: Int) -> Int {
        guard pendingSnapshot == nil else { return 0 }
        guard !units.isEmpty else { return 0 }
        let last = units.count - 1
        guard !units[last].embeddings.isEmpty else { return 0 }
        // `finalize()` feeds the deferred terminator and `</unit>` in one
        // forward pass. Removing only `</unit>` from the replay metadata
        // leaves protocolTokens one marker ahead of the KV cache (the exact
        // shape that lets a cancelled response leak into the next turn).
        let finalEmbedding = units[last].embeddings.last!
        let finalMarkerLength = max(1, Self.embeddingLength(finalEmbedding))
        units[last].embeddings.removeLast()
        if protocolTokens.last == unitEndId {
            protocolTokens.removeLast()
            if finalMarkerLength >= 2, !protocolTokens.isEmpty {
                protocolTokens.removeLast()
            }
        }
        if generatedSpecialTokens.last == unitEndId {
            generatedSpecialTokens.removeLast()
            if finalMarkerLength >= 2, !generatedSpecialTokens.isEmpty {
                generatedSpecialTokens.removeLast()
            }
        }
        rebuildFromReplay()
        let markers = MLXArray([Int32(turnEosId), Int32(unitEndId)])
        let markerEmbeddings = model.embed(markers).asType(.bfloat16)
        let output = model.forward(inputEmbeddings: markerEmbeddings, state: state)
        state = output.state
        units[last].embeddings.append(markerEmbeddings)
        units[last].generatedTokens.append(turnEosId)
        protocolTokens.append(contentsOf: [turnEosId, unitEndId])
        if windowConfig.specialTokenIDs.contains(turnEosId) {
            generatedSpecialTokens.append(turnEosId)
        } else {
            generatedTokens.append(turnEosId)
        }
        generatedSpecialTokens.append(unitEndId)
        return 1
    }

    private func rebuildFromReplay() {
        state = model.initialState()
        let prefix = systemPrefixEmbeddings.isEmpty && systemSuffixEmbeddings.isEmpty
            ? systemEmbeddings : systemPrefixEmbeddings
        let suffix = systemPrefixEmbeddings.isEmpty && systemSuffixEmbeddings.isEmpty
            ? [] : systemSuffixEmbeddings
        for embeddings in prefix {
            state = model.forward(inputEmbeddings: embeddings, state: state).state
        }
        if !contextPreviousTokenIDs.isEmpty {
            let ids = MLXArray(contextPreviousTokenIDs.map(Int32.init))
            contextPreviousEmbeddings = model.embed(ids).asType(.bfloat16)
            state = model.forward(
                inputEmbeddings: contextPreviousEmbeddings!, state: state).state
        } else {
            contextPreviousEmbeddings = nil
        }
        for embeddings in suffix {
            state = model.forward(inputEmbeddings: embeddings, state: state).state
        }
        for embeddings in systemExtraEmbeddings {
            state = model.forward(inputEmbeddings: embeddings, state: state).state
        }
        for unit in units {
            for embeddings in unit.embeddings {
                state = model.forward(inputEmbeddings: embeddings, state: state).state
            }
        }
    }

    private func normalizeForReplay(_ embeddings: MLXArray) -> MLXArray {
        embeddings.ndim == 3 ? embeddings : embeddings.expandedDimensions(axis: 0)
    }

    private static func embeddingLength(_ embeddings: MLXArray) -> Int {
        guard embeddings.ndim >= 2 else { return 1 }
        return embeddings.dim(embeddings.ndim - 2)
    }
}
