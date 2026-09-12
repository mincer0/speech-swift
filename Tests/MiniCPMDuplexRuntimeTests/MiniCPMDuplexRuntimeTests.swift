import XCTest
@testable import MiniCPMDuplexRuntime
import MiniCPMLLM

private final class RecordingEngine: MiniCPMDuplexEngine {
    var events: [String] = []
    var output = MiniCPMDuplexOutput(text: "ok", endOfTurn: true)

    func prepare(mode: MiniCPMDuplexMode, config: MiniCPMDuplexConfig) throws -> String {
        events.append("prepare:\(mode.rawValue)")
        return "prompt"
    }

    func prefill(mode: MiniCPMDuplexMode, input: MiniCPMDuplexInput) throws -> MiniCPMDuplexPrefillResult {
        events.append("prefill:\(mode.rawValue)")
        return MiniCPMDuplexPrefillResult(mode: mode, input: input)
    }

    func generate(mode: MiniCPMDuplexMode, config: MiniCPMDuplexConfig) throws -> MiniCPMDuplexOutput {
        events.append("generate:\(mode.rawValue)")
        return output
    }

    func finalize() throws { events.append("finalize") }
    func rollback() throws { events.append("rollback") }
    func interrupt() { events.append("interrupt") }
    func reset() { events.append("reset") }
    func setSlidingWindow(tokens: Int?) { events.append("window:\(tokens.map(String.init) ?? "nil")") }
}

/// Exercises the same synchronous shape as a native MLX token loop: actor
/// ``generate`` is blocked until its per-generation probe observes the
/// transport's monotonic interruption epoch.
private final class ProbeBlockingEngine: MiniCPMDuplexEngine {
    private var probe: (@Sendable () -> Bool)?
    private var blockNextGeneration = true
    let generationStarted = DispatchSemaphore(value: 0)

    func prepare(mode: MiniCPMDuplexMode, config: MiniCPMDuplexConfig) throws -> String {
        "prompt"
    }

    func prefill(mode: MiniCPMDuplexMode, input: MiniCPMDuplexInput) throws -> MiniCPMDuplexPrefillResult {
        MiniCPMDuplexPrefillResult(mode: mode, input: input)
    }

    func generate(mode: MiniCPMDuplexMode, config: MiniCPMDuplexConfig) throws -> MiniCPMDuplexOutput {
        if blockNextGeneration {
            blockNextGeneration = false
            generationStarted.signal()
            while !(probe?() ?? false) {
                Thread.sleep(forTimeInterval: 0.001)
            }
            return MiniCPMDuplexOutput(isListen: true, needsFinalize: false, finishReason: "interrupt")
        }
        return MiniCPMDuplexOutput(text: "fresh", endOfTurn: true)
    }

    func finalize() throws {}
    func rollback() throws {}
    func interrupt() {}
    func reset() {}
    func setSlidingWindow(tokens: Int?) {}
    func setInterruptionProbe(_ probe: (@Sendable () -> Bool)?) {
        self.probe = probe
    }
}

/// Mirrors a native audio/vision prefill that notices the interruption probe
/// after it has already mutated a provisional unit and reports that boundary
/// as an error.  Cancellation must be converted to `accepted == false` by the
/// runtime; otherwise the backend mistakes a normal barge-in for a fatal error.
private final class ProbeBlockingPrefillEngine: MiniCPMDuplexEngine {
    private var probe: (@Sendable () -> Bool)?
    private var blockNextPrefill = true
    let prefillStarted = DispatchSemaphore(value: 0)
    private(set) var interruptCount = 0

    func prepare(mode: MiniCPMDuplexMode, config: MiniCPMDuplexConfig) throws -> String {
        "prompt"
    }

    func prefill(mode: MiniCPMDuplexMode, input: MiniCPMDuplexInput) throws -> MiniCPMDuplexPrefillResult {
        if blockNextPrefill {
            blockNextPrefill = false
            prefillStarted.signal()
            while !(probe?() ?? false) {
                Thread.sleep(forTimeInterval: 0.001)
            }
            throw MiniCPMDuplexRuntimeError.invalidState("prefill interrupted")
        }
        return MiniCPMDuplexPrefillResult(mode: mode, input: input)
    }

    func generate(mode: MiniCPMDuplexMode, config: MiniCPMDuplexConfig) throws -> MiniCPMDuplexOutput {
        MiniCPMDuplexOutput(text: "fresh", endOfTurn: true, finishReason: "fresh")
    }

    func finalize() throws {}
    func rollback() throws {}
    func interrupt() { interruptCount += 1 }
    func reset() {}
    func setSlidingWindow(tokens: Int?) {}
    func setInterruptionProbe(_ probe: (@Sendable () -> Bool)?) {
        self.probe = probe
    }
}

final class MiniCPMDuplexRuntimeTests: XCTestCase {
    private func protocolIDs() -> MiniCPMDuplexProtocolTokenIDs {
        MiniCPMDuplexProtocolTokenIDs(
            eos: 99, unit: 1, unitEnd: 2, listen: 3, speak: 4,
            ttsBOS: 6, chunkEOS: 12, chunkTTSEOS: 14, turnEOS: 10)
    }

    func testProtocolTraceTurnEOSIsFedBeforeFinalChunkEOS() {
        let ids = protocolIDs()
        var state = MiniCPMDuplexProtocolState()
        _ = state.beginUnit()
        _ = state.beginGeneration(config: MiniCPMDuplexConfig(maxNewTokens: 3, forceListenCount: 0))

        let first = state.consume(
            sampledToken: ids.speak, index: 0, tokens: ids,
            config: MiniCPMDuplexConfig(maxNewTokens: 3, forceListenCount: 0))
        let turn = state.consume(
            sampledToken: ids.turnEOS, index: 1, tokens: ids,
            config: MiniCPMDuplexConfig(maxNewTokens: 3, forceListenCount: 0))
        let last = state.finishAtLimit(
            index: 2, tokens: ids,
            config: MiniCPMDuplexConfig(maxNewTokens: 3, forceListenCount: 0))

        XCTAssertFalse(first.shouldStop)
        XCTAssertFalse(turn.shouldStop)
        XCTAssertTrue(turn.endOfTurn)
        XCTAssertTrue(last.shouldStop)
        XCTAssertEqual(last.token, ids.chunkEOS)
        XCTAssertTrue(first.shouldFeed && turn.shouldFeed)
        XCTAssertFalse(last.shouldFeed)
        XCTAssertEqual(state.generatedIDs, [ids.turnEOS])
        XCTAssertEqual(
            state.trace.map(\.name),
            ["unit_start", "generate", "token", "token", "chunk_terminator", "chunk_eos_forced"])
    }

    func testProtocolTraceOpenTurnListenIsCoercedToTTSBOS() {
        let ids = protocolIDs()
        var state = MiniCPMDuplexProtocolState()
        _ = state.beginUnit()
        _ = state.beginGeneration(config: MiniCPMDuplexConfig(maxNewTokens: 4, forceListenCount: 0))
        _ = state.consume(
            sampledToken: ids.speak, index: 0, tokens: ids,
            config: MiniCPMDuplexConfig(maxNewTokens: 4, forceListenCount: 0))
        let step = state.consume(
            sampledToken: ids.listen, index: 1, tokens: ids,
            config: MiniCPMDuplexConfig(maxNewTokens: 4, forceListenCount: 0),
            allowListen: true)

        XCTAssertEqual(step.token, ids.ttsBOS)
        XCTAssertFalse(step.isListen)
        XCTAssertFalse(step.shouldStop)
        XCTAssertTrue(step.coercedListen)
        XCTAssertEqual(state.generatedIDs, [ids.ttsBOS])
        XCTAssertTrue(state.trace.contains { $0.name == "coerce_listen" })
    }

    func testProtocolTraceForceListenIsOneGenerationAndRollbackRestoresTurn() {
        let ids = protocolIDs()
        var state = MiniCPMDuplexProtocolState()
        _ = state.beginUnit()
        _ = state.beginGeneration(
            config: MiniCPMDuplexConfig(maxNewTokens: 3, forceListenCount: 1))
        let forced = state.consume(
            sampledToken: 42, index: 0, tokens: ids,
            config: MiniCPMDuplexConfig(maxNewTokens: 3, forceListenCount: 1))
        XCTAssertTrue(forced.isListen)
        XCTAssertTrue(forced.shouldStop)
        state.commitUnit(unitEndID: ids.unitEnd)

        _ = state.beginUnit()
        _ = state.beginGeneration(
            config: MiniCPMDuplexConfig(maxNewTokens: 3, forceListenCount: 1))
        let natural = state.consume(
            sampledToken: ids.speak, index: 0, tokens: ids,
            config: MiniCPMDuplexConfig(maxNewTokens: 3, forceListenCount: 1))
        XCTAssertFalse(natural.isListen)
        XCTAssertFalse(natural.shouldStop)
        XCTAssertTrue(state.rollbackUnit())
        XCTAssertTrue(state.currentTurnEnded)
        // Fixed upstream finalizes a forced-listen unit with both the
        // deferred listen marker and </unit>; rollback of the next unit must
        // preserve that committed sequence.
        XCTAssertEqual(state.totalIDs, [ids.listen, ids.unitEnd])
        XCTAssertEqual(state.generateCount, 1)
    }

    func testOfficialDefaultsDoNotAddStartupListenDelay() {
        let defaults = MiniCPMDuplexConfig(maxNewTokens: 3)
        XCTAssertEqual(defaults.forceListenCount, 0)
        XCTAssertEqual(defaults.topK, 20)

        let ids = protocolIDs()
        let config = MiniCPMDuplexConfig(maxNewTokens: 3, forceListenCount: 3)
        var state = MiniCPMDuplexProtocolState()

        for _ in 0 ..< 3 {
            _ = state.beginUnit()
            _ = state.beginGeneration(config: config)
            let forced = state.consume(
                sampledToken: ids.speak,
                index: 0,
                tokens: ids,
                config: config)
            XCTAssertTrue(forced.isListen)
            XCTAssertTrue(forced.shouldStop)
            state.commitUnit(unitEndID: ids.unitEnd)
        }

        _ = state.beginUnit()
        _ = state.beginGeneration(config: config)
        let natural = state.consume(
            sampledToken: ids.speak,
            index: 0,
            tokens: ids,
            config: config)
        XCTAssertFalse(natural.isListen)
        XCTAssertFalse(natural.shouldStop)
        XCTAssertEqual(state.generateCount, 4)
    }

    func testRecentSpeechOverridesOnlyLeadingGreedyDecision() {
        XCTAssertTrue(MiniCPMNativeDuplexEngine.shouldUseGreedyListenSpeakDecision(
            currentTurnEnded: true,
            configuredGreedy: true,
            allowSampledDecision: false))
        XCTAssertFalse(MiniCPMNativeDuplexEngine.shouldUseGreedyListenSpeakDecision(
            currentTurnEnded: true,
            configuredGreedy: true,
            allowSampledDecision: true))
        XCTAssertFalse(MiniCPMNativeDuplexEngine.shouldUseGreedyListenSpeakDecision(
            currentTurnEnded: false,
            configuredGreedy: true,
            allowSampledDecision: false))
    }

    func testResponseStarvationGuardIgnoresStartupSilence() {
        let defaults = MiniCPMResponseStarvationGuard()
        XCTAssertEqual(defaults.scaleAfterListenUnits, 3)
        XCTAssertEqual(defaults.forbidAfterListenUnits, 6)

        var guardState = MiniCPMResponseStarvationGuard(
            scaleAfterListenUnits: 2,
            forbidAfterListenUnits: 4)
        for _ in 0 ..< 8 {
            guardState.observeSpeechActivity(false)
            guardState.commit(isListen: true, endOfTurn: false)
        }
        XCTAssertEqual(guardState.intervention(currentTurnEnded: true), .none)
        XCTAssertFalse(guardState.hasPendingUserSpeech)
        XCTAssertEqual(guardState.consecutiveListenUnits, 0)
    }

    func testResponseStarvationGuardEscalatesOnlyAfterUserSpeech() {
        var guardState = MiniCPMResponseStarvationGuard(
            scaleAfterListenUnits: 2,
            forbidAfterListenUnits: 4,
            rescueListenScale: 0.5)
        guardState.observeSpeechActivity(true)
        XCTAssertEqual(guardState.intervention(currentTurnEnded: true), .none)

        guardState.commit(isListen: true, endOfTurn: false)
        XCTAssertEqual(guardState.intervention(currentTurnEnded: true), .none)
        guardState.commit(isListen: true, endOfTurn: false)
        XCTAssertEqual(
            guardState.intervention(currentTurnEnded: true),
            .scaleListen(0.5))
        XCTAssertEqual(guardState.intervention(currentTurnEnded: false), .none)

        guardState.commit(isListen: true, endOfTurn: false)
        guardState.commit(isListen: true, endOfTurn: false)
        XCTAssertEqual(
            guardState.intervention(currentTurnEnded: true),
            .forbidListen)
    }

    func testResponseStarvationGuardClearsOnSpeechTurnEndAndReset() {
        var guardState = MiniCPMResponseStarvationGuard(
            scaleAfterListenUnits: 1,
            forbidAfterListenUnits: 2)
        guardState.observeSpeechActivity(true)
        guardState.commit(isListen: true, endOfTurn: false)
        XCTAssertNotEqual(guardState.intervention(currentTurnEnded: true), .none)

        guardState.commit(isListen: false, endOfTurn: false)
        XCTAssertEqual(guardState.intervention(currentTurnEnded: true), .none)

        guardState.observeSpeechActivity(true)
        guardState.commit(isListen: true, endOfTurn: true)
        XCTAssertEqual(guardState.intervention(currentTurnEnded: true), .none)

        guardState.observeSpeechActivity(true)
        guardState.commit(isListen: true, endOfTurn: false)
        guardState.reset()
        XCTAssertEqual(guardState.intervention(currentTurnEnded: true), .none)
    }

    func testForceListenOverrideDoesNotConsumeConfiguredCounter() {
        let ids = protocolIDs()
        var state = MiniCPMDuplexProtocolState()
        _ = state.beginUnit()
        _ = state.beginGeneration(
            config: MiniCPMDuplexConfig(forceListenCount: 0),
            forceListenOverride: true)
        let forced = state.consume(
            sampledToken: ids.speak,
            index: 0,
            tokens: ids,
            config: MiniCPMDuplexConfig(forceListenCount: 0))
        XCTAssertTrue(forced.isListen)
        XCTAssertTrue(forced.shouldStop)

        state.commitUnit(unitEndID: ids.unitEnd)
        _ = state.beginUnit()
        _ = state.beginGeneration(
            config: MiniCPMDuplexConfig(forceListenCount: 0))
        let next = state.consume(
            sampledToken: ids.speak,
            index: 0,
            tokens: ids,
            config: MiniCPMDuplexConfig(forceListenCount: 0))
        XCTAssertFalse(next.isListen)
        XCTAssertFalse(next.shouldStop)
    }

    func testForceListenOverrideClosesOpenTurnBeforeListen() {
        let ids = protocolIDs()
        var state = MiniCPMDuplexProtocolState()
        _ = state.beginUnit()
        _ = state.beginGeneration(config: MiniCPMDuplexConfig(maxNewTokens: 3, forceListenCount: 0))
        _ = state.consume(
            sampledToken: ids.speak,
            index: 0,
            tokens: ids,
            config: MiniCPMDuplexConfig(maxNewTokens: 3, forceListenCount: 0))
        _ = state.consume(
            sampledToken: 20,
            index: 1,
            tokens: ids,
            config: MiniCPMDuplexConfig(maxNewTokens: 3, forceListenCount: 0))
        XCTAssertFalse(state.currentTurnEnded)

        XCTAssertTrue(state.closeTurnForForcedListen(turnEOS: ids.turnEOS))
        XCTAssertTrue(state.currentTurnEnded)
        _ = state.beginGeneration(
            config: MiniCPMDuplexConfig(maxNewTokens: 3, forceListenCount: 0),
            forceListenOverride: true)
        let listen = state.consume(
            sampledToken: 20,
            index: 0,
            tokens: ids,
            config: MiniCPMDuplexConfig(maxNewTokens: 3, forceListenCount: 0))
        XCTAssertTrue(listen.isListen)
        XCTAssertTrue(listen.shouldStop)
    }

    func testInterruptedTurnClosesAfterPendingUnitRollback() {
        let ids = protocolIDs()
        var state = MiniCPMDuplexProtocolState()
        let config = MiniCPMDuplexConfig(maxNewTokens: 4, forceListenCount: 0)

        // First spoken unit is already accepted, but its turn remains open.
        _ = state.beginUnit()
        _ = state.beginGeneration(config: config)
        _ = state.consume(
            sampledToken: ids.speak, index: 0, tokens: ids, config: config)
        _ = state.consume(
            sampledToken: 42, index: 1, tokens: ids, config: config)
        state.commitUnit(unitEndID: ids.unitEnd)

        // A second unit is provisional when cancel arrives. Rollback restores
        // the open-turn snapshot; interrupt must then close that turn instead
        // of returning merely because a pending unit existed.
        _ = state.beginUnit()
        _ = state.beginGeneration(config: config)
        _ = state.consume(
            sampledToken: ids.speak, index: 0, tokens: ids, config: config)
        XCTAssertFalse(state.currentTurnEnded)
        XCTAssertTrue(state.rollbackUnit())
        XCTAssertFalse(state.currentTurnEnded)

        state.closeInterruptedTurn(turnEOS: ids.turnEOS, unitEnd: ids.unitEnd)
        XCTAssertTrue(state.currentTurnEnded)
        XCTAssertEqual(state.unitIndex, 1)
        XCTAssertEqual(state.totalIDs.suffix(2), [ids.turnEOS, ids.unitEnd])
        XCTAssertEqual(state.responseIDs.last, ids.turnEOS)
    }

    func testPendingSpokenRollbackReplacesCommittedListenBoundaryExactlyOnce() {
        let ids = protocolIDs()
        let config = MiniCPMDuplexConfig(maxNewTokens: 4, forceListenCount: 0)
        var state = MiniCPMDuplexProtocolState()

        // The first committed unit is a listen boundary. This is the state
        // seen by a real barge-in after the initial warm-up units: the next
        // spoken response is provisional, but currentTurnEnded is true until
        // that response actually samples a spoken token.
        _ = state.beginUnit()
        _ = state.beginGeneration(config: config, forceListenOverride: true)
        let listen = state.consume(
            sampledToken: ids.speak, index: 0, tokens: ids, config: config)
        XCTAssertTrue(listen.isListen)
        state.commitUnit(unitEndID: ids.unitEnd)

        _ = state.beginUnit()
        _ = state.beginGeneration(config: config)
        let spoken = state.consume(
            sampledToken: ids.speak, index: 0, tokens: ids, config: config)
        XCTAssertFalse(spoken.isListen)
        XCTAssertTrue(state.rollbackUnit())
        XCTAssertTrue(state.currentTurnEnded)

        XCTAssertTrue(state.closeInterruptedTurn(
            turnEOS: ids.turnEOS, unitEnd: ids.unitEnd, force: true))
        XCTAssertEqual(state.totalIDs, [ids.turnEOS, ids.unitEnd])
        XCTAssertFalse(state.closeInterruptedTurn(
            turnEOS: ids.turnEOS, unitEnd: ids.unitEnd, force: true))
        XCTAssertEqual(state.totalIDs, [ids.turnEOS, ids.unitEnd])
        XCTAssertEqual(
            state.trace.filter { $0.name == "interrupt_turn_eos" }.count,
            1)
    }

    func testCancelledMultiUnitTurnRestoresBeforeForcedListenFence() {
        let ids = protocolIDs()
        let config = MiniCPMDuplexConfig(maxNewTokens: 4, forceListenCount: 0)
        var state = MiniCPMDuplexProtocolState()
        let turnStart = state

        // A real streamed question may already have several accepted listen
        // units before the first spoken response becomes provisional.
        for _ in 0 ..< 3 {
            _ = state.beginUnit()
            _ = state.beginGeneration(config: config)
            let listen = state.consume(
                sampledToken: ids.listen,
                index: 0,
                tokens: ids,
                config: config)
            XCTAssertTrue(listen.isListen)
            state.commitUnit(unitEndID: ids.unitEnd)
        }

        // The response contains an old-turn token, but is not committed when
        // cancel arrives.  The turn-scope checkpoint must remove both this
        // provisional response and the earlier accepted input units.
        _ = state.beginUnit()
        _ = state.beginGeneration(config: config)
        _ = state.consume(
            sampledToken: ids.speak,
            index: 0,
            tokens: ids,
            config: config)
        let oldIntent = state.consume(
            sampledToken: 42,
            index: 1,
            tokens: ids,
            config: config)
        XCTAssertFalse(oldIntent.isListen)
        XCTAssertTrue(state.totalIDs.contains(42))

        // This is the engine's active-turn restore, not the narrower
        // per-unit rollback.  No old input or response token may survive.
        state = turnStart
        XCTAssertTrue(state.totalIDs.isEmpty)
        XCTAssertFalse(state.responseIDs.contains(42))
        XCTAssertEqual(state.generateCount, 0)

        _ = state.beginUnit()
        _ = state.beginGeneration(config: config, forceListenOverride: true)
        let fence = state.consume(
            sampledToken: 42,
            index: 0,
            tokens: ids,
            config: config)
        XCTAssertTrue(fence.isListen)
        state.commitUnit(unitEndID: ids.unitEnd)

        // The following input is free to begin a new answer; it must not be
        // coerced into the cancelled response or inherit its generated IDs.
        _ = state.beginUnit()
        _ = state.beginGeneration(config: config)
        let next = state.consume(
            sampledToken: ids.speak,
            index: 0,
            tokens: ids,
            config: config)
        XCTAssertFalse(next.isListen)
        XCTAssertFalse(state.generatedIDs.contains(42))
    }

    func testCharacterLimitDefersChunkEOSWithoutFeedingIt() {
        let ids = protocolIDs()
        var state = MiniCPMDuplexProtocolState()
        _ = state.beginUnit()
        _ = state.beginGeneration(config: MiniCPMDuplexConfig(maxNewTokens: 4, forceListenCount: 0))
        _ = state.consume(
            sampledToken: ids.speak,
            index: 0,
            tokens: ids,
            config: MiniCPMDuplexConfig(maxNewTokens: 4, forceListenCount: 0))
        let step = state.finishAtCharacterLimit(
            index: 1,
            tokens: ids,
            config: MiniCPMDuplexConfig(maxNewTokens: 4, forceListenCount: 0))
        XCTAssertTrue(step.shouldStop)
        XCTAssertFalse(step.shouldFeed)
        XCTAssertEqual(step.token, ids.chunkEOS)
        XCTAssertTrue(state.trace.contains { $0.name == "character_limit" })
    }

    func testAudioHistoryRetainsIncompleteFirstChunkAcrossNextWindowAndRollback() {
        var history = MiniCPMDuplexAudioHistory()
        let first = [Float](repeating: 1, count: 16_000)
        let firstRemainder = [Float](repeating: 1.5, count: 480)
        let second = [Float](repeating: 2, count: 16_000)

        history.beginUnit()
        history.accept(first, producedEmbedding: false)
        XCTAssertEqual(history.replayChunks.count, 1)
        // The short first window has no embedding yet; do not commit it as a
        // language unit, but keep it in the frontend tail.
        XCTAssertEqual(history.replayChunks, [first])
        history.preserveTailAfterRejectedUnit()

        history.beginUnit()
        history.accept(firstRemainder, producedEmbedding: true)
        history.accept(second, producedEmbedding: true)
        history.commitUnit()
        // Accepted PCM is represented by the audio-session checkpoint, not a
        // second unbounded waveform history.
        XCTAssertTrue(history.committedChunks.isEmpty)
        XCTAssertTrue(history.tailChunks.isEmpty)

        history.beginUnit()
        history.accept([3, 4], producedEmbedding: true)
        XCTAssertTrue(history.rollbackUnit())
        XCTAssertTrue(history.committedChunks.isEmpty)
        XCTAssertTrue(history.tailChunks.isEmpty)
    }

    func testAudioHistoryBoundsUnembeddedTail() {
        var history = MiniCPMDuplexAudioHistory()
        history.beginUnit()
        let packet = [Float](repeating: 1, count: 16_000)
        for _ in 0 ..< 20 {
            history.accept(packet, producedEmbedding: false)
            let samples = history.tailChunks.reduce(0) { $0 + $1.count }
            XCTAssertLessThanOrEqual(
                samples, MiniCPMDuplexAudioHistory.maximumTailSamples)
        }
        XCTAssertEqual(
            history.tailChunks.reduce(0) { $0 + $1.count },
            MiniCPMDuplexAudioHistory.maximumTailSamples)
        XCTAssertTrue(history.committedChunks.isEmpty)
    }

    func testReferenceChannelsRemainDistinctAndLegacyReferenceFansOut() {
        let llm = [Float](repeating: 1, count: 32)
        let tts = [Float](repeating: 2, count: 48)
        let config = MiniCPMDuplexConfig(
            generateAudio: false,
            llmReferenceAudio: llm,
            llmReferenceAudioSampleRate: 16_000,
            ttsReferenceAudio: tts,
            ttsReferenceAudioSampleRate: 24_000)
        XCTAssertEqual(config.llmReferenceAudio, llm)
        XCTAssertEqual(config.llmReferenceAudioSampleRate, 16_000)
        XCTAssertEqual(config.ttsReferenceAudio, tts)
        XCTAssertEqual(config.ttsReferenceAudioSampleRate, 24_000)
        XCTAssertNil(config.referenceAudio)

        let legacy = MiniCPMDuplexConfig(referenceAudio: llm)
        XCTAssertEqual(legacy.llmReferenceAudio, llm)
        XCTAssertEqual(legacy.ttsReferenceAudio, llm)
        XCTAssertEqual(legacy.llmReferenceAudioSampleRate, 16_000)
        XCTAssertEqual(legacy.ttsReferenceAudioSampleRate, 16_000)
    }

    func testLegacyConfigArgumentOrderRemainsStable() {
        // Keep the original arguments in declaration order. New sampler/TTS
        // controls are appended after this call's final legacy argument, so
        // older source that supplies the full initializer remains valid.
        let config = MiniCPMDuplexConfig(
            maxNewTokens: 11,
            temperature: 0.55,
            topK: 9,
            topP: 0.72,
            slidingWindowTokens: 512,
            lsMode: "explicit",
            forceListenCount: 2,
            listenProbabilityScale: 0.8,
            listenTopK: 4,
            repetitionPenalty: 1.08,
            repetitionWindowSize: 128,
            generateAudio: false,
            systemPrompt: "legacy",
            referenceAudio: [0.25, -0.25],
            referenceAudioSampleRate: 22_050)

        XCTAssertEqual(config.maxNewTokens, 11)
        XCTAssertEqual(config.temperature, 0.55)
        XCTAssertEqual(config.topK, 9)
        XCTAssertEqual(config.topP, 0.72)
        XCTAssertEqual(config.slidingWindowTokens, 512)
        XCTAssertEqual(config.forceListenCount, 2)
        XCTAssertEqual(config.listenProbabilityScale, 0.8)
        XCTAssertEqual(config.listenTopK, 4)
        XCTAssertEqual(config.repetitionPenalty, 1.08)
        XCTAssertEqual(config.repetitionWindowSize, 128)
        XCTAssertFalse(config.generateAudio)
        XCTAssertEqual(config.systemPrompt, "legacy")
        XCTAssertEqual(config.referenceAudioSampleRate, 22_050)
        XCTAssertEqual(config.llmReferenceAudio, [0.25, -0.25])
        XCTAssertEqual(config.ttsReferenceAudio, [0.25, -0.25])
        XCTAssertEqual(config.lengthPenalty, 1.1)
        XCTAssertEqual(config.ttsTemperature, 0.8)
        XCTAssertEqual(config.ttsRepetitionPenalty, 1.05)
        XCTAssertFalse(config.greedyListenSpeakDecision)

        let sampledDecisions = MiniCPMDuplexConfig(greedyListenSpeakDecision: false)
        XCTAssertFalse(sampledDecisions.greedyListenSpeakDecision)
    }

    func testContinuationOnlyAcceptsOnlyZeroAudioDuringOpenSpokenTurn() {
        XCTAssertTrue(MiniCPMNativeDuplexEngine.shouldUseContinuationOnly(
            requested: true,
            currentTurnEnded: false,
            audio: [0, 0, 0],
            hasFrames: false,
            hasText: false))
        XCTAssertFalse(MiniCPMNativeDuplexEngine.shouldUseContinuationOnly(
            requested: true,
            currentTurnEnded: true,
            audio: [0, 0, 0],
            hasFrames: false,
            hasText: false))
        XCTAssertFalse(MiniCPMNativeDuplexEngine.shouldUseContinuationOnly(
            requested: true,
            currentTurnEnded: false,
            audio: [0, 0.01, 0],
            hasFrames: false,
            hasText: false))
        XCTAssertFalse(MiniCPMNativeDuplexEngine.shouldUseContinuationOnly(
            requested: true,
            currentTurnEnded: false,
            audio: [0, 0, 0],
            hasFrames: false,
            hasText: true))
        XCTAssertFalse(MiniCPMNativeDuplexEngine.shouldUseContinuationOnly(
            requested: false,
            currentTurnEnded: false,
            audio: [0, 0, 0],
            hasFrames: false,
            hasText: false))
    }

    func testSlidingWindowUsesExplicitWatermarksAndContextLimits() {
        let config = MiniCPMDuplexConfig(
            slidingWindowMode: "basic",
            basicWindowHighTokens: 8_000,
            basicWindowLowTokens: 6_000,
            contextPreviousMaxTokens: 500,
            contextMaxUnits: 24,
            contextPreviousMarker: "\n\nprevious: ")
        XCTAssertEqual(config.slidingWindowMode, "basic")
        XCTAssertEqual(config.basicWindowHighTokens, 8_000)
        XCTAssertEqual(config.basicWindowLowTokens, 6_000)
        XCTAssertEqual(config.contextPreviousMaxTokens, 500)
        XCTAssertEqual(config.contextMaxUnits, 24)

        // Legacy callers now map their cap directly to a high/low watermark;
        // no hidden `/256` unit conversion is allowed.
        let legacy = MiniCPMDuplexConfig(slidingWindowTokens: 512)
        XCTAssertEqual(legacy.slidingWindowMode, "basic")
        XCTAssertEqual(legacy.basicWindowHighTokens, 512)
        XCTAssertEqual(legacy.basicWindowLowTokens, 512)
    }

    func testEmptyMaxSliceOverrideMeansDefaultAndMismatchedListsFailClosed() throws {
        XCTAssertNil(MiniCPMNativeDuplexEngine.normalizedMaxSliceNums([]))
        XCTAssertEqual(
            MiniCPMNativeDuplexEngine.normalizedMaxSliceNums([3]),
            [3])
        XCTAssertNil(MiniCPMNativeDuplexEngine.normalizedMaxSliceNums(nil))

        // Empty input is normalized before the shape check and therefore uses
        // the processor's default slicing behavior.
        try MiniCPMNativeDuplexEngine.validateMaxSliceNums(nil, frameCount: 2)
        try MiniCPMNativeDuplexEngine.validateMaxSliceNums([3], frameCount: 2)
        try MiniCPMNativeDuplexEngine.validateMaxSliceNums([3, 4], frameCount: 2)

        XCTAssertThrowsError(
            try MiniCPMNativeDuplexEngine.validateMaxSliceNums([3, 4], frameCount: 1)) { error in
                guard case MiniCPMNativeDuplexEngineError.invalidInput(let message) = error else {
                    return XCTFail("unexpected error: \(error)")
                }
                XCTAssertTrue(message.contains("one value per frame"))
            }
        XCTAssertThrowsError(
            try MiniCPMNativeDuplexEngine.validateMaxSliceNums([0], frameCount: 2)) { error in
                guard case MiniCPMNativeDuplexEngineError.invalidInput(let message) = error else {
                    return XCTFail("unexpected error: \(error)")
                }
                XCTAssertTrue(message.contains("positive"))
            }
    }

    func testTTSPadIsPinnedInForbiddenSetLikeFixedUnifiedSource() {
        let ttsPad = 17
        XCTAssertEqual(
            MiniCPMNativeDuplexEngine.forbiddenTokenIDs(
                badTokenIds: [4, 9], ttsPadID: ttsPad),
            [4, 9, ttsPad])
        XCTAssertEqual(
            MiniCPMNativeDuplexEngine.forbiddenTokenIDs(
                badTokenIds: [4, ttsPad, 9], ttsPadID: ttsPad),
            [4, ttsPad, 9])
        XCTAssertEqual(
            MiniCPMNativeDuplexEngine.forbiddenTokenIDs(
                badTokenIds: [4, 9], ttsPadID: -1),
            [4, 9])
    }

    func testOrderedMessagesPreserveRolesAndPartOrder() {
        let messages = [
            MiniCPMDuplexMessage(
                role: "user",
                parts: [
                    .text("先看这段音频"),
                    .audio(samples: [0.1, 0.2], sampleRate: 16_000),
                    .text("再看图片"),
                    .image(Data([1, 2, 3])),
                ]),
        ]
        let input = MiniCPMDuplexInput(messages: messages)
        XCTAssertFalse(input.isEmpty)
        XCTAssertEqual(input.messages?.first?.role, "user")
        XCTAssertEqual(input.messages?.first?.parts.count, 4)
        if case let .audio(samples, rate)? = input.messages?.first?.parts[1] {
            XCTAssertEqual(samples, [0.1, 0.2])
            XCTAssertEqual(rate, 16_000)
        } else {
            XCTFail("audio part order was not preserved")
        }
    }

    func testDeferredFinalizeAndPerSessionKVBoundary() async throws {
        var engines: [RecordingEngine] = []
        let runtime = MiniCPMDuplexRuntime {
            let engine = RecordingEngine()
            engines.append(engine)
            return engine
        }

        _ = try await runtime.prepare(sessionID: "a", mode: .halfDuplex)
        _ = try await runtime.prefill(MiniCPMDuplexInput(text: "hello"), sessionID: "a")
        do {
            _ = try await runtime.prepare(sessionID: "a", mode: .halfDuplex)
            XCTFail("prepare must not discard an ungenerated prefill")
        } catch {
            // Expected pending-prefill state.
        }
        let output = try await runtime.generate(sessionID: "a")
        XCTAssertTrue(output.needsFinalize)
        do {
            _ = try await runtime.prefill(
                MiniCPMDuplexInput(text: "next"), sessionID: "a")
            XCTFail("prefill should wait for deferred finalize")
        } catch {
            // Expected pending-finalize state.
        }
        try await runtime.finalize(sessionID: "a")
        _ = try await runtime.prefill(MiniCPMDuplexInput(text: "next"), sessionID: "a")
        XCTAssertEqual(engines.count, 1)
        XCTAssertEqual(engines[0].events, ["prepare:half_duplex", "window:nil", "prefill:half_duplex", "generate:half_duplex", "finalize", "prefill:half_duplex"])
    }

    func testNonisolatedInterruptSignalStopsSynchronousGenerateAndNewEpochSurvives() async throws {
        let engine = ProbeBlockingEngine()
        let runtime = MiniCPMDuplexRuntime(engineFactory: { engine })
        _ = try await runtime.prepare(sessionID: "signal", mode: .fullDuplex)
        _ = try await runtime.prefill(MiniCPMDuplexInput(text: "first"), sessionID: "signal")

        let firstTask = Task { try await runtime.generate(sessionID: "signal") }
        XCTAssertEqual(engine.generationStarted.wait(timeout: .now() + 2), .success)
        // This call must not await the actor: it advances the lock-protected
        // epoch while the actor is still inside ProbeBlockingEngine.generate.
        runtime.requestInterrupt(sessionID: "signal")
        let interrupted = try await firstTask.value
        XCTAssertEqual(interrupted.finishReason, "interrupt")
        let interruptedSnapshot = try await runtime.snapshot(sessionID: "signal")
        XCTAssertTrue(interruptedSnapshot.interrupted)
        XCTAssertFalse(interruptedSnapshot.prefetched)

        // A subsequent prepare captures the new epoch. The prior cancel must
        // not be consumed by this generation.
        _ = try await runtime.prepare(sessionID: "signal", mode: .fullDuplex)
        _ = try await runtime.prefill(MiniCPMDuplexInput(text: "second"), sessionID: "signal")
        let fresh = try await runtime.generate(sessionID: "signal")
        XCTAssertEqual(fresh.text, "fresh")
        XCTAssertNotEqual(fresh.finishReason, "interrupt")
    }

    func testInFlightPrefillInterruptIsNonFatalAndNextPrefillCanContinue() async throws {
        let engine = ProbeBlockingPrefillEngine()
        let runtime = MiniCPMDuplexRuntime(engineFactory: { engine })
        _ = try await runtime.prepare(sessionID: "prefill-signal", mode: .fullDuplex)

        let first = Task {
            try await runtime.prefill(
                MiniCPMDuplexInput(text: "first"), sessionID: "prefill-signal")
        }
        XCTAssertEqual(engine.prefillStarted.wait(timeout: .now() + 2), .success)

        // This must advance the lock-protected epoch without waiting for the
        // actor currently blocked inside the synchronous prefill.
        runtime.requestInterrupt(sessionID: "prefill-signal")
        let cancelled = try await first.value
        XCTAssertFalse(cancelled.accepted)

        let interrupted = try await runtime.snapshot(sessionID: "prefill-signal")
        XCTAssertTrue(interrupted.interrupted)
        XCTAssertFalse(interrupted.prefetched)
        XCTAssertFalse(interrupted.needsFinalize)
        XCTAssertGreaterThanOrEqual(engine.interruptCount, 1)

        // No re-prepare is required after a normal barge-in. The next input
        // captures the new epoch and reaches the same model session.
        let next = try await runtime.prefill(
            MiniCPMDuplexInput(text: "second"), sessionID: "prefill-signal")
        XCTAssertTrue(next.accepted)
        let output = try await runtime.generate(sessionID: "prefill-signal")
        XCTAssertEqual(output.text, "fresh")
        XCTAssertNotEqual(output.finishReason, "interrupt")
    }

    func testInterruptRollsBackOnlyTargetSessionAndSlidingWindowIsLocal() async throws {
        var engines: [RecordingEngine] = []
        let runtime = MiniCPMDuplexRuntime {
            let engine = RecordingEngine()
            engines.append(engine)
            return engine
        }
        _ = try await runtime.prepare(sessionID: "a", mode: .fullDuplex, config: MiniCPMDuplexConfig(slidingWindowTokens: 128))
        _ = try await runtime.prepare(sessionID: "b", mode: .turnBased)
        _ = try await runtime.prefill(MiniCPMDuplexInput(audio: [0, 1]), sessionID: "a")
        _ = try await runtime.generate(sessionID: "a")
        _ = try await runtime.prefill(MiniCPMDuplexInput(text: "b"), sessionID: "b")
        try await runtime.interrupt(sessionID: "a")
        let a = try await runtime.snapshot(sessionID: "a")
        let b = try await runtime.snapshot(sessionID: "b")
        XCTAssertTrue(a.interrupted)
        XCTAssertFalse(a.needsFinalize)
        XCTAssertTrue(b.prefetched)
        XCTAssertEqual(a.slidingWindowTokens, 128)
        XCTAssertTrue(engines[0].events.contains("rollback"))
        XCTAssertFalse(engines[1].events.contains("rollback"))
    }

    func testResetAndCloseReleaseSessionState() async throws {
        let runtime = MiniCPMDuplexRuntime()
        _ = try await runtime.prepare(sessionID: "x", mode: .omni)
        try await runtime.reset(sessionID: "x")
        let reset = try await runtime.snapshot(sessionID: "x")
        XCTAssertFalse(reset.prepared)
        try await runtime.close(sessionID: "x")
        let remainingSessionIDs = await runtime.sessionIDs()
        XCTAssertEqual(remainingSessionIDs, [])
    }

    func testCloseRollsBackThenReleasesEngineExactlyOnce() async throws {
        var engines: [RecordingEngine] = []
        let runtime = MiniCPMDuplexRuntime {
            let engine = RecordingEngine()
            engines.append(engine)
            return engine
        }

        _ = try await runtime.prepare(sessionID: "close", mode: .fullDuplex)
        _ = try await runtime.prefill(
            MiniCPMDuplexInput(audio: [0.1, 0.2]), sessionID: "close")
        _ = try await runtime.generate(sessionID: "close")
        try await runtime.close(sessionID: "close")

        XCTAssertEqual(engines.count, 1)
        XCTAssertEqual(
            engines[0].events.filter { $0 == "reset" }.count,
            1,
            "close must release mutable model state exactly once")
        XCTAssertEqual(engines[0].events.last, "reset")
        let remainingSessionIDs = await runtime.sessionIDs()
        XCTAssertEqual(remainingSessionIDs, [])
    }

    func testRepeatGuardDetectsShortPhraseLoopOnlyAfterFourRepetitions() {
        var guardState = MiniCPMResponseRepeatGuard()
        let phrase = "记住了吗"
        for _ in 0..<3 {
            guardState.append(text: phrase + "。")
            XCTAssertFalse(guardState.tripped, "three repeats must not trip the guard")
        }
        guardState.append(text: phrase + "。")
        XCTAssertTrue(guardState.tripped, "four consecutive phrase repeats must trip the guard")
        guardState.reset()
        XCTAssertFalse(guardState.tripped)
    }

    func testRepeatGuardDetectsSentenceLengthLoop() {
        var guardState = MiniCPMResponseRepeatGuard()
        let sentence = "小猫把一颗红球放在窗台上了"  // 12+ normalized characters
        for _ in 0..<2 {
            guardState.append(text: sentence + "。")
            XCTAssertFalse(guardState.tripped, "two sentence repeats must not trip the guard")
        }
        guardState.append(text: sentence + "。")
        XCTAssertTrue(guardState.tripped, "three consecutive sentence repeats must trip the guard")
    }

    func testRepeatGuardIgnoresShortTextAndSingleCharacterRuns() {
        var guardState = MiniCPMResponseRepeatGuard()
        // "好的" repeated twice per unit is a common natural acknowledgement;
        // it only becomes a loop after the same four-in-a-row evidence.
        guardState.append(text: "好的，我明白了。")
        guardState.append(text: "好的，就这么办。")
        XCTAssertFalse(guardState.tripped, "natural acknowledgements are not decoder loops")
        // Single-character runs (laughter, singing) never trip the guard.
        var laughter = MiniCPMResponseRepeatGuard()
        for _ in 0..<6 { laughter.append(text: "哈哈哈哈") }
        XCTAssertFalse(laughter.tripped, "single-character runs are not decoder loops")
        // Text below the minimum evidence window is ignored entirely.
        var short = MiniCPMResponseRepeatGuard()
        short.append(text: "哈哈")
        short.append(text: "哈哈")
        short.append(text: "哈哈")
        XCTAssertFalse(short.tripped, "text shorter than the detection window is ignored")
    }

    func testRepeatGuardUnitCapTripsWithoutAnyPhraseRepeat() {
        var guardState = MiniCPMResponseRepeatGuard()
        for index in 0..<(MiniCPMResponseRepeatGuard.maxUnitsPerResponse - 1) {
            guardState.append(text: "第\(index)个互不相同的句子，没有循环。")
            XCTAssertFalse(guardState.tripped)
        }
        guardState.append(text: "最后一个单元触发回答长度上限。")
        XCTAssertTrue(guardState.tripped, "the upstream unit cap must end a runaway response")
    }
}
