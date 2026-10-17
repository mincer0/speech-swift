import XCTest
import MLX
@testable import MiniCPMLLM

final class MiniCPMSamplerTests: XCTestCase {
    private let protocolIDs = MiniCPMProtocolTokenIds(
        listen: 1,
        chunkEOS: 2,
        chunkTTSEOS: 3,
        turnEOS: 4,
        speak: 5,
        unitEnd: 6,
        ttsBOS: 7,
        allSpecialTokenIds: Set([1, 2, 3, 4, 5, 6, 7, 8]))

    func testSamplerStateIsDeterministicAndAdvances() {
        var left = MiniCPMSamplerState(seed: 0x1234)
        var right = MiniCPMSamplerState(seed: 0x1234)
        let first = left.nextUniform()
        XCTAssertEqual(first, right.nextUniform())
        XCTAssertEqual(left.nextUniform(), right.nextUniform())
        XCTAssertNotEqual(left.rawState, MiniCPMSamplerState(seed: 0x1234).rawState)
    }

    func testSamplerStateSnapshotRestoresDraw() {
        var state = MiniCPMSamplerState(seed: 99)
        _ = state.nextUniform()
        let snapshot = state.snapshot()
        let expected = state.nextUniform()
        _ = state.nextUniform()
        state.restore(snapshot)
        XCTAssertEqual(state.nextUniform(), expected)
    }

    func testSamplerDoesNotUseFixedHalfQuantile() {
        let config = MiniCPMSamplingConfig(
            temperature: 1, topK: 0, topP: 1, repetitionPenalty: 1,
            lengthPenalty: 1)
        var state = MiniCPMSamplerState(seed: 1)
        let logits = MLXArray([Float(0), 0, 0, 0, 0, 0, 0, 0, 0])
        let before = state.rawState
        _ = MiniCPMSampler.sample(
            logits: logits, config: config, protocol: protocolIDs, state: &state)
        let afterOne = state.rawState
        _ = MiniCPMSampler.sample(
            logits: logits, config: config, protocol: protocolIDs, state: &state)
        XCTAssertNotEqual(before, afterOne)
        XCTAssertNotEqual(afterOne, state.rawState)
    }

    func testBadTokenIsMaskedAfterNaturalProbe() {
        let config = MiniCPMSamplingConfig(
            temperature: 1, topK: 0, topP: 1, repetitionPenalty: 1,
            lengthPenalty: 1)
        // Token 8 wins the natural probe, but is forbidden for the final
        // sample. Token 0 is the next available candidate.
        let logits = MLXArray([Float(0), -10, -10, -10, -10, -10, -10, -10, 10])
        let token = MiniCPMSampler.sample(
            logits: logits, config: config, protocol: protocolIDs,
            forbidden: [8], uniform: 0).item(Int.self)
        XCTAssertEqual(token, 0)
    }

    func testZeroListenScaleCannotUnmaskForbiddenListen() {
        let config = MiniCPMSamplingConfig(
            mode: .greedy, repetitionPenalty: 1, repetitionPolicy: .officialDuplex,
            lengthPenalty: 1, listenProbabilityScale: 0)
        let logits = MLXArray([Float(-1), 100, -100, -100, -100, -100, -100, -100, -100])
        var state = MiniCPMSamplerState(seed: 1)
        XCTAssertEqual(MiniCPMSampler.sample(
            logits: logits, config: config, protocol: protocolIDs,
            forbidden: [protocolIDs.listen], state: &state).item(Int.self), 0)
    }

    func testListenRescueAlwaysLowersPositiveAndNegativeLogits() {
        for listenLogit: Float in [-1, 1] {
            let logits = MLXArray([
                listenLogit - 0.25, listenLogit,
                -100, -100, -100, -100, -100, -100, -100,
            ])
            var config = MiniCPMSamplingConfig(
                mode: .greedy, repetitionPenalty: 1,
                repetitionPolicy: .officialDuplex, lengthPenalty: 1)
            XCTAssertEqual(MiniCPMSampler.sample(
                logits: logits, config: config, protocol: protocolIDs,
                uniform: 0).item(Int.self), protocolIDs.listen)
            config.listenLogProbabilityScale = 0.5
            XCTAssertEqual(MiniCPMSampler.sample(
                logits: logits, config: config, protocol: protocolIDs,
                uniform: 0).item(Int.self), 0)
            config.listenLogProbabilityScale = 0
            XCTAssertEqual(MiniCPMSampler.sample(
                logits: logits, config: config, protocol: protocolIDs,
                uniform: 0).item(Int.self), 0)
            XCTAssertEqual(logits[protocolIDs.listen].item(Float.self), listenLogit)
        }
    }

    func testListenRescueCannotUnmaskForbiddenListen() {
        let config = MiniCPMSamplingConfig(
            mode: .greedy, repetitionPenalty: 1,
            repetitionPolicy: .officialDuplex, lengthPenalty: 1,
            listenLogProbabilityScale: 0)
        let logits = MLXArray([Float(-1), 100, -100, -100, -100, -100, -100, -100, -100])
        XCTAssertEqual(MiniCPMSampler.sample(
            logits: logits, config: config, protocol: protocolIDs,
            forbidden: [protocolIDs.listen], uniform: 0).item(Int.self), 0)
    }

    func testNaturalChunkEOSProbeWinsBeforeFiltering() {
        let config = MiniCPMSamplingConfig(
            temperature: 1, topK: 0, topP: 1, repetitionPenalty: 1,
            lengthPenalty: 1)
        let logits = MLXArray([Float(0), -10, 10, -10, -10, -10, -10, -10, -10])
        let token = MiniCPMSampler.sample(
            logits: logits, config: config, protocol: protocolIDs,
            forbidden: [2], uniform: 0).item(Int.self)
        XCTAssertEqual(token, 2)
    }

    func testProductionStateSuppressesNaturalChunkEOS() {
        let config = MiniCPMSamplingConfig(
            temperature: 1, topK: 1, topP: 1, repetitionPenalty: 1,
            lengthPenalty: 1, suppressNaturalChunkEOS: true)
        let logits = MLXArray([Float(0), -100, 100, -100, -100, -100, -100, -100, -100])
        var state = MiniCPMSamplerState(seed: 1)
        var expectedState = state
        _ = expectedState.nextUniform()
        _ = expectedState.nextUniform()
        let token = MiniCPMSampler.sample(
            logits: logits, config: config, protocol: protocolIDs,
            forbidden: [protocolIDs.chunkEOS], state: &state).item(Int.self)
        XCTAssertEqual(token, 0)
        XCTAssertEqual(state.rawState, expectedState.rawState, "A suppressed probe still consumes the final draw")
        XCTAssertEqual(state.sampledTokens, [0])
    }

    func testProductionGreedySuppressesNaturalChunkEOS() {
        let config = MiniCPMSamplingConfig(
            mode: .greedy, repetitionPenalty: 1, lengthPenalty: 1,
            suppressNaturalChunkEOS: true)
        let logits = MLXArray([Float(0), -100, 100, -100, -100, -100, -100, -100, -100])
        var state = MiniCPMSamplerState(seed: 1)
        let originalState = state
        let token = MiniCPMSampler.sample(
            logits: logits, config: config, protocol: protocolIDs,
            forbidden: [protocolIDs.chunkEOS], state: &state).item(Int.self)
        XCTAssertEqual(token, 0)
        XCTAssertEqual(state.rawState, originalState.rawState)
        XCTAssertEqual(state.sampledTokens, [0])
    }

    func testProductionUnsuppressedNaturalChunkEOSConsumesOneDraw() {
        let config = MiniCPMSamplingConfig(repetitionPenalty: 1, lengthPenalty: 1)
        let logits = MLXArray([Float(0), -100, 100, -100, -100, -100, -100, -100, -100])
        var state = MiniCPMSamplerState(seed: 1)
        var expectedState = state
        _ = expectedState.nextUniform()
        let token = MiniCPMSampler.sample(
            logits: logits, config: config, protocol: protocolIDs,
            forbidden: [protocolIDs.chunkEOS], state: &state).item(Int.self)
        XCTAssertEqual(token, protocolIDs.chunkEOS)
        XCTAssertEqual(state, expectedState)
    }

    func testProductionListenTopKEarlyReturnDoesNotConsumeFinalDrawOrHistory() {
        let config = MiniCPMSamplingConfig(
            temperature: 1, repetitionPolicy: .officialDuplex, lengthPenalty: 1,
            listenTopK: 1)
        let logits = MLXArray([Float(0), 100, -100, -100, -100, -100, -100, -100, -100])
        var state = MiniCPMSamplerState(seed: 1)
        var expectedState = state
        _ = expectedState.nextUniform()
        XCTAssertEqual(MiniCPMSampler.sample(
            logits: logits, config: config, protocol: protocolIDs,
            state: &state).item(Int.self), protocolIDs.listen)
        XCTAssertEqual(state, expectedState)
    }

    func testListenTopFiveUsesStrictZeroBasedRankAndRespectsExplicitZero() {
        // Four scores exceed listen: zero-based rank 4 belongs to top five.
        let rankFour = MLXArray([Float(10), 5, -100, -100, 9, 8, 7, -100, -100])
        // A fifth higher score puts listen exactly outside the same boundary.
        let rankFive = MLXArray([Float(10), 5, -100, -100, 9, 8, 7, 6, -100])
        for (logits, k, expected) in [(rankFour, 5, 1), (rankFive, 5, 0), (rankFour, 0, 0)] {
            let config = MiniCPMSamplingConfig(
                mode: .greedy, repetitionPenalty: 1,
                repetitionPolicy: .officialDuplex, lengthPenalty: 1, listenTopK: k)
            var state = MiniCPMSamplerState(seed: 99)
            XCTAssertEqual(MiniCPMSampler.sample(
                logits: logits, config: config, protocol: protocolIDs,
                state: &state).item(Int.self), expected)
        }
    }

    func testNaturalChunkEOSStillWinsWhenListenIsInTopFive() {
        let config = MiniCPMSamplingConfig(
            mode: .greedy, repetitionPenalty: 1,
            repetitionPolicy: .officialDuplex, lengthPenalty: 1, listenTopK: 5)
        let logits = MLXArray([Float(0), 9, 10, -100, -100, -100, -100, -100, -100])
        var state = MiniCPMSamplerState(seed: 99)
        let before = state
        XCTAssertEqual(MiniCPMSampler.sample(
            logits: logits, config: config, protocol: protocolIDs,
            forbidden: [protocolIDs.chunkEOS], state: &state).item(Int.self), protocolIDs.chunkEOS)
        XCTAssertEqual(state, before)
    }

    func testSamplerStateRecordsFilteredProtocolTokensAndRestoresHistory() {
        let config = MiniCPMSamplingConfig(
            mode: .greedy, repetitionPenalty: 1, repetitionPolicy: .officialDuplex,
            lengthPenalty: 1)
        var state = MiniCPMSamplerState(seed: 1)
        let before = state.snapshot()
        let logits = MLXArray([Float(0), 100, -100, -100, -100, -100, -100, -100, -100])
        _ = MiniCPMSampler.sample(
            logits: logits, config: config, protocol: protocolIDs, state: &state)
        XCTAssertEqual(state.sampledTokens, [protocolIDs.listen])
        let sampledSnapshot = state.snapshot()
        state.clearTokenHistory()
        XCTAssertEqual(state.rawState, sampledSnapshot.rawState)
        XCTAssertTrue(state.sampledTokens.isEmpty)
        state.restore(sampledSnapshot)
        XCTAssertEqual(state.sampledTokens, [protocolIDs.listen])
        state.restore(before)
        XCTAssertTrue(state.sampledTokens.isEmpty)
    }

    func testOfficialDuplexPenalizesNegativeHistoryWithCheckpointRule() {
        // StreamDecoder.decode divides a historical -1 by 1.05. It therefore
        // outranks -1.02; Hugging Face instead multiplies -1 and selects id 9.
        let logits = MLXArray([Float(-1), -100, -100, -100, -100, -100, -100, -100, -100, -1.02])
        var config = MiniCPMSamplingConfig(
            mode: .greedy, repetitionPenalty: 1.05, repetitionPolicy: .officialDuplex,
            lengthPenalty: 1)
        var state = MiniCPMSamplerState(seed: 1)
        XCTAssertEqual(MiniCPMSampler.sample(
            logits: logits, config: config, protocol: protocolIDs,
            previousTokens: [0], state: &state).item(Int.self), 0)
        config.repetitionPolicy = .huggingFace
        XCTAssertEqual(MiniCPMSampler.sample(
            logits: logits, config: config, protocol: protocolIDs,
            previousTokens: [0], state: &state).item(Int.self), 9)
    }

    func testOfficialDuplexIncludesProtocolHistory() {
        let logits = MLXArray([Float(5), 5.1, -100, -100, -100, -100, -100, -100, -100])
        var config = MiniCPMSamplingConfig(
            mode: .greedy, repetitionPenalty: 1.05, repetitionPolicy: .officialDuplex,
            lengthPenalty: 1)
        var state = MiniCPMSamplerState(seed: 1)
        XCTAssertEqual(MiniCPMSampler.sample(
            logits: logits, config: config, protocol: protocolIDs,
            previousTokens: [protocolIDs.listen], state: &state).item(Int.self), 0)
        config.repetitionPolicy = .huggingFace
        XCTAssertEqual(MiniCPMSampler.sample(
            logits: logits, config: config, protocol: protocolIDs,
            previousTokens: [protocolIDs.listen], state: &state).item(Int.self), protocolIDs.listen)
    }

    func testSamplerPreservesCallerLogitsAcrossMasksAndShaping() {
        let values: [Float] = [5, 5.1, -100, -100, -100, -100, -100, -100, 4]
        for shape in [[values.count], [1, values.count], [1, 1, values.count]] {
            let logits = MLXArray(values, shape)
            let config = MiniCPMSamplingConfig(
                mode: .greedy, repetitionPenalty: 1.05, repetitionPolicy: .officialDuplex,
                lengthPenalty: 1, listenProbabilityScale: 0.8)
            var state = MiniCPMSamplerState(seed: 1)
            XCTAssertEqual(MiniCPMSampler.sample(
                logits: logits, config: config, protocol: protocolIDs,
                forbidden: [8], previousTokens: [protocolIDs.listen],
                state: &state).item(Int.self), 0)
            XCTAssertEqual(logits.asArray(Float.self), values, "Raw logits must survive score shaping")
            let chatConfig = MiniCPMSamplingConfig(mode: .greedy, lengthPenalty: 1)
            XCTAssertEqual(MiniCPMSampler.sample(
                logits: logits, config: chatConfig, protocol: protocolIDs,
                state: &state).item(Int.self), protocolIDs.listen)
        }
    }

    func testOfficialDuplexIgnoresLengthPenalty() {
        let config = MiniCPMSamplingConfig(
            mode: .greedy, repetitionPenalty: 1, repetitionPolicy: .officialDuplex,
            lengthPenalty: 2)
        let logits = MLXArray([Float(9), -100, -100, -100, 9.5, -100, -100, -100, -100])
        var state = MiniCPMSamplerState(seed: 1)
        XCTAssertEqual(MiniCPMSampler.sample(
            logits: logits, config: config, protocol: protocolIDs,
            state: &state).item(Int.self), protocolIDs.turnEOS)
    }

    func testOfficialZeroRepetitionWindowUsesAllHistoryWhileChatDisablesIt() {
        let logits = MLXArray([Float(5.1), -100, -100, -100, -100, -100, -100, -100, -100, 5])
        var config = MiniCPMSamplingConfig(
            mode: .greedy, repetitionPenalty: 1.05, repetitionWindowSize: 0,
            repetitionPolicy: .officialDuplex, lengthPenalty: 1)
        var state = MiniCPMSamplerState(seed: 1)
        XCTAssertEqual(MiniCPMSampler.sample(
            logits: logits, config: config, protocol: protocolIDs,
            previousTokens: [0], state: &state).item(Int.self), 9)
        config.repetitionPolicy = .huggingFace
        XCTAssertEqual(MiniCPMSampler.sample(
            logits: logits, config: config, protocol: protocolIDs,
            previousTokens: [0], state: &state).item(Int.self), 0)
    }

    func testTopKRetainsAllTokensTiedAtThreshold() {
        let config = MiniCPMSamplingConfig(
            temperature: 1, topK: 2, topP: 1, repetitionPenalty: 1,
            lengthPenalty: 1)
        let logits = MLXArray([Float(3), -100, -100, -100, -100, -100, -100, -100, 2, 2])
        let tokens = [Float(0), 0.7, 0.999].map { draw in
            MiniCPMSampler.sample(
                logits: logits, config: config, protocol: protocolIDs,
                uniforms: [0, draw]).item(Int.self)
        }
        XCTAssertEqual(Set(tokens), Set([0, 8, 9]))
    }

    func testSpecialHistoryDoesNotReceiveRepetitionPenalty() {
        let config = MiniCPMSamplingConfig(
            temperature: 0, topK: 0, topP: 1, repetitionPenalty: 100,
            lengthPenalty: 1)
        // listen (1) is the best candidate.  It remains best when it appears
        // in history because protocol markers are not text repetition.
        let logits = MLXArray([Float(0), 10, 9, 8, 7, 6, 5, 4, 3])
        let token = MiniCPMSampler.sample(
            logits: logits, config: config, protocol: protocolIDs,
            previousTokens: [1], uniform: 0).item(Int.self)
        XCTAssertEqual(token, 1)
    }

    func testLengthPenaltySuppressesTerminatorLogit() {
        let config = MiniCPMSamplingConfig(
            temperature: 0, topK: 0, topP: 1, repetitionPenalty: 1,
            lengthPenalty: 2)
        // turn_eos (id 4) leads before the penalty, then its positive logit is
        // divided by two and loses to ordinary token 1.  chunk_tts_eos (id 3)
        // is deliberately below token 1 so this also guards that it is not
        // modified by the turn-only length penalty.
        let logits = MLXArray([Float(0), 9, 0, 8.5, 9.5, 0, 0, 0, 0])
        let token = MiniCPMSampler.sample(
            logits: logits, config: config, protocol: protocolIDs,
            uniform: 0).item(Int.self)
        XCTAssertEqual(token, 1)
    }

    func testLengthPenaltyLeavesChunkTTSEOSUntouched() {
        let config = MiniCPMSamplingConfig(
            temperature: 0, topK: 0, topP: 1, repetitionPenalty: 1,
            lengthPenalty: 2)
        // chunk_tts_eos (id 3) is the best post-probe candidate and must not
        // be divided by the turn length penalty.
        let logits = MLXArray([Float(0), 8, 0, 9.5, 0, 9, 0, 0, 0])
        let token = MiniCPMSampler.sample(
            logits: logits, config: config, protocol: protocolIDs,
            uniform: 0).item(Int.self)
        XCTAssertEqual(token, protocolIDs.chunkTTSEOS)
    }

    func testLogitTraceTopKEnvironmentGateIsStrict() {
        XCTAssertNil(MiniCPMSampler.logitTraceTopK(from: nil))
        XCTAssertNil(MiniCPMSampler.logitTraceTopK(from: ""))
        XCTAssertNil(MiniCPMSampler.logitTraceTopK(from: "0"))
        XCTAssertNil(MiniCPMSampler.logitTraceTopK(from: "21"))
        XCTAssertNil(MiniCPMSampler.logitTraceTopK(from: " 3"))
        XCTAssertNil(MiniCPMSampler.logitTraceTopK(from: "three"))
        XCTAssertEqual(MiniCPMSampler.logitTraceTopK(from: "1"), 1)
        XCTAssertEqual(MiniCPMSampler.logitTraceTopK(from: "20"), 20)
        XCTAssertEqual(
            MiniCPMSampler.logitTraceTopK(
                environment: ["MINICPM_DUPLEX_LOGIT_TRACE_TOPK": "4"]),
            4)
    }

    func testLogitTraceUsesRawTopKAndDeterministicProtocolRanks() {
        let logits: [Float] = [0, 9, 4, 7, 2, 8, 6, 1, 3]
        let trace = MiniCPMSampler.logitTrace(
            logits: logits, protocol: protocolIDs, topK: 3, sampledToken: 0)

        XCTAssertEqual(
            trace.topK,
            [
                MiniCPMLogitTraceEntry(tokenID: 1, logit: 9, rank: 1),
                MiniCPMLogitTraceEntry(tokenID: 5, logit: 8, rank: 2),
                MiniCPMLogitTraceEntry(tokenID: 3, logit: 7, rank: 3),
            ])
        XCTAssertEqual(trace.listen?.logit, 9)
        XCTAssertEqual(trace.listen?.rank, 1)
        XCTAssertEqual(trace.chunkEOS?.logit, 4)
        XCTAssertEqual(trace.chunkEOS?.rank, 5)
        XCTAssertEqual(trace.chunkTTSEOS?.rank, 3)
        XCTAssertEqual(trace.turnEOS?.rank, 7)
        XCTAssertEqual(trace.speak?.rank, 2)
        XCTAssertEqual(trace.ttsBOS?.rank, 8)
        XCTAssertEqual(trace.sampledToken, 0)
    }

    func testLogitTraceBreaksEqualLogitTiesByTokenID() {
        let trace = MiniCPMSampler.logitTrace(
            logits: [0, 5, 5, 1], protocol: protocolIDs, topK: 4, sampledToken: 2)
        XCTAssertEqual(trace.topK.map(\.tokenID), [1, 2, 3, 0])
        XCTAssertEqual(trace.listen?.rank, 1)
        XCTAssertEqual(trace.chunkEOS?.rank, 2)
    }
}
