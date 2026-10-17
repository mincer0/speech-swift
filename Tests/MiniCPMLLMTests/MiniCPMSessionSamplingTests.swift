import XCTest
import MLX
@testable import MiniCPMLLM

final class MiniCPMSessionSamplingTests: XCTestCase {
    private let ids = MiniCPMProtocolTokenIds(
        listen: 25, chunkEOS: 26, chunkTTSEOS: 27,
        turnEOS: 28, speak: 29, unitEnd: 30, ttsBOS: 31)

    private func session() -> MiniCPMSession {
        let config = MiniCPMMLXConfig(
            hiddenSize: 4, numHiddenLayers: 1, numAttentionHeads: 1,
            numKeyValueHeads: 1, headDim: 4, intermediateSize: 4,
            vocabSize: 32, maxPositionEmbeddings: 64, ropeTheta: 10_000,
            rmsNormEps: 1e-6, tieWordEmbeddings: false, eosTokenId: 0,
            quantGroupSize: 32, quantBits: 0, quantMode: "none")
        return MiniCPMSession(model: MiniCPMLanguageModel(config: config), samplerSeed: 7)
    }

    func testOfficialHistoryIncludesUncommittedSamplesAndRestoresWithSession() {
        let session = session()
        let config = MiniCPMSamplingConfig(
            mode: .greedy, repetitionPenalty: 2,
            repetitionPolicy: .officialDuplex, lengthPenalty: 1)
        var values = [Float](repeating: -100, count: 32)
        values[0] = 3
        values[1] = 2
        let logits = MLXArray(values)
        XCTAssertEqual(session.sample(logits: logits, config: config, protocol: ids).item(Int.self), 0)
        XCTAssertTrue(session.generatedTokens.isEmpty, "The unit has not been committed yet")
        let checkpoint = session.snapshot()
        XCTAssertEqual(checkpoint.samplerState.sampledTokens, [0])
        XCTAssertEqual(session.sample(logits: logits, config: config, protocol: ids).item(Int.self), 1)
        session.restore(checkpoint)
        XCTAssertEqual(session.sample(logits: logits, config: config, protocol: ids).item(Int.self), 1)
        session.reset()
        XCTAssertEqual(session.sample(logits: logits, config: config, protocol: ids).item(Int.self), 0)
    }

    func testChatUsesExplicitHistoryWhileDuplexUsesItsSampledHistory() {
        let session = session()
        var values = [Float](repeating: -100, count: 32)
        values[0] = 3
        values[1] = 2
        let logits = MLXArray(values)
        let chat = MiniCPMSamplingConfig(mode: .greedy, repetitionPenalty: 2, lengthPenalty: 1)
        XCTAssertEqual(session.sample(
            logits: logits, config: chat, protocol: ids,
            previousTokens: [0]).item(Int.self), 1)
        let duplex = MiniCPMSamplingConfig(
            mode: .greedy, repetitionPenalty: 2,
            repetitionPolicy: .officialDuplex, lengthPenalty: 1)
        XCTAssertEqual(session.sample(
            logits: logits, config: duplex, protocol: ids,
            previousTokens: [0]).item(Int.self), 0)
    }
}
