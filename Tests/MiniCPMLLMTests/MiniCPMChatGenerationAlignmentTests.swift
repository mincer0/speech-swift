import XCTest
import MLX
@testable import MiniCPMLLM

final class MiniCPMChatGenerationAlignmentTests: XCTestCase {
    func testAlignedSpansHaveOneHiddenStatePerGeneratedToken() {
        let firstHidden = MLXArray([Float(1), 2]).reshaped([1, 1, 2])
        let secondHidden = MLXArray([Float(3), 4]).reshaped([1, 1, 2])
        let spans = [
            MiniCPMChatGenerationAlignedSpan(index: 0, token: 101, hidden: firstHidden),
            MiniCPMChatGenerationAlignedSpan(index: 1, token: 102, hidden: secondHidden),
        ]
        let result = MiniCPMChatGenerationResult(
            tokens: [101, 102],
            text: "12",
            steps: [],
            alignedSpans: spans,
            finishReason: .length,
            cacheLength: 2)

        XCTAssertEqual(result.tokens.count, result.alignedSpans.count)
        XCTAssertEqual(result.tokens, result.alignedSpans.map(\.token))
        XCTAssertEqual(result.hiddenStates.count, result.tokens.count)
        XCTAssertEqual(result.hiddenStates.map(\.shape), [[1, 1, 2], [1, 1, 2]])
    }

    func testTerminatorAtFirstOrNextBoundaryNeverCreatesATTSAlignment() {
        let firstHidden = MLXArray([Float(1), 2]).reshaped([1, 1, 2])
        let textStep = MiniCPMChatGenerationStep(
            index: 0, token: 101, text: "a", isListen: false, endOfTurn: false)
        let eosStep = MiniCPMChatGenerationStep(
            index: 1, token: 151717, text: "", isListen: false, endOfTurn: true)

        let result = MiniCPMChatGenerationResult(
            tokens: [101],
            text: "a",
            steps: [textStep, eosStep],
            alignedSpans: [
                MiniCPMChatGenerationAlignedSpan(index: 0, token: 101, hidden: firstHidden),
            ],
            finishReason: .turnEOS,
            cacheLength: 2)

        XCTAssertEqual(result.alignedSpans.map(\.index), [0])
        XCTAssertEqual(result.alignedSpans.map(\.token), result.tokens)
        XCTAssertFalse(result.alignedSpans.contains { $0.token == eosStep.token })
        XCTAssertEqual(result.steps.last?.token, eosStep.token)
    }
}
