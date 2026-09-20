import Foundation
import XCTest
import MLX
@testable import MiniCPMLLM

final class MiniCPMLLMOracleTests: XCTestCase {
    private struct Oracle: Decodable {
        struct Step: Decodable {
            let hidden: [Float]
            let logits: [Float]
            let top20: [Int]
        }
        struct Rollback: Decodable {
            let cacheBefore: Int
            let cacheAfter: Int
            let removed: Int
            let step: Step
        }
        let tokenIds: [Int]
        let externalEmbeddings: [[Float]]
        let steps: [Step]
        let rollback: Rollback
    }

    func testPythonOracleTokenExternalAndRollbackTrace() throws {
        let fixtures = try XCTUnwrap(Bundle.module.resourceURL?.appendingPathComponent("Fixtures"))
        let oracleData = try Data(contentsOf: fixtures.appendingPathComponent("oracle.json"))
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        let oracle = try decoder.decode(Oracle.self, from: oracleData)
        let config = try MiniCPMMLXConfig.load(from: fixtures)
        let model = MiniCPMLanguageModel(config: config)
        try MiniCPMWeightLoader.load(model: model, from: fixtures)
        let session = MiniCPMSession(model: model)

        let ids = MLXArray(oracle.tokenIds.map(Int32.init)).expandedDimensions(axis: 0)
        let tokenOutput = session.feed(inputIds: ids)
        assertStep(tokenOutput, oracle.steps[0], label: "token")

        let externalRows = oracle.externalEmbeddings
        let external = MLXArray(externalRows.flatMap { $0 } as [Float])
            .reshaped([externalRows.count, config.hiddenSize])
        let externalOutput = session.feed(embeddings: external)
        assertStep(externalOutput, oracle.steps[1], label: "external")
        XCTAssertEqual(session.cacheLength, 4)

        session.beginUnit()
        let rollbackRow = (0..<config.hiddenSize).map { (Float($0) + 0.25) / 16 }
        let rollbackEmbedding = MLXArray(rollbackRow as [Float])
            .reshaped([1, config.hiddenSize])
        let rollbackOutput = session.feed(embeddings: rollbackEmbedding)
        assertStep(rollbackOutput, oracle.rollback.step, label: "rollback-before")
        XCTAssertEqual(session.cacheLength, oracle.rollback.cacheBefore)
        XCTAssertEqual(session.rollbackUnit(), oracle.rollback.removed)
        XCTAssertEqual(session.cacheLength, oracle.rollback.cacheAfter)
    }

    private func assertStep(
        _ output: MiniCPMForwardOutput,
        _ expected: Oracle.Step,
        label: String,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        eval(output.logits, output.hidden)
        let logits = output.logits[0, output.logits.dim(1) - 1].asType(.float32).asArray(Float.self)
        let hidden = output.hidden[0, output.hidden.dim(1) - 1].asType(.float32).asArray(Float.self)
        XCTAssertEqual(logits.count, expected.logits.count, "\(label) logits size", file: file, line: line)
        XCTAssertEqual(hidden.count, expected.hidden.count, "\(label) hidden size", file: file, line: line)
        for (actual, wanted) in zip(logits, expected.logits) {
            XCTAssertEqual(actual, wanted, accuracy: 2e-4, "\(label) logits", file: file, line: line)
        }
        for (actual, wanted) in zip(hidden, expected.hidden) {
            XCTAssertEqual(actual, wanted, accuracy: 2e-4, "\(label) hidden", file: file, line: line)
        }
        let top = logits.enumerated().sorted { $0.element > $1.element }.prefix(20).map(\.offset)
        XCTAssertEqual(top, expected.top20, "\(label) top20", file: file, line: line)
    }
}
