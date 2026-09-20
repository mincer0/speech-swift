import Foundation
import Tokenizers
import XCTest
@testable import MiniCPMLLM

final class MiniCPMTokenizerOracleTests: XCTestCase {
    private struct Oracle: Decodable {
        struct Text: Decodable {
            let text: String
            let ids: [Int]
            let decoded: String
            let skipSpecial: String
        }

        struct Chat: Decodable {
            let options: Options
            let ids: [Int]
            let decoded: String

            struct Options: Decodable {
                let enableThinking: Bool?
                let useTtsTemplate: Bool?
            }
        }

        let specialTokens: [String: Int]
        let texts: [Text]
        let chat: [Chat]
    }

    /// This is an opt-in test because the production tokenizer is an 11 MiB
    /// model asset rather than a tiny unit-test fixture.  The oracle JSON was
    /// generated with `transformers.AutoTokenizer` from the same local bundle.
    func testPythonTokenizerOracleParity() throws {
        let fixtures = try XCTUnwrap(
            Bundle.module.resourceURL?.appendingPathComponent("Fixtures"))
        let oracleData = try Data(contentsOf: fixtures.appendingPathComponent("tokenizer_oracle.json"))
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        let oracle = try decoder.decode(Oracle.self, from: oracleData)

        let environment = ProcessInfo.processInfo.environment
        let candidates = [
            environment["MINICPM_TOKENIZER_DIR"],
            "/Users/mincer/项目/s2s/models/MiniCPM-o-4_5-llm-mlx-8bit",
        ].compactMap { $0 }
        guard let path = candidates.first(where: {
            FileManager.default.fileExists(atPath: URL(fileURLWithPath: $0).appendingPathComponent("tokenizer.json").path)
        }) else {
            throw XCTSkip("Set MINICPM_TOKENIZER_DIR to run MiniCPM tokenizer parity")
        }

        let tokenizer = try MiniCPMTokenizer.loadSynchronously(
            from: URL(fileURLWithPath: path))
        XCTAssertTrue(tokenizer.hasChatTemplate)
        for (token, expected) in oracle.specialTokens {
            XCTAssertEqual(tokenizer.tokenId(token), expected, token)
        }

        for sample in oracle.texts {
            let ids = tokenizer.encode(sample.text)
            XCTAssertEqual(ids, sample.ids, sample.text)
            XCTAssertEqual(tokenizer.decode(ids), sample.decoded, sample.text)
            XCTAssertEqual(
                tokenizer.decode(ids, skipSpecialTokens: true),
                sample.skipSpecial,
                sample.text)
        }

        let messages: [Message] = [[
            "role": "user",
            "content": "你好 🥳",
        ]]
        for sample in oracle.chat {
            let ids = try tokenizer.applyChatTemplate(
                messages: messages,
                enableThinking: sample.options.enableThinking,
                useTTSTemplate: sample.options.useTtsTemplate ?? false)
            XCTAssertEqual(ids, sample.ids, String(describing: sample.options))
            XCTAssertEqual(tokenizer.decode(ids), sample.decoded)
            XCTAssertEqual(
                try tokenizer.renderChatTemplate(
                    messages: messages,
                    enableThinking: sample.options.enableThinking,
                    useTTSTemplate: sample.options.useTtsTemplate ?? false),
                sample.decoded)
        }
    }
}
