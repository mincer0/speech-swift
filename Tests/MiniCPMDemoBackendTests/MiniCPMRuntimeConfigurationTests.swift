import XCTest
@testable import MiniCPMDemoBackend
import MiniCPMDuplexRuntime

final class MiniCPMRuntimeConfigurationTests: XCTestCase {
    private func initializedConfig(_ parameters: [String: MiniCPMJSONValue]) async throws -> MiniCPMDuplexConfig {
        // Capture the real session.init/backend/adapter prepare boundary
        // without loading model weights or depending on adapter internals.
        let engine = ConfigurationCapturingEngine()
        let runtime = MiniCPMDuplexRuntime(engineFactory: { engine })
        let adapter = MiniCPMDuplexRuntimeEngineAdapter(
            runtime: runtime, sessionID: "configuration-test")
        let backend = MiniCPMDemoBackend(engine: adapter)
        let events = await backend.handle(MiniCPMDemoRequest(
            type: "session.init", payload: .object(parameters)))
        XCTAssertTrue(events.contains { $0.type == "session.created" })
        return try XCTUnwrap(engine.preparedConfig)
    }

    func testBrowserPromptAndTTSFlagSurviveSamplingEnvelope() async throws {
        let prompt = "只有听到用户清晰说话时才回答；回答完后保持倾听。"
        let config = try await initializedConfig([
            "mode": .string("full_duplex"),
            "system_prompt": .string(prompt),
            "use_tts": .bool(false),
            "config": .object([
                "force_listen_count": .number(0),
                "greedy_listen_speak_decision": .bool(false),
                "length_penalty": .number(1.1),
                "top_k": .number(20),
            ]),
        ])
        XCTAssertEqual(config.systemPrompt, prompt)
        XCTAssertFalse(config.generateAudio)
        XCTAssertEqual(config.topK, 20)
        XCTAssertEqual(config.temperature, 0.7)
        XCTAssertEqual(config.topP, 0.8)
        XCTAssertEqual(config.maxNewTokens, 20)
    }

    func testNestedAliasesOverrideTopLevelCanonicalSettings() async throws {
        let config = try await initializedConfig([
            "system_prompt": .string("top-level prompt"),
            "generate_audio": .bool(true),
            "top_k": .number(20),
            "max_new_speak_tokens_per_chunk": .number(20),
            "temperature": .number(0.7),
            "config": .object([
                "systemPrompt": .string("nested prompt"),
                "use_tts": .bool(false),
                "topK": .number(8),
                "maxNewTokens": .number(12),
                "temperature": .number(0.2),
            ]),
        ])
        XCTAssertEqual(config.systemPrompt, "nested prompt")
        XCTAssertFalse(config.generateAudio)
        XCTAssertEqual(config.topK, 8)
        XCTAssertEqual(config.maxNewTokens, 12)
        XCTAssertEqual(config.temperature, 0.2)
    }

    func testNestedGenerateAudioOverridesTopLevelUseTTS() async throws {
        let config = try await initializedConfig([
            "systemPrompt": .string("camel-case prompt"),
            "use_tts": .bool(false),
            "config": .object(["generateAudio": .bool(true)]),
        ])
        XCTAssertEqual(config.systemPrompt, "camel-case prompt")
        XCTAssertTrue(config.generateAudio)
    }

    func testLegacySamplingAndDuplexEnvelopesKeepSessionSettings() async throws {
        for envelope in ["sampling", "duplex"] {
            let config = try await initializedConfig([
                "system_prompt": .string("session prompt"),
                "use_tts": .bool(false),
                "top_p": .number(0.9),
                envelope: .object(["temperature": .number(0.3)]),
            ])
            XCTAssertEqual(config.systemPrompt, "session prompt")
            XCTAssertFalse(config.generateAudio)
            XCTAssertEqual(config.temperature, 0.3)
            XCTAssertEqual(config.topP, 0.9)
        }
    }
}

private final class ConfigurationCapturingEngine: MiniCPMDuplexEngine, @unchecked Sendable {
    var preparedConfig: MiniCPMDuplexConfig?

    func prepare(mode: MiniCPMDuplexMode, config: MiniCPMDuplexConfig) throws -> String {
        preparedConfig = config
        return config.systemPrompt
    }

    func prefill(mode: MiniCPMDuplexMode, input: MiniCPMDuplexInput) throws -> MiniCPMDuplexPrefillResult {
        MiniCPMDuplexPrefillResult(mode: mode, input: input)
    }

    func generate(mode: MiniCPMDuplexMode, config: MiniCPMDuplexConfig) throws -> MiniCPMDuplexOutput {
        MiniCPMDuplexOutput(text: "ok", endOfTurn: true, finishReason: "turn_end")
    }

    func finalize() throws {}
    func rollback() throws {}
    func interrupt() {}
    func reset() {}
}
