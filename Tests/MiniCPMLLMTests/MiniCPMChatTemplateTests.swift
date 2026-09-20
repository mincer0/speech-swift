import Foundation
import XCTest
@testable import MiniCPMLLM

final class MiniCPMChatTemplateTests: XCTestCase {
    private func builderOrSkip() throws -> MiniCPMChatTemplateBuilder {
        let environment = ProcessInfo.processInfo.environment
        let candidates = [
            environment["MINICPM_TOKENIZER_DIR"],
            "/Users/mincer/项目/s2s/models/MiniCPM-o-4_5-llm-mlx-8bit",
        ].compactMap { $0 }
        guard let path = candidates.first(where: {
            FileManager.default.fileExists(
                atPath: URL(fileURLWithPath: $0).appendingPathComponent("tokenizer.json").path)
        }) else {
            throw XCTSkip("Set MINICPM_TOKENIZER_DIR for the local MiniCPM chat-template tests")
        }
        return MiniCPMChatTemplateBuilder(
            tokenizer: try MiniCPMTokenizer.loadSynchronously(
                from: URL(fileURLWithPath: path)))
    }

    func testOrderedMediaProducesIndependentSlotsAndWrappers() throws {
        let builder = try builderOrSkip()
        let messages = [
            MiniCPMChatMessage(role: .system, text: "You are concise."),
            MiniCPMChatMessage(role: .user, parts: [
                .text("before"),
                .audio("mic", sampleRate: 16_000, sampleCount: 16_000),
                .text("between"),
                .image("camera"),
                .video("screen"),
                .text("after"),
            ]),
        ]
        let plan = try builder.build(
            messages: messages,
            options: MiniCPMChatTemplateOptions(
                addGenerationPrompt: false,
                contentSeparator: .none))

        XCTAssertEqual(plan.slots.map(\.media.key), ["mic", "camera", "screen"])
        XCTAssertEqual(plan.slots.map(\.media.kind), [.audio, .image, .video])
        XCTAssertTrue(plan.promptText.contains("<audio>./</audio>"))
        XCTAssertTrue(plan.promptText.contains("<image>./</image>"))

        let embeddingIndices = plan.segments.compactMap { segment -> Int? in
            guard case .embedding(let slot) = segment else { return nil }
            return slot.index
        }
        XCTAssertEqual(embeddingIndices, [0, 1, 2])
        XCTAssertGreaterThanOrEqual(plan.segments.count, 9)
    }

    func testGenerationPromptThinkingAndTTSMarkersFollowOptions() throws {
        let builder = try builderOrSkip()
        let user = MiniCPMChatMessage(role: .user, text: "Tell me a story.")

        let silent = try builder.build(
            messages: [user],
            options: MiniCPMChatTemplateOptions(
                addGenerationPrompt: true,
                useTTSTemplate: false,
                enableThinking: false))
        XCTAssertTrue(silent.promptText.contains("<|im_start|>assistant\n"))
        XCTAssertTrue(silent.promptText.contains("<think>\n\n</think>\n\n"))
        XCTAssertFalse(silent.promptText.contains("<|tts_bos|>"))

        let spoken = try builder.build(
            messages: [user],
            options: MiniCPMChatTemplateOptions(
                addGenerationPrompt: true,
                useTTSTemplate: true,
                enableThinking: true))
        XCTAssertTrue(spoken.promptText.contains("<|im_start|>assistant\n"))
        XCTAssertTrue(spoken.promptText.contains("<|tts_bos|>"))
        XCTAssertFalse(spoken.promptText.contains("<think>\n\n</think>\n\n"))
    }

    func testStreamingPromptPrefixesMatchPinnedDemo() throws {
        let builder = try builderOrSkip()
        let message = MiniCPMChatMessage(role: .user, parts: [
            .text("hello"), .audio("chunk", sampleRate: 16_000), .text("world"),
        ])

        let first = try builder.buildStreaming(message: message, context: .firstUser)
        XCTAssertTrue(first.promptText.hasPrefix("<|im_start|>user\n"))
        XCTAssertFalse(first.promptText.contains("<|im_end|>"))
        XCTAssertEqual(first.slots.count, 1)

        let continuation = try builder.buildStreaming(message: message, context: .continuation)
        XCTAssertFalse(continuation.promptText.contains("<|im_start|>user"))
        XCTAssertTrue(continuation.promptText.hasPrefix("hello"))

        let completed = try builder.buildStreaming(
            message: message,
            context: .nextUser(completedResponse: true))
        XCTAssertTrue(completed.promptText.hasPrefix("<|im_end|>\n<|im_start|>user\n"))

        let interrupted = try builder.buildStreaming(
            message: message,
            context: .nextUser(completedResponse: false))
        XCTAssertTrue(interrupted.promptText.hasPrefix("<|tts_eos|><|im_end|>\n<|im_start|>user\n"))
    }

    func testEmbeddingSlotsResolveWithoutFrameworkTypes() throws {
        let builder = try builderOrSkip()
        let plan = try builder.build(messages: [
            MiniCPMChatMessage(role: .user, parts: [
                .text("A"), .image("one"), .text("B"), .audio("two"),
            ])
        ])
        let resolved = try plan.resolve { slot in
            "embedding:\(slot.media.kind.rawValue):\(slot.media.key)"
        }
        let values = resolved.compactMap { segment -> String? in
            guard case .embedding(_, let value) = segment else { return nil }
            return value
        }
        XCTAssertEqual(values, ["embedding:image:one", "embedding:audio:two"])
    }
}
