import XCTest
@testable import MiniCPMDemoBackend
import MiniCPMLLM
import MiniCPMDuplexRuntime

final class MiniCPMNativeChatEngineAdapterTests: XCTestCase {
    func testStreamingVADFactoryPropagatesCreationFailure() {
        enum FactoryError: Error, Equatable {
            case unavailable
        }

        let factory: MiniCPMStreamingVADFactory = {
            throw FactoryError.unavailable
        }
        XCTAssertThrowsError(try factory()) { error in
            XCTAssertEqual(error as? FactoryError, .unavailable)
        }
    }

    func testOrderedMessageParserKeepsRoleAndPartOrder() throws {
        let input: MiniCPMJSONValue = .array([
            .object([
                "role": .string("system"),
                "content": .string("policy"),
            ]),
            .object([
                "role": .string("user"),
                "content": .array([
                    .object(["type": .string("text"), "text": .string("before")]),
                    .object(["type": .string("audio"), "data": .array([.number(0.25)]), "sample_rate": .number(16_000)]),
                    .object(["type": .string("image"), "data": .array([.number(0), .number(1)])]),
                    .object(["type": .string("text"), "text": .string("after")]),
                ]),
            ]),
        ])

        let parsed = try MiniCPMNativeChatEngineAdapter.parseMessages(input, maxVideoFrames: 4)
        XCTAssertEqual(parsed.messages.map(\.role), [.system, .user])
        XCTAssertEqual(parsed.messages[1].parts.count, 4)
        XCTAssertEqual(parsed.messages[1].parts[0], .text("before"))
        XCTAssertEqual(parsed.messages[1].parts[3], .text("after"))
        guard case .media(let audio) = parsed.messages[1].parts[1],
              case .media(let image) = parsed.messages[1].parts[2] else {
            return XCTFail("ordered audio/image slots were not retained")
        }
        XCTAssertEqual(audio.kind, .audio)
        XCTAssertEqual(image.kind, .image)
        XCTAssertEqual(parsed.media.count, 2)
    }

    func testUnsupportedRoleAndContentFailClosed() {
        let roleInput: MiniCPMJSONValue = .array([
            .object([
                "role": .string("tool"),
                "content": .string("must not flatten"),
            ])
        ])
        XCTAssertThrowsError(try MiniCPMNativeChatEngineAdapter.parseMessages(roleInput, maxVideoFrames: 4))

        let contentInput: MiniCPMJSONValue = .array([
            .object([
                "role": .string("user"),
                "content": .array([
                    .object(["type": .string("audio_chunk"), "data": .array([.number(0.1)])])
                ])
            ])
        ])
        XCTAssertThrowsError(try MiniCPMNativeChatEngineAdapter.parseMessages(contentInput, maxVideoFrames: 4))
    }

    func testDirectTextMapsToOneUserMessageWithoutDuplexFlattening() throws {
        let parsed = try MiniCPMNativeChatEngineAdapter.parseDirectInput(
            ["text": .string("hello")],
            maxVideoFrames: 4)
        XCTAssertEqual(parsed.messages, [MiniCPMChatMessage(role: .user, text: "hello")])
        XCTAssertTrue(parsed.media.isEmpty)
    }
}
