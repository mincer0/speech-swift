import Foundation
import XCTest
@testable import MiniCPMDemoBackend
import Hummingbird
import HummingbirdTesting
import HummingbirdWebSocket
import HummingbirdWSTesting

final class MiniCPMRecordingTimestampTests: XCTestCase {
    func testWireRecorderPreservesTimestampWhileExternalizingMedia() async throws {
        let root = try temporaryRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let store = try MiniCPMDemoSessionStore(root: root)
        let wire = try await MiniCPMDemoWireRecordingSession(
            store: store,
            metadata: MiniCPMDemoSessionMetadata(sessionID: "timestamp_media", mode: "audio"))
        let receivedAt = Date(timeIntervalSince1970: 1_691_234_567.125)
        var bytes = Data()
        for index in 0..<4_096 {
            var bits = Float(index % 2 == 0 ? 0.25 : -0.25).bitPattern.littleEndian
            withUnsafeBytes(of: &bits) { bytes.append(contentsOf: $0) }
        }
        let frame: MiniCPMJSONValue = .object([
            "type": .string("input.append"),
            "input": .object(["audio": .string(bytes.base64EncodedString())]),
        ])

        await wire.record(direction: .up, frame: frame, timestamp: receivedAt)

        let recording = try await store.recording(sessionID: "timestamp_media")
        XCTAssertEqual(recording.events.count, 1)
        XCTAssertEqual(recording.blobs.count, 1)
        let event = try XCTUnwrap(recording.events.first)
        XCTAssertEqual(event.timestamp, receivedAt.timeIntervalSince1970, accuracy: 0.000_001)
        let storedAudio = event.frame.objectValue?["input"]?.objectValue?["audio"]?.stringValue
        XCTAssertTrue(storedAudio?.hasPrefix("@blob/") == true)
        await wire.close(reason: "test")
    }

    func testRealtimeRecordingPreservesReceiveTimeAcrossDelayedInitialization() async throws {
        let root = try temporaryRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let store = try MiniCPMDemoSessionStore(root: root)
        let engine = RecordingTimestampInitializationEngine()
        let app = Application(
            router: Router(),
            server: makeMiniCPMDemoWebSocketServer(
                configuration: .init(autoPing: .disabled),
                makeBackend: { MiniCPMDemoBackend(engine: engine) },
                emitQueueEvents: false,
                recordingStore: store))

        _ = try await app.test(.live) { client in
            try await client.ws("/v1/realtime?mode=audio") { inbound, outbound, _ in
                var iterator = inbound.messages(maxSize: .max).makeAsyncIterator()
                try await outbound.write(.text(
                    "{\"type\":\"session.init\",\"payload\":{\"mode\":\"full_duplex\"}}"))
                await engine.waitUntilInitializationStarted()
                // The inbound init has already been received, while recorder
                // attachment still awaits the engine's readiness boundary.
                await engine.releaseInitialization()
                var created = false
                while !created, let message = try await iterator.next() {
                    if case .text(let text) = message, text.contains("session.created") {
                        created = true
                    }
                }
                XCTAssertTrue(created)
                try await outbound.write(.text("{\"type\":\"session.close\"}"))
                var closed = false
                while !closed, let message = try await iterator.next() {
                    if case .text(let text) = message, text.contains("session.closed") {
                        closed = true
                    }
                }
                XCTAssertTrue(closed)
            }
        }

        let summaries = try await store.list()
        let session = try XCTUnwrap(summaries.first)
        let recording = try await store.recording(sessionID: session.metadata.sessionID)
        let initialEvent = try XCTUnwrap(recording.events.first {
            $0.direction == .up && $0.frame.objectValue?["type"]?.stringValue == "session.init"
        })
        let startedAt = await engine.initializationStartedAt
        let initializationStartedAt = try XCTUnwrap(startedAt)
        XCTAssertLessThanOrEqual(initialEvent.timestamp, initializationStartedAt.timeIntervalSince1970,
            "session.init must retain its receive time rather than recorder attachment time")
    }

    private func temporaryRoot() throws -> URL {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("minicpm-recording-time-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        return root
    }
}

private actor RecordingTimestampInitializationEngine: MiniCPMDemoBackendEngine {
    private(set) var initializationStartedAt: Date?
    private var startedWaiter: CheckedContinuation<Void, Never>?
    private var gate: CheckedContinuation<Void, Never>?

    func initialize(mode: MiniCPMBackendMode, parameters: [String: MiniCPMJSONValue]) async throws {
        initializationStartedAt = Date()
        startedWaiter?.resume()
        startedWaiter = nil
        await withCheckedContinuation { gate = $0 }
    }

    func waitUntilInitializationStarted() async {
        if initializationStartedAt != nil { return }
        await withCheckedContinuation { startedWaiter = $0 }
    }

    func releaseInitialization() {
        gate?.resume()
        gate = nil
    }

    func prefill(mode: MiniCPMBackendMode, input: [String: MiniCPMJSONValue]) async throws {}
    func generate(mode: MiniCPMBackendMode, input: [String: MiniCPMJSONValue]) async throws -> [MiniCPMDemoOutput] { [] }
}
