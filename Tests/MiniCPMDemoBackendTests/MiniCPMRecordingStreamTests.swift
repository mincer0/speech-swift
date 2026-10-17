import XCTest
@testable import MiniCPMDemoBackend

final class MiniCPMRecordingStreamTests: XCTestCase {
    private func directory() -> URL {
        FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    }

    func testEveryAppendIsImmediatelyReadableBeforeClose() async throws {
        let root = directory()
        defer { try? FileManager.default.removeItem(at: root) }
        let recorder = try MiniCPMDemoRecordingService(sessionID: "stream", directory: root)
        for index in 0..<12 {
            let event = try await recorder.append(
                direction: .down, frame: .object(["index": .number(Double(index))]))
            XCTAssertEqual(event.sequence, Int64(index))
            let visible = try MiniCPMDemoRecordingService.readEvents(
                from: root.appendingPathComponent("stream.jsonl"))
            XCTAssertEqual(visible.count, index + 1)
            XCTAssertEqual(visible.last?.frame, event.frame)
        }
        try await recorder.close()
        try await recorder.close()
        do {
            _ = try await recorder.append(direction: .up, frame: .null)
            XCTFail("closed recorder accepted an event")
        } catch {}
    }

    func testReopenAppendsWithoutOverwritingAndContinuesSequence() async throws {
        let root = directory()
        defer { try? FileManager.default.removeItem(at: root) }
        let first = try MiniCPMDemoRecordingService(sessionID: "stream", directory: root)
        _ = try await first.append(direction: .up, frame: .string("first"))
        try await first.close()
        let second = try MiniCPMDemoRecordingService(sessionID: "stream", directory: root)
        let appended = try await second.append(direction: .down, frame: .string("second"))
        XCTAssertEqual(appended.sequence, 1)
        try await second.close()
        let events = try MiniCPMDemoRecordingService.readEvents(from: root.appendingPathComponent("stream.jsonl"))
        XCTAssertEqual(events.map(\.sequence), [0, 1])
        XCTAssertEqual(events.map(\.frame), [.string("first"), .string("second")])
    }

    func testConcurrentProducersWriteCompleteOrderedLines() async throws {
        let root = directory()
        defer { try? FileManager.default.removeItem(at: root) }
        let recorder = try MiniCPMDemoRecordingService(sessionID: "stream", directory: root)
        try await withThrowingTaskGroup(of: Void.self) { group in
            for index in 0..<24 {
                group.addTask {
                    _ = try await recorder.append(direction: .down, frame: .number(Double(index)))
                }
            }
            try await group.waitForAll()
        }
        try await recorder.close()
        let events = try MiniCPMDemoRecordingService.readEvents(from: root.appendingPathComponent("stream.jsonl"))
        XCTAssertEqual(events.map(\.sequence), Array(0..<24).map(Int64.init))
        XCTAssertEqual(Set(events.compactMap { $0.frame.numberValue }), Set((0..<24).map(Double.init)))
    }

    func testSessionSnapshotDoesNotRequireWritableStream() async throws {
        let root = directory()
        defer { try? FileManager.default.removeItem(at: root) }
        let store = try MiniCPMDemoSessionStore(root: root)
        _ = try await store.create(.init(sessionID: "read_only", mode: "audio"))
        let event = try await store.appendEvent(
            sessionID: "read_only", direction: .down, frame: .string("preserved"))
        _ = try await store.close(sessionID: "read_only")
        let stream = root.appendingPathComponent("sessions/read_only/stream.jsonl")
        try FileManager.default.setAttributes([.posixPermissions: 0o444], ofItemAtPath: stream.path)
        defer { try? FileManager.default.setAttributes([.posixPermissions: 0o644], ofItemAtPath: stream.path) }
        guard !FileManager.default.isWritableFile(atPath: stream.path) else {
            throw XCTSkip("current user can bypass read-only permissions")
        }

        let snapshot = try await store.recording(sessionID: "read_only")
        XCTAssertEqual(snapshot.events, [event])
    }
}
