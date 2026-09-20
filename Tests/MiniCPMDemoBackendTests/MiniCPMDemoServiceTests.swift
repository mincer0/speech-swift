import Foundation
import XCTest
@testable import MiniCPMDemoBackend

final class MiniCPMDemoServiceTests: XCTestCase {
    private func assertThrowsAsync(
        _ operation: () async throws -> Void,
        file: StaticString = #filePath,
        line: UInt = #line
    ) async {
        do {
            try await operation()
            XCTFail("expected operation to throw", file: file, line: line)
        } catch {
            // expected
        }
    }

    private func temporaryRoot() throws -> URL {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("minicpm-demo-services-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        return root
    }

    private func minimalWAV(samples: [Float] = [0, -0.25, 0.5]) -> Data {
        MiniCPMDemoAudioNormalizer.wavData(samples: samples, sampleRate: 16_000)
    }

    func testSessionRecordingPersistsEventsAndUniqueBlob() async throws {
        let root = try temporaryRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let store = try MiniCPMDemoSessionStore(root: root)
        _ = try await store.create(MiniCPMDemoSessionMetadata(
            sessionID: "sess_test", mode: "full_duplex", identity: ["client": .string("test")]))
        let event = try await store.appendEvent(
            sessionID: "sess_test", direction: .up,
            frame: .object(["type": .string("input.append")]))
        XCTAssertEqual(event.sequence, 0)
        let blob = try await store.storeRecordingBlob(
            sessionID: "sess_test", data: minimalWAV(),
            fileExtension: "wav", mimeType: "audio/wav", role: "user_audio")
        XCTAssertTrue(blob.relativePath.hasPrefix("blob/"))
        let beforeClose = try await store.recording(sessionID: "sess_test")
        XCTAssertEqual(beforeClose.events.count, 1)
        XCTAssertEqual(beforeClose.blobs.count, 1)
        _ = try await store.close(sessionID: "sess_test", reason: "test")
        let summaries = try await store.list()
        XCTAssertEqual(summaries.first?.metadata.closeReason, "test")
        XCTAssertFalse(summaries.first?.active ?? true)
    }

    func testSessionIdentifiersAndLimitsPreventTraversalAndOverwrite() async throws {
        let root = try temporaryRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let store = try MiniCPMDemoSessionStore(root: root, limits: .init(maxSessionBytes: 4))
        await assertThrowsAsync { _ = try await store.create(MiniCPMDemoSessionMetadata(
            sessionID: "../escape", mode: "chat")) }
        _ = try await store.create(MiniCPMDemoSessionMetadata(sessionID: "sess_safe", mode: "chat"))
        await assertThrowsAsync { _ = try await store.create(MiniCPMDemoSessionMetadata(
            sessionID: "sess_safe", mode: "chat")) }
        await assertThrowsAsync { _ = try await store.storeRecordingBlob(
            sessionID: "sess_safe", data: Data(repeating: 1, count: 5),
            fileExtension: "wav", mimeType: "audio/wav") }
    }

    func testRecordingEventLimitIsCheckedBeforeAppend() async throws {
        let root = try temporaryRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let store = try MiniCPMDemoSessionStore(
            root: root, limits: .init(maxEventBytes: 16))
        _ = try await store.create(MiniCPMDemoSessionMetadata(
            sessionID: "sess_event_limit", mode: "chat"))
        await assertThrowsAsync { _ = try await store.appendEvent(
            sessionID: "sess_event_limit", direction: .up,
            frame: .object(["text": .string("this event is larger than sixteen bytes")])) }
        let recording = try await store.recording(sessionID: "sess_event_limit")
        XCTAssertEqual(recording.events.count, 0)
    }

    func testAssetsCommentsSharesAndExpiryAreDurable() async throws {
        let root = try temporaryRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let layout = try MiniCPMDemoStorageLayout(root: root)
        let assets = try MiniCPMDemoAssetStore(layout: layout)
        let asset = try await assets.uploadReferenceAudio(name: "voice", data: minimalWAV())
        try await assets.setDefaultReferenceAudio(asset.id)
        let defaultAudio = try await assets.defaultReferenceAudio()
        XCTAssertEqual(defaultAudio.record.id, asset.id)

        let sessions = try MiniCPMDemoSessionStore(root: root)
        _ = try await sessions.create(MiniCPMDemoSessionMetadata(sessionID: "sess_share", mode: "chat"))
        let shares = try MiniCPMDemoShareStore(layout: layout)
        let comment = try await shares.setComment(sessionID: "sess_share", text: "looks good")
        XCTAssertEqual(comment.text, "looks good")
        let share = try await shares.createShare(sessionID: "sess_share", expiresIn: 60, now: Date(timeIntervalSince1970: 100))
        let resolved = try await shares.resolveShare(token: share.token, now: Date(timeIntervalSince1970: 101))
        XCTAssertEqual(resolved.sessionID, "sess_share")
        await assertThrowsAsync { _ = try await shares.resolveShare(token: share.token, now: Date(timeIntervalSince1970: 200)) }
        try await shares.revokeShare(token: share.token)
        await assertThrowsAsync { _ = try await shares.resolveShare(token: share.token) }
    }

    func testConfigETAAndPresetLoaderUseDiskState() async throws {
        let root = try temporaryRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let layout = try MiniCPMDemoStorageLayout(root: root)
        let config = try MiniCPMDemoConfigStore(layout: layout)
        try await config.recordDuration(requestType: "chat", durationSeconds: 2)
        try await config.recordDuration(requestType: "chat", durationSeconds: 4)
        try await config.recordDuration(requestType: "chat", durationSeconds: 6)
        let estimate = await config.estimate(requestType: "chat")
        // The persisted ETA uses the configured alpha=.3 exponential moving
        // average: 2 -> 2.6 -> 3.62.
        XCTAssertEqual(estimate, 3.62, accuracy: 1e-8)
        let reloaded = try MiniCPMDemoConfigStore(layout: layout)
        let reloadedEstimate = await reloaded.estimate(requestType: "chat")
        XCTAssertEqual(reloadedEstimate, 3.62, accuracy: 1e-8)

        let project = root.appendingPathComponent("project", isDirectory: true)
        try FileManager.default.createDirectory(at: project.appendingPathComponent("assets/presets/chat", isDirectory: true), withIntermediateDirectories: true)
        let fixtureSamples: [Float] = [0, -0.25, 0.5]
        try minimalWAV(samples: fixtureSamples).write(to: project.appendingPathComponent("assets/voice.wav"))
        let yaml = """
        id: sample
        order: 1
        name: Sample
        system_prompt: |
          Hello
          world
        ref_audio_path: assets/voice.wav
        """
        try Data(yaml.utf8).write(to: project.appendingPathComponent("assets/presets/chat/sample.yaml"))
        let presets = try MiniCPMDemoPresetStore(projectRoot: project)
        let listed = try await presets.list()
        XCTAssertEqual(listed["chat"]?.first?.id, "sample")
        XCTAssertEqual(listed["chat"]?.first?.values["ref_audio"]?.objectValue?["data"], .null)
        let audio = try await presets.audio(mode: "chat", presetID: "sample")
        XCTAssertEqual(
            audio.first?.data,
            MiniCPMDemoNormalizedAudio(samples: fixtureSamples).float32Data)
        XCTAssertEqual(audio.first?.sampleRate, MiniCPMDemoAudioNormalizer.targetSampleRate)
    }
}
