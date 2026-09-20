import Foundation
import XCTest
@testable import MiniCPMDemoBackend
import Hummingbird
import HummingbirdTesting
import HTTPTypes
import NIOCore

/// Small regression tests for the Gateway/static-storage parity fixes.  These
/// intentionally exercise the public service/route surfaces rather than
/// private helper implementation details.
final class MiniCPMDemoParityTests: XCTestCase {
    private struct ParsedZipEntry {
        let name: String
        let data: Data
        let crc: UInt32
    }

    private enum ZipParseError: Error {
        case malformed
    }

    private func temporaryRoot() throws -> URL {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("minicpm-demo-parity-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        return root
    }

    private func minimalWAV() -> Data {
        MiniCPMDemoAudioNormalizer.wavData(samples: [0, 0.25, -0.5], sampleRate: 16_000)
    }

    private func makeServices(root: URL) throws -> MiniCPMDemoServiceContainer {
        let web = root.appendingPathComponent("web", isDirectory: true)
        try FileManager.default.createDirectory(at: web, withIntermediateDirectories: true)
        try Data("index".utf8).write(to: web.appendingPathComponent("index.html"))
        try Data("turnbased".utf8).write(to: web.appendingPathComponent("turnbased.html"))
        try FileManager.default.createDirectory(at: web.appendingPathComponent("docs/zh", isDirectory: true), withIntermediateDirectories: true)
        try Data("docs".utf8).write(to: web.appendingPathComponent("docs/zh/index.html"))
        try FileManager.default.createDirectory(at: web.appendingPathComponent("mobile", isDirectory: true), withIntermediateDirectories: true)
        try Data("mobile".utf8).write(to: web.appendingPathComponent("mobile/index.html"))
        try Data("asset".utf8).write(to: web.appendingPathComponent("mobile/app.js"))
        return try MiniCPMDemoServiceContainer(dataDirectory: root, webRoot: web)
    }

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

    func testBundledWebResourcesTakePrecedenceOverUpstreamStatic() throws {
        let root = try temporaryRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let upstream = root.appendingPathComponent("upstream", isDirectory: true)
        let legacyWeb = upstream.appendingPathComponent("static", isDirectory: true)
        try FileManager.default.createDirectory(at: legacyWeb, withIntermediateDirectories: true)
        try Data("legacy upstream page".utf8).write(to: legacyWeb.appendingPathComponent("index.html"))

        let services = try MiniCPMDemoServiceContainer(
            dataDirectory: root.appendingPathComponent("data", isDirectory: true),
            upstreamRoot: upstream)

        XCTAssertNotEqual(services.webRoot, legacyWeb.standardizedFileURL)
        XCTAssertTrue(FileManager.default.fileExists(
            atPath: services.webRoot.appendingPathComponent("audio-duplex/audio-duplex-app.js").path))
    }

    private func temporaryArchiveNames() -> Set<String> {
        let root = FileManager.default.temporaryDirectory
        let entries = (try? FileManager.default.contentsOfDirectory(
            at: root, includingPropertiesForKeys: [.isRegularFileKey], options: [.skipsHiddenFiles])) ?? []
        return Set(entries.filter { $0.lastPathComponent.hasPrefix("minicpm-session-") && $0.pathExtension == "zip" }.map(\.lastPathComponent))
    }

    private func readUInt16(_ data: Data, at offset: Int) throws -> UInt16 {
        guard offset >= 0, offset + 2 <= data.count else { throw ZipParseError.malformed }
        return UInt16(data[offset]) | (UInt16(data[offset + 1]) << 8)
    }

    private func readUInt32(_ data: Data, at offset: Int) throws -> UInt32 {
        guard offset >= 0, offset + 4 <= data.count else { throw ZipParseError.malformed }
        return UInt32(data[offset])
            | (UInt32(data[offset + 1]) << 8)
            | (UInt32(data[offset + 2]) << 16)
            | (UInt32(data[offset + 3]) << 24)
    }

    private func crc32(_ data: Data) -> UInt32 {
        var crc: UInt32 = 0xffffffff
        for byte in data {
            crc ^= UInt32(byte)
            for _ in 0..<8 {
                crc = (crc >> 1) ^ ((crc & 1) == 1 ? 0xedb88320 : 0)
            }
        }
        return crc ^ 0xffffffff
    }

    /// Parse the classic stored ZIP emitted by the Gateway and verify each
    /// central-directory entry against its local header/payload CRC.
    private func parseStoredZip(_ data: Data) throws -> [ParsedZipEntry] {
        guard data.count >= 22 else { throw ZipParseError.malformed }
        let endSignature: UInt32 = 0x06054b50
        var endOffset: Int?
        for offset in stride(from: data.count - 22, through: max(0, data.count - 65_557), by: -1) {
            if try readUInt32(data, at: offset) == endSignature {
                endOffset = offset
                break
            }
        }
        guard let endOffset else { throw ZipParseError.malformed }
        let count = Int(try readUInt16(data, at: endOffset + 10))
        let centralSize = Int(try readUInt32(data, at: endOffset + 12))
        let centralOffset = Int(try readUInt32(data, at: endOffset + 16))
        guard centralOffset >= 0, centralSize >= 0,
              centralOffset <= data.count, centralSize <= data.count - centralOffset else {
            throw ZipParseError.malformed
        }
        var cursor = centralOffset
        var entries: [ParsedZipEntry] = []
        for _ in 0..<count {
            guard try readUInt32(data, at: cursor) == 0x02014b50 else { throw ZipParseError.malformed }
            guard try readUInt16(data, at: cursor + 10) == 0 else { throw ZipParseError.malformed }
            let crc = try readUInt32(data, at: cursor + 16)
            let compressedSize = Int(try readUInt32(data, at: cursor + 20))
            let uncompressedSize = Int(try readUInt32(data, at: cursor + 24))
            let nameLength = Int(try readUInt16(data, at: cursor + 28))
            let extraLength = Int(try readUInt16(data, at: cursor + 30))
            let commentLength = Int(try readUInt16(data, at: cursor + 32))
            let localOffset = Int(try readUInt32(data, at: cursor + 42))
            let nameStart = cursor + 46
            let nameEnd = nameStart + nameLength
            guard nameEnd <= data.count,
                  let name = String(data: data.subdata(in: nameStart..<nameEnd), encoding: .utf8) else {
                throw ZipParseError.malformed
            }
            cursor = nameEnd + extraLength + commentLength
            guard cursor <= centralOffset + centralSize,
                  try readUInt32(data, at: localOffset) == 0x04034b50 else {
                throw ZipParseError.malformed
            }
            let localNameLength = Int(try readUInt16(data, at: localOffset + 26))
            let localExtraLength = Int(try readUInt16(data, at: localOffset + 28))
            let payloadStart = localOffset + 30 + localNameLength + localExtraLength
            let payloadEnd = payloadStart + compressedSize
            guard payloadStart >= 0, payloadEnd <= data.count,
                  compressedSize == uncompressedSize else {
                throw ZipParseError.malformed
            }
            let payload = data.subdata(in: payloadStart..<payloadEnd)
            guard crc32(payload) == crc else { throw ZipParseError.malformed }
            entries.append(ParsedZipEntry(name: name, data: payload, crc: crc))
        }
        guard cursor == centralOffset + centralSize else { throw ZipParseError.malformed }
        return entries
    }

    func testStaticRouteRegistrationDoesNotDuplicateNormalizedSlashNodes() throws {
        let root = try temporaryRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let services = try makeServices(root: root)
        let router = Router<BasicRequestContext>()

        // RouterPath drops trailing empty components, so this call itself is
        // the regression gate for `/mobile[-omni]`, `/docs`, and their
        // recursive static prefixes.  A duplicate GET/HEAD would trigger
        // Hummingbird's precondition before the assertions below run.
        registerMiniCPMDemoStaticRoutes(router, services: services)
        let mobileGets = router.routes.filter { $0.method == .get && $0.path.description == "/mobile" }
        let mobileOmniGets = router.routes.filter { $0.method == .get && $0.path.description == "/mobile-omni" }
        XCTAssertEqual(mobileGets.count, 1)
        XCTAssertEqual(mobileOmniGets.count, 1)
    }

    func testDisabledAppRedirectsAndDocsMobileStaticRoutes() async throws {
        let root = try temporaryRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let services = try makeServices(root: root)
        _ = try await services.admin.setAppEnabled(appID: "turnbased", enabled: false)

        let router = Router<BasicRequestContext>()
        registerMiniCPMDemoStaticRoutes(router, services: services)
        let app = Application(router: router)
        try await app.test(.router) { client in
            let disabled = try await client.execute(uri: "/turnbased", method: .get)
            XCTAssertEqual(disabled.status.code, 302)
            XCTAssertEqual(disabled.headers[.location], "/")

            let legacy = try await client.execute(uri: "/docs/audio", method: .get)
            XCTAssertEqual(legacy.status.code, 302)
            XCTAssertEqual(legacy.headers[.location], "/docs/zh/realtime-api/audio/")

            let docsAsset = try await client.execute(uri: "/docs/zh/index.html", method: .get)
            XCTAssertEqual(docsAsset.status.code, 200)

            let mobileRedirect = try await client.execute(uri: "/mobile", method: .get)
            XCTAssertEqual(mobileRedirect.status.code, 302)
            XCTAssertEqual(mobileRedirect.headers[.location], "/mobile/")
            let mobileAsset = try await client.execute(uri: "/mobile/app.js", method: .get)
            XCTAssertEqual(mobileAsset.status.code, 200)
        }
    }

    func testMissingRecordingStreamReturnsNotFound() async throws {
        let root = try temporaryRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let store = try MiniCPMDemoSessionStore(root: root)
        _ = try await store.create(MiniCPMDemoSessionMetadata(sessionID: "missing_stream", mode: "chat"))
        let stream = try MiniCPMDemoStorageLayout(root: root)
            .sessionURL("missing_stream").appendingPathComponent("stream.jsonl")
        try FileManager.default.removeItem(at: stream)
        await assertThrowsAsync {
            _ = try await store.recording(sessionID: "missing_stream")
        }
    }

    func testOneMalformedPresetDoesNotHideValidSibling() async throws {
        let root = try temporaryRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let presetsRoot = root.appendingPathComponent("assets/presets/chat", isDirectory: true)
        try FileManager.default.createDirectory(at: presetsRoot, withIntermediateDirectories: true)
        let valid = "id: valid\nname: Valid\n"
        try Data(valid.utf8).write(to: presetsRoot.appendingPathComponent("valid.yaml"))
        // Empty YAML is skipped by the best-effort parser.
        try Data().write(to: presetsRoot.appendingPathComponent("broken.yaml"))
        let store = try MiniCPMDemoPresetStore(projectRoot: root)
        let listed = try await store.list()
        XCTAssertEqual(listed["chat"]?.map(\.id), ["valid"])
    }

    func testInvalidRawWAVAndNumericImageRecordingAreRejected() async throws {
        let root = try temporaryRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let store = try MiniCPMDemoSessionStore(root: root)
        _ = try await store.create(MiniCPMDemoSessionMetadata(sessionID: "invalid_media", mode: "omni"))
        await assertThrowsAsync {
            _ = try await store.storeRecordingBlob(
                sessionID: "invalid_media", data: Data([1, 2, 3]),
                fileExtension: "wav", mimeType: "audio/wav")
        }

        // The wire recorder is fail-safe: malformed numeric image/frame data
        // disables recording rather than persisting a fake JPEG.
        let secondStore = try MiniCPMDemoSessionStore(root: root)
        let metadata = MiniCPMDemoSessionMetadata(sessionID: "numeric_image", mode: "omni")
        let wire = try await MiniCPMDemoWireRecordingSession(store: secondStore, metadata: metadata)
        let image: MiniCPMJSONValue = .array(
            Array(repeating: MiniCPMJSONValue.number(1.0), count: 2_049))
        await wire.record(direction: .up, frame: .object(["image": image]))
        let recording = try await secondStore.recording(sessionID: "numeric_image")
        XCTAssertTrue(recording.events.isEmpty)
        XCTAssertTrue(recording.blobs.isEmpty)
    }

    func testWireRecorderExternalizesRawAudioWithoutFormat() async throws {
        let root = try temporaryRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let store = try MiniCPMDemoSessionStore(root: root)
        let metadata = MiniCPMDemoSessionMetadata(sessionID: "raw_audio", mode: "audio")
        let wire = try await MiniCPMDemoWireRecordingSession(store: store, metadata: metadata)

        var bytes = Data()
        for index in 0..<4_096 {
            let sample: Float = switch index % 4 {
            case 0: 0
            case 1: 0.25
            case 2: -0.5
            default: 0.75
            }
            var bits = sample.bitPattern.littleEndian
            withUnsafeBytes(of: &bits) { bytes.append(contentsOf: $0) }
        }
        let frame: MiniCPMJSONValue = .object([
            "type": .string("input.append"),
            "input": .object(["audio": .string(bytes.base64EncodedString())]),
        ])
        await wire.record(direction: .up, frame: frame)

        let recording = try await store.recording(sessionID: "raw_audio")
        XCTAssertEqual(recording.events.count, 1)
        XCTAssertEqual(recording.blobs.count, 1)
        XCTAssertEqual(recording.blobs[0].mimeType, "audio/wav")
        XCTAssertTrue(recording.blobs[0].relativePath.hasSuffix(".wav"))
        XCTAssertTrue(recording.events[0].frame.objectValue?["input"]?.objectValue?["audio"]?.stringValue?.hasPrefix("@blob/") == true)

        let encodedStore = try MiniCPMDemoSessionStore(root: root)
        let encodedWire = try await MiniCPMDemoWireRecordingSession(
            store: encodedStore,
            metadata: MiniCPMDemoSessionMetadata(sessionID: "encoded_audio", mode: "audio"))
        var webm = Data([0x1a, 0x45, 0xdf, 0xa3])
        webm.append(Data(repeating: 0, count: 4_096))
        await encodedWire.record(direction: .up, frame: .object([
            "format": .string("webm"),
            "audio": .string(webm.base64EncodedString()),
        ]))
        let encodedRecording = try await encodedStore.recording(sessionID: "encoded_audio")
        XCTAssertEqual(encodedRecording.blobs.count, 1)
        // Snapshot MIME is normalized from the `.webm` extension by the
        // durable store; the wire event still preserves the requested format.
        XCTAssertEqual(encodedRecording.blobs[0].mimeType, "video/webm")
        XCTAssertTrue(encodedRecording.blobs[0].relativePath.hasSuffix(".webm"))
    }

    func testSessionDownloadContainsFilesAndCleansTemporaryArchive() async throws {
        let root = try temporaryRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let web = root.appendingPathComponent("web", isDirectory: true)
        try FileManager.default.createDirectory(at: web, withIntermediateDirectories: true)
        try Data("index".utf8).write(to: web.appendingPathComponent("index.html"))
        let services = try MiniCPMDemoServiceContainer(dataDirectory: root, webRoot: web)
        _ = try await services.sessions.create(MiniCPMDemoSessionMetadata(sessionID: "zip_session", mode: "chat"))
        _ = try await services.sessions.appendEvent(
            sessionID: "zip_session", direction: .up,
            frame: .object(["type": .string("input.append")]))

        let router = Router<BasicRequestContext>()
        let scheduler = MiniCPMDemoInferenceScheduler()
        let registry = MiniCPMDemoBackendRegistry(scheduler: scheduler)
        registerMiniCPMDemoGatewayDataRoutes(router, services: services, scheduler: scheduler, registry: registry)
        let app = Application(router: router)
        let before = temporaryArchiveNames()
        try await app.test(.router) { client in
            let response = try await client.execute(uri: "/api/sessions/zip_session/download", method: .get)
            XCTAssertEqual(response.status.code, 200)
            let data = Data(buffer: response.body)
            XCTAssertEqual(Array(data.prefix(4)), [0x50, 0x4b, 0x03, 0x04])
            guard let textLength = response.headers[.contentLength],
                  let contentLength = Int(textLength) else {
                return XCTFail("ZIP response is missing Content-Length")
            }
            XCTAssertEqual(contentLength, data.count)
            let entries = try self.parseStoredZip(data)
            XCTAssertEqual(
                Set(entries.map(\.name)),
                Set(["zip_session/meta.json", "zip_session/stream.jsonl"]))
            XCTAssertTrue(entries.allSatisfy { !$0.data.isEmpty && self.crc32($0.data) == $0.crc })
        }
        XCTAssertEqual(temporaryArchiveNames(), before)
    }
}
