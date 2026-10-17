import Foundation
import XCTest
@testable import MiniCPMDemoBackend

final class MiniCPMWireEncodingTests: XCTestCase {
    private func pcmRequest(samples: Int = 16_000) -> MiniCPMDemoRequest {
        .init(type: "input.append", input: .object([
            "audio": .array((0..<samples).map { .number(Double(Float($0 % 31 - 15) / 32)) }),
            "sample_rate": .number(16_000), "speech_active": .bool(true),
            "metadata": .object(["未知": .array([.null, .string("语音 🎧\n\"\\")])]),
        ]), messageID: "input-1")
    }

    func testSharedCanonicalRequestPreservesPayloadAndExactQueueByteCount() throws {
        let fixtures: [MiniCPMDemoRequest] = [
            .init(type: "session.close"),
            .init(type: "control", payload: .null, input: .object(["x": .bool(false)]),
                  messageID: "消息 🎧", reason: "\n\"\\"),
            pcmRequest(),
            .init(type: "input.append", input: .object(["audio": .array([
                .number(-0.0), .number(0.0), .number(Double(Float.leastNonzeroMagnitude)),
                .number(Double(Float.greatestFiniteMagnitude)),
            ])])),
        ]
        for request in fixtures {
            let expectedData = try MiniCPMDemoPersistence.encoder().encode(request)
            let expectedValue = try MiniCPMDemoPersistence.decoder().decode(
                MiniCPMJSONValue.self, from: expectedData)
            let shared = try MiniCPMDemoWireRecordingSupport.requestValueAndEncodedByteCount(request)
            XCTAssertEqual(shared.value, expectedValue)
            XCTAssertEqual(shared.encodedBytes, expectedData.count)
            let item = MiniCPMDemoQueuedInput(request: request, epoch: 0, encodedBytes: shared.encodedBytes)
            let exact = MiniCPMDemoInputQueue(itemCapacity: 1, byteCapacity: shared.encodedBytes)
            try exact.enqueue(item)
            XCTAssertEqual(exact.queuedByteCount, expectedData.count)
            exact.finish()
            let insufficient = MiniCPMDemoInputQueue(itemCapacity: 1, byteCapacity: shared.encodedBytes - 1)
            XCTAssertThrowsError(try insufficient.enqueue(item))
            insufficient.finish()
        }
        let nils = try MiniCPMDemoWireRecordingSupport.requestValueAndEncodedByteCount(fixtures[0]).value
        XCTAssertEqual(nils, .object(["type": .string("session.close")]))
        let explicitNull = try MiniCPMDemoWireRecordingSupport.requestValueAndEncodedByteCount(fixtures[1]).value
        XCTAssertEqual(explicitNull.objectValue?["payload"], .null)
    }

    func testNonfiniteCanonicalInputStillFailsBeforeRecordingOrEnqueue() {
        for number in [Double.nan, .infinity, -.infinity] {
            let request = MiniCPMDemoRequest(type: "input.append", input: .object([
                "audio": .array([.number(number)]),
            ]))
            XCTAssertThrowsError(try MiniCPMDemoWireRecordingSupport.requestValueAndEncodedByteCount(request))
        }
    }

    func testOfficialWireAndRecordingShareOneEnvelopeAndTimestamp() throws {
        for type in ["response.output.delta", "session.created", "session.closed"] {
            for silent in [true, false] {
                var event = MiniCPMDemoEvent(eventID: "private-id", type: type, payload: [
                    "kind": .string("audio"), "audio": .string("AAAA"),
                    "sample_rate": .number(24_000), "silent": .bool(silent),
                    "mode": .string("full_duplex"), "reason": .string("payload-reason"),
                    "unknown": .number(.nan),
                ], sessionID: "session-1", responseID: "response-1", inputID: "input-1")
                event.text = "surface 🎧\n\"\\"
                event.reason = "surface-reason"
                let shared = try encodedOfficialMiniCPMDemoEvent(event)
                let decoded = try JSONDecoder().decode(MiniCPMJSONValue.self, from: Data(shared.wire.utf8))
                XCTAssertEqual(shared.value, decoded)
                XCTAssertEqual(decoded.objectValue?["text"], .string(event.text!))
                XCTAssertEqual(decoded.objectValue?["silent"], .bool(silent))
                XCTAssertNotNil(decoded.objectValue?["server_send_ts"]?.numberValue)
                XCTAssertNil(decoded.objectValue?["event_id"])
                XCTAssertNil(decoded.objectValue?["unknown"])
                XCTAssertEqual(decoded.objectValue?["reason"],
                               .string(type == "session.closed" ? "payload-reason" : "surface-reason"))
                XCTAssertEqual(decoded.objectValue?["mode"], type == "session.created" ? .string("full_duplex") : nil)
            }
        }
        var invalid = MiniCPMDemoEvent(eventID: "invalid", type: "response.output.delta")
        invalid.metrics = ["wall_clock_ms": .number(.infinity)]
        XCTAssertThrowsError(try encodedOfficialMiniCPMDemoEvent(invalid))
    }

    func testWireEncodingTimingProbeWhenRequested() throws {
        guard ProcessInfo.processInfo.environment["MINICPM_WIRE_BENCHMARK"] == "1" else {
            throw XCTSkip("opt-in CPU encoding benchmark")
        }
        var checksum = 0
        func median(_ body: () throws -> Void) rethrows -> Double {
            for _ in 0..<3 { try body() }
            var values: [Double] = []
            for _ in 0..<15 {
                let start = DispatchTime.now().uptimeNanoseconds
                try body()
                values.append(Double(DispatchTime.now().uptimeNanoseconds - start) / 1_000_000)
            }
            return values.sorted()[values.count / 2]
        }
        for count in [16_000, 160_000] {
            let request = pcmRequest(samples: count)
            let before = try median {
                let value = try MiniCPMDemoWireRecordingSupport.requestValueAndEncodedByteCount(request).value
                let bytes = try MiniCPMDemoWireRecordingSupport.requestValueAndEncodedByteCount(request).encodedBytes
                checksum += (value.objectValue?.count ?? 0) + bytes
            }
            let after = try median {
                let canonical = try MiniCPMDemoWireRecordingSupport.requestValueAndEncodedByteCount(request)
                checksum += (canonical.value.objectValue?.count ?? 0) + canonical.encodedBytes
            }
            print("WIRE_BENCH request_samples=\(count) old_ms=\(before) shared_ms=\(after)")
        }
        let audio = Data(repeating: 1, count: 96_000).base64EncodedString()
        let event = MiniCPMDemoEvent(eventID: "bench", type: "response.output.delta", payload: [
            "audio": .string(audio), "kind": .string("audio"),
        ])
        let before = try median {
            let wire = try encodeOfficialMiniCPMDemoEvent(event)
            let value = try JSONDecoder().decode(MiniCPMJSONValue.self, from: Data(wire.utf8))
            checksum += value.objectValue?.count ?? 0
        }
        let after = try median {
            let shared = try encodedOfficialMiniCPMDemoEvent(event)
            checksum += shared.value.objectValue?.count ?? 0
        }
        print("WIRE_BENCH output_audio_bytes=96000 old_ms=\(before) shared_ms=\(after) checksum=\(checksum)")
    }
}
