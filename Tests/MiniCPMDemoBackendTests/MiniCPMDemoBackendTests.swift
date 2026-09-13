import XCTest
@testable import MiniCPMDemoBackend
import MiniCPMDuplexRuntime
import Hummingbird
import HummingbirdTesting
import HummingbirdWebSocket
import HummingbirdWSTesting

final class MiniCPMDemoBackendTests: XCTestCase {
    func testJSONValueRoundTripPreservesNestedPayload() throws {
        let value: MiniCPMJSONValue = .object([
            "mode": .string("full_duplex"),
            "config": .object(["temperature": .number(0.7), "enabled": .bool(true)]),
            "items": .array([.string("a"), .null]),
        ])
        let data = try JSONEncoder().encode(value)
        let decoded = try JSONDecoder().decode(MiniCPMJSONValue.self, from: data)
        XCTAssertEqual(decoded, value)
    }

    func testInputQueueRejectsItemCountOverflowAndReleasesDequeuedBudget() async throws {
        let queue = MiniCPMDemoInputQueue(itemCapacity: 2, byteCapacity: 1_000)
        let item = MiniCPMDemoQueuedInput(
            request: MiniCPMDemoRequest(type: "input.append"), epoch: 0, encodedBytes: 100)

        try queue.enqueue(item)
        try queue.enqueue(item)
        XCTAssertEqual(queue.queuedItemCount, 2)
        XCTAssertEqual(queue.queuedByteCount, 200)
        XCTAssertThrowsError(try queue.enqueue(item)) { error in
            XCTAssertEqual(
                error as? MiniCPMDemoInputQueueError,
                .full(itemCount: 2, byteCount: 200, itemLimit: 2, byteLimit: 1_000))
        }

        var iterator = queue.stream.makeAsyncIterator()
        guard let dequeued = await iterator.next() else {
            return XCTFail("queue did not yield the first FIFO item")
        }
        queue.release(dequeued)
        XCTAssertEqual(queue.queuedItemCount, 1)
        XCTAssertEqual(queue.queuedByteCount, 100)
        try queue.enqueue(item)
        XCTAssertEqual(queue.queuedItemCount, 2)
        queue.finish()
    }

    func testInputQueueRejectsByteOverflowAndOversizedSingleItem() async throws {
        let queue = MiniCPMDemoInputQueue(itemCapacity: 8, byteCapacity: 10)
        let sixBytes = MiniCPMDemoQueuedInput(
            request: MiniCPMDemoRequest(type: "input.append"), epoch: 0, encodedBytes: 6)
        let fiveBytes = MiniCPMDemoQueuedInput(
            request: MiniCPMDemoRequest(type: "input.append"), epoch: 0, encodedBytes: 5)
        try queue.enqueue(sixBytes)
        XCTAssertThrowsError(try queue.enqueue(fiveBytes)) { error in
            XCTAssertEqual(
                error as? MiniCPMDemoInputQueueError,
                .full(itemCount: 1, byteCount: 6, itemLimit: 8, byteLimit: 10))
        }

        var iterator = queue.stream.makeAsyncIterator()
        guard let dequeued = await iterator.next() else {
            return XCTFail("queue did not yield the byte-budget item")
        }
        queue.release(dequeued)
        let exactLimit = MiniCPMDemoQueuedInput(
            request: MiniCPMDemoRequest(type: "input.append"), epoch: 0, encodedBytes: 10)
        try queue.enqueue(exactLimit)
        XCTAssertThrowsError(try queue.enqueue(
            MiniCPMDemoQueuedInput(
                request: MiniCPMDemoRequest(type: "input.append"), epoch: 0, encodedBytes: 11))) { error in
            XCTAssertEqual(error as? MiniCPMDemoInputQueueError, .itemTooLarge(size: 11, byteLimit: 10))
            XCTAssertTrue(error.localizedDescription.contains("close_code=1009"))
        }
        XCTAssertEqual(queue.queuedItemCount, 1)
        XCTAssertEqual(queue.queuedByteCount, 10)
        queue.finish()
    }

    func testModeAliasesKeepOmniDistinctFromAudioDuplex() throws {
        XCTAssertEqual(try MiniCPMBackendMode(rawOrAlias: "audio"), .fullDuplex)
        XCTAssertEqual(try MiniCPMBackendMode(rawOrAlias: "full"), .fullDuplex)
        XCTAssertEqual(try MiniCPMBackendMode(rawOrAlias: "video"), .omni)
        XCTAssertEqual(try MiniCPMBackendMode(rawOrAlias: "omni"), .omni)
    }

    func testRealtimePublicModesMapToBackendModesWithoutVideoDowngrade() throws {
        XCTAssertEqual(try MiniCPMRealtimeMode(rawOrAlias: "chat").backendMode, .turnBased)
        XCTAssertEqual(try MiniCPMRealtimeMode(rawOrAlias: "half").backendMode, .halfDuplex)
        XCTAssertEqual(try MiniCPMRealtimeMode(rawOrAlias: "half_duplex").backendMode, .halfDuplex)
        XCTAssertEqual(try MiniCPMRealtimeMode(rawOrAlias: "video").backendMode, .omni)
        XCTAssertEqual(try MiniCPMRealtimeMode(rawOrAlias: "audio").backendMode, .fullDuplex)
        XCTAssertThrowsError(try MiniCPMRealtimeMode(rawOrAlias: "unknown"))
    }

    func testCanonicalInputAppendConsumesForceListenAcrossBufferedAudioOnly() throws {
        let audio: MiniCPMJSONValue = .array([.number(0)])
        let request = MiniCPMDemoRequest(
            type: "input.append",
            input: .object(["audio": audio]))

        // The canonical frontend envelope must consume the latch just like
        // the legacy audio_chunk/input_audio_buffer aliases.
        let first = try translateRealtimeRequest(
            request,
            defaultMode: .fullDuplex,
            didInit: true,
            forceListenNext: true,
            allowSessionClose: true)
        XCTAssertTrue(first.consumedForceListen)
        XCTAssertEqual(first.request.input?.objectValue?["force_listen"]?.boolValue, true)

        // A short mel warm-up does not reach generate, so the latch survives
        // for the next canonical input packet.
        let awaitingAudio = MiniCPMDemoEvent(
            eventID: "buffered",
            type: "response.output.delta",
            payload: ["kind": .string("listen"), "reason": .string("awaiting_audio")])
        var latch = isAwaitingAudioEvent(awaitingAudio)
        XCTAssertTrue(latch)
        let second = try translateRealtimeRequest(
            request,
            defaultMode: .fullDuplex,
            didInit: true,
            forceListenNext: latch,
            allowSessionClose: true)
        XCTAssertTrue(second.consumedForceListen)
        XCTAssertEqual(second.request.input?.objectValue?["force_listen"]?.boolValue, true)

        // Once a real generation result is observed, the one-shot latch is
        // cleared and the following packet must not be forced again.
        let generated = MiniCPMDemoEvent(
            eventID: "generated",
            type: "response.output.delta",
            payload: ["kind": .string("text"), "text": .string("ok")])
        latch = isAwaitingAudioEvent(generated)
        XCTAssertFalse(latch)
        let third = try translateRealtimeRequest(
            request,
            defaultMode: .fullDuplex,
            didInit: true,
            forceListenNext: latch,
            allowSessionClose: true)
        XCTAssertFalse(third.consumedForceListen)
        XCTAssertNil(third.request.input?.objectValue?["force_listen"])
    }

    func testWireDecoderUnwrapsNestedVoiceAndHonorsPCM16Format() {
        let pcm16 = Data([0x00, 0x00, 0xff, 0x7f, 0x00, 0x80])
        let normalized = MiniCPMDemoWireDecoder.normalizeObject([
            "voice": .object([
                "data": .string(pcm16.base64EncodedString()),
                "format": .string("pcm_s16le"),
            ])
        ])
        guard case .array(let samples) = normalized["voice"],
              samples.compactMap(\.numberValue).count == 3 else {
            XCTFail("nested voice payload was not decoded")
            return
        }
        let values = samples.compactMap(\.numberValue)
        XCTAssertEqual(values[0], 0, accuracy: 1e-6)
        XCTAssertEqual(values[1], 32767.0 / 32768.0, accuracy: 1e-6)
        XCTAssertEqual(values[2], -1, accuracy: 1e-6)
    }

    func testStrictWireDecoderAcceptsOnlyFiniteFloat32PCM() throws {
        var bytes = Data()
        for sample in [Float(0), Float(1), Float(-1)] {
            var bits = sample.bitPattern.littleEndian
            withUnsafeBytes(of: &bits) { bytes.append(contentsOf: $0) }
        }
        let normalized = try MiniCPMDemoWireDecoder.normalizeObjectStrict([
            "audio": .string(bytes.base64EncodedString())
        ])
        guard case .array(let values) = normalized["audio"] else {
            return XCTFail("strict Float32 audio was not decoded")
        }
        XCTAssertEqual(values.compactMap(\.numberValue), [0, 1, -1])

        XCTAssertThrowsError(try MiniCPMDemoWireDecoder.normalizeObjectStrict([
            "audio": .string(Data([0, 0, 0]).base64EncodedString())
        ]))
        XCTAssertThrowsError(try MiniCPMDemoWireDecoder.normalizeObjectStrict([
            "audio": .string(Data("RIFFxxxxWAVE".utf8).base64EncodedString())
        ]))
        var nanBits = Float.nan.bitPattern.littleEndian
        var nanData = Data()
        withUnsafeBytes(of: &nanBits) { nanData.append(contentsOf: $0) }
        XCTAssertThrowsError(try MiniCPMDemoWireDecoder.normalizeObjectStrict([
            "audio": .string(nanData.base64EncodedString())
        ]))
    }

    func testStrictWireDecoderDecodesContentAudioDataKey() throws {
        var bits = Float(0.25).bitPattern.littleEndian
        var data = Data()
        withUnsafeBytes(of: &bits) { data.append(contentsOf: $0) }
        let normalized = try MiniCPMDemoWireDecoder.normalizeObjectStrict([
            "messages": .array([
                .object([
                    "role": .string("user"),
                    "content": .array([
                        .object([
                            "type": .string("audio"),
                            "data": .string(data.base64EncodedString())
                        ])
                    ])
                ])
            ])
        ])
        guard case .array(let messages) = normalized["messages"],
              case .object(let message) = messages.first,
              case .array(let content) = message["content"],
              case .object(let audio) = content.first,
              case .array(let samples) = audio["data"] else {
            return XCTFail("content audio data was not decoded")
        }
        XCTAssertEqual(samples.compactMap(\.numberValue), [0.25])
    }

    #if canImport(AVFoundation)
    func testRealMP4FixtureExtractsFramesAndAudio() throws {
        let data = Data(base64Encoded: Self.mp4FixtureBase64)!
        let payload = try MiniCPMDemoMediaDecoder.decodeVideo(
            data, stackFrames: true, maxFrames: 8, audioSampleRate: 16_000)
        XCTAssertGreaterThanOrEqual(payload.frames.count, 2)
        XCTAssertTrue(payload.frames.allSatisfy {
            $0.count > 4 && $0[$0.startIndex] == 0xff && $0[$0.startIndex + 1] == 0xd8
        })
        XCTAssertNotNil(payload.audio)
        XCTAssertGreaterThan((payload.audio ?? []).count, 100)
        XCTAssertEqual(payload.audioSampleRate, 16_000)

        let single = try MiniCPMDemoMediaDecoder.decodeVideo(
            data, stackFrames: false, maxFrames: 8, audioSampleRate: 16_000)
        XCTAssertEqual(single.frames.count, 1)
    }
    #endif

    func testDispatcherMaintainsOfficialSessionLifecycleAndTwoStepOrder() async {
        let engine = RecordingEngine()
        let backend = MiniCPMDemoBackend(engine: engine)
        let initEvents = await backend.handle(MiniCPMDemoRequest(type: "session.init", payload: .object(["mode": .string("half_duplex")])))
        XCTAssertEqual(initEvents.map(\.type), ["backend.initialized", "session.created", "backend.state"])

        let input = MiniCPMDemoRequest(
            type: "input.append",
            input: .object([
                "messages": .array([]),
                "audio": .string(Data([0, 0]).base64EncodedString()),
            ]))
        let events = await backend.handle(input)
        XCTAssertTrue(events.contains(where: { $0.type == "response.output.delta" && $0.kind == "text" }))
        let calls = await engine.calls
        XCTAssertEqual(calls, ["initialize", "prefill", "generate"])
    }

    func testCancelForwardsInterrupt() async {
        let engine = RecordingEngine()
        let backend = MiniCPMDemoBackend(engine: engine)
        _ = await backend.handle(MiniCPMDemoRequest(type: "session.init", payload: .object(["mode": .string("full_duplex")])))
        _ = await backend.handle(MiniCPMDemoRequest(type: "control", payload: .object(["type": .string("response.cancel")])))
        let interruptCount = await engine.interruptCount
        XCTAssertEqual(interruptCount, 1)
    }

    func testCancelBackendStateCarriesStructuredRuntimeTrace() async {
        let modelEngine = TraceEngine()
        let runtime = MiniCPMDuplexRuntime(engineFactory: { modelEngine })
        let adapter = MiniCPMDuplexRuntimeEngineAdapter(
            runtime: runtime, sessionID: "backend-trace")
        let backend = MiniCPMDemoBackend(engine: adapter)

        _ = await backend.handle(MiniCPMDemoRequest(
            type: "session.init",
            payload: .object(["mode": .string("full_duplex")])))
        let events = await backend.handle(MiniCPMDemoRequest(
            type: "control",
            payload: .object(["type": .string("response.cancel")])))

        guard let state = events.first(where: {
            $0.type == "backend.state" && $0.reason == "interrupt"
        }) else {
            return XCTFail("cancel did not emit an interrupt backend.state")
        }
        guard let trace = state.payload["trace"]?.objectValue else {
            return XCTFail("interrupt backend.state did not include trace")
        }
        XCTAssertEqual(trace["session_id"]?.stringValue, "backend-trace")
        XCTAssertEqual(trace["epoch"]?.numberValue, 1)
        XCTAssertEqual(trace["engine_epoch"]?.numberValue, 1)
        XCTAssertEqual(trace["unit_index"]?.numberValue, 7)
        XCTAssertEqual(trace["committed_unit_count"]?.numberValue, 3)
        XCTAssertEqual(trace["kv_cache_length"]?.numberValue, 128)
        XCTAssertEqual(trace["sliding_window_mode"]?.stringValue, "basic")
        XCTAssertEqual(trace["sliding_window_high_watermark"]?.numberValue, 256)
        XCTAssertEqual(trace["sliding_window_low_watermark"]?.numberValue, 192)
        XCTAssertEqual(trace["pending_token_count"]?.numberValue, 2)
        XCTAssertEqual(trace["pending_terminator"]?.numberValue, 99)
        XCTAssertEqual(trace["prefetched"]?.boolValue, false)
        XCTAssertEqual(trace["needs_finalize"]?.boolValue, false)
        XCTAssertEqual(trace["tts_context_present"]?.boolValue, true)
        XCTAssertEqual(trace["tts_pending_token"]?.numberValue, 42)
        XCTAssertEqual(trace["tts_committed_token_count"]?.numberValue, 5)
        XCTAssertEqual(trace["tts_cache_length"]?.numberValue, 11)
        XCTAssertEqual(trace["token2wav_prepared"]?.boolValue, true)
        XCTAssertEqual(trace["token2wav_pending_token_count"]?.numberValue, 4)
        XCTAssertEqual(trace["token2wav_decoder_positions"]?.numberValue, 8)
        XCTAssertEqual(trace["continuation_only"]?.boolValue, true)
        XCTAssertEqual(trace["audio_encoder_ms"]?.numberValue, 101)
        XCTAssertEqual(trace["audio_bridge_ms"]?.numberValue, 12)
        XCTAssertEqual(trace["llm_prefill_ms"]?.numberValue, 203)
        XCTAssertEqual(trace["llm_decode_ms"]?.numberValue, 304)
        XCTAssertEqual(trace["semantic_tts_ms"]?.numberValue, 405)
        XCTAssertEqual(trace["token2wav_ms"]?.numberValue, 506)
        XCTAssertEqual(trace["flow_ms"]?.numberValue, 307)
        XCTAssertEqual(trace["hift_ms"]?.numberValue, 199)
        XCTAssertEqual(trace["stale_output_discard_count"]?.numberValue, 0)
        XCTAssertNotNil(trace["cancel_requested_at_ns"]?.numberValue)
        XCTAssertNotNil(trace["cancel_returned_at_ns"]?.numberValue)
        XCTAssertGreaterThanOrEqual(
            trace["cancel_returned_at_ns"]?.numberValue ?? 0,
            trace["cancel_requested_at_ns"]?.numberValue ?? 0)
        XCTAssertEqual(state.payload["metrics"]?.objectValue, trace)
    }

    func testOpenStreamingChunkDoesNotFabricateTurnDone() async {
        let engine = OpenStreamingEngine()
        let backend = MiniCPMDemoBackend(engine: engine)
        _ = await backend.handle(MiniCPMDemoRequest(
            type: "session.init",
            payload: .object(["mode": .string("full_duplex")])))
        let events = await backend.handle(MiniCPMDemoRequest(
            type: "input.append",
            input: .object(["text": .string("continue")])))
        XCTAssertTrue(events.contains { $0.type == "response.output.delta" && $0.text == "好的" })
        XCTAssertFalse(events.contains { $0.type == "response.done" })
        let finalizeCount = await engine.finalizeCount
        XCTAssertEqual(finalizeCount, 0)
    }

    func testPerUnitMetricsDoNotCarryTimingOrPaddingIntoContinuation() async {
        let backend = MiniCPMDemoBackend(engine: PerUnitMetricsEngine())
        _ = await backend.handle(MiniCPMDemoRequest(
            type: "session.init",
            payload: .object(["mode": .string("full_duplex")])))

        let first = await backend.handle(MiniCPMDemoRequest(
            type: "input.append", input: .object(["text": .string("first")])))
        let firstMetrics = first.first(where: { $0.kind == "audio" })?.metrics
        XCTAssertEqual(firstMetrics?["audio_encoder_ms"]?.numberValue, 123)
        XCTAssertEqual(firstMetrics?["audio_padding_samples"]?.numberValue, 80)

        let second = await backend.handle(MiniCPMDemoRequest(
            type: "input.append", input: .object(["text": .string("continue")])))
        let secondMetrics = second.first(where: { $0.kind == "audio" })?.metrics
        XCTAssertEqual(secondMetrics?["continuation_only"]?.boolValue, true)
        XCTAssertNil(secondMetrics?["audio_encoder_ms"])
        XCTAssertNil(secondMetrics?["audio_padding_samples"])
    }

    func testResetReinitializesWithSavedSessionParametersAndPreservesCapabilityTruth() async throws {
        let engine = ResetRecordingEngine()
        let backend = MiniCPMDemoBackend(engine: engine)
        let parameters: [String: MiniCPMJSONValue] = [
            "mode": .string("full_duplex"),
            "voice": .object(["name": .string("female")]),
            "config": .object(["temperature": .number(0.2)]),
        ]
        let initialized = await backend.handle(.init(type: "session.init", payload: .object(parameters)))
        guard let capabilities = initialized
            .first(where: { $0.type == "backend.initialized" })?
            .payload["capabilities"]?.objectValue else {
            return XCTFail("missing backend capabilities")
        }
        XCTAssertEqual(capabilities["video_input"]?.boolValue, false)

        let reset = await backend.handle(.init(
            type: "control",
            payload: .object(["type": .string("reset")])))
        XCTAssertTrue(reset.contains { $0.type == "backend.state" && $0.reason == "reset" })
        let resetCount = await engine.resetCount
        let initializeParameters = await engine.initializeParameters
        XCTAssertEqual(resetCount, 1)
        XCTAssertEqual(initializeParameters.last, parameters)

        _ = await backend.handle(.init(
            type: "input.append",
            input: .object(["text": .string("after reset")])))
        let generateCount = await engine.generateCount
        XCTAssertEqual(generateCount, 1)
    }

    func testOmniCapabilityIsTheOnlyModeThatAdvertisesVideoInput() async {
        let engine = ResetRecordingEngine()
        let backend = MiniCPMDemoBackend(engine: engine)
        let initialized = await backend.handle(.init(
            type: "session.init",
            payload: .object(["mode": .string("video")])))
        let capabilities = initialized
            .first(where: { $0.type == "backend.initialized" })?
            .payload["capabilities"]?.objectValue
        XCTAssertEqual(capabilities?["video_input"]?.boolValue, true)
    }

    func testRealtimeReaderAcceptsCancelWhileGenerationIsBlockedAndDropsOldAudio() async throws {
        let engine = BlockedTransportEngine()
        let app = Application(
            router: Router(),
            server: makeMiniCPMDemoWebSocketServer(
                configuration: .init(autoPing: .disabled),
                makeBackend: { MiniCPMDemoBackend(engine: engine) },
                emitQueueEvents: false))

        _ = try await app.test(.live) { client in
            try await client.ws("/v1/realtime?mode=audio") { inbound, outbound, _ in
                var iterator = inbound.messages(maxSize: .max).makeAsyncIterator()
                try await outbound.write(.text("{\"type\":\"session.init\",\"payload\":{\"mode\":\"full_duplex\"}}"))
                var created = false
                while !created, let message = try await iterator.next() {
                    if case .text(let text) = message, text.contains("session.created") {
                        created = true
                    }
                }
                XCTAssertTrue(created)

                try await outbound.write(.text("{\"type\":\"input.append\",\"input\":{\"text\":\"first\"}}"))
                await engine.waitUntilStarted()
                try await outbound.write(.text("{\"type\":\"response.cancel\"}"))

                var sawInterrupt = false
                while !sawInterrupt, let message = try await iterator.next() {
                    if case .text(let text) = message,
                       text.contains("session.state"), text.contains("interrupt") {
                        sawInterrupt = true
                    }
                }
                XCTAssertTrue(sawInterrupt)
                let interruptCount = await engine.interruptCount
                XCTAssertEqual(interruptCount, 1)

                try await outbound.write(.text("{\"type\":\"input.append\",\"input\":{\"text\":\"second\"}}"))
                var sawNew = false
                var sawOld = false
                while !sawNew, let message = try await iterator.next() {
                    if case .text(let text) = message {
                        sawOld = sawOld || text.contains(Data("old-audio".utf8).base64EncodedString())
                        sawNew = text.contains(Data("new-audio".utf8).base64EncodedString())
                    }
                }
                XCTAssertFalse(sawOld, "cancelled transaction leaked stale audio")
                XCTAssertTrue(sawNew)

                try await outbound.write(.text("{\"type\":\"session.close\"}"))
                var sawClosed = false
                while !sawClosed, let message = try await iterator.next() {
                    if case .text(let text) = message, text.contains("session.closed") {
                        sawClosed = true
                    }
                }
                XCTAssertTrue(sawClosed)
            }
        }
    }

    func testRealtimeDisconnectInterruptsBlockedGenerationAndReleasesLease() async throws {
        let engine = BlockedTransportEngine()
        let scheduler = MiniCPMDemoInferenceScheduler(maxConcurrent: 1)
        let registry = MiniCPMDemoBackendRegistry(scheduler: scheduler)
        let app = Application(
            router: Router(),
            server: makeMiniCPMDemoWebSocketServer(
                configuration: .init(autoPing: .disabled),
                makeBackend: { MiniCPMDemoBackend(engine: engine) },
                registry: registry,
                scheduler: scheduler,
                emitQueueEvents: false))

        _ = try await app.test(.live) { client in
            try await client.ws("/v1/realtime?mode=audio") { inbound, outbound, _ in
                var iterator = inbound.messages(maxSize: .max).makeAsyncIterator()
                try await outbound.write(.text("{\"type\":\"session.init\",\"payload\":{\"mode\":\"full_duplex\"}}"))
                while let message = try await iterator.next() {
                    if case .text(let text) = message, text.contains("session.created") { break }
                }
                try await outbound.write(.text("{\"type\":\"input.append\",\"input\":{\"text\":\"blocked\"}}"))
                await engine.waitUntilStarted()
                // Returning without session.close exercises the transport EOF
                // path.  The server must interrupt the gate before releasing
                // its scheduler lease/registry entry.
            }
        }

        let interruptCount = await engine.interruptCount
        XCTAssertEqual(interruptCount, 1)
        let snapshot = await scheduler.snapshot()
        XCTAssertEqual(snapshot.active, 0)
        XCTAssertEqual(snapshot.queued, 0)
        let registryCount = await registry.count()
        XCTAssertEqual(registryCount, 0)

        // Re-admission must work immediately after transport teardown.  This
        // catches a lease that was not released after interrupting generation.
        guard case .granted(let followupLease, _) = try await scheduler.enqueue(
            sessionID: "after_disconnect") else {
            return XCTFail("teardown left the scheduler lease occupied")
        }
        await scheduler.release(followupLease)
    }

    func testServerTeardownHasSingleLeaseReleaseAndGenerationEpochReturn() throws {
        let sourceURL = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent() // MiniCPMDemoBackendTests
            .deletingLastPathComponent() // Tests
            .deletingLastPathComponent()
            .appendingPathComponent("Sources/MiniCPMDemoBackend/Server.swift")
        let source = try String(contentsOf: sourceURL, encoding: .utf8)

        let teardownRelease = "if let lease, let scheduler { await scheduler.release(lease) }"
        XCTAssertEqual(
            source.components(separatedBy: teardownRelease).count - 1,
            1,
            "teardown must release its lease exactly once")
        XCTAssertTrue(
            source.contains("inputQueue.finish()\n            throw error"),
            "a failed generation worker must close its FIFO before exiting")
        XCTAssertTrue(
            source.contains("inputID: request.messageID"),
            "backpressure must identify the exact rejected input for bounded retry")

        guard let emitStart = source.range(of: "private func emitMiniCPMDemoEvents("),
              let emitEnd = source.range(
                of: "\nprivate func writeMiniCPMDemoControlEvent",
                range: emitStart.upperBound..<source.endIndex) else {
            return XCTFail("could not locate event emitter")
        }
        let emitter = String(source[emitStart.lowerBound..<emitEnd.lowerBound])
        XCTAssertEqual(
            emitter.components(separatedBy: "\n            return\n").count - 1,
            1,
            "stale generation output should have one early return")
    }

    func testShortAudioPrefillBuffersWithoutGeneratingUntilSecondChunk() async throws {
        let engine = TwoChunkEngine()
        let backend = MiniCPMDemoBackend(engine: engine)
        _ = await backend.handle(MiniCPMDemoRequest(
            type: "session.init",
            payload: .object(["mode": .string("full_duplex")])))
        let chunk = MiniCPMJSONValue.array(
            (0..<16_000).map { _ in MiniCPMJSONValue.number(0) })
        let first = await backend.handle(MiniCPMDemoRequest(
            type: "input.append",
            input: .object(["audio": chunk])))
        XCTAssertFalse(first.contains { $0.type == "response.done" })
        let firstGenerateCount = await engine.generateCount
        XCTAssertEqual(firstGenerateCount, 0)

        let second = await backend.handle(MiniCPMDemoRequest(
            type: "input.append",
            input: .object(["audio": chunk])))
        XCTAssertTrue(second.contains {
            $0.type == "response.output.delta" && $0.kind == "listen"
        })
        let secondGenerateCount = await engine.generateCount
        XCTAssertEqual(secondGenerateCount, 1)
        try await backend.commitPendingResponse()
    }

    func testFinalizeWaitsForTransportCommit() async throws {
        let engine = RecordingEngine()
        let backend = MiniCPMDemoBackend(engine: engine)
        _ = await backend.handle(MiniCPMDemoRequest(
            type: "session.init",
            payload: .object(["mode": .string("turn_based")])))
        _ = await backend.handle(MiniCPMDemoRequest(
            type: "input.append",
            input: .object(["text": .string("hello")])))
        let before = await engine.finalizeCount
        XCTAssertEqual(before, 0)
        try await backend.commitPendingResponse()
        let after = await engine.finalizeCount
        XCTAssertEqual(after, 1)
    }

    func testDefaultVoiceIsPreparedBeforeFirstPrefillAndRefAudioStaysLLMOnly() async throws {
        let engine = CapturingDuplexEngine()
        let runtime = MiniCPMDuplexRuntime(engineFactory: { engine })
        let reference = [Float(0.1), 0.2]
        let defaultVoice = [Float(0.3), 0.4, 0.5]
        let adapter = MiniCPMDuplexRuntimeEngineAdapter(
            runtime: runtime,
            sessionID: "voice-test",
            defaultVoicePCM: defaultVoice,
            defaultVoiceSampleRate: 22_050)

        try await adapter.initialize(
            mode: .fullDuplex,
            parameters: [
                "voice": .object([
                    "ref_audio": .array(reference.map { .number(Double($0)) }),
                    "sample_rate": .number(16_000),
                ])
            ])

        guard let prepared = engine.preparedConfig else {
            return XCTFail("runtime.prepare was not called")
        }
        XCTAssertEqual(prepared.llmReferenceAudio, reference)
        XCTAssertEqual(prepared.llmReferenceAudioSampleRate, 16_000)
        XCTAssertEqual(prepared.ttsReferenceAudio, defaultVoice)
        XCTAssertEqual(prepared.ttsReferenceAudioSampleRate, 22_050)

        _ = try await adapter.prefillResult(
            mode: .fullDuplex,
            input: ["text": .string("hello")])
        XCTAssertNil(engine.lastInput?.voicePCM, "prepared default voice must not be injected twice")
    }

    func testDuplexDefaultsAndRecentSpeechDecisionMetadataReachRuntime() async throws {
        let engine = CapturingDuplexEngine()
        let runtime = MiniCPMDuplexRuntime(engineFactory: { engine })
        let adapter = MiniCPMDuplexRuntimeEngineAdapter(
            runtime: runtime, sessionID: "recent-speech")

        try await adapter.initialize(mode: .fullDuplex, parameters: [:])
        XCTAssertEqual(engine.preparedConfig?.forceListenCount, 0)
        XCTAssertEqual(engine.preparedConfig?.topK, 20)
        XCTAssertTrue(engine.preparedConfig?.greedyListenSpeakDecision == false)

        _ = try await adapter.prefillResult(mode: .fullDuplex, input: [
            "text": .string("hello"),
            "speech_active": .bool(true),
            "allow_sampled_listen_speak_decision": .bool(true),
            "continuation_tick": .bool(true),
        ])
        XCTAssertEqual(engine.lastInput?.metadata["speech_active"], "true")
        XCTAssertEqual(
            engine.lastInput?.metadata["allow_sampled_listen_speak_decision"],
            "true")
        XCTAssertEqual(engine.lastInput?.allowSampledListenSpeakDecision, true)
        XCTAssertEqual(engine.lastInput?.continuationOnly, true)
        XCTAssertEqual(engine.lastInput?.metadata["continuation_tick"], "true")
    }

    func testExplicitTTSReferenceOverridesDefaultAndKeepsLLMReferenceSeparate() async throws {
        let engine = CapturingDuplexEngine()
        let runtime = MiniCPMDuplexRuntime(engineFactory: { engine })
        let reference = [Float(0.1), 0.2]
        let defaultVoice = [Float(0.3), 0.4, 0.5]
        let explicitTTS = [Float(0.6), 0.7, 0.8, 0.9]
        let adapter = MiniCPMDuplexRuntimeEngineAdapter(
            runtime: runtime,
            sessionID: "voice-explicit-tts",
            defaultVoicePCM: defaultVoice,
            defaultVoiceSampleRate: 22_050)

        try await adapter.initialize(
            mode: .fullDuplex,
            parameters: [
                "voice": .object([
                    "ref_audio": .array(reference.map { .number(Double($0)) }),
                    "sample_rate": .number(16_000),
                    "tts_ref_audio": .array(explicitTTS.map { .number(Double($0)) }),
                    "tts_sample_rate": .number(24_000),
                ])
            ])

        XCTAssertEqual(engine.preparedConfig?.llmReferenceAudio, reference)
        XCTAssertEqual(engine.preparedConfig?.llmReferenceAudioSampleRate, 16_000)
        XCTAssertEqual(engine.preparedConfig?.ttsReferenceAudio, explicitTTS)
        XCTAssertEqual(engine.preparedConfig?.ttsReferenceAudioSampleRate, 24_000)
    }

    func testRefAudioFallsBackToTTSReferenceWhenDedicatedChannelMissing() async throws {
        let engine = CapturingDuplexEngine()
        let runtime = MiniCPMDuplexRuntime(engineFactory: { engine })
        let adapter = MiniCPMDuplexRuntimeEngineAdapter(runtime: runtime, sessionID: "voice-fallback")
        let reference = [Float(0.1), 0.2]
        try await adapter.initialize(
            mode: .fullDuplex,
            parameters: ["voice": .object([
                "ref_audio": .array(reference.map { .number(Double($0)) }),
                "sample_rate": .number(16_000),
            ])])
        XCTAssertEqual(engine.preparedConfig?.llmReferenceAudio, reference)
        XCTAssertEqual(engine.preparedConfig?.ttsReferenceAudio, reference)
    }

    func testChatModesRequireAndUseDedicatedChatEngine() async throws {
        let duplex = CapturingDuplexEngine()
        let runtime = MiniCPMDuplexRuntime(engineFactory: { duplex })
        let chat = RecordingChatEngine()
        let adapter = MiniCPMDuplexRuntimeEngineAdapter(
            runtime: runtime,
            sessionID: "chat-router",
            chatEngine: chat)
        try await adapter.initialize(mode: .turnBased, parameters: [:])
        _ = try await adapter.prefillResult(mode: .turnBased, input: [
            "messages": .array([.object([
                "role": .string("user"),
                "content": .string("hello"),
            ])])
        ])
        _ = try await adapter.generate(mode: .turnBased, input: [:])
        let calls = await chat.calls
        XCTAssertEqual(calls, ["initialize", "prefill", "generate"])
        XCTAssertNil(duplex.preparedConfig, "chat mode must not prepare the duplex runtime")
    }

    func testInferenceSchedulerIsFIFOAndReleasesQueuedSession() async throws {
        let scheduler = MiniCPMDemoInferenceScheduler(maxConcurrent: 1, serviceEstimateMS: 100)
        guard case .granted(let first, _) = try await scheduler.enqueue(sessionID: "first") else {
            return XCTFail("first lease should be granted")
        }
        guard case .queued(let ticket, let queuedStatus) = try await scheduler.enqueue(sessionID: "second") else {
            return XCTFail("second session should queue")
        }
        XCTAssertEqual(queuedStatus.position, 1)
        let waiter = Task { try await scheduler.wait(ticket) }
        await scheduler.release(first)
        let second = try await waiter.value
        XCTAssertEqual(second.sessionID, "second")
        await scheduler.release(second)
        let snapshot = await scheduler.snapshot()
        XCTAssertEqual(snapshot.active, 0)
        XCTAssertEqual(snapshot.queued, 0)
    }

    func testRegistryCloseReleasesConnectionLeaseByMappedSchedulerID() async throws {
        let scheduler = MiniCPMDemoInferenceScheduler(maxConcurrent: 1)
        guard case .granted = try await scheduler.enqueue(sessionID: "conn_lease") else {
            return XCTFail("connection lease should be granted")
        }
        let registry = MiniCPMDemoBackendRegistry(scheduler: scheduler)
        await registry.register(
            sessionID: "sess_backend",
            backend: MiniCPMDemoBackend(),
            schedulerSessionID: "conn_lease")

        let closed = await registry.close(sessionID: "sess_backend", reason: "test")
        XCTAssertTrue(closed)
        let snapshot = await scheduler.snapshot()
        XCTAssertEqual(snapshot.active, 0)
        XCTAssertEqual(snapshot.queued, 0)
    }

    func testJSONLRecorderCapturesLifecycleEvents() async throws {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("minicpm-demo-recorder-\(UUID().uuidString).jsonl")
        let recorder = try MiniCPMDemoJSONLRecorder(url: url)
        let backend = MiniCPMDemoBackend(recorder: recorder)
        _ = await backend.handle(MiniCPMDemoRequest(type: "session.init", payload: .object(["mode": .string("turn_based")])))
        _ = await backend.handle(MiniCPMDemoRequest(type: "session.close"))
        recorder.finish()
        let lines = try String(contentsOf: url, encoding: .utf8)
            .split(separator: "\n")
        XCTAssertTrue(lines.contains { $0.contains("session.created") })
        XCTAssertTrue(lines.contains { $0.contains("session.closed") })
        try? FileManager.default.removeItem(at: url)
    }

    #if canImport(AVFoundation)
    private static let mp4FixtureBase64 = """
    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
    """.replacingOccurrences(of: "\\n", with: "")
    #endif

}

private final class CapturingDuplexEngine: MiniCPMDuplexEngine, @unchecked Sendable {
    var preparedConfig: MiniCPMDuplexConfig?
    var lastInput: MiniCPMDuplexInput?

    func prepare(mode: MiniCPMDuplexMode, config: MiniCPMDuplexConfig) throws -> String {
        preparedConfig = config
        return "prompt"
    }

    func prefill(mode: MiniCPMDuplexMode, input: MiniCPMDuplexInput) throws -> MiniCPMDuplexPrefillResult {
        lastInput = input
        return MiniCPMDuplexPrefillResult(mode: mode, input: input)
    }

    func generate(mode: MiniCPMDuplexMode, config: MiniCPMDuplexConfig) throws -> MiniCPMDuplexOutput {
        MiniCPMDuplexOutput(text: "ok", endOfTurn: true, finishReason: "turn_end")
    }

    func finalize() throws {}
    func rollback() throws {}
    func interrupt() {}
    func reset() {}
}

/// Native-runtime-shaped engine used to prove that adapter.stop/close signals
/// the nonisolated runtime epoch before awaiting actor rollback. Its first
/// synchronous generation remains blocked until the probe flips.
private final class AdapterProbeEngine: MiniCPMDuplexEngine, @unchecked Sendable {
    private var probe: (@Sendable () -> Bool)?
    private var blockNextGeneration = true
    let generationStarted = DispatchSemaphore(value: 0)

    func prepare(mode: MiniCPMDuplexMode, config: MiniCPMDuplexConfig) throws -> String { "prompt" }

    func prefill(mode: MiniCPMDuplexMode, input: MiniCPMDuplexInput) throws -> MiniCPMDuplexPrefillResult {
        MiniCPMDuplexPrefillResult(mode: mode, input: input)
    }

    func generate(mode: MiniCPMDuplexMode, config: MiniCPMDuplexConfig) throws -> MiniCPMDuplexOutput {
        if blockNextGeneration {
            blockNextGeneration = false
            generationStarted.signal()
            while !(probe?() ?? false) {
                Thread.sleep(forTimeInterval: 0.001)
            }
            return MiniCPMDuplexOutput(isListen: true, needsFinalize: false, finishReason: "interrupt")
        }
        return MiniCPMDuplexOutput(audio: [0.1], endOfTurn: true, finishReason: "fresh")
    }

    func finalize() throws {}
    func rollback() throws {}
    func interrupt() {}
    func reset() {}
    func setInterruptionProbe(_ probe: (@Sendable () -> Bool)?) { self.probe = probe }
}

/// Native-runtime-shaped prefill probe.  The first prefill exits through the
/// interruption path only after the adapter's nonisolated cancel signal is
/// delivered; a later prefill must remain usable in the same session.
private final class AdapterPrefillProbeEngine: MiniCPMDuplexEngine, @unchecked Sendable {
    private var probe: (@Sendable () -> Bool)?
    private var blockNextPrefill = true
    let prefillStarted = DispatchSemaphore(value: 0)

    func prepare(mode: MiniCPMDuplexMode, config: MiniCPMDuplexConfig) throws -> String { "prompt" }

    func prefill(mode: MiniCPMDuplexMode, input: MiniCPMDuplexInput) throws -> MiniCPMDuplexPrefillResult {
        if blockNextPrefill {
            blockNextPrefill = false
            prefillStarted.signal()
            while !(probe?() ?? false) {
                Thread.sleep(forTimeInterval: 0.001)
            }
            throw MiniCPMDuplexRuntimeError.invalidState("prefill interrupted")
        }
        return MiniCPMDuplexPrefillResult(mode: mode, input: input)
    }

    func generate(mode: MiniCPMDuplexMode, config: MiniCPMDuplexConfig) throws -> MiniCPMDuplexOutput {
        MiniCPMDuplexOutput(text: "fresh", endOfTurn: true, finishReason: "fresh")
    }

    func finalize() throws {}
    func rollback() throws {}
    func interrupt() {}
    func reset() {}
    func setSlidingWindow(tokens: Int?) {}
    func setInterruptionProbe(_ probe: (@Sendable () -> Bool)?) { self.probe = probe }
}

extension MiniCPMDemoBackendTests {
    func testBackendCloseSignalsRuntimeBeforeAwaitingRollbackAndDropsStaleOutput() async throws {
        let modelEngine = AdapterProbeEngine()
        let runtime = MiniCPMDuplexRuntime(engineFactory: { modelEngine })
        let adapter = MiniCPMDuplexRuntimeEngineAdapter(
            runtime: runtime, sessionID: "backend-close-signal")
        let backend = MiniCPMDemoBackend(engine: adapter)

        _ = await backend.handle(MiniCPMDemoRequest(
            type: "session.init",
            payload: .object(["mode": .string("full_duplex")])))
        let generationTask = Task {
            await backend.handle(MiniCPMDemoRequest(
                type: "input.append",
                input: .object(["text": .string("blocked")])))
        }
        XCTAssertEqual(modelEngine.generationStarted.wait(timeout: .now() + 2), .success)

        // Backend.close invokes stop before close. stop must flip the runtime
        // epoch synchronously, otherwise this await would wait for the model
        // turn to finish and the close path could never release the session.
        let closeEvents = await backend.handle(MiniCPMDemoRequest(
            type: "session.close", reason: "test_close"))
        let generationEvents = await generationTask.value
        XCTAssertTrue(closeEvents.contains { $0.type == "session.closed" })
        XCTAssertFalse(generationEvents.contains {
            $0.type == "response.output.delta" && $0.kind == "audio"
        }, "close must not leak output from the interrupted generation")
    }

    func testBackendCancelDuringPrefillKeepsSessionActiveForNextInput() async throws {
        let modelEngine = AdapterPrefillProbeEngine()
        let runtime = MiniCPMDuplexRuntime(engineFactory: { modelEngine })
        let adapter = MiniCPMDuplexRuntimeEngineAdapter(
            runtime: runtime, sessionID: "backend-prefill-cancel")
        let backend = MiniCPMDemoBackend(engine: adapter)

        _ = await backend.handle(MiniCPMDemoRequest(
            type: "session.init",
            payload: .object(["mode": .string("full_duplex")])))
        let firstTask = Task {
            await backend.handle(MiniCPMDemoRequest(
                type: "input.append",
                input: .object(["text": .string("first")])))
        }
        XCTAssertEqual(modelEngine.prefillStarted.wait(timeout: .now() + 2), .success)

        let cancelEvents = await backend.handle(MiniCPMDemoRequest(
            type: "control",
            payload: .object(["type": .string("response.cancel")])))
        let firstEvents = await firstTask.value
        XCTAssertTrue(cancelEvents.contains {
            $0.type == "backend.state" && $0.reason == "interrupt"
        })
        XCTAssertFalse(firstEvents.contains { $0.type == "error" })

        let secondEvents = await backend.handle(MiniCPMDemoRequest(
            type: "input.append",
            input: .object(["text": .string("second")])))
        XCTAssertTrue(secondEvents.contains {
            $0.type == "response.output.delta" && $0.kind == "text" && $0.text == "fresh"
        })
        XCTAssertFalse(secondEvents.contains { $0.type == "error" })
        let health = await backend.health()
        XCTAssertEqual(health["worker_status"]?.stringValue, "active")
    }

    func testContinueResponseStreamsNextUnitFromSyntheticContinuationTick() async throws {
        let modelEngine = ScriptedDuplexEngine(script: [.spoken, .spoken, .listen])
        let runtime = MiniCPMDuplexRuntime(engineFactory: { modelEngine })
        let adapter = MiniCPMDuplexRuntimeEngineAdapter(
            runtime: runtime, sessionID: "backend-continue-response")
        let backend = MiniCPMDemoBackend(engine: adapter)

        _ = await backend.handle(MiniCPMDemoRequest(
            type: "session.init",
            payload: .object(["mode": .string("full_duplex")])))
        let firstEvents = await backend.handle(MiniCPMDemoRequest(
            type: "input.append",
            input: .object(["text": .string("hello")])))
        let firstAudio = firstEvents.first {
            $0.type == "response.output.delta" && $0.kind == "audio"
        }
        XCTAssertNotNil(firstAudio, "the first input must open a spoken response")
        try await backend.commitPendingResponse()

        // No new transport packet has arrived.  The driver must still produce
        // the next spoken unit from a synthetic zero-filled continuation tick.
        let continuedEvents = try await requireContinue(backend)
        let continuedAudio = continuedEvents.first {
            $0.type == "response.output.delta" && $0.kind == "audio"
        }
        XCTAssertNotNil(continuedAudio)
        XCTAssertEqual(continuedAudio?.responseID, firstAudio?.responseID,
                       "continuation units stay inside the same open response")
        XCTAssertEqual(continuedAudio?.inputID?.hasPrefix("cont_"), true)

        let prefillInputs = await modelEngine.prefillInputs
        let continuationPrefills = prefillInputs.filter(\.continuationOnly)
        XCTAssertEqual(continuationPrefills.count, 1)
        XCTAssertEqual(continuationPrefills.first?.audio?.count, 16_000)
        XCTAssertTrue(continuationPrefills.first?.audio?.allSatisfy({ $0 == 0 }) ?? false)

        try await backend.commitPendingResponse()
        let closingEvents = try await requireContinue(backend)
        XCTAssertTrue(closingEvents.contains {
            $0.type == "response.output.delta" && $0.kind == "listen"
        })
        try await backend.commitPendingResponse()
        let exhausted = await backend.continueResponse()
        XCTAssertNil(exhausted, "a closed turn must not be driven further")
    }

    func testContinueResponseStopsAtPlaybackCushionBound() async throws {
        // Each scripted unit emits two seconds of audio, so the un-played
        // client buffer crosses maxBufferedAudioSeconds after the second
        // synthetic unit and the driver must stop.
        let modelEngine = ScriptedDuplexEngine(
            script: Array(repeating: .spoken, count: 8), samplesPerUnit: 48_000)
        let runtime = MiniCPMDuplexRuntime(engineFactory: { modelEngine })
        let adapter = MiniCPMDuplexRuntimeEngineAdapter(
            runtime: runtime, sessionID: "backend-continue-cushion")
        let backend = MiniCPMDemoBackend(engine: adapter)

        _ = await backend.handle(MiniCPMDemoRequest(
            type: "session.init",
            payload: .object(["mode": .string("full_duplex")])))
        _ = await backend.handle(MiniCPMDemoRequest(
            type: "input.append",
            input: .object(["text": .string("hello")])))
        try await backend.commitPendingResponse()

        var generated = 0
        while let events = await backend.continueResponse() {
            XCTAssertFalse(events.contains { $0.type == "error" })
            try await backend.commitPendingResponse()
            generated += 1
        }
        XCTAssertEqual(
            generated, Int(MiniCPMDemoBackend.maxBufferedAudioSeconds / 2.0),
            "run-ahead must stop once the un-played client buffer is full")
        let exhausted = await backend.continueResponse()
        XCTAssertNil(exhausted)
    }

    func testContinueResponseStopsAtSyntheticStreakCapAndRealInputResumes() async throws {
        // Tiny audio units never fill the playback cushion, so the
        // consecutive-synthetic-unit backstop is what stops the driver.
        let modelEngine = ScriptedDuplexEngine(
            script: Array(repeating: .spoken, count: 32), samplesPerUnit: 2_400)
        let runtime = MiniCPMDuplexRuntime(engineFactory: { modelEngine })
        let adapter = MiniCPMDuplexRuntimeEngineAdapter(
            runtime: runtime, sessionID: "backend-continue-streak")
        let backend = MiniCPMDemoBackend(engine: adapter)

        _ = await backend.handle(MiniCPMDemoRequest(
            type: "session.init",
            payload: .object(["mode": .string("full_duplex")])))
        _ = await backend.handle(MiniCPMDemoRequest(
            type: "input.append",
            input: .object(["text": .string("hello")])))
        try await backend.commitPendingResponse()

        var generated = 0
        while let events = await backend.continueResponse() {
            XCTAssertFalse(events.contains { $0.type == "error" })
            try await backend.commitPendingResponse()
            generated += 1
        }
        XCTAssertEqual(
            generated, MiniCPMDemoBackend.maxConsecutiveSyntheticUnits,
            "the synthetic streak backstop must stop the driver")

        // A real client packet ends the streak, so the driver may run again.
        _ = await backend.handle(MiniCPMDemoRequest(
            type: "input.append",
            input: .object(["text": .string("again")])))
        try await backend.commitPendingResponse()
        let resumed = try await requireContinue(backend)
        XCTAssertFalse(resumed.contains { $0.type == "error" })
    }

    func testSilentDuplexPacketsArePromotedToContinuationTicks() async throws {
        let modelEngine = ScriptedDuplexEngine(script: [.spoken, .spoken])
        let runtime = MiniCPMDuplexRuntime(engineFactory: { modelEngine })
        let adapter = MiniCPMDuplexRuntimeEngineAdapter(
            runtime: runtime, sessionID: "backend-silent-promotion")
        let backend = MiniCPMDemoBackend(engine: adapter)

        _ = await backend.handle(MiniCPMDemoRequest(
            type: "session.init",
            payload: .object(["mode": .string("full_duplex")])))
        _ = await backend.handle(MiniCPMDemoRequest(
            type: "input.append",
            input: .object(["text": .string("hello")])))
        try await backend.commitPendingResponse()
        // A browser-style silent packet without any client-side flag.
        let zeros = [Float](repeating: 0, count: 16_000)
        _ = await backend.handle(MiniCPMDemoRequest(
            type: "input.append",
            input: .object([
                "audio": .string(Self.base64OfFloat32(zeros)),
                "format": .string("pcm_f32le"),
                "sample_rate": .number(16_000),
                "speech_active": .bool(false),
            ])))
        try await backend.commitPendingResponse()
        let prefillInputs = await modelEngine.prefillInputs
        XCTAssertGreaterThanOrEqual(prefillInputs.count, 2)
        let promoted = prefillInputs.last
        XCTAssertEqual(promoted?.continuationOnly, true,
                       "all-zero duplex audio must be promoted server-side")
        // Non-zero audio must keep the regular encode path.
        let loud = [Float](repeating: 0.25, count: 16_000)
        _ = await backend.handle(MiniCPMDemoRequest(
            type: "input.append",
            input: .object([
                "audio": .string(Self.base64OfFloat32(loud)),
                "format": .string("pcm_f32le"),
                "sample_rate": .number(16_000),
                "speech_active": .bool(true),
            ])))
        try await backend.commitPendingResponse()
        let after = await modelEngine.prefillInputs
        XCTAssertEqual(after.last?.continuationOnly, false,
                       "non-zero audio must not be promoted")
    }

    func testContinueResponseIsRefusedWithoutAnOpenResponse() async throws {
        let modelEngine = ScriptedDuplexEngine(script: [.listen])
        let runtime = MiniCPMDuplexRuntime(engineFactory: { modelEngine })
        let adapter = MiniCPMDuplexRuntimeEngineAdapter(
            runtime: runtime, sessionID: "backend-continue-refuses")
        let backend = MiniCPMDemoBackend(engine: adapter)

        _ = await backend.handle(MiniCPMDemoRequest(
            type: "session.init",
            payload: .object(["mode": .string("full_duplex")])))
        let listenOnly = await backend.continueResponse()
        XCTAssertNil(listenOnly, "no response has been opened yet")

        _ = await backend.handle(MiniCPMDemoRequest(
            type: "input.append",
            input: .object(["text": .string("hello")])))
        let afterListen = await backend.continueResponse()
        XCTAssertNil(afterListen, "a listen marker closes the response and the driver")
    }

    private func requireContinue(
        _ backend: MiniCPMDemoBackend
    ) async throws -> [MiniCPMDemoEvent] {
        guard let events = await backend.continueResponse() else {
            XCTFail("expected the driver to continue the open response")
            return []
        }
        return events
    }
}

private extension MiniCPMDemoBackendTests {
    static func base64OfFloat32(_ samples: [Float]) -> String {
        var data = Data(capacity: samples.count * 4)
        for sample in samples {
            var bits = sample.bitPattern.littleEndian
            withUnsafeBytes(of: &bits) { data.append(contentsOf: $0) }
        }
        return data.base64EncodedString()
    }
}

/// Scripted duplex engine unit kinds for the continuation-driver tests.
private enum ScriptedDuplexUnit { case spoken, listen }

/// Scripted duplex engine: prefill always accepts and records its inputs;
/// generate walks a fixed script of spoken/listen units.
private final class ScriptedDuplexEngine: MiniCPMDuplexEngine, @unchecked Sendable {
    private let lock = NSLock()
    private var script: [ScriptedDuplexUnit]
    private let samplesPerUnit: Int
    private(set) var prefillInputs: [MiniCPMDuplexInput] = []

    init(script: [ScriptedDuplexUnit], samplesPerUnit: Int = 2_400) {
        self.script = script
        self.samplesPerUnit = samplesPerUnit
    }

    func prepare(mode: MiniCPMDuplexMode, config: MiniCPMDuplexConfig) throws -> String { "prompt" }

    func prefill(mode: MiniCPMDuplexMode, input: MiniCPMDuplexInput) throws -> MiniCPMDuplexPrefillResult {
        lock.lock()
        prefillInputs.append(input)
        lock.unlock()
        return MiniCPMDuplexPrefillResult(mode: mode, input: input)
    }

    func generate(mode: MiniCPMDuplexMode, config: MiniCPMDuplexConfig) throws -> MiniCPMDuplexOutput {
        lock.lock()
        let unit = script.isEmpty ? ScriptedDuplexUnit.listen : script.removeFirst()
        lock.unlock()
        switch unit {
        case .spoken:
            return MiniCPMDuplexOutput(
                text: "片段",
                audio: [Float](repeating: 0.1, count: samplesPerUnit),
                isListen: false,
                endOfTurn: false,
                needsFinalize: true,
                finishReason: "chunk")
        case .listen:
            return MiniCPMDuplexOutput(
                isListen: true,
                endOfTurn: true,
                needsFinalize: true,
                finishReason: "listen")
        }
    }

    func finalize() throws {}
    func rollback() throws {}
    func interrupt() {}
    func reset() {}
}

private final class TraceEngine: MiniCPMDuplexEngine, @unchecked Sendable {
    private var epoch: UInt64 = 0

    func prepare(mode: MiniCPMDuplexMode, config: MiniCPMDuplexConfig) throws -> String {
        "trace-prompt"
    }

    func prefill(mode: MiniCPMDuplexMode, input: MiniCPMDuplexInput) throws -> MiniCPMDuplexPrefillResult {
        MiniCPMDuplexPrefillResult(mode: mode, input: input)
    }

    func generate(mode: MiniCPMDuplexMode, config: MiniCPMDuplexConfig) throws -> MiniCPMDuplexOutput {
        MiniCPMDuplexOutput(isListen: true, finishReason: "listen")
    }

    func finalize() throws {}
    func rollback() throws {}
    func interrupt() {}
    func reset() {}
    func setGenerationEpoch(_ epoch: UInt64) { self.epoch = epoch }

    func diagnosticSnapshot() -> MiniCPMDuplexEngineDiagnosticSnapshot? {
        MiniCPMDuplexEngineDiagnosticSnapshot(
            generationEpoch: epoch,
            unitIndex: 7,
            currentTurnEnded: true,
            committedUnitCount: 3,
            kvCacheLength: 128,
            slidingWindowMode: "basic",
            slidingWindowHighWatermark: 256,
            slidingWindowLowWatermark: 192,
            pendingTokenCount: 2,
            pendingTerminator: 99,
            prefetched: false,
            needsFinalize: false,
            ttsContextPresent: true,
            ttsPendingToken: 42,
            ttsCommittedTokenCount: 5,
            ttsCacheLength: 11,
            token2WavPrepared: true,
            token2WavPendingTokenCount: 4,
            token2WavDecoderPositions: 8,
            continuationOnly: true,
            audioEncoderMS: 101,
            audioBridgeMS: 12,
            llmPrefillMS: 203,
            llmDecodeMS: 304,
            semanticTTSMS: 405,
            token2WavMS: 506,
            flowMS: 307,
            hiftMS: 199)
    }
}

private actor RecordingEngine: MiniCPMDemoBackendEngine {
    var calls: [String] = []
    var interruptCount = 0
    var finalizeCount = 0

    func initialize(mode: MiniCPMBackendMode, parameters: [String: MiniCPMJSONValue]) async throws {
        calls.append("initialize")
    }

    func prefill(mode: MiniCPMBackendMode, input: [String: MiniCPMJSONValue]) async throws {
        calls.append("prefill")
    }

    func generate(mode: MiniCPMBackendMode, input: [String: MiniCPMJSONValue]) async throws -> [MiniCPMDemoOutput] {
        calls.append("generate")
        return [MiniCPMDemoOutput(kind: "text", text: "ok", done: true)]
    }

    func interrupt() async {
        interruptCount += 1
    }

    func finalize() async throws {
        finalizeCount += 1
    }
}

private actor ResetRecordingEngine: MiniCPMDemoBackendEngine {
    var initializeParameters: [[String: MiniCPMJSONValue]] = []
    var resetCount = 0
    var generateCount = 0

    func initialize(mode: MiniCPMBackendMode, parameters: [String: MiniCPMJSONValue]) async throws {
        initializeParameters.append(parameters)
    }

    func prefill(mode: MiniCPMBackendMode, input: [String: MiniCPMJSONValue]) async throws {}

    func generate(mode: MiniCPMBackendMode, input: [String: MiniCPMJSONValue]) async throws -> [MiniCPMDemoOutput] {
        generateCount += 1
        return [MiniCPMDemoOutput(kind: "text", text: "ok", done: true, reason: "turn_end")]
    }

    func reset() async {
        resetCount += 1
    }
}

private actor BlockedTransportEngine: MiniCPMDemoBackendEngine {
    private var started = false
    private var startedWaiter: CheckedContinuation<Void, Never>?
    private var gate: CheckedContinuation<Void, Never>?
    private(set) var interruptCount = 0
    private var generateCount = 0

    func initialize(mode: MiniCPMBackendMode, parameters: [String: MiniCPMJSONValue]) async throws {}
    func prefill(mode: MiniCPMBackendMode, input: [String: MiniCPMJSONValue]) async throws {}

    func waitUntilStarted() async {
        if started { return }
        await withCheckedContinuation { continuation in
            startedWaiter = continuation
        }
    }

    func generate(mode: MiniCPMBackendMode, input: [String: MiniCPMJSONValue]) async throws -> [MiniCPMDemoOutput] {
        generateCount += 1
        if generateCount == 1 {
            started = true
            startedWaiter?.resume()
            startedWaiter = nil
            await withCheckedContinuation { continuation in
                gate = continuation
            }
            return [MiniCPMDemoOutput(kind: "audio", audio: Data("old-audio".utf8).base64EncodedString(), done: false)]
        }
        return [MiniCPMDemoOutput(kind: "audio", audio: Data("new-audio".utf8).base64EncodedString(), done: false)]
    }

    func interrupt() async {
        interruptCount += 1
        gate?.resume()
        gate = nil
    }
}

private actor RecordingChatEngine: MiniCPMDemoChatEngine {
    var calls: [String] = []

    func initialize(mode: MiniCPMBackendMode, parameters: [String: MiniCPMJSONValue]) async throws {
        calls.append("initialize")
    }

    func prefill(mode: MiniCPMBackendMode, input: [String: MiniCPMJSONValue]) async throws {
        calls.append("prefill")
    }

    func generate(mode: MiniCPMBackendMode, input: [String: MiniCPMJSONValue]) async throws -> [MiniCPMDemoOutput] {
        calls.append("generate")
        return [MiniCPMDemoOutput(kind: "text", text: "chat", done: true)]
    }
}

private actor OpenStreamingEngine: MiniCPMDemoBackendEngine {
    var finalizeCount = 0

    func initialize(mode: MiniCPMBackendMode, parameters: [String: MiniCPMJSONValue]) async throws {}
    func prefill(mode: MiniCPMBackendMode, input: [String: MiniCPMJSONValue]) async throws {}
    func generate(mode: MiniCPMBackendMode, input: [String: MiniCPMJSONValue]) async throws -> [MiniCPMDemoOutput] {
        [MiniCPMDemoOutput(kind: "text", text: "好的", done: false, reason: "streaming")]
    }
    func finalize() async throws { finalizeCount += 1 }
}

private actor PerUnitMetricsEngine: MiniCPMDemoBackendEngine {
    private var generation = 0

    func initialize(mode: MiniCPMBackendMode, parameters: [String: MiniCPMJSONValue]) async throws {}
    func prefill(mode: MiniCPMBackendMode, input: [String: MiniCPMJSONValue]) async throws {}
    func generate(mode: MiniCPMBackendMode, input: [String: MiniCPMJSONValue]) async throws -> [MiniCPMDemoOutput] {
        generation += 1
        let metrics: [String: MiniCPMJSONValue] = generation == 1
            ? [
                "audio_encoder_ms": .number(123),
                "audio_padding_samples": .number(80),
                "continuation_only": .bool(false),
            ]
            : ["continuation_only": .bool(true)]
        return [MiniCPMDemoOutput(
            kind: "audio",
            audio: "AAAAAA==",
            done: false,
            needsFinalize: false,
            reason: "streaming",
            metrics: metrics)]
    }
}

private actor TwoChunkEngine: MiniCPMDemoBackendEngine {
    var prefillCount = 0
    var generateCount = 0

    func initialize(mode: MiniCPMBackendMode, parameters: [String: MiniCPMJSONValue]) async throws {}
    func prefill(mode: MiniCPMBackendMode, input: [String: MiniCPMJSONValue]) async throws {}
    func prefillResult(mode: MiniCPMBackendMode, input: [String: MiniCPMJSONValue]) async throws -> MiniCPMDemoPrefillResult {
        prefillCount += 1
        return MiniCPMDemoPrefillResult(accepted: prefillCount >= 2, needsMoreAudio: prefillCount < 2)
    }
    func generate(mode: MiniCPMBackendMode, input: [String: MiniCPMJSONValue]) async throws -> [MiniCPMDemoOutput] {
        generateCount += 1
        return [MiniCPMDemoOutput(kind: "text", text: "done", done: true, reason: "turn_end")]
    }
}
