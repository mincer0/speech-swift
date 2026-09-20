import Foundation
import XCTest
import MLX
import MLXNN

@testable import MiniCPMTTSSemantic

final class MiniCPMTTSSemanticTests: XCTestCase {
    /// ``shouldCancel`` is @Sendable because the native duplex engine may
    /// probe it across an async boundary.  The tiny-model tests run
    /// synchronously, but using the same locked probe catches accidental
    /// non-Sendable test closures and makes the cancellation point explicit.
    private final class CancellationProbe: @unchecked Sendable {
        private let lock = NSLock()
        private var count = 0
        private let cancelAt: Int

        init(cancelAt: Int) { self.cancelAt = cancelAt }

        func shouldCancel() -> Bool {
            lock.lock()
            defer { lock.unlock() }
            count += 1
            return count >= cancelAt
        }
    }

    private func tinyModel() -> MiniCPMTTSSemantic {
        MiniCPMTTSSemantic(configuration: MiniCPMTTSSemanticConfiguration(
            llmDim: 4,
            hiddenSize: 8,
            intermediateSize: 16,
            numHiddenLayers: 1,
            numAttentionHeads: 2,
            numKeyValueHeads: 2,
            numAudioTokens: 7,
            numTextTokens: 10,
            maxPositionEmbeddings: 64,
            audioBOSTokenID: 9,
            textEOSTokenID: 8
        ))
    }

    /// Return a parity resource only when the opt-in parity gate is enabled.
    /// Once either the gate or a resource path is supplied, configuration is
    /// explicit: a missing counterpart or a missing path is a test failure,
    /// never a silent fallback to a developer-local fixture.
    private func parityResource(_ key: String, label: String) throws -> String? {
        let environment = ProcessInfo.processInfo.environment
        let parityEnabled = environment["MINICPM_SWIFT_BUNDLE_PARITY"] == "1"
        let configured = environment[key] != nil
        guard parityEnabled || configured else {
            return nil
        }
        guard parityEnabled else {
            XCTFail("\(key) is set but MINICPM_SWIFT_BUNDLE_PARITY is not 1")
            throw MiniCPMTTSSemanticError.invalidInput(
                "\(key) requires MINICPM_SWIFT_BUNDLE_PARITY=1")
        }
        guard let path = environment[key], !path.isEmpty else {
            XCTFail("MINICPM_SWIFT_BUNDLE_PARITY=1 requires \(key)")
            throw MiniCPMTTSSemanticError.invalidInput(
                "missing explicit \(label) path")
        }
        guard FileManager.default.fileExists(atPath: path) else {
            XCTFail("configured \(label) path does not exist: \(path)")
            throw MiniCPMTTSSemanticError.invalidInput(
                "missing explicit \(label) path")
        }
        return path
    }

    func testBundledOracleMetadataUsesStableSourceLabels() throws {
        let fixture = try XCTUnwrap(
            Bundle.module.resourceURL?.appendingPathComponent(
                "Fixtures/minicpm-tts-condition-oracle.json"))
        let object = try XCTUnwrap(
            try JSONSerialization.jsonObject(with: Data(contentsOf: fixture))
                as? [String: Any])
        let source = try XCTUnwrap(object["source"] as? [String: Any])

        XCTAssertEqual(source["weights_root"] as? String, "<models>/MiniCPM-o-4_5")
        XCTAssertEqual(source["upstream_repo"] as? String, "OpenBMB/MiniCPM-o-Demo")
        for (key, value) in source {
            guard let string = value as? String else { continue }
            XCTAssertFalse(
                string.hasPrefix("/"),
                "source metadata \(key) must not contain a machine-local absolute path")
        }
    }

    func testConditionAddsAudioBOSAndNormalizesProjectedHidden() throws {
        let model = tinyModel()
        let hidden = MLXArray.zeros([2, 4])
        let condition = try model.buildCondition(
            hidden: hidden,
            tokenIDs: MLXArray([Int32(1), Int32(2)]))
        XCTAssertEqual(condition.shape, [1, 3, 8])
        eval(condition)
        XCTAssertEqual(condition.dtype, model.textEmbedding.weight.dtype)
    }

    func testConditionExplicitTextEOSAndAudioBOSOrdering() throws {
        let model = tinyModel()
        let eosOnly = try model.buildCondition(
            hidden: MLXArray.zeros([0, 4]),
            tokenIDs: MLXArray.zeros([0], dtype: .int32),
            addAudioBOS: true,
            addTextEOS: true)
        XCTAssertEqual(eosOnly.shape, [1, 2, 8])

        let expectedEOS = model.textEmbedding(
            MLXArray([Int32(model.configuration.textEOSTokenID)]))
            .expandedDimensions(axis: 0)
        let expectedBOS = model.textEmbedding(
            MLXArray([Int32(model.configuration.audioBOSTokenID)]))
            .expandedDimensions(axis: 0)
        eval(eosOnly, expectedEOS, expectedBOS)
        XCTAssertEqual(
            eosOnly[0, 0].asArray(Float.self),
            expectedEOS[0, 0].asArray(Float.self))
        XCTAssertEqual(
            eosOnly[0, 1].asArray(Float.self),
            expectedBOS[0, 0].asArray(Float.self))
    }

    func testConditionRejectsHiddenTokenMisalignment() throws {
        let model = tinyModel()
        XCTAssertThrowsError(try model.buildCondition(
            hidden: MLXArray.zeros([2, 4]),
            tokenIDs: MLXArray([Int32(1)]))) { error in
            guard case MiniCPMTTSSemanticError.invalidInput = error else {
                return XCTFail("expected hidden/token length validation, got \(error)")
            }
        }
    }

    func testSessionKeepsNewestCodePendingAndCommitsEarlierCodes() throws {
        let model = tinyModel()
        let session = model.makeSession()
        let embeddings = MLXArray.zeros([1, 2, 8])
        let result = try model.generateChunk(
            inputsEmbeds: embeddings,
            session: session,
            temperature: 0,
            forceNoStop: true,
            maxNewToken: 3,
            minNewTokens: 3)

        XCTAssertEqual(result.sampledTokenCount, 3)
        XCTAssertEqual(result.committedTokens.count, 2)
        XCTAssertNotNil(result.pendingToken)
        XCTAssertEqual(session.committedTokens, result.committedTokens)
        XCTAssertEqual(session.cacheOffset, 4) // 2 condition + 2 fed codes

        let next = try model.generateChunk(
            inputsEmbeds: MLXArray.zeros([1, 1, 8]),
            session: session,
            temperature: 0,
            forceNoStop: true,
            maxNewToken: 1)
        XCTAssertEqual(next.committedTokens, [])
        XCTAssertEqual(next.sampledTokenCount, 1)
        XCTAssertEqual(session.cacheOffset, 6) // pending code + new condition
    }

    func testCancellationRollsBackTinySessionAndAllowsResume() throws {
        let model = tinyModel()
        let session = model.makeSession()
        // Seed a look-ahead token so rollback must restore both KV and the
        // pending code that would otherwise be consumed at the next call.
        let seeded = try model.generateChunk(
            inputsEmbeds: MLXArray.zeros([1, 1, 8]),
            session: session,
            temperature: 0,
            forceNoStop: true,
            maxNewToken: 1)
        XCTAssertEqual(seeded.sampledTokenCount, 1)
        XCTAssertNotNil(session.pendingToken)
        let checkpoint = session.snapshot()

        // Calls are: entry guard, token-loop guard, post-sample guard.  The
        // third probe fires after decoder KV has been touched but before the
        // generated token is committed to session state.
        let probe = CancellationProbe(cancelAt: 3)
        XCTAssertThrowsError(try model.generateChunk(
            inputsEmbeds: MLXArray.zeros([1, 1, 8]),
            session: session,
            temperature: 0,
            forceNoStop: true,
            maxNewToken: 2,
            shouldCancel: { probe.shouldCancel() })) { error in
            guard case MiniCPMTTSSemanticError.cancelled = error else {
                return XCTFail("expected semantic cancellation, got \(error)")
            }
        }

        // Cancellation itself is an abort signal; the owning transaction
        // restores the immutable checkpoint before accepting another unit.
        try session.rollback(to: checkpoint)
        XCTAssertEqual(session.cacheOffset, checkpoint.cacheOffset)
        XCTAssertEqual(session.pendingToken, checkpoint.pendingToken)
        XCTAssertEqual(session.committedTokens, checkpoint.committedTokens)
        XCTAssertEqual(session.lastCommittedTokens, checkpoint.lastCommittedTokens)
        XCTAssertEqual(session.textStartPosition, checkpoint.textStartPosition)
        XCTAssertEqual(session.chunkIndex, checkpoint.chunkIndex)
        XCTAssertEqual(session.turnActive, checkpoint.turnActive)

        let resumed = try model.generateChunk(
            inputsEmbeds: MLXArray.zeros([1, 1, 8]),
            session: session,
            temperature: 0,
            forceNoStop: true,
            maxNewToken: 1)
        XCTAssertEqual(resumed.sampledTokenCount, 1)
        XCTAssertNotNil(resumed.pendingToken)
        // A one-token generation remains look-ahead only.
        XCTAssertEqual(session.committedTokens, checkpoint.committedTokens)
    }

    func testFullSequenceFacadeReturnsPendingCodeAndResets() throws {
        let model = tinyModel()
        let result = try model.generate(
            inputsEmbeds: MLXArray.zeros([1, 2, 8]),
            temperature: 0,
            forceNoStop: true,
            maxNewToken: 3,
            minNewTokens: 3)

        XCTAssertEqual(result.sampledTokenCount, 3)
        XCTAssertEqual(result.tokens.count, 3)
        XCTAssertFalse(result.eosEmitted)
        XCTAssertGreaterThan(result.cacheOffset, 0)
        XCTAssertTrue(result.sessionWasReset)
    }

    func testInterleavedFacadeCarriesPendingCodeBetweenConditions() throws {
        let model = tinyModel()
        let session = model.makeSession()
        let result = try model.interleavedGenerate(
            speakerEmbeds: MLXArray.zeros([1, 1, 8]),
            conditions: [
                MLXArray.zeros([1, 2, 8]),
                MLXArray.zeros([1, 1, 8]),
            ],
            session: session,
            temperature: 0,
            forceNoStop: true,
            maxNewToken: 3,
            minNewTokens: 3,
            // Retain the supplied session so this test can inspect the
            // pending look-ahead code carried into the next condition.
            resetAfterGeneration: false)

        XCTAssertEqual(result.chunkTokens.count, 2)
        XCTAssertEqual(result.chunkTokens.map(\.count), [3, 3])
        XCTAssertEqual(result.tokens.count, 6)
        XCTAssertEqual(result.sampledTokenCount, 6)
        XCTAssertFalse(result.eosEmitted)
        XCTAssertFalse(result.sessionWasReset)
        XCTAssertNotNil(session.pendingToken)
        XCTAssertEqual(session.chunkIndex, 2)
        session.finishTurn()
    }

    func testSessionsDoNotShareKVOrPendingState() throws {
        let model = tinyModel()
        let first = model.makeSession()
        let second = model.makeSession()
        let embeddings = MLXArray.zeros([1, 1, 8])
        _ = try model.generateChunk(
            inputsEmbeds: embeddings,
            session: first,
            temperature: 0,
            forceNoStop: true,
            maxNewToken: 1)
        XCTAssertEqual(second.cacheOffset, 0)
        XCTAssertNil(second.pendingToken)
    }

    func testLayerCacheUsesGeometricCapacityAndLogicalOffset() throws {
        let cache = MiniCPMTTSLayerCache()
        for index in 0..<600 {
            let key = MLXArray.ones([1, 2, 1, 4]) * Float(index)
            _ = cache.update(keys: key, values: key)
            XCTAssertEqual(cache.offset, index + 1)
            XCTAssertEqual(cache.keys?.dim(2), index + 1)
            XCTAssertEqual(cache.values?.dim(2), index + 1)
        }

        XCTAssertGreaterThanOrEqual(cache.capacity, 600)
        // 32 -> 64 -> ... -> 1024: no per-token reallocation.
        XCTAssertLessThanOrEqual(cache.capacityGrowthCount, 6)
        XCTAssertGreaterThan(cache.capacity, cache.offset)

        eval(cache.keys!, cache.values!)
        XCTAssertEqual(cache.keys![0, 0, 0, 0].item(Float.self), 0)
        XCTAssertEqual(cache.values![0, 0, 599, 0].item(Float.self), 599)
    }

    func testLongStreamingStoryKeepsCacheGrowthLogarithmic() throws {
        let model = tinyModel()
        let session = model.makeSession()
        for _ in 0..<32 {
            let result = try model.generateStreamingChunk(
                inputsEmbeds: MLXArray.zeros([1, 1, 8]),
                session: session,
                temperature: 0,
                forceNoStop: true,
                maxNewToken: 26)
            XCTAssertEqual(result.sampledTokenCount, 26, "chunk (index)")
            XCTAssertEqual(result.committedTokenCount, 25, "chunk (index)")
            XCTAssertNil(result.pendingToken, "protocol chunk hides look-ahead")
            XCTAssertFalse(result.endedByEOS)
        }

        let cache = try XCTUnwrap(session.layerCaches.first)
        XCTAssertGreaterThan(session.cacheOffset, 800)
        XCTAssertLessThanOrEqual(cache.capacityGrowthCount, 6)
        XCTAssertGreaterThanOrEqual(cache.capacity, session.cacheOffset)

        session.finishTurn()
        XCTAssertEqual(session.cacheOffset, 0)
        XCTAssertEqual(cache.capacity, 0)
    }

    func testStreamingFirstMiddleFinalOracleAndRollback() throws {
        let model = tinyModel()
        let session = model.makeSession()

        let first = try model.generateStreamingChunk(
            inputsEmbeds: MLXArray.zeros([1, 2, 8]),
            session: session,
            temperature: 0,
            forceNoStop: true,
            maxNewToken: 26)
        XCTAssertTrue(first.isFirstChunk)
        XCTAssertFalse(first.isFinalChunk)
        XCTAssertEqual(first.sampledTokenCount, 26)
        XCTAssertEqual(first.committedTokenCount, 25)
        XCTAssertNil(first.pendingToken)
        XCTAssertNotNil(session.pendingToken, "look-ahead stays in the semantic session")
        XCTAssertEqual(first.textStartPosition, 0)
        XCTAssertEqual(first.nextTextStartPosition, 27) // 2 + 25 returned
        XCTAssertEqual(session.cacheOffset, 27)
        XCTAssertEqual(session.textStartPosition, 27)
        XCTAssertEqual(session.chunkIndex, 1)
        XCTAssertTrue(session.turnActive)

        let checkpoint = session.snapshot()
        let checkpointCapacity = try XCTUnwrap(session.layerCaches.first).capacity
        let middle = try model.generateStreamingChunk(
            inputsEmbeds: MLXArray.zeros([1, 3, 8]),
            session: session,
            temperature: 0,
            forceNoStop: true,
            maxNewToken: 26)
        XCTAssertFalse(middle.isFirstChunk)
        XCTAssertFalse(middle.isFinalChunk)
        XCTAssertEqual(middle.sampledTokenCount, 26)
        XCTAssertEqual(middle.committedTokenCount, 25)
        XCTAssertNil(middle.pendingToken)
        XCTAssertNotNil(session.pendingToken, "look-ahead stays in the semantic session")
        XCTAssertEqual(middle.textStartPosition, 27)
        XCTAssertEqual(middle.nextTextStartPosition, 55) // +3 +25 returned
        // The hidden look-ahead code is fed before this condition, but is not
        // included in text-position bookkeeping.
        XCTAssertEqual(session.cacheOffset, 56) // pending + condition + 25 fed

        try session.rollback(to: checkpoint)
        XCTAssertEqual(session.cacheOffset, checkpoint.cacheOffset)
        XCTAssertEqual(session.pendingToken, checkpoint.pendingToken)
        XCTAssertEqual(session.committedTokens, checkpoint.committedTokens)
        XCTAssertEqual(session.textStartPosition, checkpoint.textStartPosition)
        XCTAssertEqual(session.chunkIndex, checkpoint.chunkIndex)
        XCTAssertEqual(try XCTUnwrap(session.layerCaches.first).capacity, checkpointCapacity)
        XCTAssertTrue(session.turnActive)

        let replay = try model.generateStreamingChunk(
            inputsEmbeds: MLXArray.zeros([1, 3, 8]),
            session: session,
            temperature: 0,
            forceNoStop: true,
            maxNewToken: 26)
        XCTAssertEqual(replay.committedTokens, middle.committedTokens)
        XCTAssertEqual(replay.pendingToken, middle.pendingToken)
        XCTAssertEqual(replay.nextTextStartPosition, middle.nextTextStartPosition)

        let final = try model.generateStreamingChunk(
            inputsEmbeds: MLXArray.zeros([1, 1, 8]),
            session: session,
            endOfTurn: true,
            temperature: 0,
            forceNoStop: true,
            maxNewToken: 26)
        XCTAssertFalse(final.isFirstChunk)
        XCTAssertTrue(final.isFinalChunk)
        XCTAssertFalse(final.sessionWasReset)
        XCTAssertFalse(final.endedByEOS) // final flag is sufficient to flush
        XCTAssertEqual(final.sampledTokenCount, 26)
        XCTAssertEqual(final.committedTokenCount, 25)
        XCTAssertEqual(final.textStartPosition, 55)
        XCTAssertEqual(final.nextTextStartPosition, 81) // +1 +25 returned
        XCTAssertEqual(final.cacheOffset, 83) // pending + condition + 25 fed
        // The caller flushes token2wav before releasing semantic KV.
        XCTAssertEqual(session.cacheOffset, 83)
        XCTAssertNotNil(session.pendingToken)
        session.finishTurn()
        XCTAssertEqual(session.cacheOffset, 0)
        XCTAssertNil(session.pendingToken)
        XCTAssertEqual(session.textStartPosition, 0)
        XCTAssertEqual(session.chunkIndex, 0)
        XCTAssertFalse(session.turnActive)
    }

    func testRollbackRestoresCapacityAfterAnInterruptedGrowth() throws {
        let model = tinyModel()
        let session = model.makeSession()
        _ = try model.generateStreamingChunk(
            inputsEmbeds: MLXArray.zeros([1, 1, 8]),
            session: session,
            temperature: 0,
            forceNoStop: true,
            maxNewToken: 26)
        let checkpoint = session.snapshot()
        let cache = try XCTUnwrap(session.layerCaches.first)
        let capacityBefore = cache.capacity

        // Two chunks are enough to cross the initial 32-entry allocation and
        // exercise the copy-on-write path during geometric growth.
        _ = try model.generateStreamingChunk(
            inputsEmbeds: MLXArray.zeros([1, 1, 8]),
            session: session,
            temperature: 0,
            forceNoStop: true,
            maxNewToken: 26)
        _ = try model.generateStreamingChunk(
            inputsEmbeds: MLXArray.zeros([1, 1, 8]),
            session: session,
            temperature: 0,
            forceNoStop: true,
            maxNewToken: 26)
        XCTAssertGreaterThan(cache.capacity, capacityBefore)
        XCTAssertGreaterThan(session.cacheOffset, checkpoint.cacheOffset)

        try session.rollback(to: checkpoint)
        XCTAssertEqual(session.cacheOffset, checkpoint.cacheOffset)
        XCTAssertEqual(cache.capacity, capacityBefore)
        XCTAssertEqual(session.pendingToken, checkpoint.pendingToken)
        XCTAssertEqual(session.committedTokens, checkpoint.committedTokens)
        XCTAssertEqual(session.textStartPosition, checkpoint.textStartPosition)
        XCTAssertEqual(session.chunkIndex, checkpoint.chunkIndex)
    }

    func testStreamingEOSResetsSessionAfterCapturingResult() throws {
        let model = tinyModel()
        let condition = MLXArray.zeros([1, 2, 8])
        // Probe the deterministic first-step argmax and use it as EOS.  This
        // makes the test independent of random fixture weights while proving
        // EOS itself is never committed or left pending.
        let probe = model.makeSession()
        let hidden = model.decoder(condition, caches: probe.layerCaches, offset: 0)
        let logits = model.audioHead(hidden[0, hidden.dim(1) - 1]).asType(.float32)
        eval(logits)
        let eos = argMax(logits, axis: 0).item(Int.self)

        let session = model.makeSession()
        let result = try model.generateStreamingChunk(
            inputsEmbeds: condition,
            session: session,
            temperature: 0,
            repetitionPenalty: 1,
            eosToken: eos,
            maxNewToken: 26)
        XCTAssertTrue(result.endedByEOS)
        XCTAssertTrue(result.isFinalChunk)
        XCTAssertFalse(result.sessionWasReset)
        XCTAssertEqual(result.sampledTokenCount, 1)
        XCTAssertEqual(result.committedTokens, [])
        XCTAssertNil(result.pendingToken)
        XCTAssertGreaterThan(session.cacheOffset, 0)
        session.finishTurn()
        XCTAssertEqual(session.cacheOffset, 0)
        XCTAssertEqual(session.textStartPosition, 0)
        XCTAssertEqual(session.chunkIndex, 0)
        XCTAssertFalse(session.turnActive)
    }

    func testLongStoryFinalThenNextTurnStartsFreshWithoutReplay() throws {
        let model = tinyModel()
        let session = model.makeSession()
        let condition = MLXArray.zeros([1, 1, 8])

        // A story spans many semantic units.  None of the middle units may
        // silently end the TTS turn or alter the strict returned-token
        // cadence.
        for _ in 0 ..< 8 {
            let chunk = try model.generateStreamingChunk(
                inputsEmbeds: condition,
                session: session,
                temperature: 0,
                forceNoStop: true,
                maxNewToken: 26)
            XCTAssertFalse(chunk.isFinalChunk)
            XCTAssertEqual(chunk.committedTokenCount, 25)
            XCTAssertNil(chunk.pendingToken, "protocol chunk hides look-ahead")
            XCTAssertEqual(
                chunk.nextTextStartPosition,
                chunk.textStartPosition + 26,
                "condition length 1 + 25 returned codes")
        }
        XCTAssertTrue(session.turnActive)
        XCTAssertGreaterThan(session.cacheOffset, 0)

        // End-of-turn is observable before the Token2Wav flush.  The guard
        // below prevents a stale replay if a caller accidentally submits the
        // next unit before releasing the semantic cache.
        let final = try model.generateStreamingChunk(
            inputsEmbeds: condition,
            session: session,
            endOfTurn: true,
            temperature: 0,
            forceNoStop: true,
            maxNewToken: 26)
        XCTAssertTrue(final.isFinalChunk)
        XCTAssertFalse(session.turnActive)
        XCTAssertThrowsError(try model.generateStreamingChunk(
            inputsEmbeds: condition,
            session: session,
            temperature: 0,
            forceNoStop: true,
            maxNewToken: 26))

        // The next turn must explicitly finish/flush the old one first.  It
        // then starts at text position zero with no pending code or KV.
        session.finishTurn()
        let next = try model.generateStreamingChunk(
            inputsEmbeds: condition,
            session: session,
            temperature: 0,
            forceNoStop: true,
            maxNewToken: 26)
        XCTAssertTrue(next.isFirstChunk)
        XCTAssertEqual(next.textStartPosition, 0)
        XCTAssertEqual(session.textStartPosition, 26) // 1 condition + 25 returned
        XCTAssertEqual(session.chunkIndex, 1)
        XCTAssertTrue(session.turnActive)
        XCTAssertEqual(session.committedTokens.count, 25)
    }

    func testRealBundleLoadsAndMatchesGreedyOracle() throws {
        guard let path = try parityResource(
            "MINICPM_TTS_BUNDLE_PATH", label: "MiniCPM TTS bundle")
        else {
            throw XCTSkip("set MINICPM_SWIFT_BUNDLE_PARITY=1 and MINICPM_TTS_BUNDLE_PATH")
        }

        let model = try MiniCPMTTSSemantic.fromDirectory(URL(fileURLWithPath: path))
        let hidden = MLXArray.arange(0, 3 * 4096, dtype: .float32)
            .reshaped(3, 4096) / MLXArray(Float(1000))
        let condition = try model.buildCondition(
            hidden: hidden,
            tokenIDs: MLXArray([Int32(22), Int32(33), Int32(44)]))
        let result = try model.generateChunk(
            inputsEmbeds: condition,
            session: model.makeSession(),
            temperature: 0,
            forceNoStop: true,
            maxNewToken: 3,
            minNewTokens: 3)

        // Oracle generated by Python MLX from the original full MiniCPM
        // checkpoint.  This exercises strict key translation, BF16 loading,
        // RoPE, SDPA, KV state, and the weight-normalized audio head.
        XCTAssertEqual(result.committedTokens, [1736, 5379])
        XCTAssertEqual(result.pendingToken, 5645)
        XCTAssertEqual(result.cacheOffset, 6)
    }

    func testOfficialConditionOracleAndThreeChunkTrace() throws {
        guard let directory = try parityResource(
            "MINICPM_TTS_CONDITION_ORACLE_DIR", label: "MiniCPM TTS condition oracle")
        else {
            throw XCTSkip("set MINICPM_SWIFT_BUNDLE_PARITY=1 and MINICPM_TTS_CONDITION_ORACLE_DIR")
        }
        let root = URL(fileURLWithPath: directory)
        let fixtureURL = root.appendingPathComponent("minicpm-tts-condition-oracle.safetensors")
        let metadataURL = root.appendingPathComponent("minicpm-tts-condition-oracle.json")
        let arrays = try MLX.loadArrays(url: fixtureURL)
        let metadata = try XCTUnwrap(
            try JSONSerialization.jsonObject(with: Data(contentsOf: metadataURL)) as? [String: Any])
        let sourceMetadata = try XCTUnwrap(metadata["source"] as? [String: Any])
        XCTAssertEqual(
            sourceMetadata["oracle_backend"] as? String,
            "official_pytorch",
            "semantic oracle must come from the fixed official PyTorch TTS")
        XCTAssertEqual(
            sourceMetadata["upstream_commit"] as? String,
            "ba7fa9cc6ad63c894f1bd5e5afac28466953519d")
        XCTAssertEqual(
            sourceMetadata["modeling_sha256"] as? String,
            "877f20cea7282bb0d05684393a1580646d1f264a2299a42f82467fd99f4028af")
        XCTAssertEqual(
            sourceMetadata["unified_modeling_sha256"] as? String,
            "9fceb75f969cfafc6381adf2b59255529be7fb34d64ad784b8417d224ffbabb9")
        XCTAssertEqual(sourceMetadata["condition_verified_with_fixed_source"] as? Bool, true)
        let modelPath = try XCTUnwrap(try parityResource(
            "MINICPM_TTS_BUNDLE_PATH", label: "MiniCPM TTS bundle"))
        let model = try MiniCPMTTSSemantic.fromDirectory(URL(fileURLWithPath: modelPath))

        func assertClose(_ actual: MLXArray, _ expected: MLXArray, _ label: String) throws {
            XCTAssertEqual(actual.shape, expected.shape, "\(label) shape")
            let lhs = actual.asType(.float32)
            let rhs = expected.asType(.float32)
            let delta = MLX.maximum(lhs - rhs, rhs - lhs)
            let maxAbs = delta.max().item(Float.self)
            let meanAbs = delta.mean().item(Float.self)
            let dot = (lhs * rhs).sum().item(Float.self)
            let norm = MLX.sqrt((lhs * lhs).sum() * (rhs * rhs).sum()).item(Float.self)
            let cosine = norm > 0 ? dot / norm : 1
            if ProcessInfo.processInfo.environment["MINICPM_TTS_DIAGNOSTICS"] == "1" {
                print("TTS_CONDITION_TRACE \(label) max=\(maxAbs) mean=\(meanAbs) cosine=\(cosine)")
            }
            XCTAssertLessThanOrEqual(maxAbs, 0.03125, "\(label) maxAbs=\(maxAbs)")
            XCTAssertLessThanOrEqual(meanAbs, 0.001, "\(label) meanAbs=\(meanAbs)")
            XCTAssertGreaterThan(cosine, 0.99999, "\(label) cosine=\(cosine)")
        }

        func assertLogitsClose(_ actual: MLXArray, _ expected: MLXArray, _ label: String) throws {
            XCTAssertEqual(actual.shape, expected.shape, "\(label) shape")
            let lhs = actual.asType(.float32)
            let rhs = expected.asType(.float32)
            let delta = MLX.maximum(lhs - rhs, rhs - lhs)
            let maxAbs = delta.max().item(Float.self)
            let meanAbs = delta.mean().item(Float.self)
            let dot = (lhs * rhs).sum().item(Float.self)
            let norm = MLX.sqrt((lhs * lhs).sum() * (rhs * rhs).sum()).item(Float.self)
            let cosine = norm > 0 ? dot / norm : 1
            // Each BF16 matmul/attention path can differ by one BF16 ULP
            // after accumulation.  The measured first-step oracle is
            // max=0.0785, mean=0.0178; retain headroom for another legal ULP
            // without allowing a materially different decoder.
            XCTAssertLessThanOrEqual(maxAbs, 0.125, "\(label) maxAbs=\(maxAbs)")
            XCTAssertLessThanOrEqual(meanAbs, 0.025, "\(label) meanAbs=\(meanAbs)")
            XCTAssertGreaterThan(cosine, 0.99998, "\(label) cosine=\(cosine)")
            XCTAssertEqual(
                argMax(lhs, axis: 0).item(Int.self),
                argMax(rhs, axis: 0).item(Int.self),
                "\(label) top-1")
        }

        // Keep one numerical decoder oracle in addition to token IDs.  BF16
        // greedy decoding is allowed to choose a different ID when two
        // logits round to the same value, but the logits themselves must
        // still match the Python MLX implementation.  This catches weight
        // layout, RoPE, attention-mask, and KV bugs before the first tie.
        let logitsFixtureURL = root.appendingPathComponent("minicpm-tts-logits-oracle.safetensors")
        if FileManager.default.fileExists(atPath: logitsFixtureURL.path) {
            let logitsArrays = try MLX.loadArrays(url: logitsFixtureURL)
            let probeCondition = try model.buildCondition(
                hidden: try XCTUnwrap(arrays["chunk0.hidden"]),
                tokenIDs: MLXArray([Int32(22), Int32(33), Int32(44)]))
            let probeSession = model.makeSession()
            let hidden = model.decoder(
                probeCondition,
                caches: probeSession.layerCaches,
                offset: 0)
            let actualLogits = model.audioHead(
                hidden[0, hidden.dim(1) - 1]).asType(.float32)
            try assertLogitsClose(
                actualLogits,
                try XCTUnwrap(logitsArrays["chunk0.first_logits"]),
                "chunk0.first_logits")
        }

        // results=[] in the official helper is audio BOS only.
        let empty = try model.buildCondition(
            hidden: MLXArray.zeros([0, model.configuration.llmDim], dtype: .bfloat16),
            tokenIDs: MLXArray.zeros([0], dtype: .int32))
        try assertClose(empty, try XCTUnwrap(arrays["empty.expected"]), "empty")

        let conditionCases: [(String, [Int32])] = [
            ("filtered", [22, 33, 44]), // raw duplex first <|speak|> was filtered
            ("control", [55, 151717]), // <|turn_eos|> remains in official results
        ]
        for (name, tokens) in conditionCases {
            let hidden = try XCTUnwrap(arrays["\(name).hidden"])
            let actual = try model.buildCondition(
                hidden: hidden,
                tokenIDs: MLXArray(tokens.map { Int32($0) }))
            try assertClose(actual, try XCTUnwrap(arrays["\(name).expected"]), name)
        }

        guard let chunkDefinitions = metadata["chunks"] as? [[String: Any]],
              let trace = metadata["trace"] as? [[String: Any]]
        else {
            throw MiniCPMTTSSemanticError.invalidInput("oracle metadata has no chunk trace")
        }
        XCTAssertEqual(chunkDefinitions.count, 3)
        XCTAssertEqual(trace.count, chunkDefinitions.count)

        func intList(_ value: Any?) -> [Int] {
            (value as? [Any] ?? []).compactMap { ($0 as? NSNumber)?.intValue }
        }

        let session = model.makeSession()
        for (index, definition) in chunkDefinitions.enumerated() {
            let traceEntry = trace[index]
            let hiddenKey = try XCTUnwrap(definition["hidden_key"] as? String)
            let expectedKey = try XCTUnwrap(definition["expected_key"] as? String)
            let tokens = intList(definition["tokens"])
            let condition = try model.buildCondition(
                hidden: try XCTUnwrap(arrays[hiddenKey]),
                tokenIDs: MLXArray(tokens.map { Int32($0) }))
            try assertClose(condition, try XCTUnwrap(arrays[expectedKey]), "chunk\(index)")

            let isFinal = (traceEntry["is_final"] as? Bool) ?? false
            let result = try model.generateStreamingChunk(
                inputsEmbeds: condition,
                session: session,
                endOfTurn: isFinal,
                temperature: 0,
                repetitionPenalty: 1,
                forceNoStop: true,
                maxNewToken: 26)
            XCTAssertEqual(result.conditionTokenCount, (traceEntry["condition_length"] as? NSNumber)?.intValue)
            XCTAssertEqual(result.textStartPosition, (traceEntry["text_start"] as? NSNumber)?.intValue)
            XCTAssertEqual(result.nextTextStartPosition, (traceEntry["next_text_start"] as? NSNumber)?.intValue)
            XCTAssertEqual(result.sampledTokenCount, (traceEntry["sampled_count"] as? NSNumber)?.intValue)
            let expectedCommitted = intList(traceEntry["committed"])
            XCTAssertEqual(result.committedTokens.count, expectedCommitted.count)
            XCTAssertEqual(
                result.committedTokens.count,
                25,
                "a full non-EOS chunk must expose 25 committed codes")
            XCTAssertTrue(
                result.committedTokens.allSatisfy { $0 >= 0 && $0 < model.configuration.numAudioTokens },
                "committed audio code outside vocabulary")
            if let stablePrefix = traceEntry["stable_prefix"] {
                let prefix = intList(stablePrefix)
                XCTAssertGreaterThanOrEqual(
                    result.committedTokens.count,
                    prefix.count,
                    "stable prefix longer than generated sequence")
                XCTAssertEqual(
                    Array(result.committedTokens.prefix(prefix.count)),
                    prefix,
                    "stable prefix before BF16 tie (chunk\(index))")
            }
            // The protocol facade hides the sampled look-ahead token from
            // Token2Wav, while the session retains it for the next decoder KV
            // update.  This distinction is what makes offsets 29 -> 60 -> 89
            // match the official PyTorch trace.
            XCTAssertNil(result.pendingToken, "look-ahead is hidden from Token2Wav")
            // The official fixture records its reference pending ID for
            // diagnostics, but BF16 argmax ties can select a different legal
            // ID in Swift.  Retention and KV length are the parity contract;
            // exact IDs are checked only through the stable-prefix rule.
            XCTAssertNotNil(session.pendingToken, "look-ahead is retained in the semantic session")
            if let pending = session.pendingToken {
                XCTAssertGreaterThanOrEqual(pending, 0)
                XCTAssertLessThan(pending, model.configuration.numAudioTokens)
            }
            XCTAssertEqual(result.cacheOffset, (traceEntry["cache_offset"] as? NSNumber)?.intValue)
            XCTAssertEqual(result.isFirstChunk, (traceEntry["is_first"] as? Bool) ?? false)
            XCTAssertEqual(result.isFinalChunk, isFinal)
            XCTAssertTrue(
                (traceEntry["force_flush"] as? Bool) ?? false,
                "fixed MiniCPM-o unified duplex force-flushes every TTS chunk")
            XCTAssertFalse(result.endedByEOS)
            XCTAssertFalse(result.sessionWasReset)

            if isFinal {
                // Deferred-finalize contract: flush token2wav first, then
                // release semantic KV and text position.
                XCTAssertGreaterThan(session.cacheOffset, 0)
                session.finishTurn()
            }
        }
        XCTAssertEqual(session.cacheOffset, 0)
        XCTAssertEqual(session.textStartPosition, 0)
        XCTAssertEqual(session.chunkIndex, 0)
    }

    /// Diagnostic-only layer trace used while reconciling the native decoder
    /// with the pinned official PyTorch Llama implementation.  The trace is
    /// deliberately opt-in and prints every operation's error rather than
    /// changing the production tolerance of the semantic parity gate.
    func testDiagnosticOfficialLayerTrace() throws {
        let environment = ProcessInfo.processInfo.environment
        guard let tracePath = environment["MINICPM_TTS_LAYER_TRACE_PATH"],
              !tracePath.isEmpty,
              let conditionPath = environment["MINICPM_TTS_CONDITION_ORACLE_DIR"],
              !conditionPath.isEmpty,
              let bundlePath = environment["MINICPM_TTS_BUNDLE_PATH"],
              !bundlePath.isEmpty
        else {
            throw XCTSkip(
                "set MINICPM_TTS_LAYER_TRACE_PATH, MINICPM_TTS_CONDITION_ORACLE_DIR, and MINICPM_TTS_BUNDLE_PATH")
        }

        let model = try MiniCPMTTSSemantic.fromDirectory(URL(fileURLWithPath: bundlePath))
        let conditionArrays = try MLX.loadArrays(
            url: URL(fileURLWithPath: conditionPath)
                .appendingPathComponent("minicpm-tts-condition-oracle.safetensors"))
        var condition = try XCTUnwrap(conditionArrays["chunk0.expected"])
        if environment["MINICPM_TTS_DIAGNOSTICS_REBUILT"] == "1" {
            let hidden = try XCTUnwrap(conditionArrays["chunk0.hidden"])
            condition = try model.buildCondition(
                hidden: hidden,
                tokenIDs: MLXArray([Int32(22), Int32(33), Int32(44)]))
        }
        let trace = try MLX.loadArrays(url: URL(fileURLWithPath: tracePath))

        func report(_ actual: MLXArray, _ key: String) throws {
            guard let expected = trace[key] else {
                return XCTFail("missing diagnostic trace tensor (\(key))")
            }
            XCTAssertEqual(actual.shape, expected.shape, "\(key) shape")
            let lhs = actual.asType(.float32)
            let rhs = expected.asType(.float32)
            let delta = MLX.maximum(lhs - rhs, rhs - lhs)
            let maxAbs = delta.max().item(Float.self)
            let meanAbs = delta.mean().item(Float.self)
            let dot = (lhs * rhs).sum().item(Float.self)
            let norm = MLX.sqrt((lhs * lhs).sum() * (rhs * rhs).sum()).item(Float.self)
            let cosine = norm > 0 ? dot / norm : 1
            print("TTS_LAYER_TRACE \(key) max=\(maxAbs) mean=\(meanAbs) cosine=\(cosine)")
            if ProcessInfo.processInfo.environment["MINICPM_TTS_DIAGNOSTICS"] == "1",
               key == "layer_10.mlp_down"
            {
                let flatDelta = delta.reshaped(-1)
                let index = argMax(flatDelta, axis: 0).item(Int.self)
                print(
                    "TTS_LAYER_TRACE_DETAIL \(key) index=\(index) actual=\(lhs.reshaped(-1)[index].item(Float.self)) expected=\(rhs.reshaped(-1)[index].item(Float.self))")
            }
        }

        let session = model.makeSession()
        let caches = session.layerCaches
        let batch = condition.dim(0)
        let sequence = condition.dim(1)
        // ``tri - 1`` is only a 0/-1 indicator.  The official additive mask
        // uses the dtype's large negative sentinel for disallowed positions;
        // omitting that scale lets future tokens leak into the diagnostic and
        // falsely attributes the resulting drift to RoPE/SDPA.
        let causal = (MLXArray.tri(sequence, m: sequence, k: 0, type: Float.self) - 1)
            * Float.greatestFiniteMagnitude
        let mask = MLXFast.ScaledDotProductAttentionMaskMode.array(
            causal.reshaped(batch, 1, sequence, sequence).asType(condition.dtype))

        try report(condition, "embedding_input")
        var hidden = condition
        for (index, layer) in model.decoder.layers.enumerated() {
            let prefix = String(format: "layer_%02d", index)
            try report(hidden, prefix + ".input")
            let inputNorm = layer.inputLayerNorm(hidden)
            try report(inputNorm, prefix + ".input_norm")
            // Split the attention path at the same boundaries captured by
            // the official PyTorch diagnostic exporter.  This tells us
            // whether the first divergence is in Linear layout, RoPE, or
            // SDPA/masking rather than inferring it from the fused output.
            let rawQ = layer.attention.query(inputNorm)
            let rawK = layer.attention.key(inputNorm)
            let rawV = layer.attention.value(inputNorm)
            try report(rawQ, prefix + ".q_proj")
            try report(rawK, prefix + ".k_proj")
            try report(rawV, prefix + ".v_proj")
            let qShape = [batch, sequence, layer.attention.numHeads, layer.attention.headDimension]
            let kShape = [batch, sequence, layer.attention.numKVHeads, layer.attention.headDimension]
            let qHeads = rawQ.reshaped(qShape).transposed(0, 2, 1, 3)
            let kHeads = rawK.reshaped(kShape).transposed(0, 2, 1, 3)
            let qRotated = layer.attention.rope(qHeads, offset: 0)
            let kRotated = layer.attention.rope(kHeads, offset: 0)
            try report(qRotated, prefix + ".q_rotated")
            try report(kRotated, prefix + ".k_rotated")
            let attended = layer.attention(
                inputNorm,
                cache: caches[index],
                offset: 0,
                mask: mask)
            try report(attended, prefix + ".attention")
            let residual = hidden + attended
            let postNorm = layer.postAttentionLayerNorm(residual)
            try report(postNorm, prefix + ".post_attention_norm")
            let mlpGate = layer.mlp.gate(postNorm)
            let mlpUp = layer.mlp.up(postNorm)
            let mlpActivation = silu(mlpGate)
            let mlp = layer.mlp.down(mlpActivation * mlpUp)
            try report(mlpGate, prefix + ".mlp_gate")
            try report(mlpUp, prefix + ".mlp_up")
            try report(mlpActivation, prefix + ".mlp_activation")
            try report(mlp, prefix + ".mlp_down")
            try report(mlp, prefix + ".mlp")
            hidden = residual + mlp
            try report(hidden, prefix + ".output")
        }
        let final = model.decoder.norm(hidden)
        try report(final, "final_output")
        let logits = model.audioHead(final[0, final.dim(1) - 1]).asType(.float32)
        try report(logits, "first_logits")
    }
}
