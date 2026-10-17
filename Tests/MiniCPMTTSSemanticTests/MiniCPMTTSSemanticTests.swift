import Foundation
import XCTest
import MLX
import MLXNN
import MiniCPMLLM

@testable import MiniCPMTTSSemantic

final class MiniCPMTTSSemanticTests: XCTestCase {
    func testRepetitionPenaltyHostCountsPreserveOriginalFP32Operations() {
        let model = tinyModel()
        let size = model.configuration.numAudioTokens
        let logits = MLXArray((0..<size).map { Float($0 % 9 - 4) / 8 })
        let generated = [0, 0, 2, 3, 2, -1, size, 1, 4, 5, 6, 7, 2, 2, 2, 3, 3, 1]
        let recent = Array(generated.suffix(16))
        let ids = MLXArray(recent.map(Int32.init)).expandedDimensions(axis: 1)
        let vocab = MLXArray(0..<Int32(size)).expandedDimensions(axis: 0)
        let frequencies = (ids .== vocab).asType(.float32).sum(axis: 0)
        for penalty: Float in [0.75, 1, 1.05, 2] {
            let alpha = MLX.exp(MLX.log(MLXArray(penalty)) * frequencies)
            let expected = MLX.where(logits .< MLXArray(Float(0)), logits * alpha, logits / alpha)
            let actual = model.applyRepetitionPenalty(
                logits, generated: generated, penalty: penalty)
            XCTAssertEqual(actual.asData(access: .copy).data,
                           expected.asData(access: .copy).data)
        }
    }
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

    func testExactBF16AttentionKeepsOfficialSingleTokenVectorRoute() {
        XCTAssertFalse(MiniCPMTTSAudioAttention.shouldUseExactBF16SDPA(
            enabled: true, sequence: 1,
            queryDType: .bfloat16, keyDType: .bfloat16, valueDType: .bfloat16))
        XCTAssertTrue(MiniCPMTTSAudioAttention.shouldUseExactBF16SDPA(
            enabled: true, sequence: 4,
            queryDType: .bfloat16, keyDType: .bfloat16, valueDType: .bfloat16))
        XCTAssertFalse(MiniCPMTTSAudioAttention.shouldUseExactBF16SDPA(
            enabled: false, sequence: 1,
            queryDType: .bfloat16, keyDType: .bfloat16, valueDType: .bfloat16))
        for dtype in [DType.float16, .float32] {
            XCTAssertFalse(MiniCPMTTSAudioAttention.shouldUseExactBF16SDPA(
                enabled: true, sequence: 1,
                queryDType: dtype, keyDType: dtype, valueDType: dtype))
        }
    }

    func testDiagnosticCachedSingleTokenProbeSeesEntirePrefix() {
        // Equal scores give a uniform distribution over all three keys.
        // Looking only at the first key would yield 3 rather than 1.
        let query = MLXArray.zeros([1, 1, 1, 2], dtype: .bfloat16)
        let key = MLXArray.zeros([1, 1, 3, 2], dtype: .bfloat16)
        let value = MLXArray([Float(3), 3, 0, 0, 0, 0], [1, 1, 3, 2]).asType(.bfloat16)
        let mask = MiniCPMExactBF16SDPA.rightAlignedAdditiveMask(
            batch: 1, queryHeads: 1, queryTokens: 1, keyTokens: 3)
        XCTAssertEqual(mask.asType(.float32).asArray(Float.self), [0, 0, 0])
        let output = MiniCPMExactBF16SDPA.attend(
            query: query, key: key, value: value, scale: 1,
            additiveMask: mask)
        XCTAssertEqual(output.shape, [1, 1, 1, 2])
        XCTAssertEqual(output.asType(.float32).asArray(Float.self), [1, 1])
    }

    private func tinyOfficialOracle() throws -> ([String: MLXArray], [String: Any]) {
        let root = try XCTUnwrap(Bundle.module.resourceURL)
        let arrays = try MLX.loadArrays(url: root.appendingPathComponent(
            "Fixtures/minicpm-tts-tiny-oracle.safetensors"))
        let metadata = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(
            contentsOf: root.appendingPathComponent("Fixtures/minicpm-tts-tiny-oracle.json")))
            as? [String: Any])
        XCTAssertEqual(metadata["oracle_backend"] as? String, "official_pytorch_tiny")
        XCTAssertEqual(metadata["modeling_sha256"] as? String,
                       "877f20cea7282bb0d05684393a1580646d1f264a2299a42f82467fd99f4028af")
        return (arrays, metadata)
    }

    func testCheckpointDTypeRoundsRotaryFrequencyBufferLikeOfficialHF() throws {
        let (arrays, _) = try tinyOfficialOracle()
        let input = try XCTUnwrap(arrays["rotary.input"])
        let expected = try XCTUnwrap(arrays["rotary.expected"])
        let actual = MiniCPMTTSHFStyleRoPE(dimensions: 4)(input, offset: 53)
        XCTAssertEqual(actual.dtype, .bfloat16)
        XCTAssertEqual(actual.asType(.float32).asArray(Float.self),
                       expected.asType(.float32).asArray(Float.self))
    }

    func testBF16SiLURoundsAfterFullActivationLikeOfficialTorch() throws {
        let (arrays, _) = try tinyOfficialOracle()
        let input = try XCTUnwrap(arrays["activation.input"])
        let expected = try XCTUnwrap(arrays["activation.expected"])
        let actual = MiniCPMTTSMLP.activate(input)
        XCTAssertEqual(actual.asType(.float32).asArray(Float.self),
                       expected.asType(.float32).asArray(Float.self))
        XCTAssertNotEqual(silu(input).asType(.float32).asArray(Float.self),
                          expected.asType(.float32).asArray(Float.self))
    }

    func testBF16HeadWeightNormalizationMatchesTorchMPS() throws {
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
        let arrays = try MLX.loadArrays(url: root.appendingPathComponent(
            "Fixtures/minicpm-tts-weight-norm-mps.safetensors"))
        let direction = try XCTUnwrap(arrays["direction"])
        let magnitude = try XCTUnwrap(arrays["magnitude"])
        let expected = try XCTUnwrap(arrays["expected"])
        let actual = MiniCPMTTSWeightNormalization.reconstruct(
            direction: direction, magnitude: magnitude)
        XCTAssertEqual(actual.dtype, .bfloat16)
        XCTAssertEqual(actual.asType(.float32).asArray(Float.self),
                       expected.asType(.float32).asArray(Float.self))
        let v32 = direction.asType(.float32)
        let old = (v32 * magnitude.asType(.float32)
            / sqrt((v32 * v32).sum(axis: -1, keepDims: true))).asType(.bfloat16)
        XCTAssertNotEqual(old.asType(.float32).asArray(Float.self),
                          expected.asType(.float32).asArray(Float.self))
    }

    func testProductionKernelPolicyUsesValidatedLinearAndGeneralAttention() {
        XCTAssertTrue(MiniCPMTTSKernelPolicy.usesMPSGraph(
            kernel: "LINEAR", defaultEnabled: true, environment: [:]))
        XCTAssertTrue(MiniCPMTTSKernelPolicy.usesMPSGraph(
            kernel: "SDPA", defaultEnabled: true, environment: [:]))
        XCTAssertFalse(MiniCPMTTSKernelPolicy.usesMPSGraph(
            kernel: "RMSNORM", defaultEnabled: false, environment: [:]))
        XCTAssertFalse(MiniCPMTTSKernelPolicy.usesMPSGraph(
            kernel: "LINEAR", defaultEnabled: true,
            environment: ["MINICPM_TTS_BF16_LINEAR": "mlx", "MINICPM_TTS_EXACT_KERNELS": "mpsgraph"]))
        XCTAssertTrue(MiniCPMTTSKernelPolicy.usesMPSGraph(
            kernel: "RMSNORM", defaultEnabled: false,
            environment: ["MINICPM_TTS_EXACT_KERNELS": "mpsgraph"]))
    }


    /// Exercise the production bundle loader with independently exported tiny
    /// Torch weights, including its weight-normalized head reconstruction.
    private func loadTinyOfficialModel(
        arrays: [String: MLXArray], metadata: [String: Any], dtype: DType = .float32
    ) throws -> MiniCPMTTSSemantic {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let config = try XCTUnwrap(metadata["config"] as? [String: Any])
        try JSONSerialization.data(withJSONObject: config).write(to: directory.appendingPathComponent("config.json"))
        var weights: [String: MLXArray] = [:]
        for (key, value) in arrays where key.hasPrefix("weight.") {
            weights[String(key.dropFirst(7))] = value.asType(dtype)
        }
        try MLX.save(arrays: weights, url: directory.appendingPathComponent("model.safetensors"))
        return try MiniCPMTTSSemantic.fromDirectory(directory)
    }

    func testTinyOfficialTeacherForcedLogitsAndProductionStreaming() throws {
        let (arrays, metadata) = try tinyOfficialOracle()
        let model = try loadTinyOfficialModel(arrays: arrays, metadata: metadata)
        let trace = try XCTUnwrap(metadata["trace"] as? [[String: Any]])
        let teacher = model.makeSession()
        let streaming = model.makeSession()
        for (index, entry) in trace.enumerated() {
            let ids = try XCTUnwrap(entry["tokens"] as? [Int])
            let condition = try model.buildCondition(
                hidden: try XCTUnwrap(arrays["chunk\(index).hidden"]),
                tokenIDs: MLXArray(ids.map(Int32.init)))
            let expectedCondition = try XCTUnwrap(arrays["chunk\(index).condition"])
            XCTAssertLessThan(MLX.abs(condition - expectedCondition).max().item(Float.self), 0.0005)
            let sampled = try XCTUnwrap(arrays["chunk\(index).sampled"]).asArray(Int32.self)
            let referenceLogits = try XCTUnwrap(arrays["chunk\(index).logits"])
            for step in 0..<sampled.count {
                let input = step == 0 ? condition : model.audioEmbedding(
                    MLXArray([sampled[step - 1]])).expandedDimensions(axis: 0)
                let hidden = model.decoder(input, caches: teacher.layerCaches, offset: teacher.cacheOffset)
                let logits = model.audioHead(hidden[0, hidden.dim(1) - 1])
                XCTAssertLessThan(
                    MLX.abs(logits - referenceLogits[step]).max().item(Float.self), 0.0005,
                    "official teacher-forced chunk \(index), step \(step)")
            }
            let result = try model.generateStreamingChunk(
                inputsEmbeds: condition, session: streaming,
                endOfTurn: index == trace.count - 1,
                temperature: 0, forceNoStop: true, maxNewToken: 26)
            XCTAssertEqual(result.committedTokens, entry["committed"] as? [Int])
            XCTAssertEqual(result.cacheOffset, entry["cache_offset"] as? Int)
            XCTAssertEqual(teacher.cacheOffset, result.cacheOffset)
            XCTAssertEqual(streaming.textStartPosition, result.cacheOffset)
            XCTAssertNil(streaming.pendingToken)
        }
    }

    func testProductionLoaderKeepsAudioHeadInCheckpointDType() throws {
        let (arrays, metadata) = try tinyOfficialOracle()
        for dtype in [DType.bfloat16, .float16, .float32] {
            let model = try loadTinyOfficialModel(arrays: arrays, metadata: metadata, dtype: dtype)
            XCTAssertEqual(model.textEmbedding.weight.dtype, dtype)
            XCTAssertEqual(model.audioHead.weight.dtype, dtype)
        }
    }

    func testBF16ConditionMatchesOfficialNormalizationBoundary() throws {
        let (arrays, _) = try tinyOfficialOracle()
        let model = tinyModel()
        let projected = try XCTUnwrap(arrays["normalization.input"])
        try model.update(parameters: ModuleParameters.unflattened([
            "textEmbedding.weight": MLXArray.zeros([10, 8], dtype: .bfloat16),
            "semanticProjector.linear1.weight": MLXArray.zeros([8, 4], dtype: .bfloat16),
            "semanticProjector.linear1.bias": projected,
            "semanticProjector.linear2.weight": MLXArray.eye(8).asType(.bfloat16),
            "semanticProjector.linear2.bias": MLXArray.zeros([8], dtype: .bfloat16),
        ]), verify: .none)
        let condition = try model.buildCondition(
            hidden: MLXArray.zeros([1, 4], dtype: .bfloat16),
            tokenIDs: MLXArray([Int32(1)]), addAudioBOS: false)
        XCTAssertEqual(condition[0, 0].asType(.float32).asArray(Float.self),
                       try XCTUnwrap(arrays["normalization.expected"]).asType(.float32).asArray(Float.self))
    }

    func testControlOnlyFinalConditionsRemainAvailableForWaveformFlush() throws {
        let model = tinyModel()
        try model.audioHead.update(parameters: ModuleParameters.unflattened([
            "weight": MLXArray.zeros([7, 8]),
        ]), verify: .none)
        let conditions = [
            try model.buildCondition(
                hidden: MLXArray.zeros([0, 4]), tokenIDs: MLXArray.zeros([0], dtype: .int32)),
            try model.buildCondition(
                hidden: MLXArray.zeros([1, 4]), tokenIDs: MLXArray([Int32(8)])),
        ]
        for condition in conditions {
            let session = model.makeSession()
            _ = try model.generateStreamingChunk(
                inputsEmbeds: MLXArray.zeros([1, 2, 8]), session: session,
                temperature: 0, eosToken: 0, forceNoStop: true)
            let previousOffset = session.cacheOffset
            let final = try model.generateStreamingChunk(
                inputsEmbeds: condition, session: session, endOfTurn: true,
                temperature: 0, eosToken: 0)
            XCTAssertTrue(final.isFinalChunk)
            XCTAssertTrue(final.endedByEOS)
            XCTAssertEqual(final.sampledTokenCount, 1)
            XCTAssertEqual(final.cacheOffset, previousOffset + condition.dim(1))
            XCTAssertFalse(final.sessionWasReset, "caller must flush waveform before teardown")
            XCTAssertNil(session.pendingToken)
            XCTAssertFalse(session.turnActive)
        }
    }

    func testStreamingFirstAndFinalAllowEOSButMiddleMasksIt() throws {
        let model = tinyModel()
        try model.audioHead.update(parameters: ModuleParameters.unflattened([
            "weight": MLXArray.zeros([7, 8]),
        ]), verify: .none)
        let session = model.makeSession()
        let condition = MLXArray.zeros([1, 2, 8])
        let first = try model.generateStreamingChunk(
            inputsEmbeds: condition, session: session, temperature: 0, eosToken: 0)
        XCTAssertTrue(first.endedByEOS)
        XCTAssertFalse(first.isFinalChunk)
        XCTAssertEqual(first.sampledTokenCount, 1)
        XCTAssertEqual(session.cacheOffset, 2)
        let middle = try model.generateStreamingChunk(
            inputsEmbeds: condition, session: session, temperature: 0, eosToken: 0)
        XCTAssertFalse(middle.endedByEOS)
        XCTAssertEqual(middle.sampledTokenCount, 26)
        XCTAssertEqual(middle.committedTokenCount, 25)
        XCTAssertFalse(middle.committedTokens.contains(0))
        let final = try model.generateStreamingChunk(
            inputsEmbeds: condition, session: session, endOfTurn: true,
            temperature: 0, eosToken: 0)
        XCTAssertTrue(final.endedByEOS)
        XCTAssertTrue(final.isFinalChunk)
        XCTAssertEqual(final.sampledTokenCount, 1)
        XCTAssertEqual(session.cacheOffset, 31)
        XCTAssertEqual(session.cacheOffset, session.textStartPosition)
        XCTAssertNil(session.pendingToken)
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
        XCTAssertNil(session.pendingToken, "official streaming discards the newest sample")
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
        XCTAssertNil(session.pendingToken, "official streaming discards the newest sample")
        XCTAssertEqual(middle.textStartPosition, 27)
        XCTAssertEqual(middle.nextTextStartPosition, 55) // +3 +25 returned
        XCTAssertEqual(session.cacheOffset, 55) // new condition + 25 fed codes
        XCTAssertEqual(session.cacheOffset, session.textStartPosition)

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
        XCTAssertEqual(final.cacheOffset, 81) // condition + 25 fed codes
        // The caller flushes token2wav before releasing semantic KV.
        XCTAssertEqual(session.cacheOffset, 81)
        XCTAssertNil(session.pendingToken)
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

    func testStreamingFirstChunkAllowsEOSWithoutClosingTurn() throws {
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
        XCTAssertFalse(result.isFinalChunk)
        XCTAssertTrue(session.turnActive)
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

    func testRealBundleLoadsCheckpointDTypesAndCacheContract() throws {
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

        XCTAssertEqual(model.audioHead.weight.dtype, model.textEmbedding.weight.dtype)
        XCTAssertEqual(result.committedTokens.count, 2)
        XCTAssertNotNil(result.pendingToken)
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
        guard (metadata["format"] as? Int ?? 0) >= 2 else {
            throw MiniCPMTTSSemanticError.invalidInput(
                "legacy decoder trace is not an oracle; regenerate with export_minicpm_tts_semantic_oracle.py")
        }
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
        let traceContract = try XCTUnwrap(metadata["trace_contract"] as? [String: Any])
        XCTAssertEqual(traceContract["teacher_forcing_required"] as? Bool, true)
        XCTAssertEqual(
            traceContract["stable_prefix_contract"] as? String,
            "unique argmax before first exact tie; no free-running token equality from that tie onward, including subsequent chunks")
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

        }

        // The official exporter stores logits in the main fixture. Missing
        // logits are a failure, never a silent skip or another MLX oracle.
        let probeCondition = try XCTUnwrap(arrays["chunk0.expected"])
        let probeSession = model.makeSession()
        let probeHidden = model.decoder(
            probeCondition, caches: probeSession.layerCaches, offset: 0)
        try assertLogitsClose(
            model.audioHead(probeHidden[0, probeHidden.dim(1) - 1]).asType(.float32),
            try XCTUnwrap(arrays["chunk0.first_logits"]), "chunk0.first_logits")

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
        let teacherSession = model.makeSession()
        var freeRunningPrefixIsValid = true
        for (index, definition) in chunkDefinitions.enumerated() {
            let traceEntry = trace[index]
            let hiddenKey = try XCTUnwrap(definition["hidden_key"] as? String)
            let expectedKey = try XCTUnwrap(definition["expected_key"] as? String)
            let tokens = intList(definition["tokens"])
            let condition = try model.buildCondition(
                hidden: try XCTUnwrap(arrays[hiddenKey]),
                tokenIDs: MLXArray(tokens.map { Int32($0) }))
            try assertClose(condition, try XCTUnwrap(arrays[expectedKey]), "chunk\(index)")

            let referenceCodes = intList(traceEntry["committed"])
            let referenceLogits = try XCTUnwrap(arrays["chunk\(index).teacher_logits"])
            XCTAssertEqual(referenceLogits.dim(0), referenceCodes.count + 1)
            // The official exporter uses multinomial with a tiny positive
            // temperature.  Swift greedy argmax can select another exact
            // BF16 tie, after which later free-running inputs diverge.  Check
            // prefix metadata against the logits rather than trusting a
            // hard-coded number of supposedly stable tokens.
            let rows = referenceLogits.asType(.float32).asArray(Float.self)
            let width = referenceLogits.dim(1)
            let eos = model.configuration.numAudioTokens - 1
            var firstTie: Int?
            for step in 0 ..< referenceLogits.dim(0) {
                let row = Array(rows[(step * width) ..< ((step + 1) * width)])
                let candidates = row.enumerated().filter { $0.offset != eos }
                let maximum = try XCTUnwrap(candidates.map(\.element).max())
                if candidates.filter({ $0.element == maximum }).count > 1 {
                    firstTie = step
                    break
                }
            }
            let stableCount = freeRunningPrefixIsValid
                ? min(referenceCodes.count, firstTie ?? referenceCodes.count)
                : 0
            XCTAssertEqual(intList(traceEntry["stable_prefix"]), Array(referenceCodes.prefix(stableCount)))
            XCTAssertEqual(traceEntry["free_running_prefix_valid_at_chunk_start"] as? Bool, freeRunningPrefixIsValid)
            if let firstTie {
                let tieMetadata = try XCTUnwrap(traceEntry["first_argmax_tie"] as? [String: Any])
                XCTAssertEqual(tieMetadata["step_index"] as? Int, firstTie)
                XCTAssertEqual((tieMetadata["top_gap"] as? NSNumber)?.floatValue, 0)
                freeRunningPrefixIsValid = false
            }
            for step in 0 ..< referenceLogits.dim(0) {
                let input = step == 0 ? condition : model.audioEmbedding(
                    MLXArray([Int32(referenceCodes[step - 1])])).expandedDimensions(axis: 0)
                let hidden = model.decoder(
                    input, caches: teacherSession.layerCaches, offset: teacherSession.cacheOffset)
                try assertLogitsClose(
                    model.audioHead(hidden[0, hidden.dim(1) - 1]).asType(.float32),
                    referenceLogits[step], "chunk\(index).step\(step)")
            }
            XCTAssertEqual(teacherSession.cacheOffset, (traceEntry["cache_offset"] as? NSNumber)?.intValue)

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
                    "stable prefix before first exact BF16 argmax tie (chunk\(index))")
            }
            XCTAssertNil(result.pendingToken)
            XCTAssertNil(session.pendingToken, "newest official sample is discarded")
            XCTAssertEqual(session.cacheOffset, session.textStartPosition)
            XCTAssertEqual(result.cacheOffset, (traceEntry["cache_offset"] as? NSNumber)?.intValue)
            XCTAssertEqual(result.isFirstChunk, (traceEntry["is_first"] as? Bool) ?? false)
            XCTAssertEqual(result.isFinalChunk, isFinal)
            XCTAssertEqual(
                traceEntry["force_flush"] as? Bool,
                index == 0,
                "modeling_minicpmo.py requests token2wav force_flush only for the first chunk; final flushing also uses end_of_turn")
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
        if ProcessInfo.processInfo.environment["MINICPM_TTS_ENCODED_DECODE"] == "1" {
            XCTAssertGreaterThanOrEqual(
                model.decoder.encodedDecodeCount, 78,
                "the official trace must exercise encoded decode rather than silently fall back")
        }
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
        if let embedding = trace["embedding_input"] {
            condition = embedding.asType(model.audioEmbedding.weight.dtype)
        }
        let localOperators = environment["MINICPM_TTS_DIAGNOSTICS_LOCAL"] == "1"
        let offset = trace["layer_00.cache_keys_before"]?.dim(2) ?? 0

        func report(_ actual: MLXArray, _ key: String, variant: String = "") throws {
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
            print("TTS_LAYER_TRACE \(key)\(variant) max=\(maxAbs) mean=\(meanAbs) cosine=\(cosine)")
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
        for (index, cache) in caches.enumerated() {
            let prefix = String(format: "layer_%02d", index)
            if let keys = trace[prefix + ".cache_keys_before"],
               let values = trace[prefix + ".cache_values_before"], keys.dim(2) > 0 {
                _ = cache.update(
                    keys: keys.asType(condition.dtype),
                    values: values.asType(condition.dtype))
            }
        }
        let batch = condition.dim(0)
        let sequence = condition.dim(1)
        // ``tri - 1`` is only a 0/-1 indicator.  The official additive mask
        // uses the dtype's large negative sentinel for disallowed positions;
        // omitting that scale lets future tokens leak into the diagnostic and
        // falsely attributes the resulting drift to RoPE/SDPA.
        let keyLength = offset + sequence
        let causal = (MLXArray.tri(sequence, m: keyLength, k: offset, type: Float.self) - 1)
            * Float.greatestFiniteMagnitude
        let mask: MLXFast.ScaledDotProductAttentionMaskMode = sequence == 1 && offset > 0
            ? .none
            : .array(causal.reshaped(batch, 1, sequence, keyLength).asType(condition.dtype))

        func hfRMSNorm(_ input: MLXArray, _ norm: RMSNorm) -> MLXArray {
            let value = input.asType(.float32)
            let scale = rsqrt((value * value).mean(axis: -1, keepDims: true) + norm.eps)
            return (value * scale).asType(input.dtype) * norm.weight
        }

        try report(condition, "embedding_input")
        var hidden = condition
        for (index, layer) in model.decoder.layers.enumerated() {
            let prefix = String(format: "layer_%02d", index)
            if localOperators {
                hidden = try XCTUnwrap(trace[prefix + ".input"]).asType(condition.dtype)
            }
            try report(hidden, prefix + ".input")
            let inputNorm = layer.inputLayerNorm(hidden)
            try report(inputNorm, prefix + ".input_norm")
            try report(hfRMSNorm(hidden, layer.inputLayerNorm), prefix + ".input_norm", variant: ".hf_candidate")
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
            let qRotated = layer.attention.rope(qHeads, offset: offset)
            let kRotated = layer.attention.rope(kHeads, offset: offset)
            try report(qRotated, prefix + ".q_rotated")
            try report(kRotated, prefix + ".k_rotated")
            let attended = layer.attention(
                inputNorm,
                cache: caches[index],
                offset: offset,
                mask: mask)
            try report(attended, prefix + ".attention")
            let residual = hidden + attended
            let postNorm = layer.postAttentionLayerNorm(residual)
            try report(postNorm, prefix + ".post_attention_norm")
            let normInput = localOperators
                ? try XCTUnwrap(trace[prefix + ".residual_before_mlp"]).asType(condition.dtype)
                : residual
            try report(layer.postAttentionLayerNorm(normInput), prefix + ".post_attention_norm", variant: ".native_local")
            try report(hfRMSNorm(normInput, layer.postAttentionLayerNorm), prefix + ".post_attention_norm", variant: ".hf_candidate")
            let mlpGate = layer.mlp.gate(postNorm)
            let mlpUp = layer.mlp.up(postNorm)
            let mlpActivation = MiniCPMTTSMLP.activate(mlpGate)
            let mlp = layer.mlp.down(mlpActivation * mlpUp)
            try report(mlpGate, prefix + ".mlp_gate")
            try report(mlpUp, prefix + ".mlp_up")
            try report(mlpActivation, prefix + ".mlp_activation")
            let oracleGate = try XCTUnwrap(trace[prefix + ".mlp_gate"]).asType(condition.dtype)
            let floatGate = oracleGate.asType(.float32)
            try report(oracleGate / (1 + exp(-oracleGate)), prefix + ".mlp_activation", variant: ".bf16_div_candidate")
            try report((floatGate * sigmoid(floatGate)).asType(condition.dtype), prefix + ".mlp_activation", variant: ".f32_sigmoid_candidate")
            try report((floatGate / (1 + exp(-floatGate))).asType(condition.dtype), prefix + ".mlp_activation", variant: ".f32_div_candidate")
            try report(mlp, prefix + ".mlp_down")
            try report(mlp, prefix + ".mlp")
            hidden = residual + mlp
            try report(hidden, prefix + ".output")
        }
        if localOperators {
            let last = String(format: "layer_%02d.output", model.decoder.layers.count - 1)
            hidden = try XCTUnwrap(trace[last]).asType(condition.dtype)
        }
        let final = model.decoder.norm(hidden)
        try report(final, "final_output")
        try report(hfRMSNorm(hidden, model.decoder.norm), "final_output", variant: ".hf_candidate")
        let logits = model.audioHead(final[0, final.dim(1) - 1]).asType(.float32)
        try report(logits, "first_logits")
    }
}

final class E2EMiniCPMTTSHeadTests: XCTestCase {
    func testDiagnosticCheckpointHeadWeightAndProjection() throws {
        let environment = ProcessInfo.processInfo.environment
        guard let conditionPath = environment["MINICPM_TTS_CONDITION_ORACLE_DIR"],
              let bundlePath = environment["MINICPM_TTS_BUNDLE_PATH"] else {
            throw XCTSkip("set TTS condition oracle and bundle paths")
        }
        let arrays = try MLX.loadArrays(url: URL(fileURLWithPath: conditionPath)
            .appendingPathComponent("head-mps-oracle.safetensors"))
        let model = try MiniCPMTTSSemantic.fromDirectory(URL(fileURLWithPath: bundlePath))
        let expected = try XCTUnwrap(arrays["expected"])
        XCTAssertEqual(model.audioHead.weight.asType(.float32).asArray(Float.self),
                       expected.asType(.float32).asArray(Float.self))
        if let tracePath = environment["MINICPM_TTS_LAYER_TRACE_PATH"] {
            let trace = try MLX.loadArrays(url: URL(fileURLWithPath: tracePath))
            let final = try XCTUnwrap(trace["final_output"])
                .asType(model.audioHead.weight.dtype)
            let logits = try XCTUnwrap(trace["first_logits"])
            for (name, actual) in [
                ("full", model.audioHead(final)[0, final.dim(1) - 1]),
                ("last", model.audioHead(final[0, final.dim(1) - 1])),
            ] {
                let delta = abs(actual.asType(.float32) - logits)
                print("TTS_HEAD_TRACE \(name) max=\(delta.max().item(Float.self)) "
                      + "mean=\(delta.mean().item(Float.self))")
            }
        }
    }

}
