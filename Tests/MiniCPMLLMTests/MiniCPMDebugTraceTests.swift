import Foundation
import XCTest
import MLX
import MLXCommon
import MLXNN
@testable import MiniCPMLLM

/// Opt-in layer trace used while reconciling the tiny MLX fixture with the
/// Python oracle.  It is skipped unless MINICPM_WRITE_TRACE points at a file,
/// so ordinary test runs do not allocate or serialize intermediate tensors.
final class MiniCPMDebugTraceTests: XCTestCase {
    private static let modelTraceTokenIDs = [151643, 198, 108, 151645]

    private static func parseTraceTokenIDs(for text: String?) throws -> [Int]? {
        guard let text else { return nil }
        let fields = text.split(separator: ",", omittingEmptySubsequences: false)
        guard !fields.isEmpty,
              fields.allSatisfy({
                  !String($0).trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
              })
        else {
            throw MiniCPMLLMError.invalidState(
                "MINICPM_TRACE_TOKEN_IDS must be a non-empty comma-separated sequence")
        }
        var tokenIDs: [Int] = []
        tokenIDs.reserveCapacity(fields.count)
        for field in fields {
            let tokenText = String(field).trimmingCharacters(in: .whitespacesAndNewlines)
            guard let tokenID = Int(tokenText), tokenID >= 0 else {
                throw MiniCPMLLMError.invalidState(
                    "MINICPM_TRACE_TOKEN_IDS contains an invalid token ID: \(tokenText)")
            }
            tokenIDs.append(tokenID)
        }
        return tokenIDs
    }

    private static func selectedTraceTokenIDs(
        tokenCountText: String?, tokenIDsText: String?
    ) throws -> [Int] {
        if let tokenIDs = try parseTraceTokenIDs(for: tokenIDsText) {
            return tokenIDs
        }
        let valueText = tokenCountText ?? "\(modelTraceTokenIDs.count)"
        guard let tokenCount = Int(valueText) else {
            throw MiniCPMLLMError.invalidState(
                "MINICPM_TRACE_TOKEN_COUNT must be an integer, got \(valueText)")
        }
        return try traceTokenIDs(for: tokenCount)
    }

    private static func traceTokenIDs(for tokenCount: Int) throws -> [Int] {
        guard tokenCount >= 1, tokenCount <= modelTraceTokenIDs.count else {
            throw MiniCPMLLMError.invalidState(
                "MINICPM_TRACE_TOKEN_COUNT must be in [1, \(modelTraceTokenIDs.count)]")
        }
        return Array(modelTraceTokenIDs.prefix(tokenCount))
    }

    private static func traceCachedPrefixCount(
        for text: String?, tokenCount: Int = modelTraceTokenIDs.count
    ) throws -> Int {
        let valueText = text ?? "0"
        guard let value = Int(valueText), value >= 0,
              value < tokenCount else {
            throw MiniCPMLLMError.invalidState(
                "MINICPM_TRACE_CACHED_PREFIX_COUNT must be in "
                    + "[0, \(tokenCount - 1)]")
        }
        return value
    }

    private static func traceAllLayersEnabled(for text: String?) -> Bool {
        text == "1"
    }

    func testTraceTokenCountSelection() throws {
        XCTAssertEqual(try Self.traceTokenIDs(for: 4), [151643, 198, 108, 151645])
        XCTAssertEqual(try Self.traceTokenIDs(for: 1), [151643])
        XCTAssertThrowsError(try Self.traceTokenIDs(for: 0))
        XCTAssertThrowsError(try Self.traceTokenIDs(for: 5))
    }

    func testTraceTokenIDsSelectionAndPrecedence() throws {
        XCTAssertEqual(
            try Self.parseTraceTokenIDs(for: "151643, 151645,198"),
            [151643, 151645, 198])
        XCTAssertEqual(
            try Self.selectedTraceTokenIDs(
                tokenCountText: "0", tokenIDsText: "151643,151645,198"),
            [151643, 151645, 198])
        XCTAssertEqual(
            try Self.selectedTraceTokenIDs(tokenCountText: "1", tokenIDsText: nil),
            [151643])
        XCTAssertNil(try Self.parseTraceTokenIDs(for: nil))
    }

    func testTraceTokenIDsRejectEmptyOrInvalidValues() throws {
        for value in ["", ",", "151643,", ",151643", "not-an-int", "-1"] {
            XCTAssertThrowsError(try Self.parseTraceTokenIDs(for: value), "value=\(value)")
        }
    }

    func testTraceCachedPrefixCountSelection() throws {
        XCTAssertEqual(try Self.traceCachedPrefixCount(for: nil), 0)
        XCTAssertEqual(try Self.traceCachedPrefixCount(for: "0"), 0)
        XCTAssertEqual(try Self.traceCachedPrefixCount(for: "1"), 1)
        XCTAssertEqual(try Self.traceCachedPrefixCount(for: "3"), 3)
        XCTAssertThrowsError(try Self.traceCachedPrefixCount(for: "-1"))
        XCTAssertThrowsError(try Self.traceCachedPrefixCount(for: "4"))
        XCTAssertThrowsError(try Self.traceCachedPrefixCount(for: "not-an-int"))
        XCTAssertEqual(
            try Self.traceCachedPrefixCount(for: "2", tokenCount: 3), 2)
        XCTAssertThrowsError(try Self.traceCachedPrefixCount(for: "3", tokenCount: 3))
    }

    func testTraceAllLayersFlagSelection() {
        XCTAssertTrue(Self.traceAllLayersEnabled(for: "1"))
        XCTAssertFalse(Self.traceAllLayersEnabled(for: nil))
        XCTAssertFalse(Self.traceAllLayersEnabled(for: "0"))
    }

    /// Keep opt-in traces on the same dense-BF16 prefill route as production.
    /// The helper deliberately mirrors MiniCPMAttention's guards so Float32
    /// tiny fixtures, packed bundles, and any unequal-length/decode probe stay
    /// on the established fused MLX implementation.
    private func traceAttention(
        config: MiniCPMMLXConfig,
        qHeads: MLXArray,
        kHeads: MLXArray,
        vHeads: MLXArray,
        scale: Float
    ) -> MLXArray {
        let length = qHeads.dim(2)
        let canUseExact = config.quantBits == 0
            && ProcessInfo.processInfo.environment["MINICPM_MLX_BF16_SDPA"] != "1"
            && length > 1
            && qHeads.dtype == .bfloat16
            && kHeads.dtype == .bfloat16
            && vHeads.dtype == .bfloat16
            && qHeads.dim(2) == kHeads.dim(2)
            && kHeads.dim(2) == vHeads.dim(2)
        if canUseExact {
            return MiniCPMExactBF16SDPA.attend(
                query: qHeads,
                key: kHeads,
                value: vHeads,
                scale: scale).reshaped(
                    qHeads.dim(0), length, qHeads.dim(1) * qHeads.dim(3))
        }
        let mask: MLXFast.ScaledDotProductAttentionMaskMode = length <= 1
            ? .none
            : .causal
        return SDPA.attendAndMerge(
            qHeads: qHeads,
            kHeads: kHeads,
            vHeads: vHeads,
            scale: scale,
            mask: mask)
    }

    /// Tiny, model-free guard for the dense BF16 rotary contract.  The
    /// expected inverse-frequency values exercise the BF16 storage cast, the
    /// known position-one output exercises FP32 frequency math followed by
    /// BF16 cos/sin, and the offset-five check catches accidental offset
    /// resets in incremental decoding.
    func testExactBF16RoPESemantics() {
        let inverse = MiniCPMExactBF16RoPE.inverseFrequency(
            headDim: 8, ropeTheta: 1_000_000)
        eval(inverse)
        XCTAssertEqual(inverse.dtype, .bfloat16)
        XCTAssertEqual(
            inverse.asType(.float32).asArray(Float.self),
            [1.0, 0.03173828125, 0.00099945068359375, 0.000031709671020507812])

        let input = MLXArray([
            Float(1), 2, 3, 4, 5, 6, 7, 8,
            Float(1), 2, 3, 4, 5, 6, 7, 8,
        ], [1, 1, 2, 8]).asType(.bfloat16)
        let positionOne = MiniCPMExactBF16RoPE.apply(
            input, inverseFrequency: inverse, offset: 0)
        let offsetFive = MiniCPMExactBF16RoPE.apply(
            input, inverseFrequency: inverse, offset: 5)
        eval(positionOne, offsetFive)

        let first = positionOne[0, 0, 0].asType(.float32).asArray(Float.self)
        XCTAssertEqual(first, [1, 2, 3, 4, 5, 6, 7, 8])
        let second = positionOne[0, 0, 1].asType(.float32).asArray(Float.self)
        XCTAssertEqual(second, [-3.65625, 1.8125, 3, 4, 3.53125, 6.0625, 7, 8])
        let shifted = offsetFive[0, 0, 0].asType(.float32).asArray(Float.self)
        XCTAssertEqual(shifted, [5.0625, 1.03125, 2.96875, 4, 0.45703125, 6.25, 7, 8])
    }

    func testWriteFixtureTrace() throws {
        if let modelPath = ProcessInfo.processInfo.environment["MINICPM_TRACE_MODEL"] {
            try writeModelTrace(modelPath: modelPath)
            return
        }
        guard let outputPath = ProcessInfo.processInfo.environment["MINICPM_WRITE_TRACE"] else {
            return
        }
        let fixtures = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().appendingPathComponent("Fixtures")
        let config = try MiniCPMMLXConfig.load(from: fixtures)
        let model = MiniCPMLanguageModel(config: config)
        try MiniCPMWeightLoader.load(model: model, from: fixtures)
        let ids = MLXArray([Int32(1), Int32(2)]).expandedDimensions(axis: 0)
        let x = model.embed(ids)
        let layer = model.layers[0]
        let attention = layer.selfAttention
        var trace: [String: [Float]] = [:]

        func record(_ name: String, _ array: MLXArray) {
            eval(array)
            trace[name] = array.asType(.float32).asArray(Float.self)
        }
        record("embed", x)
        let inputNorm = layer.inputLayerNorm(x)
        record("input_norm", inputNorm)
        let qProjection = attention.qProj(inputNorm)
        let kProjection = attention.kProj(inputNorm)
        let vProjection = attention.vProj(inputNorm)
        record("q_proj", qProjection)
        record("k_proj", kProjection)
        record("v_proj", vProjection)
        let b = x.dim(0), t = x.dim(1)
        var q = attention.qNorm(qProjection.reshaped(b, t, attention.numQHeads, attention.headDim))
        var k = attention.kNorm(kProjection.reshaped(b, t, attention.numKVHeads, attention.headDim))
        var v = vProjection.reshaped(b, t, attention.numKVHeads, attention.headDim)
        q = q.transposed(0, 2, 1, 3)
        k = k.transposed(0, 2, 1, 3)
        v = v.transposed(0, 2, 1, 3)
        record("q_norm_reshaped", q)
        record("k_norm_reshaped", k)
        record("v_reshaped", v)
        q = attention.rope(q, offset: 0)
        k = attention.rope(k, offset: 0)
        record("q_rope", q)
        record("k_rope", k)
        let attn = traceAttention(
            config: config, qHeads: q, kHeads: k, vHeads: v, scale: attention.scale)
        record("sdpa_merge", attn)
        let outputProjection = attention.oProj(attn)
        record("o_proj", outputProjection)
        let residual = x + outputProjection
        record("residual", residual)
        let postNorm = layer.postAttentionLayerNorm(residual)
        record("post_norm", postNorm)
        let gate = layer.mlp.gateProj(postNorm)
        let up = layer.mlp.upProj(postNorm)
        record("gate", gate)
        record("up", up)
        let activatedGate = config.quantBits == 0
            ? MiniCPMExactBF16SiLU.apply(gate)
            : silu(gate)
        let down = layer.mlp.downProj(activatedGate * up)
        record("down", down)
        let output = residual + down
        record("layer_out", output)
        let finalNorm = model.norm(output)
        record("final_norm", finalNorm)
        let logits = model.lmHead!(finalNorm)
        record("logits", logits)

        let data = try JSONSerialization.data(withJSONObject: trace, options: [.sortedKeys])
        try data.write(to: URL(fileURLWithPath: outputPath))
    }

    /// Opt-in production-bundle diagnostic for two decode-only failure modes:
    /// (1) a cache write that does not preserve the post-RoPE K/V bytes and
    /// (2) a shape-specific MPSGraph plan that changes after a T=4 prefill.
    /// Loading the real bundle is intentionally skipped unless the caller
    /// supplies MINICPM_TRACE_PRODUCTION_MODEL.
    func testProductionModelKVAndPlanPersistenceDiagnostics() throws {
        let environment = ProcessInfo.processInfo.environment
        guard let modelPath = environment["MINICPM_TRACE_PRODUCTION_MODEL"],
              !modelPath.isEmpty else {
            throw XCTSkip("set MINICPM_TRACE_PRODUCTION_MODEL to enable the production diagnostic")
        }
        let modelURL = URL(fileURLWithPath: modelPath, isDirectory: true)
        guard FileManager.default.fileExists(atPath: modelURL.path) else {
            throw XCTSkip("production model bundle is missing at \(modelURL.path)")
        }

        let config = try MiniCPMMLXConfig.load(from: modelURL)
        let model = MiniCPMLanguageModel(config: config)
        try MiniCPMWeightLoader.load(model: model, from: modelURL)
        guard let layer = model.layers.first else {
            throw XCTSkip("production model has no transformer layers")
        }

        let tokenID: Int32 = 151643
        let t1IDs = MLXArray([tokenID]).expandedDimensions(axis: 0)
        let attention = layer.selfAttention
        let input = model.embed(t1IDs).asType(.bfloat16)
        let inputNorm = layer.inputLayerNorm(input)
        var manualK = attention.kNorm(
            attention.kProj(inputNorm).reshaped(
                1, 1, attention.numKVHeads, attention.headDim))
        var manualV = attention.vProj(inputNorm).reshaped(
            1, 1, attention.numKVHeads, attention.headDim)
        manualK = manualK.transposed(0, 2, 1, 3)
        manualV = manualV.transposed(0, 2, 1, 3)
        if config.quantBits == 0 && environment["MINICPM_FUSED_ROPE"] != "1" {
            let inverse = MiniCPMExactBF16RoPE.inverseFrequency(
                headDim: attention.headDim, ropeTheta: config.ropeTheta)
            manualK = MiniCPMExactBF16RoPE.apply(
                manualK, inverseFrequency: inverse, offset: 0)
        } else {
            manualK = attention.rope(manualK, offset: 0)
        }

        let fresh = model.forward(inputIds: t1IDs, state: model.initialState())
        eval(manualK, manualV, fresh.hidden, fresh.logits)
        guard let firstCache = fresh.state.kvCaches.first,
              let cache = firstCache else {
            throw XCTSkip("production forward did not produce a layer-0 KV cache")
        }
        reportProductionTensorComparison(
            "layer0_post_rope_k", expected: manualK, actual: cache.0)
        reportProductionTensorComparison(
            "layer0_v", expected: manualV, actual: cache.1)

        // Force a distinct prefill shape and wait for both outputs before the
        // second T=1 call.  Any changed T=1 bytes then indicate persistent
        // plan state rather than an unevaluated/lazy T=4 graph.
        let t4IDs = MLXArray(Array(repeating: tokenID, count: 4), [1, 4])
        let prefill = model.forward(inputIds: t4IDs, state: model.initialState())
        eval(prefill.hidden, prefill.logits)
        let afterPrefill = model.forward(inputIds: t1IDs, state: model.initialState())
        eval(afterPrefill.hidden, afterPrefill.logits)
        reportProductionTensorComparison(
            "t1_hidden_after_t4", expected: fresh.hidden, actual: afterPrefill.hidden)
        reportProductionTensorComparison(
            "t1_logits_after_t4", expected: fresh.logits, actual: afterPrefill.logits)
    }

    private func reportProductionTensorComparison(
        _ label: String,
        expected: MLXArray,
        actual: MLXArray
    ) {
        eval(expected, actual)
        let expectedValues = expected.asType(.float32).asArray(Float.self)
        let actualValues = actual.asType(.float32).asArray(Float.self)
        let expectedStrides = expected.asData(access: .noCopy).strides
        let actualStrides = actual.asData(access: .noCopy).strides
        let sharedCount = min(expectedValues.count, actualValues.count)
        var mismatchCount = abs(expectedValues.count - actualValues.count)
        var firstMismatch: Int?
        for index in 0 ..< sharedCount {
            if expectedValues[index].bitPattern != actualValues[index].bitPattern {
                mismatchCount += 1
                if firstMismatch == nil { firstMismatch = index }
            }
        }
        let first: String
        if let index = firstMismatch {
            first = "index=\(index) expected=\(expectedValues[index]) actual=\(actualValues[index])"
        } else if expectedValues.count != actualValues.count {
            first = "count expected=\(expectedValues.count) actual=\(actualValues.count)"
        } else {
            first = "none"
        }
        print(
            "MINICPM_TRACE_PRODUCTION \(label) "
                + "expected_shape=\(expected.shape) actual_shape=\(actual.shape) "
                + "expected_stride=\(expectedStrides) actual_stride=\(actualStrides) "
                + "neq=\(mismatchCount) first_mismatch=\(first)")
    }

    /// Inject the official layer-0 BF16 intermediates one operation at a time.
    ///
    /// The ordinary trace above is useful for finding the first drift in a
    /// complete forward pass, but it cannot tell whether a mismatch is caused
    /// by an upstream tensor or by the operation consuming it.  This probe
    /// restores every official ``.f32`` tensor to BF16 before passing it to
    /// the next Swift operator.  Consequently each reported comparison has
    /// exactly the same BF16 input on both sides of the operation boundary.
    /// It is deliberately opt-in: loading the real dense bundle consumes
    /// roughly 15 GB and this test never calls the full model forward path.
    func testOracleInjectionDenseBF16() throws {
        guard ProcessInfo.processInfo.environment["MINICPM_ORACLE_INJECTION"] == "1" else {
            return
        }

        let environment = ProcessInfo.processInfo.environment
        guard let modelPath = environment["MINICPM_ORACLE_INJECTION_MODEL"],
              !modelPath.isEmpty,
              let tracePath = environment["MINICPM_ORACLE_INJECTION_TRACE_DIR"],
              !tracePath.isEmpty else {
            return
        }
        let modelURL = URL(fileURLWithPath: modelPath, isDirectory: true)
        let oracleURL = URL(fileURLWithPath: tracePath, isDirectory: true)
        let outputURL: URL
        if let outputPath = environment["MINICPM_ORACLE_INJECTION_OUT_DIR"],
           !outputPath.isEmpty {
            outputURL = URL(fileURLWithPath: outputPath, isDirectory: true)
        } else {
            outputURL = FileManager.default.temporaryDirectory.appendingPathComponent(
                "minicpm-swift-oracle-injection-\(UUID().uuidString)", isDirectory: true)
        }

        guard FileManager.default.fileExists(atPath: modelURL.path) else {
            throw XCTSkip("dense BF16 bundle is missing at \(modelURL.path)")
        }
        guard FileManager.default.fileExists(atPath: oracleURL.path) else {
            throw XCTSkip("official layer-0 trace is missing at \(oracleURL.path)")
        }
        try FileManager.default.createDirectory(
            at: outputURL, withIntermediateDirectories: true)

        // The current exporter prefixes layer-0 tensors with `layer_00_`,
        // while older injection fixtures used bare operation names.  Accept
        // both layouts so this opt-in probe remains useful across traces.
        func tensorURL(_ name: String) -> URL? {
            let aliases: [String]
            switch name {
            case "q_norm_trans": aliases = [name, "q"]
            case "k_norm_trans": aliases = [name, "k"]
            case "residual": aliases = [name, "attn_residual"]
            default: aliases = [name]
            }
            let candidates = aliases.flatMap { alias in
                [
                    oracleURL.appendingPathComponent("\(alias).f32"),
                    oracleURL.appendingPathComponent("layer_00_\(alias).f32"),
                ]
            }
            return candidates.first {
                FileManager.default.fileExists(atPath: $0.path)
            }
        }

        // The single model construction/load is intentional.  All operations
        // below use this one layer and never invoke MiniCPMLanguageModel.forward.
        let config = try MiniCPMMLXConfig.load(from: modelURL)
        guard config.quantBits == 0 else {
            throw XCTSkip("oracle injection requires the dense BF16 bundle")
        }
        let model = MiniCPMLanguageModel(config: config)
        try MiniCPMWeightLoader.load(model: model, from: modelURL)
        guard let layer = model.layers.first else {
            throw XCTSkip("dense bundle has no transformer layers")
        }
        let attention = layer.selfAttention
        let batch = 1
        let length = 4
        let qWidth = attention.numQHeads * attention.headDim
        let kvWidth = attention.numKVHeads * attention.headDim

        func oracle(_ name: String, _ shape: [Int]) throws -> MLXArray {
            let url = tensorURL(name) ?? oracleURL.appendingPathComponent("\(name).f32")
            let data: Data
            do {
                data = try Data(contentsOf: url)
            } catch {
                throw NSError(
                    domain: "MiniCPMOracleInjection",
                    code: 1,
                    userInfo: [NSLocalizedDescriptionKey:
                        "missing official tensor \(url.path): \(error)"])
            }
            guard data.count % MemoryLayout<Float>.stride == 0 else {
                throw NSError(
                    domain: "MiniCPMOracleInjection",
                    code: 2,
                    userInfo: [NSLocalizedDescriptionKey:
                        "official tensor \(url.path) is not float32"])
            }
            let values = data.withUnsafeBytes { raw in
                Array(raw.bindMemory(to: Float.self))
            }
            let expectedCount = shape.reduce(1, *)
            guard values.count == expectedCount else {
                throw NSError(
                    domain: "MiniCPMOracleInjection",
                    code: 3,
                    userInfo: [NSLocalizedDescriptionKey:
                        "official tensor \(name) has \(values.count) values, expected "
                            + "\(expectedCount) for shape \(shape)"])
            }
            // The Python exporter writes values after every op in BF16.  Cast
            // back to BF16 before injection so no operation receives a hidden
            // FP32 path merely because the fixture is serialized as .f32.
            return MLXArray(values, shape).asType(.bfloat16)
        }

        func optionalOracle(_ name: String, _ shape: [Int]) throws -> MLXArray? {
            guard tensorURL(name) != nil else { return nil }
            return try oracle(name, shape)
        }

        func values(_ array: MLXArray) -> [Float] {
            eval(array)
            return array.asType(.float32).asArray(Float.self)
        }

        /// Diagnostic-only dense Linear candidate.  MLX's regular Linear is
        /// intentionally left untouched; this path promotes both BF16 input
        /// and BF16 weight to FP32 for the matmul, then stores the result back
        /// as BF16 like the official dense checkpoint's activation tensors.
        func denseLinearFP32Accum(_ input: MLXArray, _ linear: Linear) -> MLXArray {
            let inputFP32 = input.asType(.float32)
            let weightFP32 = linear.weight.asType(.float32)
            return matmul(inputFP32, weightFP32.transposed()).asType(.bfloat16)
        }

        func write(_ name: String, _ values: [Float]) throws {
            let url = outputURL.appendingPathComponent("\(name).f32")
            let data = values.withUnsafeBufferPointer { Data(buffer: $0) }
            try data.write(to: url)
        }

        @discardableResult
        func report(
            _ name: String,
            actual: [Float],
            reference: String?
        ) throws -> (maxAbs: Double, meanAbs: Double, rms: Double, neq: Int)? {
            try write(name, actual)
            guard let reference else {
                print("ORACLE_INJECTION \(name): wrote \(actual.count) values (no official reference)")
                return nil
            }
            guard tensorURL(reference) != nil else {
                print("ORACLE_INJECTION \(name): wrote \(actual.count) values (no official reference)")
                return nil
            }
            let url = tensorURL(reference)!
            let data = try Data(contentsOf: url)
            guard data.count % MemoryLayout<Float>.stride == 0 else {
                throw NSError(
                    domain: "MiniCPMOracleInjection",
                    code: 4,
                    userInfo: [NSLocalizedDescriptionKey:
                        "official reference \(url.path) is not float32"])
            }
            let expected = data.withUnsafeBytes { raw in
                Array(raw.bindMemory(to: Float.self))
            }
            guard expected.count == actual.count else {
                throw NSError(
                    domain: "MiniCPMOracleInjection",
                    code: 5,
                    userInfo: [NSLocalizedDescriptionKey:
                        "\(name) has \(actual.count) values, reference has "
                            + "\(expected.count)"])
            }
            var maxAbs = 0.0
            var sumAbs = 0.0
            var sumSq = 0.0
            var neq = 0
            for (a, e) in zip(actual, expected) {
                let delta = abs(Double(a) - Double(e))
                maxAbs = max(maxAbs, delta)
                sumAbs += delta
                sumSq += delta * delta
                if a != e { neq += 1 }
            }
            let count = Double(actual.count)
            let meanAbs = count == 0 ? 0 : sumAbs / count
            let rms = count == 0 ? 0 : sqrt(sumSq / count)
            print(String(format:
                "ORACLE_INJECTION %@: max=%.9g mean=%.9g rms=%.9g neq=%d/%d",
                name, maxAbs, meanAbs, rms, neq, actual.count))
            return (maxAbs, meanAbs, rms, neq)
        }

        let oracleQProjection = try oracle("q_proj", [batch, length, qWidth])
        let kProjection = try oracle("k_proj", [batch, length, kvWidth])
        let vProjection = try oracle("v_proj", [batch, length, kvWidth])

        // Compare the current BF16 Linear and the FP32-accumulate candidate
        // from the same official input_norm tensor.  The official q_proj
        // itself remains the injected input for q/k norm below.
        let oracleInputNorm = try oracle("input_norm", [batch, length, config.hiddenSize])
        let qProjectionCurrent = attention.qProj(oracleInputNorm)
        try report("q_proj", actual: values(qProjectionCurrent), reference: "q_proj")
        let qProjectionFP32 = denseLinearFP32Accum(oracleInputNorm, attention.qProj)
        try report("q_proj_fp32acc", actual: values(qProjectionFP32), reference: "q_proj")

        // q/k norm: inject the serialized projection rather than Swift's
        // projection, then compare the normalized head-shaped tensors.
        let qNorm = attention.qNorm(oracleQProjection.reshaped(
            batch, length, attention.numQHeads, attention.headDim))
        let kNorm = attention.kNorm(kProjection.reshaped(
            batch, length, attention.numKVHeads, attention.headDim))
        try report("q_norm", actual: values(qNorm), reference: "q_norm")
        try report("k_norm", actual: values(kNorm), reference: "k_norm")

        // Exact BF16 RoPE receives the official transposed q/k norm tensors.
        // (The fused RoPE implementation is intentionally not used here.)
        let qNormTrans = try oracle(
            "q_norm_trans", [batch, attention.numQHeads, length, attention.headDim])
        let kNormTrans = try oracle(
            "k_norm_trans", [batch, attention.numKVHeads, length, attention.headDim])
        let inverse = MiniCPMExactBF16RoPE.inverseFrequency(
            headDim: attention.headDim, ropeTheta: config.ropeTheta)
        let qRope = MiniCPMExactBF16RoPE.apply(
            qNormTrans, inverseFrequency: inverse, offset: 0)
        let kRope = MiniCPMExactBF16RoPE.apply(
            kNormTrans, inverseFrequency: inverse, offset: 0)
        try report("q_rope", actual: values(qRope), reference: "q_rope")
        try report("k_rope", actual: values(kRope), reference: "k_rope")

        // Feed official q/k RoPE and v into the fused SDPA kernel.  The
        // exporter records the merged tensor as [B,T,H,D], while the Swift
        // helper returns [B,T,H*D], so only a view reshape is needed.
        let oracleQRope = try oracle(
            "q_rope", [batch, attention.numQHeads, length, attention.headDim])
        let oracleKRope = try oracle(
            "k_rope", [batch, attention.numKVHeads, length, attention.headDim])
        let vHeads = vProjection.reshaped(
            batch, length, attention.numKVHeads, attention.headDim)
            .transposed(0, 2, 1, 3)
        let attentionMerged = traceAttention(
            config: config,
            qHeads: oracleQRope,
            kHeads: oracleKRope,
            vHeads: vHeads,
            scale: attention.scale)
        let attentionMergedForTrace = attentionMerged.reshaped(
            batch, length, attention.numQHeads, attention.headDim)
        try report("sdpa_merge", actual: values(attentionMergedForTrace), reference: "sdpa_merge")

        // o_proj receives the official SDPA output, not the Swift SDPA
        // output above.  This isolates the projection kernel and its weights.
        let oracleSDPA = try oracle(
            "sdpa_merge", [batch, length, attention.numQHeads, attention.headDim])
        let oProjection = attention.oProj(oracleSDPA.reshaped(
            batch, length, attention.numQHeads * attention.headDim))
        try report("o_proj", actual: values(oProjection), reference: "o_proj")
        let oProjectionFP32 = denseLinearFP32Accum(
            oracleSDPA.reshaped(batch, length, attention.numQHeads * attention.headDim),
            attention.oProj)
        try report("o_proj_fp32acc", actual: values(oProjectionFP32), reference: "o_proj")

        // Post-attention RMSNorm receives the official residual, then both
        // MLP projections independently receive the official post-norm input.
        let oracleResidual = try oracle("residual", [batch, length, config.hiddenSize])
        let postNorm = layer.postAttentionLayerNorm(oracleResidual)
        try report("post_norm", actual: values(postNorm), reference: "post_norm")

        let oraclePostNorm = try oracle("post_norm", [batch, length, config.hiddenSize])
        let gate = layer.mlp.gateProj(oraclePostNorm)
        let up = layer.mlp.upProj(oraclePostNorm)
        let gateValues = values(gate)
        let upValues = values(up)
        try report("gate", actual: gateValues, reference: "gate")
        try report("up", actual: upValues, reference: "up")
        let gateFP32 = denseLinearFP32Accum(oraclePostNorm, layer.mlp.gateProj)
        let upFP32 = denseLinearFP32Accum(oraclePostNorm, layer.mlp.upProj)
        try report("gate_fp32acc", actual: values(gateFP32), reference: "gate")
        try report("up_fp32acc", actual: values(upFP32), reference: "up")

        // Prefer official silu/mul fixtures when present.  Existing traces
        // predate those two files, so the fallback computes each from the
        // restored BF16 gate/up tensors and still writes both diagnostics.
        let gateShape = [batch, length, config.intermediateSize]
        let oracleGate = try oracle("gate", gateShape)
        let oracleUp = try oracle("up", gateShape)
        let siluGate = MiniCPMExactBF16SiLU.apply(oracleGate)
        let siluValues = values(siluGate)
        try report("silu_gate", actual: siluValues, reference: "silu_gate")
        let siluForMul = try optionalOracle("silu_gate", gateShape) ?? siluGate
        let mul = siluForMul * oracleUp
        let mulValues = values(mul)
        try report("mul", actual: mulValues, reference: "mul")
        let mulForDown = try optionalOracle("mul", gateShape) ?? mul
        let down = layer.mlp.downProj(mulForDown)
        let downValues = values(down)
        let downMetrics = try report("down", actual: downValues, reference: "down")
        let downFP32 = denseLinearFP32Accum(mulForDown, layer.mlp.downProj)
        try report("down_fp32acc", actual: values(downFP32), reference: "down")

        // Probe whether a particular reduction order is responsible for the
        // remaining down-projection drift.  Each chunk performs FP32 partial
        // matmul and accumulates its result in a Swift Float32 buffer in
        // strictly increasing K order, then stores the final tensor as BF16.
        // Partials are evaluated before the next chunk, so no chain of lazy
        // MLX graphs is retained across candidates.
        func downChunkFP32Accum(_ input: MLXArray, chunkK: Int) -> MLXArray {
            let inputFP32 = input.asType(.float32)
            let weightFP32 = layer.mlp.downProj.weight.asType(.float32)
            let outputWidth = layer.mlp.downProj.weight.dim(0)
            let inputWidth = layer.mlp.downProj.weight.dim(1)
            var accumulated = [Float](
                repeating: 0, count: batch * length * outputWidth)
            var start = 0
            while start < inputWidth {
                let end = min(start + chunkK, inputWidth)
                let partial = matmul(
                    inputFP32[0..., 0..., start ..< end],
                    weightFP32[0..., start ..< end].transposed())
                eval(partial)
                let values = partial.asType(.float32).asArray(Float.self)
                for index in accumulated.indices {
                    accumulated[index] += values[index]
                }
                start = end
            }
            return MLXArray(accumulated, [batch, length, outputWidth]).asType(.bfloat16)
        }

        let oracleDown = try oracle("down", [batch, length, config.hiddenSize])
        let oracleDownValues = values(oracleDown)
        for chunkK in [32, 64, 128, 256, 512, 1024, 2048, 4096] {
            let candidate = downChunkFP32Accum(mulForDown, chunkK: chunkK)
            let candidateValues = values(candidate)
            let name = "down_fp32acc_chunk\(chunkK)"
            let candidateMetrics = try report(
                name, actual: candidateValues, reference: "down")
            if let baseline = downMetrics,
               let candidateMetrics,
               candidateMetrics.rms < baseline.rms,
               candidateValues.count > 2276,
               oracleDownValues.count > 2276 {
                print(String(format:
                    "ORACLE_INJECTION %@: idx2276_actual=%.9g idx140_actual=%.9g "
                        + "reference_idx2276=%.9g reference_idx140=%.9g",
                    name,
                    candidateValues[2276], candidateValues[140],
                    oracleDownValues[2276], oracleDownValues[140]))
            }
        }
        print("ORACLE_INJECTION_OUT \(outputURL.path)")
    }

    /// Print compact layer-0 intermediates for a real bundle.  This opt-in
    /// path intentionally avoids writing a huge tensor fixture; matching
    /// shape/sum/sum-of-squares/edge values is enough to identify the first
    /// divergent operation against the Python oracle.
    private func writeModelTrace(modelPath: String) throws {
        let environment = ProcessInfo.processInfo.environment
        let tokenIDs = try Self.selectedTraceTokenIDs(
            tokenCountText: environment["MINICPM_TRACE_TOKEN_COUNT"],
            tokenIDsText: environment["MINICPM_TRACE_TOKEN_IDS"])
        let cachedPrefixCount = try Self.traceCachedPrefixCount(
            for: environment["MINICPM_TRACE_CACHED_PREFIX_COUNT"],
            tokenCount: tokenIDs.count)
        let modelURL = URL(fileURLWithPath: modelPath)
        let config = try MiniCPMMLXConfig.load(from: modelURL)
        let model = MiniCPMLanguageModel(config: config)
        try MiniCPMWeightLoader.load(model: model, from: modelURL)
        func applySiLU(_ input: MLXArray) -> MLXArray {
            config.quantBits == 0 ? MiniCPMExactBF16SiLU.apply(input) : silu(input)
        }
        let ids = MLXArray(tokenIDs.map { Int32($0) })
            .expandedDimensions(axis: 0)
        let x = model.embed(ids).asType(.bfloat16)
        let layer = model.layers[0]
        let attention = layer.selfAttention
        let binaryDirectory = environment["MINICPM_TRACE_BIN_DIR"].map {
            URL(fileURLWithPath: $0, isDirectory: true)
        }
        let traceAllLayers = Self.traceAllLayersEnabled(
            for: environment["MINICPM_TRACE_ALL_LAYERS"])
        if let binaryDirectory {
            try? FileManager.default.createDirectory(
                at: binaryDirectory, withIntermediateDirectories: true)
            let currentTokenIDs = cachedPrefixCount > 0
                ? [tokenIDs[cachedPrefixCount]] : tokenIDs
            let provenance: [String: Any] = [
                "mode": cachedPrefixCount > 0 ? "cached_decode" : "prefill",
                "prefix_count": cachedPrefixCount,
                "current_token_ids": currentTokenIDs,
                "current_length": currentTokenIDs.count,
                "input_length": tokenIDs.count,
                "token_ids": tokenIDs,
            ]
            if let data = try? JSONSerialization.data(
                withJSONObject: provenance, options: [.sortedKeys]) {
                try? data.write(to: binaryDirectory.appendingPathComponent("provenance.json"))
            }
        }
        let provenanceMode = cachedPrefixCount > 0 ? "cached_decode" : "prefill"
        let provenanceCurrentIDs = cachedPrefixCount > 0
            ? [tokenIDs[cachedPrefixCount]] : tokenIDs
        print(
            "TRACE provenance mode=\(provenanceMode) prefix_count=\(cachedPrefixCount) "
                + "token_ids=\(tokenIDs) current_token_ids=\(provenanceCurrentIDs) "
                + "current_length=\(provenanceCurrentIDs.count)")

        func summary(_ name: String, _ array: MLXArray) {
            eval(array)
            let values = array.asType(.float32).asArray(Float.self)
            if let binaryDirectory {
                let data = values.withUnsafeBufferPointer { Data(buffer: $0) }
                try? data.write(to: binaryDirectory.appendingPathComponent("\(name).f32"))
            }
            let sum = values.reduce(0.0) { $0 + Double($1) }
            let sumSq = values.reduce(0.0) { $0 + Double($1) * Double($1) }
            let first = Array(values.prefix(8))
            let last = Array(values.suffix(8))
            print("TRACE \(name) shape=\(array.shape) sum=\(sum) sumsq=\(sumSq) first8=\(first) last8=\(last)")
            if values.count == 32 * 4 * 128 || values.count == 8 * 4 * 128 {
                let heads = values.count == 32 * 4 * 128 ? 32 : 8
                let positions = (0..<4).map { position in
                    (0..<heads).reduce(0.0) { partial, head in
                        let start = (head * 4 + position) * 128
                        return partial + values[start..<(start + 128)].reduce(0.0) { $0 + Double($1) }
                    }
                }
                print("TRACE \(name) position_sums=\(positions)")
            }
        }

        func summaryAndReturn(_ name: String, _ array: MLXArray) -> MLXArray {
            summary(name, array)
            return array
        }

        let useExactRoPE = ProcessInfo.processInfo.environment["MINICPM_TRACE_CUSTOM_ROPE"] == "1"
            || (config.quantBits == 0
                && ProcessInfo.processInfo.environment["MINICPM_FUSED_ROPE"] != "1")
        let requestedLayer = Int(
            ProcessInfo.processInfo.environment["MINICPM_TRACE_LAYER"] ?? "0") ?? 0
        guard requestedLayer >= 0, requestedLayer < model.layers.count else {
            throw MiniCPMLLMError.invalidState(
                "MINICPM_TRACE_LAYER must be in [0, \(model.layers.count - 1)]")
        }

        if cachedPrefixCount > 0 {
            // Build every layer's prefix KV through the real production
            // forward path once.  The successor token is then replayed by
            // hand so the target layer can expose its prefix and post-append
            // caches without running the full logits/E2E path.
            let prefixIDs = MLXArray(
                tokenIDs.prefix(cachedPrefixCount).map { Int32($0) })
                .expandedDimensions(axis: 0)
            let currentIDs = MLXArray([Int32(tokenIDs[cachedPrefixCount])])
                .expandedDimensions(axis: 0)
            let prefixState = model.forward(
                inputIds: prefixIDs, state: model.initialState()).state
            guard prefixState.sequenceLength == cachedPrefixCount else {
                throw MiniCPMLLMError.invalidState(
                    "cached prefix length was \(prefixState.sequenceLength), expected "
                        + "\(cachedPrefixCount)")
            }

            func prefixCache(at index: Int) throws -> MiniCPMKVCache {
                guard index >= 0, index < prefixState.kvCaches.count,
                      let pair = prefixState.kvCaches[index] else {
                    throw MiniCPMLLMError.invalidState(
                        "missing prefix KV cache for layer \(index)")
                }
                return MiniCPMKVCache(keys: pair.0, values: pair.1, base: 0)
            }

            func replayCachedLayer(
                _ input: MLXArray,
                layer: MiniCPMTransformerLayer,
                cache: MiniCPMKVCache
            ) -> (hidden: MLXArray, cache: MiniCPMKVCache) {
                let currentAttention = layer.selfAttention
                let inputNorm = layer.inputLayerNorm(input)
                let qProjection = currentAttention.qProj(inputNorm)
                let kProjection = currentAttention.kProj(inputNorm)
                let vProjection = currentAttention.vProj(inputNorm)
                var q = currentAttention.qNorm(qProjection.reshaped(
                    1, 1, currentAttention.numQHeads, currentAttention.headDim))
                var k = currentAttention.kNorm(kProjection.reshaped(
                    1, 1, currentAttention.numKVHeads, currentAttention.headDim))
                let v = vProjection.reshaped(
                    1, 1, currentAttention.numKVHeads, currentAttention.headDim)
                q = q.transposed(0, 2, 1, 3)
                k = k.transposed(0, 2, 1, 3)
                let vHeads = v.transposed(0, 2, 1, 3)
                let inverse = MiniCPMExactBF16RoPE.inverseFrequency(
                    headDim: currentAttention.headDim, ropeTheta: config.ropeTheta)
                if useExactRoPE {
                    q = MiniCPMExactBF16RoPE.apply(
                        q, inverseFrequency: inverse, offset: cachedPrefixCount)
                    k = MiniCPMExactBF16RoPE.apply(
                        k, inverseFrequency: inverse, offset: cachedPrefixCount)
                } else {
                    q = currentAttention.rope(q, offset: cachedPrefixCount)
                    k = currentAttention.rope(k, offset: cachedPrefixCount)
                }
                let nextCache = cache.appending(keys: k, values: vHeads)
                let (cacheKeys, cacheValues) = nextCache.window()
                let attentionOutput = SDPA.attendAndMerge(
                    qHeads: q,
                    kHeads: cacheKeys,
                    vHeads: cacheValues,
                    scale: currentAttention.scale,
                    mask: MLXFast.ScaledDotProductAttentionMaskMode.none)
                let outputProjection = currentAttention.oProj(attentionOutput)
                let residual = input + outputProjection
                let postNorm = layer.postAttentionLayerNorm(residual)
                let gate = layer.mlp.gateProj(postNorm)
                let up = layer.mlp.upProj(postNorm)
                let down = layer.mlp.downProj(applySiLU(gate) * up)
                return (hidden: residual + down, cache: nextCache)
            }

            var hidden = model.embed(currentIDs).asType(.bfloat16)
            if requestedLayer > 0 {
                for index in 0 ..< requestedLayer {
                    let cache = try prefixCache(at: index)
                    hidden = replayCachedLayer(
                        hidden, layer: model.layers[index], cache: cache).hidden
                    if traceAllLayers {
                        summary(String(format: "layer_%02d_layer_out", index), hidden)
                    }
                }
            }

            let currentLayer = model.layers[requestedLayer]
            let currentAttention = currentLayer.selfAttention
            let prefix = String(format: "layer_%02d_", requestedLayer)
            let targetPrefixCache = try prefixCache(at: requestedLayer)
            let targetPrefixWindow = targetPrefixCache.window()
            summary(prefix + "input", hidden)
            let inputNorm = summaryAndReturn(
                prefix + "input_norm", currentLayer.inputLayerNorm(hidden))
            let qProjection = summaryAndReturn(
                prefix + "q_proj", currentAttention.qProj(inputNorm))
            let kProjection = summaryAndReturn(
                prefix + "k_proj", currentAttention.kProj(inputNorm))
            let vProjection = summaryAndReturn(
                prefix + "v_proj", currentAttention.vProj(inputNorm))
            var q = currentAttention.qNorm(qProjection.reshaped(
                1, 1, currentAttention.numQHeads, currentAttention.headDim))
            var k = currentAttention.kNorm(kProjection.reshaped(
                1, 1, currentAttention.numKVHeads, currentAttention.headDim))
            var v = vProjection.reshaped(
                1, 1, currentAttention.numKVHeads, currentAttention.headDim)
            q = summaryAndReturn(prefix + "q", q.transposed(0, 2, 1, 3))
            k = summaryAndReturn(prefix + "k", k.transposed(0, 2, 1, 3))
            v = summaryAndReturn(prefix + "v", v.transposed(0, 2, 1, 3))
            let inverse = MiniCPMExactBF16RoPE.inverseFrequency(
                headDim: currentAttention.headDim, ropeTheta: config.ropeTheta)
            if useExactRoPE {
                q = summaryAndReturn(
                    prefix + "q_rope",
                    MiniCPMExactBF16RoPE.apply(
                        q, inverseFrequency: inverse, offset: cachedPrefixCount))
                k = summaryAndReturn(
                    prefix + "k_rope",
                    MiniCPMExactBF16RoPE.apply(
                        k, inverseFrequency: inverse, offset: cachedPrefixCount))
            } else {
                q = summaryAndReturn(
                    prefix + "q_rope", currentAttention.rope(q, offset: cachedPrefixCount))
                k = summaryAndReturn(
                    prefix + "k_rope", currentAttention.rope(k, offset: cachedPrefixCount))
            }
            summary(prefix + "prefix_cache_k", targetPrefixWindow.0)
            summary(prefix + "prefix_cache_v", targetPrefixWindow.1)
            let targetCache = targetPrefixCache.appending(keys: k, values: v)
            let targetWindow = targetCache.window()
            summary(prefix + "cache_k", targetWindow.0)
            summary(prefix + "cache_v", targetWindow.1)
            let attentionOutput = summaryAndReturn(
                prefix + "sdpa_merge",
                SDPA.attendAndMerge(
                    qHeads: q,
                    kHeads: targetWindow.0,
                    vHeads: targetWindow.1,
                    scale: currentAttention.scale,
                    mask: MLXFast.ScaledDotProductAttentionMaskMode.none))
            let outputProjection = summaryAndReturn(
                prefix + "o_proj", currentAttention.oProj(attentionOutput))
            let residual = summaryAndReturn(
                prefix + "attn_residual", hidden + outputProjection)
            let postNorm = summaryAndReturn(
                prefix + "post_norm", currentLayer.postAttentionLayerNorm(residual))
            let gate = summaryAndReturn(prefix + "gate", currentLayer.mlp.gateProj(postNorm))
            let up = summaryAndReturn(prefix + "up", currentLayer.mlp.upProj(postNorm))
            let siluGate = summaryAndReturn(prefix + "silu_gate", applySiLU(gate))
            let mul = summaryAndReturn(prefix + "mul", siluGate * up)
            let down = summaryAndReturn(prefix + "down", currentLayer.mlp.downProj(mul))
            hidden = summaryAndReturn(prefix + "layer_out", residual + down)
            if traceAllLayers, requestedLayer + 1 < model.layers.count {
                for index in (requestedLayer + 1) ..< model.layers.count {
                    let cache = try prefixCache(at: index)
                    hidden = replayCachedLayer(
                        hidden, layer: model.layers[index], cache: cache).hidden
                    summary(String(format: "layer_%02d_layer_out", index), hidden)
                }
            }
            return
        }

        if requestedLayer > 0 {
            // Evaluate the prefix without retaining its intermediates, then
            // record every operation in the requested layer.  This mirrors
            // the targeted Python exporter and keeps the trace small while
            // locating a sudden layer-output divergence.
            let b = x.dim(0), t = x.dim(1)
            var hidden = x
            for index in 0 ..< requestedLayer {
                let currentLayer = model.layers[index]
                let currentAttention = currentLayer.selfAttention
                let currentInputNorm = currentLayer.inputLayerNorm(hidden)
                let currentQProjection = currentAttention.qProj(currentInputNorm)
                let currentKProjection = currentAttention.kProj(currentInputNorm)
                let currentVProjection = currentAttention.vProj(currentInputNorm)
                var currentQ = currentAttention.qNorm(
                    currentQProjection.reshaped(
                        b, t, currentAttention.numQHeads, currentAttention.headDim))
                var currentK = currentAttention.kNorm(
                    currentKProjection.reshaped(
                        b, t, currentAttention.numKVHeads, currentAttention.headDim))
                let currentV = currentVProjection.reshaped(
                    b, t, currentAttention.numKVHeads, currentAttention.headDim)
                currentQ = currentQ.transposed(0, 2, 1, 3)
                currentK = currentK.transposed(0, 2, 1, 3)
                let inverse = MiniCPMExactBF16RoPE.inverseFrequency(
                    headDim: currentAttention.headDim, ropeTheta: config.ropeTheta)
                if useExactRoPE {
                    currentQ = MiniCPMExactBF16RoPE.apply(
                        currentQ, inverseFrequency: inverse, offset: 0)
                    currentK = MiniCPMExactBF16RoPE.apply(
                        currentK, inverseFrequency: inverse, offset: 0)
                } else {
                    currentQ = currentAttention.rope(currentQ, offset: 0)
                    currentK = currentAttention.rope(currentK, offset: 0)
                }
                let currentAttentionOutput = traceAttention(
                    config: config,
                    qHeads: currentQ,
                    kHeads: currentK,
                    vHeads: currentV.transposed(0, 2, 1, 3),
                    scale: currentAttention.scale)
                let currentResidual = hidden + currentAttention.oProj(currentAttentionOutput)
                let currentPostNorm = currentLayer.postAttentionLayerNorm(currentResidual)
                let currentGate = currentLayer.mlp.gateProj(currentPostNorm)
                let currentUp = currentLayer.mlp.upProj(currentPostNorm)
                let currentDown = currentLayer.mlp.downProj(
                    applySiLU(currentGate) * currentUp)
                hidden = currentResidual + currentDown
                _ = index
            }

            let currentLayer = model.layers[requestedLayer]
            let currentAttention = currentLayer.selfAttention
            let prefix = String(format: "layer_%02d_", requestedLayer)
            summary(prefix + "input", hidden)
            let inputNorm = summaryAndReturn(
                prefix + "input_norm", currentLayer.inputLayerNorm(hidden))
            let qProjection = summaryAndReturn(prefix + "q_proj", currentAttention.qProj(inputNorm))
            let kProjection = summaryAndReturn(prefix + "k_proj", currentAttention.kProj(inputNorm))
            let vProjection = summaryAndReturn(prefix + "v_proj", currentAttention.vProj(inputNorm))
            var q = currentAttention.qNorm(
                qProjection.reshaped(b, t, currentAttention.numQHeads, currentAttention.headDim))
            var k = currentAttention.kNorm(
                kProjection.reshaped(b, t, currentAttention.numKVHeads, currentAttention.headDim))
            var v = vProjection.reshaped(b, t, currentAttention.numKVHeads, currentAttention.headDim)
            q = summaryAndReturn(prefix + "q", q.transposed(0, 2, 1, 3))
            k = summaryAndReturn(prefix + "k", k.transposed(0, 2, 1, 3))
            v = summaryAndReturn(prefix + "v", v.transposed(0, 2, 1, 3))
            let inverse = MiniCPMExactBF16RoPE.inverseFrequency(
                headDim: currentAttention.headDim, ropeTheta: config.ropeTheta)
            if useExactRoPE {
                q = summaryAndReturn(
                    prefix + "q_rope",
                    MiniCPMExactBF16RoPE.apply(q, inverseFrequency: inverse, offset: 0))
                k = summaryAndReturn(
                    prefix + "k_rope",
                    MiniCPMExactBF16RoPE.apply(k, inverseFrequency: inverse, offset: 0))
            } else {
                q = summaryAndReturn(prefix + "q_rope", currentAttention.rope(q, offset: 0))
                k = summaryAndReturn(prefix + "k_rope", currentAttention.rope(k, offset: 0))
            }
            let attn = summaryAndReturn(
                prefix + "sdpa_merge",
                traceAttention(
                    config: config,
                    qHeads: q,
                    kHeads: k,
                    vHeads: v,
                    scale: currentAttention.scale))
            let outputProjection = summaryAndReturn(
                prefix + "o_proj", currentAttention.oProj(attn))
            let residual = summaryAndReturn(prefix + "attn_residual", hidden + outputProjection)
            let postNorm = summaryAndReturn(
                prefix + "post_norm", currentLayer.postAttentionLayerNorm(residual))
            let gate = summaryAndReturn(prefix + "gate", currentLayer.mlp.gateProj(postNorm))
            let up = summaryAndReturn(prefix + "up", currentLayer.mlp.upProj(postNorm))
            let siluGate = summaryAndReturn(prefix + "silu_gate", applySiLU(gate))
            let mul = summaryAndReturn(prefix + "mul", siluGate * up)
            let down = summaryAndReturn(prefix + "down", currentLayer.mlp.downProj(mul))
            _ = summaryAndReturn(prefix + "layer_out", residual + down)
            return
        }

        summary("embed", x)
        let inputNorm = layer.inputLayerNorm(x)
        summary("input_norm", inputNorm)
        let qProjection = attention.qProj(inputNorm)
        let kProjection = attention.kProj(inputNorm)
        let vProjection = attention.vProj(inputNorm)
        summary("q_proj", qProjection)
        summary("k_proj", kProjection)
        summary("v_proj", vProjection)
        let b = x.dim(0), t = x.dim(1)
        var q = attention.qNorm(qProjection.reshaped(b, t, attention.numQHeads, attention.headDim))
        var k = attention.kNorm(kProjection.reshaped(b, t, attention.numKVHeads, attention.headDim))
        var v = vProjection.reshaped(b, t, attention.numKVHeads, attention.headDim)
        summary("q_norm", q)
        summary("k_norm", k)
        q = q.transposed(0, 2, 1, 3)
        k = k.transposed(0, 2, 1, 3)
        v = v.transposed(0, 2, 1, 3)
        summary("q_norm_trans", q)
        summary("k_norm_trans", k)
        if useExactRoPE {
            func customRoPE(_ input: MLXArray, prefix: String) -> MLXArray {
                let half = attention.headDim / 2
                let positions = MLXArray((0..<input.dim(2)).map { Float($0) }, [input.dim(2), 1])
                let inverse = MiniCPMExactBF16RoPE.inverseFrequency(
                    headDim: attention.headDim, ropeTheta: config.ropeTheta)
                summary("position_ids", positions)
                summary("inv_freq", inverse)
                let freqs = matmul(positions, inverse.asType(.float32))
                summary("freqs", freqs)
                let cosBase = concatenated([cos(freqs), cos(freqs)], axis: -1)
                let sinBase = concatenated([sin(freqs), sin(freqs)], axis: -1)
                let cosFull = cosBase.asType(input.dtype)
                let sinFull = sinBase.asType(input.dtype)
                summary("cos_bf16", cosFull)
                summary("sin_bf16", sinFull)
                let cosHalf = cosFull[0..., 0 ..< half]
                    .expandedDimensions(axis: 0).expandedDimensions(axis: 0)
                let sinHalf = sinFull[0..., 0 ..< half]
                    .expandedDimensions(axis: 0).expandedDimensions(axis: 0)
                let first = input[0..., 0..., 0..., 0 ..< half]
                let second = input[0..., 0..., 0..., half ..< attention.headDim]
                let rotateHalf = concatenated([-second, first], axis: -1)
                summary("\(prefix)_rotate_half", rotateHalf)
                let qCos = input * concatenated([cosHalf, cosHalf], axis: -1)
                let qSin = rotateHalf * concatenated([sinHalf, sinHalf], axis: -1)
                summary("\(prefix)_cos", qCos)
                summary("\(prefix)_sin", qSin)
                return qCos + qSin
            }
            q = customRoPE(q, prefix: "q")
            k = customRoPE(k, prefix: "k")
        } else {
            q = attention.rope(q, offset: 0)
            k = attention.rope(k, offset: 0)
        }
        summary("q_rope", q)
        summary("k_rope", k)
        let attn = traceAttention(
            config: config, qHeads: q, kHeads: k, vHeads: v, scale: attention.scale)
        summary("sdpa_merge", attn)
        let outputProjection = attention.oProj(attn)
        summary("o_proj", outputProjection)
        let residual = x + outputProjection
        summary("residual", residual)
        let postNorm = layer.postAttentionLayerNorm(residual)
        summary("post_norm", postNorm)
        let gate = layer.mlp.gateProj(postNorm)
        let up = layer.mlp.upProj(postNorm)
        summary("gate", gate)
        summary("up", up)
        let down = layer.mlp.downProj(applySiLU(gate) * up)
        summary("down", down)
        let output = residual + down
        summary("layer_out", output)
        summary("layer_00_out", output)

        // Keep the trace compact after layer 0: one output tensor per block
        // is enough to locate the first layer where small BF16 differences
        // amplify.  The same fused SDPA path and exact/fused RoPE choice are
        // used as in production.
        var hidden = output
        for (layerIndex, currentLayer) in model.layers.dropFirst().enumerated() {
            let currentAttention = currentLayer.selfAttention
            let currentInputNorm = currentLayer.inputLayerNorm(hidden)
            let currentQProjection = currentAttention.qProj(currentInputNorm)
            let currentKProjection = currentAttention.kProj(currentInputNorm)
            let currentVProjection = currentAttention.vProj(currentInputNorm)
            var currentQ = currentAttention.qNorm(
                currentQProjection.reshaped(
                    b, t, currentAttention.numQHeads, currentAttention.headDim))
            var currentK = currentAttention.kNorm(
                currentKProjection.reshaped(
                    b, t, currentAttention.numKVHeads, currentAttention.headDim))
            let currentV = currentVProjection.reshaped(
                b, t, currentAttention.numKVHeads, currentAttention.headDim)
            currentQ = currentQ.transposed(0, 2, 1, 3)
            currentK = currentK.transposed(0, 2, 1, 3)
            let currentQOffset = 0
            if useExactRoPE {
                let inverse = MiniCPMExactBF16RoPE.inverseFrequency(
                    headDim: currentAttention.headDim, ropeTheta: config.ropeTheta)
                currentQ = MiniCPMExactBF16RoPE.apply(
                    currentQ, inverseFrequency: inverse, offset: currentQOffset)
                currentK = MiniCPMExactBF16RoPE.apply(
                    currentK, inverseFrequency: inverse, offset: currentQOffset)
            } else {
                currentQ = currentAttention.rope(currentQ, offset: currentQOffset)
                currentK = currentAttention.rope(currentK, offset: currentQOffset)
            }
            let currentAttentionOutput = traceAttention(
                config: config,
                qHeads: currentQ,
                kHeads: currentK,
                vHeads: currentV.transposed(0, 2, 1, 3),
                scale: currentAttention.scale)
            let currentResidual = hidden + currentAttention.oProj(currentAttentionOutput)
            let currentPostNorm = currentLayer.postAttentionLayerNorm(currentResidual)
            let currentGate = currentLayer.mlp.gateProj(currentPostNorm)
            let currentUp = currentLayer.mlp.upProj(currentPostNorm)
            let currentDown = currentLayer.mlp.downProj(
                applySiLU(currentGate) * currentUp)
            hidden = currentResidual + currentDown
            summary(String(format: "layer_%02d_out", layerIndex + 1), hidden)
        }
    }
}
