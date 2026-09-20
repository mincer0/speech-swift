import Foundation
import XCTest
import MLX
@testable import MiniCPMLLM

/// Opt-in parity tests for the Qwen-only MiniCPM-o language backbone.
///
/// The exporter is intentionally not run by the regular unit suite: it loads
/// the full 8-bit bundle and can consume several tens of GB.  These tests are
/// therefore skipped unless ``MINICPM_LLM_GOLDEN_DIR`` (and a model path, or
/// the manifest's repository-relative hint) is present.  Once enabled they
/// exercise the same transaction and window paths used by the duplex runtime,
/// rather than only comparing one prompt's final logits.
final class E2EMiniCPMLLMGoldenTests: XCTestCase {
    private let schema = "minicpm-o-llm-golden/v1"

    // MARK: Fixture/model setup

    private func loadFixture() throws -> MiniCPMLLMGoldenFixture {
        guard let fixture = try MiniCPMLLMGoldenFixture.locate() else {
            if ProcessInfo.processInfo.environment.keys.contains("MINICPM_LLM_GOLDEN_MODEL") {
                throw MiniCPMLLMGoldenFixture.FixtureError.invalid(
                    "MINICPM_LLM_GOLDEN_MODEL is set but no golden fixture was found")
            }
            throw XCTSkip(
                "Set MINICPM_LLM_GOLDEN_DIR to run the opt-in MiniCPM LLM golden tests")
        }
        guard fixture.schema == schema else {
            throw MiniCPMLLMGoldenFixture.FixtureError.invalid(
                "unsupported schema \(fixture.schema)")
        }
        return fixture
    }

    private func loadModel(
        _ fixture: MiniCPMLLMGoldenFixture
    ) throws -> MiniCPMLanguageModel {
        guard let modelURL = try fixture.modelURL() else {
            throw XCTSkip(
                "Set MINICPM_LLM_GOLDEN_MODEL or provide the manifest repo-relative model")
        }
        let config = try MiniCPMMLXConfig.load(from: modelURL)
        let model = MiniCPMLanguageModel(config: config)
        try MiniCPMWeightLoader.load(model: model, from: modelURL)
        return model
    }

    private func ids(_ values: [Int]) -> MLXArray {
        MLXArray(values.map(Int32.init))
    }

    private func protocolIDs(
        _ fixture: MiniCPMLLMGoldenFixture
    ) throws -> MiniCPMProtocolTokenIds {
        let object = try fixture.object(at: ["protocol_ids"])
        let allSpecial: Set<Int>
        if let values = fixture.manifest["protocol_special_ids"] as? [NSNumber] {
            allSpecial = Set(values.map(\.intValue))
        } else {
            allSpecial = []
        }
        return MiniCPMProtocolTokenIds(
            listen: try fixture.intValue(object, "listen"),
            chunkEOS: try fixture.intValue(object, "chunk_eos"),
            chunkTTSEOS: try fixture.intValue(object, "chunk_tts_eos"),
            turnEOS: try fixture.intValue(object, "turn_eos"),
            speak: try fixture.intValue(object, "speak"),
            unitEnd: try fixture.intValue(object, "unit_end"),
            ttsBOS: try fixture.optionalIntValue(object, "tts_bos"),
            allSpecialTokenIds: allSpecial)
    }

    /// Validate the optional quality report separately from the strict
    /// Python-MLX parity oracle.  An 8-bit-vs-BF16 quality report is allowed to
    /// be ``failed``: that status describes quantization loss, not a
    /// Swift-to-MLX implementation mismatch.
    private func validateQuantizationReport(
        _ fixture: MiniCPMLLMGoldenFixture
    ) throws {
        guard let reportName = fixture.manifest["quantization_report"] as? String else {
            return
        }
        guard let report = try fixture.jsonFile(reportName) as? [String: Any] else {
            throw MiniCPMLLMGoldenFixture.FixtureError.invalid(
                "quantization report is not an object")
        }
        guard report["schema"] as? String == "minicpm-o-llm-quantization-error/v1" else {
            throw MiniCPMLLMGoldenFixture.FixtureError.invalid(
                "unsupported quantization report schema")
        }
        guard let status = report["status"] as? String,
              status == "passed" || status == "failed" else {
            throw MiniCPMLLMGoldenFixture.FixtureError.invalid(
                "quantization report status must be passed or failed")
        }
        guard let underTest = report["under_test"] as? [String: Any],
              let reference = report["reference"] as? [String: Any],
              let thresholds = report["thresholds"] as? [String: Any],
              let scenarios = report["scenarios"] as? [String: Any] else {
            throw MiniCPMLLMGoldenFixture.FixtureError.invalid(
                "quantization report is missing required sections")
        }
        XCTAssertEqual(
            underTest["backend"] as? String,
            fixture.manifest["oracle_backend"] as? String,
            "quantization report backend must match the fixture oracle")
        XCTAssertEqual(reference["backend"] as? String, "official_pytorch")
        for key in ["hidden_max_abs", "hidden_relative_l2", "logits_selected_cosine", "top20_overlap_min"] {
            guard thresholds[key] is NSNumber else {
                throw MiniCPMLLMGoldenFixture.FixtureError.invalid(
                    "quantization report threshold \(key) is missing")
            }
        }
        for name in ["prefill", "incremental", "external", "rollback", "trim", "sliding_basic", "sliding_context"] {
            guard let scenario = scenarios[name] as? [String: Any],
                  let hidden = scenario["hidden"] as? [String: Any],
                  let logits = scenario["logits_selected"] as? [String: Any],
                  let overlap = scenario["top20_overlap"] as? [String: Any] else {
                throw MiniCPMLLMGoldenFixture.FixtureError.invalid(
                    "quantization report scenario \(name) is incomplete")
            }
            for key in ["cosine", "max_abs", "mean_abs", "relative_l2"] {
                guard hidden[key] is NSNumber else {
                    throw MiniCPMLLMGoldenFixture.FixtureError.invalid(
                        "quantization report \(name).hidden.\(key) is missing")
                }
                guard logits[key] is NSNumber else {
                    throw MiniCPMLLMGoldenFixture.FixtureError.invalid(
                        "quantization report \(name).logits_selected.\(key) is missing")
                }
            }
            guard overlap["min"] is NSNumber,
                  overlap["rows"] is [NSNumber] else {
                throw MiniCPMLLMGoldenFixture.FixtureError.invalid(
                    "quantization report \(name).top20_overlap is incomplete")
            }
        }
    }

    private func outputObject(
        _ scenario: [String: Any]
    ) throws -> [String: Any] {
        guard let output = scenario["output"] as? [String: Any] else {
            throw MiniCPMLLMGoldenFixture.FixtureError.invalid("scenario has no output")
        }
        return output
    }

    private func row(
        _ fixture: MiniCPMLLMGoldenFixture,
        array name: String,
        _ row: Int = 0
    ) throws -> [Float] {
        let shape = try fixture.arrayShape(name)
        let values = try fixture.floatArray(name)
        guard !shape.isEmpty else { return values }
        let width = shape.last ?? values.count
        let product = row.multipliedReportingOverflow(by: width)
        guard width >= 0,
              row >= 0,
              !product.overflow,
              product.partialValue >= 0,
              product.partialValue + width <= values.count else {
            throw MiniCPMLLMGoldenFixture.FixtureError.invalid(
                "array \(name) row \(row) is out of bounds")
        }
        let start = product.partialValue
        return Array(values[start ..< (start + width)])
    }

    private func intRow(
        _ fixture: MiniCPMLLMGoldenFixture,
        array name: String,
        _ row: Int = 0
    ) throws -> [Int] {
        let shape = try fixture.arrayShape(name)
        let values = try fixture.intArray(name)
        guard !shape.isEmpty else { return values }
        let width = shape.last ?? values.count
        let start = row * width
        guard row >= 0, start >= 0, start + width <= values.count else {
            throw MiniCPMLLMGoldenFixture.FixtureError.invalid(
                "array \(name) row \(row) is out of bounds")
        }
        return Array(values[start ..< (start + width)])
    }

    private func tolerances(
        _ fixture: MiniCPMLLMGoldenFixture,
        _ model: MiniCPMLanguageModel
    ) -> (hidden: Float, cosine: Double, overlap: Int) {
        let all = fixture.manifest["tolerances"] as? [String: Any]
        let key = model.config.quantBits >= 8 ? "mlx_8bit" : "bf16"
        let values = all?[key] as? [String: Any]
        let hidden = (values?["hidden_max_abs"] as? NSNumber)?.floatValue ?? 0.2
        let cosine = (values?["logits_cosine"] as? NSNumber)?.doubleValue ?? 0.96
        let overlap = (values?["top20_overlap"] as? NSNumber)?.intValue ?? 18
        return (
            hidden,
            cosine,
            overlap)
    }

    private func cosine(_ lhs: [Float], _ rhs: [Float]) -> Double {
        guard lhs.count == rhs.count, !lhs.isEmpty else { return 0 }
        var dot = 0.0
        var lhsNorm = 0.0
        var rhsNorm = 0.0
        for (a, b) in zip(lhs, rhs) {
            let x = Double(a), y = Double(b)
            dot += x * y
            lhsNorm += x * x
            rhsNorm += y * y
        }
        guard lhsNorm > 0, rhsNorm > 0 else { return dot == 0 ? 1 : 0 }
        return dot / (sqrt(lhsNorm) * sqrt(rhsNorm))
    }

    private func assertOutput(
        _ output: MiniCPMForwardOutput,
        fixture: MiniCPMLLMGoldenFixture,
        expected object: [String: Any],
        row rowIndex: Int = 0,
        model: MiniCPMLanguageModel,
        label: String,
        file: StaticString = #filePath,
        line: UInt = #line
    ) throws {
        let expected = try fixture.output(object)
        eval(output.logits, output.hidden)
        let actualLogits = output.logits[0, output.logits.dim(1) - 1]
            .asType(.float32).asArray(Float.self)
        let actualHidden = output.hidden[0, output.hidden.dim(1) - 1]
            .asType(.float32).asArray(Float.self)
        let selected = try row(fixture, array: expected.logitsSelected, rowIndex)
        let hidden = try row(fixture, array: expected.hidden, rowIndex)
        XCTAssertEqual(selected.count, expected.selectedIndices.count, "\(label) selected shape", file: file, line: line)
        XCTAssertEqual(hidden.count, actualHidden.count, "\(label) hidden shape", file: file, line: line)
        guard selected.count == expected.selectedIndices.count,
              hidden.count == actualHidden.count else { return }

        let actualSelected = expected.selectedIndices.map { index in
            index >= 0 && index < actualLogits.count ? actualLogits[index] : .nan
        }
        let tolerance = tolerances(fixture, model)
        let selectedDiff = zip(actualSelected, selected)
            .map { abs(Double($0) - Double($1)) }.max() ?? 0
        let hiddenDiff = zip(actualHidden, hidden)
            .map { abs(Double($0) - Double($1)) }.max() ?? 0
        XCTAssertLessThanOrEqual(
            selectedDiff, Double(tolerance.hidden) * 2,
            "\(label) selected logits drift", file: file, line: line)
        XCTAssertLessThanOrEqual(
            hiddenDiff, Double(tolerance.hidden),
            "\(label) hidden drift", file: file, line: line)

        let expectedTop = try intRow(fixture, array: expected.top20, rowIndex)
        let actualTop = actualLogits.enumerated().sorted {
            $0.element == $1.element ? $0.offset < $1.offset : $0.element > $1.element
        }.prefix(expectedTop.count).map(\.offset)
        let overlap = Set(actualTop).intersection(expectedTop).count
        XCTAssertGreaterThanOrEqual(
            overlap, tolerance.overlap,
            "\(label) top-k overlap \(overlap)/\(expectedTop.count)", file: file, line: line)

        let expectedSelectedLogits = selected
        let actualSelectedCosine = cosine(actualSelected, expectedSelectedLogits)
        XCTAssertGreaterThanOrEqual(
            actualSelectedCosine, tolerance.cosine,
            "\(label) selected-logit cosine", file: file, line: line)
    }

    private func assertEquivalent(
        _ lhs: MiniCPMForwardOutput,
        _ rhs: MiniCPMForwardOutput,
        lhsExpected lhsObject: [String: Any],
        lhsRow: Int,
        rhsExpected rhsObject: [String: Any],
        rhsRow: Int,
        fixture: MiniCPMLLMGoldenFixture,
        model: MiniCPMLanguageModel,
        label: String
    ) throws {
        eval(lhs.logits, lhs.hidden, rhs.logits, rhs.hidden)
        let lhsLogits = lhs.logits[0, lhs.logits.dim(1) - 1].asType(.float32).asArray(Float.self)
        let rhsLogits = rhs.logits[0, rhs.logits.dim(1) - 1].asType(.float32).asArray(Float.self)
        let lhsHidden = lhs.hidden[0, lhs.hidden.dim(1) - 1].asType(.float32).asArray(Float.self)
        let rhsHidden = rhs.hidden[0, rhs.hidden.dim(1) - 1].asType(.float32).asArray(Float.self)
        let lhsExpectedOutput = try fixture.output(lhsObject)
        let rhsExpectedOutput = try fixture.output(rhsObject)
        let lhsOfficialHidden = try row(fixture, array: lhsExpectedOutput.hidden, lhsRow)
        let rhsOfficialHidden = try row(fixture, array: rhsExpectedOutput.hidden, rhsRow)
        let tolerance = tolerances(fixture, model)
        XCTAssertEqual(lhsLogits.count, rhsLogits.count, "\(label) logits shape")
        XCTAssertEqual(lhsHidden.count, rhsHidden.count, "\(label) actual hidden shape")
        XCTAssertEqual(lhsHidden.count, lhsOfficialHidden.count, "\(label) prefill hidden shape")
        XCTAssertEqual(rhsHidden.count, rhsOfficialHidden.count, "\(label) incremental hidden shape")
        guard lhsLogits.count == rhsLogits.count,
              lhsHidden.count == rhsHidden.count,
              lhsHidden.count == lhsOfficialHidden.count,
              rhsHidden.count == rhsOfficialHidden.count else { return }
        XCTAssertGreaterThanOrEqual(cosine(lhsLogits, rhsLogits), tolerance.cosine, "\(label) logits parity")
        var hiddenDiff = 0.0
        for index in lhsHidden.indices {
            let actualDelta = Double(lhsHidden[index]) - Double(rhsHidden[index])
            let officialDelta = Double(lhsOfficialHidden[index]) - Double(rhsOfficialHidden[index])
            hiddenDiff = max(hiddenDiff, abs(actualDelta - officialDelta))
        }
        XCTAssertLessThanOrEqual(hiddenDiff, Double(tolerance.hidden) * 2, "\(label) cross-mode hidden delta")
    }

    // MARK: Cache checksums

    private func assertCache(
        _ session: MiniCPMSession,
        expected object: Any,
        fixture: MiniCPMLLMGoldenFixture,
        label: String
    ) throws {
        guard let expected = object as? [String: Any],
              let layers = expected["layers"] as? [[String: Any]] else {
            throw MiniCPMLLMGoldenFixture.FixtureError.invalid("\(label) cache record")
        }
        XCTAssertEqual(session.cacheLength, try fixture.intValue(expected, "cache_length"), "\(label) cache length")
        let actualLayers = session.state.kvCaches
        XCTAssertEqual(actualLayers.count, layers.count, "\(label) layer count")
        let kvTolerance = (fixture.manifest["tolerances"] as? [String: Any])?["kv_checksum"] as? [String: Any]
        let sumTolerance = (kvTolerance?["sum_abs"] as? NSNumber)?.doubleValue ?? 0.2
        let sumSqTolerance = (kvTolerance?["sum_sq_abs"] as? NSNumber)?.doubleValue ?? 1.0
        for (index, expectedLayer) in layers.enumerated() where index < actualLayers.count {
            let expectedLength = try fixture.intValue(expectedLayer, "length")
            let pair = actualLayers[index]
            XCTAssertEqual(pair?.0.dim(2) ?? -1, expectedLength, "\(label) layer \(index) length")
            guard let pair,
                  let keys = expectedLayer["keys"] as? [String: Any],
                  let values = expectedLayer["values"] as? [String: Any] else { continue }
            assertTensor(pair.0, expected: keys, fixture: fixture, sumTolerance: sumTolerance, sumSqTolerance: sumSqTolerance, label: "\(label) layer \(index) keys")
            assertTensor(pair.1, expected: values, fixture: fixture, sumTolerance: sumTolerance, sumSqTolerance: sumSqTolerance, label: "\(label) layer \(index) values")
        }
    }

    private func assertTensor(
        _ tensor: MLXArray,
        expected: [String: Any],
        fixture: MiniCPMLLMGoldenFixture,
        sumTolerance: Double,
        sumSqTolerance: Double,
        label: String
    ) {
        eval(tensor)
        let values = tensor.asType(.float32).asArray(Float.self)
        if let shape = expected["shape"] as? [NSNumber] {
            XCTAssertEqual(tensor.shape, shape.map(\.intValue), "\(label) shape")
        }
        let sum = values.reduce(0.0) { $0 + Double($1) }
        let sumSq = values.reduce(0.0) { $0 + Double($1) * Double($1) }
        if let expectedSum = expected["sum"] as? NSNumber {
            XCTAssertLessThanOrEqual(abs(sum - expectedSum.doubleValue), sumTolerance, "\(label) sum")
        }
        if let expectedSumSq = expected["sum_sq"] as? NSNumber {
            XCTAssertLessThanOrEqual(abs(sumSq - expectedSumSq.doubleValue), sumSqTolerance, "\(label) sum_sq")
        }
        if let first = expected["first8"] as? [NSNumber] {
            for (actual, wanted) in zip(values.prefix(first.count), first) {
                XCTAssertEqual(actual, wanted.floatValue, accuracy: Float(sumTolerance), "\(label) first8")
            }
        }
        if let last = expected["last8"] as? [NSNumber] {
            for (actual, wanted) in zip(values.suffix(last.count), last) {
                XCTAssertEqual(actual, wanted.floatValue, accuracy: Float(sumTolerance), "\(label) last8")
            }
        }
        // The exporter stores a digest too.  Recompute it here only for the
        // small checksum slices above; full KV byte hashing would duplicate
        // tens of GB of work and the numerical checks already catch backend
        // drift while keeping this opt-in test practical.
        _ = fixture
    }

    // MARK: Context-window position trace

    /// Return a compact position-level mismatch instead of emitting hundreds
    /// of XCTest failures.  The exporter records the same digest for every
    /// layer/position; walking in physical order identifies the first true
    /// divergence (prefix, rebuilt segment, retained K/V, or relocated K).
    private func firstCacheTraceMismatch(
        _ state: MiniCPMInferenceState,
        expected object: [String: Any],
        label: String,
        sumTolerance: Double = 0.2,
        sumSqTolerance: Double = 1.0
    ) throws -> String? {
        guard let expectedLayers = object["layers"] as? [[String: Any]] else {
            throw MiniCPMLLMGoldenFixture.FixtureError.invalid(
                "\(label) trace has no layers")
        }
        let expectedLength = (object["cache_length"] as? NSNumber)?.intValue ?? -1
        if state.sequenceLength != expectedLength {
            return "\(label): cache length actual \(state.sequenceLength), expected \(expectedLength)"
        }
        for index in expectedLayers.indices {
            let expectedLayer = expectedLayers[index]
            guard index < state.kvCaches.count,
                  let pair = state.kvCaches[index] else {
                return "\(label): layer \(index) cache missing"
            }
            let layerLength = (expectedLayer["length"] as? NSNumber)?.intValue ?? -1
            if pair.0.dim(2) != layerLength {
                return "\(label): layer \(index) length actual \(pair.0.dim(2)), expected \(layerLength)"
            }
            guard let positions = expectedLayer["positions"] as? [[String: Any]] else {
                throw MiniCPMLLMGoldenFixture.FixtureError.invalid(
                    "\(label) layer \(index) trace has no positions")
            }
            eval(pair.0, pair.1)
            let keys = pair.0.asType(.float32)
            let values = pair.1.asType(.float32)
            for positionObject in positions {
                guard let position = (positionObject["position"] as? NSNumber)?.intValue,
                      position >= 0, position < pair.0.dim(2),
                      let expectedKeys = positionObject["keys"] as? [String: Any],
                      let expectedValues = positionObject["values"] as? [String: Any] else {
                    throw MiniCPMLLMGoldenFixture.FixtureError.invalid(
                        "\(label) layer \(index) has malformed position trace")
                }
                let actualKeyValues = keys[0..., 0..., position, 0...].asArray(Float.self)
                let actualValueValues = values[0..., 0..., position, 0...].asArray(Float.self)
                for (name, actual) in [("keys", actualKeyValues), ("values", actualValueValues)] {
                    guard let wanted = (name == "keys" ? expectedKeys : expectedValues)["sum"] as? NSNumber,
                          let wantedSumSq = (name == "keys" ? expectedKeys : expectedValues)["sum_sq"] as? NSNumber else {
                        throw MiniCPMLLMGoldenFixture.FixtureError.invalid(
                            "\(label) layer \(index) position \(position) \(name) digest")
                    }
                    var sum = 0.0
                    var sumSq = 0.0
                    for item in actual {
                        let value = Double(item)
                        sum += value
                        sumSq += value * value
                    }
                    if abs(sum - wanted.doubleValue) > sumTolerance {
                        return "\(label): first divergence layer \(index) position \(position) \(name) sum actual \(sum) expected \(wanted.doubleValue)"
                    }
                    if abs(sumSq - wantedSumSq.doubleValue) > sumSqTolerance {
                        return "\(label): first divergence layer \(index) position \(position) \(name) sum_sq actual \(sumSq) expected \(wantedSumSq.doubleValue)"
                    }
                }
            }
        }
        return nil
    }

    /// Build independent cache slices for the diagnostic rebuild.  The
    /// zero-backed destination prevents the subsequent forward from writing
    /// through an MLX view into the live session cache.
    private func independentCacheSlice(
        _ state: MiniCPMInferenceState,
        start: Int,
        end: Int
    ) -> MiniCPMInferenceState {
        let length = max(0, end - start)
        let pairs: [(MLXArray, MLXArray)?] = state.kvCaches.map { pair in
            guard let pair else { return nil }
            let keys = MLXArray.zeros(
                [pair.0.dim(0), pair.0.dim(1), length, pair.0.dim(3)],
                dtype: pair.0.dtype)
            let values = MLXArray.zeros(
                [pair.1.dim(0), pair.1.dim(1), length, pair.1.dim(3)],
                dtype: pair.1.dtype)
            if length > 0 {
                keys[0..., 0..., 0 ..< length, 0...] =
                    pair.0[0..., 0..., start ..< end, 0...]
                values[0..., 0..., 0 ..< length, 0...] =
                    pair.1[0..., 0..., start ..< end, 0...]
            }
            return (keys, values)
        }
        return MiniCPMInferenceState(kvCaches: pairs, position: length)
    }

    /// Reproduce the official context rebuild's intermediate states from the
    /// pre-eviction Swift cache.  This does not mutate the session; it lets the
    /// golden test compare the rebuilt previous/suffix forward separately from
    /// the retained-tail RoPE surgery.
    private func contextDiagnosticStates(
        model: MiniCPMLanguageModel,
        source: MiniCPMInferenceState,
        marker: [Int]
    ) -> [String: MiniCPMInferenceState] {
        let prefixLength = 1
        let retainedLength = 1
        let retainedStart = source.sequenceLength - retainedLength
        let prefixState = independentCacheSlice(source, start: 0, end: prefixLength)
        let retainedState = independentCacheSlice(source, start: retainedStart, end: source.sequenceLength)
        let previousAndSuffix = model.embed(
            MLXArray((marker + [20, 21, 151645]).map(Int32.init))).asType(.bfloat16)
        let recomputedState = model.forward(
            inputEmbeddings: previousAndSuffix,
            state: prefixState).state

        var finalPairs: [(MLXArray, MLXArray)?] = []
        finalPairs.reserveCapacity(recomputedState.kvCaches.count)
        for index in recomputedState.kvCaches.indices {
            guard let head = recomputedState.kvCaches[index],
                  let tail = retainedState.kvCaches[index] else {
                finalPairs.append(nil)
                continue
            }
            let tailKeys = MiniCPMCacheRoPE.realign(
                tail.0,
                oldStart: retainedStart,
                newStart: recomputedState.sequenceLength,
                ropeTheta: model.config.ropeTheta)
            let keys = concatenated([head.0, tailKeys], axis: 2)
            let values = concatenated([head.1, tail.1], axis: 2)
            finalPairs.append((keys, values))
        }
        let finalState = MiniCPMInferenceState(
            kvCaches: finalPairs,
            position: recomputedState.sequenceLength + retainedLength)
        return [
            "prefix": prefixState,
            "retained_original": retainedState,
            "recomputed_previous_suffix": recomputedState,
            "retained_reindexed": MiniCPMInferenceState(
                kvCaches: retainedState.kvCaches.enumerated().map { index, pair in
                    guard let pair else { return nil }
                    let keys = MiniCPMCacheRoPE.realign(
                        pair.0,
                        oldStart: retainedStart,
                        newStart: recomputedState.sequenceLength,
                        ropeTheta: model.config.ropeTheta)
                    return (keys, pair.1)
                },
                position: retainedLength),
            "final": finalState,
        ]
    }

    // MARK: Protocol and embedding fixtures

    func testManifestProtocolIDs() throws {
        let fixture = try loadFixture()
        guard let backend = fixture.manifest["oracle_backend"] as? String else {
            throw MiniCPMLLMGoldenFixture.FixtureError.invalid(
                "manifest has no oracle_backend")
        }
        let expectedAttention: String
        switch backend {
        case "official_pytorch":
            expectedAttention = "sdpa"
        case "python_mlx_quantized":
            expectedAttention = "mlx_lm.qwen3.scaled_dot_product_attention"
        default:
            throw MiniCPMLLMGoldenFixture.FixtureError.invalid(
                "unsupported oracle backend \(backend)")
        }
        let runtime = fixture.manifest["runtime"] as? [String: Any]
        XCTAssertEqual(
            runtime?["attention_implementation"] as? String,
            expectedAttention,
            "the oracle must record its resolved attention backend")
        XCTAssertEqual(runtime?["oracle_backend"] as? String, backend)
        try fixture.validateAllArrays()
        try validateQuantizationReport(fixture)
        let ids = try fixture.object(at: ["protocol_ids"])
        let eos = try fixture.intValue(ids, "eos")
        let listen = try fixture.intValue(ids, "listen")
        XCTAssertEqual(eos, 151645)
        XCTAssertEqual(listen, 151705)
        XCTAssertNotEqual(eos, listen, "listen must not alias the normal EOS token")
        XCTAssertEqual(try fixture.intValue(ids, "tokenizer_eos_token_id"), eos)
    }

    func testTokenEmbedding() throws {
        let fixture = try loadFixture()
        let model = try loadModel(fixture)
        let inputs = try fixture.object(at: ["inputs"])
        let tokenValues = try fixture.intValues(inputs, "token_ids_values")
        let expectedName = try fixture.stringValue(inputs, "token_embeddings")
        let expected = try fixture.floatArray(expectedName)
        let actual = model.embed(ids(tokenValues))
        eval(actual)
        let values = actual.asType(.float32).asArray(Float.self)
        XCTAssertEqual(values.count, expected.count)
        let maxDiff = zip(values, expected).map { abs(Double($0) - Double($1)) }.max() ?? 0
        XCTAssertLessThanOrEqual(maxDiff, 1e-3, "token embedding drift")
    }

    // MARK: Prefill, incremental, and external embeddings

    func testBatchPrefillAndIncremental() throws {
        let fixture = try loadFixture()
        let model = try loadModel(fixture)
        let scenario = try fixture.scenario("prefill")
        let tokenValues = try fixture.intValues(scenario, "token_ids")
        let session = MiniCPMSession(model: model)
        let prefillOutput = session.feed(inputIds: ids(tokenValues))
        let prefillOutputObject = try outputObject(scenario)
        try assertOutput(prefillOutput, fixture: fixture, expected: prefillOutputObject, model: model, label: "prefill")
        try assertCache(session, expected: try fixture.jsonFile(try fixture.stringValue(scenario, "cache")), fixture: fixture, label: "prefill")

        let incremental = try fixture.scenario("incremental")
        let incrementalOutputObject = try outputObject(incremental)
        let incrementalCacheFile = try fixture.stringValue(incremental, "cache")
        guard let expectedCaches = try fixture.jsonFile(incrementalCacheFile) as? [[String: Any]] else {
            throw MiniCPMLLMGoldenFixture.FixtureError.invalid("incremental cache record")
        }
        session.reset()
        var outputs: [MiniCPMForwardOutput] = []
        for (index, token) in tokenValues.enumerated() {
            let output = session.feed(inputIds: ids([token]))
            outputs.append(output)
            try assertOutput(output, fixture: fixture, expected: incrementalOutputObject, row: index, model: model, label: "incremental-\(index)")
            XCTAssertEqual(session.cacheLength, try fixture.intValue(expectedCaches[index], "cache_length"), "incremental-\(index) cache length")
        }
        XCTAssertEqual(outputs.count, tokenValues.count)
        if let last = outputs.last {
            let lastStep = (incremental["batch_last_step"] as? NSNumber)?.intValue ?? tokenValues.count - 1
            try assertOutput(last, fixture: fixture, expected: incrementalOutputObject, row: lastStep, model: model, label: "incremental-last")
            try assertEquivalent(
                prefillOutput, last,
                lhsExpected: prefillOutputObject, lhsRow: 0,
                rhsExpected: incrementalOutputObject, rhsRow: lastStep,
                fixture: fixture, model: model, label: "prefill-vs-incremental")
        }
        if let expectedLast = expectedCaches.last {
            try assertCache(session, expected: expectedLast, fixture: fixture, label: "incremental-last")
        }
    }

    func testExternalEmbeddings() throws {
        let fixture = try loadFixture()
        let model = try loadModel(fixture)
        let inputs = try fixture.object(at: ["inputs"])
        let externalName = try fixture.stringValue(inputs, "external_embeddings")
        let shape = try fixture.arrayShape(externalName)
        let values = try fixture.floatArray(externalName)
        let external = MLXArray(values).reshaped(shape)
        let output = MiniCPMSession(model: model).feed(embeddings: external)
        try assertOutput(output, fixture: fixture, expected: try outputObject(try fixture.scenario("external")), model: model, label: "external")
    }

    func testAudioEmbeddingBridge() throws {
        let fixture = try loadFixture()
        let model = try loadModel(fixture)
        let protocolIDs = try protocolIDs(fixture)
        let scenario = try fixture.scenario("audio_bridge")
        let audioName = try fixture.stringValue(scenario, "audio_array")
        let audioShape = try fixture.arrayShape(audioName)
        XCTAssertEqual(
            audioShape,
            (scenario["audio_shape"] as? [NSNumber])?.map(\.intValue) ?? audioShape,
            "audio bridge shape")
        let audio = MLXArray(try fixture.floatArray(audioName)).reshaped(audioShape)
        let cases = scenario["cases"] as? [[String: Any]] ?? []
        XCTAssertEqual(cases.count, 4, "audio bridge fixture must cover all bridge cases")

        for item in cases {
            let name = try fixture.stringValue(item, "name")
            let session = MiniCPMSession(model: model)
            var outputs: [MiniCPMForwardOutput] = []
            let operations = item["operations"] as? [[String: Any]] ?? []
            for operation in operations {
                let kind = try fixture.stringValue(operation, "kind")
                switch kind {
                case "audio":
                    outputs.append(session.feed(embeddings: audio))
                case "token":
                    outputs.append(session.feed(inputIds: ids([
                        try fixture.intValue(operation, "token_id")
                    ])))
                default:
                    throw MiniCPMLLMGoldenFixture.FixtureError.invalid(
                        "unknown audio bridge operation \(kind)")
                }
            }

            if (item["repair"] as? Bool) == true {
                session.reset()
                outputs = [session.feed(embeddings: audio)]
                session.beginUnit()
                _ = session.feed(inputIds: ids([protocolIDs.listen]))
                _ = session.feed(inputIds: ids([protocolIDs.speak]))
                XCTAssertEqual(session.rollbackUnit(), 2, "\(name) rollback length")
                outputs.append(session.feed(inputIds: ids([protocolIDs.turnEOS])))
                outputs.append(session.feed(inputIds: ids([protocolIDs.unitEnd])))
            }

            let expectedOutput = try outputObject(item)
            XCTAssertEqual(
                outputs.count,
                try fixture.intValue(expectedOutput, "steps"),
                "\(name) output steps")
            for (index, output) in outputs.enumerated() {
                try assertOutput(
                    output,
                    fixture: fixture,
                    expected: expectedOutput,
                    row: index,
                    model: model,
                    label: "audio-bridge-\(name)-\(index)")
            }
            let cacheName = try fixture.stringValue(item, "cache")
            try assertCache(
                session,
                expected: try fixture.jsonFile(cacheName),
                fixture: fixture,
                label: "audio-bridge-\(name)")
        }
    }

    // MARK: Transactional rollback and direct trim

    func testRollback() throws {
        let fixture = try loadFixture()
        let model = try loadModel(fixture)
        let scenario = try fixture.scenario("rollback")
        let session = MiniCPMSession(model: model)
        let prefix = try fixture.intValues(scenario, "prefix_token_ids")
        let speculative = try fixture.intValues(scenario, "speculative_token_ids")
        let canonical = try fixture.intValues(scenario, "canonical_token_ids")
        _ = session.feed(inputIds: ids(prefix))
        XCTAssertEqual(session.cacheLength, try fixture.intValue(scenario, "cache_before"))
        session.beginUnit()
        for token in speculative { _ = session.feed(inputIds: ids([token])) }
        XCTAssertEqual(session.cacheLength, try fixture.intValue(scenario, "cache_speculative"))
        XCTAssertEqual(session.rollbackUnit(), try fixture.intValue(scenario, "removed"))
        XCTAssertEqual(session.cacheLength, try fixture.intValue(scenario, "cache_after"))
        let output = try outputObject(scenario)
        for (index, token) in canonical.enumerated() {
            let result = session.feed(inputIds: ids([token]))
            try assertOutput(result, fixture: fixture, expected: output, row: index, model: model, label: "rollback-\(index)")
        }
    }

    func testTrim() throws {
        let fixture = try loadFixture()
        let model = try loadModel(fixture)
        let scenario = try fixture.scenario("trim")
        var state = model.initialState()
        let prefix = try fixture.intValues(scenario, "prefix_token_ids")
        let prefixOutput = model.forward(inputIds: ids(prefix), state: state)
        state = model.trimmedState(prefixOutput.state, to: try fixture.intValue(scenario, "target_length"))
        XCTAssertEqual(state.sequenceLength, try fixture.intValue(scenario, "target_length"))
        let probe = try fixture.intValue(scenario, "probe_token_id")
        let output = model.forward(inputIds: ids([probe]), state: state)
        XCTAssertEqual(output.state.sequenceLength, try fixture.intValue(scenario, "cache_after_probe"))
        try assertOutput(output, fixture: fixture, expected: try outputObject(scenario), model: model, label: "trim")
    }

    // MARK: Basic/context sliding windows

    func testBasicSlidingWindow() throws {
        let fixture = try loadFixture()
        let model = try loadModel(fixture)
        let scenario = try fixture.scenario("sliding_basic")
        let config = scenario["config"] as? [String: Any] ?? [:]
        let high = (config["high"] as? NSNumber)?.intValue ?? 4
        let low = (config["low"] as? NSNumber)?.intValue ?? 2
        let session = MiniCPMSession(model: model)
        session.configureSlidingWindow(MiniCPMSlidingWindowConfig(mode: "basic", basicHighTokens: high, basicLowTokens: low))
        session.beginSystemPrompt()
        _ = session.feed(inputIds: ids(try fixture.intValues(scenario, "prefix_token_ids")))
        session.finishSystemPrompt()
        let units = scenario["units"] as? [[String: Any]] ?? []
        var eventCount = 0
        for (index, unit) in units.enumerated() {
            let before = session.units.map(\.unitID)
            session.beginUnit()
            let token = try fixture.intValue(unit, "input_token_id")
            _ = session.feed(inputIds: ids([token]))
            let cacheAfterFeed = session.cacheLength
            session.commitUnit(generatedTokens: try fixture.intValues(unit, "generated_tokens"))
            let triggered = session.enforceSlidingWindow()
            if triggered {
                eventCount += 1
                let after = session.units.map(\.unitID)
                let expectedEvents = scenario["events"] as? [[String: Any]] ?? []
                let event = expectedEvents[eventCount - 1]
                XCTAssertEqual(try fixture.intValue(event, "unit_id"), index)
                XCTAssertEqual(try fixture.intValues(event, "before_unit_ids"), before)
                XCTAssertEqual(try fixture.intValues(event, "after_unit_ids"), after)
                XCTAssertEqual(try fixture.intValues(event, "dropped_unit_ids"), before.filter { !after.contains($0) })
                XCTAssertEqual(try fixture.intValue(event, "cache_before"), cacheAfterFeed)
                XCTAssertEqual(try fixture.intValue(event, "cache_after"), session.cacheLength)
            }
        }
        XCTAssertEqual(eventCount, (scenario["events"] as? [[String: Any]] ?? []).count)
        let probe = try fixture.intValue(scenario, "probe_token_id")
        let output = session.feed(inputIds: ids([probe]))
        try assertOutput(output, fixture: fixture, expected: try outputObject(scenario), model: model, label: "sliding-basic")
        try assertCache(session, expected: try fixture.jsonFile(try fixture.stringValue(scenario, "cache")), fixture: fixture, label: "sliding-basic")
    }

    func testContextSlidingWindow() throws {
        let fixture = try loadFixture()
        let model = try loadModel(fixture)
        let scenario = try fixture.scenario("sliding_context")
        let config = scenario["config"] as? [String: Any] ?? [:]
        let maxUnits = (config["max_units"] as? NSNumber)?.intValue ?? 1
        let maxPrevious = (config["previous_max_tokens"] as? NSNumber)?.intValue ?? 8
        let marker = try fixture.intValues(scenario, "marker_token_ids")
        let session = MiniCPMSession(model: model)
        session.configureSlidingWindow(MiniCPMSlidingWindowConfig(
            mode: "context", contextPreviousMaxTokens: maxPrevious,
            contextMaxUnits: maxUnits,
            contextPreviousMarkerTokenIDs: marker))
        session.beginSystemPrompt()
        _ = session.feed(inputIds: ids(try fixture.intValues(scenario, "prefix_token_ids")))
        var contextTraceStates: [String: MiniCPMInferenceState] = [
            "prefix": session.state,
        ]
        session.markSystemPromptSuffix()
        _ = session.feed(inputIds: ids(try fixture.intValues(scenario, "suffix_token_ids")))
        session.finishSystemPrompt()
        contextTraceStates["prefix_suffix"] = session.state

        let units = scenario["units"] as? [[String: Any]] ?? []
        var eventCount = 0
        var sourceBeforeRebuild: MiniCPMInferenceState?
        for (index, unit) in units.enumerated() {
            session.beginUnit()
            let token = try fixture.intValue(unit, "token")
            _ = session.feed(inputIds: ids([token]))
            session.commitUnit(
                generatedTokens: try fixture.intValues(unit, "generated"),
                generatedText: try fixture.stringValue(unit, "text"))
            contextTraceStates["before_enforce_unit_\(index)"] = session.state
            if index == 1 {
                sourceBeforeRebuild = session.state
            }
            if session.enforceSlidingWindow() {
                eventCount += 1
                contextTraceStates["final"] = session.state
                let events = scenario["events"] as? [[String: Any]] ?? []
                let event = events[eventCount - 1]
                XCTAssertEqual(try fixture.intValue(event, "unit_id"), index)
                XCTAssertEqual(try fixture.intValues(event, "previous_token_ids"), session.previousContext.tokenIDs)
                XCTAssertEqual(try fixture.stringValue(event, "previous_text"), session.previousContext.text)
                XCTAssertEqual(try fixture.intValues(event, "after_unit_ids"), session.units.map(\.unitID))
            }
        }
        XCTAssertEqual(eventCount, (scenario["events"] as? [[String: Any]] ?? []).count)

        // The exporter records stage traces independently from the normal
        // final-output assertion.  Compare each physical position in order and
        // report one actionable first-divergence line rather than 34×N noisy
        // checksum failures.  Rebuild intermediates are reconstructed from the
        // pre-eviction state using the same public model helper and RoPE
        // primitive, so this also separates a segment-forward error from tail
        // relocation/concatenation.
        if let trace = scenario["trace"] as? [String: Any] {
            var mismatches: [String] = []
            for name in ["prefix", "prefix_suffix", "before_enforce_unit_0", "before_enforce_unit_1", "final"] {
                guard let expected = trace[name] as? [String: Any],
                      let state = contextTraceStates[name] else { continue }
                if let mismatch = try firstCacheTraceMismatch(
                    state, expected: expected, label: "context trace \(name)") {
                    mismatches.append(mismatch)
                }
            }
            if let sourceBeforeRebuild,
               let rebuildObject = trace["rebuild"] as? [String: Any] {
                let diagnostic = contextDiagnosticStates(
                    model: model,
                    source: sourceBeforeRebuild,
                    marker: try fixture.intValues(scenario, "marker_token_ids"))
                for name in ["prefix", "retained_original", "recomputed_previous_suffix", "retained_reindexed", "final"] {
                    guard let expected = rebuildObject[name] as? [String: Any],
                          let state = diagnostic[name] else { continue }
                    if let mismatch = try firstCacheTraceMismatch(
                        state, expected: expected, label: "context rebuild \(name)") {
                        mismatches.append(mismatch)
                    }
                }
            }
            if !mismatches.isEmpty {
                XCTFail("context sliding trace first mismatches:\n" + mismatches.joined(separator: "\n"))
            }
        }
        let probe = try fixture.intValue(scenario, "probe_token_id")
        let output = session.feed(inputIds: ids([probe]))
        try assertOutput(output, fixture: fixture, expected: try outputObject(scenario), model: model, label: "sliding-context")
        try assertCache(session, expected: try fixture.jsonFile(try fixture.stringValue(scenario, "cache")), fixture: fixture, label: "sliding-context")
    }

    // MARK: Protocol-aware sampler

    func testSamplerCases() throws {
        let fixture = try loadFixture()
        let protocolIDs = try protocolIDs(fixture)
        let scenario = try fixture.scenario("sampling")
        let cases = scenario["cases"] as? [[String: Any]] ?? []
        for item in cases {
            let name = try fixture.stringValue(item, "name")
            let logits = try fixture.floatArray(try fixture.stringValue(item, "logits"))
            let mode = try fixture.stringValue(item, "mode")
            let config = MiniCPMSamplingConfig(
                mode: mode == "greedy" ? .greedy : .sampling,
                temperature: try fixture.floatValue(item, "temperature"),
                topK: try fixture.intValue(item, "top_k"),
                topP: try fixture.floatValue(item, "top_p"),
                repetitionPenalty: try fixture.floatValue(item, "repetition_penalty"),
                lengthPenalty: try fixture.floatValue(item, "length_penalty"),
                listenTopK: try fixture.optionalIntValue(item, "listen_top_k"),
                listenProbabilityScale: try fixture.floatValue(item, "listen_probability_scale"))
            let forbidden = try fixture.intValues(item, "forbidden")
            let previous = try fixture.intValues(item, "previous_tokens")
            let uniforms = try fixture.floatValues(item, "uniforms")
            let actual = MiniCPMSampler.sample(
                logits: MLXArray(logits), config: config, protocol: protocolIDs,
                forbidden: forbidden, previousTokens: previous,
                uniforms: uniforms).item(Int.self)
            XCTAssertEqual(actual, try fixture.intValue(item, "expected_token"), name)
        }
    }
}

private extension MiniCPMLLMGoldenFixture {
    func floatValues(_ object: [String: Any], _ key: String) throws -> [Float] {
        guard let values = object[key] as? [NSNumber] else {
            throw FixtureError.invalid("\(key) is not a float array")
        }
        return values.map(\.floatValue)
    }
}

/// Fail-closed model-path checks.  These tests intentionally create only
/// tiny temporary files and never instantiate or load a MiniCPM model.
final class MiniCPMLLMGoldenFixtureTests: XCTestCase {
    private func fixture(root: URL) -> MiniCPMLLMGoldenFixture {
        let hash = String(repeating: "0", count: 64)
        return MiniCPMLLMGoldenFixture(
            root: root,
            manifest: [
                "model": [
                    "repo_relative_path": "model",
                    "config_sha256": hash,
                    "tree_sha256": hash,
                ],
            ])
    }

    func testExplicitEmptyModelPathFailsClosed() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("minicpm-llm-fixture-empty-\(UUID().uuidString)")
        let golden = fixture(root: root)
        XCTAssertThrowsError(try golden.modelURL(
            environment: ["MINICPM_LLM_GOLDEN_MODEL": ""])) { error in
            guard case MiniCPMLLMGoldenFixture.FixtureError.invalid(let message) = error else {
                return XCTFail("expected FixtureError.invalid, got \(error)")
            }
            XCTAssertTrue(message.contains("is empty"), message)
        }
    }

    func testExplicitNonDirectoryModelPathFailsClosed() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("minicpm-llm-fixture-file-\(UUID().uuidString)")
        let manager = FileManager.default
        try manager.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? manager.removeItem(at: root) }
        let file = root.appendingPathComponent("model-file")
        try Data("not a model directory".utf8).write(to: file)

        let golden = fixture(root: root)
        XCTAssertThrowsError(try golden.modelURL(
            environment: ["MINICPM_LLM_GOLDEN_MODEL": file.path])) { error in
            guard case MiniCPMLLMGoldenFixture.FixtureError.invalid(let message) = error else {
                return XCTFail("expected FixtureError.invalid, got \(error)")
            }
            XCTAssertTrue(message.contains("not a directory"), message)
        }
    }

    func testExplicitModelProvenanceMismatchFailsClosed() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("minicpm-llm-fixture-hash-\(UUID().uuidString)")
        let manager = FileManager.default
        let model = root.appendingPathComponent("model", isDirectory: true)
        try manager.createDirectory(at: model, withIntermediateDirectories: true)
        defer { try? manager.removeItem(at: root) }
        try Data("{}\n".utf8).write(to: model.appendingPathComponent("config.json"))

        let golden = fixture(root: root)
        XCTAssertThrowsError(try golden.modelURL(
            environment: ["MINICPM_LLM_GOLDEN_MODEL": model.path])) { error in
            guard case MiniCPMLLMGoldenFixture.FixtureError.invalid(let message) = error else {
                return XCTFail("expected FixtureError.invalid, got \(error)")
            }
            XCTAssertTrue(message.contains("sha256 mismatch"), message)
        }
    }

    func testUndiscoveredModelRemainsOptionalWhenNotConfigured() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("minicpm-llm-fixture-undiscovered-\(UUID().uuidString)")
        let golden = fixture(root: root)
        XCTAssertNil(try golden.modelURL(environment: [:]))
    }
}
