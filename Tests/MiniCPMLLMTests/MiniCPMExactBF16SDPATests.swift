import Foundation
import Metal
import XCTest
import MLX
import MLXCommon
@testable import MiniCPMLLM

/// Focused production-helper coverage for the dense BF16 MPSGraph SDPA path.
/// These tests never load the full MiniCPM checkpoint.
final class MiniCPMExactBF16SDPATests: XCTestCase {
    private let queryShape = [1, 4, 3, 8]
    private let keyValueShape = [1, 2, 3, 8]
    private let outputShape = [1, 3, 4, 8]

    func testCommittedNoGradFixtureIsBitExact() throws {
        guard MTLCreateSystemDefaultDevice() != nil else {
            throw XCTSkip("Metal GPU device is unavailable")
        }
        let directory = URL(fileURLWithPath: fixtureDirectory, isDirectory: true)
        guard isDirectory(directory) else {
            XCTFail("committed exact SDPA fixture is unavailable: \(directory.path)")
            return
        }
        let q = try readFloat32(
            directory.appendingPathComponent("query.f32"), count: queryShape)
            .asBF16(shape: queryShape)
        let k = try readFloat32(
            directory.appendingPathComponent("key.f32"), count: keyValueShape)
            .asBF16(shape: keyValueShape)
        let v = try readFloat32(
            directory.appendingPathComponent("value.f32"), count: keyValueShape)
            .asBF16(shape: keyValueShape)
        let reference = try readFloat32(
            directory.appendingPathComponent("output.f32"), count: outputShape)

        let output = MiniCPMExactBF16SDPA.attend(
            query: q,
            key: k,
            value: v,
            scale: 1 / sqrt(Float(queryShape[3])))
        eval(output)
        XCTAssertEqual(output.shape, outputShape)
        XCTAssertEqual(output.dtype, .bfloat16)
        XCTAssertEqual(output.asType(.float32).asArray(Float.self), reference)

        // Same-shape calls exercise the cached graph plan and must preserve
        // the same BF16 bytes, not merely numerically close values.
        let repeated = MiniCPMExactBF16SDPA.attend(
            query: q,
            key: k,
            value: v,
            scale: 1 / sqrt(Float(queryShape[3])))
        eval(repeated)
        XCTAssertEqual(repeated.asType(.float32).asArray(Float.self), reference)
    }

    func testDynamicBatchGQAShapeAndScalePlanKey() throws {
        guard MTLCreateSystemDefaultDevice() != nil else {
            throw XCTSkip("Metal GPU device is unavailable")
        }
        let qShape = [2, 4, 3, 8]
        let kvShape = [2, 2, 3, 8]
        let qCount = qShape.reduce(1, *)
        let kvCount = kvShape.reduce(1, *)
        let qValues: [Float] = (0 ..< qCount).map { index in
            Float((index * 13) % 29 - 14) / 8
        }
        let kValues: [Float] = (0 ..< kvCount).map { index in
            Float((index * 7) % 23 - 11) / 7
        }
        let vValues: [Float] = (0 ..< kvCount).map { index in
            Float((index * 5) % 19 - 9) / 6
        }
        let q = MLXArray(qValues, qShape).asType(.bfloat16)
        let k = MLXArray(kValues, kvShape).asType(.bfloat16)
        let v = MLXArray(vValues, kvShape).asType(.bfloat16)

        let scaleA = MiniCPMExactBF16SDPA.attend(
            query: q, key: k, value: v, scale: 0.5)
        let scaleB = MiniCPMExactBF16SDPA.attend(
            query: q, key: k, value: v, scale: 0.25)
        eval(scaleA, scaleB)
        XCTAssertEqual(scaleA.shape, [2, 3, 4, 8])
        XCTAssertEqual(scaleA.dtype, .bfloat16)
        XCTAssertTrue(scaleA.asType(.float32).asArray(Float.self).allSatisfy(\.isFinite))
        XCTAssertTrue(scaleB.asType(.float32).asArray(Float.self).allSatisfy(\.isFinite))
        XCTAssertNotEqual(
            scaleA.asType(.float32).asArray(Float.self),
            scaleB.asType(.float32).asArray(Float.self),
            "graph plans must not reuse a different scale constant")
    }

    func testRectangularAdditiveMaskUsesRightAlignedCausalRows() throws {
        guard MTLCreateSystemDefaultDevice() != nil else {
            throw XCTSkip("Metal GPU device is unavailable")
        }

        let batch = 1
        let heads = 2
        let queryTokens = 2
        let keyTokens = 3
        let headDimension = 4
        let query = MLXArray.zeros(
            [batch, heads, queryTokens, headDimension],
            dtype: .bfloat16)
        let key = MLXArray.zeros(
            [batch, heads, keyTokens, headDimension],
            dtype: .bfloat16)
        let valueFloats: [Float] = (0 ..< batch * heads * keyTokens * headDimension)
            .map { Float($0 + 1) }
        let value = MLXArray(
            valueFloats,
            [batch, heads, keyTokens, headDimension]).asType(.bfloat16)
        let mask = MiniCPMExactBF16SDPA.rightAlignedAdditiveMask(
            batch: batch,
            queryHeads: heads,
            queryTokens: queryTokens,
            keyTokens: keyTokens)
        eval(mask)

        XCTAssertEqual(mask.shape, [batch, heads, queryTokens, keyTokens])
        XCTAssertEqual(mask.dtype, .bfloat16)
        let maskValues = mask.asType(.float32).asArray(Float.self)
        XCTAssertEqual(maskValues[0], 0)
        XCTAssertEqual(maskValues[1], 0)
        XCTAssertEqual(maskValues[2], MiniCPMExactBF16SDPA.bfloat16FinfoMin)
        XCTAssertEqual(maskValues[3], 0)
        XCTAssertEqual(maskValues[4], 0)
        XCTAssertEqual(maskValues[5], 0)

        let output = MiniCPMExactBF16SDPA.attend(
            query: query,
            key: key,
            value: value,
            scale: 1 / sqrt(Float(headDimension)),
            additiveMask: mask)
        eval(output)
        XCTAssertEqual(output.shape, [batch, queryTokens, heads, headDimension])

        // Zero Q/K produces a uniform BF16 softmax over the allowed rows:
        // query row zero sees keys 0 and 1, while row one sees all three.
        let expected: [Float] = [
            3, 4, 5, 6,
            15, 16, 17, 18,
            5, 6, 7, 8,
            17, 18, 19, 20,
        ]
        XCTAssertEqual(output.asType(.float32).asArray(Float.self), expected)
    }

    func testQuantizedAndOptOutRoutingFlags() throws {
        let denseConfig = try loadConfig(bits: 0)
        let denseModel = MiniCPMLanguageModel(config: denseConfig)
        let expectedDense = ProcessInfo.processInfo.environment["MINICPM_MLX_BF16_SDPA"] != "1"
        XCTAssertEqual(denseModel.layers[0].selfAttention.exactBF16SDPAEnabled, expectedDense)

        let quantizedConfig = try loadConfig(bits: 4)
        let quantizedModel = MiniCPMLanguageModel(config: quantizedConfig)
        XCTAssertFalse(quantizedModel.layers[0].selfAttention.exactBF16SDPAEnabled)
    }

    func testCachedMultiTokenUsesExactRouteButDecodeStaysFused() throws {
        guard MTLCreateSystemDefaultDevice() != nil else {
            throw XCTSkip("Metal GPU device is unavailable")
        }
        let denseConfig = try loadConfig(bits: 0)
        let denseModel = MiniCPMLanguageModel(config: denseConfig)
        let attention = denseModel.layers[0].selfAttention
        let cachedKeys = MLXArray.zeros(
            [1, denseConfig.numKeyValueHeads, 2, denseConfig.headDim],
            dtype: .bfloat16)
        let cached = MiniCPMKVCache(keys: cachedKeys, values: cachedKeys)

        XCTAssertEqual(
            attention.shouldUseExactBF16SDPA(
                cache: nil, length: 2, dtype: .bfloat16),
            attention.exactBF16SDPAEnabled)
        XCTAssertEqual(
            attention.shouldUseExactBF16SDPA(
                cache: cached, length: 2, dtype: .bfloat16),
            attention.exactBF16SDPAEnabled,
            "cached context rebuilds must use exact BF16 SDPA")
        XCTAssertFalse(
            attention.shouldUseExactBF16SDPA(
                cache: cached, length: 1, dtype: .bfloat16),
            "cached one-token decode remains on fused SDPA")
        XCTAssertFalse(
            attention.shouldUseExactBF16SDPA(
                cache: cached, length: 2, dtype: .float32),
            "non-BF16 inputs remain on fused SDPA")

        let quantizedModel = MiniCPMLanguageModel(config: try loadConfig(bits: 4))
        XCTAssertFalse(
            quantizedModel.layers[0].selfAttention.shouldUseExactBF16SDPA(
                cache: cached, length: 2, dtype: .bfloat16),
            "quantized layers must retain fused SDPA")

        // Exercise the production attention call with a two-token segment
        // against a two-token cached prefix.  The exact branch must pass the
        // appended [prefix + segment] K/V (qLen=2, kLen=4) to the helper;
        // the helper's right-aligned causal-mask math is covered separately by
        // MiniCPMMPSGraphSDPATests.
        let x = MLXArray(
            (0 ..< (2 * denseConfig.hiddenSize)).map { Float($0 + 1) / 10 },
            [1, 2, denseConfig.hiddenSize]).asType(.bfloat16)
        let (attentionOutput, nextCache) = attention(
            x, cache: cached, offset: cached.count)
        eval(attentionOutput)
        XCTAssertEqual(
            attentionOutput.shape,
            [1, 2, denseConfig.numAttentionHeads * denseConfig.headDim])
        XCTAssertEqual(nextCache.count, 4)
    }

    func testBitsZeroFloat32WeightsStayOnFusedFallback() throws {
        let config = try loadConfig(bits: 0)
        let model = MiniCPMLanguageModel(config: config)
        let ids = MLXArray([Int32(1), Int32(2)]).expandedDimensions(axis: 0)
        let output = model.forward(inputIds: ids, state: model.initialState())
        eval(output.hidden, output.logits)
        XCTAssertEqual(output.hidden.shape, [1, 2, config.hiddenSize])
        XCTAssertEqual(output.logits.shape, [1, 2, config.vocabSize])
        XCTAssertTrue(output.hidden.asType(.float32).asArray(Float.self).allSatisfy(\.isFinite))
    }

    private func loadConfig(bits: Int) throws -> MiniCPMMLXConfig {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("MiniCPMExactBF16SDPA-\(UUID().uuidString).json")
        defer { try? FileManager.default.removeItem(at: url) }
        let json = #"""
        {
          "model_type": "qwen3",
          "hidden_size": 32,
          "num_hidden_layers": 1,
          "intermediate_size": 32,
          "num_attention_heads": 4,
          "num_key_value_heads": 2,
          "head_dim": 8,
          "rms_norm_eps": 1e-6,
          "vocab_size": 16,
          "max_position_embeddings": 32,
          "rope_theta": 10000,
          "tie_word_embeddings": false,
          "quantization": {
            "bits": \#(bits),
            "group_size": 32,
            "mode": "affine"
          }
        }
        """#
        try Data(json.utf8).write(to: url)
        return try MiniCPMMLXConfig.load(from: url)
    }

    private func readFloat32(_ url: URL, count shape: [Int]) throws -> [Float] {
        let data: Data
        do {
            data = try Data(contentsOf: url)
        } catch {
            throw NSError(
                domain: "MiniCPMExactBF16SDPAFixture",
                code: 1,
                userInfo: [NSLocalizedDescriptionKey:
                    "missing SDPA fixture tensor \(url.path): \(error)"])
        }
        let expectedBytes = shape.reduce(1, *) * MemoryLayout<Float>.stride
        guard data.count == expectedBytes else {
            throw NSError(
                domain: "MiniCPMExactBF16SDPAFixture",
                code: 2,
                userInfo: [NSLocalizedDescriptionKey:
                    "fixture tensor \(url.lastPathComponent) has \(data.count) bytes; "
                        + "expected \(expectedBytes)"])
        }
        return data.withUnsafeBytes { raw in
            Array(raw.bindMemory(to: Float.self))
        }
    }

    private func isDirectory(_ url: URL) -> Bool {
        var value: ObjCBool = false
        return FileManager.default.fileExists(atPath: url.path, isDirectory: &value)
            && value.boolValue
    }

    private var fixtureDirectory: String {
        if let bundled = Bundle.module.url(
            forResource: "ExactBF16SDPA",
            withExtension: nil,
            subdirectory: "Fixtures") {
            return bundled.path
        }
        return URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .appendingPathComponent("Fixtures/ExactBF16SDPA", isDirectory: true)
            .path
    }
}

private extension Array where Element == Float {
    func asBF16(shape: [Int]) -> MLXArray {
        MLXArray(self, shape).asType(.bfloat16)
    }
}
