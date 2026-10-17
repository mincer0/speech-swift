import Foundation
import XCTest
import MLX
import MLXFast
import MLXNN
import MiniCPMLLM

@testable import MiniCPMTTSSemantic

final class E2EMiniCPMTTSEncodedDecodeTests: XCTestCase {
    private typealias Decode = MiniCPMTTSEncodedBF16Decode

    private func array(shape: [Int], phase: Int, divisor: Float) -> MLXArray {
        let values = (0 ..< shape.reduce(1, *)).map {
            Float(($0 * 17 + phase) % 97 - 48) / divisor
        }
        return MLXArray(values, shape).asType(.bfloat16)
    }

    private func norm(phase: Int = 0) -> RMSNorm {
        let norm = RMSNorm(dimensions: 768, eps: 1e-6)
        norm.update(parameters: ModuleParameters.unflattened([
            "weight": MLXArray((0 ..< 768).map {
                Float(0.75) + Float(($0 * 7 + phase) % 37) / 64
            }, [768]).asType(.bfloat16),
        ]))
        return norm
    }

    private func layer(phase: Int) -> Decode.Layer {
        func projection(_ output: Int, _ input: Int, _ offset: Int) -> Linear {
            MiniCPMExactBF16Linear(weight: array(
                shape: [output, input], phase: phase + offset,
                divisor: Float(input).squareRoot() * 24))
        }
        return Decode.Layer(
            inputNorm: norm(phase: phase), postNorm: norm(phase: phase + 3),
            query: projection(768, 768, 11), key: projection(768, 768, 29),
            value: projection(768, 768, 43), output: projection(768, 768, 53),
            gate: projection(3072, 768, 7), up: projection(3072, 768, 31),
            down: projection(768, 3072, 61),
            inverseFrequency: MiniCPMTTSHFStyleRoPE(dimensions: 64).inverseFrequency,
            eps: 1e-6)
    }

    private func cache(count: Int, capacity: Int, phase: Int) -> Decode.Cache {
        Decode.Cache(
            keys: array(shape: [1, 12, capacity, 64], phase: phase, divisor: 48),
            values: array(shape: [1, 12, capacity, 64], phase: phase + 23, divisor: 32),
            count: count, capacity: capacity)
    }

    private func rope(
        _ input: MLXArray, inverseFrequency: MLXArray, offset: Int
    ) -> MLXArray {
        let positions = MLXArray([Float(offset)], [1, 1])
        let frequencies = matmul(
            positions, inverseFrequency.asType(input.dtype).asType(.float32))
        let cosine = concatenated([cos(frequencies), cos(frequencies)], axis: -1)
            .asType(input.dtype).expandedDimensions(axis: 0).expandedDimensions(axis: 0)
        let sine = concatenated([sin(frequencies), sin(frequencies)], axis: -1)
            .asType(input.dtype).expandedDimensions(axis: 0).expandedDimensions(axis: 0)
        let first = input[0..., 0..., 0..., 0 ..< 32]
        let second = input[0..., 0..., 0..., 32 ..< 64]
        return input * cosine + concatenated([-second, first], axis: -1) * sine
    }

    /// The ordinary Q=1 decoder expression, including its MLX SDPA route and
    /// materialized K/V slices. This never calls the encoded helper.
    private func reference(
        input: MLXArray, layers: [Decode.Layer], caches: [Decode.Cache], finalNorm: RMSNorm
    ) -> Decode.Result {
        var hidden = input
        var updated: [Decode.Cache] = []
        for (layer, cache) in zip(layers, caches) {
            let projections = MiniCPMTTSBatchedBF16Projection.apply(
                layer.inputNorm(hidden), projections: [layer.query, layer.key, layer.value],
                environment: [:])
            let query = rope(
                projections[0].reshaped(1, 1, 12, 64).transposed(0, 2, 1, 3),
                inverseFrequency: layer.inverseFrequency, offset: cache.count)
            let key = rope(
                projections[1].reshaped(1, 1, 12, 64).transposed(0, 2, 1, 3),
                inverseFrequency: layer.inverseFrequency, offset: cache.count)
            let value = projections[2].reshaped(1, 1, 12, 64).transposed(0, 2, 1, 3)
            let keys: MLXArray
            let values: MLXArray
            if cache.count > 0, let oldKeys = cache.keys, let oldValues = cache.values {
                keys = concatenated(
                    [oldKeys[0..., 0..., 0 ..< cache.count, 0...], key], axis: 2)
                values = concatenated(
                    [oldValues[0..., 0..., 0 ..< cache.count, 0...], value], axis: 2)
            } else {
                keys = key
                values = value
            }
            let mask: MLXFast.ScaledDotProductAttentionMaskMode = cache.count > 0
                ? .none : .array(MLXArray.zeros([1, 1, 1, 1], dtype: .bfloat16))
            let attended = MLXFast.scaledDotProductAttention(
                queries: query, keys: keys, values: values, scale: 0.125, mask: mask)
                .transposed(0, 2, 1, 3).reshaped(1, 1, 768)
            let residual = hidden + layer.output(attended)
            let mlp = MiniCPMTTSBatchedBF16Projection.apply(
                layer.postNorm(residual), projections: [layer.gate, layer.up], environment: [:])
            hidden = residual + layer.down(MiniCPMExactBF16SiLU.apply(mlp[0]) * mlp[1])
            var capacity = max(32, cache.capacity)
            while capacity < cache.count + 1 { capacity *= 2 }
            let storedKeys = MLXArray.zeros([1, 12, capacity, 64], dtype: .bfloat16)
            let storedValues = MLXArray.zeros([1, 12, capacity, 64], dtype: .bfloat16)
            storedKeys[0..., 0..., 0 ..< cache.count + 1, 0...] = keys
            storedValues[0..., 0..., 0 ..< cache.count + 1, 0...] = values
            updated.append(Decode.Cache(
                keys: storedKeys, values: storedValues, count: cache.count + 1, capacity: capacity))
        }
        return Decode.Result(hidden: finalNorm(hidden), caches: updated)
    }

    private func assertBitwiseEqual(
        _ actual: MLXArray, _ expected: MLXArray,
        file: StaticString = #filePath, line: UInt = #line
    ) {
        XCTAssertEqual(actual.shape, expected.shape, file: file, line: line)
        XCTAssertEqual(actual.dtype, .bfloat16, file: file, line: line)
        XCTAssertEqual(expected.dtype, .bfloat16, file: file, line: line)
        XCTAssertEqual(
            actual.asData(access: .copy).data, expected.asData(access: .copy).data,
            file: file, line: line)
        // The returned array must also remain a valid ordinary MLX input.
        let consumed = actual + MLXArray.zeros(actual.shape, dtype: .bfloat16)
        let consumedReference = expected + MLXArray.zeros(expected.shape, dtype: .bfloat16)
        XCTAssertEqual(
            consumed.asData(access: .copy).data, consumedReference.asData(access: .copy).data,
            file: file, line: line)
    }

    private func assertEqual(
        _ actual: Decode.Result, _ expected: Decode.Result,
        file: StaticString = #filePath, line: UInt = #line
    ) throws {
        assertBitwiseEqual(actual.hidden, expected.hidden, file: file, line: line)
        XCTAssertEqual(actual.caches.count, expected.caches.count, file: file, line: line)
        for (actualCache, expectedCache) in zip(actual.caches, expected.caches) {
            XCTAssertEqual(actualCache.count, expectedCache.count, file: file, line: line)
            XCTAssertEqual(actualCache.capacity, expectedCache.capacity, file: file, line: line)
            let count = actualCache.count
            let actualKeys = try XCTUnwrap(actualCache.keys, file: file, line: line)
            let actualValues = try XCTUnwrap(actualCache.values, file: file, line: line)
            let expectedKeys = try XCTUnwrap(expectedCache.keys, file: file, line: line)
            let expectedValues = try XCTUnwrap(expectedCache.values, file: file, line: line)
            assertBitwiseEqual(
                actualKeys[0..., 0..., 0 ..< count, 0...],
                expectedKeys[0..., 0..., 0 ..< count, 0...], file: file, line: line)
            assertBitwiseEqual(
                actualValues[0..., 0..., 0 ..< count, 0...],
                expectedValues[0..., 0..., 0 ..< count, 0...], file: file, line: line)
        }
    }

    private func requireSupport() throws {
        guard Decode.isAvailable else {
            throw XCTSkip("Pinned BF16 decoder SDPA/RMS Metal library is unavailable")
        }
    }

    func testMultipleDecoderLayersAndCapacityBoundaryKeepOriginalCachesReadOnly() throws {
        try requireSupport()
        let layers = [layer(phase: 3), layer(phase: 37)]
        let finalNorm = norm(phase: 11)
        var actualCaches = [
            cache(count: 31, capacity: 32, phase: 5),
            cache(count: 31, capacity: 32, phase: 43),
        ]
        var expectedCaches = actualCaches
        let originalCaches = actualCaches
        let originalBytes = actualCaches.map {
            ($0.keys!.asData(access: .copy).data, $0.values!.asData(access: .copy).data)
        }
        var firstResult: Decode.Result?
        var firstBytes: Data?
        var firstCacheBytes: [(Data, Data)] = []
        // 31→32→33 checks exact capacity, geometric growth, non-zero RoPE
        // positions, different attention tails, and repeated graph encodes.
        for phase in [13, 59] {
            let input = array(shape: [1, 1, 768], phase: phase, divisor: 24)
            let expected = reference(
                input: input, layers: layers, caches: expectedCaches, finalNorm: finalNorm)
            let actual = try XCTUnwrap(Decode.apply(
                input: input, layers: layers, caches: actualCaches, finalNorm: finalNorm,
                forceCommandBufferRollover: phase == 59))
            try assertEqual(actual, expected)
            if firstResult == nil {
                firstResult = actual
                firstBytes = actual.hidden.asData(access: .copy).data
                firstCacheBytes = try actual.caches.map {
                    let keys = try XCTUnwrap($0.keys)
                    let values = try XCTUnwrap($0.values)
                    return (keys.asData(access: .copy).data, values.asData(access: .copy).data)
                }
            }
            actualCaches = actual.caches
            expectedCaches = expected.caches
        }
        for (index, cache) in originalCaches.enumerated() {
            XCTAssertEqual(cache.keys!.asData(access: .copy).data, originalBytes[index].0)
            XCTAssertEqual(cache.values!.asData(access: .copy).data, originalBytes[index].1)
            XCTAssertEqual(cache.count, 31)
        }
        XCTAssertEqual(firstResult?.hidden.asData(access: .copy).data, firstBytes)
        let retainedFirst = try XCTUnwrap(firstResult)
        for (cache, bytes) in zip(retainedFirst.caches, firstCacheBytes) {
            XCTAssertEqual(try XCTUnwrap(cache.keys).asData(access: .copy).data, bytes.0)
            XCTAssertEqual(try XCTUnwrap(cache.values).asData(access: .copy).data, bytes.1)
        }
        XCTAssertEqual(actualCaches.map(\.capacity), [64, 64])
    }

    func testEmptyCacheAndLongestSupportedSinglePassPrefixMatchReference() throws {
        try requireSupport()
        let layers = [layer(phase: 17)]
        let finalNorm = norm(phase: 29)
        let input = array(shape: [1, 1, 768], phase: 31, divisor: 24)
        // Build one cache from a real multi-token prefill's norm, projection
        // and RoPE, then decode its next position using the encoded route.
        let prefill = array(shape: [1, 7, 768], phase: 47, divisor: 24)
        let prefillProjections = MiniCPMTTSBatchedBF16Projection.apply(
            layers[0].inputNorm(prefill),
            projections: [layers[0].query, layers[0].key, layers[0].value],
            environment: [:])
        let prefillKey = MiniCPMTTSHFStyleRoPE(dimensions: 64)(
            prefillProjections[1].reshaped(1, 7, 12, 64).transposed(0, 2, 1, 3))
        let prefillValue = prefillProjections[2]
            .reshaped(1, 7, 12, 64).transposed(0, 2, 1, 3)
        let prefilledKeys = MLXArray.zeros([1, 12, 32, 64], dtype: .bfloat16)
        let prefilledValues = MLXArray.zeros([1, 12, 32, 64], dtype: .bfloat16)
        prefilledKeys[0..., 0..., 0 ..< 7, 0...] = prefillKey
        prefilledValues[0..., 0..., 0 ..< 7, 0...] = prefillValue
        for cache in [
            Decode.Cache(keys: nil, values: nil, count: 0, capacity: 0),
            Decode.Cache(keys: prefilledKeys, values: prefilledValues, count: 7, capacity: 32),
            self.cache(count: 1022, capacity: 1024, phase: 19),
        ] {
            let expected = reference(
                input: input, layers: layers, caches: [cache], finalNorm: finalNorm)
            let actual = try XCTUnwrap(Decode.apply(
                input: input, layers: layers, caches: [cache], finalNorm: finalNorm))
            try assertEqual(actual, expected)
        }
    }

    func testLongPrefixDispatchBoundariesMatchReferenceAndKeepCachesReadOnly() throws {
        try requireSupport()
        let layers = [layer(phase: 17), layer(phase: 53)]
        let finalNorm = norm(phase: 29)
        // N includes this decode's appended token. N=1024 switches d/s
        // devices to the pinned two-pass kernel; 1025 also grows the cache.
        // The same partial/sums/maxs buffers are reused across both layers.
        for length in [1024, 1025, 2048, 4096] {
            var capacity = 32
            while capacity < length - 1 { capacity *= 2 }
            let caches = [
                cache(count: length - 1, capacity: capacity, phase: 19),
                cache(count: length - 1, capacity: capacity, phase: 43),
            ]
            let originalBytes = try caches.map { cache in
                let keys = try XCTUnwrap(cache.keys)
                let values = try XCTUnwrap(cache.values)
                return (keys.asData(access: .copy).data, values.asData(access: .copy).data)
            }
            let input = array(shape: [1, 1, 768], phase: length % 97, divisor: 24)
            let expected = reference(
                input: input, layers: layers, caches: caches, finalNorm: finalNorm)
            let started = DispatchTime.now().uptimeNanoseconds
            let actual = try XCTUnwrap(Decode.apply(
                input: input, layers: layers, caches: caches, finalNorm: finalNorm,
                forceCommandBufferRollover: length == 1025))
            let elapsed = Double(DispatchTime.now().uptimeNanoseconds - started) / 1e6
            print("encoded_decode_long_context N=\(length) layers=\(layers.count) apply_ms=\(elapsed)")
            try assertEqual(actual, expected)
            for (cache, bytes) in zip(caches, originalBytes) {
                XCTAssertEqual(try XCTUnwrap(cache.keys).asData(access: .copy).data, bytes.0)
                XCTAssertEqual(try XCTUnwrap(cache.values).asData(access: .copy).data, bytes.1)
                XCTAssertEqual(cache.count, length - 1)
            }
        }
    }

    func testParameterUpdatesAreVisibleWithoutMutatingEarlierResult() throws {
        try requireSupport()
        let layer = layer(phase: 23)
        let finalNorm = norm(phase: 41)
        let caches = [cache(count: 7, capacity: 32, phase: 11)]
        let input = array(shape: [1, 1, 768], phase: 41, divisor: 24)
        let original = try XCTUnwrap(Decode.apply(
            input: input, layers: [layer], caches: caches, finalNorm: finalNorm))
        let originalHidden = original.hidden.asData(access: .copy).data
        let originalKeys = original.caches[0].keys!.asData(access: .copy).data
        for (index, projection) in [
            layer.query, layer.key, layer.value, layer.output, layer.gate, layer.up, layer.down,
        ].enumerated() {
            try projection.update(parameters: ModuleParameters.unflattened([
                "weight": array(
                    shape: projection.weight.shape, phase: index * 13 + 67,
                    divisor: Float(projection.weight.dim(1)).squareRoot() * 24),
            ]), verify: .all)
        }
        for (index, norm) in [layer.inputNorm, layer.postNorm, finalNorm].enumerated() {
            try norm.update(parameters: ModuleParameters.unflattened([
                "weight": array(shape: [768], phase: index * 17 + 41, divisor: 64),
            ]), verify: .all)
        }
        let actual = try XCTUnwrap(Decode.apply(
            input: input, layers: [layer], caches: caches, finalNorm: finalNorm))
        let expected = reference(
            input: input, layers: [layer], caches: caches, finalNorm: finalNorm)
        try assertEqual(actual, expected)
        XCTAssertNotEqual(actual.hidden.asData(access: .copy).data, originalHidden)
        XCTAssertEqual(original.hidden.asData(access: .copy).data, originalHidden)
        XCTAssertEqual(original.caches[0].keys!.asData(access: .copy).data, originalKeys)
    }

    func testUnsupportedConfigurationReturnsNilBeforeChangingInputCaches() throws {
        let layer = layer(phase: 7)
        let finalNorm = norm()
        let input = array(shape: [1, 1, 768], phase: 13, divisor: 24)
        let cache = cache(count: 3, capacity: 32, phase: 31)
        let original = cache.keys!.asData(access: .copy).data
        func apply(
            input overrideInput: MLXArray? = nil, layers: [Decode.Layer]? = nil,
            caches: [Decode.Cache]? = nil, final overrideFinal: RMSNorm? = nil
        ) -> Decode.Result? {
            Decode.apply(
                input: overrideInput ?? input, layers: layers ?? [layer],
                caches: caches ?? [cache], finalNorm: overrideFinal ?? finalNorm)
        }
        XCTAssertNil(apply(input: input.asType(.float32)))
        XCTAssertNil(apply(input: array(shape: [1, 2, 768], phase: 3, divisor: 24)))
        XCTAssertNil(apply(input: array(shape: [2, 1, 768], phase: 3, divisor: 24)))
        XCTAssertNil(apply(layers: []))
        XCTAssertNil(apply(caches: []))
        XCTAssertNil(apply(final: RMSNorm(dimensions: 768, eps: 1e-6)))
        XCTAssertNil(apply(caches: [Decode.Cache(
            keys: nil, values: cache.values, count: 0, capacity: 0)]))
        XCTAssertNil(apply(caches: [Decode.Cache(
            keys: cache.keys, values: cache.values, count: 3, capacity: 64)]))
        let tooLong = self.cache(count: 4096, capacity: 4096, phase: 19)
        XCTAssertNil(apply(caches: [tooLong]))
        let unsupported = Decode.Layer(
            inputNorm: layer.inputNorm, postNorm: layer.postNorm,
            query: Linear(weight: layer.query.weight), key: layer.key, value: layer.value,
            output: layer.output, gate: layer.gate, up: layer.up, down: layer.down,
            inverseFrequency: layer.inverseFrequency, eps: layer.eps)
        XCTAssertNil(apply(layers: [unsupported]))
        let unsupportedFrequency = Decode.Layer(
            inputNorm: layer.inputNorm, postNorm: layer.postNorm,
            query: layer.query, key: layer.key, value: layer.value,
            output: layer.output, gate: layer.gate, up: layer.up, down: layer.down,
            inverseFrequency: layer.inverseFrequency.asType(.bfloat16), eps: layer.eps)
        XCTAssertNil(apply(layers: [unsupportedFrequency]))
        let exactNorm = MiniCPMExactBF16RMSNorm(dimensions: 768, eps: 1e-6)
        exactNorm.update(parameters: ModuleParameters.unflattened([
            "weight": layer.inputNorm.weight,
        ]))
        XCTAssertNil(apply(final: exactNorm))
        XCTAssertEqual(cache.keys!.asData(access: .copy).data, original)
        XCTAssertEqual(cache.count, 3)
    }
}
