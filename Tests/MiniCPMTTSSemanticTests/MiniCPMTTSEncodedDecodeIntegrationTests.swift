import XCTest
import MLX
import MLXNN
import MLXRandom

@testable import MiniCPMTTSSemantic

final class E2EMiniCPMTTSEncodedDecodeIntegrationTests: XCTestCase {
    private func input(_ tokens: Int, phase: Int) -> MLXArray {
        MLXArray((0 ..< tokens * 768).map {
            Float(($0 * 17 + phase) % 97 - 48) / 64
        }, [1, tokens, 768]).asType(.bfloat16)
    }

    private func assertEqual(
        _ actual: MLXArray, _ expected: MLXArray,
        file: StaticString = #filePath, line: UInt = #line
    ) {
        XCTAssertEqual(actual.shape, expected.shape, file: file, line: line)
        XCTAssertEqual(actual.dtype, expected.dtype, file: file, line: line)
        XCTAssertEqual(actual.asData(access: .copy).data,
                       expected.asData(access: .copy).data, file: file, line: line)
    }

    private func assertCachesEqual(
        _ actual: MiniCPMTTSSession, _ expected: MiniCPMTTSSession,
        file: StaticString = #filePath, line: UInt = #line
    ) throws {
        for (actual, expected) in zip(actual.layerCaches, expected.layerCaches) {
            XCTAssertEqual(actual.count, expected.count, file: file, line: line)
            XCTAssertEqual(actual.capacity, expected.capacity, file: file, line: line)
            XCTAssertEqual(actual.capacityGrowthCount, expected.capacityGrowthCount,
                           file: file, line: line)
            assertEqual(try XCTUnwrap(actual.keys), try XCTUnwrap(expected.keys),
                        file: file, line: line)
            assertEqual(try XCTUnwrap(actual.values), try XCTUnwrap(expected.values),
                        file: file, line: line)
        }
    }

    func testCommittedDecodeGrowthRollbackAndLaterPrefillMatchCanonical() throws {
        guard MiniCPMTTSEncodedBF16Decode.isAvailable else {
            throw XCTSkip("Pinned decoder Metal pipelines unavailable")
        }
        MLXRandom.seed(19)
        let decoder = MiniCPMTTSAudioDecoder(.init(numHiddenLayers: 2))
        try decoder.update(parameters: ModuleParameters.unflattened(
            decoder.parameters().flattened().map { ($0.0, $0.1.asType(.bfloat16)) }),
            verify: .all)
        let actual = MiniCPMTTSSession(layerCount: 2)
        let expected = MiniCPMTTSSession(layerCount: 2)
        let prefill = input(31, phase: 7)
        assertEqual(
            decoder(prefill, caches: actual.layerCaches, offset: 0, encodedDecode: true),
            decoder(prefill, caches: expected.layerCaches, offset: 0, encodedDecode: false))
        XCTAssertEqual(decoder.encodedDecodeCount, 0)
        try assertCachesEqual(actual, expected)
        let snapshot = actual.snapshot()
        let expectedSnapshot = expected.snapshot()

        for phase in [23, 47] {
            let next = input(1, phase: phase)
            assertEqual(
                decoder(next, caches: actual.layerCaches, offset: actual.cacheOffset,
                        encodedDecode: true),
                decoder(next, caches: expected.layerCaches, offset: expected.cacheOffset,
                        encodedDecode: false))
            try assertCachesEqual(actual, expected)
        }
        XCTAssertEqual(decoder.encodedDecodeCount, 2)
        XCTAssertEqual(actual.layerCaches.map(\.capacity), [64, 64])
        for _ in 0 ..< 2 {
            try actual.rollback(to: snapshot)
            try expected.rollback(to: expectedSnapshot)
            XCTAssertEqual(actual.cacheOffset, 31)
            let next = input(1, phase: 71)
            assertEqual(
                decoder(next, caches: actual.layerCaches, offset: actual.cacheOffset,
                        encodedDecode: true),
                decoder(next, caches: expected.layerCaches, offset: expected.cacheOffset,
                        encodedDecode: false))
            try assertCachesEqual(actual, expected)
        }
        XCTAssertEqual(decoder.encodedDecodeCount, 4)
        // A new multi-token condition must consume the committed cache using
        // the established prefill route, with no duplicate count advancement.
        let laterPrefill = input(2, phase: 83)
        assertEqual(
            decoder(laterPrefill, caches: actual.layerCaches, offset: actual.cacheOffset,
                    encodedDecode: true),
            decoder(laterPrefill, caches: expected.layerCaches, offset: expected.cacheOffset,
                    encodedDecode: false))
        XCTAssertEqual(decoder.encodedDecodeCount, 4)
        XCTAssertEqual(actual.cacheOffset, 34)
        try assertCachesEqual(actual, expected)

        // Once a cache exists, canonical attention takes RoPE position from
        // its length even when the caller retains an earlier chunk offset.
        let shiftedPosition = actual.cacheOffset + 7
        let shifted = input(1, phase: 89)
        assertEqual(
            decoder(shifted, caches: actual.layerCaches, offset: shiftedPosition,
                    encodedDecode: true),
            decoder(shifted, caches: expected.layerCaches, offset: shiftedPosition,
                    encodedDecode: false))
        XCTAssertEqual(decoder.encodedDecodeCount, 5,
                       "an existing cache takes precedence over the caller's offset")
        XCTAssertEqual(actual.cacheOffset, 35)
        try assertCachesEqual(actual, expected)
    }
}
