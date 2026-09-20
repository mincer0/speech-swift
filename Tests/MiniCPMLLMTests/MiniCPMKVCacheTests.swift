import XCTest
import Foundation
import Metal
import MLX
import MLXCommon
@testable import MiniCPMLLM

/// Regression tests for the MiniCPM cache geometry.  These deliberately use
/// the cache helper directly instead of loading the 8B checkpoint, so they run
/// quickly and still catch the two failures that matter for a long duplex
/// session: copying the whole prefix on every token and exposing spare
/// capacity to SDPA.
final class MiniCPMKVCacheTests: XCTestCase {
    private func tinyConfig() -> MiniCPMMLXConfig {
        MiniCPMMLXConfig(
            hiddenSize: 4,
            numHiddenLayers: 1,
            numAttentionHeads: 1,
            numKeyValueHeads: 1,
            headDim: 4,
            intermediateSize: 4,
            vocabSize: 32,
            maxPositionEmbeddings: 64,
            ropeTheta: 10_000,
            rmsNormEps: 1e-6,
            tieWordEmbeddings: false,
            eosTokenId: 0,
            quantGroupSize: 32,
            quantBits: 0,
            quantMode: "none")
    }

    private func token(_ value: Float) -> MLXArray {
        MLXArray([value, value + 0.5]).reshaped([1, 1, 1, 2])
    }

    private func flattenedValues(_ cache: MiniCPMKVCache) -> [Float] {
        let (_, values) = cache.window()
        eval(values)
        return values.asType(.float32).asArray(Float.self)
    }

    /// Independent float32 RoPE reference for cache-surgery assertions. The
    /// production forward helper intentionally stores inverse frequencies as
    /// BF16; sliding-window reindexing must instead follow the pinned
    /// utility's direct-float32 inverse-frequency path.
    private func referenceRoPE(
        _ input: MLXArray,
        start: Int,
        ropeTheta: Float
    ) -> MLXArray {
        let headDim = input.dim(3)
        let half = headDim / 2
        let length = input.dim(2)
        let inverse = MLXArray((0 ..< half).map { index in
            Float(pow(Double(ropeTheta), -Double(index) / Double(half)))
        }, [1, half])
        let positions = MLXArray(
            (0 ..< length).map { Float(start + $0) }, [length, 1])
        let angles = matmul(positions, inverse)
        let cosines = concatenated([cos(angles), cos(angles)], axis: -1)
            .asType(input.dtype)
            .expandedDimensions(axis: 0)
            .expandedDimensions(axis: 0)
        let sines = concatenated([sin(angles), sin(angles)], axis: -1)
            .asType(input.dtype)
            .expandedDimensions(axis: 0)
            .expandedDimensions(axis: 0)
        let first = input[0..., 0..., 0..., 0 ..< half]
        let second = input[0..., 0..., 0..., half ..< headDim]
        let rotated = concatenated([-second, first], axis: -1)
        return input * cosines + rotated * sines
    }

    private func referenceReindex(
        _ input: MLXArray,
        oldStart: Int,
        newStart: Int,
        ropeTheta: Float
    ) -> MLXArray {
        let headDim = input.dim(3)
        let half = headDim / 2
        let length = input.dim(2)
        let inverse = MLXArray((0 ..< half).map { index in
            Float(pow(Double(ropeTheta), -Double(index) / Double(half)))
        }, [1, half])

        func cosSin(start: Int) -> (MLXArray, MLXArray) {
            let positions = MLXArray(
                (0 ..< length).map { Float(start + $0) }, [length, 1])
            let angles = matmul(positions, inverse)
            let cosines = concatenated([cos(angles), cos(angles)], axis: -1)
                .asType(input.dtype)
                .expandedDimensions(axis: 0)
                .expandedDimensions(axis: 0)
            let sines = concatenated([sin(angles), sin(angles)], axis: -1)
                .asType(input.dtype)
                .expandedDimensions(axis: 0)
                .expandedDimensions(axis: 0)
            return (cosines, sines)
        }

        func rotateHalf(_ value: MLXArray) -> MLXArray {
            let first = value[0..., 0..., 0..., 0 ..< half]
            let second = value[0..., 0..., 0..., half ..< headDim]
            return concatenated([-second, first], axis: -1)
        }

        let (oldCos, oldSin) = cosSin(start: oldStart)
        let unrotated = oldCos * input - oldSin * rotateHalf(input)
        let (newCos, newSin) = cosSin(start: newStart)
        return newCos * unrotated + newSin * rotateHalf(unrotated)
    }

    func testWindowExcludesSpareCapacityAndGrowthIsGeometric() {
        let first = MiniCPMKVCache(keys: token(1), values: token(1))
        XCTAssertEqual(first.count, 1)
        XCTAssertEqual(first.capacity, 1)

        // The first append grows once, then leaves a 512-token runway.  The
        // logical window must contain exactly two tokens, not the zero-filled
        // spare entries.
        let second = first.appending(keys: token(2), values: token(2))
        XCTAssertEqual(first.count, 1, "append must not move the old view")
        XCTAssertEqual(second.count, 2)
        XCTAssertGreaterThanOrEqual(second.capacity, 513)
        XCTAssertEqual(second.growthCount, 1)
        XCTAssertEqual(flattenedValues(second), [1, 1.5, 2, 2.5])
    }

    func testSnapshotRollbackMovesLogicalBoundaryWithoutCopyingPrefix() {
        let initial = MiniCPMKVCache(keys: token(10), values: token(10))
        let committed = initial.appending(keys: token(11), values: token(11))
        let snapshotCount = committed.count
        let snapshotGrowth = committed.growthCount

        // This write lands in spare capacity.  The snapshot's count remains
        // two and its valid prefix is unchanged; rollback therefore only has
        // to restore the old view metadata.
        let speculative = committed.appending(keys: token(12), values: token(12))
        XCTAssertEqual(committed.count, snapshotCount)
        XCTAssertEqual(committed.growthCount, snapshotGrowth)
        XCTAssertEqual(flattenedValues(committed), [10, 10.5, 11, 11.5])
        XCTAssertEqual(speculative.count, snapshotCount + 1)
        XCTAssertEqual(flattenedValues(speculative), [10, 10.5, 11, 11.5, 12, 12.5])
    }

    func testLongAppendUsesFewCapacityGrowths() {
        var cache = MiniCPMKVCache(keys: token(0), values: token(0))
        for index in 1 ..< 4096 {
            let value = Float(index)
            cache = cache.appending(keys: token(value), values: token(value))
        }

        XCTAssertEqual(cache.count, 4096)
        XCTAssertGreaterThanOrEqual(cache.capacity, cache.count)
        // A doubling strategy should need only a handful of reallocations,
        // not one per token as concatenation-based caches do.
        XCTAssertLessThan(cache.growthCount, 16)
    }

    func testLegacyTupleStateStillReportsTheLogicalLength() {
        let keys = token(3)
        let values = token(4)
        let state = MiniCPMInferenceState(
            kvCaches: [(keys, values)], position: 1)

        XCTAssertEqual(state.sequenceLength, 1)
        XCTAssertEqual(state.kvCaches.count, 1)
        XCTAssertEqual(state.kvCaches[0]?.0.dim(2), 1)
        XCTAssertEqual(state.cacheGrowthCount, 0)
    }

    func testSpareCapacityWindowMatchesDirectSDPA() {
        // Keep every token/head/channel distinct so a bad stride write cannot
        // hide behind repeated values.  The cache starts with one token,
        // grows while appending the second, then writes the final two tokens
        // into spare capacity: that is the path that used to expose the
        // wrong sequence geometry to SDPA.
        let query = MLXArray([
            Float(0.25), -0.5, 0.75, -1.0,
            1.25, -1.5, 1.75, -2.0,
        ], [1, 2, 1, 4]).asType(.bfloat16)
        let keyValues: [Float] = (0 ..< 24).map { index in
            Float(index + 1) / 8
        }
        let valueValues: [Float] = (0 ..< 24).map { index in
            Float((index * 5) % 23 - 11) / 7
        }
        // Attention projections arrive as a transpose of [B,T,H,D], not a
        // row-contiguous [B,H,T,D] allocation.  Keep that stride pattern in
        // the fixture so the regression exercises the real assignment path.
        let fullKeys = MLXArray(keyValues, [1, 3, 2, 4])
            .asType(.bfloat16).transposed(0, 2, 1, 3)
        let fullValues = MLXArray(valueValues, [1, 3, 2, 4])
            .asType(.bfloat16).transposed(0, 2, 1, 3)
        let scale = 1 / sqrt(Float(query.dim(3)))

        let direct = SDPA.attendAndMerge(
            qHeads: query,
            kHeads: fullKeys,
            vHeads: fullValues,
            scale: scale)

        let prefixKeys = fullKeys[0..., 0..., 0 ..< 1, 0...]
        let prefixValues = fullValues[0..., 0..., 0 ..< 1, 0...]
        let firstSuffixKeys = fullKeys[0..., 0..., 1 ..< 2, 0...]
        let firstSuffixValues = fullValues[0..., 0..., 1 ..< 2, 0...]
        let finalSuffixKeys = fullKeys[0..., 0..., 2 ..< 3, 0...]
        let finalSuffixValues = fullValues[0..., 0..., 2 ..< 3, 0...]
        let cache = MiniCPMKVCache(keys: prefixKeys, values: prefixValues)
            .appending(keys: firstSuffixKeys, values: firstSuffixValues)
            .appending(keys: finalSuffixKeys, values: finalSuffixValues)
        let (windowKeys, windowValues) = cache.window()

        eval(direct, fullKeys, fullValues, windowKeys, windowValues)
        XCTAssertEqual(cache.count, 3)
        XCTAssertEqual(windowKeys.shape, fullKeys.shape)
        XCTAssertEqual(windowValues.shape, fullValues.shape)
        XCTAssertEqual(
            windowKeys.asType(.float32).asArray(Float.self),
            fullKeys.asType(.float32).asArray(Float.self))
        XCTAssertEqual(
            windowValues.asType(.float32).asArray(Float.self),
            fullValues.asType(.float32).asArray(Float.self))

        let cached = SDPA.attendAndMerge(
            qHeads: query,
            kHeads: windowKeys,
            vHeads: windowValues,
            scale: scale)
        let difference = MLX.abs(direct - cached).max()
        eval(cached, difference)
        XCTAssertLessThan(
            difference.item(Float.self), 1e-3,
            "SDPA output changed when keys/values came from the cache window")
    }

    func testSpareCapacityWindowMatchesExactBF16ProjectionOutput() throws {
        guard MTLCreateSystemDefaultDevice() != nil else {
            throw XCTSkip("Metal GPU device is unavailable")
        }

        var identity = Array(repeating: Float(0), count: 64)
        for index in 0 ..< 8 {
            identity[index * 8 + index] = 1
        }
        let weight = MLXArray(identity, [8, 8]).asType(.bfloat16)
        let keyProjection = MiniCPMExactBF16Linear(weight: weight, bias: nil)
        let valueProjection = MiniCPMExactBF16Linear(weight: weight, bias: nil)
        let features = MLXArray((0 ..< 24).map { index in
            Float((index * 3) % 17 - 8) / 5
        }, [1, 3, 8]).asType(.bfloat16)
        let sequenceKeys = keyProjection(features)
        let sequenceValues = valueProjection(features)
        let fullKeys = sequenceKeys.reshaped(1, 3, 2, 4).transposed(0, 2, 1, 3)
        let fullValues = sequenceValues.reshaped(1, 3, 2, 4).transposed(0, 2, 1, 3)
        let query = fullKeys[0..., 0..., 2 ..< 3, 0...]
        let scale = 1 / sqrt(Float(query.dim(3)))

        let direct = SDPA.attendAndMerge(
            qHeads: query, kHeads: fullKeys, vHeads: fullValues, scale: scale)
        let cache = MiniCPMKVCache(
            keys: fullKeys[0..., 0..., 0 ..< 1, 0...],
            values: fullValues[0..., 0..., 0 ..< 1, 0...])
            .appending(
                keys: fullKeys[0..., 0..., 1 ..< 2, 0...],
                values: fullValues[0..., 0..., 1 ..< 2, 0...])
            .appending(
                keys: fullKeys[0..., 0..., 2 ..< 3, 0...],
                values: fullValues[0..., 0..., 2 ..< 3, 0...])
        let (windowKeys, windowValues) = cache.window()
        let cached = SDPA.attendAndMerge(
            qHeads: query, kHeads: windowKeys, vHeads: windowValues, scale: scale)
        let difference = MLX.abs(direct - cached).max()
        eval(direct, cached, difference)
        XCTAssertEqual(windowKeys.shape, fullKeys.shape)
        XCTAssertLessThan(
            difference.item(Float.self), 1e-3,
            "SDPA output changed for exact BF16 projection-backed cache values")
    }

    func testBasicDropPreservesValuesReindexesSuffixAndSnapshots() throws {
        let headDim = 4
        let ropeTheta: Float = 10_000
        let rawKeys = MLXArray(
            (0 ..< 16).map { Float($0 + 1) / 7 }, [1, 1, 4, headDim])
            .asType(.bfloat16)
        let values = MLXArray(
            (0 ..< 16).map { Float($0 + 100) }, [1, 1, 4, headDim])
            .asType(.bfloat16)
        let encodedKeys = referenceRoPE(rawKeys, start: 0, ropeTheta: ropeTheta)
        let cache = MiniCPMKVCache(keys: encodedKeys, values: values)
        let oldKeys = encodedKeys.asType(.float32).asArray(Float.self)
        let oldValues = values.asType(.float32).asArray(Float.self)

        // Remove the middle token at position one. The suffix originally at
        // positions two and three must move to positions one and two, while
        // all V vectors remain byte-for-byte unchanged.
        let compacted = try XCTUnwrap(cache.dropping(
            preserve: 1, length: 1, ropeTheta: ropeTheta))
        let (compactedKeys, compactedValues) = compacted.window()
        let expectedKeys = concatenated([
            encodedKeys[0..., 0..., 0 ..< 1, 0...],
            referenceReindex(
                encodedKeys[0..., 0..., 2 ..< 4, 0...],
                oldStart: 2,
                newStart: 1,
                ropeTheta: ropeTheta),
        ], axis: 2)
        let expectedValues = concatenated([
            values[0..., 0..., 0 ..< 1, 0...],
            values[0..., 0..., 2 ..< 4, 0...],
        ], axis: 2)
        eval(compactedKeys, compactedValues, expectedKeys, expectedValues)
        XCTAssertEqual(compacted.count, 3)
        XCTAssertEqual(
            compactedKeys.asType(.float32).asArray(Float.self),
            expectedKeys.asType(.float32).asArray(Float.self))
        XCTAssertEqual(
            compactedValues.asType(.float32).asArray(Float.self),
            expectedValues.asType(.float32).asArray(Float.self))

        // Surgery allocates fresh storage: a pre-surgery snapshot must retain
        // both its original logical length and its original tensor contents.
        let (snapshotKeys, snapshotValues) = cache.window()
        eval(snapshotKeys, snapshotValues)
        XCTAssertEqual(cache.count, 4)
        XCTAssertEqual(snapshotKeys.asType(.float32).asArray(Float.self), oldKeys)
        XCTAssertEqual(snapshotValues.asType(.float32).asArray(Float.self), oldValues)

        // A second contiguous deletion must reindex the remaining suffix from
        // its already-compressed position, not from the original position.
        let compactedTwice = try XCTUnwrap(compacted.dropping(
            preserve: 1, length: 1, ropeTheta: ropeTheta))
        let (twiceKeys, twiceValues) = compactedTwice.window()
        let expectedTwiceKeys = concatenated([
            encodedKeys[0..., 0..., 0 ..< 1, 0...],
            referenceReindex(
                compactedKeys[0..., 0..., 2 ..< 3, 0...],
                oldStart: 2,
                newStart: 1,
                ropeTheta: ropeTheta),
        ], axis: 2)
        let expectedTwiceValues = concatenated([
            values[0..., 0..., 0 ..< 1, 0...],
            values[0..., 0..., 3 ..< 4, 0...],
        ], axis: 2)
        eval(twiceKeys, twiceValues, expectedTwiceKeys, expectedTwiceValues)
        XCTAssertEqual(compactedTwice.count, 2)
        XCTAssertEqual(
            twiceKeys.asType(.float32).asArray(Float.self),
            expectedTwiceKeys.asType(.float32).asArray(Float.self))
        XCTAssertEqual(
            twiceValues.asType(.float32).asArray(Float.self),
            expectedTwiceValues.asType(.float32).asArray(Float.self))

        // The public state surgery carries the same value-semantic guarantee
        // and compresses position alongside cache length.
        let config = MiniCPMMLXConfig(
            hiddenSize: 4,
            numHiddenLayers: 1,
            numAttentionHeads: 1,
            numKeyValueHeads: 1,
            headDim: 4,
            intermediateSize: 4,
            vocabSize: 8,
            maxPositionEmbeddings: 32,
            ropeTheta: ropeTheta,
            rmsNormEps: 1e-6,
            tieWordEmbeddings: false,
            eosTokenId: 0,
            quantGroupSize: 32,
            quantBits: 0,
            quantMode: "none")
        let model = MiniCPMLanguageModel(config: config)
        let state = MiniCPMInferenceState(
            kvCaches: [(encodedKeys, values)], position: 4)
        let compactedState = try XCTUnwrap(model.droppingState(
            state, preserveLength: 1, droppingLength: 1))
        XCTAssertEqual(state.sequenceLength, 4)
        XCTAssertEqual(state.position, 4)
        XCTAssertEqual(compactedState.sequenceLength, 3)
        XCTAssertEqual(compactedState.position, 3)
        XCTAssertEqual(
            state.kvCaches[0]!.0.asType(.float32).asArray(Float.self), oldKeys)
        XCTAssertEqual(
            state.kvCaches[0]!.1.asType(.float32).asArray(Float.self), oldValues)
    }

    func testBasicWindowSessionDropsMultipleUnitsWithoutReplay() {
        let config = MiniCPMMLXConfig(
            hiddenSize: 4,
            numHiddenLayers: 1,
            numAttentionHeads: 1,
            numKeyValueHeads: 1,
            headDim: 4,
            intermediateSize: 4,
            vocabSize: 8,
            maxPositionEmbeddings: 32,
            ropeTheta: 10_000,
            rmsNormEps: 1e-6,
            tieWordEmbeddings: false,
            eosTokenId: 0,
            quantGroupSize: 32,
            quantBits: 0,
            quantMode: "none")
        let model = MiniCPMLanguageModel(config: config)
        let session = MiniCPMSession(model: model)
        session.configureSlidingWindow(MiniCPMSlidingWindowConfig(
            mode: "basic", basicHighTokens: 3, basicLowTokens: 2))
        let embedding = MLXArray([Float(1), 2, 3, 4], [1, 1, 4]).asType(.bfloat16)

        session.beginSystemPrompt()
        _ = session.feed(embeddings: embedding)
        session.finishSystemPrompt()
        for _ in 0 ..< 3 {
            session.beginUnit()
            _ = session.feed(embeddings: embedding)
            session.commitUnit()
        }
        let snapshot = session.snapshot()
        XCTAssertEqual(session.cacheLength, 4)
        XCTAssertEqual(session.units.map(\.unitID), [0, 1, 2])

        XCTAssertTrue(session.enforceSlidingWindow())
        XCTAssertEqual(session.cacheLength, 2)
        XCTAssertEqual(session.state.position, 2)
        XCTAssertEqual(session.units.map(\.unitID), [2])
        XCTAssertEqual(session.windowStats()["system_preserve_length"] as? Int, 1)

        // The snapshot captured before surgery remains a four-token state and
        // can restore the pre-window metadata/cache view exactly.
        XCTAssertEqual(snapshot.inference.sequenceLength, 4)
        XCTAssertEqual(snapshot.inference.position, 4)
        session.restore(snapshot)
        XCTAssertEqual(session.cacheLength, 4)
        XCTAssertEqual(session.units.map(\.unitID), [0, 1, 2])
    }

    func testInterruptedTurnReplacesFinalizedTerminatorInReplayMetadata() {
        let model = MiniCPMLanguageModel(config: tinyConfig())
        let session = MiniCPMSession(model: model)
        let system = MLXArray([Float(1), 2, 3, 4], [1, 1, 4]).asType(.bfloat16)
        let input = MLXArray([Float(-1), -2, -3, -4], [1, 1, 4]).asType(.bfloat16)
        let finalizedMarkers = MLXArray(
            [Float(5), 6, 7, 8, 9, 10, 11, 12], [1, 2, 4]).asType(.bfloat16)

        session.beginSystemPrompt()
        _ = session.feed(embeddings: system)
        session.finishSystemPrompt()
        session.beginUnit()
        _ = session.feed(embeddings: input)
        _ = session.feed(embeddings: finalizedMarkers)
        session.recordProtocolToken(42, special: true)
        session.recordProtocolToken(2, special: true)
        session.commitUnit()

        XCTAssertEqual(session.protocolTokens, [42, 2])
        XCTAssertEqual(session.cacheLength, 4)

        XCTAssertEqual(
            session.closeInterruptedTurn(turnEosId: 10, unitEndId: 2),
            1)
        // The old chunk/listen terminator and </unit> were one KV embedding;
        // replay metadata must replace both with turn_eos + </unit>.
        XCTAssertEqual(session.protocolTokens, [10, 2])
        XCTAssertEqual(session.cacheLength, 4)
    }

    func testContextRebuildCopiesTailReindexesAndPreservesSnapshot() throws {
        let config = tinyConfig()
        let model = MiniCPMLanguageModel(config: config)
        let rawKeys = MLXArray(
            (0 ..< 20).map { Float($0 + 1) / 7 }, [1, 1, 5, 4])
            .asType(.bfloat16)
        let oldKeys = referenceRoPE(rawKeys, start: 0, ropeTheta: config.ropeTheta)
        let oldValues = MLXArray(
            (0 ..< 20).map { Float($0 + 100) }, [1, 1, 5, 4])
            .asType(.bfloat16)
        let state = MiniCPMInferenceState(
            kvCaches: [(oldKeys, oldValues)], position: 5)
        let segment = MLXArray([Float(0.25), 0.5, -0.75, 1.0], [1, 1, 4])
            .asType(.bfloat16)

        let rebuilt = try XCTUnwrap(model.rebuildingContextState(
            state,
            prefixLength: 1,
            segmentEmbeddings: segment,
            retainedLength: 2))
        let (newKeys, newValues) = try XCTUnwrap(rebuilt.kvCaches[0])
        let expectedTailKeys = referenceReindex(
            oldKeys[0..., 0..., 3 ..< 5, 0...],
            oldStart: 3,
            newStart: 2,
            ropeTheta: config.ropeTheta)
        let expectedTailValues = oldValues[0..., 0..., 3 ..< 5, 0...]
        eval(newKeys, newValues, expectedTailKeys, expectedTailValues)

        XCTAssertEqual(rebuilt.sequenceLength, 4)
        XCTAssertEqual(rebuilt.position, 4)
        XCTAssertEqual(
            newValues[0..., 0..., 0 ..< 1, 0...]
                .asType(.float32).asArray(Float.self),
            oldValues[0..., 0..., 0 ..< 1, 0...]
                .asType(.float32).asArray(Float.self))
        XCTAssertEqual(
            newValues[0..., 0..., 2 ..< 4, 0...]
                .asType(.float32).asArray(Float.self),
            expectedTailValues.asType(.float32).asArray(Float.self))
        XCTAssertEqual(
            newKeys[0..., 0..., 2 ..< 4, 0...]
                .asType(.float32).asArray(Float.self),
            expectedTailKeys.asType(.float32).asArray(Float.self))

        // The middle segment is recomputed from the new embedding; it is not
        // copied out of the old [prefix][old segment] cache.
        let oldMiddleValues = oldValues[0..., 0..., 1 ..< 2, 0...]
            .asType(.float32).asArray(Float.self)
        let newSegmentValues = newValues[0..., 0..., 1 ..< 2, 0...]
            .asType(.float32).asArray(Float.self)
        XCTAssertNotEqual(newSegmentValues, oldMiddleValues)

        // The old state remains a valid snapshot after prefix forwarding and
        // final-cache concatenation.
        XCTAssertEqual(state.sequenceLength, 5)
        XCTAssertEqual(state.position, 5)
        XCTAssertEqual(
            state.kvCaches[0]!.0.asType(.float32).asArray(Float.self),
            oldKeys.asType(.float32).asArray(Float.self))
        XCTAssertEqual(
            state.kvCaches[0]!.1.asType(.float32).asArray(Float.self),
            oldValues.asType(.float32).asArray(Float.self))
    }

    func testContextWindowEvictsContinuouslyAndRebuildsPreviousAndExtras() {
        let model = MiniCPMLanguageModel(config: tinyConfig())
        let session = MiniCPMSession(model: model)
        session.configureSlidingWindow(MiniCPMSlidingWindowConfig(
            mode: "context",
            contextPreviousMaxTokens: 8,
            contextMaxUnits: 1,
            contextPreviousMarkerTokenIDs: [9, 10]))
        let prefix = MLXArray([Float(1), 2, 3, 4], [1, 1, 4]).asType(.bfloat16)
        let suffix = MLXArray([Float(5), 6, 7, 8], [1, 1, 4]).asType(.bfloat16)
        let extra = MLXArray([Float(9), 10, 11, 12], [1, 1, 4]).asType(.bfloat16)
        let unitEmbedding = MLXArray([Float(-1), -2, -3, -4], [1, 1, 4])
            .asType(.bfloat16)

        session.beginSystemPrompt()
        _ = session.feed(embeddings: prefix)
        session.markSystemPromptSuffix()
        _ = session.feed(embeddings: suffix)
        session.finishSystemPrompt()
        _ = session.feed(embeddings: extra)
        for index in 0 ..< 3 {
            session.beginUnit()
            _ = session.feed(embeddings: unitEmbedding)
            session.commitUnit(
                generatedTokens: [20 + index],
                generatedText: "u\(index)")
        }

        let snapshot = session.snapshot()
        XCTAssertEqual(session.cacheLength, 6) // prefix + suffix + extra + 3 units
        XCTAssertEqual(session.units.map(\.unitID), [0, 1, 2])

        // One call must evict both oldest units while the max-unit watermark
        // is crossed, rebuilding the previous segment after each removal.
        XCTAssertTrue(session.enforceSlidingWindow())
        XCTAssertEqual(session.units.map(\.unitID), [2])
        XCTAssertEqual(session.previousContext.tokenIDs, [9, 10, 20, 21])
        XCTAssertEqual(session.cacheLength, 8) // 1 + 4 + 1 + 1 + 1
        XCTAssertEqual(session.state.position, 8)
        XCTAssertEqual(session.windowStats()["system_preserve_length"] as? Int, 7)
        XCTAssertEqual(
            snapshot.inference.sequenceLength,
            6,
            "pre-eviction snapshot must retain its old logical boundary")
        XCTAssertEqual(snapshot.inference.position, 6)
        XCTAssertEqual(snapshot.contextPreviousTokenIDs, [])
        XCTAssertNil(snapshot.contextPreviousEmbeddings)
    }

    func testContextWindowDropsListenUnitWithEmptyPrevious() {
        let model = MiniCPMLanguageModel(config: tinyConfig())
        let session = MiniCPMSession(model: model)
        session.configureSlidingWindow(MiniCPMSlidingWindowConfig(
            mode: "context",
            contextPreviousMaxTokens: 8,
            contextMaxUnits: 1,
            contextPreviousMarkerTokenIDs: [9, 10]))
        let embedding = MLXArray([Float(1), 2, 3, 4], [1, 1, 4]).asType(.bfloat16)

        session.beginSystemPrompt()
        _ = session.feed(embeddings: embedding)
        session.markSystemPromptSuffix()
        _ = session.feed(embeddings: embedding)
        session.finishSystemPrompt()

        session.beginUnit()
        _ = session.feed(embeddings: embedding)
        // Even malformed listen metadata must not be promoted into previous
        // context; the official extractor skips listen units unconditionally.
        session.commitUnit(
            generatedTokens: [77, 78],
            generatedText: "listen-must-not-enter-previous",
            isListen: true)
        session.beginUnit()
        _ = session.feed(embeddings: embedding)
        session.commitUnit(generatedTokens: [42], generatedText: "kept")
        XCTAssertEqual(session.cacheLength, 4)

        XCTAssertTrue(session.enforceSlidingWindow())
        XCTAssertEqual(session.units.map(\.unitID), [1])
        XCTAssertEqual(session.previousContext.tokenIDs, [])
        XCTAssertEqual(session.previousContext.text, "")
        XCTAssertEqual(session.cacheLength, 3) // prefix + suffix + retained unit
        XCTAssertEqual(session.windowStats()["system_preserve_length"] as? Int, 2)
    }

}
