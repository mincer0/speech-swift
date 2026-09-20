import Foundation
import Metal
import MetalPerformanceShadersGraph
import MLX
import MLXFast
import XCTest
@testable import MiniCPMLLM

/// Opt-in, model-free SDPA parity probe for the official layer-1 trace.
///
/// The input files contain the official BF16 tensors serialized as float32.
/// Keeping this probe independent of a model bundle makes it useful when
/// bisecting attention backends: every candidate receives the same BF16 bytes
/// (or those exact bytes promoted to float32 for the FP32 candidate).
final class MiniCPMMPSGraphSDPATests: XCTestCase {
    private let batch = 1
    private let queryHeads = 32
    private let keyValueHeads = 8
    private let tokens = 4
    private let headDimension = 128

    private struct Metrics {
        let maxAbs: Double
        let meanAbs: Double
        let rms: Double
        let neq: Int
    }

    private enum ProbeDType {
        case bfloat16
        case float32

        var mlx: DType {
            switch self {
            case .bfloat16: return .bfloat16
            case .float32: return .float32
            }
        }

        var mps: MPSDataType {
            switch self {
            case .bfloat16: return .bFloat16
            case .float32: return .float32
            }
        }
    }

    func testQ1PyTorchMPSVectorParityCandidates() throws {
        guard ProcessInfo.processInfo.environment["MINICPM_Q1_SDPA_VECTOR_PROBE"] == "1" else {
            throw XCTSkip("set MINICPM_Q1_SDPA_VECTOR_PROBE=1 to run the qLen=1 probe")
        }
        let directoryPath = ProcessInfo.processInfo.environment["MINICPM_Q1_SDPA_VECTOR_DIR"]
            ?? "/tmp/minicpm-q1-sdpa-vector-v1"
        let directory = URL(fileURLWithPath: directoryPath, isDirectory: true)
        guard isDirectory(directory) else {
            throw XCTSkip("qLen=1 SDPA fixture is unavailable: \(directory.path)")
        }

        let qShape = [1, 32, 1, 128]
        let scale = 1 / sqrt(Float(qShape[3]))
        for keyLength in [3, 4] {
            let kvShape = [1, 32, keyLength, 128]
            let q = MLXArray(try readFloat32(
                from: directory.appendingPathComponent("k\(keyLength)_q.f32"),
                expectedCount: qShape.reduce(1, *)), qShape).asType(.bfloat16)
            let k = MLXArray(try readFloat32(
                from: directory.appendingPathComponent("k\(keyLength)_k.f32"),
                expectedCount: kvShape.reduce(1, *)), kvShape).asType(.bfloat16)
            let v = MLXArray(try readFloat32(
                from: directory.appendingPathComponent("k\(keyLength)_v.f32"),
                expectedCount: kvShape.reduce(1, *)), kvShape).asType(.bfloat16)
            let reference = try readFloat32(
                from: directory.appendingPathComponent("k\(keyLength)_output.f32"),
                expectedCount: qShape.reduce(1, *))

            let fused = MLXFast.scaledDotProductAttention(
                queries: q, keys: k, values: v, scale: scale, mask: .none)
            let scores = matmul(q * scale, k.transposed(0, 1, 3, 2))
            let probabilities = softmax(scores.asType(.float32), axis: -1)
                .asType(.bfloat16)
            let explicit = matmul(probabilities, v)
            let general = MiniCPMExactBF16SDPA.attend(
                query: q, key: k, value: v, scale: scale)
                .transposed(0, 2, 1, 3)
            eval(fused, explicit, general)

            let fusedMetrics = metrics(
                fused.asType(.float32).asArray(Float.self), against: reference)
            let explicitMetrics = metrics(
                explicit.asType(.float32).asArray(Float.self), against: reference)
            let generalMetrics = metrics(
                general.asType(.float32).asArray(Float.self), against: reference)
            report("q1/k\(keyLength) MLX fused", fusedMetrics)
            report("q1/k\(keyLength) MLX explicit", explicitMetrics)
            report("q1/k\(keyLength) MPSGraph general", generalMetrics)
            XCTAssertTrue(fusedMetrics.maxAbs.isFinite)
            XCTAssertTrue(explicitMetrics.maxAbs.isFinite)
            XCTAssertTrue(generalMetrics.maxAbs.isFinite)
        }
    }

    func testQ1PyTorchMPSVectorGQAParityCandidates() throws {
        guard ProcessInfo.processInfo.environment["MINICPM_Q1_SDPA_VECTOR_PROBE"] == "1" else {
            throw XCTSkip("set MINICPM_Q1_SDPA_VECTOR_PROBE=1 to run the qLen=1 GQA probe")
        }
        let directoryPath = ProcessInfo.processInfo.environment["MINICPM_Q1_SDPA_VECTOR_GQA_DIR"]
            ?? "/tmp/minicpm-q1-sdpa-vector-gqa-v1"
        let directory = URL(fileURLWithPath: directoryPath, isDirectory: true)
        guard isDirectory(directory) else {
            throw XCTSkip("qLen=1 GQA fixture is unavailable: \(directory.path)")
        }

        let qShape = [1, 32, 1, 128]
        let scale = 1 / sqrt(Float(qShape[3]))
        for keyLength in [3, 4] {
            let kvShape = [1, 8, keyLength, 128]
            let q = MLXArray(try readFloat32(
                from: directory.appendingPathComponent("k\(keyLength)_q.f32"),
                expectedCount: qShape.reduce(1, *)), qShape).asType(.bfloat16)
            let k = MLXArray(try readFloat32(
                from: directory.appendingPathComponent("k\(keyLength)_k8.f32"),
                expectedCount: kvShape.reduce(1, *)), kvShape).asType(.bfloat16)
            let v = MLXArray(try readFloat32(
                from: directory.appendingPathComponent("k\(keyLength)_v8.f32"),
                expectedCount: kvShape.reduce(1, *)), kvShape).asType(.bfloat16)
            let reference = try readFloat32(
                from: directory.appendingPathComponent("k\(keyLength)_output.f32"),
                expectedCount: qShape.reduce(1, *))

            let native = MLXFast.scaledDotProductAttention(
                queries: q, keys: k, values: v, scale: scale, mask: .none)
            let repeatedShape = [1, 8, 4, keyLength, 128]
            let repeatedK = broadcast(k.expandedDimensions(axis: 2), to: repeatedShape)
                .reshaped([1, 32, keyLength, 128]).contiguous()
            let repeatedV = broadcast(v.expandedDimensions(axis: 2), to: repeatedShape)
                .reshaped([1, 32, keyLength, 128]).contiguous()
            let repeated = MLXFast.scaledDotProductAttention(
                queries: q, keys: repeatedK, values: repeatedV, scale: scale, mask: .none)
            let general = MiniCPMExactBF16SDPA.attend(
                query: q, key: k, value: v, scale: scale)
                .transposed(0, 2, 1, 3)
            eval(native, repeated, general)

            let nativeMetrics = metrics(
                native.asType(.float32).asArray(Float.self), against: reference)
            let repeatedMetrics = metrics(
                repeated.asType(.float32).asArray(Float.self), against: reference)
            let generalMetrics = metrics(
                general.asType(.float32).asArray(Float.self), against: reference)
            report("q1/k\(keyLength) MLX native-GQA", nativeMetrics)
            report("q1/k\(keyLength) MLX repeated-KV", repeatedMetrics)
            report("q1/k\(keyLength) MPSGraph repeated-KV", generalMetrics)
            XCTAssertTrue(nativeMetrics.maxAbs.isFinite)
            XCTAssertTrue(repeatedMetrics.maxAbs.isFinite)
            XCTAssertTrue(generalMetrics.maxAbs.isFinite)
        }
    }

    private var queryShape: [Int] { [batch, queryHeads, tokens, headDimension] }
    private var keyValueShape: [Int] { [batch, keyValueHeads, tokens, headDimension] }
    private var mergedShape: [Int] { [batch, tokens, queryHeads, headDimension] }

    func testExactSDPACachedDecodeQ1K2AndK4Alternation() throws {
        guard MTLCreateSystemDefaultDevice() != nil else {
            throw XCTSkip("Metal GPU device is unavailable")
        }

        let qHeads = 4
        let kvHeads = 2
        let headDimension = 8
        let scale = 1 / sqrt(Float(headDimension))
        let query = MLXArray(
            [Float](repeating: 0, count: qHeads * headDimension),
            [1, qHeads, 1, headDimension]).asType(.bfloat16)

        func values(forKeyLength keyLength: Int) -> (
            key: MLXArray,
            value: MLXArray,
            expected: MLXArray
        ) {
            let key = MLXArray(
                [Float](repeating: 0, count: kvHeads * keyLength * headDimension),
                [1, kvHeads, keyLength, headDimension]).asType(.bfloat16)
            let valueValues: [Float] = Array(0 ..< kvHeads * keyLength * headDimension).map {
                index in
                Float((index % 11) - 5)
            }
            let value = MLXArray(
                valueValues,
                [1, kvHeads, keyLength, headDimension]).asType(.bfloat16)
            let repeats = qHeads / kvHeads
            let repeatedKey = broadcast(
                key.expandedDimensions(axis: 2),
                to: [1, kvHeads, repeats, keyLength, headDimension]).reshaped(
                    [1, qHeads, keyLength, headDimension]).contiguous()
            let repeatedValue = broadcast(
                value.expandedDimensions(axis: 2),
                to: [1, kvHeads, repeats, keyLength, headDimension]).reshaped(
                    [1, qHeads, keyLength, headDimension]).contiguous()
            let expected = MLXFast.scaledDotProductAttention(
                queries: query,
                keys: repeatedKey,
                values: repeatedValue,
                scale: scale,
                mask: .none).transposed(0, 2, 1, 3)
            return (key, value, expected)
        }

        let k2 = values(forKeyLength: 2)
        let k4 = values(forKeyLength: 4)
        let first2 = MiniCPMExactBF16SDPA.attend(
            query: query, key: k2.key, value: k2.value, scale: scale)
        let first4 = MiniCPMExactBF16SDPA.attend(
            query: query, key: k4.key, value: k4.value, scale: scale)
        eval(first2, first4, k2.expected, k4.expected)
        XCTAssertEqual(first2.shape, [1, 1, qHeads, headDimension])
        XCTAssertEqual(first4.shape, [1, 1, qHeads, headDimension])
        XCTAssertEqual(
            first2.asType(.float32).asArray(Float.self),
            k2.expected.asType(.float32).asArray(Float.self))
        XCTAssertEqual(
            first4.asType(.float32).asArray(Float.self),
            k4.expected.asType(.float32).asArray(Float.self))

        // For qLen=2/kLen=4, right-aligned causal masking allows three keys
        // for query row zero and all four keys for query row one.  Zero Q/K
        // makes the expected BF16 averages independent of softmax rounding.
        let query2 = MLXArray(
            [Float](repeating: 0, count: qHeads * 2 * headDimension),
            [1, qHeads, 2, headDimension]).asType(.bfloat16)
        eval(k4.value)
        let k4Values = k4.value.asType(.float32).asArray(Float.self)
        var expectedQ2Values = [Float](
            repeating: 0, count: 2 * qHeads * headDimension)
        let repeats = qHeads / kvHeads
        for queryRow in 0 ..< 2 {
            let allowedKeys = 4 - 2 + queryRow + 1
            for head in 0 ..< qHeads {
                let kvHead = head / repeats
                for dimension in 0 ..< headDimension {
                    var sum: Float = 0
                    for token in 0 ..< allowedKeys {
                        let source = (kvHead * 4 + token) * headDimension + dimension
                        sum += k4Values[source]
                    }
                    let destination = (queryRow * qHeads + head) * headDimension + dimension
                    expectedQ2Values[destination] = sum / Float(allowedKeys)
                }
            }
        }
        let expectedQ2 = MLXArray(
            expectedQ2Values, [1, 2, qHeads, headDimension]).asType(.bfloat16)
        let actualQ2 = MiniCPMExactBF16SDPA.attend(
            query: query2, key: k4.key, value: k4.value, scale: scale)
        eval(actualQ2, expectedQ2)
        XCTAssertEqual(actualQ2.shape, [1, 2, qHeads, headDimension])
        XCTAssertEqual(
            actualQ2.asType(.float32).asArray(Float.self),
            expectedQ2.asType(.float32).asArray(Float.self))

        let baseline2 = first2.asType(.float32).asArray(Float.self)
        let baseline4 = first4.asType(.float32).asArray(Float.self)
        for _ in 0 ..< 4 {
            let repeated2 = MiniCPMExactBF16SDPA.attend(
                query: query, key: k2.key, value: k2.value, scale: scale)
            let repeated4 = MiniCPMExactBF16SDPA.attend(
                query: query, key: k4.key, value: k4.value, scale: scale)
            eval(repeated2, repeated4)
            XCTAssertEqual(repeated2.asType(.float32).asArray(Float.self), baseline2)
            XCTAssertEqual(repeated4.asType(.float32).asArray(Float.self), baseline4)
        }
    }

    func testLayer1SDPAParityCandidates() throws {
        guard ProcessInfo.processInfo.environment["MINICPM_MPSGRAPH_SDPA_PROBE"] == "1" else {
            throw XCTSkip("set MINICPM_MPSGRAPH_SDPA_PROBE=1 to run the SDPA probe")
        }
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw XCTSkip("Metal GPU device is unavailable")
        }

        let environment = ProcessInfo.processInfo.environment
        let directoryPath = environment["MINICPM_MPSGRAPH_SDPA_TRACE_DIR"]
            ?? "/tmp/minicpm-python-layer1-20260805-1040"
        let directory = URL(fileURLWithPath: directoryPath, isDirectory: true)
        guard isDirectory(directory) else {
            throw XCTSkip("layer-1 SDPA trace directory is unavailable: \(directory.path)")
        }

        let qValues = try readFloat32(
            from: directory.appendingPathComponent("layer_01_q_rope.f32"),
            expectedCount: queryShape.reduce(1, *))
        let kValues = try readFloat32(
            from: directory.appendingPathComponent("layer_01_k_rope.f32"),
            expectedCount: keyValueShape.reduce(1, *))
        let vValues = try readFloat32(
            from: directory.appendingPathComponent("layer_01_v.f32"),
            expectedCount: keyValueShape.reduce(1, *))
        let reference = try readFloat32(
            from: directory.appendingPathComponent("layer_01_sdpa_merge.f32"),
            expectedCount: mergedShape.reduce(1, *))
        guard qValues.allSatisfy(\.isFinite),
              kValues.allSatisfy(\.isFinite),
              vValues.allSatisfy(\.isFinite),
              reference.allSatisfy(\.isFinite) else {
            XCTFail("layer-1 SDPA trace contains non-finite values")
            return
        }

        let scale = 1 / sqrt(Float(headDimension))
        let qBF16 = MLXArray(qValues, queryShape).asType(.bfloat16)
        let kBF16 = MLXArray(kValues, keyValueShape).asType(.bfloat16)
        let vBF16 = MLXArray(vValues, keyValueShape).asType(.bfloat16)
        let repeated = try explicitMLXSDPA(
            query: qBF16, key: kBF16, value: vBF16, scale: scale)
        report("MLX explicit-repeat BF16", metrics(repeated, against: reference))

        // A direct GQA call is included beside the explicit repeat to expose
        // any backend difference caused by MLX's internal KV-head tiling.
        let directHeads = MLXFast.scaledDotProductAttention(
            queries: qBF16,
            keys: kBF16,
            values: vBF16,
            scale: scale,
            mask: .causal)
        let direct = mergedValues(directHeads)
        eval(directHeads, direct)
        report(
            "MLX native-GQA BF16",
            metrics(direct.asType(.float32).asArray(Float.self), against: reference))

        let bf16Explicit = try runMPSGraphExplicit(
            query: qBF16,
            key: kBF16,
            value: vBF16,
            scale: scale,
            dtype: .bfloat16,
            device: device)
        report("MPSGraph explicit BF16", metrics(bf16Explicit, against: reference))

        let repeatedKey = explicitRepeatKV(kBF16)
        let repeatedValue = explicitRepeatKV(vBF16)
        let pytorchGeneral = try runMPSGraphPyTorchGeneral(
            query: qBF16,
            key: repeatedKey,
            value: repeatedValue,
            scale: scale,
            device: device)
        report("MPSGraph PyTorch sdpa_general_mps", metrics(pytorchGeneral, against: reference))
        let fp32Softmax = try runMPSGraphPyTorchFP32Softmax(
            query: qBF16,
            key: repeatedKey,
            value: repeatedValue,
            scale: scale,
            device: device)
        report("MPSGraph BF16-QK FP32-softmax", metrics(fp32Softmax, against: reference))
        let eager = try runMPSGraphPyTorchEager(
            query: qBF16,
            key: repeatedKey,
            value: repeatedValue,
            scale: scale,
            device: device)
        report("MPSGraph PyTorch eager", metrics(eager, against: reference))

        let qFloat = MLXArray(qValues, queryShape)
        let kFloat = MLXArray(kValues, keyValueShape)
        let vFloat = MLXArray(vValues, keyValueShape)
        let fp32Explicit = try runMPSGraphExplicit(
            query: qFloat,
            key: kFloat,
            value: vFloat,
            scale: scale,
            dtype: .float32,
            device: device)
        report("MPSGraph explicit FP32", metrics(fp32Explicit, against: reference))

        if #available(macOS 15.0, iOS 18.0, *) {
            let native = try runMPSGraphNative(
                query: qBF16,
                key: kBF16,
                value: vBF16,
                scale: scale,
                device: device)
            let nativeMetrics = metrics(native, against: reference)
            report("MPSGraph native SDPA BF16", nativeMetrics)
            // Keep an exact candidate as a regression guard when the SDK and
            // device reproduce the official BF16 output byte-for-byte.  On a
            // backend with a different accumulation order the metrics remain
            // diagnostic, without introducing an arbitrary tolerance gate.
            if nativeMetrics.neq == 0 {
                XCTAssertEqual(nativeMetrics.maxAbs, 0)
            }
        } else {
            print("MPSGraph native SDPA unavailable on this SDK")
        }
    }

    private func explicitMLXSDPA(
        query: MLXArray,
        key: MLXArray,
        value: MLXArray,
        scale: Float
    ) throws -> [Float] {
        guard query.dtype == .bfloat16,
              key.dtype == .bfloat16,
              value.dtype == .bfloat16 else {
            throw ProbeError.invalid("explicit MLX candidate expects BF16 inputs")
        }
        let repeatedKey = explicitRepeatKV(key)
        let repeatedValue = explicitRepeatKV(value)
        let heads = MLXFast.scaledDotProductAttention(
            queries: query,
            keys: repeatedKey,
            values: repeatedValue,
            scale: scale,
            mask: .causal)
        let merged = mergedValues(heads)
        eval(repeatedKey, repeatedValue, heads, merged)
        return merged.asType(.float32).asArray(Float.self)
    }

    private func explicitRepeatKV(_ input: MLXArray) -> MLXArray {
        let repeats = input.dim(1) > 0 ? queryHeads / input.dim(1) : 0
        precondition(repeats > 0 && queryHeads % input.dim(1) == 0)
        let repeatedShape = [input.dim(0), input.dim(1), repeats, input.dim(2), input.dim(3)]
        return broadcast(
            input.expandedDimensions(axis: 2), to: repeatedShape).reshaped(
                [input.dim(0), input.dim(1) * repeats, input.dim(2), input.dim(3)]).contiguous()
    }

    private func mergedValues(_ heads: MLXArray) -> MLXArray {
        heads.transposed(0, 2, 1, 3)
    }

    private func runMPSGraphExplicit(
        query: MLXArray,
        key: MLXArray,
        value: MLXArray,
        scale: Float,
        dtype: ProbeDType,
        device: any MTLDevice
    ) throws -> [Float] {
        precondition(query.dtype == dtype.mlx)
        precondition(key.dtype == dtype.mlx)
        precondition(value.dtype == dtype.mlx)
        let graph = MPSGraph()
        let queryTensor = graph.placeholder(
            shape: queryShape.map(NSNumber.init), dataType: dtype.mps, name: "sdpa_q")
        let keyTensor = graph.placeholder(
            shape: keyValueShape.map(NSNumber.init), dataType: dtype.mps, name: "sdpa_k")
        let valueTensor = graph.placeholder(
            shape: keyValueShape.map(NSNumber.init), dataType: dtype.mps, name: "sdpa_v")
        let maskTensor = causalMaskTensor(graph: graph, dtype: dtype)
        // MPSGraph matrixMultiplication does not broadcast the head axis from
        // 8 KV heads to 32 query heads.  Mirror PyTorch repeat_kv explicitly
        // before the two matmuls in this reference candidate.
        let repeatedKey = repeatedKVHeads(
            graph: graph, tensor: keyTensor, name: "sdpa_k_repeat")
        let repeatedValue = repeatedKVHeads(
            graph: graph, tensor: valueTensor, name: "sdpa_v_repeat")
        let keyTranspose = graph.transpose(
            repeatedKey, permutation: [0, 1, 3, 2], name: "sdpa_k_transpose")
        let scores = graph.matrixMultiplication(
            primary: queryTensor, secondary: keyTranspose, name: "sdpa_scores")
        let scaleTensor = graph.constant(
            Double(scale), dataType: dtype.mps)
        let scaledScores = graph.multiplication(
            scores, scaleTensor, name: "sdpa_scale")
        let maskedScores = graph.addition(
            scaledScores, maskTensor, name: "sdpa_causal_mask")
        let probabilities = graph.softMax(
            with: maskedScores, axis: -1, name: "sdpa_softmax")
        let attendedHeads = graph.matrixMultiplication(
            primary: probabilities, secondary: repeatedValue, name: "sdpa_values")
        let merged = graph.transpose(
            attendedHeads, permutation: [0, 2, 1, 3], name: "sdpa_merge")
        return try runGraph(
            graph: graph,
            queryTensor: queryTensor,
            keyTensor: keyTensor,
            valueTensor: valueTensor,
            query: query,
            key: key,
            value: value,
            queryShape: queryShape,
            keyValueShape: keyValueShape,
            outputTensor: merged,
            outputShape: mergedShape,
            dtype: dtype,
            device: device)
    }

    /// Reproduce PyTorch 2.8's ``sdpa_general_mps`` graph sequence.  This is
    /// deliberately separate from the generic candidates above: PyTorch
    /// keeps QK^T in BF16, upcasts only for the FP32 scale multiply, casts
    /// back to BF16, then applies a bool band mask and BF16 softmax/matmul.
    private func runMPSGraphPyTorchGeneral(
        query: MLXArray,
        key: MLXArray,
        value: MLXArray,
        scale: Float,
        device: any MTLDevice
    ) throws -> [Float] {
        precondition(query.dtype == .bfloat16)
        precondition(key.dtype == .bfloat16)
        precondition(value.dtype == .bfloat16)
        let qShape = query.shape
        let kvShape = key.shape
        precondition(qShape == [batch, queryHeads, tokens, headDimension])
        precondition(kvShape == qShape)

        let graph = MPSGraph()
        let qTensor = graph.placeholder(
            shape: qShape.map(NSNumber.init), dataType: .bFloat16, name: "pytorch_sdpa_q")
        let kTensor = graph.placeholder(
            shape: kvShape.map(NSNumber.init), dataType: .bFloat16, name: "pytorch_sdpa_k")
        let vTensor = graph.placeholder(
            shape: kvShape.map(NSNumber.init), dataType: .bFloat16, name: "pytorch_sdpa_v")
        let kTranspose = graph.transpose(
            kTensor, permutation: [0, 1, 3, 2], name: "pytorch_sdpa_k_transpose")
        let qk = graph.matrixMultiplication(
            primary: qTensor, secondary: kTranspose, name: "pytorch_sdpa_qk")
        let scaleTensor = graph.constant(
            Double(scale), shape: [1].map(NSNumber.init), dataType: .float32)
        var scaled = qk
        if scaled.dataType != .float32 {
            scaled = graph.cast(scaled, to: .float32, name: "pytorch_sdpa_qk_fp32")
        }
        scaled = graph.multiplication(
            scaled, scaleTensor, name: "pytorch_sdpa_scale_fp32")
        if scaled.dataType != qTensor.dataType {
            scaled = graph.cast(scaled, to: qTensor.dataType, name: "pytorch_sdpa_scale_bf16")
        }

        let ones = graph.constant(
            1.0,
            shape: [tokens, tokens].map(NSNumber.init),
            dataType: .bool)
        let causalMask = graph.bandPart(
            ones, numLower: -1, numUpper: 0, name: "pytorch_sdpa_band")
        let minusInf = graph.constant(
            -1e20,
            shape: [batch, queryHeads, tokens, tokens].map(NSNumber.init),
            dataType: scaled.dataType)
        let masked = graph.select(
            predicate: causalMask,
            trueTensor: scaled,
            falseTensor: minusInf,
            name: "pytorch_sdpa_select")
        let softmax = graph.softMax(
            with: masked, axis: 3, name: "pytorch_sdpa_softmax_bf16")
        let attended = graph.matrixMultiplication(
            primary: softmax, secondary: vTensor, name: "pytorch_sdpa_v_bf16")
        let merged = graph.transpose(
            attended, permutation: [0, 2, 1, 3], name: "pytorch_sdpa_merge")

        print(
            "MINICPM_SDPA pytorch_stages "
                + "q=\(String(describing: qTensor.shape))/\(qTensor.dataType) "
                + "qk=\(String(describing: qk.shape))/\(qk.dataType) "
                + "scaled=\(String(describing: scaled.shape))/\(scaled.dataType) "
                + "mask=\(String(describing: causalMask.shape))/\(causalMask.dataType) "
                + "softmax=\(String(describing: softmax.shape))/\(softmax.dataType) "
                + "out=\(String(describing: merged.shape))/\(merged.dataType)")

        return try runGraph(
            graph: graph,
            queryTensor: qTensor,
            keyTensor: kTensor,
            valueTensor: vTensor,
            query: query,
            key: key,
            value: value,
            queryShape: qShape,
            keyValueShape: kvShape,
            outputTensor: merged,
            outputShape: mergedShape,
            dtype: .bfloat16,
            device: device)
    }

    /// Keep the QK/scale/mask/softmax path in FP32 while feeding BF16 Q/K/V
    /// buffers.  MPSGraph accepts the mixed-dtype value matmul on the SDKs
    /// used by this probe and materializes its result as FP32; quantizing that
    /// result back to BF16 below makes the comparison match the BF16 output
    /// contract of the surrounding attention block.
    private func runMPSGraphPyTorchFP32Softmax(
        query: MLXArray,
        key: MLXArray,
        value: MLXArray,
        scale: Float,
        device: any MTLDevice
    ) throws -> [Float] {
        precondition(query.dtype == .bfloat16)
        precondition(key.dtype == .bfloat16)
        precondition(value.dtype == .bfloat16)
        let qShape = query.shape
        let kvShape = key.shape
        precondition(qShape == [batch, queryHeads, tokens, headDimension])
        precondition(kvShape == qShape)

        let graph = MPSGraph()
        let qTensor = graph.placeholder(
            shape: qShape.map(NSNumber.init), dataType: .bFloat16,
            name: "pytorch_sdpa_fp32_q")
        let kTensor = graph.placeholder(
            shape: kvShape.map(NSNumber.init), dataType: .bFloat16,
            name: "pytorch_sdpa_fp32_k")
        let vTensor = graph.placeholder(
            shape: kvShape.map(NSNumber.init), dataType: .bFloat16,
            name: "pytorch_sdpa_fp32_v")
        let kTranspose = graph.transpose(
            kTensor, permutation: [0, 1, 3, 2], name: "pytorch_sdpa_fp32_k_transpose")

        // PyTorch's MPS implementation computes the first QK^T matmul in the
        // input dtype.  Only the scale multiply (and all following mask and
        // softmax operations) are intentionally kept in FP32 here.
        let qk = graph.matrixMultiplication(
            primary: qTensor, secondary: kTranspose, name: "pytorch_sdpa_fp32_qk")
        var scaled = graph.cast(qk, to: .float32, name: "pytorch_sdpa_fp32_qk_cast")
        let scaleTensor = graph.constant(
            Double(scale), shape: [1].map(NSNumber.init), dataType: .float32)
        scaled = graph.multiplication(
            scaled, scaleTensor, name: "pytorch_sdpa_fp32_scale")

        let ones = graph.constant(
            1.0,
            shape: [tokens, tokens].map(NSNumber.init),
            dataType: .bool)
        let causalMask = graph.bandPart(
            ones, numLower: -1, numUpper: 0, name: "pytorch_sdpa_fp32_band")
        let minusInf = graph.constant(
            -1e20,
            shape: [batch, queryHeads, tokens, tokens].map(NSNumber.init),
            dataType: .float32)
        let masked = graph.select(
            predicate: causalMask,
            trueTensor: scaled,
            falseTensor: minusInf,
            name: "pytorch_sdpa_fp32_select")
        let softmax = graph.softMax(
            with: masked, axis: 3, name: "pytorch_sdpa_fp32_softmax")

        // Keep V as a BF16 feed.  MPSGraph promotes this mixed matmul's
        // result to FP32, allowing us to observe the unquantized path before
        // the explicit BF16 round trip used for parity reporting.
        let attended = graph.matrixMultiplication(
            primary: softmax, secondary: vTensor, name: "pytorch_sdpa_fp32_v_bf16")
        let mergedFP32 = graph.transpose(
            attended, permutation: [0, 2, 1, 3], name: "pytorch_sdpa_fp32_merge")
        let merged = graph.cast(
            mergedFP32, to: .bFloat16, name: "pytorch_sdpa_fp32_output_bf16")

        print(
            "MINICPM_SDPA pytorch_fp32_softmax_stages "
                + "q=\(String(describing: qTensor.shape))/\(qTensor.dataType) "
                + "qk=\(String(describing: qk.shape))/\(qk.dataType) "
                + "scaled=\(String(describing: scaled.shape))/\(scaled.dataType) "
                + "mask=\(String(describing: causalMask.shape))/\(causalMask.dataType) "
                + "softmax=\(String(describing: softmax.shape))/\(softmax.dataType) "
                + "out=\(String(describing: merged.shape))/\(merged.dataType)")

        return try runGraph(
            graph: graph,
            queryTensor: qTensor,
            keyTensor: kTensor,
            valueTensor: vTensor,
            query: query,
            key: key,
            value: value,
            queryShape: qShape,
            keyValueShape: kvShape,
            outputTensor: merged,
            outputShape: mergedShape,
            dtype: .bfloat16,
            device: device)
    }

    /// Mirror the configured Qwen3 eager attention path: QK is produced from
    /// BF16 inputs, masking and softmax are evaluated in FP32, probabilities
    /// are explicitly rounded back to BF16, and the value matmul is BF16.
    /// This differs from PyTorch's fused SDPA graph even though both expose a
    /// BF16 result to the transformer block.
    private func runMPSGraphPyTorchEager(
        query: MLXArray,
        key: MLXArray,
        value: MLXArray,
        scale: Float,
        device: any MTLDevice
    ) throws -> [Float] {
        precondition(query.dtype == .bfloat16)
        precondition(key.dtype == .bfloat16)
        precondition(value.dtype == .bfloat16)
        let qShape = query.shape
        let kvShape = key.shape
        precondition(qShape == [batch, queryHeads, tokens, headDimension])
        precondition(kvShape == qShape)

        let graph = MPSGraph()
        let qTensor = graph.placeholder(
            shape: qShape.map(NSNumber.init), dataType: .bFloat16,
            name: "pytorch_eager_q")
        let kTensor = graph.placeholder(
            shape: kvShape.map(NSNumber.init), dataType: .bFloat16,
            name: "pytorch_eager_k")
        let vTensor = graph.placeholder(
            shape: kvShape.map(NSNumber.init), dataType: .bFloat16,
            name: "pytorch_eager_v")
        let kTranspose = graph.transpose(
            kTensor, permutation: [0, 1, 3, 2], name: "pytorch_eager_k_transpose")
        let qk = graph.matrixMultiplication(
            primary: qTensor, secondary: kTranspose, name: "pytorch_eager_qk")
        var scores = graph.cast(qk, to: .float32, name: "pytorch_eager_qk_fp32")
        let scaleTensor = graph.constant(
            Double(scale), shape: [1].map(NSNumber.init), dataType: .float32)
        scores = graph.multiplication(
            scores, scaleTensor, name: "pytorch_eager_scale_fp32")

        let ones = graph.constant(
            1.0,
            shape: [tokens, tokens].map(NSNumber.init),
            dataType: .bool)
        let causalMask = graph.bandPart(
            ones, numLower: -1, numUpper: 0, name: "pytorch_eager_band")
        let minusInf = graph.constant(
            -1e20,
            shape: [batch, queryHeads, tokens, tokens].map(NSNumber.init),
            dataType: .float32)
        let masked = graph.select(
            predicate: causalMask,
            trueTensor: scores,
            falseTensor: minusInf,
            name: "pytorch_eager_select")
        let softmaxFP32 = graph.softMax(
            with: masked, axis: 3, name: "pytorch_eager_softmax_fp32")
        let probabilities = graph.cast(
            softmaxFP32, to: .bFloat16, name: "pytorch_eager_probabilities_bf16")
        let attended = graph.matrixMultiplication(
            primary: probabilities,
            secondary: vTensor,
            name: "pytorch_eager_v_bf16")
        let merged = graph.transpose(
            attended, permutation: [0, 2, 1, 3], name: "pytorch_eager_merge")

        return try runGraph(
            graph: graph,
            queryTensor: qTensor,
            keyTensor: kTensor,
            valueTensor: vTensor,
            query: query,
            key: key,
            value: value,
            queryShape: qShape,
            keyValueShape: kvShape,
            outputTensor: merged,
            outputShape: mergedShape,
            dtype: .bfloat16,
            device: device)
    }

    private func repeatedKVHeads(
        graph: MPSGraph,
        tensor: MPSGraphTensor,
        name: String
    ) -> MPSGraphTensor {
        let repeats = queryHeads / keyValueHeads
        let singleton = graph.reshape(
            tensor,
            shape: [batch, keyValueHeads, 1, tokens, headDimension].map(NSNumber.init),
            name: "\(name)_singleton")
        let broadcast = graph.broadcast(
            singleton,
            shape: [batch, keyValueHeads, repeats, tokens, headDimension].map(NSNumber.init),
            name: "\(name)_broadcast")
        return graph.reshape(
            broadcast,
            shape: [batch, queryHeads, tokens, headDimension].map(NSNumber.init),
            name: name)
    }

    @available(macOS 15.0, iOS 18.0, *)
    private func runMPSGraphNative(
        query: MLXArray,
        key: MLXArray,
        value: MLXArray,
        scale: Float,
        device: any MTLDevice
    ) throws -> [Float] {
        let graph = MPSGraph()
        let queryTensor = graph.placeholder(
            shape: queryShape.map(NSNumber.init), dataType: .bFloat16, name: "native_sdpa_q")
        let keyTensor = graph.placeholder(
            shape: keyValueShape.map(NSNumber.init), dataType: .bFloat16, name: "native_sdpa_k")
        let valueTensor = graph.placeholder(
            shape: keyValueShape.map(NSNumber.init), dataType: .bFloat16, name: "native_sdpa_v")
        let maskTensor = causalMaskTensor(graph: graph, dtype: .bfloat16)
        let attendedHeads = graph.scaledDotProductAttention(
            query: queryTensor,
            key: keyTensor,
            value: valueTensor,
            mask: maskTensor,
            scale: scale,
            name: "native_sdpa")
        let merged = graph.transpose(
            attendedHeads, permutation: [0, 2, 1, 3], name: "native_sdpa_merge")
        return try runGraph(
            graph: graph,
            queryTensor: queryTensor,
            keyTensor: keyTensor,
            valueTensor: valueTensor,
            query: query,
            key: key,
            value: value,
            queryShape: queryShape,
            keyValueShape: keyValueShape,
            outputTensor: merged,
            outputShape: mergedShape,
            dtype: .bfloat16,
            device: device)
    }

    private func causalMaskTensor(
        graph: MPSGraph,
        dtype: ProbeDType
    ) -> MPSGraphTensor {
        let negativeInfinity = -Float.infinity
        let values = (0 ..< tokens).flatMap { row in
            (0 ..< tokens).map { column in column <= row ? Float(0) : negativeInfinity }
        }
        let mask = MLXArray(values, [1, 1, tokens, tokens]).asType(dtype.mlx)
        let data = mask.asData(access: .copy).data
        return graph.constant(
            data,
            shape: [1, 1, tokens, tokens].map(NSNumber.init),
            dataType: dtype.mps)
    }

    private func runGraph(
        graph: MPSGraph,
        queryTensor: MPSGraphTensor,
        keyTensor: MPSGraphTensor,
        valueTensor: MPSGraphTensor,
        query: MLXArray,
        key: MLXArray,
        value: MLXArray,
        queryShape: [Int],
        keyValueShape: [Int],
        outputTensor: MPSGraphTensor,
        outputShape: [Int],
        dtype: ProbeDType,
        device: any MTLDevice,
        outputDType: ProbeDType? = nil
    ) throws -> [Float] {
        let resultDType = outputDType ?? dtype
        let queue = try XCTUnwrap(device.makeCommandQueue(), "Metal command queue unavailable")
        let qBuffer = try noCopyBuffer(query, device: device, label: "SDPA query")
        let kBuffer = try noCopyBuffer(key, device: device, label: "SDPA key")
        let vBuffer = try noCopyBuffer(value, device: device, label: "SDPA value")
        let qData = MPSGraphTensorData(
            qBuffer, shape: queryShape.map(NSNumber.init), dataType: dtype.mps)
        let kData = MPSGraphTensorData(
            kBuffer, shape: keyValueShape.map(NSNumber.init), dataType: dtype.mps)
        let vData = MPSGraphTensorData(
            vBuffer, shape: keyValueShape.map(NSNumber.init), dataType: dtype.mps)
        let results = graph.run(
            with: queue,
            feeds: [queryTensor: qData, keyTensor: kData, valueTensor: vData],
            targetTensors: [outputTensor],
            targetOperations: nil)
        guard let result = results[outputTensor] else {
            throw ProbeError.invalid("MPSGraph returned no SDPA result")
        }
        guard result.shape.map(\.intValue) == outputShape else {
            throw ProbeError.invalid(
                "MPSGraph SDPA shape \(result.shape) != \(outputShape)")
        }
        guard result.dataType == resultDType.mps else {
            throw ProbeError.invalid(
                "MPSGraph SDPA dtype \(result.dataType) != \(resultDType.mps)")
        }
        return readMPSGraphValues(result, dtype: resultDType)
    }

    /// Quantize an FP32 graph output through MLX's BF16 conversion and expose
    /// the resulting values as Float for the common metrics path.
    private func bf16RoundTrip(_ values: [Float]) -> [Float] {
        let quantized = MLXArray(values, mergedShape).asType(.bfloat16)
        return quantized.asType(.float32).asArray(Float.self)
    }

    private func noCopyBuffer(
        _ array: MLXArray,
        device: any MTLDevice,
        label: String
    ) throws -> any MTLBuffer {
        eval(array)
        let backing = array.asData(access: .noCopy)
        guard backing.dType == array.dtype,
              backing.strides == contiguousStrides(for: array.shape) else {
            throw ProbeError.invalid("\(label) is not contiguous \(array.dtype)")
        }
        guard let buffer = array.asMTLBuffer(device: device, noCopy: true) else {
            throw ProbeError.invalid("MLX no-copy MTLBuffer unavailable for \(label)")
        }
        guard buffer.length >= array.nbytes else {
            throw ProbeError.invalid("MLX MTLBuffer is too small for \(label)")
        }
        return buffer
    }

    private func readMPSGraphValues(
        _ data: MPSGraphTensorData,
        dtype: ProbeDType
    ) -> [Float] {
        let ndarray = data.mpsndarray()
        let count = data.shape.map(\.intValue).reduce(1, *)
        switch dtype {
        case .bfloat16:
            var bits = [UInt16](repeating: 0, count: count)
            ndarray.readBytes(&bits, strideBytes: nil)
            return bits.map(floatFromBFloat16)
        case .float32:
            var values = [Float](repeating: 0, count: count)
            ndarray.readBytes(&values, strideBytes: nil)
            return values
        }
    }

    private func floatFromBFloat16(_ bits: UInt16) -> Float {
        Float(bitPattern: UInt32(bits) << 16)
    }

    private func metrics(_ actual: [Float], against expected: [Float]) -> Metrics {
        guard actual.count == expected.count, !actual.isEmpty else {
            return Metrics(maxAbs: .infinity, meanAbs: .infinity, rms: .infinity, neq: -1)
        }
        var maxAbs = 0.0
        var sumAbs = 0.0
        var sumSquares = 0.0
        var neq = 0
        for (lhs, rhs) in zip(actual, expected) {
            let delta = abs(Double(lhs) - Double(rhs))
            maxAbs = max(maxAbs, delta)
            sumAbs += delta
            sumSquares += delta * delta
            if lhs.bitPattern != rhs.bitPattern { neq += 1 }
        }
        let count = Double(actual.count)
        return Metrics(
            maxAbs: maxAbs,
            meanAbs: sumAbs / count,
            rms: sqrt(sumSquares / count),
            neq: neq)
    }

    private func report(_ label: String, _ result: Metrics) {
        print(
            "MINICPM_SDPA \(label) max_abs=\(result.maxAbs) "
                + "mean_abs=\(result.meanAbs) rms=\(result.rms) neq=\(result.neq)")
    }

    private func readFloat32(from url: URL, expectedCount: Int) throws -> [Float] {
        let data: Data
        do {
            data = try Data(contentsOf: url)
        } catch {
            throw XCTSkip("missing SDPA trace tensor \(url.path): \(error)")
        }
        guard data.count == expectedCount * MemoryLayout<Float>.stride else {
            throw ProbeError.invalid(
                "\(url.lastPathComponent) has \(data.count) bytes; expected "
                    + "\(expectedCount * MemoryLayout<Float>.stride)")
        }
        return data.withUnsafeBytes { raw in
            Array(raw.bindMemory(to: Float.self))
        }
    }

    private func isDirectory(_ url: URL) -> Bool {
        var directory: ObjCBool = false
        return FileManager.default.fileExists(atPath: url.path, isDirectory: &directory)
            && directory.boolValue
    }

    private func contiguousStrides(for shape: [Int]) -> [Int] {
        guard !shape.isEmpty else { return [] }
        var strides = Array(repeating: 1, count: shape.count)
        if shape.count > 1 {
            for index in stride(from: shape.count - 2, through: 0, by: -1) {
                strides[index] = strides[index + 1] * shape[index + 1]
            }
        }
        return strides
    }

    private enum ProbeError: Error, LocalizedError {
        case invalid(String)

        var errorDescription: String? {
            switch self {
            case .invalid(let message): return message
            }
        }
    }
}
