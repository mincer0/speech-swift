import Foundation
import Metal
import MetalPerformanceShadersGraph
import MLX

/// A complete BF16 attention plan for the MiniCPM Whisper encoder.
///
/// The official PyTorch MPS path materializes the already-scaled BF16 Q/K/V
/// projections, performs QK^T, an optional neutral additive-mask operation,
/// BF16 softmax, and BF16 P·V.  Keeping those operations in one MPSGraph is
/// important: replacing only softmax still leaves MLX's BF16 value matmul as
/// a different reduction boundary, and that one-ULP difference is amplified
/// by the recurrent encoder stack.
///
/// This helper is opt-in through `MINICPM_AUDIO_ATTENTION_VARIANT=mpsgraph`.
/// It remains opt-in until the complete offline and streaming golden gates
/// select it; the existing MLX path is retained as the fallback for unit
/// configurations and for a safe rollback.
public enum MiniCPMAudioBF16Attention {
    public struct Result {
        public let scores: MLXArray
        public let probabilities: MLXArray?
        public let context: MLXArray

        fileprivate init(
            scores: MLXArray,
            probabilities: MLXArray?,
            context: MLXArray
        ) {
            self.scores = scores
            self.probabilities = probabilities
            self.context = context
        }
    }

    private struct PlanKey: Hashable {
        let queryShape: [Int]
        let keyShape: [Int]
        let valueShape: [Int]
        let hasNeutralMask: Bool
    }

    private final class ExecutionContext {
        static let shared = ExecutionContext()

        let device: any MTLDevice
        let commandQueue: any MTLCommandQueue
        let lock = NSLock()

        private init() {
            guard let device = MTLCreateSystemDefaultDevice() else {
                preconditionFailure("MiniCPM audio attention requires a Metal device")
            }
            guard let commandQueue = device.makeCommandQueue() else {
                preconditionFailure("MiniCPM audio attention could not create a command queue")
            }
            self.device = device
            self.commandQueue = commandQueue
        }
    }

    private final class Plan {
        let graph: MPSGraph
        let query: MPSGraphTensor
        let key: MPSGraphTensor
        let value: MPSGraphTensor
        let scores: MPSGraphTensor
        let probabilities: MPSGraphTensor
        let context: MPSGraphTensor

        init(
            graph: MPSGraph,
            query: MPSGraphTensor,
            key: MPSGraphTensor,
            value: MPSGraphTensor,
            scores: MPSGraphTensor,
            probabilities: MPSGraphTensor,
            context: MPSGraphTensor
        ) {
            self.graph = graph
            self.query = query
            self.key = key
            self.value = value
            self.scores = scores
            self.probabilities = probabilities
            self.context = context
        }
    }

    private static let execution = ExecutionContext.shared
    private static let planLock = NSLock()
    private static var plans: [PlanKey: Plan] = [:]

    /// Run exact BF16 attention on tensors shaped `[B, H, Q/K, D]`.
    ///
    /// `query` must already include the Whisper scale (`1/sqrt(D)`).  When
    /// `neutralMask` is true, the graph spells the same zero BF16 additive
    /// mask boundary used by streaming PyTorch calls.  Set
    /// `captureIntermediates` when the caller needs trace tensors; production
    /// calls bind only the final context output.
    public static func apply(
        query: MLXArray,
        key: MLXArray,
        value: MLXArray,
        neutralMask: Bool,
        captureIntermediates: Bool
    ) -> Result {
        validate(query: query, key: key, value: value)
        let q = rowContiguous(query)
        let k = rowContiguous(key)
        let v = rowContiguous(value)
        let keyID = PlanKey(
            queryShape: q.shape,
            keyShape: k.shape,
            valueShape: v.shape,
            hasNeutralMask: neutralMask)

        planLock.lock()
        let plan: Plan
        if let cached = plans[keyID] {
            plan = cached
        } else {
            let graph = MPSGraph()
            let queryTensor = graph.placeholder(
                shape: q.shape.map(NSNumber.init),
                dataType: .bFloat16,
                name: "minicpm_audio_attention_query")
            let keyTensor = graph.placeholder(
                shape: k.shape.map(NSNumber.init),
                dataType: .bFloat16,
                name: "minicpm_audio_attention_key")
            let valueTensor = graph.placeholder(
                shape: v.shape.map(NSNumber.init),
                dataType: .bFloat16,
                name: "minicpm_audio_attention_value")
            let transposedKey = graph.transpose(
                keyTensor,
                permutation: [0, 1, 3, 2],
                name: "minicpm_audio_attention_key_transpose")
            var scoreTensor = graph.matrixMultiplication(
                primary: queryTensor,
                secondary: transposedKey,
                name: "minicpm_audio_attention_qk")
            precondition(
                scoreTensor.dataType == .bFloat16,
                "MiniCPM audio attention scores must remain BF16")
            if neutralMask {
                // The streaming mask is all zeros but PyTorch still executes
                // the BF16 addition.  Use a full [B,H,Q,K] constant so the
                // graph has the same broadcast-free operation boundary as the
                // official MPS trace.
                let zero = graph.constant(
                    0.0,
                    shape: [q.dim(0), q.dim(1), q.dim(2), k.dim(2)]
                        .map(NSNumber.init),
                    dataType: .bFloat16)
                scoreTensor = graph.addition(
                    scoreTensor,
                    zero,
                    name: "minicpm_audio_attention_neutral_mask")
            }
            precondition(
                scoreTensor.dataType == .bFloat16,
                "MiniCPM audio masked scores must remain BF16")
            let probabilityTensor = graph.softMax(
                with: scoreTensor,
                axis: 3,
                name: "minicpm_audio_attention_softmax")
            precondition(
                probabilityTensor.dataType == .bFloat16,
                "MiniCPM audio attention probabilities must remain BF16")
            let contextTensor = graph.matrixMultiplication(
                primary: probabilityTensor,
                secondary: valueTensor,
                name: "minicpm_audio_attention_pv")
            precondition(
                contextTensor.dataType == .bFloat16,
                "MiniCPM audio attention context must remain BF16")
            let created = Plan(
                graph: graph,
                query: queryTensor,
                key: keyTensor,
                value: valueTensor,
                scores: scoreTensor,
                probabilities: probabilityTensor,
                context: contextTensor)
            plans[keyID] = created
            plan = created
        }
        planLock.unlock()

        let device = execution.device
        let queryBuffer = noCopyBuffer(q, device: device, label: "query")
        let keyBuffer = noCopyBuffer(k, device: device, label: "key")
        let valueBuffer = noCopyBuffer(v, device: device, label: "value")
        let contextShape = [q.dim(0), q.dim(1), q.dim(2), v.dim(3)]
        let context = MLXArray.zeros(contextShape, dtype: .bfloat16)
        let contextBuffer = noCopyBuffer(context, device: device, label: "context")
        let scores: MLXArray
        let probabilities: MLXArray?
        if captureIntermediates {
            scores = MLXArray.zeros(
                [q.dim(0), q.dim(1), q.dim(2), k.dim(2)], dtype: .bfloat16)
            probabilities = MLXArray.zeros(
                [q.dim(0), q.dim(1), q.dim(2), k.dim(2)], dtype: .bfloat16)
        } else {
            // These values are not consumed by production callers.  Keep
            // correctly-shaped placeholders in the result without binding
            // extra graph outputs, which avoids an unnecessary device copy.
            scores = MLXArray.zeros(
                [q.dim(0), q.dim(1), q.dim(2), k.dim(2)], dtype: .bfloat16)
            probabilities = nil
        }

        let queryData = MPSGraphTensorData(
            queryBuffer,
            shape: q.shape.map(NSNumber.init),
            dataType: .bFloat16)
        let keyData = MPSGraphTensorData(
            keyBuffer,
            shape: k.shape.map(NSNumber.init),
            dataType: .bFloat16)
        let valueData = MPSGraphTensorData(
            valueBuffer,
            shape: v.shape.map(NSNumber.init),
            dataType: .bFloat16)
        let contextData = MPSGraphTensorData(
            contextBuffer,
            shape: contextShape.map(NSNumber.init),
            dataType: .bFloat16)
        var resultDictionary: [MPSGraphTensor: MPSGraphTensorData] = [
            plan.context: contextData,
        ]
        if captureIntermediates {
            let scoreBuffer = noCopyBuffer(scores, device: device, label: "scores")
            let probabilityBuffer = noCopyBuffer(
                probabilities!, device: device, label: "probabilities")
            resultDictionary[plan.scores] = MPSGraphTensorData(
                scoreBuffer,
                shape: scores.shape.map(NSNumber.init),
                dataType: .bFloat16)
            resultDictionary[plan.probabilities] = MPSGraphTensorData(
                probabilityBuffer,
                shape: probabilities!.shape.map(NSNumber.init),
                dataType: .bFloat16)
        }

        execution.lock.lock()
        defer { execution.lock.unlock() }
        plan.graph.run(
            with: execution.commandQueue,
            feeds: [
                plan.query: queryData,
                plan.key: keyData,
                plan.value: valueData,
            ],
            targetOperations: nil,
            resultsDictionary: resultDictionary)
        if captureIntermediates {
            eval(scores, probabilities!, context)
        } else {
            eval(context)
        }
        return Result(
            scores: scores,
            probabilities: probabilities,
            context: context)
    }

    private static func validate(
        query: MLXArray,
        key: MLXArray,
        value: MLXArray
    ) {
        precondition(query.dtype == .bfloat16, "MiniCPM audio attention query must be BF16")
        precondition(key.dtype == .bfloat16, "MiniCPM audio attention key must be BF16")
        precondition(value.dtype == .bfloat16, "MiniCPM audio attention value must be BF16")
        precondition(query.ndim == 4 && key.ndim == 4 && value.ndim == 4)
        precondition(key.shape == value.shape)
        precondition(query.dim(0) == key.dim(0))
        precondition(query.dim(1) == key.dim(1))
        precondition(query.dim(2) > 0 && key.dim(2) > 0)
        precondition(query.dim(3) == key.dim(3))
    }

    private static func rowContiguous(_ array: MLXArray) -> MLXArray {
        eval(array)
        let backing = array.asData(access: .noCopy)
        guard isRowContiguous(shape: array.shape, strides: backing.strides) else {
            return array.contiguous()
        }
        return array
    }

    private static func noCopyBuffer(
        _ array: MLXArray,
        device: any MTLDevice,
        label: String
    ) -> any MTLBuffer {
        eval(array)
        let backing = array.asData(access: .noCopy)
        precondition(
            backing.dType == .bfloat16
                && isRowContiguous(shape: array.shape, strides: backing.strides),
            "MiniCPM audio attention \(label) must be row-contiguous BF16")
        guard let buffer = array.asMTLBuffer(device: device, noCopy: true) else {
            preconditionFailure(
                "MiniCPM audio attention could not obtain \(label) MTLBuffer")
        }
        precondition(
            buffer.length >= array.nbytes,
            "MiniCPM audio attention \(label) buffer is too small")
        return buffer
    }

    private static func isRowContiguous(shape: [Int], strides: [Int]) -> Bool {
        guard shape.count == strides.count else { return false }
        let expected = contiguousStrides(for: shape)
        return zip(zip(shape, strides), expected).allSatisfy { pair, expectedStride in
            let (dimension, stride) = pair
            return dimension == 1 || stride == expectedStride
        }
    }

    private static func contiguousStrides(for shape: [Int]) -> [Int] {
        guard !shape.isEmpty else { return [] }
        var strides = Array(repeating: 1, count: shape.count)
        if shape.count > 1 {
            for index in stride(from: shape.count - 2, through: 0, by: -1) {
                strides[index] = strides[index + 1] * shape[index + 1]
            }
        }
        return strides
    }
}
