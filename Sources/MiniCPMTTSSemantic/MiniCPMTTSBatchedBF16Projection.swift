import Foundation
import Metal
import MetalPerformanceShadersGraph
import MLX
import MLXNN
import MiniCPMLLM

/// Submit independent BF16 projections of the same input together. Each
/// projection retains its own weight placeholder, transpose, and matmul;
/// weights and outputs are never concatenated or cast to another dtype.
enum MiniCPMTTSBatchedBF16Projection {
    private struct PlanKey: Hashable {
        let inputShape: [Int]
        let weightShapes: [[Int]]
    }

    private final class Plan {
        let graph: MPSGraph
        let input: MPSGraphTensor
        let weights: [MPSGraphTensor]
        let outputs: [MPSGraphTensor]

        init(inputShape: [Int], weightShapes: [[Int]]) {
            let graph = MPSGraph()
            let input = graph.placeholder(
                shape: inputShape.map(NSNumber.init), dataType: .bFloat16,
                name: "minicpm_bf16_projection_input")
            var weights: [MPSGraphTensor] = []
            var outputs: [MPSGraphTensor] = []
            for (index, shape) in weightShapes.enumerated() {
                let weight = graph.placeholder(
                    shape: shape.map(NSNumber.init), dataType: .bFloat16,
                    name: "minicpm_bf16_projection_weight_\(index)")
                let transposed = graph.transpose(
                    weight, permutation: [1, 0],
                    name: "minicpm_bf16_projection_transpose_\(index)")
                let output = graph.matrixMultiplication(
                    primary: input, secondary: transposed,
                    name: "minicpm_bf16_projection_matmul_\(index)")
                precondition(
                    output.shape?.map(\.intValue) == Array(inputShape.dropLast()) + [shape[0]]
                        && output.dataType == .bFloat16,
                    "MiniCPM grouped projection inferred an unexpected shape or dtype")
                weights.append(weight)
                outputs.append(output)
            }
            self.graph = graph
            self.input = input
            self.weights = weights
            self.outputs = outputs
        }
    }

    private final class ExecutionContext {
        let device: any MTLDevice
        let queue: any MTLCommandQueue
        let lock = NSLock()

        init() {
            guard let device = MTLCreateSystemDefaultDevice(),
                  let queue = device.makeCommandQueue()
            else {
                preconditionFailure("MiniCPM grouped BF16 projections require a Metal queue")
            }
            self.device = device
            self.queue = queue
        }
    }

    private static let context = ExecutionContext()
    private static let planLock = NSLock()
    private static var plans: [PlanKey: Plan] = [:]

    static func canBatch(
        input: MLXArray,
        projections: [Linear],
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> Bool {
        guard environment["MINICPM_TTS_BATCHED_PROJECTIONS"] != "0",
              projections.count > 1,
              input.dtype == .bfloat16,
              input.ndim >= 1,
              input.shape.allSatisfy({ $0 > 0 })
        else { return false }
        return projections.allSatisfy {
            $0 is MiniCPMExactBF16Linear && $0.bias == nil
                && $0.weight.dtype == .bfloat16 && $0.weight.ndim == 2
                && $0.weight.shape.allSatisfy({ $0 > 0 })
                && $0.weight.dim(1) == input.dim(-1)
        }
    }

    static func apply(
        _ input: MLXArray,
        projections: [Linear],
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> [MLXArray] {
        guard canBatch(input: input, projections: projections, environment: environment) else {
            return projections.map { $0(input) }
        }

        let execution = context
        let graphInput = input.contiguous()
        let preparedWeights = projections.map {
            // canBatch has checked that every projection uses this exact
            // implementation. Reuse its storage-owning cached weight feed.
            ($0 as! MiniCPMExactBF16Linear).preparedBF16Weight(device: execution.device)
        }
        let outputs = preparedWeights.map {
            MLXArray.zeros(Array(input.shape.dropLast()) + [$0.shape[0]], dtype: .bfloat16)
        }
        // Materialize all destination arrays in one MLX submission before
        // MPSGraph writes through their no-copy buffers.
        eval([graphInput] + outputs)

        let key = PlanKey(inputShape: input.shape, weightShapes: preparedWeights.map(\.shape))
        planLock.lock()
        let plan: Plan
        if let cached = plans[key] {
            plan = cached
        } else {
            plan = Plan(inputShape: key.inputShape, weightShapes: key.weightShapes)
            plans[key] = plan
        }
        planLock.unlock()

        let inputData = tensorData(graphInput, device: execution.device)
        let outputData = outputs.map { tensorData($0, device: execution.device) }
        var feeds: [MPSGraphTensor: MPSGraphTensorData] = [plan.input: inputData]
        var results: [MPSGraphTensor: MPSGraphTensorData] = [:]
        for index in preparedWeights.indices {
            feeds[plan.weights[index]] = preparedWeights[index].tensorData
            results[plan.outputs[index]] = outputData[index]
        }

        execution.lock.lock()
        defer { execution.lock.unlock() }
        // TensorData retains the buffer wrappers, not the MLX storage owners.
        withExtendedLifetime((graphInput, preparedWeights, outputs)) {
            plan.graph.run(
                with: execution.queue, feeds: feeds, targetOperations: nil,
                resultsDictionary: results)
        }
        return outputs
    }

    private static func tensorData(
        _ array: MLXArray, device: any MTLDevice
    ) -> MPSGraphTensorData {
        let backing = array.asData(access: .noCopy)
        precondition(array.dtype == .bfloat16 && backing.dType == .bfloat16)
        var expectedStride = 1
        for axis in array.shape.indices.reversed() {
            precondition(array.dim(axis) == 1 || backing.strides[axis] == expectedStride,
                         "MiniCPM grouped projection requires row-contiguous storage")
            expectedStride *= array.dim(axis)
        }
        guard let buffer = array.asMTLBuffer(device: device, noCopy: true) else {
            preconditionFailure("MiniCPM grouped projection cannot bridge MLX storage")
        }
        precondition(buffer.length >= array.nbytes)
        return MPSGraphTensorData(
            buffer, shape: array.shape.map(NSNumber.init), dataType: .bFloat16)
    }
}
