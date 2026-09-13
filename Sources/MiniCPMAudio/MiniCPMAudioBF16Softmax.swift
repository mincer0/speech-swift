import Foundation
import Metal
import MetalPerformanceShadersGraph
import MLX

/// BF16 softmax with the same reduction boundary as PyTorch's MPS backend.
///
/// The ordinary MLX softmax is numerically sound, but its BF16 reduction can
/// round at a different point from `torch.nn.functional.softmax` on MPS.  In
/// the MiniCPM Whisper encoder that one-ULP difference is fed into 24
/// recurrently-connected layers and becomes large enough to fail the frozen
/// audio golden.  MPSGraph's BF16 `softMax` is the closest available native
/// operation and is also what the exact MiniCPM LLM attention path uses.
///
/// Plans are cached by shape and all executions are serialized on one command
/// queue.  The MLX arrays are evaluated and bridged without a copy only after
/// their row-contiguous layout has been checked.
public enum MiniCPMAudioBF16Softmax {
    private struct PlanKey: Hashable {
        let shape: [Int]
        let axis: Int
    }

    private final class ExecutionContext {
        static let shared = ExecutionContext()

        let device: any MTLDevice
        let commandQueue: any MTLCommandQueue
        let lock = NSLock()

        private init() {
            guard let device = MTLCreateSystemDefaultDevice() else {
                preconditionFailure("MiniCPM audio BF16 softmax requires a Metal device")
            }
            guard let commandQueue = device.makeCommandQueue() else {
                preconditionFailure("MiniCPM audio BF16 softmax could not create a command queue")
            }
            self.device = device
            self.commandQueue = commandQueue
        }
    }

    private final class Plan {
        let graph: MPSGraph
        let input: MPSGraphTensor
        let output: MPSGraphTensor

        init(graph: MPSGraph, input: MPSGraphTensor, output: MPSGraphTensor) {
            self.graph = graph
            self.input = input
            self.output = output
        }
    }

    private static let context = ExecutionContext.shared
    private static let planLock = NSLock()
    private static var plans: [PlanKey: Plan] = [:]

    /// Apply BF16 softmax to the requested axis.
    public static func apply(_ scores: MLXArray, axis: Int = -1) -> MLXArray {
        precondition(scores.dtype == .bfloat16, "MiniCPM audio softmax input must be BF16")
        precondition(scores.ndim > 0, "MiniCPM audio softmax input must be non-scalar")
        let normalizedAxis = axis >= 0 ? axis : scores.ndim + axis
        precondition(
            normalizedAxis >= 0 && normalizedAxis < scores.ndim,
            "MiniCPM audio softmax axis is out of range")

        let input = rowContiguous(scores)
        let key = PlanKey(shape: input.shape, axis: normalizedAxis)
        planLock.lock()
        let plan: Plan
        if let cached = plans[key] {
            plan = cached
        } else {
            let graph = MPSGraph()
            let placeholder = graph.placeholder(
                shape: input.shape.map(NSNumber.init),
                dataType: .bFloat16,
                name: "minicpm_audio_bf16_softmax_input")
            let output = graph.softMax(
                with: placeholder,
                axis: normalizedAxis,
                name: "minicpm_audio_bf16_softmax")
            precondition(
                output.dataType == .bFloat16,
                "MiniCPM audio MPSGraph softmax must remain BF16")
            let created = Plan(graph: graph, input: placeholder, output: output)
            plans[key] = created
            plan = created
        }
        planLock.unlock()

        eval(input)
        let inputBacking = input.asData(access: .noCopy)
        precondition(
            inputBacking.dType == .bfloat16
                && isRowContiguous(shape: input.shape, strides: inputBacking.strides),
            "MiniCPM audio softmax input must be row-contiguous")
        guard let inputBuffer = input.asMTLBuffer(device: context.device, noCopy: true) else {
            preconditionFailure("MiniCPM audio softmax could not obtain input MTLBuffer")
        }

        let result = MLXArray.zeros(input.shape, dtype: .bfloat16)
        guard let outputBuffer = result.asMTLBuffer(device: context.device, noCopy: true) else {
            preconditionFailure("MiniCPM audio softmax could not obtain output MTLBuffer")
        }
        let inputData = MPSGraphTensorData(
            inputBuffer,
            shape: input.shape.map(NSNumber.init),
            dataType: .bFloat16)
        let outputData = MPSGraphTensorData(
            outputBuffer,
            shape: input.shape.map(NSNumber.init),
            dataType: .bFloat16)

        context.lock.lock()
        defer { context.lock.unlock() }
        plan.graph.run(
            with: context.commandQueue,
            feeds: [plan.input: inputData],
            targetOperations: nil,
            resultsDictionary: [plan.output: outputData])
        return result
    }

    private static func rowContiguous(_ array: MLXArray) -> MLXArray {
        eval(array)
        let backing = array.asData(access: .noCopy)
        guard isRowContiguous(shape: array.shape, strides: backing.strides) else {
            return array.contiguous()
        }
        return array
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
