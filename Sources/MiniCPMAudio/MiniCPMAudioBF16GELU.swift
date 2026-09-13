import Foundation
import Metal
import MetalPerformanceShadersGraph
import MLX

/// MPSGraph implementation of the exact (non-approximate) GELU expression
/// used by the MiniCPM Whisper encoder's PyTorch reference.
///
/// Keep the graph itself BF16.  MPSGraph still evaluates the transcendental
/// operation with device precision, but retaining BF16 at every tensor and
/// constant boundary matches PyTorch MPS's `gelu(approximate: none)` path.
/// Promoting the input or constants to FP32 changes the erf rounding and is
/// measurably non-equivalent on this encoder.
public enum MiniCPMAudioBF16GELU {
    private struct PlanKey: Hashable {
        let shape: [Int]
    }

    private final class ExecutionContext {
        static let shared = ExecutionContext()

        let device: any MTLDevice
        let commandQueue: any MTLCommandQueue
        let lock = NSLock()

        private init() {
            guard let device = MTLCreateSystemDefaultDevice() else {
                preconditionFailure("MiniCPM audio GELU requires a Metal device")
            }
            guard let commandQueue = device.makeCommandQueue() else {
                preconditionFailure("MiniCPM audio GELU could not create a command queue")
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

    /// Compute exact GELU for a BF16 tensor and return BF16.
    public static func apply(_ input: MLXArray) -> MLXArray {
        precondition(input.dtype == .bfloat16, "MiniCPM audio GELU input must be BF16")
        let contiguous = rowContiguous(input)
        let key = PlanKey(shape: contiguous.shape)

        planLock.lock()
        let plan: Plan
        if let cached = plans[key] {
            plan = cached
        } else {
            let graph = MPSGraph()
            let placeholder = graph.placeholder(
                shape: contiguous.shape.map(NSNumber.init),
                dataType: .bFloat16,
                name: "minicpm_audio_gelu_input")
            let sqrtHalf = graph.constant(
                Double(0.7071067811865475244),
                shape: [1].map(NSNumber.init),
                dataType: .bFloat16)
            let half = graph.constant(
                0.5,
                shape: [1].map(NSNumber.init),
                dataType: .bFloat16)
            let one = graph.constant(
                1.0,
                shape: [1].map(NSNumber.init),
                dataType: .bFloat16)
            let scaled = graph.multiplication(
                placeholder,
                sqrtHalf,
                name: "minicpm_audio_gelu_scale")
            let erfValue = graph.erf(
                with: scaled,
                name: "minicpm_audio_gelu_erf")
            let cdf = graph.multiplication(
                graph.addition(erfValue, one, name: "minicpm_audio_gelu_add_one"),
                half,
                name: "minicpm_audio_gelu_half_cdf")
            let result = graph.multiplication(
                cdf,
                placeholder,
                name: "minicpm_audio_gelu_multiply")
            let output = result
            precondition(
                output.dataType == .bFloat16,
                "MiniCPM audio MPSGraph GELU must return BF16")
            let created = Plan(graph: graph, input: placeholder, output: output)
            plans[key] = created
            plan = created
        }
        planLock.unlock()

        eval(contiguous)
        let backing = contiguous.asData(access: .noCopy)
        precondition(
            backing.dType == .bfloat16
                && isRowContiguous(shape: contiguous.shape, strides: backing.strides),
            "MiniCPM audio GELU input must be row-contiguous")
        guard let inputBuffer = contiguous.asMTLBuffer(
            device: context.device, noCopy: true) else {
            preconditionFailure("MiniCPM audio GELU could not obtain input MTLBuffer")
        }
        let output = MLXArray.zeros(contiguous.shape, dtype: .bfloat16)
        guard let outputBuffer = output.asMTLBuffer(
            device: context.device, noCopy: true) else {
            preconditionFailure("MiniCPM audio GELU could not obtain output MTLBuffer")
        }
        let inputData = MPSGraphTensorData(
            inputBuffer,
            shape: contiguous.shape.map(NSNumber.init),
            dataType: .bFloat16)
        let outputData = MPSGraphTensorData(
            outputBuffer,
            shape: contiguous.shape.map(NSNumber.init),
            dataType: .bFloat16)

        context.lock.lock()
        defer { context.lock.unlock() }
        plan.graph.run(
            with: context.commandQueue,
            feeds: [plan.input: inputData],
            targetOperations: nil,
            resultsDictionary: [plan.output: outputData])
        return output
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
