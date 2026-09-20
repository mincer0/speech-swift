import Foundation
import Metal
import MetalPerformanceShadersGraph
import MLX
import MLXNN

/// Diagnostic dense, bias-free BF16 projection for the MiniCPM semantic decoder.
///
/// The pinned PyTorch/MPS reference performs these projections as BF16 matrix
/// multiplications.  MLX's ordinary `Linear` is numerically very close, but
/// its reduction/fusion choices can differ by an extra BF16 ULP on the long
/// 3,072-wide TTS MLP.  Keeping the graph boundary explicitly BF16 makes the
/// diagnostic path use the same MPSGraph matmul family as the MiniCPM LLM.  It
/// is opt-in with `MINICPM_TTS_BF16_LINEAR=mpsgraph`; ordinary MLX `Linear`
/// remains the production default until this kernel passes the full oracle and
/// stable-token gates.
final class MiniCPMTTSExactBF16Linear: Linear {
    private struct PlanKey: Hashable {
        let inputShape: [Int]
        let weightShape: [Int]
    }

    private final class ExecutionContext {
        static let shared = ExecutionContext()

        let device: any MTLDevice
        let commandQueue: any MTLCommandQueue
        let lock = NSLock()

        private init() {
            guard let device = MTLCreateSystemDefaultDevice() else {
                preconditionFailure("MiniCPM TTS exact BF16 Linear requires a Metal device")
            }
            guard let commandQueue = device.makeCommandQueue() else {
                preconditionFailure("MiniCPM TTS exact BF16 Linear could not create a command queue")
            }
            self.device = device
            self.commandQueue = commandQueue
        }
    }

    private final class Plan {
        let graph: MPSGraph
        let input: MPSGraphTensor
        let weight: MPSGraphTensor
        let output: MPSGraphTensor

        init(
            graph: MPSGraph,
            input: MPSGraphTensor,
            weight: MPSGraphTensor,
            output: MPSGraphTensor
        ) {
            self.graph = graph
            self.input = input
            self.weight = weight
            self.output = output
        }
    }

    private static let execution = ExecutionContext.shared
    private let planLock = NSLock()
    private var plans: [PlanKey: Plan] = [:]

    override init(_ inputDimensions: Int, _ outputDimensions: Int, bias: Bool = true) {
        super.init(inputDimensions, outputDimensions, bias: bias)
    }

    override init(weight: MLXArray, bias: MLXArray? = nil) {
        super.init(weight: weight, bias: bias)
    }

    override func callAsFunction(_ input: MLXArray) -> MLXArray {
        guard input.dtype == .bfloat16,
              weight.dtype == .bfloat16,
              bias == nil
        else {
            return super.callAsFunction(input)
        }

        precondition(weight.ndim == 2, "MiniCPM TTS exact Linear requires rank-2 weights")
        precondition(input.ndim >= 1, "MiniCPM TTS exact Linear requires a non-empty input rank")
        precondition(
            input.shape.last == weight.shape.last,
            "MiniCPM TTS exact Linear input width does not match weight")

        let inputShape = input.shape
        let weightShape = weight.shape
        let plan = plan(for: inputShape, weightShape: weightShape)
        let graphInput = contiguousIfNeeded(input)
        let inputBuffer = noCopyBuffer(graphInput, label: "TTS BF16 input")
        let weightBuffer = noCopyBuffer(weight, label: "TTS BF16 weight")

        var outputShape = Array(inputShape.dropLast())
        outputShape.append(weightShape[0])
        let output = MLXArray.zeros(outputShape, dtype: .bfloat16)
        let outputBuffer = noCopyBuffer(output, label: "TTS BF16 output")
        let inputData = MPSGraphTensorData(
            inputBuffer,
            shape: inputShape.map(NSNumber.init),
            dataType: .bFloat16)
        let weightData = MPSGraphTensorData(
            weightBuffer,
            shape: weightShape.map(NSNumber.init),
            dataType: .bFloat16)
        let outputData = MPSGraphTensorData(
            outputBuffer,
            shape: outputShape.map(NSNumber.init),
            dataType: .bFloat16)

        let context = Self.execution
        context.lock.lock()
        defer { context.lock.unlock() }
        plan.graph.run(
            with: context.commandQueue,
            feeds: [plan.input: inputData, plan.weight: weightData],
            targetOperations: nil,
            resultsDictionary: [plan.output: outputData])
        return output
    }

    private func plan(for inputShape: [Int], weightShape: [Int]) -> Plan {
        planLock.lock()
        defer { planLock.unlock() }
        let key = PlanKey(inputShape: inputShape, weightShape: weightShape)
        if let plan = plans[key] { return plan }

        let graph = MPSGraph()
        let input = graph.placeholder(
            shape: inputShape.map(NSNumber.init),
            dataType: .bFloat16,
            name: "minicpm_tts_bf16_linear_input")
        let weight = graph.placeholder(
            shape: weightShape.map(NSNumber.init),
            dataType: .bFloat16,
            name: "minicpm_tts_bf16_linear_weight")
        let transposedWeight = graph.transpose(
            weight,
            permutation: [1, 0],
            name: "minicpm_tts_bf16_linear_weight_transpose")
        let output = graph.matrixMultiplication(
            primary: input,
            secondary: transposedWeight,
            name: "minicpm_tts_bf16_linear_matmul")
        var expected = Array(inputShape.dropLast())
        expected.append(weightShape[0])
        precondition(
            output.shape?.map(\.intValue) == expected,
            "MiniCPM TTS exact Linear inferred an unexpected output shape")
        precondition(
            output.dataType == .bFloat16,
            "MiniCPM TTS exact Linear inferred a non-BF16 output")

        let created = Plan(graph: graph, input: input, weight: weight, output: output)
        plans[key] = created
        return created
    }

    private func contiguousIfNeeded(_ input: MLXArray) -> MLXArray {
        eval(input)
        let data = input.asData(access: .noCopy)
        guard isContiguous(shape: input.shape, strides: data.strides) else {
            return input.contiguous()
        }
        return input
    }

    private func noCopyBuffer(_ array: MLXArray, label: String) -> any MTLBuffer {
        eval(array)
        let data = array.asData(access: .noCopy)
        precondition(
            array.dtype == .bfloat16 && data.dType == .bfloat16
                && isContiguous(shape: array.shape, strides: data.strides),
            "(label) must be a row-contiguous BF16 array")
        guard let buffer = array.asMTLBuffer(device: Self.execution.device, noCopy: true) else {
            preconditionFailure("MLX no-copy MTLBuffer is unavailable for (label)")
        }
        precondition(buffer.length >= array.nbytes, "MLX no-copy MTLBuffer is too small for (label)")
        return buffer
    }

    private func isContiguous(shape: [Int], strides: [Int]) -> Bool {
        guard shape.count == strides.count else { return false }
        guard !shape.isEmpty else { return true }
        var expected = Array(repeating: 1, count: shape.count)
        if shape.count > 1 {
            for index in stride(from: shape.count - 2, through: 0, by: -1) {
                expected[index] = expected[index + 1] * shape[index + 1]
            }
        }
        return zip(zip(shape, strides), expected).allSatisfy { pair, expectedStride in
            let (dimension, stride) = pair
            return dimension == 1 || stride == expectedStride
        }
    }
}
