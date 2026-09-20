import Foundation
import Metal
import MetalPerformanceShadersGraph
import MLX
import MLXNN

/// Dense MiniCPM Linear backed by an MPSGraph BF16 matmul.
///
/// The module intentionally keeps ``Linear``'s inherited ``weight`` and
/// ``bias`` parameters.  The strict MiniCPM weight loader can therefore use
/// the same parameter keys for this class and for the ordinary MLX Linear.
/// Only the dense BF16 path enters MPSGraph; all other dtypes use the exact
/// implementation supplied by ``Linear``.
public final class MiniCPMExactBF16Linear: Linear {
    private struct PlanKey: Hashable {
        let inputShape: [Int]
        let weightShape: [Int]
    }

    /// MPSGraph execution shares one Metal command queue across every exact
    /// BF16 Linear instance.  MPSGraph's no-copy TensorData bridge aliases
    /// MLX-owned buffers, so serializing graph submissions avoids cross-queue
    /// hazards when many projections are evaluated in one model step.
    private final class SharedExecutionContext {
        static let shared = SharedExecutionContext()

        let device: any MTLDevice
        let commandQueue: any MTLCommandQueue
        let executionLock = NSLock()

        private init() {
            guard let device = MTLCreateSystemDefaultDevice() else {
                preconditionFailure("MiniCPMExactBF16Linear requires a Metal GPU device")
            }
            guard let commandQueue = device.makeCommandQueue() else {
                preconditionFailure(
                    "MiniCPMExactBF16Linear could not create a Metal command queue")
            }
            self.device = device
            self.commandQueue = commandQueue
        }
    }

    private final class Plan {
        let graph: MPSGraph
        let inputTensor: MPSGraphTensor
        let weightTensor: MPSGraphTensor
        let outputTensor: MPSGraphTensor

        init(
            graph: MPSGraph,
            inputTensor: MPSGraphTensor,
            weightTensor: MPSGraphTensor,
            outputTensor: MPSGraphTensor
        ) {
            self.graph = graph
            self.inputTensor = inputTensor
            self.weightTensor = weightTensor
            self.outputTensor = outputTensor
        }
    }

    private static let sharedExecutionContext = SharedExecutionContext.shared
    private let planLock = NSLock()
    private var plans: [PlanKey: Plan] = [:]

    public override init(_ inputDimensions: Int, _ outputDimensions: Int, bias: Bool = true) {
        super.init(inputDimensions, outputDimensions, bias: bias)
    }

    public override init(weight: MLXArray, bias: MLXArray? = nil) {
        super.init(weight: weight, bias: bias)
    }

    public override func callAsFunction(_ input: MLXArray) -> MLXArray {
        // This class remains usable by model-independent fixtures that use
        // Float32 (or another MLX dtype): retain Linear's established path for
        // those calls instead of introducing an implicit cast.
        guard input.dtype == .bfloat16, weight.dtype == .bfloat16 else {
            return super.callAsFunction(input)
        }

        // MiniCPM constructs every projection with bias == false.  MPSGraph
        // bias support is intentionally not hidden behind a fallback: a
        // future caller that requests a BF16 bias gets an explicit failure.
        precondition(
            bias == nil,
            "MiniCPMExactBF16Linear does not support BF16 bias; construct it with bias: false")
        precondition(weight.ndim == 2, "MiniCPMExactBF16Linear requires a rank-2 weight")
        precondition(
            input.ndim >= 1,
            "MiniCPMExactBF16Linear requires an input with at least one dimension")
        precondition(
            input.shape.last == weight.shape.last,
            "MiniCPMExactBF16Linear input width does not match weight")

        let inputShape = input.shape
        let weightShape = weight.shape
        let plan = plan(for: inputShape, weightShape: weightShape)
        let device = metalDevice
        // MPSGraph's TensorData bridge is deliberately no-copy.  MLX views
        // (for example, a transpose) can have valid BF16 values but
        // non-row-contiguous strides, which cannot be represented by that
        // bridge.  Materialize those views in MLX first; this remains a GPU
        // operation and does not introduce a Data/CPU copy or a GEMM fallback.
        let graphInput = rowContiguousInput(input)
        let inputBuffer = noCopyBuffer(graphInput, device: device, label: "BF16 input")
        let weightBuffer = noCopyBuffer(weight, device: device, label: "BF16 weight")

        var outputShape = Array(inputShape.dropLast())
        outputShape.append(weightShape[0])
        let output = MLXArray.zeros(outputShape, dtype: .bfloat16)
        let outputBuffer = noCopyBuffer(output, device: device, label: "BF16 output")

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
        precondition(
            outputData.shape.map(\.intValue) == outputShape
                && outputData.dataType == .bFloat16,
            "MPSGraph BF16 output TensorData shape/dtype mismatch")

        let results: [MPSGraphTensor: MPSGraphTensorData] = [
            plan.outputTensor: outputData,
        ]
        let executionContext = Self.sharedExecutionContext
        executionContext.executionLock.lock()
        defer { executionContext.executionLock.unlock() }
        plan.graph.run(
            with: executionContext.commandQueue,
            feeds: [plan.inputTensor: inputData, plan.weightTensor: weightData],
            targetOperations: nil,
            resultsDictionary: results)
        return output
    }

    private func rowContiguousInput(_ input: MLXArray) -> MLXArray {
        eval(input)
        let backing = input.asData(access: .noCopy)
        guard !isRowContiguous(shape: input.shape, strides: backing.strides) else {
            return input
        }
        return input.contiguous()
    }

    private var metalDevice: any MTLDevice { Self.sharedExecutionContext.device }

    private func plan(for inputShape: [Int], weightShape: [Int]) -> Plan {
        planLock.lock()
        defer { planLock.unlock() }

        let key = PlanKey(inputShape: inputShape, weightShape: weightShape)
        if let cached = plans[key] { return cached }

        let graph = MPSGraph()
        let inputTensor = graph.placeholder(
            shape: inputShape.map(NSNumber.init),
            dataType: .bFloat16,
            name: "minicpm_bf16_linear_input")
        let weightTensor = graph.placeholder(
            shape: weightShape.map(NSNumber.init),
            dataType: .bFloat16,
            name: "minicpm_bf16_linear_weight")
        let transposedWeight = graph.transpose(
            weightTensor,
            permutation: [1, 0],
            name: "minicpm_bf16_linear_weight_transpose")
        let outputTensor = graph.matrixMultiplication(
            primary: inputTensor,
            secondary: transposedWeight,
            name: "minicpm_bf16_linear_matmul")
        var expectedOutputShape = Array(inputShape.dropLast())
        expectedOutputShape.append(weightShape[0])
        precondition(
            outputTensor.shape?.map(\.intValue) == expectedOutputShape,
            "MPSGraph BF16 Linear inferred an unexpected output shape")
        precondition(
            outputTensor.dataType == .bFloat16,
            "MPSGraph BF16 Linear inferred a non-BF16 output")

        let created = Plan(
            graph: graph,
            inputTensor: inputTensor,
            weightTensor: weightTensor,
            outputTensor: outputTensor)
        plans[key] = created
        return created
    }

    private func noCopyBuffer(
        _ array: MLXArray,
        device: any MTLDevice,
        label: String
    ) -> any MTLBuffer {
        eval(array)
        precondition(array.dtype == .bfloat16, "\(label) must be BF16")
        let backing = array.asData(access: .noCopy)
        precondition(
            backing.dType == .bfloat16
                && isRowContiguous(shape: array.shape, strides: backing.strides),
            "\(label) must be row-contiguous; refusing an implicit copy")
        guard let buffer = array.asMTLBuffer(device: device, noCopy: true) else {
            preconditionFailure("MLX no-copy MTLBuffer is unavailable for \(label)")
        }
        precondition(
            buffer.length >= array.nbytes,
            "MLX no-copy MTLBuffer is too small for \(label)")
        return buffer
    }

    /// MLX can retain arbitrary strides on axes whose shape is one (for
    /// example, a broadcast view). Such axes never advance the logical
    /// address, so the no-copy bridge only requires standard row strides on
    /// axes with more than one element.
    private func isRowContiguous(shape: [Int], strides: [Int]) -> Bool {
        guard shape.count == strides.count else { return false }
        let expected = contiguousStrides(for: shape)
        return zip(zip(shape, strides), expected).allSatisfy { pair, expectedStride in
            let (dimension, stride) = pair
            return dimension == 1 || stride == expectedStride
        }
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
}
