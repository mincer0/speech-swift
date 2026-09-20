import Foundation
import Metal
import MetalPerformanceShadersGraph
import MLX
import MLXNN

/// Dense MiniCPM RMSNorm backed by an MPSGraph BF16 graph.
///
/// The dense model's reference path is the MPS implementation used by
/// PyTorch: BF16 input is promoted to FP32, squared and reduced in FP32,
/// reciprocal-square-rooted, normalized in FP32, converted back to BF16, and
/// finally multiplied by the BF16 weight.  Pow(2) and mean deliberately run
/// as separate MPSGraph submissions, mirroring PyTorch's eager MPS dispatch;
/// writing the resulting FP32 variance into MLX preserves the next operation
/// boundary before the remaining pointwise work and BF16 rounding.
///
/// Only an actual dense-BF16 call enters this path.  Float32, mixed-precision,
/// and quantized-model calls retain ``RMSNorm``'s established fused behavior.
/// Shared dense-BF16 RMSNorm implementation used by MiniCPM language and
/// semantic-TTS decoders.  The public surface is intentionally tiny so other
/// native MiniCPM components can reuse the already-validated arithmetic rather
/// than maintaining subtly different MPSGraph copies.
public final class MiniCPMExactBF16RMSNorm: RMSNorm {
    private struct PlanKey: Hashable {
        let inputShape: [Int]
        let weightShape: [Int]
        let epsilonBits: UInt32
        let inputDType: DType
        let weightDType: DType
    }

    /// MPSGraph's no-copy TensorData bridge aliases MLX-owned buffers.  All
    /// exact RMSNorm instances therefore share one Metal device, command
    /// queue, and submission lock, just like the exact Linear and SDPA paths.
    private final class SharedExecutionContext {
        static let shared = SharedExecutionContext()

        let device: any MTLDevice
        let commandQueue: any MTLCommandQueue
        let executionLock = NSLock()

        private init() {
            guard let device = MTLCreateSystemDefaultDevice() else {
                preconditionFailure("MiniCPMExactBF16RMSNorm requires a Metal GPU device")
            }
            guard let commandQueue = device.makeCommandQueue() else {
                preconditionFailure(
                    "MiniCPMExactBF16RMSNorm could not create a Metal command queue")
            }
            self.device = device
            self.commandQueue = commandQueue
        }
    }

    private final class Plan {
        let powerGraph: MPSGraph
        let powerInputTensor: MPSGraphTensor
        let exponentTensor: MPSGraphTensor
        let squaredTensor: MPSGraphTensor
        let meanGraph: MPSGraph
        let meanInputTensor: MPSGraphTensor
        let varianceTensor: MPSGraphTensor

        init(
            powerGraph: MPSGraph,
            powerInputTensor: MPSGraphTensor,
            exponentTensor: MPSGraphTensor,
            squaredTensor: MPSGraphTensor,
            meanGraph: MPSGraph,
            meanInputTensor: MPSGraphTensor,
            varianceTensor: MPSGraphTensor
        ) {
            self.powerGraph = powerGraph
            self.powerInputTensor = powerInputTensor
            self.exponentTensor = exponentTensor
            self.squaredTensor = squaredTensor
            self.meanGraph = meanGraph
            self.meanInputTensor = meanInputTensor
            self.varianceTensor = varianceTensor
        }
    }

    private static let sharedExecutionContext = SharedExecutionContext.shared
    private let planLock = NSLock()
    private var plans: [PlanKey: Plan] = [:]

    public override init(dimensions: Int, eps: Float = 1e-5) {
        super.init(dimensions: dimensions, eps: eps)
    }

    public override func callAsFunction(_ input: MLXArray) -> MLXArray {
        // Preserve RMSNorm's established behavior outside the dense BF16
        // route.  In particular, quantized weights and mixed precision must
        // never be implicitly cast into the MPSGraph path.
        guard input.dtype == .bfloat16, weight.dtype == .bfloat16 else {
            return super.callAsFunction(input)
        }
        if let error = Self.validationError(input: input, weight: weight) {
            preconditionFailure(error)
        }

        let graphInput = rowContiguous(input)
        let graphWeight = rowContiguous(weight)
        let inputShape = graphInput.shape
        let weightShape = graphWeight.shape
        let plan = plan(
            inputShape: inputShape,
            weightShape: weightShape,
            inputDType: graphInput.dtype,
            weightDType: graphWeight.dtype)
        let device = Self.sharedExecutionContext.device
        // PyTorch MPS executes BF16->FP32, pow(2), and mean as three eager
        // operations.  Keep the cast materialized before entering the two
        // separate MPSGraph plans below; a single graph is free to fuse and
        // algebraically rewrite these operations, which changes reduction
        // rounding at real q_norm boundaries.
        let inputFloat = rowContiguous(graphInput.asType(.float32))
        let inputBuffer = noCopyBuffer(
            inputFloat, device: device, label: "FP32 input", expectedDType: .float32)
        // PyTorch's tensor-scalar pow graph receives the scalar through a
        // rank-0 placeholder.  Keeping it as a feed prevents MPSGraph from
        // constant-folding pow(x, 2) back into its numerically different
        // square primitive.
        let exponent = MLXArray(Float(2))
        let exponentBuffer = noCopyBuffer(
            exponent, device: device, label: "FP32 exponent", expectedDType: .float32)
        let squared = MLXArray.zeros(inputShape, dtype: .float32)
        let squaredBuffer = noCopyBuffer(
            squared, device: device, label: "FP32 squared", expectedDType: .float32)
        let varianceShape = Array(inputShape.dropLast()) + [1]
        let variance = MLXArray.zeros(varianceShape, dtype: .float32)
        let varianceBuffer = noCopyBuffer(
            variance, device: device, label: "FP32 variance", expectedDType: .float32)
        let inputData = MPSGraphTensorData(
            inputBuffer,
            shape: inputShape.map(NSNumber.init),
            dataType: .float32)
        let exponentData = MPSGraphTensorData(
            exponentBuffer,
            shape: [],
            dataType: .float32)
        let squaredData = MPSGraphTensorData(
            squaredBuffer,
            shape: inputShape.map(NSNumber.init),
            dataType: .float32)
        let varianceData = MPSGraphTensorData(
            varianceBuffer,
            shape: varianceShape.map(NSNumber.init),
            dataType: .float32)

        precondition(
            squaredData.shape.map(\.intValue) == inputShape
                && squaredData.dataType == .float32,
            "MPSGraph RMSNorm squared TensorData shape/dtype mismatch")
        precondition(
            varianceData.shape.map(\.intValue) == varianceShape
                && varianceData.dataType == .float32,
            "MPSGraph BF16 RMSNorm variance TensorData shape/dtype mismatch")
        precondition(
            plan.varianceTensor.shape?.map(\.intValue) == varianceShape,
            "MPSGraph BF16 RMSNorm inferred an unexpected variance shape")
        precondition(
            plan.varianceTensor.dataType == .float32,
            "MPSGraph BF16 RMSNorm variance must remain FP32")

        let executionContext = Self.sharedExecutionContext
        executionContext.executionLock.lock()
        defer { executionContext.executionLock.unlock() }
        plan.powerGraph.run(
            with: executionContext.commandQueue,
            feeds: [
                plan.powerInputTensor: inputData,
                plan.exponentTensor: exponentData,
            ],
            targetOperations: nil,
            resultsDictionary: [plan.squaredTensor: squaredData])
        plan.meanGraph.run(
            with: executionContext.commandQueue,
            feeds: [plan.meanInputTensor: squaredData],
            targetOperations: nil,
            resultsDictionary: [plan.varianceTensor: varianceData])

        // The variance is deliberately materialized into an MLX FP32 array
        // before the remaining pointwise operations.  This is a GPU-backed
        // no-copy bridge, but it also preserves the operation boundary that
        // PyTorch MPS uses instead of letting MPSGraph fuse reduction,
        // square-root, and normalization into one kernel.
        eval(variance)
        // Do not replace this with ``MLX.rsqrt``.  Its Metal fast primitive
        // rounds this BF16 boundary differently from PyTorch's MPS path,
        // producing 0.20214844 instead of the reference 0.2041015625.  The
        // reference computes the epsilon add and FP32 square-root first,
        // then takes the FP32 reciprocal before the BF16 cast below.
        let varianceWithEpsilon = variance + MLXArray(eps)
        let root = MLX.sqrt(varianceWithEpsilon)
        let inverseRoot = MLXArray(Float(1)) / root
        let normalized = (inputFloat * inverseRoot).asType(.bfloat16)
        return normalized * graphWeight
    }

    /// Validate graph geometry without deliberately crashing an XCTest
    /// process. Production callers still fail fast through `precondition`.
    public static func validationError(input: MLXArray, weight: MLXArray) -> String? {
        guard input.ndim >= 1 else {
            return "MiniCPMExactBF16RMSNorm requires rank >= 1 input"
        }
        guard weight.ndim == 1 else {
            return "MiniCPMExactBF16RMSNorm requires a rank-1 weight"
        }
        guard input.shape.last == weight.shape[0] else {
            return "MiniCPMExactBF16RMSNorm input width does not match weight"
        }
        guard input.shape.allSatisfy({ $0 > 0 }) else {
            return "MiniCPMExactBF16RMSNorm does not support zero-shaped input"
        }
        guard weight.shape[0] > 0 else {
            return "MiniCPMExactBF16RMSNorm does not support zero-shaped weight"
        }
        return nil
    }

    private func plan(
        inputShape: [Int],
        weightShape: [Int],
        inputDType: DType,
        weightDType: DType
    ) -> Plan {
        planLock.lock()
        defer { planLock.unlock() }

        let key = PlanKey(
            inputShape: inputShape,
            weightShape: weightShape,
            epsilonBits: eps.bitPattern,
            inputDType: inputDType,
            weightDType: weightDType)
        if let cached = plans[key] { return cached }

        let powerGraph = MPSGraph()
        let powerInputTensor = powerGraph.placeholder(
            shape: inputShape.map(NSNumber.init),
            dataType: .float32,
            name: "minicpm_bf16_rmsnorm_power_input_fp32")
        // Qwen3 spells this operation as `hidden_states.pow(2)`.  On MPS,
        // MPSGraph's dedicated square op can differ from power(x, 2) by a
        // couple of FP32 reduction ULPs.  That is enough to cross a BF16
        // rounding midpoint in real q_norm heads, so retain the same primitive
        // used by PyTorch instead of applying the algebraic square rewrite.
        let exponent = powerGraph.placeholder(
            shape: [],
            dataType: .float32,
            name: "minicpm_bf16_rmsnorm_exponent_fp32")
        let squared = powerGraph.power(
            powerInputTensor,
            exponent,
            name: "minicpm_bf16_rmsnorm_pow2_fp32")

        // Keep mean in a second graph so MPSGraph cannot fuse pow(2) into the
        // reduction.  PyTorch eager MPS dispatches these as separate cached
        // graphs, and its BF16 reference bytes depend on that boundary.
        let meanGraph = MPSGraph()
        let meanInputTensor = meanGraph.placeholder(
            shape: inputShape.map(NSNumber.init),
            dataType: .float32,
            name: "minicpm_bf16_rmsnorm_mean_input_fp32")
        let variance = meanGraph.mean(
            of: meanInputTensor,
            axes: [NSNumber(value: inputShape.count - 1)],
            name: "minicpm_bf16_rmsnorm_mean_fp32")
        let varianceShape = Array(inputShape.dropLast()) + [1]
        precondition(
            variance.shape?.map(\.intValue) == varianceShape,
            "MPSGraph BF16 RMSNorm inferred an unexpected variance shape")
        precondition(
            variance.dataType == .float32,
            "MPSGraph BF16 RMSNorm variance must remain FP32")

        let created = Plan(
            powerGraph: powerGraph,
            powerInputTensor: powerInputTensor,
            exponentTensor: exponent,
            squaredTensor: squared,
            meanGraph: meanGraph,
            meanInputTensor: meanInputTensor,
            varianceTensor: variance)
        plans[key] = created
        return created
    }

    private func rowContiguous(_ array: MLXArray) -> MLXArray {
        eval(array)
        let backing = array.asData(access: .noCopy)
        guard !isRowContiguous(shape: array.shape, strides: backing.strides) else {
            return array
        }
        // `contiguous()` is an MLX GPU operation.  It is required before the
        // no-copy MPSGraph bridge and does not create a CPU/Data fallback.
        return array.contiguous()
    }

    private func noCopyBuffer(
        _ array: MLXArray,
        device: any MTLDevice,
        label: String,
        expectedDType: DType = .bfloat16
    ) -> any MTLBuffer {
        eval(array)
        precondition(array.dtype == expectedDType, "\(label) has an unexpected dtype")
        let backing = array.asData(access: .noCopy)
        precondition(
            backing.dType == expectedDType
                && backing.shape == array.shape
                && isRowContiguous(shape: array.shape, strides: backing.strides),
            "\(label) must be row-contiguous \(expectedDType); refusing an implicit copy")
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
