import Foundation
import Metal
import MetalPerformanceShaders
import MetalPerformanceShadersGraph
import MLX
import MLXNN
import MiniCPMLLM

/// An optional BF16 MLP execution path with one final CPU wait. The gate/up
/// and down matmuls remain separate MPSGraph plans; only the existing
/// pointwise activation and multiplication run between them on the queue.
enum MiniCPMTTSEncodedBF16MLP {
    private struct PlanKey: Hashable {
        let inputShape: [Int]
        let gateShape: [Int]
        let upShape: [Int]
        let downShape: [Int]
    }

    private final class Plan {
        let projections: MPSGraph
        let projectionInput: MPSGraphTensor
        let gateWeight: MPSGraphTensor
        let upWeight: MPSGraphTensor
        let gateOutput: MPSGraphTensor
        let upOutput: MPSGraphTensor
        let down: MPSGraph
        let downInput: MPSGraphTensor
        let downWeight: MPSGraphTensor
        let downOutput: MPSGraphTensor

        init?(key: PlanKey) {
            // Match the independent weight placeholders, transposes and
            // matmuls in MiniCPMTTSBatchedBF16Projection. No weight packing or
            // cross-matmul algebraic rewriting is introduced here.
            let projections = MPSGraph()
            projections.options = .none
            let input = projections.placeholder(
                shape: key.inputShape.map(NSNumber.init), dataType: .bFloat16,
                name: "minicpm_bf16_projection_input")
            var weights: [MPSGraphTensor] = []
            var outputs: [MPSGraphTensor] = []
            for (index, shape) in [key.gateShape, key.upShape].enumerated() {
                let weight = projections.placeholder(
                    shape: shape.map(NSNumber.init), dataType: .bFloat16,
                    name: "minicpm_bf16_projection_weight_\(index)")
                let transposed = projections.transpose(
                    weight, permutation: [1, 0],
                    name: "minicpm_bf16_projection_transpose_\(index)")
                let output = projections.matrixMultiplication(
                    primary: input, secondary: transposed,
                    name: "minicpm_bf16_projection_matmul_\(index)")
                guard output.dataType == .bFloat16,
                      output.shape?.map(\.intValue)
                        == Array(key.inputShape.dropLast()) + [shape[0]]
                else { return nil }
                weights.append(weight)
                outputs.append(output)
            }

            let down = MPSGraph()
            down.options = .none
            let intermediateShape = Array(key.inputShape.dropLast()) + [key.gateShape[0]]
            let downInput = down.placeholder(
                shape: intermediateShape.map(NSNumber.init), dataType: .bFloat16,
                name: "minicpm_bf16_linear_input")
            let downWeight = down.placeholder(
                shape: key.downShape.map(NSNumber.init), dataType: .bFloat16,
                name: "minicpm_bf16_linear_weight")
            let downTransposed = down.transpose(
                downWeight, permutation: [1, 0],
                name: "minicpm_bf16_linear_weight_transpose")
            let downOutput = down.matrixMultiplication(
                primary: downInput, secondary: downTransposed,
                name: "minicpm_bf16_linear_matmul")
            guard downOutput.dataType == .bFloat16,
                  downOutput.shape?.map(\.intValue)
                    == Array(key.inputShape.dropLast()) + [key.downShape[0]]
            else { return nil }

            self.projections = projections
            self.projectionInput = input
            self.gateWeight = weights[0]
            self.upWeight = weights[1]
            self.gateOutput = outputs[0]
            self.upOutput = outputs[1]
            self.down = down
            self.downInput = downInput
            self.downWeight = downWeight
            self.downOutput = downOutput
        }
    }

    private final class ExecutionContext {
        let device: any MTLDevice
        let queue: any MTLCommandQueue
        let pointwise: any MTLComputePipelineState
        let lock = NSLock()
        var plans: [PlanKey: Plan] = [:]

        init?() {
            guard let device = MTLCreateSystemDefaultDevice(),
                  device.supportsFamily(.apple7),
                  let queue = device.makeCommandQueue()
            else { return nil }
            let options = MTLCompileOptions()
            // Pinned MLX also disables fast math for its Metal kernels.
            options.fastMathEnabled = false
            options.languageVersion = .version3_1
            guard let library = try? device.makeLibrary(
                source: MiniCPMTTSEncodedBF16MLP.pointwiseSource, options: options),
                  let function = library.makeFunction(name: "minicpm_bf16_silu_product"),
                  let pointwise = try? device.makeComputePipelineState(function: function)
            else { return nil }
            self.device = device
            self.queue = queue
            self.pointwise = pointwise
        }
    }

    private final class ExecutionState: @unchecked Sendable {
        private let lock = NSLock()
        private var failed = false

        func recordFailure() {
            lock.lock()
            failed = true
            lock.unlock()
        }

        var didFail: Bool {
            lock.lock()
            defer { lock.unlock() }
            return failed
        }
    }

    private struct TensorBridge {
        let owner: MLXArray
        let buffer: any MTLBuffer
        let tensorData: MPSGraphTensorData

        init?(_ array: MLXArray, device: any MTLDevice) {
            let backing = array.asData(access: .noCopy)
            guard array.dtype == .bfloat16, backing.dType == .bfloat16,
                  backing.strides.count == array.ndim
            else { return nil }
            var expectedStride = 1
            for axis in array.shape.indices.reversed() {
                guard array.dim(axis) == 1 || backing.strides[axis] == expectedStride else {
                    return nil
                }
                expectedStride *= array.dim(axis)
            }
            guard let buffer = array.asMTLBuffer(device: device, noCopy: true),
                  buffer.length >= array.nbytes
            else { return nil }
            self.owner = array
            self.buffer = buffer
            self.tensorData = MPSGraphTensorData(
                buffer, shape: array.shape.map(NSNumber.init), dataType: .bFloat16)
        }
    }

    private static let context = ExecutionContext()

    static var isAvailable: Bool { context != nil }

    static func apply(
        input: MLXArray, gate: Linear, up: Linear, down: Linear
    ) -> MLXArray? {
        guard input.dtype == .bfloat16, input.ndim >= 1,
              input.shape.allSatisfy({ $0 > 0 }),
              let exactGate = exactProjection(gate),
              let exactUp = exactProjection(up),
              let exactDown = exactProjection(down),
              gate.weight.dim(1) == input.dim(-1),
              up.weight.shape == gate.weight.shape,
              down.weight.dim(1) == gate.weight.dim(0),
              let execution = context
        else { return nil }

        let intermediateShape = Array(input.shape.dropLast()) + [gate.weight.dim(0)]
        let outputShape = Array(input.shape.dropLast()) + [down.weight.dim(0)]
        let graphInput = input.contiguous()
        let gateOutput = MLXArray.zeros(intermediateShape, dtype: .bfloat16)
        let upOutput = MLXArray.zeros(intermediateShape, dtype: .bfloat16)
        let product = MLXArray.zeros(intermediateShape, dtype: .bfloat16)
        let output = MLXArray.zeros(outputShape, dtype: .bfloat16)
        guard product.size <= Int(UInt32.max) else { return nil }

        let gateWeight = exactGate.preparedBF16Weight(device: execution.device)
        let upWeight = exactUp.preparedBF16Weight(device: execution.device)
        let downWeight = exactDown.preparedBF16Weight(device: execution.device)
        // All dynamic arrays must be available before the external queue can
        // write their backing. Nothing re-enters MLX until the final wait.
        eval([graphInput, gateOutput, upOutput, product, output])
        guard let inputBridge = TensorBridge(graphInput, device: execution.device),
              let gateBridge = TensorBridge(gateOutput, device: execution.device),
              let upBridge = TensorBridge(upOutput, device: execution.device),
              let productBridge = TensorBridge(product, device: execution.device),
              let outputBridge = TensorBridge(output, device: execution.device)
        else { return nil }

        execution.lock.lock()
        defer { execution.lock.unlock() }
        let key = PlanKey(
            inputShape: input.shape, gateShape: gateWeight.shape,
            upShape: upWeight.shape, downShape: downWeight.shape)
        let plan: Plan
        if let cached = execution.plans[key] {
            plan = cached
        } else {
            guard let created = Plan(key: key) else { return nil }
            execution.plans[key] = created
            plan = created
        }
        guard let initialCommandBuffer = execution.queue.makeCommandBuffer() else { return nil }
        let commandBuffer = MPSCommandBuffer(commandBuffer: initialCommandBuffer)
        let state = ExecutionState()
        let descriptor = MPSGraphExecutionDescriptor()
        descriptor.waitUntilCompleted = false
        descriptor.completionHandler = { _, error in
            if error != nil { state.recordFailure() }
        }
        initialCommandBuffer.addCompletedHandler { completed in
            if completed.status == .error { state.recordFailure() }
        }

        return withExtendedLifetime((
            inputBridge, gateBridge, upBridge, productBridge, outputBridge,
            gateWeight, upWeight, downWeight, descriptor
        )) {
            plan.projections.encode(
                to: commandBuffer,
                feeds: [plan.projectionInput: inputBridge.tensorData,
                        plan.gateWeight: gateWeight.tensorData,
                        plan.upWeight: upWeight.tensorData],
                targetOperations: nil,
                resultsDictionary: [plan.gateOutput: gateBridge.tensorData,
                                    plan.upOutput: upBridge.tensorData],
                executionDescriptor: descriptor)

            // MPSGraph may commitAndContinue while encoding. Obtain the
            // current underlying buffer after every graph call rather than
            // keeping the buffer used to initialize MPSCommandBuffer.
            if let encoder = commandBuffer.rootCommandBuffer.makeComputeCommandEncoder() {
                encoder.setComputePipelineState(execution.pointwise)
                encoder.setBuffer(gateBridge.buffer, offset: 0, index: 0)
                encoder.setBuffer(upBridge.buffer, offset: 0, index: 1)
                encoder.setBuffer(productBridge.buffer, offset: 0, index: 2)
                var count = UInt32(product.size)
                encoder.setBytes(&count, length: MemoryLayout<UInt32>.size, index: 3)
                let groupWidth = min(256, execution.pointwise.maxTotalThreadsPerThreadgroup)
                encoder.dispatchThreads(
                    MTLSize(width: product.size, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: groupWidth, height: 1, depth: 1))
                encoder.endEncoding()
                commandBuffer.rootCommandBuffer.addCompletedHandler { completed in
                    if completed.status == .error { state.recordFailure() }
                }

                plan.down.encode(
                    to: commandBuffer,
                    feeds: [plan.downInput: productBridge.tensorData,
                            plan.downWeight: downWeight.tensorData],
                    targetOperations: nil,
                    resultsDictionary: [plan.downOutput: outputBridge.tensorData],
                    executionDescriptor: descriptor)
            } else {
                state.recordFailure()
            }

            // Even an encoder failure must drain any graph work already
            // submitted before releasing the no-copy MLX storage owners.
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
            guard commandBuffer.rootCommandBuffer.status == .completed, !state.didFail else {
                return nil
            }
            return output
        }
    }

    private static func exactProjection(_ projection: Linear) -> MiniCPMExactBF16Linear? {
        guard let exact = projection as? MiniCPMExactBF16Linear,
              exact.bias == nil, exact.weight.dtype == .bfloat16,
              exact.weight.ndim == 2, exact.weight.shape.allSatisfy({ $0 > 0 })
        else { return nil }
        return exact
    }

    private static let pointwiseSource = #"""
        #include <metal_stdlib>
        using namespace metal;

        kernel void minicpm_bf16_silu_product(
            device const bfloat* gate [[buffer(0)]],
            device const bfloat* up [[buffer(1)]],
            device bfloat* product [[buffer(2)]],
            constant uint& count [[buffer(3)]],
            uint index [[thread_position_in_grid]]) {
            if (index >= count) return;
            const float x = float(gate[index]);
            // Match pinned MLX Sigmoid, including its stable positive branch.
            const float y = 1.0f / (1.0f + metal::exp(metal::abs(x)));
            const float sigmoid = x < 0.0f ? y : 1.0f - y;
            // Both BF16 boundaries are required: SiLU rounds before the
            // separate BF16 multiplication by up.
            const bfloat activated = bfloat(x * sigmoid);
            product[index] = bfloat(float(activated) * float(up[index]));
        }
        """#
}
