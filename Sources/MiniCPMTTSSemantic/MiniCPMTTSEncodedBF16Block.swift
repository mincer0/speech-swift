import Foundation
import Metal
import MetalPerformanceShaders
import MetalPerformanceShadersGraph
import MLX
import MLXNN
import MiniCPMLLM

/// Optional post-attention execution on a private queue. Each BF16 matmul
/// keeps its original MPSGraph boundary; residuals, RMSNorm and the MLP's
/// pointwise operations run between those graphs before one final CPU wait.
enum MiniCPMTTSEncodedBF16Block {
    private struct PlanKey: Hashable {
        let inputShape: [Int]
        let outputWeightShape: [Int]
        let gateShape: [Int]
        let upShape: [Int]
        let downShape: [Int]
    }

    private final class ProjectionPlan {
        let graph: MPSGraph
        let input: MPSGraphTensor
        let weights: [MPSGraphTensor]
        let outputs: [MPSGraphTensor]

        init?(inputShape: [Int], weightShapes: [[Int]], grouped: Bool) {
            let graph = MPSGraph()
            graph.options = .none
            let prefix = grouped ? "minicpm_bf16_projection" : "minicpm_bf16_linear"
            let input = graph.placeholder(
                shape: inputShape.map(NSNumber.init), dataType: .bFloat16,
                name: prefix + "_input")
            var weights: [MPSGraphTensor] = []
            var outputs: [MPSGraphTensor] = []
            for (index, shape) in weightShapes.enumerated() {
                let suffix = grouped ? "_\(index)" : ""
                let weight = graph.placeholder(
                    shape: shape.map(NSNumber.init), dataType: .bFloat16,
                    name: prefix + "_weight" + suffix)
                let transposed = graph.transpose(
                    weight, permutation: [1, 0],
                    name: grouped ? prefix + "_transpose" + suffix
                        : prefix + "_weight_transpose")
                let output = graph.matrixMultiplication(
                    primary: input, secondary: transposed,
                    name: prefix + "_matmul" + suffix)
                guard output.dataType == .bFloat16,
                      output.shape?.map(\.intValue)
                        == Array(inputShape.dropLast()) + [shape[0]]
                else { return nil }
                weights.append(weight)
                outputs.append(output)
            }
            self.graph = graph
            self.input = input
            self.weights = weights
            self.outputs = outputs
        }

        func encode(
            to commandBuffer: MPSCommandBuffer, inputData: MPSGraphTensorData,
            weightData: [MPSGraphTensorData], outputData: [MPSGraphTensorData],
            descriptor: MPSGraphExecutionDescriptor
        ) {
            var feeds: [MPSGraphTensor: MPSGraphTensorData] = [input: inputData]
            var results: [MPSGraphTensor: MPSGraphTensorData] = [:]
            for index in weights.indices {
                feeds[weights[index]] = weightData[index]
                results[outputs[index]] = outputData[index]
            }
            graph.encode(
                to: commandBuffer, feeds: feeds, targetOperations: nil,
                resultsDictionary: results, executionDescriptor: descriptor)
        }
    }

    private final class Plan {
        let outputProjection: ProjectionPlan
        let gateUp: ProjectionPlan
        let down: ProjectionPlan

        init?(key: PlanKey) {
            let intermediateShape = Array(key.inputShape.dropLast()) + [key.gateShape[0]]
            guard let outputProjection = ProjectionPlan(
                inputShape: key.inputShape, weightShapes: [key.outputWeightShape], grouped: false),
                  let gateUp = ProjectionPlan(
                    inputShape: key.inputShape,
                    weightShapes: [key.gateShape, key.upShape], grouped: true),
                  let down = ProjectionPlan(
                    inputShape: intermediateShape, weightShapes: [key.downShape], grouped: false)
            else { return nil }
            self.outputProjection = outputProjection
            self.gateUp = gateUp
            self.down = down
        }
    }

    private final class ExecutionContext {
        let device: any MTLDevice
        let queue: any MTLCommandQueue
        let add: any MTLComputePipelineState
        let rms: any MTLComputePipelineState
        let siluProduct: any MTLComputePipelineState
        let lock = NSLock()
        var plans: [PlanKey: Plan] = [:]

        init?() {
            guard let device = MTLCreateSystemDefaultDevice(),
                  device.supportsFamily(.apple7), let queue = device.makeCommandQueue()
            else { return nil }
            let options = MTLCompileOptions()
            options.fastMathEnabled = false
            options.languageVersion = .version3_1
            guard let library = try? device.makeLibrary(
                source: MiniCPMTTSEncodedBF16Block.kernelSource, options: options),
                  let addFunction = library.makeFunction(name: "minicpm_block_residual_add"),
                  let rmsFunction = library.makeFunction(name: "minicpm_block_rms_single_row"),
                  let siluFunction = library.makeFunction(name: "minicpm_block_silu_product"),
                  let add = try? device.makeComputePipelineState(function: addFunction),
                  let rms = try? device.makeComputePipelineState(function: rmsFunction),
                  let silu = try? device.makeComputePipelineState(function: siluFunction),
                  rms.threadExecutionWidth == 32
            else { return nil }
            self.device = device
            self.queue = queue
            self.add = add
            self.rms = rms
            self.siluProduct = silu
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

    private enum Slot: Int {
        case residual, attended, projected, attentionResidual, normWeight, normalized
        case gate, up, product, down, output
    }

    private static let context = ExecutionContext()
    private static let profileExecution = ProcessInfo.processInfo.environment[
        "MINICPM_TTS_EXECUTION_PROFILE"] == "1"
    static var isAvailable: Bool { context != nil }

    private static func profileTime() -> UInt64 {
        profileExecution ? DispatchTime.now().uptimeNanoseconds : 0
    }

    static func apply(
        residual: MLXArray, attended: MLXArray, outputProjection: Linear,
        norm: RMSNorm, gate: Linear, up: Linear, down: Linear, epsilon: Float
    ) -> MLXArray? {
        guard residual.dtype == .bfloat16, attended.dtype == .bfloat16,
              residual.ndim >= 1, residual.shape == attended.shape,
              residual.shape.allSatisfy({ $0 > 0 }),
              residual.dim(-1) <= 4096, residual.size <= Int(UInt32.max),
              type(of: norm) == RMSNorm.self,
              norm.weight.dtype == .bfloat16,
              norm.weight.shape == [residual.dim(-1)],
              epsilon.isFinite, epsilon >= 0, epsilon == norm.eps,
              let exactOutput = exactProjection(outputProjection),
              let exactGate = exactProjection(gate),
              let exactUp = exactProjection(up),
              let exactDown = exactProjection(down)
        else { return nil }
        let hidden = residual.dim(-1)
        let intermediate = gate.weight.dim(0)
        guard outputProjection.weight.shape == [hidden, hidden],
              gate.weight.dim(1) == hidden, up.weight.shape == gate.weight.shape,
              down.weight.shape == [hidden, intermediate],
              let execution = context
        else { return nil }
        // Match normalization.cpp exactly: four reads per thread, rounded
        // up to whole 32-thread SIMD groups (192 threads at width 768).
        let rmsGroupSize = ((hidden + 3) / 4 + 31) / 32 * 32
        guard rmsGroupSize <= execution.rms.maxTotalThreadsPerThreadgroup else { return nil }

        let shape = residual.shape
        let intermediateShape = Array(shape.dropLast()) + [intermediate]
        let started = profileTime()
        let arrays: [MLXArray] = [
            residual.contiguous(), attended.contiguous(),
            MLXArray.zeros(shape, dtype: .bfloat16),
            MLXArray.zeros(shape, dtype: .bfloat16), norm.weight.contiguous(),
            MLXArray.zeros(shape, dtype: .bfloat16),
            MLXArray.zeros(intermediateShape, dtype: .bfloat16),
            MLXArray.zeros(intermediateShape, dtype: .bfloat16),
            MLXArray.zeros(intermediateShape, dtype: .bfloat16),
            MLXArray.zeros(shape, dtype: .bfloat16),
            MLXArray.zeros(shape, dtype: .bfloat16),
        ]
        let allocated = profileTime()
        guard arrays[Slot.product.rawValue].size <= Int(UInt32.max) else { return nil }
        let preparedWeights = [exactOutput, exactGate, exactUp, exactDown].map {
            $0.preparedBF16Weight(device: execution.device)
        }
        let weightsPrepared = profileTime()
        // RMSNorm has no cache invalidation hook here, so its current weight
        // is materialized with the dynamic arrays on every call. This also
        // preserves Module.update semantics without a stale norm feed.
        eval(arrays)
        let evaluated = profileTime()
        var bridges: [TensorBridge] = []
        for array in arrays {
            guard let bridge = TensorBridge(array, device: execution.device) else { return nil }
            bridges.append(bridge)
        }
        let bridged = profileTime()
        func bridge(_ slot: Slot) -> TensorBridge { bridges[slot.rawValue] }

        execution.lock.lock()
        defer { execution.lock.unlock() }
        let key = PlanKey(
            inputShape: shape, outputWeightShape: preparedWeights[0].shape,
            gateShape: preparedWeights[1].shape, upShape: preparedWeights[2].shape,
            downShape: preparedWeights[3].shape)
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
        trackErrors(commandBuffer.rootCommandBuffer, state: state)
        let planned = profileTime()

        return withExtendedLifetime((bridges, preparedWeights, descriptor)) {
            plan.outputProjection.encode(
                to: commandBuffer, inputData: bridge(.attended).tensorData,
                weightData: [preparedWeights[0].tensorData],
                outputData: [bridge(.projected).tensorData], descriptor: descriptor)

            let added = encodePointwise(
                commandBuffer: commandBuffer, pipeline: execution.add,
                buffers: [bridge(.residual).buffer, bridge(.projected).buffer,
                          bridge(.attentionResidual).buffer],
                count: residual.size, state: state)
            let normalized = added && encodeRMS(
                commandBuffer: commandBuffer, pipeline: execution.rms,
                input: bridge(.attentionResidual).buffer, weight: bridge(.normWeight).buffer,
                output: bridge(.normalized).buffer, epsilon: epsilon, hidden: hidden,
                rows: residual.size / hidden, groupSize: rmsGroupSize, state: state)
            if normalized {
                plan.gateUp.encode(
                    to: commandBuffer, inputData: bridge(.normalized).tensorData,
                    weightData: [preparedWeights[1].tensorData, preparedWeights[2].tensorData],
                    outputData: [bridge(.gate).tensorData, bridge(.up).tensorData],
                    descriptor: descriptor)
                if encodePointwise(
                    commandBuffer: commandBuffer, pipeline: execution.siluProduct,
                    buffers: [bridge(.gate).buffer, bridge(.up).buffer, bridge(.product).buffer],
                    count: arrays[Slot.product.rawValue].size, state: state)
                {
                    plan.down.encode(
                        to: commandBuffer, inputData: bridge(.product).tensorData,
                        weightData: [preparedWeights[3].tensorData],
                        outputData: [bridge(.down).tensorData], descriptor: descriptor)
                    _ = encodePointwise(
                        commandBuffer: commandBuffer, pipeline: execution.add,
                        buffers: [bridge(.attentionResidual).buffer, bridge(.down).buffer,
                                  bridge(.output).buffer],
                        count: residual.size, state: state)
                }
            }
            // All custom encoders fetch MPSCommandBuffer's current buffer:
            // any preceding graph encode may have called commitAndContinue.
            // Drain already submitted work even after an encoding failure.
            let encoded = profileTime()
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
            let completed = profileTime()
            if profileExecution {
                let timestamps = [started, allocated, weightsPrepared, evaluated,
                                  bridged, planned, encoded, completed]
                let names = ["allocation_ms", "weights_ms", "eval_ms", "bridge_ms",
                             "plan_ms", "encode_ms", "wait_ms"]
                var metrics: [String: Any] = ["shape": shape]
                for index in names.indices {
                    metrics[names[index]] = Double(timestamps[index + 1] - timestamps[index]) / 1e6
                }
                metrics["total_ms"] = Double(completed - started) / 1e6
                if let data = try? JSONSerialization.data(withJSONObject: metrics,
                                                         options: [.sortedKeys]) {
                    FileHandle.standardError.write(Data("minicpm_tts_block_profile ".utf8) + data + Data([10]))
                }
            }
            guard commandBuffer.rootCommandBuffer.status == .completed, !state.didFail else {
                return nil
            }
            return arrays[Slot.output.rawValue]
        }
    }

    private static func exactProjection(_ projection: Linear) -> MiniCPMExactBF16Linear? {
        guard let exact = projection as? MiniCPMExactBF16Linear,
              exact.bias == nil, exact.weight.dtype == .bfloat16,
              exact.weight.ndim == 2, exact.weight.shape.allSatisfy({ $0 > 0 })
        else { return nil }
        return exact
    }

    private static func trackErrors(
        _ commandBuffer: any MTLCommandBuffer, state: ExecutionState
    ) {
        commandBuffer.addCompletedHandler { completed in
            if completed.status == .error { state.recordFailure() }
        }
    }

    private static func encodePointwise(
        commandBuffer: MPSCommandBuffer, pipeline: any MTLComputePipelineState,
        buffers: [any MTLBuffer], count: Int, state: ExecutionState
    ) -> Bool {
        guard let encoder = commandBuffer.rootCommandBuffer.makeComputeCommandEncoder() else {
            state.recordFailure()
            return false
        }
        encoder.setComputePipelineState(pipeline)
        for (index, buffer) in buffers.enumerated() {
            encoder.setBuffer(buffer, offset: 0, index: index)
        }
        var count = UInt32(count)
        encoder.setBytes(&count, length: MemoryLayout<UInt32>.size, index: 3)
        encoder.dispatchThreads(
            MTLSize(width: Int(count), height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(
                width: min(256, pipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
        encoder.endEncoding()
        trackErrors(commandBuffer.rootCommandBuffer, state: state)
        return true
    }

    private static func encodeRMS(
        commandBuffer: MPSCommandBuffer, pipeline: any MTLComputePipelineState,
        input: any MTLBuffer, weight: any MTLBuffer, output: any MTLBuffer,
        epsilon: Float, hidden: Int, rows: Int, groupSize: Int, state: ExecutionState
    ) -> Bool {
        guard let encoder = commandBuffer.rootCommandBuffer.makeComputeCommandEncoder() else {
            state.recordFailure()
            return false
        }
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(input, offset: 0, index: 0)
        encoder.setBuffer(weight, offset: 0, index: 1)
        encoder.setBuffer(output, offset: 0, index: 2)
        var epsilon = epsilon
        var axisSize = UInt32(hidden)
        var weightStride = UInt32(1)
        encoder.setBytes(&epsilon, length: MemoryLayout<Float>.size, index: 3)
        encoder.setBytes(&axisSize, length: MemoryLayout<UInt32>.size, index: 4)
        encoder.setBytes(&weightStride, length: MemoryLayout<UInt32>.size, index: 5)
        encoder.dispatchThreads(
            MTLSize(width: rows * groupSize, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: groupSize, height: 1, depth: 1))
        encoder.endEncoding()
        trackErrors(commandBuffer.rootCommandBuffer, state: state)
        return true
    }

    private static let kernelSource = #"""
        #include <metal_stdlib>
        #include <metal_common>
        #include <metal_simdgroup>
        using namespace metal;

        kernel void minicpm_block_residual_add(
            device const bfloat* left [[buffer(0)]],
            device const bfloat* right [[buffer(1)]],
            device bfloat* out [[buffer(2)]],
            constant uint& count [[buffer(3)]],
            uint index [[thread_position_in_grid]]) {
            if (index < count) out[index] = left[index] + right[index];
        }

        kernel void minicpm_block_silu_product(
            device const bfloat* gate [[buffer(0)]],
            device const bfloat* up [[buffer(1)]],
            device bfloat* product [[buffer(2)]],
            constant uint& count [[buffer(3)]],
            uint index [[thread_position_in_grid]]) {
            if (index >= count) return;
            const float x = float(gate[index]);
            const float y = 1.0f / (1.0f + metal::exp(metal::abs(x)));
            const float sigmoid = x < 0.0f ? y : 1.0f - y;
            const bfloat activated = bfloat(x * sigmoid);
            product[index] = bfloat(float(activated) * float(up[index]));
        }

        // Adapted from pinned mlx-swift 0.31.6, revision
        // 0bb916c67f4b9e5c682cbe02a42c701c93ab5021:
        // Source/Cmlx/mlx/mlx/backend/metal/kernels/rms_norm.metal
        // rms_single_row<T=bfloat, N_READS=4>. Dispatch follows
        // Source/Cmlx/mlx/mlx/backend/metal/normalization.cpp.
        // Copyright © 2024 Apple Inc.
        //
        // MIT License
        // Copyright © 2023 Apple Inc.
        //
        // Permission is hereby granted, free of charge, to any person obtaining a copy
        // of this software and associated documentation files (the "Software"), to deal
        // in the Software without restriction, including without limitation the rights
        // to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
        // copies of the Software, and to permit persons to whom the Software is
        // furnished to do so, subject to the following conditions:
        //
        // The above copyright notice and this permission notice shall be included in all
        // copies or substantial portions of the Software.
        //
        // THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
        // IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
        // FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
        // AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
        // LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
        // OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
        // SOFTWARE.
        kernel void minicpm_block_rms_single_row(
            const device bfloat* x [[buffer(0)]],
            const device bfloat* w [[buffer(1)]],
            device bfloat* out [[buffer(2)]],
            constant float& eps [[buffer(3)]],
            constant uint& axis_size [[buffer(4)]],
            constant uint& w_stride [[buffer(5)]],
            uint gid [[threadgroup_position_in_grid]],
            uint lid [[thread_position_in_threadgroup]],
            uint simd_lane_id [[thread_index_in_simdgroup]],
            uint simd_group_id [[simdgroup_index_in_threadgroup]]) {
            constexpr int SIMD_SIZE = 32;
            constexpr int N_READS = 4;
            threadgroup float local_inv_mean[1];
            threadgroup float local_sums[SIMD_SIZE];
            float acc = 0;
            x += gid * size_t(axis_size) + lid * N_READS;
            w += w_stride * lid * N_READS;
            if (lid * N_READS + N_READS <= axis_size) {
                for (int i = 0; i < N_READS; i++) {
                    float xi = x[i];
                    acc += xi * xi;
                }
            } else {
                for (int i = 0; i < N_READS; i++) {
                    if ((lid * N_READS + i) < axis_size) {
                        float xi = x[i];
                        acc += xi * xi;
                    }
                }
            }
            acc = simd_sum(acc);
            if (simd_group_id == 0) local_sums[simd_lane_id] = 0;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (simd_lane_id == 0) local_sums[simd_group_id] = acc;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (simd_group_id == 0) {
                acc = simd_sum(local_sums[simd_lane_id]);
                if (simd_lane_id == 0) {
                    local_inv_mean[0] = metal::precise::rsqrt(acc / axis_size + eps);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            out += gid * size_t(axis_size) + lid * N_READS;
            if (lid * N_READS + N_READS <= axis_size) {
                for (int i = 0; i < N_READS; i++) {
                    out[i] = w[w_stride * i] * static_cast<bfloat>(x[i] * local_inv_mean[0]);
                }
            } else {
                for (int i = 0; i < N_READS; i++) {
                    if ((lid * N_READS + i) < axis_size) {
                        out[i] = w[w_stride * i] * static_cast<bfloat>(x[i] * local_inv_mean[0]);
                    }
                }
            }
        }
        """#
}
