import Foundation
import Metal
import MetalPerformanceShaders
import MetalPerformanceShadersGraph
import MLX
import MLXNN
import MiniCPMLLM

/// Optional single-token decoder execution with one entry MLX evaluation and
/// one final CPU wait. Matmuls retain the original, separate BF16 graph
/// boundaries. The inputs and all existing K/V backing arrays remain read-only.
enum MiniCPMTTSEncodedBF16Decode {
    struct Layer {
        let inputNorm: RMSNorm
        let postNorm: RMSNorm
        let query: Linear
        let key: Linear
        let value: Linear
        let output: Linear
        let gate: Linear
        let up: Linear
        let down: Linear
        let inverseFrequency: MLXArray
        let eps: Float
    }

    /// These are full [1, 12, capacity, 64] backing arrays, not logical slices.
    struct Cache {
        let keys: MLXArray?
        let values: MLXArray?
        let count: Int
        let capacity: Int
    }

    struct Result {
        let hidden: MLXArray
        let caches: [Cache]
    }

    private static let hiddenSize = 768
    private static let intermediateSize = 3072
    private static let heads = 12
    private static let headDimension = 64
    private static let rmsThreads = 192

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

    private final class LibraryBundleAnchor: NSObject {}

    private final class TwoPassAttentionPlan {
        let first: any MTLComputePipelineState
        let second: any MTLComputePipelineState
        let blocks: Int

        init?(device: any MTLDevice, library: any MTLLibrary, blocks: Int) {
            let constants = MTLFunctionConstantValues()
            for index in 20...25 {
                var value = false
                constants.setConstantValue(&value, type: .bool, index: index)
            }
            var blockCount = Int32(blocks)
            constants.setConstantValue(&blockCount, type: .int, index: 26)
            guard let firstFunction = try? library.makeFunction(
                name: "sdpa_vector_2pass_1_bfloat16_t_64_64", constantValues: constants),
                  let secondFunction = library.makeFunction(
                    name: "sdpa_vector_2pass_2_bfloat16_t_64"),
                  let first = try? device.makeComputePipelineState(function: firstFunction),
                  let second = try? device.makeComputePipelineState(function: secondFunction),
                  first.threadExecutionWidth == 32,
                  second.threadExecutionWidth == 32,
                  first.maxTotalThreadsPerThreadgroup >= 32,
                  second.maxTotalThreadsPerThreadgroup >= 1024 else { return nil }
            self.first = first
            self.second = second
            self.blocks = blocks
        }
    }

    private struct TwoPassAttentionScratch {
        let blocks: Int
        let intermediate: any MTLBuffer
        let sums: any MTLBuffer
        let maxs: any MTLBuffer

        init?(blocks: Int, device: any MTLDevice) {
            // The pinned two-pass ABI stores BF16 [1,12,1,blocks,64]
            // partials and FP32 [1,12,1,blocks] sums and maxima.
            let partialBytes = heads * blocks * headDimension * 2
            let reductionBytes = heads * blocks * MemoryLayout<Float>.size
            guard let intermediate = device.makeBuffer(
                length: partialBytes, options: .storageModeShared),
                  let sums = device.makeBuffer(
                    length: reductionBytes, options: .storageModeShared),
                  let maxs = device.makeBuffer(
                    length: reductionBytes, options: .storageModeShared) else { return nil }
            self.blocks = blocks
            self.intermediate = intermediate
            self.sums = sums
            self.maxs = maxs
        }
    }

    private final class ExecutionContext {
        let device: any MTLDevice
        let queue: any MTLCommandQueue
        let rms: any MTLComputePipelineState
        let sdpa: any MTLComputePipelineState
        let sdpaZeroMask: any MTLComputePipelineState
        let sdpaTwoPass: TwoPassAttentionPlan?
        let rope: any MTLComputePipelineState
        let copyAppend: any MTLComputePipelineState
        let add: any MTLComputePipelineState
        let siluProduct: any MTLComputePipelineState
        let qkv: ProjectionPlan
        let output: ProjectionPlan
        let gateUp: ProjectionPlan
        let down: ProjectionPlan
        let lock = NSLock()

        init?() {
            guard ProcessInfo.processInfo.environment["MLX_METAL_GPU_ARCH", default: ""].isEmpty,
                  let device = MTLCreateSystemDefaultDevice(),
                  device.supportsFamily(.apple7), let queue = device.makeCommandQueue(),
                  let mlxLibrary = MiniCPMTTSEncodedBF16Decode.loadMLXLibrary(device: device),
                  // Pinned mlx-swift 0.31.6 / revision
                  // 0bb916c67f4b9e5c682cbe02a42c701c93ab5021:
                  // Source/Cmlx/mlx/mlx/backend/metal/kernels/rms_norm.metal
                  // and kernels/sdpa_vector.h, instantiated by
                  // kernels/scaled_dot_product_attention.metal.
                  let rmsFunction = mlxLibrary.makeFunction(name: "rmsbfloat16"),
                  let rms = try? device.makeComputePipelineState(function: rmsFunction)
            else { return nil }
            func attention(masked: Bool) -> (any MTLComputePipelineState)? {
                let constants = MTLFunctionConstantValues()
                for index in 20...25 {
                    // Q=1 is physically contiguous, never causal, and has no
                    // sinks. The empty-cache reference supplies an all-zero
                    // additive mask; keep its score + 0 operation as well.
                    var value = (index == 20 || index == 24) && masked
                    constants.setConstantValue(&value, type: .bool, index: index)
                }
                guard let function = try? mlxLibrary.makeFunction(
                    name: "sdpa_vector_bfloat16_t_64_64", constantValues: constants)
                else { return nil }
                return try? device.makeComputePipelineState(function: function)
            }
            // Match pinned MLX device.cpp and SDPA dispatch. For Q=1/GQA=1
            // and logical N<=4096, d/s architectures always use these block
            // counts once N reaches 1024. Other architectures stay single-pass.
            let sdpaTwoPass: TwoPassAttentionPlan?
            let architecture = device.architecture.name.last
            if architecture == "d" || architecture == "s" {
                guard let plan = TwoPassAttentionPlan(
                    device: device, library: mlxLibrary,
                    blocks: architecture == "d" ? 128 : 64) else { return nil }
                sdpaTwoPass = plan
            } else {
                sdpaTwoPass = nil
            }
            let options = MTLCompileOptions()
            options.fastMathEnabled = false
            options.languageVersion = .version3_1
            guard let sdpa = attention(masked: false),
                  let sdpaZeroMask = attention(masked: true),
                  let custom = try? device.makeLibrary(
                    source: MiniCPMTTSEncodedBF16Decode.kernelSource, options: options),
                  let ropeFunction = custom.makeFunction(name: "minicpm_decode_rope"),
                  let copyFunction = custom.makeFunction(name: "minicpm_decode_copy_append"),
                  let addFunction = custom.makeFunction(name: "minicpm_decode_residual_add"),
                  let siluFunction = custom.makeFunction(name: "minicpm_decode_silu_product"),
                  let rope = try? device.makeComputePipelineState(function: ropeFunction),
                  let copy = try? device.makeComputePipelineState(function: copyFunction),
                  let add = try? device.makeComputePipelineState(function: addFunction),
                  let silu = try? device.makeComputePipelineState(function: siluFunction),
                  rms.threadExecutionWidth == 32,
                  rms.maxTotalThreadsPerThreadgroup >= rmsThreads,
                  sdpa.threadExecutionWidth == 32,
                  sdpaZeroMask.threadExecutionWidth == 32,
                  sdpa.maxTotalThreadsPerThreadgroup >= 1024,
                  sdpaZeroMask.maxTotalThreadsPerThreadgroup >= 1024,
                  let qkv = ProjectionPlan(
                    inputShape: [1, 1, hiddenSize],
                    weightShapes: Array(repeating: [hiddenSize, hiddenSize], count: 3),
                    grouped: true),
                  let output = ProjectionPlan(
                    inputShape: [1, 1, hiddenSize],
                    weightShapes: [[hiddenSize, hiddenSize]], grouped: false),
                  let gateUp = ProjectionPlan(
                    inputShape: [1, 1, hiddenSize],
                    weightShapes: Array(repeating: [intermediateSize, hiddenSize], count: 2),
                    grouped: true),
                  let down = ProjectionPlan(
                    inputShape: [1, 1, intermediateSize],
                    weightShapes: [[hiddenSize, intermediateSize]], grouped: false)
            else { return nil }
            self.device = device
            self.queue = queue
            self.rms = rms
            self.sdpa = sdpa
            self.sdpaZeroMask = sdpaZeroMask
            self.sdpaTwoPass = sdpaTwoPass
            self.rope = rope
            self.copyAppend = copy
            self.add = add
            self.siluProduct = silu
            self.qkv = qkv
            self.output = output
            self.gateUp = gateUp
            self.down = down
        }
    }

    private final class ExecutionState: @unchecked Sendable {
        private let lock = NSLock()
        private var failed = false
        // Retain each underlying buffer so ObjectIdentifier cannot be reused
        // after MPSGraph commits and replaces an earlier buffer.
        private var trackedBuffers: [ObjectIdentifier: any MTLCommandBuffer] = [:]

        func recordFailure() {
            lock.lock()
            failed = true
            lock.unlock()
        }

        var didFail: Bool {
            lock.lock()
            defer { lock.unlock() }
            return failed || trackedBuffers.values.contains { $0.status == .error }
        }

        func track(_ commandBuffer: any MTLCommandBuffer) {
            lock.lock()
            let identity = ObjectIdentifier(commandBuffer)
            let inserted = trackedBuffers[identity] == nil
            if inserted { trackedBuffers[identity] = commandBuffer }
            lock.unlock()
            guard inserted else { return }
            commandBuffer.addCompletedHandler { [self] completed in
                if completed.status == .error { recordFailure() }
            }
        }
    }

    private struct TensorBridge {
        // Only tensors crossing the MLX boundary need an MLX owner. Private
        // scratch retains its Metal buffer directly until the final wait.
        let owner: MLXArray?
        let buffer: any MTLBuffer
        let tensorData: MPSGraphTensorData

        init?(_ array: MLXArray, device: any MTLDevice) {
            let backing = array.asData(access: .noCopy)
            guard array.dtype == .bfloat16, backing.dType == .bfloat16,
                  backing.strides.count == array.ndim else { return nil }
            var expectedStride = 1
            for axis in array.shape.indices.reversed() {
                guard array.dim(axis) == 1 || backing.strides[axis] == expectedStride else {
                    return nil
                }
                expectedStride *= array.dim(axis)
            }
            guard let buffer = array.asMTLBuffer(device: device, noCopy: true),
                  buffer.length >= array.nbytes else { return nil }
            self.owner = array
            self.buffer = buffer
            self.tensorData = MPSGraphTensorData(
                buffer, shape: array.shape.map(NSNumber.init), dataType: .bFloat16)
        }

        init?(scratchShape: [Int], device: any MTLDevice) {
            guard !scratchShape.isEmpty, scratchShape.allSatisfy({ $0 > 0 }) else {
                return nil
            }
            let byteCount = scratchShape.reduce(2, *)
            guard let buffer = device.makeBuffer(length: byteCount, options: .storageModeShared)
            else { return nil }
            self.owner = nil
            self.buffer = buffer
            self.tensorData = MPSGraphTensorData(
                buffer, shape: scratchShape.map(NSNumber.init), dataType: .bFloat16)
        }
    }

    private enum Slot: Int {
        case inputNormWeight, postNormWeight, cosine, sine, normalizedInput
        case query, key, value, rotatedQuery, rotatedKey, attended
        case projected, attentionResidual, normalizedPost, gate, up, product, down, output
        case cacheKeys, cacheValues
    }

    private enum PreparedTensor {
        case array(MLXArray)
        case scratch(shape: [Int])
    }

    private struct PreparedLayer {
        let tensors: [PreparedTensor]
        // Q/K/V/O/gate/up/down, in that order.
        let weights: [MiniCPMExactBF16Linear.BF16WeightBridge]
        let oldCache: Cache
        let newCapacity: Int
        let eps: Float
    }

    private struct EncodedLayer {
        let prepared: PreparedLayer
        let bridges: [TensorBridge]
        let oldKeys: TensorBridge?
        let oldValues: TensorBridge?

        func bridge(_ slot: Slot) -> TensorBridge { bridges[slot.rawValue] }
    }

    private static let context = ExecutionContext()
    static var isAvailable: Bool { context != nil }
    private static let profileEnabled = ProcessInfo.processInfo.environment[
        "MINICPM_TTS_EXECUTION_PROFILE"] == "1"
    private static func profileTime() -> UInt64 {
        profileEnabled ? DispatchTime.now().uptimeNanoseconds : 0
    }

    static func apply(
        input: MLXArray, layers: [Layer], caches: [Cache], finalNorm: RMSNorm,
        forceCommandBufferRollover: Bool = false
    ) -> Result? {
        guard ProcessInfo.processInfo.environment["MLX_METAL_GPU_ARCH", default: ""].isEmpty,
              input.dtype == .bfloat16, input.shape == [1, 1, hiddenSize],
              !layers.isEmpty, layers.count == caches.count,
              validNorm(finalNorm), let count = caches.first?.count,
              count >= 0, count < 4096,
              caches.allSatisfy({ $0.count == count && validCache($0) }),
              layers.allSatisfy(validLayer), let execution = context
        else { return nil }
        execution.lock.lock()
        defer { execution.lock.unlock() }
        let started = profileTime()

        let inputArray = input.contiguous()
        let finalWeight = finalNorm.weight.contiguous()
        let finalOutput = zeroArray([1, 1, hiddenSize])
        let positions = MLXArray([Float(count)], [1, 1])
        var prepared: [PreparedLayer] = []
        var evaluatedArrays = [inputArray, finalWeight, finalOutput]
        for (layer, cache) in zip(layers, caches) {
            let projections = [layer.query, layer.key, layer.value, layer.output,
                               layer.gate, layer.up, layer.down]
            // validLayer has checked all casts before any GPU work is encoded.
            let weights = projections.map {
                ($0 as! MiniCPMExactBF16Linear).preparedBF16Weight(device: execution.device)
            }
            let frequencies = matmul(
                positions, layer.inverseFrequency.asType(.bfloat16).asType(.float32))
            let cosine = concatenated([cos(frequencies), cos(frequencies)], axis: -1)
                .asType(.bfloat16).expandedDimensions(axis: 0).expandedDimensions(axis: 0)
            let sine = concatenated([sin(frequencies), sin(frequencies)], axis: -1)
                .asType(.bfloat16).expandedDimensions(axis: 0).expandedDimensions(axis: 0)
            var capacity = max(32, cache.capacity)
            while capacity < count + 1 { capacity *= 2 }
            let shape = [1, 1, hiddenSize]
            let wide = [1, 1, intermediateSize]
            // Slot order is shared by the graph and custom encoders. Every
            // scratch tensor is fully written before its first read, so it
            // needs no MLX allocation, evaluation, or zero-fill operation.
            let tensors: [PreparedTensor] = [
                .array(layer.inputNorm.weight.contiguous()),
                .array(layer.postNorm.weight.contiguous()),
                .array(cosine), .array(sine), .scratch(shape: shape),
                .scratch(shape: shape), .scratch(shape: shape), .scratch(shape: shape),
                .scratch(shape: [1, heads, 1, headDimension]),
                .scratch(shape: [1, heads, 1, headDimension]), .scratch(shape: shape),
                .scratch(shape: shape), .scratch(shape: shape), .scratch(shape: shape),
                .scratch(shape: wide), .scratch(shape: wide), .scratch(shape: wide),
                .scratch(shape: shape), .scratch(shape: shape),
                .array(cacheArray(capacity: capacity)), .array(cacheArray(capacity: capacity)),
            ]
            prepared.append(PreparedLayer(
                tensors: tensors, weights: weights, oldCache: cache,
                newCapacity: capacity, eps: layer.eps))
            evaluatedArrays.append(contentsOf: tensors.compactMap { tensor -> MLXArray? in
                if case .array(let array) = tensor { return array }
                return nil
            })
            if let keys = cache.keys, let values = cache.values {
                evaluatedArrays.append(contentsOf: [keys, values])
            }
        }
        // Complete every MLX dependency and allocation before entering the
        // private queue. No MLX operation is evaluated between decoder layers.
        let preparedAt = profileTime()
        eval(evaluatedArrays)
        let evaluatedAt = profileTime()
        guard let inputBridge = TensorBridge(inputArray, device: execution.device),
              let finalWeightBridge = TensorBridge(finalWeight, device: execution.device),
              let finalOutputBridge = TensorBridge(finalOutput, device: execution.device),
              let zeroMaskBridge = TensorBridge(scratchShape: [1], device: execution.device)
        else { return nil }
        // The empty-cache attention kernel retains the original BF16 + 0
        // score operation. Its sole scratch input is initialized on the CPU.
        zeroMaskBridge.buffer.contents().storeBytes(of: UInt16(0), as: UInt16.self)
        var encodedLayers: [EncodedLayer] = []
        // Layers execute serially. Each scratch slot's last read precedes
        // the next layer's write, so reuse one buffer per slot within this
        // invocation. In particular the previous layer's output is consumed
        // by RMS and the first residual add before the final add replaces it.
        // Returned caches and final hidden storage never use these slots.
        var scratchBridges: [Int: TensorBridge] = [:]
        for layer in prepared {
            var bridges: [TensorBridge] = []
            for (slot, tensor) in layer.tensors.enumerated() {
                let bridge: TensorBridge?
                switch tensor {
                case .array(let array):
                    bridge = TensorBridge(array, device: execution.device)
                case .scratch(let shape):
                    if let existing = scratchBridges[slot] {
                        guard existing.tensorData.shape.map(\.intValue) == shape else { return nil }
                        bridge = existing
                    } else {
                        bridge = TensorBridge(scratchShape: shape, device: execution.device)
                        scratchBridges[slot] = bridge
                    }
                }
                guard let bridge else { return nil }
                bridges.append(bridge)
            }
            var oldKeys: TensorBridge?
            var oldValues: TensorBridge?
            if let keys = layer.oldCache.keys, let values = layer.oldCache.values {
                guard let keyBridge = TensorBridge(keys, device: execution.device),
                      let valueBridge = TensorBridge(values, device: execution.device)
                else { return nil }
                oldKeys = keyBridge
                oldValues = valueBridge
            }
            encodedLayers.append(EncodedLayer(
                prepared: layer, bridges: bridges, oldKeys: oldKeys, oldValues: oldValues))
        }
        // Validate all externally returned owners before any private work is
        // encoded. A malformed slot must fall back without partial execution.
        var resultCaches: [Cache] = []
        for layer in encodedLayers {
            guard let keys = layer.bridge(.cacheKeys).owner,
                  let values = layer.bridge(.cacheValues).owner else { return nil }
            resultCaches.append(Cache(
                keys: keys, values: values, count: count + 1,
                capacity: layer.prepared.newCapacity))
        }
        // The two attention passes finish before the next layer reuses these
        // private buffers. Allocate them once per invocation, never per layer.
        let attentionScratch: TwoPassAttentionScratch?
        if count + 1 >= 1024, let plan = execution.sdpaTwoPass {
            guard let scratch = TwoPassAttentionScratch(
                blocks: plan.blocks, device: execution.device) else { return nil }
            attentionScratch = scratch
        } else {
            attentionScratch = nil
        }
        guard let initial = execution.queue.makeCommandBuffer() else { return nil }
        let bridgedAt = profileTime()
        let commandBuffer = MPSCommandBuffer(commandBuffer: initial)
        let state = ExecutionState()
        state.track(initial)
        let descriptor = MPSGraphExecutionDescriptor()
        descriptor.waitUntilCompleted = false
        descriptor.completionHandler = { _, error in
            if error != nil { state.recordFailure() }
        }
        return withExtendedLifetime(
            (encodedLayers, inputBridge, finalWeightBridge, finalOutputBridge,
             zeroMaskBridge, attentionScratch, resultCaches, descriptor)
        ) {
            var current = inputBridge
            var succeeded = true
            for (index, layer) in encodedLayers.enumerated() {
                if forceCommandBufferRollover && index > 0 {
                    state.track(commandBuffer.rootCommandBuffer)
                    commandBuffer.commitAndContinue()
                }
                guard !state.didFail, encodeLayer(
                    layer, input: current, zeroMask: zeroMaskBridge,
                    attentionScratch: attentionScratch,
                    execution: execution, commandBuffer: commandBuffer,
                    descriptor: descriptor, state: state) else {
                    succeeded = false
                    break
                }
                current = layer.bridge(.output)
            }
            if succeeded {
                succeeded = encodeRMS(
                    input: current, weight: finalWeightBridge, output: finalOutputBridge,
                    eps: finalNorm.eps, execution: execution,
                    commandBuffer: commandBuffer, state: state)
            }
            // MPSGraph may have internally committed earlier command buffers.
            // Drain the entire wrapper before releasing any backing owner,
            // including when a later custom encoder could not be created.
            let finalCommandBuffer = commandBuffer.rootCommandBuffer
            state.track(finalCommandBuffer)
            let encodedAt = profileTime()
            commandBuffer.commit()
            finalCommandBuffer.waitUntilCompleted()
            if profileEnabled {
                let finished = profileTime()
                let record: [String: Any] = [
                    "kind": "tts_encoded_decode", "count": count,
                    "attention_passes": attentionScratch == nil ? 1 : 2,
                    "sdpa_blocks": attentionScratch?.blocks ?? 0,
                    "prepare_ms": Double(preparedAt - started) / 1e6,
                    "eval_ms": Double(evaluatedAt - preparedAt) / 1e6,
                    "bridge_ms": Double(bridgedAt - evaluatedAt) / 1e6,
                    "encode_ms": Double(encodedAt - bridgedAt) / 1e6,
                    "wait_ms": Double(finished - encodedAt) / 1e6,
                    "total_ms": Double(finished - started) / 1e6,
                ]
                if let data = try? JSONSerialization.data(withJSONObject: record),
                   let line = String(data: data, encoding: .utf8) {
                    fputs(line + "\n", stderr)
                }
            }
            guard succeeded, finalCommandBuffer.status == .completed,
                  !state.didFail else { return nil }
            return Result(hidden: finalOutput, caches: resultCaches)
        }
    }

    private static func encodeLayer(
        _ layer: EncodedLayer, input: TensorBridge, zeroMask: TensorBridge,
        attentionScratch: TwoPassAttentionScratch?,
        execution: ExecutionContext, commandBuffer: MPSCommandBuffer,
        descriptor: MPSGraphExecutionDescriptor, state: ExecutionState
    ) -> Bool {
        func bridge(_ slot: Slot) -> TensorBridge { layer.bridge(slot) }
        let weights = layer.prepared.weights
        guard encodeRMS(
            input: input, weight: bridge(.inputNormWeight), output: bridge(.normalizedInput),
            eps: layer.prepared.eps, execution: execution,
            commandBuffer: commandBuffer, state: state) else { return false }
        execution.qkv.encode(
            to: commandBuffer, inputData: bridge(.normalizedInput).tensorData,
            weightData: Array(weights[0...2]).map(\.tensorData),
            outputData: [bridge(.query).tensorData, bridge(.key).tensorData,
                         bridge(.value).tensorData], descriptor: descriptor)
        guard encodeCustom(
            execution.rope, commandBuffer: commandBuffer, state: state,
            threads: hiddenSize / 2, buffers: [
                bridge(.query).buffer, bridge(.key).buffer, bridge(.cosine).buffer,
                bridge(.sine).buffer, bridge(.rotatedQuery).buffer, bridge(.rotatedKey).buffer,
            ]),
              encodeCopyAppend(
                layer, execution: execution, commandBuffer: commandBuffer, state: state),
              encodeAttention(
                layer, zeroMask: zeroMask, execution: execution,
                attentionScratch: attentionScratch,
                commandBuffer: commandBuffer, state: state)
        else { return false }
        execution.output.encode(
            to: commandBuffer, inputData: bridge(.attended).tensorData,
            weightData: [weights[3].tensorData], outputData: [bridge(.projected).tensorData],
            descriptor: descriptor)
        guard encodeCustom(
            execution.add, commandBuffer: commandBuffer, state: state, threads: hiddenSize,
            buffers: [input.buffer, bridge(.projected).buffer, bridge(.attentionResidual).buffer]),
              encodeRMS(
                input: bridge(.attentionResidual), weight: bridge(.postNormWeight),
                output: bridge(.normalizedPost), eps: layer.prepared.eps,
                execution: execution, commandBuffer: commandBuffer, state: state)
        else { return false }
        execution.gateUp.encode(
            to: commandBuffer, inputData: bridge(.normalizedPost).tensorData,
            weightData: [weights[4].tensorData, weights[5].tensorData],
            outputData: [bridge(.gate).tensorData, bridge(.up).tensorData],
            descriptor: descriptor)
        guard encodeCustom(
            execution.siluProduct, commandBuffer: commandBuffer, state: state,
            threads: intermediateSize,
            buffers: [bridge(.gate).buffer, bridge(.up).buffer, bridge(.product).buffer])
        else { return false }
        execution.down.encode(
            to: commandBuffer, inputData: bridge(.product).tensorData,
            weightData: [weights[6].tensorData], outputData: [bridge(.down).tensorData],
            descriptor: descriptor)
        return encodeCustom(
            execution.add, commandBuffer: commandBuffer, state: state, threads: hiddenSize,
            buffers: [bridge(.attentionResidual).buffer, bridge(.down).buffer, bridge(.output).buffer])
    }

    private static func makeEncoder(
        commandBuffer: MPSCommandBuffer, state: ExecutionState
    ) -> (any MTLComputeCommandEncoder)? {
        // Never retain the underlying buffer across an MPSGraph encode.
        let current = commandBuffer.rootCommandBuffer
        state.track(current)
        guard let encoder = current.makeComputeCommandEncoder() else {
            state.recordFailure()
            return nil
        }
        return encoder
    }

    private static func encodeCustom(
        _ pipeline: any MTLComputePipelineState, commandBuffer: MPSCommandBuffer,
        state: ExecutionState, threads: Int, buffers: [any MTLBuffer]
    ) -> Bool {
        guard let encoder = makeEncoder(commandBuffer: commandBuffer, state: state) else {
            return false
        }
        encoder.setComputePipelineState(pipeline)
        for (index, buffer) in buffers.enumerated() {
            encoder.setBuffer(buffer, offset: 0, index: index)
        }
        encoder.dispatchThreads(
            MTLSize(width: threads, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(
                width: min(256, pipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
        encoder.endEncoding()
        return true
    }

    private static func encodeRMS(
        input: TensorBridge, weight: TensorBridge, output: TensorBridge, eps: Float,
        execution: ExecutionContext, commandBuffer: MPSCommandBuffer, state: ExecutionState
    ) -> Bool {
        guard let encoder = makeEncoder(commandBuffer: commandBuffer, state: state) else {
            return false
        }
        encoder.setComputePipelineState(execution.rms)
        encoder.setBuffer(input.buffer, offset: 0, index: 0)
        encoder.setBuffer(weight.buffer, offset: 0, index: 1)
        encoder.setBuffer(output.buffer, offset: 0, index: 2)
        var eps = eps
        var axis = UInt32(hiddenSize)
        var stride = UInt32(1)
        encoder.setBytes(&eps, length: MemoryLayout<Float>.size, index: 3)
        encoder.setBytes(&axis, length: MemoryLayout<UInt32>.size, index: 4)
        encoder.setBytes(&stride, length: MemoryLayout<UInt32>.size, index: 5)
        encoder.dispatchThreads(
            MTLSize(width: rmsThreads, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: rmsThreads, height: 1, depth: 1))
        encoder.endEncoding()
        return true
    }

    private static func encodeCopyAppend(
        _ layer: EncodedLayer, execution: ExecutionContext,
        commandBuffer: MPSCommandBuffer, state: ExecutionState
    ) -> Bool {
        guard let encoder = makeEncoder(commandBuffer: commandBuffer, state: state) else {
            return false
        }
        encoder.setComputePipelineState(execution.copyAppend)
        let key = layer.bridge(.rotatedKey).buffer
        let value = layer.bridge(.value).buffer
        let buffers = [
            layer.oldKeys?.buffer ?? key, layer.oldValues?.buffer ?? value, key, value,
            layer.bridge(.cacheKeys).buffer, layer.bridge(.cacheValues).buffer,
        ]
        for (index, buffer) in buffers.enumerated() {
            encoder.setBuffer(buffer, offset: 0, index: index)
        }
        var count = UInt32(layer.prepared.oldCache.count)
        var oldCapacity = UInt32(layer.prepared.oldCache.capacity)
        var newCapacity = UInt32(layer.prepared.newCapacity)
        encoder.setBytes(&count, length: MemoryLayout<UInt32>.size, index: 6)
        encoder.setBytes(&oldCapacity, length: MemoryLayout<UInt32>.size, index: 7)
        encoder.setBytes(&newCapacity, length: MemoryLayout<UInt32>.size, index: 8)
        encoder.dispatchThreads(
            MTLSize(width: heads * (Int(count) + 1) * headDimension, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(
                width: min(256, execution.copyAppend.maxTotalThreadsPerThreadgroup),
                height: 1, depth: 1))
        encoder.endEncoding()
        return true
    }

    private static func encodeAttention(
        _ layer: EncodedLayer, zeroMask: TensorBridge, execution: ExecutionContext,
        attentionScratch: TwoPassAttentionScratch?,
        commandBuffer: MPSCommandBuffer, state: ExecutionState
    ) -> Bool {
        if layer.prepared.oldCache.count + 1 >= 1024, let plan = execution.sdpaTwoPass {
            guard let scratch = attentionScratch, scratch.blocks == plan.blocks else {
                return false
            }
            return encodeTwoPassAttention(
                layer, plan: plan, scratch: scratch,
                commandBuffer: commandBuffer, state: state)
        }
        guard let encoder = makeEncoder(commandBuffer: commandBuffer, state: state) else {
            return false
        }
        let empty = layer.prepared.oldCache.count == 0
        encoder.setComputePipelineState(empty ? execution.sdpaZeroMask : execution.sdpa)
        encoder.setBuffer(layer.bridge(.rotatedQuery).buffer, offset: 0, index: 0)
        encoder.setBuffer(layer.bridge(.cacheKeys).buffer, offset: 0, index: 1)
        encoder.setBuffer(layer.bridge(.cacheValues).buffer, offset: 0, index: 2)
        encoder.setBuffer(layer.bridge(.attended).buffer, offset: 0, index: 3)
        var gqa = Int32(1)
        var count = Int32(layer.prepared.oldCache.count + 1)
        var headStride = UInt64(layer.prepared.newCapacity * headDimension)
        var sequenceStride = UInt64(headDimension)
        var scale = Float(0.125)
        encoder.setBytes(&gqa, length: MemoryLayout<Int32>.size, index: 4)
        encoder.setBytes(&count, length: MemoryLayout<Int32>.size, index: 5)
        encoder.setBytes(&headStride, length: MemoryLayout<UInt64>.size, index: 6)
        encoder.setBytes(&sequenceStride, length: MemoryLayout<UInt64>.size, index: 7)
        encoder.setBytes(&headStride, length: MemoryLayout<UInt64>.size, index: 8)
        encoder.setBytes(&sequenceStride, length: MemoryLayout<UInt64>.size, index: 9)
        encoder.setBytes(&scale, length: MemoryLayout<Float>.size, index: 10)
        if empty {
            encoder.setBuffer(zeroMask.buffer, offset: 0, index: 12)
            var stride = Int32(0)
            for index in 13...15 {
                encoder.setBytes(&stride, length: MemoryLayout<Int32>.size, index: index)
            }
        }
        encoder.dispatchThreadgroups(
            MTLSize(width: heads, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 1024, height: 1, depth: 1))
        encoder.endEncoding()
        return true
    }

    private static func encodeTwoPassAttention(
        _ layer: EncodedLayer, plan: TwoPassAttentionPlan, scratch: TwoPassAttentionScratch,
        commandBuffer: MPSCommandBuffer, state: ExecutionState
    ) -> Bool {
        guard let first = makeEncoder(commandBuffer: commandBuffer, state: state) else {
            return false
        }
        first.setComputePipelineState(plan.first)
        let buffers = [
            layer.bridge(.rotatedQuery).buffer, layer.bridge(.cacheKeys).buffer,
            layer.bridge(.cacheValues).buffer, scratch.intermediate, scratch.sums, scratch.maxs,
        ]
        for (index, buffer) in buffers.enumerated() {
            first.setBuffer(buffer, offset: 0, index: index)
        }
        var count = Int32(layer.prepared.oldCache.count + 1)
        var headStride = UInt64(layer.prepared.newCapacity * headDimension)
        var sequenceStride = UInt64(headDimension)
        var scale = Float(0.125)
        first.setBytes(&count, length: MemoryLayout<Int32>.size, index: 7)
        first.setBytes(&headStride, length: MemoryLayout<UInt64>.size, index: 8)
        first.setBytes(&sequenceStride, length: MemoryLayout<UInt64>.size, index: 9)
        first.setBytes(&headStride, length: MemoryLayout<UInt64>.size, index: 10)
        first.setBytes(&sequenceStride, length: MemoryLayout<UInt64>.size, index: 11)
        first.setBytes(&scale, length: MemoryLayout<Float>.size, index: 12)
        first.dispatchThreadgroups(
            MTLSize(width: heads, height: 1, depth: plan.blocks),
            threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        first.endEncoding()

        // A separate serial encoder orders the partial writes before the
        // second pass, matching MLX's buffer barrier between dispatches.
        guard let second = makeEncoder(commandBuffer: commandBuffer, state: state) else {
            return false
        }
        second.setComputePipelineState(plan.second)
        second.setBuffer(scratch.intermediate, offset: 0, index: 0)
        second.setBuffer(scratch.sums, offset: 0, index: 1)
        second.setBuffer(scratch.maxs, offset: 0, index: 2)
        second.setBuffer(layer.bridge(.attended).buffer, offset: 0, index: 3)
        var blocks = Int32(plan.blocks)
        second.setBytes(&blocks, length: MemoryLayout<Int32>.size, index: 4)
        second.dispatchThreadgroups(
            MTLSize(width: heads, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 1024, height: 1, depth: 1))
        second.endEncoding()
        return true
    }

    private static func validNorm(_ norm: RMSNorm) -> Bool {
        type(of: norm) == RMSNorm.self && norm.weight.dtype == .bfloat16
            && norm.weight.shape == [hiddenSize] && norm.eps.isFinite && norm.eps > 0
    }

    private static func validLayer(_ layer: Layer) -> Bool {
        guard validNorm(layer.inputNorm), validNorm(layer.postNorm),
              layer.eps == layer.inputNorm.eps, layer.eps == layer.postNorm.eps,
              layer.inverseFrequency.dtype == .float32,
              layer.inverseFrequency.shape == [1, headDimension / 2]
        else { return false }
        let projections = [layer.query, layer.key, layer.value, layer.output,
                           layer.gate, layer.up, layer.down]
        let shapes = Array(repeating: [hiddenSize, hiddenSize], count: 4)
            + [[intermediateSize, hiddenSize], [intermediateSize, hiddenSize],
               [hiddenSize, intermediateSize]]
        return zip(projections, shapes).allSatisfy { projection, shape in
            projection is MiniCPMExactBF16Linear && projection.bias == nil
                && projection.weight.dtype == .bfloat16 && projection.weight.shape == shape
        }
    }

    private static func validCache(_ cache: Cache) -> Bool {
        if cache.keys == nil || cache.values == nil {
            return cache.keys == nil && cache.values == nil && cache.count == 0 && cache.capacity == 0
        }
        guard cache.capacity >= 32, cache.capacity <= 4096,
              (cache.capacity & (cache.capacity - 1)) == 0,
              cache.count >= 0, cache.count <= cache.capacity,
              let keys = cache.keys, let values = cache.values
        else { return false }
        let shape = [1, heads, cache.capacity, headDimension]
        return keys.dtype == .bfloat16 && values.dtype == .bfloat16
            && keys.shape == shape && values.shape == shape
    }

    private static func zeroArray(_ shape: [Int]) -> MLXArray {
        // Eager MLX storage avoids a zero-fill primitive/command submission
        // for every graph destination. These bytes are owned by this result.
        MLXArray(Data(count: shape.reduce(1, *) * 2), shape, dtype: .bfloat16)
    }

    private static func cacheArray(capacity: Int) -> MLXArray {
        let shape = [1, heads, capacity, headDimension]
        // Large backing caches avoid CPU Data zero-fill; these lazy zeros
        // join the existing single entry eval and preserve zero padding.
        // Small destinations retain eager storage to avoid extra dispatches.
        return capacity >= 1024 ? MLXArray.zeros(shape, dtype: .bfloat16) : zeroArray(shape)
    }

    private static func loadMLXLibrary(device: any MTLDevice) -> (any MTLLibrary)? {
        // Match MLX's colocated mlx.metallib and SwiftPM-resource fallbacks.
        // The anchor also finds the xctest binary containing statically
        // linked MLX, where Bundle.main may instead name Apple's test runner.
        let anchor = Bundle(for: LibraryBundleAnchor.self)
        let bundles = [anchor, Bundle.main] + Bundle.allBundles + Bundle.allFrameworks
        var candidates: [URL] = []
        let roots = [anchor.executableURL, Bundle.main.executableURL]
            .compactMap { $0?.resolvingSymlinksInPath().deletingLastPathComponent() }
        for root in roots {
            candidates.append(root.appendingPathComponent("mlx.metallib"))
            candidates.append(root.appendingPathComponent("Resources/mlx.metallib"))
        }
        for bundle in bundles {
            for root in [bundle.bundleURL, bundle.resourceURL].compactMap({ $0 }) {
                let resourceBundleURL = root.appendingPathComponent("mlx-swift_Cmlx.bundle")
                if let resources = Bundle(url: resourceBundleURL)?.resourceURL {
                    candidates.append(resources.appendingPathComponent("default.metallib"))
                }
                // SwiftPM may also emit a flat resource directory without
                // a macOS Contents/Resources bundle layout.
                candidates.append(resourceBundleURL.appendingPathComponent("default.metallib"))
            }
            if bundle.bundleURL.lastPathComponent == "mlx-swift_Cmlx.bundle"
                || bundle.bundleIdentifier == "mlx-swift_Cmlx" {
                if let root = bundle.resourceURL {
                    candidates.append(root.appendingPathComponent("default.metallib"))
                }
            }
        }
        var seen: Set<String> = []
        for url in candidates where seen.insert(url.path).inserted {
            guard FileManager.default.fileExists(atPath: url.path),
                  let library = try? device.makeLibrary(URL: url),
                  library.functionNames.contains("sdpa_vector_bfloat16_t_64_64"),
                  library.functionNames.contains("sdpa_vector_2pass_1_bfloat16_t_64_64"),
                  library.functionNames.contains("sdpa_vector_2pass_2_bfloat16_t_64"),
                  library.functionNames.contains("rmsbfloat16")
            else { continue }
            return library
        }
        return nil
    }

    private static let kernelSource = #"""
        #include <metal_stdlib>
        using namespace metal;

        kernel void minicpm_decode_rope(
            const device bfloat* query [[buffer(0)]],
            const device bfloat* key [[buffer(1)]],
            const device bfloat* cosine [[buffer(2)]],
            const device bfloat* sine [[buffer(3)]],
            device bfloat* query_out [[buffer(4)]],
            device bfloat* key_out [[buffer(5)]],
            uint index [[thread_position_in_grid]]) {
            if (index >= 384) return;
            const uint head = index / 32;
            const uint first = head * 64 + index % 32;
            const uint second = first + 32;
            const bfloat c = cosine[index % 32];
            const bfloat s = sine[index % 32];
            const bfloat q1c = bfloat(float(query[first]) * float(c));
            const bfloat q2c = bfloat(float(query[second]) * float(c));
            const bfloat q2s = bfloat(float(bfloat(-float(query[second]))) * float(s));
            const bfloat q1s = bfloat(float(query[first]) * float(s));
            const bfloat k1c = bfloat(float(key[first]) * float(c));
            const bfloat k2c = bfloat(float(key[second]) * float(c));
            const bfloat k2s = bfloat(float(bfloat(-float(key[second]))) * float(s));
            const bfloat k1s = bfloat(float(key[first]) * float(s));
            query_out[first] = bfloat(float(q1c) + float(q2s));
            query_out[second] = bfloat(float(q2c) + float(q1s));
            key_out[first] = bfloat(float(k1c) + float(k2s));
            key_out[second] = bfloat(float(k2c) + float(k1s));
        }

        kernel void minicpm_decode_copy_append(
            const device bfloat* old_keys [[buffer(0)]],
            const device bfloat* old_values [[buffer(1)]],
            const device bfloat* key [[buffer(2)]],
            const device bfloat* value [[buffer(3)]],
            device bfloat* keys_out [[buffer(4)]],
            device bfloat* values_out [[buffer(5)]],
            constant uint& count [[buffer(6)]],
            constant uint& old_capacity [[buffer(7)]],
            constant uint& new_capacity [[buffer(8)]],
            uint index [[thread_position_in_grid]]) {
            const uint length = count + 1;
            if (index >= 12 * length * 64) return;
            const uint head = index / (length * 64);
            const uint token = (index / 64) % length;
            const uint dim = index % 64;
            const uint dst = (head * new_capacity + token) * 64 + dim;
            if (token < count) {
                const uint src = (head * old_capacity + token) * 64 + dim;
                keys_out[dst] = old_keys[src];
                values_out[dst] = old_values[src];
            } else {
                keys_out[dst] = key[head * 64 + dim];
                values_out[dst] = value[head * 64 + dim];
            }
        }

        kernel void minicpm_decode_residual_add(
            const device bfloat* left [[buffer(0)]],
            const device bfloat* right [[buffer(1)]],
            device bfloat* out [[buffer(2)]],
            uint index [[thread_position_in_grid]]) {
            if (index < 768) out[index] = left[index] + right[index];
        }

        kernel void minicpm_decode_silu_product(
            const device bfloat* gate [[buffer(0)]],
            const device bfloat* up [[buffer(1)]],
            device bfloat* product [[buffer(2)]],
            uint index [[thread_position_in_grid]]) {
            if (index >= 3072) return;
            const float x = float(gate[index]);
            const float y = 1.0f / (1.0f + metal::exp(metal::abs(x)));
            const float sigmoid = x < 0.0f ? y : 1.0f - y;
            const bfloat activated = bfloat(x * sigmoid);
            product[index] = bfloat(float(activated) * float(up[index]));
        }
        """#
}
