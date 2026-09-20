import Foundation
import Metal
import MetalPerformanceShadersGraph
import MLX

/// Exact dense-BF16 SDPA used by MiniCPM prefill and cached decode.
///
/// The graph mirrors PyTorch 2.8's ``sdpa_general_mps`` operation order:
/// BF16 QK^T, FP32 scale multiply, cast back to BF16, then either a square
/// causal select with ``-1e20`` or a full rectangular BF16 additive mask,
/// followed by BF16 softmax and BF16 value matmul.  The rectangular mask is
/// right-aligned so the final query row can see the full cached prefix.
/// Its internal entry point accepts tensors already arranged as
/// ``[B,H,T,D]`` and returns ``[B,T,H,D]``.  K/V heads are explicitly
/// repeated in the graph for grouped-query attention rather than relying on
/// MPSGraph's head-axis broadcasting.
/// Shared dense-BF16 scaled-dot-product attention implementation used by
/// MiniCPM language and semantic-TTS decoders.
public final class MiniCPMExactBF16SDPA {
    private struct PlanKey: Hashable {
        let queryShape: [Int]
        let keyValueShape: [Int]
        let scaleBits: UInt32
        /// A rectangular cached call receives a full additive mask through a
        /// graph placeholder.  Keep this bit in the plan key so a square
        /// causal-select graph is never reused for a masked rectangular call
        /// (or vice versa).
        let hasAdditiveMask: Bool
    }

    /// All exact SDPA plans submit through one queue.  MPSGraph's no-copy
    /// TensorData bridge aliases MLX-owned buffers, so a shared execution lock
    /// keeps submissions from separate attention contexts ordered on the same
    /// Metal device and avoids queue accumulation across sequence lengths.
    private final class SharedExecutionContext {
        static let shared = SharedExecutionContext()

        let device: any MTLDevice
        let commandQueue: any MTLCommandQueue
        let executionLock = NSLock()

        private init() {
            guard let device = MTLCreateSystemDefaultDevice() else {
                preconditionFailure("MiniCPM exact SDPA requires a Metal GPU device")
            }
            guard let commandQueue = device.makeCommandQueue() else {
                preconditionFailure("MiniCPM exact SDPA could not create a Metal command queue")
            }
            self.device = device
            self.commandQueue = commandQueue
        }
    }

    private final class Plan {
        let graph: MPSGraph
        let queryTensor: MPSGraphTensor
        let keyTensor: MPSGraphTensor
        let valueTensor: MPSGraphTensor
        let additiveMaskTensor: MPSGraphTensor?
        let outputTensor: MPSGraphTensor

        init(
            graph: MPSGraph,
            queryTensor: MPSGraphTensor,
            keyTensor: MPSGraphTensor,
            valueTensor: MPSGraphTensor,
            additiveMaskTensor: MPSGraphTensor?,
            outputTensor: MPSGraphTensor
        ) {
            self.graph = graph
            self.queryTensor = queryTensor
            self.keyTensor = keyTensor
            self.valueTensor = valueTensor
            self.additiveMaskTensor = additiveMaskTensor
            self.outputTensor = outputTensor
        }
    }

    private static let sharedExecutionContext = SharedExecutionContext.shared
    private static let planLock = NSLock()
    private static var plans: [PlanKey: Plan] = [:]

    /// ``torch.finfo(torch.bfloat16).min``.  Keep the value in Float32 while
    /// constructing the host-side mask, then round it once to BF16.  Using
    /// ``-Float.greatestFiniteMagnitude`` can round to BF16 ``-inf`` on some
    /// conversion paths, which is not the value used by PyTorch's additive
    /// mask.
    public static let bfloat16FinfoMin: Float = -3.3895313892515355e38

    /// Build the right-aligned causal mask used when a multi-token segment
    /// attends a longer cached prefix.  The result is a full
    /// ``[B,Hq,Q,K]`` BF16 tensor: zero means allowed and
    /// ``torch.bfloat16.finfo.min`` means blocked.
    public static func rightAlignedAdditiveMask(
        batch: Int,
        queryHeads: Int,
        queryTokens: Int,
        keyTokens: Int
    ) -> MLXArray {
        precondition(batch > 0, "MiniCPM exact SDPA mask requires a non-empty batch")
        precondition(queryHeads > 0, "MiniCPM exact SDPA mask requires query heads")
        precondition(queryTokens > 0, "MiniCPM exact SDPA mask requires query tokens")
        precondition(
            keyTokens >= queryTokens,
            "MiniCPM exact SDPA mask requires key length >= query length")

        let prefix = keyTokens - queryTokens
        let rowSize = queryHeads * queryTokens * keyTokens
        var values = Array(
            repeating: bfloat16FinfoMin,
            count: batch * rowSize)
        for batchIndex in 0 ..< batch {
            for head in 0 ..< queryHeads {
                for queryIndex in 0 ..< queryTokens {
                    // The query segment is right-aligned to the cached key
                    // axis.  Row zero sees prefix+1 keys; the final row sees
                    // the complete key/value sequence.
                    let allowedThrough = prefix + queryIndex
                    let rowBase = ((batchIndex * queryHeads + head)
                        * queryTokens + queryIndex) * keyTokens
                    for keyIndex in 0 ... allowedThrough {
                        values[rowBase + keyIndex] = 0
                    }
                }
            }
        }
        return MLXArray(
            values,
            [batch, queryHeads, queryTokens, keyTokens]).asType(.bfloat16)
    }

    private init() {}

    /// Run exact BF16 attention.
    ///
    /// - Parameters:
    ///   - query: BF16 query heads with shape ``[B,Hq,Q,D]``.
    ///   - key: BF16 key heads with shape ``[B,Hkv,K,D]``, where ``Q <= K``.
    ///   - value: BF16 value heads with shape ``[B,Hkv,K,D]``.
    ///   - scale: FP32 scale applied after the BF16 QK^T matmul.
    ///   - additiveMask: optional full BF16 additive mask with shape
    ///     ``[B,Hq,Q,K]``.  Entries are added after the BF16 scale cast;
    ///     callers should use zero for allowed positions and
    ///     ``torch.bfloat16.finfo.min`` for blocked positions.
    /// - Returns: BF16 attention values with shape ``[B,T,Hq,D]``.
    ///
    /// This operation is intentionally strict.  It does not reinterpret
    /// mismatched shapes or dtypes.  Non-contiguous views are explicitly
    /// materialized before the no-copy bridge, and the resulting backing
    /// storage is checked before MPSGraph receives it.
    public static func attend(
        query: MLXArray,
        key: MLXArray,
        value: MLXArray,
        scale: Float,
        additiveMask: MLXArray? = nil
    ) -> MLXArray {
        let geometry = validate(
            query: query,
            key: key,
            value: value,
            scale: scale,
            additiveMask: additiveMask)
        let device = Self.sharedExecutionContext.device

        // MPSGraph TensorData is a no-copy bridge.  Materialize the transposed
        // Q/K/V views at this explicit boundary, then validate their backing
        // storage again before feeding the graph.  This keeps the production
        // caller free to use the natural [B,H,T,D] transpose layout without
        // silently passing incorrect strides to MPSGraph.
        let contiguousQuery = rowContiguous(query)
        let contiguousKey = rowContiguous(key)
        let contiguousValue = rowContiguous(value)
        let contiguousMask = additiveMask.map(rowContiguous)
        let queryBuffer = noCopyBuffer(contiguousQuery, device: device, label: "query")
        let keyBuffer = noCopyBuffer(contiguousKey, device: device, label: "key")
        let valueBuffer = noCopyBuffer(contiguousValue, device: device, label: "value")
        let maskBuffer = contiguousMask.map {
            noCopyBuffer($0, device: device, label: "additive mask")
        }

        planLock.lock()
        let plan = plan(
            queryShape: geometry.queryShape,
            keyValueShape: geometry.keyValueShape,
            queryHeads: geometry.queryHeads,
            keyValueHeads: geometry.keyValueHeads,
            queryTokens: geometry.queryTokens,
            keyTokens: geometry.keyTokens,
            headDimension: geometry.headDimension,
            scale: scale,
            hasAdditiveMask: additiveMask != nil)
        planLock.unlock()

        let outputShape = [
            geometry.batch,
            geometry.queryTokens,
            geometry.queryHeads,
            geometry.headDimension,
        ]
        let output = MLXArray.zeros(outputShape, dtype: .bfloat16)
        let outputBuffer = noCopyBuffer(output, device: device, label: "output")
        let queryData = MPSGraphTensorData(
            queryBuffer,
            shape: geometry.queryShape.map(NSNumber.init),
            dataType: .bFloat16)
        let keyData = MPSGraphTensorData(
            keyBuffer,
            shape: geometry.keyValueShape.map(NSNumber.init),
            dataType: .bFloat16)
        let valueData = MPSGraphTensorData(
            valueBuffer,
            shape: geometry.keyValueShape.map(NSNumber.init),
            dataType: .bFloat16)
        let maskData = maskBuffer.map {
            MPSGraphTensorData(
                $0,
                shape: geometry.additiveMaskShape.map(NSNumber.init),
                dataType: .bFloat16)
        }
        let outputData = MPSGraphTensorData(
            outputBuffer,
            shape: outputShape.map(NSNumber.init),
            dataType: .bFloat16)

        precondition(
            plan.outputTensor.shape?.map(\.intValue) == outputShape,
            "MiniCPM exact SDPA output shape changed during graph execution")
        precondition(
            plan.outputTensor.dataType == .bFloat16,
            "MiniCPM exact SDPA output must remain BF16")
        precondition(
            outputData.shape.map(\.intValue) == outputShape
                && outputData.dataType == .bFloat16,
            "MiniCPM exact SDPA output TensorData shape/dtype mismatch")

        let executionContext = Self.sharedExecutionContext
        executionContext.executionLock.lock()
        defer { executionContext.executionLock.unlock() }
        var feeds: [MPSGraphTensor: MPSGraphTensorData] = [
            plan.queryTensor: queryData,
            plan.keyTensor: keyData,
            plan.valueTensor: valueData,
        ]
        if let maskTensor = plan.additiveMaskTensor {
            guard let maskData else {
                preconditionFailure(
                    "MiniCPM exact SDPA graph requires an additive mask feed")
            }
            feeds[maskTensor] = maskData
        } else {
            precondition(
                maskData == nil,
                "MiniCPM exact SDPA received an additive mask for an unmasked graph")
        }
        plan.graph.run(
            with: executionContext.commandQueue,
            feeds: feeds,
            targetOperations: nil,
            resultsDictionary: [plan.outputTensor: outputData])
        return output
    }

    private struct Geometry {
        let queryShape: [Int]
        let keyValueShape: [Int]
        let batch: Int
        let queryHeads: Int
        let keyValueHeads: Int
        let queryTokens: Int
        let keyTokens: Int
        let headDimension: Int
        let additiveMaskShape: [Int]
    }

    private static func validate(
        query: MLXArray,
        key: MLXArray,
        value: MLXArray,
        scale: Float,
        additiveMask: MLXArray?
    ) -> Geometry {
        precondition(query.dtype == .bfloat16, "MiniCPM exact SDPA query must be BF16")
        precondition(key.dtype == .bfloat16, "MiniCPM exact SDPA key must be BF16")
        precondition(value.dtype == .bfloat16, "MiniCPM exact SDPA value must be BF16")
        precondition(query.ndim == 4, "MiniCPM exact SDPA query must have shape [B,H,T,D]")
        precondition(key.ndim == 4, "MiniCPM exact SDPA key must have shape [B,H,T,D]")
        precondition(value.ndim == 4, "MiniCPM exact SDPA value must have shape [B,H,T,D]")
        precondition(key.shape == value.shape, "MiniCPM exact SDPA key/value shapes must match")
        precondition(query.dim(0) == key.dim(0), "MiniCPM exact SDPA batch dimensions must match")
        precondition(query.dim(0) > 0, "MiniCPM exact SDPA requires a non-empty batch")
        precondition(
            query.dim(2) <= key.dim(2),
            "MiniCPM exact SDPA requires query length no greater than key length")
        precondition(query.dim(3) == key.dim(3), "MiniCPM exact SDPA head dimensions must match")
        precondition(query.dim(1) > 0, "MiniCPM exact SDPA requires at least one query head")
        precondition(key.dim(1) > 0, "MiniCPM exact SDPA requires at least one KV head")
        precondition(query.dim(2) > 0, "MiniCPM exact SDPA requires a non-empty query length")
        precondition(key.dim(2) > 0, "MiniCPM exact SDPA requires a non-empty key length")
        precondition(query.dim(3) > 0, "MiniCPM exact SDPA requires a non-empty head dimension")
        precondition(
            query.dim(1) % key.dim(1) == 0,
            "MiniCPM exact SDPA query heads must be divisible by KV heads")
        precondition(scale.isFinite && scale > 0, "MiniCPM exact SDPA scale must be finite and positive")
        let additiveMaskShape = [query.dim(0), query.dim(1), query.dim(2), key.dim(2)]
        if let additiveMask {
            precondition(
                additiveMask.dtype == .bfloat16,
                "MiniCPM exact SDPA additive mask must be BF16")
            precondition(
                additiveMask.shape == additiveMaskShape,
                "MiniCPM exact SDPA additive mask must have shape [B,Hq,Q,K]")
        }
        return Geometry(
            queryShape: query.shape,
            keyValueShape: key.shape,
            batch: query.dim(0),
            queryHeads: query.dim(1),
            keyValueHeads: key.dim(1),
            queryTokens: query.dim(2),
            keyTokens: key.dim(2),
            headDimension: query.dim(3),
            additiveMaskShape: additiveMaskShape)
    }

    private static func plan(
        queryShape: [Int],
        keyValueShape: [Int],
        queryHeads: Int,
        keyValueHeads: Int,
        queryTokens: Int,
        keyTokens: Int,
        headDimension: Int,
        scale: Float,
        hasAdditiveMask: Bool
    ) -> Plan {
        let key = PlanKey(
            queryShape: queryShape,
            keyValueShape: keyValueShape,
            scaleBits: scale.bitPattern,
            hasAdditiveMask: hasAdditiveMask)
        if let cached = plans[key] { return cached }

        let graph = MPSGraph()
        let queryTensor = graph.placeholder(
            shape: queryShape.map(NSNumber.init),
            dataType: .bFloat16,
            name: "minicpm_exact_sdpa_query")
        let keyTensor = graph.placeholder(
            shape: keyValueShape.map(NSNumber.init),
            dataType: .bFloat16,
            name: "minicpm_exact_sdpa_key")
        let valueTensor = graph.placeholder(
            shape: keyValueShape.map(NSNumber.init),
            dataType: .bFloat16,
            name: "minicpm_exact_sdpa_value")
        let repeatedKey = repeatKV(
            graph: graph,
            tensor: keyTensor,
            batch: queryShape[0],
            keyValueHeads: keyValueHeads,
            queryHeads: queryHeads,
            tokens: keyTokens,
            headDimension: headDimension,
            name: "minicpm_exact_sdpa_key_repeat")
        let repeatedValue = repeatKV(
            graph: graph,
            tensor: valueTensor,
            batch: queryShape[0],
            keyValueHeads: keyValueHeads,
            queryHeads: queryHeads,
            tokens: keyTokens,
            headDimension: headDimension,
            name: "minicpm_exact_sdpa_value_repeat")
        let keyTranspose = graph.transpose(
            repeatedKey,
            permutation: [0, 1, 3, 2],
            name: "minicpm_exact_sdpa_key_transpose")
        let qk = graph.matrixMultiplication(
            primary: queryTensor,
            secondary: keyTranspose,
            name: "minicpm_exact_sdpa_qk_bf16")
        precondition(qk.dataType == .bFloat16, "MiniCPM exact SDPA QK must remain BF16")

        let scaleTensor = graph.constant(
            Double(scale),
            shape: [1].map(NSNumber.init),
            dataType: .float32)
        var scaled = graph.cast(
            qk,
            to: .float32,
            name: "minicpm_exact_sdpa_qk_fp32")
        scaled = graph.multiplication(
            scaled,
            scaleTensor,
            name: "minicpm_exact_sdpa_scale_fp32")
        scaled = graph.cast(
            scaled,
            to: .bFloat16,
            name: "minicpm_exact_sdpa_scale_bf16")

        let additiveMaskTensor: MPSGraphTensor?
        let masked: MPSGraphTensor
        if hasAdditiveMask {
            let placeholder = graph.placeholder(
                shape: [queryShape[0], queryHeads, queryTokens, keyTokens]
                    .map(NSNumber.init),
                dataType: .bFloat16,
                name: "minicpm_exact_sdpa_additive_mask")
            precondition(
                scaled.dataType == .bFloat16,
                "MiniCPM exact SDPA masked scores must be BF16 before addition")
            masked = graph.addition(
                scaled,
                placeholder,
                name: "minicpm_exact_sdpa_additive_mask_add")
            additiveMaskTensor = placeholder
        } else {
            // Square prefill (and direct qLen=1 probes) retain the original
            // causal-select graph.  The cached rectangular caller supplies a
            // full additive mask instead, matching PyTorch's
            // sdpa_general_mps path exactly.
            let ones = graph.constant(
                1.0,
                shape: [queryTokens, keyTokens].map(NSNumber.init),
                dataType: .bool)
            let causalMask = graph.bandPart(
                ones,
                numLower: -1,
                numUpper: keyTokens - queryTokens,
                name: "minicpm_exact_sdpa_band")
            let minusInf = graph.constant(
                -1e20,
                shape: [queryShape[0], queryHeads, queryTokens, keyTokens].map(NSNumber.init),
                dataType: .bFloat16)
            masked = graph.select(
                predicate: causalMask,
                trueTensor: scaled,
                falseTensor: minusInf,
                name: "minicpm_exact_sdpa_select")
            additiveMaskTensor = nil
        }
        precondition(
            masked.dataType == .bFloat16,
            "MiniCPM exact SDPA masked scores must remain BF16")
        let softmax = graph.softMax(
            with: masked,
            axis: 3,
            name: "minicpm_exact_sdpa_softmax_bf16")
        precondition(softmax.dataType == .bFloat16, "MiniCPM exact SDPA softmax must remain BF16")
        let attended = graph.matrixMultiplication(
            primary: softmax,
            secondary: repeatedValue,
            name: "minicpm_exact_sdpa_v_bf16")
        precondition(attended.dataType == .bFloat16, "MiniCPM exact SDPA value matmul must remain BF16")
        let outputTensor = graph.transpose(
            attended,
            permutation: [0, 2, 1, 3],
            name: "minicpm_exact_sdpa_merge")
        let outputShape = [queryShape[0], queryTokens, queryHeads, headDimension]
        precondition(
            outputTensor.shape?.map(\.intValue) == outputShape,
            "MiniCPM exact SDPA inferred an unexpected output shape")
        precondition(
            outputTensor.dataType == .bFloat16,
            "MiniCPM exact SDPA inferred a non-BF16 output")

        let created = Plan(
            graph: graph,
            queryTensor: queryTensor,
            keyTensor: keyTensor,
            valueTensor: valueTensor,
            additiveMaskTensor: additiveMaskTensor,
            outputTensor: outputTensor)
        plans[key] = created
        return created
    }

    private static func repeatKV(
        graph: MPSGraph,
        tensor: MPSGraphTensor,
        batch: Int,
        keyValueHeads: Int,
        queryHeads: Int,
        tokens: Int,
        headDimension: Int,
        name: String
    ) -> MPSGraphTensor {
        let repeats = queryHeads / keyValueHeads
        guard repeats > 0 else {
            preconditionFailure("MiniCPM exact SDPA requires positive KV repetition")
        }
        if repeats == 1 { return tensor }
        let singleton = graph.reshape(
            tensor,
            shape: [batch, keyValueHeads, 1, tokens, headDimension].map(NSNumber.init),
            name: "\(name)_singleton")
        let broadcast = graph.broadcast(
            singleton,
            shape: [batch, keyValueHeads, repeats, tokens, headDimension].map(NSNumber.init),
            name: "\(name)_broadcast")
        return graph.reshape(
            broadcast,
            shape: [batch, queryHeads, tokens, headDimension].map(NSNumber.init),
            name: name)
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
        precondition(array.dtype == .bfloat16, "MiniCPM exact SDPA \(label) must be BF16")
        let backing = array.asData(access: .noCopy)
        precondition(
            backing.dType == .bfloat16
                && isRowContiguous(shape: array.shape, strides: backing.strides),
            "MiniCPM exact SDPA \(label) must be row-contiguous")
        guard let buffer = array.asMTLBuffer(device: device, noCopy: true) else {
            preconditionFailure("MiniCPM exact SDPA no-copy MTLBuffer unavailable for \(label)")
        }
        precondition(
            buffer.length >= array.nbytes,
            "MiniCPM exact SDPA MTLBuffer is too small for \(label)")
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
