import Foundation
import Metal
import MetalPerformanceShadersGraph
import XCTest
import MLX
import MLXCommon
import MLXNN
@testable import MiniCPMLLM

/// Opt-in feasibility probe for a dense BF16 layer-0 MLP down projection.
///
/// This intentionally does not add a numerical tolerance gate: the useful
/// result of this test is the pair of metrics printed for the existing MLX
/// implementation and the MPSGraph implementation against the same official
/// trace.  Shape, storage type, finiteness, and the no-copy contract are hard
/// checks; numerical differences remain visible instead of being hidden by a
/// broad cross-backend threshold.
final class E2EMiniCPMMPSGraphLinearTests: XCTestCase {
    private struct Metrics {
        let maxAbs: Double
        let meanAbs: Double
        let rms: Double
        let neq: Int
    }

    func testMPSGraphBF16DownProjectionNoCopy() throws {
        let environment = ProcessInfo.processInfo.environment
        guard environment["MINICPM_MPSGRAPH_LINEAR_PROBE"] == "1" else {
            throw XCTSkip("set MINICPM_MPSGRAPH_LINEAR_PROBE=1 to run the real-model probe")
        }

        guard let modelPath = environment["MINICPM_MPSGRAPH_LINEAR_MODEL"],
              !modelPath.isEmpty else {
            XCTFail("MINICPM_MPSGRAPH_LINEAR_MODEL is required when the probe is enabled")
            return
        }
        guard let tracePath = environment["MINICPM_MPSGRAPH_LINEAR_TRACE_DIR"],
              !tracePath.isEmpty else {
            XCTFail("MINICPM_MPSGRAPH_LINEAR_TRACE_DIR is required when the probe is enabled")
            return
        }

        let modelURL = URL(fileURLWithPath: modelPath, isDirectory: true)
        let traceURL = URL(fileURLWithPath: tracePath, isDirectory: true)
        guard isDirectory(modelURL) else {
            XCTFail("dense BF16 model bundle is not a directory: \(modelURL.path)")
            return
        }
        guard isDirectory(traceURL) else {
            XCTFail("official layer trace is not a directory: \(traceURL.path)")
            return
        }

        // ``--trace-layer 0`` writes the authoritative targeted names.  Keep
        // the old unprefixed names as a compatibility fallback for existing
        // traces generated before targeted-layer naming was made explicit.
        let inputValues = try readFloat32(
            from: traceTensorURL(named: "mul", in: traceURL))
        let referenceValues = try readFloat32(
            from: traceTensorURL(named: "down", in: traceURL))
        guard inputValues.allSatisfy(\.isFinite), referenceValues.allSatisfy(\.isFinite) else {
            XCTFail("official mul.f32/down.f32 contain non-finite values")
            return
        }

        // Construct and load exactly one dense model.  This probe never calls
        // the full forward path: only layer 0's down projection is evaluated.
        let config = try MiniCPMMLXConfig.load(from: modelURL)
        guard config.quantBits == 0 else {
            XCTFail("MPSGraph BF16 probe requires a dense bundle (quantization bits must be zero)")
            return
        }
        let model = MiniCPMLanguageModel(config: config)
        try MiniCPMWeightLoader.load(model: model, from: modelURL)
        guard let layer = model.layers.first else {
            XCTFail("dense bundle has no transformer layers")
            return
        }
        let downProj = layer.mlp.downProj
        let weight = downProj.weight
        eval(weight)

        guard weight.dtype == .bfloat16 else {
            XCTFail("layer-0 down_proj weight must be BF16, got \(weight.dtype)")
            return
        }
        guard weight.shape == [config.hiddenSize, config.intermediateSize] else {
            XCTFail(
                "layer-0 down_proj weight shape \(weight.shape) does not match "
                    + "[\(config.hiddenSize), \(config.intermediateSize)]")
            return
        }

        let inputWidth = weight.shape[1]
        let outputWidth = weight.shape[0]
        guard inputWidth > 0, outputWidth > 0,
              inputValues.count % inputWidth == 0 else {
            XCTFail(
                "mul.f32 element count \(inputValues.count) is incompatible with "
                    + "down_proj input width \(inputWidth)")
            return
        }
        let tokenCount = inputValues.count / inputWidth
        guard tokenCount > 0,
              referenceValues.count == tokenCount * outputWidth else {
            XCTFail(
                "official down.f32 has \(referenceValues.count) values; expected "
                    + "\(tokenCount * outputWidth) for [1, \(tokenCount), \(outputWidth)]")
            return
        }

        // The exporter records layer traces as [batch, tokens, channels].
        // Restoring mul.f32 to BF16 here is intentional: both MLX and
        // MPSGraph receive the exact same BF16 activation bytes.
        let inputShape = [1, tokenCount, inputWidth]
        let outputShape = [1, tokenCount, outputWidth]
        let input = MLXArray(inputValues, inputShape).asType(.bfloat16)
        eval(input)
        guard input.dtype == .bfloat16 else {
            XCTFail("restored official mul.f32 input is not BF16")
            return
        }
        let restoredInputValues = input.asType(.float32).asArray(Float.self)
        guard restoredInputValues.allSatisfy(\.isFinite) else {
            XCTFail("restored official mul.f32 BF16 input produced non-finite values")
            return
        }

        guard let metalDevice = MTLCreateSystemDefaultDevice() else {
            throw XCTSkip("Metal GPU device is unavailable")
        }
        let inputBuffer = try noCopyBuffer(
            for: input, device: metalDevice, label: "mul BF16 input")
        let weightBuffer = try noCopyBuffer(
            for: weight, device: metalDevice, label: "down_proj BF16 weight")

        // Existing MLX baseline, evaluated from the same BF16 input and the
        // same loaded BF16 Linear weight as the graph path.
        let mlxBaseline = downProj(input)
        eval(mlxBaseline)
        guard mlxBaseline.shape == outputShape else {
            XCTFail("MLX down_proj shape \(mlxBaseline.shape) != \(outputShape)")
            return
        }
        guard mlxBaseline.dtype == .bfloat16 else {
            XCTFail("MLX down_proj baseline must remain BF16, got \(mlxBaseline.dtype)")
            return
        }
        let mlxValues = mlxBaseline.asType(.float32).asArray(Float.self)
        guard mlxValues.allSatisfy(\.isFinite) else {
            XCTFail("MLX down_proj baseline produced non-finite values")
            return
        }

        // MPSGraph receives both contiguous BF16 arrays through the MTL
        // buffers above.  The transpose is a graph operation, not a Swift or
        // MLX pre-transpose/copy of the weight.
        let graph = MPSGraph()
        let inputTensor = graph.placeholder(
            shape: inputShape.map(NSNumber.init),
            dataType: .bFloat16,
            name: "mul_bf16")
        let weightTensor = graph.placeholder(
            shape: weight.shape.map(NSNumber.init),
            dataType: .bFloat16,
            name: "down_proj_weight_bf16")
        let transposedWeight = graph.transpose(
            weightTensor, permutation: [1, 0], name: "down_proj_weight_transpose")
        let outputTensor = graph.matrixMultiplication(
            primary: inputTensor,
            secondary: transposedWeight,
            name: "down_proj_bf16_matmul")
        XCTAssertEqual(outputTensor.shape?.map(\.intValue), outputShape)
        XCTAssertEqual(outputTensor.dataType, .bFloat16)

        let inputData = MPSGraphTensorData(
            inputBuffer,
            shape: inputShape.map(NSNumber.init),
            dataType: .bFloat16)
        let weightData = MPSGraphTensorData(
            weightBuffer,
            shape: weight.shape.map(NSNumber.init),
            dataType: .bFloat16)
        let graphResults = graph.run(
            feeds: [inputTensor: inputData, weightTensor: weightData],
            targetTensors: [outputTensor],
            targetOperations: nil)
        guard let graphOutputData = graphResults[outputTensor] else {
            XCTFail("MPSGraph returned no down_proj output")
            return
        }
        guard graphOutputData.shape.map(\.intValue) == outputShape else {
            XCTFail("MPSGraph output shape \(graphOutputData.shape) != \(outputShape)")
            return
        }
        guard graphOutputData.dataType == .bFloat16 else {
            XCTFail("MPSGraph output must remain BF16, got \(graphOutputData.dataType)")
            return
        }

        let graphOutputNDArray = graphOutputData.mpsndarray()
        guard graphOutputNDArray.dataType == .bFloat16 else {
            XCTFail("MPSGraph NDArray output must remain BF16, got \(graphOutputNDArray.dataType)")
            return
        }
        var graphBits = [UInt16](repeating: 0, count: referenceValues.count)
        graphOutputNDArray.readBytes(
            &graphBits,
            strideBytes: nil as UnsafeMutablePointer<Int>?)
        let graphValues = graphBits.map(Self.floatFromBFloat16)
        guard graphValues.allSatisfy(\.isFinite) else {
            XCTFail("MPSGraph BF16 down_proj produced non-finite values")
            return
        }

        let baselineMetrics = try metrics(mlxValues, against: referenceValues)
        let graphMetrics = try metrics(graphValues, against: referenceValues)
        printMetrics("MLX baseline", baselineMetrics, count: referenceValues.count)
        printMetrics("MPSGraph BF16", graphMetrics, count: referenceValues.count)

        // Reverse bridge: let MPSGraph write directly into a preallocated,
        // contiguous MLX BF16 array.  Keeping this separate from the normal
        // target-tensor run above proves the production-style output handoff
        // instead of merely comparing two independently allocated results.
        let writeThrough = MLXArray.zeros(outputShape, dtype: .bfloat16)
        eval(writeThrough)
        guard writeThrough.shape == outputShape, writeThrough.dtype == .bfloat16 else {
            XCTFail(
                "preallocated MPSGraph output must be contiguous BF16 with shape "
                    + "\(outputShape); got \(writeThrough.shape)/\(writeThrough.dtype)")
            return
        }
        let writeThroughBuffer = try noCopyBuffer(
            for: writeThrough, device: metalDevice, label: "MPSGraph output")
        let writeThroughData = MPSGraphTensorData(
            writeThroughBuffer,
            shape: outputShape.map(NSNumber.init),
            dataType: .bFloat16)
        guard writeThroughData.shape.map(\.intValue) == outputShape,
              writeThroughData.dataType == .bFloat16 else {
            XCTFail("MPSGraph resultsDictionary output TensorData shape/dtype mismatch")
            return
        }
        guard let commandQueue = metalDevice.makeCommandQueue() else {
            throw XCTSkip("Metal command queue is unavailable for MPSGraph output bridge")
        }

        let resultsDictionary: [MPSGraphTensor: MPSGraphTensorData] = [
            outputTensor: writeThroughData,
        ]
        let firstRunStart = DispatchTime.now().uptimeNanoseconds
        graph.run(
            with: commandQueue,
            feeds: [inputTensor: inputData, weightTensor: weightData],
            targetOperations: nil,
            resultsDictionary: resultsDictionary)
        let firstRunMilliseconds = Double(
            DispatchTime.now().uptimeNanoseconds - firstRunStart) / 1_000_000.0

        // A couple of synchronous warm runs make the optional first/warm
        // timing useful without changing any correctness or tolerance gate.
        var warmRunMilliseconds: [Double] = []
        for _ in 0..<2 {
            let start = DispatchTime.now().uptimeNanoseconds
            graph.run(
                with: commandQueue,
                feeds: [inputTensor: inputData, weightTensor: weightData],
                targetOperations: nil,
                resultsDictionary: resultsDictionary)
            warmRunMilliseconds.append(
                Double(DispatchTime.now().uptimeNanoseconds - start) / 1_000_000.0)
        }
        print(
            "MPSGRAPH_LINEAR output_bridge_ms first=\(firstRunMilliseconds) "
                + "warm=\(warmRunMilliseconds)")

        // Reading the buffer through MLX after the synchronous graph call is
        // the key reverse-bridge check.  The addition is an actual MLX op,
        // not a host-side byte read, and should preserve BF16 values when
        // adding a BF16 zero tensor.
        let writeThroughDirectValues = writeThrough.asType(.float32).asArray(Float.self)
        guard writeThroughDirectValues.allSatisfy(\.isFinite) else {
            XCTFail("MPSGraph direct MLX output contains non-finite values")
            return
        }
        let zero = MLXArray.zeros(outputShape, dtype: .bfloat16)
        eval(zero)
        let mlxConsumed = writeThrough + zero
        eval(mlxConsumed)
        guard mlxConsumed.shape == outputShape, mlxConsumed.dtype == .bfloat16 else {
            XCTFail("MLX consumer changed bridged output shape/dtype")
            return
        }
        let mlxConsumedValues = mlxConsumed.asType(.float32).asArray(Float.self)
        guard mlxConsumedValues.allSatisfy(\.isFinite) else {
            XCTFail("MLX consumer produced non-finite values from bridged output")
            return
        }
        let writeThroughMetrics = try metrics(
            writeThroughDirectValues, against: referenceValues)
        let consumedMetrics = try metrics(mlxConsumedValues, against: referenceValues)
        printMetrics(
            "MPSGraph write-through", writeThroughMetrics, count: referenceValues.count)
        printMetrics(
            "MLX consumed write-through", consumedMetrics, count: referenceValues.count)
    }

    private func isDirectory(_ url: URL) -> Bool {
        var directory: ObjCBool = false
        return FileManager.default.fileExists(atPath: url.path, isDirectory: &directory)
            && directory.boolValue
    }

    private func readFloat32(from url: URL) throws -> [Float] {
        let data = try Data(contentsOf: url)
        guard data.count % MemoryLayout<Float>.stride == 0 else {
            throw NSError(
                domain: "MiniCPMMPSGraphLinearProbe",
                code: 1,
                userInfo: [NSLocalizedDescriptionKey:
                    "trace tensor is not little-endian float32: \(url.path)"])
        }
        return data.withUnsafeBytes { raw in
            Array(raw.bindMemory(to: Float.self))
        }
    }

    private func traceTensorURL(named name: String, in directory: URL) throws -> URL {
        let targeted = directory.appendingPathComponent("layer_00_\(name).f32")
        let legacy = directory.appendingPathComponent("\(name).f32")
        if FileManager.default.fileExists(atPath: targeted.path) { return targeted }
        if FileManager.default.fileExists(atPath: legacy.path) { return legacy }
        throw NSError(
            domain: "MiniCPMMPSGraphLinearProbe",
            code: 3,
            userInfo: [NSLocalizedDescriptionKey:
                "missing \(name).f32; tried \(targeted.path) and \(legacy.path)"])
    }

    /// Return an MTLBuffer only when MLX's no-copy path is demonstrably used.
    /// ``asMTLBuffer(noCopy: true)`` otherwise falls back to a copy for a
    /// non-contiguous array, so inspect the backing strides and pointer before
    /// accepting the buffer instead of silently allowing that fallback.
    private func noCopyBuffer(
        for array: MLXArray,
        device: any MTLDevice,
        label: String
    ) throws -> any MTLBuffer {
        eval(array)
        let backing = array.asData(access: .noCopy)
        let expectedStrides = contiguousStrides(for: array.shape)
        guard backing.dType == .bfloat16, backing.strides == expectedStrides else {
            throw XCTSkip(
                "\(label) is not contiguous BF16; refusing an implicit no-copy fallback")
        }
        guard let buffer = array.asMTLBuffer(device: device, noCopy: true) else {
            throw XCTSkip("MLX no-copy MTLBuffer is unavailable for \(label)")
        }
        guard buffer.length >= array.nbytes else {
            throw XCTSkip("MLX no-copy MTLBuffer is too small for \(label)")
        }

        let sourceAddress = backing.data.withUnsafeBytes {
            (raw: UnsafeRawBufferPointer) -> Int? in
            raw.baseAddress.map { Int(bitPattern: $0) }
        }
        let bufferAddress = Int(bitPattern: buffer.contents())
        guard let sourceAddress, sourceAddress == bufferAddress else {
            throw XCTSkip(
                "MLX no-copy MTLBuffer did not alias \(label) backing storage")
        }
        return buffer
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

    private func metrics(_ actual: [Float], against reference: [Float]) throws -> Metrics {
        guard actual.count == reference.count else {
            throw NSError(
                domain: "MiniCPMMPSGraphLinearProbe",
                code: 2,
                userInfo: [NSLocalizedDescriptionKey:
                    "metric length mismatch: \(actual.count) vs \(reference.count)"])
        }
        var maxAbs = 0.0
        var sumAbs = 0.0
        var sumSquares = 0.0
        var neq = 0
        for (value, expected) in zip(actual, reference) {
            let delta = abs(Double(value) - Double(expected))
            maxAbs = max(maxAbs, delta)
            sumAbs += delta
            sumSquares += delta * delta
            if value != expected { neq += 1 }
        }
        let count = Double(actual.count)
        return Metrics(
            maxAbs: maxAbs,
            meanAbs: count == 0 ? 0 : sumAbs / count,
            rms: count == 0 ? 0 : sqrt(sumSquares / count),
            neq: neq)
    }

    private func printMetrics(_ label: String, _ metrics: Metrics, count: Int) {
        print(String(format:
            "MPSGRAPH_LINEAR %@ max=%.9g mean=%.9g rms=%.9g neq=%d/%d",
            label,
            metrics.maxAbs,
            metrics.meanAbs,
            metrics.rms,
            metrics.neq,
            count))
    }

    private static func floatFromBFloat16(_ bits: UInt16) -> Float {
        Float(bitPattern: UInt32(bits) << 16)
    }
}
