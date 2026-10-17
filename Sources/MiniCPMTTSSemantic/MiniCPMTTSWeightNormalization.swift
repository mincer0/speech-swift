import Foundation
import Metal
import MetalPerformanceShadersGraph
import MLX

/// Reconstruct the audio head's dim=0 weight parametrization once at load.
/// PyTorch's BF16 MPS weight_norm keeps the entire graph in BF16 and exports
/// both norms and weights. A Float32 reconstruction changes the checkpoint's
/// effective weights, even when it is cast back to BF16 afterwards.
enum MiniCPMTTSWeightNormalization {
    static func reconstruct(direction: MLXArray, magnitude: MLXArray) -> MLXArray {
        precondition(direction.ndim == 2)
        precondition(magnitude.shape == [direction.dim(0), 1])
        precondition(direction.dtype == magnitude.dtype)
        guard direction.dtype == .bfloat16 else {
            let norm = MLX.sqrt((direction * direction).sum(axis: -1, keepDims: true))
            return (direction / norm) * magnitude
        }
        guard let device = MTLCreateSystemDefaultDevice(),
              let queue = device.makeCommandQueue()
        else {
            preconditionFailure("MiniCPM BF16 weight normalization requires a Metal device")
        }
        let v = direction.contiguous()
        let g = magnitude.contiguous()
        let norms = MLXArray.zeros(magnitude.shape, dtype: .bfloat16)
        let weights = MLXArray.zeros(direction.shape, dtype: .bfloat16)

        func tensorData(_ array: MLXArray) -> MPSGraphTensorData {
            eval(array)
            let backing = array.asData(access: .noCopy)
            precondition(backing.dType == .bfloat16)
            var expectedStride = 1
            for axis in array.shape.indices.reversed() {
                precondition(array.dim(axis) == 1 || backing.strides[axis] == expectedStride)
                expectedStride *= array.dim(axis)
            }
            guard let buffer = array.asMTLBuffer(device: device, noCopy: true) else {
                preconditionFailure("MiniCPM weight normalization cannot bridge MLX storage")
            }
            precondition(buffer.length >= array.nbytes)
            return MPSGraphTensorData(
                buffer, shape: array.shape.map(NSNumber.init), dataType: .bFloat16)
        }
        let vData = tensorData(v)
        let gData = tensorData(g)
        let normData = tensorData(norms)
        let weightData = tensorData(weights)
        let graph = MPSGraph()
        let vTensor = graph.placeholder(
            shape: v.shape.map(NSNumber.init), dataType: .bFloat16, name: nil)
        let gTensor = graph.placeholder(
            shape: g.shape.map(NSNumber.init), dataType: .bFloat16, name: nil)
        let squared = graph.square(with: vTensor, name: nil)
        let summed = graph.reductionSum(with: squared, axes: [1], name: nil)
        let normTensor = graph.squareRoot(with: summed, name: nil)
        let unit = graph.division(vTensor, normTensor, name: nil)
        let weightTensor = graph.multiplication(unit, gTensor, name: nil)
        // TensorData's no-copy buffer wrappers do not retain their MLX owners.
        withExtendedLifetime((v, g, norms, weights)) {
            graph.run(
                with: queue, feeds: [vTensor: vData, gTensor: gData],
                targetOperations: nil,
                resultsDictionary: [normTensor: normData, weightTensor: weightData])
        }
        return weights
    }
}
