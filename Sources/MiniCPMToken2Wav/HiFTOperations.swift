import Foundation
import MLX
import MLXNN

final class MiniCPMSnake: Module {
    @ParameterInfo(key: "alpha") var alpha: MLXArray

    init(channels: Int) {
        _alpha.wrappedValue = MLXArray.ones([channels])
        super.init()
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        let value = alpha.reshaped([1, -1, 1])
        let periodic = sin(value * input)
        return input + periodic * periodic / (value + 1e-9)
    }
}

/// Standard PyTorch-style Conv1d with symmetric integer padding and NCL I/O.
final class MiniCPMConv1d: Module {
    @ParameterInfo(key: "weight") var weight: MLXArray
    @ParameterInfo(key: "bias") var bias: MLXArray
    let stride: Int
    let padding: Int
    let dilation: Int

    init(
        inputChannels: Int,
        outputChannels: Int,
        kernelSize: Int,
        stride: Int = 1,
        padding: Int = 0,
        dilation: Int = 1
    ) {
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        let scale = Float(Foundation.sqrt(1.0 / Double(inputChannels * kernelSize)))
        _weight.wrappedValue = MLXRandom.uniform(
            low: -scale, high: scale, [outputChannels, kernelSize, inputChannels])
        _bias.wrappedValue = MLXArray.zeros([outputChannels])
        super.init()
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        var value = input.transposed(0, 2, 1)
        value = conv1d(
            value, weight, stride: stride, padding: padding, dilation: dilation)
        value = value + bias
        return value.transposed(0, 2, 1)
    }
}

/// PyTorch ConvTranspose1d with NCL I/O. Converted weights use [out, kernel, in].
final class MiniCPMConvTranspose1d: Module {
    @ParameterInfo(key: "weight") var weight: MLXArray
    @ParameterInfo(key: "bias") var bias: MLXArray
    let stride: Int
    let padding: Int

    init(
        inputChannels: Int,
        outputChannels: Int,
        kernelSize: Int,
        stride: Int,
        padding: Int
    ) {
        self.stride = stride
        self.padding = padding
        let scale = Float(Foundation.sqrt(1.0 / Double(inputChannels * kernelSize)))
        _weight.wrappedValue = MLXRandom.uniform(
            low: -scale, high: scale, [outputChannels, kernelSize, inputChannels])
        _bias.wrappedValue = MLXArray.zeros([outputChannels])
        super.init()
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        var value = input.transposed(0, 2, 1)
        value = convTransposed1d(value, weight, stride: stride, padding: padding)
        value = value + bias
        return value.transposed(0, 2, 1)
    }
}

final class MiniCPMHiFTResidualBlock: Module {
    @ModuleInfo(key: "convs1") var convolutions1: [MiniCPMConv1d]
    @ModuleInfo(key: "convs2") var convolutions2: [MiniCPMConv1d]
    @ModuleInfo(key: "activations1") var activations1: [MiniCPMSnake]
    @ModuleInfo(key: "activations2") var activations2: [MiniCPMSnake]

    init(channels: Int, kernelSize: Int, dilations: [Int]) {
        var first: [MiniCPMConv1d] = []
        var second: [MiniCPMConv1d] = []
        var firstActivations: [MiniCPMSnake] = []
        var secondActivations: [MiniCPMSnake] = []
        for dilation in dilations {
            first.append(MiniCPMConv1d(
                inputChannels: channels,
                outputChannels: channels,
                kernelSize: kernelSize,
                padding: (kernelSize * dilation - dilation) / 2,
                dilation: dilation))
            second.append(MiniCPMConv1d(
                inputChannels: channels,
                outputChannels: channels,
                kernelSize: kernelSize,
                padding: (kernelSize - 1) / 2))
            firstActivations.append(MiniCPMSnake(channels: channels))
            secondActivations.append(MiniCPMSnake(channels: channels))
        }
        _convolutions1 = ModuleInfo(wrappedValue: first, key: "convs1")
        _convolutions2 = ModuleInfo(wrappedValue: second, key: "convs2")
        _activations1 = ModuleInfo(wrappedValue: firstActivations, key: "activations1")
        _activations2 = ModuleInfo(wrappedValue: secondActivations, key: "activations2")
        super.init()
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        var value = input
        for index in convolutions1.indices {
            var residual = activations1[index](value)
            residual = convolutions1[index](residual)
            residual = activations2[index](residual)
            residual = convolutions2[index](residual)
            value = value + residual
        }
        return value
    }
}

@inline(__always)
func miniCPMELU(_ input: MLXArray) -> MLXArray {
    MLX.where(input .> 0, input, exp(input) - 1)
}

@inline(__always)
func miniCPMLeakyReLU(_ input: MLXArray, slope: Float) -> MLXArray {
    maximum(input, slope * input)
}
