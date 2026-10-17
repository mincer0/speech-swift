import MLX
import MLXFast

/// Shared-memory candidate for the audio encoder's sequential FP32 affine
/// boundary. Tiling reuses loads while each output retains channel-order FMA,
/// a separate bias add, and one destination-dtype store.
enum MiniCPMAudioExactTiledLinear {
    static func apply(
        input: MLXArray,
        weight: MLXArray,
        bias: MLXArray?
    ) -> MLXArray? {
        guard input.ndim == 3, weight.ndim == 2,
              input.dtype == .bfloat16 || input.dtype == .float32,
              weight.dtype == input.dtype,
              input.dim(0) > 0, input.dim(1) > 0, input.dim(2) > 0,
              weight.dim(0) > 0, weight.dim(1) == input.dim(2)
        else { return nil }
        let batch = input.dim(0)
        let sequenceLength = input.dim(1)
        let inputChannels = input.dim(2)
        let outputChannels = weight.dim(0)
        if let bias {
            guard bias.ndim == 1, bias.dim(0) == outputChannels,
                  bias.dtype == input.dtype
            else { return nil }
        }
        // Metal address arithmetic uses uint; scalar shape arguments use
        // Int32, matching the original sequential kernel's interface.
        let addressLimit = Int(UInt32.max)
        guard [batch, sequenceLength, inputChannels, outputChannels]
            .allSatisfy({ $0 <= Int(Int32.max) }),
              input.size <= addressLimit, weight.size <= addressLimit,
              batch <= addressLimit / sequenceLength,
              batch * sequenceLength <= addressLimit / outputChannels
        else { return nil }

        let contiguousInput = input.contiguous()
        let contiguousWeight = weight.contiguous()
        let contiguousBias = (bias ?? MLXArray.zeros(
            [outputChannels], dtype: input.dtype)).contiguous()
        // Pad both axes so even boundary groups contain all 256 threads.
        // Every thread must reach both barriers in every input-channel tile.
        let outputGrid = ((outputChannels + 31) / 32) * 32
        let sequenceGrid = ((sequenceLength + 7) / 8) * 8
        return kernel(
            [contiguousInput, contiguousWeight, contiguousBias,
             Int32(sequenceLength), Int32(inputChannels), Int32(outputChannels)],
            template: [("T", input.dtype)],
            grid: (outputGrid, sequenceGrid, batch),
            threadGroup: (32, 8, 1),
            outputShapes: [[batch, sequenceLength, outputChannels]],
            outputDTypes: [input.dtype]
        )[0]
    }

    private static let kernel = MLXFast.metalKernel(
        name: "minicpm_audio_linear_mps_exact_tiled",
        inputNames: ["x", "weight", "bias", "sequence_length", "input_channels", "output_channels"],
        outputNames: ["out"],
        source: """
            uint localOutput = thread_position_in_threadgroup.x;
            uint localSequence = thread_position_in_threadgroup.y;
            uint threadIndex = localSequence * 32 + localOutput;
            uint firstOutput = threadgroup_position_in_grid.x * 32;
            uint firstSequence = threadgroup_position_in_grid.y * 8;
            uint outputChannel = firstOutput + localOutput;
            uint sequence = firstSequence + localSequence;
            uint batch = thread_position_in_grid.z;
            uint sequenceLength = uint(sequence_length);
            uint inputChannels = uint(input_channels);
            uint outputChannels = uint(output_channels);
            bool validOutput = outputChannel < outputChannels && sequence < sequenceLength;

            threadgroup float inputTile[8 * 32];
            // Transpose the shared layout so output lanes read consecutive
            // weights for each channel. Padding also avoids stride-32 bank
            // conflicts while loading consecutive global input channels.
            threadgroup float weightTile[32 * 33];
            float accumulator = 0.0f;
            for (uint firstChannel = 0; firstChannel < inputChannels; firstChannel += 32) {
                uint inputChannel = firstChannel + localOutput;
                inputTile[threadIndex] = sequence < sequenceLength && inputChannel < inputChannels
                    ? float(x[(batch * sequenceLength + sequence) * inputChannels + inputChannel])
                    : 0.0f;

                // Each SIMD group loads consecutive input-channel weights;
                // four loads per thread cover all 32 output-channel rows.
                for (uint part = 0; part < 4; part++) {
                    uint tileIndex = threadIndex + part * 256;
                    uint tileOutput = tileIndex / 32;
                    uint tileChannel = tileIndex % 32;
                    uint weightOutput = firstOutput + tileOutput;
                    uint weightChannel = firstChannel + tileChannel;
                    weightTile[tileChannel * 33 + tileOutput] = weightOutput < outputChannels && weightChannel < inputChannels
                        ? float(weight[weightOutput * inputChannels + weightChannel])
                        : 0.0f;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                if (validOutput) {
                    uint channelCount = min(uint(32), inputChannels - firstChannel);
                    for (uint channel = 0; channel < channelCount; channel++) {
                        accumulator = metal::fma(
                            inputTile[localSequence * 32 + channel],
                            weightTile[channel * 33 + localOutput],
                            accumulator);
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            if (validOutput) {
                uint outputOffset = (batch * sequenceLength + sequence) * outputChannels
                    + outputChannel;
                accumulator += float(bias[outputChannel]);
                out[outputOffset] = T(accumulator);
            }
        """,
        header: """
            #include <metal_stdlib>
            using namespace metal;
        """
    )
}
