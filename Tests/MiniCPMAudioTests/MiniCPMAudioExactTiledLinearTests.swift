import Foundation
import MLX
import MLXFast
@testable import MiniCPMAudio
import XCTest

final class E2EMiniCPMAudioExactTiledLinearTests: XCTestCase {
    private func array(_ shape: [Int], phase: Int, dtype: DType) -> MLXArray {
        let values = (0..<shape.reduce(1, *)).map { index in
            Float((index * 17 + phase) % 97 - 48) / 113
        }
        return MLXArray(values, shape).asType(dtype)
    }

    private func assertMatchesSequential(
        input: MLXArray,
        weight: MLXArray,
        bias: MLXArray?,
        file: StaticString = #filePath,
        line: UInt = #line
    ) throws {
        let expected = sequential(input: input, weight: weight, bias: bias)
        let actual = try XCTUnwrap(MiniCPMAudioExactTiledLinear.apply(
            input: input, weight: weight, bias: bias), file: file, line: line)
        eval(actual, expected)
        let context = "input=\(input.shape), weight=\(weight.shape), dtype=\(input.dtype), bias=\(bias != nil)"
        XCTAssertEqual(actual.shape, expected.shape, context, file: file, line: line)
        XCTAssertEqual(actual.dtype, input.dtype, context, file: file, line: line)
        XCTAssertEqual(
            actual.asData(access: .copy).data,
            expected.asData(access: .copy).data,
            context, file: file, line: line)
    }

    private func assertPartialTilesMatch(dtype: DType) throws {
        for (batch, sequenceLength, inputChannels, outputChannels) in [
            (1, 1, 1, 1),
            (1, 7, 31, 17),
            (2, 9, 33, 35),
            (1, 17, 65, 67),
            (2, 3, 97, 129),
        ] {
            let input = array([batch, sequenceLength, inputChannels], phase: 3, dtype: dtype)
            let weight = array([outputChannels, inputChannels], phase: 31, dtype: dtype)
            for hasBias in [false, true] {
                let bias = hasBias ? array([outputChannels], phase: 53, dtype: dtype) : nil
                try assertMatchesSequential(input: input, weight: weight, bias: bias)
            }
        }
    }

    func testBF16PartialTilesMatchSequentialBitwise() throws {
        try assertPartialTilesMatch(dtype: .bfloat16)
    }

    func testFloat32PartialTilesMatchSequentialBitwise() throws {
        try assertPartialTilesMatch(dtype: .float32)
    }

    func testBF16EncoderProjectionWidthsMatchSequentialBitwise() throws {
        // Real encoder widths and the 50-position streaming chunk exercise
        // many channel tiles and an incomplete final sequence group.
        for (inputChannels, outputChannels) in [
            (1_024, 1_024), (1_024, 4_096), (4_096, 1_024),
        ] {
            let input = array([1, 50, inputChannels], phase: 11, dtype: .bfloat16)
            let weight = array([outputChannels, inputChannels], phase: 37, dtype: .bfloat16)
            let bias = array([outputChannels], phase: 71, dtype: .bfloat16)
            try assertMatchesSequential(input: input, weight: weight, bias: bias)
            if inputChannels == outputChannels {
                try assertMatchesSequential(input: input, weight: weight, bias: nil)
            }
        }
    }

    func testNoncontiguousFeedsMatchSequentialBitwise() throws {
        for dtype: DType in [.bfloat16, .float32] {
            let input = array([2, 3, 65], phase: 13, dtype: dtype).transposed(1, 0, 2)
            let weight = array([65, 35], phase: 43, dtype: dtype).transposed()
            let bias = asStrided(array([70], phase: 73, dtype: dtype), [35], strides: [2])
            XCTAssertNotEqual(input.asData(access: .noCopy).strides, [130, 65, 1])
            XCTAssertNotEqual(weight.asData(access: .noCopy).strides, [65, 1])
            XCTAssertEqual(bias.asData(access: .noCopy).strides, [2])
            try assertMatchesSequential(input: input, weight: weight, bias: bias)
        }
    }

    func testUnsupportedDtypesAndShapesReturnNil() {
        let input = MLXArray.zeros([1, 2, 3], dtype: .bfloat16)
        let weight = MLXArray.zeros([5, 3], dtype: .bfloat16)
        XCTAssertNil(MiniCPMAudioExactTiledLinear.apply(
            input: input.asType(.float16), weight: weight.asType(.float16), bias: nil))
        XCTAssertNil(MiniCPMAudioExactTiledLinear.apply(
            input: input, weight: weight.asType(.float32), bias: nil))
        XCTAssertNil(MiniCPMAudioExactTiledLinear.apply(
            input: input.reshaped(2, 3), weight: weight, bias: nil))
        XCTAssertNil(MiniCPMAudioExactTiledLinear.apply(
            input: input, weight: weight.reshaped(1, 5, 3), bias: nil))
        XCTAssertNil(MiniCPMAudioExactTiledLinear.apply(
            input: input, weight: MLXArray.zeros([5, 4], dtype: .bfloat16), bias: nil))
        XCTAssertNil(MiniCPMAudioExactTiledLinear.apply(
            input: input, weight: weight, bias: MLXArray.zeros([1, 5], dtype: .bfloat16)))
        XCTAssertNil(MiniCPMAudioExactTiledLinear.apply(
            input: input, weight: weight, bias: MLXArray.zeros([4], dtype: .bfloat16)))
        XCTAssertNil(MiniCPMAudioExactTiledLinear.apply(
            input: input, weight: weight, bias: MLXArray.zeros([5], dtype: .float32)))
        XCTAssertNil(MiniCPMAudioExactTiledLinear.apply(
            input: MLXArray.zeros([1, 0, 3], dtype: .bfloat16), weight: weight, bias: nil))
    }

    func testOptInBoundedEncoderProjectionBenchmark() throws {
        guard ProcessInfo.processInfo.environment["MINICPM_AUDIO_TILED_LINEAR_BENCHMARK"] == "1" else {
            throw XCTSkip("set MINICPM_AUDIO_TILED_LINEAR_BENCHMARK=1 for the bounded kernel benchmark")
        }
        let warmupCount = 2
        let repetitionCount = 5
        for (inputChannels, outputChannels) in [
            (1_024, 1_024), (1_024, 4_096), (4_096, 1_024),
        ] {
            let input = array([1, 50, inputChannels], phase: 11, dtype: .bfloat16)
            let weight = array([outputChannels, inputChannels], phase: 37, dtype: .bfloat16)
            let bias = array([outputChannels], phase: 71, dtype: .bfloat16)
            // Exclude feed creation from both paths. Each measured iteration
            // creates a fresh output graph and waits for its actual completion.
            eval(input, weight, bias)
            let scalar = { self.sequential(input: input, weight: weight, bias: bias) }
            let tiled = {
                try XCTUnwrap(MiniCPMAudioExactTiledLinear.apply(
                    input: input, weight: weight, bias: bias))
            }
            for _ in 0..<warmupCount {
                let reference = scalar()
                eval(reference)
                let candidate = try tiled()
                eval(candidate)
                XCTAssertEqual(
                    candidate.asData(access: .copy).data,
                    reference.asData(access: .copy).data)
            }

            var scalarMS: [Double] = []
            var tiledMS: [Double] = []
            for repetition in 0..<repetitionCount {
                // Alternate order to reduce consistent first-run bias.
                if repetition.isMultiple(of: 2) {
                    scalarMS.append(benchmarkElapsedMS(scalar))
                    tiledMS.append(try benchmarkElapsedMS(tiled))
                } else {
                    tiledMS.append(try benchmarkElapsedMS(tiled))
                    scalarMS.append(benchmarkElapsedMS(scalar))
                }
            }
            let scalarMedian = scalarMS.sorted()[repetitionCount / 2]
            let tiledMedian = tiledMS.sorted()[repetitionCount / 2]
            print("[audio-linear-benchmark] shape=[1,50,\(inputChannels)] out=\(outputChannels) "
                + "dtype=bf16 warmup=\(warmupCount) repeats=\(repetitionCount) "
                + "scalar_median_ms=\(String(format: "%.6f", scalarMedian)) "
                + "tiled_median_ms=\(String(format: "%.6f", tiledMedian)) "
                + "scalar_samples_ms=\(scalarMS) tiled_samples_ms=\(tiledMS)")
        }
    }

    private func benchmarkElapsedMS(_ operation: () throws -> MLXArray) rethrows -> Double {
        let started = DispatchTime.now().uptimeNanoseconds
        let output = try operation()
        eval(output)
        return Double(DispatchTime.now().uptimeNanoseconds - started) / 1_000_000
    }

    private func sequential(input: MLXArray, weight: MLXArray, bias: MLXArray?) -> MLXArray {
        let x = input.contiguous()
        let w = weight.contiguous()
        let outputChannels = w.dim(0)
        let b = (bias ?? MLXArray.zeros([outputChannels], dtype: input.dtype)).contiguous()
        return Self.sequentialKernel(
            [x, w, b, Int32(x.dim(1)), Int32(w.dim(1)), Int32(outputChannels)],
            template: [("T", input.dtype)],
            grid: (outputChannels, x.dim(1), x.dim(0)),
            threadGroup: (1, min(32, x.dim(1)), 1),
            outputShapes: [[x.dim(0), x.dim(1), outputChannels]],
            outputDTypes: [input.dtype]
        )[0]
    }

    // Preserve the original production sequential kernel independently of
    // environment switches and of the candidate's shared-memory loading.
    private static let sequentialKernel = MLXFast.metalKernel(
        name: "minicpm_audio_linear_mps_sequential_test_reference",
        inputNames: ["x", "weight", "bias", "sequence_length", "input_channels", "output_channels"],
        outputNames: ["out"],
        source: """
            uint outputChannel = thread_position_in_grid.x;
            uint sequence = thread_position_in_grid.y;
            uint batch = thread_position_in_grid.z;
            uint sequenceLength = uint(sequence_length);
            uint inputChannels = uint(input_channels);
            uint outputChannels = uint(output_channels);
            if (outputChannel >= outputChannels || sequence >= sequenceLength) {
                return;
            }

            uint inputOffset = (batch * sequenceLength + sequence) * inputChannels;
            uint weightOffset = outputChannel * inputChannels;
            float accumulator = 0.0f;
            for (uint channel = 0; channel < inputChannels; channel++) {
                accumulator = metal::fma(
                    float(x[inputOffset + channel]),
                    float(weight[weightOffset + channel]),
                    accumulator);
            }
            uint outputOffset = (batch * sequenceLength + sequence) * outputChannels
                + outputChannel;
            accumulator += float(bias[outputChannel]);
            out[outputOffset] = T(accumulator);
        """,
        header: """
            #include <metal_stdlib>
            using namespace metal;
        """
    )
}
