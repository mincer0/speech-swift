import Foundation
import MiniCPMToken2Wav
import MLX

private struct Request: Decodable {
    let id: String?
    let op: String
    let tokens: [Int32]?
    let lastChunk: Bool?
    let forceFlush: Bool?
    let isFinal: Bool?
    // Raw reference PCM for the native prompt front-end.  The samples are
    // little-endian Float32 mono; `audioBase64` is accepted as a compatibility
    // alias for clients that already use that field name.
    let pcmBase64: String?
    let audioBase64: String?
    let samples: Int?
    let sampleRate: Int?
    // Until the ONNX S3Tokenizer/CAM++ weights are converted, callers may
    // inject these two model outputs while the mel front-end remains native.
    let promptTokens: [Int32]?
    let speakerEmbedding: [Float]?
    let speakerEmbeddingBase64: String?
    /// Test-only CAM++ provider probe: a precomputed [1,500,80] Float32 mel
    /// can be sent to the CoreML adapter so ONNX/CoreML parity is measured on
    /// identical tensors rather than on two independently implemented fbank
    /// front-ends.
    let speakerMelBase64: String?
    let includeSpeakerMel: Bool?
}

private struct ChunkResponse: Encodable {
    let audioBase64: String
    let sampleCount: Int
    let inputTokenCount: Int
    let stableTokenCount: Int
    let melFrameCount: Int
    let peakAbsoluteAmplitude: Float?
    let isFinal: Bool
}

private struct Response: Encodable {
    let id: String?
    let event: String?
    let ok: Bool
    let error: String?
    let audioBase64: String?
    let sampleCount: Int?
    let inputTokenCount: Int?
    let stableTokenCount: Int?
    let melFrameCount: Int?
    let peakAbsoluteAmplitude: Float?
    let latencyMilliseconds: Double?
    let context: MiniCPMToken2WavContextStatistics?
    let tokenizerMelFrameCount: Int?
    let promptTokenCount: Int?
    let promptMelFrameCount: Int?
    let speakerMelFrameCount: Int?
    let promptTokenPreview: [Int32]?
    /// First eight values of the native CAM++ vector.  This is intentionally
    /// diagnostic-only and lets the ONNX/CoreML migration harness compare the
    /// provider without writing a second model-specific executable.
    let speakerEmbeddingPreview: [Float]?
    let speakerEmbeddingOutputBase64: String?
    /// Optional 40,000-value CAM++ input tensor for fbank parity diagnostics.
    /// It is omitted from normal requests to keep the JSONL response small.
    let speakerMelBase64: String?
    /// `append` can produce more than one stable window when a large token
    /// batch arrives.  Keep the old top-level fields for one-window clients,
    /// while exposing every generated chunk here.
    let chunks: [ChunkResponse]?
}

private let encoder: JSONEncoder = {
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.sortedKeys]
    return encoder
}()

private func write(_ response: Response) throws {
    var data = try encoder.encode(response)
    data.append(0x0A)
    FileHandle.standardOutput.write(data)
}

private func failure(id: String?, _ error: Error) -> Response {
    Response(
        id: id,
        event: nil,
        ok: false,
        error: error.localizedDescription,
        audioBase64: nil,
        sampleCount: nil,
        inputTokenCount: nil,
        stableTokenCount: nil,
        melFrameCount: nil,
        peakAbsoluteAmplitude: nil,
        latencyMilliseconds: nil,
        context: nil,
        tokenizerMelFrameCount: nil,
        promptTokenCount: nil,
        promptMelFrameCount: nil,
        speakerMelFrameCount: nil,
        promptTokenPreview: nil,
        speakerEmbeddingPreview: nil,
        speakerEmbeddingOutputBase64: nil,
        speakerMelBase64: nil,
        chunks: nil)
}

private func floatPCM16(_ samples: [Float]) -> Data {
    var data = Data(capacity: samples.count * 2)
    for sample in samples {
        let clipped = Swift.max(-1, Swift.min(1, sample))
        var value = Int16(clipped * Float(Int16.max)).littleEndian
        withUnsafeBytes(of: &value) { data.append(contentsOf: $0) }
    }
    return data
}

private func chunkResponse(_ chunk: MiniCPMToken2WavChunk) -> ChunkResponse {
    let samples = chunk.waveform.asArray(Float.self)
    return ChunkResponse(
        audioBase64: floatPCM16(samples).base64EncodedString(),
        sampleCount: samples.count,
        inputTokenCount: chunk.inputTokenCount,
        stableTokenCount: chunk.stableTokenCount,
        melFrameCount: chunk.melFrameCount,
        peakAbsoluteAmplitude: samples.map { Swift.abs($0) }.max(),
        isFinal: chunk.isFinal)
}

private func decodeFloat32PCM(
    _ encoded: String,
    expectedSamples: Int?
) throws -> [Float] {
    guard let data = Data(base64Encoded: encoded) else {
        throw MiniCPMToken2WavError.invalidInput(
            "pcmBase64 must be valid base64")
    }
    guard data.count % MemoryLayout<Float>.size == 0 else {
        throw MiniCPMToken2WavError.invalidInput(
            "PCM byte length must be a multiple of 4")
    }
    let count = data.count / MemoryLayout<Float>.size
    if let expectedSamples, expectedSamples != count {
        throw MiniCPMToken2WavError.invalidInput(
            "samples=\(expectedSamples) does not match PCM payload count \(count)")
    }
    var values = [Float](repeating: 0, count: count)
    data.withUnsafeBytes { bytes in
        for i in 0..<count {
            let offset = i * MemoryLayout<Float>.size
            let raw = bytes.loadUnaligned(fromByteOffset: offset, as: UInt32.self)
            values[i] = Float(bitPattern: UInt32(littleEndian: raw))
        }
    }
    guard !values.isEmpty, values.allSatisfy({ $0.isFinite }) else {
        throw MiniCPMToken2WavError.invalidInput(
            "PCM payload must contain finite Float32 samples")
    }
    return values
}

private func decodeFloat32Vector(_ encoded: String) throws -> [Float] {
    try decodeFloat32PCM(encoded, expectedSamples: nil)
}

private func float32Base64(_ values: [Float]) -> String {
    var data = Data(capacity: values.count * MemoryLayout<Float>.size)
    for value in values {
        var bits = value.bitPattern.littleEndian
        withUnsafeBytes(of: &bits) { data.append(contentsOf: $0) }
    }
    return data.base64EncodedString()
}

private func required(
    _ name: String, from arrays: [String: MLXArray]
) throws -> MLXArray {
    guard let value = arrays[name] else {
        throw MiniCPMToken2WavError.invalidInput(
            "prompt fixture is missing \(name)")
    }
    return value
}

@main
enum MiniCPMToken2WavHelperCommand {
    static func main() throws {
        guard (2 ... 4).contains(CommandLine.arguments.count) else {
            throw MiniCPMToken2WavError.invalidInput(
                "usage: minicpm-token2wav-helper <model-dir> [prompt.safetensors] [ode-steps]")
        }
        let modelDirectory = URL(
            fileURLWithPath: CommandLine.arguments[1], isDirectory: true)
        // Keep the historical `<model> <fixture> [steps]` form, while also
        // allowing `<model>` or `<model> <steps>` for a raw-PCM
        // `prepare_prompt` session.
        let promptURL: URL?
        let odeSteps: Int
        if CommandLine.arguments.count == 2 {
            promptURL = nil
            odeSteps = 10
        } else if CommandLine.arguments.count == 3,
                  let parsed = Int(CommandLine.arguments[2]) {
            promptURL = nil
            odeSteps = parsed
        } else {
            promptURL = URL(fileURLWithPath: CommandLine.arguments[2])
            odeSteps = CommandLine.arguments.count == 4
                ? Int(CommandLine.arguments[3]) ?? 10 : 10
        }
        guard odeSteps > 0 else {
            throw MiniCPMToken2WavError.invalidInput(
                "ode-steps must be positive")
        }

        var flowConfiguration = MiniCPMFlowConfiguration()
        flowConfiguration.odeSteps = odeSteps
        let pipeline = try MiniCPMToken2WavPipeline(
            modelDirectory: modelDirectory,
            flowConfiguration: flowConfiguration)
        if let promptURL {
            let arrays = try MLX.loadArrays(url: promptURL)
            try pipeline.prepare(prompt: MiniCPMToken2WavPrompt(
                tokens: try required("prompt_tokens", from: arrays),
                mel: try required("prompt_mel", from: arrays),
                speakerEmbedding: try required("speaker_embedding", from: arrays)))
        }
        try write(Response(
            id: nil,
            event: "ready",
            ok: true,
            error: nil,
            audioBase64: nil,
            sampleCount: nil,
            inputTokenCount: nil,
            stableTokenCount: nil,
            melFrameCount: nil,
            peakAbsoluteAmplitude: nil,
            latencyMilliseconds: nil,
            context: pipeline.contextStatistics(),
            tokenizerMelFrameCount: nil,
            promptTokenCount: nil,
            promptMelFrameCount: nil,
            speakerMelFrameCount: nil,
            promptTokenPreview: nil,
            speakerEmbeddingPreview: nil,
            speakerEmbeddingOutputBase64: nil,
            speakerMelBase64: nil,
            chunks: nil))

        let promptPreparer = MiniCPMToken2WavPromptPreparer()
        let environment = ProcessInfo.processInfo.environment
        let tokenizerPath = environment["MINICPM_SPEECH_TOKENIZER_WEIGHTS"]
            .map { URL(fileURLWithPath: $0) }
        let camPlusPlusPath = environment["MINICPM_CAMPPLUS_COREML"]
            .map { URL(fileURLWithPath: $0) }
        let nativeEncoder: MiniCPMNativePromptEncoder?
        if tokenizerPath != nil || camPlusPlusPath != nil {
            nativeEncoder = try MiniCPMNativePromptEncoder.discover(
                in: modelDirectory, environment: environment)
        } else if promptURL == nil {
            // A model-only process cannot synthesize a voice from raw PCM.
            // Discover the converted S3Tokenizer/CAM++ sidecars before the
            // ready event so a missing prompt component fails closed instead
            // of producing a ready-but-silent server.  Test harnesses that
            // intentionally inject both arrays can opt out explicitly.
            if environment["MINICPM_ALLOW_INJECTED_PROMPT"] == "1" {
                nativeEncoder = nil
            } else {
                nativeEncoder = try MiniCPMNativePromptEncoder.discover(
                    in: modelDirectory, environment: environment)
            }
        } else {
            nativeEncoder = nil
        }

        while let line = readLine(strippingNewline: true) {
            if line.isEmpty { continue }
            var requestID: String?
            do {
                let request = try JSONDecoder().decode(
                    Request.self, from: Data(line.utf8))
                requestID = request.id
                switch request.op {
                case "ping", "stats":
                    try write(Response(
                        id: request.id,
                        event: request.op,
                        ok: true,
                        error: nil,
                        audioBase64: nil,
                        sampleCount: nil,
                        inputTokenCount: nil,
                        stableTokenCount: nil,
                        melFrameCount: nil,
                        peakAbsoluteAmplitude: nil,
                        latencyMilliseconds: nil,
                        context: pipeline.contextStatistics(),
                        tokenizerMelFrameCount: nil,
                        promptTokenCount: nil,
                        promptMelFrameCount: nil,
                        speakerMelFrameCount: nil,
                        promptTokenPreview: nil,
                        speakerEmbeddingPreview: nil,
                        speakerEmbeddingOutputBase64: nil,
                        speakerMelBase64: nil,
                        chunks: nil))
                case "reset", "interrupt":
                    pipeline.interruptAndReset()
                    try write(Response(
                        id: request.id,
                        event: request.op,
                        ok: true,
                        error: nil,
                        audioBase64: nil,
                        sampleCount: nil,
                        inputTokenCount: nil,
                        stableTokenCount: nil,
                        melFrameCount: nil,
                        peakAbsoluteAmplitude: nil,
                        latencyMilliseconds: nil,
                        context: pipeline.contextStatistics(),
                        tokenizerMelFrameCount: nil,
                        promptTokenCount: nil,
                        promptMelFrameCount: nil,
                        speakerMelFrameCount: nil,
                        promptTokenPreview: nil,
                        speakerEmbeddingPreview: nil,
                        speakerEmbeddingOutputBase64: nil,
                        speakerMelBase64: nil,
                        chunks: nil))
                case "prepare_prompt":
                    guard let encoded = request.pcmBase64 ?? request.audioBase64 else {
                        throw MiniCPMToken2WavError.invalidInput(
                            "prepare_prompt requires pcmBase64 (Float32 mono PCM)")
                    }
                    let referenceRate = request.sampleRate ?? 16_000
                    let pcm = try decodeFloat32PCM(encoded, expectedSamples: request.samples)
                    let injectedTokens = request.promptTokens.map {
                        MLXArray($0, [1, $0.count])
                    }
                    let injectedSpeaker: MLXArray?
                    if let values = request.speakerEmbedding {
                        injectedSpeaker = MLXArray(values, [1, values.count])
                    } else if let encodedSpeaker = request.speakerEmbeddingBase64 {
                        let values = try decodeFloat32Vector(encodedSpeaker)
                        injectedSpeaker = MLXArray(values, [1, values.count])
                    } else if let encodedMel = request.speakerMelBase64 {
                        guard let camPlusPlusPath else {
                            throw MiniCPMToken2WavError.invalidInput(
                                "speakerMelBase64 requires MINICPM_CAMPPLUS_COREML")
                        }
                        let values = try decodeFloat32Vector(encodedMel)
                        guard values.count == 500 * 80 else {
                            throw MiniCPMToken2WavError.invalidInput(
                                "speakerMelBase64 must contain 40000 Float32 values")
                        }
                        let cam = try MiniCPMCoreMLCamPlusPlusEncoder(
                            modelURL: camPlusPlusPath)
                        injectedSpeaker = try cam.encode(
                            speakerMel: MLXArray(values, [1, 500, 80]))
                    } else {
                        injectedSpeaker = nil
                    }
                    let profile = try promptPreparer.prepare(
                        pcm: pcm,
                        sampleRate: referenceRate,
                        encoder: nativeEncoder,
                        promptTokens: injectedTokens,
                        speakerEmbedding: injectedSpeaker)
                    try pipeline.prepare(prompt: profile.prompt)
                    try write(Response(
                        id: request.id,
                        event: "prepare_prompt",
                        ok: true,
                        error: nil,
                        audioBase64: nil,
                        sampleCount: nil,
                        inputTokenCount: nil,
                        stableTokenCount: nil,
                        melFrameCount: nil,
                        peakAbsoluteAmplitude: nil,
                        latencyMilliseconds: nil,
                        context: pipeline.contextStatistics(),
                        tokenizerMelFrameCount: profile.tokenizerMelFrameCount,
                        promptTokenCount: profile.promptTokenCount,
                        promptMelFrameCount: profile.promptMelFrameCount,
                        speakerMelFrameCount: profile.speakerMelFrameCount,
                        promptTokenPreview: profile.prompt.tokens
                            .asType(.int32).reshaped([-1]).asArray(Int32.self)
                            .prefix(8).map { $0 },
                        speakerEmbeddingPreview: profile.prompt.speakerEmbedding
                            .asType(.float32).reshaped([-1]).asArray(Float.self)
                            .prefix(8).map { $0 },
                        speakerEmbeddingOutputBase64: float32Base64(
                            profile.prompt.speakerEmbedding.asType(.float32)
                                .reshaped([-1]).asArray(Float.self)),
                        speakerMelBase64: request.includeSpeakerMel == true
                            ? float32Base64(profile.speakerMel.asType(.float32)
                                .reshaped([-1]).asArray(Float.self)) : nil,
                        chunks: nil))
                case "append":
                    guard let tokens = request.tokens else {
                        throw MiniCPMToken2WavError.invalidInput(
                            "append request is missing tokens")
                    }
                    let started = ContinuousClock.now
                    let isFinal = request.isFinal ?? request.lastChunk ?? false
                    let chunks = try pipeline.append(
                        tokens,
                        forceFlush: request.forceFlush ?? false,
                        isFinal: isFinal)
                    let payloads = chunks.map(chunkResponse)
                    let first = payloads.first
                    let duration = started.duration(to: .now)
                    let milliseconds = Double(duration.components.seconds) * 1_000
                        + Double(duration.components.attoseconds) / 1e15
                    try write(Response(
                        id: request.id,
                        event: "append",
                        ok: true,
                        error: nil,
                        audioBase64: first?.audioBase64,
                        sampleCount: first?.sampleCount,
                        inputTokenCount: first?.inputTokenCount,
                        stableTokenCount: first?.stableTokenCount,
                        melFrameCount: first?.melFrameCount,
                        peakAbsoluteAmplitude: first?.peakAbsoluteAmplitude,
                        latencyMilliseconds: milliseconds,
                        context: pipeline.contextStatistics(),
                        tokenizerMelFrameCount: nil,
                        promptTokenCount: nil,
                        promptMelFrameCount: nil,
                        speakerMelFrameCount: nil,
                        promptTokenPreview: nil,
                        speakerEmbeddingPreview: nil,
                        speakerEmbeddingOutputBase64: nil,
                        speakerMelBase64: nil,
                        chunks: payloads))
                case "stream":
                    guard let tokens = request.tokens else {
                        throw MiniCPMToken2WavError.invalidInput(
                            "stream request is missing tokens")
                    }
                    let started = ContinuousClock.now
                    let chunk = try pipeline.synthesizeWindow(
                        tokens, isFinal: request.lastChunk ?? false)
                    let payload = chunkResponse(chunk)
                    let duration = started.duration(to: .now)
                    let milliseconds = Double(duration.components.seconds) * 1_000
                        + Double(duration.components.attoseconds) / 1e15
                    try write(Response(
                        id: request.id,
                        event: "stream",
                        ok: true,
                        error: nil,
                        audioBase64: payload.audioBase64,
                        sampleCount: payload.sampleCount,
                        inputTokenCount: payload.inputTokenCount,
                        stableTokenCount: payload.stableTokenCount,
                        melFrameCount: payload.melFrameCount,
                        peakAbsoluteAmplitude: payload.peakAbsoluteAmplitude,
                        latencyMilliseconds: milliseconds,
                        context: pipeline.contextStatistics(),
                        tokenizerMelFrameCount: nil,
                        promptTokenCount: nil,
                        promptMelFrameCount: nil,
                        speakerMelFrameCount: nil,
                        promptTokenPreview: nil,
                        speakerEmbeddingPreview: nil,
                        speakerEmbeddingOutputBase64: nil,
                        speakerMelBase64: nil,
                        chunks: [payload]))
                case "shutdown":
                    pipeline.releaseContext()
                    try write(Response(
                        id: request.id,
                        event: "shutdown",
                        ok: true,
                        error: nil,
                        audioBase64: nil,
                        sampleCount: nil,
                        inputTokenCount: nil,
                        stableTokenCount: nil,
                        melFrameCount: nil,
                        peakAbsoluteAmplitude: nil,
                        latencyMilliseconds: nil,
                        context: nil,
                        tokenizerMelFrameCount: nil,
                        promptTokenCount: nil,
                        promptMelFrameCount: nil,
                        speakerMelFrameCount: nil,
                        promptTokenPreview: nil,
                        speakerEmbeddingPreview: nil,
                        speakerEmbeddingOutputBase64: nil,
                        speakerMelBase64: nil,
                        chunks: nil))
                    return
                default:
                    throw MiniCPMToken2WavError.invalidInput(
                        "unknown helper operation \(request.op)")
                }
            } catch {
                try write(failure(id: requestID, error))
            }
        }
        pipeline.releaseContext()
    }
}
