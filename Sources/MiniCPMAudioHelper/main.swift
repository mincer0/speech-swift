import Foundation
import MiniCPMAudio
import MLX

private struct Request: Decodable {
    let id: String?
    let op: String
    let featuresBase64: String?
    let frames: Int?
    let prefixExtraFrames: Int?
    let suffixExtraFrames: Int?
    /// Little-endian Float32 PCM at 16 kHz.  `audioBase64` is accepted as a
    /// compatibility alias for clients that call the payload "audio".
    let pcmBase64: String?
    let audioBase64: String?
    let samples: Int?
    let sampleRate: Int?
    let isFinal: Bool?
}

private struct Response: Encodable {
    let id: String?
    let event: String?
    let ok: Bool
    let error: String?
    let embeddingsBase64: String?
    let audioTokenCount: Int?
    let embeddingDimension: Int?
    let cacheLength: Int?
    let cacheResetCount: Int?
    let didResetCache: Bool?
    let latencyMilliseconds: Double?
}

private let encoder: JSONEncoder = {
    let value = JSONEncoder()
    value.outputFormatting = [.sortedKeys]
    return value
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
        embeddingsBase64: nil,
        audioTokenCount: nil,
        embeddingDimension: nil,
        cacheLength: nil,
        cacheResetCount: nil,
        didResetCache: nil,
        latencyMilliseconds: nil)
}

private func floats(fromBase64 string: String, expectedCount: Int) throws -> [Float] {
    guard let data = Data(base64Encoded: string), data.count == expectedCount * 4 else {
        throw MiniCPMAudioError.invalidFeatures(
            "base64 feature payload length does not match its frame count")
    }
    var values = [Float](repeating: 0, count: expectedCount)
    _ = values.withUnsafeMutableBytes { destination in
        data.copyBytes(to: destination)
    }
    return values
}

private func base64(_ values: [Float]) -> String {
    values.withUnsafeBytes { Data($0).base64EncodedString() }
}

private func pcmFloats(
    fromBase64 string: String,
    expectedSamples: Int?
) throws -> [Float] {
    guard let data = Data(base64Encoded: string), data.count.isMultiple(of: 4) else {
        throw MiniCPMAudioError.invalidFeatures(
            "base64 PCM payload must contain whole little-endian Float32 samples")
    }
    let count = data.count / 4
    if let expectedSamples, expectedSamples != count {
        throw MiniCPMAudioError.invalidFeatures(
            "base64 PCM sample count does not match its payload length")
    }
    var values = [Float](repeating: 0, count: count)
    _ = values.withUnsafeMutableBytes { destination in
        data.copyBytes(to: destination)
    }
    guard values.allSatisfy(\.isFinite) else {
        throw MiniCPMAudioError.invalidFeatures(
            "PCM payload contains NaN or infinity")
    }
    return values
}

private func response(
    id: String?,
    event: String,
    model: MiniCPMAudioModel,
    session: MiniCPMAudioStreamingSession,
    encoded: MiniCPMStreamingAudioEncoding? = nil,
    latencyMilliseconds: Double? = nil
) -> Response {
    var embeddingsBase64: String?
    var tokenCount: Int?
    var dimension: Int?
    if let output = encoded?.output {
        eval(output.embeddings)
        let values = output.embeddings.asType(.float32).asArray(Float.self)
        embeddingsBase64 = base64(values)
        tokenCount = output.embeddings.dim(1)
        dimension = output.embeddings.dim(2)
    }
    return Response(
        id: id,
        event: event,
        ok: true,
        error: nil,
        embeddingsBase64: embeddingsBase64,
        audioTokenCount: tokenCount,
        embeddingDimension: dimension ?? model.configuration.projectionSize,
        cacheLength: session.cache?.length ?? 0,
        cacheResetCount: session.cacheResetCount,
        didResetCache: encoded?.didResetCache ?? false,
        latencyMilliseconds: latencyMilliseconds)
}

@main
enum MiniCPMAudioHelperCommand {
    static func main() throws {
        guard CommandLine.arguments.count == 2 else {
            throw MiniCPMAudioError.invalidFeatures(
                "usage: minicpm-audio-helper <audio-model-dir>")
        }
        let model = try MiniCPMAudioModel.fromDirectory(URL(
            fileURLWithPath: CommandLine.arguments[1], isDirectory: true))
        let session = MiniCPMAudioStreamingSession(model: model)
        try write(Response(
            id: nil,
            event: "ready",
            ok: true,
            error: nil,
            embeddingsBase64: nil,
            audioTokenCount: nil,
            embeddingDimension: model.configuration.projectionSize,
            cacheLength: 0,
            cacheResetCount: 0,
            didResetCache: false,
            latencyMilliseconds: nil))

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
                        embeddingsBase64: nil,
                        audioTokenCount: nil,
                        embeddingDimension: model.configuration.projectionSize,
                        cacheLength: session.cache?.length ?? 0,
                        cacheResetCount: session.cacheResetCount,
                        didResetCache: false,
                        latencyMilliseconds: nil))
                case "reset", "interrupt":
                    session.reset()
                    Memory.clearCache()
                    try write(Response(
                        id: request.id,
                        event: request.op,
                        ok: true,
                        error: nil,
                        embeddingsBase64: nil,
                        audioTokenCount: nil,
                        embeddingDimension: model.configuration.projectionSize,
                        cacheLength: 0,
                        cacheResetCount: 0,
                        didResetCache: false,
                        latencyMilliseconds: nil))
                case "accept_audio", "encode_audio", "encode_pcm":
                    let payload = request.pcmBase64 ?? request.audioBase64
                    guard let payload else {
                        throw MiniCPMAudioError.invalidFeatures(
                            "\(request.op) requires pcmBase64 (Float32 PCM at 16 kHz)")
                    }
                    if let sampleRate = request.sampleRate, sampleRate != 16_000 {
                        throw MiniCPMAudioError.invalidFeatures(
                            "MiniCPM audio expects 16000 Hz PCM; resample before calling the helper")
                    }
                    let samples = try pcmFloats(
                        fromBase64: payload,
                        expectedSamples: request.samples)
                    let started = ContinuousClock.now
                    let encodedAudio = try session.acceptAudio(
                        samples,
                        isFinal: request.isFinal ?? false)
                    let duration = started.duration(to: .now)
                    let milliseconds = Double(duration.components.seconds) * 1_000
                        + Double(duration.components.attoseconds) / 1e15
                    try write(response(
                        id: request.id,
                        event: "accept_audio",
                        model: model,
                        session: session,
                        encoded: encodedAudio,
                        latencyMilliseconds: milliseconds))
                case "encode_reference", "encode_offline":
                    let payload = request.pcmBase64 ?? request.audioBase64
                    guard let payload else {
                        throw MiniCPMAudioError.invalidFeatures(
                            "\(request.op) requires pcmBase64 (Float32 PCM at 16 kHz)")
                    }
                    if let sampleRate = request.sampleRate, sampleRate != 16_000 {
                        throw MiniCPMAudioError.invalidFeatures(
                            "MiniCPM audio expects 16000 Hz PCM; resample before calling the helper")
                    }
                    let samples = try pcmFloats(
                        fromBase64: payload,
                        expectedSamples: request.samples)
                    let started = ContinuousClock.now
                    // This path intentionally uses a fresh extractor and no
                    // cache; reference audio can never contaminate the live
                    // streaming session.
                    let encodedAudio = try session.encodeReferenceAudio(samples)
                    let duration = started.duration(to: .now)
                    let milliseconds = Double(duration.components.seconds) * 1_000
                        + Double(duration.components.attoseconds) / 1e15
                    try write(response(
                        id: request.id,
                        event: "encode_reference",
                        model: model,
                        session: session,
                        encoded: encodedAudio,
                        latencyMilliseconds: milliseconds))
                case "encode_features":
                    guard let encoded = request.featuresBase64,
                          let frames = request.frames,
                          frames > 0 else {
                        throw MiniCPMAudioError.invalidFeatures(
                            "encode_features requires a non-empty payload and frame count")
                    }
                    let values = try floats(
                        fromBase64: encoded,
                        expectedCount: frames * model.configuration.melBins)
                    let features = MLXArray(
                        values, [1, frames, model.configuration.melBins])
                    let started = ContinuousClock.now
                    let encodedAudio = try session.encodeFeatures(
                        features,
                        prefixExtraFrames: request.prefixExtraFrames ?? 0,
                        suffixExtraFrames: request.suffixExtraFrames ?? 0)
                    eval(encodedAudio.output.embeddings)
                    let embeddings = encodedAudio.output.embeddings
                        .asType(.float32).asArray(Float.self)
                    let duration = started.duration(to: .now)
                    let milliseconds = Double(duration.components.seconds) * 1_000
                        + Double(duration.components.attoseconds) / 1e15
                    try write(Response(
                        id: request.id,
                        event: "encode_features",
                        ok: true,
                        error: nil,
                        embeddingsBase64: base64(embeddings),
                        audioTokenCount: encodedAudio.output.embeddings.dim(1),
                        embeddingDimension: encodedAudio.output.embeddings.dim(2),
                        cacheLength: session.cache?.length ?? 0,
                        cacheResetCount: session.cacheResetCount,
                        didResetCache: encodedAudio.didResetCache,
                        latencyMilliseconds: milliseconds))
                case "shutdown":
                    session.releaseEncoderContext()
                    try write(Response(
                        id: request.id,
                        event: "shutdown",
                        ok: true,
                        error: nil,
                        embeddingsBase64: nil,
                        audioTokenCount: nil,
                        embeddingDimension: nil,
                        cacheLength: nil,
                        cacheResetCount: nil,
                        didResetCache: nil,
                        latencyMilliseconds: nil))
                    return
                default:
                    throw MiniCPMAudioError.invalidFeatures(
                        "unknown helper operation \(request.op)")
                }
            } catch {
                try write(failure(id: requestID, error))
            }
        }
        session.releaseEncoderContext()
    }
}
