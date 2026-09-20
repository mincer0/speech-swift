import Foundation
import MiniCPMToken2Wav
import MLX

private struct ChunkReport: Encodable {
    let round: Int
    let index: Int
    let inputTokens: Int
    let stableTokens: Int
    let melFrames: Int
    let samples: Int
    let final: Bool
    let latencyMilliseconds: Double
}

private struct SmokeReport: Encodable {
    let odeSteps: Int
    let rounds: Int
    let generatedTokens: Int
    let firstRoundSamples: Int
    let durationSeconds: Double
    let peakAbsoluteAmplitude: Float
    let allFinite: Bool
    let repeatedRoundsMatchLength: Bool
    let repeatedRoundsMatchContent: Bool
    let maximumRepeatedRoundDifference: Float
    let maximumBoundaryJump: Float
    let maximumDecoderPositions: Int
    let maximumConformerOutputPositions: Int
    let outputWAV: String
    let chunks: [ChunkReport]
}

private func appendLittleEndian<T: FixedWidthInteger>(
    _ value: T, to data: inout Data
) {
    var littleEndian = value.littleEndian
    withUnsafeBytes(of: &littleEndian) { data.append(contentsOf: $0) }
}

private func writeWAV(samples: [Float], sampleRate: Int, to url: URL) throws {
    var pcm = Data(capacity: samples.count * 2)
    for sample in samples {
        let clipped = Swift.max(-1, Swift.min(1, sample))
        appendLittleEndian(Int16(clipped * Float(Int16.max)), to: &pcm)
    }
    var data = Data()
    data.append(contentsOf: "RIFF".utf8)
    appendLittleEndian(UInt32(36 + pcm.count), to: &data)
    data.append(contentsOf: "WAVE".utf8)
    data.append(contentsOf: "fmt ".utf8)
    appendLittleEndian(UInt32(16), to: &data)
    appendLittleEndian(UInt16(1), to: &data)
    appendLittleEndian(UInt16(1), to: &data)
    appendLittleEndian(UInt32(sampleRate), to: &data)
    appendLittleEndian(UInt32(sampleRate * 2), to: &data)
    appendLittleEndian(UInt16(2), to: &data)
    appendLittleEndian(UInt16(16), to: &data)
    data.append(contentsOf: "data".utf8)
    appendLittleEndian(UInt32(pcm.count), to: &data)
    data.append(pcm)
    try FileManager.default.createDirectory(
        at: url.deletingLastPathComponent(),
        withIntermediateDirectories: true)
    try data.write(to: url, options: .atomic)
}

private func required(
    _ name: String, from arrays: [String: MLXArray]
) throws -> MLXArray {
    guard let value = arrays[name] else {
        throw MiniCPMToken2WavError.invalidInput(
            "smoke fixture is missing \(name)")
    }
    return value
}

private func tokenValues(_ array: MLXArray) -> [Int32] {
    array.asType(.int32).reshaped([-1]).asArray(Int32.self)
}

@main
enum MiniCPMToken2WavSmokeCommand {
    static func main() throws {
        guard (4 ... 6).contains(CommandLine.arguments.count) else {
            FileHandle.standardError.write(Data(
                "usage: minicpm-token2wav-smoke <model-dir> <fixture.safetensors> <output.wav> [ode-steps] [rounds]\n".utf8))
            throw MiniCPMToken2WavError.invalidInput("invalid smoke arguments")
        }
        let modelDirectory = URL(
            fileURLWithPath: CommandLine.arguments[1], isDirectory: true)
        let fixture = try MLX.loadArrays(
            url: URL(fileURLWithPath: CommandLine.arguments[2]))
        let outputURL = URL(fileURLWithPath: CommandLine.arguments[3])
        let odeSteps = CommandLine.arguments.count >= 5
            ? Int(CommandLine.arguments[4]) ?? 10 : 10
        let rounds = CommandLine.arguments.count >= 6
            ? Int(CommandLine.arguments[5]) ?? 2 : 2
        guard odeSteps > 0, rounds > 0 else {
            throw MiniCPMToken2WavError.invalidInput(
                "ode-steps and rounds must be positive")
        }

        var flowConfiguration = MiniCPMFlowConfiguration()
        flowConfiguration.odeSteps = odeSteps
        let pipeline = try MiniCPMToken2WavPipeline(
            modelDirectory: modelDirectory,
            flowConfiguration: flowConfiguration)
        let prompt = MiniCPMToken2WavPrompt(
            tokens: try required("prompt_tokens", from: fixture),
            mel: try required("prompt_mel", from: fixture),
            speakerEmbedding: try required("speaker_embedding", from: fixture))
        let generated = tokenValues(
            try required("generated_tokens", from: fixture))
        guard !generated.isEmpty else {
            throw MiniCPMToken2WavError.invalidInput(
                "smoke fixture contains no generated tokens")
        }

        let prepareStart = ContinuousClock.now
        try pipeline.prepare(prompt: prompt)
        let prepareDuration = prepareStart.duration(to: .now)
        FileHandle.standardError.write(Data(
            "prepared prompt in \(prepareDuration)\n".utf8))

        var reports: [ChunkReport] = []
        var sampleCounts: [Int] = []
        var roundWaveforms: [[Float]] = []
        var firstRoundChunkSamples: [Int] = []
        var firstRoundSamples: [Float] = []
        var maximumDecoderPositions = 0
        var maximumConformerOutputPositions = 0

        for round in 0 ..< rounds {
            if round > 0 { pipeline.resetForNewTurn() }
            var emitted: [Float] = []
            var offset = 0
            let feedPattern = [7, 13, 5, 19, 11]
            var feedIndex = 0
            var chunkIndex = 0
            while offset < generated.count {
                let count = Swift.min(
                    feedPattern[feedIndex % feedPattern.count],
                    generated.count - offset)
                let end = offset + count
                let isFinal = end == generated.count
                let start = ContinuousClock.now
                let chunks = try pipeline.append(
                    Array(generated[offset ..< end]),
                    forceFlush: feedIndex % 3 == 2 && !isFinal,
                    isFinal: isFinal)
                let duration = start.duration(to: .now)
                let totalMilliseconds = duration.components.seconds * 1_000
                    + Int64(duration.components.attoseconds / 1_000_000_000_000_000)
                let perChunk = chunks.isEmpty
                    ? 0 : Double(totalMilliseconds) / Double(chunks.count)
                for chunk in chunks {
                    let samples = chunk.waveform.asArray(Float.self)
                    emitted.append(contentsOf: samples)
                    if round == 0 {
                        firstRoundChunkSamples.append(samples.count)
                    }
                    reports.append(ChunkReport(
                        round: round,
                        index: chunkIndex,
                        inputTokens: chunk.inputTokenCount,
                        stableTokens: chunk.stableTokenCount,
                        melFrames: chunk.melFrameCount,
                        samples: samples.count,
                        final: chunk.isFinal,
                        latencyMilliseconds: perChunk))
                    chunkIndex += 1
                }
                if let statistics = pipeline.contextStatistics() {
                    maximumDecoderPositions = Swift.max(
                        maximumDecoderPositions, statistics.decoderPositions)
                    maximumConformerOutputPositions = Swift.max(
                        maximumConformerOutputPositions,
                        statistics.conformerOutputPositions)
                }
                offset = end
                feedIndex += 1
            }
            sampleCounts.append(emitted.count)
            if round == 0 { firstRoundSamples = emitted }
            roundWaveforms.append(emitted)
        }

        guard !firstRoundSamples.isEmpty else {
            throw MiniCPMToken2WavError.invalidInput(
                "native pipeline emitted no waveform")
        }
        try writeWAV(samples: firstRoundSamples, sampleRate: 24_000, to: outputURL)
        let finite = firstRoundSamples.allSatisfy(\.isFinite)
        let peak = firstRoundSamples.map { Swift.abs($0) }.max() ?? 0
        var maximumRepeatedRoundDifference: Float = 0
        for waveform in roundWaveforms.dropFirst() {
            guard waveform.count == firstRoundSamples.count else {
                maximumRepeatedRoundDifference = .greatestFiniteMagnitude
                continue
            }
            for (lhs, rhs) in zip(firstRoundSamples, waveform) {
                maximumRepeatedRoundDifference = Swift.max(
                    maximumRepeatedRoundDifference, Swift.abs(lhs - rhs))
            }
        }
        let repeatedRoundsMatchContent = maximumRepeatedRoundDifference <= 1e-4
        var maximumBoundaryJump: Float = 0
        var boundary = 0
        for sampleCount in firstRoundChunkSamples.dropLast() {
            boundary += sampleCount
            guard boundary > 0, boundary < firstRoundSamples.count else { continue }
            maximumBoundaryJump = Swift.max(
                maximumBoundaryJump,
                Swift.abs(firstRoundSamples[boundary] - firstRoundSamples[boundary - 1]))
        }
        let report = SmokeReport(
            odeSteps: odeSteps,
            rounds: rounds,
            generatedTokens: generated.count,
            firstRoundSamples: firstRoundSamples.count,
            durationSeconds: Double(firstRoundSamples.count) / 24_000,
            peakAbsoluteAmplitude: peak,
            allFinite: finite,
            repeatedRoundsMatchLength: Set(sampleCounts).count == 1,
            repeatedRoundsMatchContent: repeatedRoundsMatchContent,
            maximumRepeatedRoundDifference: maximumRepeatedRoundDifference,
            maximumBoundaryJump: maximumBoundaryJump,
            maximumDecoderPositions: maximumDecoderPositions,
            maximumConformerOutputPositions: maximumConformerOutputPositions,
            outputWAV: outputURL.path,
            chunks: reports)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        FileHandle.standardOutput.write(try encoder.encode(report))
        FileHandle.standardOutput.write(Data("\n".utf8))

        guard finite, peak > 0, Set(sampleCounts).count == 1,
              repeatedRoundsMatchContent else {
            throw MiniCPMToken2WavError.invalidInput(
                "native token2wav smoke validation failed")
        }
    }
}
