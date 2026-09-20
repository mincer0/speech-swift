import Foundation

#if canImport(AVFoundation)
import AVFoundation
#endif

/// Canonical audio boundary used by the MiniCPM-o Demo APIs.
///
/// The browser protocol sends audio as little-endian Float32 samples at 16 kHz
/// mono.  Files uploaded through the asset and preset APIs can be WAV, MP3,
/// M4A or another format understood by AVFoundation; those files are decoded
/// and converted once at the HTTP boundary so model code never has to guess
/// whether a base64 value contains a WAV container or raw samples.
public struct MiniCPMDemoNormalizedAudio: Sendable, Equatable {
    public let samples: [Float]
    public let sampleRate: Int

    public init(samples: [Float], sampleRate: Int = 16_000) {
        self.samples = samples
        self.sampleRate = sampleRate
    }

    public var durationSeconds: Double {
        guard sampleRate > 0 else { return 0 }
        return Double(samples.count) / Double(sampleRate)
    }

    /// Raw little-endian Float32 PCM for the MiniCPM-o wire protocol.
    public var float32Data: Data {
        var data = Data(capacity: samples.count * MemoryLayout<Float>.size)
        for sample in samples {
            let value = sample.isFinite ? sample : 0
            var bits = value.bitPattern.littleEndian
            withUnsafeBytes(of: &bits) { data.append(contentsOf: $0) }
        }
        return data
    }
}

public enum MiniCPMDemoAudioNormalizer {
    public static let targetSampleRate = 16_000

    /// Decode an audio file and return mono 16 kHz Float32 samples.
    public static func normalize(
        data: Data,
        mimeType: String? = nil,
        fileExtension: String? = nil
    ) throws -> MiniCPMDemoNormalizedAudio {
        guard !data.isEmpty else {
            throw MiniCPMDemoServiceError.invalidMedia("empty audio")
        }

        if let wav = try? decodeWAV(data) {
            return MiniCPMDemoNormalizedAudio(
                samples: resample(wav.samples, from: wav.sampleRate, to: targetSampleRate),
                sampleRate: targetSampleRate)
        }

        #if canImport(AVFoundation)
        return try decodeWithAVFoundation(data: data, mimeType: mimeType, fileExtension: fileExtension)
        #else
        _ = mimeType
        _ = fileExtension
        throw MiniCPMDemoServiceError.invalidMedia(
            "unsupported audio container (AVFoundation is unavailable)")
        #endif
    }

    /// Build the PCM16 WAV representation used by session recordings and
    /// uploaded reference-audio files.  This intentionally has no AVFoundation
    /// dependency so recordings remain valid in command-line/test builds.
    public static func wavData(samples: [Float], sampleRate: Int) -> Data {
        let rate = max(1, sampleRate)
        var pcm = Data(capacity: samples.count * 2)
        for raw in samples {
            let value = raw.isFinite ? max(-1, min(1, raw)) : 0
            let scaled = value < 0 ? value * 32768 : value * 32767
            var sample = Int16(max(-32768, min(32767, Int(scaled.rounded())))).littleEndian
            withUnsafeBytes(of: &sample) { pcm.append(contentsOf: $0) }
        }

        var result = Data(capacity: 44 + pcm.count)
        result.append(contentsOf: "RIFF".utf8)
        appendUInt32LE(&result, UInt32(min(UInt64(UInt32.max), UInt64(36 + pcm.count))))
        result.append(contentsOf: "WAVE".utf8)
        result.append(contentsOf: "fmt ".utf8)
        appendUInt32LE(&result, 16)
        appendUInt16LE(&result, 1) // PCM
        appendUInt16LE(&result, 1) // mono
        appendUInt32LE(&result, UInt32(rate))
        appendUInt32LE(&result, UInt32(rate * 2))
        appendUInt16LE(&result, 2)
        appendUInt16LE(&result, 16)
        result.append(contentsOf: "data".utf8)
        appendUInt32LE(&result, UInt32(min(UInt64(UInt32.max), UInt64(pcm.count))))
        result.append(pcm)
        return result
    }

    /// Decode a raw Float32 payload used by a few old recorder call sites and
    /// wrap it as PCM16 WAV. Returns nil for non-Float32 data.
    public static func wavData(
        fromRawFloat32 data: Data,
        sampleRate: Int
    ) -> Data? {
        guard data.count > 0, data.count % MemoryLayout<Float>.size == 0 else { return nil }
        let count = data.count / MemoryLayout<Float>.size
        let samples: [Float] = data.withUnsafeBytes { raw in
            guard let base = raw.bindMemory(to: Float.self).baseAddress else { return [] }
            return (0..<count).map { base[$0] }
        }
        guard samples.contains(where: { $0.isFinite }) else { return nil }
        return wavData(samples: samples, sampleRate: sampleRate)
    }

    public static func samples(fromRawFloat32 data: Data) -> [Float]? {
        guard data.count > 0, data.count % MemoryLayout<Float>.size == 0 else { return nil }
        let count = data.count / MemoryLayout<Float>.size
        return data.withUnsafeBytes { raw in
            guard let base = raw.bindMemory(to: Float.self).baseAddress else { return nil }
            let result = (0..<count).map { base[$0] }
            return result.allSatisfy(\.isFinite) ? result : nil
        }
    }

    private struct DecodedWAV {
        let samples: [Float]
        let sampleRate: Int
    }

    private static func decodeWAV(_ data: Data) throws -> DecodedWAV {
        guard data.count >= 12,
              data.subdata(in: 0..<4) == Data("RIFF".utf8),
              data.subdata(in: 8..<12) == Data("WAVE".utf8) else {
            throw MiniCPMDemoServiceError.invalidMedia("not a RIFF/WAVE file")
        }

        var offset = 12
        var format: UInt16?
        var channels: Int?
        var sampleRate: Int?
        var bitsPerSample: Int?
        var audioData: Data?

        while offset + 8 <= data.count {
            let chunkID = data.subdata(in: offset..<(offset + 4))
            let chunkSize = Int(readUInt32LE(data, offset + 4))
            offset += 8
            guard chunkSize >= 0, offset + chunkSize <= data.count else {
                throw MiniCPMDemoServiceError.invalidMedia("truncated WAV chunk")
            }
            if chunkID == Data("fmt ".utf8), chunkSize >= 16 {
                format = readUInt16LE(data, offset)
                channels = Int(readUInt16LE(data, offset + 2))
                sampleRate = Int(readUInt32LE(data, offset + 4))
                bitsPerSample = Int(readUInt16LE(data, offset + 14))
            } else if chunkID == Data("data".utf8) {
                audioData = data.subdata(in: offset..<(offset + chunkSize))
            }
            offset += chunkSize + (chunkSize % 2)
        }

        guard let format, let channels, let sampleRate, let bitsPerSample,
              channels > 0, sampleRate > 0, let audioData,
              format == 1 || format == 3 else {
            throw MiniCPMDemoServiceError.invalidMedia("unsupported WAV format")
        }

        let bytesPerSample = (bitsPerSample + 7) / 8
        let frameBytes = channels * bytesPerSample
        guard frameBytes > 0, audioData.count >= frameBytes else {
            throw MiniCPMDemoServiceError.invalidMedia("WAV contains no samples")
        }
        let frameCount = audioData.count / frameBytes
        var samples = [Float](repeating: 0, count: frameCount)
        audioData.withUnsafeBytes { raw in
            let bytes = raw.bindMemory(to: UInt8.self)
            for frame in 0..<frameCount {
                var sum: Float = 0
                for channel in 0..<channels {
                    let start = frame * frameBytes + channel * bytesPerSample
                    sum += decodeSample(
                        bytes: bytes, offset: start, format: format, bits: bitsPerSample)
                }
                samples[frame] = sum / Float(channels)
            }
        }
        return DecodedWAV(samples: samples, sampleRate: sampleRate)
    }

    private static func decodeSample(
        bytes: UnsafeBufferPointer<UInt8>,
        offset: Int,
        format: UInt16,
        bits: Int
    ) -> Float {
        guard offset >= 0, offset + ((bits + 7) / 8) <= bytes.count else { return 0 }
        switch (format, bits) {
        case (3, 32):
            var bitsValue: UInt32 = 0
            for i in 0..<4 { bitsValue |= UInt32(bytes[offset + i]) << UInt32(8 * i) }
            return Float(bitPattern: bitsValue).isFinite ? Float(bitPattern: bitsValue) : 0
        case (1, 8):
            return (Float(bytes[offset]) - 128) / 128
        case (1, 16):
            let raw = Int16(bitPattern: UInt16(bytes[offset]) | UInt16(bytes[offset + 1]) << 8)
            return Float(raw) / 32768
        case (1, 24):
            var raw = Int32(bytes[offset]) | Int32(bytes[offset + 1]) << 8 | Int32(bytes[offset + 2]) << 16
            if raw & 0x800000 != 0 { raw |= ~0xFFFFFF }
            return Float(raw) / 8_388_608
        case (1, 32):
            let raw = Int32(bitPattern: UInt32(bytes[offset]) | UInt32(bytes[offset + 1]) << 8 | UInt32(bytes[offset + 2]) << 16 | UInt32(bytes[offset + 3]) << 24)
            return Float(raw) / 2_147_483_648
        default:
            return 0
        }
    }

    private static func resample(_ input: [Float], from sourceRate: Int, to targetRate: Int) -> [Float] {
        guard !input.isEmpty, sourceRate > 0, targetRate > 0, sourceRate != targetRate else {
            return input
        }
        let outputCount = max(1, Int((Double(input.count) * Double(targetRate) / Double(sourceRate)).rounded()))
        var output = [Float](repeating: 0, count: outputCount)
        let ratio = Double(sourceRate) / Double(targetRate)
        for index in 0..<outputCount {
            let position = Double(index) * ratio
            let lower = min(input.count - 1, Int(position))
            let upper = min(input.count - 1, lower + 1)
            let fraction = Float(position - Double(lower))
            output[index] = input[lower] + (input[upper] - input[lower]) * fraction
        }
        return output
    }

    #if canImport(AVFoundation)
    private static func decodeWithAVFoundation(
        data: Data,
        mimeType: String?,
        fileExtension: String?
    ) throws -> MiniCPMDemoNormalizedAudio {
        let ext = (fileExtension ?? mimeTypeToExtension(mimeType)).trimmingCharacters(in: CharacterSet(charactersIn: ". ")).lowercased()
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("minicpm-demo-audio-\(UUID().uuidString).\(ext.isEmpty ? "audio" : ext)")
        defer { try? FileManager.default.removeItem(at: url) }
        try data.write(to: url, options: [.atomic])

        let input = try AVAudioFile(forReading: url)
        let inputFormat = input.processingFormat
        guard let target = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: Double(targetSampleRate),
            channels: 1,
            interleaved: false),
            let converter = AVAudioConverter(from: inputFormat, to: target) else {
            throw MiniCPMDemoServiceError.invalidMedia("cannot create audio converter")
        }
        let inputFrames = AVAudioFrameCount(input.length)
        guard let sourceBuffer = AVAudioPCMBuffer(
            pcmFormat: inputFormat, frameCapacity: inputFrames) else {
            throw MiniCPMDemoServiceError.invalidMedia("cannot allocate source audio buffer")
        }
        try input.read(into: sourceBuffer)
        let capacity = AVAudioFrameCount(ceil(Double(sourceBuffer.frameLength) * Double(targetSampleRate) / inputFormat.sampleRate)) + 2_048
        guard let targetBuffer = AVAudioPCMBuffer(pcmFormat: target, frameCapacity: max(1, capacity)) else {
            throw MiniCPMDemoServiceError.invalidMedia("cannot allocate target audio buffer")
        }
        var conversionError: NSError?
        var supplied = false
        let status = converter.convert(to: targetBuffer, error: &conversionError) { _, outStatus in
            if supplied {
                outStatus.pointee = .endOfStream
                return nil
            }
            supplied = true
            outStatus.pointee = .haveData
            return sourceBuffer
        }
        guard status == .haveData || status == .inputRanDry || status == .endOfStream,
              conversionError == nil,
              let channel = targetBuffer.floatChannelData?[0] else {
            throw MiniCPMDemoServiceError.invalidMedia(
                "audio conversion failed: \(conversionError?.localizedDescription ?? "unknown error")")
        }
        let samples = Array(UnsafeBufferPointer(start: channel, count: Int(targetBuffer.frameLength)))
        guard !samples.isEmpty else {
            throw MiniCPMDemoServiceError.invalidMedia("audio contains no samples")
        }
        return MiniCPMDemoNormalizedAudio(samples: samples, sampleRate: targetSampleRate)
    }

    private static func mimeTypeToExtension(_ mimeType: String?) -> String {
        switch mimeType?.lowercased().split(separator: ";", maxSplits: 1).first.map(String.init) {
        case "audio/mpeg": return "mp3"
        case "audio/mp4": return "m4a"
        case "audio/webm": return "webm"
        case "audio/wav": return "wav"
        default: return "audio"
        }
    }
    #endif

    private static func readUInt16LE(_ data: Data, _ offset: Int) -> UInt16 {
        UInt16(data[offset]) | UInt16(data[offset + 1]) << 8
    }

    private static func readUInt32LE(_ data: Data, _ offset: Int) -> UInt32 {
        UInt32(data[offset]) | UInt32(data[offset + 1]) << 8
            | UInt32(data[offset + 2]) << 16 | UInt32(data[offset + 3]) << 24
    }

    private static func appendUInt16LE(_ data: inout Data, _ value: UInt16) {
        var little = value.littleEndian
        withUnsafeBytes(of: &little) { data.append(contentsOf: $0) }
    }

    private static func appendUInt32LE(_ data: inout Data, _ value: UInt32) {
        var little = value.littleEndian
        withUnsafeBytes(of: &little) { data.append(contentsOf: $0) }
    }
}
