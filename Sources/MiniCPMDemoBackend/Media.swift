import Foundation

#if canImport(AVFoundation)
import AVFoundation
import CoreMedia
import ImageIO
import UniformTypeIdentifiers
#endif

/// Decoded media carried by one turn-based message content item. Video
/// containers are never passed to the vision processor directly: MiniCPM's
/// image processor expects one encoded image per frame.
public struct MiniCPMDemoMediaPayload: Sendable, Equatable {
    public var frames: [Data]
    public var audio: [Float]?
    public var audioSampleRate: Int?

    public init(frames: [Data] = [], audio: [Float]? = nil, audioSampleRate: Int? = nil) {
        self.frames = frames
        self.audio = audio
        self.audioSampleRate = audioSampleRate
    }
}

public enum MiniCPMDemoMediaError: Error, LocalizedError, Sendable {
    case unsupportedPlatform
    case invalidContainer(String)
    case noVideoTrack
    case frameDecodeFailed(String)
    case audioDecodeFailed(String)

    public var errorDescription: String? {
        switch self {
        case .unsupportedPlatform:
            return "MP4 extraction requires AVFoundation"
        case .invalidContainer(let message):
            return "invalid video container: \(message)"
        case .noVideoTrack:
            return "video container has no video track"
        case .frameDecodeFailed(let message):
            return "video frame extraction failed: \(message)"
        case .audioDecodeFailed(let message):
            return "video audio extraction failed: \(message)"
        }
    }
}

/// Small, synchronous AVFoundation bridge used only at the turn-based
/// transport boundary. The model hot path still receives ordinary JPEG/Data
/// frames and Float32 samples, so no AVAsset object crosses into MLX actors.
public enum MiniCPMDemoMediaDecoder {
    #if canImport(AVFoundation)
    /// Decode an MP4/MOV payload. With stackFrames=false one representative
    /// frame is returned; with it enabled frames are sampled at the source
    /// frame rate (bounded to avoid a long movie becoming an unbounded prompt).
    public static func decodeVideo(
        _ data: Data,
        stackFrames: Bool = true,
        maxFrames: Int = 32,
        audioSampleRate: Int = 16_000
    ) throws -> MiniCPMDemoMediaPayload {
        guard !data.isEmpty else {
            throw MiniCPMDemoMediaError.invalidContainer("empty payload")
        }
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("minicpm-video-\(UUID().uuidString).mp4")
        do {
            try data.write(to: url, options: .atomic)
        } catch {
            throw MiniCPMDemoMediaError.invalidContainer("cannot stage payload: \(error.localizedDescription)")
        }
        defer { try? FileManager.default.removeItem(at: url) }

        let asset = AVURLAsset(url: url)
        let videoTracks = asset.tracks(withMediaType: .video)
        guard let videoTrack = videoTracks.first else {
            throw MiniCPMDemoMediaError.noVideoTrack
        }
        let duration = asset.duration.seconds
        guard duration.isFinite, duration >= 0 else {
            throw MiniCPMDemoMediaError.invalidContainer("invalid duration")
        }

        let frameLimit = max(1, maxFrames)
        let nominalRate = Double(videoTrack.nominalFrameRate)
        let estimatedCount = nominalRate > 0 && duration > 0
            ? max(1, Int(ceil(duration * nominalRate)))
            : 1
        let requestedCount = stackFrames ? min(frameLimit, estimatedCount) : 1
        var times: [CMTime] = []
        times.reserveCapacity(requestedCount)
        if requestedCount == 1 {
            times = [.zero]
        } else {
            let last = max(0, duration - (1.0 / max(nominalRate, 1.0)) * 0.25)
            for index in 0..<requestedCount {
                let fraction = Double(index) / Double(requestedCount - 1)
                times.append(CMTime(seconds: fraction * last, preferredTimescale: 600))
            }
        }

        let generator = AVAssetImageGenerator(asset: asset)
        generator.appliesPreferredTrackTransform = true
        generator.requestedTimeToleranceBefore = .zero
        generator.requestedTimeToleranceAfter = .zero
        var frames: [Data] = []
        frames.reserveCapacity(times.count)
        for time in times {
            do {
                let image = try generator.copyCGImage(at: time, actualTime: nil)
                guard let encoded = jpegData(image) else {
                    throw MiniCPMDemoMediaError.frameDecodeFailed("cannot encode frame as JPEG")
                }
                frames.append(encoded)
            } catch let error as MiniCPMDemoMediaError {
                throw error
            } catch {
                throw MiniCPMDemoMediaError.frameDecodeFailed(error.localizedDescription)
            }
        }
        guard !frames.isEmpty else {
            throw MiniCPMDemoMediaError.frameDecodeFailed("no decodable frames")
        }

        let audio = try decodeAudio(asset: asset, sampleRate: audioSampleRate)
        return MiniCPMDemoMediaPayload(
            frames: frames,
            audio: audio,
            audioSampleRate: audio == nil ? nil : audioSampleRate)
    }

    private static func jpegData(_ image: CGImage) -> Data? {
        let output = NSMutableData()
        guard let destination = CGImageDestinationCreateWithData(
            output,
            UTType.jpeg.identifier as CFString,
            1,
            nil) else {
            return nil
        }
        CGImageDestinationAddImage(destination, image, [
            kCGImageDestinationLossyCompressionQuality: 0.85
        ] as CFDictionary)
        guard CGImageDestinationFinalize(destination) else { return nil }
        return output as Data
    }

    private static func decodeAudio(asset: AVAsset, sampleRate: Int) throws -> [Float]? {
        guard let track = asset.tracks(withMediaType: .audio).first else { return nil }
        guard sampleRate > 0 else {
            throw MiniCPMDemoMediaError.audioDecodeFailed("sample rate must be positive")
        }
        let reader: AVAssetReader
        do {
            reader = try AVAssetReader(asset: asset)
        } catch {
            throw MiniCPMDemoMediaError.audioDecodeFailed(error.localizedDescription)
        }
        let settings: [String: Any] = [
            AVFormatIDKey: kAudioFormatLinearPCM,
            AVSampleRateKey: sampleRate,
            AVNumberOfChannelsKey: 1,
            AVLinearPCMBitDepthKey: 32,
            AVLinearPCMIsFloatKey: true,
            AVLinearPCMIsBigEndianKey: false,
            AVLinearPCMIsNonInterleaved: false
        ]
        let output = AVAssetReaderTrackOutput(track: track, outputSettings: settings)
        output.alwaysCopiesSampleData = false
        guard reader.canAdd(output) else {
            throw MiniCPMDemoMediaError.audioDecodeFailed("reader rejected audio output")
        }
        reader.add(output)
        guard reader.startReading() else {
            throw MiniCPMDemoMediaError.audioDecodeFailed(
                reader.error?.localizedDescription ?? "reader did not start")
        }

        var samples: [Float] = []
        while let sampleBuffer = output.copyNextSampleBuffer() {
            guard let block = CMSampleBufferGetDataBuffer(sampleBuffer) else {
                CMSampleBufferInvalidate(sampleBuffer)
                continue
            }
            let length = CMBlockBufferGetDataLength(block)
            guard length.isMultiple(of: MemoryLayout<Float>.size) else {
                CMSampleBufferInvalidate(sampleBuffer)
                throw MiniCPMDemoMediaError.audioDecodeFailed(
                    "converted PCM byte length is not divisible by 4")
            }
            var bytes = Data(count: length)
            let status = bytes.withUnsafeMutableBytes { destination in
                CMBlockBufferCopyDataBytes(
                    block,
                    atOffset: 0,
                    dataLength: length,
                    destination: destination.baseAddress!)
            }
            CMSampleBufferInvalidate(sampleBuffer)
            guard status == kCMBlockBufferNoErr else {
                throw MiniCPMDemoMediaError.audioDecodeFailed("cannot read PCM sample buffer")
            }
            bytes.withUnsafeBytes { raw in
                for offset in stride(from: 0, to: length, by: MemoryLayout<Float>.size) {
                    let bits = raw.loadUnaligned(fromByteOffset: offset, as: UInt32.self)
                    samples.append(Float(bitPattern: UInt32(littleEndian: bits)))
                }
            }
        }
        if reader.status == .failed {
            throw MiniCPMDemoMediaError.audioDecodeFailed(
                reader.error?.localizedDescription ?? "reader failed")
        }
        guard samples.allSatisfy(\.isFinite) else {
            throw MiniCPMDemoMediaError.audioDecodeFailed("PCM contains NaN or infinity")
        }
        return samples.isEmpty ? nil : samples
    }
    #else
    public static func decodeVideo(
        _: Data,
        stackFrames: Bool = true,
        maxFrames: Int = 32,
        audioSampleRate: Int = 16_000
    ) throws -> MiniCPMDemoMediaPayload {
        throw MiniCPMDemoMediaError.unsupportedPlatform
    }
    #endif
}
