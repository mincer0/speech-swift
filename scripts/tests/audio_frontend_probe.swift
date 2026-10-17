import Foundation

/// Standalone DSP probe: compile with Configuration.swift and
/// WhisperFeatureExtractor.swift; no MLX, SwiftPM, or model weights.
@main
struct AudioFrontendProbe {
    static func main() throws {
        let processor = MiniCPMStreamingMelProcessor()
        var snapshots: [String: MiniCPMStreamingMelProcessor.Snapshot] = [:]
        while let line = readLine() {
            do {
                let request = try JSONSerialization.jsonObject(with: Data(line.utf8))
                    as! [String: Any]
                var features: MiniCPMLogMelFeatures?
                switch request["op"] as? String ?? "accept" {
                case "snapshot":
                    snapshots[request["id"] as! String] = processor.snapshot()
                case "restore":
                    processor.restore(try requireSnapshot(
                        snapshots, request["id"] as! String))
                case "reset":
                    processor.reset()
                default:
                    let payload = request["samplesBase64"] as? String ?? ""
                    guard let data = Data(base64Encoded: payload), data.count % 4 == 0 else {
                        throw MiniCPMAudioError.invalidFeatures("invalid Float32 PCM payload")
                    }
                    let samples = data.withUnsafeBytes { bytes in
                        stride(from: 0, to: bytes.count, by: 4).map {
                            bytes.loadUnaligned(fromByteOffset: $0, as: Float.self)
                        }
                    }
                    features = try processor.accept(
                        samples, isFinal: request["isFinal"] as? Bool ?? false)
                }
                var response: [String: Any] = [
                    "ok": true,
                    "totalSamples": processor.totalSampleCount,
                    "bufferedSamples": processor.bufferedSampleCount,
                    "bufferStartSample": processor.bufferStartSampleIndex,
                    "emittedCoreFrames": processor.emittedFrameCount,
                    "nextChunkSamples": processor.nextChunkSampleCount,
                ]
                if let features {
                    let payload = features.data.withUnsafeBufferPointer { Data(buffer: $0) }
                    let positions = (features.timeFrames + 1) / 2
                        - (features.prefixExtraFrames + 1) / 2
                        - (features.suffixExtraFrames + 1) / 2
                    response["frames"] = features.timeFrames
                    response["prefixExtraFrames"] = features.prefixExtraFrames
                    response["suffixExtraFrames"] = features.suffixExtraFrames
                    response["encoderPositions"] = positions
                    response["audioTokens"] = positions / 5
                    response["featuresBase64"] = payload.base64EncodedString()
                }
                try write(response)
            } catch {
                try write(["ok": false, "error": String(describing: error)])
            }
        }
    }

    static func requireSnapshot(
        _ snapshots: [String: MiniCPMStreamingMelProcessor.Snapshot], _ id: String
    ) throws -> MiniCPMStreamingMelProcessor.Snapshot {
        guard let snapshot = snapshots[id] else {
            throw MiniCPMAudioError.invalidFeatures("unknown snapshot")
        }
        return snapshot
    }

    static func write(_ value: [String: Any]) throws {
        let data = try JSONSerialization.data(withJSONObject: value, options: [.sortedKeys])
        FileHandle.standardOutput.write(data)
        FileHandle.standardOutput.write(Data([10]))
    }
}
