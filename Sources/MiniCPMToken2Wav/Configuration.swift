import Foundation

/// MiniCPM-o 4.5's bundled CosyVoice2 HiFT vocoder configuration.
public struct MiniCPMHiFTConfiguration: Sendable {
    public var inputChannels = 80
    public var baseChannels = 512
    public var harmonicCount = 8
    public var sampleRate = 24_000
    public var sineAmplitude: Float = 0.1
    public var noiseStandardDeviation: Float = 0.003
    public var voicedThreshold: Float = 10
    public var upsampleRates = [8, 5, 3]
    public var upsampleKernelSizes = [16, 11, 7]
    public var fftSize = 16
    public var fftHopLength = 4
    public var residualKernelSizes = [3, 7, 11]
    public var residualDilations = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
    public var sourceResidualKernelSizes = [7, 7, 11]
    public var sourceResidualDilations = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
    public var leakyReLUSlope: Float = 0.1
    public var audioLimit: Float = 0.99

    public init() {}

    public var sourceUpsampleScale: Int {
        upsampleRates.reduce(1, *) * fftHopLength
    }
}

public enum MiniCPMToken2WavError: Error, LocalizedError {
    case missingModelFile(URL)
    case invalidWeights(String)
    case invalidInput(String)
    case pipelineNotPrepared
    case turnClosed
    /// Cooperative cancellation at a Flow/HiFT boundary.  The pipeline
    /// restores its token/window/cache snapshot before rethrowing this case.
    case cancelled

    public var errorDescription: String? {
        switch self {
        case .missingModelFile(let url):
            "Missing MiniCPM token2wav model file: \(url.path)"
        case .invalidWeights(let reason):
            "Invalid MiniCPM token2wav weights: \(reason)"
        case .invalidInput(let reason):
            "Invalid MiniCPM token2wav input: \(reason)"
        case .pipelineNotPrepared:
            "MiniCPM token2wav pipeline has no prepared voice prompt"
        case .turnClosed:
            "MiniCPM token2wav turn is closed; call resetForNewTurn() first"
        case .cancelled:
            "MiniCPM token2wav synthesis was cancelled"
        }
    }
}
