import Foundation

public struct MiniCPMFlowConfiguration: Sendable {
    public var tokenVocabularySize = 6_561
    public var tokenDimension = 512
    public var melDimension = 80
    public var speakerEmbeddingDimension = 192
    public var encoderBlocks = 6
    public var encoderUpsampleBlocks = 4
    public var encoderHeads = 8
    public var encoderFeedForwardDimension = 2_048
    public var preLookahead = 3
    public var upsampleStride = 2
    public var decoderHiddenDimension = 512
    public var decoderBlocks = 16
    public var decoderHeads = 8
    public var decoderHeadDimension = 64
    public var decoderMLPRatio = 4
    public var decoderInputChannels = 320
    public var classifierFreeGuidance: Float = 0.7
    /// ODE steps for the flow-matching decoder. 10 is the official value.
    /// Lowering to 8 cut `token2wav`/flow time by ~1/5 with no audible
    /// complaint in a long-story session; verify by ear before going lower
    /// (each step is one full DiT forward over the mel frames).
    public var odeSteps = 8
    public var maximumStreamingFrames = 100

    public init() {}
}
