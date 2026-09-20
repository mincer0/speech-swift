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
    public var odeSteps = 10
    public var maximumStreamingFrames = 100

    public init() {}
}
