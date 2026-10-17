/// Qualifies sustained microphone speech before interrupting an open answer.
/// Audio is inspected in complete 20 ms frames; the caller retains the input
/// unchanged for the model. This policy expects echo-processed microphone PCM.
struct MiniCPMBargeInPolicy: Sendable {
    struct Decision: Equatable, Sendable {
        let interrupt: Bool
        let forceListen: Bool
    }

    private static let framesPerSecond = 50
    private static let onsetRMS = 0.010
    private static let releaseRMS = 0.004
    private static let strongMultiplier = 1.8
    private static let requiredCandidateFrames = 24
    private static let requiredWeightedVoicedFrames = 10
    private static let maximumCandidateQuietFrames = 12
    private static let requiredReleaseQuietFrames = 24

    private var sampleRate: Int?
    private var frameTargetSamples = 0
    private var frameSampleCount = 0
    private var frameSquareSum = 0.0
    private var frameIsFinite = true
    // Carry fractional samples so rates such as 11,025 Hz still account for
    // exactly 50 frames per second without padding a partial input packet.
    private var frameRemainder = 0
    private var frameRemainderAccumulator = 0

    private var candidateFrames = 0
    private var weightedVoicedFrames = 0
    private var candidateQuietFrames = 0
    private var latched = false
    private var releaseQuietFrames = 0

    init() {}

    mutating func reset() {
        self = Self()
    }

    mutating func observe(
        audio: [Float], sampleRate: Int, answerOpen: Bool, playbackActive: Bool = false
    ) -> Decision {
        var interrupt = false
        var forceListen = latched
        // The final model token can precede audible completion. Buffered
        // output still needs the same sustained onset qualification.
        let canInterrupt = answerOpen || playbackActive

        // Invalid timing cannot create speech evidence or release an existing
        // interruption. A rate change starts a fresh acoustic timeline while
        // retaining the listen latch until valid quiet audio releases it.
        guard sampleRate >= Self.framesPerSecond else {
            clearAcousticTimeline(sampleRate: nil)
            return Decision(interrupt: false, forceListen: forceListen)
        }
        if self.sampleRate != sampleRate {
            clearAcousticTimeline(sampleRate: sampleRate)
        }

        guard canInterrupt || latched else {
            clearAcousticTimeline(sampleRate: sampleRate)
            return Decision(interrupt: false, forceListen: false)
        }

        for sample in audio {
            frameSampleCount += 1
            if sample.isFinite {
                let value = Double(sample)
                frameSquareSum += value * value
            } else {
                frameIsFinite = false
            }
            guard frameSampleCount == frameTargetSamples else { continue }

            let finiteFrame = frameIsFinite
            let rms = (frameSquareSum / Double(frameTargetSamples)).squareRoot()
            advanceFrame()

            guard finiteFrame else {
                clearCandidate()
                releaseQuietFrames = 0
                continue
            }

            if latched {
                forceListen = true
                if rms < Self.releaseRMS {
                    releaseQuietFrames = min(
                        releaseQuietFrames + 1, Self.requiredReleaseQuietFrames)
                } else {
                    releaseQuietFrames = 0
                }
                continue
            }

            guard canInterrupt else {
                clearCandidate()
                continue
            }

            if rms > Self.onsetRMS {
                let weight = rms >= Self.onsetRMS * Self.strongMultiplier ? 2 : 1
                candidateFrames += weight
                weightedVoicedFrames += weight
                candidateQuietFrames = 0
            } else if candidateFrames > 0 {
                candidateFrames += 1
                candidateQuietFrames += 1
                if candidateQuietFrames >= Self.maximumCandidateQuietFrames {
                    clearCandidate()
                }
            }

            // Gaps preserve elapsed candidate time, but an irreversible cut
            // still requires speech in the current frame. Quiet alone must
            // never complete an otherwise subthreshold onset.
            if rms > Self.onsetRMS,
               candidateFrames >= Self.requiredCandidateFrames,
               weightedVoicedFrames >= Self.requiredWeightedVoicedFrames {
                latched = true
                releaseQuietFrames = 0
                interrupt = true
                forceListen = true
                clearCandidate()
            }
        }

        // Release only at the input boundary: speech after an earlier pause
        // in this same input must keep the latch armed. The release input
        // itself still listens, so its microphone audio is accepted first.
        // An incomplete voiced/nonfinite frame also vetoes release without
        // padding it into a complete quiet frame.
        let trailingPartialIsQuiet = frameSampleCount == 0
            || (frameIsFinite
                && (frameSquareSum / Double(frameSampleCount)).squareRoot() < Self.releaseRMS)
        if latched, releaseQuietFrames >= Self.requiredReleaseQuietFrames,
           trailingPartialIsQuiet {
            latched = false
            releaseQuietFrames = 0
            clearCandidate()
        }

        return Decision(interrupt: interrupt, forceListen: forceListen)
    }

    private mutating func clearCandidate() {
        candidateFrames = 0
        weightedVoicedFrames = 0
        candidateQuietFrames = 0
    }

    private mutating func clearAcousticTimeline(sampleRate: Int?) {
        self.sampleRate = sampleRate
        frameSampleCount = 0
        frameSquareSum = 0
        frameIsFinite = true
        frameTargetSamples = (sampleRate ?? 0) / Self.framesPerSecond
        frameRemainder = (sampleRate ?? 0) % Self.framesPerSecond
        frameRemainderAccumulator = frameRemainder
        clearCandidate()
        releaseQuietFrames = 0
    }

    private mutating func advanceFrame() {
        frameSampleCount = 0
        frameSquareSum = 0
        frameIsFinite = true
        frameTargetSamples = (sampleRate ?? 0) / Self.framesPerSecond
        frameRemainderAccumulator += frameRemainder
        if frameRemainderAccumulator >= Self.framesPerSecond {
            frameTargetSamples += 1
            frameRemainderAccumulator -= Self.framesPerSecond
        }
    }
}
