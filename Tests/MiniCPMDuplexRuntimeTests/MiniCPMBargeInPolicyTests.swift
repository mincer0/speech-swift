import XCTest
@testable import MiniCPMDuplexRuntime

final class MiniCPMBargeInPolicyTests: XCTestCase {
    private let rate = 16_000

    private func frames(_ count: Int, amplitude: Float, sampleRate: Int = 16_000) -> [Float] {
        Array(repeating: amplitude, count: count * sampleRate / 50)
    }

    private func observe(
        _ policy: inout MiniCPMBargeInPolicy,
        _ audio: [Float],
        answerOpen: Bool = true,
        sampleRate: Int = 16_000
    ) -> MiniCPMBargeInPolicy.Decision {
        policy.observe(audio: audio, sampleRate: sampleRate, answerOpen: answerOpen)
    }

    func testWeakAndStrongEightyMillisecondTransientsDoNotInterrupt() {
        for amplitude in [Float(0.012), 0.025] {
            var policy = MiniCPMBargeInPolicy()
            let input = frames(4, amplitude: amplitude) + frames(50, amplitude: 0)
            XCTAssertEqual(observe(&policy, input), .init(interrupt: false, forceListen: false))
        }
    }

    func testStrongSpeechInterruptsAtTwoHundredFortyMilliseconds() {
        var policy = MiniCPMBargeInPolicy()
        XCTAssertFalse(observe(&policy, frames(11, amplitude: 0.025)).interrupt)
        XCTAssertEqual(
            observe(&policy, frames(1, amplitude: 0.025)),
            .init(interrupt: true, forceListen: true))
        XCTAssertEqual(
            observe(&policy, frames(12, amplitude: 0.025)),
            .init(interrupt: false, forceListen: true))
    }

    func testEndedAnswerWithBufferedPlaybackStillRequiresSustainedSpeech() {
        var policy = MiniCPMBargeInPolicy()
        XCTAssertFalse(policy.observe(
            audio: frames(4, amplitude: 0.025) + frames(12, amplitude: 0),
            sampleRate: rate, answerOpen: false, playbackActive: true).interrupt)
        XCTAssertFalse(policy.observe(
            audio: frames(11, amplitude: 0.025), sampleRate: rate,
            answerOpen: false, playbackActive: true).interrupt)
        XCTAssertTrue(policy.observe(
            audio: frames(1, amplitude: 0.025), sampleRate: rate,
            answerOpen: false, playbackActive: true).interrupt)
    }

    func testFinalModelTokenPreservesOnsetWhilePlaybackRemainsBuffered() {
        var policy = MiniCPMBargeInPolicy()
        XCTAssertFalse(observe(&policy, frames(11, amplitude: 0.025)).interrupt)
        XCTAssertTrue(policy.observe(
            audio: frames(1, amplitude: 0.025), sampleRate: rate,
            answerOpen: false, playbackActive: true).interrupt)
    }

    func testDrainedPlaybackClearsUnqualifiedOnsetAfterModelTurnEnd() {
        var policy = MiniCPMBargeInPolicy()
        XCTAssertFalse(observe(&policy, frames(11, amplitude: 0.025)).interrupt)
        XCTAssertFalse(policy.observe(
            audio: frames(50, amplitude: 0.025), sampleRate: rate,
            answerOpen: false, playbackActive: false).interrupt)
        XCTAssertFalse(policy.observe(
            audio: frames(1, amplitude: 0.025), sampleRate: rate,
            answerOpen: false, playbackActive: true).interrupt)
    }

    func testModerateSpeechRequiresFourHundredEightyMilliseconds() {
        var policy = MiniCPMBargeInPolicy()
        XCTAssertFalse(observe(&policy, frames(23, amplitude: 0.012)).interrupt)
        XCTAssertTrue(observe(&policy, frames(1, amplitude: 0.012)).interrupt)
    }

    func testCrossInputWordAndPartialFrameAreRetainedWithoutPadding() {
        var policy = MiniCPMBargeInPolicy()
        XCTAssertFalse(observe(&policy, frames(5, amplitude: 0.025)).interrupt)
        XCTAssertFalse(observe(&policy, frames(6, amplitude: 0.025)).interrupt)
        XCTAssertFalse(observe(&policy, Array(repeating: 0.025, count: rate / 50 - 1)).interrupt)
        XCTAssertTrue(observe(&policy, [0.025]).interrupt)
    }

    func testShortFragmentedSpeechGapKeepsEvidence() {
        var policy = MiniCPMBargeInPolicy()
        XCTAssertFalse(observe(&policy, frames(5, amplitude: 0.025)).interrupt)
        XCTAssertFalse(observe(&policy, frames(11, amplitude: 0)).interrupt)
        XCTAssertTrue(observe(&policy, frames(2, amplitude: 0.025)).interrupt)
    }

    func testTwoHundredFortyMillisecondGapClearsEvidence() {
        var policy = MiniCPMBargeInPolicy()
        XCTAssertFalse(observe(&policy, frames(5, amplitude: 0.025)).interrupt)
        XCTAssertFalse(observe(&policy, frames(12, amplitude: 0)).interrupt)
        XCTAssertFalse(observe(&policy, frames(11, amplitude: 0.025)).interrupt)
        XCTAssertTrue(observe(&policy, frames(1, amplitude: 0.025)).interrupt)
    }

    func testQuietCandidateMaturityCannotTriggerAnInterruption() {
        var policy = MiniCPMBargeInPolicy()
        XCTAssertFalse(observe(&policy, frames(8, amplitude: 0.025)).interrupt)
        XCTAssertEqual(
            observe(&policy, frames(8, amplitude: 0)),
            .init(interrupt: false, forceListen: false))
        XCTAssertTrue(observe(&policy, frames(1, amplitude: 0.012)).interrupt)

        policy.reset()
        let isolated = frames(8, amplitude: 0.025) + frames(12, amplitude: 0)
        XCTAssertEqual(observe(&policy, isolated), .init(interrupt: false, forceListen: false))
        XCTAssertFalse(observe(&policy, frames(11, amplitude: 0.025)).interrupt)
    }

    func testClosedAnswerCannotAccumulateOnsetOrPartialEvidence() {
        var policy = MiniCPMBargeInPolicy()
        XCTAssertFalse(observe(&policy, frames(11, amplitude: 0.025)).interrupt)
        XCTAssertEqual(
            observe(&policy, [0.025], answerOpen: false),
            .init(interrupt: false, forceListen: false))
        XCTAssertFalse(observe(&policy, frames(11, amplitude: 0.025)).interrupt)
        XCTAssertFalse(observe(&policy, Array(repeating: 0.025, count: rate / 50 - 1)).interrupt)
        XCTAssertTrue(observe(&policy, [0.025]).interrupt)
    }

    func testLatchKeepsListeningToQuietSpeechAfterAnswerCloses() {
        var policy = MiniCPMBargeInPolicy()
        XCTAssertTrue(observe(&policy, frames(12, amplitude: 0.025)).interrupt)
        XCTAssertEqual(
            observe(&policy, frames(80, amplitude: 0.006), answerOpen: false),
            .init(interrupt: false, forceListen: true))
        XCTAssertEqual(
            observe(&policy, [], answerOpen: false),
            .init(interrupt: false, forceListen: true))
    }

    func testReleaseRequiresContinuousQuietAndReleaseInputStillListens() {
        var policy = MiniCPMBargeInPolicy()
        XCTAssertTrue(observe(&policy, frames(12, amplitude: 0.025)).interrupt)
        XCTAssertTrue(observe(&policy, frames(23, amplitude: 0), answerOpen: false).forceListen)
        XCTAssertTrue(observe(&policy, frames(1, amplitude: 0.006), answerOpen: false).forceListen)
        XCTAssertTrue(observe(&policy, frames(23, amplitude: 0), answerOpen: false).forceListen)
        XCTAssertEqual(
            observe(&policy, frames(1, amplitude: 0), answerOpen: false),
            .init(interrupt: false, forceListen: true))
        XCTAssertEqual(
            observe(&policy, frames(1, amplitude: 0), answerOpen: false),
            .init(interrupt: false, forceListen: false))
    }

    func testInterruptAndReleaseInOneInputStillReportsInterruptAndListen() {
        var policy = MiniCPMBargeInPolicy()
        let input = frames(12, amplitude: 0.025) + frames(30, amplitude: 0)
        XCTAssertEqual(observe(&policy, input), .init(interrupt: true, forceListen: true))
        XCTAssertEqual(observe(&policy, []), .init(interrupt: false, forceListen: false))
    }

    func testReleaseDoesNotRearmWithinTheSameInput() {
        var policy = MiniCPMBargeInPolicy()
        let input = frames(12, amplitude: 0.025)
            + frames(24, amplitude: 0)
            + frames(12, amplitude: 0.025)
        XCTAssertEqual(observe(&policy, input), .init(interrupt: true, forceListen: true))
        XCTAssertEqual(
            observe(&policy, frames(30, amplitude: 0.006), answerOpen: false),
            .init(interrupt: false, forceListen: true))
        XCTAssertEqual(
            observe(&policy, frames(24, amplitude: 0), answerOpen: false),
            .init(interrupt: false, forceListen: true))
        XCTAssertEqual(
            observe(&policy, [], answerOpen: false),
            .init(interrupt: false, forceListen: false))
    }

    func testVoicedPartialAfterReleasePauseKeepsLatch() {
        var policy = MiniCPMBargeInPolicy()
        let input = frames(12, amplitude: 0.025) + frames(24, amplitude: 0) + [0.006]
        XCTAssertEqual(observe(&policy, input), .init(interrupt: true, forceListen: true))
        XCTAssertEqual(
            observe(&policy, frames(1, amplitude: 0.006), answerOpen: false),
            .init(interrupt: false, forceListen: true))
    }

    func testPartialQuietFrameDoesNotPrematurelyReleaseLatch() {
        var policy = MiniCPMBargeInPolicy()
        XCTAssertTrue(observe(&policy, frames(12, amplitude: 0.025)).interrupt)
        XCTAssertTrue(observe(&policy, frames(23, amplitude: 0), answerOpen: false).forceListen)
        XCTAssertTrue(observe(&policy, Array(repeating: 0, count: rate / 50 - 1), answerOpen: false).forceListen)
        XCTAssertTrue(observe(&policy, [], answerOpen: false).forceListen)
        XCTAssertTrue(observe(&policy, [0], answerOpen: false).forceListen)
        XCTAssertFalse(observe(&policy, [], answerOpen: false).forceListen)
    }

    func testResetClearsCandidatePartialAndLatch() {
        var policy = MiniCPMBargeInPolicy()
        _ = observe(&policy, frames(11, amplitude: 0.025) + [0.025])
        policy.reset()
        XCTAssertFalse(observe(&policy, frames(11, amplitude: 0.025)).interrupt)
        XCTAssertFalse(observe(&policy, Array(repeating: 0.025, count: rate / 50 - 1)).interrupt)
        XCTAssertTrue(observe(&policy, [0.025]).interrupt)
        policy.reset()
        XCTAssertEqual(observe(&policy, [], answerOpen: false), .init(interrupt: false, forceListen: false))
    }

    func testNonfiniteFrameClearsOnsetEvidenceAndCannotReleaseLatch() {
        for invalid in [Float.nan, .infinity, -.infinity] {
            var policy = MiniCPMBargeInPolicy()
            XCTAssertFalse(observe(&policy, frames(11, amplitude: 0.025)).interrupt)
            XCTAssertFalse(observe(&policy, frames(1, amplitude: invalid)).interrupt)
            XCTAssertFalse(observe(&policy, frames(11, amplitude: 0.025)).interrupt)
            XCTAssertTrue(observe(&policy, frames(1, amplitude: 0.025)).interrupt)
            XCTAssertTrue(observe(&policy, frames(24, amplitude: invalid), answerOpen: false).forceListen)
            XCTAssertTrue(observe(&policy, frames(23, amplitude: 0), answerOpen: false).forceListen)
            XCTAssertTrue(observe(&policy, frames(1, amplitude: 0), answerOpen: false).forceListen)
            XCTAssertFalse(observe(&policy, [], answerOpen: false).forceListen)
        }
    }

    func testRateChangeClearsOnsetAndPartialEvidence() {
        var policy = MiniCPMBargeInPolicy()
        XCTAssertFalse(observe(&policy, frames(11, amplitude: 0.025) + [0.025]).interrupt)
        XCTAssertFalse(observe(&policy, frames(11, amplitude: 0.025, sampleRate: 8_000), sampleRate: 8_000).interrupt)
        XCTAssertFalse(observe(&policy, Array(repeating: 0.025, count: 159), sampleRate: 8_000).interrupt)
        XCTAssertTrue(observe(&policy, [0.025], sampleRate: 8_000).interrupt)
    }

    func testRateChangeAndInvalidRateCannotDropListenLatch() {
        var policy = MiniCPMBargeInPolicy()
        XCTAssertTrue(observe(&policy, frames(12, amplitude: 0.025)).interrupt)
        XCTAssertTrue(observe(&policy, frames(23, amplitude: 0), answerOpen: false).forceListen)
        XCTAssertTrue(observe(&policy, [0], answerOpen: false, sampleRate: 0).forceListen)
        XCTAssertTrue(observe(&policy, [], answerOpen: false, sampleRate: -16_000).forceListen)
        XCTAssertTrue(observe(&policy, frames(23, amplitude: 0, sampleRate: 48_000), answerOpen: false, sampleRate: 48_000).forceListen)
        XCTAssertTrue(observe(&policy, frames(1, amplitude: 0, sampleRate: 48_000), answerOpen: false, sampleRate: 48_000).forceListen)
        XCTAssertFalse(observe(&policy, [], answerOpen: false, sampleRate: 48_000).forceListen)
    }

    func testFractionalFrameSampleCountsPreserveElapsedTime() {
        var policy = MiniCPMBargeInPolicy()
        let sampleRate = 11_025
        let samplesBeforeOnset = 12 * sampleRate / 50 - 1
        XCTAssertFalse(observe(&policy, Array(repeating: 0.025, count: samplesBeforeOnset), sampleRate: sampleRate).interrupt)
        XCTAssertTrue(observe(&policy, [0.025], sampleRate: sampleRate).interrupt)
    }
}
