/**
 * Preserve the continuous audio timeline expected by MiniCPM duplex mode.
 * Startup silence is withheld so the model cannot speak before the user, but
 * silence after the first speech chunk is sent as zeros to mark elapsed time.
 */
export class LiveInputTimeline {
    constructor({
        activityRmsThreshold = 0.0035,
        responseWindowChunks = 3,
        preRollChunks = 1,
    } = {}) {
        this.activityRmsThreshold = activityRmsThreshold;
        this.responseWindowChunks = Math.max(0, Math.floor(responseWindowChunks));
        this.preRollChunks = Math.max(0, Math.floor(preRollChunks));
        this.reset();
    }

    reset() {
        this.started = false;
        this.recentSpeechChunks = 0;
        this.startupFrames = [];
    }

    // An application-generated continuation is a new response opportunity,
    // even though it has no microphone speech chunk to open the window.
    openResponseWindow() {
        this.started = true;
        this.recentSpeechChunks = this.responseWindowChunks;
        this.startupFrames = [];
    }

    prepare(samples, activityRms) {
        const active = Number.isFinite(activityRms)
            && activityRms >= this.activityRmsThreshold;

        if (!this.started && !active) {
            if (this.preRollChunks > 0 && samples?.length) {
                this.startupFrames.push(samples.slice());
                while (this.startupFrames.length > this.preRollChunks) {
                    this.startupFrames.shift();
                }
            }
            return {
                send: false, samples, zeroFilled: false,
                speechActive: false, responseWindowActive: false, prefixSamples: [],
            };
        }

        const prefixSamples = !this.started && active
            ? this.startupFrames.splice(0)
            : [];
        if (active) {
            this.started = true;
            this.recentSpeechChunks = this.responseWindowChunks;
        }
        // Stable-mode answer gating: while the user is actively speaking the
        // leading listen/speak decision stays deterministic (greedy listen),
        // so the model stops interjecting fragments mid-question. The sampled
        // window opens only on the post-speech pause that follows.
        const responseWindowActive = !active && this.recentSpeechChunks > 0;
        if (!active) {
            if (this.recentSpeechChunks > 0) this.recentSpeechChunks--;
            return {
                send: true,
                samples: new Float32Array(samples.length),
                zeroFilled: true,
                speechActive: false,
                responseWindowActive, prefixSamples,
            };
        }

        return {
            send: true, samples, zeroFilled: false,
            speechActive: true, responseWindowActive, prefixSamples,
        };
    }
}
