/**
 * Keeps a finite file-mode session clocked long enough to receive and finish
 * the assistant turn. MiniCPM duplex generation advances with input chunks,
 * so stopping exactly at EOF can truncate a response that is still speaking.
 */
export class FileResponseDrain {
    constructor({ startGraceChunks = 5, maxChunks = 60 } = {}) {
        this.startGraceChunks = Math.max(0, startGraceChunks);
        this.maxChunks = Math.max(this.startGraceChunks, maxChunks);
        this.reset();
    }

    reset() {
        this.chunks = 0;
        this.sawSpeaking = false;
    }

    next(modelState, playbackActive = false) {
        const state = modelState || 'listening';
        if (state === 'speaking') this.sawSpeaking = true;

        const responseComplete = (this.sawSpeaking && state !== 'speaking')
            || (!this.sawSpeaking && state === 'end_of_turn');
        if (responseComplete && playbackActive && this.chunks < this.maxChunks) {
            this.chunks += 1;
            return { continue: true, reason: 'playback', chunks: this.chunks };
        }
        if (this.sawSpeaking && state !== 'speaking') {
            return { continue: false, reason: 'response_complete', chunks: this.chunks };
        }
        if (!this.sawSpeaking && state === 'end_of_turn') {
            return { continue: false, reason: 'response_complete', chunks: this.chunks };
        }
        if (this.chunks >= this.maxChunks) {
            return { continue: false, reason: 'timeout', chunks: this.chunks };
        }
        if (!this.sawSpeaking && this.chunks >= this.startGraceChunks) {
            return { continue: false, reason: 'no_response', chunks: this.chunks };
        }

        this.chunks += 1;
        return { continue: true, reason: this.sawSpeaking ? 'speaking' : 'start_grace', chunks: this.chunks };
    }
}
