/**
 * echo-reference.js — far-end reference diagnostics for browser AEC.
 *
 * Browser AEC (requested by getUserMedia) remains the primary cancellation
 * path.  The short far-end history also provides a conservative software
 * fallback for browsers/devices whose hardware AEC is ineffective: only a
 * strongly correlated playback component is subtracted, while unrelated mic
 * speech is preserved.  This is echo suppression, not VAD or turn detection.
 */

function rms(samples) {
    if (!samples || samples.length === 0) return 0;
    let sum = 0;
    for (let i = 0; i < samples.length; i++) sum += samples[i] * samples[i];
    return Math.sqrt(sum / samples.length);
}

function resampleLinear(samples, fromRate, toRate) {
    if (!samples || samples.length === 0) return new Float32Array(0);
    if (!fromRate || !toRate || fromRate === toRate) return samples;
    const length = Math.max(1, Math.round(samples.length * toRate / fromRate));
    const output = new Float32Array(length);
    const ratio = fromRate / toRate;
    for (let i = 0; i < length; i++) {
        const position = i * ratio;
        const left = Math.min(samples.length - 1, Math.floor(position));
        const right = Math.min(samples.length - 1, left + 1);
        const fraction = position - left;
        output[i] = samples[left] + (samples[right] - samples[left]) * fraction;
    }
    return output;
}

function normalizedCorrelation(a, b) {
    const n = Math.min(a.length, b.length);
    if (n < 8) return 0;
    let aa = 0;
    let bb = 0;
    let ab = 0;
    for (let i = 0; i < n; i++) {
        aa += a[i] * a[i];
        bb += b[i] * b[i];
        ab += a[i] * b[i];
    }
    const denominator = Math.sqrt(aa * bb);
    return denominator > 1e-9 ? ab / denominator : 0;
}

export class FarEndEchoReference {
    constructor(options = {}) {
        this.sampleRate = options.sampleRate || 16000;
        this.maxSamples = Math.max(
            this.sampleRate,
            Math.round((options.historyMs || 2000) / 1000 * this.sampleRate),
        );
        this._farEnd = new Float32Array(0);
        this._lastPlaybackAt = 0;
        this._last = null;
    }

    reset() {
        this._farEnd = new Float32Array(0);
        this._lastPlaybackAt = 0;
        this._last = null;
    }

    /** Add raw scheduled playback audio, not merely network arrival audio. */
    pushPlayback(samples, sampleRate = this.sampleRate, timestamp = performance.now()) {
        const normalized = resampleLinear(samples, sampleRate, this.sampleRate);
        if (normalized.length === 0) return;
        const joined = new Float32Array(this._farEnd.length + normalized.length);
        joined.set(this._farEnd);
        joined.set(normalized, this._farEnd.length);
        this._farEnd = joined.length > this.maxSamples
            ? joined.slice(joined.length - this.maxSamples)
            : joined;
        this._lastPlaybackAt = timestamp;
    }

    /**
     * Analyze a microphone chunk against the most recent far-end window.
     * `suppressionSuggested` is telemetry only; callers must not use it as a
     * full-duplex input gate.
     */
    analyzeInput(samples, sampleRate = this.sampleRate, timestamp = performance.now()) {
        const mic = resampleLinear(samples, sampleRate, this.sampleRate);
        const far = this._farEnd.slice(Math.max(0, this._farEnd.length - mic.length));
        const micRms = rms(mic);
        const farRms = rms(far);
        const correlation = normalizedCorrelation(mic, far);
        const playbackAgeMs = this._lastPlaybackAt > 0
            ? Math.max(0, timestamp - this._lastPlaybackAt)
            : null;
        // A strong positive correlation while playback is recent is a useful
        // warning, but the browser's AEC and the model still receive `mic`.
        const suppressionSuggested = farRms > 0.01
            && correlation > 0.55
            && (playbackAgeMs === null || playbackAgeMs < 1500);
        const result = {
            micRms,
            farRms,
            correlation,
            playbackAgeMs,
            suppressionSuggested,
        };
        this._last = result;
        return result;
    }

    /**
     * Remove a strongly correlated far-end component from one mic chunk.
     * The returned samples stay at the requested input rate.  A least-squares
     * gain prevents the far-end reference from changing the user's voice when
     * the correlation is weak or the speaker is silent.
     */
    suppressInput(samples, sampleRate = this.sampleRate, timestamp = performance.now()) {
        if (!samples || samples.length === 0) {
            return { samples, diagnostics: this.analyzeInput(samples, sampleRate, timestamp) };
        }
        const mic = resampleLinear(samples, sampleRate, this.sampleRate);
        const diagnostics = this.analyzeInput(samples, sampleRate, timestamp);
        const far = this._farEnd.slice(Math.max(0, this._farEnd.length - mic.length));
        if (!diagnostics.suppressionSuggested || far.length !== mic.length) {
            return { samples, diagnostics, suppressed: false, gain: 0 };
        }

        let farEnergy = 0;
        let crossEnergy = 0;
        for (let i = 0; i < mic.length; i++) {
            farEnergy += far[i] * far[i];
            crossEnergy += mic[i] * far[i];
        }
        const gain = Math.max(0, Math.min(1.5, crossEnergy / Math.max(farEnergy, 1e-9)));
        if (gain < 0.02) return { samples, diagnostics, suppressed: false, gain };

        const filtered = new Float32Array(mic.length);
        let residualEnergy = 0;
        for (let i = 0; i < mic.length; i++) {
            const value = mic[i] - gain * far[i];
            filtered[i] = value;
            residualEnergy += value * value;
        }
        const residualRms = Math.sqrt(residualEnergy / Math.max(1, filtered.length));
        const accepted = residualRms <= diagnostics.micRms * 1.05;
        const output = accepted && sampleRate === this.sampleRate
            ? filtered
            : samples;
        return {
            samples: output,
            diagnostics: { ...diagnostics, echoSuppressed: accepted, echoGain: gain, residualRms },
            suppressed: accepted,
            gain,
        };
    }

    get last() { return this._last; }
}
