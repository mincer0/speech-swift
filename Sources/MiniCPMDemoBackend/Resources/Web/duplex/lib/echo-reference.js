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
        this._segments = [];
        this._maxDelayMs = 200;
        this._last = null;
    }

    reset() {
        this._segments = [];
        this._last = null;
    }

    /** Add raw scheduled playback audio, not merely network arrival audio. */
    pushPlayback(samples, sampleRate = this.sampleRate, timestamp = performance.now()) {
        const normalized = resampleLinear(samples, sampleRate, this.sampleRate);
        if (normalized.length === 0) return;
        if (!Number.isFinite(timestamp)) return;
        this._segments.push({
            start: timestamp,
            end: timestamp + normalized.length / this.sampleRate * 1000,
            samples: Float32Array.from(normalized),
        });
    }

    _alignedReference(mic, timestamp) {
        const durationMs = mic.length / this.sampleRate * 1000;
        const oldest = timestamp - durationMs - this._maxDelayMs
            - this.maxSamples / this.sampleRate * 1000;
        this._segments = this._segments.filter(segment => segment.end > oldest);
        const delaySamples = Math.round(this._maxDelayMs / 1000 * this.sampleRate);
        const start = timestamp - durationMs - this._maxDelayMs;
        const timeline = new Float32Array(mic.length + delaySamples);
        const covered = new Uint8Array(timeline.length);
        // Build the time-aligned history once, then inspect cheap views for
        // each delay. Reconstructing it per candidate stalls the capture UI.
        for (const segment of this._segments) {
            if (segment.start >= timestamp || segment.end <= start) continue;
            const first = Math.max(0, Math.ceil((segment.start - start) * this.sampleRate / 1000 - 1e-6));
            const end = Math.min(timeline.length, Math.ceil((segment.end - start) * this.sampleRate / 1000 - 1e-6));
            const offset = (start - segment.start) * this.sampleRate / 1000;
            for (let i = first; i < end; i++) {
                const position = Math.max(0, offset + i);
                const left = Math.min(segment.samples.length - 1, Math.floor(position));
                const right = Math.min(segment.samples.length - 1, left + 1);
                timeline[i] = segment.samples[left]
                    + (segment.samples[right] - segment.samples[left]) * (position - left);
                covered[i] = 1;
            }
        }
        const missing = new Uint32Array(timeline.length + 1);
        for (let i = 0; i < timeline.length; i++) missing[i + 1] = missing[i] + (1 - covered[i]);
        let best = { far: new Float32Array(0), correlation: 0, delayMs: null };
        // Timestamp is the END of the captured mic chunk; playback timestamps
        // are the START of scheduled sources. Only a complete past window can
        // suppress input. Uncertain timing/coverage leaves browser AEC in charge.
        for (let delayMs = 0; delayMs <= this._maxDelayMs; delayMs += 10) {
            const first = delaySamples - Math.round(delayMs / 1000 * this.sampleRate);
            const end = first + mic.length;
            if (mic.length === 0 || missing[end] !== missing[first]) continue;
            const far = timeline.subarray(first, end);
            const correlation = normalizedCorrelation(mic, far);
            if (best.delayMs === null || correlation > best.correlation) {
                best = { far, correlation, delayMs };
            }
        }
        return best;
    }

    _analyze(mic, timestamp) {
        const aligned = this._alignedReference(mic, timestamp);
        const micRms = rms(mic);
        const farRms = rms(aligned.far);
        const { correlation, delayMs } = aligned;
        const suppressionSuggested = delayMs !== null && farRms > 0.01 && correlation > 0.55;
        const diagnostics = {
            micRms, farRms, correlation,
            playbackAgeMs: delayMs,
            referenceAligned: delayMs !== null,
            suppressionSuggested,
        };
        this._last = diagnostics;
        return { ...aligned, diagnostics };
    }

    /**
     * Analyze a microphone chunk against the most recent far-end window.
     * `suppressionSuggested` is telemetry only; callers must not use it as a
     * full-duplex input gate.
     */
    analyzeInput(samples, sampleRate = this.sampleRate, timestamp = performance.now()) {
        const mic = resampleLinear(samples, sampleRate, this.sampleRate);
        return this._analyze(mic, timestamp).diagnostics;
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
        const { diagnostics, far } = this._analyze(mic, timestamp);
        if (!diagnostics.suppressionSuggested || sampleRate !== this.sampleRate) {
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
        const output = accepted ? filtered : samples;
        return {
            samples: output,
            diagnostics: { ...diagnostics, echoSuppressed: accepted, echoGain: gain, residualRms },
            suppressed: accepted,
            gain,
        };
    }

    get last() { return this._last; }
}
