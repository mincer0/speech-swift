/**
 * lib/audio-player.js — AudioBufferSourceNode pre-scheduled audio player (zero DOM dependency)
 *
 * Reports playback metrics via the onMetrics callback instead of writing to DOM directly.
 * The consumer (UI layer / page) wires onMetrics to update the display.
 *
 * @import { resampleAudio } from './duplex-utils.js'
 */

import { resampleAudio } from './duplex-utils.js';

const DEFAULT_MAX_LEARNED_STARTUP_MS = 1000;
const GENERATION_JITTER_MARGIN_MS = 100;

/** Remove only padding explicitly declared by the MiniCPM wire metadata. */
export function trimAudioPadding(samples, paddingSamples = 0) {
    const padding = Math.max(0, Math.min(
        samples.length,
        Math.floor(Number(paddingSamples) || 0),
    ));
    return padding === 0 ? samples : samples.subarray(padding);
}

/**
 * Keep startup latency bounded. The server's per-chunk wall clock is used only
 * as a small jitter-buffer hint: one chunk of observed work plus a fixed
 * margin, capped at one second. It cannot recreate the old 2.5-4 second
 * cumulative delay, while it gives the scheduler enough headroom for the
 * measured 1.0-1.2 second chunk cadence.
 */
export function adaptivePlaybackDelayMs(
    requestedDelayMs,
    generationLatencyMs,
    audioDurationMs,
    learnedDelayMs = 0,
    maxDelayMs = DEFAULT_MAX_LEARNED_STARTUP_MS,
) {
    const maxDelay = Math.max(0, Number(maxDelayMs) || 0);
    const requested = Math.max(0, Number(requestedDelayMs) || 0);
    const learned = Math.max(0, Number(learnedDelayMs) || 0);
    const generation = Math.max(0, Number(generationLatencyMs) || 0);
    const duration = Math.max(0, Number(audioDurationMs) || 0);
    const observed = generation > 0
        ? generation + GENERATION_JITTER_MARGIN_MS
        : duration > 0 ? Math.min(duration, 250) : 0;
    return Math.min(maxDelay, Math.max(requested, learned, observed));
}

export class AudioPlayer {
    /**
     * @param {object} [options]
     * @param {number} [options.outputSampleRate] - Expected output sample rate (e.g. 24000)
     * @param {function} [options.getPlaybackDelayMs] - Returns playback delay in ms (default: () => 200)
     */
    constructor(options = {}) {
        this._outputSR_expected = options.outputSampleRate || 24000;
        this._getDelayMs = options.getPlaybackDelayMs || (() => 200);
        this._maxStartupDelayMs = Math.max(
            0,
            Number(options.maxStartupDelayMs ?? 2500) || 0,
        );

        this._ctx = null;
        this._outputSR = 0;
        this._turnActive = false;
        this._turnIdx = 0;
        this._enqueueCount = 0;
        this._nextTime = 0;
        this._playing = false;
        this._sources = [];
        this._delayTimer = null;
        this._pendingChunks = [];  // {resampled: Float32Array, raw: Float32Array}[]
        this._playbackGeneration = 0;
        this._resumeRequest = null;
        this._resumeOperation = null;
        this._adaptiveDelayMs = 0;
        this._learnedDelayMs = 0;
        this._generationLatencyMs = 0;
        this._audioDurationMs = 0;

        // Monitoring metrics
        this._firstChunkTime = 0;
        this._playbackStartTime = 0;
        this._playbackStartCtxTime = 0;
        this._gapCount = 0;
        this._totalShiftMs = 0;
        this._rebufferCount = 0;
        this._lastGapMs = 0;
        this._lastAheadMs = 0;
        this._lastArrivalTime = 0;
        this._aheadInterval = null;

        /**
         * Metrics callback: (data) => void
         * data shape: { ahead, gapCount, totalShift, rebufferCount, turn, pdelay? }
         * Called inside requestAnimationFrame for batched UI updates.
         */
        this.onMetrics = null;

        /**
         * Gap callback: (gapInfo) => void
         * gapInfo shape: { gap_idx, gap_ms, total_shift_ms, chunk_idx, turn }
         */
        this.onGap = null;

        /**
         * Raw audio callback for session recording.
         * Fires from _scheduleChunk with decoded PCM and the actual scheduled
         * playback time (performance.now()-based), NOT the arrival time.
         * (samples: Float32Array, sampleRate: number, playbackTimestamp: number) => void
         */
        this.onRawAudio = null;

        /**
         * Explicit playback cancellation callback for session recording.
         * Receives the performance.now() cutoff; normal turn endings preserve
         * queued audio and do not call it.
         */
        this.onPlaybackCancel = null;

        /**
         * Device recovery replaced the scheduled playback timeline. Consumers
         * must discard references to the old schedule before retained audio
         * is scheduled again. Recording cancellation has its own callback.
         */
        this.onPlaybackTimelineReset = null;

        /** Output-context recovery failed; queued PCM remains available to retry. */
        this.onPlaybackError = null;

        /**
         * Far-end reference callback for browser AEC diagnostics.  It fires
         * with raw (expected 24 kHz) samples and their scheduled playback
         * timestamp.  This is observational only; it never gates input.
         */
        this.onFarEndReference = null;
    }

    // Public read-only accessors
    get turnActive() { return this._turnActive; }
    get playing() { return this._playing; }
    get ctx() { return this._ctx; }
    get nextTime() { return this._nextTime; }
    get gapCount() { return this._gapCount; }
    get totalShiftMs() { return this._totalShiftMs; }
    get lastAheadMs() { return this._lastAheadMs; }
    get turnIdx() { return this._turnIdx; }
    get hasBufferedAudio() {
        if (this._pendingChunks.length > 0 || this._sources.length > 0) return true;
        return !!this._ctx && this._nextTime > this._ctx.currentTime + 0.01;
    }

    init() {
        if (!this._ctx || this._ctx.state === 'closed') {
            this._ctx = new AudioContext();
            this._outputSR = this._ctx.sampleRate;
            console.log(`[AudioPlayer] init: outputSR=${this._outputSR}`);
        }
        this._stopAllSources();
        this._turnActive = false;
        this._turnIdx = 0;
        this._enqueueCount = 0;
        this._nextTime = 0;
        this._playing = false;
        this._pendingChunks = [];
        if (this._delayTimer) { clearTimeout(this._delayTimer); this._delayTimer = null; }
    }

    /** Start a new SPEAK turn */
    beginTurn() {
        if (this._turnActive) return;
        // A model turn can finish before its queued audio has played. Keep
        // that timeline across normal turn boundaries so the next chunk
        // follows the previous tail; explicit stopAll/stop handles interrupts.
        const continuePlayback = this.hasBufferedAudio;
        const waitingForResume = !!this._resumeRequest;
        if (!continuePlayback) this._stopAllSources();
        else this._invalidatePlaybackStart();
        this._turnActive = true;
        this._turnIdx++;
        if (!this._playing) {
            this._nextTime = 0;
            this._playbackStartTime = 0;
            this._playbackStartCtxTime = 0;
        }
        if (!this._pendingChunks.length) this._firstChunkTime = 0;
        this._gapCount = 0;
        this._totalShiftMs = 0;
        this._rebufferCount = 0;
        this._lastGapMs = 0;
        this._lastAheadMs = this._playing
            ? Math.max(0, (this._nextTime - this._ctx.currentTime) * 1000)
            : 0;
        this._lastArrivalTime = 0;
        this._adaptiveDelayMs = Math.min(
            this._maxStartupDelayMs,
            Math.max(0, Number(this._getDelayMs()) || 0),
        );
        this._generationLatencyMs = 0;
        this._audioDurationMs = 0;
        if (this._delayTimer) { clearTimeout(this._delayTimer); this._delayTimer = null; }
        if (this._playing) this._startAheadMonitor();
        else if (waitingForResume && this._pendingChunks.length) this._startPlayback();
        else if (this._pendingChunks.length) this._armStartupTimer();
        console.log(`[AudioPlayer] === turn #${this._turnIdx} begin ===`);
    }

    /**
     * Enqueue a SPEAK audio chunk for playback.
     * @param {string} base64Data - Base64-encoded Float32 audio at outputSampleRate
     * @param {number} [arrivalTime] - performance.now() timestamp of arrival
     * @param {number} [generationLatencyMs] - diagnostic only
     * @param {number} [audioPaddingSamples] - leading transport padding at 24 kHz
     */
    playChunk(base64Data, arrivalTime, generationLatencyMs, audioPaddingSamples) {
        if (!base64Data || !this._ctx) return;
        const t0 = performance.now();

        const binary = atob(base64Data);
        const len = binary.length;
        const bytes = new Uint8Array(len);
        for (let i = 0; i < len; i += 1024) {
            const end = Math.min(i + 1024, len);
            for (let j = i; j < end; j++) bytes[j] = binary.charCodeAt(j);
        }
        const decoded = new Float32Array(bytes.buffer);
        const samples = trimAudioPadding(decoded, audioPaddingSamples);
        if (samples.length === 0) return;

        const resampled = resampleAudio(samples, this._outputSR_expected, this._outputSR);
        const raw = (this.onRawAudio || this.onFarEndReference) ? samples : null;
        const audioDurationMs = samples.length / this._outputSR_expected * 1000;
        this._enqueueCount++;
        if (Number.isFinite(Number(generationLatencyMs)) && Number(generationLatencyMs) > 0) {
            const sample = Number(generationLatencyMs);
            this._generationLatencyMs = this._generationLatencyMs > 0
                ? this._generationLatencyMs * 0.65 + sample * 0.35
                : sample;
        }
        this._audioDurationMs = this._audioDurationMs > 0
            ? this._audioDurationMs * 0.65 + audioDurationMs * 0.35
            : audioDurationMs;

        if (!this._firstChunkTime) {
            this._firstChunkTime = arrivalTime || t0;
        }
        this._lastArrivalTime = arrivalTime || t0;

        const interrupted = this._playing && this._ctx.state !== 'running';
        if (interrupted) this._retainInterruptedPlayback();
        if (this._playing) {
            this._scheduleChunk(resampled, raw);
            this._lastAheadMs = (this._nextTime - this._ctx.currentTime) * 1000;
            this._emitMetrics();
        } else {
            this._pendingChunks.push({ resampled, raw });
            this._adaptiveDelayMs = Math.max(
                this._adaptiveDelayMs,
                adaptivePlaybackDelayMs(
                    this._getDelayMs(),
                    generationLatencyMs,
                    audioDurationMs,
                    this._learnedDelayMs,
                    this._maxStartupDelayMs,
                ),
            );
            // Startup buffering already completed before a running output
            // context was interrupted. Recover its retained tail directly.
            if (interrupted) this._startPlayback();
            else this._armStartupTimer();
        }
    }

    _retainInterruptedPlayback() {
        const stoppedAt = this._ctx.currentTime;
        const playedUntilMs = this._playbackStartTime
            + (stoppedAt - this._playbackStartCtxTime) * 1000;
        const remaining = [];
        for (const source of this._sources) {
            const chunk = source._playbackChunk;
            if (!chunk) continue;
            const playedSeconds = Math.max(0, stoppedAt - chunk.startTime);
            const playedSamples = Math.min(chunk.samples.length,
                Math.floor(playedSeconds * this._outputSR));
            if (playedSamples === chunk.samples.length) continue;
            const rawOffset = chunk.rawSamples
                ? Math.min(chunk.rawSamples.length,
                    Math.floor(playedSeconds * this._outputSR_expected))
                : 0;
            remaining.push({
                resampled: chunk.samples.subarray(playedSamples),
                raw: chunk.rawSamples?.subarray(rawOffset) || null,
            });
        }
        remaining.push(...this._pendingChunks);
        this._stopAllSources();
        this._pendingChunks = remaining;
        if (this.onPlaybackTimelineReset) this.onPlaybackTimelineReset();
        // Previously scheduled recording entries used the old context clock.
        // Retain the audible prefix, then place this tail at its recovered
        // wall-clock time through the ordinary onRawAudio callback.
        if (this.onPlaybackCancel) this.onPlaybackCancel(playedUntilMs);
    }

    _armStartupTimer() {
        if (this._playing || this._resumeRequest || this._pendingChunks.length === 0) return;
        if (this._delayTimer) { clearTimeout(this._delayTimer); this._delayTimer = null; }
        const elapsed = this._firstChunkTime
            ? Math.max(0, performance.now() - this._firstChunkTime)
            : 0;
        const remaining = Math.max(0, this._adaptiveDelayMs - elapsed);
        if (remaining <= 0) {
            this._startPlayback();
            return;
        }
        this._delayTimer = setTimeout(() => {
            this._delayTimer = null;
            if (this._pendingChunks.length > 0) this._startPlayback();
        }, remaining);
    }

    /** Current SPEAK turn ended */
    endTurn() {
        if (!this._turnActive) return;
        if (!this._playing && this._pendingChunks.length > 0) {
            if (this._delayTimer) { clearTimeout(this._delayTimer); this._delayTimer = null; }
            this._startPlayback();
        }
        this._turnActive = false;
        this._stopAheadMonitor();
        if (this._gapCount > 0) {
            this._learnedDelayMs = Math.min(
                this._maxStartupDelayMs,
                Math.max(this._learnedDelayMs, Math.min(
                    this._maxStartupDelayMs,
                    this._adaptiveDelayMs + this._lastGapMs + 150,
                )),
            );
        } else {
            this._learnedDelayMs *= 0.75;
        }
        const ahead = this._playing
            ? ((this._nextTime - this._ctx.currentTime) * 1000).toFixed(0)
            : '0';
        console.log(`[AudioPlayer] === turn #${this._turnIdx} end (remaining=${ahead}ms) ===`);
    }

    _startPlayback() {
        if (this._playing || this._resumeRequest || !this._ctx || !this._pendingChunks.length) return;
        if (this._ctx.state !== 'running') {
            this._resumeBeforePlayback();
            return;
        }
        this._playing = true;
        this._playbackStartTime = performance.now();
        this._playbackStartCtxTime = this._ctx.currentTime;

        this._nextTime = this._ctx.currentTime;
        for (const chunk of this._pendingChunks) {
            this._scheduleChunk(chunk.resampled, chunk.raw);
        }
        this._pendingChunks = [];

        const pdelay = this._firstChunkTime ? (this._playbackStartTime - this._firstChunkTime) : 0;
        this._lastAheadMs = (this._nextTime - this._ctx.currentTime) * 1000;
        this._emitMetrics({
            pdelay,
            bufferTargetMs: this._adaptiveDelayMs,
            bufferedChunks: this._enqueueCount,
        });
        if (this._turnActive) this._startAheadMonitor();

        console.log(`[AudioPlayer] playback started (buffered=${this._lastAheadMs.toFixed(0)}ms, pdelay=${pdelay.toFixed(0)}ms, target=${this._adaptiveDelayMs.toFixed(0)}ms)`);
    }

    _resumeBeforePlayback() {
        const context = this._ctx;
        const request = { context, generation: this._playbackGeneration };
        this._resumeRequest = request;
        // A normal turn boundary or cancellation invalidates its scheduling
        // callback, but an AudioContext.resume() cannot itself be cancelled.
        // Reuse that operation if a fresh turn needs the same context.
        let operation = this._resumeOperation;
        if (!operation || operation.context !== context) {
            operation = { context, promise: null };
            this._resumeOperation = operation;
            try {
                operation.promise = Promise.resolve(context.resume());
            } catch (error) {
                operation.promise = Promise.reject(error);
            }
            const clearOperation = () => {
                if (this._resumeOperation === operation) this._resumeOperation = null;
            };
            operation.promise.then(clearOperation, clearOperation);
        }
        const isCurrent = () => this._resumeRequest === request
            && this._ctx === context
            && this._playbackGeneration === request.generation;
        operation.promise.then(() => {
            if (!isCurrent()) return;
            this._resumeRequest = null;
            if (context.state !== 'running') {
                this._reportPlaybackError(new Error(`Audio output is ${context.state} after resume`));
                return;
            }
            // Establish the recording/playback anchor only once audio time
            // can advance. Chunks arriving during recovery remain in order.
            this._startPlayback();
        }, error => {
            if (!isCurrent()) return;
            this._resumeRequest = null;
            this._reportPlaybackError(error);
        }).catch(error => {
            // Scheduling failures must not escape the asynchronous recovery
            // callback as an unhandled promise rejection.
            if (this._ctx === context && this._playbackGeneration === request.generation) {
                this._reportPlaybackError(error);
            }
        });
    }

    _reportPlaybackError(error) {
        const failure = error instanceof Error ? error : new Error(String(error));
        console.warn(`[AudioPlayer] Unable to resume audio output: ${failure.message}`);
        if (this.onPlaybackError) {
            try { this.onPlaybackError(failure); }
            catch (callbackError) { console.warn('[AudioPlayer] Playback error callback failed', callbackError); }
        }
    }

    _invalidatePlaybackStart() {
        this._playbackGeneration++;
        this._resumeRequest = null;
    }

    _scheduleChunk(samples, rawSamples) {
        const buffer = this._ctx.createBuffer(1, samples.length, this._outputSR);
        buffer.getChannelData(0).set(samples);
        const source = this._ctx.createBufferSource();
        source.buffer = buffer;
        source.connect(this._ctx.destination);

        const now = this._ctx.currentTime;
        if (this._nextTime < now) {
            const gapMs = (now - this._nextTime) * 1000;
            if (gapMs > 10) {
                this._gapCount++;
                this._rebufferCount++;
                this._lastGapMs = gapMs;
                this._totalShiftMs += gapMs;
                this._emitMetrics();
                if (this.onGap) {
                    const info = {
                        gap_idx: this._gapCount,
                        gap_ms: gapMs,
                        total_shift_ms: this._totalShiftMs,
                        chunk_idx: this._enqueueCount,
                        turn: this._turnIdx,
                    };
                    setTimeout(() => this.onGap(info), 0);
                }
            }
            this._nextTime = now;
        }

        if (this.onRawAudio && rawSamples) {
            const playbackMs = this._playbackStartTime +
                (this._nextTime - this._playbackStartCtxTime) * 1000;
            this.onRawAudio(rawSamples, this._outputSR_expected, playbackMs);
            if (this.onFarEndReference) {
                this.onFarEndReference(rawSamples, this._outputSR_expected, playbackMs);
            }
        } else if (this.onFarEndReference && rawSamples) {
            const playbackMs = this._playbackStartTime +
                (this._nextTime - this._playbackStartCtxTime) * 1000;
            this.onFarEndReference(rawSamples, this._outputSR_expected, playbackMs);
        }

        source._playbackChunk = { samples, rawSamples, startTime: this._nextTime };
        source.start(this._nextTime);
        this._nextTime += buffer.duration;

        this._sources.push(source);
        source.onended = () => {
            const idx = this._sources.indexOf(source);
            if (idx >= 0) this._sources.splice(idx, 1);
        };
    }

    /** Emit metrics via callback inside rAF */
    _emitMetrics(extra) {
        if (!this.onMetrics) return;
        const data = {
            ahead: this._lastAheadMs,
            gapCount: this._gapCount,
            totalShift: this._totalShiftMs,
            rebufferCount: this._rebufferCount,
            lastGapMs: this._lastGapMs,
            bufferTargetMs: this._adaptiveDelayMs,
            generationLatencyMs: this._generationLatencyMs,
            audioDurationMs: this._audioDurationMs,
            turn: this._turnIdx,
            ...extra,
        };
        requestAnimationFrame(() => {
            if (this.onMetrics) this.onMetrics(data);
        });
    }

    _startAheadMonitor() {
        this._stopAheadMonitor();
        this._aheadInterval = setInterval(() => {
            if (!this._playing || !this._ctx) {
                this._stopAheadMonitor();
                return;
            }
            const ahead = (this._nextTime - this._ctx.currentTime) * 1000;
            this._lastAheadMs = Math.max(0, ahead);
            this._emitMetrics();
        }, 200);
    }

    _stopAheadMonitor() {
        if (this._aheadInterval) { clearInterval(this._aheadInterval); this._aheadInterval = null; }
    }

    _stopAllSources() {
        this._invalidatePlaybackStart();
        if (this._delayTimer) { clearTimeout(this._delayTimer); this._delayTimer = null; }
        this._stopAheadMonitor();
        for (const src of this._sources) {
            try { src.stop(); } catch (_) {}
            try { src.disconnect(); } catch (_) {}
        }
        this._sources = [];
        this._playing = false;
        this._pendingChunks = [];
        this._nextTime = 0;
    }

    /** Stop all playback immediately (public API) */
    stopAll() {
        const now = performance.now();
        const frozenPlayback = this._playing && this._ctx
            && (this._ctx.state === 'suspended' || this._ctx.state === 'interrupted');
        // Audio time stops during an output interruption. Cancellation can
        // arrive before another chunk detects that interruption, so crop the
        // recording at the last audible sample rather than the later wall time.
        // A startup queue or a repeated stop has no active playback anchor.
        const cutoff = frozenPlayback
            ? Math.min(now, this._playbackStartTime
                + Math.max(0, this._ctx.currentTime - this._playbackStartCtxTime) * 1000)
            : now;
        this._stopAllSources();
        this._turnActive = false;
        if (this.onPlaybackCancel) this.onPlaybackCancel(cutoff);
    }

    /** Full session stop */
    stop() {
        console.log(`[AudioPlayer] session stop (${this._enqueueCount} chunks total)`);
        this.stopAll();
        const ctx = this._ctx;
        this._ctx = null;
        this._outputSR = 0;
        if (ctx && ctx.state !== 'closed') ctx.close().catch(() => {});
    }
}
