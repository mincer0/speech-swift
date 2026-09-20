/**
 * capture-processor.js — AudioWorklet processor for mixing + capturing audio
 *
 * Runs on the audio rendering thread. Receives mixed audio from the graph
 * (browser auto-sums all connected inputs), passes it through to output
 * (for MediaStreamDestination + monitor), and posts 1-second PCM chunks
 * to the main thread via MessagePort.  Browser AudioContext constructors are
 * allowed to ignore a requested sampleRate, so this processor also performs
 * a small streaming linear resample to the requested target rate.  The wire
 * protocol therefore always receives real 16 kHz PCM rather than “16,000
 * samples at whatever rate the browser selected”.
 *
 * Commands via port.postMessage:
 *   { command: 'start' }  — begin accumulating and emitting chunks
 *   { command: 'stop' }   — stop accumulating, flush buffer
 *
 * Emits via port.postMessage:
 *   { type: 'chunk', audio: Float32Array }  — 1-second PCM chunk (Transferable)
 */
class CaptureProcessor extends AudioWorkletProcessor {
    constructor(options) {
        super();
        const processorOptions = options.processorOptions || {};
        this._chunkSize = processorOptions.chunkSize || 16000;
        this._targetRate = processorOptions.targetSampleRate || 16000;
        // `sampleRate` is the AudioWorkletGlobalScope's actual input rate;
        // unlike AudioContext({sampleRate}), this value cannot be ignored.
        this._sourceRate = sampleRate;
        this._ratio = Math.max(1e-6, this._sourceRate / this._targetRate);
        // Streaming linear-resampler state.  Keeping only the previous source
        // sample and a phase avoids concatenating/slicing a Float32Array on
        // every 128-frame render quantum (which would trigger GC on the audio
        // rendering thread).
        this._previousSample = 0;
        this._hasPrevious = false;
        this._phase = 0;
        // The only steady-state allocation is one target-rate chunk.  It is
        // transferred to the main thread and replaced only at chunk boundary.
        this._chunkBuffer = new Float32Array(this._chunkSize);
        this._chunkPos = 0;
        this._active = false;

        this.port.onmessage = (e) => {
            const { command } = e.data;
            if (command === 'start') {
                this._active = true;
                this._previousSample = 0;
                this._hasPrevious = false;
                this._phase = 0;
                this._chunkBuffer = new Float32Array(this._chunkSize);
                this._chunkPos = 0;
            } else if (command === 'stop') {
                if (this._chunkPos > 0) {
                    const remaining = this._chunkBuffer.slice(0, this._chunkPos);
                    this.port.postMessage(
                        { type: 'chunk', audio: remaining, final: true },
                        [remaining.buffer]
                    );
                }
                this._active = false;
                this._chunkBuffer = new Float32Array(this._chunkSize);
                this._chunkPos = 0;
                this._previousSample = 0;
                this._hasPrevious = false;
                this._phase = 0;
            }
        };
    }

    process(inputs, outputs) {
        const input = inputs[0]?.[0];
        const output = outputs[0]?.[0];

        // Always pass-through: enables MediaStreamDestination + monitor downstream
        if (input && output) {
            output.set(input);
        }

        if (!this._active || !input || input.length === 0) {
            return true;
        }

        // Resample before accumulating.  At equal rates this is effectively
        // a copy; at 48/44.1 kHz it emits 16 kHz samples with phase carried
        // across render quanta, avoiding boundary clicks and drift.
        this._appendResampled(input);

        return true;
    }

    _appendResampled(input) {
        let start = 0;
        if (!this._hasPrevious) {
            if (input.length === 0) return;
            this._previousSample = input[0];
            this._hasPrevious = true;
            this._phase = 0;
            start = 1;
        }

        for (let i = start; i < input.length; i++) {
            const current = input[i];
            // Target samples are placed at `phase` source samples into the
            // [previous,current] interval.  `phase` may be >1 for downsample
            // ratios; in that case this interval contributes no output.
            while (this._phase <= 1) {
                const fraction = this._phase;
                this._appendTargetSample(
                    this._previousSample
                        + (current - this._previousSample) * fraction,
                );
                this._phase += this._ratio;
            }
            this._phase -= 1;
            this._previousSample = current;
        }
    }

    _appendTargetSample(value) {
        this._chunkBuffer[this._chunkPos++] = value;
        if (this._chunkPos === this._chunkSize) {
            const chunk = this._chunkBuffer;
            this.port.postMessage({ type: 'chunk', audio: chunk }, [chunk.buffer]);
            // The transferred ArrayBuffer is detached; allocate exactly one
            // replacement for the next chunk, never per render quantum.
            this._chunkBuffer = new Float32Array(this._chunkSize);
            this._chunkPos = 0;
        }
    }
}

registerProcessor('capture-processor', CaptureProcessor);
