import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import vm from 'node:vm';
import test from 'node:test';

const root = new URL('../../Sources/MiniCPMDemoBackend/Resources/Web/', import.meta.url);
const text = file => readFileSync(new URL(file, root), 'utf8');
const app = text('audio-duplex/audio-duplex-app.js');

function fileHarness() {
    let now = 0;
    let nextTimer = 0;
    const timers = new Map();
    const audio = { paused: true, currentTime: 0, play() { this.paused = false; }, pause() { this.paused = true; } };
    const context = {
        Float32Array, SAMPLE_RATE_IN: 16000, CHUNK_MS: 1000, performance: { now: () => now },
        FILE_MAX_DURATION: 300, URL: { createObjectURL: () => 'file-source', revokeObjectURL() {} },
        setTimeout(fn, delay) { const id = ++nextTimer; timers.set(id, { fn, at: now + delay }); return id; },
        clearTimeout(id) { timers.delete(id); },
        document: { getElementById: () => audio }, addSystemLog() {},
    };
    vm.runInNewContext(text('duplex/lib/file-response-drain.js').replace('export class', 'class')
        + '\nthis.FileResponseDrain = FileResponseDrain;', context);
    const begin = app.indexOf('class FileAudioProvider');
    const end = app.indexOf('// ============================================================================', begin);
    vm.runInNewContext(app.slice(begin, end) + '\nthis.FileAudioProvider = FileAudioProvider;', context);
    const provider = new context.FileAudioProvider({});
    provider.running = true;
    provider._allAudio = Array.from({ length: 12 }, (_, index) => new Float32Array([index + 1]));
    provider._mainEnd = provider._grandTotal = 12;
    return {
        provider, audio, timers,
        advance(ms) {
            const target = now + ms;
            while (true) {
                const next = [...timers].sort((a, b) => a[1].at - b[1].at)[0];
                if (!next || next[1].at > target) break;
                now = next[1].at;
                timers.delete(next[0]); next[1].fn();
            }
            now = target;
        },
    };
}

test('synchronous input hold freezes file position and resumes with the next unconsumed packet', () => {
    const h = fileHarness();
    const packets = [];
    h.provider.onChunk = chunk => {
        packets.push(chunk.audio[0]);
        if (packets.length === 4) h.provider.pause();
    };
    h.provider._feedNext(); h.advance(3000);
    assert.deepEqual(packets, [1, 2, 3, 4]);
    assert.equal(h.provider.paused, true);
    assert.equal(h.audio.paused, true);
    assert.equal(h.timers.size, 0, 'the producer must not rearm after its callback paused it');
    h.advance(60000);
    assert.equal(h.provider._chunkIdx, 4);
    h.provider.resume(); h.advance(2000);
    assert.deepEqual(packets, [1, 2, 3, 4, 5, 6, 7]);
});

test('file EOF waits for committed input and does not spend response grace on unprocessed packets', () => {
    const h = fileHarness();
    h.provider._chunkIdx = h.provider._grandTotal;
    let pending = true;
    let ended = false;
    let sent = 0;
    h.provider.getInputPending = () => pending;
    h.provider.onEnd = () => { ended = true; };
    h.provider.onChunk = () => { sent++; pending = true; };
    h.provider._feedNext(); h.advance(30000);
    assert.equal(ended, false);
    assert.equal(sent, 0);
    assert.equal(h.provider._responseDrain.chunks, 0);
    pending = false; h.advance(1000);
    assert.equal(sent, 1);
    h.advance(20000);
    assert.equal(sent, 1, 'do not create more drain ticks before the last tick commits');
});

test('input flow suspends capture without stopping output or overriding user pause', async () => {
    const calls = [];
    const context = {
        session: { _sessionReady: true, running: true, inputFlowHeld: true, paused: false },
        media: { pause() { calls.push('file.pause'); }, resume() { calls.push('file.resume'); } },
        audioCtxIn: { state: 'running', suspend() { calls.push('mic.suspend'); return Promise.resolve(); },
            resume() { calls.push('mic.resume'); return Promise.resolve(); } },
        currentMode: 'live', setStatusLamp(state) { calls.push(state); },
        document: { querySelector: () => ({}) }, addSystemLog() {},
    };
    vm.runInNewContext(app.slice(app.indexOf('function syncInputCaptureFlow'),
        app.indexOf('function buildDuplexSystemPrompt')), context);
    context.syncInputCaptureFlow();
    assert.deepEqual(calls, ['file.pause', 'mic.suspend', 'preparing']);
    context.session.inputFlowHeld = false; context.session.paused = true;
    calls.length = 0; context.syncInputCaptureFlow();
    assert.deepEqual(calls, ['file.pause', 'mic.suspend', 'preparing']);
    context.session.paused = false;
    calls.length = 0; context.syncInputCaptureFlow();
    assert.deepEqual(calls, ['file.resume', 'mic.resume', 'live']);
    context.session._sessionReady = false;
    calls.length = 0; context.syncInputCaptureFlow();
    assert.deepEqual(calls, [], 'cleanup must not restart a source');
});

test('reconnect readiness resumes capture suspended by an old input hold and resets startup gating', () => {
    const calls = [];
    const context = {
        session: { _sessionReady: false, running: true, inputFlowHeld: false, paused: false },
        media: null, sessionRecorder: null,
        captureNodeLive: { port: { postMessage(message) { calls.push(message); } } },
        audioCtxIn: { state: 'suspended', resume() { calls.push('resume'); return Promise.resolve(); },
            suspend() { calls.push('suspend'); return Promise.resolve(); } },
        currentMode: 'live', liveInputTimeline: { reset() { calls.push('reset'); } },
        setStatusLamp() {}, document: { querySelector: () => ({}) }, addSystemLog() {},
    };
    vm.runInNewContext(app.slice(app.indexOf('function syncInputCaptureFlow'),
        app.indexOf('function buildDuplexSystemPrompt')), context);
    const begin = app.indexOf('session.onReconnectState =');
    vm.runInNewContext(app.slice(begin, app.indexOf('session.onPauseStateChange =', begin)), context);
    context.session.onReconnectState('reconnecting');
    assert.deepEqual(calls, ['suspend']);
    calls.length = 0;
    context.session._sessionReady = true;
    context.session.onReconnectState('connected');
    assert.equal(calls[0], 'reset');
    assert.equal(calls[1].command, 'start');
    assert.equal(calls[1].captureEpoch, 1);
    assert.equal(calls[2], 'resume');
});

test('cleanup pause-state callback cannot enable the stopped page pause button', () => {
    const calls = [];
    const context = { session: { running: false },
        setDefaultPauseBtnState() { calls.push('enabled'); } };
    const begin = app.indexOf('session.onPauseStateChange =');
    vm.runInNewContext(app.slice(begin, app.indexOf('session.onQueueUpdate =', begin)), context);
    context.session.onPauseStateChange('active');
    assert.deepEqual(calls, []);
});

test('file preparation starts recording before the synchronous first packet', async () => {
    const h = fileHarness();
    const calls = [];
    h.provider._getAudioDuration = async () => 1;
    h.provider._decodeFileAudio = async () => {};
    h.provider._buildPaddedArrays = () => {};
    h.provider._feedNext = () => calls.push('first packet');
    h.provider.onReady = () => calls.push('recording started');
    await h.provider.start();
    assert.deepEqual(calls, ['recording started', 'first packet']);
});

test('stopped file preparation cannot revive capture or replace a new playback source', async () => {
    const h = fileHarness();
    let finishDuration;
    h.provider._disconnectMic = () => {};
    h.provider._getAudioDuration = () => new Promise(resolve => { finishDuration = resolve; });
    let decoded = false;
    h.provider._decodeFileAudio = async () => { decoded = true; };
    const start = h.provider.start();
    h.provider.stop(); h.audio.src = 'new-source';
    finishDuration(1);
    await assert.rejects(start, /cancelled/);
    assert.equal(decoded, false);
    assert.equal(h.provider.running, false);
    assert.equal(h.audio.src, 'new-source');
});

test('late live worklet packets cannot enter a replacement source with the same epoch', () => {
    const node = { _captureEpoch: 1, port: {} };
    const recorded = [];
    const context = {
        node, captureNodeLive: node,
        session: { running: true, paused: false, prepareInputChunk: chunk => chunk, onMetrics() {} },
        liveInputTimeline: { prepare: () => ({ send: false }) }, rms: () => 0, MIC_ACTIVITY_RMS: 0.02,
        sessionRecorder: { pushLeft: chunk => recorded.push(chunk[0]) },
    };
    const begin = app.indexOf('node.port.onmessage =');
    vm.runInNewContext(app.slice(begin, app.indexOf('    startWaveformDrawing();', begin)), context);
    const chunk = { data: { type: 'chunk', captureEpoch: 1, audio: new Float32Array([1]) } };
    node.port.onmessage(chunk);
    context.captureNodeLive = { _captureEpoch: 1, port: {} };
    node.port.onmessage(chunk);
    assert.deepEqual(recorded, [1]);
    context.captureNodeLive = node;
    node._captureEpoch = 2;
    node.port.onmessage(chunk);
    assert.deepEqual(recorded, [1], 'prior complete packets must be rejected after timeline reset');
});

test('manual resume records the synchronous first file packet before capture reopens', () => {
    const h = fileHarness();
    const context = {
        console: { log() {} }, performance: { now: () => 0 },
        Float32Array, Blob, ArrayBuffer, DataView, resampleAudio: samples => samples,
    };
    vm.runInNewContext(text('duplex/lib/session-recorder.js')
        .replace(/^import .*;$/m, '').replace('export class', 'class')
        + '\nthis.SessionRecorder = SessionRecorder;', context);
    const recorder = new context.SessionRecorder();
    recorder.start(); recorder.pause(); h.provider.paused = true;
    h.provider.onChunk = chunk => recorder.pushLeft(chunk.audio);
    Object.assign(context, {
        session: { running: true }, sessionRecorder: recorder,
        setDefaultPauseBtnState() {}, resetLiveCaptureTimeline() {},
        syncInputCaptureFlow: () => h.provider.resume(),
    });
    const begin = app.indexOf('session.onPauseStateChange =');
    vm.runInNewContext(app.slice(begin, app.indexOf('session.onQueueUpdate =', begin)), context);
    context.session.onPauseStateChange('active');
    assert.equal(h.provider._chunkIdx, 1);
    assert.equal(recorder._leftChunks.length, 1);
    assert.equal(recorder._leftChunks[0][0], 1);
});

test('mixed EOF freezes its graph while queued file input finishes and respects user pause', () => {
    const h = fileHarness();
    h.provider._audioMode = 'mixed';
    h.provider._chunkIdx = h.provider._grandTotal;
    h.provider._micCtx = { state: 'running', suspend() { this.state = 'suspended'; },
        resume() { this.state = 'running'; } };
    h.provider._disconnectMic = () => {};
    let pending = true;
    let canResume = true;
    let ended = false;
    const captured = [];
    const holdChanges = [];
    h.provider.getInputPending = () => pending;
    h.provider.canResumeInput = () => canResume;
    h.provider.getResponseState = () => 'end_of_turn';
    h.provider.onEnd = () => { ended = true; };
    h.provider.onChunk = chunk => captured.push(chunk.audio[0]);
    h.provider.onInputHoldChange = () => holdChanges.push(h.provider._waitingForInputDrain);
    h.provider._handleMicChunk(new Float32Array(16000).fill(1));
    h.provider._handleMicChunk(new Float32Array(16000).fill(2));
    h.advance(15000);
    assert.equal(h.provider._micCtx.state, 'suspended');
    assert.equal(ended, false);
    assert.deepEqual(captured, [1, 2], 'preserve complete packets posted before context suspension');
    pending = false; canResume = false;
    h.advance(10000);
    assert.equal(ended, false);
    canResume = true; h.advance(1000);
    assert.equal(ended, true);
    assert.deepEqual(holdChanges, [true, false], 'recording must follow the EOF capture hold');
});

test('quick mixed pause and resume queues both context operations despite stale async state', async () => {
    const h = fileHarness();
    h.provider._audioMode = 'mixed';
    const calls = [];
    let operations = Promise.resolve();
    const ctx = { state: 'running',
        suspend() { calls.push('suspend'); operations = operations.then(() => { this.state = 'suspended'; }); return operations; },
        resume() { calls.push('resume'); operations = operations.then(() => { this.state = 'running'; }); return operations; } };
    h.provider._micCtx = ctx;
    h.provider.pause(); h.provider.resume();
    await operations;
    assert.deepEqual(calls, ['suspend', 'resume']);
    assert.equal(ctx.state, 'running');
    assert.equal(h.provider.paused, false);
});

test('input-only holds keep assistant recording and leave an aligned silent capture gap', async () => {
    let now = 1000;
    const context = { Float32Array, Uint8Array, DataView, ArrayBuffer, Blob,
        resampleAudio: x => x, performance: { now: () => now }, console: { log() {}, warn() {} } };
    vm.runInNewContext(text('duplex/lib/session-recorder.js').replace(/^import .*;$/gm, '')
        .replace('export class', 'class') + '\nthis.SessionRecorder = SessionRecorder;', context);
    const recorder = new context.SessionRecorder(10, 10);
    recorder.start(); recorder.pushLeft(new Float32Array(10).fill(1));
    now += 1000; recorder.pauseInputCapture();
    now += 2000; recorder.pushRight(new Float32Array(10).fill(0.5), 10, now);
    recorder.pause(); now += 10000; recorder.resume();
    now += 1000; recorder.resumeInputCapture();
    recorder.pushLeft(new Float32Array(10).fill(0.25)); now += 1000;
    const result = recorder.stop();
    assert.equal(result.leftSamples, 50, '3s capture hold, excluding 10s explicit session pause');
    assert.equal(result.rightSamples, 10, 'assistant audio must survive the input hold');
    const bytes = new DataView(await result.blob.arrayBuffer());
    assert.equal(bytes.getInt16(44, true), 32767);
    assert.equal(bytes.getInt16(44 + 10 * 4, true), 0);
    assert.equal(bytes.getInt16(44 + 30 * 4 + 2, true), 16383);
    assert.equal(bytes.getInt16(44 + 40 * 4, true), 8191);
});

test('already posted capture received during a hold is not counted twice in the recording gap', () => {
    let now = 0;
    const context = { Float32Array, Uint8Array, DataView, ArrayBuffer, Blob,
        resampleAudio: x => x, performance: { now: () => now }, console: { log() {}, warn() {} } };
    vm.runInNewContext(text('duplex/lib/session-recorder.js').replace(/^import .*;$/gm, '')
        .replace('export class', 'class') + '\nthis.SessionRecorder = SessionRecorder;', context);
    const recorder = new context.SessionRecorder(10, 10);
    recorder.start();
    now = 1000; recorder.pushLeft(new Float32Array(10)); recorder.pauseInputCapture();
    now = 2000; recorder.pushLeft(new Float32Array(10));
    now = 4000; recorder.resumeInputCapture();
    assert.equal(recorder.stop().leftSamples, 40);
});
