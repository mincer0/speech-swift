import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import vm from 'node:vm';
import test from 'node:test';

const web = new URL('../../Sources/MiniCPMDemoBackend/Resources/Web/', import.meta.url);
const source = file => readFileSync(new URL(file, web), 'utf8');
const app = source('omni/omni-app.js');

function harness() {
    let now = 0;
    let timerID = 0;
    let playingAt = null;
    let videoPosition = 0;
    const timers = new Map();
    const captureCommands = [];
    const video = {
        paused: true, style: {},
        get currentTime() { return videoPosition + (playingAt === null ? 0 : (now - playingAt) / 1000); },
        set currentTime(value) { videoPosition = value; playingAt = this.paused ? null : now; },
        play() { if (this.paused) playingAt = now; this.paused = false; return Promise.resolve(); },
        pause() { videoPosition = this.currentTime; playingAt = null; this.paused = true; },
    };
    const genericElement = { style: {}, classList: { add() {}, remove() {}, toggle() {} } };
    const canvas = { getContext: () => ({}), style: {} };
    const context = {
        Float32Array, Uint8Array, SAMPLE_RATE_IN: 16000, SAMPLE_RATE_OUT: 24000, CHUNK_MS: 1000,
        performance: { now: () => now },
        setTimeout(fn, delay = 0) {
            const id = ++timerID;
            timers.set(id, { fn, at: now + delay });
            return id;
        },
        clearTimeout: id => timers.delete(id),
        requestAnimationFrame: fn => fn(),
        document: {
            getElementById: id => id === 'videoEl' ? video : id === 'frameCanvas' ? canvas : genericElement,
            querySelector: () => ({ textContent: '' }),
        },
        URL: { createObjectURL: () => 'blob:test', revokeObjectURL() {} },
        console: { warn() {} }, addSystemEntry() {},
        setStatusLamp() {}, setPauseBtnState() {}, setButtonStates() {},
        _omniStartWaveform() {}, _omniStopWaveform() {},
        currentMode: 'file', session: null, media: null, sessionRecorder: null,
        navigator: { mediaDevices: { getUserMedia: async () => ({ getTracks: () => [] }) } },
        omniDeviceSelector: { getSelectedMicId: () => null },
        AudioContext: class {
            constructor() { this.state = 'running'; this.audioWorklet = { addModule: async () => {} }; }
        },
        AudioPlayer: class {
            constructor() { this.turnActive = false; }
            stopAll() {}
            stop() {}
            endTurn() {}
        },
        FarEndEchoReference: class { reset() {} analyzeInput() { return {}; } },
        taskContinuationForAssistantText: () => null,
        WebSocket: { OPEN: 1 },
    };
    vm.runInNewContext(source('duplex/lib/file-response-drain.js')
        .replace('export class', 'class') + '\nthis.FileResponseDrain = FileResponseDrain;', context);
    const begin = app.indexOf('class MediaProvider');
    const end = app.indexOf('// ============================================================================', app.indexOf('class FileMediaProvider'));
    vm.runInNewContext(app.slice(begin, end)
        + '\nthis.FileMediaProvider = FileMediaProvider; this.LiveMediaProvider = LiveMediaProvider;', context);
    vm.runInNewContext(app.slice(app.indexOf('function syncOmniRecorderInputFlow'),
        app.indexOf('// 排队倒计时')), context);
    vm.runInNewContext(source('duplex/lib/realtime-session.js').replace(/^import .*;$/gm, '')
        .replace('export class RealtimeSession', 'class RealtimeSession')
        + '\nthis.RealtimeSession = RealtimeSession;', context);
    const file = new context.FileMediaProvider({});
    file.running = true;
    file._allAudio = Array.from({ length: 12 }, (_, index) => new Float32Array([index + 1]));
    file._allFrames = Array.from({ length: 12 }, (_, index) => `frame-${index + 1}`);
    file._mainEnd = file._grandTotal = 12;
    file._lastFrame = 'frame-12';
    context.media = file;
    return {
        context, file, video, timers, captureCommands,
        session(config = {}) {
            const session = new context.RealtimeSession('omni-test', { enableInputFlowControl: true, ...config });
            session._started = true;
            session._cleaned = false;
            session._sessionReady = true;
            session._usesInputCommitted = true;
            session.ws = { readyState: 1, send() {}, close() {} };
            context.session = session;
            context.ownerSession = session;
            context.ownerMedia = context.media;
            const hookBegin = app.indexOf('session.onInputFlowChange = () => {');
            const hookEnd = app.indexOf('session.onForceListenChange', hookBegin);
            vm.runInNewContext(app.slice(hookBegin, hookEnd), context);
            return session;
        },
        advance(ms) {
            const target = now + ms;
            let iterations = 0;
            while (true) {
                const next = [...timers].filter(([, timer]) => timer.at <= target)
                    .sort((a, b) => a[1].at - b[1].at || a[0] - b[0])[0];
                if (!next) break;
                assert.ok(++iterations < 10000, 'timer loop did not make progress');
                const [id, timer] = next;
                timers.delete(id);
                now = timer.at;
                timer.fn();
            }
            now = target;
        },
    };
}

test('omni file/video FIFO holds at four pending packets against a five-second worker, preserving frames and video position', () => {
    const h = harness();
    const session = h.session();
    const queue = [];
    const sent = [];
    const committed = [];
    let busy = false;
    let ended = false;
    let heldAt = null;
    let maxPending = 0;
    const run = () => {
        if (busy || !queue.length) return;
        busy = true;
        h.context.setTimeout(() => {
            const packet = queue.shift();
            committed.push(packet.input.audio);
            busy = false;
            session._handleMessage({ type: 'input.committed', input_id: packet.message_id });
            run();
        }, 5000);
    };
    session.ws.send = payload => {
        const event = JSON.parse(payload);
        sent.push(event);
        if (event.type === 'input.append') { queue.push(event); run(); }
    };
    const onFlow = session.onInputFlowChange;
    session.onInputFlowChange = state => {
        onFlow(state);
        if (state === 'held' && heldAt === null) heldAt = h.video.currentTime;
    };
    h.file.getInputPending = () => session._recentInputs.size > 0 || session._pendingInputs.length > 0;
    h.file.onChunk = chunk => {
        session.sendChunk({ audio_base64: `pcm-${chunk.audio[0]}`, frame_base64_list: [chunk.frameBase64] });
        maxPending = Math.max(maxPending, session._pendingInputs.length);
    };
    h.file.onEnd = () => { ended = true; };
    h.file._feedNext();
    h.advance(10000);
    assert.equal(session.inputFlowHeld, true);
    assert.equal(h.video.currentTime, heldAt, 'input hold must freeze the source video clock');
    h.advance(150000);
    const realFileInputs = sent.filter(event => event.input?.audio !== 'pcm-0');
    assert.deepEqual(realFileInputs.map(event => event.input.audio),
        Array.from({ length: 12 }, (_, index) => `pcm-${index + 1}`));
    assert.deepEqual(realFileInputs.map(event => event.input.video_frames[0]),
        Array.from({ length: 12 }, (_, index) => `frame-${index + 1}`));
    assert.deepEqual(committed, sent.map(event => event.input.audio));
    assert.ok(sent.every(event => event.type === 'input.append'), 'holding input must not cancel the response');
    assert.equal(maxPending, 4);
    assert.equal(session.paused, false);
    assert.equal(ended, true, 'EOF should finish after all file and response grace packets commit');
});

test('video EOF waits for committed input, resumes its poll after manual pause, and retains playback drain', () => {
    const h = harness();
    h.file._chunkIdx = h.file._grandTotal;
    let pending = true;
    let ended = false;
    const packets = [];
    h.file.getInputPending = () => pending;
    h.file.getResponseState = () => 'end_of_turn';
    h.file.getPlaybackActive = () => true;
    h.file.onChunk = packet => packets.push(packet);
    h.file.onEnd = () => { ended = true; };
    h.file._feedNext();
    h.advance(10000);
    assert.equal(ended, false);
    assert.equal(packets.length, 0);
    h.file.pause(); h.advance(10000);
    pending = false;
    h.file.resume();
    assert.equal(packets.length, 1);
    assert.equal(packets[0].frameBase64, 'frame-12');
    assert.equal(ended, false, 'assistant buffered audio still needs playback time');
    h.file.getPlaybackActive = () => false;
    h.advance(1000);
    assert.equal(ended, true);
});

test('mixed EOF preserves posted packets once and waits for input/flow/user pause before resuming', () => {
    const h = harness();
    h.file._audioMode = 'mixed';
    h.file._chunkIdx = h.file._grandTotal;
    h.file._disconnectMic = () => {};
    h.file._micCtx = { state: 'running', suspend() { this.state = 'suspended'; }, resume() { this.state = 'running'; } };
    let pending = true;
    let mayResume = false;
    let ended = false;
    const packets = [];
    h.file.getInputPending = () => pending;
    h.file.canResumeInput = () => mayResume;
    h.file.getResponseState = () => 'end_of_turn';
    h.file.onChunk = packet => packets.push(packet.audio[0]);
    h.file.onEnd = () => { ended = true; };
    h.file._handleMicChunk(new Float32Array([1]));
    h.file._handleMicChunk(new Float32Array([2]));
    h.advance(10000);
    assert.deepEqual(packets, [1, 2]);
    assert.equal(h.file._micCtx.state, 'suspended');
    pending = false; h.advance(10000);
    assert.equal(ended, false);
    h.file.resume();
    assert.equal(h.file.paused, true, 'a waiting mixed source must not override user/flow pause');
    mayResume = true; h.advance(1000);
    assert.equal(ended, true);
    assert.deepEqual(packets, [1, 2], 'a completed response must not replay the deferred microphone packet');
});

test('manual resume restores the recorder before the first synchronous file packet', () => {
    const h = harness();
    const session = h.session();
    let recorderPaused = true;
    const calls = [];
    h.context.sessionRecorder = {
        pause() { recorderPaused = true; },
        resume() { recorderPaused = false; calls.push('recorder.resume'); },
    };
    h.file.paused = true;
    h.file.onChunk = () => {
        calls.push('file.packet');
        assert.equal(recorderPaused, false, 'first packet after resume must remain in the recording');
    };
    session.onPauseStateChange('active');
    assert.deepEqual(calls.slice(0, 2), ['recorder.resume', 'file.packet']);
});

test('live and mixed rapid pause/resume queue both asynchronous context operations', async () => {
    for (const type of ['live', 'mixed']) {
        const h = harness();
        const provider = type === 'live' ? new h.context.LiveMediaProvider() : h.file;
        provider.running = true;
        provider._audioMode = 'mixed';
        let operations = Promise.resolve();
        const calls = [];
        const ctx = { state: 'running',
            suspend() { calls.push('suspend'); operations = operations.then(() => { this.state = 'suspended'; }); return operations; },
            resume() { calls.push('resume'); operations = operations.then(() => { this.state = 'running'; }); return operations; } };
        if (type === 'live') provider._audioCtx = ctx;
        else provider._micCtx = ctx;
        provider.pause(); provider.resume();
        await operations;
        assert.deepEqual(calls, ['suspend', 'resume']);
        assert.equal(ctx.state, 'running');
    }
});

test('reconnect readiness preserves manual pause and resets worklet partial input before capture resumes', () => {
    const h = harness();
    const session = h.session();
    const calls = [];
    h.context.media = {
        pause() { calls.push('capture.pause'); },
        resume() { calls.push('capture.resume'); },
        resetCaptureTimeline() { calls.push('capture.reset'); },
    };
    h.context.sessionRecorder = {
        pauseInputCapture() { calls.push('record.input.hold'); },
        resumeInputCapture() { calls.push('record.input.resume'); },
    };
    session._sessionReady = false;
    session.onReconnectState('reconnecting');
    assert.deepEqual(calls, ['record.input.hold', 'capture.pause']);
    session._sessionReady = true;
    session.paused = true;
    calls.length = 0;
    session.onReconnectState('connected');
    assert.deepEqual(calls, ['capture.reset', 'record.input.resume', 'capture.pause']);
    session.paused = false;
    calls.length = 0;
    session.onReconnectState('connected');
    assert.deepEqual(calls, ['capture.reset', 'record.input.resume', 'capture.resume']);
});

test('file and mixed sources await recording readiness before advancing their first source clock', async () => {
    for (const type of ['video', 'mixed']) {
        const h = harness();
        h.file.running = false;
        h.file._audioMode = type;
        h.file._getVideoDuration = async () => 12;
        h.file._extractFrames = async () => Array.from({ length: 12 }, (_, index) => `frame-${index + 1}`);
        h.file._decodeVideoAudio = async () => { h.file._videoAudioChunks = h.file._allAudio; };
        h.file._setupMic = async () => {};
        const calls = [];
        h.file._connectMic = () => {
            calls.push('prepare.graph');
            h.file._captureNode = { port: { postMessage: () => calls.push('capture.start') } };
            h.file._fileSrcNode = { start: () => calls.push('file.start') };
        };
        let ready;
        h.file.onReady = () => new Promise(resolve => { ready = () => { calls.push('recorder.ready'); resolve(); }; });
        h.file.onChunk = () => calls.push('file.packet');
        const start = h.file.start();
        await new Promise(resolve => setImmediate(resolve));
        assert.equal(typeof ready, 'function');
        assert.ok(!calls.includes('file.packet') && !calls.includes('file.start') && !calls.includes('capture.start'));
        assert.equal(h.video.currentTime, 0);
        ready(); await start;
        const first = type === 'video' ? 'file.packet' : 'capture.start';
        assert.ok(calls.indexOf('recorder.ready') < calls.indexOf(first));
    }
});

test('stopping a provider during recording preparation cannot restart capture when readiness completes', async () => {
    const h = harness();
    h.file.running = false;
    h.file._getVideoDuration = async () => 12;
    h.file._extractFrames = async () => h.file._allFrames;
    h.file._decodeVideoAudio = async () => { h.file._videoAudioChunks = h.file._allAudio; };
    h.file._disconnectMic = () => {};
    let ready;
    h.file.onReady = () => new Promise(resolve => { ready = resolve; });
    let sent = 0;
    h.file.onChunk = () => { sent++; };
    const start = h.file.start();
    await new Promise(resolve => setImmediate(resolve));
    h.file.stop(); ready(); await start;
    assert.equal(sent, 0);
    assert.equal(h.file.running, false);
});

test('a delayed old media preparation retains its original ownership and cannot start a new session recorder', async () => {
    const h = harness();
    const ownerSession = {};
    const ownerMedia = h.file;
    let recorderStarts = 0;
    h.context.ownerSession = ownerSession;
    h.context.ownerMedia = ownerMedia;
    h.context.session = ownerSession;
    h.context.media = ownerMedia;
    h.context.sessionRecorder = { async start() { recorderStarts++; } };
    const begin = app.indexOf('ownerMedia.onReady = async () => {');
    const end = app.indexOf('await ownerMedia.start();', begin);
    vm.runInNewContext(app.slice(begin, end), h.context);
    const freshSession = {};
    const freshMedia = {};
    h.context.session = freshSession;
    h.context.media = freshMedia;
    await assert.rejects(ownerMedia.onReady(), /Media preparation cancelled/);
    assert.equal(recorderStarts, 0);
    assert.equal(ownerMedia.running, false);
    assert.equal(h.context.session, freshSession);
    assert.equal(h.context.media, freshMedia);
});

test('recording readiness cancelled midway closes its local recorder without disturbing the new owner', async () => {
    const h = harness();
    const ownerSession = {};
    h.context.ownerSession = ownerSession;
    h.context.ownerMedia = h.file;
    h.context.session = ownerSession;
    h.context.media = h.file;
    h.context.recordingSettings = null;
    let ready;
    let stopCount = 0;
    const recorder = {
        start() { return new Promise(resolve => { ready = resolve; }); },
        async stop() { stopCount++; },
    };
    h.context.sessionRecorder = recorder;
    const begin = app.indexOf('ownerMedia.onReady = async () => {');
    const end = app.indexOf('await ownerMedia.start();', begin);
    vm.runInNewContext(app.slice(begin, end), h.context);
    const preparing = h.file.onReady();
    await new Promise(resolve => setImmediate(resolve));
    const freshSession = {};
    h.context.session = freshSession;
    h.context.media = {};
    h.context.sessionRecorder = {};
    ready();
    await assert.rejects(preparing, /Media preparation cancelled/);
    assert.equal(stopCount, 1);
    assert.equal(h.context.session, freshSession);
});

test('a cancelled old video decode cannot overwrite the freshly started video element', async () => {
    const h = harness();
    let durationReady;
    h.file.running = false;
    h.file._getVideoDuration = () => new Promise(resolve => { durationReady = resolve; });
    h.file._disconnectMic = () => {};
    const starting = h.file.start();
    h.file.stop();
    h.video.src = 'blob:fresh-session';
    durationReady(12);
    await assert.rejects(starting, /Media preparation cancelled/);
    assert.equal(h.video.src, 'blob:fresh-session');
    assert.equal(h.file.running, false);
});

test('microphone acquisition finishing after cancellation closes its newly acquired track', async () => {
    const h = harness();
    const provider = new h.context.LiveMediaProvider();
    provider._videoStream = { getTracks: () => [] };
    let microphoneReady;
    let trackStops = 0;
    h.context.navigator.mediaDevices.getUserMedia = () => new Promise(resolve => { microphoneReady = resolve; });
    const starting = provider.start();
    provider.stop();
    microphoneReady({ getTracks: () => [{ stop() { trackStops++; } }] });
    await assert.rejects(starting, /Media preparation cancelled/);
    assert.equal(trackStops, 1);
    assert.equal(provider.running, false);
});

test('worklet capture epochs reject delayed complete packets and clear partial audio on manual/resume boundaries', () => {
    const h = harness();
    let Processor;
    const workerContext = {
        Float32Array, sampleRate: 10,
        AudioWorkletProcessor: class { constructor() { this.port = {}; } },
        registerProcessor(_name, klass) { Processor = klass; },
    };
    vm.runInNewContext(source('duplex/lib/capture-processor.js'), workerContext);
    const processor = new Processor({ processorOptions: { chunkSize: 10, targetSampleRate: 10 } });
    const messages = [];
    processor.port.postMessage = message => messages.push(message);
    const provider = new h.context.LiveMediaProvider();
    provider.running = true;
    provider._captureNode = { port: { postMessage: message => processor.port.onmessage({ data: message }) } };
    const received = [];
    provider.onChunk = chunk => received.push(chunk.audio[0]);
    provider._captureFrame = () => 'frame';
    const begin = app.indexOf('this._captureNode.port.onmessage = (e) => {');
    const end = app.indexOf('\n    }', begin);
    const handlerContext = { provider };
    vm.runInNewContext('(function() { ' + app.slice(begin, end) + ' }).call(provider);', handlerContext);
    provider.resetCaptureTimeline();
    for (let index = 0; index < 10; index++) processor._appendTargetSample(1);
    for (let index = 0; index < 6; index++) processor._appendTargetSample(2);
    provider.resetCaptureTimeline();
    provider._captureNode.port.onmessage({ data: messages[0] });
    assert.deepEqual(received, [], 'late old-epoch complete packets must not enter a fresh session');
    for (let index = 0; index < 10; index++) processor._appendTargetSample(3);
    provider._captureNode.port.onmessage({ data: messages[1] });
    assert.deepEqual(received, [3]);
    assert.deepEqual(Array.from(messages[1].audio), Array(10).fill(3), 'fresh packets must not retain old partial audio');
});

test('a stale startup callback cannot replace fresh provider callbacks or begin its old source', async () => {
    const h = harness();
    let staleStarts = 0;
    h.context.ownerSession = {};
    h.context.ownerMedia = { start() { staleStarts++; } };
    h.context.session = {};
    const freshChunk = () => {};
    h.context.media = { onChunk: freshChunk };
    const begin = app.indexOf('async () => {\n                if (session !== ownerSession');
    const end = app.indexOf('\n            }\n        );', begin);
    const callback = vm.runInNewContext('(' + app.slice(begin, end) + '\n})', h.context);
    await assert.rejects(callback(), /Media preparation cancelled/);
    assert.equal(staleStarts, 0);
    assert.equal(h.context.media.onChunk, freshChunk);
});

test('stopping stale providers clears only their own resources and leaves the fresh video and waveform intact', () => {
    for (const mode of ['file', 'live']) {
        const h = harness();
        const stale = mode === 'file' ? h.file : new h.context.LiveMediaProvider();
        let waveformStops = 0;
        let trackStops = 0;
        h.context._omniStopWaveform = () => { waveformStops++; };
        if (mode === 'file') stale._micStream = { getTracks: () => [{ stop() { trackStops++; } }] };
        else stale._audioStream = { getTracks: () => [{ stop() { trackStops++; } }] };
        const freshStream = {};
        h.video.src = 'blob:fresh-source';
        h.video.srcObject = freshStream;
        h.video.style.display = 'block';
        h.video.play();
        h.context.media = {};
        stale.stop();
        assert.equal(trackStops, 1);
        assert.equal(waveformStops, 0);
        assert.equal(h.video.src, 'blob:fresh-source');
        assert.equal(h.video.srcObject, freshStream);
        assert.equal(h.video.paused, false);
        assert.equal(h.video.style.display, 'block');
    }
});
