import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { existsSync, readFileSync, readdirSync } from 'node:fs';
import { dirname, join, relative, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import vm from 'node:vm';
import test from 'node:test';

const here = dirname(fileURLToPath(import.meta.url));
const webRoot = resolve(here, '../../Sources/MiniCPMDemoBackend/Resources/Web');

function text(path) {
    return readFileSync(join(webRoot, path), 'utf8');
}

function allFiles(root = webRoot) {
    const result = [];
    for (const entry of readdirSync(root, { withFileTypes: true })) {
        const path = join(root, entry.name);
        if (entry.isDirectory()) result.push(...allFiles(path));
        else result.push(path);
    }
    return result;
}

function resourcePath(reference, fromFile) {
    if (/^(?:https?:|wss?:|data:|mailto:|#|javascript:)/i.test(reference)) return null;
    const clean = reference.split(/[?#]/, 1)[0];
    if (!clean) return null;
    if (clean.startsWith('/')) {
        // The mounted web root serves `/static/...` and root pages.
        return join(webRoot, clean.replace(/^\/static\//, '').replace(/^\//, ''));
    }
    return resolve(dirname(fromFile), clean);
}

function playbackHarness() {
    let now = 1_000;
    let nextTimer = 1;
    const timers = new Map();
    const sources = [];
    const audioContext = {
        sampleRate: 24_000,
        currentTime: 0,
        state: 'running',
        destination: {},
        close() { this.state = 'closed'; return Promise.resolve(); },
        createBuffer(_channels, count, rate) {
            const samples = new Float32Array(count);
            return { duration: count / rate, getChannelData: () => samples };
        },
        createBufferSource() {
            const source = {
                stopCount: 0,
                connect() {},
                disconnect() {},
                start(time) { this.startedAt = time; },
                stop(time) { this.stopCount++; this.stoppedAt = time; },
            };
            sources.push(source);
            return source;
        },
    };
    const setTimer = (fn, delay, interval = false) => {
        const id = nextTimer++;
        timers.set(id, { fn, at: now + delay, interval });
        return id;
    };
    const context = {
        AudioContext: function () { return audioContext; },
        Float32Array,
        Uint8Array,
        atob: encoded => Buffer.from(encoded, 'base64').toString('binary'),
        resampleAudio: samples => samples,
        performance: { now: () => now },
        setTimeout: (fn, delay = 0) => setTimer(fn, delay),
        clearTimeout: id => timers.delete(id),
        setInterval: (fn, delay) => setTimer(fn, delay, true),
        clearInterval: id => timers.delete(id),
        requestAnimationFrame: callback => callback(),
        console: { log() {}, warn() {} },
    };
    const source = text('duplex/lib/audio-player.js')
        .replace(/^import .*;$/gm, '')
        .replace(/^export /gm, '')
        .concat('\nthis.AudioPlayer = AudioPlayer;');
    vm.runInNewContext(source, context);
    return {
        context,
        audioContext,
        sources,
        now: () => now,
        audio: Buffer.from(new Float32Array(24_000).buffer).toString('base64'),
        advance(ms) {
            now += ms;
            audioContext.currentTime += ms / 1000;
            for (const [id, timer] of [...timers]) {
                if (!timer.interval && timer.at <= now) {
                    timers.delete(id);
                    timer.fn();
                }
            }
        },
    };
}

function realtimeHarness(config = {}) {
    const harness = playbackHarness();
    harness.context.FarEndEchoReference = class {
        constructor() { this.resetCount = 0; this.playbackCount = 0; }
        pushPlayback() { this.playbackCount++; }
        reset() { this.resetCount++; this.playbackCount = 0; }
        analyzeInput() { return {}; }
    };
    harness.context.taskContinuationForAssistantText = () => null;
    harness.context.WebSocket = { OPEN: 1 };
    const source = text('duplex/lib/realtime-session.js')
        .replace(/^import .*;$/gm, '')
        .replace('export class RealtimeSession', 'class RealtimeSession')
        .concat('\nthis.RealtimeSession = RealtimeSession;');
    vm.runInNewContext(source, harness.context);
    const session = new harness.context.RealtimeSession('test', { getPlaybackDelayMs: () => 0, ...config });
    session._sessionReady = true;
    session._usesInputCommitted = true;
    session.audioPlayer.init();
    session.onSpeakStart = () => ({});
    const sent = [];
    session.ws = {
        readyState: 1,
        send: payload => sent.push(JSON.parse(payload)),
        close() { this.readyState = 3; },
    };
    return {
        ...harness,
        session,
        sent,
        speak(spokenText = '回答。') {
            session._handleMessage({ type: 'response.output.delta', kind: 'text', text: spokenText });
            session._handleMessage({ type: 'response.output.delta', kind: 'audio', audio: harness.audio });
        },
    };
}

test('audio padding applies only to the packet that explicitly declares it', () => {
    const harness = realtimeHarness();
    harness.session._handleMessage({
        type: 'response.output.delta', kind: 'audio', audio: harness.audio,
        metrics: { audio_padding_samples: 3840 },
    });
    harness.session._handleMessage({
        type: 'response.output.delta', kind: 'audio', audio: harness.audio,
    });
    harness.advance(250);
    assert.equal(harness.sources.length, 2);
    assert.equal(harness.sources[0].buffer.duration, 0.84);
    assert.equal(harness.sources[1].buffer.duration, 1,
        'an unmarked packet must retain all of its speech samples');
});

function recorderHarness() {
    const harness = playbackHarness();
    harness.context.Blob = Blob;
    for (const [file, className] of [
        ['session-recorder.js', 'SessionRecorder'],
        ['session-video-recorder.js', 'SessionVideoRecorder'],
    ]) {
        const source = text(`duplex/lib/${file}`)
            .replace(/^import .*;$/gm, '')
            .replace(`export class ${className}`, `class ${className}`)
            .concat(`\nthis.${className} = ${className};`);
        vm.runInNewContext(source, harness.context);
    }
    return harness;
}

function recordedRealtimeHarness(config = {}) {
    const harness = realtimeHarness(config);
    harness.context.Blob = Blob;
    const source = text('duplex/lib/session-recorder.js')
        .replace(/^import .*;$/gm, '')
        .replace('export class SessionRecorder', 'class SessionRecorder')
        .concat('\nthis.SessionRecorder = SessionRecorder;');
    vm.runInNewContext(source, harness.context);
    const recorder = new harness.context.SessionRecorder(24_000, 24_000);
    recorder.start();
    const cancellations = [];
    harness.session.audioPlayer.onRawAudio = (samples, rate, timestamp) =>
        recorder.pushRight(samples, rate, timestamp);
    harness.session.audioPlayer.onPlaybackCancel = timestamp => {
        cancellations.push(timestamp);
        recorder.cancelRight(timestamp);
    };
    return { ...harness, recorder, cancellations };
}

function connectionHarness(config = {}) {
    const harness = realtimeHarness(config);
    const sockets = [];
    class MockWebSocket {
        static OPEN = 1;
        constructor() {
            this.readyState = 0;
            this.sent = [];
            sockets.push(this);
        }
        open() { this.readyState = 1; this.onopen?.(); }
        send(payload) { this.sent.push(JSON.parse(payload)); }
        deliver(payload) { this.onmessage?.({ data: JSON.stringify(payload) }); }
        close() {
            if (this.readyState === 3) return;
            this.readyState = 3;
            this.onclose?.();
        }
    }
    harness.context.WebSocket = MockWebSocket;
    harness.session.config.getWsUrl = () => 'ws://test';
    harness.session._sessionArgs = { systemPrompt: 'prompt', preparePayload: {} };
    harness.session._sessionReady = false;
    return { ...harness, sockets, MockWebSocket };
}

const flushConnection = () => new Promise(resolve => setImmediate(resolve));

test('initial connection waits for session.created before input and user controls', async () => {
    const { session, sockets, advance } = connectionHarness();
    const starting = session.start('prompt', {}, async () => {
        session.sendChunk({ audio_base64: 'media-started', speech_active: true });
    });
    const socket = sockets[0];
    session.sendChunk({ audio_base64: 'before-open', speech_active: true });
    session.pauseToggle();
    session.toggleForceListen();
    socket.open();
    await flushConnection();
    session.sendChunk({ audio_base64: 'open-before-init', speech_active: true });
    assert.equal(session._sendControl('session.pause'), false);
    session._cancelCurrentResponse();
    assert.equal(session._awaitingCancelAck, false, 'an unsent cancellation must not wait forever for an ACK');
    session.pauseToggle();
    session.toggleForceListen();
    assert.deepEqual(socket.sent, []);
    assert.equal(session.paused, false);
    assert.equal(session.forceListenActive, false);
    assert.equal(session._pendingInputs.length, 0);
    advance(100);
    assert.deepEqual(socket.sent.map(msg => msg.type), ['session.init']);
    session.sendChunk({ audio_base64: 'init-before-created', speech_active: true });
    const oldStartupError = socket.onerror;
    const oldStartupClose = socket.onclose;
    socket.deliver({ type: 'session.created', session_id: 'initial', metrics: { backend: 'swift_mlx' } });
    await starting;
    oldStartupError();
    oldStartupClose();
    assert.equal(session._sessionReady, true);
    assert.equal(session.running, true, 'late startup callbacks must not stop an established session');
    assert.deepEqual(socket.sent.filter(msg => msg.type === 'input.append').map(msg => msg.input.audio),
        ['media-started']);
    session.stop();
    assert.equal(session._sessionReady, false);
});

test('reconnect discards old input and waits for fresh session.created before capture and controls', async () => {
    const { session, sockets, MockWebSocket, advance } = connectionHarness();
    session._reset();
    session._started = true;
    session._sessionReady = true;
    const oldSocket = new MockWebSocket();
    oldSocket.open();
    session.ws = oldSocket;
    for (const name of ['old-one', 'old-two', 'old-pending']) {
        session.sendChunk({ audio_base64: name, speech_active: true });
    }
    assert.equal(session._pendingInputs.length, 1);
    oldSocket.readyState = 3;
    const reconnecting = session._handleTransportClose();
    const socket = sockets[1];
    assert.equal(session._sessionReady, false);
    assert.equal(session._pendingInputs.length, 0);
    assert.equal(session._recentInputs.size, 0);
    session.sendChunk({ audio_base64: 'reconnect-before-open', speech_active: true });
    socket.open();
    await flushConnection();
    session.sendChunk({ audio_base64: 'reconnect-open', speech_active: true });
    session.pauseToggle();
    session.toggleForceListen();
    assert.deepEqual(socket.sent, []);
    assert.equal(session.paused, false);
    assert.equal(session.forceListenActive, false);
    advance(100);
    assert.deepEqual(socket.sent.map(msg => msg.type), ['session.init']);
    session.sendChunk({ audio_base64: 'reconnect-initializing', speech_active: true });
    socket.deliver({ type: 'session.created', session_id: 'fresh', metrics: { backend: 'swift_mlx' } });
    await reconnecting;
    session.sendChunk({ audio_base64: 'fresh-capture', speech_active: true });
    assert.deepEqual(socket.sent.filter(msg => msg.type === 'input.append').map(msg => msg.input.audio),
        ['fresh-capture']);
    assert.equal(session._reconnecting, false);
    session.stop();
});

test('failed reconnect cannot reopen input readiness through a stale session.created', async () => {
    const { session, sockets, MockWebSocket, advance } = connectionHarness({
        reconnectAttempts: 1, reconnectBaseDelayMs: 100,
    });
    session._reset();
    session._started = true;
    const oldSocket = new MockWebSocket();
    oldSocket.readyState = 3;
    session.ws = oldSocket;
    const reconnecting = session._handleTransportClose();
    const failedSocket = sockets[1];
    failedSocket.open();
    await flushConnection();
    const staleDeliver = failedSocket.onmessage;
    failedSocket.deliver({ type: 'error', error: { message: 'init failed' } });
    await flushConnection();
    assert.equal(session._sessionReady, false);
    staleDeliver({ data: JSON.stringify({ type: 'session.created', session_id: 'stale' }) });
    assert.equal(session._sessionReady, false);
    session.sendChunk({ audio_base64: 'failed-capture', speech_active: true });
    advance(100);
    await flushConnection();
    const retrySocket = sockets[2];
    retrySocket.open();
    await flushConnection();
    retrySocket.deliver({ type: 'session.queue_done' });
    session.sendChunk({ audio_base64: 'retry-before-created', speech_active: true });
    assert.deepEqual(retrySocket.sent.map(msg => msg.type), ['session.init']);
    retrySocket.deliver({ type: 'session.created', session_id: 'retry-ok' });
    await reconnecting;
    session.sendChunk({ audio_base64: 'retry-fresh', speech_active: true });
    assert.deepEqual(retrySocket.sent.filter(msg => msg.type === 'input.append').map(msg => msg.input.audio),
        ['retry-fresh']);
    session.stop();
});

test('stopping during either reconnect phase cancels its wait without sending controls before init', async () => {
    for (const phase of ['opening', 'initializing']) {
        const { session, sockets, MockWebSocket, advance } = connectionHarness();
        session._reset();
        session._started = true;
        const oldSocket = new MockWebSocket();
        oldSocket.readyState = 3;
        session.ws = oldSocket;
        const reconnecting = session._handleTransportClose();
        const socket = sockets[1];
        if (phase === 'initializing') {
            socket.open();
            await flushConnection();
        }
        const staleDeliver = socket.onmessage;
        session.stop();
        await reconnecting;
        if (staleDeliver) staleDeliver({ data: JSON.stringify({ type: 'session.created', session_id: 'cancelled' }) });
        advance(30_000);
        assert.equal(session._sessionReady, false);
        assert.equal(session.running, false);
        assert.equal(session._reconnectReject, null);
        assert.deepEqual(socket.sent, [], `${phase} reconnect sent a session control before creation`);
    }
});

test('initial socket close, error and cancellation reject creation waits both before and after queue_done', async () => {
    for (const action of ['close', 'error', 'stop', 'cancelQueue']) {
        for (const assigned of [false, true]) {
            const { session, sockets, advance } = connectionHarness();
            let cleanupCount = 0;
            session.onCleanup = () => { cleanupCount++; };
            const starting = session.start('prompt', {}, null);
            const rejected = assert.rejects(starting, /closed|failed|cancelled/);
            const socket = sockets[0];
            socket.open();
            await flushConnection();
            if (assigned) socket.deliver({ type: 'session.queue_done' });
            const staleDeliver = socket.onmessage;
            if (action === 'close') socket.close();
            else if (action === 'error') socket.onerror();
            else session[action]();
            await rejected;
            advance(30_000);
            staleDeliver({ data: JSON.stringify({ type: 'session.created', session_id: 'too-late' }) });
            assert.equal(cleanupCount, 1, `${action}/${assigned} cleanup must happen exactly once`);
            assert.equal(session._sessionReady, false);
            assert.equal(session.running, false);
            assert.equal(session.ws, null);
            assert.equal(session._queueReject, null);
            assert.equal(session._sessionInitTimer, null);
            assert.deepEqual(socket.sent.map(msg => msg.type), assigned ? ['session.init'] : [],
                `${action}/${assigned} must not send a delayed init or a premature session.close`);
        }
    }
});

test('initial connection cancellation before socket open cancels its connection wait', async () => {
    for (const action of ['stop', 'cancelQueue']) {
        const { session, sockets, advance } = connectionHarness();
        const starting = session.start('prompt', {}, null);
        const rejected = assert.rejects(starting, /cancelled/);
        session[action]();
        await rejected;
        advance(30_000);
        assert.equal(session._sessionReady, false);
        assert.equal(session._queueReject, null);
        assert.deepEqual(sockets[0].sent, []);
    }
});

test('automatic continuation does not mutate a reconnecting unready session', () => {
    const { session, context, sent } = connectionHarness();
    context.taskContinuationForAssistantText = () => 'Continue';
    let continuationCount = 0;
    session.onAutoContinuation = () => { continuationCount++; };
    session._maybeAutoContinue('story');
    assert.equal(session._autoContinuationUsed, false);
    assert.equal(session.modelState, 'listening');
    assert.equal(continuationCount, 0);
    assert.deepEqual(sent, []);
    assert.equal(session._pendingInputs.length, 0);
});

test('vendored tree contains the complete MiniCPM demo page set', () => {
    for (const page of [
        'index.html', 'turnbased.html', 'half-duplex/half_duplex.html',
        'audio-duplex/audio_duplex.html', 'omni/omni.html',
        'realtime/realtime.html', 'mobile-omni/index.html', 'admin.html',
        'session-viewer.html', 'prototype-content-editor.html',
    ]) assert.ok(existsSync(join(webRoot, page)), page);

    const files = allFiles().filter(path => !path.endsWith('UPSTREAM_SOURCE.md'));
    assert.ok(files.length >= 60, `expected the full static dependency closure, got ${files.length}`);
    assert.match(text('UPSTREAM_SOURCE.md'), /ba7fa9cc6ad63c894f1bd5e5afac28466953519d/);
});

test('HTML and JavaScript dependency references resolve inside the mounted root', () => {
    for (const file of allFiles().filter(path => /\.(?:html|js|mjs)$/.test(path))) {
        const source = readFileSync(file, 'utf8');
        const references = [];
        for (const match of source.matchAll(/(?:src|href)=["']([^"']+)["']/g)) references.push(match[1]);
        for (const match of source.matchAll(/(?:from\s*|import\s*\()["']([^"']+)["']/g)) references.push(match[1]);
        for (const reference of references) {
            const path = resourcePath(reference, file);
            if (!path || !/\.(?:html|css|js|mjs|png|mp4|wav|onnx)$/.test(path)) continue;
            assert.ok(existsSync(path), `${relative(webRoot, file)} -> ${reference}`);
        }
    }
});

test('capture and playback contracts are explicit and sample-rate safe', () => {
    const worklet = text('duplex/lib/capture-processor.js');
    assert.match(worklet, /targetSampleRate/);
    assert.match(worklet, /this\._sourceRate\s*=\s*sampleRate/);
    assert.match(worklet, /registerProcessor\('capture-processor'/);
    assert.doesNotMatch(worklet, /_sourceBuffer/);
    assert.doesNotMatch(worklet, /output\.push/);
    const player = text('duplex/lib/audio-player.js');
    assert.match(player, /outputSampleRate.*24000/);
    assert.match(player, /_pendingChunks/);
    assert.match(player, /onFarEndReference/);
    for (const page of ['audio-duplex/audio-duplex-app.js', 'omni/omni-app.js']) {
        const source = text(page);
        assert.match(source, /echoCancellation:\s*true/);
        assert.match(source, /noiseSuppression:\s*true/);
        assert.match(source, /autoGainControl:\s*true/);
        assert.match(source, /targetSampleRate:\s*SAMPLE_RATE_IN/);
        assert.doesNotMatch(source, /readyState\s*===\s*WebSocket\.OPEN\s*&&\s*!aiSpeaking/);
    }
    const audioDuplex = text('audio-duplex/audio-duplex-app.js');
    assert.match(audioDuplex, /MIC_ACTIVITY_RMS/);
    assert.match(audioDuplex, /echo_gate/);
    const half = text('half-duplex/half-duplex-app.js');
    assert.match(half, /echoCancellation:\s*true/);
    assert.match(half, /!aiSpeaking/);
});

test('slow generation uses a bounded adaptive jitter buffer', async () => {
    const modulePath = pathToFileURL(join(webRoot, 'duplex/lib/audio-player.js')).href;
    const { adaptivePlaybackDelayMs, trimAudioPadding } = await import(modulePath);

    assert.equal(
        adaptivePlaybackDelayMs(200, 800, 1000),
        900,
        'a measured chunk gets a bounded jitter-buffer reserve');
    assert.equal(
        adaptivePlaybackDelayMs(200, 1200, 1000),
        1000,
        'a slow chunk is capped at one second instead of accumulating delay');
    assert.equal(
        adaptivePlaybackDelayMs(200, 2500, 1000),
        1000,
        'very slow generation still cannot create a multi-second startup wait');
    assert.equal(
        adaptivePlaybackDelayMs(200, 800, 1000, 1800),
        1000,
        'learned underrun reserve is capped at one second');
    assert.equal(
        adaptivePlaybackDelayMs(200, 1200, 1000, 0, 2500),
        1300,
        'the runtime can reserve more than one chunk for measured jitter');
    assert.equal(
        adaptivePlaybackDelayMs(200, 2500, 1000, 0, 2500),
        2500,
        'the runtime reserve remains capped instead of waiting indefinitely');

    const padded = new Float32Array([0, 0, 0.1, 0.2]);
    const trimmed = Array.from(trimAudioPadding(padded, 2));
    assert.equal(trimmed.length, 2);
    assert.ok(Math.abs(trimmed[0] - 0.1) < 1e-6);
    assert.ok(Math.abs(trimmed[1] - 0.2) < 1e-6);
    assert.deepEqual(
        Array.from(trimAudioPadding(padded, 99)),
        [],
        'invalid oversized padding cannot expose stale samples');

    const session = text('duplex/lib/realtime-session.js');
    assert.match(session, /msg\.metrics\?\.wall_clock_ms/);
    assert.match(session, /audio_padding_samples/);
    assert.match(session, /case 'input\.committed'/);
});

test('normal audio turns preserve queued tails and stopAll clears playback', () => {
    const harness = playbackHarness();
    const player = new harness.context.AudioPlayer({ getPlaybackDelayMs: () => 0 });
    const recordedAt = [];
    player.onRawAudio = (_samples, _rate, timestamp) => recordedAt.push(timestamp);
    player.init();
    player.beginTurn();
    player.playChunk(harness.audio, harness.now());
    assert.equal(harness.sources.length, 0, 'a fresh timeline must retain startup buffering');
    harness.advance(250);
    assert.equal(harness.sources.length, 1);
    const previousEnd = player.nextTime;
    const playbackStartTime = player._playbackStartTime;
    const playbackStartCtxTime = player._playbackStartCtxTime;

    harness.advance(200);
    player.endTurn();
    player.beginTurn();
    player.playChunk(harness.audio, harness.now());
    assert.equal(harness.sources[0].stopCount, 0, 'a normal new turn cut the previous tail');
    assert.equal(harness.sources[1].startedAt, previousEnd, 'new audio must follow the queued tail');
    assert.equal(player._playbackStartTime, playbackStartTime);
    assert.equal(player._playbackStartCtxTime, playbackStartCtxTime);
    assert.equal(recordedAt[1] - recordedAt[0], 1000, 'recording timestamps lost their playback anchor');
    assert.equal(player._aheadInterval !== null, true, 'new turn must restart queue monitoring');

    player.stopAll();
    assert.deepEqual(harness.sources.map(source => source.stopCount), [1, 1]);
    assert.equal(player.hasBufferedAudio, false, 'stopped audio must not remain in the playback clock');
    assert.equal(player.playing, false);
    assert.equal(player.nextTime, 0);
    player.endTurn();
    player.beginTurn();
    player.playChunk(harness.audio, harness.now());
    assert.equal(harness.sources.length, 2, 'an empty new turn must use startup buffering again');
    assert.equal(player._pendingChunks.length, 1);
    player.stopAll();
    assert.equal(player._pendingChunks.length, 0, 'stopAll must also discard unstarted chunks');
    assert.equal(player._delayTimer, null);
});

test('explicit cancellation preserves played WAV samples and removes queued future samples', async () => {
    const harness = recorderHarness();
    const recorder = new harness.context.SessionRecorder(4, 4);
    recorder.start();
    recorder.pushLeft(new Float32Array(8));
    recorder.pushRight(new Float32Array([0.1, 0.2, 0.3, 0.4]), 4, 1000);
    recorder.pushRight(new Float32Array([0.5, 0.6, 0.7, 0.8]), 4, 2000);
    harness.advance(500);
    recorder.cancelRight(harness.now());
    assert.equal(recorder._rightEntries.length, 1);
    assert.equal(recorder._rightEntries[0].data.length, 2, 'cancellation must retain the played prefix');
    assert.equal(recorder._rightNextOffset, 2, 'cancelled tail must not reserve recording time');
    const result = recorder.stop();
    assert.equal(result.rightSamples, 2);
    const wav = new DataView(await result.blob.arrayBuffer());
    assert.ok(wav.getInt16(46, true) > 0);
    assert.ok(wav.getInt16(50, true) > 0);
    for (let index = 2; index < 8; index++) {
        assert.equal(wav.getInt16(46 + index * 4, true), 0, `cancelled sample ${index} leaked into exported WAV`);
    }
});

test('normal turn endings preserve recording tails while stopAll cancels them', () => {
    const harness = recorderHarness();
    const player = new harness.context.AudioPlayer({ getPlaybackDelayMs: () => 0 });
    const recorder = new harness.context.SessionRecorder(24_000, 24_000);
    recorder.start();
    let cancellations = 0;
    player.onRawAudio = (samples, rate, timestamp) => recorder.pushRight(samples, rate, timestamp);
    player.onPlaybackCancel = timestamp => {
        cancellations++;
        recorder.cancelRight(timestamp);
    };
    player.init();
    player.beginTurn();
    player.playChunk(harness.audio, harness.now());
    harness.advance(250);
    player.playChunk(harness.audio, harness.now());
    harness.advance(200);
    player.endTurn();
    player.beginTurn();
    assert.equal(cancellations, 0, 'ordinary turn changes must not cancel recording audio');
    assert.equal(recorder._rightEntries.length, 2);
    player.stopAll();
    assert.equal(cancellations, 1);
    assert.equal(player.turnActive, false);
    assert.equal(recorder._rightEntries.length, 1);
    assert.equal(recorder._rightEntries[0].data.length, 4800, 'only the 200 ms actually played should remain');
    assert.equal(recorder._rightNextOffset, 10_800, 'recorded prefix retains its original wall-clock offset');
});

test('cancelled WAV tails do not push new audio forward after pause and resume', () => {
    const harness = recorderHarness();
    const recorder = new harness.context.SessionRecorder(4, 4);
    recorder.start();
    recorder.pushRight(new Float32Array(8), 4, 1000);
    harness.advance(500);
    recorder.cancelRight(harness.now());
    recorder.pause();
    harness.advance(2000);
    recorder.resume();
    recorder.pushRight(new Float32Array(4), 4, harness.now());
    assert.deepEqual(Array.from(recorder._rightEntries, entry => entry.offset), [0, 2]);
    harness.advance(250);
    recorder.pause();
    harness.advance(1000);
    recorder.cancelRight(harness.now());
    assert.deepEqual(Array.from(recorder._rightEntries, entry => entry.data.length), [2, 1]);
    assert.equal(recorder._rightNextOffset, 3, 'ongoing paused time must not count as played audio');
});

test('stopping WAV recording independently crops future playback before finalization', () => {
    const harness = recorderHarness();
    const recorder = new harness.context.SessionRecorder(4, 4);
    recorder.start();
    recorder.pushRight(new Float32Array(4), 4, 2000);
    const result = recorder.stop();
    assert.equal(result.rightSamples, 0, 'cleanup may finalize the recorder before stopping AudioPlayer');
    assert.equal(result.blob.size, 0);
});

test('session cleanup crops frozen playback before finalizing its WAV recorder', () => {
    const harness = realtimeHarness();
    harness.context.Blob = Blob;
    const source = text('duplex/lib/session-recorder.js')
        .replace(/^import .*;$/gm, '')
        .replace('export class SessionRecorder', 'class SessionRecorder')
        .concat('\nthis.SessionRecorder = SessionRecorder;');
    vm.runInNewContext(source, harness.context);
    const recorder = new harness.context.SessionRecorder(24_000, 24_000);
    recorder.start();
    const { session } = harness;
    session._cleaned = false;
    session.audioPlayer.onRawAudio = (samples, rate, timestamp) => recorder.pushRight(samples, rate, timestamp);
    const order = [];
    session.audioPlayer.onPlaybackCancel = timestamp => {
        order.push('crop');
        assert.equal(timestamp, 1650);
        recorder.cancelRight(timestamp);
    };
    let result;
    session.onCleanup = () => {
        order.push('finalize');
        assert.equal(session.audioPlayer.ctx, null, 'the output must stop before page cleanup');
        result = recorder.stop();
    };
    harness.speak(); harness.advance(250); harness.speak(); harness.advance(400);
    const frozenAudioTime = harness.audioContext.currentTime;
    harness.audioContext.state = 'interrupted';
    harness.advance(2000);
    harness.audioContext.currentTime = frozenAudioTime;
    session.cleanup();
    assert.deepEqual(order, ['crop', 'finalize']);
    assert.equal(result.rightSamples, 9600, 'finalization must retain only the 400 ms heard before interruption');
    assert.deepEqual(harness.sources.map(source => source.stopCount), [1, 1]);
    assert.equal(session.audioPlayer.hasBufferedAudio, false);
    session.cleanup();
    assert.deepEqual(order, ['crop', 'finalize'], 'cleanup must remain idempotent');
});

test('video recording follows player timestamps and cancels right sources before final flush', async () => {
    const harness = recorderHarness();
    const recorder = new harness.context.SessionVideoRecorder({}, 4, 4);
    recorder._recording = true;
    recorder._recCtx = harness.audioContext;
    recorder._stereoNode = { disconnect() {} };
    recorder._rightNextTime = 0;
    recorder._rightSources = [];
    recorder._startTime = harness.now();
    recorder.pushRight(new Float32Array(4), 4, harness.now() + 1000);
    recorder.pushRight(new Float32Array(4), 4, harness.now() + 2000);
    assert.deepEqual(harness.sources.map(source => source.startedAt), [1, 2]);
    assert.equal(recorder._rightNextTime, 3);
    harness.advance(1250);
    recorder.cancelRight(harness.now());
    assert.deepEqual(harness.sources.map(source => source.stopCount), [1, 1]);
    assert.deepEqual(harness.sources.map(source => source.stoppedAt), [1.25, 1.25]);
    assert.equal(harness.sources[0].stoppedAt - harness.sources[0].startedAt, 0.25,
        'the active source keeps its already rendered prefix');
    assert.ok(harness.sources[1].stoppedAt < harness.sources[1].startedAt,
        'the future source is cancelled before it can enter the recording');
    assert.equal(recorder._rightSources.length, 0);
    assert.equal(recorder._rightNextTime, 1.25, 'new audio must not wait behind cancelled video-recording sources');
    recorder.pushRight(new Float32Array(4), 4, harness.now());
    assert.equal(harness.sources[2].startedAt, 1.25);
    let flushed = false;
    recorder._mediaRecorder = {
        mimeType: 'video/webm',
        requestData() {
            assert.equal(harness.sources[2].stopCount, 1, 'stop must cancel right audio before flushing the encoder');
            flushed = true;
        },
        stop() { queueMicrotask(() => this.onstop()); },
    };
    await recorder.stop();
    assert.equal(flushed, true);
    assert.equal(recorder._rightSources.length, 0);
});

test('both duplex pages wire explicit playback cancellation into their recorder', () => {
    for (const page of ['audio-duplex/audio-duplex-app.js', 'omni/omni-app.js']) {
        assert.match(text(page), /onPlaybackCancel\s*=\s*ts\s*=>\s*sessionRecorder\?\.cancelRight\(ts\)/);
    }
});

test('normal listen preserves audio while backend interrupt clears playback without continuing', () => {
    const harness = playbackHarness();
    const continuationCandidates = [];
    harness.context.FarEndEchoReference = class { pushPlayback() {} reset() {} };
    harness.context.taskContinuationForAssistantText = text => {
        continuationCandidates.push(text);
        return null;
    };
    harness.context.WebSocket = { OPEN: 1 };
    const source = text('duplex/lib/realtime-session.js')
        .replace(/^import .*;$/gm, '')
        .replace('export class RealtimeSession', 'class RealtimeSession')
        .concat('\nthis.RealtimeSession = RealtimeSession;');
    vm.runInNewContext(source, harness.context);
    const session = new harness.context.RealtimeSession('test', { getPlaybackDelayMs: () => 0 });
    session._sessionReady = true;
    session.audioPlayer.init();
    let speakEnds = 0;
    let autoContinuations = 0;
    session.onSpeakStart = () => ({});
    session.onSpeakEnd = () => { speakEnds++; };
    session.onAutoContinuation = () => { autoContinuations++; };
    const sent = [];
    session.ws = { readyState: 1, send: payload => sent.push(payload) };
    const speak = (responseID, spokenText) => {
        session._handleMessage({
            type: 'response.output.delta', kind: 'text', response_id: responseID, text: spokenText,
        });
        session._handleMessage({
            type: 'response.output.delta', kind: 'audio', response_id: responseID, audio: harness.audio,
        });
    };
    const listen = () => session._handleMessage({
        type: 'response.output.delta', kind: 'listen', reason: 'turn_end',
    });
    speak('response-one', '上一轮回答。');
    harness.advance(250);
    harness.advance(200);
    listen();
    assert.equal(harness.sources[0].stopCount, 0, 'ordinary listen must let queued speech finish');
    assert.equal(session.audioPlayer.hasBufferedAudio, true);
    assert.equal(session.modelState, 'listening');
    assert.equal(session.currentSpeakText, '');

    const previousEnd = session.audioPlayer.nextTime;
    const lastTTFS = session._lastTTFS;
    const resultCount = session._resultCount;
    session._handleMessage({
        type: 'response.output.delta', kind: 'audio', silent: true, audio: harness.audio,
    });
    assert.equal(session.modelState, 'listening', 'listen PCM must not enter speaking state');
    assert.equal(session._wasListening, true, 'listen PCM must not consume the next real TTFS');
    assert.equal(session._lastTTFS, lastTTFS);
    assert.equal(session._resultCount, resultCount);
    listen();
    assert.equal(harness.sources[0].stopCount, 0, 'a listen silence block must not cut queued speech');
    assert.equal(harness.sources.length, 1, 'listen PCM must not add a pause to the playback queue');
    speak('response-two', '好呀，你想听什么样的故事呢？');
    assert.equal(harness.sources[1].startedAt, previousEnd);

    continuationCandidates.length = 0;
    harness.context.taskContinuationForAssistantText = text => {
        continuationCandidates.push(text);
        return 'Continue the story';
    };
    session._handleMessage({ type: 'session.state', reason: 'interrupt', state: 'active' });
    assert.deepEqual(harness.sources.map(source => source.stopCount), [1, 1]);
    assert.equal(session.audioPlayer.hasBufferedAudio, false);
    assert.equal(session.audioPlayer.turnActive, false);
    assert.equal(session.audioPlayer.playing, false);
    assert.equal(session.modelState, 'listening');
    assert.equal(session.currentSpeakText, '');
    assert.equal(session._speakHandle, null);
    assert.equal(speakEnds, 2, 'interrupt must close the speaking UI');
    assert.deepEqual(continuationCandidates, [], 'interrupt must not evaluate automatic continuation');
    assert.equal(autoContinuations, 0);
    assert.deepEqual(sent, [], 'interrupt must not submit a new model input');

    speak('response-three', '新的回答。');
    harness.advance(250);
    session._handleMessage({ type: 'session.state', reason: 'force_listen', state: 'active' });
    assert.equal(session.audioPlayer.hasBufferedAudio, true, 'control acknowledgements must preserve speech');
    session._handleMessage({ type: 'session.state', reason: 'reset', state: 'active' });
    assert.equal(session.audioPlayer.hasBufferedAudio, false, 'official reset must clear speech');
    assert.equal(session.currentSpeakText, '');
});

test('model barge-in stops scheduled playback and crops unheard WAV without cancelling fresh input', () => {
    for (const reason of ['model_barge_in', 'model_interrupted']) {
        for (const endedButBuffered of [false, true]) {
            const harness = recordedRealtimeHarness({ maxInputInFlight: 1 });
            const { session, recorder, cancellations, sent } = harness;
            let speakEnds = 0;
            session.onSpeakEnd = () => { speakEnds++; };
            harness.speak('正在讲的故事。');
            harness.advance(250);
            session._handleMessage({
                type: 'response.output.delta', kind: 'audio', audio: harness.audio,
            });
            harness.advance(200);
            assert.equal(recorder._rightEntries.length, 2);
            assert.equal(session.echoReference.playbackCount, 2);
            if (endedButBuffered) {
                session._handleMessage({
                    type: 'response.output.delta', kind: 'listen', reason: 'turn_end',
                });
                assert.equal(session.audioPlayer.turnActive, false);
                assert.equal(session.audioPlayer.hasBufferedAudio, true);
            }

            const continuationCandidates = [];
            harness.context.taskContinuationForAssistantText = text => {
                continuationCandidates.push(text);
                return 'Continue the story';
            };
            session.sendChunk({ audio_base64: 'new-onset', speech_active: true });
            session.sendChunk({ audio_base64: 'new-second', speech_active: true });
            session.sendChunk({ audio_base64: 'new-third', speech_active: true });
            const recent = Array.from(session._recentInputs);
            const pending = Array.from(session._pendingInputs);
            const queuedBytes = session._pendingInputBytes;
            const controlsBefore = sent.length;
            session._handleMessage({ type: 'response.output.delta', kind: 'listen', reason });

            assert.deepEqual(harness.sources.map(source => source.stopCount), [1, 1]);
            assert.equal(session.audioPlayer.hasBufferedAudio, false);
            assert.equal(session.audioPlayer.turnActive, false);
            assert.equal(session.echoReference.playbackCount, 0);
            assert.equal(session.modelState, 'listening');
            assert.equal(session.currentSpeakText, '');
            assert.equal(speakEnds, 1, 'barge-in must close the speaking UI only once');
            assert.deepEqual(cancellations, [harness.now()]);
            assert.equal(recorder._rightEntries.length, 1);
            assert.equal(recorder._rightEntries[0].data.length, 4800,
                'WAV retains only the 200 ms already played');
            assert.deepEqual(Array.from(session._recentInputs), recent);
            assert.deepEqual(Array.from(session._pendingInputs), pending);
            assert.equal(session._pendingInputBytes, queuedBytes);
            assert.equal(session._pendingCancelAcks, 0);
            assert.equal(sent.length, controlsBefore, 'model listen must not send response.cancel');
            assert.deepEqual(continuationCandidates, [], 'an interrupted story must not continue itself');

            session._handleMessage({ type: 'input.committed', input_id: sent[0].message_id });
            harness.advance(0);
            session._handleMessage({ type: 'input.committed', input_id: sent[1].message_id });
            harness.advance(0);
            assert.deepEqual(sent.map(event => event.input.audio),
                ['new-onset', 'new-second', 'new-third']);
            assert.equal(recorder.stop().rightSamples, 4800);
        }
    }
});

test('input reports buffered output after natural model turn end until drained or stopped', () => {
    for (const stopEarly of [false, true]) {
        const harness = realtimeHarness({ maxInputInFlight: 4 });
        const { session, sent } = harness;
        session.sendChunk({ audio_base64: 'before-playback' });
        assert.equal(sent.at(-1).input.playback_active, false);
        harness.speak('完整的回答。');
        harness.advance(250);
        session._handleMessage({ type: 'response.output.delta', kind: 'listen', reason: 'turn_end' });
        assert.equal(session.audioPlayer.turnActive, false);
        session.sendChunk({ audio_base64: 'during-tail' });
        assert.equal(sent.at(-1).input.playback_active, true);
        if (stopEarly) session.audioPlayer.stopAll();
        else {
            harness.advance(2000);
            for (const source of harness.sources) source.onended?.();
        }
        session.sendChunk({ audio_base64: 'after-tail' });
        assert.equal(sent.at(-1).input.playback_active, false);
    }
});

test('model barge-in does not consume explicit cancel acknowledgements or release queued PCM early', () => {
    const harness = realtimeHarness({ maxInputInFlight: 1 });
    const { session, sent } = harness;
    session._pendingCancelAcks = 2;
    session.sendChunk({ audio_base64: 'fresh-after-cancel', speech_active: true });
    const pending = Array.from(session._pendingInputs);
    session._handleMessage({
        type: 'response.output.delta', kind: 'listen', reason: 'model_barge_in',
    });
    harness.advance(0);
    assert.equal(session._pendingCancelAcks, 2);
    assert.deepEqual(Array.from(session._pendingInputs), pending);
    assert.equal(sent.length, 0);
    session._handleMessage({ type: 'session.state', reason: 'interrupt', state: 'active' });
    harness.advance(0);
    assert.equal(session._pendingCancelAcks, 1);
    assert.equal(sent.length, 0);
    session._handleMessage({ type: 'session.state', reason: 'interrupt', state: 'active' });
    harness.advance(0);
    assert.equal(session._pendingCancelAcks, 0);
    assert.equal(sent[0].input.audio, 'fresh-after-cancel');
});

test('ordinary turn end and model listen preserve scheduled playback and complete WAV tails', () => {
    for (const reason of ['turn_end', 'model_listen']) {
        const harness = recordedRealtimeHarness();
        const { session, recorder, cancellations } = harness;
        harness.speak('完整的回答。');
        harness.advance(250);
        session._handleMessage({
            type: 'response.output.delta', kind: 'audio', audio: harness.audio,
        });
        harness.advance(200);
        session._handleMessage({ type: 'response.output.delta', kind: 'listen', reason });
        assert.deepEqual(harness.sources.map(source => source.stopCount), [0, 0]);
        assert.equal(session.audioPlayer.hasBufferedAudio, true);
        assert.deepEqual(cancellations, []);
        assert.equal(recorder._rightEntries.length, 2);
        assert.ok(recorder._rightEntries.every(entry => entry.data.length === 24_000));
        harness.advance(2000);
        assert.equal(recorder.stop().rightSamples, 48_000);
    }
});

test('pause cancels queued speech before the server pause and resumes with fresh input', () => {
    const harness = realtimeHarness();
    const { session, sent } = harness;
    let speakEnds = 0;
    session.onSpeakEnd = () => { speakEnds++; };
    harness.speak('暂停前的回答。');
    harness.advance(250);
    assert.equal(session.echoReference.playbackCount, 1);
    session._recentInputs.set('old-input', { event: { message_id: 'old-input' } });
    session._pendingInputs.push({ event: { message_id: 'old-pending' }, sequence: 1, retries: 0 });
    session._schedulePendingInputRetry();

    session.pauseToggle();
    assert.deepEqual(sent.map(message => message.type), ['response.cancel', 'session.pause']);
    assert.equal(session.paused, true);
    assert.equal(session.pauseState, 'paused');
    assert.equal(session._awaitingCancelAck, true);
    assert.equal(session._pendingInputs.length, 0);
    assert.equal(session._recentInputs.size, 0);
    assert.equal(session._inputRetryTimer, null);
    assert.equal(session.audioPlayer.hasBufferedAudio, false);
    assert.equal(harness.sources[0].stopCount, 1);
    assert.equal(session.echoReference.playbackCount, 0);
    assert.equal(speakEnds, 1);

    const resultCount = session._resultCount;
    harness.speak('旧的晚到输出。');
    assert.equal(session._resultCount, resultCount, 'paused output is excluded from speech statistics');
    assert.equal(session.currentSpeakText, '');
    assert.equal(harness.sources.length, 1, 'late audio must not restart the speaker');
    session.sendChunk({ audio_base64: harness.audio });
    assert.equal(sent.length, 2, 'pause also stops client input');

    session._handleMessage({ type: 'session.state', reason: 'interrupt', state: 'active' });
    session._handleMessage({ type: 'session.state', reason: 'paused', state: 'paused' });
    assert.equal(session._awaitingCancelAck, false);
    harness.speak('取消确认后的旧输出。');
    assert.equal(session.currentSpeakText, '', 'the pause stays active after cancellation acknowledgement');
    session.pauseToggle();
    assert.deepEqual(sent.map(message => message.type), ['response.cancel', 'session.pause', 'session.resume']);
    assert.equal(session.paused, false);
    harness.speak('恢复后的新回答。');
    harness.advance(250);
    assert.equal(session.currentSpeakText, '恢复后的新回答。');
    assert.equal(harness.sources.length, 2);
    assert.equal(harness.sources[1].stopCount, 0);
});

test('force listen cancels current speech and rejects late output until disabled', () => {
    const harness = realtimeHarness();
    const { session, sent } = harness;
    harness.speak('正在回答。');
    harness.advance(250);
    session.toggleForceListen();
    assert.equal(session.forceListenActive, true);
    assert.equal(session._awaitingCancelAck, true);
    assert.deepEqual(sent.map(message => message.type), ['response.cancel']);
    assert.equal(session.audioPlayer.hasBufferedAudio, false);
    assert.equal(session.echoReference.playbackCount, 0);
    harness.speak('取消确认前的旧回答。');
    assert.equal(session.currentSpeakText, '');
    session._handleMessage({ type: 'session.state', reason: 'interrupt', state: 'active' });
    assert.equal(session._awaitingCancelAck, false);
    harness.speak('取消确认后的旧回答。');
    assert.equal(harness.sources.length, 1, 'forced listening rejects audio even after cancel acknowledgement');
    harness.context.taskContinuationForAssistantText = () => 'Continue';
    session._maybeAutoContinue('讲故事吗？');
    assert.equal(sent.length, 1, 'forced listening must not create automatic continuation input');
    session.sendChunk({ audio_base64: harness.audio });
    assert.equal(sent[1].input.force_listen, true);

    session.toggleForceListen();
    assert.equal(session.forceListenActive, false);
    harness.speak('重新开始回答。');
    harness.advance(250);
    assert.equal(session.currentSpeakText, '重新开始回答。');
    assert.equal(harness.sources.length, 2);

    session.toggleForceListen();
    assert.equal(session._awaitingCancelAck, true);
    session.toggleForceListen();
    assert.equal(session._awaitingCancelAck, true, 'turning force listen off must still wait for the server cancellation');
    session._handleMessage({ type: 'session.state', reason: 'interrupt', state: 'active' });
    assert.equal(session._awaitingCancelAck, false);
});

test('a fast resume waits for cancellation acknowledgement before accepting speech', () => {
    const harness = realtimeHarness();
    const { session, sent } = harness;
    session.pauseToggle();
    session.pauseToggle();
    assert.deepEqual(sent.map(message => message.type), ['response.cancel', 'session.pause', 'session.resume']);
    assert.equal(session.paused, false);
    assert.equal(session._awaitingCancelAck, true);
    harness.speak('暂停之前晚到的输出。');
    assert.equal(session.currentSpeakText, '');
    assert.equal(session.audioPlayer.hasBufferedAudio, false);
    session._handleMessage({ type: 'session.state', reason: 'interrupt', state: 'active' });
    harness.speak('新的回答。');
    assert.equal(session.currentSpeakText, '新的回答。');
});

test('fresh microphone PCM waits for cancel acknowledgement without being discarded by that acknowledgement', () => {
    for (const mode of ['force-listen', 'force-off', 'fast-resume']) {
        const harness = realtimeHarness({ maxInputInFlight: 1 });
        const { session, sent } = harness;
        if (mode === 'force-listen' || mode === 'force-off') {
            session.toggleForceListen();
            if (mode === 'force-off') session.toggleForceListen();
        }
        else {
            session.pauseToggle();
            session.pauseToggle();
        }
        const controlCount = sent.length;
        session.sendChunk({ audio_base64: 'fresh-one', speech_active: true });
        session.sendChunk({ audio_base64: 'fresh-two', speech_active: true });
        harness.advance(1000);
        assert.equal(sent.length, controlCount, `${mode} submitted new audio before cancel completed`);
        assert.equal(session._pendingInputs.length, 2);
        session._handleMessage({ type: 'response.backpressure', input_id: 'cancelled-input', retry_after_ms: 500 });
        assert.equal(session._inputBackpressureUntil, 0, 'late cancelled-input backpressure must not stall fresh PCM');
        session._handleMessage({ type: 'session.state', reason: 'interrupt', state: 'active' });
        harness.advance(0);
        assert.equal(sent[controlCount].input.audio, 'fresh-one');
        assert.equal(sent[controlCount].input.force_listen, mode === 'force-listen' ? true : undefined);
        session._handleMessage({ type: 'input.committed', input_id: sent[controlCount].message_id });
        harness.advance(0);
        assert.equal(sent[controlCount + 1].input.audio, 'fresh-two');
        assert.equal(session._pendingInputs.length, 0);
    }
});

test('slow model preserves every speech and silence packet in input order', () => {
    const harness = realtimeHarness({ maxInputInFlight: 2 });
    const { session, sent } = harness;
    const packets = [
        { audio_base64: 'speech-one', speech_active: true },
        { audio_base64: 'speech-two', speech_active: true },
        { audio_base64: 'silence-one', speech_active: false },
        { audio_base64: 'speech-three', speech_active: true },
        { audio_base64: 'speech-four', speech_active: true },
        { audio_base64: 'silence-two', speech_active: false },
    ];
    for (const packet of packets) session.sendChunk(packet);
    assert.deepEqual(sent.map(message => message.input.audio), ['speech-one', 'speech-two']);
    assert.equal(session._pendingInputs.length, 4);
    assert.equal(session._inputRetryTimer, null, 'full input windows wait for completion without polling');

    session._handleMessage({ type: 'input.committed', input_id: sent[0].message_id });
    // The newly captured packet must not overtake older pending audio in the
    // interval between the completion event and the queue's scheduled drain.
    session.sendChunk({ audio_base64: 'speech-five', speech_active: true });
    harness.advance(0);
    assert.equal(sent[2].input.audio, 'silence-one');
    assert.equal(session._recentInputs.size, 2);
    for (let index = 1; index < 7; index++) {
        session._handleMessage({ type: 'input.committed', input_id: sent[index].message_id });
        harness.advance(0);
        assert.ok(session._recentInputs.size <= 2, 'FIFO drain must preserve the inflight limit');
    }
    assert.deepEqual(sent.map(message => message.input.audio), [
        ...packets.map(packet => packet.audio_base64), 'speech-five',
    ]);
    assert.deepEqual(sent.map(message => message.input.speech_active), [true, true, false, true, true, false, true]);
    assert.equal(session._pendingInputs.length, 0);
    assert.equal(session._pendingInputBytes, 0);
});

test('a rejected input is retried before later queued audio', () => {
    const harness = realtimeHarness({ maxInputInFlight: 1 });
    const { session, sent } = harness;
    for (const name of ['first-speech', 'second-speech', 'silence']) {
        session.sendChunk({ audio_base64: name, speech_active: name !== 'silence' });
    }
    const firstID = sent[0].message_id;
    session._handleMessage({ type: 'response.backpressure', input_id: firstID, retry_after_ms: 100 });
    session.sendChunk({ audio_base64: 'third-speech', speech_active: true });
    harness.advance(99);
    assert.equal(sent.length, 1, 'backoff must hold the entire ordered queue');
    harness.advance(1);
    assert.equal(sent[1].message_id, firstID);
    assert.equal(sent[1].input.audio, 'first-speech');
    for (let index = 1; index < 5; index++) {
        session._handleMessage({ type: 'input.committed', input_id: sent[index].message_id });
        harness.advance(0);
    }
    assert.deepEqual(sent.map(message => message.input.audio), [
        'first-speech', 'first-speech', 'second-speech', 'silence', 'third-speech',
    ]);
    assert.equal(session._pendingInputBytes, 0);
});

test('native input windows wait for committed even when text, audio or listen arrives earlier', () => {
    const harness = realtimeHarness({ maxInputInFlight: 1 });
    const { session, sent } = harness;
    session.sendChunk({ audio_base64: 'first', speech_active: true });
    session.sendChunk({ audio_base64: 'second', speech_active: true });
    const inputID = sent[0].message_id;
    for (const delta of [
        { kind: 'text', text: '回答。' },
        { kind: 'audio', silent: true, audio: harness.audio },
        { kind: 'listen', reason: 'turn_end' },
    ]) {
        session._handleMessage({ type: 'response.output.delta', input_id: inputID, ...delta });
        harness.advance(0);
        assert.equal(sent.length, 1, `${delta.kind} prematurely released the native input window`);
    }
    session._handleMessage({ type: 'response.metrics', input_id: inputID, metrics: {} });
    harness.advance(0);
    assert.equal(sent.length, 1, 'metrics is not an input completion acknowledgement');
    session._handleMessage({ type: 'input.committed', input_id: inputID });
    harness.advance(0);
    assert.equal(sent[1].input.audio, 'second');
});

test('older gateways complete inputs with audio, listen or done while text is never an ack', () => {
    for (const completion of [
        { type: 'response.output.delta', kind: 'audio', silent: true },
        { type: 'response.output.delta', kind: 'listen', reason: 'turn_end' },
        { type: 'response.output.delta', kind: 'done' },
        { type: 'response.done' },
    ]) {
        const harness = realtimeHarness({ maxInputInFlight: 1 });
        const { session, sent } = harness;
        session._usesInputCommitted = false;
        session.sendChunk({ audio_base64: 'first', speech_active: true });
        session.sendChunk({ audio_base64: 'second', speech_active: true });
        const inputID = sent[0].message_id;
        session._handleMessage({ type: 'response.output.delta', kind: 'text', text: '回答。', input_id: inputID });
        harness.advance(0);
        assert.equal(sent.length, 1);
        session._handleMessage({ ...completion, input_id: inputID });
        harness.advance(0);
        assert.equal(sent[1].input.audio, 'second', `${completion.kind || completion.type} failed legacy completion`);
    }
});

test('ambiguous or out-of-order rejection cancels the timeline rather than replaying wrong audio', () => {
    for (const rejectedInput of ['older', 'older-after-completion', 'unknown']) {
        const harness = realtimeHarness({ maxInputInFlight: 2 });
        const { session, sent } = harness;
        session.sendChunk({ audio_base64: 'first', speech_active: true });
        session.sendChunk({ audio_base64: 'second', speech_active: true });
        session.sendChunk({ audio_base64: 'third', speech_active: true });
        if (rejectedInput === 'older-after-completion') {
            session._handleMessage({ type: 'input.committed', input_id: sent[1].message_id });
        }
        session._handleMessage({
            type: 'response.backpressure',
            ...(rejectedInput !== 'unknown' ? { input_id: sent[0].message_id } : {}),
            retry_after_ms: 100,
        });
        assert.equal(session.paused, true, `${rejectedInput} rejection must stop the inconsistent timeline`);
        assert.deepEqual(sent.slice(-2).map(message => message.type), ['response.cancel', 'session.pause']);
        harness.advance(1000);
        assert.equal(sent.filter(message => message.type === 'input.append').length, 2);
        assert.equal(session._pendingInputs.length, 0);
    }
});

test('queue overflow explicitly cancels and pauses the input timeline', () => {
    const harness = realtimeHarness({ maxInputInFlight: 1, maxPendingInputPackets: 2 });
    const { session, sent } = harness;
    const logs = [];
    const metrics = [];
    session.onSystemLog = message => logs.push(message);
    session.onMetrics = message => metrics.push(message);
    harness.speak();
    harness.advance(250);
    for (const name of ['speech-one', 'silence', 'speech-two', 'speech-three']) {
        session.sendChunk({ audio_base64: name, speech_active: name !== 'silence' });
    }
    assert.deepEqual(sent.map(message => message.type), ['input.append', 'response.cancel', 'session.pause']);
    assert.ok(logs.some(message => /backlog.*Session paused/.test(message)));
    assert.ok(metrics.some(message => message.type === 'input_overflow' && message.pending === 2));
    assert.equal(session.paused, true);
    assert.equal(session.pauseState, 'paused');
    assert.equal(session._awaitingCancelAck, true);
    assert.equal(session._pendingInputs.length, 0);
    assert.equal(session._recentInputs.size, 0);
    assert.equal(session._pendingInputBytes, 0);
    assert.equal(session._inputRetryTimer, null);
    assert.equal(session.audioPlayer.hasBufferedAudio, false);
    assert.equal(session.echoReference.playbackCount, 0);
    session.sendChunk({ audio_base64: 'still-paused', speech_active: true });
    harness.advance(1000);
    assert.equal(sent.length, 3, 'overflow must stop capture admission until the user resumes');
    session._handleMessage({ type: 'session.state', reason: 'interrupt', state: 'active' });
    session.pauseToggle();
    session.sendChunk({ audio_base64: 'fresh-speech', speech_active: true });
    assert.deepEqual(sent.map(message => message.type), [
        'input.append', 'response.cancel', 'session.pause', 'session.resume', 'input.append',
    ]);
    assert.equal(sent[4].input.audio, 'fresh-speech', 'resuming must not replay cancelled backlog');
});

test('payload and retry limits pause instead of dropping an input', () => {
    const oversized = realtimeHarness({ maxInputInFlight: 1, maxPendingInputBytes: 1024 });
    oversized.session.sendChunk({ audio_base64: 'first', speech_active: true });
    oversized.session.sendChunk({ audio_base64: 'a'.repeat(1024), speech_active: true });
    assert.equal(oversized.session.paused, true);
    assert.deepEqual(oversized.sent.map(message => message.type), ['input.append', 'response.cancel', 'session.pause']);

    const retrying = realtimeHarness({ maxInputInFlight: 1 });
    retrying.session.sendChunk({ audio_base64: 'speech', speech_active: true });
    const inputID = retrying.sent[0].message_id;
    for (let attempt = 0; attempt < 9; attempt++) {
        retrying.session._handleMessage({ type: 'response.backpressure', input_id: inputID, retry_after_ms: 100 });
        retrying.advance(100);
    }
    assert.equal(retrying.session.paused, true);
    assert.deepEqual(retrying.sent.slice(-2).map(message => message.type), ['response.cancel', 'session.pause']);
    assert.equal(retrying.sent.filter(message => message.type === 'input.append').length, 9);
    assert.equal(retrying.session._pendingInputs.length, 0);
});

test('cancel, reset, pause and transport close clear the complete input backlog', async () => {
    for (const action of ['cancel', 'interrupt', 'reset', 'pause', 'disconnect', 'cleanup']) {
        const harness = realtimeHarness({ maxInputInFlight: 1 });
        const { session, sent } = harness;
        session._cleaned = false;
        session.sendChunk({ audio_base64: 'first', speech_active: true });
        session.sendChunk({ audio_base64: 'second', speech_active: true });
        session.sendChunk({ audio_base64: 'silence', speech_active: false });
        assert.equal(session._pendingInputs.length, 2);
        if (action === 'cancel') session._cancelCurrentResponse();
        else if (action === 'interrupt' || action === 'reset') {
            session._handleMessage({ type: 'session.state', reason: action, state: 'active' });
        } else if (action === 'pause') session.pauseToggle();
        else if (action === 'cleanup') session.cleanup();
        else {
            session._started = true;
            session.ws.readyState = 3;
            await session._handleTransportClose();
        }
        assert.equal(session._pendingInputs.length, 0, `${action} retained pending audio`);
        assert.equal(session._pendingInputBytes, 0, `${action} retained pending payload bytes`);
        assert.equal(session._recentInputs.size, 0, `${action} retained old input acknowledgements`);
        assert.equal(session._inputRetryTimer, null, `${action} retained an input retry timer`);
        harness.advance(1000);
        assert.equal(sent.filter(message => message.type === 'input.append').length, 1, `${action} replayed cancelled audio`);
    }
});

test('explicit interruption and cleanup discard cancelled far-end references', () => {
    const harness = realtimeHarness();
    const { session } = harness;
    harness.speak();
    harness.advance(250);
    session._handleMessage({ type: 'response.output.delta', kind: 'listen', reason: 'turn_end' });
    assert.equal(session.echoReference.resetCount, 0, 'normal turn endings preserve the audible tail reference');
    assert.equal(session.echoReference.playbackCount, 1);
    session._handleMessage({ type: 'backend.state', reason: 'interrupt', state: 'active' });
    assert.equal(session.echoReference.playbackCount, 0);
    assert.equal(session.echoReference.resetCount, 1);
    harness.speak();
    harness.advance(250);
    assert.equal(session.echoReference.playbackCount, 1);
    session._cleaned = false;
    session.cleanup();
    assert.equal(session.echoReference.playbackCount, 0);
    assert.equal(session.audioPlayer.hasBufferedAudio, false);
    const closedText = session.currentSpeakText;
    harness.speak('清理后的晚到输出。');
    assert.equal(session.currentSpeakText, closedText, 'closed sessions ignore late text and audio');
});

test('AudioWorklet resampler emits target-rate chunks when the context stays at 48 kHz', () => {
    let Processor;
    const context = {
        sampleRate: 48000,
        Float32Array,
        AudioWorkletProcessor: class {
            constructor() {
                this.port = { onmessage: null, messages: [], postMessage: message => {
                    this.port.messages.push(message);
                } };
            }
        },
        registerProcessor: (_name, value) => { Processor = value; },
    };
    vm.runInNewContext(text('duplex/lib/capture-processor.js'), context);
    assert.ok(Processor, 'capture processor did not register');
    const node = new Processor({ processorOptions: { chunkSize: 16000, targetSampleRate: 16000 } });
    node.port.onmessage({ data: { command: 'start' } });
    const input = new Float32Array(128);
    input.fill(0.25);
    for (let i = 0; i < 1_600; i++) {
        node.process([[input]], [[new Float32Array(128)]]);
    }
    const fullChunks = node.port.messages.filter(item => item.type === 'chunk' && !item.final);
    assert.ok(fullChunks.length >= 3, `only ${fullChunks.length} full chunks emitted`);
    assert.ok(fullChunks.every(item => item.audio.length === 16000));
    node.port.onmessage({ data: { command: 'stop' } });
    assert.ok(node.port.messages.some(item => item.type === 'chunk' && item.final));
    const emitted = node.port.messages
        .filter(item => item.type === 'chunk')
        .reduce((total, item) => total + item.audio.length, 0);
    const sourceSamples = 1_600 * 128;
    const expected = Math.floor((sourceSamples - 1) / 3) + 1;
    assert.equal(emitted, expected, 'resampler drifted instead of preserving the target sample count');
});

test('live input keeps post-speech silence on the MiniCPM duplex timeline', async () => {
    const modulePath = pathToFileURL(join(webRoot, 'duplex/lib/live-input-timeline.js')).href;
    const { LiveInputTimeline } = await import(modulePath);
    const timeline = new LiveInputTimeline({ activityRmsThreshold: 0.01 });
    const silence = new Float32Array([0.001, -0.001]);
    const speech = new Float32Array([0.1, -0.1]);

    const startup = timeline.prepare(silence, 0.001);
    assert.equal(startup.send, false, 'startup silence must not trigger proactive speech');
    assert.equal(startup.responseWindowActive, false);

    const firstSpeech = timeline.prepare(speech, 0.1);
    assert.equal(firstSpeech.send, true);
    assert.equal(firstSpeech.samples, speech);
    assert.equal(firstSpeech.zeroFilled, false);
    assert.equal(firstSpeech.speechActive, true);
    // Stable-mode gating: while the user is actively speaking the window
    // stays closed so the model cannot interject fragments mid-question.
    assert.equal(firstSpeech.responseWindowActive, false);
    assert.equal(firstSpeech.prefixSamples.length, 1, 'first speech must retain one startup pre-roll frame');
    assert.deepEqual(Array.from(firstSpeech.prefixSamples[0]), Array.from(silence));

    const trailingSilence = timeline.prepare(silence, 0.001);
    assert.equal(trailingSilence.send, true, 'post-speech silence must preserve elapsed time');
    assert.equal(trailingSilence.zeroFilled, true);
    assert.equal(trailingSilence.speechActive, false);
    assert.equal(trailingSilence.responseWindowActive, true);
    assert.deepEqual(Array.from(trailingSilence.samples), [0, 0]);

    assert.equal(timeline.prepare(silence, 0.001).responseWindowActive, true);
    assert.equal(timeline.prepare(silence, 0.001).responseWindowActive, true);
    assert.equal(
        timeline.prepare(silence, 0.001).responseWindowActive,
        false,
        'long silence must restore deterministic listen decisions');

    const reopened = timeline.prepare(speech, 0.1);
    assert.equal(reopened.responseWindowActive, false, 'active speech keeps the sampled window closed');
    const postSpeech = timeline.prepare(silence, 0.001);
    assert.equal(postSpeech.responseWindowActive, true, 'the post-speech pause reopens the window');

    timeline.reset();
    assert.equal(timeline.prepare(silence, 0.001).send, false);

    timeline.openResponseWindow();
    const continuationSilence = timeline.prepare(silence, 0.001);
    assert.equal(continuationSilence.send, true);
    assert.equal(continuationSilence.responseWindowActive, true);
    assert.equal(continuationSilence.zeroFilled, true);
});

test('story confirmation gets one bounded internal continuation', async () => {
    const modulePath = pathToFileURL(join(webRoot, 'duplex/lib/task-continuation.js')).href;
    const { taskContinuationForAssistantText } = await import(modulePath);

    assert.match(
        taskContinuationForAssistantText('好呀，你想听什么样的故事呢？'),
        /三至五句话/);
    assert.match(
        taskContinuationForAssistantText('要不要听一个故事？'),
        /直接开始/);
    assert.equal(taskContinuationForAssistantText('从前有一只小兔子。'), null);
    assert.equal(taskContinuationForAssistantText('你想听什么音乐？'), null);
});

test('file replay preserves clear speech level and emits exact one-second packets', async () => {
    const modulePath = pathToFileURL(join(webRoot, 'duplex/lib/duplex-utils.js')).href;
    const { peakLimitedGain, splitFixedAudioChunks } = await import(modulePath);

    assert.equal(peakLimitedGain(new Float32Array([0.2, -0.3])), 1);
    assert.ok(Math.abs(peakLimitedGain(new Float32Array([1])) - 0.98) < 1e-6);

    const chunks = splitFixedAudioChunks(new Float32Array([0.1, 0.2, 0.3]), 2, 2);
    assert.equal(chunks.length, 2);
    assert.deepEqual(Array.from(chunks[0]), Array.from(new Float32Array([0.2, 0.4])));
    assert.deepEqual(Array.from(chunks[1]), Array.from(new Float32Array([0.6, 0])));
    assert.throws(() => splitFixedAudioChunks(new Float32Array([1]), 0), RangeError);
});

test('all live pages use the official realtime envelope and reconnect diagnostics', () => {
    const session = text('duplex/lib/realtime-session.js');
    assert.match(session, /session\.init/);
    assert.match(session, /input\.append/);
    assert.match(session, /_handleTransportClose/);
    assert.match(session, /_connectForReconnect/);
    assert.doesNotMatch(session, /_outbox/);
    assert.doesNotMatch(session, /\.splice\(0\)/);
    assert.match(session, /FarEndEchoReference/);
    assert.match(session, /prepareInputChunk/);
    assert.match(session, /allow_sampled_listen_speak_decision/);
    assert.match(session, /response\.backpressure/);
    assert.match(session, /get modelState\(\)/);
    assert.match(session, /onAutoContinuation/);
    assert.match(session, /conversation context reset/);
    assert.doesNotMatch(session, /audio stream resumed/);
    const audioDuplex = text('audio-duplex/audio-duplex-app.js');
    assert.match(audioDuplex, /force_listen_count:\s*0/);
    assert.match(audioDuplex, /top_k:\s*20/);
    assert.match(audioDuplex, /responseWindowActive/);
    assert.match(audioDuplex, /activity\.zeroFilled && session\.modelState === 'speaking'/);
    assert.doesNotMatch(audioDuplex, /用户要求讲故事、解释、列举或执行连续任务时/);
    const echo = text('duplex/lib/echo-reference.js');
    assert.match(echo, /suppressInput/);
    assert.match(echo, /echoSuppressed/);
    const half = text('half-duplex/half-duplex-app.js');
    assert.match(half, /\/v1\/realtime\?mode=half/);
    assert.doesNotMatch(half, /`\$\{proto\}:\/\/\$\{location\.host\}\/ws\/half_duplex\//);
    assert.match(half, /scheduleHalfReconnect/);
    assert.match(half, /halfOutbox/);
});

test('file mode drains an active assistant turn without waiting forever', async () => {
    const modulePath = pathToFileURL(join(webRoot, 'duplex/lib/file-response-drain.js')).href;
    const { FileResponseDrain } = await import(modulePath);

    const noResponse = new FileResponseDrain({ startGraceChunks: 2, maxChunks: 4 });
    assert.equal(noResponse.next('listening').continue, true);
    assert.equal(noResponse.next('listening').continue, true);
    assert.deepEqual(noResponse.next('listening'), {
        continue: false, reason: 'no_response', chunks: 2,
    });

    const response = new FileResponseDrain({ startGraceChunks: 2, maxChunks: 4 });
    assert.equal(response.next('listening').reason, 'start_grace');
    assert.equal(response.next('speaking').reason, 'speaking');
    assert.equal(response.next('speaking').continue, true);
    assert.deepEqual(response.next('listening'), {
        continue: false, reason: 'response_complete', chunks: 3,
    });

    const timeout = new FileResponseDrain({ startGraceChunks: 1, maxChunks: 2 });
    assert.equal(timeout.next('speaking').continue, true);
    assert.equal(timeout.next('speaking').continue, true);
    assert.deepEqual(timeout.next('speaking'), {
        continue: false, reason: 'timeout', chunks: 2,
    });

    const queuedPlayback = new FileResponseDrain({ startGraceChunks: 1, maxChunks: 3 });
    assert.equal(queuedPlayback.next('speaking').continue, true);
    assert.deepEqual(queuedPlayback.next('listening', true), {
        continue: true, reason: 'playback', chunks: 2,
    });
    assert.deepEqual(queuedPlayback.next('listening', false), {
        continue: false, reason: 'response_complete', chunks: 2,
    });
});

test('realtime session parses cleanly and has one KV-cache guard', () => {
    const sessionPath = join(webRoot, 'duplex/lib/realtime-session.js');
    assert.doesNotThrow(() => {
        execFileSync(process.execPath, ['--check', sessionPath], { stdio: 'pipe' });
    });

    const session = text('duplex/lib/realtime-session.js');
    const kvGuardHeaders = session.match(/^\s*_checkKvCache\(result\) \{$/gm) || [];
    assert.equal(kvGuardHeaders.length, 1, 'KV-cache guard method must be defined once');
    assert.doesNotMatch(
        session,
        /_checkKvCache\(result\) \{\s*\n\s*_checkKvCache\(result\) \{/,
        'duplicate adjacent KV-cache method headers found',
    );
});

test('duplex preparation is cancellable and decorative audio cannot block media startup', () => {
    const session = text('duplex/lib/realtime-session.js');
    assert.match(session, /_startGeneration/);
    assert.match(session, /_assertStartActive\(startGeneration\)/);
    assert.match(session, /cleanup\(\) \{[\s\S]*this\._startGeneration\+\+/);
    assert.ok(
        session.indexOf('this.audioPlayer.init();') < session.indexOf('await this.onPrepared();'),
        'speaker routing hook must run after AudioPlayer owns an AudioContext',
    );

    const devices = text('lib/audio-device-selector.js');
    assert.match(devices, /permissionStream\?\.getTracks\(\)\.forEach\(track => track\.stop\(\)\)/);

    const chimes = text('duplex/lib/queue-chimes.js');
    assert.match(chimes, /Promise\.race/);
    assert.match(chimes, /ctx\.state !== 'running'/);

    for (const page of ['audio-duplex/audio-duplex-app.js', 'omni/omni-app.js']) {
        const source = text(page);
        assert.match(source, /if \(session\.sessionId\)[\s\S]*session\.stop\(\)/);
        assert.match(source, /else if \(_queuePhase\) session\.cancelQueue\(\)/);
    }
});

test('stopping after session.created cannot revive a pending preparation', async () => {
    const source = text('duplex/lib/realtime-session.js')
        .replace(/^import .*;$/gm, '')
        .replace('export class RealtimeSession', 'class RealtimeSession')
        .concat('\nthis.RealtimeSession = RealtimeSession;');

    const sockets = [];
    class MockWebSocket {
        static OPEN = 1;
        constructor() {
            this.readyState = MockWebSocket.OPEN;
            this.sent = [];
            sockets.push(this);
            queueMicrotask(() => this.onopen?.());
        }
        send(payload) { this.sent.push(JSON.parse(payload)); }
        close() { this.readyState = 3; }
        deliver(payload) { this.onmessage?.({ data: JSON.stringify(payload) }); }
    }
    class MockAudioPlayer {
        constructor() { this.turnActive = false; }
        init() {}
        stop() {}
        stopAll() {}
        endTurn() {}
    }
    class MockEchoReference { pushPlayback() {} reset() {} }
    const context = {
        AudioPlayer: MockAudioPlayer,
        FarEndEchoReference: MockEchoReference,
        WebSocket: MockWebSocket,
        performance: { now: () => 0 },
        setTimeout,
        clearTimeout,
        console,
    };
    vm.runInNewContext(source, context);

    const session = new context.RealtimeSession('test', { getWsUrl: () => 'ws://test' });
    let releasePreparation;
    session.onPrepared = () => new Promise(resolve => { releasePreparation = resolve; });
    let cleanupCount = 0;
    session.onCleanup = () => { cleanupCount += 1; };
    const start = session.start('prompt', {}, async () => {});
    await new Promise(resolve => setImmediate(resolve));
    sockets[0].deliver({ type: 'session.queue_done' });
    sockets[0].deliver({ type: 'session.created', session_id: 'session-one' });
    await new Promise(resolve => setImmediate(resolve));

    session.stop();
    assert.equal(cleanupCount, 1);
    assert.ok(sockets[0].sent.some(message => message.type === 'session.close'));
    releasePreparation();
    await assert.rejects(start, /cancelled/);
    assert.equal(session.running, false);
    assert.equal(cleanupCount, 1);
});

test('queue cancellation clears delayed init and backpressure retries the rejected input id', async () => {
    const source = text('duplex/lib/realtime-session.js')
        .replace(/^import .*;$/gm, '')
        .replace('export class RealtimeSession', 'class RealtimeSession')
        .concat('\nthis.RealtimeSession = RealtimeSession;');

    let now = 0;
    let nextTimer = 1;
    const timers = new Map();
    const setTimer = (fn, delay = 0) => {
        const id = nextTimer++;
        timers.set(id, { fn, at: now + delay });
        return id;
    };
    const clearTimer = id => timers.delete(id);
    const runTimers = () => {
        const ready = [...timers.entries()].filter(([, timer]) => timer.at <= now);
        for (const [id, timer] of ready) {
            timers.delete(id);
            timer.fn();
        }
    };

    const sockets = [];
    class MockWebSocket {
        static OPEN = 1;
        constructor() {
            this.readyState = MockWebSocket.OPEN;
            this.sent = [];
            sockets.push(this);
            queueMicrotask(() => this.onopen?.());
        }
        send(payload) { this.sent.push(JSON.parse(payload)); }
        close() { this.readyState = 3; this.onclose?.(); }
    }
    class MockAudioPlayer {
        constructor() { this.turnActive = false; }
        init() {}
        stop() {}
        stopAll() {}
        endTurn() {}
    }
    class MockEchoReference {
        pushPlayback() {}
        reset() {}
        analyzeInput() { return {}; }
    }
    const context = {
        AudioPlayer: MockAudioPlayer,
        FarEndEchoReference: MockEchoReference,
        taskContinuationForAssistantText: () => null,
        WebSocket: MockWebSocket,
        performance: { now: () => now },
        setTimeout: setTimer,
        clearTimeout: clearTimer,
        requestAnimationFrame: callback => callback(),
        console,
    };
    vm.runInNewContext(source, context);

    const queued = new context.RealtimeSession('test', { getWsUrl: () => 'ws://test' });
    const starting = queued.start('prompt', {}, null);
    await new Promise(resolve => setImmediate(resolve));
    assert.ok(timers.size > 0, 'session.init fallback timer was not armed');
    queued.cancelQueue();
    await assert.rejects(starting, /cancelled/);
    assert.equal(timers.size, 0, 'cleanup left the session.init timer armed');

    const session = new context.RealtimeSession('test', { getWsUrl: () => 'ws://test' });
    session.ws = new MockWebSocket();
    session._sessionReady = true;
    session.sendChunk({
        audio_base64: 'AAAAAA==',
        speech_active: true,
        allow_sampled_listen_speak_decision: true,
        continuation_tick: true,
    });
    const first = session.ws.sent[0];
    assert.match(first.message_id, /^client_in_/);
    assert.equal(first.input.input_id, first.message_id);
    assert.equal(first.input.continuation_tick, true);

    session._handleMessage({
        type: 'response.backpressure',
        input_id: first.message_id,
        retry_after_ms: 100,
    });
    now = 100;
    runTimers();
    assert.equal(session.ws.sent.length, 2);
    assert.equal(session.ws.sent[1].message_id, first.message_id);
    assert.equal(session.ws.sent[1].input.input_id, first.message_id);

    // Older gateways do not echo input_id; retry the newest tracked input.
    session._handleMessage({
        type: 'response.backpressure',
        retry_after_ms: 100,
    });
    now = 200;
    runTimers();
    assert.equal(session.ws.sent.length, 3);
    assert.equal(session.ws.sent[2].message_id, first.message_id);
    assert.equal(session.ws.sent[2].input.input_id, first.message_id);

    const windowed = new context.RealtimeSession('test', {
        getWsUrl: () => 'ws://test',
        maxInputInFlight: 1,
    });
    windowed.ws = new MockWebSocket();
    windowed._sessionReady = true;
    windowed.sendChunk({ audio_base64: 'AAAAAA==', speech_active: false });
    windowed.sendChunk({ audio_base64: 'AQAAAA==', speech_active: false });
    const windowFirst = windowed.ws.sent[0];
    assert.equal(windowed.ws.sent.length, 1, 'the acknowledgement window leaked a second input');
    assert.equal(timers.size, 0, 'a full window must wait for an acknowledgement, not poll');

    windowed._handleMessage({
        type: 'input.committed',
        input_id: windowFirst.message_id,
        metrics: { kv_cache_length: 1 },
    });
    runTimers();
    assert.equal(windowed.ws.sent.length, 2, 'the pending input did not leave after commit acknowledgement');
    assert.notEqual(windowed.ws.sent[1].message_id, windowFirst.message_id);
});

test('session recorder removes paused wall-clock time from AI channel offsets', async () => {
    const originalPerformance = globalThis.performance;
    let now = 1_000;
    Object.defineProperty(globalThis, 'performance', {
        configurable: true,
        value: { now: () => now },
    });
    try {
        const modulePath = pathToFileURL(join(webRoot, 'duplex/lib/session-recorder.js')).href;
        const { SessionRecorder } = await import(`${modulePath}?pause-offset-test`);
        const recorder = new SessionRecorder(2, 2);
        recorder.start();
        recorder.pushLeft(new Float32Array(2));
        now = 2_000;
        recorder.pause();
        now = 5_000;
        recorder.resume();
        recorder.pushLeft(new Float32Array(2));
        now = 6_000;
        recorder.pushRight(new Float32Array([0.5, 0.5]), 2, now);
        assert.equal(recorder._rightEntries[0].offset, 4);
    } finally {
        Object.defineProperty(globalThis, 'performance', {
            configurable: true,
            value: originalPerformance,
        });
    }
});
