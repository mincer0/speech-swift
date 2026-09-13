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
