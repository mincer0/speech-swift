import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import vm from 'node:vm';
import test from 'node:test';

// Isolate the real session code from browser audio APIs. The fake clock runs
// both producer and backend deadlines so 1 Hz input is exercised against a
// serial worker that takes five seconds to commit each packet.
function sessionHarness(config = {}) {
    let now = 0;
    let nextTimer = 1;
    const timers = new Map();
    const sent = [];
    const logs = [];
    const metrics = [];
    const flow = [];
    const context = {
        performance: { now: () => now },
        setTimeout(fn, delay = 0) {
            const id = nextTimer++;
            timers.set(id, { fn, at: now + delay });
            return id;
        },
        clearTimeout: id => timers.delete(id),
        requestAnimationFrame: fn => fn(),
        Float32Array,
        Uint8Array,
        console: { warn() {} },
        WebSocket: { OPEN: 1 },
        AudioPlayer: class {
            constructor() { this.turnActive = false; }
            stopAll() { this.turnActive = false; }
            stop() { this.stopAll(); }
            endTurn() { this.turnActive = false; }
        },
        FarEndEchoReference: class {
            reset() {}
            analyzeInput() { return {}; }
        },
        taskContinuationForAssistantText: () => null,
    };
    const source = readFileSync(new URL(
        '../../Sources/MiniCPMDemoBackend/Resources/Web/duplex/lib/realtime-session.js',
        import.meta.url,
    ), 'utf8').replace(/^import .*;$/gm, '')
        .replace('export class RealtimeSession', 'class RealtimeSession')
        .concat('\nthis.RealtimeSession = RealtimeSession;');
    vm.runInNewContext(source, context);
    const session = new context.RealtimeSession('flow-test', config);
    session._started = true;
    session._cleaned = false;
    session._sessionReady = true;
    session._usesInputCommitted = true;
    session.ws = {
        readyState: 1,
        send(payload) { sent.push(JSON.parse(payload)); },
        close() { this.readyState = 3; },
    };
    session.onSystemLog = log => logs.push(log);
    session.onMetrics = metric => metrics.push(metric);
    session.onInputFlowChange = state => flow.push(state);
    return {
        session, sent, logs, metrics, flow, context,
        now: () => now,
        send(name) { session.sendChunk({ audio_base64: name, speech_active: true }); },
        commit(inputID) {
            session._handleMessage({ type: 'input.committed', input_id: inputID });
        },
        advance(ms) {
            const target = now + ms;
            let executions = 0;
            while (true) {
                const next = [...timers].filter(([, timer]) => timer.at <= target)
                    .sort((a, b) => a[1].at - b[1].at || a[0] - b[0])[0];
                if (!next) break;
                assert.ok(++executions < 10000, 'timer loop did not make progress');
                const [id, timer] = next;
                timers.delete(id);
                now = timer.at;
                timer.fn();
            }
            now = target;
        },
    };
}

test('1 Hz capture holds and resumes against five-second commits without cancelling or losing FIFO audio', () => {
    const harness = sessionHarness({ enableInputFlowControl: true });
    const { session, context, sent, logs, metrics } = harness;
    const captured = [];
    const committed = [];
    const backend = [];
    let backendBusy = false;
    let captureHeld = false;
    let maximumPending = 0;
    let resumes = 0;
    session.onInputFlowChange = state => {
        captureHeld = state === 'held';
        if (state === 'active') {
            assert.equal(session._pendingInputs.length, 0);
            assert.ok(session._recentInputs.size < session.config.maxInputInFlight);
            resumes++;
        }
    };
    const runBackend = () => {
        if (backendBusy || !backend.length) return;
        backendBusy = true;
        context.setTimeout(() => {
            const packet = backend.shift();
            committed.push(packet.input.audio);
            backendBusy = false;
            harness.commit(packet.message_id);
            runBackend();
        }, 5000);
    };
    session.ws.send = payload => {
        const event = JSON.parse(payload);
        sent.push(event);
        if (event.type === 'input.append') {
            backend.push(event);
            runBackend();
        }
    };
    const tick = () => {
        if (!captureHeld && captured.length < 24) {
            const packet = `speech-${captured.length + 1}`;
            captured.push(packet);
            harness.send(packet);
            maximumPending = Math.max(maximumPending, session._pendingInputs.length);
        }
        if (captured.length < 24) context.setTimeout(tick, 1000);
    };
    context.setTimeout(tick, 1000);
    harness.advance(160000);
    assert.equal(captured.length, 24);
    assert.deepEqual(committed, captured, 'every captured packet must commit in source order');
    assert.deepEqual(sent.map(event => event.input.audio), captured);
    assert.ok(sent.every(event => event.type === 'input.append'), 'flow hold must not cancel or pause the session');
    assert.equal(maximumPending, 4);
    assert.ok(resumes >= 3, 'sustained input should enter several hold/resume cycles');
    assert.equal(session.paused, false);
    assert.equal(session.inputFlowHeld, false);
    assert.ok(logs.some(log => log.includes('input capture temporarily paused')));
    assert.ok(logs.some(log => log.includes('input capture resumed')));
    assert.ok(!metrics.some(metric => metric.type === 'input_overflow'));
});

test('hold is synchronous, keeps an already posted worklet packet, and waits for committed capacity', () => {
    const harness = sessionHarness({
        enableInputFlowControl: true, maxInputInFlight: 1, inputFlowHighWaterPackets: 1,
    });
    const { session, flow, sent } = harness;
    harness.send('first');
    harness.send('second');
    assert.equal(session.inputFlowHeld, true);
    assert.deepEqual(flow, ['held'], 'producer must be notified before sendChunk returns');
    harness.send('already-posted-third');
    assert.deepEqual(Array.from(session._pendingInputs, entry => entry.event.input.audio),
        ['second', 'already-posted-third']);
    for (const expected of ['second', 'already-posted-third']) {
        harness.commit(sent.at(-1).message_id);
        harness.advance(0);
        assert.equal(sent.at(-1).input.audio, expected);
        assert.equal(session.inputFlowHeld, true, 'an empty pending queue still needs an input credit');
    }
    harness.commit(sent.at(-1).message_id);
    assert.equal(session.inputFlowHeld, false);
    assert.deepEqual(flow, ['held', 'active']);
    assert.equal(session.paused, false);
});

test('unadapted producers retain the hard overflow guard and flow control is opt in', () => {
    const legacy = sessionHarness({ maxInputInFlight: 1 });
    for (let index = 0; index < 10; index++) legacy.send(`legacy-${index}`);
    assert.equal(legacy.session.inputFlowHeld, false);
    assert.deepEqual(legacy.flow, []);
    assert.equal(legacy.session._pendingInputs.length, 9);

    const harness = sessionHarness({
        enableInputFlowControl: true, maxInputInFlight: 1,
        maxPendingInputPackets: 3, inputFlowHighWaterPackets: 1,
    });
    for (let index = 0; index < 5; index++) harness.send(`uncooperative-${index}`);
    assert.equal(harness.session.paused, true);
    assert.equal(harness.session.inputFlowHeld, false);
    assert.deepEqual(harness.sent.map(event => event.type),
        ['input.append', 'response.cancel', 'session.pause']);
    assert.ok(harness.metrics.some(metric => metric.type === 'input_overflow'));
});

test('flow watermark defaults and explicit values are bounded by the packet limit', () => {
    assert.equal(sessionHarness({ maxPendingInputPackets: 2 }).session.config.inputFlowHighWaterPackets, 2);
    assert.equal(sessionHarness({ maxPendingInputPackets: 16 }).session.config.inputFlowHighWaterPackets, 4);
    assert.equal(sessionHarness({ maxPendingInputPackets: 3, inputFlowHighWaterPackets: 12 })
        .session.config.inputFlowHighWaterPackets, 3);
    assert.equal(sessionHarness({ inputFlowHighWaterPackets: 0 }).session.config.inputFlowHighWaterPackets, 1);
});

test('explicit pause, stop, reset, cleanup and disconnect end a hold without reviving capture', async () => {
    for (const action of ['pause', 'stop', 'reset', 'cleanup', 'disconnect']) {
        const harness = sessionHarness({
            enableInputFlowControl: true, maxInputInFlight: 1, inputFlowHighWaterPackets: 1,
        });
        const { session } = harness;
        let captureRunning = true;
        session.onInputFlowChange = state => {
            captureRunning = state === 'active' && session.running && !session.paused
                && session._sessionReady && !session._intentionalClose;
        };
        harness.send('first');
        harness.send('queued');
        assert.equal(captureRunning, false);
        if (action === 'pause') session.pauseToggle();
        else if (action === 'stop') session.stop();
        else if (action === 'reset') session._reset();
        else if (action === 'cleanup') session.cleanup();
        else {
            session._sessionArgs = null;
            await session._handleTransportClose();
        }
        assert.equal(session.inputFlowHeld, false, `${action} retained hold`);
        assert.equal(captureRunning, false, `${action} revived capture`);
        assert.equal(session._pendingInputs.length, 0, `${action} retained cancelled packets`);
        assert.ok(!harness.logs.some(log => log.includes('input capture resumed')),
            `${action} should not be described as catching up`);
    }
});

test('explicit cancellation queues synchronously resumed fresh input until the interrupt ACK', () => {
    const harness = sessionHarness({
        enableInputFlowControl: true, maxInputInFlight: 1, inputFlowHighWaterPackets: 2,
    });
    const { session, sent } = harness;
    harness.send('old-first');
    harness.send('old-second');
    harness.send('old-third');
    session.onInputFlowChange = state => {
        if (state === 'active') harness.send('fresh-after-cancel');
    };
    session._cancelCurrentResponse();
    assert.deepEqual(sent.map(event => event.type), ['input.append', 'response.cancel']);
    assert.equal(session._pendingInputs.length, 1);
    assert.equal(session._pendingInputs[0].event.input.audio, 'fresh-after-cancel');
    session._handleMessage({ type: 'session.state', reason: 'interrupt', state: 'active' });
    harness.advance(0);
    assert.equal(sent.at(-1).input.audio, 'fresh-after-cancel');
    assert.equal(sent.filter(event => event.type === 'input.append').length, 2);
});

test('two rapid pause and resume cycles retain fresh audio until both cancel ACKs arrive', () => {
    for (const ackType of ['session.state', 'backend.state']) {
        const h = sessionHarness({ enableInputFlowControl: true });
        const { session, sent } = h;
        session.pauseToggle(); session.pauseToggle();
        session.pauseToggle(); session.pauseToggle();
        h.send('fresh-after-second-resume');
        assert.equal(session._pendingCancelAcks, 2);
        assert.equal(session._awaitingCancelAck, true);
        assert.equal(sent.filter(event => event.type === 'response.cancel').length, 2);

        session._handleMessage({ type: ackType, reason: 'interrupt', state: 'active' });
        h.advance(0);
        assert.equal(session._pendingCancelAcks, 1);
        assert.equal(session._awaitingCancelAck, true, 'the first ACK must not reopen the input window');
        assert.equal(sent.filter(event => event.type === 'input.append').length, 0);
        h.send('fresh-after-first-ACK');
        const resultCount = session._resultCount;
        session._handleMessage({ type: 'response.output.delta', kind: 'text', text: 'stale speech' });
        assert.equal(session._resultCount, resultCount, 'speech stays blocked until every cancel is acknowledged');

        session._handleMessage({ type: ackType, reason: 'interrupt', state: 'active' });
        assert.equal(session._pendingCancelAcks, 0);
        assert.equal(session._awaitingCancelAck, false);
        h.advance(0);
        assert.deepEqual(sent.filter(event => event.type === 'input.append').map(event => event.input.audio),
            ['fresh-after-second-resume', 'fresh-after-first-ACK']);
        assert.equal(session._pendingInputs.length, 0);
    }
});

test('stop, cleanup and reset clear every pending cancel without replaying queued input', () => {
    for (const action of ['stop', 'cleanup', '_reset']) {
        const h = sessionHarness();
        h.session._cancelCurrentResponse(); h.session._cancelCurrentResponse();
        h.send('waiting-for-both-cancels');
        assert.equal(h.session._pendingCancelAcks, 2);
        h.session[action]();
        h.advance(0);
        assert.equal(h.session._pendingCancelAcks, 0, `${action} kept cancel acknowledgements from an old timeline`);
        assert.equal(h.session._awaitingCancelAck, false);
        assert.equal(h.session._pendingInputs.length, 0);
        assert.equal(h.sent.filter(event => event.type === 'input.append').length, 0);
    }
});

test('reconnect clears pending cancel ACKs before admitting input to the new session', async () => {
    const h = sessionHarness();
    h.session._cancelCurrentResponse(); h.session._cancelCurrentResponse();
    h.send('old-session-packet');
    h.session._sessionArgs = { systemPrompt: 'prompt', preparePayload: {} };
    h.session._connectForReconnect = async () => {
        assert.equal(h.session._pendingCancelAcks, 0, 'the new transport cannot inherit old cancel barriers');
        assert.equal(h.session._pendingInputs.length, 0);
        h.session._sessionReady = true;
    };
    await h.session._handleTransportClose();
    h.send('new-session-packet');
    h.advance(0);
    assert.equal(h.session._awaitingCancelAck, false);
    assert.deepEqual(h.sent.filter(event => event.type === 'input.append').map(event => event.input.audio),
        ['new-session-packet']);
});

test('unsent cancel attempts do not consume an ACK slot from an earlier sent cancel', () => {
    const h = sessionHarness();
    h.session._cancelCurrentResponse();
    h.session.ws.readyState = 0;
    h.session._cancelCurrentResponse();
    assert.equal(h.session._pendingCancelAcks, 1);
    assert.equal(h.sent.filter(event => event.type === 'response.cancel').length, 1);
    h.session._handleMessage({ type: 'session.state', reason: 'interrupt', state: 'active' });
    assert.equal(h.session._pendingCancelAcks, 0);
    assert.equal(h.session._awaitingCancelAck, false);
});

test('manual pause without an automatic hold resets active input cadence before synchronous resume', () => {
    const h = sessionHarness();
    const warnings = [];
    h.context.console.warn = message => warnings.push(message);
    h.send('first');
    h.session.pauseToggle();
    h.session._handleMessage({ type: 'session.state', reason: 'interrupt', state: 'active' });
    h.advance(6000);
    h.session.onPauseStateChange = state => { if (state === 'active') h.send('resumed'); };
    h.session.pauseToggle();
    assert.deepEqual(warnings, [], 'an intentional pause is not a feed timer failure');
    h.advance(4000); h.send('late');
    assert.equal(warnings.length, 1, 'unexpected active capture delays must remain visible');
    assert.match(warnings[0], /Feed cadence slip/);
});

const windowMetrics = (length, overrides = {}) => ({
    backend: 'swift_mlx', kv_cache_length: length, engine_epoch: 1,
    sliding_window_mode: 'basic', sliding_window_high_watermark: 100,
    sliding_window_low_watermark: 70, ...overrides,
});

test('interrupt, reset and generation epoch changes reset KV comparison instead of auto-stopping', () => {
    for (const action of ['interrupt', 'reset', 'epoch']) {
        const harness = sessionHarness({ getStopOnSlidingWindow: () => true });
        harness.session._handleMetrics(windowMetrics(112));
        if (action === 'epoch') harness.session._handleMetrics(windowMetrics(70, { engine_epoch: 2 }));
        else {
            harness.session._handleMessage({
                type: 'session.state', reason: action, state: 'active', metrics: windowMetrics(70),
            });
        }
        harness.advance(0);
        assert.equal(harness.session.running, true, `${action} incorrectly stopped the session`);
        assert.ok(!harness.logs.some(log => log.includes('KV pruned')), `${action} was labelled pruning`);
        assert.equal(harness.session._lastKvCacheLength, 70);
    }
});

test('KV decreases with off, unknown, context or missing window evidence never trigger sliding-window stop', () => {
    for (const mode of ['off', undefined, 'context', 'unknown']) {
        const harness = sessionHarness({ getStopOnSlidingWindow: () => true });
        harness.session._handleMetrics(windowMetrics(112, { sliding_window_mode: mode }));
        harness.session._handleMetrics(windowMetrics(70, { sliding_window_mode: mode }));
        harness.advance(0);
        assert.equal(harness.session.running, true, `${mode} incorrectly stopped the session`);
        assert.ok(!harness.logs.some(log => log.includes('KV pruned')));
    }
    const legacy = sessionHarness({ getStopOnSlidingWindow: () => true });
    legacy.session._usesInputCommitted = false;
    legacy.session._handleMessage({ type: 'response.listen', kv_cache_length: 112 });
    legacy.session._handleMessage({ type: 'response.listen', kv_cache_length: 70 });
    legacy.advance(0);
    assert.equal(legacy.session.running, true);
    assert.deepEqual(legacy.logs, []);
});

test('basic-window decrease needs both watermark crossings and a known same native generation', () => {
    for (const [previous, current, overrides] of [
        [90, 70, {}], [112, 85, {}], [112, 70, { engine_epoch: undefined }],
        [112, 70, { sliding_window_low_watermark: undefined }],
    ]) {
        const harness = sessionHarness({ getStopOnSlidingWindow: () => true });
        harness.session._handleMetrics(windowMetrics(previous, overrides));
        harness.session._handleMetrics(windowMetrics(current, overrides));
        harness.advance(0);
        assert.equal(harness.session.running, true);
        assert.ok(!harness.logs.some(log => log.includes('KV pruned')));
    }
});

test('confirmed basic-window high-to-low transition preserves the stop-on-sliding-window setting', () => {
    const harness = sessionHarness({ getStopOnSlidingWindow: () => true });
    harness.session._handleMetrics(windowMetrics(112));
    harness.session._handleMetrics(windowMetrics(70));
    assert.ok(harness.logs.some(log => log === '✂ KV pruned: 112 → 70'));
    harness.advance(0);
    assert.equal(harness.session.running, false);
    assert.equal(harness.sent.at(-1).type, 'session.close');
});

test('older gateways require explicit basic-window evidence to retain sliding-window detection', () => {
    for (const type of ['response.listen', 'result']) {
        const harness = sessionHarness({ getStopOnSlidingWindow: () => true });
        harness.session._usesInputCommitted = false;
        const metrics = length => ({
            type, is_listen: true, kv_cache_length: length,
            sliding_window_mode: 'basic', sliding_window_high_watermark: 100,
            sliding_window_low_watermark: 70,
        });
        harness.session._handleMessage(metrics(112));
        harness.session._handleMessage(metrics(70));
        assert.ok(harness.logs.some(log => log.includes('KV pruned')));
        harness.advance(0);
        assert.equal(harness.session.running, false);
    }
});
