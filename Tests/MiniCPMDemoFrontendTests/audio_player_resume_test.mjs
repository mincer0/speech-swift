import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import vm from 'node:vm';
import test from 'node:test';

const playerSource = readFileSync(new URL(
    '../../Sources/MiniCPMDemoBackend/Resources/Web/duplex/lib/audio-player.js', import.meta.url,
), 'utf8').replace(/^import .*;$/gm, '').replace(/^export /gm, '')
    + '\nthis.AudioPlayer = AudioPlayer;';

function deferred() {
    let resolve;
    let reject;
    const promise = new Promise((yes, no) => { resolve = yes; reject = no; });
    return { promise, resolve, reject };
}

function harness() {
    let now = 1_000;
    let nextTimer = 0;
    const timers = new Map();
    const sources = [];
    const recordings = [];
    const errors = [];
    const warnings = [];
    const resumes = [];
    const cancellations = [];
    const context = {
        sampleRate: 24_000, currentTime: 0, state: 'suspended', destination: {},
        resume() {
            const result = deferred();
            resumes.push(result);
            return result.promise;
        },
        close() { this.state = 'closed'; return Promise.resolve(); },
        createBuffer(_channels, count, rate) {
            const samples = new Float32Array(count);
            return { duration: count / rate, getChannelData: () => samples, samples };
        },
        createBufferSource() {
            const source = {
                connect() {}, disconnect() {}, stopCount: 0,
                start(time) { this.startedAt = time; },
                stop() { this.stopCount++; },
            };
            sources.push(source);
            return source;
        },
    };
    const scope = {
        AudioContext: function () { return context; },
        Float32Array, Uint8Array,
        atob: value => Buffer.from(value, 'base64').toString('binary'),
        resampleAudio: samples => samples,
        performance: { now: () => now },
        setTimeout(fn, delay = 0) {
            const id = ++nextTimer;
            timers.set(id, { fn, at: now + delay });
            return id;
        },
        clearTimeout: id => timers.delete(id),
        setInterval: () => ++nextTimer, clearInterval() {},
        requestAnimationFrame: callback => callback(),
        console: { log() {}, warn: (...args) => warnings.push(args) },
    };
    vm.runInNewContext(playerSource, scope);
    const player = new scope.AudioPlayer({ getPlaybackDelayMs: () => 0 });
    player.onRawAudio = (samples, rate, timestamp) => recordings.push({ samples, rate, timestamp });
    player.onPlaybackError = error => errors.push(error);
    player.onPlaybackCancel = timestamp => cancellations.push(timestamp);
    player.init();
    player.beginTurn();
    return {
        player, context, sources, recordings, errors, warnings, resumes, cancellations, scope,
        enqueue(value = 0.1) {
            const samples = value instanceof Float32Array ? value : new Float32Array(24_000).fill(value);
            player.playChunk(Buffer.from(samples.buffer).toString('base64'), now);
        },
        advance(ms) {
            now += ms;
            if (context.state === 'running') context.currentTime += ms / 1000;
            for (const [id, timer] of [...timers]) {
                if (timer.at <= now) { timers.delete(id); timer.fn(); }
            }
        },
        now: () => now,
        async recover(index = resumes.length - 1) {
            context.state = 'running';
            resumes[index].resolve();
            await flush();
        },
    };
}

const flush = () => new Promise(resolve => setImmediate(resolve));

function attachRecorder(h) {
    const source = readFileSync(new URL(
        '../../Sources/MiniCPMDemoBackend/Resources/Web/duplex/lib/session-recorder.js', import.meta.url,
    ), 'utf8').replace(/^import .*;$/gm, '').replace(/^export /gm, '')
        + '\nthis.SessionRecorder = SessionRecorder;';
    h.scope.Blob = Blob;
    vm.runInNewContext(source, h.scope);
    const recorder = new h.scope.SessionRecorder(24_000, 24_000);
    recorder.start();
    h.player.onRawAudio = (samples, rate, timestamp) => {
        h.recordings.push({ samples, rate, timestamp });
        recorder.pushRight(samples, rate, timestamp);
    };
    h.player.onPlaybackCancel = timestamp => {
        h.cancellations.push(timestamp);
        recorder.cancelRight(timestamp);
    };
    return recorder;
}

test('suspended output waits for recovery before scheduling PCM or recording timestamps', async () => {
    const h = harness();
    h.enqueue(); h.advance(250);
    assert.equal(h.resumes.length, 1);
    assert.equal(h.player.playing, false);
    assert.equal(h.sources.length, 0);
    assert.equal(h.recordings.length, 0);
    h.advance(650);
    await h.recover();
    assert.equal(h.player.playing, true);
    assert.equal(h.sources.length, 1);
    assert.equal(h.sources[0].startedAt, h.context.currentTime);
    assert.equal(h.recordings[0].timestamp, h.now(), 'recording must use the recovered output clock');
    assert.equal(h.player._playbackStartTime, h.now());
    assert.equal(h.errors.length, 0);
    h.player.stopAll();
});

test('chunks arriving during output recovery keep FIFO order and use one resume operation', async () => {
    const h = harness();
    h.enqueue(0.1); h.advance(250);
    h.enqueue(0.2); h.advance(750);
    h.enqueue(0.3); h.player.endTurn();
    assert.equal(h.resumes.length, 1);
    assert.equal(h.sources.length, 0);
    await h.recover();
    assert.deepEqual(h.sources.map(source => source.startedAt), [0, 1, 2]);
    assert.deepEqual(h.recordings.map(entry => Math.round(entry.samples[0] * 10)), [1, 2, 3]);
    assert.deepEqual(h.recordings.map(entry => entry.timestamp), [h.now(), h.now() + 1000, h.now() + 2000]);
    assert.equal(h.player._aheadInterval, null, 'recovery must not reopen monitoring for an ended turn');
    h.player.stopAll();
});

test('rejected recovery is reported once, retains PCM, and permits a later successful retry', async () => {
    const h = harness();
    h.enqueue(); h.advance(250);
    h.resumes[0].reject(new Error('device recovery denied'));
    await flush();
    assert.equal(h.errors.length, 1);
    assert.match(h.errors[0].message, /device recovery denied/);
    assert.equal(h.warnings.length, 1);
    assert.equal(h.player.playing, false);
    assert.equal(h.player._pendingChunks.length, 1);
    assert.equal(h.sources.length, 0);
    assert.equal(h.recordings.length, 0);
    h.player._startPlayback();
    assert.equal(h.resumes.length, 2);
    h.advance(600);
    await h.recover();
    assert.equal(h.sources.length, 1);
    assert.equal(h.recordings[0].timestamp, h.now());
    assert.equal(h.errors.length, 1);
    h.player.stopAll();
});

test('synchronous recovery exceptions do not escape as unhandled rejections', async () => {
    const h = harness();
    h.context.resume = () => { throw new Error('output already closed'); };
    h.enqueue(); h.advance(250);
    await flush();
    assert.equal(h.errors.length, 1);
    assert.match(h.errors[0].message, /output already closed/);
    assert.equal(h.player.playing, false);
    assert.equal(h.sources.length, 0);
    h.player.stopAll();
});

test('cancellation while recovery is pending cannot resurrect cancelled playback', async () => {
    const h = harness();
    h.enqueue(); h.advance(250);
    h.player.stopAll();
    h.advance(1000);
    await h.recover();
    assert.equal(h.player.playing, false);
    assert.equal(h.player.turnActive, false);
    assert.equal(h.player.hasBufferedAudio, false);
    assert.equal(h.sources.length, 0);
    assert.equal(h.recordings.length, 0);
    assert.equal(h.errors.length, 0);
});

test('a fresh turn during cancelled recovery schedules only its own PCM', async () => {
    const h = harness();
    h.enqueue(0.1); h.advance(250);
    h.player.stopAll();
    h.player.beginTurn();
    h.enqueue(0.8); h.advance(250);
    assert.equal(h.resumes.length, 1, 'a fresh request may share the uncancellable context operation');
    await h.recover();
    assert.equal(h.sources.length, 1, 'the obsolete recovery callback must not schedule the old queue');
    assert.equal(Math.round(h.recordings[0].samples[0] * 10), 8);
    assert.equal(h.recordings[0].timestamp, h.now());
    h.player.stopAll();
});

test('normal turn boundaries during recovery preserve the previous unplayed tail', async () => {
    const h = harness();
    h.enqueue(0.1); h.advance(250);
    h.player.endTurn();
    h.player.beginTurn();
    h.enqueue(0.2);
    assert.equal(h.resumes.length, 1);
    await h.recover();
    assert.deepEqual(h.sources.map(source => source.startedAt), [0, 1]);
    assert.deepEqual(h.recordings.map(entry => Math.round(entry.samples[0] * 10)), [1, 2]);
    h.player.stopAll();
});

test('obsolete recovery rejection after a stop is consumed without reporting a new-session error', async () => {
    const h = harness();
    h.enqueue(); h.advance(250);
    h.player.stop();
    h.resumes[0].reject(new Error('old context closed'));
    await flush();
    assert.equal(h.sources.length, 0);
    assert.equal(h.recordings.length, 0);
    assert.equal(h.errors.length, 0);
    assert.equal(h.warnings.length, 0);
    assert.equal(h.player.ctx, null);
});

test('mid-playback context suspension retains the unheard tail and realigns the recording clock', async () => {
    for (const state of ['suspended', 'interrupted']) {
        const h = harness();
        const recorderSource = readFileSync(new URL(
            '../../Sources/MiniCPMDemoBackend/Resources/Web/duplex/lib/session-recorder.js', import.meta.url,
        ), 'utf8').replace(/^import .*;$/gm, '').replace(/^export /gm, '')
            + '\nthis.SessionRecorder = SessionRecorder;';
        vm.runInNewContext(recorderSource, h.scope);
        const recorder = new h.scope.SessionRecorder(24_000, 24_000);
        recorder.start();
        h.player.onRawAudio = (samples, rate, timestamp) => {
            h.recordings.push({ samples, rate, timestamp });
            recorder.pushRight(samples, rate, timestamp);
        };
        h.player.onPlaybackCancel = timestamp => {
            h.cancellations.push(timestamp);
            recorder.cancelRight(timestamp);
        };
        h.context.state = 'running';
        h.enqueue(Float32Array.from({ length: 24_000 }, (_, index) => index / 24_000));
        h.advance(250); h.enqueue(0.2);
        h.advance(400);
        h.context.state = state;
        h.advance(2000); h.enqueue(0.3);
        assert.equal(h.resumes.length, 1, `a ${state} context must resume before new scheduling`);
        assert.equal(h.player.playing, false);
        assert.deepEqual(h.sources.map(source => source.stopCount), [1, 1]);
        assert.equal(h.sources.length, 2, 'no new source may start against the frozen context clock');
        assert.deepEqual(h.cancellations, [1650], 'recording must crop at the audible prefix, before the freeze');
        assert.equal(recorder._rightEntries.length, 1);
        assert.equal(recorder._rightEntries[0].data.length, 9600, 'retain exactly the 400 ms already heard');
        assert.deepEqual(Array.from(h.player._pendingChunks, chunk => chunk.raw.length), [14_400, 24_000, 24_000]);

        h.advance(100); await h.recover();
        assert.deepEqual(h.sources.slice(2).map(source => source.startedAt), [0.65, 1.25, 2.25]);
        const recovered = h.recordings.slice(2);
        assert.ok(Math.abs(recovered[0].samples[0] - 0.4) < 1e-6, 'a recovery must not repeat the played prefix');
        assert.deepEqual(recovered.map(entry => entry.timestamp), [3750, 4350, 5350]);
        assert.deepEqual(Array.from(recorder._rightEntries, entry => entry.offset), [6000, 66000, 80400, 104400]);
        assert.deepEqual(Array.from(recorder._rightEntries, entry => entry.data.length), [9600, 14400, 24000, 24000]);
        assert.equal(h.errors.length, 0);
        h.player.stopAll();
    }
});

test('failed mid-playback recovery retains all unheard PCM until a successful retry', async () => {
    const h = harness();
    h.context.state = 'running';
    h.enqueue(0.1); h.advance(250); h.advance(400);
    h.context.state = 'suspended'; h.advance(1000); h.enqueue(0.2);
    h.resumes[0].reject(new Error('audio device unavailable'));
    await flush();
    assert.equal(h.player.playing, false);
    assert.equal(h.errors.length, 1);
    assert.match(h.errors[0].message, /audio device unavailable/);
    assert.deepEqual(Array.from(h.player._pendingChunks, chunk => chunk.raw.length), [14400, 24000]);
    assert.equal(h.sources.length, 1);
    assert.equal(h.sources[0].stopCount, 1);
    assert.deepEqual(h.cancellations, [1650]);

    h.advance(600); h.player._startPlayback();
    assert.equal(h.resumes.length, 2);
    await h.recover();
    assert.deepEqual(h.sources.slice(1).map(source => source.startedAt), [0.65, 1.25]);
    assert.deepEqual(h.recordings.slice(1).map(entry => entry.timestamp), [3250, 3850]);
    assert.deepEqual(h.recordings.slice(1).map(entry => entry.samples.length), [14400, 24000]);
    assert.equal(h.player._pendingChunks.length, 0);
    h.player.stopAll();
});

test('device recovery clears the old session echo schedule and rebuilds it without replacing recording cancellation', async () => {
    for (const state of ['suspended', 'interrupted']) {
        const h = harness();
        const recorder = attachRecorder(h);
        for (const [file, name] of [
            ['echo-reference.js', 'FarEndEchoReference'],
            ['realtime-session.js', 'RealtimeSession'],
        ]) {
            const source = readFileSync(new URL(
                `../../Sources/MiniCPMDemoBackend/Resources/Web/duplex/lib/${file}`, import.meta.url,
            ), 'utf8').replace(/^import .*;$/gm, '').replace(/^export /gm, '')
                + `\nthis.${name} = ${name};`;
            vm.runInNewContext(source, h.scope);
        }
        const session = new h.scope.RealtimeSession('recovery-test');
        // Use the session's real independent hooks with this instrumented
        // player; the page-owned recording callback must remain untouched.
        h.player.onFarEndReference = session.audioPlayer.onFarEndReference;
        h.player.onPlaybackTimelineReset = session.audioPlayer.onPlaybackTimelineReset;
        const echo = session.echoReference;
        let resets = 0;
        const reset = echo.reset.bind(echo);
        echo.reset = () => { resets++; reset(); };

        h.context.state = 'running';
        h.enqueue(Float32Array.from({ length: 24_000 }, (_, index) => index / 24_000));
        h.advance(250);
        h.enqueue(0.2);
        assert.deepEqual(Array.from(echo._segments, segment => segment.start), [1250, 2250]);
        h.advance(400);
        h.context.state = state;
        h.advance(2000);
        h.enqueue(0.3);
        assert.equal(resets, 1);
        assert.equal(echo._segments.length, 0,
            'unheard old future audio must stop participating in echo suppression');
        assert.deepEqual(h.cancellations, [1650], 'the recorder still receives its audible cutoff');
        assert.equal(recorder._rightEntries.length, 1);
        assert.equal(recorder._rightEntries[0].data.length, 9600);

        h.advance(100);
        await h.recover();
        assert.deepEqual(Array.from(echo._segments, segment => segment.start), [3750, 4350, 5350]);
        assert.deepEqual(Array.from(echo._segments, segment => segment.samples.length), [9600, 16000, 16000]);
        assert.deepEqual(h.sources.map(source => source.stopCount), [1, 1, 0, 0, 0]);
        assert.ok(Math.abs(h.recordings[2].samples[0] - 0.4) < 1e-6,
            'recovery must not replay the already heard prefix');
        assert.deepEqual(Array.from(recorder._rightEntries, entry => entry.data.length),
            [9600, 14400, 24000, 24000]);
        assert.deepEqual(h.cancellations, [1650], 'rescheduling must not crop the WAV twice');

        h.player.endTurn();
        assert.equal(resets, 1, 'ordinary turn endings preserve the current echo schedule');
        assert.equal(echo._segments.length, 3);
        h.player.stopAll();
    }
});

test('manual cancellation during mid-playback recovery never restarts retained audio', async () => {
    const h = harness();
    h.context.state = 'running';
    h.enqueue(0.1); h.advance(250); h.advance(400);
    h.context.state = 'suspended'; h.advance(1000); h.enqueue(0.2);
    assert.equal(h.player._pendingChunks.length, 2);
    h.player.stopAll();
    h.advance(1000); await h.recover();
    assert.equal(h.sources.length, 1, 'a stale recovery must not recreate cancelled sources');
    assert.equal(h.recordings.length, 1, 'cancelled PCM must not return to the recording');
    assert.equal(h.player.hasBufferedAudio, false);
    assert.equal(h.player.turnActive, false);
    assert.equal(h.player.playing, false);
    assert.equal(h.errors.length, 0);
});

test('pause or stop without a later packet crops a frozen output at its audible prefix', () => {
    for (const state of ['suspended', 'interrupted']) {
        for (const action of ['stopAll', 'stop']) {
            const h = harness();
            const recorder = attachRecorder(h);
            h.context.state = 'running';
            h.enqueue(0.1); h.advance(250); h.enqueue(0.2);
            h.advance(400);
            h.context.state = state;
            h.advance(2000);
            h.player.endTurn();
            h.player[action]();

            assert.deepEqual(h.cancellations, [1650], `${state}/${action} used the frozen wall-clock cutoff`);
            assert.deepEqual(h.sources.map(source => source.stopCount), [1, 1]);
            assert.equal(h.resumes.length, 0, 'cancellation must not depend on another incoming packet');
            assert.equal(recorder._rightEntries.length, 1);
            assert.equal(recorder._rightEntries[0].data.length, 9600, 'retain only the 400 ms actually heard');
            assert.equal(recorder._rightNextOffset, 15600);

            h.player.stopAll();
            assert.equal(h.cancellations.at(-1), h.now(), 'a repeated stop must not reuse the old playback anchor');
            assert.equal(recorder.stop().rightSamples, 9600, 'WAV finalization must not restore the unheard tail');
        }
    }
});

test('stopping a suspended startup queue uses current wall time without a playback anchor', async () => {
    for (const recoveryPending of [false, true]) {
        const h = harness();
        h.enqueue(0.1);
        h.advance(recoveryPending ? 250 : 100);
        assert.equal(h.player.playing, false);
        assert.equal(h.sources.length, 0);
        h.player.stopAll();
        assert.deepEqual(h.cancellations, [h.now()]);
        assert.equal(h.player.hasBufferedAudio, false);
        if (recoveryPending) await h.recover();
        else h.advance(1000);
        assert.equal(h.sources.length, 0, 'a stopped startup queue must not be revived');
        assert.equal(h.recordings.length, 0);
    }
});

test('running output cancellation keeps the current wall-clock prefix', () => {
    const h = harness();
    const recorder = attachRecorder(h);
    h.context.state = 'running';
    h.enqueue(0.1); h.advance(250); h.enqueue(0.2);
    h.advance(400);
    h.player.stopAll();
    assert.deepEqual(h.cancellations, [h.now()]);
    assert.equal(recorder._rightEntries.length, 1);
    assert.equal(recorder._rightEntries[0].data.length, 9600);
    assert.equal(recorder.stop().rightSamples, 9600);
});
