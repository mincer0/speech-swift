import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import vm from 'node:vm';
import test from 'node:test';

const recorderSource = readFileSync(new URL(
    '../../Sources/MiniCPMDemoBackend/Resources/Web/duplex/lib/session-video-recorder.js', import.meta.url), 'utf8')
    .replace(/^import .*;$/gm, '').replace('export class', 'class')
    + '\nthis.SessionVideoRecorder = SessionVideoRecorder;';
const workletSource = readFileSync(new URL(
    '../../Sources/MiniCPMDemoBackend/Resources/Web/duplex/lib/stereo-recorder-processor.js', import.meta.url), 'utf8');

async function flushPromises() {
    for (let index = 0; index < 12; index++) await Promise.resolve();
}

function recordingHarness({ moduleError = null, constructorError = null, startError = null } = {}) {
    let wall = 0;
    let nextTimer = 0;
    const timers = new Map();
    const order = [];
    const nodes = [];
    const sources = [];
    const mediaRecorders = [];
    const streamSources = [];
    const contextCalls = [];
    const videoTrack = { kind: 'video', stopCount: 0, stop() { this.stopCount++; } };
    const audioTrack = { kind: 'audio' };
    class MockMediaStream {
        constructor(tracks = []) { this.tracks = tracks; }
        getVideoTracks() { return this.tracks.filter(track => track.kind === 'video'); }
        getAudioTracks() { return this.tracks.filter(track => track.kind === 'audio'); }
    }
    const ctx = {
        state: 'running', currentTime: 0,
        audioWorklet: {
            async addModule() {
                order.push('worklet-module');
                if (moduleError) throw moduleError;
            },
        },
        createBuffer(_channels, length, rate) {
            const data = new Float32Array(length);
            return { duration: length / rate, getChannelData: () => data };
        },
        createBufferSource() {
            const source = {
                connect() {}, disconnect() { this.disconnected = true; },
                start(at, offset = 0) {
                    this.startedAt = at; this.offset = offset;
                    order.push('buffer-start');
                },
                stop() { this.stopped = true; },
            };
            sources.push(source);
            return source;
        },
        createMediaStreamSource() {
            const source = {
                connect() {}, disconnect() { this.disconnected = true; },
            };
            streamSources.push(source);
            return source;
        },
        createMediaStreamDestination() {
            return { stream: new MockMediaStream([audioTrack]) };
        },
        async suspend() { contextCalls.push('suspend'); this.state = 'suspended'; },
        async resume() { contextCalls.push('resume'); this.state = 'running'; },
        async close() { contextCalls.push('close'); this.state = 'closed'; },
    };
    class MockAudioWorkletNode {
        constructor() {
            this.port = {
                onmessage: null, messages: [],
                postMessage(message) { this.messages.push(message); },
            };
            nodes.push(this);
        }
        connect() { order.push('audio-connected'); }
        disconnect() { this.disconnected = true; }
    }
    class MockMediaRecorder {
        constructor(stream, options) {
            if (constructorError) throw constructorError;
            this.stream = stream;
            this.mimeType = options.mimeType;
            this.state = 'inactive';
            this.stopCount = 0;
            mediaRecorders.push(this);
        }
        start() {
            order.push('media-start-request');
            if (startError) throw startError;
            this.state = 'recording';
        }
        pause() { this.state = 'paused'; order.push('media-pause'); }
        resume() { this.state = 'recording'; order.push('media-resume'); }
        requestData() { this.ondataavailable?.({ data: new Blob(['encoded']) }); }
        stop() {
            this.stopCount++;
            this.state = 'inactive';
            queueMicrotask(() => this.onstop?.());
        }
        static isTypeSupported() { return true; }
    }
    const videoEl = {
        captureStream() {
            order.push('video-acquired');
            return new MockMediaStream([videoTrack]);
        },
    };
    const realm = {
        AudioContext: function () { return ctx; },
        AudioWorkletNode: MockAudioWorkletNode,
        MediaStream: MockMediaStream, MediaRecorder: MockMediaRecorder,
        Blob, Float32Array, navigator: { userAgent: 'Chrome' },
        performance: { now: () => wall },
        console: { log() {}, warn() {} },
        resampleAudio: samples => samples,
        setTimeout(callback) { const id = ++nextTimer; timers.set(id, callback); return id; },
        clearTimeout(id) { timers.delete(id); },
        cancelAnimationFrame() {},
    };
    vm.runInNewContext(recorderSource, realm);
    return {
        recorder: new realm.SessionVideoRecorder(videoEl, 4, 4),
        ctx, order, nodes, sources, mediaRecorders, streamSources, contextCalls, videoTrack, timers,
        advance(milliseconds) { wall += milliseconds; ctx.currentTime += milliseconds / 1000; },
        acknowledgeAudioReady() {
            order.push('audio-ready');
            nodes[0].port.onmessage?.({ data: { type: 'ready' } });
        },
        acknowledgeMediaStart() {
            order.push('media-started');
            mediaRecorders[0].onstart?.();
        },
        expireTimers() {
            const callbacks = [...timers.values()];
            timers.clear();
            for (const callback of callbacks) callback();
        },
    };
}

async function startFileRecording(harness) {
    const pending = harness.recorder.start({}, { leftAudioBuffer: new Float32Array(40) });
    await flushPromises();
    harness.acknowledgeAudioReady();
    await flushPromises();
    harness.acknowledgeMediaStart();
    await pending;
}

function processorHarness() {
    const messages = [];
    let Processor;
    vm.runInNewContext(workletSource, {
        Float32Array,
        AudioWorkletProcessor: class {
            constructor() {
                this.port = { postMessage: message => messages.push(structuredClone(message)) };
            }
        },
        registerProcessor(_name, implementation) { Processor = implementation; },
    });
    const processor = new Processor();
    return {
        messages,
        send(message) { processor.port.onmessage({ data: message }); },
        render(length = 2, inputs = []) {
            const output = [new Float32Array(length), new Float32Array(length)];
            assert.equal(processor.process(inputs, [output]), true);
            return output.map(channel => Array.from(channel));
        },
    };
}

test('video recorder waits for audio rendering before video acquisition and for onstart before file audio', async () => {
    const h = recordingHarness();
    const pending = h.recorder.start({}, { leftAudioBuffer: new Float32Array(40) });
    await flushPromises();
    assert.equal(h.order.includes('audio-connected'), true);
    assert.equal(h.order.includes('video-acquired'), false);
    assert.equal(h.mediaRecorders.length, 0);
    assert.equal(h.sources.length, 0);
    h.nodes[0].port.onmessage({ data: { type: 'unrelated' } });
    await flushPromises();
    assert.equal(h.order.includes('video-acquired'), false);

    h.advance(3000);
    h.acknowledgeAudioReady();
    await flushPromises();
    assert.equal(h.mediaRecorders.length, 1);
    assert.equal(h.order.indexOf('audio-ready') < h.order.indexOf('video-acquired'), true);
    assert.equal(h.order.includes('media-start-request'), true);
    assert.equal(h.sources.length, 0, 'file source must wait for the encoder start event');
    assert.equal(h.recorder.recording, false);
    h.advance(2000);
    h.acknowledgeMediaStart();
    await pending;
    assert.equal(h.recorder.recording, true);
    assert.equal(h.sources.length, 1);
    assert.equal(h.sources[0].startedAt, 5);
    assert.equal(h.order.indexOf('media-started') < h.order.indexOf('buffer-start'), true);
    assert.equal(h.timers.size, 0);
    h.advance(1000);
    const result = await h.recorder.stop();
    assert.equal(result.durationSec, 1, 'preparation waits are outside the recorded duration');
    assert.equal(h.videoTrack.stopCount, 1);
});

test('audio render readiness timeout cleans up its graph without acquiring video or starting file audio', async () => {
    const h = recordingHarness();
    const pending = h.recorder.start({}, { leftAudioStream: {} });
    const rejected = assert.rejects(pending, /Recording audio did not start rendering/);
    await flushPromises();
    assert.equal(h.streamSources.length, 1);
    h.expireTimers();
    await rejected;
    assert.equal(h.ctx.state, 'closed');
    assert.deepEqual(h.contextCalls, ['close']);
    assert.equal(h.streamSources[0].disconnected, true);
    assert.equal(h.nodes[0].disconnected, true);
    assert.equal(h.nodes[0].port.onmessage, null);
    assert.equal(h.order.includes('video-acquired'), false);
    assert.equal(h.mediaRecorders.length, 0);
    assert.equal(h.sources.length, 0);
    assert.equal(h.recorder.recording, false);
});

test('worklet module preparation failure closes its context before acquiring video', async () => {
    const failure = new Error('Worklet module failed');
    const h = recordingHarness({ moduleError: failure });
    await assert.rejects(h.recorder.start(), error => error === failure);
    assert.equal(h.ctx.state, 'closed');
    assert.deepEqual(h.contextCalls, ['close']);
    assert.equal(h.nodes.length, 0);
    assert.equal(h.order.includes('video-acquired'), false);
    assert.equal(h.mediaRecorders.length, 0);
    assert.equal(h.recorder.recording, false);
    assert.equal(h.timers.size, 0);
});

for (const failureStage of ['constructor', 'start']) {
    test(`media encoder ${failureStage} failure releases acquired video and prepared audio`, async () => {
        const failure = new Error(`Encoder ${failureStage} failed`);
        const h = recordingHarness({ [`${failureStage}Error`]: failure });
        const pending = h.recorder.start({}, { leftAudioBuffer: new Float32Array(40) });
        const rejected = assert.rejects(pending, error => error === failure);
        await flushPromises();
        h.acknowledgeAudioReady();
        await rejected;
        assert.equal(h.order.includes('video-acquired'), true);
        assert.equal(h.videoTrack.stopCount, 1);
        assert.equal(h.ctx.state, 'closed');
        assert.deepEqual(h.contextCalls, ['close']);
        assert.equal(h.nodes[0].disconnected, true);
        assert.equal(h.nodes[0].port.onmessage, null);
        assert.equal(h.sources.length, 0);
        assert.equal(h.recorder.recording, false);
        assert.equal(h.timers.size, 0);
        if (failureStage === 'start') {
            assert.equal(h.mediaRecorders[0].onstart, null);
            assert.equal(h.mediaRecorders[0].onerror, null);
            assert.equal(h.mediaRecorders[0].ondataavailable, null);
        }
    });
}

test('media encoder startup error stops the partially started recorder and releases its tracks', async () => {
    const h = recordingHarness();
    const failure = new Error('Encoder startup failed');
    const pending = h.recorder.start({}, { leftAudioStream: {} });
    const rejected = assert.rejects(pending, error => error === failure);
    await flushPromises();
    h.acknowledgeAudioReady();
    await flushPromises();
    const media = h.mediaRecorders[0];
    media.onerror({ error: failure });
    await rejected;
    assert.equal(media.state, 'inactive');
    assert.equal(media.stopCount, 1);
    assert.equal(media.onstart, null);
    assert.equal(media.onerror, null);
    assert.equal(media.ondataavailable, null);
    assert.equal(h.streamSources[0].disconnected, true);
    assert.equal(h.videoTrack.stopCount, 1);
    assert.equal(h.ctx.state, 'closed');
    assert.equal(h.nodes[0].disconnected, true);
    assert.equal(h.recorder.recording, false);
    assert.equal(h.timers.size, 0);
});

test('media encoder startup timeout cleans resources without beginning buffered file audio', async () => {
    const h = recordingHarness();
    const pending = h.recorder.start({}, { leftAudioBuffer: new Float32Array(40) });
    const rejected = assert.rejects(pending, /Media recorder did not start/);
    await flushPromises();
    h.acknowledgeAudioReady();
    await flushPromises();
    const media = h.mediaRecorders[0];
    assert.equal(media.state, 'recording');
    assert.equal(h.sources.length, 0);
    h.expireTimers();
    await rejected;
    assert.equal(media.state, 'inactive');
    assert.equal(media.stopCount, 1);
    assert.equal(media.onstart, null);
    assert.equal(media.onerror, null);
    assert.equal(media.ondataavailable, null);
    assert.equal(h.videoTrack.stopCount, 1);
    assert.equal(h.ctx.state, 'closed');
    assert.equal(h.nodes[0].disconnected, true);
    assert.equal(h.sources.length, 0);
    assert.equal(h.recorder.recording, false);
    assert.equal(h.timers.size, 0);
});

test('manual video recording pause keeps the audio clock running and restores consumed and future sources', async () => {
    const h = recordingHarness();
    await startFileRecording(h);
    h.recorder.pushRight(new Float32Array(16), 4, 0);
    h.recorder.pushRight(new Float32Array(8), 4, 8000);
    const [originalLeft, activeRight, futureRight] = h.sources;
    h.advance(1500);
    h.recorder.pause();
    assert.equal(h.ctx.state, 'running');
    assert.deepEqual(h.contextCalls, [], 'manual pause must not suspend the audio track clock');
    assert.equal(originalLeft.stopped, true);
    assert.equal(activeRight.stopped, true);
    assert.equal(futureRight.stopped, true);
    assert.equal(h.mediaRecorders[0].state, 'paused');
    h.advance(10_000);
    h.recorder.resume();
    const [resumedActiveRight, resumedFutureRight, resumedLeft] = h.sources.slice(3);
    assert.equal(resumedActiveRight.startedAt, 11.5);
    assert.equal(resumedActiveRight.offset, 1.5);
    assert.equal(resumedFutureRight.startedAt, 18);
    assert.equal(resumedFutureRight.offset, 0);
    assert.equal(resumedLeft.startedAt, 11.5);
    assert.equal(resumedLeft.offset, 1.5);
    assert.equal(h.mediaRecorders[0].state, 'recording');
    assert.deepEqual(h.contextCalls, []);
    const commands = h.nodes[0].port.messages.map(message => message.command);
    assert.deepEqual(commands.slice(-2), ['pause', 'resume']);
    await h.recorder.stop();
});

test('stereo worklet emits readiness only on its first rendered block', () => {
    const h = processorHarness();
    assert.deepEqual(h.messages, [], 'construction alone does not prove that the graph renders');
    assert.deepEqual(h.render(), [[0, 0], [0, 0]]);
    assert.deepEqual(h.messages, [{ type: 'ready' }]);
    h.render();
    h.send({ command: 'pause' });
    h.render();
    h.send({ command: 'resume' });
    h.render();
    assert.deepEqual(h.messages, [{ type: 'ready' }]);
});

test('stereo worklet pause outputs silence while retaining partial buffers and queued legacy audio', () => {
    const h = processorHarness();
    h.send({ channel: 'left', audio: [1, 2, 3, 4, 5, 6] });
    h.send({ channel: 'right', audio: [11, 12, 13, 14, 15, 16] });
    assert.deepEqual(h.render(), [[1, 2], [11, 12]]);
    h.send({ command: 'pause' });
    h.send({ channel: 'left', audio: [7, 8] });
    h.send({ channel: 'right', audio: [17, 18] });
    assert.deepEqual(h.render(), [[0, 0], [0, 0]]);
    assert.deepEqual(h.render(), [[0, 0], [0, 0]]);
    h.send({ command: 'resume' });
    assert.deepEqual(h.render(), [[3, 4], [13, 14]]);
    assert.deepEqual(h.render(), [[5, 6], [15, 16]]);
    assert.deepEqual(h.render(), [[7, 8], [17, 18]]);
    assert.deepEqual(h.render(), [[0, 0], [0, 0]]);
    assert.deepEqual(h.messages, [{ type: 'ready' }]);
});
