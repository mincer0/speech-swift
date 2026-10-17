import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import vm from 'node:vm';
import test from 'node:test';

const source = readFileSync(new URL('../../Sources/MiniCPMDemoBackend/Resources/Web/duplex/lib/session-video-recorder.js', import.meta.url), 'utf8')
    .replace(/^import .*;$/gm, '').replace('export class', 'class') + '\nthis.SessionVideoRecorder = SessionVideoRecorder;';
function harness() {
    let wall = 0;
    const sources = [];
    const calls = [];
    const ctx = { state: 'running', currentTime: 0,
        createBuffer(_channels, length, rate) { return { duration: length / rate, getChannelData: () => new Float32Array(length) }; },
        createBufferSource() {
            const src = { connect() {}, disconnect() {},
                start(at, offset = 0) { this.at = at; this.offset = offset; },
                stop() { this.stopped = true; }, };
            sources.push(src); return src;
        },
        suspend() { calls.push('suspend'); return Promise.resolve(); },
        resume() { calls.push('resume'); return Promise.resolve(); },
        close() { return Promise.resolve(); },
    };
    const context = { console: { log() {} }, Blob, performance: { now: () => wall },
        resampleAudio: samples => samples, cancelAnimationFrame() {} };
    vm.runInNewContext(source, context);
    const recorder = new context.SessionVideoRecorder({}, 4, 4);
    recorder._recording = true; recorder._recCtx = ctx;
    recorder._stereoNode = { disconnect() {}, port: { postMessage() {} } };
    recorder._leftBuffer = { duration: 10 };
    recorder._leftBufferOffset = 0;
    recorder._leftBufferStartedAt = null;
    recorder._leftBufSource = null;
    recorder._rightSources = []; recorder._rightNextTime = 0;
    recorder._resumeLeftBuffer();
    return { recorder, ctx, sources, calls,
        advance(ms, audio = true) { wall += ms; if (audio) ctx.currentTime += ms / 1000; } };
}

test('video input hold freezes file audio at its consumed offset while assistant recording continues', () => {
    const h = harness(); const r = h.recorder;
    h.advance(1250); r.pauseInputCapture();
    assert.equal(h.sources[0].stopped, true);
    assert.equal(r._leftBufferOffset, 1.25);
    assert.deepEqual(h.calls, []);
    h.advance(5000);
    r.pushRight(new Float32Array(4), 4, 6250);
    assert.equal(h.sources[1].at, 6.25);
    assert.equal(h.sources[1].stopped, undefined);
    r.resumeInputCapture();
    assert.equal(h.sources[2].offset, 1.25);
    assert.equal(h.sources[2].at, 6.25);
    h.sources[0].onended();
    assert.equal(r._leftBufSource, h.sources[2], 'late stop callback must not erase replacement');
    h.advance(750); r.pauseInputCapture(); r.resumeInputCapture();
    assert.equal(h.sources[3].offset, 2);
});

test('manual pause and input hold do not advance the file or resume it before the user', async () => {
    const h = harness(); const r = h.recorder;
    h.advance(1000); r.pauseInputCapture(); r.pause();
    h.advance(3000); r.resumeInputCapture();
    assert.equal(h.sources.length, 1);
    r.resume(); await Promise.resolve();
    assert.equal(h.sources[1].offset, 1);
    assert.equal(h.sources[1].at, 4);
    assert.deepEqual(h.calls, [], 'audio track clock must stay continuous while encoding is paused');
    h.sources[1].onended(); r.pauseInputCapture(); r.resumeInputCapture();
    assert.equal(h.sources.length, 2, 'EOF must not restart the complete file');
});

test('video stop duration excludes manual pause and asynchronous encoder flush time', async () => {
    const h = harness(); const r = h.recorder;
    h.advance(1000); r.pause(); h.advance(3000); r.resume();
    await Promise.resolve(); h.advance(1000);
    r._mediaRecorder = { mimeType: 'video/webm', state: 'recording',
        requestData() { assert.equal(h.sources[0].stopped, true); },
        stop() { h.advance(500); queueMicrotask(() => this.onstop()); } };
    const result = await r.stop();
    assert.equal(result.durationSec, 2);
});

test('manual pause preserves partial and future assistant sources on the continuous recording clock', () => {
    const h = harness(); const r = h.recorder;
    r.pushRight(new Float32Array(12), 4, 0);
    r.pushRight(new Float32Array(4), 4, 0);
    h.advance(1000); r.pause();
    assert.equal(h.sources[1].stopped, true);
    assert.equal(h.sources[2].stopped, true);
    h.advance(5000); r.resume();
    assert.equal(h.sources[3].at, 6);
    assert.equal(h.sources[3].offset, 1);
    assert.equal(h.sources[4].at, 8);
    assert.equal(h.sources[4].offset, 0);
    assert.equal(r._rightNextTime, 9);
    assert.equal(h.sources[5].offset, 1, 'left input resumes at its consumed position');
    r.pause(); r.cancelRight(); h.advance(1000); r.resume();
    assert.equal(r._rightSources.length, 0, 'cancel while paused clears preserved assistant audio');
});
