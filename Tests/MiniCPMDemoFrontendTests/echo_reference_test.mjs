import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import vm from 'node:vm';

const source = readFileSync(new URL(
    '../../Sources/MiniCPMDemoBackend/Resources/Web/duplex/lib/echo-reference.js', import.meta.url,
), 'utf8').replace(/^export /gm, '') + '\nthis.FarEndEchoReference = FarEndEchoReference;';
const scope = { Float32Array, Uint8Array, performance: { now: () => 10_000 } };
vm.runInNewContext(source, scope);
const FarEndEchoReference = scope.FarEndEchoReference;
const rate = 16_000;
const wave = (hz = 211, count = rate) => Float32Array.from({ length: count }, (_, i) =>
    0.1 * Math.sin(2 * Math.PI * hz * i / rate) + 0.04 * Math.sin(2 * Math.PI * 673 * i / rate));
const energy = samples => samples.reduce((sum, value) => sum + value * value, 0) / samples.length;

test('future scheduled playback cannot erase identical current microphone speech', () => {
    const reference = new FarEndEchoReference();
    const mic = wave();
    reference.pushPlayback(mic, rate, 5_000);
    const result = reference.suppressInput(mic, rate, 1_000);
    assert.equal(result.suppressed, false);
    assert.equal(result.samples, mic);
    assert.equal(result.diagnostics.referenceAligned, false);
    assert.equal(result.diagnostics.suppressionSuggested, false);
});

test('mic window uses its matching played interval even with future queued audio', () => {
    const reference = new FarEndEchoReference();
    const played = wave();
    reference.pushPlayback(played, rate, 0);
    reference.pushPlayback(wave(367, rate * 5), rate, 1_000);
    const mic = Float32Array.from(played, value => value * 0.4);
    const result = reference.suppressInput(mic, rate, 1_000);
    assert.equal(result.suppressed, true);
    assert.ok(energy(result.samples) < 1e-14);
    assert.ok(Math.abs(result.gain - 0.4) < 1e-6);
});

test('adjacent playback chunks cover one complete microphone window', () => {
    const reference = new FarEndEchoReference();
    const played = wave();
    reference.pushPlayback(played.slice(0, rate / 2), rate, 1_000);
    reference.pushPlayback(played.slice(rate / 2), rate, 1_500);
    const result = reference.suppressInput(played, rate, 2_000);
    assert.equal(result.suppressed, true);
    assert.ok(energy(result.samples) < 1e-14);
});

test('partial or stale reference leaves the entire microphone chunk intact', () => {
    const mic = wave();
    for (const start of [500, -5_000]) {
        const reference = new FarEndEchoReference();
        reference.pushPlayback(mic, rate, start);
        const result = reference.suppressInput(mic, rate, 1_000);
        assert.equal(result.samples, mic);
        assert.equal(result.suppressed, false);
    }
});

test('aligned delayed echo subtraction preserves independent near-end speech', () => {
    const reference = new FarEndEchoReference();
    const played = wave();
    // Integer-cycle orthogonal near-end tone. The echo projection must not
    // remove it even though the echo dominates the microphone signal.
    const near = Float32Array.from({ length: rate }, (_, i) => 0.02 * Math.sin(2 * Math.PI * 479 * i / rate));
    const mic = Float32Array.from(played, (value, i) => 0.7 * value + near[i]);
    reference.pushPlayback(played, rate, 0);
    const result = reference.suppressInput(mic, rate, 1_040);
    assert.equal(result.suppressed, true);
    assert.equal(result.diagnostics.playbackAgeMs, 40);
    const error = Float32Array.from(result.samples, (value, i) => value - near[i]);
    assert.ok(energy(error) < 1e-14);
    assert.ok(energy(result.samples) > energy(near) * 0.999);
});

test('unrelated speech and unsupported input rate remain unchanged', () => {
    const reference = new FarEndEchoReference();
    reference.pushPlayback(wave(), rate, 0);
    const near = Float32Array.from({ length: rate }, (_, i) => 0.1 * Math.sin(2 * Math.PI * 479 * i / rate));
    assert.equal(reference.suppressInput(near, rate, 1_000).samples, near);
    const at48k = new Float32Array(rate * 3).fill(0.1);
    const result = reference.suppressInput(at48k, 48_000, 1_000);
    assert.equal(result.samples, at48k);
    assert.equal(result.suppressed, false);
});

test('playback resampling and reset respect the same timestamp contract', () => {
    const reference = new FarEndEchoReference();
    const played = new Float32Array(24_000).fill(0.1);
    reference.pushPlayback(played, 24_000, 0);
    const mic = new Float32Array(rate).fill(0.04);
    assert.equal(reference.suppressInput(mic, rate, 1_000).suppressed, true);
    reference.reset();
    const result = reference.suppressInput(mic, rate, 1_000);
    assert.equal(result.samples, mic);
    assert.equal(result.suppressed, false);
    assert.equal(result.diagnostics.referenceAligned, false);
});
