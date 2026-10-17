# MiniCPM duplex throughput follow-up — 2026-10-17

The continuity and interruption repair was first preserved on `fix/minicpm-quality-20261010` at `2c022d4`. This follow-up removes redundant work while retaining the existing numerical operations, per-event transport acceptance, recording timestamps, cancellation checks and exact queue/storage accounting.

## Changes retained

- Audio attention now exports scores and probabilities only when both a trace prefix and a trace consumer exist. Ordinary encoder layers passed a prefix even without a consumer, which caused 24 layers to allocate, clear and write two unused BF16 attention matrices per packet. Context computation still uses the same MPSGraph. Offline and streaming K=50/500/1450 comparisons require identical context bytes with and without diagnostic outputs.
- Each normalized input request is encoded and decoded once. The recorder reuses the canonical value, and the queue reuses its exact encoded byte count at the original enqueue point. No input fields change between these points.
- Each official output event returns both its wire string and its existing JSON value. The recorder reuses that value after the send succeeds. `server_send_ts` is created once; `sentAt` is still taken after transport acceptance. Every event still checks the generation epoch separately. The full event batch must still succeed before finalize.

## Measurements

Release CPU benchmark, three warmups and fifteen measured iterations; medians:

| Operation | Previous | Shared result |
| --- | ---: | ---: |
| Canonical request, 16,000 numeric PCM samples | 95.282 ms | 49.401 ms |
| Canonical request, 160,000 numeric PCM samples | 981.714 ms | 491.191 ms |
| Output event, 96,000 PCM bytes before Base64 | 0.397 ms | 0.277 ms |

These measure serialization only. Encoding the next input can overlap processing of the previous output, so these savings do not add directly to end-to-end latency.

The same recorded microphone input was replayed through real model weights, with thirty original packets and six trailing silence packets:

| Public protocol measurement | Before | After |
| --- | ---: | ---: |
| Committed inputs | 36 | 36 |
| Voiced packets | 23 | 22 |
| Same-response arrival interval, median | 1046.20 ms | 1033.39 ms |
| Same-response arrival interval, maximum | 1103.77 ms | 1075.62 ms |
| Extra path wait, median | 50.56 ms | 31.98 ms |
| Voiced-step compute RTF | 1.0395 | 1.0554 |

Extra path wait is successive voiced arrival time within one response minus the current step's `wall_clock_ms`; it includes recording, finalize, serialization and transport. Both runs had zero protocol errors, only input27 interrupted, and no old-response voiced output followed that interruption. All 22 optimized voiced WAVs contain finite, nonzero PCM. Free generation changed the answer text and packet lengths, and resource contention was not controlled; these are not same-text model speed measurements. Audio encoder median was 135.18→130.56 ms, which cannot isolate the trace-only benefit from load variation.

Playback scheduling simulation still reports 13 gaps, 1235.6 ms total and 620.7 ms maximum. In this run input9 produced a 280 ms “好的，” prefix, then input10 arrived about 990 ms later. That first short prefix creates most of the maximum gap; a lower transport overhead does not provide enough generated audio to cover it. The prior run's maximum was 103.8 ms with different generated text. No claim is made that this follow-up eliminates audible gaps or improves every playback metric. Simulation uses a clock and recorded sample lengths, not an output device.

## Validation

- 78 targeted Swift tests passed, including two BF16 trace/context comparisons and four wire encoding tests (one is the opt-in CPU benchmark).
- Six real-weight audio golden tests passed at the existing tolerances: offline encoding/projector, streaming caches, actual PCM frontend, extra context and snapshot/rollback; the optional operator trace test was skipped because no trace path was supplied.
- 129 frontend tests passed. Browser resources did not change in this follow-up.
- Public real-model cancellation, buffered-tail interruption and zero-text-limit forced-listen probes are repeated against the final binary; their results and current service identity are in the local handoff evidence.
- LLM, semantic TTS and Token2Wav numerical sources remain at the previously checked versions. The audio caller's trace export selection is the only changed numerical-path source; its graph arithmetic is unchanged and was checked again above.

CPU benchmark command (after a release test build):

```sh
MINICPM_WIRE_BENCHMARK=1 swift test -c release --disable-sandbox -j 2 --no-parallel \
  --filter MiniCPMWireEncodingTests
```

Model-loading builds/tests/benchmarks must run sequentially, with the model service stopped before building. Preserve other running workloads.

## Candidates considered and next experiments

| Candidate | Decision and required evidence |
| --- | --- |
| Reuse request/event serialization | Retained; measured above, exact byte/field and failure checks pass. |
| Skip unused attention diagnostic outputs | Retained; BF16 context bytes and real-weight golden checks pass. |
| Construct canonical request JSON directly | Next CPU candidate: avoid decoding a known typed request back into a value; must preserve nil/null, nonfinite validation, unknown-key handling and exact queue bytes. |
| Add TTS audio-head graph to existing command buffer | Larger candidate: preserve the existing `[768]` input and separate BF16 matmul graph, encode after final RMS, wait once. Test fixed teacher inputs, all 78 semantic steps, weight updates, cache rollover, fallback and old-result lifetimes. |
| Reuse private TTS scratch across tokens | Possible allocation/bridge reduction under the existing execution lock. Never reuse returned hidden/cache buffers; snapshots and rollback retain them. Measure fixed-token bridge and complete decode times. |
| Cache Hamming crossfade windows | Small candidate: cache immutable windows by overlap with exactly the same Double→Float calculation and arithmetic order. Measure before adding state. |
| Reuse duplicate frontend STFT rows within one accept | Preserve reflected edges, sliding context and final-chunk behavior. CPU benchmark first; likely smaller than model costs. |
| Batch recording or defer recording until an event group completes | Rejected for this pass: changes send-failure prefixes, cancellation boundaries, concurrent event sequence and immediate snapshot visibility. |
| Increase queues/buffers, drop microphone packets, ACK early | Excluded: these do not improve generation throughput and can obscure the failure. |

A separate quality issue remains in the inherited configuration: `MiniCPMFlowConfiguration.odeSteps` defaults to 8 while the source comment and exported reference specify 10. The loader does not override it; the current flow golden fixture explicitly uses 2 steps and does not establish 8-versus-10 perceptual equivalence. This follow-up does not change the step count or claim full waveform parity. Evaluate fixed semantic tokens and noise at 8 and 10 steps, including listening and latency, before changing this quality/performance tradeoff.

Local raw evidence is under `test-output/swift-mps-audit-20261017/throughput-followup/`: `swift-tests.log`, `audio-golden.log`, `cpu-benchmark.json`, `comparison.json`, `optimized-replay/`, `optimized-player.json`, `pcm-check.json`, and the protocol probe summaries. Session audio stays local and is not part of the Git commit.
