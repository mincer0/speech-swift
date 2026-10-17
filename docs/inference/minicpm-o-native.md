# MiniCPM-o Native Streaming Inference

This page describes the current native server and direct Swift APIs.  Runtime
model execution uses MLX Swift, MPSGraph, Accelerate, and (for CAM++/optional
VAD) CoreML;
Python, PyTorch, and ONNX Runtime are not production dependencies.  Conversion
scripts and golden-oracle exporters are offline tools only.

## Interruption qualification and recording throughput

Ordinary audio attention does not export diagnostic score/probability tensors without a trace consumer. Input recording and queue accounting share one canonical serialization; sent output and its recording share one envelope while retaining per-event send acceptance and timestamps. Measurements and remaining performance candidates are documented in [the throughput follow-up](../benchmarks/minicpm-duplex-throughput-2026-10-17.md).

Duplex interruption uses sustained 20 ms microphone evidence across input packets, separately from the sensitive transport activity flag. Buffered playback remains eligible after the model has ended its turn, using the client playback_active flag without adding a duplicate turn boundary. A candidate must still be voiced when it qualifies; vocabulary rank alone never authorizes clearing an open answer. Qualified input closes the old spoken turn without removing the current microphone unit or accepted history, then holds listen until the input ends in 480 ms of quiet. The release packet still listens. The compatibility reason remains `model_barge_in`; it does not claim model confidence or semantic speech classification. One-second transport packets still bound live response latency.

Output-context recovery clears obsolete echo-reference schedules before requeuing unheard audio. The recording writer opens only on the first append so snapshots do not need write permission. Writes remain ordered and immediately readable, with a hardware flush at normal close instead of each event. Capacity checks still inspect actual storage; optimized traversal must preserve hidden-file and symlink accounting. An unclean process/device shutdown may lose recent unflushed log data. No audio queue limits or numerical gates are relaxed.

## Build and launch

Build the server and helpers from this package:

```bash
swift build -c release -j 1 --scratch-path .build-minicpm-release \
  --product minicpm-mlx-server --disable-sandbox
./scripts/build_mlx_metallib.sh release --scratch-path .build-minicpm-release
```

The dedicated scratch directory prevents unrelated package products from
leaving incompatible transitive Swift/C modules in the server's release module
search path. `-j 1` keeps peak build memory bounded. The metallib step is
required because command-line SwiftPM does not compile MLX Metal shaders.
Keep this server scratch directory separate from `swift test` output. The
pinned EventSource package tests `canImport(AsyncHTTPClient)` for an optional
adapter; a cached module from HummingbirdTesting can enable that source even
though a server-only build has neither its compile dependencies nor its link
objects. A dedicated server scratch directory avoids the resulting missing
C-module errors without changing dependency pins.

The shader builder targets macOS 15 explicitly with
`-mmacosx-version-min=15.0` and retains `-fno-fast-math`. Newer Xcode defaults
can reject the pinned kernels without that deployment target. Its cache key
includes compiler flags, Metal compiler and SDK versions, and kernel sources
and headers. It copies the resulting library into test bundles for the
selected build configuration only. The verified release build compiled all
33 selected Metal sources into a 107,844,874-byte library and passed the
real-weight audio and three-chunk semantic TTS gates; an older library from a
different configuration is not a substitute for this build step.

Production startup is fail-closed.  The model root must contain the five
canonical sibling bundles and the Token2Wav bundle must contain the completed
`speech_tokenizer.safetensors` and `MiniCPM-CamPlusPlus.mlmodelc` sidecars.
The server discovers those sidecars automatically before loading
`MiniCPMNativeModels`; it does not discover or execute the original ONNX
graphs.

```bash
.build-minicpm-release/release/minicpm-mlx-server \
  --model-dir /path/to/models \
  --voice-reference /path/to/reference.wav \
  --host 127.0.0.1 --port 7860 \
  --max-concurrent 1
```

`--fake` is reserved for protocol-only transport smoke tests.  It does not
exercise model inference and must not be used as a production fallback.

## Server options and environment

Three optional semantic-TTS execution paths reduce waits between MLX and
MPSGraph while retaining the established operator graphs, BF16 projection
boundaries and numerical gates:

| Environment switch | Scope |
| --- | --- |
| `MINICPM_TTS_ENCODED_MLP=1` | Encode gate/up, SiLU/product and down together. |
| `MINICPM_TTS_ENCODED_BLOCK=1` | Also encode the attention output projection, residual additions and standard MLX RMSNorm. |
| `MINICPM_TTS_ENCODED_DECODE=1` | Encode a supported single-token BF16 decode across the complete 20-layer speech decoder and final RMSNorm, with one entry MLX evaluation and one final CPU wait. |

All three switches default to disabled in source. Unsupported shapes, dtypes,
projection implementations, normalization settings or device/kernel support
use the established path. Full encoded decode leaves prefill on that path.
The encoded routes retain MLX storage until submitted work completes and drain
that work before falling back after an execution failure. Custom Metal encoders and completion
checks use the current `MPSCommandBuffer.rootCommandBuffer`, including after
MPSGraph rolls over to another command buffer.

The full encoded decode supports a logical KV length, including the appended
token, of at most 4,096. Its attention dispatch follows the pinned MLX
implementation: on GPU architectures whose names end in `d` or `s`, a logical
key length of at least 1,024 uses the original two-pass attention kernels
(128 or 64 blocks respectively). Other supported architectures keep the
single-pass dispatch. Existing cache owners remain read-only; cache replacement
occurs only after every layer succeeds.

A prior 34-input free-generation replay produced 25.92 seconds of valid
speech with 23.7151 seconds of recorded voiced-packet computation (RTF 0.9149),
34 committed inputs and no reported protocol errors. It includes the long-cache
extension but reached only 682 TTS cache positions. Free-generation runs produce
different text, so this is not a same-text speed comparison. Aggregate RTF alone
does not establish uninterrupted playback on all devices; microphone/AEC and
perceived continuity still require acceptance. See the current local handoff
for the longer-session probe, evidence and limits. A later recorded long answer
had voiced-packet RTF 1.068 under its observed load; the earlier 0.9149 must not
be treated as a throughput guarantee for later sessions.

The executable defaults are:

| Option | Default | Description |
| --- | --- | --- |
| `--host`, `--port` | `127.0.0.1`, `7860` | HTTP/WebSocket bind |
| `--model-dir` | required | Root of the five native model bundles |
| `--audio-bundle` | inferred | Explicit audio bundle override |
| `--token2wav-bundle` | inferred | Explicit Flow/HiFT + prompt-sidecar override |
| `--voice-reference` | required | PCM16 WAV or little-endian raw PCM16; raw input defaults to 16 kHz |
| `--max-concurrent` | `1` | Active MLX sessions |
| `--max-queue-size` | `1000` | Maximum queued sessions; `0` rejects queueing |
| `--queue-timeout-ms` | `120000` | FIFO timeout; `0` disables it |
| `--data-dir` | `./data` | Persistent gateway data |
| `--preset-root`, `--upstream-root` | unset | Optional Demo static/preset roots |
| `--vad-model-dir` | standard Silero cache | Explicit offline CoreML VAD bundle/cache |
| `--disable-vad` | off | Half-duplex requires explicit `utterance_end` |

Prompt sidecar discovery accepts these process environment overrides:

```text
MINICPM_SPEECH_TOKENIZER_WEIGHTS=/absolute/path/speech_tokenizer.safetensors
MINICPM_CAMPPLUS_COREML=/absolute/path/MiniCPM-CamPlusPlus.mlmodelc
```

The repository's `start_minicpm_native.sh` launcher additionally reads
`MINICPM_NATIVE_BINARY`, `MINICPM_NATIVE_MODEL_ROOT`, `MINICPM_NATIVE_VOICE_REFERENCE`,
`MINICPM_NATIVE_UPSTREAM_ROOT`, `MINICPM_NATIVE_DATA_DIR`,
`MINICPM_NATIVE_HOST`, `MINICPM_NATIVE_PORT`,
`MINICPM_NATIVE_MAX_CONCURRENT`, `MINICPM_NATIVE_ENABLE_VAD` (default `0`),
and `MINICPM_NATIVE_VAD_MODEL_DIR`.  Its `--check` mode verifies the release
binary, all five model directories, voice file, writable data directory, and
the optional pinned Demo checkout (`ba7fa9cc6ad63c894f1bd5e5afac28466953519d`) without
starting the server.  It never builds, downloads, or substitutes a fake model.

The source model pin is a separate identity:
`073dbbc8c5bc0af2d789e1ce12e7c17a6be746e1` is the MiniCPM-o model repository
revision; `ba7fa9cc6ad63c894f1bd5e5afac28466953519d` is the Demo implementation
commit.

## Realtime input and controls

The native endpoint exposes `input.committed` as the completion
acknowledgement for each accepted input, including inputs that produce no
model delta. The bundled browser uses that acknowledgement for the native
`swift_mlx` backend, rather than treating a text delta as input completion.
It keeps two inputs in flight by default and retains unsent speech, silence,
and continuation packets in a single FIFO. Rejected inputs keep their
original sequence when retried.

The default unsent FIFO limits are 16 packets and 8 MiB of accounted JSON
storage. Browser configuration can adjust these with
`maxPendingInputPackets` and `maxPendingInputBytes`; `maxInputInFlight`
controls the acknowledgement window. Exceeding either FIFO limit explicitly
cancels the response and pauses the session, with an overflow diagnostic.
Resume starts a fresh input timeline after the cancellation acknowledgement.
The audio-duplex and Omni pages enable input flow control before that hard limit:
four pending packets suspend microphone capture or file advancement while
the model and output player keep running. Already captured packets retain
their FIFO order. When the queue drains and an input credit is available,
capture resumes automatically. This is an explicit capture pause when
processing is slower than real time, rather than a throughput improvement.
The source AudioContext is suspended without restarting its worklet, so a
partial packet and mixed-file playback position survive the hold. The WAV
recorder keeps assistant playback and leaves silence in the input channel for
the hold interval. Video recording freezes buffered file audio at its consumed
offset while assistant audio continues; input holds also preserve source video
position. File EOF waits for outstanding commits before deciding
that the assistant has finished. Generic clients can opt into the same
contract with `enableInputFlowControl` and `onInputFlowChange`; the hard FIFO
limit remains a fallback for producers that do not pause.
An open WebSocket does not admit inputs or user controls until
`session.created`. Reconnect clears the old FIFO and drops capture produced
while the fresh session is initializing, then admits new capture after its
creation acknowledgement. Stopping during either reconnect stage cancels the
pending wait immediately; a late acknowledgement from a failed connection
cannot reopen the input gate. Reconnect and explicit resume reset the capture
worklet's partial packet and epoch, rejecting late packets from the previous
timeline. Automatic flow release preserves the partial packet. Recorders resume
before a file provider can synchronously deliver its next packet. Media and
recording preparation retain their session owner so cancelled preparation cannot
revive capture or close a replacement session.
The first connection also keeps a cancellation callback through
`session.created`: close or error during initialization rejects that wait and
cleans up the session, instead of leaving the browser in `Connecting`.

Session initialization merges top-level prompt/TTS settings with the nested
sampling configuration; a nested spelling overrides a top-level alias.
System and reference conditioning are evaluated before session readiness is
reported. Cache decreases on interrupt, reset or a new engine epoch establish
a fresh display baseline; they are not sliding-window pruning events.
Automatic stop for basic sliding windows requires a same-epoch transition
across the reported high/low watermarks. A context-window decrease without
explicit pruning evidence does not trigger automatic stop.

Portable browser regression checks, including sustained 1 Hz input against
a simulated five-second-per-packet backend:

```sh
node --test Tests/MiniCPMDemoFrontendTests/*.mjs
```

Listen clock PCM carries `silent: true` on the wire. The browser ignores it
for playback, speech UI and first-audio latency while retaining protocol
metrics. Normal turn boundaries preserve queued waveform tails. Pause sends
`response.cancel` followed by `session.pause`; Resume sends `session.resume`.
Force Listen cancels current speech and suppresses late output. The browser
handles the public `session.state` interrupt/reset events and the internal
`backend.state` spelling, stops scheduled audio, clears cancelled echo
references, and prevents cancellation from starting automatic continuation.

Natural interruption in full-duplex and Omni mode keeps the interrupting
microphone unit. While an answer is open, its first decision on an input marked
`speech_active` uses the native bridge's `listen_top_k = 5` fallback when no
explicit value is configured. A zero-based listen rank below five selects
listen; explicit zero disables this fallback. The sampler's natural chunk-EOS
probe still runs first. Silence and later tokens retain the existing sampling
policy. A model listen closes the old spoken turn before the listen marker is
committed and emits `reason: model_barge_in`. The browser immediately stops
unheard output and crops its WAV tail without clearing the new input FIFO or
its acknowledgement state. Ordinary listen/turn-end boundaries retain tails.

Explicit cancellation rolls back only the provisional unit and closes accepted
speech with canonical turn markers, retaining the question and accepted history.
Replacing those markers copies the retained KV prefix into independent storage;
it does not overwrite snapshots or normally replay the entire context. Deferred
acceptance checks the generation epoch again inside the runtime actor.

Browser AEC remains the primary echo cancellation path. The software reference
stores scheduled source start times, aligns only a fully covered past microphone
window, and checks delays from 0 to 200 ms in 10 ms steps. Future queued playback
and incomplete reference windows cannot suppress input. This conservative
fallback does not establish live acoustic performance; residual echo or a coarse
`speech_active` signal can still affect interruption sensitivity.

Server-generated zero-audio continuation is disabled by default. Set
`MINICPM_SYNTHETIC_CONTINUATION=1` only for diagnostics: synthetic packets
advance the model's input timeline independently of the real audio stream.

## Audio encoding

Load converted audio weights once and keep one session per conversation:

```swift
let model = try MiniCPMAudioModel.fromDirectory(audioBundleURL)
let session = MiniCPMAudioStreamingSession(model: model)

if let result = try session.acceptAudio(oneSecondOfPCM) {
    let embeddings = result.output.embeddings
    // [1, time, 4096], with streaming KV state retained by `session`.
}
```

When an existing frontend already produces `[1,time,80]` mel features, use
`encodeFeatures`.  `prefixExtraFrames` and `suffixExtraFrames` remove
redundant convolution context before positions are appended to the KV cache.
`releaseEncoderContext()` drops generated encoder context and `reset()` also
resets the streaming mel state.  The model enforces its 1,500 source-position
limit and the streaming session resets before an incoming chunk would exceed
it. Regular PCM chunks include the actual convolution overlap: the first
chunk has 102 mel frames and subsequent chunks have 104. After context
removal each one-second chunk has 50 encoder positions and ten audio tokens.
A final flush has no right overlap. Normalization retains the official
20–30 second PCM window, sliding by ten seconds; snapshot/restore includes
that window and its normalization peak.

Streaming attention uses the real key length, matching the official encoder.
`MINICPM_AUDIO_KV_BUCKET=1` explicitly enables the optional 256-position
padding experiment. It changes BF16 reductions and is disabled by default;
the FAST and linear-kernel options do not enable it implicitly.

`MINICPM_AUDIO_LINEAR_TILED=1` selects an optional shared-memory implementation
of the encoder's exact linear projections. It retains the original sequential
FP32 FMA order, separate bias addition and final BF16 rounding boundary. Its
transposed weight tile uses a stride of 33 to avoid shared-memory bank
conflicts without changing that arithmetic. The switch defaults to disabled
in source, and unsupported inputs fall back to the established exact kernel.
This option must pass the same bitwise and real-weight audio gates; it does
not enable the numerically different built-in matmul route.

The waveform frontend uses Float32 CPU pocketFFT, a fixed Float32 periodic
Hann window and Float32 mel reduction, matching the pinned torch frontend.
The Float64 NumPy fallback has different BF16 rounding at the encoder input.
Standalone DSP probes compiled without MLX retain the matrix DFT fallback
and verify framing and normalization only.

## Duplex text and semantic speech

Duplex defaults are temperature 0.7, top-k 20 and top-p 0.8. The duplex
sampler follows the pinned StreamDecoder repetition rule and records only
filtered samples, including protocol markers. Its natural chunk EOS and
listen-top-k early returns consume no final sample. Chat retains the separate
Hugging Face repetition rule. A zero duplex repetition window means all
history; a zero chat window disables its history.

The decoder ends text chunks using protocol tokens and its token budget,
without a visible-character limit. Semantic TTS allows immediate EOS for the
first and final chunks. Middle chunks sample 26 codes and return 25; the last
sample is discarded rather than fed before the next condition. The head
retains checkpoint dtype and reconstructs its weight-normalization graph in
BF16 on MPSGraph once at load. Condition normalization rounds its norm to
that dtype before division.
Llama RoPE also rounds its inverse-frequency buffer to the checkpoint dtype
before promoting it for Float32 trigonometry. BF16 SiLU evaluates the full
activation in Float32 and rounds the final result once, matching the official
speech decoder's operation boundaries.
Dense BF16 TTS projections and multi-token SDPA use the shared MPSGraph
kernels; single-token SDPA and RMSNorm keep their MLX kernels. This combination
passes the pinned official three-chunk, 78-step teacher-logit gate. It adds
synchronous graph submissions, so latency must be measured in a release build.
`MINICPM_TTS_EXACT_KERNELS=mlx` selects the faster MLX diagnostic route;
individual `MINICPM_TTS_BF16_LINEAR`, `MINICPM_TTS_BF16_SDPA` and
`MINICPM_TTS_BF16_RMSNORM` overrides accept `mlx` or `mpsgraph`.

Q/K/V and gate/up projections that share an input are submitted together to
MPSGraph. Each retains its own BF16 matmul and weight; batching does not
concatenate weights or change rounding boundaries. The full 78-step
real-weight gate passed with its original tolerances; that earlier batching
change reduced release execution time for the gate from 17.422 to 11.040
seconds. Set `MINICPM_TTS_BATCHED_PROJECTIONS=0` to use the individual
submissions for diagnostics. These are historical gate timings, not current semantic-unit
latency or a streaming real-time claim. The replay measurement above describes
the newer encoded-decode and tiled-audio configuration with its stated limits.

The response starvation guard is armed only by real user speech and stands
down when model speech starts. Its rescue uses a separate
`listenLogProbabilityScale`: adding `log(scale)` to the listen score reduces
the listen odds even when the score is negative. This is separate from the
official `listenProbabilityScale` raw-logit multiplication, whose semantics
remain unchanged. Startup silence does not arm the guard.

Every non-listen audio unit reaches semantic TTS, including a control-only
final unit. The final waveform drains buffered look-ahead and vocoder
overlap before turn teardown. Rejected provisional units restore the
semantic and waveform checkpoints so already committed speech context
survives a retry.
The runtime also preserves `audioPaddingSamples` through the transport
adapter, so the web player can trim the reported leading silence from short
non-final waveform blocks.
Normal browser turn boundaries preserve queued waveform tails and their
playback timestamps. An explicit interrupt clears that queue and the speaking
state without triggering automatic continuation.
The player also reports the cancellation timestamp to the WAV and video
recorders. WAV export retains the played prefix and cuts future scheduled
samples; video recording schedules audio from the same playback timestamp
and stops future sources before flushing its encoder. Stopping a recording
applies the same cutoff, so exported media does not include cancelled speech.
Video export duration excludes manual pause intervals and encoder flush time.
Video recording waits for the audio graph's first render and MediaRecorder's
start event before advancing file input. During manual pauses the recording
audio clock continues alongside the video clock; encoding pauses and queued
audio sources retain their playback positions. This keeps exported audio and
video timestamps aligned after asynchronous preparation and pause/resume.
Wire recording preserves the original receive/send timestamp across initialization
buffering and media externalization. Use these event timestamps to measure
initialization; metadata creation remains the recorder-ready time.

## Token-to-waveform pipeline

Prepare the voice prompt once per pipeline.  Prompt tokens are 25 Hz, prompt
mel is 50 Hz, and the mel frame count is exactly twice the token count:

```swift
var flowConfiguration = MiniCPMFlowConfiguration()
flowConfiguration.odeSteps = 10

let pipeline = try MiniCPMToken2WavPipeline(
    modelDirectory: token2WavBundleURL,
    flowConfiguration: flowConfiguration)

try pipeline.prepare(prompt: MiniCPMToken2WavPrompt(
    tokens: promptTokens,
    mel: promptMel,
    speakerEmbedding: speakerEmbedding))

let chunks = try pipeline.append(newSpeechTokens)
let finalChunks = try pipeline.append([], isFinal: true)
```

For raw reference PCM, the native model set performs the same preparation
through `MiniCPMNativeModels.prepareVoicePrompt`.  A production server passes
its `--voice-reference` samples to each per-session chat engine and prepares
Token2Wav lazily.  `setTTSVoicePrompt` can replace the reference between turns
when no response is pending; a fixed prompt fixture, if loaded, is immutable.

The pipeline keeps a 25-token stable prefix and three-token look-ahead buffer.
`forceFlush` emits a shorter stable prefix, `isFinal` flushes the remaining
look-ahead and releases generated Flow/HiFT context, `resetForNewTurn()` keeps
the immutable prompt, and `interruptAndReset()`/`releaseContext()` release
prompt and generated state as documented by the API.

## JSONL helpers

The long-lived helpers keep native weights resident:

```text
minicpm-audio-helper <audio-model-dir>
minicpm-token2wav-helper <token2wav-model-dir> [prompt.safetensors] [ode-steps]
```

`minicpm-token2wav-helper` defaults to 10 ODE steps.  Its `prepare_prompt`
operation accepts little-endian Float32 mono PCM and uses the discovered
S3Tokenizer/CAM++ sidecars:

```json
{"id":"ref-1","op":"prepare_prompt","pcmBase64":"<Float32 PCM>","samples":16000,"sampleRate":16000}
```

`promptTokens` and `speakerEmbedding` are explicit test injection fields only.
When deliberately running without sidecars, set
`MINICPM_ALLOW_INJECTED_PROMPT=1`; a supplied fixed prompt fixture does not
need that override.  Normal responses encode generated waveform as
little-endian PCM16 base64 and diagnostics as structured JSON.  `shutdown`
releases resident context.

## Validation and real-model gates

Build-only and shape/unit tests do not prove model parity.  The real-weight
tests are opt-in and skip unless their gates are configured:

```bash
swift test --filter MiniCPMAudioTests --disable-sandbox
swift test --filter MiniCPMToken2WavTests --disable-sandbox
swift test --filter MiniCPMPromptParityTests --disable-sandbox
```

Relevant environment gates are:

```text
MINICPM_AUDIO_MLX_PATH       MINICPM_AUDIO_GOLDEN_DIR
MINICPM_FLOW_MLX_PATH        MINICPM_FLOW_GOLDEN_DIR
MINICPM_HIFT_MLX_PATH        MINICPM_HIFT_GOLDEN_DIR
MINICPM_TOKEN2WAV_MLX_PATH   MINICPM_TOKEN2WAV_FIXTURE
MINICPM_PROMPT_GOLDEN_DIR
MINICPM_SPEECH_TOKENIZER_WEIGHTS   (optional provider gate)
MINICPM_CAMPPLUS_COREML           (optional provider gate)
```

The prompt fixture itself is generated with the pinned Demo implementation
commit, while converted model artifacts use the pinned model repository
revision.  No E2E result is implied by these commands or by the existence of
the sidecars; record the actual run and result before declaring parity.


The bundled tiny semantic oracle runs the pinned original PyTorch
`generate_chunk` against a small Llama decoder. It checks all 78
teacher-forced steps across three chunks, cache boundaries, and early EOS.
The standalone frontend test executes real Swift DSP and the official Python
frontend over packet splits and a 31-second sliding-window case:

```bash
python scripts/tests/test_minicpm_audio_streaming_oracle.py
python -m unittest scripts/tests/test_export_minicpm_o_llm_golden_sampling.py \
  scripts/tests/test_export_minicpm_o_llm_mlx_golden_metrics.py
```

Legacy audio fixtures without `official_StreamingMelProcessorExact` provenance
and semantic decoder fixtures without version-2 teacher logits must be
regenerated with the corresponding exporters. Tiny model parity and sampling
logic tests do not establish full checkpoint or perceptual audio parity.

The optional encoded-decode helper passed six checks, including bitwise
boundaries at logical N=1,024/1,025/2,048/4,096, cache immutability, command-buffer
rollover and session integration. A real-weight 20-layer timing probe uses
synthetic zero-KV prefixes 1,022/1,023/2,047/4,093 and asserts that all three decode
steps per prefix use the encoded path. The final three-chunk, 78-step official
teacher-logit gate passed in 1.961 seconds with unchanged tolerances. These gate
timings are not streaming RTF or perceptual quality measurements.

The final longer replay appended 30 explicit one-second silent inputs after the
34 captured inputs. All 64 inputs committed without protocol errors; 48.28 seconds
of valid speech required 43.9787 seconds of voiced-packet computation (RTF 0.9109).
The TTS cache reached 1,286 positions, covering the two-pass boundary in an actual
session. This does not establish performance above the 4,096-position fast-path
limit or replace browser microphone/AEC and listening acceptance.

The WebSocket probe registers each input's phase and turn before sending it,
and fixes each response's ownership when its first output arrives. Delayed
startup/post-silence speech, earlier-turn output and server continuation do
not inherit the currently receiving question. Initial silence strictly waits
for `input.committed` within the configured turn timeout; a missing commit is
a reported failure, never a paced fallback. Text alone is not an input
completion acknowledgement. Silent PCM is checked for zeros and excluded
from answer WAVs, and declared leading audio padding is trimmed before
measuring first audio or duration. Failed turns still produce their JSON
diagnostics even if they have no speech audio.
