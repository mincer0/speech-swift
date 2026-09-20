# MiniCPM-o Native Streaming Inference

This page describes the current native server and direct Swift APIs.  Runtime
model execution is MLX Swift, Accelerate, and (for CAM++/optional VAD) CoreML;
Python, PyTorch, and ONNX Runtime are not production dependencies.  Conversion
scripts and golden-oracle exporters are offline tools only.

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
`MINICPM_NATIVE_MODEL_ROOT`, `MINICPM_NATIVE_VOICE_REFERENCE`,
`MINICPM_NATIVE_UPSTREAM_ROOT`, `MINICPM_NATIVE_DATA_DIR`,
`MINICPM_NATIVE_HOST`, `MINICPM_NATIVE_PORT`,
`MINICPM_NATIVE_MAX_CONCURRENT`, `MINICPM_NATIVE_ENABLE_VAD` (default `0`),
and `MINICPM_NATIVE_VAD_MODEL_DIR`.  Its `--check` mode verifies the release
binary, all five model directories, voice file, writable data directory, and
the pinned Demo checkout (`ba7fa9cc6ad63c894f1bd5e5afac28466953519d`) without
starting the server.  It never builds, downloads, or substitutes a fake model.

The source model pin is a separate identity:
`073dbbc8c5bc0af2d789e1ce12e7c17a6be746e1` is the MiniCPM-o model repository
revision; `ba7fa9cc6ad63c894f1bd5e5afac28466953519d` is the Demo implementation
commit.

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
it.

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
