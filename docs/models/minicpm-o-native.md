# MiniCPM-o 4.5 Native Audio Runtime

The native MiniCPM-o runtime is split into MLX Swift components.  The
production server loads local safetensors and CoreML bundles directly; it does
not start Python, PyTorch, or ONNX Runtime.  Python/PyTorch/ONNX tooling is
reserved for offline conversion and golden-oracle generation.

## Reproducibility pins

When publishing or reproducing artifacts, record both identities below:

| Identity | Revision | Used for |
| --- | --- | --- |
| `OpenBMB/MiniCPM-o_4_5` model repository | `073dbbc8c5bc0af2d789e1ce12e7c17a6be746e1` | Source weights and model configuration |
| `OpenBMB/MiniCPM-o-Demo` implementation | `ba7fa9cc6ad63c894f1bd5e5afac28466953519d` | Gateway/session behavior and prompt oracle |

The model revision is not a code revision, and the Demo commit is not a model
weight revision.

## Bundle layout and startup gate

Pass the parent directory to `MiniCPMNativeModels.load` or to
`minicpm-mlx-server --model-dir`.  The loader expects these sibling
directories:

```text
MiniCPM-o-4_5-llm-mlx-8bit/
MiniCPM-o-4_5-audio-mlx/
MiniCPM-o-4_5-vision-mlx/
MiniCPM-o-4_5-tts-mlx/
MiniCPM-o-4_5-token2wav-mlx/
  flow.safetensors
  hift.safetensors
  speech_tokenizer.safetensors
  MiniCPM-CamPlusPlus.mlmodelc/       # compiled CoreML sidecar
  MiniCPM-CamPlusPlus.mlpackage/      # source package may be retained
```

The production server requires all five model directories and a PCM16 voice
reference.  Before loading weights it automatically discovers
`speech_tokenizer.safetensors` and the CAM++ CoreML sidecar in the effective
Token2Wav directory.  A missing sidecar is a deterministic startup failure;
there is no zero-voice or ONNX fallback.  `--fake` is an explicit
protocol-only smoke mode, not a model fallback.

## Audio input

`MiniCPMAudioModel` implements MiniCPM-o's 16-kHz streaming audio path:

- 400-sample Whisper log-mel frontend, 80 bins;
- 24-layer, 1,024-dimensional Whisper encoder;
- per-layer streaming KV caches;
- 1,024 -> 4,096 multimodal projection; and
- five-frame average pooling.

The converted audio directory contains `config.json` and
`audio_encoder.safetensors`.  The published runtime configuration is 1,500
maximum source positions, so `MiniCPMAudioStreamingSession` releases its
generated context and restarts positions before a chunk would exceed that
limit.  `releaseEncoderContext()` drops generated context without resetting
the streaming frontend; `reset()` resets both.

## Semantic TTS and Token2Wav

The semantic TTS directory contains `config.json` and `model.safetensors`.
The Token2Wav directory contains the converted Flow and HiFT weights:

1. The six-block plus four-block upsample Conformer maps 25-Hz speech tokens
   to 50-Hz conditioning frames.
2. The 16-block DiT v5 with cosine-Euler flow matching generates 80-bin mel
   frames.  The default runtime configuration is `odeSteps = 10`,
   classifier-free guidance `0.7`, a three-token pre-lookahead, and at most
   100 generated mel frames in the streaming cache.
3. HiFT produces 24-kHz PCM with F0 prediction, SineGen2/source injection,
   transposed convolutions, residual blocks, STFT, and ISTFT.

`MiniCPMToken2WavPipeline` retains an immutable prompt prefix, up to 100
recent generated mel frames, eight HiFT mel frames, and 3,840 source/waveform
samples.  Consecutive chunks use the 160-ms overlap from the reference path.
Regular `append` calls keep 25 stable tokens plus three look-ahead tokens;
`forceFlush` and `isFinal` close shorter/final windows.

## Native reference-prompt providers

`MiniCPMNativePromptEncoder` combines two completed sidecars:

- `speech_tokenizer.safetensors` is loaded by the MLX
  `SpeechTokenizerModel` using the six-block `miniCPMV2` configuration.  Its
  input is `[1,128,T]` and its output is 25-Hz integer codes from an 8-channel
  ternary FSQ (`3^8 = 6,561` codes).
- `MiniCPM-CamPlusPlus.mlmodelc` is loaded by CoreML (Neural Engine is
  available through the `.all` compute-unit configuration).  It accepts
  `[1,500,80]` Kaldi fbank features and returns `[1,192]`.

The Swift front-end creates those features from raw mono PCM at 16 kHz and
24 kHz as documented in
[`minicpm-prompt-onnx-migration.md`](minicpm-prompt-onnx-migration.md).
`prepareVoicePrompt` fails closed when either sidecar is unavailable.  A fixed
`prompt.safetensors` fixture is supported for diagnostics, but it must contain
`prompt_tokens`, `prompt_mel`, and a `[1,192]` `speaker_embedding` and is not a
production substitute for dynamic reference audio.

## Runtime dependency boundary

| Component | Production provider | Offline-only tooling |
| --- | --- | --- |
| LLM/audio/vision/TTS/Flow/HiFT | MLX Swift + Accelerate | conversion scripts and golden exporters |
| S3Tokenizer | MLX Swift + `speech_tokenizer.safetensors` | ONNX graph importer/converter |
| CAM++ | CoreML `.mlmodelc`/`.mlpackage` | ONNX -> TorchScript -> CoreML converter |
| VAD (half-duplex, when enabled) | CoreML Silero | local fixture generation |

No production code path imports Python, Torch, or ONNX Runtime.  The converted
artifacts should be built once, copied into the bundle, and treated as
immutable runtime inputs.

## Current server parameters

`MiniCPMMLXServer` defaults are:

| Option | Default | Notes |
| --- | --- | --- |
| `--host` / `--port` | `127.0.0.1` / `7860` | HTTP/WebSocket bind |
| `--model-dir` | none | Required in production; contains the five sibling bundles |
| `--audio-bundle` | inferred sibling | Optional explicit audio bundle |
| `--token2wav-bundle` | inferred sibling | Optional explicit Token2Wav + prompt sidecar bundle |
| `--voice-reference` | none | Required PCM16 WAV or little-endian raw PCM16 |
| `--max-concurrent` | `1` | Active model sessions |
| `--max-queue-size` | `1000` | `0` rejects queueing |
| `--queue-timeout-ms` | `120000` | `0` disables timeout |
| `--data-dir` | `./data` | Gateway/session storage |
| `--upstream-root` / `--preset-root` | unset | Optional Demo static/preset roots |
| `--vad-model-dir` | standard Silero cache | Explicit offline CoreML VAD bundle/cache |
| `--disable-vad` | off | Half-duplex then requires `utterance_end` |

The optional environment overrides for prompt sidecars are
`MINICPM_SPEECH_TOKENIZER_WEIGHTS` and `MINICPM_CAMPPLUS_COREML`.  The
repository launcher additionally exposes `MINICPM_NATIVE_MODEL_ROOT`,
`MINICPM_NATIVE_VOICE_REFERENCE`, `MINICPM_NATIVE_UPSTREAM_ROOT`,
`MINICPM_NATIVE_DATA_DIR`, `MINICPM_NATIVE_HOST`, `MINICPM_NATIVE_PORT`,
`MINICPM_NATIVE_MAX_CONCURRENT`, `MINICPM_NATIVE_ENABLE_VAD` (default `0`),
and `MINICPM_NATIVE_VAD_MODEL_DIR`.

## Real-model gates

The expensive tests are opt-in and skip when their model/fixture variables are
unset.  The relevant gates are:

```text
MINICPM_AUDIO_MLX_PATH       MINICPM_AUDIO_GOLDEN_DIR
MINICPM_FLOW_MLX_PATH        MINICPM_FLOW_GOLDEN_DIR
MINICPM_HIFT_MLX_PATH        MINICPM_HIFT_GOLDEN_DIR
MINICPM_TOKEN2WAV_MLX_PATH   MINICPM_TOKEN2WAV_FIXTURE
MINICPM_PROMPT_GOLDEN_DIR    (plus optional S3/CAM++ sidecar paths)
MINICPM_LLM_GOLDEN_DIR       MINICPM_LLM_GOLDEN_MODEL (optional override)
MINICPM_VISION_RUN_MODEL=1   MINICPM_VISION_GOLDEN  MINICPM_VISION_BUNDLE
MINICPM_SWIFT_BUNDLE_PARITY=1
MINICPM_TTS_BUNDLE_PATH      MINICPM_TTS_CONDITION_ORACLE_DIR
```

These variables describe the numerical/model gates only.  Their presence is
not evidence that an end-to-end server run has completed; record the actual
test command and result separately.
