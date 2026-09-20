# MiniCPM-o Token2Wav prompt migration

The production prompt path is native.  `MiniCPMToken2WavPromptPreparer` builds
all three feature tensors with Swift/Accelerate, `SpeechTokenizerModel` reads
the converted S3Tokenizer safetensors, and CoreML runs the converted CAM++
sidecar.  Production inference does not import Python, PyTorch, or ONNX
Runtime.  Python/PyTorch (and ONNX tooling) are allowed only for offline
conversion and for generating a golden/oracle fixture.

## Source identity

Keep these two pins separate when reproducing an artifact:

| Identity | Pin | Meaning |
| --- | --- | --- |
| MiniCPM-o model repository | `073dbbc8c5bc0af2d789e1ce12e7c17a6be746e1` | Source model weights/config revision used by the converters |
| MiniCPM-o Demo implementation | `ba7fa9cc6ad63c894f1bd5e5afac28466953519d` | Upstream Demo gateway and prompt-oracle implementation revision |

The first value is a model-repository revision; the second is a Demo code
commit.  Neither value is an alias for the other.

## Source graphs and completed native artifacts

The original release stores these graphs under
`assets/token2wav/`.  They remain offline conversion inputs, not runtime
providers:

| Source graph | Native artifact | Runtime contract |
| --- | --- | --- |
| `speech_tokenizer_v2_25hz.onnx` | `speech_tokenizer.safetensors` | MLX `SpeechTokenizerModel` with `SpeechTokenizerConfig.miniCPMV2` (6 blocks, 1,280 state, 20 heads, 8-way ternary FSQ) |
| `campplus.onnx` | `MiniCPM-CamPlusPlus.mlmodelc` (a `.mlpackage` is also accepted and compiled at load) | CoreML input `[1,500,80]`, output `[1,192]` |

The completed sidecars live beside `flow.safetensors` and `hift.safetensors`
in the Token2Wav bundle.  `MiniCPMMLXServer` resolves them before loading the
production model set; a missing converted sidecar is a startup error.  It
never silently loads the ONNX graph or substitutes an all-zero prompt.

The checked-in source graphs are useful for auditing the conversion.  The S3
graph has 1,358 nodes and 102 initializers; CAM++ has 3,206 nodes and 617
initializers and is normally invoked at `[1,500,80] -> [1,192]`.

## Native prompt graph

1. **S3Tokenizer mel:** resample to 16 kHz, then produce `[1,128,T]` with a
   400-sample FFT, 160-sample hop, Slaney mel bank, log10 clipping, and the
   dropped final STFT frame.  The six-block S3Tokenizer-v2 encoder emits
   25-Hz integer codes through two stride-2 convolutions and affine FSQ.
2. **Flow prompt mel:** resample to 24 kHz and produce `[1,80,T]` at 50 Hz
   (1,920 FFT, 480 hop, magnitude mel, natural-log floor).  The preparer
   pads or crops this tensor to exactly `2 * promptTokenCount` frames.
3. **CAM++ fbank:** resample to 16 kHz and produce Kaldi-compatible `[1,T,80]`
   features (400-sample window, 160-sample hop, Povey window, CMN).  Short
   input is tiled and long input is center-cropped to `[1,500,80]`; CoreML
   returns the 192-dimensional speaker embedding.

`preparePromptAudio` requires both native model outputs (or explicitly
supplied test overrides) and validates all shapes.  It fails closed when a
tokenizer, speaker encoder, or prompt array is missing.

## Offline conversion

Run these commands on a conversion machine, not in the server process:

```bash
python scripts/convert_minicpm_s3tokenizer_onnx_to_safetensors.py \
  /path/MiniCPM-o-4_5/assets/token2wav/speech_tokenizer_v2_25hz.onnx \
  /path/MiniCPM-o-4_5-token2wav-mlx/speech_tokenizer.safetensors \
  --dry-run

python scripts/convert_minicpm_s3tokenizer_onnx_to_safetensors.py \
  /path/MiniCPM-o-4_5/assets/token2wav/speech_tokenizer_v2_25hz.onnx \
  /path/MiniCPM-o-4_5-token2wav-mlx/speech_tokenizer.safetensors

python scripts/convert_minicpm_campplus_coreml.py \
  /path/MiniCPM-o-4_5/assets/token2wav/campplus.onnx \
  /path/MiniCPM-o-4_5-token2wav-mlx/MiniCPM-CamPlusPlus.mlpackage
```

The S3Tokenizer converter records the source model revision and Demo
implementation commit in bundle metadata; keep the same two pins alongside
the CAM++ conversion output.  The converters' Python/PyTorch/ONNX
dependencies must not be added to a production runtime image.

## Sidecar discovery and overrides

`MiniCPMNativePromptEncoder.discoverAssets(in:environment:)` checks these
environment overrides first, then the Token2Wav directory:

```text
MINICPM_SPEECH_TOKENIZER_WEIGHTS=/absolute/path/speech_tokenizer.safetensors
MINICPM_CAMPPLUS_COREML=/absolute/path/MiniCPM-CamPlusPlus.mlmodelc
```

Accepted default filenames are `speech_tokenizer.safetensors`,
`s3tokenizer.safetensors`, `speech_tokenizer_v2_25hz.safetensors` and
`MiniCPM-CamPlusPlus.mlmodelc`, `campplus.mlmodelc`, `CamPlusPlus.mlmodelc`
(or the corresponding `.mlpackage` names).  The server passes the discovered
URLs directly to `MiniCPMNativeModels.load`.

## JSONL helper

`minicpm-token2wav-helper` accepts a raw Float32 mono reference without
precomputed arrays:

```json
{"id":"ref-1","op":"prepare_prompt","pcmBase64":"<little-endian Float32 mono PCM>","samples":16000,"sampleRate":16000}
```

The response reports tokenizer-mel, token, flow-mel, and fixed CAM++ frame
counts and installs the prompt in the resident Flow cache.  `promptTokens` and
`speakerEmbedding` fields are test-only injection hooks; use them only with
`MINICPM_ALLOW_INJECTED_PROMPT=1` when deliberately testing the frontend
without sidecars.  They are not a production fallback.

## Parity gate

Prompt frontend parity is opt-in and independently gated:

```bash
MINICPM_PROMPT_GOLDEN_DIR=/path/to/prompt-golden \
MINICPM_SPEECH_TOKENIZER_WEIGHTS=/path/to/speech_tokenizer.safetensors \
MINICPM_CAMPPLUS_COREML=/path/to/MiniCPM-CamPlusPlus.mlmodelc \
swift test --filter MiniCPMPromptParityTests --disable-sandbox
```

`MINICPM_PROMPT_GOLDEN_DIR` is required for the fixture.  The two sidecar
variables enable their corresponding numerical provider checks and may be set
independently.  This document describes the gate; it does not claim that an
E2E or golden run has been executed.
