# MiniCPM-o Demo web resources

This directory is a vendored browser frontend from the upstream
[MiniCPM-o-Demo](https://github.com/OpenBMB/MiniCPM-o-Demo) repository at
commit `ba7fa9cc6ad63c894f1bd5e5afac28466953519d` (2026-07-20).  The upstream
frontend is distributed with that repository.  The pinned tree does not contain
a top-level `LICENSE` file, so this vendoring note is an attribution record,
not a new license grant; retain upstream notices and review the repository's
current license terms before redistributing a build.

The static tree is intentionally kept as a complete dependency closure: the
turn-based, half-duplex, audio full-duplex, omni, generic realtime, mobile
omni, home, FAQ, settings/editor, admin, session viewer, and diagnostic pages
plus their shared JavaScript/CSS/worklet/media assets are all under `Web/`.

Local integration changes:

- `duplex/lib/capture-processor.js` resamples the actual AudioWorklet input
  rate to the requested target (16 kHz for model input) while preserving
  pass-through audio for monitoring/recording.
- Full-duplex and omni microphone constraints request browser AEC,
  noise-suppression, and auto-gain; they continue sending input while the
  model speaks.
- `duplex/lib/echo-reference.js` and `AudioPlayer.onFarEndReference` expose
  far-end energy/correlation diagnostics without implementing an unsafe
  speaker-energy input gate.
- `duplex/lib/realtime-session.js` keeps a bounded reconnect outbox, retries
  `/v1/realtime`, and replays only real `input.append` frames after a fresh
  `session.init`; because the current wire protocol has no resume token, it
  explicitly reports that the new session's LLM/TTS context was reset.
- The half-duplex page uses `/v1/realtime?mode=half_duplex`; legacy event
  handlers remain documented in the page for older gateway deployments.

The server must mount this directory as a static web root so `/index.html`,
`/static/...`, `/turnbased`, `/half_duplex`, `/audio_duplex`, `/omni`, and
`/v1/realtime` (including `?mode=audio`, `?mode=video`, and `?mode=half`)
remain same-origin.  See the frontend test README for a
read-only mount checklist.
