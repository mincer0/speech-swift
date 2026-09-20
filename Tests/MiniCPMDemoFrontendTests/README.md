# MiniCPM demo frontend checks

These are dependency-free Node tests for the vendored static frontend. They do
not start a server, open port 7860, request a microphone, load a model, or run
Swift/MLX tests.

From the package root:

```sh
node --test Tests/MiniCPMDemoFrontendTests/frontend_static_test.mjs
```

Browser acceptance (after the server mounts
`Sources/MiniCPMDemoBackend/Resources/Web` as its static root):

1. Open `/`, `/turnbased`, `/half_duplex`, `/audio_duplex`, `/omni`, and
   `/realtime`; verify each page loads its CSS/JS without a 404.
2. On audio full-duplex and omni, grant microphone permission and inspect
   `MediaStreamTrack.getSettings()` for `echoCancellation`,
   `noiseSuppression`, and `autoGainControl`. Keep speaking while the model
   speaks; input chunks must continue to arrive (barge-in is model-side, not a
   client energy gate).
3. Verify the worklet sends one-second 16 kHz Float32 chunks even when the
   browser chooses a 44.1/48 kHz AudioContext. Verify the player queues 24 kHz
   output continuously across multiple `response.output.delta` events.
4. Drop the WebSocket in devtools. The session should show
   “Reconnecting…”, retry `/v1/realtime`, then replay only buffered
   `input.append` frames after a new `session.created` event. No local/fake
   model response should appear.
5. Toggle Force Listen while audio is playing. Playback stops immediately and
   the next real input frame carries `force_listen: true`.
6. Stop the session and download the recorder output; left is 16 kHz captured
   input and right is scheduled 24 kHz model audio resampled to 16 kHz.

Required server routes are `/v1/realtime` (WebSocket), `/api/*` preset/default
voice helpers, and static `/static/*`. The half-duplex page no longer uses the
legacy `/ws/half_duplex/:id` route, although it still understands legacy event
names for compatibility.

