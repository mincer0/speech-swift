/**
 * lib/realtime-session.js — OpenAI Realtime-style session manager
 *
 * Realtime session client that speaks the API V2 protocol:
 *   session.init / input.append / response.output.delta(kind=...)
 *
 * Keeps the established callback interface so UI code remains thin.
 */

import { AudioPlayer } from './audio-player.js';
import { FarEndEchoReference } from './echo-reference.js';
import { taskContinuationForAssistantText } from './task-continuation.js';

export class RealtimeSession {
    constructor(prefix, config = {}) {
        this.prefix = prefix;
        this.config = {
            getMaxKvTokens: config.getMaxKvTokens || (() => 8192),
            getPlaybackDelayMs: config.getPlaybackDelayMs || (() => 200),
            getStopOnSlidingWindow: config.getStopOnSlidingWindow || (() => false),
            outputSampleRate: config.outputSampleRate || 24000,
            reconnectAttempts: Number.isFinite(config.reconnectAttempts)
                ? Math.max(0, config.reconnectAttempts) : 3,
            reconnectBaseDelayMs: Number.isFinite(config.reconnectBaseDelayMs)
                ? Math.max(100, config.reconnectBaseDelayMs) : 500,
            // Keep a small acknowledgement window so a slow model cannot
            // accumulate seconds of stale microphone audio in the gateway.
            // The server still enforces its own byte/item budget.
            maxInputInFlight: Number.isFinite(config.maxInputInFlight)
                ? Math.max(1, Math.min(4, Math.floor(config.maxInputInFlight))) : 2,
            getWsUrl: config.getWsUrl || (() => {
                const proto = location.protocol === 'https:' ? 'wss' : 'ws';
                const url = `${proto}://${location.host}/v1/realtime`;
                return window.ClientIdentity ? window.ClientIdentity.appendToUrl(url) : url;
            }),
        };

        this.ws = null;
        this.audioPlayer = new AudioPlayer({
            outputSampleRate: this.config.outputSampleRate,
            getPlaybackDelayMs: this.config.getPlaybackDelayMs,
        });
        this.sessionId = '';
        this.recordingSessionId = '';
        this.chunksSent = 0;
        this.paused = false;
        this.pauseState = 'active';
        this.forceListenActive = false;
        this.currentSpeakText = '';
        this._speakHandle = null;
        this._started = false;
        this._intentionalClose = false;
        this._reconnecting = false;
        this._reconnectPromise = null;
        this._reconnectGeneration = 0;
        this._startGeneration = 0;
        this._sessionArgs = null;
        this.echoReference = new FarEndEchoReference({ sampleRate: 16000 });

        this._sessionStartTime = 0;
        this._lastListenTime = 0;
        this._wasListening = true;
        this._lastTTFS = 0;
        this._lastResultTime = 0;
        this._firstServerTs = 0;
        this._firstClientTs = 0;
        this._resultCount = 0;
        this._lastDriftMs = null;
        this._lastKvCacheLength = 0;
        this._lastFrameMetrics = {};
        this._inputBackpressureUntil = 0;
        this._inputSequence = 0;
        this._recentInputs = new Map();
        this._pendingInput = null;
        this._inputRetryTimer = null;
        this._sessionInitTimer = null;
        this._modelState = 'listening';
        this._autoContinuationUsed = false;
        this._cleaned = true;

        // Protocol event log for the data flow panel
        this._eventLog = [];
        this._maxEventLog = 200;

        this.audioPlayer.onMetrics = (data) => {
            this.onMetrics({
                type: 'audio',
                ahead: data.ahead,
                gapCount: data.gapCount,
                totalShift: data.totalShift,
                rebufferCount: data.rebufferCount,
                lastGapMs: data.lastGapMs,
                bufferTargetMs: data.bufferTargetMs,
                generationLatencyMs: data.generationLatencyMs,
                audioDurationMs: data.audioDurationMs,
                turn: data.turn,
                pdelay: data.pdelay,
            });
        };
        this.audioPlayer.onFarEndReference = (samples, sampleRate, timestamp) => {
            this.echoReference.pushPlayback(samples, sampleRate, timestamp);
        };
    }

    get running() { return this._started; }
    get eventLog() { return this._eventLog; }
    get modelState() { return this._modelState; }

    // ==== Hooks ====
    onSystemLog(text) {}
    onQueueUpdate(data) {}
    onQueueDone() {}
    onSpeakStart(text) { return null; }
    onSpeakUpdate(handle, text) {}
    onSpeakEnd() {}
    onListenResult(result) {}
    onExtraResult(result, recvTime) {}
    async onPrepared() {}
    onCleanup() {}
    onMetrics(data) {}
    onRunningChange(running) {}
    onForceListenChange(active) {}
    onPauseStateChange(state) {}
    /** New: protocol event logged (for data flow panel). */
    onProtocolEvent(entry) {}
    onEchoDiagnostics(data) {}
    onReconnectState(state) {}
    onAutoContinuation(text) {}

    // ==== Protocol event logging ====
    _logProtoEvent(dir, type, summary, full) {
        const entry = {
            ts: Date.now(),
            dir, // 'client' | 'server'
            type,
            summary: summary || '',
            full: full || null,
        };
        this._eventLog.push(entry);
        if (this._eventLog.length > this._maxEventLog) this._eventLog.shift();
        this.onProtocolEvent(entry);
    }

    // ==== Core API ====

    async start(systemPrompt, preparePayload, startMediaFn) {
        this._reset();
        this._intentionalClose = false;
        const startGeneration = ++this._startGeneration;
        this._sessionArgs = { systemPrompt, preparePayload, startMediaFn };
        this.sessionId = '';
        this.recordingSessionId = '';
        this.onMetrics({ type: 'state', sessionState: 'Connecting...' });

        const wsUrl = this.config.getWsUrl();

        try {
            await new Promise((resolve, reject) => {
                this.ws = new WebSocket(wsUrl);
                this.ws.onopen = () => resolve();
                this.ws.onerror = () => reject(new Error('WebSocket connection failed'));
                this.ws.onclose = () => {
                    if (!this._started) reject(new Error('WebSocket closed before ready'));
                };
            });
            this._assertStartActive(startGeneration);

            // Wait for queue + send session.init
            await new Promise((resolve, reject) => {
                let queueDone = false;
                let initSent = false;
                this._queueReject = reject;

                const sendSessionInit = () => {
                    if (initSent
                        || startGeneration !== this._startGeneration
                        || this._intentionalClose
                        || !this.ws
                        || this.ws.readyState !== WebSocket.OPEN) return;
                    initSent = true;
                    this._clearSessionInitTimer();

                    const sessionInit = {
                        type: 'session.init',
                        payload: {
                            system_prompt: systemPrompt,
                            ...preparePayload,
                        },
                    };
                    this.ws.send(JSON.stringify(sessionInit));
                    this._logProtoEvent('client', 'session.init',
                        `system_prompt="${systemPrompt.slice(0, 40)}…"`, sessionInit);
                };

                this.ws.onmessage = (e) => {
                    const msg = JSON.parse(e.data);

                    if (msg.type === 'session.queued') {
                        this._logProtoEvent('server', 'session.queued',
                            `pos=${msg.position}`, msg);
                        this.onQueueUpdate({
                            position: msg.position,
                            estimated_wait_s: msg.estimated_wait_s,
                            ticket_id: msg.ticket_id,
                            queue_length: msg.queue_length,
                        });
                    } else if (msg.type === 'session.queue_update') {
                        this._logProtoEvent('server', 'session.queue_update',
                            `pos=${msg.position}`, msg);
                        this.onQueueUpdate({
                            position: msg.position,
                            estimated_wait_s: msg.estimated_wait_s,
                            queue_length: msg.queue_length,
                        });
                    } else if (msg.type === 'session.queue_done') {
                        queueDone = true;
                        this._queueReject = null;
                        this._logProtoEvent('server', 'session.queue_done', '', msg);
                        this.onQueueDone();
                        this.onQueueUpdate(null);
                        this.onSystemLog('Worker assigned, preparing...');
                        sendSessionInit();

                    // Backward compat: old protocol queue messages
                    } else if (msg.type === 'queued') {
                        this._logProtoEvent('server', 'queued (compat)', `pos=${msg.position}`, msg);
                        this.onQueueUpdate({
                            position: msg.position,
                            estimated_wait_s: msg.estimated_wait_s,
                            ticket_id: msg.ticket_id,
                            queue_length: msg.queue_length,
                        });
                    } else if (msg.type === 'queue_done') {
                        queueDone = true;
                        this._queueReject = null;
                        this._logProtoEvent('server', 'queue_done (compat)', '', msg);
                        this.onQueueDone();
                        this.onQueueUpdate(null);
                        this.onSystemLog('Worker assigned, preparing...');
                        sendSessionInit();

                    } else if (msg.type === 'session.created') {
                        this._clearSessionInitTimer();
                        this._queueReject = null;
                        this.sessionId = msg.session_id || '';
                        this.recordingSessionId = this.sessionId;
                        this._logProtoEvent('server', 'session.created',
                            `session_id=${this.sessionId}`, msg);
                        this.onQueueUpdate(null);
                        this.onMetrics({ type: 'state', sessionId: this.sessionId });
                        this.onSystemLog(`Session created: ${this.sessionId} (${msg.prompt_length || '?'} tokens)`);
                        resolve();
                    } else if (msg.type === 'error') {
                        this._clearSessionInitTimer();
                        this._queueReject = null;
                        this._logProtoEvent('server', 'error',
                            `${msg.error?.code}: ${msg.error?.message}`, msg);
                        const errMsg = msg.error?.message || msg.error || 'Unknown error';
                        reject(new Error(errMsg));
                    }
                };

                this._sessionInitTimer = setTimeout(() => {
                    this._sessionInitTimer = null;
                    if (!queueDone) sendSessionInit();
                }, 100);
            });
            this._assertStartActive(startGeneration);

            this.audioPlayer.init();
            await this.onPrepared();
            this._assertStartActive(startGeneration);
            if (startMediaFn) await startMediaFn();
            this._assertStartActive(startGeneration);

            this._started = true;
            this.onRunningChange(true);
            this.ws.onmessage = (e) => this._handleMessage(JSON.parse(e.data));
            this.ws.onclose = () => this._handleTransportClose();
        } catch (err) {
            if (startGeneration === this._startGeneration && !this._intentionalClose) {
                this.cleanup();
            }
            throw err;
        }
    }

    /**
     * Send audio chunk using the API V2 protocol.
     * Accepts the OLD format { type: 'audio_chunk', audio_base64, ... }
     * and translates to the new { type: 'input.append', input: { audio, ... } }
     */
    sendChunk(msg) {
        if (this.paused) return;
        // Input cadence probe. MiniCPM duplex expects one input chunk per
        // second; the callers self-correct their timers, yet the gap between
        // the server's audio deltas grows to ~1.4 s late in a long session
        // while the server-side stage timings stay flat (~762 ms/unit). That
        // surplus must therefore originate in the browser, so measure the real
        // inter-send interval here instead of guessing where it goes. Logs at
        // most one line every 3 s, and only when the target is actually missed.
        {
            const now = typeof performance !== 'undefined'
                ? performance.now()
                : Date.now();
            if (this._lastSendAt != null) {
                const delta = now - this._lastSendAt;
                if (delta > 1200 && now - (this._lastSendLogAt || 0) > 3000) {
                    this._lastSendLogAt = now;
                    const note = `Feed cadence slip: ${delta.toFixed(0)} ms between input chunks (target 1000 ms).`;
                    if (typeof window !== 'undefined'
                        && typeof window.addSystemLog === 'function') {
                        window.addSystemLog(note);
                    } else {
                        console.warn(note);
                    }
                }
            }
            this._lastSendAt = now;
        }
        if (msg.speech_active === true) this._autoContinuationUsed = false;

        const inputID = this._nextInputID();

        const newMsg = {
            type: 'input.append',
            message_id: inputID,
            input: {
                audio: msg.audio_base64,
                input_id: inputID,
            },
        };

        if (this.forceListenActive || msg.force_listen) {
            newMsg.input.force_listen = true;
        }
        if (msg.frame_base64_list) {
            newMsg.input.video_frames = msg.frame_base64_list;
        }
        if (msg.max_slice_nums) {
            newMsg.input.max_slice_nums = msg.max_slice_nums;
        }
        if (typeof msg.speech_active === 'boolean') {
            newMsg.input.speech_active = msg.speech_active;
        }
        if (typeof msg.allow_sampled_listen_speak_decision === 'boolean') {
            newMsg.input.allow_sampled_listen_speak_decision =
                msg.allow_sampled_listen_speak_decision;
        }
        if (msg.continuation_tick === true) {
            newMsg.input.continuation_tick = true;
        }

        this._observeInput(msg);
        this._submitInput({
            event: newMsg,
            priority: (msg.speech_active === true || msg.force_listen || this.forceListenActive)
                ? 3 : 1,
            retries: 0,
        });
    }

    _nextInputID() {
        this._inputSequence++;
        return `client_in_${String(this._inputSequence).padStart(8, '0')}`;
    }

    _submitInput(entry) {
        if (this._intentionalClose || !this.ws || this.ws.readyState !== WebSocket.OPEN) {
            // Reconnect creates a fresh model timeline, so old audio is never
            // replayed across a transport boundary.
            return false;
        }
        if (performance.now() < this._inputBackpressureUntil) {
            this._queuePendingInput(entry);
            this._schedulePendingInputRetry();
            return false;
        }
        if (this._recentInputs.size >= this.config.maxInputInFlight) {
            this._queuePendingInput(entry);
            this.onMetrics({
                type: 'input_window',
                inFlight: this._recentInputs.size,
                limit: this.config.maxInputInFlight,
            });
            return false;
        }
        return this._sendInput(entry);
    }

    _sendInput(entry) {
        if (!this.ws || this.ws.readyState !== WebSocket.OPEN) return false;
        this.ws.send(JSON.stringify(entry.event));
        this.chunksSent++;

        const inputID = entry.event.message_id;
        if (inputID) {
            this._recentInputs.set(inputID, entry);
            while (this._recentInputs.size > 16) {
                this._recentInputs.delete(this._recentInputs.keys().next().value);
            }
        }

        const input = entry.event.input || {};
        const hasVideo = input.video_frames ? ` +${input.video_frames.length}fr` : '';
        const retry = entry.retries > 0 ? ` retry=${entry.retries}` : '';
        this._logProtoEvent('client', 'input.append',
            `#${this.chunksSent}${hasVideo}${input.force_listen ? ' force' : ''}${retry}`,
            entry.event);
        this.onMetrics({ type: 'result', chunksSent: this.chunksSent });
        return true;
    }

    _queuePendingInput(entry, replaceEqual = false) {
        if (!this._pendingInput
            || entry.priority > this._pendingInput.priority
            || (replaceEqual && entry.priority === this._pendingInput.priority)
            // For audio, latest-wins bounds latency instead of replaying an
            // old packet after the user has already moved on.
            || (entry.priority === this._pendingInput.priority
                && (entry.priority === 1 || entry.priority === 3))) {
            this._pendingInput = entry;
        }
    }

    _schedulePendingInputRetry() {
        if (this._inputRetryTimer || !this._pendingInput || this.paused) return;
        const delay = Math.max(0, this._inputBackpressureUntil - performance.now());
        this._inputRetryTimer = setTimeout(() => {
            this._inputRetryTimer = null;
            if (!this._pendingInput || this.paused || this._intentionalClose) return;
            if (performance.now() < this._inputBackpressureUntil) {
                this._schedulePendingInputRetry();
                return;
            }
            const pending = this._pendingInput;
            this._pendingInput = null;
            // Re-enter the normal admission path.  A retry timer must not
            // bypass the acknowledgement window or a newly received limit.
            if (!this._submitInput(pending) && !this._pendingInput) {
                this._queuePendingInput(pending);
            }
        }, delay);
    }

    _clearPendingInputs() {
        if (this._inputRetryTimer) {
            clearTimeout(this._inputRetryTimer);
            this._inputRetryTimer = null;
        }
        this._pendingInput = null;
        this._recentInputs.clear();
        this._inputBackpressureUntil = 0;
    }

    _takeLatestRecentInput() {
        let latestID = null;
        let latest = null;
        for (const [inputID, entry] of this._recentInputs) {
            latestID = inputID;
            latest = entry;
        }
        if (latestID) this._recentInputs.delete(latestID);
        return latest ? { inputID: latestID, entry: latest } : null;
    }

    _clearSessionInitTimer() {
        if (this._sessionInitTimer) {
            clearTimeout(this._sessionInitTimer);
            this._sessionInitTimer = null;
        }
    }

    toggleForceListen() {
        if (!this.ws || this.ws.readyState !== WebSocket.OPEN) return;
        this.forceListenActive = !this.forceListenActive;
        this.onForceListenChange(this.forceListenActive);
        if (this.forceListenActive) {
            this.onSystemLog('Force Listen ON');
            this.audioPlayer.stopAll();
            if (this.audioPlayer.turnActive) this.audioPlayer.endTurn();
        } else {
            if (this.audioPlayer.turnActive) this.audioPlayer.endTurn();
            this.onSystemLog('Force Listen OFF');
        }
    }

    pauseToggle() {
        if (!this.ws || this.ws.readyState !== WebSocket.OPEN) return;
        if (this.pauseState === 'active') {
            this.paused = true;
            this.pauseState = 'paused';
            this.onPauseStateChange('paused');
            this.onMetrics({ type: 'state', sessionState: 'Paused' });
            this.onSystemLog('Session paused');
        } else if (this.pauseState === 'paused') {
            this.paused = false;
            this.pauseState = 'active';
            this.onPauseStateChange('active');
            this.onMetrics({ type: 'state', sessionState: 'Active' });
            this.onSystemLog('Session resumed');
            this._schedulePendingInputRetry();
        }
    }

    stop() {
        this._intentionalClose = true;
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            const msg = { type: 'session.close', reason: 'user_stop' };
            this.ws.send(JSON.stringify(msg));
            this._logProtoEvent('client', 'session.close', 'user_stop');
        }
        this.cleanup();
    }

    cancelQueue() {
        this._intentionalClose = true;
        const reject = this._queueReject;
        this._queueReject = null;
        this.cleanup();
        if (reject) reject(new Error('Queue cancelled by user'));
    }

    cleanup() {
        if (this._cleaned) return;
        this._cleaned = true;
        this._intentionalClose = true;
        this._startGeneration++;
        this._reconnectGeneration++;
        this._clearSessionInitTimer();
        this._clearPendingInputs();
        this.onCleanup();
        this.audioPlayer.stop();
        if (this.ws) {
            this.ws.onclose = null;
            try { this.ws.close(); } catch (_) {}
            this.ws = null;
        }
        this._started = false;
        this._reconnecting = false;
        this.paused = false;
        this.pauseState = 'active';
        this.forceListenActive = false;
        this.onRunningChange(false);
        this.onForceListenChange(false);
        this.onPauseStateChange('active');
        this.onMetrics({ type: 'state', sessionState: 'Stopped' });
        this._sessionArgs = null;
        this._queueReject = null;
    }

    // ==== Internal ====

    _reset() {
        this._clearSessionInitTimer();
        this._clearPendingInputs();
        this._cleaned = false;
        this._sessionStartTime = performance.now();
        this._lastListenTime = 0;
        this._wasListening = true;
        this._lastTTFS = 0;
        this._lastResultTime = 0;
        this._firstServerTs = 0;
        this._firstClientTs = 0;
        this._resultCount = 0;
        this._lastDriftMs = null;
        this._lastKvCacheLength = 0;
        this._lastFrameMetrics = {};
        this._inputBackpressureUntil = 0;
        this._inputSequence = 0;
        this._modelState = 'listening';
        this._autoContinuationUsed = false;
        this.chunksSent = 0;
        this.currentSpeakText = '';
        this._speakHandle = null;
        this.paused = false;
        this.pauseState = 'active';
        this.forceListenActive = false;
        this._queueReject = null;
        this._eventLog = [];
        this.echoReference.reset();
    }

    _assertStartActive(generation) {
        if (generation !== this._startGeneration || this._intentionalClose) {
            throw new Error('Session start cancelled');
        }
    }

    _observeInput(msg) {
        const samples = msg?.audio_samples instanceof Float32Array
            ? msg.audio_samples
            : (msg?.audio_samples || this._decodeFloat32(msg?.audio_base64));
        if (!samples || samples.length === 0) return;
        const diagnostics = this.echoReference.analyzeInput(samples, 16000);
        this.onEchoDiagnostics(diagnostics);
        this.onMetrics({ type: 'echo', ...diagnostics });
    }

    _decodeFloat32(base64) {
        if (typeof base64 !== 'string' || typeof atob !== 'function') return null;
        try {
            const binary = atob(base64);
            if (binary.length < 4 || binary.length % 4 !== 0) return null;
            const bytes = new Uint8Array(binary.length);
            for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
            return new Float32Array(bytes.buffer);
        } catch (_) {
            return null;
        }
    }

    prepareInputChunk(samples, sampleRate = 16000) {
        return this.echoReference.suppressInput(samples, sampleRate).samples;
    }

    async _handleTransportClose() {
        if (this._intentionalClose || !this._started) {
            if (!this._intentionalClose) this.cleanup();
            return;
        }
        if (this._reconnecting) return;
        this._reconnecting = true;
        const generation = ++this._reconnectGeneration;
        this._clearPendingInputs();
        this._lastListenTime = 0;
        this._wasListening = true;
        this._lastTTFS = 0;
        this._firstServerTs = 0;
        this._firstClientTs = 0;
        this._lastKvCacheLength = 0;
        this._lastFrameMetrics = {};
        this._modelState = 'listening';
        this._autoContinuationUsed = false;
        // There is no resume token in the current backend protocol: a
        // reconnect creates a new model session rather than continuing the
        // old LLM/TTS KV context. Stop queued old audio and close the visible
        // assistant turn before creating that fresh session.
        this.audioPlayer.stopAll();
        if (this.audioPlayer.turnActive) this.audioPlayer.endTurn();
        if (this._speakHandle) this.onSpeakEnd();
        this._speakHandle = null;
        this.currentSpeakText = '';
        this.onReconnectState('reconnecting');
        this.onMetrics({ type: 'state', sessionState: 'Reconnecting...' });
        this.onSystemLog('Network disconnected — reconnecting…');
        try {
            const args = this._sessionArgs;
            if (!args) throw new Error('session arguments unavailable');
            let lastError = null;
            for (let attempt = 0; attempt <= this.config.reconnectAttempts; attempt++) {
                if (generation !== this._reconnectGeneration || this._intentionalClose) return;
                const delay = attempt === 0 ? 0
                    : this.config.reconnectBaseDelayMs * Math.pow(2, attempt - 1);
                if (delay > 0) await new Promise(resolve => setTimeout(resolve, delay));
                if (generation !== this._reconnectGeneration || this._intentionalClose) return;
                try {
                    await this._connectForReconnect(args.systemPrompt, args.preparePayload);
                    if (generation !== this._reconnectGeneration || this._intentionalClose) {
                        if (this.ws) {
                            this.ws.onclose = null;
                            try { this.ws.close(); } catch (_) {}
                            this.ws = null;
                        }
                        return;
                    }
                    this.onReconnectState('connected');
                    this.onSystemLog('Reconnected as a new session — conversation context reset');
                    this.onMetrics({ type: 'state', sessionState: 'Active', sessionId: this.sessionId });
                    this._reconnecting = false;
                    return;
                } catch (error) {
                    lastError = error;
                    if (this.ws) {
                        this.ws.onclose = null;
                        try { this.ws.close(); } catch (_) {}
                        this.ws = null;
                    }
                    this.onSystemLog(`Reconnect attempt ${attempt + 1} failed: ${error.message}`);
                }
            }
            throw lastError || new Error('reconnect attempts exhausted');
        } catch (error) {
            if (generation !== this._reconnectGeneration || this._intentionalClose) return;
            this._reconnecting = false;
            this.onReconnectState('failed');
            this.onSystemLog(`Reconnect failed: ${error.message}`);
            this.cleanup();
        }
    }

    async _connectForReconnect(systemPrompt, preparePayload) {
        const wsUrl = this.config.getWsUrl();
        const ws = new WebSocket(wsUrl);
        this.ws = ws;
        await new Promise((resolve, reject) => {
            const timer = setTimeout(() => reject(new Error('WebSocket reconnect timeout')), 10_000);
            ws.onopen = () => { clearTimeout(timer); resolve(); };
            ws.onerror = () => { clearTimeout(timer); reject(new Error('WebSocket reconnect failed')); };
            ws.onclose = () => { clearTimeout(timer); reject(new Error('WebSocket closed during reconnect')); };
        });

        await new Promise((resolve, reject) => {
            const timer = setTimeout(() => reject(new Error('session.init reconnect timeout')), 30_000);
            ws.onclose = () => {
                clearTimeout(timer);
                reject(new Error('WebSocket closed during session reconnect'));
            };
            let initSent = false;
            const sendInit = () => {
                if (initSent || ws.readyState !== WebSocket.OPEN) return;
                initSent = true;
                ws.send(JSON.stringify({
                    type: 'session.init',
                    payload: { system_prompt: systemPrompt, ...preparePayload },
                }));
            };
            ws.onmessage = event => {
                let msg;
                try { msg = JSON.parse(event.data); } catch (_) { return; }
                if (msg.type === 'session.queued' || msg.type === 'session.queue_update') {
                    this.onQueueUpdate({
                        position: msg.position,
                        estimated_wait_s: msg.estimated_wait_s,
                        queue_length: msg.queue_length,
                    });
                    return;
                }
                if (msg.type === 'session.queue_done' || msg.type === 'queue_done') {
                    this.onQueueUpdate(null);
                    sendInit();
                    return;
                }
                if (msg.type === 'session.created') {
                    clearTimeout(timer);
                    this.sessionId = msg.session_id || '';
                    this.recordingSessionId = this.sessionId;
                    this.onMetrics({ type: 'state', sessionId: this.sessionId });
                    resolve();
                    return;
                }
                if (msg.type === 'error') {
                    clearTimeout(timer);
                    reject(new Error(msg.error?.message || msg.error || 'session.init failed'));
                }
            };
            // The Swift backend can emit queue_done, while compatibility
            // gateways accept session.init immediately.
            setTimeout(sendInit, 100);
        });

        ws.onmessage = event => {
            try { this._handleMessage(JSON.parse(event.data)); }
            catch (_) { this.onSystemLog('Ignored malformed realtime message'); }
        };
        ws.onclose = () => this._handleTransportClose();
    }

    _handleMessage(msg) {
        const type = msg.type || '';
        if (type !== 'response.backpressure' && msg.input_id) {
            this._recentInputs.delete(msg.input_id);
            // A completed input frees one slot in the client window.  The
            // pending packet is sent on the next turn of the event loop so
            // this handler never recursively processes model output.
            this._schedulePendingInputRetry();
        }

        switch (type) {
            case 'input.committed':
                this._logProtoEvent('server', 'input.committed',
                    `input=${msg.input_id || '?'}`, msg);
                this.onMetrics({
                    type: 'input_window',
                    inFlight: this._recentInputs.size,
                    limit: this.config.maxInputInFlight,
                });
                break;

            case 'response.metrics':
                this._logProtoEvent('server', 'response.metrics',
                    `kv=${msg.kv_cache_length}`, msg);
                this._handleMetrics(msg);
                break;

            case 'response.listen':
                this._logProtoEvent('server', 'response.listen',
                    'listen', msg);
                this._handleListen(msg);
                break;

            case 'response.output_audio.delta':
                this._logProtoEvent('server', 'response.output_audio.delta',
                    `"${(msg.text||'').slice(0,30)}" eot=${msg.end_of_turn}`, msg);
                this._handleSpeak(msg);
                break;

            case 'response.output.delta':
                this._handleOutputDelta(msg);
                break;

            case 'session.closed':
                this._logProtoEvent('server', 'session.closed',
                    `reason=${msg.reason}`, msg);
                this.onSystemLog(`Session closed: ${msg.reason}`);
                this.cleanup();
                break;

            case 'error':
                this._logProtoEvent('server', 'error',
                    `${msg.error?.code}: ${msg.error?.message}`, msg);
                this.onSystemLog(`Error: ${msg.error?.message || msg.error}`);
                this._modelState = 'error';
                this.cleanup();
                break;

            case 'response.backpressure': {
                const retryAfter = Number(msg.retry_after_ms) || 250;
                this._inputBackpressureUntil = Math.max(
                    this._inputBackpressureUntil,
                    performance.now() + Math.max(100, retryAfter));
                const rejectedID = msg.input_id || '';
                const tracked = rejectedID
                    ? (() => {
                        const entry = this._recentInputs.get(rejectedID);
                        if (entry) this._recentInputs.delete(rejectedID);
                        return entry ? { inputID: rejectedID, entry } : null;
                    })()
                    : this._takeLatestRecentInput();
                const rejected = tracked?.entry || null;
                const retryID = tracked?.inputID || rejectedID;
                if (rejected) {
                    const retryEntry = { ...rejected, retries: rejected.retries + 1 };
                    if (retryEntry.retries <= 8) {
                        this._queuePendingInput(retryEntry, true);
                        this._schedulePendingInputRetry();
                    } else {
                        this.onSystemLog(`Input ${retryID || '?'} dropped after 8 bounded retries`);
                    }
                }
                this._logProtoEvent('server', 'response.backpressure',
                    `input=${retryID || '?'} retry=${retryAfter}ms`, msg);
                this.onSystemLog(rejected
                    ? `Input busy — retrying ${retryID || 'latest'} in ${retryAfter}ms`
                    : `Input busy — retrying new input in ${retryAfter}ms`);
                this.onMetrics({ type: 'backpressure', retryAfterMs: retryAfter });
                break;
            }

            // Backward compat: old protocol events
            case 'result':
                this._handleResultCompat(msg);
                break;
            case 'stopped':
                this.onSystemLog('Session stopped');
                this.cleanup();
                break;
            case 'timeout':
                this.onSystemLog(`Timeout: ${msg.reason}`);
                this.cleanup();
                break;
            case 'queued':
            case 'queue_update':
            case 'session.queued':
            case 'session.queue_update':
                this.onQueueUpdate({
                    position: msg.position,
                    estimated_wait_s: msg.estimated_wait_s,
                    queue_length: msg.queue_length,
                });
                break;
            case 'queue_done':
            case 'session.queue_done':
                this.onQueueUpdate(null);
                break;
        }
    }

    /** Network drift: latency change vs the first result (ported from duplex-session.js). */
    _updateDrift(msg) {
        this._lastDriftMs = null;
        const serverSendSec = msg && msg.server_send_ts;
        if (!serverSendSec) return;
        const clientRecvSec = Date.now() / 1000;
        if (!this._firstServerTs) {
            this._firstServerTs = serverSendSec;
            this._firstClientTs = clientRecvSec;
        }
        this._lastDriftMs = (clientRecvSec - serverSendSec
            - (this._firstClientTs - this._firstServerTs)) * 1000;
    }

    _applyFrameMetrics(msg) {
        this._updateDrift(msg);
        if (msg && typeof msg.metrics === 'object' && msg.metrics !== null) {
            this._lastFrameMetrics = msg.metrics;
            return;
        }
        if (msg && (msg.kv_cache_length !== undefined || msg.wall_clock_ms !== undefined || msg.generate_ms !== undefined)) {
            this._lastFrameMetrics = {
                ...this._lastFrameMetrics,
                kv_cache_length: msg.kv_cache_length,
                wall_clock_ms: msg.wall_clock_ms,
                generate_ms: msg.generate_ms,
            };
        }
    }

    _handleOutputDelta(msg) {
        const kind = msg.kind || '';
        this._applyFrameMetrics(msg);
        this._logProtoEvent('server', `response.output.delta/${kind}`,
            kind === 'text' ? `"${(msg.text || '').slice(0, 30)}"`
                : kind === 'audio' ? `audio=${msg.audio ? msg.audio.length : 0}`
                : kind || 'unknown',
            msg);

        if (kind === 'listen') {
            this._handleListen(msg);
        } else if (kind === 'text') {
            this._handleSpeak({
                ...msg,
                audio: undefined,
            });
        } else if (kind === 'audio') {
            this._handleSpeak({
                ...msg,
                text: '',
            });
        }
    }

    /** Handle new protocol response.listen */
    _handleListen(msg) {
        this._applyFrameMetrics(msg);
        const recvTime = performance.now();
        this._resultCount++;
        this._lastListenTime = recvTime;
        this._wasListening = true;
        this._modelState = 'listening';

        if (this.audioPlayer.turnActive) this.audioPlayer.endTurn();

        const result = {
            is_listen: true,
            kv_cache_length: this._lastFrameMetrics.kv_cache_length,
        };

        this._checkKvCache(result);
        this._emitMetrics(result, recvTime);

        const completedText = this.currentSpeakText;
        if (this._speakHandle) {
            this.onSpeakEnd();
        }
        this._speakHandle = null;
        this.currentSpeakText = '';
        if (completedText) this.onSystemLog('— end of turn —');
        this.onListenResult(result);
        this.onExtraResult(result, recvTime);
        this._lastResultTime = recvTime;
        this._maybeAutoContinue(completedText);
    }

    _maybeAutoContinue(completedText) {
        const text = taskContinuationForAssistantText(completedText);
        if (!text || this._autoContinuationUsed || this.paused || this._intentionalClose) return;
        if (!this.ws || this.ws.readyState !== WebSocket.OPEN) return;

        this._autoContinuationUsed = true;
        const inputID = this._nextInputID();
        const event = {
            type: 'input.append',
            message_id: inputID,
            input: {
                text,
                input_id: inputID,
                allow_sampled_listen_speak_decision: true,
            },
        };
        this._submitInput({ event, priority: 2, retries: 0 });
        this._modelState = 'waiting_response';
        this.onAutoContinuation(text);
        this.onSystemLog('Continuing the requested story...');
    }

    /** Handle new protocol response.output_audio.delta */
    _handleSpeak(msg) {
        this._applyFrameMetrics(msg);
        const recvTime = performance.now();
        this._resultCount++;
        this._modelState = msg.end_of_turn ? 'end_of_turn' : 'speaking';

        if (this._wasListening) {
            this._wasListening = false;
            this._lastTTFS = this._lastListenTime > 0
                ? recvTime - this._lastListenTime : 0;
        }

        if (msg.audio) {
            if (!this.audioPlayer.turnActive) this.audioPlayer.beginTurn();
            this.audioPlayer.playChunk(
                msg.audio,
                recvTime,
                msg.metrics?.wall_clock_ms ?? this._lastFrameMetrics.wall_clock_ms,
                msg.metrics?.audio_padding_samples
                    ?? this._lastFrameMetrics.audio_padding_samples,
            );
        }

        const result = {
            is_listen: false,
            text: msg.text || '',
            audio_data: msg.audio,
            end_of_turn: msg.end_of_turn || false,
            kv_cache_length: this._lastFrameMetrics.kv_cache_length,
        };

        this._checkKvCache(result);
        this._emitMetrics(result, recvTime);

        if (result.text) {
            this.currentSpeakText += result.text;
            if (!this._speakHandle) {
                this._speakHandle = this.onSpeakStart(this.currentSpeakText);
            } else {
                this.onSpeakUpdate(this._speakHandle, this.currentSpeakText);
            }
        }

        this.onExtraResult(result, recvTime);
        this._lastResultTime = recvTime;
    }

    /** Handle old protocol 'result' for backward compat (when gateway doesn't translate) */
    _handleResultCompat(result) {
        if (result.is_listen) {
            this._handleListen({
                kv_cache_length: result.kv_cache_length,
            });
        } else {
            this._handleSpeak({
                text: result.text,
                audio: result.audio_data,
                end_of_turn: result.end_of_turn,
                kv_cache_length: result.kv_cache_length,
            });
        }
    }

    _emitMetrics(result, recvTime) {
        const maxKv = this.config.getMaxKvTokens();
        const metrics = this._lastFrameMetrics || {};
        requestAnimationFrame(() => {
            this.onMetrics({
                type: 'result',
                latencyMs: metrics.wall_clock_ms || metrics.generate_ms,
                costAllMs: metrics.generate_ms,
                driftMs: this._lastDriftMs,
                kvCacheLength: result.kv_cache_length,
                maxKvTokens: maxKv,
                ttfsMs: (!result.is_listen && this._lastTTFS) ? this._lastTTFS : null,
                modelState: result.is_listen ? 'listening' : (result.end_of_turn ? 'end_of_turn' : 'speaking'),
                chunksSent: this.chunksSent,
                visionSlices: metrics.vision_slices,
                visionTokens: metrics.vision_tokens,
            });
            if (!result.is_listen && this._lastTTFS) this._lastTTFS = 0;
        });
    }

    _handleMetrics(metrics) {
        this._lastFrameMetrics = metrics || {};
        const maxKv = this.config.getMaxKvTokens();
        const kvCacheLength = this._lastFrameMetrics.kv_cache_length;
        this._checkKvCache({ kv_cache_length: kvCacheLength });
        this.onMetrics({
            type: 'result',
            latencyMs: this._lastFrameMetrics.wall_clock_ms || this._lastFrameMetrics.generate_ms,
            costAllMs: this._lastFrameMetrics.generate_ms,
            driftMs: this._lastDriftMs,
            kvCacheLength,
            maxKvTokens: maxKv,
            chunksSent: this.chunksSent,
            visionSlices: this._lastFrameMetrics.vision_slices,
            visionTokens: this._lastFrameMetrics.vision_tokens,
        });
    }

    _checkKvCache(result) {
        const maxKv = this.config.getMaxKvTokens();
        const curKv = result.kv_cache_length;
        if (curKv !== undefined && curKv > 0) {
            if (curKv >= maxKv) {
                this.onSystemLog(`⚠ KV cache (${curKv.toLocaleString()}) reached limit. Auto-stopping.`);
                setTimeout(() => this.stop(), 0);
            } else if (this._lastKvCacheLength > 0 && curKv < this._lastKvCacheLength) {
                const prev = this._lastKvCacheLength;
                this.onSystemLog(`✂ KV pruned: ${prev.toLocaleString()} → ${curKv.toLocaleString()}`);
                if (this.config.getStopOnSlidingWindow()) {
                    this.onSystemLog('⚠ Stop-on-sliding-window. Auto-stopping.');
                    setTimeout(() => this.stop(), 0);
                }
            }
            this._lastKvCacheLength = curKv;
        }
    }

}
