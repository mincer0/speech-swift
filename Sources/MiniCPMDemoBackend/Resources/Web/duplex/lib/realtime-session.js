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
            // Keep a small acknowledgement window and retain unsent audio
            // locally in timeline order. Backlog limits pause the session
            // explicitly instead of silently dropping microphone packets.
            maxInputInFlight: Number.isFinite(config.maxInputInFlight)
                ? Math.max(1, Math.min(4, Math.floor(config.maxInputInFlight))) : 2,
            maxPendingInputPackets: Number.isFinite(config.maxPendingInputPackets)
                ? Math.max(1, Math.min(64, Math.floor(config.maxPendingInputPackets))) : 16,
            maxPendingInputBytes: Number.isFinite(config.maxPendingInputBytes)
                ? Math.max(1024, Math.min(32 * 1024 * 1024, Math.floor(config.maxPendingInputBytes)))
                : 8 * 1024 * 1024,
            // Producers opt in only after they can synchronously hold capture.
            // Keep the hard queue limits for producers that do not cooperate.
            enableInputFlowControl: config.enableInputFlowControl === true,
            getWsUrl: config.getWsUrl || (() => {
                const proto = location.protocol === 'https:' ? 'wss' : 'ws';
                const url = `${proto}://${location.host}/v1/realtime`;
                return window.ClientIdentity ? window.ClientIdentity.appendToUrl(url) : url;
            }),
        };
        this.config.inputFlowHighWaterPackets = Number.isFinite(config.inputFlowHighWaterPackets)
            ? Math.max(1, Math.min(this.config.maxPendingInputPackets,
                Math.floor(config.inputFlowHighWaterPackets)))
            : Math.min(4, this.config.maxPendingInputPackets);

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
        this._pendingCancelAcks = 0;
        this.currentSpeakText = '';
        this._speakHandle = null;
        this._started = false;
        this._intentionalClose = false;
        this._reconnecting = false;
        // An open transport is not yet a model session. Microphone capture
        // can keep running during reconnect, but its old timeline must never
        // precede session.init or be replayed into the fresh model context.
        this._sessionReady = false;
        this._reconnectPromise = null;
        this._reconnectReject = null;
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
        this._lastKvEngineEpoch = null;
        this._lastFrameMetrics = {};
        this._inputBackpressureUntil = 0;
        this._inputSequence = 0;
        this._recentInputs = new Map();
        this._pendingInputs = [];
        this._pendingInputBytes = 0;
        this._inputFlowHeld = false;
        this._inputAdmissionSequence = 0;
        this._lastSubmittedInputSequence = 0;
        this._usesInputCommitted = false;
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
        this.audioPlayer.onPlaybackTimelineReset = () => this.echoReference.reset();
        this.audioPlayer.onPlaybackError = error => {
            this._logProtoEvent('client', 'audio.playback_error', error.message);
            this.onSystemLog(`Audio playback could not start: ${error.message}`);
        };
    }

    get running() { return this._started; }
    get eventLog() { return this._eventLog; }
    get modelState() { return this._modelState; }
    get inputFlowHeld() { return this._inputFlowHeld; }
    get _awaitingCancelAck() { return this._pendingCancelAcks > 0; }

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
    onInputFlowChange(state) {}
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
                const fail = error => {
                    if (this._queueReject === fail) this._queueReject = null;
                    reject(error);
                };
                this._queueReject = fail;
                this.ws.onopen = () => {
                    if (this._queueReject === fail) this._queueReject = null;
                    resolve();
                };
                this.ws.onerror = () => fail(new Error('WebSocket connection failed'));
                this.ws.onclose = () => fail(new Error('WebSocket closed before ready'));
            });
            this._assertStartActive(startGeneration);

            // Wait for queue + send session.init
            await new Promise((resolve, reject) => {
                let queueDone = false;
                let initSent = false;
                const fail = error => {
                    this._clearSessionInitTimer();
                    if (this._queueReject === fail) this._queueReject = null;
                    reject(error);
                };
                this._queueReject = fail;
                this.ws.onerror = () => {
                    if (this._started || startGeneration !== this._startGeneration || this._intentionalClose) return;
                    if (this._sessionReady) this.cleanup();
                    fail(new Error('WebSocket failed during session initialization'));
                };
                this.ws.onclose = () => {
                    if (this._started || startGeneration !== this._startGeneration || this._intentionalClose) return;
                    if (this._sessionReady) this.cleanup();
                    fail(new Error('WebSocket closed during session initialization'));
                };

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
                    if (startGeneration !== this._startGeneration || this._intentionalClose) return;
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
                        this._logProtoEvent('server', 'queue_done (compat)', '', msg);
                        this.onQueueDone();
                        this.onQueueUpdate(null);
                        this.onSystemLog('Worker assigned, preparing...');
                        sendSessionInit();

                    } else if (msg.type === 'session.created') {
                        this._clearSessionInitTimer();
                        this._queueReject = null;
                        this._sessionReady = true;
                        this.sessionId = msg.session_id || '';
                        this._usesInputCommitted = msg.metrics?.backend === 'swift_mlx';
                        this._lastFrameMetrics = msg.metrics || {};
                        this._resetKvBaseline(this._lastFrameMetrics);
                        this.recordingSessionId = this.sessionId;
                        this._logProtoEvent('server', 'session.created',
                            `session_id=${this.sessionId}`, msg);
                        this.onQueueUpdate(null);
                        this.onMetrics({ type: 'state', sessionId: this.sessionId });
                        this.onSystemLog(`Session created: ${this.sessionId} (${msg.prompt_length || '?'} tokens)`);
                        resolve();
                    } else if (msg.type === 'error') {
                        this._logProtoEvent('server', 'error',
                            `${msg.error?.code}: ${msg.error?.message}`, msg);
                        const errMsg = msg.error?.message || msg.error || 'Unknown error';
                        fail(new Error(errMsg));
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
            this.ws.onerror = null;
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
        if (this.paused || !this._sessionReady) return;
        // Measure unexpected slips during active capture. Deliberate flow
        // holds reset the baseline so their processing wait is not reported
        // as a browser timer failure.
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
                // Model turn end can arrive before the audible tail drains.
                playback_active: this.audioPlayer.hasBufferedAudio === true,
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
            retries: 0,
        });
    }

    _nextInputID() {
        this._inputSequence++;
        return `client_in_${String(this._inputSequence).padStart(8, '0')}`;
    }

    _submitInput(entry) {
        if (this.paused || !this._sessionReady || this._intentionalClose
            || !this.ws || this.ws.readyState !== WebSocket.OPEN) {
            // Reconnect creates a fresh model timeline, so old audio is never
            // replayed across a transport boundary.
            return false;
        }
        if (!Number.isFinite(entry.sequence)) entry.sequence = ++this._inputAdmissionSequence;
        if (this._pendingInputs.length
            || this._awaitingCancelAck
            || performance.now() < this._inputBackpressureUntil
            || this._recentInputs.size >= this.config.maxInputInFlight) {
            if (!this._queuePendingInput(entry)) return false;
            this._schedulePendingInputRetry();
            this.onMetrics({
                type: 'input_window',
                inFlight: this._recentInputs.size,
                limit: this.config.maxInputInFlight,
                pending: this._pendingInputs.length,
            });
            return false;
        }
        return this._sendInput(entry);
    }

    _sendInput(entry) {
        if (!this._sessionReady || !this.ws || this.ws.readyState !== WebSocket.OPEN) return false;
        this.ws.send(JSON.stringify(entry.event));
        this._lastSubmittedInputSequence = Math.max(this._lastSubmittedInputSequence, entry.sequence);
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

    _queuePendingInput(entry) {
        const inputID = entry.event.message_id;
        if (this._pendingInputs.some(pending => pending.event.message_id === inputID)) return true;
        entry.queuedBytes = JSON.stringify(entry.event).length * 2;
        if (this._pendingInputs.length >= this.config.maxPendingInputPackets
            || this._pendingInputBytes + entry.queuedBytes > this.config.maxPendingInputBytes) {
            this._pauseInputTimeline('Input backlog exceeded its audio queue limit');
            return false;
        }
        // A rejected packet retains its original sequence and returns before
        // newer unsent packets; speech, silence and continuation share one FIFO.
        const index = this._pendingInputs.findIndex(pending => pending.sequence > entry.sequence);
        if (index < 0) this._pendingInputs.push(entry);
        else this._pendingInputs.splice(index, 0, entry);
        this._pendingInputBytes += entry.queuedBytes;
        this._updateInputFlowControl();
        return true;
    }

    _setInputFlowHeld(held, reason = 'drain') {
        if (this._inputFlowHeld === held) return;
        this._inputFlowHeld = held;
        this._lastSendAt = null;
        const state = held ? 'held' : 'active';
        if (held) {
            this.onSystemLog('Processing queued audio; input capture temporarily paused.');
        } else if (reason === 'drain') {
            this.onSystemLog('Queued audio caught up; input capture resumed.');
        }
        this.onMetrics({
            type: 'input_flow',
            state,
            held,
            reason,
            pending: this._pendingInputs.length,
            pendingBytes: this._pendingInputBytes,
            inFlight: this._recentInputs.size,
            highWaterPackets: this.config.inputFlowHighWaterPackets,
        });
        // The producer must pause synchronously on held. Already posted
        // worklet messages still enter the same FIFO instead of being lost.
        // On explicit clear, the producer checks the session lifecycle before
        // resuming; clearing a stopped or paused session is not a drain.
        this.onInputFlowChange(state);
    }

    _updateInputFlowControl() {
        if (!this.config.enableInputFlowControl) return;
        if (!this._inputFlowHeld
            && this._pendingInputs.length >= this.config.inputFlowHighWaterPackets) {
            this._setInputFlowHeld(true);
        } else if (this._inputFlowHeld && !this._pendingInputs.length
            && this._recentInputs.size < this.config.maxInputInFlight
            && this._sessionReady && !this.paused && !this._awaitingCancelAck
            && !this._intentionalClose) {
            this._setInputFlowHeld(false);
        }
    }

    _pauseInputTimeline(reason) {
        this.onSystemLog(`${reason}. Session paused; resume to start fresh input.`);
        this.onMetrics({
            type: 'input_overflow',
            reason,
            pending: this._pendingInputs.length,
            pendingBytes: this._pendingInputBytes,
            packetLimit: this.config.maxPendingInputPackets,
            byteLimit: this.config.maxPendingInputBytes,
        });
        this.paused = true;
        this.pauseState = 'paused';
        this._cancelCurrentResponse();
        this._sendControl('session.pause');
        this.onPauseStateChange('paused');
        this.onMetrics({ type: 'state', sessionState: 'Paused' });
    }

    _schedulePendingInputRetry() {
        if (this._inputRetryTimer || !this._pendingInputs.length || this.paused || !this._sessionReady
            || this._awaitingCancelAck || this._intentionalClose
            || this._recentInputs.size >= this.config.maxInputInFlight) return;
        const delay = Math.max(0, this._inputBackpressureUntil - performance.now());
        this._inputRetryTimer = setTimeout(() => {
            this._inputRetryTimer = null;
            if (this.paused || !this._sessionReady || this._awaitingCancelAck || this._intentionalClose
                || !this.ws || this.ws.readyState !== WebSocket.OPEN) return;
            if (performance.now() < this._inputBackpressureUntil) {
                this._schedulePendingInputRetry();
                return;
            }
            while (this._pendingInputs.length
                && this._recentInputs.size < this.config.maxInputInFlight) {
                const pending = this._pendingInputs.shift();
                this._pendingInputBytes -= pending.queuedBytes;
                this._sendInput(pending);
            }
            this._updateInputFlowControl();
        }, delay);
    }

    _clearPendingInputs() {
        // Cancellation, explicit pause and reconnect begin a fresh timeline
        // even when the queue never reached the automatic hold watermark.
        this._lastSendAt = null;
        if (this._inputRetryTimer) {
            clearTimeout(this._inputRetryTimer);
            this._inputRetryTimer = null;
        }
        this._pendingInputs = [];
        this._pendingInputBytes = 0;
        this._recentInputs.clear();
        this._inputBackpressureUntil = 0;
        this._lastSubmittedInputSequence = 0;
        this._setInputFlowHeld(false, 'clear');
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

    _sendControl(type) {
        if (!this._sessionReady || !this.ws || this.ws.readyState !== WebSocket.OPEN) return false;
        this.ws.send(JSON.stringify({ type }));
        this._logProtoEvent('client', type, '');
        return true;
    }

    _stopCurrentSpeech() {
        this.audioPlayer.stopAll();
        // References are collected when sources are scheduled. Cancelled
        // future audio must not suppress the user's next microphone chunk.
        this.echoReference.reset();
        if (this._speakHandle) this.onSpeakEnd();
        this._speakHandle = null;
        this.currentSpeakText = '';
        this._modelState = 'listening';
        this._wasListening = true;
        this._lastListenTime = performance.now();
        this._lastTTFS = 0;
        this.onMetrics({ type: 'result', modelState: 'listening' });
    }

    _cancelCurrentResponse() {
        // Establish the barrier before the flow callback can synchronously
        // restart a producer with a fresh packet. Each successful cancel has
        // its own interrupt ACK; one ACK cannot reopen input past a later
        // cancellation that is still pending.
        this._pendingCancelAcks++;
        this._resetKvBaseline();
        this._clearPendingInputs();
        this._stopCurrentSpeech();
        let sent = false;
        try {
            sent = this._sendControl('response.cancel');
        } finally {
            if (!sent) this._pendingCancelAcks = Math.max(0, this._pendingCancelAcks - 1);
        }
    }

    toggleForceListen() {
        if (!this._sessionReady || !this.ws || this.ws.readyState !== WebSocket.OPEN) return;
        this.forceListenActive = !this.forceListenActive;
        this.onForceListenChange(this.forceListenActive);
        if (this.forceListenActive) {
            this.onSystemLog('Force Listen ON');
            this._cancelCurrentResponse();
        } else {
            this.onSystemLog('Force Listen OFF');
        }
    }

    pauseToggle() {
        if (!this._sessionReady || !this.ws || this.ws.readyState !== WebSocket.OPEN) return;
        if (this.pauseState === 'active') {
            this.paused = true;
            this.pauseState = 'paused';
            // Invalidate already queued inputs before pausing the worker;
            // otherwise an old input can reach a paused backend and fail.
            this._cancelCurrentResponse();
            this._sendControl('session.pause');
            this.onPauseStateChange('paused');
            this.onMetrics({ type: 'state', sessionState: 'Paused' });
            this.onSystemLog('Session paused');
        } else if (this.pauseState === 'paused') {
            this._sendControl('session.resume');
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
        if (this._sessionReady && this.ws && this.ws.readyState === WebSocket.OPEN) {
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
        this._sessionReady = false;
        this._startGeneration++;
        this._reconnectGeneration++;
        const rejectReconnect = this._reconnectReject;
        this._reconnectReject = null;
        if (rejectReconnect) rejectReconnect(new Error('Reconnect cancelled by user'));
        const rejectStartup = this._queueReject;
        this._queueReject = null;
        if (rejectStartup) rejectStartup(new Error('Session start cancelled by user'));
        this._clearSessionInitTimer();
        this._clearPendingInputs();
        // Crop cancelled output on its audio clock before the page finalizes
        // the recording, including when the output context is interrupted.
        this.audioPlayer.stop();
        this.onCleanup();
        this.echoReference.reset();
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
        this._pendingCancelAcks = 0;
        this.onRunningChange(false);
        this.onForceListenChange(false);
        this.onPauseStateChange('active');
        this.onMetrics({ type: 'state', sessionState: 'Stopped' });
        this._sessionArgs = null;
        this._queueReject = null;
    }

    // ==== Internal ====

    _reset() {
        this._sessionReady = false;
        this._clearSessionInitTimer();
        this._clearPendingInputs();
        this._cleaned = false;
        this._sessionReady = false;
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
        this._lastKvEngineEpoch = null;
        this._lastFrameMetrics = {};
        this._inputBackpressureUntil = 0;
        this._inputSequence = 0;
        this._inputAdmissionSequence = 0;
        this._usesInputCommitted = false;
        this._modelState = 'listening';
        this._autoContinuationUsed = false;
        this.chunksSent = 0;
        this.currentSpeakText = '';
        this._speakHandle = null;
        this.paused = false;
        this.pauseState = 'active';
        this.forceListenActive = false;
        this._pendingCancelAcks = 0;
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
        this._sessionReady = false;
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
        this._lastKvEngineEpoch = null;
        this._lastFrameMetrics = {};
        this._modelState = 'listening';
        this._autoContinuationUsed = false;
        // There is no resume token in the current backend protocol: a
        // reconnect creates a new model session rather than continuing the
        // old LLM/TTS KV context. Stop queued old audio and close the visible
        // assistant turn before creating that fresh session.
        this._stopCurrentSpeech();
        this._lastListenTime = 0;
        this._pendingCancelAcks = 0;
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
                    this._sessionReady = false;
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
        this._sessionReady = false;
        const wsUrl = this.config.getWsUrl();
        const ws = new WebSocket(wsUrl);
        this.ws = ws;
        await new Promise((resolve, reject) => {
            const fail = error => {
                clearTimeout(timer);
                if (this._reconnectReject === fail) this._reconnectReject = null;
                reject(error);
            };
            const timer = setTimeout(() => fail(new Error('WebSocket reconnect timeout')), 10_000);
            this._reconnectReject = fail;
            ws.onopen = () => {
                clearTimeout(timer);
                if (this._reconnectReject === fail) this._reconnectReject = null;
                resolve();
            };
            ws.onerror = () => fail(new Error('WebSocket reconnect failed'));
            ws.onclose = () => fail(new Error('WebSocket closed during reconnect'));
        });
        if (this._intentionalClose || this.ws !== ws) throw new Error('Reconnect cancelled by user');

        await new Promise((resolve, reject) => {
            const fail = error => {
                clearTimeout(timer);
                clearTimeout(initTimer);
                if (this._reconnectReject === fail) this._reconnectReject = null;
                reject(error);
            };
            const timer = setTimeout(() => fail(new Error('session.init reconnect timeout')), 30_000);
            let initTimer = null;
            this._reconnectReject = fail;
            ws.onclose = () => fail(new Error('WebSocket closed during session reconnect'));
            ws.onerror = () => fail(new Error('WebSocket failed during session reconnect'));
            let initSent = false;
            const sendInit = () => {
                if (initSent || this._intentionalClose || this.ws !== ws
                    || ws.readyState !== WebSocket.OPEN) return;
                initSent = true;
                ws.send(JSON.stringify({
                    type: 'session.init',
                    payload: { system_prompt: systemPrompt, ...preparePayload },
                }));
            };
            ws.onmessage = event => {
                if (this._intentionalClose || this.ws !== ws) return;
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
                    clearTimeout(initTimer);
                    if (this._reconnectReject === fail) this._reconnectReject = null;
                    this._sessionReady = true;
                    this.sessionId = msg.session_id || '';
                    this._usesInputCommitted = msg.metrics?.backend === 'swift_mlx';
                    this._lastFrameMetrics = msg.metrics || {};
                    this._resetKvBaseline(this._lastFrameMetrics);
                    this.recordingSessionId = this.sessionId;
                    this.onMetrics({ type: 'state', sessionId: this.sessionId });
                    resolve();
                    return;
                }
                if (msg.type === 'error') {
                    fail(new Error(msg.error?.message || msg.error || 'session.init failed'));
                }
            };
            // The Swift backend can emit queue_done, while compatibility
            // gateways accept session.init immediately.
            initTimer = setTimeout(sendInit, 100);
        });

        ws.onmessage = event => {
            try { this._handleMessage(JSON.parse(event.data)); }
            catch (_) { this.onSystemLog('Ignored malformed realtime message'); }
        };
        ws.onclose = () => this._handleTransportClose();
    }

    _handleMessage(msg) {
        const type = msg.type || '';
        if (type === 'input.committed' || msg.metrics?.backend === 'swift_mlx') {
            this._usesInputCommitted = true;
        }
        const legacyCompletion = !this._usesInputCommitted
            && (type === 'response.output_audio.delta' || type === 'response.listen'
                || type === 'response.done'
                || (type === 'response.output.delta'
                    && (msg.kind === 'audio' || msg.kind === 'listen' || msg.kind === 'done'))
                || (type === 'result' && (msg.is_listen || msg.audio_data || msg.end_of_turn)));
        if ((type === 'input.committed' || legacyCompletion) && msg.input_id) {
            this._recentInputs.delete(msg.input_id);
            // A completed input frees one slot in the client window.  The
            // pending packet is sent on the next turn of the event loop so
            // this handler never recursively processes model output.
            this._schedulePendingInputRetry();
            this._updateInputFlowControl();
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

            case 'session.state':
            case 'backend.state':
                this._logProtoEvent('server', type,
                    `reason=${msg.reason || ''}`, msg);
                if (msg.reason === 'interrupt' || msg.reason === 'reset') {
                    const isCancelAck = msg.reason === 'interrupt' && this._awaitingCancelAck;
                    if (isCancelAck) this._pendingCancelAcks--;
                    // Interrupt rollback and reset can shorten KV without a
                    // sliding window. Do not compare the new timeline with
                    // the provisional cache from the cancelled generation.
                    this._lastFrameMetrics = msg.metrics || {};
                    this._resetKvBaseline(msg.metrics || msg);
                    // Local cancellation already cleared the previous queue.
                    // Audio captured while waiting for its ACK belongs to the
                    // fresh timeline and must not be discarded by that ACK.
                    if (!isCancelAck) this._clearPendingInputs();
                    this._stopCurrentSpeech();
                    this._handleListen(msg, { autoContinue: false });
                    this._schedulePendingInputRetry();
                }
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
                if (this.paused || this._awaitingCancelAck || this._intentionalClose) break;
                const retryAfter = Number(msg.retry_after_ms) || 250;
                this._inputBackpressureUntil = Math.max(
                    this._inputBackpressureUntil,
                    performance.now() + Math.max(100, retryAfter));
                const rejectedID = msg.input_id || '';
                if (!rejectedID && this._recentInputs.size > 1) {
                    this._pauseInputTimeline('Input rejection did not identify which audio packet was refused');
                    break;
                }
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
                    if (rejected.sequence < this._lastSubmittedInputSequence) {
                        this._pauseInputTimeline(`Input ${retryID || '?'} was refused after later audio was submitted`);
                        break;
                    }
                    const retryEntry = { ...rejected, retries: rejected.retries + 1 };
                    if (retryEntry.retries <= 8) {
                        if (!this._queuePendingInput(retryEntry)) break;
                        this._schedulePendingInputRetry();
                    } else {
                        this._pauseInputTimeline(`Input ${retryID || '?'} could not be admitted after 8 retries`);
                        break;
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
            this._lastFrameMetrics = { ...this._lastFrameMetrics };
            for (const key of [
                'kv_cache_length', 'wall_clock_ms', 'generate_ms', 'backend',
                'engine_epoch', 'sliding_window_mode',
                'sliding_window_high_watermark', 'sliding_window_low_watermark',
            ]) {
                if (msg[key] !== undefined) this._lastFrameMetrics[key] = msg[key];
            }
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
    _handleListen(msg, { autoContinue = true } = {}) {
        const modelInterrupted = msg.reason === 'model_barge_in'
            || msg.reason === 'model_interrupted';
        // The runtime accepted an interruption during an answer or its tail. Stop
        // its unheard output while retaining the user's new input timeline.
        if (modelInterrupted) this._stopCurrentSpeech();
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

        this._checkKvCache({ ...this._lastFrameMetrics, ...result });
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
        if (autoContinue && !modelInterrupted) this._maybeAutoContinue(completedText);
    }

    _maybeAutoContinue(completedText) {
        const text = taskContinuationForAssistantText(completedText);
        if (!text || !this._sessionReady || this._autoContinuationUsed || this.paused
            || this.forceListenActive || this._awaitingCancelAck || this._intentionalClose) return;
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
        this._submitInput({ event, retries: 0 });
        this._modelState = 'waiting_response';
        this.onAutoContinuation(text);
        this.onSystemLog('Continuing the requested story...');
    }

    /** Handle new protocol response.output_audio.delta */
    _handleSpeak(msg) {
        this._applyFrameMetrics(msg);
        // A listen clock marker is not speech. AudioContext time advances
        // without a source, so queuing this PCM would insert a one-second
        // pause ahead of the next answer and corrupt TTFS/UI statistics.
        if (msg.silent === true || this._intentionalClose || this.paused || this.forceListenActive
            || this._awaitingCancelAck) return;
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
                // Padding describes this packet only. Reusing the previous
                // frame's metrics would trim real speech from a later packet
                // whose optional metrics are absent.
                msg.metrics?.audio_padding_samples,
            );
        }

        const result = {
            is_listen: false,
            text: msg.text || '',
            audio_data: msg.audio,
            end_of_turn: msg.end_of_turn || false,
            kv_cache_length: this._lastFrameMetrics.kv_cache_length,
        };

        this._checkKvCache({ ...this._lastFrameMetrics, ...result });
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
                ...result,
                kv_cache_length: result.kv_cache_length,
            });
        } else {
            this._handleSpeak({
                ...result,
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
        this._checkKvCache(this._lastFrameMetrics);
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

    _resetKvBaseline(metrics = {}) {
        this._lastKvCacheLength = Number.isFinite(metrics.kv_cache_length)
            ? Math.max(0, metrics.kv_cache_length) : 0;
        this._lastKvEngineEpoch = Number.isFinite(metrics.engine_epoch)
            ? metrics.engine_epoch : null;
    }

    _checkKvCache(result) {
        const maxKv = this.config.getMaxKvTokens();
        const curKv = result.kv_cache_length;
        if (Number.isFinite(curKv) && curKv >= 0) {
            const epoch = Number.isFinite(result.engine_epoch) ? result.engine_epoch : null;
            const native = this._usesInputCommitted || result.backend === 'swift_mlx';
            const sameEpoch = epoch !== null
                ? epoch === this._lastKvEngineEpoch
                : !native && this._lastKvEngineEpoch === null;
            // Cache decreases also occur during ordinary rollback. The
            // native backend exposes window mode, watermarks and epoch, but
            // no pruning counter: only a basic-window high-to-low transition
            // in the same generation is sufficient evidence here. Context
            // windows cannot be inferred from a decrease alone.
            const basicWindowPruned = result.sliding_window_mode === 'basic'
                && Number.isFinite(result.sliding_window_high_watermark)
                && Number.isFinite(result.sliding_window_low_watermark)
                && result.sliding_window_high_watermark > result.sliding_window_low_watermark
                && this._lastKvCacheLength >= result.sliding_window_high_watermark
                && curKv <= result.sliding_window_low_watermark
                && curKv < this._lastKvCacheLength && sameEpoch;
            if (curKv >= maxKv) {
                this.onSystemLog(`⚠ KV cache (${curKv.toLocaleString()}) reached limit. Auto-stopping.`);
                setTimeout(() => this.stop(), 0);
            } else if (basicWindowPruned) {
                const prev = this._lastKvCacheLength;
                this.onSystemLog(`✂ KV pruned: ${prev.toLocaleString()} → ${curKv.toLocaleString()}`);
                if (this.config.getStopOnSlidingWindow()) {
                    this.onSystemLog('⚠ Stop-on-sliding-window. Auto-stopping.');
                    setTimeout(() => this.stop(), 0);
                }
            }
            this._lastKvCacheLength = curKv;
            this._lastKvEngineEpoch = epoch;
        }
    }

}
