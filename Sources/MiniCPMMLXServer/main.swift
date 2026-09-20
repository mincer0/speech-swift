import Foundation
import Hummingbird
import HummingbirdCore
import HummingbirdWebSocket
import NIOCore
import MiniCPMDemoBackend
import MiniCPMDuplexRuntime
import MiniCPMToken2Wav
import SpeechVAD
import AudioCommon

/// Standalone Swift transport for the Torch-free MiniCPM MLX backend.
///
/// Production startup loads the split MiniCPM-o MLX bundles and creates one
/// runtime/session factory per WebSocket conversation.  The deterministic
/// no-op engine is deliberately opt-in (`--fake`) so a missing or misspelled
/// checkpoint can never look like a healthy model server.
@main
struct MiniCPMMLXServer {
    struct Options: Sendable {
        var host = "127.0.0.1"
        var port = 7860
        var showHelp = false
        var fake = false
        var modelDirectory: String?
        var audioBundle: String?
        var token2WavBundle: String?
        var voiceReference: String?
        var maxConcurrent = 1
        var maxQueueSize = 1_000
        var queueTimeoutMS: Int? = 120_000
        var dataDirectory: String?
        var presetRoot: String?
        var upstreamRoot: String?
        /// Optional local CoreML Silero bundle. When omitted, the standard
        /// Silero CoreML Hugging Face cache is used (and populated on first
        /// startup). The provider is only attached to half-duplex sessions;
        /// full/omni duplex keeps the model's native listen/speak policy.
        var vadModelDirectory: String?
        /// Disable the default half-duplex Silero provider. In this mode a
        /// half-duplex client must send an explicit utterance_end marker.
        var disableVAD = false

        init(arguments: [String] = Array(CommandLine.arguments.dropFirst())) {
            var index = 0
            while index < arguments.count {
                switch arguments[index] {
                case "--host" where index + 1 < arguments.count:
                    host = arguments[index + 1]
                    index += 2
                case "--port" where index + 1 < arguments.count:
                    if let value = Int(arguments[index + 1]), (1...65_535).contains(value) {
                        port = value
                    }
                    index += 2
                case "--model-dir" where index + 1 < arguments.count:
                    modelDirectory = arguments[index + 1]
                    index += 2
                case "--audio-bundle" where index + 1 < arguments.count:
                    audioBundle = arguments[index + 1]
                    index += 2
                case "--token2wav-bundle" where index + 1 < arguments.count:
                    token2WavBundle = arguments[index + 1]
                    index += 2
                case "--voice-reference" where index + 1 < arguments.count:
                    voiceReference = arguments[index + 1]
                    index += 2
                case "--max-concurrent" where index + 1 < arguments.count:
                    if let value = Int(arguments[index + 1]), value > 0 { maxConcurrent = value }
                    index += 2
                case "--max-queue-size" where index + 1 < arguments.count:
                    if let value = Int(arguments[index + 1]), value >= 0 { maxQueueSize = value }
                    index += 2
                case "--queue-timeout-ms" where index + 1 < arguments.count:
                    if let value = Int(arguments[index + 1]), value >= 0 { queueTimeoutMS = value }
                    index += 2
                case "--data-dir" where index + 1 < arguments.count:
                    dataDirectory = arguments[index + 1]
                    index += 2
                case "--preset-root" where index + 1 < arguments.count:
                    presetRoot = arguments[index + 1]
                    index += 2
                case "--upstream-root" where index + 1 < arguments.count:
                    upstreamRoot = arguments[index + 1]
                    index += 2
                case "--vad-model-dir" where index + 1 < arguments.count:
                    vadModelDirectory = arguments[index + 1]
                    index += 2
                case "--disable-vad":
                    disableVAD = true
                    index += 1
                case "--fake":
                    fake = true
                    index += 1
                case "--help", "-h":
                    showHelp = true
                    index = arguments.count
                default:
                    index += 1
                }
            }
        }
    }

    static func main() async {
        do {
            try await run(options: Options())
        } catch {
            let message = "minicpm-mlx-server: \(error.localizedDescription)\n"
            FileHandle.standardError.write(Data(message.utf8))
            // A model-load/configuration error must be observable by launch
            // scripts and supervisors.  Returning normally would report exit
            // status 0 and make a failed production startup look healthy.
            Foundation.exit(EXIT_FAILURE)
        }
    }

    static func run(options: Options, backend: MiniCPMDemoBackend? = nil) async throws {
        if options.showHelp {
            print("Usage: minicpm-mlx-server --model-dir DIR [options]")
            print("  --host HOST                 Bind address (default 127.0.0.1)")
            print("  --port PORT                 Bind port (default 7860)")
            print("  --model-dir DIR             MiniCPM MLX bundle root (required in production)")
            print("  --audio-bundle DIR          Converted MiniCPM audio bundle")
            print("  --token2wav-bundle DIR      Converted CosyVoice/HiFT bundle (also contains native voice-prompt sidecars)")
            print("  --voice-reference PATH      PCM16 WAV/raw reference (required for production audio)")
            print("  --max-concurrent N          Maximum active MLX sessions (default 1)")
            print("  --max-queue-size N          Maximum queued sessions (default 1000; 0 rejects queueing)")
            print("  --queue-timeout-ms MS       FIFO queue timeout (default 120000; 0 disables)")
            print("  --data-dir DIR              Persistent Gateway data directory (default ./data)")
            print("  --preset-root DIR           Upstream project or assets/presets root")
            print("  --upstream-root DIR         Pinned MiniCPM-o-Demo checkout (static + presets)")
            print("  --vad-model-dir DIR         CoreML Silero VAD bundle/cache (default standard qwen3-speech cache)")
            print("  --disable-vad               Disable half-duplex Silero VAD; require explicit utterance_end")
            print("  --fake                      Use protocol-only no-op engine")
            return
        }
        let dataRoot = URL(
            fileURLWithPath: options.dataDirectory
                ?? FileManager.default.currentDirectoryPath + "/data",
            isDirectory: true)
        let detectedUpstream = options.upstreamRoot
            ?? (FileManager.default.fileExists(atPath: "/tmp/MiniCPM-o-Demo") ? "/tmp/MiniCPM-o-Demo" : nil)
        let upstreamRoot = detectedUpstream.map { URL(fileURLWithPath: $0, isDirectory: true) }
        let presetRoot = options.presetRoot.map { URL(fileURLWithPath: $0, isDirectory: true) }
        let services = try MiniCPMDemoServiceContainer(
            dataDirectory: dataRoot,
            upstreamRoot: upstreamRoot,
            presetRoot: presetRoot)
        let initialETA = await services.config.estimate(requestType: "chat")
        let scheduler = MiniCPMDemoInferenceScheduler(
            maxConcurrent: options.maxConcurrent,
            maxQueueSize: options.maxQueueSize,
            serviceEstimateMS: initialETA * 1_000)
        var temporaryModelRoot: URL?
        defer {
            if let temporaryModelRoot {
                try? FileManager.default.removeItem(at: temporaryModelRoot)
            }
        }
        // Protocol-only/factory-injected launches must not trigger a VAD
        // download. Production MiniCPM sessions get the default CoreML
        // Silero factory below; full/omni still never call it.
        let vadFactory: MiniCPMStreamingVADFactory?
        if options.fake || backend != nil {
            vadFactory = nil
        } else {
            vadFactory = try await makeVADFactory(options: options)
        }
        let makeBackend: @Sendable () -> MiniCPMDemoBackend
        let healthBackend: MiniCPMDemoBackend
        if options.fake {
            // An injected backend is useful for transport tests; otherwise
            // `--fake` is the only path that may construct the no-op engine.
            if let backend {
                makeBackend = { backend }
                healthBackend = backend
            } else {
                makeBackend = { MiniCPMDemoBackend(engine: MiniCPMDemoNoopEngine()) }
                healthBackend = makeBackend()
            }
        } else if let backend {
            // Embedding applications can supply their own already-loaded
            // backend.  This is explicit dependency injection, not a silent
            // fallback: command-line production startup still loads weights.
            makeBackend = { backend }
            healthBackend = backend
        } else {
            let production = try makeProductionBackend(options: options, vadFactory: vadFactory)
            makeBackend = production.makeBackend
            healthBackend = makeBackend()
            temporaryModelRoot = production.temporaryRoot
        }
        let registry = MiniCPMDemoBackendRegistry(scheduler: scheduler)
        let router = Router(options: [.autoGenerateHeadEndpoints])
        router.get("/health") { _, _ in
            var health = await healthBackend.health()
            let queue = await scheduler.snapshot()
            health["active_workers"] = .number(Double(queue.active))
            health["total_workers"] = .number(Double(queue.maxConcurrent))
            health["queue_length"] = .number(Double(queue.queued))
            health["max_queue_size"] = .number(Double(await scheduler.maxQueueSize))
            let body = try JSONEncoder().encode(health)
            return Response(
                status: .ok,
                headers: [.contentType: "application/json"],
                body: .init(byteBuffer: .init(data: body)))
        }
        registerMiniCPMDemoRealtimeRoutes(router, scheduler: scheduler, registry: registry)
        registerMiniCPMDemoGatewayDataRoutes(
            router, services: services, scheduler: scheduler, registry: registry)
        registerMiniCPMDemoStaticRoutes(router, services: services)
        router.get("/queue") { _, _ in
            try await miniCPMDemoQueueResponse(scheduler: scheduler)
        }
        router.post("/sessions/:session_id/close") { request, context in
            let sessionID = try context.parameters.require("session_id")
            var reason = "client_closed"
            let body = try await request.body.collect(upTo: 64 * 1024)
            if body.readableBytes > 0 {
                let data = Data(buffer: body)
                if let payload = try? JSONDecoder().decode([String: MiniCPMJSONValue].self, from: data),
                   let requestedReason = payload["reason"]?.stringValue,
                   !requestedReason.isEmpty {
                    reason = requestedReason
                }
            }
            guard await registry.close(sessionID: sessionID, reason: reason) else {
                throw HTTPError(.notFound, message: "unknown MiniCPM backend session")
            }
            let response = MiniCPMJSONValue.object([
                "ok": .bool(true),
                "session_id": .string(sessionID),
                "closed": .bool(true),
            ])
            let responseData = try JSONEncoder().encode(response)
            return Response(
                status: .ok,
                headers: [.contentType: "application/json"],
                body: .init(byteBuffer: .init(data: responseData)))
        }

        // Official turn-based clients may carry a high-resolution MP4 in one
        // JSON frame. Keep the transport cap aligned with the backend schema
        // (128 MiB); the handler still rejects malformed media before MLX.
        let websocketConfiguration = WebSocketServerConfiguration(
            maxFrameSize: 128 * 1024 * 1024,
            autoPing: .disabled)
        let websocketServer = makeMiniCPMDemoWebSocketServer(
            configuration: websocketConfiguration,
            makeBackend: makeBackend,
            registry: registry,
            scheduler: scheduler,
            queueTimeoutMS: options.queueTimeoutMS,
            emitQueueEvents: true,
            recordingStore: services.sessions)
        let app = Application(
            router: router,
            server: websocketServer,
            configuration: .init(address: .hostname(options.host, port: options.port)))
        try await app.run()
    }

    private struct ProductionBackend {
        let makeBackend: @Sendable () -> MiniCPMDemoBackend
        let temporaryRoot: URL?
    }

    private final class NativeModelsBox: @unchecked Sendable {
        let models: MiniCPMNativeModels

        init(_ models: MiniCPMNativeModels) {
            self.models = models
        }
    }

    /// Build a framework-neutral per-session VAD factory. The model bundle on
    /// disk is shared, but every factory invocation loads a fresh CoreML model
    /// object and therefore owns independent Silero LSTM/context state.
    ///
    /// Half-duplex uses this provider by default. `--vad-model-dir` is an
    /// explicit offline bundle/cache override; without it we resolve the
    /// normal AudioCommon/HuggingFace cache and allow the first launch to
    /// populate it. `--disable-vad` is the only way to opt out.
    private static func makeVADFactory(options: Options) async throws -> MiniCPMStreamingVADFactory? {
        if options.disableVAD {
            let line = "[minicpm-mlx] Silero VAD disabled; half-duplex requires explicit utterance_end\n"
            FileHandle.standardError.write(Data(line.utf8))
            return nil
        }

        let explicitCache = options.vadModelDirectory.map { !$0.isEmpty } ?? false
        let cache: URL
        if let path = options.vadModelDirectory, !path.isEmpty {
            cache = URL(fileURLWithPath: path, isDirectory: true).standardizedFileURL
        } else {
            cache = try HuggingFaceDownloader.getCacheDirectory(
                for: SileroVADModel.defaultCoreMLModelId)
        }
        if explicitCache {
            try requireDirectory(cache, label: "VAD model bundle")
        }
        // Validate one model during startup. Subsequent session factories use
        // the same local cache; an explicit path is treated as offline while
        // the default cache may be populated by the normal downloader.
        let offlineMode = explicitCache
        _ = try await SileroVADModel.fromPretrained(
            modelId: SileroVADModel.defaultCoreMLModelId,
            engine: .coreml,
            cacheDir: cache,
            offlineMode: offlineMode,
            progressHandler: { fraction, message in
                let percent = Int((fraction * 100).rounded())
                let line = "[minicpm-mlx] VAD \(percent)% \(message)\n"
                FileHandle.standardError.write(Data(line.utf8))
            })
        let location = explicitCache ? "explicit bundle \(cache.path)" : "cache \(cache.path)"
        let readyLine = "[minicpm-mlx] CoreML Silero VAD ready (\(location)); full/omni do not instantiate it\n"
        FileHandle.standardError.write(Data(readyLine.utf8))
        return { [cache, offlineMode] in
            let box = VADLoadBox()
            let semaphore = DispatchSemaphore(value: 0)
            Task.detached(priority: .userInitiated) {
                do {
                    box.model = try await SileroVADModel.fromPretrained(
                        modelId: SileroVADModel.defaultCoreMLModelId,
                        engine: .coreml,
                        cacheDir: cache,
                        offlineMode: offlineMode)
                } catch {
                    box.error = error
                }
                semaphore.signal()
            }
            semaphore.wait()
            guard let model = box.model else {
                // The factory is throwing: a session must not silently fall
                // back to explicit-boundary/no-op behavior after startup
                // validation has succeeded. Preserve the original error so
                // the WebSocket/session layer can report it to the client.
                let detail = box.error?.localizedDescription ?? "unknown VAD load failure"
                let line = "[minicpm-mlx] ERROR: CoreML Silero VAD session load failed: \(detail); rejecting half-duplex session\n"
                FileHandle.standardError.write(Data(line.utf8))
                throw box.error ?? MiniCPMDemoBackendError.engine(
                    "CoreML Silero VAD session load failed: \(detail)")
            }
            return MiniCPMSileroStreamingVADProvider(model: model)
        }
    }

    private final class VADLoadBox: @unchecked Sendable {
        var model: SileroVADModel?
        var error: Error?
    }

    private final class MiniCPMSileroStreamingVADProvider: MiniCPMStreamingVADProvider {
        private let processor: StreamingVADProcessor

        init(model: SileroVADModel) {
            self.processor = StreamingVADProcessor(model: model, config: .sileroDefault)
        }

        func process(samples: [Float], sampleRate: Int) -> MiniCPMStreamingVADEvents {
            let normalized = sampleRate == SileroVADModel.sampleRate
                ? samples
                : AudioFileLoader.resample(samples, from: sampleRate, to: SileroVADModel.sampleRate)
            return Self.map(processor.process(samples: normalized))
        }

        func flush(sampleRate: Int) -> MiniCPMStreamingVADEvents {
            Self.map(processor.flush())
        }

        func reset() { processor.reset() }

        private static func map(_ events: [VADEvent]) -> MiniCPMStreamingVADEvents {
            MiniCPMStreamingVADEvents(
                speechStarted: events.contains {
                    if case .speechStarted = $0 { return true }
                    return false
                },
                speechEnded: events.contains {
                    if case .speechEnded = $0 { return true }
                    return false
                })
        }
    }

    /// Resolve and load the split MiniCPM-o MLX checkpoint set.  The model
    /// loader accepts one root with canonical component names; when callers
    /// provide an explicit audio or Token2Wav bundle we create a short-lived
    /// symlink farm instead of copying multi-gigabyte weights.
    private static func makeProductionBackend(
        options: Options,
        vadFactory: MiniCPMStreamingVADFactory?
    ) throws -> ProductionBackend {
        guard let modelDirectory = options.modelDirectory, !modelDirectory.isEmpty else {
            throw MiniCPMDemoBackendError.engine(
                "--model-dir is required for production MiniCPM MLX startup; pass --fake for protocol smoke tests")
        }
        let root = URL(fileURLWithPath: modelDirectory, isDirectory: true).standardizedFileURL
        try requireDirectory(root, label: "model root")

        // Token2Wav has no meaningful zero-voice fallback.  Until a fixed
        // prompt fixture or CAM++/S3Tokenizer flags are supplied, production
        // audio requires a real reference and must fail closed otherwise.
        guard let voiceReference = options.voiceReference, !voiceReference.isEmpty else {
            throw MiniCPMDemoBackendError.engine(
                "--voice-reference is required for production audio output (provide a PCM16 WAV or raw PCM16 file)")
        }
        let voiceURL = URL(fileURLWithPath: voiceReference).standardizedFileURL
        try requireFile(voiceURL, label: "voice reference")
        let voice = try loadVoiceReference(voiceURL)

        let inferred = MiniCPMNativeModelDirectories.infer(from: root)
        let customAudio = try optionalDirectory(options.audioBundle, label: "audio bundle")
        let customToken2Wav = try optionalDirectory(options.token2WavBundle, label: "Token2Wav bundle")
        // Dynamic Token2Wav voice prompts are produced by the converted
        // S3Tokenizer + CAM++ sidecars, not by the flow/HiFT weights alone.
        // Resolve them from the effective bundle before constructing the
        // temporary symlink farm.  Passing the original absolute URLs is
        // intentional: the symlink farm is short-lived and these assets can
        // be loaded directly by CoreML/MLX without copying hundreds of MiB.
        let effectiveToken2Wav = customToken2Wav ?? inferred.token2wav
        let promptAssets = try MiniCPMNativePromptEncoder.discoverAssets(
            in: effectiveToken2Wav)
        let hasOverrides = customAudio != nil || customToken2Wav != nil
        let loadRoot: URL
        let temporaryRoot: URL?
        if hasOverrides {
            let staging = FileManager.default.temporaryDirectory
                .appendingPathComponent("minicpm-mlx-\(UUID().uuidString)", isDirectory: true)
            try FileManager.default.createDirectory(at: staging, withIntermediateDirectories: true)
            do {
                let audio = customAudio ?? inferred.audio
                let token2wav = effectiveToken2Wav
                try link(inferred.llm, in: staging, as: "MiniCPM-o-4_5-llm-mlx-8bit")
                try link(audio, in: staging, as: "MiniCPM-o-4_5-audio-mlx")
                try link(inferred.vision, in: staging, as: "MiniCPM-o-4_5-vision-mlx")
                try link(inferred.tts, in: staging, as: "MiniCPM-o-4_5-tts-mlx")
                try link(token2wav, in: staging, as: "MiniCPM-o-4_5-token2wav-mlx")
            } catch {
                try? FileManager.default.removeItem(at: staging)
                throw error
            }
            loadRoot = staging
            temporaryRoot = staging
        } else {
            loadRoot = root
            temporaryRoot = nil
        }

        let models = try MiniCPMNativeModels.load(
            from: loadRoot,
            loadVision: true,
            loadTTS: true,
            speechTokenizerWeightsURL: promptAssets.speechTokenizerWeights,
            camPlusPlusModelURL: promptAssets.camPlusPlusModel,
            progressHandler: { fraction, message in
                let percent = Int((fraction * 100).rounded())
                let line = "[minicpm-mlx] \(percent)% \(message)\n"
                FileHandle.standardError.write(Data(line.utf8))
            })
        // Keep immutable weights in one box, but create a fresh runtime,
        // adapter, and session identifier for every WebSocket upgrade.
        let modelBox = NativeModelsBox(models)
        return ProductionBackend(
            makeBackend: {
                let runtime = MiniCPMDuplexRuntime(engineFactory: { modelBox.models.makeEngine() })
                let adapter = MiniCPMDuplexRuntimeEngineAdapter(
                    runtime: runtime,
                    sessionID: "backend_\(UUID().uuidString.replacingOccurrences(of: "-", with: "").prefix(12))",
                    defaultVoicePCM: voice.samples,
                    defaultVoiceSampleRate: voice.sampleRate,
                    // Turn/half-duplex use an independent ChatSession over
                    // the same immutable model weights. Full/omni remain on
                    // the DuplexRuntime path inside the adapter.
                    chatEngine: MiniCPMNativeChatEngineAdapter(
                        models: modelBox.models,
                        vadFactory: vadFactory))
                return MiniCPMDemoBackend(engine: adapter)
            },
            temporaryRoot: temporaryRoot)
    }

    private static func requireDirectory(_ url: URL, label: String) throws {
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: url.path, isDirectory: &isDirectory), isDirectory.boolValue else {
            throw MiniCPMDemoBackendError.engine("\(label) does not exist or is not a directory: \(url.path)")
        }
    }

    private static func requireFile(_ url: URL, label: String) throws {
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: url.path, isDirectory: &isDirectory), !isDirectory.boolValue else {
            throw MiniCPMDemoBackendError.engine("\(label) does not exist or is a directory: \(url.path)")
        }
    }

    private static func optionalDirectory(_ path: String?, label: String) throws -> URL? {
        guard let path, !path.isEmpty else { return nil }
        let url = URL(fileURLWithPath: path, isDirectory: true).standardizedFileURL
        try requireDirectory(url, label: label)
        return url
    }

    private static func link(_ source: URL, in directory: URL, as name: String) throws {
        try requireDirectory(source, label: "MiniCPM component")
        let destination = directory.appendingPathComponent(name, isDirectory: true)
        try FileManager.default.createSymbolicLink(at: destination, withDestinationURL: source)
    }

    private static func loadVoiceReference(_ url: URL) throws -> (samples: [Float], sampleRate: Int) {
        let data = try Data(contentsOf: url)
        if data.count >= 12,
           data.prefix(4) == Data("RIFF".utf8),
           data[8..<12] == Data("WAVE".utf8) {
            var offset = 12
            var channels = 1
            var sampleRate = 16_000
            var bits = 16
            var format: UInt16 = 1
            var payload: Data.SubSequence?
            while offset + 8 <= data.count {
                let id = String(data: data[offset..<(offset + 4)], encoding: .ascii)
                let size = data[(offset + 4)..<(offset + 8)].withUnsafeBytes {
                    $0.loadUnaligned(as: UInt32.self)
                }
                let start = offset + 8
                let end = start + Int(size)
                guard end >= start, end <= data.count else { break }
                if id == "fmt ", end - start >= 16 {
                    format = data[start..<(start + 2)].withUnsafeBytes {
                        $0.loadUnaligned(as: UInt16.self)
                    }
                    channels = Int(data[(start + 2)..<(start + 4)].withUnsafeBytes {
                        $0.loadUnaligned(as: UInt16.self)
                    })
                    sampleRate = Int(data[(start + 4)..<(start + 8)].withUnsafeBytes {
                        $0.loadUnaligned(as: UInt32.self)
                    })
                    bits = Int(data[(start + 14)..<(start + 16)].withUnsafeBytes {
                        $0.loadUnaligned(as: UInt16.self)
                    })
                } else if id == "data" {
                    payload = data[start..<end]
                    break
                }
                offset = end + (end.isMultiple(of: 2) ? 0 : 1)
            }
            guard format == 1, bits == 16, channels > 0, let payload, payload.count.isMultiple(of: 2) else {
                throw MiniCPMDemoBackendError.engine("voice reference must be PCM16 WAV: \(url.path)")
            }
            let values: [Float] = stride(from: payload.startIndex, to: payload.endIndex - 1, by: 2).map { index in
                let sampleBits = UInt16(payload[index]) | UInt16(payload[index + 1]) << 8
                return Float(Int16(bitPattern: sampleBits)) / 32768.0
            }
            guard channels > 1 else { return (values, sampleRate) }
            let frames = values.count / channels
            return ((0..<frames).map { frame in
                let start = frame * channels
                return values[start..<(start + channels)].reduce(0, +) / Float(channels)
            }, sampleRate)
        }
        guard data.count.isMultiple(of: 2), !data.isEmpty else {
            throw MiniCPMDemoBackendError.engine(
                "raw voice reference must contain little-endian PCM16 samples: \(url.path)")
        }
        let samples = stride(from: 0, to: data.count - 1, by: 2).map { index in
            let sampleBits = UInt16(data[index]) | UInt16(data[index + 1]) << 8
            return Float(Int16(bitPattern: sampleBits)) / 32768.0
        }
        return (samples, 16_000)
    }
}
