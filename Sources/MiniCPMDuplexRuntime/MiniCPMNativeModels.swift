import Foundation
import MLX
import MiniCPMAudio
import MiniCPMLLM
import MiniCPMTTSSemantic
import MiniCPMToken2Wav
import MiniCPMVision

/// Locations of the independent native MiniCPM-o bundles.
///
/// The official checkpoint is deliberately split into these directories by
/// the conversion scripts.  Keeping the paths explicit makes a missing
/// component fail at startup instead of silently falling back to a random
/// text-only model.
public struct MiniCPMNativeModelDirectories: @unchecked Sendable, Equatable {
    public let root: URL
    public let llm: URL
    public let audio: URL
    public let vision: URL
    public let tts: URL
    public let token2wav: URL

    public init(
        root: URL,
        llm: URL? = nil,
        audio: URL? = nil,
        vision: URL? = nil,
        tts: URL? = nil,
        token2wav: URL? = nil
    ) {
        self.root = root
        self.llm = llm ?? root.appendingPathComponent("MiniCPM-o-4_5-llm-mlx-8bit")
        self.audio = audio ?? root.appendingPathComponent("MiniCPM-o-4_5-audio-mlx")
        self.vision = vision ?? root.appendingPathComponent("MiniCPM-o-4_5-vision-mlx")
        self.tts = tts ?? root.appendingPathComponent("MiniCPM-o-4_5-tts-mlx")
        self.token2wav = token2wav ?? root.appendingPathComponent("MiniCPM-o-4_5-token2wav-mlx")
    }

    /// Resolve the common layout when ``root`` itself is one component (for
    /// example a downloaded `MiniCPM-o-4_5-llm-mlx-8bit` directory).
    public static func infer(from root: URL) -> Self {
        let name = root.lastPathComponent.lowercased()
        if name.contains("llm-mlx") {
            let parent = root.deletingLastPathComponent()
            return Self(root: parent, llm: root)
        }
        if name.contains("audio-mlx") {
            let parent = root.deletingLastPathComponent()
            return Self(root: parent, audio: root)
        }
        if name.contains("vision-mlx") {
            let parent = root.deletingLastPathComponent()
            return Self(root: parent, vision: root)
        }
        if name.contains("tts-mlx") {
            let parent = root.deletingLastPathComponent()
            return Self(root: parent, tts: root)
        }
        if name.contains("token2wav-mlx") {
            let parent = root.deletingLastPathComponent()
            return Self(root: parent, token2wav: root)
        }
        // The original PyTorch model directory is commonly passed as the
        // root argument. Its native siblings live beside it under models/.
        if name == "minicpm-o-4_5" {
            let parent = root.deletingLastPathComponent()
            return Self(root: parent)
        }
        return Self(root: root)
    }
}

/// Optional fixed Token2Wav prompt.  A fixed profile is useful on machines
/// where CAM++/S3Tokenizer conversion is not installed yet.  It is accepted
/// only when all three arrays are present and shape-validated by the pipeline.
public struct MiniCPMNativePromptProfile: @unchecked Sendable {
    public let prompt: MiniCPMToken2WavPrompt

    public init(prompt: MiniCPMToken2WavPrompt) {
        self.prompt = prompt
    }

    public static func load(from url: URL) throws -> Self {
        let file: URL
        var directory: ObjCBool = false
        if FileManager.default.fileExists(atPath: url.path, isDirectory: &directory),
           directory.boolValue {
            let candidates = [
                "prompt.safetensors", "prompt_fixture.safetensors",
                "token2wav_prompt.safetensors",
            ]
            guard let candidate = candidates
                .map({ url.appendingPathComponent($0) })
                .first(where: { FileManager.default.fileExists(atPath: $0.path) })
            else {
                throw MiniCPMNativeModelsError.missingPromptFile(url)
            }
            file = candidate
        } else {
            file = url
        }

        let arrays: [String: MLXArray]
        do {
            arrays = try MLX.loadArrays(url: file)
        } catch {
            throw MiniCPMNativeModelsError.invalidPrompt(
                "cannot load \(file.path): \(error.localizedDescription)")
        }
        func required(_ names: [String]) throws -> MLXArray {
            for name in names {
                if let value = arrays[name] { return value }
            }
            throw MiniCPMNativeModelsError.invalidPrompt(
                "\(file.lastPathComponent) is missing one of \(names.joined(separator: ", "))")
        }

        let promptTokens = try required(["prompt_tokens", "tokens", "flow_prompt_speech_token"])
        let promptMel = try required(["prompt_mel", "mel", "prompt_feat", "prompt_speech_feat"])
        let speaker = try required(["speaker_embedding", "speaker", "prompt_spk_embedding"])
        guard promptTokens.ndim == 2, promptTokens.dim(0) == 1,
              promptTokens.dim(1) > 0,
              speaker.ndim == 2, speaker.shape == [1, 192]
        else {
            throw MiniCPMNativeModelsError.invalidPrompt(
                "fixed prompt has invalid token or speaker shape")
        }
        return Self(prompt: MiniCPMToken2WavPrompt(
            tokens: promptTokens.asType(.int32),
            mel: promptMel.asType(.float32),
            speakerEmbedding: speaker.asType(.float32)))
    }
}

public enum MiniCPMNativeModelsError: Error, LocalizedError, Sendable {
    case missingComponent(String, URL)
    case missingPromptFile(URL)
    case invalidPrompt(String)
    case unsupported(String)
    case loadFailed(String)

    public var errorDescription: String? {
        switch self {
        case .missingComponent(let component, let url):
            return "MiniCPM native \(component) bundle is missing: \(url.path)"
        case .missingPromptFile(let url):
            return "MiniCPM fixed prompt fixture not found under \(url.path)"
        case .invalidPrompt(let message):
            return "Invalid MiniCPM fixed prompt: \(message)"
        case .unsupported(let message):
            return "MiniCPM native runtime unsupported: \(message)"
        case .loadFailed(let message):
            return "MiniCPM native model load failed: \(message)"
        }
    }
}

/// Shared immutable model weights.  Each call to ``makeEngine`` creates a
/// fresh set of session/cache objects; no attention KV or TTS state is shared
/// between conversations.
public final class MiniCPMNativeModels {
    public let directories: MiniCPMNativeModelDirectories
    public let llm: MiniCPMLanguageModel
    public let tokenizer: MiniCPMTokenizer
    public let audio: MiniCPMAudioModel
    public let vision: MiniCPMVisionModel?
    public let ttsSemantic: MiniCPMTTSSemantic?
    public let flow: MiniCPMToken2WavFlowModel?
    public let hift: MiniCPMHiFTGenerator?
    public let promptEncoder: MiniCPMNativePromptEncoder?
    public let fixedPrompt: MiniCPMNativePromptProfile?

    private init(
        directories: MiniCPMNativeModelDirectories,
        llm: MiniCPMLanguageModel,
        tokenizer: MiniCPMTokenizer,
        audio: MiniCPMAudioModel,
        vision: MiniCPMVisionModel?,
        ttsSemantic: MiniCPMTTSSemantic?,
        flow: MiniCPMToken2WavFlowModel?,
        hift: MiniCPMHiFTGenerator?,
        promptEncoder: MiniCPMNativePromptEncoder?,
        fixedPrompt: MiniCPMNativePromptProfile?
    ) {
        self.directories = directories
        self.llm = llm
        self.tokenizer = tokenizer
        self.audio = audio
        self.vision = vision
        self.ttsSemantic = ttsSemantic
        self.flow = flow
        self.hift = hift
        self.promptEncoder = promptEncoder
        self.fixedPrompt = fixedPrompt
    }

    /// Load the native bundles.  ``loadVision`` and ``loadTTS`` are explicit
    /// so an audio-only/text-only diagnostic can avoid allocating unused
    /// weights.  When ``loadTTS`` is true both semantic and Token2Wav bundles
    /// are required; there is no silent zero-voice fallback.
    public static func load(
        from root: URL,
        loadVision: Bool = true,
        loadTTS: Bool = true,
        fixedPromptURL: URL? = nil,
        speechTokenizerWeightsURL: URL? = nil,
        camPlusPlusModelURL: URL? = nil,
        progressHandler: ((Double, String) -> Void)? = nil
    ) throws -> MiniCPMNativeModels {
        try load(
            directories: MiniCPMNativeModelDirectories.infer(from: root),
            loadVision: loadVision,
            loadTTS: loadTTS,
            fixedPromptURL: fixedPromptURL,
            speechTokenizerWeightsURL: speechTokenizerWeightsURL,
            camPlusPlusModelURL: camPlusPlusModelURL,
            progressHandler: progressHandler)
    }

    /// Variant for CLIs and services that expose individual bundle overrides.
    public static func load(
        directories: MiniCPMNativeModelDirectories,
        loadVision: Bool = true,
        loadTTS: Bool = true,
        fixedPromptURL: URL? = nil,
        speechTokenizerWeightsURL: URL? = nil,
        camPlusPlusModelURL: URL? = nil,
        progressHandler: ((Double, String) -> Void)? = nil
    ) throws -> MiniCPMNativeModels {
        try requireDirectory(directories.llm, name: "LLM")
        try requireDirectory(directories.audio, name: "audio")
        if loadVision { try requireDirectory(directories.vision, name: "vision") }
        if loadTTS {
            try requireDirectory(directories.tts, name: "semantic TTS")
            try requireDirectory(directories.token2wav, name: "Token2Wav")
        }

        progressHandler?(0.02, "Loading MiniCPM tokenizer...")
        let tokenizer: MiniCPMTokenizer
        do {
            tokenizer = try MiniCPMTokenizer.loadSynchronously(
                from: tokenizerDirectory(for: directories))
        } catch {
            throw MiniCPMNativeModelsError.loadFailed(
                "tokenizer: \(error.localizedDescription)")
        }

        progressHandler?(0.08, "Loading MiniCPM Qwen3 backbone...")
        let config: MiniCPMMLXConfig
        let llm: MiniCPMLanguageModel
        do {
            config = try MiniCPMMLXConfig.load(from: directories.llm)
            llm = MiniCPMLanguageModel(config: config)
            try MiniCPMWeightLoader.load(model: llm, from: directories.llm) { fraction, message in
                progressHandler?(0.08 + fraction * 0.34, message)
            }
        } catch {
            throw MiniCPMNativeModelsError.loadFailed(
                "LLM: \(error.localizedDescription)")
        }

        progressHandler?(0.45, "Loading MiniCPM audio encoder...")
        let audio: MiniCPMAudioModel
        do {
            audio = try MiniCPMAudioModel.fromDirectory(directories.audio)
        } catch {
            throw MiniCPMNativeModelsError.loadFailed(
                "audio: \(error.localizedDescription)")
        }

        let vision: MiniCPMVisionModel?
        if loadVision {
            do {
                vision = try MiniCPMVisionModel.fromDirectory(directories.vision) { fraction, message in
                    progressHandler?(0.45 + fraction * 0.18, message)
                }
            } catch {
                throw MiniCPMNativeModelsError.loadFailed(
                    "vision: \(error.localizedDescription)")
            }
        } else {
            vision = nil
        }

        let ttsSemantic: MiniCPMTTSSemantic?
        let flow: MiniCPMToken2WavFlowModel?
        let hift: MiniCPMHiFTGenerator?
        if loadTTS {
            do {
                ttsSemantic = try MiniCPMTTSSemantic.fromDirectory(directories.tts)
                flow = try MiniCPMToken2WavFlowModel.fromDirectory(directories.token2wav)
                hift = try MiniCPMHiFTGenerator.fromDirectory(directories.token2wav)
                ttsSemantic?.train(false)
                flow?.train(false)
                hift?.train(false)
            } catch {
                throw MiniCPMNativeModelsError.loadFailed(
                    "TTS/Token2Wav: \(error.localizedDescription)")
            }
        } else {
            ttsSemantic = nil
            flow = nil
            hift = nil
        }

        let fixedPrompt: MiniCPMNativePromptProfile?
        if let fixedPromptURL {
            fixedPrompt = try MiniCPMNativePromptProfile.load(from: fixedPromptURL)
        } else {
            fixedPrompt = nil
        }

        let promptEncoder: MiniCPMNativePromptEncoder?
        if speechTokenizerWeightsURL != nil || camPlusPlusModelURL != nil {
            do {
                promptEncoder = try MiniCPMNativePromptEncoder(
                    speechTokenizerWeightsURL: speechTokenizerWeightsURL,
                    camPlusPlusModelURL: camPlusPlusModelURL)
            } catch {
                throw MiniCPMNativeModelsError.loadFailed(
                    "voice prompt encoder: \(error.localizedDescription)")
            }
        } else {
            promptEncoder = nil
        }

        progressHandler?(1.0, "MiniCPM native model set ready")
        return MiniCPMNativeModels(
            directories: directories,
            llm: llm,
            tokenizer: tokenizer,
            audio: audio,
            vision: vision,
            ttsSemantic: ttsSemantic,
            flow: flow,
            hift: hift,
            promptEncoder: promptEncoder,
            fixedPrompt: fixedPrompt)
    }

    private static func tokenizerDirectory(
        for directories: MiniCPMNativeModelDirectories
    ) -> URL {
        let tokenizerName = "tokenizer.json"
        let direct = directories.llm.appendingPathComponent(tokenizerName)
        if FileManager.default.fileExists(atPath: direct.path) {
            return directories.llm
        }

        // Dense and quantized bundles share exactly the same tokenizer. The
        // BF16 converter intentionally writes weights/config only, so resolve
        // tokenizer assets from the canonical sibling instead of duplicating
        // them into every 16 GB weight directory.
        let sibling = directories.root
            .appendingPathComponent("MiniCPM-o-4_5-llm-mlx-8bit")
        if FileManager.default.fileExists(
            atPath: sibling.appendingPathComponent(tokenizerName).path) {
            return sibling
        }
        return directories.llm
    }

    /// Build a fresh Token2Wav pipeline backed by shared immutable flow/HiFT
    /// weights.  A prompt is mandatory; if neither a fixed profile nor an
    /// encoder is configured this returns nil instead of synthesizing silence.
    public func makeToken2WavPipeline() throws -> MiniCPMToken2WavPipeline? {
        guard let flow, let hift else { return nil }
        let pipeline = MiniCPMToken2WavPipeline(flow: flow, hift: hift)
        if let fixedPrompt {
            try pipeline.prepare(prompt: fixedPrompt.prompt)
        }
        return pipeline
    }

    /// Prepare a per-engine voice prompt from raw PCM.  This method fails
    /// closed when CAM++/S3Tokenizer or a fixed profile is unavailable.
    public func prepareVoicePrompt(
        pcm: [Float],
        sampleRate: Int,
        pipeline: MiniCPMToken2WavPipeline
    ) throws -> MiniCPMToken2WavPromptProfile {
        if fixedPrompt != nil {
            throw MiniCPMNativeModelsError.unsupported(
                "a fixed prompt is already installed; do not replace it with raw PCM")
        }
        guard let promptEncoder else {
            throw MiniCPMNativeModelsError.unsupported(
                "raw voice PCM requires S3Tokenizer and CAM++ adapters")
        }
        do {
            return try pipeline.preparePromptAudio(
                pcm: pcm,
                sampleRate: sampleRate,
                encoder: promptEncoder)
        } catch {
            throw MiniCPMNativeModelsError.loadFailed(
                "voice prompt preparation: \(error.localizedDescription)")
        }
    }

    public func makeEngine() -> MiniCPMNativeDuplexEngine {
        MiniCPMNativeDuplexEngine(models: self)
    }

    private static func requireDirectory(_ url: URL, name: String) throws {
        var directory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: url.path, isDirectory: &directory),
              directory.boolValue
        else {
            throw MiniCPMNativeModelsError.missingComponent(name, url)
        }
    }
}
