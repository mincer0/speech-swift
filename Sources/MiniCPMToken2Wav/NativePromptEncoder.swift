import Foundation
import MLX
import CosyVoiceTTS

/// Native model adapter for dynamic reference prompts.
///
/// The S3Tokenizer V2 ONNX graph is represented by the existing MLX
/// `SpeechTokenizerModel`; its six-block MiniCPM configuration is loaded from
/// the safetensors bundle produced by the offline converter.  CAM++ is kept as
/// a small CoreML wrapper because the 3D-Speaker export already targets the
/// Neural Engine and its 617-node graph is not an MLX-native checkpoint yet.
/// Both paths are Torch/ONNX-Runtime-free at inference time.
public final class MiniCPMNativePromptEncoder: MiniCPMToken2WavPromptEncoder {
    private let speechTokenizer: SpeechTokenizerModel?
    private let camPlusPlus: MiniCPMCoreMLCamPlusPlusEncoder?

    /// Paths for the two native voice-prompt sidecars.  Discovery is kept
    /// separate from model construction so production startup and tests can
    /// validate a bundle without allocating S3Tokenizer/CAM++ weights.
    public struct AssetURLs: Sendable, Equatable {
        public let speechTokenizerWeights: URL
        public let camPlusPlusModel: URL

        public init(
            speechTokenizerWeights: URL,
            camPlusPlusModel: URL
        ) {
            self.speechTokenizerWeights = speechTokenizerWeights
            self.camPlusPlusModel = camPlusPlusModel
        }
    }

    /// Resolve the native prompt sidecars that belong to an MLX Token2Wav
    /// bundle.  The original MiniCPM release contains ONNX graphs under
    /// `assets/token2wav`; those graphs are source assets for the offline
    /// converters, not runtime providers.  We therefore never silently treat
    /// an ONNX path as a loaded encoder: a missing converted sidecar is a
    /// deterministic startup error instead of a prompt that renders silence.
    public static func discover(
        in modelDirectory: URL,
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) throws -> MiniCPMNativePromptEncoder {
        let assets = try discoverAssets(in: modelDirectory, environment: environment)
        return try MiniCPMNativePromptEncoder(
            speechTokenizerWeightsURL: assets.speechTokenizerWeights,
            camPlusPlusModelURL: assets.camPlusPlusModel)
    }

    /// Discover sidecar URLs without loading either model.  S3Tokenizer must
    /// be a regular safetensors file; a compiled CoreML model is represented
    /// by a `.mlmodelc` directory, so directories are valid for CAM++.
    public static func discoverAssets(
        in modelDirectory: URL,
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) throws -> AssetURLs {
        let tokenizerCandidates = [
            environment["MINICPM_SPEECH_TOKENIZER_WEIGHTS"],
            modelDirectory.appendingPathComponent("speech_tokenizer.safetensors").path,
            modelDirectory.appendingPathComponent("s3tokenizer.safetensors").path,
            modelDirectory.appendingPathComponent("speech_tokenizer_v2_25hz.safetensors").path,
        ].compactMap { $0 }.map { URL(fileURLWithPath: $0) }
        let camCandidates = [
            environment["MINICPM_CAMPPLUS_COREML"],
            modelDirectory.appendingPathComponent("MiniCPM-CamPlusPlus.mlmodelc").path,
            modelDirectory.appendingPathComponent("campplus.mlmodelc").path,
            modelDirectory.appendingPathComponent("CamPlusPlus.mlmodelc").path,
            modelDirectory.appendingPathComponent("MiniCPM-CamPlusPlus.mlpackage").path,
            modelDirectory.appendingPathComponent("campplus.mlpackage").path,
            modelDirectory.appendingPathComponent("CamPlusPlus.mlpackage").path,
        ].compactMap { $0 }.map { URL(fileURLWithPath: $0) }

        let tokenizerURL = tokenizerCandidates.first {
            var isDirectory: ObjCBool = false
            return FileManager.default.fileExists(
                atPath: $0.path, isDirectory: &isDirectory) && !isDirectory.boolValue
        }
        let camURL = camCandidates.first {
            FileManager.default.fileExists(atPath: $0.path)
        }
        if tokenizerURL == nil {
            let onnx = modelDirectory
                .appendingPathComponent("assets/token2wav/speech_tokenizer_v2_25hz.onnx")
            if FileManager.default.fileExists(atPath: onnx.path) {
                throw MiniCPMToken2WavPromptPreparationError.missingModelAsset(
                    "S3Tokenizer ONNX exists at \(onnx.path), but no converted "
                    + "speech_tokenizer.safetensors was found; run "
                    + "convert_minicpm_s3tokenizer_onnx_to_safetensors.py")
            }
            throw MiniCPMToken2WavPromptPreparationError.missingModelAsset(
                "no S3Tokenizer safetensors sidecar in \(modelDirectory.path)")
        }
        if camURL == nil {
            let onnx = modelDirectory
                .appendingPathComponent("assets/token2wav/campplus.onnx")
            if FileManager.default.fileExists(atPath: onnx.path) {
                throw MiniCPMToken2WavPromptPreparationError.missingModelAsset(
                    "CAM++ ONNX exists at \(onnx.path), but no compiled "
                    + "MiniCPM-CamPlusPlus.mlmodelc was found; run "
                    + "convert_minicpm_campplus_coreml.py")
            }
            throw MiniCPMToken2WavPromptPreparationError.missingModelAsset(
                "no CAM++ CoreML sidecar in \(modelDirectory.path)")
        }
        // Both URLs are known non-nil after the fail-closed checks above.
        return AssetURLs(
            speechTokenizerWeights: tokenizerURL!,
            camPlusPlusModel: camURL!)
    }

    /// - Parameters:
    ///   - speechTokenizerWeightsURL: safetensors converted from
    ///     `speech_tokenizer_v2_25hz.onnx`. Pass nil only for a test override.
    ///   - camPlusPlusModelURL: compiled `CamPlusPlus.mlmodelc` (or a compiled
    ///     CoreML model URL). Pass nil only when the caller injects a speaker
    ///     embedding for a test.
    public init(
        speechTokenizerWeightsURL: URL? = nil,
        camPlusPlusModelURL: URL? = nil
    ) throws {
        if let speechTokenizerWeightsURL {
            // MiniCPM-o's bundled graph is S3Tokenizer-v2: six blocks,
            // attention LayerNorm eps=1e-6, explicit mel-length masking and
            // affine FSQ quantization.  Keep these semantics scoped to this
            // adapter; CosyVoice-v3 callers continue using the default config.
            let config = SpeechTokenizerConfig.miniCPMV2
            let tokenizer = SpeechTokenizerModel(config: config)
            try CosyVoiceWeightLoader.loadSpeechTokenizer(
                tokenizer, from: speechTokenizerWeightsURL)
            tokenizer.train(false)
            self.speechTokenizer = tokenizer
        } else {
            speechTokenizer = nil
        }

        if let camPlusPlusModelURL {
            self.camPlusPlus = try MiniCPMCoreMLCamPlusPlusEncoder(
                modelURL: camPlusPlusModelURL)
        } else {
            camPlusPlus = nil
        }
    }

    public func encodeSpeechTokens(tokenizerMel: MLXArray) throws -> MLXArray {
        guard let speechTokenizer else {
            throw MiniCPMToken2WavPromptPreparationError.missingSpeechTokenizer
        }
        guard tokenizerMel.ndim == 3, tokenizerMel.dim(1) == 128 else {
            throw MiniCPMToken2WavPromptPreparationError.invalidTokens(
                "S3 tokenizer mel must have shape [1, 128, T]")
        }
        let melLengths = MLXArray([Int32(tokenizerMel.dim(2))])
        let tokens = speechTokenizer.encode(
            mel: tokenizerMel, melLengths: melLengths)
        eval(tokens)
        return tokens.asType(.int32)
    }

    public func encodeSpeakerEmbedding(speakerMel: MLXArray) throws -> MLXArray {
        guard let camPlusPlus else {
            throw MiniCPMToken2WavPromptPreparationError.missingSpeakerEncoder
        }
        return try camPlusPlus.encode(speakerMel: speakerMel)
    }
}

#if canImport(CoreML)
import CoreML

/// CAM++ CoreML adapter.  `MiniCPMCamPlusPlusMelExtractor` owns all audio
/// feature extraction, so this wrapper only marshals `[1,500,80]` into the
/// compiled model and returns the 192-dimensional speaker vector as MLX.
public final class MiniCPMCoreMLCamPlusPlusEncoder {
    private let model: MLModel
    private let inputName: String
    private let outputName: String
    private let inputDataType: MLMultiArrayDataType

    public init(modelURL: URL) throws {
        let configuration = MLModelConfiguration()
        configuration.computeUnits = .all
        // `MLModel(contentsOf:)` accepts compiled `.mlmodelc` bundles.  When a
        // source `.mlpackage` is supplied, compile it once at startup and
        // retain only the loaded model; production bundles use `.mlmodelc` so
        // this branch is normally not taken.
        let loadURL: URL
        if modelURL.pathExtension.lowercased() == "mlpackage" {
            loadURL = try MLModel.compileModel(at: modelURL)
        } else {
            loadURL = modelURL
        }
        model = try MLModel(contentsOf: loadURL, configuration: configuration)
        guard let input = model.modelDescription.inputDescriptionsByName[
            "mel_features"] ?? model.modelDescription.inputDescriptionsByName.values.first,
              let output = model.modelDescription.outputDescriptionsByName[
                  "embedding"] ?? model.modelDescription.outputDescriptionsByName.values.first
        else {
            throw MiniCPMToken2WavPromptPreparationError.missingSpeakerEncoder
        }
        inputName = input.name
        outputName = output.name
        guard input.type == .multiArray,
              let constraint = input.multiArrayConstraint
        else {
            throw MiniCPMToken2WavPromptPreparationError.invalidSpeakerEmbedding(
                "CAM++ input must be a multi-array")
        }
        inputDataType = constraint.dataType
    }

    public func encode(speakerMel: MLXArray) throws -> MLXArray {
        guard speakerMel.shape == [1, 500, 80] else {
            throw MiniCPMToken2WavPromptPreparationError.invalidSpeakerEmbedding(
                "CAM++ mel must have shape [1, 500, 80]")
        }
        let values = speakerMel.asType(.float32).asArray(Float.self)
        let featureArray = try MLMultiArray(
            shape: [1, 500, 80], dataType: inputDataType)
        switch inputDataType {
        case .float16:
            let pointer = featureArray.dataPointer.assumingMemoryBound(to: Float16.self)
            for index in values.indices { pointer[index] = Float16(values[index]) }
        case .float32:
            let pointer = featureArray.dataPointer.assumingMemoryBound(to: Float.self)
            for index in values.indices { pointer[index] = values[index] }
        default:
            throw MiniCPMToken2WavPromptPreparationError.invalidSpeakerEmbedding(
                "CAM++ input dtype is unsupported: \(inputDataType)")
        }
        let provider = try MLDictionaryFeatureProvider(dictionary: [
            inputName: MLFeatureValue(multiArray: featureArray),
        ])
        let result = try model.prediction(from: provider)
        guard let output = result.featureValue(for: outputName)?.multiArrayValue else {
            throw MiniCPMToken2WavPromptPreparationError.missingSpeakerEncoder
        }
        let count = output.count
        guard count == 192 else {
            throw MiniCPMToken2WavPromptPreparationError.invalidSpeakerEmbedding(
                "CAM++ output has \(count) values, expected 192")
        }
        var embedding = [Float](repeating: 0, count: count)
        switch output.dataType {
        case .float16:
            let source = output.dataPointer.assumingMemoryBound(to: Float16.self)
            for i in 0..<count { embedding[i] = Float(source[i]) }
        case .float32:
            let source = output.dataPointer.assumingMemoryBound(to: Float.self)
            for i in 0..<count { embedding[i] = source[i] }
        default:
            throw MiniCPMToken2WavPromptPreparationError.invalidSpeakerEmbedding(
                "CAM++ output dtype is unsupported: \(output.dataType)")
        }
        guard embedding.allSatisfy({ $0.isFinite }) else {
            throw MiniCPMToken2WavPromptPreparationError.invalidSpeakerEmbedding(
                "CAM++ output contains NaN or infinity")
        }
        return MLXArray(embedding, [1, 192])
    }
}
#else
/// The package targets Apple platforms. This fallback keeps the API explicit
/// for non-CoreML builds rather than accidentally using a Torch/ONNX path.
public final class MiniCPMCoreMLCamPlusPlusEncoder {
    public init(modelURL: URL) throws {
        throw MiniCPMToken2WavPromptPreparationError.missingSpeakerEncoder
    }

    public func encode(speakerMel: MLXArray) throws -> MLXArray {
        throw MiniCPMToken2WavPromptPreparationError.missingSpeakerEncoder
    }
}
#endif
