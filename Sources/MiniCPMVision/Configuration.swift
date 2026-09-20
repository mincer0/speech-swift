import Foundation

public enum MiniCPMVisionComputePrecision: String, Codable, Equatable, Sendable {
    case float32
    case bfloat16
}

/// Errors raised by the native MiniCPM-o vision path.
public enum MiniCPMVisionError: Error, LocalizedError, Sendable {
    case invalidConfiguration(String)
    case invalidInput(String)
    case missingModelFile(String)
    case invalidWeights(String)

    public var errorDescription: String? {
        switch self {
        case .invalidConfiguration(let message):
            return "Invalid MiniCPM vision configuration: \(message)"
        case .invalidInput(let message):
            return "Invalid MiniCPM vision input: \(message)"
        case .missingModelFile(let file):
            return "Missing MiniCPM vision model file: \(file)"
        case .invalidWeights(let message):
            return "Invalid MiniCPM vision weights: \(message)"
        }
    }
}

/// Geometry of the SigLIP tower and MiniCPM resampler.
///
/// The values are the fields used by MiniCPM-o 4.5's converted visual bundle.
/// Keeping this separate from the image processor is intentional: callers can
/// feed already packed tensors to the model without importing CoreGraphics or
/// decoding an image.
public struct MiniCPMVisionConfiguration: Codable, Equatable, Sendable {
    public var visionHiddenSize: Int
    public var visionIntermediateSize: Int
    public var visionLayers: Int
    public var visionHeads: Int
    public var visionChannels: Int
    public var imageSize: Int
    public var patchSize: Int
    public var layerNormEpsilon: Float
    public var queryNum: Int
    public var outputDim: Int
    public var resamplerHeads: Int
    public var maxGridHeight: Int
    public var maxGridWidth: Int
    public var computePrecision: MiniCPMVisionComputePrecision

    public init(
        visionHiddenSize: Int = 1_152,
        visionIntermediateSize: Int = 4_304,
        visionLayers: Int = 27,
        visionHeads: Int = 16,
        visionChannels: Int = 3,
        imageSize: Int = 980,
        patchSize: Int = 14,
        layerNormEpsilon: Float = 1e-6,
        queryNum: Int = 64,
        outputDim: Int = 4_096,
        resamplerHeads: Int? = nil,
        maxGridHeight: Int = 70,
        maxGridWidth: Int = 70,
        computePrecision: MiniCPMVisionComputePrecision = .float32
    ) {
        self.visionHiddenSize = visionHiddenSize
        self.visionIntermediateSize = visionIntermediateSize
        self.visionLayers = visionLayers
        self.visionHeads = visionHeads
        self.visionChannels = visionChannels
        self.imageSize = imageSize
        self.patchSize = patchSize
        self.layerNormEpsilon = layerNormEpsilon
        self.queryNum = queryNum
        self.outputDim = outputDim
        self.resamplerHeads = resamplerHeads ?? max(1, outputDim / 128)
        self.maxGridHeight = maxGridHeight
        self.maxGridWidth = maxGridWidth
        self.computePrecision = computePrecision
    }

    /// Fields in the bundle's nested `vision_config` object.
    private struct VisionConfigJSON: Decodable {
        let hiddenSize: Int?
        let intermediateSize: Int?
        let layers: Int?
        let heads: Int?
        let channels: Int?
        let imageSize: Int?
        let patchSize: Int?
        let layerNormEpsilon: Float?

        enum CodingKeys: String, CodingKey {
            case hiddenSize = "hidden_size"
            case intermediateSize = "intermediate_size"
            case layers = "num_hidden_layers"
            case heads = "num_attention_heads"
            case channels = "num_channels"
            case imageSize = "image_size"
            case patchSize = "patch_size"
            case layerNormEpsilon = "layer_norm_eps"
        }
    }

    private struct BundleJSON: Decodable {
        let visionConfig: VisionConfigJSON?
        let queryNum: Int?
        let hiddenSize: Int?
        let patchSize: Int?
        let computeDType: String?

        enum CodingKeys: String, CodingKey {
            case visionConfig = "vision_config"
            case queryNum = "query_num"
            case hiddenSize = "hidden_size"
            case patchSize = "patch_size"
            case computeDType = "compute_dtype"
        }
    }

    /// Parse either the converted bundle root (`vision_config` nested) or a
    /// standalone SigLIP config. Unknown fields are ignored by design.
    public static func fromBundleConfig(at url: URL) throws -> Self {
        let data: Data
        do {
            data = try Data(contentsOf: url)
        } catch {
            throw MiniCPMVisionError.missingModelFile(url.lastPathComponent)
        }

        let decoder = JSONDecoder()
        let root: BundleJSON
        do {
            root = try decoder.decode(BundleJSON.self, from: data)
        } catch {
            throw MiniCPMVisionError.invalidConfiguration(error.localizedDescription)
        }

        let vision = root.visionConfig
        let computePrecision: MiniCPMVisionComputePrecision
        if let value = root.computeDType {
            guard let decoded = MiniCPMVisionComputePrecision(rawValue: value) else {
                throw MiniCPMVisionError.invalidConfiguration(
                    "unsupported compute_dtype: \(value)"
                )
            }
            computePrecision = decoded
        } else {
            // Legacy bundles already execute in FP32 because the native image
            // processor emits Float32 pixels. Make that behavior explicit.
            computePrecision = .float32
        }
        let config = Self(
            visionHiddenSize: vision?.hiddenSize ?? 1_152,
            visionIntermediateSize: vision?.intermediateSize ?? 4_304,
            visionLayers: vision?.layers ?? 27,
            visionHeads: vision?.heads ?? 16,
            visionChannels: vision?.channels ?? 3,
            imageSize: vision?.imageSize ?? 980,
            patchSize: vision?.patchSize ?? root.patchSize ?? 14,
            layerNormEpsilon: vision?.layerNormEpsilon ?? 1e-6,
            queryNum: root.queryNum ?? 64,
            outputDim: root.hiddenSize ?? 4_096,
            maxGridHeight: (vision?.imageSize ?? 980) / max(vision?.patchSize ?? root.patchSize ?? 14, 1),
            maxGridWidth: (vision?.imageSize ?? 980) / max(vision?.patchSize ?? root.patchSize ?? 14, 1),
            computePrecision: computePrecision
        )
        // A malformed or manually edited bundle should fail before allocating
        // 27 layers of parameters.
        try config.validate()
        return config
    }

    public func validate() throws {
        guard visionHiddenSize > 0,
              visionIntermediateSize > 0,
              visionLayers > 0,
              visionHeads > 0,
              visionHiddenSize.isMultiple(of: visionHeads),
              visionChannels == 3,
              imageSize > 0,
              patchSize > 0,
              imageSize.isMultiple(of: patchSize),
              layerNormEpsilon > 0,
              queryNum > 0,
              outputDim > 0,
              resamplerHeads > 0,
              outputDim.isMultiple(of: resamplerHeads),
              maxGridHeight > 0,
              maxGridWidth > 0
        else {
            throw MiniCPMVisionError.invalidConfiguration(
                "unsupported SigLIP/resampler geometry"
            )
        }
    }

    public var visionPositionCount: Int {
        let side = imageSize / patchSize
        return side * side
    }
}
