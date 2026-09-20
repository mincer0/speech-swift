import Foundation
import CryptoKit

/// JSON contract emitted by `scripts/minicpm_vision_golden.py`.
///
/// The fixture intentionally stores packed processor inputs and float32
/// embeddings as separate binary files.  This keeps the manifest readable and
/// lets a future parity test feed each image part independently without
/// importing NumPy or depending on a particular image decoder.
struct MiniCPMVisionGoldenManifest: Decodable {
    let schemaVersion: Int
    let fixtureKind: String
    let oracle: Oracle
    let sourceModel: SourceModel
    let bundle: Bundle
    let parity: Parity
    let processorParity: ProcessorParity?
    let cases: [GoldenCase]

    enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case fixtureKind = "fixture_kind"
        case oracle
        case sourceModel = "source_model"
        case bundle
        case parity
        case processorParity = "processor_parity"
        case cases
    }

    struct Oracle: Decodable {
        let sourceRoot: String
        let gitCommit: String
        let processorFile: String
        let visionFile: String
        let resamplerFile: String
        let sourceFileSHA256: [String: String]
        let computeDType: String
        let serializedDType: String

        enum CodingKeys: String, CodingKey {
            case sourceRoot = "source_root"
            case gitCommit = "git_commit"
            case processorFile = "processor_file"
            case visionFile = "vision_file"
            case resamplerFile = "resampler_file"
            case sourceFileSHA256 = "source_file_sha256"
            case computeDType = "compute_dtype"
            case serializedDType = "serialized_dtype"
        }
    }

    struct SourceModel: Decodable {
        let path: String
        let treeSHA256: String
        let configSHA256: String
        let visualSourceShards: [String]
        let visionConfig: [String: JSONValue]
        let queryNum: Int
        let hiddenSize: Int

        enum CodingKeys: String, CodingKey {
            case path
            case treeSHA256 = "tree_sha256"
            case configSHA256 = "config_sha256"
            case visualSourceShards = "visual_source_shards"
            case visionConfig = "vision_config"
            case queryNum = "query_num"
            case hiddenSize = "hidden_size"
        }
    }

    struct Bundle: Decodable {
        let path: String
        let treeSHA256: String
        let bundleKind: String

        enum CodingKeys: String, CodingKey {
            case path
            case treeSHA256 = "tree_sha256"
            case bundleKind = "bundle_kind"
        }
    }

    struct Parity: Decodable {
        let embeddingShape: [Int]
        let recommendedMaxAbs: Double
        let recommendedMeanAbs: Double
        let recommendedCosine: Double

        enum CodingKeys: String, CodingKey {
            case embeddingShape = "embedding_shape"
            case recommendedMaxAbs = "recommended_max_abs"
            case recommendedMeanAbs = "recommended_mean_abs"
            case recommendedCosine = "recommended_cosine"
        }
    }

    struct ProcessorParity: Decodable {
        let recommendedMaxAbs: Double
        let recommendedMeanAbs: Double

        enum CodingKeys: String, CodingKey {
            case recommendedMaxAbs = "recommended_max_abs"
            case recommendedMeanAbs = "recommended_mean_abs"
        }
    }

    struct GoldenCase: Decodable {
        let name: String
        let kind: String
        let inputImageSizes: [[Int]]
        let inputFiles: [String]?
        let maxSliceNums: Int?
        let perFrameMaxSliceNums: [Int]?
        let groups: [GoldenGroup]
        let partCount: Int
        let parts: [GoldenPart]

        enum CodingKeys: String, CodingKey {
            case name
            case kind
            case inputImageSizes = "input_image_sizes"
            case inputFiles = "input_files"
            case maxSliceNums = "max_slice_nums"
            case perFrameMaxSliceNums = "per_frame_max_slice_nums"
            case groups
            case partCount = "part_count"
            case parts
        }
    }

    struct GoldenGroup: Decodable {
        let sourceIndex: Int
        let frameIndex: Int?
        let originalSize: [Int]
        let maxSliceNums: Int
        let grid: [Int]?
        let partStart: Int
        let partCount: Int

        enum CodingKeys: String, CodingKey {
            case sourceIndex = "source_index"
            case frameIndex = "frame_index"
            case originalSize = "original_size"
            case maxSliceNums = "max_slice_nums"
            case grid
            case partStart = "part_start"
            case partCount = "part_count"
        }
    }

    struct GoldenPart: Decodable {
        let order: Int
        let sourceIndex: Int
        let frameIndex: Int?
        let role: String
        let originalSize: [Int]
        let resizedSize: [Int]
        let targetSize: [Int]
        let pixelsFile: String
        let pixelsShape: [Int]
        let pixelsDType: String
        let pixelsSHA256: String
        let embeddingFile: String
        let embeddingShape: [Int]
        let embeddingDType: String
        let embeddingSHA256: String
        let keyValues: [GoldenKeyValue]
        /// Optional in schema v1 so older fixtures remain readable. New
        /// fixtures include compact diagnostics for every vision stage.
        let stageStatistics: [String: StageStatistics]?

        enum CodingKeys: String, CodingKey {
            case order
            case sourceIndex = "source_index"
            case frameIndex = "frame_index"
            case role
            case originalSize = "original_size"
            case resizedSize = "resized_size"
            case targetSize = "target_size"
            case pixelsFile = "pixels_file"
            case pixelsShape = "pixels_shape"
            case pixelsDType = "pixels_dtype"
            case pixelsSHA256 = "pixels_sha256"
            case embeddingFile = "embedding_file"
            case embeddingShape = "embedding_shape"
            case embeddingDType = "embedding_dtype"
            case embeddingSHA256 = "embedding_sha256"
            case keyValues = "key_values"
            case stageStatistics = "stage_statistics"
        }
    }

    struct StageStatistics: Decodable {
        // These summaries make the fixture fail closed and localize drift
        // during oracle regeneration. The current Swift model gate compares
        // only final embeddings; it does not expose a public per-stage API.
        let shape: [Int]
        let min: Double
        let max: Double
        let maxAbs: Double
        let mean: Double
        let meanAbs: Double
        let cosine: Double
        let cosineReference: String?
        let keyValues: [GoldenKeyValue]

        enum CodingKeys: String, CodingKey {
            case shape
            case min
            case max
            case maxAbs = "max_abs"
            case mean
            case meanAbs = "mean_abs"
            case cosine
            case cosineReference = "cosine_reference"
            case keyValues = "key_values"
        }
    }

    struct GoldenKeyValue: Decodable {
        let flatIndex: Int
        let value: Double

        enum CodingKeys: String, CodingKey {
            case flatIndex = "flat_index"
            case value
        }
    }

    /// Minimal representation for unknown nested config fields.  The config
    /// is retained for provenance but is not interpreted by the schema gate.
    enum JSONValue: Decodable {
        case string(String)
        case number(Double)
        case bool(Bool)
        case object([String: JSONValue])
        case array([JSONValue])
        case null

        init(from decoder: Decoder) throws {
            let container = try decoder.singleValueContainer()
            if container.decodeNil() { self = .null }
            else if let value = try? container.decode(String.self) { self = .string(value) }
            else if let value = try? container.decode(Double.self) { self = .number(value) }
            else if let value = try? container.decode(Bool.self) { self = .bool(value) }
            else if let value = try? container.decode([String: JSONValue].self) { self = .object(value) }
            else { self = .array(try container.decode([JSONValue].self)) }
        }
    }
}

extension MiniCPMVisionGoldenManifest {
    static func load(at directory: URL) throws -> Self {
        let manifestURL = directory.appendingPathComponent("manifest.json")
        let data = try Data(contentsOf: manifestURL)
        return try JSONDecoder().decode(Self.self, from: data)
    }

    /// Validate ordering/shape/path invariants without loading a model.  This
    /// is intentionally safe to run in CI where the 1GB vision bundle is not
    /// present; the numerical gate can be enabled later with the fixture and
    /// model environment variables.
    func validate(on disk: URL) throws {
        guard schemaVersion == 1 else { throw FixtureError.invalid("schema_version") }
        guard fixtureKind == "minicpm_o_vision_siglip_resampler_oracle" else {
            throw FixtureError.invalid("fixture_kind")
        }
        guard oracle.gitCommit.count == 40,
              Self.isStableProvenance(oracle.sourceRoot),
              oracle.processorFile.hasSuffix("processing_minicpmo.py"),
              oracle.visionFile.hasSuffix("modeling_navit_siglip.py"),
              oracle.resamplerFile.hasSuffix("modeling_minicpmo.py"),
              oracle.computeDType == "float32" || oracle.computeDType == "bfloat16",
              oracle.serializedDType == "float32-le"
        else { throw FixtureError.invalid("oracle provenance") }
        guard Self.isStableProvenance(sourceModel.path),
              Self.isStableProvenance(bundle.path)
        else { throw FixtureError.invalid("absolute provenance path") }
        guard sourceModel.queryNum > 0,
              sourceModel.hiddenSize > 0,
              parity.embeddingShape == [sourceModel.queryNum, sourceModel.hiddenSize]
        else { throw FixtureError.invalid("embedding dimensions") }
        if let processorParity {
            guard processorParity.recommendedMaxAbs > 0,
                  processorParity.recommendedMeanAbs > 0,
                  processorParity.recommendedMeanAbs <= processorParity.recommendedMaxAbs
            else { throw FixtureError.invalid("processor parity thresholds") }
        }
        guard !bundle.treeSHA256.isEmpty,
              !sourceModel.treeSHA256.isEmpty,
              !sourceModel.configSHA256.isEmpty
        else { throw FixtureError.invalid("bundle/model hash") }

        for goldenCase in cases {
            if let inputFiles = goldenCase.inputFiles {
                guard inputFiles.count == goldenCase.inputImageSizes.count,
                      inputFiles.allSatisfy({ !$0.isEmpty })
                else { throw FixtureError.invalid("input image accounting: \(goldenCase.name)") }
                for inputFile in inputFiles {
                    guard FileManager.default.fileExists(
                        atPath: disk.appendingPathComponent(inputFile).path)
                    else { throw FixtureError.invalid("missing input image: \(goldenCase.name)/\(inputFile)") }
                }
            }
            guard goldenCase.partCount == goldenCase.parts.count,
                  !goldenCase.parts.isEmpty,
                  goldenCase.groups.reduce(0) { $0 + $1.partCount } == goldenCase.partCount
            else { throw FixtureError.invalid("part accounting: \(goldenCase.name)") }
            var expectedOrder = 0
            for group in goldenCase.groups {
                guard group.partStart >= 0,
                      group.partCount > 0,
                      group.sourceIndex >= 0,
                      group.sourceIndex < goldenCase.inputImageSizes.count,
                      group.partStart + group.partCount <= goldenCase.parts.count,
                      group.originalSize.count == 2,
                      group.maxSliceNums > 0
                else { throw FixtureError.invalid("group geometry: \(goldenCase.name)") }
                for localIndex in 0..<group.partCount {
                    let part = goldenCase.parts[group.partStart + localIndex]
                    guard part.order == expectedOrder,
                          part.sourceIndex == group.sourceIndex,
                          part.role == (localIndex == 0 ? "overview" : "slice"),
                          part.targetSize.count == 2,
                          part.pixelsShape.count == 3,
                          part.pixelsShape[0] == 3,
                          part.pixelsShape[1] > 0,
                          part.pixelsShape[2] == part.targetSize[0] * part.targetSize[1] * part.pixelsShape[1],
                          part.embeddingShape == parity.embeddingShape,
                          part.pixelsDType == "float32-le",
                          part.embeddingDType == "float32-le",
                          part.pixelsSHA256.count == 64,
                          part.embeddingSHA256.count == 64,
                          !part.keyValues.isEmpty,
                          part.keyValues.allSatisfy({ key in
                              key.flatIndex >= 0
                                  && key.flatIndex < part.embeddingShape.reduce(1, *)
                              && key.value.isFinite
                          })
                    else { throw FixtureError.invalid("part contract: \(goldenCase.name)#\(expectedOrder)") }
                    if let stageStatistics = part.stageStatistics {
                        try Self.validateStageStatistics(
                            stageStatistics,
                            part: part,
                            caseName: goldenCase.name
                        )
                    }
                    let pixelsURL = disk.appendingPathComponent(part.pixelsFile)
                    let embeddingURL = disk.appendingPathComponent(part.embeddingFile)
                    guard FileManager.default.fileExists(atPath: pixelsURL.path),
                          FileManager.default.fileExists(atPath: embeddingURL.path)
                    else { throw FixtureError.invalid("missing binary part: \(goldenCase.name)#\(expectedOrder)") }
                    let pixelBytes = try Data(contentsOf: pixelsURL)
                    let embeddingBytes = try Data(contentsOf: embeddingURL)
                    guard pixelBytes.count == part.pixelsShape.reduce(1, *).multipliedReportingOverflow(by: 4).partialValue,
                          embeddingBytes.count == part.embeddingShape.reduce(1, *).multipliedReportingOverflow(by: 4).partialValue
                    else { throw FixtureError.invalid("binary shape: \(goldenCase.name)#\(expectedOrder)") }
                    guard Self.sha256(pixelBytes) == part.pixelsSHA256,
                          Self.sha256(embeddingBytes) == part.embeddingSHA256
                    else { throw FixtureError.invalid("binary hash: \(goldenCase.name)#\(expectedOrder)") }
                    expectedOrder += 1
                }
            }
        }
    }

    enum FixtureError: Error, LocalizedError {
        case invalid(String)

        var errorDescription: String? {
            switch self {
            case .invalid(let message): return "Invalid MiniCPM vision golden fixture: \(message)"
            }
        }
    }

    private static func sha256(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    private static let requiredStageNames: Set<String> = [
        "processor",
        "patch_embedding",
        "positional",
        "encoder",
        "projector",
        "final_embedding",
    ]

    private static func isStableProvenance(_ value: String) -> Bool {
        guard !value.isEmpty,
              !value.hasPrefix("/"),
              !value.hasPrefix("~"),
              !value.contains("://")
        else { return false }
        // Reject Windows drive-qualified absolute paths as well, even though
        // fixtures are normally generated on macOS.
        return value.range(of: #"^[A-Za-z]:[\\/]"#, options: .regularExpression) == nil
    }

    private static func validateStageStatistics(
        _ stages: [String: StageStatistics],
        part: GoldenPart,
        caseName: String
    ) throws {
        guard Set(stages.keys) == requiredStageNames else {
            throw FixtureError.invalid("stage names: \(caseName)#\(part.order)")
        }
        let elementCount = part.embeddingShape.reduce(1, *)
        for name in requiredStageNames {
            guard let stage = stages[name],
                  !stage.shape.isEmpty,
                  stage.shape.allSatisfy({ $0 > 0 }),
                  stage.min.isFinite,
                  stage.max.isFinite,
                  stage.maxAbs.isFinite,
                  stage.mean.isFinite,
                  stage.meanAbs.isFinite,
                  stage.cosine.isFinite,
                  stage.maxAbs >= 0,
                  stage.meanAbs >= 0,
                  stage.meanAbs <= stage.maxAbs + 1e-9,
                  stage.cosine >= -1.0 - 1e-9,
                  stage.cosine <= 1.0 + 1e-9,
                  !stage.keyValues.isEmpty,
                  stage.keyValues.allSatisfy({ key in
                      key.flatIndex >= 0
                          && key.flatIndex < stage.shape.reduce(1, *)
                          && key.value.isFinite
                  })
            else {
                throw FixtureError.invalid("stage statistics: \(caseName)#\(part.order)/\(name)")
            }
            // A final-stage diagnostic should describe the same logical
            // embedding even though it retains an explicit batch dimension.
            if name == "final_embedding" {
                let shape = stages[name]?.shape ?? []
                guard shape.count == 3,
                      shape[0] == 1,
                      Array(shape.dropFirst()) == part.embeddingShape,
                      shape.reduce(1, *) == elementCount
                else {
                    throw FixtureError.invalid("final stage shape: \(caseName)#\(part.order)")
                }
            }
        }
    }
}
