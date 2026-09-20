import CoreGraphics
import Foundation
import MLX
import MLXNN
@testable import MiniCPMVision
import XCTest

final class MiniCPMVisionConfigurationTests: XCTestCase {
    func testReadsConvertedBundleConfiguration() throws {
        let json = """
        {
          "model_type": "minicpmo_vision_mlx",
          "vision_config": {
            "hidden_size": 1152,
            "intermediate_size": 4304,
            "num_hidden_layers": 27,
            "num_attention_heads": 16,
            "num_channels": 3,
            "image_size": 980,
            "patch_size": 14,
            "layer_norm_eps": 0.000001
          },
          "query_num": 64,
          "hidden_size": 4096,
          "patch_size": 14,
          "compute_dtype": "float32"
        }
        """
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
            .appendingPathExtension("json")
        try Data(json.utf8).write(to: root)
        defer { try? FileManager.default.removeItem(at: root) }
        let config = try MiniCPMVisionConfiguration.fromBundleConfig(at: root)
        XCTAssertEqual(config.visionHiddenSize, 1_152)
        XCTAssertEqual(config.visionIntermediateSize, 4_304)
        XCTAssertEqual(config.visionLayers, 27)
        XCTAssertEqual(config.visionHeads, 16)
        XCTAssertEqual(config.queryNum, 64)
        XCTAssertEqual(config.outputDim, 4_096)
        XCTAssertEqual(config.computePrecision, .float32)
    }

    func testRejectsUnsupportedComputeDType() throws {
        let json = """
        {
          "vision_config": {},
          "compute_dtype": "float16"
        }
        """
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
            .appendingPathExtension("json")
        try Data(json.utf8).write(to: root)
        defer { try? FileManager.default.removeItem(at: root) }
        XCTAssertThrowsError(try MiniCPMVisionConfiguration.fromBundleConfig(at: root)) { error in
            guard case MiniCPMVisionError.invalidConfiguration(let message) = error else {
                return XCTFail("expected invalidConfiguration, got \(error)")
            }
            XCTAssertTrue(message.contains("unsupported compute_dtype"))
        }
    }

    func testTinyConfigurationValidation() throws {
        let config = MiniCPMVisionConfiguration(
            visionHiddenSize: 8,
            visionIntermediateSize: 16,
            visionLayers: 1,
            visionHeads: 2,
            imageSize: 4,
            patchSize: 2,
            queryNum: 2,
            outputDim: 4,
            resamplerHeads: 1,
            maxGridHeight: 2,
            maxGridWidth: 2
        )
        try config.validate()
        XCTAssertEqual(config.visionPositionCount, 4)
    }
}

final class MiniCPMVisionPackedProcessorTests: XCTestCase {
    private var processor: MiniCPMImageProcessor {
        MiniCPMImageProcessor(configuration: .init(
            maxSliceNums: 9,
            scaleResolution: 448,
            patchSize: 14
        ))
    }

    private func solidImage(width: Int, height: Int) throws -> CGImage {
        let colorSpace = CGColorSpace(name: CGColorSpace.sRGB)
            ?? CGColorSpaceCreateDeviceRGB()
        var pixels = [UInt8](repeating: 0, count: width * height * 4)
        let image: CGImage? = pixels.withUnsafeMutableBytes { rawBuffer in
            guard let context = CGContext(
                data: rawBuffer.baseAddress,
                width: width,
                height: height,
                bitsPerComponent: 8,
                bytesPerRow: width * 4,
                space: colorSpace,
                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
                    | CGBitmapInfo.byteOrder32Big.rawValue
            ) else {
                return nil
            }
            return context.makeImage()
        }
        return try XCTUnwrap(image)
    }

    func testGridMatchesOfficialMiniCPMRuleForWideImage() throws {
        let grid = try XCTUnwrap(
            processor.getSlicedGrid(imageSize: (width: 1_000, height: 500))
        )
        XCTAssertEqual(grid.columns, 2)
        XCTAssertEqual(grid.rows, 1)
    }

    func testRefineSizeAndSliceGeometryMatchOfficialWideImage() throws {
        // Fixed-source oracle (MiniCPMO45/processing_minicpmo.py @
        // ba7fa9cc6ad63c894f1bd5e5afac28466953519d):
        // 1000x500, max_slice_nums=9 -> grid 2x1, overview 630x322,
        // refine image 896x448, then two 448x448 slices in row-major order.
        // This catches using patchSize when rounding the full dimensions in
        // get_refine_size (which produced the incorrect 504x504 geometry).
        let image = try solidImage(width: 1_000, height: 500)
        let groups = try processor.processWithMetadata(
            [image],
            maxSliceNums: 9
        )
        let group = try XCTUnwrap(groups.first)
        XCTAssertEqual(group.overview.metadata.originalSize.width, 1_000)
        XCTAssertEqual(group.overview.metadata.originalSize.height, 500)
        XCTAssertEqual(group.overview.metadata.resizedSize.width, 630)
        XCTAssertEqual(group.overview.metadata.resizedSize.height, 322)
        XCTAssertEqual(group.slices.count, 2)
        XCTAssertEqual(
            group.slices.map { [$0.metadata.resizedSize.width, $0.metadata.resizedSize.height] },
            [[448, 448], [448, 448]]
        )
        XCTAssertEqual(
            group.slices.map { [$0.metadata.targetSize.height, $0.metadata.targetSize.width] },
            [[32, 32], [32, 32]]
        )
    }

    func testPackedPatchOrderMatchesUnfoldLayout() throws {
        let tiny = MiniCPMImageProcessor(configuration: .init(
            patchSize: 2
        ))
        // CHW, 3×4×4. This is the packed order from the reference
        // `reshape_by_patch`: patch rows are grouped before patch-grid rows,
        // patch-grid columns, and patch columns (equivalent to
        // reshape(C,nH,P,nW,P).transpose(C,P,nH,nW,P)).
        let chw = (0..<48).map(Float.init)
        let packed = try tiny.packNormalizedCHW(
            chw,
            width: 4,
            height: 4,
            targetSize: MiniCPMTargetSize(height: 2, width: 2)
        )
        eval(packed.pixels)
        let values = packed.pixels.asArray(Float.self)
        XCTAssertEqual(packed.pixels.shape, [3, 2, 8])
        XCTAssertEqual(
            Array(values[0..<16]),
            [0, 1, 2, 3, 8, 9, 10, 11, 4, 5, 6, 7, 12, 13, 14, 15]
        )
    }

    func testTinyPackedModelProducesGroupedShape() throws {
        let config = MiniCPMVisionConfiguration(
            visionHiddenSize: 8,
            visionIntermediateSize: 16,
            visionLayers: 1,
            visionHeads: 2,
            imageSize: 4,
            patchSize: 2,
            queryNum: 2,
            outputDim: 4,
            resamplerHeads: 1,
            maxGridHeight: 2,
            maxGridWidth: 2
        )
        let model = MiniCPMVisionModel(configuration: config)
        let pixels = MLXArray.zeros([3, 2, 8])
        let input = MiniCPMPackedImage(
            pixels: pixels,
            targetSize: MiniCPMTargetSize(height: 2, width: 2)
        )
        let groups = try model.encode([
            MiniCPMImageGroup(overview: input, slices: [input, input])
        ])
        eval(groups[0].overview, groups[0].slices)
        XCTAssertEqual(groups.count, 1)
        XCTAssertEqual(groups[0].overview.shape, [2, 4])
        XCTAssertEqual(groups[0].slices.count, 2)
        XCTAssertTrue(groups[0].slices.allSatisfy { $0.shape == [2, 4] })
    }

    func testResamplerPositionTableMatchesOfficialFrequencyBase() {
        // get_1d_sincos_pos_embed(embed_dim=4) creates two frequencies and
        // divides by embed_dim / 2 (=2), i.e. 10000^(-i / 2), not by the
        // per-axis output width (=4).  Check column=1 where the second
        // frequency is exactly 0.01 for D=8.
        let table = MiniCPMResampler.makePositionTable(
            height: 1,
            width: 2,
            dimensions: 8
        )
        XCTAssertEqual(table.count, 16)
        let secondColumn = Array(table[8..<16])
        XCTAssertEqual(secondColumn[0], sin(1), accuracy: 1e-6)
        XCTAssertEqual(secondColumn[1], sin(0.01), accuracy: 1e-6)
        XCTAssertEqual(secondColumn[2], cos(1), accuracy: 1e-6)
        XCTAssertEqual(secondColumn[3], cos(0.01), accuracy: 1e-6)
        XCTAssertEqual(secondColumn[4], 0, accuracy: 1e-6)
        XCTAssertEqual(secondColumn[5], 0, accuracy: 1e-6)
        XCTAssertEqual(secondColumn[6], 1, accuracy: 1e-6)
        XCTAssertEqual(secondColumn[7], 1, accuracy: 1e-6)
    }

    func testResamplerExpandsPositionCacheForLargerTargetGrid() throws {
        let config = MiniCPMVisionConfiguration(
            visionHiddenSize: 8,
            visionIntermediateSize: 16,
            visionLayers: 1,
            visionHeads: 2,
            imageSize: 4,
            patchSize: 2,
            queryNum: 2,
            outputDim: 4,
            resamplerHeads: 1,
            maxGridHeight: 1,
            maxGridWidth: 1
        )
        let resampler = MiniCPMResampler(configuration: config)
        let output = try resampler(
            MLXArray.zeros([1, 2, 8]),
            targetSizes: MLXArray([Int32(1), Int32(2)]).expandedDimensions(axis: 0)
        )
        eval(output)
        XCTAssertEqual(output.shape, [1, 2, 4])
    }

    func testPerFrameSliceLimitsPreserveOverviewAndSliceOrder() throws {
        let wide = try XCTUnwrap(
            processor.getSlicedGrid(imageSize: (width: 1_000, height: 500), maxSliceNums: 9)
        )
        let square = try processor.getSlicedGrid(
            imageSize: (width: 224, height: 224), maxSliceNums: 1
        )
        XCTAssertEqual(wide.columns, 2)
        XCTAssertEqual(wide.rows, 1)
        XCTAssertNil(square)

        // The Swift adapter processes per-frame limits independently.  This
        // contract yields frame 0's overview first, then its slices, followed
        // by frame 1's overview; a wide frame cannot change frame 1's budget.
        let frame0 = MiniCPMImageGroup(
            overview: MiniCPMPackedImage(
                pixels: MLXArray.zeros([3, 14, 14 * 32]),
                targetSize: MiniCPMTargetSize(height: 1, width: 32)
            ),
            slices: [
                MiniCPMPackedImage(
                    pixels: MLXArray.zeros([3, 14, 14 * 32]),
                    targetSize: MiniCPMTargetSize(height: 1, width: 32)
                ),
                MiniCPMPackedImage(
                    pixels: MLXArray.zeros([3, 14, 14 * 32]),
                    targetSize: MiniCPMTargetSize(height: 1, width: 32)
                ),
            ]
        )
        let frame1 = MiniCPMImageGroup(
            overview: MiniCPMPackedImage(
                pixels: MLXArray.zeros([3, 14, 14 * 16]),
                targetSize: MiniCPMTargetSize(height: 4, width: 4)
            )
        )
        let groups = [frame0, frame1]
        XCTAssertEqual(groups.count, 2)
        XCTAssertEqual(groups[0].all.count, 3)
        XCTAssertEqual(groups[1].all.count, 1)
        XCTAssertEqual(groups[0].all[0].targetSize, MiniCPMTargetSize(height: 1, width: 32))
        XCTAssertEqual(groups[0].all[1].targetSize, MiniCPMTargetSize(height: 1, width: 32))
        XCTAssertEqual(groups[1].all[0].targetSize, MiniCPMTargetSize(height: 4, width: 4))
        XCTAssertEqual(groups[0].all[0].pixels.shape, [3, 14, 14 * 32])
        XCTAssertEqual(groups[0].all[1].pixels.shape, [3, 14, 14 * 32])
        XCTAssertEqual(groups[1].all[0].pixels.shape, [3, 14, 14 * 16])
    }
}

final class MiniCPMVisionGoldenFixtureTests: XCTestCase {
    func testDecodesPythonManifestTopLevelSnakeCaseKeys() throws {
        let data = Data(
            """
            {
              "schema_version": 1,
              "fixture_kind": "minicpm_o_vision_siglip_resampler_oracle",
              "oracle": {
                "source_root": "fixed-upstream/MiniCPM-o-Demo",
                "git_commit": "0123456789012345678901234567890123456789",
                "processor_file": "processing_minicpmo.py",
                "vision_file": "modeling_navit_siglip.py",
                "resampler_file": "modeling_minicpmo.py",
                "source_file_sha256": {},
                "compute_dtype": "float32",
                "serialized_dtype": "float32-le"
              },
              "source_model": {
                "path": "source-model/MiniCPM-o-4_5",
                "tree_sha256": "model-tree",
                "config_sha256": "config",
                "visual_source_shards": [],
                "vision_config": {},
                "query_num": 1,
                "hidden_size": 1
              },
              "bundle": {
                "path": "mlx-bundle/MiniCPM-o-4_5-vision-mlx",
                "tree_sha256": "bundle-tree",
                "bundle_kind": "mlx_vision"
              },
              "parity": {
                "embedding_shape": [1, 1],
                "recommended_max_abs": 0.5,
                "recommended_mean_abs": 0.1,
                "recommended_cosine": 0.9
              },
              "processor_parity": {
                "recommended_max_abs": 0.2,
                "recommended_mean_abs": 0.02
              },
              "cases": []
            }
            """.utf8
        )
        let manifest = try JSONDecoder().decode(
            MiniCPMVisionGoldenManifest.self,
            from: data
        )
        XCTAssertEqual(manifest.schemaVersion, 1)
        XCTAssertEqual(manifest.fixtureKind, "minicpm_o_vision_siglip_resampler_oracle")
        XCTAssertEqual(manifest.sourceModel.path, "source-model/MiniCPM-o-4_5")
        XCTAssertEqual(manifest.processorParity?.recommendedMaxAbs, 0.2)
    }

    func testFixedSourceGoldenFixtureSchemaIfPresent() throws {
        guard let path = ProcessInfo.processInfo.environment["MINICPM_VISION_GOLDEN"] else {
            throw XCTSkip("set MINICPM_VISION_GOLDEN to validate a generated fixed-source fixture")
        }
        let directory = URL(fileURLWithPath: path, isDirectory: true)
        let manifest = try MiniCPMVisionGoldenManifest.load(at: directory)
        try manifest.validate(on: directory)
        XCTAssertTrue(manifest.cases.contains { $0.kind == "hd_slices" })
        XCTAssertTrue(manifest.cases.contains { $0.kind == "multi_image" })
        XCTAssertTrue(manifest.cases.contains { $0.kind == "video_frames_per_frame_limits" })
    }

    func testOptionalSwiftProcessorMatchesFixedSourcePixels() throws {
        guard let fixturePath = ProcessInfo.processInfo.environment["MINICPM_VISION_GOLDEN"] else {
            throw XCTSkip("set MINICPM_VISION_GOLDEN to validate processor pixel parity")
        }
        let fixtureURL = URL(fileURLWithPath: fixturePath, isDirectory: true)
        let manifest = try MiniCPMVisionGoldenManifest.load(at: fixtureURL)
        try manifest.validate(on: fixtureURL)
        let processor = MiniCPMImageProcessor()
        let maxAbs = Float(manifest.processorParity?.recommendedMaxAbs ?? 0.2)
        let meanAbs = manifest.processorParity?.recommendedMeanAbs ?? 0.02
        var exercised = false

        for goldenCase in manifest.cases {
            guard let inputFiles = goldenCase.inputFiles else { continue }
            exercised = true
            let images = try inputFiles.map {
                try processor.image(from: fixtureURL.appendingPathComponent($0))
            }
            let groups: [MiniCPMProcessedImageGroup]
            if let limits = goldenCase.perFrameMaxSliceNums {
                groups = try processor.processWithMetadata(images, maxSliceNums: limits)
            } else if let limit = goldenCase.maxSliceNums {
                groups = try processor.processWithMetadata(images, maxSliceNums: limit)
            } else {
                throw MiniCPMVisionGoldenManifest.FixtureError.invalid(
                    "processor case has no slice limit: \(goldenCase.name)"
                )
            }
            let actualParts = groups.flatMap { [$0.overview] + $0.slices }
            XCTAssertEqual(
                actualParts.count,
                goldenCase.parts.count,
                "processor part count mismatch in \(goldenCase.name)"
            )
            for (actual, expectedPart) in zip(actualParts, goldenCase.parts) {
                eval(actual.packed.pixels)
                let actualValues = actual.packed.pixels.asArray(Float.self)
                let expectedURL = fixtureURL.appendingPathComponent(expectedPart.pixelsFile)
                let expectedValues = try Self.readFloat32(expectedURL)
                XCTAssertEqual(actualValues.count, expectedValues.count)
                var maximum: Float = 0
                var maximumIndex = 0
                var total: Double = 0
                for (index, pair) in zip(actualValues, expectedValues).enumerated() {
                    let lhs = pair.0
                    let rhs = pair.1
                    let delta = abs(lhs - rhs)
                    if delta > maximum {
                        maximum = delta
                        maximumIndex = index
                    }
                    total += Double(delta)
                }
                let average = total / Double(max(actualValues.count, 1))
                if ProcessInfo.processInfo.environment["MINICPM_VISION_PROCESSOR_DIAG"] == "1" {
                    let pairs = zip(actualValues, expectedValues).enumerated().filter { abs($0.element.0 - $0.element.1) > 0.2 }.prefix(20)
                    let first = pairs.map { index, pair in "\(index):\(pair.0)/\(pair.1)" }
                    print("[Processor diagnostic] \(goldenCase.name)#\(expectedPart.order) count=\(pairs.count) max=\(maximum) index=\(maximumIndex) actual=\(actualValues[maximumIndex]) expected=\(expectedValues[maximumIndex]) mean=\(average) first=\(first)")
                }
                XCTAssertLessThanOrEqual(
                    maximum,
                    maxAbs,
                    "processor max abs mismatch in \(goldenCase.name)#\(expectedPart.order)"
                )
                XCTAssertLessThanOrEqual(
                    average,
                    meanAbs,
                    "processor mean abs mismatch in \(goldenCase.name)#\(expectedPart.order)"
                )
                XCTAssertEqual(actual.metadata.resizedSize.width, expectedPart.resizedSize[0])
                XCTAssertEqual(actual.metadata.resizedSize.height, expectedPart.resizedSize[1])
                XCTAssertEqual(actual.packed.targetSize.height, expectedPart.targetSize[0])
                XCTAssertEqual(actual.packed.targetSize.width, expectedPart.targetSize[1])
            }
        }
        guard exercised else {
            throw XCTSkip("fixture has no processor input_files; regenerate with the fixed exporter")
        }
    }

    func testOptionalSwiftModelMatchesFixedSourceGolden() throws {
        guard ProcessInfo.processInfo.environment["MINICPM_VISION_RUN_MODEL"] == "1",
              let fixturePath = ProcessInfo.processInfo.environment["MINICPM_VISION_GOLDEN"],
              let bundlePath = ProcessInfo.processInfo.environment["MINICPM_VISION_BUNDLE"]
        else {
            throw XCTSkip(
                "set MINICPM_VISION_RUN_MODEL=1, MINICPM_VISION_GOLDEN and MINICPM_VISION_BUNDLE for numeric parity"
            )
        }
        let fixtureURL = URL(fileURLWithPath: fixturePath, isDirectory: true)
        let manifest = try MiniCPMVisionGoldenManifest.load(at: fixtureURL)
        try manifest.validate(on: fixtureURL)
        let model = try MiniCPMVisionModel.fromDirectory(
            URL(fileURLWithPath: bundlePath, isDirectory: true)
        )
        guard let oraclePrecision = MiniCPMVisionComputePrecision(
            rawValue: manifest.oracle.computeDType
        ) else {
            throw MiniCPMVisionGoldenManifest.FixtureError.invalid(
                "unsupported oracle compute dtype: \(manifest.oracle.computeDType)"
            )
        }
        XCTAssertEqual(
            model.configuration.computePrecision,
            oraclePrecision,
            "vision bundle compute_dtype must match the golden oracle"
        )
        guard model.configuration.computePrecision == oraclePrecision else { return }
        for goldenCase in manifest.cases {
            for part in goldenCase.parts {
                let pixelURL = fixtureURL.appendingPathComponent(part.pixelsFile)
                let expectedURL = fixtureURL.appendingPathComponent(part.embeddingFile)
                let pixelValues = try Self.readFloat32(pixelURL)
                let expected = try Self.readFloat32(expectedURL)
                let pixels = MLXArray(pixelValues, part.pixelsShape).asType(
                    oraclePrecision == .float32 ? .float32 : .bfloat16
                )
                let target = MLXArray(
                    [Int32(part.targetSize[0]), Int32(part.targetSize[1])]
                )
                let actual = try model.encode(pixelValues: pixels, targetSizes: target)[0]
                eval(actual)
                let values = actual.asArray(Float.self)
                XCTAssertEqual(values.count, expected.count)
                let pair = zip(values, expected)
                var maxAbs: Float = 0
                var sumAbs: Double = 0
                var dot: Double = 0
                var actualNorm: Double = 0
                var expectedNorm: Double = 0
                for (lhs, rhs) in pair {
                    let delta = abs(lhs - rhs)
                    maxAbs = max(maxAbs, delta)
                    sumAbs += Double(delta)
                    dot += Double(lhs) * Double(rhs)
                    actualNorm += Double(lhs) * Double(lhs)
                    expectedNorm += Double(rhs) * Double(rhs)
                }
                let meanAbs = sumAbs / Double(max(values.count, 1))
                let cosine = dot / max(sqrt(actualNorm * expectedNorm), 1e-12)
                print("[Vision diagnostic] \(goldenCase.name)#\(part.order) max=\(maxAbs) mean=\(meanAbs) cosine=\(cosine)")
                XCTAssertLessThanOrEqual(
                    Double(maxAbs), manifest.parity.recommendedMaxAbs,
                    "max abs mismatch in \(goldenCase.name)#\(part.order)"
                )
                XCTAssertLessThanOrEqual(
                    meanAbs, manifest.parity.recommendedMeanAbs,
                    "mean abs mismatch in \(goldenCase.name)#\(part.order)"
                )
                XCTAssertGreaterThanOrEqual(
                    cosine, manifest.parity.recommendedCosine,
                    "cosine mismatch in \(goldenCase.name)#\(part.order)"
                )
                for key in part.keyValues {
                    XCTAssertLessThan(key.flatIndex, values.count)
                    XCTAssertEqual(
                        values[key.flatIndex], Float(key.value), accuracy: Float(manifest.parity.recommendedMaxAbs)
                    )
                }
            }
        }
    }

    func testDiagnosticStagesForFirstGoldenPart() throws {
        guard let fixturePath = ProcessInfo.processInfo.environment["MINICPM_VISION_GOLDEN"],
              let bundlePath = ProcessInfo.processInfo.environment["MINICPM_VISION_BUNDLE"]
        else {
            throw XCTSkip("set MINICPM_VISION_GOLDEN and MINICPM_VISION_BUNDLE for diagnostics")
        }
        let fixtureURL = URL(fileURLWithPath: fixturePath, isDirectory: true)
        let manifest = try MiniCPMVisionGoldenManifest.load(at: fixtureURL)
        let goldenCase = try XCTUnwrap(manifest.cases.first)
        let part = try XCTUnwrap(goldenCase.parts.first)
        let model = try MiniCPMVisionModel.fromDirectory(
            URL(fileURLWithPath: bundlePath, isDirectory: true)
        )
        let pixels = try Self.readFloat32(fixtureURL.appendingPathComponent(part.pixelsFile))
        let diagnosticDType: DType =
            ProcessInfo.processInfo.environment["MINICPM_VISION_DIAGNOSTIC_FP32"] == "1"
            ? .float32
            : .bfloat16
        let pixelTensor = MLXArray(pixels, part.pixelsShape).asType(diagnosticDType)
        let sizes = MLXArray([Int32(part.targetSize[0]), Int32(part.targetSize[1])])
        let batched = pixelTensor.expandedDimensions(axis: 0)
        let targetBatch = sizes.expandedDimensions(axis: 0)
        let patchInput = batched.transposed(0, 2, 3, 1)
        let patch = model.vision.embeddings.patchEmbedding(patchInput)
            .reshaped([1, part.targetSize[0] * part.targetSize[1], model.configuration.visionHiddenSize])
        let embeddings = try model.vision.embeddings(batched, targetSizes: targetBatch)
        let positional = embeddings - patch
        let firstLayer = model.vision.encoder.layers[0]
        let norm1 = firstLayer.firstNorm(embeddings)
        let q = firstLayer.attention.queryProjection(norm1)
        let k = firstLayer.attention.keyProjection(norm1)
        let v = firstLayer.attention.valueProjection(norm1)
        let heads = model.configuration.visionHeads
        let headDimension = model.configuration.visionHiddenSize / heads
        let length = q.dim(1)
        let qHeads = q.reshaped([1, length, heads, headDimension])
            .transposed(0, 2, 1, 3)
        let kHeads = k.reshaped([1, length, heads, headDimension])
            .transposed(0, 2, 1, 3)
        let vHeads = v.reshaped([1, length, heads, headDimension])
            .transposed(0, 2, 1, 3)
        let scores = matmul(qHeads, kHeads.transposed(0, 1, 3, 2))
            * (1 / sqrt(Float(headDimension)))
        let probabilities = softmax(scores.asType(.float32), axis: -1)
            .asType(q.dtype)
        let context = matmul(probabilities, vHeads)
            .transposed(0, 2, 1, 3)
            .reshaped([1, length, model.configuration.visionHiddenSize])
        let attn = firstLayer.attention(norm1)
        let norm2Input = embeddings + attn
        let norm2 = firstLayer.secondNorm(norm2Input)
        let fc1 = firstLayer.mlp.first(norm2)
        let gelu = geluApproximate(fc1)
        let fc2 = firstLayer.mlp.second(gelu)
        let debugComponents: [(String, MLXArray)] = [
            ("embeddings", embeddings), ("norm1", norm1),
            ("q", q), ("k", k), ("v", v),
            ("scores", scores), ("probabilities", probabilities),
            ("context", context),
            ("attn", attn), ("norm2", norm2), ("fc1", fc1),
            ("gelu", gelu), ("fc2", fc2),
        ]
        for (name, value) in debugComponents {
            eval(value)
            let values = value.asArray(Float.self)
            let indexes = [0, 1, 2, values.count / 2, values.count - 1]
            print("[Vision component diagnostic] \(name) dtype=\(value.dtype) keyActual=\(indexes.map { values[$0] })")
            try compareWithLayerOracle(name: name, actual: values)
        }
        let normValues = norm1.asArray(Float.self)
        let normData = normValues.withUnsafeBufferPointer { Data(buffer: $0) }
        try normData.write(to: URL(fileURLWithPath: "/tmp/swift_norm1.f32"))
        let patchValues = patch.asArray(Float.self)
        let patchData = patchValues.withUnsafeBufferPointer { Data(buffer: $0) }
        try patchData.write(to: URL(fileURLWithPath: "/tmp/swift_patch.f32"))
        let embeddingValues = embeddings.asArray(Float.self)
        let embeddingData = embeddingValues.withUnsafeBufferPointer { Data(buffer: $0) }
        try embeddingData.write(to: URL(fileURLWithPath: "/tmp/swift_embeddings.f32"))
        let qValues = q.asArray(Float.self)
        let qData = qValues.withUnsafeBufferPointer { Data(buffer: $0) }
        try qData.write(to: URL(fileURLWithPath: "/tmp/swift_q.f32"))
        if let expectedNormData = try? Data(contentsOf: URL(fileURLWithPath: "/tmp/q_x.f32")) {
            let expectedNorm = expectedNormData.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
            let qFromExpected = firstLayer.attention.queryProjection(
                MLXArray(expectedNorm, [1, part.targetSize[0] * part.targetSize[1], model.configuration.visionHiddenSize])
            ).asType(.bfloat16)
            eval(qFromExpected)
            let values = qFromExpected.asArray(Float.self)
            let indexes = [0, 1, 2, values.count / 2, values.count - 1]
            print("[Vision q expected-input diagnostic] \(indexes.map { values[$0] })")
        }
        var encoded = embeddings
        let diagnosticLayers: Set<Int> = [0, 1, 2, 3, 4, 5, 10, 15, 20, 25, 26]
        for (index, layer) in model.vision.encoder.layers.enumerated() {
            encoded = layer(encoded)
            eval(encoded)
            try compareWithLayerOracle(
                name: String(format: "layer_%02d", index),
                actual: encoded.asArray(Float.self))
            if diagnosticLayers.contains(index) {
                let values = encoded.asArray(Float.self)
                let indexes = [0, 1, 2, values.count / 2, values.count - 1]
                print("[Vision layer diagnostic] \(index) dtype=\(encoded.dtype) keyActual=\(indexes.map { values[$0] })")
            }
        }
        let towerOutput = model.vision.postLayerNorm(encoded)
        eval(towerOutput)
        try compareWithLayerOracle(
            name: "post_layernorm",
            actual: towerOutput.asArray(Float.self))
        let projected = model.resampler.keyValueProjection(towerOutput)
        let output = try model.resampler(towerOutput, targetSizes: targetBatch)
        let stages: [(String, MLXArray)] = [
            ("patch_embedding", patch),
            ("positional", positional),
            ("encoder", encoded),
            ("projector", projected),
            ("final_embedding", output),
        ]
        for (name, value) in stages {
            eval(value)
            let values = value.asArray(Float.self)
            let expected = try XCTUnwrap(part.stageStatistics?[name])
            let actual = expected.keyValues.map { values[$0.flatIndex] }
            let target = expected.keyValues.map { Float($0.value) }
            print("[Vision stage diagnostic] \(name) dtype=\(value.dtype) shape=\(value.shape) keyActual=\(actual) keyExpected=\(target)")
        }
        let expectedOutput = try Self.readFloat32(fixtureURL.appendingPathComponent(part.embeddingFile))
        let outputValues = output.asArray(Float.self)
        let castValues = output.asType(.bfloat16).asArray(Float.self)
        for (label, values) in [("f32", outputValues), ("bf16", castValues)] {
            var maximum: Float = 0
            var total: Double = 0
            var dot: Double = 0
            var lhsNorm: Double = 0
            var rhsNorm: Double = 0
            var maximumIndex = 0
            for (index, pair) in zip(values, expectedOutput).enumerated() {
                let (lhs, rhs) = pair
                let delta = abs(lhs - rhs)
                if delta > maximum {
                    maximum = delta
                    maximumIndex = index
                }
                total += Double(delta)
                dot += Double(lhs) * Double(rhs)
                lhsNorm += Double(lhs) * Double(lhs)
                rhsNorm += Double(rhs) * Double(rhs)
            }
            print("[Vision output diagnostic] \(label) max=\(maximum) index=\(maximumIndex) actual=\(values[maximumIndex]) expected=\(expectedOutput[maximumIndex]) mean=\(total / Double(values.count)) cosine=\(dot / max(sqrt(lhsNorm * rhsNorm), 1e-12))")
        }
    }

    private func compareWithLayerOracle(name: String, actual: [Float]) throws {
        guard let root = ProcessInfo.processInfo.environment["MINICPM_VISION_LAYER_ORACLE"] else {
            return
        }
        let file = URL(fileURLWithPath: root, isDirectory: true)
            .appendingPathComponent("\(name).f32")
        guard FileManager.default.fileExists(atPath: file.path) else {
            print("[Vision oracle diagnostic] \(name) missing")
            return
        }
        let expected = try Self.readFloat32(file)
        guard actual.count == expected.count else {
            XCTFail("Vision oracle count mismatch for \(name): \(actual.count) vs \(expected.count)")
            return
        }
        var maximum: Float = 0
        var maximumIndex = 0
        var total: Double = 0
        var dot: Double = 0
        var actualNorm: Double = 0
        var expectedNorm: Double = 0
        for (index, pair) in zip(actual, expected).enumerated() {
            let (lhs, rhs) = pair
            let delta = abs(lhs - rhs)
            if delta > maximum {
                maximum = delta
                maximumIndex = index
            }
            total += Double(delta)
            dot += Double(lhs) * Double(rhs)
            actualNorm += Double(lhs) * Double(lhs)
            expectedNorm += Double(rhs) * Double(rhs)
        }
        let mean = total / Double(max(actual.count, 1))
        let cosine = dot / max(sqrt(actualNorm * expectedNorm), 1e-12)
        print(
            "[Vision oracle diagnostic] \(name) max=\(maximum) index=\(maximumIndex) "
                + "actual=\(actual[maximumIndex]) expected=\(expected[maximumIndex]) "
                + "mean=\(mean) cosine=\(cosine)"
        )
    }

    private static func readFloat32(_ url: URL) throws -> [Float] {
        let data = try Data(contentsOf: url)
        guard data.count.isMultiple(of: MemoryLayout<Float>.size) else {
            throw MiniCPMVisionGoldenManifest.FixtureError.invalid("unaligned float32 file: \(url.lastPathComponent)")
        }
        return data.withUnsafeBytes { rawBuffer in
            Array(rawBuffer.bindMemory(to: Float.self))
        }
    }
}

final class MiniCPMVisionWeightMappingTests: XCTestCase {
    func testAliasesAndPackedResamplerQKVAreCanonicalized() {
        let q = MLXArray.zeros([4, 4])
        let packed = MLXArray.zeros([12, 4])
        let raw: [String: MLXArray] = [
            "vpm.embeddings.patch_embedding.weight": MLXArray.zeros([8, 3, 2, 2]),
            "resampler.attn.in_proj_weight": packed,
            "resampler.attn.in_proj_bias": MLXArray.zeros([12]),
            "resampler.attn.out_proj.weight": q,
        ]
        let sanitized = MiniCPMVisionWeightLoader.sanitize(raw)
        XCTAssertNotNil(sanitized["vision.embeddings.patch_embedding.weight"])
        XCTAssertEqual(sanitized["vision.embeddings.patch_embedding.weight"]?.shape, [8, 2, 2, 3])
        XCTAssertEqual(sanitized["resampler.attn.q_proj.weight"]?.shape, [4, 4])
        XCTAssertEqual(sanitized["resampler.attn.k_proj.weight"]?.shape, [4, 4])
        XCTAssertEqual(sanitized["resampler.attn.v_proj.weight"]?.shape, [4, 4])
        XCTAssertEqual(sanitized["resampler.attn.q_proj.bias"]?.shape, [4])
    }
}
