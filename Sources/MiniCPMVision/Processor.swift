import Accelerate
import CoreGraphics
import Foundation
import ImageIO
import MLX

/// Image preprocessing settings kept alongside the native MLX visual model.
public struct MiniCPMImageProcessorConfiguration: Sendable {
    public var maxSliceNums: Int
    public var scaleResolution: Int
    public var patchSize: Int
    public var imageFeatureSize: Int
    public var normMean: (Float, Float, Float)
    public var normStd: (Float, Float, Float)
    public var sliceMode: Bool

    public init(
        maxSliceNums: Int = 9,
        scaleResolution: Int = 448,
        patchSize: Int = 14,
        imageFeatureSize: Int = 64,
        normMean: (Float, Float, Float) = (0.5, 0.5, 0.5),
        normStd: (Float, Float, Float) = (0.5, 0.5, 0.5),
        sliceMode: Bool = true
    ) {
        self.maxSliceNums = maxSliceNums
        self.scaleResolution = scaleResolution
        self.patchSize = patchSize
        self.imageFeatureSize = imageFeatureSize
        self.normMean = normMean
        self.normStd = normStd
        self.sliceMode = sliceMode
    }
}

/// Metadata for one packed image. `isOverview` is retained for protocol
/// adapters that build `<image>`/`<slice>` markers from a flat sequence.
public struct MiniCPMPackedImageMetadata: Sendable {
    public let originalSize: (width: Int, height: Int)
    public let resizedSize: (width: Int, height: Int)
    public let targetSize: MiniCPMTargetSize
    public let isOverview: Bool

    public init(
        originalSize: (width: Int, height: Int),
        resizedSize: (width: Int, height: Int),
        targetSize: MiniCPMTargetSize,
        isOverview: Bool
    ) {
        self.originalSize = originalSize
        self.resizedSize = resizedSize
        self.targetSize = targetSize
        self.isOverview = isOverview
    }
}

/// Packed image plus geometry diagnostics produced by the native processor.
public struct MiniCPMProcessedImage {
    public let packed: MiniCPMPackedImage
    public let metadata: MiniCPMPackedImageMetadata

    public init(packed: MiniCPMPackedImage, metadata: MiniCPMPackedImageMetadata) {
        self.packed = packed
        self.metadata = metadata
    }
}

/// Image group with diagnostics for source overview and HD tiles.
public struct MiniCPMProcessedImageGroup {
    public let overview: MiniCPMProcessedImage
    public let slices: [MiniCPMProcessedImage]

    public init(overview: MiniCPMProcessedImage, slices: [MiniCPMProcessedImage] = []) {
        self.overview = overview
        self.slices = slices
    }

    public var modelInput: MiniCPMImageGroup {
        MiniCPMImageGroup(
            overview: overview.packed,
            slices: slices.map(\.packed)
        )
    }
}

/// Native CoreGraphics/ImageIO implementation of MiniCPM-o's image processor.
///
/// It deliberately emits the unusual packed patch layout consumed by the
/// SigLIP tower: `[C, patchSize, patchCount * patchSize]`. This avoids a
/// Python dependency in the production Swift runtime while keeping the source
/// overview and HD slice grouping explicit.
public final class MiniCPMImageProcessor: @unchecked Sendable {
    public let configuration: MiniCPMImageProcessorConfiguration

    public init(configuration: MiniCPMImageProcessorConfiguration = .init()) {
        self.configuration = configuration
    }

    // MARK: Public decoding/processing API

    public func image(from data: Data) throws -> CGImage {
        guard let source = CGImageSourceCreateWithData(data as CFData, nil),
              let image = CGImageSourceCreateImageAtIndex(source, 0, nil)
        else {
            throw MiniCPMVisionError.invalidInput("unable to decode image data")
        }
        return image
    }

    public func image(from url: URL) throws -> CGImage {
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
              let image = CGImageSourceCreateImageAtIndex(source, 0, nil)
        else {
            throw MiniCPMVisionError.invalidInput("unable to decode image at \(url.path)")
        }
        return image
    }

    /// Process one image per group, matching the common Demo frame-list path.
    public func process(
        _ images: [CGImage],
        maxSliceNums: Int? = nil
    ) throws -> MiniCPMImageBatch {
        let groups = try images.map { try processGroup($0, maxSliceNums: maxSliceNums) }
        return MiniCPMImageBatch(groups: groups.map(\.modelInput))
    }

    /// Process a frame batch with an independent slice budget per frame.
    /// Each frame is sliced in isolation and groups are returned in input
    /// order, so a wide frame cannot change its neighbour's marker budget.
    public func process(
        _ images: [CGImage],
        maxSliceNums limits: [Int]
    ) throws -> MiniCPMImageBatch {
        guard limits.count == images.count else {
            throw MiniCPMVisionError.invalidInput(
                "per-frame maxSliceNums count \(limits.count) does not match image count \(images.count)"
            )
        }
        guard limits.allSatisfy({ $0 > 0 }) else {
            throw MiniCPMVisionError.invalidInput("per-frame maxSliceNums values must be positive")
        }
        let groups = try zip(images, limits).map { image, limit in
            try processGroup(image, maxSliceNums: limit)
        }
        return MiniCPMImageBatch(groups: groups.map(\.modelInput))
    }

    /// Process explicit groups when one multimodal unit contains several
    /// independent images. Each inner group keeps its own slice budget.
    public func process(
        groups: [[CGImage]],
        maxSliceNums: Int? = nil
    ) throws -> MiniCPMImageBatch {
        var output: [MiniCPMImageGroup] = []
        output.reserveCapacity(groups.count)
        for group in groups {
            for image in group {
                output.append(try processGroup(image, maxSliceNums: maxSliceNums).modelInput)
            }
        }
        return MiniCPMImageBatch(groups: output)
    }

    /// Diagnostics-preserving variant used by parity tests and protocol code.
    public func processWithMetadata(
        _ images: [CGImage],
        maxSliceNums: Int? = nil
    ) throws -> [MiniCPMProcessedImageGroup] {
        try images.map { try processGroup($0, maxSliceNums: maxSliceNums) }
    }

    /// Metadata-preserving counterpart to the per-frame slice-budget API.
    public func processWithMetadata(
        _ images: [CGImage],
        maxSliceNums limits: [Int]
    ) throws -> [MiniCPMProcessedImageGroup] {
        guard limits.count == images.count else {
            throw MiniCPMVisionError.invalidInput(
                "per-frame maxSliceNums count \(limits.count) does not match image count \(images.count)"
            )
        }
        guard limits.allSatisfy({ $0 > 0 }) else {
            throw MiniCPMVisionError.invalidInput("per-frame maxSliceNums values must be positive")
        }
        return try zip(images, limits).map { image, limit in
            try processGroup(image, maxSliceNums: limit)
        }
    }

    /// Convert a normalized CHW image to MiniCPM's packed patch layout. This
    /// helper is public so a caller with an existing native pixel buffer can
    /// bypass CoreGraphics without reimplementing the unfold ordering.
    public func packNormalizedCHW(
        _ chw: [Float],
        width: Int,
        height: Int,
        targetSize: MiniCPMTargetSize
    ) throws -> MiniCPMPackedImage {
        let channels = 3
        guard width > 0, height > 0,
              width.isMultiple(of: configuration.patchSize),
              height.isMultiple(of: configuration.patchSize),
              targetSize.patchCount == (width / configuration.patchSize) * (height / configuration.patchSize),
              chw.count == channels * width * height
        else {
            throw MiniCPMVisionError.invalidInput(
                "CHW image must be [3,H,W], divisible by patch size and match target grid"
            )
        }
        let patch = configuration.patchSize
        let gridHeight = height / patch
        let gridWidth = width / patch
        let packedWidth = gridHeight * gridWidth * patch
        var packed = [Float](repeating: 0, count: channels * patch * packedWidth)

        for channel in 0..<channels {
            for patchRow in 0..<patch {
                for blockRow in 0..<gridHeight {
                    for blockColumn in 0..<gridWidth {
                        for patchColumn in 0..<patch {
                            let sourceRow = blockRow * patch + patchRow
                            let sourceColumn = blockColumn * patch + patchColumn
                            let source = channel * height * width
                                + sourceRow * width + sourceColumn
                            let destination = channel * patch * packedWidth
                                + patchRow * packedWidth
                                + (blockRow * gridWidth + blockColumn) * patch
                                + patchColumn
                            packed[destination] = chw[source]
                        }
                    }
                }
            }
        }
        return MiniCPMPackedImage(
            pixels: MLXArray(packed, [channels, patch, packedWidth]),
            targetSize: targetSize
        )
    }

    // MARK: Exact MiniCPM grid rules

    public func findBestResize(
        originalSize: (width: Int, height: Int),
        allowUpscale: Bool = false
    ) -> (width: Int, height: Int) {
        findBestResizeExact(
            originalSize: (Double(originalSize.width), Double(originalSize.height)),
            allowUpscale: allowUpscale
        )
    }

    public func getSlicedGrid(
        imageSize: (width: Int, height: Int),
        maxSliceNums: Int? = nil
    ) throws -> (columns: Int, rows: Int)? {
        let limit = maxSliceNums ?? configuration.maxSliceNums
        guard limit > 0, imageSize.width > 0, imageSize.height > 0 else {
            throw MiniCPMVisionError.invalidInput("image and slice limits must be positive")
        }
        let logRatio = log(Double(imageSize.width) / Double(imageSize.height))
        let ratio = Double(imageSize.width * imageSize.height)
            / Double(configuration.scaleResolution * configuration.scaleResolution)
        let multiple = min(Int(ceil(ratio)), limit)
        if multiple <= 1 { return nil }

        var candidates: [Int] = []
        for count in [multiple - 1, multiple, multiple + 1]
            where count != 1 && count <= limit
        {
            candidates.append(count)
        }

        var best = (columns: 1, rows: 1)
        var minimum = Double.infinity
        for count in candidates {
            guard count > 0 else { continue }
            for columns in 1...count where count.isMultiple(of: columns) {
                let rows = count / columns
                let error = abs(logRatio - log(Double(columns) / Double(rows)))
                if error < minimum {
                    minimum = error
                    best = (columns, rows)
                }
            }
        }
        return best
    }

    // MARK: Implementation

    private func processGroup(
        _ image: CGImage,
        maxSliceNums: Int?
    ) throws -> MiniCPMProcessedImageGroup {
        // PIL ``convert("RGB")`` discards alpha without compositing.  Make
        // that conversion explicit before any CoreGraphics resize so a
        // transparent source cannot be premultiplied against black.
        let sourceImage = try opaqueRGBImageIfNeeded(image)
        let original = (width: sourceImage.width, height: sourceImage.height)
        let limit = maxSliceNums ?? configuration.maxSliceNums
        let grid = configuration.sliceMode
            ? try getSlicedGrid(imageSize: original, maxSliceNums: limit)
            : nil
        let sourceSize: (width: Int, height: Int)
        let tileImages: [(CGImage, (width: Int, height: Int))]
        if let grid {
            sourceSize = findBestResize(originalSize: original, allowUpscale: false)
            let source = try resized(sourceImage, width: sourceSize.width, height: sourceSize.height)
            let refined = getRefineSize(originalSize: original, grid: grid)
            let full = try resized(sourceImage, width: refined.width, height: refined.height)
            let tiles = try split(full, grid: grid)
            tileImages = tiles.map { ($0, (width: refined.width / grid.columns, height: refined.height / grid.rows)) }
            let overview = try makeProcessed(
                source,
                originalSize: original,
                resizedSize: sourceSize,
                isOverview: true
            )
            let slices = try tileImages.map { tile, tileSize in
                try makeProcessed(
                    tile,
                    originalSize: original,
                    resizedSize: tileSize,
                    isOverview: false
                )
            }
            return MiniCPMProcessedImageGroup(overview: overview, slices: slices)
        } else {
            sourceSize = findBestResize(originalSize: original, allowUpscale: true)
            let source = try resized(sourceImage, width: sourceSize.width, height: sourceSize.height)
            let overview = try makeProcessed(
                source,
                originalSize: original,
                resizedSize: sourceSize,
                isOverview: true
            )
            return MiniCPMProcessedImageGroup(overview: overview)
        }
    }

    private func makeProcessed(
        _ image: CGImage,
        originalSize: (width: Int, height: Int),
        resizedSize: (width: Int, height: Int),
        isOverview: Bool
    ) throws -> MiniCPMProcessedImage {
        let dimensions = try rgbaPixels(image)
        let patch = configuration.patchSize
        guard dimensions.width.isMultiple(of: patch),
              dimensions.height.isMultiple(of: patch)
        else {
            throw MiniCPMVisionError.invalidInput(
                "resized image \(dimensions.width)x\(dimensions.height) is not patch divisible"
            )
        }
        let target = MiniCPMTargetSize(
            height: dimensions.height / patch,
            width: dimensions.width / patch
        )
        let chw = normalize(
            rgba: dimensions.pixels,
            width: dimensions.width,
            height: dimensions.height
        )
        let packed = try packNormalizedCHW(
            chw,
            width: dimensions.width,
            height: dimensions.height,
            targetSize: target
        )
        return MiniCPMProcessedImage(
            packed: packed,
            metadata: MiniCPMPackedImageMetadata(
                originalSize: originalSize,
                resizedSize: resizedSize,
                targetSize: target,
                isOverview: isOverview
            )
        )
    }

    private func getRefineSize(
        originalSize: (width: Int, height: Int),
        grid: (columns: Int, rows: Int)
    ) -> (width: Int, height: Int) {
        // The reference processor first rounds the *full* source dimensions
        // to a multiple of the grid count (not to patchSize), then derives the
        // fractional per-tile geometry.  Passing the already divided value to
        // ensureDivide(_:patchSize:) looks equivalent for many even inputs,
        // but changes the result whenever the grid count and patch size differ
        // (for example 1000x500 with a 2x1 grid: 896x448 vs 504x504).
        let refineWidth = ensureDivide(
            Double(originalSize.width),
            grid.columns
        )
        let refineHeight = ensureDivide(
            Double(originalSize.height),
            grid.rows
        )
        // Keep the fractional per-tile geometry until the official resize
        // routine applies its ``int``/rounding steps.  Truncating these values
        // before calling ``findBestResize`` changes the grid for odd aspect
        // ratios and diverges from PIL's implementation.
        let best = findBestResizeExact(
            originalSize: (
                Double(refineWidth) / Double(grid.columns),
                Double(refineHeight) / Double(grid.rows)
            ),
            allowUpscale: true
        )
        return (best.width * grid.columns, best.height * grid.rows)
    }

    private func split(
        _ image: CGImage,
        grid: (columns: Int, rows: Int)
    ) throws -> [CGImage] {
        let tileWidth = image.width / grid.columns
        let tileHeight = image.height / grid.rows
        guard tileWidth > 0, tileHeight > 0 else {
            throw MiniCPMVisionError.invalidInput("slice grid is larger than resized image")
        }
        var result: [CGImage] = []
        result.reserveCapacity(grid.columns * grid.rows)
        for row in 0..<grid.rows {
            for column in 0..<grid.columns {
                let rect = CGRect(
                    x: column * tileWidth,
                    y: row * tileHeight,
                    width: tileWidth,
                    height: tileHeight
                )
                guard let tile = image.cropping(to: rect) else {
                    throw MiniCPMVisionError.invalidInput("unable to crop image slice")
                }
                result.append(tile)
            }
        }
        return result
    }

    private func opaqueRGBImageIfNeeded(_ image: CGImage) throws -> CGImage {
        switch image.alphaInfo {
        case .first, .last, .premultipliedFirst, .premultipliedLast:
            let rgba = try rawProviderPixels(image)
            let colorSpace = CGColorSpace(name: CGColorSpace.sRGB)
                ?? CGColorSpaceCreateDeviceRGB()
            var bytes = rgba.pixels
            for index in stride(from: 3, to: bytes.count, by: 4) {
                bytes[index] = 255
            }
            guard let provider = CGDataProvider(data: Data(bytes) as CFData),
                  let result = CGImage(
                    width: rgba.width,
                    height: rgba.height,
                    bitsPerComponent: 8,
                    bitsPerPixel: 32,
                    bytesPerRow: rgba.width * 4,
                    space: colorSpace,
                    bitmapInfo: CGBitmapInfo(
                        rawValue: CGImageAlphaInfo.noneSkipLast.rawValue
                            | CGBitmapInfo.byteOrder32Big.rawValue
                    ),
                    provider: provider,
                    decode: nil,
                    shouldInterpolate: false,
                    intent: .defaultIntent
                  )
            else {
                throw MiniCPMVisionError.invalidInput("unable to create opaque RGB image")
            }
            return result
        default:
            return image
        }
    }

    /// Return un-premultiplied 8-bit RGBA bytes from an ImageIO provider.
    /// Fixture PNGs use packed RGBA/RGB layouts; callers fall back to the
    /// CGContext path for uncommon source formats.
    private func rawProviderPixels(_ image: CGImage) throws ->
        (pixels: [UInt8], width: Int, height: Int)
    {
        let width = image.width
        let height = image.height
        guard image.bitsPerComponent == 8,
              image.bitsPerPixel == 32,
              image.bytesPerRow >= width * 4,
              let providerData = image.dataProvider?.data,
              let source = CFDataGetBytePtr(providerData),
              CFDataGetLength(providerData) >= image.bytesPerRow * height
        else {
            throw MiniCPMVisionError.invalidInput("unsupported image pixel format")
        }
        var bytes = [UInt8](repeating: 0, count: width * height * 4)
        let alpha = image.alphaInfo
        let alphaFirst = alpha == .first || alpha == .premultipliedFirst
        let alphaLast = alpha == .last || alpha == .premultipliedLast
        let skipFirst = alpha == .noneSkipFirst
        let skipLast = alpha == .noneSkipLast
        guard alphaFirst || alphaLast || skipFirst || skipLast else {
            throw MiniCPMVisionError.invalidInput("unsupported image alpha layout")
        }
        for row in 0..<height {
            let rowBase = row * image.bytesPerRow
            for column in 0..<width {
                let sourceBase = rowBase + column * 4
                let destinationBase = (row * width + column) * 4
                let rgbBase = (alphaFirst || skipFirst) ? sourceBase + 1 : sourceBase
                var red = source[rgbBase]
                var green = source[rgbBase + 1]
                var blue = source[rgbBase + 2]
                let value: UInt8
                if alphaFirst {
                    value = source[sourceBase]
                } else if alphaLast {
                    value = source[sourceBase + 3]
                } else {
                    value = 255
                }
                // CoreGraphics may expose premultiplied channels for a
                // decoded source.  Undo that representation before dropping
                // alpha, matching PIL's channel-preserving conversion.
                if (alpha == .premultipliedFirst || alpha == .premultipliedLast),
                   value > 0,
                   value < 255
                {
                    red = UInt8(min(255, Int((Int(red) * 255 + Int(value) / 2) / Int(value))))
                    green = UInt8(min(255, Int((Int(green) * 255 + Int(value) / 2) / Int(value))))
                    blue = UInt8(min(255, Int((Int(blue) * 255 + Int(value) / 2) / Int(value))))
                }
                bytes[destinationBase] = red
                bytes[destinationBase + 1] = green
                bytes[destinationBase + 2] = blue
                bytes[destinationBase + 3] = value
            }
        }
        return (bytes, width, height)
    }

    private func resized(_ image: CGImage, width: Int, height: Int) throws -> CGImage {
        guard width > 0, height > 0 else {
            throw MiniCPMVisionError.invalidInput("resize dimensions must be positive")
        }
        // CoreGraphics' interpolation kernels are implementation-defined and
        // differ materially from the PIL BICUBIC kernel used by the Python
        // processor.  Materialize RGB bytes first, then apply Pillow's
        // separable 8-bit bicubic filter below.  This also keeps the resize
        // path independent of premultiplied-alpha bitmap contexts.
        let source = try rgbaPixels(image)
        var sourceRGB = [UInt8](repeating: 0, count: source.width * source.height * 3)
        for index in 0..<(source.width * source.height) {
            sourceRGB[index * 3] = source.pixels[index * 4]
            sourceRGB[index * 3 + 1] = source.pixels[index * 4 + 1]
            sourceRGB[index * 3 + 2] = source.pixels[index * 4 + 2]
        }
        let resizedRGB = pillowBicubicResize(
            sourceRGB,
            sourceWidth: source.width,
            sourceHeight: source.height,
            destinationWidth: width,
            destinationHeight: height
        )

        var bytes = [UInt8](repeating: 255, count: width * height * 4)
        for index in 0..<(width * height) {
            bytes[index * 4] = resizedRGB[index * 3]
            bytes[index * 4 + 1] = resizedRGB[index * 3 + 1]
            bytes[index * 4 + 2] = resizedRGB[index * 3 + 2]
        }
        let colorSpace = CGColorSpace(name: CGColorSpace.sRGB)
            ?? CGColorSpaceCreateDeviceRGB()
        guard let provider = CGDataProvider(data: Data(bytes) as CFData),
              let result = CGImage(
                  width: width,
                  height: height,
                  bitsPerComponent: 8,
                  bitsPerPixel: 32,
                  bytesPerRow: width * 4,
                  space: colorSpace,
                  bitmapInfo: CGBitmapInfo(
                      rawValue: CGImageAlphaInfo.noneSkipLast.rawValue
                          | CGBitmapInfo.byteOrder32Big.rawValue
                  ),
                  provider: provider,
                  decode: nil,
                  shouldInterpolate: false,
                  intent: .defaultIntent
              )
        else {
            throw MiniCPMVisionError.invalidInput("unable to materialize resized image")
        }
        return result
    }

    /// Pillow's BICUBIC resize for 8-bit images is separable and quantizes
    /// each axis' coefficients to 22 fractional bits.  Reproducing that
    /// integer path (rather than evaluating a floating-point approximation)
    /// makes generated pixels agree with ``Image.Resampling.BICUBIC`` while
    /// retaining a pure Swift implementation.
    private func pillowBicubicResize(
        _ source: [UInt8],
        sourceWidth: Int,
        sourceHeight: Int,
        destinationWidth: Int,
        destinationHeight: Int
    ) -> [UInt8] {
        let horizontal = makeBicubicCoefficients(input: sourceWidth, output: destinationWidth)
        let vertical = makeBicubicCoefficients(input: sourceHeight, output: destinationHeight)
        let channels = 3

        // Pillow materializes a horizontal 8-bit image before the vertical
        // pass, so retain the same rounding/clipping boundary here.
        var horizontalPixels = [UInt8](repeating: 0, count: destinationWidth * sourceHeight * channels)
        for row in 0..<sourceHeight {
            for column in 0..<destinationWidth {
                let coefficients = horizontal[column]
                for channel in 0..<channels {
                    var sum: Int64 = 1 << 21 // 22-bit fixed-point rounding bias
                    for tap in 0..<coefficients.weights.count {
                        let sourceColumn = coefficients.start + tap
                        let sourceIndex = (row * sourceWidth + sourceColumn) * channels + channel
                        sum += Int64(source[sourceIndex]) * Int64(coefficients.weights[tap])
                    }
                    horizontalPixels[(row * destinationWidth + column) * channels + channel] =
                        clipBicubicByte(sum)
                }
            }
        }

        var output = [UInt8](repeating: 0, count: destinationWidth * destinationHeight * channels)
        for row in 0..<destinationHeight {
            let coefficients = vertical[row]
            for column in 0..<destinationWidth {
                for channel in 0..<channels {
                    var sum: Int64 = 1 << 21
                    for tap in 0..<coefficients.weights.count {
                        let sourceRow = coefficients.start + tap
                        let sourceIndex = (sourceRow * destinationWidth + column) * channels + channel
                        sum += Int64(horizontalPixels[sourceIndex]) * Int64(coefficients.weights[tap])
                    }
                    output[(row * destinationWidth + column) * channels + channel] =
                        clipBicubicByte(sum)
                }
            }
        }
        return output
    }

    private struct BicubicCoefficients {
        let start: Int
        let weights: [Int32]
    }

    private func makeBicubicCoefficients(input: Int, output: Int) -> [BicubicCoefficients] {
        let scale = Double(input) / Double(output)
        let filterScale = max(scale, 1.0)
        let support = 2.0 * filterScale
        var coefficients: [BicubicCoefficients] = []
        coefficients.reserveCapacity(output)
        for outputIndex in 0..<output {
            let center = (Double(outputIndex) + 0.5) * scale
            var start = Int(center - support + 0.5)
            start = max(start, 0)
            var end = Int(center + support + 0.5)
            end = min(end, input)
            let count = max(end - start, 0)
            var floating = [Double](repeating: 0, count: count)
            var total = 0.0
            for tap in 0..<count {
                let distance = (Double(tap + start) - center + 0.5) / filterScale
                let weight = bicubicKernel(distance)
                floating[tap] = weight
                total += weight
            }
            if total != 0 {
                for tap in floating.indices {
                    floating[tap] /= total
                }
            }
            let fixed = floating.map { weight -> Int32 in
                let scaled = weight * Double(1 << 22)
                // C's cast to int truncates toward zero after the +/- 0.5
                // adjustment used by Pillow's normalize_coeffs_8bpc().
                return Int32(weight < 0 ? Int(scaled - 0.5) : Int(scaled + 0.5))
            }
            coefficients.append(BicubicCoefficients(start: start, weights: fixed))
        }
        return coefficients
    }

    private func bicubicKernel(_ value: Double) -> Double {
        let x = abs(value)
        if x < 1.0 {
            return ((1.5 * x - 2.5) * x) * x + 1.0
        }
        if x < 2.0 {
            return (((x - 5.0) * x + 8.0) * x - 4.0) * -0.5
        }
        return 0.0
    }

    private func clipBicubicByte(_ fixedPoint: Int64) -> UInt8 {
        let value = fixedPoint >> 22
        if value <= 0 { return 0 }
        if value >= 255 { return 255 }
        return UInt8(value)
    }

    private func rgbaPixels(_ image: CGImage) throws -> (pixels: [UInt8], width: Int, height: Int) {
        let width = image.width
        let height = image.height
        let colorSpace = CGColorSpace(name: CGColorSpace.sRGB)
            ?? CGColorSpaceCreateDeviceRGB()
        var bytes = [UInt8](repeating: 0, count: width * height * 4)
        let success = bytes.withUnsafeMutableBytes { rawBuffer -> Bool in
            guard let context = CGContext(
                data: rawBuffer.baseAddress,
                width: width,
                height: height,
                bitsPerComponent: 8,
                bytesPerRow: width * 4,
                space: colorSpace,
                // Match PIL ``convert("RGB")`` semantics; do not premultiply
                // RGB by an input alpha channel before normalization.
                bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue
                    | CGBitmapInfo.byteOrder32Big.rawValue
            ) else { return false }
            context.setBlendMode(.copy)
            context.translateBy(x: 0, y: CGFloat(height))
            context.scaleBy(x: 1, y: -1)
            context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
            return true
        }
        guard success else {
            throw MiniCPMVisionError.invalidInput("unable to create image buffer")
        }
        return (bytes, width, height)
    }

    private func normalize(
        rgba: [UInt8],
        width: Int,
        height: Int
    ) -> [Float] {
        let means = [configuration.normMean.0, configuration.normMean.1, configuration.normMean.2]
        let stds = [configuration.normStd.0, configuration.normStd.1, configuration.normStd.2]
        var chw = [Float](repeating: 0, count: 3 * width * height)
        for channel in 0..<3 {
            for row in 0..<height {
                for column in 0..<width {
                    let rgbaIndex = (row * width + column) * 4 + channel
                    let value = Float(rgba[rgbaIndex]) / 255
                    chw[channel * width * height + row * width + column] =
                        (value - means[channel]) / stds[channel]
                }
            }
        }
        return chw
    }

    /// Python's round is ties-to-even; Swift's default rounded() is ties-away.
    private func findBestResizeExact(
        originalSize: (width: Double, height: Double),
        allowUpscale: Bool
    ) -> (width: Int, height: Int) {
        var width = originalSize.width
        var height = originalSize.height
        let area = width * height
        if area > Double(configuration.scaleResolution * configuration.scaleResolution)
            || allowUpscale
        {
            let ratio = width / max(height, 1)
            // This deliberately mirrors the reference Python implementation:
            // height is truncated first, then width is truncated from that
            // integer height before patch rounding.
            height = Double(Int(Double(configuration.scaleResolution) / sqrt(max(ratio, 1e-12))))
            width = Double(Int(height * ratio))
        }
        return (
            width: ensureDivide(width, configuration.patchSize),
            height: ensureDivide(height, configuration.patchSize)
        )
    }

    /// Python's round is ties-to-even; Swift's default rounded() is ties-away.
    private func ensureDivide(_ value: Double, _ patch: Int) -> Int {
        let quotient = value / Double(patch)
        let lower = Int(floor(quotient))
        let fraction = quotient - Double(lower)
        let rounded: Int
        if fraction < 0.5 {
            rounded = lower
        } else if fraction > 0.5 {
            rounded = lower + 1
        } else {
            rounded = lower.isMultiple(of: 2) ? lower : lower + 1
        }
        return max(rounded * patch, patch)
    }
}
