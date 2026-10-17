import Foundation
import XCTest
import MLX

@testable import MiniCPMTTSSemantic

final class E2EMiniCPMTTSLongCacheTimingTests: XCTestCase {
    func testOptInOfficialDecoderLongCacheTiming() throws {
        let environment = ProcessInfo.processInfo.environment
        guard environment["MINICPM_TTS_LONG_CACHE_BENCHMARK"] == "1" else {
            throw XCTSkip("set MINICPM_TTS_LONG_CACHE_BENCHMARK=1 for bounded real-weight timing")
        }
        let path = try XCTUnwrap(environment["MINICPM_TTS_BUNDLE_PATH"])
        let model = try MiniCPMTTSSemantic.fromDirectory(URL(fileURLWithPath: path))
        let input = MLXArray((0..<768).map { Float($0 % 97 - 48) / 64 }, [1, 1, 768])
            .asType(.bfloat16)
        eval(input)
        for prefix in [1022, 1023, 2047, 4093] {
            let session = model.makeSession()
            let zeros = MLXArray.zeros([1, 12, prefix, 64], dtype: .bfloat16)
            for cache in session.layerCaches { cache.update(keys: zeros, values: zeros) }
            eval(session.layerCaches.flatMap { [$0.keys!, $0.values!] })
            var timings: [Double] = []
            let initialCount = model.decoder.encodedDecodeCount
            for _ in 0..<3 {
                let started = DispatchTime.now().uptimeNanoseconds
                let result = model.decoder(
                    input, caches: session.layerCaches, offset: session.cacheOffset,
                    encodedDecode: true)
                eval(result)
                timings.append(Double(DispatchTime.now().uptimeNanoseconds - started) / 1e6)
            }
            XCTAssertEqual(model.decoder.encodedDecodeCount - initialCount, 3,
                           "long-cache timing must exercise the encoded route")
            XCTAssertEqual(session.cacheOffset, prefix + 3)
            XCTAssertTrue(timings.allSatisfy { $0.isFinite && $0 > 0 })
            print("[tts-long-cache] prefix=\(prefix) real_layers=\(session.layerCaches.count) "
                + "synthetic_zero_kv=true samples_ms=\(timings) median_ms=\(timings.sorted()[1])")
        }
    }
}
