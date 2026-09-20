import Foundation
import XCTest
@testable import MiniCPMDemoBackend
import Hummingbird

/// The standalone server shares one Router between all Gateway surfaces. Keep
/// this smoke test aligned with `MiniCPMMLXServer/main.swift`: a duplicate
/// method/path is a Hummingbird precondition failure while registering routes,
/// so an HTTP request test would never get a chance to report the regression.
final class MiniCPMDemoRouteRegistrationTests: XCTestCase {
    private func temporaryRoot() throws -> URL {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("minicpm-demo-routes-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        return root
    }

    func testMainRouteRegistrationOrderHasUniqueResponders() throws {
        let root = try temporaryRoot()
        defer { try? FileManager.default.removeItem(at: root) }

        let webRoot = root.appendingPathComponent("web", isDirectory: true)
        try FileManager.default.createDirectory(at: webRoot, withIntermediateDirectories: true)
        try Data("index".utf8).write(to: webRoot.appendingPathComponent("index.html"))
        let services = try MiniCPMDemoServiceContainer(dataDirectory: root, webRoot: webRoot)
        let scheduler = MiniCPMDemoInferenceScheduler()
        let registry = MiniCPMDemoBackendRegistry(scheduler: scheduler)
        let router = Router<BasicRequestContext>(options: [.autoGenerateHeadEndpoints])

        // Keep this order and the control route shape in sync with main.swift.
        router.get("/health") { _, _ in Response(status: .ok) }
        registerMiniCPMDemoRealtimeRoutes(router, scheduler: scheduler, registry: registry)
        registerMiniCPMDemoGatewayDataRoutes(
            router, services: services, scheduler: scheduler, registry: registry)
        registerMiniCPMDemoStaticRoutes(router, services: services)
        router.get("/queue") { _, _ in Response(status: .ok) }
        router.post("/sessions/:session_id/close") { _, _ in Response(status: .ok) }

        // `buildResponder` performs the same route validation/application of
        // auto-generated HEAD endpoints used by the real server startup.
        _ = router.buildResponder()
        let keys = router.routes.map { "\($0.method.rawValue) \($0.path.description)" }
        XCTAssertEqual(keys.count, Set(keys).count)
        XCTAssertEqual(
            router.routes.filter { $0.method == .get && $0.path.description == "/cache" }.count,
            1)
    }
}
