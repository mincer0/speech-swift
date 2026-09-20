import Foundation
import Hummingbird
import HummingbirdCore
import NIOCore

/// Mount the vendored upstream Demo pages and their static dependency closure.
/// Every file lookup is confined below the selected Web root; URL path
/// components are never interpreted as filesystem paths before that check.
public func registerMiniCPMDemoStaticRoutes(
    _ router: Router<BasicRequestContext>,
    services: MiniCPMDemoServiceContainer
) {
    let root = services.webRoot

    func page(_ path: String, _ file: String, appID: String? = nil) {
        router.get(RouterPath(path)) { request, _ in
            if let appID,
               !(await services.admin.listApps(includeDisabled: false).contains { $0.appID == appID }) {
                return MiniCPMDemoStaticHTTP.redirect("/")
            }
            return try MiniCPMDemoStaticHTTP.fileResponse(
                request: request,
                file: try MiniCPMDemoStaticHTTP.confined(root: root, relativePath: file),
                fallbackStatus: .notFound)
        }
        router.head(RouterPath(path)) { request, _ in
            if let appID,
               !(await services.admin.listApps(includeDisabled: false).contains { $0.appID == appID }) {
                return MiniCPMDemoStaticHTTP.redirect("/")
            }
            return try MiniCPMDemoStaticHTTP.fileResponse(
                request: request,
                file: try MiniCPMDemoStaticHTTP.confined(root: root, relativePath: file),
                fallbackStatus: .notFound)
        }
    }

    page("/", "index.html")
    page("/turnbased", "turnbased.html", appID: "turnbased")
    page("/omni", "omni/omni.html", appID: "omni")
    page("/half_duplex", "half-duplex/half_duplex.html", appID: "half_duplex_audio")
    page("/audio_duplex", "audio-duplex/audio_duplex.html", appID: "audio_duplex")
    page("/realtime", "realtime/realtime.html")
    page("/admin", "admin.html")

    // RouterPath intentionally ignores trailing slashes.  Registering both
    // `/mobile-omni` and `/mobile-omni/` (or `/mobile` and `/mobile/`) would
    // therefore add a second GET/HEAD handler to the same trie node and hit
    // Hummingbird's `already has a handler` precondition.  Keep one route and
    // inspect the original URI to retain the upstream redirect semantics.
    router.get("/mobile-omni") { request, _ in
        guard request.uri.path.hasSuffix("/") else {
            return MiniCPMDemoStaticHTTP.redirect("/mobile-omni/")
        }
        guard await services.admin.listApps(includeDisabled: false).contains(where: { $0.appID == "omni" }) else {
            return MiniCPMDemoStaticHTTP.redirect("/")
        }
        return try MiniCPMDemoStaticHTTP.fileResponse(
            request: request, file: try MiniCPMDemoStaticHTTP.confined(root: root, relativePath: "mobile-omni/index.html"), fallbackStatus: .notFound)
    }
    router.head("/mobile-omni") { request, _ in
        guard request.uri.path.hasSuffix("/") else {
            return MiniCPMDemoStaticHTTP.redirect("/mobile-omni/")
        }
        guard await services.admin.listApps(includeDisabled: false).contains(where: { $0.appID == "omni" }) else {
            return MiniCPMDemoStaticHTTP.redirect("/")
        }
        return try MiniCPMDemoStaticHTTP.fileResponse(
            request: request, file: try MiniCPMDemoStaticHTTP.confined(root: root, relativePath: "mobile-omni/index.html"), fallbackStatus: .notFound)
    }
    router.get("/mobile") { request, _ in
        guard request.uri.path.hasSuffix("/") else {
            return MiniCPMDemoStaticHTTP.redirect("/mobile/")
        }
        return try MiniCPMDemoStaticHTTP.fileResponse(
            request: request, file: try MiniCPMDemoStaticHTTP.confined(root: root, relativePath: "mobile/index.html"), fallbackStatus: .notFound)
    }
    router.head("/mobile") { request, _ in
        guard request.uri.path.hasSuffix("/") else {
            return MiniCPMDemoStaticHTTP.redirect("/mobile/")
        }
        return try MiniCPMDemoStaticHTTP.fileResponse(
            request: request, file: try MiniCPMDemoStaticHTTP.confined(root: root, relativePath: "mobile/index.html"), fallbackStatus: .notFound)
    }

    // Session viewer is intentionally gated by persisted metadata so a
    // guessed ID cannot turn the page into an arbitrary file oracle.
    router.get("/s/:session_id") { request, context in
        let id = try context.parameters.require("session_id")
        _ = try await services.sessions.metadata(sessionID: id)
        return try MiniCPMDemoStaticHTTP.fileResponse(
            request: request, file: try MiniCPMDemoStaticHTTP.confined(root: root, relativePath: "session-viewer.html"), fallbackStatus: .notFound)
    }
    router.head("/s/:session_id") { request, context in
        let id = try context.parameters.require("session_id")
        _ = try await services.sessions.metadata(sessionID: id)
        return try MiniCPMDemoStaticHTTP.fileResponse(
            request: request, file: try MiniCPMDemoStaticHTTP.confined(root: root, relativePath: "session-viewer.html"), fallbackStatus: .notFound)
    }

    // The upstream gateway hosts the generated Fumadocs site from the same
    // static root and retains these redirects for older links.
    router.get("/docs") { request, _ in
        if request.uri.path.hasSuffix("/") {
            return try MiniCPMDemoStaticHTTP.fileResponse(
                request: request,
                file: try MiniCPMDemoStaticHTTP.confined(root: root, relativePath: "docs/index.html"),
                fallbackStatus: .notFound)
        }
        return MiniCPMDemoStaticHTTP.redirect("/docs/zh/")
    }
    router.head("/docs") { request, _ in
        if request.uri.path.hasSuffix("/") {
            return try MiniCPMDemoStaticHTTP.fileResponse(
                request: request,
                file: try MiniCPMDemoStaticHTTP.confined(root: root, relativePath: "docs/index.html"),
                fallbackStatus: .notFound)
        }
        return MiniCPMDemoStaticHTTP.redirect("/docs/zh/")
    }
    router.get("/docs/overview") { _, _ in MiniCPMDemoStaticHTTP.redirect("/docs/zh/realtime-api/overview/") }
    router.head("/docs/overview") { _, _ in MiniCPMDemoStaticHTTP.redirect("/docs/zh/realtime-api/overview/") }
    router.get("/docs/video") { _, _ in MiniCPMDemoStaticHTTP.redirect("/docs/zh/realtime-api/video/") }
    router.head("/docs/video") { _, _ in MiniCPMDemoStaticHTTP.redirect("/docs/zh/realtime-api/video/") }
    router.get("/docs/audio") { _, _ in MiniCPMDemoStaticHTTP.redirect("/docs/zh/realtime-api/audio/") }
    router.head("/docs/audio") { _, _ in MiniCPMDemoStaticHTTP.redirect("/docs/zh/realtime-api/audio/") }
    router.get("/docs/en") { _, _ in MiniCPMDemoStaticHTTP.redirect("/docs/en/") }
    router.head("/docs/en") { _, _ in MiniCPMDemoStaticHTTP.redirect("/docs/en/") }
    router.get("/docs/en/overview") { _, _ in MiniCPMDemoStaticHTTP.redirect("/docs/en/realtime-api/overview/") }
    router.head("/docs/en/overview") { _, _ in MiniCPMDemoStaticHTTP.redirect("/docs/en/realtime-api/overview/") }
    router.get("/docs/en/video") { _, _ in MiniCPMDemoStaticHTTP.redirect("/docs/en/realtime-api/video/") }
    router.head("/docs/en/video") { _, _ in MiniCPMDemoStaticHTTP.redirect("/docs/en/realtime-api/video/") }
    router.get("/docs/en/audio") { _, _ in MiniCPMDemoStaticHTTP.redirect("/docs/en/realtime-api/audio/") }
    router.head("/docs/en/audio") { _, _ in MiniCPMDemoStaticHTTP.redirect("/docs/en/realtime-api/audio/") }

    for prefix in ["/static/**", "/assets/**", "/shared/**", "/duplex/**", "/lib/**", "/omni/**", "/half-duplex/**", "/audio-duplex/**", "/realtime/**", "/mobile-omni/**", "/mobile/**", "/docs/**", "/tools/**"] {
        router.get(RouterPath(prefix)) { request, context in
            let path = context.parameters.getCatchAll().map(String.init).joined(separator: "/")
            let relative = prefix.hasPrefix("/static") ? path : String(prefix.dropFirst()).replacingOccurrences(of: "**", with: path)
            return try MiniCPMDemoStaticHTTP.fileResponse(
                request: request, file: try MiniCPMDemoStaticHTTP.confined(root: root, relativePath: relative), fallbackStatus: .notFound)
        }
        router.head(RouterPath(prefix)) { request, context in
            let path = context.parameters.getCatchAll().map(String.init).joined(separator: "/")
            let relative = prefix.hasPrefix("/static") ? path : String(prefix.dropFirst()).replacingOccurrences(of: "**", with: path)
            return try MiniCPMDemoStaticHTTP.fileResponse(
                request: request, file: try MiniCPMDemoStaticHTTP.confined(root: root, relativePath: relative), fallbackStatus: .notFound)
        }
    }
}

private enum MiniCPMDemoStaticHTTP {
    static func confined(root: URL, relativePath: String) throws -> URL {
        guard !relativePath.isEmpty,
              !relativePath.hasPrefix("/"),
              !relativePath.split(separator: "/").contains("..") else {
            throw HTTPError(.badRequest, message: "Path traversal detected")
        }
        let file = root.appendingPathComponent(relativePath)
        let base = root.resolvingSymlinksInPath().standardizedFileURL.path
        let resolved = file.resolvingSymlinksInPath().standardizedFileURL.path
        guard resolved == base || resolved.hasPrefix(base + "/") else {
            throw HTTPError(.badRequest, message: "Path traversal detected")
        }
        guard FileManager.default.fileExists(atPath: file.path),
              (try? file.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile) == true else {
            throw HTTPError(.notFound, message: "Static resource not found")
        }
        return file
    }

    static func fileResponse(
        request: Request,
        file: URL,
        fallbackStatus: HTTPResponse.Status
    ) throws -> Response {
        guard FileManager.default.fileExists(atPath: file.path) else {
            throw HTTPError(fallbackStatus, message: "File not found")
        }
        var headers: HTTPFields = [.contentType: mimeType(for: file.pathExtension)]
        let size = (try file.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0
        headers[.contentLength] = String(size)
        let body: ResponseBody = request.method == .head
            ? .init()
            : .init(contentLength: size) { writer in
                let handle = try FileHandle(forReadingFrom: file)
                defer { try? handle.close() }
                while let data = try handle.read(upToCount: 64 * 1024), !data.isEmpty {
                    try await writer.write(ByteBuffer(data: data))
                }
                try await writer.finish(nil)
            }
        return Response(status: .ok, headers: headers, body: body)
    }

    static func redirect(_ location: String) -> Response {
        var headers: HTTPFields = [:]
        headers[.location] = location
        return Response(status: HTTPResponse.Status(code: 302, reasonPhrase: "Found"), headers: headers)
    }

    private static func mimeType(for extension: String) -> String {
        switch `extension`.lowercased() {
        case "html": return "text/html; charset=utf-8"
        case "css": return "text/css; charset=utf-8"
        case "js", "mjs": return "text/javascript; charset=utf-8"
        case "json": return "application/json"
        case "png": return "image/png"
        case "jpg", "jpeg": return "image/jpeg"
        case "svg": return "image/svg+xml"
        case "wav": return "audio/wav"
        case "mp3": return "audio/mpeg"
        case "webm": return "video/webm"
        case "mp4": return "video/mp4"
        case "woff": return "font/woff"
        case "woff2": return "font/woff2"
        default: return "application/octet-stream"
        }
    }
}
