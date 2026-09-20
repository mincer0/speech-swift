import Foundation

/// Shared dependency graph for the local, single-process Gateway.  Routes
/// receive this value instead of constructing stores per request; all durable
/// stores are actors and therefore safe to share across HTTP and WebSocket
/// handlers.
public struct MiniCPMDemoServiceContainer: Sendable {
    public let layout: MiniCPMDemoStorageLayout
    public let sessions: MiniCPMDemoSessionStore
    public let assets: MiniCPMDemoAssetStore
    public let shares: MiniCPMDemoShareStore
    public let config: MiniCPMDemoConfigStore
    public let admin: MiniCPMDemoAdminService
    public let presets: MiniCPMDemoPresetStore?
    public let webRoot: URL
    public let upstreamRoot: URL?

    public init(
        dataDirectory: URL,
        upstreamRoot: URL? = nil,
        presetRoot: URL? = nil,
        limits: MiniCPMDemoStorageLimits = .init(),
        webRoot: URL? = nil
    ) throws {
        let layout = try MiniCPMDemoStorageLayout(root: dataDirectory)
        let sessions = try MiniCPMDemoSessionStore(root: dataDirectory, limits: limits)
        let builtinDefaultURL = upstreamRoot?.appendingPathComponent(
            "assets/ref_audio/ref_minicpm_signature.wav")
        let assets = try MiniCPMDemoAssetStore(
            layout: layout,
            limits: limits,
            defaultReferenceURL: builtinDefaultURL)
        let shares = try MiniCPMDemoShareStore(layout: layout, limits: limits)
        let config = try MiniCPMDemoConfigStore(layout: layout)
        let admin = try MiniCPMDemoAdminService(sessionStore: sessions, assetStore: assets)

        let projectRoot = Self.resolveProjectRoot(
            presetRoot: presetRoot,
            upstreamRoot: upstreamRoot)
        let presetStore: MiniCPMDemoPresetStore?
        if let projectRoot,
           FileManager.default.fileExists(
               atPath: projectRoot.appendingPathComponent("assets/presets", isDirectory: true).path) {
            presetStore = try MiniCPMDemoPresetStore(projectRoot: projectRoot)
        } else {
            presetStore = nil
        }

        // Upstream supplies presets and reference media, but its static pages
        // may lag behind the transport/AEC fixes shipped by this package.
        let candidateWebRoot = webRoot
            ?? Self.bundledWebRoot()
            ?? upstreamRoot.map { $0.appendingPathComponent("static", isDirectory: true) }
        guard let candidateWebRoot,
              FileManager.default.fileExists(atPath: candidateWebRoot.path) else {
            throw MiniCPMDemoServiceError.notFound("MiniCPM Demo Web resources")
        }

        self.layout = layout
        self.sessions = sessions
        self.assets = assets
        self.shares = shares
        self.config = config
        self.admin = admin
        self.presets = presetStore
        self.webRoot = candidateWebRoot.standardizedFileURL
        self.upstreamRoot = upstreamRoot?.standardizedFileURL
    }

    private static func resolveProjectRoot(
        presetRoot: URL?,
        upstreamRoot: URL?
    ) -> URL? {
        let candidate = (presetRoot ?? upstreamRoot)?.standardizedFileURL
        guard let candidate else { return nil }
        let direct = candidate.appendingPathComponent("assets/presets", isDirectory: true)
        if FileManager.default.fileExists(atPath: direct.path) { return candidate }
        // Accept --preset-root pointing directly at `assets/presets` while
        // keeping all relative audio paths confined to the repository root.
        if candidate.lastPathComponent == "presets" {
            let parent = candidate.deletingLastPathComponent()
            let project = parent.deletingLastPathComponent()
            if FileManager.default.fileExists(
                atPath: candidate.path),
               FileManager.default.fileExists(
                   atPath: project.appendingPathComponent("assets", isDirectory: true).path) {
                return project
            }
        }
        return nil
    }

    private static func bundledWebRoot() -> URL? {
        #if SWIFT_PACKAGE
        return Bundle.module.resourceURL?.appendingPathComponent("Web", isDirectory: true)
        #else
        return nil
        #endif
    }
}
