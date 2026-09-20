import Foundation

public struct MiniCPMDemoAppRecord: Codable, Sendable, Equatable {
    public let appID: String
    public let name: String
    public let route: String
    public var enabled: Bool
    public var updatedAt: Date

    enum CodingKeys: String, CodingKey {
        case appID = "app_id"
        case name, route, enabled
        case updatedAt = "updated_at"
    }

    public init(
        appID: String,
        name: String,
        route: String,
        enabled: Bool = true,
        updatedAt: Date = Date()
    ) {
        self.appID = appID
        self.name = name
        self.route = route
        self.enabled = enabled
        self.updatedAt = updatedAt
    }
}

public struct MiniCPMDemoAdminSnapshot: Codable, Sendable, Equatable {
    public let apps: [MiniCPMDemoAppRecord]
    public let sessions: [MiniCPMDemoSessionSummary]
    public let storage: MiniCPMDemoStorageStats

    public init(
        apps: [MiniCPMDemoAppRecord],
        sessions: [MiniCPMDemoSessionSummary],
        storage: MiniCPMDemoStorageStats
    ) {
        self.apps = apps
        self.sessions = sessions
        self.storage = storage
    }
}

/// Persistent Admin-facing registry.  App toggles survive process restarts;
/// session/storage queries delegate to the durable session and asset stores so
/// the API never reports an in-memory approximation.
public actor MiniCPMDemoAdminService {
    public let sessionStore: MiniCPMDemoSessionStore
    public let assetStore: MiniCPMDemoAssetStore
    private let appsURL: URL
    private var apps: [String: MiniCPMDemoAppRecord]

    public init(
        sessionStore: MiniCPMDemoSessionStore,
        assetStore: MiniCPMDemoAssetStore,
        defaults: [MiniCPMDemoAppRecord] = MiniCPMDemoAdminService.defaultApps()
    ) throws {
        self.sessionStore = sessionStore
        self.assetStore = assetStore
        self.appsURL = sessionStore.layout.admin.appendingPathComponent("apps.json")
        if FileManager.default.fileExists(atPath: appsURL.path) {
            let loaded = try MiniCPMDemoPersistence.read(
                [MiniCPMDemoAppRecord].self, from: appsURL)
            self.apps = Dictionary(uniqueKeysWithValues: loaded.map { ($0.appID, $0) })
        } else {
            self.apps = Dictionary(uniqueKeysWithValues: defaults.map { ($0.appID, $0) })
            try MiniCPMDemoPersistence.write(defaults, to: appsURL)
        }
    }

    public static func defaultApps() -> [MiniCPMDemoAppRecord] {
        [
            MiniCPMDemoAppRecord(appID: "turnbased", name: "Turn-based Chat", route: "/turnbased"),
            MiniCPMDemoAppRecord(appID: "half_duplex_audio", name: "Half-Duplex Audio", route: "/half_duplex"),
            MiniCPMDemoAppRecord(appID: "omni", name: "Omni Full-Duplex", route: "/omni"),
            MiniCPMDemoAppRecord(appID: "audio_duplex", name: "Audio Full-Duplex", route: "/audio_duplex"),
        ]
    }

    public func listApps(includeDisabled: Bool = true) -> [MiniCPMDemoAppRecord] {
        apps.values
            .filter { includeDisabled || $0.enabled }
            .sorted { Self.appComesBefore($0.appID, $1.appID) }
    }

    public func setAppEnabled(
        appID: String,
        enabled: Bool,
        now: Date = Date()
    ) throws -> MiniCPMDemoAppRecord {
        guard var app = apps[appID] else { throw MiniCPMDemoServiceError.notFound("app \(appID)") }
        app.enabled = enabled
        app.updatedAt = now
        apps[appID] = app
        try MiniCPMDemoPersistence.write(
            apps.values.sorted { Self.appComesBefore($0.appID, $1.appID) }, to: appsURL)
        return app
    }

    private static func appComesBefore(_ lhs: String, _ rhs: String) -> Bool {
        let defaults = defaultApps()
        let left = defaults.firstIndex(where: { $0.appID == lhs }) ?? defaults.count
        let right = defaults.firstIndex(where: { $0.appID == rhs }) ?? defaults.count
        return left == right ? lhs < rhs : left < right
    }

    public func snapshot(includeActiveSessions: Bool = true) async throws -> MiniCPMDemoAdminSnapshot {
        let sessions = try await sessionStore.list(includeActive: includeActiveSessions)
        let assets = await assetStore.count()
        let storage = try await sessionStore.storageStats(assetCount: assets)
        return MiniCPMDemoAdminSnapshot(
            apps: listApps(), sessions: sessions, storage: storage)
    }

    public func cleanup(
        retention: TimeInterval? = nil,
        maxTotalBytes: Int64? = nil,
        now: Date = Date()
    ) async throws -> MiniCPMDemoCleanupReport {
        let report = try await sessionStore.cleanup(
            now: now, retention: retention, maxTotalBytes: maxTotalBytes)
        _ = try await assetStore.purgeMissingFiles()
        return report
    }
}
