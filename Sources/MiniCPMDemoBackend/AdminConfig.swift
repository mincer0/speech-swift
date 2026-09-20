import Foundation

public struct MiniCPMDemoFrontendDefaults: Codable, Sendable, Equatable {
    public var playbackDelayMS: Int
    public var defaultLanguage: String?

    enum CodingKeys: String, CodingKey {
        case playbackDelayMS = "playback_delay_ms"
        case defaultLanguage = "default_lang"
    }

    public init(playbackDelayMS: Int = 200, defaultLanguage: String? = nil) {
        self.playbackDelayMS = max(0, playbackDelayMS)
        self.defaultLanguage = defaultLanguage
    }
}

public struct MiniCPMDemoETAConfig: Codable, Sendable, Equatable {
    public var chatSeconds: Double
    public var halfDuplexSeconds: Double
    public var audioDuplexSeconds: Double
    public var omniDuplexSeconds: Double
    public var emaAlpha: Double
    public var emaMinimumSamples: Int

    enum CodingKeys: String, CodingKey {
        case chatSeconds = "eta_chat_s"
        case halfDuplexSeconds = "eta_half_duplex_s"
        case audioDuplexSeconds = "eta_audio_duplex_s"
        case omniDuplexSeconds = "eta_omni_duplex_s"
        case emaAlpha = "ema_alpha"
        case emaMinimumSamples = "ema_min_samples"
    }

    public init(
        chatSeconds: Double = 15,
        halfDuplexSeconds: Double = 180,
        audioDuplexSeconds: Double = 120,
        omniDuplexSeconds: Double = 90,
        emaAlpha: Double = 0.3,
        emaMinimumSamples: Int = 3
    ) {
        self.chatSeconds = max(0, chatSeconds)
        self.halfDuplexSeconds = max(0, halfDuplexSeconds)
        self.audioDuplexSeconds = max(0, audioDuplexSeconds)
        self.omniDuplexSeconds = max(0, omniDuplexSeconds)
        self.emaAlpha = min(1, max(0, emaAlpha))
        self.emaMinimumSamples = max(1, emaMinimumSamples)
    }

    public func baseline(for requestType: String) -> Double {
        switch requestType {
        case "chat": return chatSeconds
        case "half_duplex_audio": return halfDuplexSeconds
        case "audio_duplex": return audioDuplexSeconds
        case "omni_duplex": return omniDuplexSeconds
        default: return 30
        }
    }
}

public struct MiniCPMDemoETAStatus: Codable, Sendable, Equatable {
    public let config: MiniCPMDemoETAConfig
    public let ema: [String: Double]
    public let samples: [String: Int]

    public init(config: MiniCPMDemoETAConfig, ema: [String: Double], samples: [String: Int]) {
        self.config = config
        self.ema = ema
        self.samples = samples
    }
}

/// Persistent frontend defaults and ETA/EMA configuration.  EMA samples are
/// stored in `admin/eta.json`, so an Admin reload does not silently reset ETA
/// estimates to an in-memory placeholder.
public actor MiniCPMDemoConfigStore {
    public let layout: MiniCPMDemoStorageLayout
    private let defaultsURL: URL
    private let etaURL: URL
    private var defaults: MiniCPMDemoFrontendDefaults
    private var eta: MiniCPMDemoETAConfig
    private var ema: [String: Double] = [:]
    private var sampleCounts: [String: Int] = [:]

    public init(
        layout: MiniCPMDemoStorageLayout,
        defaults: MiniCPMDemoFrontendDefaults = .init(),
        eta: MiniCPMDemoETAConfig = .init()
    ) throws {
        self.layout = layout
        self.defaultsURL = layout.admin.appendingPathComponent("frontend_defaults.json")
        self.etaURL = layout.admin.appendingPathComponent("eta.json")
        if FileManager.default.fileExists(atPath: defaultsURL.path) {
            self.defaults = try MiniCPMDemoPersistence.read(
                MiniCPMDemoFrontendDefaults.self, from: defaultsURL)
        } else {
            self.defaults = defaults
            try MiniCPMDemoPersistence.write(defaults, to: defaultsURL)
        }
        if FileManager.default.fileExists(atPath: etaURL.path) {
            let saved = try MiniCPMDemoPersistence.read(
                MiniCPMDemoPersistedETA.self, from: etaURL)
            self.eta = saved.config
            self.ema = saved.ema
            self.sampleCounts = saved.samples
        } else {
            self.eta = eta
            try MiniCPMDemoPersistence.write(
                MiniCPMDemoPersistedETA(config: eta, ema: [:], samples: [:]), to: etaURL)
        }
    }

    public func frontendDefaults() -> MiniCPMDemoFrontendDefaults { defaults }

    public func updateFrontendDefaults(_ value: MiniCPMDemoFrontendDefaults) throws {
        defaults = value
        try MiniCPMDemoPersistence.write(value, to: defaultsURL)
    }

    public func etaStatus() -> MiniCPMDemoETAStatus {
        MiniCPMDemoETAStatus(config: eta, ema: ema, samples: sampleCounts)
    }

    public func updateETA(_ value: MiniCPMDemoETAConfig) throws -> MiniCPMDemoETAStatus {
        eta = value
        try persistETA()
        return etaStatus()
    }

    public func recordDuration(requestType: String, durationSeconds: Double) throws {
        guard durationSeconds.isFinite, durationSeconds >= 0 else {
            throw MiniCPMDemoServiceError.invalidMedia("invalid ETA duration")
        }
        let count = sampleCounts[requestType, default: 0]
        if count == 0 {
            ema[requestType] = durationSeconds
        } else {
            let old = ema[requestType] ?? durationSeconds
            ema[requestType] = eta.emaAlpha * durationSeconds + (1 - eta.emaAlpha) * old
        }
        sampleCounts[requestType] = count + 1
        try persistETA()
    }

    public func estimate(requestType: String) -> Double {
        guard sampleCounts[requestType, default: 0] >= eta.emaMinimumSamples else {
            return eta.baseline(for: requestType)
        }
        return ema[requestType] ?? eta.baseline(for: requestType)
    }

    private func persistETA() throws {
        try MiniCPMDemoPersistence.write(
            MiniCPMDemoPersistedETA(config: eta, ema: ema, samples: sampleCounts), to: etaURL)
    }
}

private struct MiniCPMDemoPersistedETA: Codable, Sendable {
    let config: MiniCPMDemoETAConfig
    let ema: [String: Double]
    let samples: [String: Int]
}
