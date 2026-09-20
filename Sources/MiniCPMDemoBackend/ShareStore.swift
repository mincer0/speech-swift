import CryptoKit
import Foundation

public struct MiniCPMDemoComment: Codable, Sendable, Equatable {
    public let sessionID: String
    public let text: String
    public let updatedAt: Date

    public init(sessionID: String, text: String, updatedAt: Date = Date()) {
        self.sessionID = sessionID
        self.text = text
        self.updatedAt = updatedAt
    }
}

public struct MiniCPMDemoShareInfo: Codable, Sendable, Equatable {
    public let token: String
    public let sessionID: String
    public let createdAt: Date
    public let expiresAt: Date?

    public init(token: String, sessionID: String, createdAt: Date = Date(), expiresAt: Date? = nil) {
        self.token = token
        self.sessionID = sessionID
        self.createdAt = createdAt
        self.expiresAt = expiresAt
    }
}

private struct MiniCPMDemoPersistedShare: Codable, Sendable, Equatable {
    let tokenHash: String
    let sessionID: String
    let createdAt: Date
    let expiresAt: Date?
    var revoked: Bool
}

/// Durable comments and capability-style share links.  Plaintext tokens are
/// returned only at creation time; disk stores SHA-256 token hashes, so a
/// leaked registry cannot be used to open sessions.  Share resolution still
/// checks expiry, revocation and session existence on every request.
public actor MiniCPMDemoShareStore {
    public let layout: MiniCPMDemoStorageLayout
    public let limits: MiniCPMDemoStorageLimits
    private let registryURL: URL
    private var shares: [String: MiniCPMDemoPersistedShare] = [:]

    public init(
        layout: MiniCPMDemoStorageLayout,
        limits: MiniCPMDemoStorageLimits = .init()
    ) throws {
        self.layout = layout
        self.limits = limits
        self.registryURL = layout.admin.appendingPathComponent("shares.json")
        if FileManager.default.fileExists(atPath: registryURL.path) {
            let loaded = try MiniCPMDemoPersistence.read(
                [MiniCPMDemoPersistedShare].self, from: registryURL)
            self.shares = Dictionary(uniqueKeysWithValues: loaded.map { ($0.tokenHash, $0) })
        }
    }

    public func setComment(
        sessionID: String,
        text: String,
        now: Date = Date()
    ) throws -> MiniCPMDemoComment {
        _ = try existingSessionDirectory(sessionID)
        let value = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard value.count <= limits.maxCommentCharacters else {
            throw MiniCPMDemoServiceError.invalidComment("comment exceeds max characters")
        }
        let path = try layout.sessionCommentURL(sessionID)
        if value.isEmpty {
            try? FileManager.default.removeItem(at: path)
        } else {
            do { try Data(value.utf8).write(to: path, options: [.atomic]) }
            catch { throw MiniCPMDemoServiceError.persistence("write comment: \(error.localizedDescription)") }
        }
        return MiniCPMDemoComment(sessionID: sessionID, text: value, updatedAt: now)
    }

    public func comment(sessionID: String) throws -> MiniCPMDemoComment {
        let path = try layout.sessionCommentURL(sessionID)
        guard FileManager.default.fileExists(atPath: path.path) else {
            _ = try existingSessionDirectory(sessionID)
            return MiniCPMDemoComment(sessionID: sessionID, text: "")
        }
        do {
            return MiniCPMDemoComment(
                sessionID: sessionID,
                text: try String(contentsOf: path, encoding: .utf8),
                updatedAt: (try? path.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? Date())
        } catch {
            throw MiniCPMDemoServiceError.persistence("read comment: \(error.localizedDescription)")
        }
    }

    public func createShare(
        sessionID: String,
        expiresIn: TimeInterval? = nil,
        now: Date = Date()
    ) throws -> MiniCPMDemoShareInfo {
        _ = try existingSessionDirectory(sessionID)
        let expiration = expiresIn.map { now.addingTimeInterval(max(1, $0)) }
        var token: String
        var hash: String
        repeat {
            token = (UUID().uuidString + UUID().uuidString)
                .replacingOccurrences(of: "-", with: "").lowercased()
            hash = Self.hash(token)
        } while shares[hash] != nil
        shares[hash] = MiniCPMDemoPersistedShare(
            tokenHash: hash, sessionID: sessionID, createdAt: now,
            expiresAt: expiration, revoked: false)
        try persist()
        return MiniCPMDemoShareInfo(
            token: token, sessionID: sessionID, createdAt: now, expiresAt: expiration)
    }

    public func resolveShare(token: String, now: Date = Date()) throws -> MiniCPMDemoShareInfo {
        let hash = Self.hash(token)
        guard let share = shares[hash] else { throw MiniCPMDemoServiceError.notFound("share") }
        guard !share.revoked else { throw MiniCPMDemoServiceError.revoked("share") }
        if let expiresAt = share.expiresAt, expiresAt <= now {
            throw MiniCPMDemoServiceError.expired("share")
        }
        _ = try existingSessionDirectory(share.sessionID)
        return MiniCPMDemoShareInfo(
            token: token, sessionID: share.sessionID,
            createdAt: share.createdAt, expiresAt: share.expiresAt)
    }

    public func revokeShare(token: String) throws {
        let hash = Self.hash(token)
        guard var share = shares[hash] else { throw MiniCPMDemoServiceError.notFound("share") }
        share.revoked = true
        shares[hash] = share
        try persist()
    }

    @discardableResult
    public func purgeExpired(now: Date = Date()) throws -> Int {
        let before = shares.count
        shares = shares.filter { _, share in
            guard let expiresAt = share.expiresAt else { return true }
            return expiresAt > now && !share.revoked
        }
        if shares.count != before { try persist() }
        return before - shares.count
    }

    public func shareCount() -> Int { shares.count }

    private func existingSessionDirectory(_ sessionID: String) throws -> URL {
        let directory = try layout.sessionURL(sessionID)
        guard FileManager.default.fileExists(atPath: directory.path) else {
            throw MiniCPMDemoServiceError.notFound(sessionID)
        }
        return directory
    }

    private func persist() throws {
        try MiniCPMDemoPersistence.write(
            shares.values.sorted { $0.createdAt < $1.createdAt }, to: registryURL)
    }

    private static func hash(_ token: String) -> String {
        SHA256.hash(data: Data(token.utf8)).map { String(format: "%02x", $0) }.joined()
    }
}
