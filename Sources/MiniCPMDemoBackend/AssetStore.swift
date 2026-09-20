import Foundation

public enum MiniCPMDemoAssetKind: String, Codable, Sendable {
    case referenceAudio = "ref_audio"
    case upload
}

public struct MiniCPMDemoAssetRecord: Codable, Sendable, Equatable {
    public let id: String
    public let name: String
    public let filename: String
    public let kind: MiniCPMDemoAssetKind
    public let mimeType: String
    public let sizeBytes: Int64
    public let sampleRate: Int?
    public let durationMS: Int?
    public let createdAt: Date

    enum CodingKeys: String, CodingKey {
        case id, name, filename, kind
        case mimeType = "mime_type"
        case sizeBytes = "size_bytes"
        case sampleRate = "sample_rate"
        case durationMS = "duration_ms"
        case createdAt = "created_at"
    }

    public init(
        id: String,
        name: String,
        filename: String,
        kind: MiniCPMDemoAssetKind = .referenceAudio,
        mimeType: String,
        sizeBytes: Int64,
        sampleRate: Int? = nil,
        durationMS: Int? = nil,
        createdAt: Date = Date()
    ) {
        self.id = id
        self.name = name
        self.filename = filename
        self.kind = kind
        self.mimeType = mimeType
        self.sizeBytes = sizeBytes
        self.sampleRate = sampleRate
        self.durationMS = durationMS
        self.createdAt = createdAt
    }
}

public struct MiniCPMDemoAssetDownload: Sendable, Equatable {
    public let record: MiniCPMDemoAssetRecord
    public let data: Data

    public init(record: MiniCPMDemoAssetRecord, data: Data) {
        self.record = record
        self.data = data
    }
}

/// Durable reference-audio/asset registry.  Uploads are written as unique
/// files and indexed by `registry.json`; the service never trusts a filename
/// supplied by a client and never returns a path outside its asset directory.
public actor MiniCPMDemoAssetStore {
    public let layout: MiniCPMDemoStorageLayout
    public let limits: MiniCPMDemoStorageLimits
    private let directory: URL
    private let registryURL: URL
    private let defaultURL: URL
    private let builtinDefaultURL: URL?
    private var records: [String: MiniCPMDemoAssetRecord] = [:]

    public init(
        layout: MiniCPMDemoStorageLayout,
        limits: MiniCPMDemoStorageLimits = .init(),
        defaultReferenceURL: URL? = nil
    ) throws {
        // Keep reference audio under the same path as the Python gateway.
        // Older Swift builds accidentally nested this as `assets/assets`; the
        // migration below keeps those registries/files available after the
        // layout correction.
        let directory = layout.assets.appendingPathComponent("ref_audio", isDirectory: true)
        let registryURL = directory.appendingPathComponent("registry.json")
        let defaultURL = layout.admin.appendingPathComponent("default_ref_audio.json")
        let builtinDefaultURL = defaultReferenceURL?.standardizedFileURL
        try MiniCPMDemoPersistence.ensureDirectory(directory)

        // Actor initializers are nonisolated until all stored properties have
        // been initialized.  Keep all bootstrap/migration work in local state
        // and static helpers, then publish the completed registry in one step.
        var initialRecords = try Self.loadInitialRecords(
            registryURL: registryURL,
            directory: directory,
            legacyDirectory: layout.assets.appendingPathComponent("assets", isDirectory: true))
        try Self.bootstrapBuiltinDefaultIfNeeded(
            records: &initialRecords,
            layout: layout,
            limits: limits,
            directory: directory,
            registryURL: registryURL,
            defaultURL: defaultURL,
            builtinDefaultURL: builtinDefaultURL)

        self.layout = layout
        self.limits = limits
        self.directory = directory
        self.registryURL = registryURL
        self.defaultURL = defaultURL
        self.builtinDefaultURL = builtinDefaultURL
        self.records = initialRecords
    }

    public init(
        root: URL,
        limits: MiniCPMDemoStorageLimits = .init(),
        defaultReferenceURL: URL? = nil
    ) throws {
        try self.init(
            layout: MiniCPMDemoStorageLayout(root: root),
            limits: limits,
            defaultReferenceURL: defaultReferenceURL)
    }

    public func uploadReferenceAudio(
        name: String,
        data: Data,
        mimeType: String = "audio/wav",
        sampleRate: Int? = 16_000,
        durationMS: Int? = nil
    ) throws -> MiniCPMDemoAssetRecord {
        let normalized = try MiniCPMDemoAudioNormalizer.normalize(
            data: data, mimeType: mimeType, fileExtension: Self.extension(for: mimeType))
        let wav = MiniCPMDemoAudioNormalizer.wavData(
            samples: normalized.samples, sampleRate: normalized.sampleRate)
        return try upload(
            name: name, data: wav, kind: .referenceAudio,
            mimeType: "audio/wav", fileExtension: "wav",
            sampleRate: normalized.sampleRate,
            durationMS: Int((normalized.durationSeconds * 1_000).rounded()))
    }

    public func upload(
        name: String,
        data: Data,
        kind: MiniCPMDemoAssetKind = .upload,
        mimeType: String,
        fileExtension: String,
        sampleRate: Int? = nil,
        durationMS: Int? = nil
    ) throws -> MiniCPMDemoAssetRecord {
        let cleanName = try Self.normalizedName(name)
        guard !data.isEmpty else { throw MiniCPMDemoServiceError.invalidMedia("empty asset") }
        guard data.count <= limits.maxAssetBytes else {
            throw MiniCPMDemoServiceError.limitExceeded("max_asset_bytes")
        }
        let ext = try Self.normalizedExtension(fileExtension)
        guard limits.maxTotalBytes >= Int64(data.count) else {
            throw MiniCPMDemoServiceError.limitExceeded("max_total_bytes")
        }
        let total = MiniCPMDemoPersistence.directorySize(layout.root)
        guard total <= limits.maxTotalBytes - Int64(data.count) else {
            throw MiniCPMDemoServiceError.limitExceeded("max_total_bytes")
        }
        let id = UUID().uuidString.lowercased()
        let filename = "\(id).\(ext)"
        let path = directory.appendingPathComponent(filename)
        guard !FileManager.default.fileExists(atPath: path.path) else {
            throw MiniCPMDemoServiceError.alreadyExists(id)
        }
        guard FileManager.default.createFile(atPath: path.path, contents: data) else {
            throw MiniCPMDemoServiceError.persistence("cannot create asset file")
        }
        let record = MiniCPMDemoAssetRecord(
            id: id, name: cleanName, filename: filename, kind: kind,
            mimeType: mimeType, sizeBytes: Int64(data.count),
            sampleRate: sampleRate, durationMS: durationMS)
        records[id] = record
        do { try persist() }
        catch {
            try? FileManager.default.removeItem(at: path)
            records.removeValue(forKey: id)
            throw error
        }
        return record
    }

    public func list(kind: MiniCPMDemoAssetKind? = nil) -> [MiniCPMDemoAssetRecord] {
        records.values
            .filter { kind == nil || $0.kind == kind }
            .sorted { $0.createdAt > $1.createdAt }
    }

    public func get(_ id: String) throws -> MiniCPMDemoAssetRecord {
        guard let record = records[id] else { throw MiniCPMDemoServiceError.notFound(id) }
        return record
    }

    public func fileURL(_ id: String) throws -> URL {
        let record = try get(id)
        let path = directory.appendingPathComponent(record.filename)
        let resolvedDirectory = directory.resolvingSymlinksInPath().standardizedFileURL.path
        let resolved = path.resolvingSymlinksInPath().standardizedFileURL.path
        guard resolved.hasPrefix(resolvedDirectory + "/") else {
            throw MiniCPMDemoServiceError.invalidIdentifier("asset path escapes storage")
        }
        guard FileManager.default.fileExists(atPath: path.path) else {
            throw MiniCPMDemoServiceError.notFound(id)
        }
        return path
    }

    public func download(_ id: String) throws -> MiniCPMDemoAssetDownload {
        let record = try get(id)
        let path = try fileURL(id)
        do { return MiniCPMDemoAssetDownload(record: record, data: try Data(contentsOf: path)) }
        catch { throw MiniCPMDemoServiceError.persistence("read asset: \(error.localizedDescription)") }
    }

    @discardableResult
    public func delete(_ id: String) throws -> Bool {
        guard let record = records.removeValue(forKey: id) else { return false }
        let path = directory.appendingPathComponent(record.filename)
        try? FileManager.default.removeItem(at: path)
        try persist()
        return true
    }

    public func count() -> Int { records.count }

    public func setDefaultReferenceAudio(_ id: String?) throws {
        if let id { _ = try get(id) }
        if let id {
            try MiniCPMDemoPersistence.write(["asset_id": id], to: defaultURL)
        } else {
            try? FileManager.default.removeItem(at: defaultURL)
        }
    }

    public func defaultReferenceAudio() throws -> MiniCPMDemoAssetDownload {
        guard FileManager.default.fileExists(atPath: defaultURL.path) else {
            throw MiniCPMDemoServiceError.notFound("default_ref_audio")
        }
        let value = try MiniCPMDemoPersistence.read(
            [String: String].self, from: defaultURL)
        guard let id = value["asset_id"] else {
            throw MiniCPMDemoServiceError.persistence("default_ref_audio missing asset_id")
        }
        return try download(id)
    }

    /// Register the pinned upstream signature clip the first time a fresh
    /// Swift data directory is used.  The Python gateway resolves this file
    /// lazily from `assets/ref_audio/ref_minicpm_signature.wav`; persisting it
    /// here keeps the Swift `/api/default_ref_audio` and asset list stable
    /// across restarts while still allowing an explicit user-selected default.
    private static func loadInitialRecords(
        registryURL: URL,
        directory: URL,
        legacyDirectory: URL
    ) throws -> [String: MiniCPMDemoAssetRecord] {
        if FileManager.default.fileExists(atPath: registryURL.path) {
            let loaded = try MiniCPMDemoPersistence.read(
                [MiniCPMDemoAssetRecord].self, from: registryURL)
            return Dictionary(uniqueKeysWithValues: loaded.map { ($0.id, $0) })
        }

        // Best-effort migration from the pre-parity `assets/assets`
        // directory.  A corrupt legacy registry must not prevent a fresh
        // store from starting; valid rows are copied one by one.
        var records: [String: MiniCPMDemoAssetRecord] = [:]
        let legacyRegistry = legacyDirectory.appendingPathComponent("registry.json")
        if FileManager.default.fileExists(atPath: legacyRegistry.path),
           let loaded = try? MiniCPMDemoPersistence.read(
               [MiniCPMDemoAssetRecord].self, from: legacyRegistry) {
            for record in loaded {
                let source = legacyDirectory.appendingPathComponent(record.filename)
                let destination = directory.appendingPathComponent(record.filename)
                guard FileManager.default.fileExists(atPath: source.path),
                      !FileManager.default.fileExists(atPath: destination.path) else { continue }
                try? FileManager.default.copyItem(at: source, to: destination)
                if FileManager.default.fileExists(atPath: destination.path) {
                    records[record.id] = record
                }
            }
            if !records.isEmpty {
                try persist(records: records, to: registryURL)
            }
        }
        return records
    }

    private static func bootstrapBuiltinDefaultIfNeeded(
        records: inout [String: MiniCPMDemoAssetRecord],
        layout: MiniCPMDemoStorageLayout,
        limits: MiniCPMDemoStorageLimits,
        directory: URL,
        registryURL: URL,
        defaultURL: URL,
        builtinDefaultURL: URL?
    ) throws {
        guard let source = builtinDefaultURL,
              FileManager.default.fileExists(atPath: source.path) else { return }
        let sourceData = try Data(contentsOf: source)
        let normalized = try MiniCPMDemoAudioNormalizer.normalize(
            data: sourceData, mimeType: "audio/wav", fileExtension: "wav")
        let wav = MiniCPMDemoAudioNormalizer.wavData(
            samples: normalized.samples, sampleRate: normalized.sampleRate)
        let id = "builtin_default"
        let filename = "\(id).wav"
        let destination = directory.appendingPathComponent(filename)
        // A process can crash after writing the builtin file but before the
        // registry/default pointer is persisted.  Reconstruct the row instead
        // of treating that partial state as a valid bootstrap.
        if !FileManager.default.fileExists(atPath: destination.path) {
            // Also recognize the literal `(id).wav` name emitted by the old
            // implementation and move it to the canonical filename.
            let legacyDestination = directory.appendingPathComponent("(id).wav")
            if FileManager.default.fileExists(atPath: legacyDestination.path) {
                try? FileManager.default.moveItem(at: legacyDestination, to: destination)
            }
        }
        if FileManager.default.fileExists(atPath: destination.path),
           records[id] == nil {
            let attrs = try? FileManager.default.attributesOfItem(atPath: destination.path)
            let size = (attrs?[.size] as? NSNumber)?.int64Value ?? Int64(wav.count)
            records[id] = MiniCPMDemoAssetRecord(
                id: id,
                name: source.lastPathComponent,
                filename: filename,
                kind: .referenceAudio,
                mimeType: "audio/wav",
                sizeBytes: size,
                sampleRate: normalized.sampleRate,
                durationMS: Int((normalized.durationSeconds * 1_000).rounded()))
            try persist(records: records, to: registryURL)
        }
        if FileManager.default.fileExists(atPath: destination.path) {
            if !FileManager.default.fileExists(atPath: defaultURL.path) {
                try MiniCPMDemoPersistence.write(["asset_id": id], to: defaultURL)
            }
            return
        }
        guard limits.maxAssetBytes >= Int64(wav.count),
              MiniCPMDemoPersistence.directorySize(layout.root) <= limits.maxTotalBytes - Int64(wav.count) else {
            throw MiniCPMDemoServiceError.limitExceeded("max_total_bytes")
        }
        guard FileManager.default.createFile(atPath: destination.path, contents: wav) else {
            throw MiniCPMDemoServiceError.persistence("cannot create builtin default asset")
        }
        let record = MiniCPMDemoAssetRecord(
            id: id,
            name: source.lastPathComponent,
            filename: filename,
            kind: .referenceAudio,
            mimeType: "audio/wav",
            sizeBytes: Int64(wav.count),
            sampleRate: normalized.sampleRate,
            durationMS: Int((normalized.durationSeconds * 1_000).rounded()))
        records[id] = record
        try persist(records: records, to: registryURL)
        try MiniCPMDemoPersistence.write(["asset_id": id], to: defaultURL)
    }

    /// Remove registry rows whose file was manually deleted or lost after a
    /// crash. This keeps Admin listings truthful without deleting valid files.
    @discardableResult
    public func purgeMissingFiles() throws -> Int {
        let before = records.count
        records = records.filter { _, record in
            FileManager.default.fileExists(
                atPath: directory.appendingPathComponent(record.filename).path)
        }
        if records.count != before { try persist() }
        return before - records.count
    }

    private func persist() throws {
        try Self.persist(records: records, to: registryURL)
    }

    private static func persist(
        records: [String: MiniCPMDemoAssetRecord],
        to registryURL: URL
    ) throws {
        try MiniCPMDemoPersistence.write(
            records.values.sorted { $0.createdAt < $1.createdAt }, to: registryURL)
    }

    private static func normalizedName(_ name: String) throws -> String {
        let value = name.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !value.isEmpty, value.count <= 240, !value.contains("\0") else {
            throw MiniCPMDemoServiceError.invalidName(name)
        }
        return value
    }

    private static func normalizedExtension(_ value: String) throws -> String {
        let ext = value.trimmingCharacters(in: CharacterSet(charactersIn: ". ")).lowercased()
        guard !ext.isEmpty, ext.count <= 8,
              ext.unicodeScalars.allSatisfy({ $0.isASCII && ($0.value >= 48 && $0.value <= 57
                  || $0.value >= 97 && $0.value <= 122) }) else {
            throw MiniCPMDemoServiceError.invalidMedia("unsafe extension")
        }
        return ext
    }

    private static func `extension`(for mimeType: String) -> String {
        let type = mimeType.lowercased().split(separator: ";", maxSplits: 1).first.map(String.init) ?? ""
        switch type {
        case "audio/wav": return "wav"
        case "audio/webm": return "webm"
        case "image/jpeg": return "jpg"
        case "image/png": return "png"
        case "video/mp4": return "mp4"
        case "video/webm": return "webm"
        default: return "bin"
        }
    }
}
