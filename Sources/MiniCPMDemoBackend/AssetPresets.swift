import Foundation

public struct MiniCPMDemoPreset: Codable, Sendable, Equatable {
    public let mode: String
    public let id: String
    public let values: [String: MiniCPMJSONValue]

    public init(mode: String, id: String, values: [String: MiniCPMJSONValue]) {
        self.mode = mode
        self.id = id
        self.values = values
    }

    public var order: Int { Int(values["order"]?.numberValue ?? 999) }
    public var name: String { values["name"]?.stringValue ?? id }
}

public struct MiniCPMDemoPresetAudio: Sendable, Equatable {
    public let mode: String
    public let presetID: String
    public let path: String
    public let name: String
    public let data: Data
    public let mimeType: String
    public let sampleRate: Int
    public let durationSeconds: Double

    public init(
        mode: String,
        presetID: String,
        path: String,
        name: String,
        data: Data,
        mimeType: String,
        sampleRate: Int = MiniCPMDemoAudioNormalizer.targetSampleRate,
        durationSeconds: Double = 0
    ) {
        self.mode = mode
        self.presetID = presetID
        self.path = path
        self.name = name
        self.data = data
        self.mimeType = mimeType
        self.sampleRate = sampleRate
        self.durationSeconds = durationSeconds
    }
}

/// Filesystem-backed preset loader for the pinned Demo assets.  It implements
/// the subset of YAML used by `assets/presets/*/*.yaml` without adding a YAML
/// runtime dependency; arbitrary unknown scalar fields are preserved as
/// `MiniCPMJSONValue` and malformed files are skipped like the Python gateway.
public actor MiniCPMDemoPresetStore {
    public let projectRoot: URL
    public let presetsRoot: URL

    public init(projectRoot: URL) throws {
        self.projectRoot = projectRoot.standardizedFileURL
        self.presetsRoot = self.projectRoot.appendingPathComponent("assets/presets", isDirectory: true)
        guard FileManager.default.fileExists(atPath: presetsRoot.path) else {
            throw MiniCPMDemoServiceError.notFound("assets/presets")
        }
    }

    public func list() throws -> [String: [MiniCPMDemoPreset]] {
        // A single malformed/partially copied mode must not make the whole
        // preset endpoint fail.  This mirrors the Python loader's best-effort
        // behaviour while still preserving valid siblings.
        guard let modes = try? FileManager.default.contentsOfDirectory(
            at: presetsRoot, includingPropertiesForKeys: [.isDirectoryKey], options: [.skipsHiddenFiles]) else {
            return [:]
        }
        var result: [String: [MiniCPMDemoPreset]] = [:]
        for modeURL in modes where (try? modeURL.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) == true {
            let mode = modeURL.lastPathComponent
            guard (try? MiniCPMDemoStorageLayout.safeComponent(mode, label: "preset mode")) != nil else { continue }
            guard let files = try? FileManager.default.contentsOfDirectory(
                at: modeURL, includingPropertiesForKeys: [.isRegularFileKey], options: [.skipsHiddenFiles]) else {
                continue
            }
            let presets = files.compactMap { file -> MiniCPMDemoPreset? in
                guard file.pathExtension.lowercased() == "yaml" || file.pathExtension.lowercased() == "yml",
                      (try? file.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile) == true else { return nil }
                guard let text = try? String(contentsOf: file, encoding: .utf8),
                      let values = Self.parseYAML(text) else { return nil }
                let id = values["id"]?.stringValue ?? file.deletingPathExtension().lastPathComponent
                guard (try? MiniCPMDemoStorageLayout.safeComponent(id, label: "preset id")) != nil else { return nil }
                return MiniCPMDemoPreset(mode: mode, id: id, values: Self.resolveAudioMetadata(values, projectRoot: projectRoot))
            }.sorted { $0.order == $1.order ? $0.id < $1.id : $0.order < $1.order }
            if !presets.isEmpty { result[mode] = presets }
        }
        return result
    }

    public func audio(mode: String, presetID: String) throws -> [MiniCPMDemoPresetAudio] {
        try MiniCPMDemoStorageLayout.safeComponent(mode, label: "preset mode")
        try MiniCPMDemoStorageLayout.safeComponent(presetID, label: "preset id")
        guard let preset = try list()[mode]?.first(where: { $0.id == presetID }) else {
            throw MiniCPMDemoServiceError.notFound("preset \(mode)/\(presetID)")
        }
        var references: [String] = []
        if case .object(let ref)? = preset.values["ref_audio"],
           let path = ref["path"]?.stringValue { references.append(path) }
        if case .array(let content)? = preset.values["system_content"] {
            for item in content {
                if case .object(let object) = item,
                   object["type"]?.stringValue == "audio",
                   let path = object["path"]?.stringValue { references.append(path) }
            }
        }
        return try references.map { path in
            let file = try confinedAssetPath(path)
            let sourceData: Data
            do { sourceData = try Data(contentsOf: file) }
            catch { throw MiniCPMDemoServiceError.persistence("read preset audio: \(error.localizedDescription)") }
            let normalized = try MiniCPMDemoAudioNormalizer.normalize(
                data: sourceData,
                mimeType: Self.mimeType(for: file.pathExtension),
                fileExtension: file.pathExtension)
            return MiniCPMDemoPresetAudio(
                mode: mode, presetID: presetID, path: path,
                name: file.lastPathComponent,
                data: normalized.float32Data,
                mimeType: "audio/x-pcm-f32",
                sampleRate: normalized.sampleRate,
                durationSeconds: normalized.durationSeconds)
        }
    }

    private func confinedAssetPath(_ relativePath: String) throws -> URL {
        guard !relativePath.hasPrefix("/"),
              !relativePath.split(separator: "/").contains("..") else {
            throw MiniCPMDemoServiceError.invalidIdentifier("preset audio path")
        }
        let file = projectRoot.appendingPathComponent(relativePath)
        let base = projectRoot.resolvingSymlinksInPath().standardizedFileURL.path
        let resolved = file.resolvingSymlinksInPath().standardizedFileURL.path
        guard resolved.hasPrefix(base + "/") else {
            throw MiniCPMDemoServiceError.invalidIdentifier("preset audio path escapes project root")
        }
        guard FileManager.default.fileExists(atPath: file.path),
              (try? file.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile) == true else {
            throw MiniCPMDemoServiceError.notFound(relativePath)
        }
        return file
    }

    private static func resolveAudioMetadata(
        _ values: [String: MiniCPMJSONValue], projectRoot: URL
    ) -> [String: MiniCPMJSONValue] {
        var result = values
        if let path = result["ref_audio_path"]?.stringValue {
            result.removeValue(forKey: "ref_audio_path")
            result["ref_audio"] = .object([
                "data": .null,
                "path": .string(path),
                "name": .string(URL(fileURLWithPath: path).lastPathComponent),
                "duration": .number(audioDuration(path: path, projectRoot: projectRoot)),
            ])
        }
        if case .array(let content)? = result["system_content"] {
            result["system_content"] = .array(content.map { item in
                guard case .object(var object) = item,
                      object["type"]?.stringValue == "audio",
                      let path = object["path"]?.stringValue else { return item }
                object["data"] = .null
                object["name"] = .string(URL(fileURLWithPath: path).lastPathComponent)
                object["duration"] = .number(audioDuration(path: path, projectRoot: projectRoot))
                return .object(object)
            })
        }
        _ = projectRoot
        return result
    }

    private static func audioDuration(path: String, projectRoot: URL) -> Double {
        guard !path.hasPrefix("/"),
              !path.split(separator: "/").contains("..") else { return 0 }
        let file = projectRoot.appendingPathComponent(path)
        let base = projectRoot.resolvingSymlinksInPath().standardizedFileURL.path
        let resolved = file.resolvingSymlinksInPath().standardizedFileURL.path
        guard resolved == base || resolved.hasPrefix(base + "/") else { return 0 }
        guard (try? file.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile) == true else { return 0 }
        guard let data = try? Data(contentsOf: file),
              let normalized = try? MiniCPMDemoAudioNormalizer.normalize(
                  data: data,
                  mimeType: mimeType(for: file.pathExtension),
                  fileExtension: file.pathExtension) else { return 0 }
        return normalized.durationSeconds
    }

    private static func parseYAML(_ text: String) -> [String: MiniCPMJSONValue]? {
        let lines = text.replacingOccurrences(of: "\r\n", with: "\n").split(separator: "\n", omittingEmptySubsequences: false).map(String.init)
        var result: [String: MiniCPMJSONValue] = [:]
        var index = 0
        while index < lines.count {
            let line = lines[index]
            index += 1
            guard !line.trimmingCharacters(in: .whitespaces).isEmpty,
                  !line.trimmingCharacters(in: .whitespaces).hasPrefix("#"),
                  leadingSpaces(line) == 0,
                  let (key, value) = splitKeyValue(line) else { continue }
            if value == "|" || value == ">" {
                let (block, next) = readBlock(lines, from: index)
                index = next
                result[key] = .string(value == ">" ? block.replacingOccurrences(of: "\n", with: " ") : block)
            } else if key == "system_content" {
                let (items, next) = readSystemContent(lines, from: index)
                index = next
                result[key] = .array(items)
            } else {
                result[key] = scalar(value)
            }
        }
        return result.isEmpty ? nil : result
    }

    private static func readSystemContent(_ lines: [String], from start: Int) -> ([MiniCPMJSONValue], Int) {
        var items: [MiniCPMJSONValue] = []
        var index = start
        while index < lines.count {
            let line = lines[index]
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            guard !trimmed.isEmpty else { index += 1; continue }
            let indent = leadingSpaces(line)
            guard indent > 0, trimmed.hasPrefix("-") else { break }
            var object: [String: MiniCPMJSONValue] = [:]
            let first = String(trimmed.dropFirst()).trimmingCharacters(in: .whitespaces)
            if let pair = splitKeyValue(first) { object[pair.0] = scalar(pair.1) }
            index += 1
            while index < lines.count {
                let child = lines[index]
                let childTrimmed = child.trimmingCharacters(in: .whitespaces)
                let childIndent = leadingSpaces(child)
                guard !childTrimmed.isEmpty, childIndent > indent, !childTrimmed.hasPrefix("-") else { break }
                guard let pair = splitKeyValue(childTrimmed) else { index += 1; continue }
                index += 1
                if pair.1 == "|" || pair.1 == ">" {
                    let (block, next) = readBlock(lines, from: index)
                    index = next
                    object[pair.0] = .string(pair.1 == ">" ? block.replacingOccurrences(of: "\n", with: " ") : block)
                } else {
                    object[pair.0] = scalar(pair.1)
                }
            }
            items.append(.object(object))
        }
        return (items, index)
    }

    private static func readBlock(_ lines: [String], from start: Int) -> (String, Int) {
        var index = start
        var values: [String] = []
        var minimumIndent: Int?
        while index < lines.count {
            let line = lines[index]
            if !line.trimmingCharacters(in: .whitespaces).isEmpty {
                let indent = leadingSpaces(line)
                guard indent > 0 else { break }
                minimumIndent = min(minimumIndent ?? indent, indent)
            }
            values.append(line)
            index += 1
        }
        let strip = minimumIndent ?? 0
        return (values.map { String($0.dropFirst(min(strip, $0.count))) }.joined(separator: "\n").trimmingCharacters(in: .newlines), index)
    }

    private static func splitKeyValue(_ line: String) -> (String, String)? {
        guard let colon = line.firstIndex(of: ":") else { return nil }
        let key = line[..<colon].trimmingCharacters(in: .whitespaces)
        let value = line[line.index(after: colon)...].trimmingCharacters(in: .whitespaces)
        guard !key.isEmpty else { return nil }
        return (key, value)
    }

    private static func scalar(_ raw: String) -> MiniCPMJSONValue {
        let value = raw.trimmingCharacters(in: .whitespaces)
        if value == "null" || value == "~" { return .null }
        if value.lowercased() == "true" { return .bool(true) }
        if value.lowercased() == "false" { return .bool(false) }
        if let number = Double(value) { return .number(number) }
        if value.hasPrefix("\"") && value.hasSuffix("\"") || value.hasPrefix("'") && value.hasSuffix("'") {
            return .string(String(value.dropFirst().dropLast()))
        }
        return .string(value)
    }

    private static func leadingSpaces(_ line: String) -> Int {
        line.prefix { $0 == " " }.count
    }

    private static func mimeType(for ext: String) -> String {
        switch ext.lowercased() {
        case "wav": return "audio/wav"
        case "mp3": return "audio/mpeg"
        case "jpg", "jpeg": return "image/jpeg"
        case "png": return "image/png"
        default: return "application/octet-stream"
        }
    }
}
