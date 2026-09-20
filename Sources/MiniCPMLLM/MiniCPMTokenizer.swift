import Foundation
import Hub
import Tokenizers

/// Every control token used by MiniCPM-o's turn/duplex protocol.
///
/// The ids are resolved from the loaded Hugging Face tokenizer at startup;
/// they are not copied from a particular checkpoint.  This matters for
/// converted bundles, where a vocabulary can be identical while the added
/// token table is reordered.  Keeping the complete table here also prevents
/// the language model and the official duplex coordinator from silently using
/// different ids for the same marker.
public struct MiniCPMSpecialTokenIDs: Sendable, Equatable {
    public let bos: Int
    public let eos: Int
    public let pad: Int
    public let unknown: Int

    public let unit: Int
    public let unitEnd: Int
    public let listen: Int
    public let speak: Int
    public let interrupt: Int
    public let ttsBOS: Int
    public let ttsEOS: Int
    public let ttsPad: Int
    public let turnBOS: Int
    public let turnEOS: Int
    public let chunkBOS: Int
    public let chunkEOS: Int
    public let chunkTTSBOS: Int
    public let chunkTTSEOS: Int

    public let audioStart: Int
    public let audio: Int
    public let audioEnd: Int
    public let speakerBOS: Int
    public let speaker: Int
    public let speakerEOS: Int

    public let vadStart: Int
    public let vadEnd: Int
    public let visionStart: Int?
    public let visionEnd: Int?
    public let image: Int?
    public let imageEnd: Int?
    public let imagePad: Int?
    public let slice: Int?
    public let sliceEnd: Int?

    public init(
        bos: Int,
        eos: Int,
        pad: Int,
        unknown: Int,
        unit: Int,
        unitEnd: Int,
        listen: Int,
        speak: Int,
        interrupt: Int,
        ttsBOS: Int,
        ttsEOS: Int,
        ttsPad: Int,
        turnBOS: Int,
        turnEOS: Int,
        chunkBOS: Int,
        chunkEOS: Int,
        chunkTTSBOS: Int,
        chunkTTSEOS: Int,
        audioStart: Int,
        audio: Int,
        audioEnd: Int,
        speakerBOS: Int,
        speaker: Int,
        speakerEOS: Int,
        vadStart: Int,
        vadEnd: Int,
        visionStart: Int?,
        visionEnd: Int?,
        image: Int?,
        imageEnd: Int?,
        imagePad: Int?,
        slice: Int?,
        sliceEnd: Int?
    ) {
        self.bos = bos
        self.eos = eos
        self.pad = pad
        self.unknown = unknown
        self.unit = unit
        self.unitEnd = unitEnd
        self.listen = listen
        self.speak = speak
        self.interrupt = interrupt
        self.ttsBOS = ttsBOS
        self.ttsEOS = ttsEOS
        self.ttsPad = ttsPad
        self.turnBOS = turnBOS
        self.turnEOS = turnEOS
        self.chunkBOS = chunkBOS
        self.chunkEOS = chunkEOS
        self.chunkTTSBOS = chunkTTSBOS
        self.chunkTTSEOS = chunkTTSEOS
        self.audioStart = audioStart
        self.audio = audio
        self.audioEnd = audioEnd
        self.speakerBOS = speakerBOS
        self.speaker = speaker
        self.speakerEOS = speakerEOS
        self.vadStart = vadStart
        self.vadEnd = vadEnd
        self.visionStart = visionStart
        self.visionEnd = visionEnd
        self.image = image
        self.imageEnd = imageEnd
        self.imagePad = imagePad
        self.slice = slice
        self.sliceEnd = sliceEnd
    }
}

public enum MiniCPMTokenizerError: Error, LocalizedError, Sendable {
    case missingSpecialToken(String)
    case invalidTokenizer(String)

    public var errorDescription: String? {
        switch self {
        case .missingSpecialToken(let token):
            return "MiniCPM tokenizer is missing required special token: \(token)"
        case .invalidTokenizer(let message):
            return "Invalid MiniCPM tokenizer: \(message)"
        }
    }
}

/// Exact MiniCPM-o tokenizer adapter backed by swift-transformers.
///
/// ``ChatTokenizer`` in the Qwen3 chat target is intentionally a small
/// fallback implementation.  It is useful for lightweight demos but cannot
/// guarantee byte-level BPE parity for CJK, combining marks, or emoji.  The
/// production duplex path uses this wrapper around the upstream
/// ``AutoTokenizer`` implementation instead.  Added/special tokens are
/// recognized inside ordinary text, and all chat-template rendering is done
/// by the same Jinja implementation used by the rest of swift-transformers.
public final class MiniCPMTokenizer: @unchecked Sendable {
    private let tokenizer: any Tokenizers.Tokenizer
    public let specialTokens: MiniCPMSpecialTokenIDs
    /// Token ids marked as invalid by MiniCPM-o's `tokenizer_config.json`.
    /// They include punctuation/control pieces that the official streaming
    /// decoder masks before its final sample.  The list is kept data-driven
    /// because it is large and can change between converted bundles.
    public let badTokenIds: [Int]

    public var bosTokenId: Int { specialTokens.bos }
    public var eosTokenId: Int { specialTokens.eos }
    public var padTokenId: Int { specialTokens.pad }
    public var unknownTokenId: Int { specialTokens.unknown }
    public var hasChatTemplate: Bool { tokenizer.hasChatTemplate }

    public init(tokenizer: any Tokenizers.Tokenizer, badTokenIds: [Int] = []) throws {
        self.tokenizer = tokenizer
        self.badTokenIds = Array(Set(badTokenIds.filter { $0 >= 0 })).sorted()

        func required(_ token: String) throws -> Int {
            guard let id = tokenizer.convertTokenToId(token), id >= 0,
                  tokenizer.convertIdToToken(id) == token else {
                throw MiniCPMTokenizerError.missingSpecialToken(token)
            }
            return id
        }
        func optional(_ token: String) -> Int? {
            guard let id = tokenizer.convertTokenToId(token), id >= 0,
                  tokenizer.convertIdToToken(id) == token else { return nil }
            return id
        }

        // `Tokenizer` intentionally does not expose padToken (it is not a
        // universal property across decoder-only models); MiniCPM-o declares
        // `<|endoftext|>` as its pad token in tokenizer_config.json.
        guard let bos = tokenizer.bosTokenId,
              let eos = tokenizer.eosTokenId,
              let pad = tokenizer.convertTokenToId("<|endoftext|>"),
              tokenizer.convertIdToToken(pad) == "<|endoftext|>",
              let unknown = tokenizer.unknownTokenId
        else {
            throw MiniCPMTokenizerError.invalidTokenizer(
                "bos/eos/pad/unknown ids are not all defined")
        }

        // These markers form the official TDM protocol and are required even
        // for an audio-only runtime.  Vision/audio conditioning can then be
        // added without changing the protocol id table.
        specialTokens = MiniCPMSpecialTokenIDs(
            bos: bos,
            eos: eos,
            pad: pad,
            unknown: unknown,
            unit: try required("<unit>"),
            unitEnd: try required("</unit>"),
            listen: try required("<|listen|>"),
            speak: try required("<|speak|>"),
            interrupt: try required("<|interrupt|>"),
            ttsBOS: try required("<|tts_bos|>"),
            ttsEOS: try required("<|tts_eos|>"),
            ttsPad: try required("<|tts_pad|>"),
            turnBOS: try required("<|turn_bos|>"),
            turnEOS: try required("<|turn_eos|>"),
            chunkBOS: try required("<|chunk_bos|>"),
            chunkEOS: try required("<|chunk_eos|>"),
            chunkTTSBOS: try required("<|chunk_tts_bos|>"),
            chunkTTSEOS: try required("<|chunk_tts_eos|>"),
            audioStart: try required("<|audio_start|>"),
            audio: try required("<|audio|>"),
            audioEnd: try required("<|audio_end|>"),
            speakerBOS: try required("<|spk_bos|>"),
            speaker: try required("<|spk|>"),
            speakerEOS: try required("<|spk_eos|>"),
            vadStart: try required("<|vad_start|>"),
            vadEnd: try required("<|vad_end|>"),
            visionStart: optional("<|vision_start|>"),
            visionEnd: optional("<|vision_end|>"),
            image: optional("<image>"),
            imageEnd: optional("</image>"),
            imagePad: optional("<|image_pad|>"),
            slice: optional("<slice>"),
            sliceEnd: optional("</slice>"))
    }

    /// Load a local bundle.  ``AutoTokenizer`` also merges a sibling
    /// ``chat_template.jinja``/``chat_template.json`` when present.
    public static func load(from directory: URL, strict: Bool = false) async throws -> MiniCPMTokenizer {
        let tokenizer = try await AutoTokenizer.from(modelFolder: directory, strict: strict)
        let badTokenIds = try readBadTokenIds(from: directory)
        return try MiniCPMTokenizer(tokenizer: tokenizer, badTokenIds: badTokenIds)
    }

    /// Synchronous local loader for command-line tools that already run off
    /// the async coordinator.  This path intentionally reads only local files
    /// and never contacts the Hub.
    public static func loadSynchronously(from directory: URL, strict: Bool = false) throws -> MiniCPMTokenizer {
        let dataURL = directory.appendingPathComponent("tokenizer.json")
        let configURL = directory.appendingPathComponent("tokenizer_config.json")
        guard FileManager.default.fileExists(atPath: dataURL.path) else {
            throw MiniCPMTokenizerError.invalidTokenizer(
                "tokenizer.json not found at \(dataURL.path)")
        }
        let hub = HubApi()
        var config = FileManager.default.fileExists(atPath: configURL.path)
            ? try hub.configuration(fileURL: configURL)
            : Config([String: Config]())

        // Keep parity with AutoTokenizer.from(modelFolder:) for converted
        // bundles that store the template in a sidecar file.
        let jinjaURL = directory.appendingPathComponent("chat_template.jinja")
        if FileManager.default.fileExists(atPath: jinjaURL.path),
           let template = try? String(contentsOf: jinjaURL, encoding: .utf8),
           var dictionary = config.dictionary() {
            dictionary["chat_template"] = Config(template)
            config = Config(dictionary)
        } else {
            let jsonURL = directory.appendingPathComponent("chat_template.json")
            if FileManager.default.fileExists(atPath: jsonURL.path),
               let template = try? hub.configuration(fileURL: jsonURL)["chat_template"].string(),
               var dictionary = config.dictionary() {
                dictionary["chat_template"] = Config(template)
                config = Config(dictionary)
            }
        }
        let tokenizerData = try hub.configuration(fileURL: dataURL)
        let tokenizer = try AutoTokenizer.from(
            tokenizerConfig: config,
            tokenizerData: tokenizerData,
            strict: strict)
        return try MiniCPMTokenizer(
            tokenizer: tokenizer,
            badTokenIds: badTokenIds(from: config))
    }

    /// The complete set of special/control ids known to this tokenizer.  A
    /// duplex sampler can pass this set as
    /// `MiniCPMProtocolTokenIds.allSpecialTokenIds` to keep protocol markers
    /// out of text repetition history.
    public var allSpecialTokenIds: Set<Int> {
        var ids: Set<Int> = [
            specialTokens.bos, specialTokens.eos, specialTokens.pad,
            specialTokens.unknown, specialTokens.unit, specialTokens.unitEnd,
            specialTokens.listen, specialTokens.speak, specialTokens.interrupt,
            specialTokens.ttsBOS, specialTokens.ttsEOS, specialTokens.ttsPad,
            specialTokens.turnBOS, specialTokens.turnEOS,
            specialTokens.chunkBOS, specialTokens.chunkEOS,
            specialTokens.chunkTTSBOS, specialTokens.chunkTTSEOS,
            specialTokens.audioStart, specialTokens.audio, specialTokens.audioEnd,
            specialTokens.speakerBOS, specialTokens.speaker, specialTokens.speakerEOS,
            specialTokens.vadStart, specialTokens.vadEnd,
        ]
        for id in [
            specialTokens.visionStart, specialTokens.visionEnd,
            specialTokens.image, specialTokens.imageEnd, specialTokens.imagePad,
            specialTokens.slice, specialTokens.sliceEnd,
        ].compactMap({ $0 }) {
            ids.insert(id)
        }
        return ids
    }

    private static func readBadTokenIds(from directory: URL) throws -> [Int] {
        let configURL = directory.appendingPathComponent("tokenizer_config.json")
        guard FileManager.default.fileExists(atPath: configURL.path) else { return [] }
        let config = try HubApi().configuration(fileURL: configURL)
        return badTokenIds(from: config)
    }

    private static func badTokenIds(from config: Config) -> [Int] {
        config[Config.Key("bad_token_ids")].array()?.compactMap { $0.integer() } ?? []
    }

    /// Resolve an exact token id without exposing the backend's sentinel
    /// behavior for unknown tokens.
    public func tokenId(_ token: String) -> Int? {
        guard let id = tokenizer.convertTokenToId(token),
              tokenizer.convertIdToToken(id) == token else { return nil }
        return id
    }

    public func tokenString(_ id: Int) -> String? {
        tokenizer.convertIdToToken(id)
    }

    /// Encode with no implicit BOS/EOS.  MiniCPM's Python tokenizer uses this
    /// mode for protocol fragments and chat-template output.
    public func encode(_ text: String, addSpecialTokens: Bool = false) -> [Int] {
        tokenizer.encode(text: text, addSpecialTokens: addSpecialTokens)
    }

    public func decode(_ ids: [Int], skipSpecialTokens: Bool = false) -> String {
        tokenizer.decode(tokens: ids, skipSpecialTokens: skipSpecialTokens)
    }

    public func applyChatTemplate(
        messages: [Message],
        addGenerationPrompt: Bool = true,
        enableThinking: Bool? = nil,
        useTTSTemplate: Bool = false,
        tools: [ToolSpec]? = nil,
        truncation: Bool = false,
        maxLength: Int? = nil
    ) throws -> [Int] {
        var context: [String: any Sendable] = [:]
        if let enableThinking { context["enable_thinking"] = enableThinking }
        if useTTSTemplate { context["use_tts_template"] = true }
        return try tokenizer.applyChatTemplate(
            messages: messages,
            chatTemplate: nil,
            addGenerationPrompt: addGenerationPrompt,
            truncation: truncation,
            maxLength: maxLength,
            tools: tools,
            additionalContext: context.isEmpty ? nil : context)
    }

    /// Render the same template as text.  Decoding with special tokens kept
    /// preserves protocol markers exactly and is useful for parity diagnostics.
    public func renderChatTemplate(
        messages: [Message],
        addGenerationPrompt: Bool = true,
        enableThinking: Bool? = nil,
        useTTSTemplate: Bool = false,
        tools: [ToolSpec]? = nil
    ) throws -> String {
        let ids = try applyChatTemplate(
            messages: messages,
            addGenerationPrompt: addGenerationPrompt,
            enableThinking: enableThinking,
            useTTSTemplate: useTTSTemplate,
            tools: tools)
        return decode(ids, skipSpecialTokens: false)
    }
}
