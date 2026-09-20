import Foundation
import Tokenizers

/// Roles accepted by the MiniCPM-o chat template.  The official template has
/// branches for tool messages as well, but the native turn/half-duplex surface
/// intentionally fails closed for those roles until a tool-call adapter is
/// added.  Silently rendering a tool message as user text changes the model's
/// conversation state.
public enum MiniCPMChatRole: String, Codable, Sendable, Equatable {
    case system
    case user
    case assistant
}

/// A stable handle for an audio/image/video value that is encoded outside the
/// tokenizer.  The handle is copied into an embedding slot and lets a caller
/// keep the actual PCM/image bytes in its own store (or in a Swift/MLX
/// encoder) without making the chat template depend on a media framework.
public struct MiniCPMExternalMedia: Sendable, Equatable, Codable {
    public enum Kind: String, Codable, Sendable {
        case audio
        case image
        case video
    }

    public let key: String
    public let kind: Kind
    public let sampleRate: Int?
    public let sampleCount: Int?

    public init(
        key: String,
        kind: Kind,
        sampleRate: Int? = nil,
        sampleCount: Int? = nil
    ) {
        self.key = key
        self.kind = kind
        self.sampleRate = sampleRate
        self.sampleCount = sampleCount
    }

    public static func audio(
        _ key: String,
        sampleRate: Int? = nil,
        sampleCount: Int? = nil
    ) -> MiniCPMExternalMedia {
        MiniCPMExternalMedia(
            key: key,
            kind: .audio,
            sampleRate: sampleRate,
            sampleCount: sampleCount)
    }

    public static func image(_ key: String) -> MiniCPMExternalMedia {
        MiniCPMExternalMedia(key: key, kind: .image)
    }

    public static func video(_ key: String) -> MiniCPMExternalMedia {
        MiniCPMExternalMedia(key: key, kind: .video)
    }
}

/// Ordered multimodal content.  Parts are deliberately not represented as a
/// `[String]`: the order is part of MiniCPM's input semantics and the media
/// value is replaced by an embedding slot only after the chat template has
/// rendered the surrounding role markers.
public enum MiniCPMChatContentPart: Sendable, Equatable, Codable {
    case text(String)
    case media(MiniCPMExternalMedia)

    public static func audio(
        _ key: String,
        sampleRate: Int? = nil,
        sampleCount: Int? = nil
    ) -> MiniCPMChatContentPart {
        .media(.audio(key, sampleRate: sampleRate, sampleCount: sampleCount))
    }

    public static func image(_ key: String) -> MiniCPMChatContentPart {
        .media(.image(key))
    }

    public static func video(_ key: String) -> MiniCPMChatContentPart {
        .media(.video(key))
    }

    private enum CodingKeys: String, CodingKey {
        case type
        case text
        case key
        case sampleRate = "sample_rate"
        case sampleCount = "sample_count"
    }

    // Tokenizers also exports a `Decoder` protocol.  Qualify the standard
    // library protocol here so Codable keeps using Swift's keyed containers.
    public init(from decoder: Swift.Decoder) throws {
        let values = try decoder.container(keyedBy: CodingKeys.self)
        let type = try values.decode(String.self, forKey: .type)
        switch type {
        case "text":
            self = .text(try values.decode(String.self, forKey: .text))
        case "audio", "image", "video":
            guard let key = try values.decodeIfPresent(String.self, forKey: .key),
                  !key.isEmpty
            else {
                throw MiniCPMChatTemplateError.invalidMedia("media key is empty")
            }
            let kind = MiniCPMExternalMedia.Kind(rawValue: type)
            guard let kind else {
                throw MiniCPMChatTemplateError.invalidMedia("unknown media kind: \(type)")
            }
            self = .media(MiniCPMExternalMedia(
                key: key,
                kind: kind,
                sampleRate: try values.decodeIfPresent(Int.self, forKey: .sampleRate),
                sampleCount: try values.decodeIfPresent(Int.self, forKey: .sampleCount)))
        default:
            throw MiniCPMChatTemplateError.invalidMedia("unknown content type: \(type)")
        }
    }

    public func encode(to encoder: Swift.Encoder) throws {
        var values = encoder.container(keyedBy: CodingKeys.self)
        switch self {
        case .text(let value):
            try values.encode("text", forKey: .type)
            try values.encode(value, forKey: .text)
        case .media(let media):
            try values.encode(media.kind.rawValue, forKey: .type)
            try values.encode(media.key, forKey: .key)
            try values.encodeIfPresent(media.sampleRate, forKey: .sampleRate)
            try values.encodeIfPresent(media.sampleCount, forKey: .sampleCount)
        }
    }
}

/// A role-bearing message used by the framework-neutral builder.
public struct MiniCPMChatMessage: Sendable, Equatable, Codable {
    public let role: MiniCPMChatRole
    public let parts: [MiniCPMChatContentPart]
    /// Optional Qwen3 reasoning content used when replaying an assistant
    /// message.  The upstream Jinja template keeps this separate from the
    /// visible assistant content.
    public let reasoningContent: String?

    public init(
        role: MiniCPMChatRole,
        parts: [MiniCPMChatContentPart],
        reasoningContent: String? = nil
    ) {
        self.role = role
        self.parts = parts
        self.reasoningContent = reasoningContent
    }

    public init(role: MiniCPMChatRole, text: String, reasoningContent: String? = nil) {
        self.init(role: role, parts: [.text(text)], reasoningContent: reasoningContent)
    }

    public var isEmpty: Bool {
        parts.isEmpty || parts.allSatisfy {
            switch $0 {
            case .text(let value): return value.isEmpty
            case .media(let media): return media.key.isEmpty
            }
        }
    }
}

public enum MiniCPMChatContentSeparator: Sendable, Equatable {
    case none
    case newline

    fileprivate var value: String {
        switch self {
        case .none: return ""
        case .newline: return "\n"
        }
    }
}

/// Options shared by turn-based chat and complete (non-streaming) prefill.
/// ``contentSeparator`` follows the upstream processor: non-omni chat joins
/// parts with a newline, while streaming/omni input uses no separator.
public struct MiniCPMChatTemplateOptions: Sendable, Equatable {
    public var addGenerationPrompt: Bool
    public var useTTSTemplate: Bool
    public var enableThinking: Bool
    public var contentSeparator: MiniCPMChatContentSeparator
    /// Audio content makes the upstream chat path enable the TTS template.
    /// Keep this on by default, while still allowing diagnostics to disable
    /// the implicit behavior.
    public var enableTTSTemplateForAudio: Bool

    public init(
        addGenerationPrompt: Bool = true,
        useTTSTemplate: Bool = false,
        enableThinking: Bool = false,
        contentSeparator: MiniCPMChatContentSeparator = .newline,
        enableTTSTemplateForAudio: Bool = true
    ) {
        self.addGenerationPrompt = addGenerationPrompt
        self.useTTSTemplate = useTTSTemplate
        self.enableThinking = enableThinking
        self.contentSeparator = contentSeparator
        self.enableTTSTemplateForAudio = enableTTSTemplateForAudio
    }
}

/// The two prompt forms used by upstream ``streaming_prefill``.
public enum MiniCPMStreamingPromptContext: Sendable, Equatable {
    /// First segment of the first user turn: no ``</unit>``/``<|im_end|>``
    /// is emitted because more audio segments are appended to this turn.
    case firstUser
    /// A subsequent segment of the same turn; only ordered content is fed.
    case continuation
    /// First segment of a new user turn.  ``completedResponse`` selects the
    /// upstream ``<|im_end|>`` versus ``<|tts_eos|><|im_end|>`` repair marker.
    case nextUser(completedResponse: Bool)
    /// System/assistant prefill uses the normal chat template without a
    /// generation prompt.
    case systemOrAssistant
}

public struct MiniCPMExternalEmbeddingSlot: Sendable, Equatable, Codable {
    public let index: Int
    public let media: MiniCPMExternalMedia
    public let role: MiniCPMChatRole
    public let messageIndex: Int
    public let partIndex: Int

    public init(
        index: Int,
        media: MiniCPMExternalMedia,
        role: MiniCPMChatRole,
        messageIndex: Int,
        partIndex: Int
    ) {
        self.index = index
        self.media = media
        self.role = role
        self.messageIndex = messageIndex
        self.partIndex = partIndex
    }
}

/// A token or an external embedding slot in exact source order.  The enum is
/// intentionally independent of MLX; a DuplexEngine can resolve slots to an
/// ``MLXArray`` (or a remote/Swift embedding) without changing template code.
public enum MiniCPMChatInputSegment: Sendable, Equatable {
    case tokens([Int])
    case embedding(MiniCPMExternalEmbeddingSlot)
}

public enum MiniCPMChatTemplateError: Error, LocalizedError, Sendable, Equatable {
    case emptyMessages
    case emptyMessage(Int)
    case unsupportedRole(String)
    case firstRoleMustBeSystemOrUser
    case invalidMedia(String)
    case mediaMarkerNotFound(Int)
    case mediaMarkerCollision
    case missingImageMarkers
    case missingAudioMarkers
    case missingChatTemplate

    public var errorDescription: String? {
        switch self {
        case .emptyMessages:
            return "MiniCPM chat template requires at least one message"
        case .emptyMessage(let index):
            return "MiniCPM chat message \(index) has no content"
        case .unsupportedRole(let role):
            return "MiniCPM chat template does not support role: \(role)"
        case .firstRoleMustBeSystemOrUser:
            return "MiniCPM chat template requires the first role to be system or user"
        case .invalidMedia(let message):
            return "Invalid MiniCPM media content: \(message)"
        case .mediaMarkerNotFound(let index):
            return "MiniCPM chat template lost external media marker \(index)"
        case .mediaMarkerCollision:
            return "MiniCPM chat template media marker collides with message text"
        case .missingImageMarkers:
            return "MiniCPM tokenizer has no image marker tokens"
        case .missingAudioMarkers:
            return "MiniCPM tokenizer has no audio marker tokens"
        case .missingChatTemplate:
            return "MiniCPM tokenizer has no chat template"
        }
    }
}

/// The output of ``MiniCPMChatTemplateBuilder``.  ``segments`` is the only
/// representation that should be fed to a model: text is tokenized, while
/// every media part remains a separately addressable slot in source order.
public struct MiniCPMChatInputPlan: Sendable, Equatable {
    public let segments: [MiniCPMChatInputSegment]
    public let slots: [MiniCPMExternalEmbeddingSlot]
    /// Canonical prompt text with ``<audio>./</audio>`` and
    /// ``<image>./</image>`` placeholders.  This is useful for diagnostics
    /// and mirrors the upstream processor's input string.
    public let promptText: String
    public let messages: [MiniCPMChatMessage]

    public init(
        segments: [MiniCPMChatInputSegment],
        slots: [MiniCPMExternalEmbeddingSlot],
        promptText: String,
        messages: [MiniCPMChatMessage]
    ) {
        self.segments = segments
        self.slots = slots
        self.promptText = promptText
        self.messages = messages
    }

    public var tokenCount: Int {
        segments.reduce(into: 0) { result, segment in
            if case .tokens(let values) = segment { result += values.count }
        }
    }

    /// Resolve each slot lazily.  No MLX/Torch type appears in this API; the
    /// caller chooses its embedding representation and can stream or cache it
    /// independently of the prompt/tokenizer.
    public func resolve<Embedding>(
        _ provider: (MiniCPMExternalEmbeddingSlot) throws -> Embedding
    ) rethrows -> [MiniCPMResolvedInputSegment<Embedding>] {
        try segments.map { segment in
            switch segment {
            case .tokens(let values): return .tokens(values)
            case .embedding(let slot): return .embedding(slot, try provider(slot))
            }
        }
    }
}

public enum MiniCPMResolvedInputSegment<Embedding> {
    case tokens([Int])
    case embedding(MiniCPMExternalEmbeddingSlot, Embedding)
}

/// Native renderer for the pinned MiniCPM-o chat template.  It delegates
/// role/assistant-reasoning/thinking/TTS formatting to the loaded tokenizer's
/// Jinja template and only substitutes media sentinels after rendering.  This
/// is what preserves the exact upstream whitespace and generation prompt while
/// keeping multimodal parts ordered.
public final class MiniCPMChatTemplateBuilder: @unchecked Sendable {
    public let tokenizer: MiniCPMTokenizer

    public init(tokenizer: MiniCPMTokenizer) {
        self.tokenizer = tokenizer
    }

    public func build(
        messages: [MiniCPMChatMessage],
        options: MiniCPMChatTemplateOptions = .init()
    ) throws -> MiniCPMChatInputPlan {
        try validate(messages)
        let state = try prepareMessages(messages, separator: options.contentSeparator)
        return try makeCompletePlan(
            messages: messages,
            state: state,
            options: options)
    }

    private func makeCompletePlan(
        messages: [MiniCPMChatMessage],
        state: PreparedState,
        options: MiniCPMChatTemplateOptions
    ) throws -> MiniCPMChatInputPlan {
        let effectiveTTS = options.useTTSTemplate
            || (options.enableTTSTemplateForAudio && state.hasAudio)
        let rendered = try render(
            state.templateMessages,
            addGenerationPrompt: options.addGenerationPrompt,
            enableThinking: options.enableThinking,
            useTTSTemplate: effectiveTTS)
        return try makePlan(
            rendered: rendered,
            messages: messages,
            prepared: state.preparedParts,
            slots: state.slots,
            templateOptions: options,
            effectiveTTS: effectiveTTS)
    }

    /// Build one streaming prefill segment using the exact prefix rules in
    /// ``MiniCPMO45/modeling_minicpmo.py``.  Unlike ``build``, user segments
    /// intentionally do not close the ``<|im_start|>user`` block.
    public func buildStreaming(
        message: MiniCPMChatMessage,
        context: MiniCPMStreamingPromptContext,
        separator: MiniCPMChatContentSeparator = .none,
        enableThinking: Bool = false,
        useTTSTemplate: Bool = true
    ) throws -> MiniCPMChatInputPlan {
        try validate([message], allowFirstAssistant: context == .systemOrAssistant && message.role == .assistant)
        let state = try prepareMessages([message], separator: separator)
        let content = state.preparedParts.first?.textWithMarkers ?? ""
        let prefix: String
        switch context {
        case .firstUser:
            guard message.role == .user else {
                throw MiniCPMChatTemplateError.unsupportedRole(message.role.rawValue)
            }
            prefix = "<|im_start|>user\n"
        case .continuation:
            prefix = ""
        case .nextUser(let completedResponse):
            guard message.role == .user else {
                throw MiniCPMChatTemplateError.unsupportedRole(message.role.rawValue)
            }
            prefix = completedResponse
                ? "<|im_end|>\n<|im_start|>user\n"
                : "<|tts_eos|><|im_end|>\n<|im_start|>user\n"
        case .systemOrAssistant:
            return try makeCompletePlan(
                messages: [message],
                state: state,
                options: MiniCPMChatTemplateOptions(
                    addGenerationPrompt: false,
                    useTTSTemplate: useTTSTemplate,
                    enableThinking: enableThinking,
                    contentSeparator: separator,
                    enableTTSTemplateForAudio: false))
        }

        let rendered = prefix + content
        // User streaming segments never use the assistant TTS generation
        // suffix.  ``makePlan`` still emits marker tokens around each slot.
        return try makePlan(
            rendered: rendered,
            messages: [message],
            prepared: state.preparedParts,
            slots: state.slots,
            templateOptions: MiniCPMChatTemplateOptions(
                addGenerationPrompt: false,
                useTTSTemplate: false,
                enableThinking: enableThinking,
                contentSeparator: separator,
                enableTTSTemplateForAudio: false),
            effectiveTTS: false)
    }

    private struct PreparedPart {
        let messageIndex: Int
        let role: MiniCPMChatRole
        let parts: [MiniCPMChatContentPart]
        let textWithMarkers: String
    }

    private struct PreparedState {
        let templateMessages: [Message]
        let preparedParts: [PreparedPart]
        let slots: [MiniCPMExternalEmbeddingSlot]
        let hasAudio: Bool
    }

    private func validate(
        _ messages: [MiniCPMChatMessage],
        allowFirstAssistant: Bool = false
    ) throws {
        guard !messages.isEmpty else { throw MiniCPMChatTemplateError.emptyMessages }
        guard messages[0].role == .system
            || messages[0].role == .user
            || (allowFirstAssistant && messages[0].role == .assistant)
        else {
            throw MiniCPMChatTemplateError.firstRoleMustBeSystemOrUser
        }
        for (index, message) in messages.enumerated() {
            guard !message.isEmpty else { throw MiniCPMChatTemplateError.emptyMessage(index) }
            for part in message.parts {
                if case .media(let media) = part, media.key.isEmpty {
                    throw MiniCPMChatTemplateError.invalidMedia("empty key at message \(index)")
                }
            }
        }
    }

    private func prepareMessages(
        _ messages: [MiniCPMChatMessage],
        separator: MiniCPMChatContentSeparator
    ) throws -> PreparedState {
        var templateMessages: [Message] = []
        var preparedParts: [PreparedPart] = []
        var slots: [MiniCPMExternalEmbeddingSlot] = []
        var hasAudio = false
        var markerIndex = 0

        for (messageIndex, message) in messages.enumerated() {
            var pieces: [String] = []
            for (partIndex, part) in message.parts.enumerated() {
                switch part {
                case .text(let value):
                    pieces.append(value)
                case .media(let media):
                    let marker = Self.marker(markerIndex)
                    // A marker collision would make a user's literal text
                    // look like a model input slot after template rendering.
                    if messages.contains(where: { item in
                        item.parts.contains { part in
                            if case .text(let value) = part { return value.contains(marker) }
                            return false
                        }
                    }) {
                        throw MiniCPMChatTemplateError.mediaMarkerCollision
                    }
                    pieces.append(marker)
                    slots.append(MiniCPMExternalEmbeddingSlot(
                        index: markerIndex,
                        media: media,
                        role: message.role,
                        messageIndex: messageIndex,
                        partIndex: partIndex))
                    markerIndex += 1
                    hasAudio = hasAudio || media.kind == .audio
                }
            }
            let text = pieces.joined(separator: separator.value)
            templateMessages.append(Self.messageDictionary(
                role: message.role,
                content: text,
                reasoningContent: message.reasoningContent))
            preparedParts.append(PreparedPart(
                messageIndex: messageIndex,
                role: message.role,
                parts: message.parts,
                textWithMarkers: text))
        }
        return PreparedState(
            templateMessages: templateMessages,
            preparedParts: preparedParts,
            slots: slots,
            hasAudio: hasAudio)
    }

    private func render(
        _ messages: [Message],
        addGenerationPrompt: Bool,
        enableThinking: Bool,
        useTTSTemplate: Bool
    ) throws -> String {
        guard tokenizer.hasChatTemplate else {
            throw MiniCPMChatTemplateError.missingChatTemplate
        }
        return try tokenizer.renderChatTemplate(
            messages: messages,
            addGenerationPrompt: addGenerationPrompt,
            enableThinking: enableThinking,
            useTTSTemplate: useTTSTemplate)
    }

    private func makePlan(
        rendered: String,
        messages: [MiniCPMChatMessage],
        prepared: [PreparedPart],
        slots: [MiniCPMExternalEmbeddingSlot],
        templateOptions: MiniCPMChatTemplateOptions,
        effectiveTTS: Bool
    ) throws -> MiniCPMChatInputPlan {
        var segments: [MiniCPMChatInputSegment] = []
        var cursor = rendered.startIndex
        var promptText = rendered

        for slot in slots {
            let marker = Self.marker(slot.index)
            guard let range = rendered.range(of: marker, range: cursor..<rendered.endIndex) else {
                throw MiniCPMChatTemplateError.mediaMarkerNotFound(slot.index)
            }
            let prefix = String(rendered[cursor..<range.lowerBound])
            if !prefix.isEmpty {
                segments.append(.tokens(tokenizer.encode(prefix)))
            }
            let wrapper = try markerTokens(for: slot.media.kind)
            segments.append(.tokens(wrapper.0))
            segments.append(.embedding(slot))
            segments.append(.tokens(wrapper.1))
            promptText = promptText.replacingOccurrences(of: marker, with: wrapper.2)
            cursor = range.upperBound
        }
        let suffix = String(rendered[cursor..<rendered.endIndex])
        if !suffix.isEmpty {
            segments.append(.tokens(tokenizer.encode(suffix)))
        }

        // ``prepared`` is intentionally consumed here even for the no-slot
        // case: the message array is retained in the plan so callers can
        // reconstruct media ordering after a snapshot/rollback.
        _ = prepared
        _ = templateOptions
        _ = effectiveTTS
        return MiniCPMChatInputPlan(
            segments: segments,
            slots: slots,
            promptText: promptText,
            messages: messages)
    }

    private func markerTokens(
        for kind: MiniCPMExternalMedia.Kind
    ) throws -> (before: [Int], after: [Int], placeholder: String) {
        switch kind {
        case .audio:
            return (
                [tokenizer.specialTokens.audioStart],
                [tokenizer.specialTokens.audioEnd],
                "<audio>./</audio>")
        case .image, .video:
            guard let image = tokenizer.specialTokens.image,
                  let imageEnd = tokenizer.specialTokens.imageEnd
            else { throw MiniCPMChatTemplateError.missingImageMarkers }
            return ([image], [imageEnd], "<image>./</image>")
        }
    }

    private static func marker(_ index: Int) -> String {
        "MINICPMEXTERNALSLOT\(index)END"
    }

    private static func messageDictionary(
        role: MiniCPMChatRole,
        content: String,
        reasoningContent: String?
    ) -> Message {
        var value: Message = [
            "role": role.rawValue,
            "content": content,
        ]
        if let reasoningContent {
            value["reasoning_content"] = reasoningContent
        }
        return value
    }
}

/// Short alias used by callers that think in terms of input construction
/// rather than Jinja rendering.
public typealias MiniCPMChatInputBuilder = MiniCPMChatTemplateBuilder
