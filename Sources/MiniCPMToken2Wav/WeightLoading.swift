import Foundation
import MLX
import MLXCommon
import MLXNN

extension MiniCPMHiFTGenerator {
    public static func fromDirectory(_ directory: URL) throws -> MiniCPMHiFTGenerator {
        let weightsURL = directory.appending(path: "hift.safetensors")
        guard FileManager.default.fileExists(atPath: weightsURL.path) else {
            throw MiniCPMToken2WavError.missingModelFile(weightsURL)
        }
        let model = MiniCPMHiFTGenerator()
        try model.loadWeights(from: weightsURL)
        return model
    }

    public func loadWeights(from url: URL) throws {
        let weights = try CommonWeightLoader.loadSafetensors(url: url)
        let modelKeys = Set(parameters().flattened().map(\.0))
        var remapped: [String: MLXArray] = [:]
        remapped.reserveCapacity(weights.count)
        for (name, tensor) in weights {
            var candidates = [name]
            if name.hasPrefix("source_downs.") {
                candidates.append("sourceDowns." + name.dropFirst("source_downs.".count))
            } else if name.hasPrefix("source_resblocks.") {
                candidates.append("sourceResidualBlocks."
                    + name.dropFirst("source_resblocks.".count))
            }
            candidates += candidates.map {
                $0.replacingOccurrences(of: ".convs1.", with: ".convolutions1.")
                    .replacingOccurrences(of: ".convs2.", with: ".convolutions2.")
            }
            candidates += candidates.map {
                $0.hasSuffix(".alpha") ? String($0.dropLast("alpha".count)) + "_alpha" : $0
            }
            guard let target = candidates.first(where: modelKeys.contains) else {
                throw MiniCPMToken2WavError.invalidWeights(
                    "no model parameter matches \(name)")
            }
            remapped[target] = tensor
        }

        do {
            try update(parameters: ModuleParameters.unflattened(remapped), verify: .all)
        } catch {
            throw MiniCPMToken2WavError.invalidWeights(error.localizedDescription)
        }
    }
}

extension MiniCPMToken2WavFlowModel {
    public static func fromDirectory(
        _ directory: URL,
        configuration: MiniCPMFlowConfiguration = .init()
    ) throws -> MiniCPMToken2WavFlowModel {
        let weightsURL = directory.appending(path: "flow.safetensors")
        guard FileManager.default.fileExists(atPath: weightsURL.path) else {
            throw MiniCPMToken2WavError.missingModelFile(weightsURL)
        }
        let model = MiniCPMToken2WavFlowModel(configuration: configuration)
        try model.loadWeights(from: weightsURL)
        return model
    }

    public func loadWeights(from url: URL) throws {
        let weights = try CommonWeightLoader.loadSafetensors(url: url)
        let modelKeys = Set(parameters().flattened().map(\.0))
        var remapped: [String: MLXArray] = [:]
        remapped.reserveCapacity(weights.count)

        for (name, tensor) in weights {
            var target = name
            target = target
                .replacingOccurrences(of: "encoder.embed.out.0.", with: "encoder.embed.linear.")
                .replacingOccurrences(of: "encoder.embed.out.1.", with: "encoder.embed.normalization.")
                .replacingOccurrences(of: "encoder.up_embed.out.0.", with: "encoder.up_embed.linear.")
                .replacingOccurrences(of: "encoder.up_embed.out.1.", with: "encoder.up_embed.normalization.")
                .replacingOccurrences(of: ".t_embedder.mlp.0.", with: ".t_embedder.linear1.")
                .replacingOccurrences(of: ".t_embedder.mlp.2.", with: ".t_embedder.linear2.")
                .replacingOccurrences(of: ".adaLN_modulation.1.", with: ".adaLN_modulation.")
                .replacingOccurrences(of: ".conv.block.1.", with: ".conv.firstConv.")
                .replacingOccurrences(of: ".conv.block.3.", with: ".conv.normalization.")
                .replacingOccurrences(of: ".conv.block.6.", with: ".conv.secondConv.")

            var candidates = [target]
            candidates += candidates.map {
                $0.replacingOccurrences(of: "encoder.up_encoders.", with: "encoder.upsampleEncoders.")
                    .replacingOccurrences(of: "encoder.up_embed.", with: "encoder.upsampleEmbed.")
            }
            guard let resolved = candidates.first(where: modelKeys.contains) else {
                throw MiniCPMToken2WavError.invalidWeights(
                    "no flow model parameter matches \(name) (mapped as \(target))")
            }
            remapped[resolved] = tensor
        }

        do {
            try update(parameters: ModuleParameters.unflattened(remapped), verify: .all)
        } catch {
            throw MiniCPMToken2WavError.invalidWeights(error.localizedDescription)
        }
    }
}
