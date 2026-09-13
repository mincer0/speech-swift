import Foundation
import MLX
import MLXNN

extension MiniCPMAudioModel {
    public static let configurationFile = "config.json"
    public static let weightsFile = "audio_encoder.safetensors"

    /// Load an audio-only bundle produced by
    /// `scripts/convert_minicpm_o_audio_to_mlx.py`.
    public static func fromDirectory(_ directory: URL) throws -> MiniCPMAudioModel {
        let configURL = directory.appendingPathComponent(configurationFile)
        let weightsURL = directory.appendingPathComponent(weightsFile)
        guard FileManager.default.fileExists(atPath: configURL.path) else {
            throw MiniCPMAudioError.missingModelFile(configurationFile)
        }
        guard FileManager.default.fileExists(atPath: weightsURL.path) else {
            throw MiniCPMAudioError.missingModelFile(weightsFile)
        }

        let configuration = try JSONDecoder().decode(
            MiniCPMAudioConfiguration.self,
            from: Data(contentsOf: configURL)
        )
        try configuration.validate()
        let model = MiniCPMAudioModel(configuration: configuration)
        do {
            let weights = try MLX.loadArrays(url: weightsURL)
            try model.update(
                parameters: ModuleParameters.unflattened(weights),
                verify: .all
            )
            model.train(false)
            eval(model)
        } catch {
            throw MiniCPMAudioError.invalidConfiguration(
                "incompatible audio weights: \(error.localizedDescription)"
            )
        }
        return model
    }
}
