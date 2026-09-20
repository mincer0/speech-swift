import MLX

/// The dense MiniCPM bundle follows PyTorch's BF16 SiLU contract:
///
///     BF16(round(float32(x) * sigmoid(float32(x))))
///
/// MLX's fused ``silu`` kernel evaluates the activation in a different
/// precision order for BF16 inputs.  The difference is visible in the dense
/// MiniCPM oracle even though the operation is mathematically equivalent.
/// Keep the explicit cast sequence here so the production MLP and its parity
/// regression share one implementation.  Quantized MiniCPM layers continue to
/// use MLX's established fused activation and never call this helper.
public enum MiniCPMExactBF16SiLU {
    public static func apply(_ input: MLXArray) -> MLXArray {
        guard input.dtype == .bfloat16 else {
            // This helper is intended for dense BF16 activations.  Retaining
            // the ordinary MLX expression for other dtypes keeps it useful in
            // model-independent probes without silently changing their dtype.
            return input * sigmoid(input)
        }
        let inputFloat = input.asType(.float32)
        return (inputFloat * sigmoid(inputFloat)).asType(.bfloat16)
    }
}
