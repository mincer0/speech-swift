#!/usr/bin/env python3
"""Export a tiny PyTorch/MPS BF16 audio-head weight_norm regression fixture."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=Path(__file__).parents[1]
        / "Tests/MiniCPMTTSSemanticTests/Fixtures/minicpm-tts-weight-norm-mps.safetensors",
    )
    args = parser.parse_args()
    import torch
    from safetensors.torch import save_file

    if not torch.backends.mps.is_available():
        parser.error("the fixture requires the PyTorch MPS weight_norm kernel")
    v = (((torch.arange(3 * 768, dtype=torch.int32) * 37) % 255 - 127)
         .float() / 4096).reshape(3, 768).bfloat16()
    g = torch.tensor([[0.75], [0.625], [1.125]], dtype=torch.bfloat16)
    with torch.inference_mode():
        expected = torch._weight_norm(v.to("mps"), g.to("mps"), 0).cpu()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file({"direction": v, "magnitude": g, "expected": expected}, str(args.output))
    args.output.with_suffix(".json").write_text(json.dumps({
        "oracle_backend": "pytorch_mps_weight_norm",
        "torch_version": torch.__version__,
        "torch_git_version": torch.version.git_version,
        "direction_shape": list(v.shape), "dtype": "bfloat16", "dim": 0,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
