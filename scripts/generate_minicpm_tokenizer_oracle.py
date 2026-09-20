#!/usr/bin/env python3
"""Generate the cross-language MiniCPM tokenizer oracle.

The Swift parity test deliberately does not ship the 11 MiB tokenizer files
inside the test bundle.  Run this script whenever the local MiniCPM tokenizer
bundle changes, then run the opt-in Swift test against the same directory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer


SPECIAL_TOKENS = [
    "<|im_start|>",
    "<|im_end|>",
    "<|endoftext|>",
    "<unk>",
    "<unit>",
    "</unit>",
    "<|listen|>",
    "<|speak|>",
    "<|interrupt|>",
    "<|tts_bos|>",
    "<|tts_eos|>",
    "<|tts_pad|>",
    "<|turn_bos|>",
    "<|turn_eos|>",
    "<|chunk_bos|>",
    "<|chunk_eos|>",
    "<|chunk_tts_bos|>",
    "<|chunk_tts_eos|>",
    "<|audio_start|>",
    "<|audio|>",
    "<|audio_end|>",
    "<|spk_bos|>",
    "<|spk|>",
    "<|spk_eos|>",
    "<|vad_start|>",
    "<|vad_end|>",
    "<|vision_start|>",
    "<|vision_end|>",
    "<image>",
    "</image>",
    "<|image_pad|>",
    "<slice>",
    "</slice>",
]

TEXTS = [
    "Hello world",
    "你好，世界！",
    "emoji 🥳🎵 café e\u0301",
    "<|listen|>前缀 <|tts_bos|> 后缀",
    "混合<|chunk_eos|>emoji🥳",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parents[1]
        / "Tests/MiniCPMLLMTests/Fixtures/tokenizer_oracle.json",
    )
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(
        str(args.model_dir), trust_remote_code=True, local_files_only=True
    )

    output = {
        "special_tokens": {
            token: tokenizer.convert_tokens_to_ids(token)
            for token in SPECIAL_TOKENS
        },
        "texts": [],
        "chat": [],
    }
    for text in TEXTS:
        ids = tokenizer.encode(text, add_special_tokens=False)
        output["texts"].append(
            {
                "text": text,
                "ids": ids,
                "decoded": tokenizer.decode(ids, skip_special_tokens=False),
                "skip_special": tokenizer.decode(ids, skip_special_tokens=True),
            }
        )

    messages = [{"role": "user", "content": "你好 🥳"}]
    for options in (
        {},
        {"enable_thinking": False},
        {"use_tts_template": True},
        {"enable_thinking": False, "use_tts_template": True},
    ):
        ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            **options,
        )
        output["chat"].append(
            {
                "options": options,
                "ids": ids,
                "decoded": tokenizer.decode(ids, skip_special_tokens=False),
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
