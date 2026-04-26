#!/usr/bin/env python3
"""
make_short_prompt.py

Tokenize a text string with a Qwen3.5 GGUF and write int32 LE token IDs to
short_prompt.bin next to this script.

Usage:
    python3 make_short_prompt.py --text "You are a helpful assistant." \
        --model-path /path/to/Qwen3.5-27B-Q4_K_M.gguf

Requires the `gguf` Python package:
    pip install gguf

If the package is unavailable, the script falls back to the hardcoded 16-token
fixture already committed in short_prompt.bin and prints a warning.
"""

import argparse
import pathlib
import struct
import sys

SCRIPT_DIR = pathlib.Path(__file__).parent

FALLBACK_TOKENS = [
    151644, 8948, 198, 2610, 525, 264, 10950, 17847,
    13, 151645, 198, 151644, 872, 198, 2610, 7291,
]


def tokenize_via_gguf(text: str, model_path: str):
    try:
        from gguf import GGUFReader  # type: ignore
    except ImportError:
        return None

    # GGUFReader gives access to metadata but not a tokenizer runtime.
    # For actual tokenization we need llama-cpp-python or similar.
    try:
        from llama_cpp import Llama  # type: ignore
    except ImportError:
        return None

    llm = Llama(model_path=model_path, vocab_only=True, verbose=False)
    tokens = llm.tokenize(text.encode(), add_bos=True, special=True)
    return tokens


def main():
    parser = argparse.ArgumentParser(description="Generate short_prompt.bin from text.")
    parser.add_argument("--text", default="You are a helpful assistant.")
    parser.add_argument("--model-path", default="")
    parser.add_argument("--out", default=str(SCRIPT_DIR / "short_prompt.bin"))
    args = parser.parse_args()

    tokens = None
    if args.model_path:
        tokens = tokenize_via_gguf(args.text, args.model_path)
        if tokens is None:
            print(
                "WARNING: llama_cpp Python package not available. "
                "Writing hardcoded fallback fixture.",
                file=sys.stderr,
            )

    if tokens is None:
        tokens = FALLBACK_TOKENS

    out_path = pathlib.Path(args.out)
    data = struct.pack("<" + "i" * len(tokens), *tokens)
    out_path.write_bytes(data)
    print(f"wrote {len(tokens)} tokens to {out_path}")


if __name__ == "__main__":
    main()
