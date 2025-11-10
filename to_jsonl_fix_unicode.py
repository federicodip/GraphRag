#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert a JSON array of text chunks (isaw.txt) into JSONL,
decoding bare 'uXXXX' / 'uXXXXXX' codepoint strings into real Unicode.

Usage:
  python to_jsonl_fix_unicode.py isaw.txt isaw.jsonl --article-id isaw-papers-1-2011
"""

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path

# matches: u2643, u00b0, u1f45, u03b1, etc. (4–6 hex digits)
UHEX = re.compile(r"u([0-9a-fA-F]{4,6})")

def decode_bare_u_sequences(s: str) -> str:
    """Replace 'uXXXX' / 'uXXXXXX' with the corresponding Unicode character."""
    return UHEX.sub(lambda m: chr(int(m.group(1), 16)), s)

def normalize_text(s: str) -> str:
    """Unicode NFC + replace NBSP with regular space."""
    s = s.replace("\u00A0", " ")
    return unicodedata.normalize("NFC", s)

def clean_text(s: str) -> str:
    """Decode bare-u sequences then normalize."""
    return normalize_text(decode_bare_u_sequences(s))

def load_chunks(input_path: Path) -> list[str]:
    """Read a JSON array of strings from file."""
    # 'utf-8-sig' strips a BOM if present (common on Windows)
    raw = input_path.read_text(encoding="utf-8-sig")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        msg = (
            f"Failed to parse {input_path} as JSON.\n"
            "Ensure the file is a valid JSON array of strings, e.g. [\"...\", \"...\"]\n"
            f"JSON error: {e}"
        )
        raise SystemExit(msg)
    if not isinstance(data, list) or not all(isinstance(x, str) for x in data):
        raise SystemExit("Input must be a JSON array of strings.")
    return data

def write_jsonl(chunks: list[str], output_path: Path, article_id: str):
    """Write JSONL with cleaned text (no raw text kept)."""
    with output_path.open("w", encoding="utf-8") as f:
        for i, text_raw in enumerate(chunks):
            text_clean = clean_text(text_raw)
            obj = {
                "articleId": article_id,
                "chunkId": f"{article_id}:{i:04d}",
                "seq": i,
                "text": text_clean,
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="Path to isaw.txt (JSON array of strings)")
    parser.add_argument("output", type=Path, nargs="?", help="Path to output JSONL (default: input stem + .jsonl)")
    parser.add_argument("--article-id", default="isaw-article-1", help="Stable id/slug for this article")
    args = parser.parse_args()

    input_path: Path = args.input
    output_path: Path = args.output or input_path.with_suffix(".jsonl")

    chunks = load_chunks(input_path)
    write_jsonl(chunks, output_path, args.article_id)

    # quick sanity check for residual 'uXXXX' sequences in cleaned text
    residual = 0
    for s in chunks:
        if UHEX.search(clean_text(s)):
            residual += 1
    if residual:
        print(f"WARNING: {residual} chunks still contain 'uXXXX' patterns after cleaning.", file=sys.stderr)

    print(f"Wrote {len(chunks)} lines to {output_path}")

if __name__ == "__main__":
    main()
