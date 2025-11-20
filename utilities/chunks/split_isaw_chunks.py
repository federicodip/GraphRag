#!/usr/bin/env python3
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

# Match integers or decimals like 7 or 7.1, 7.12; accept "Papers" or "Paper"
ISAW_RE = re.compile(r'ISAW\s*Papers?\s*(\d+(?:\.\d+)?)\b', re.IGNORECASE)

def extract_isaw_id(chunk: str) -> str | None:
    """
    Return the last ISAW Papers identifier found in the chunk, e.g. '7' or '7.3'.
    We take the final occurrence to avoid earlier mentions in the prose.
    """
    hits = ISAW_RE.findall(chunk)
    return hits[-1] if hits else None

def natural_isaw_sort_key(ident: str):
    # Sort 7, 7.1, 7.2, 8 ... numerically by each dot-separated part
    return [int(p) for p in ident.split('.')]

def split_by_subchapter(input_path: Path) -> dict[str, list[str]]:
    with input_path.open('r', encoding='utf-8') as f:
        chunks = json.load(f)

    if not isinstance(chunks, list) or any(not isinstance(c, str) for c in chunks):
        raise ValueError("Input must be a JSON array of strings.")

    groups: dict[str, list[str]] = defaultdict(list)
    skipped = 0

    for chunk in chunks:
        ident = extract_isaw_id(chunk)
        if ident is None:
            skipped += 1
            continue
        groups[ident].append(chunk)

    if skipped:
        print(f"Warning: {skipped} chunk(s) lacked an 'ISAW Papers <id>' tag and were skipped.")
    return groups

def write_outputs(groups: dict[str, list[str]], out_dir: Path) -> None:
    for ident in sorted(groups, key=natural_isaw_sort_key):
        safe_ident = ident.replace('.', '_')  # e.g., 7.1 -> 7_1
        out_path = out_dir / f"isaw_paper{safe_ident}.txt"
        with out_path.open('w', encoding='utf-8') as f:
            json.dump(groups[ident], f, ensure_ascii=False, indent=2)
        print(f"Wrote {out_path} ({len(groups[ident])} chunk(s)).")

def main():
    ap = argparse.ArgumentParser(
        description="Split ISAW chunks (JSON array of strings) into per-paper/subchapter files."
    )
    ap.add_argument("input_file", help="Path to JSON array file (e.g., chunks_isaw_papers_all.txt)")
    args = ap.parse_args()

    in_path = Path(args.input_file).expanduser().resolve()
    if not in_path.exists():
        raise FileNotFoundError(in_path)

    groups = split_by_subchapter(in_path)
    write_outputs(groups, in_path.parent)

if __name__ == "__main__":
    main()
