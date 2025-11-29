#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
generate_meta_stubs.py

Generate stub meta JSON files for ISAW articles based on chunks JSONL.

- Reads JSONL files (one article per file) to get articleId.
- Writes meta stubs to data/articles/<articleId>.meta.json
  if they don't already exist (or if --overwrite is given).

This does NOT try to guess title/year/url/authors.
You fill those manually afterwards.

Usage (what you want for articles 3+):
  python generate_meta_stubs.py \
      --jsonl-dir data/chunks \
      --meta-dir  data/articles \
      --pattern "isaw_paper*.jsonl"
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any, Set


def read_article_id_from_jsonl(path: Path) -> str:
    """
    Read all lines in a JSONL file and extract distinct articleIds.
    Expect exactly one distinct non-null articleId.
    """
    ids: Set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Bad JSON on line {line_no} in {path}: {e}") from e
            aid = obj.get("articleId")
            if aid:
                ids.add(aid)

    if not ids:
        raise ValueError(f"No articleId found in {path}")
    if len(ids) > 1:
        raise ValueError(f"Multiple articleIds in {path}: {ids}")
    return ids.pop()


def make_meta_stub(article_id: str, journal: str = "ISAW Papers") -> Dict[str, Any]:
    """
    Build a minimal stub meta dict for manual completion.
    """
    return {
        "articleId": article_id,
        "title": "",          # TODO: fill in title
        "year": None,         # TODO: fill in publication year (int)
        "journal": journal,
        "url": "",            # TODO: fill in canonical URL
        "authors": [],        # TODO: list of {name, order, role, corresponding}
    }


def main():
    ap = argparse.ArgumentParser(
        description="Generate stub meta JSON files from chunks JSONL."
    )
    ap.add_argument(
        "--jsonl-dir",
        type=Path,
        default=Path("data/chunks"),
        help="Directory containing chunks JSONL files.",
    )
    ap.add_argument(
        "--meta-dir",
        type=Path,
        default=Path("data/articles"),
        help="Directory to write meta JSON files.",
    )
    ap.add_argument(
        "--pattern",
        default="isaw_paper*.jsonl",
        help="Glob pattern for JSONL files in --jsonl-dir (default: isaw_paper*.jsonl).",
    )
    ap.add_argument(
        "--journal",
        default="ISAW Papers",
        help="Default journal name to put in stubs.",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing meta files instead of skipping them.",
    )

    args = ap.parse_args()

    jsonl_dir = args.jsonl_dir.expanduser().resolve()
    meta_dir = args.meta_dir.expanduser().resolve()

    if not jsonl_dir.exists():
        raise SystemExit(f"JSONL directory not found: {jsonl_dir}")

    meta_dir.mkdir(parents=True, exist_ok=True)

    jsonl_files = sorted(jsonl_dir.glob(args.pattern))
    if not jsonl_files:
        raise SystemExit(f"No JSONL files matching pattern {args.pattern!r} in {jsonl_dir}")

    print(f"JSONL dir: {jsonl_dir}")
    print(f"Meta dir : {meta_dir}")
    print(f"Pattern  : {args.pattern!r}")
    print(f"Overwrite: {args.overwrite}")
    print("----")

    created = 0
    skipped = 0
    for jsonl_path in jsonl_files:
        try:
            article_id = read_article_id_from_jsonl(jsonl_path)
        except ValueError as e:
            print(f"[SKIP] {jsonl_path.name}: {e}")
            skipped += 1
            continue

        meta_path = meta_dir / f"{article_id}.meta.json"
        if meta_path.exists() and not args.overwrite:
            print(f"[SKIP existing] {meta_path.name}")
            skipped += 1
            continue

        meta = make_meta_stub(article_id, journal=args.journal)
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[WRITE] {meta_path.name} (articleId={article_id})")
        created += 1

    print("----")
    print(f"Done. created={created}, skipped={skipped}")


if __name__ == "__main__":
    main()
