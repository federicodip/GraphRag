#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Bulk-ingest ALL ISAW articles that have:
  - a meta file:   <meta-dir>/<articleId>.meta.json
  - a chunks file: <chunks-dir>/<articleId>.jsonl  (or any *.jsonl containing that articleId in its filename)

This script REUSES the ingest() logic from ingest_articles.py so the graph
schema + embeddings stay consistent.

Usage (from project root):

  python ingest_all_from_meta.py \
      --meta-dir   data/articles \
      --chunks-dir data/chunks \
      --embed-model sentence-transformers/all-MiniLM-L6-v2

Env vars (same as before):
  NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD/NEO4J_PASS, NEO4J_DATABASE
"""

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Import your existing ingest pipeline
from ingest_articles import (
    ingest,
    read_jsonl,
    load_meta,
    NEO4J_URI,
    NEO4J_USER,
    NEO4J_PASS,
    NEO4J_DATABASE,
    DEFAULT_EMBED_MODEL,
)

from sentence_transformers import SentenceTransformer
import spacy
from tqdm import tqdm


def find_chunks_file(chunks_dir: Path, article_id: str) -> Path | None:
    """
    Prefer <articleId>.jsonl; if that doesn't exist, fall back to any *.jsonl
    whose filename contains the articleId as a substring.
    """
    direct = chunks_dir / f"{article_id}.jsonl"
    if direct.exists():
        return direct

    candidates = list(chunks_dir.glob("*.jsonl"))
    matches = [p for p in candidates if article_id in p.name]

    if not matches:
        return None
    if len(matches) > 1:
        print(f"[WARN] Multiple chunk files for {article_id}: {[m.name for m in matches]}. Using {matches[0].name}")
    return matches[0]


def main():
    if not NEO4J_PASS:
        raise SystemExit(
            "Missing Neo4j password. Set NEO4J_PASSWORD or NEO4J_PASS in .env or env vars."
        )

    ap = argparse.ArgumentParser(
        description="Bulk-ingest all articles that have meta + chunks, using ingest_articles.ingest()."
    )
    ap.add_argument(
        "--meta-dir",
        default="data/articles",
        help="Directory with <articleId>.meta.json files.",
    )
    ap.add_argument(
        "--chunks-dir",
        default="data/chunks",
        help="Directory with *.jsonl chunk files.",
    )
    ap.add_argument(
        "--embed-model",
        default=DEFAULT_EMBED_MODEL,
        help="SentenceTransformer model name.",
    )

    args = ap.parse_args()

    meta_dir = Path(args.meta_dir).expanduser().resolve()
    chunks_dir = Path(args.chunks_dir).expanduser().resolve()

    if not meta_dir.exists():
        raise SystemExit(f"Meta directory not found: {meta_dir}")
    if not chunks_dir.exists():
        raise SystemExit(f"Chunks directory not found: {chunks_dir}")

    print(f"Meta dir:   {meta_dir}")
    print(f"Chunks dir: {chunks_dir}")
    print(f"Neo4j DB:   {NEO4J_URI} / {NEO4J_DATABASE}")
    print(f"Embedder:   {args.embed_model}")

    print("Loading embedder...")
    embedder = SentenceTransformer(args.embed_model)

    print("Loading spaCy model...")
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        import subprocess, sys

        subprocess.check_call(
            [sys.executable, "-m", "spacy", "download", "en_core_web_sm"]
        )
        nlp = spacy.load("en_core_web_sm")

    meta_files = sorted(meta_dir.glob("*.meta.json"))
    if not meta_files:
        raise SystemExit(f"No *.meta.json files found in {meta_dir}")

    print(f"Found {len(meta_files)} meta files. Starting ingest...\n")

    ingested = 0
    skipped_no_chunks = 0
    skipped_bad_chunks = 0

    for meta_path in tqdm(meta_files, desc="Articles", unit="article"):
        try:
            meta = load_meta(meta_path)
        except Exception as e:
            print(f"[SKIP] Failed to load meta {meta_path.name}: {e}")
            continue

        article_id = meta.get("articleId")
        if not article_id:
            print(f"[SKIP] {meta_path.name}: no 'articleId' in meta.")
            continue

        chunks_path = find_chunks_file(chunks_dir, article_id)
        if not chunks_path:
            print(f"[SKIP] {article_id}: no chunks JSONL found in {chunks_dir}")
            skipped_no_chunks += 1
            continue

        try:
            chunks = read_jsonl(chunks_path)
        except Exception as e:
            print(f"[SKIP] Failed to read chunks {chunks_path.name}: {e}")
            skipped_bad_chunks += 1
            continue

        ids = {c.get("articleId") for c in chunks}
        ids.discard(None)
        if len(ids) != 1:
            print(
                f"[SKIP] {chunks_path.name}: expected exactly 1 articleId in chunks, found {ids}"
            )
            skipped_bad_chunks += 1
            continue

        chunks_article_id = ids.pop()
        if chunks_article_id != article_id:
            print(
                f"[SKIP] Meta/chunks mismatch: meta articleId={article_id}, chunks articleId={chunks_article_id}"
            )
            skipped_bad_chunks += 1
            continue

        print(f"\n=== Ingesting {article_id} from {chunks_path.name} ===")
        try:
            ingest(
                NEO4J_URI,
                NEO4J_USER,
                NEO4J_PASS,
                NEO4J_DATABASE,
                chunks,
                meta,
                embedder,
                nlp,
            )
            ingested += 1
            print(f"=== Done {article_id} ===")
        except Exception as e:
            print(f"[ERROR] Ingest failed for {article_id}: {e}")
            skipped_bad_chunks += 1

    print("\nAll done.")
    print(f"Ingested OK:           {ingested}")
    print(f"Skipped (no chunks):   {skipped_no_chunks}")
    print(f"Skipped (bad chunks):  {skipped_bad_chunks}")


if __name__ == "__main__":
    main()
