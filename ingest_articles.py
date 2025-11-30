#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Ingest one or many ISAW articles into Neo4j.

Reads:
  - JSONL of chunks
  - metadata JSON for the article (like isaw-papers-1-2011.meta.json)

Does NOT create/save meta files. It only reads them and writes nodes/edges.

Single-article usage (compatible with your old flow):
  python ingest_articles.py --jsonl data/chunks/isaw2.jsonl \
                            --meta  data/articles/isaw-papers-2-2012.meta.json

Multi-article usage (what you want for 3+):
  python ingest_articles.py --jsonl-dir data/chunks \
                            --meta-dir  data/articles \
                            --pattern "isaw_paper*.jsonl"

Assumptions:
  - Each JSONL line: {"articleId": "...", "chunkId": "...", "seq": 0, "text": "..."}
  - For each articleId, there is a meta file:
        <meta-dir>/<articleId>.meta.json
    with a JSON object:
        {"articleId": "...", "title": "...", "year": ..., "journal": "...", "url": "...", "authors": [...]}
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple, Iterable

# Load .env BEFORE anything uses env vars
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

from neo4j import GraphDatabase
from tqdm import tqdm

import spacy
from sentence_transformers import SentenceTransformer

# -----------------------------
# Secrets & connection from env
# -----------------------------
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASS = os.getenv("NEO4J_PASSWORD") or os.getenv("NEO4J_PASS")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "graphrag")

if not NEO4J_PASS:
    raise SystemExit(
        "Missing Neo4j password. Set NEO4J_PASSWORD in a local .env file or as an environment variable."
    )

DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CONCEPT_ALLOWLIST = {"Terms", "Exaltations", "Triplicities", "Houses", "Decans"}


# ---------- I/O helpers ----------

def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Bad JSON on line {line_no} in {path}: {e}") from e
    return rows


def load_meta(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_next_pairs(chunks: List[Dict[str, Any]]) -> List[Tuple[str, str]]:
    s = sorted(chunks, key=lambda x: x["seq"])
    return [(s[i]["chunkId"], s[i + 1]["chunkId"]) for i in range(len(s) - 1)]


def extract_mentions(nlp, text: str) -> Dict[str, List[str]]:
    doc = nlp(text)
    persons = sorted({
        ent.text.strip()
        for ent in doc.ents
        if ent.label_ == "PERSON" and len(ent.text.strip()) >= 2
    })
    concepts = sorted({c for c in CONCEPT_ALLOWLIST if c.lower() in text.lower()})
    return {"persons": persons, "concepts": concepts}


# ---------- Cypher ----------

CYPHER_MERGE_ARTICLE = """
MERGE (a:Article {articleId: $articleId})
SET a.title = $title, a.year = $year, a.journal = $journal, a.url = $url
"""

CYPHER_MERGE_CHUNK = """
MERGE (c:Chunk {chunkId: $chunkId})
SET c.seq = $seq, c.text = $text, c.textEmbedding = $embedding
"""

CYPHER_REL_HAS_CHUNK = """
MATCH (a:Article {articleId: $articleId}), (c:Chunk {chunkId: $chunkId})
MERGE (a)-[:HAS_CHUNK]->(c)
MERGE (c)-[:PART_OF]->(a)
"""

CYPHER_REL_NEXT = """
MATCH (c1:Chunk {chunkId: $c1}), (c2:Chunk {chunkId: $c2})
MERGE (c1)-[:NEXT]->(c2)
"""

CYPHER_MERGE_PERSON = """
MERGE (p:Person {name: $name})
ON CREATE SET p.aliases = COALESCE($aliases, []),
              p.orcid = $orcid, p.wikidataId = $wikidataId,
              p.birth = $birth, p.death = $death
"""

CYPHER_REL_AUTHORED = """
MATCH (p:Person {name: $name}), (a:Article {articleId: $articleId})
MERGE (p)-[r:AUTHORED]->(a)
ON CREATE SET r.`order` = $order, r.role = $role, r.corresponding = $corresponding
"""

CYPHER_MERGE_CONCEPT = "MERGE (k:Concept {name: $name})"

CYPHER_REL_MENTIONS_PERSON = """
MATCH (c:Chunk {chunkId: $chunkId}), (p:Person {name: $name})
MERGE (c)-[:MENTIONS]->(p)
"""

CYPHER_REL_MENTIONS_CONCEPT = """
MATCH (c:Chunk {chunkId: $chunkId}), (k:Concept {name: $name})
MERGE (c)-[:MENTIONS]->(k)
"""


# ---------- Core ingest ----------

def ingest(uri, user, password, database,
           chunks: List[Dict[str, Any]],
           meta: Dict[str, Any],
           embedder,
           nlp) -> None:
    driver = GraphDatabase.driver(uri, auth=(user, password))
    article_id = meta["articleId"]

    with driver.session(database=database) as session:
        # Article
        session.run(
            CYPHER_MERGE_ARTICLE,
            articleId=article_id,
            title=meta.get("title"),
            year=meta.get("year"),
            journal=meta.get("journal"),
            url=meta.get("url"),
        )

        # Authors (accept list of dicts OR list of strings)
        raw_authors = meta.get("authors") or []
        for idx, a_raw in enumerate(raw_authors, start=1):
            if isinstance(a_raw, str):
                # Simple format: "Alexander Jones"
                a = {
                    "name": a_raw,
                    "order": idx,
                    "role": "author",
                    "corresponding": False,
                }
            elif isinstance(a_raw, dict):
                a = a_raw
            else:
                print(f"[WARN] Unsupported author entry in meta for {article_id}: {a_raw!r}")
                continue

            name = (a.get("name") or "").strip()
            if not name:
                print(f"[WARN] Author entry without name in meta for {article_id}: {a!r}")
                continue

            session.run(
                CYPHER_MERGE_PERSON,
                name=name,
                aliases=a.get("aliases"),
                orcid=a.get("orcid"),
                wikidataId=a.get("wikidataId"),
                birth=a.get("birth"),
                death=a.get("death"),
            )
            session.run(
                CYPHER_REL_AUTHORED,
                name=name,
                articleId=article_id,
                order=a.get("order", idx),
                role=a.get("role", "author"),
                corresponding=bool(a.get("corresponding", False)),
            )


        # Chunks + embeddings + mentions
        texts = [c["text"] for c in chunks]
        embeddings = embedder.encode(
            texts,
            show_progress_bar=True,
            normalize_embeddings=True,
        )

        for c, emb in tqdm(
            zip(chunks, embeddings),
            total=len(chunks),
            desc=f"Inserting chunks for {article_id}",
        ):
            session.run(
                CYPHER_MERGE_CHUNK,
                chunkId=c["chunkId"],
                seq=c["seq"],
                text=c["text"],
                embedding=[float(x) for x in emb],
            )
            session.run(
                CYPHER_REL_HAS_CHUNK,
                articleId=article_id,
                chunkId=c["chunkId"],
            )

            m = extract_mentions(nlp, c["text"])
            for person_name in m["persons"]:
                session.run(
                    CYPHER_MERGE_PERSON,
                    name=person_name,
                    aliases=None,
                    orcid=None,
                    wikidataId=None,
                    birth=None,
                    death=None,
                )
                session.run(
                    CYPHER_REL_MENTIONS_PERSON,
                    chunkId=c["chunkId"],
                    name=person_name,
                )

            for concept_name in m["concepts"]:
                session.run(CYPHER_MERGE_CONCEPT, name=concept_name)
                session.run(
                    CYPHER_REL_MENTIONS_CONCEPT,
                    chunkId=c["chunkId"],
                    name=concept_name,
                )

        # NEXT chain
        for c1, c2 in build_next_pairs(chunks):
            session.run(CYPHER_REL_NEXT, c1=c1, c2=c2)

    driver.close()


# ---------- Iteration helpers ----------

def iter_articles_from_dir(jsonl_dir: Path, pattern: str) -> Iterable[tuple[Path, str]]:
    """
    Yield (jsonl_path, article_id) for each JSONL in jsonl_dir matching pattern.
    Assumes each file contains exactly one articleId.
    """
    for jsonl_path in sorted(jsonl_dir.glob(pattern)):
        chunks = read_jsonl(jsonl_path)
        ids = {c.get("articleId") for c in chunks}
        ids.discard(None)
        if len(ids) != 1:
            print(f"[SKIP] {jsonl_path}: expected exactly 1 articleId, found {ids}")
            continue
        article_id = ids.pop()
        yield jsonl_path, article_id


# ---------- Main ----------

def main():
    ap = argparse.ArgumentParser(
        description="Ingest one or many ISAW articles (chunks JSONL + meta JSON) into Neo4j."
    )
    # Single-article mode
    ap.add_argument("--jsonl", help="Path to a single chunks JSONL file.")
    ap.add_argument("--meta", help="Path to the corresponding meta JSON file.")

    # Multi-article mode
    ap.add_argument("--jsonl-dir", default="data/chunks",
                    help="Directory with *.jsonl files (multi-article mode).")
    ap.add_argument("--meta-dir", default="data/articles",
                    help="Directory with <articleId>.meta.json files.")
    ap.add_argument("--pattern", default="*.jsonl",
                    help="Glob pattern for JSONL files in --jsonl-dir (e.g. 'isaw_paper*.jsonl').")

    ap.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL,
                    help="SentenceTransformer model name.")

    args = ap.parse_args()

    print(f"Loading embedder: {args.embed_model}")
    embedder = SentenceTransformer(args.embed_model)

    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        import subprocess, sys
        subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
        nlp = spacy.load("en_core_web_sm")

    meta_dir = Path(args.meta_dir).expanduser().resolve()

    # --- Single-article mode ---
    if args.jsonl and args.meta:
        jsonl_path = Path(args.jsonl).expanduser().resolve()
        meta_path = Path(args.meta).expanduser().resolve()

        if not jsonl_path.exists():
            raise SystemExit(f"Missing chunks file: {jsonl_path}")
        if not meta_path.exists():
            raise SystemExit(f"Missing meta file:   {meta_path}")

        chunks = read_jsonl(jsonl_path)
        meta = load_meta(meta_path)

        ids = {c.get("articleId") for c in chunks}
        ids.discard(None)
        if len(ids) != 1:
            raise SystemExit(f"{jsonl_path}: expected 1 articleId in chunks, found {ids}")
        article_id = ids.pop()
        if article_id != meta.get("articleId"):
            raise SystemExit(
                f"ArticleId mismatch between chunks ({article_id}) and meta ({meta.get('articleId')})"
            )

        print(f"Ingesting {article_id} from {jsonl_path.name}")
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
        print("Done (single article).")
        return

    # --- Multi-article mode ---
    jsonl_dir = Path(args.jsonl_dir).expanduser().resolve()
    if not jsonl_dir.exists():
        raise SystemExit(f"JSONL directory not found: {jsonl_dir}")

    print(f"Multi-article mode: scanning {jsonl_dir} with pattern {args.pattern!r}")
    for jsonl_path, article_id in iter_articles_from_dir(jsonl_dir, args.pattern):
        meta_path = meta_dir / f"{article_id}.meta.json"
        if not meta_path.exists():
            print(f"[SKIP] {jsonl_path.name}: meta file not found: {meta_path.name}")
            continue

        chunks = read_jsonl(jsonl_path)
        meta = load_meta(meta_path)
        if meta.get("articleId") != article_id:
            print(
                f"[SKIP] {jsonl_path.name}: articleId in meta ({meta.get('articleId')}) "
                f"does not match chunks ({article_id})"
            )
            continue

        print(f"=== Ingesting {article_id} from {jsonl_path.name} ===")
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
        print(f"=== Done {article_id} ===")

    print("All done.")


if __name__ == "__main__":
    main()
