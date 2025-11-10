#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ingest_one_article.py

Reads:
  - JSONL of chunks
  - metadata JSON for the article
Embeds chunks, creates Article/Chunk/Person/Concept nodes and relations in Neo4j.

Secrets/config:
  - Create a .env file in the project root (NOT committed) with:
      NEO4J_URI=bolt://localhost:7687
      NEO4J_USER=neo4j
      NEO4J_PASSWORD=...         # REQUIRED
      NEO4J_DATABASE=graphrag     # optional
Install once:
  pip install python-dotenv neo4j tqdm spacy sentence-transformers
  python -m spacy download en_core_web_sm
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple

# Load .env BEFORE anything uses env vars
try:
    from dotenv import load_dotenv
    load_dotenv()  # loads .env from current working directory or parents
except Exception:
    # If python-dotenv isn't installed, we just proceed (env can still be set by OS)
    pass

from neo4j import GraphDatabase
from tqdm import tqdm

import spacy
from sentence_transformers import SentenceTransformer

# -----------------------------
# Static file locations (adjust if needed)
# -----------------------------
CONFIG = {
    "JSONL_PATH": r"C:\Users\feder\Desktop\GraphRag\data\chunks\isaw2.jsonl",
    "META_PATH":  r"C:\Users\feder\Desktop\GraphRag\data\articles\isaw-papers-2-2012.meta.json",
    "EMBED_MODEL": "sentence-transformers/all-MiniLM-L6-v2",
}

# -----------------------------
# Secrets & connection from env
# -----------------------------
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
# Accept either NEO4J_PASSWORD or NEO4J_PASS
NEO4J_PASS = os.getenv("NEO4J_PASSWORD") or os.getenv("NEO4J_PASS")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "graphrag")

if not NEO4J_PASS:
    raise SystemExit(
        "Missing Neo4j password. Set NEO4J_PASSWORD in a local .env file or as an environment variable."
    )

CONCEPT_ALLOWLIST = {"Terms", "Exaltations", "Triplicities", "Houses", "Decans"}


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
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


def ingest(uri, user, password, database, chunks, meta, embedder, nlp):
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

        # Authors
        for a in meta.get("authors", []):
            session.run(
                CYPHER_MERGE_PERSON,
                name=a["name"],
                aliases=a.get("aliases"),
                orcid=a.get("orcid"),
                wikidataId=a.get("wikidataId"),
                birth=a.get("birth"),
                death=a.get("death"),
            )
            session.run(
                CYPHER_REL_AUTHORED,
                name=a["name"],
                articleId=article_id,
                order=a.get("order"),
                role=a.get("role", "author"),
                corresponding=bool(a.get("corresponding", False)),
            )

        # Chunks + embeddings + mentions
        texts = [c["text"] for c in chunks]
        embeddings = embedder.encode(texts, show_progress_bar=True, normalize_embeddings=True)

        for c, emb in tqdm(zip(chunks, embeddings), total=len(chunks), desc="Inserting chunks"):
            session.run(
                CYPHER_MERGE_CHUNK,
                chunkId=c["chunkId"],
                seq=c["seq"],
                text=c["text"],
                embedding=[float(x) for x in emb],  # Neo4j supports list<float>
            )
            session.run(CYPHER_REL_HAS_CHUNK, articleId=article_id, chunkId=c["chunkId"])

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
                session.run(CYPHER_REL_MENTIONS_PERSON, chunkId=c["chunkId"], name=person_name)

            for concept_name in m["concepts"]:
                session.run(CYPHER_MERGE_CONCEPT, name=concept_name)
                session.run(CYPHER_REL_MENTIONS_CONCEPT, chunkId=c["chunkId"], name=concept_name)

        # NEXT chain
        for c1, c2 in build_next_pairs(chunks):
            session.run(CYPHER_REL_NEXT, c1=c1, c2=c2)

    driver.close()


def main():
    jsonl_path = Path(CONFIG["JSONL_PATH"])
    meta_path = Path(CONFIG["META_PATH"])

    if not jsonl_path.exists():
        raise SystemExit(f"Missing chunks file: {jsonl_path}")
    if not meta_path.exists():
        raise SystemExit(f"Missing meta file:   {meta_path}")

    chunks = read_jsonl(jsonl_path)
    meta = load_meta(meta_path)

    if {c["articleId"] for c in chunks} != {meta["articleId"]}:
        raise SystemExit("ArticleId mismatch between chunks and meta")

    print(f"Loaded {len(chunks)} chunks for article {meta['articleId']}")
    print("Loading embedder:", CONFIG["EMBED_MODEL"])
    embedder = SentenceTransformer(CONFIG["EMBED_MODEL"])

    # spaCy model (auto-install if missing)
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        import subprocess, sys
        subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
        nlp = spacy.load("en_core_web_sm")

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
    print("Done.")


if __name__ == "__main__":
    main()
