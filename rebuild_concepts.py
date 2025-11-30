#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Rebuild :Concept nodes from existing Chunk text.

- Fetches all (:Chunk) nodes and their `text`.
- Uses spaCy to extract candidate concepts (noun phrases + some non-person entities).
- Keeps only phrases that appear at least MIN_GLOBAL_FREQ times.
- MERGE (:Concept {name}) and (c:Chunk)-[:MENTIONS]->(k:Concept).

This does NOT delete existing Concept nodes. It just adds more.
"""

import os
from collections import Counter, defaultdict
from typing import List, Set

from dotenv import load_dotenv
from neo4j import GraphDatabase
import spacy

# -----------------------------
# Config
# -----------------------------

load_dotenv()

NEO4J_URI  = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASS = os.getenv("NEO4J_PASSWORD") or os.getenv("NEO4J_PASS")
NEO4J_DB   = os.getenv("NEO4J_DATABASE", "graphrag")

# How often a phrase must appear (across all chunks) to be kept as a Concept
MIN_GLOBAL_FREQ = 3

# Max tokens in a concept phrase
MAX_TOKENS = 6


# -----------------------------
# NLP setup
# -----------------------------

def load_nlp():
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        # Try to download if missing
        import subprocess, sys
        subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
        return spacy.load("en_core_web_sm")


nlp = load_nlp()


def clean_phrase(text: str) -> str:
    """Basic cleanup: strip whitespace/punctuation, collapse spaces, title-case."""
    t = text.strip()
    # drop quotes and trailing punctuation, keep inner apostrophes
    t = t.strip(" .,:;!?\"“”'()[]{}")
    t = " ".join(t.split())
    return t


def is_good_concept(text: str) -> bool:
    """Heuristic filters for candidate concepts."""
    t = text.strip()
    if len(t) < 3:
        return False
    if len(t) > 80:
        return False
    # Avoid anything with obvious junk
    if any(ch.isdigit() for ch in t):
        return False
    if any(ext in t.lower() for ext in [".csv", ".json", "http://", "https://"]):
        return False
    # Mostly punctuation?
    if sum(ch.isalpha() for ch in t) < 2:
        return False
    return True


def extract_concepts_from_text(text: str) -> Set[str]:
    """
    Extract candidate concept phrases from text:
    - noun chunks
    - entities of non-person, non-place types (ORG, EVENT, WORK_OF_ART, etc.)
    """
    doc = nlp(text)
    candidates = set()

    # 1) Noun chunks
    for nc in doc.noun_chunks:
        # Filter by length
        tokens = [tok for tok in nc if not tok.is_space]
        if not tokens or len(tokens) > MAX_TOKENS:
            continue
        phrase = clean_phrase(nc.text)
        if not phrase:
            continue
        if is_good_concept(phrase):
            candidates.add(phrase)

    # 2) Named entities (excluding people and GPE/LOC)
    bad_labels = {"PERSON", "GPE", "LOC"}
    for ent in doc.ents:
        if ent.label_ in bad_labels:
            continue
        tokens = [tok for tok in ent if not tok.is_space]
        if not tokens or len(tokens) > MAX_TOKENS:
            continue
        phrase = clean_phrase(ent.text)
        if not phrase:
            continue
        if is_good_concept(phrase):
            candidates.add(phrase)

    return candidates


# -----------------------------
# Neo4j helpers
# -----------------------------

CYPHER_FETCH_CHUNKS = """
MATCH (c:Chunk)
RETURN c.chunkId AS cid, c.text AS text
"""

CYPHER_MERGE_CONCEPT = """
MERGE (k:Concept {name: $name})
"""

CYPHER_LINK_CHUNK_CONCEPT = """
MATCH (c:Chunk {chunkId: $cid})
MATCH (k:Concept {name: $name})
MERGE (c)-[:MENTIONS]->(k)
"""


def main():
    if not NEO4J_PASS:
        raise SystemExit("Neo4j password missing (NEO4J_PASSWORD or NEO4J_PASS).")

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))
    with driver.session(database=NEO4J_DB) as session:
        print(f"Connected to Neo4j DB='{NEO4J_DB}' at {NEO4J_URI} as user='{NEO4J_USER}'")

        print("Fetching all chunks...")
        result = session.run(CYPHER_FETCH_CHUNKS)
        rows = list(result)
        print(f"Fetched {len(rows)} chunks.")

        # First pass: extract candidate concepts + global counts
        concept_counts = Counter()
        chunk_to_concepts = {}

        for idx, rec in enumerate(rows, start=1):
            cid = rec["cid"]
            text = rec["text"] or ""
            if not text.strip():
                chunk_to_concepts[cid] = set()
                continue

            concepts = extract_concepts_from_text(text)
            chunk_to_concepts[cid] = concepts
            concept_counts.update(concepts)

            if idx % 500 == 0:
                print(f"Processed {idx}/{len(rows)} chunks...")

        print(f"Found {len(concept_counts)} unique concept candidates.")

        # Decide which concepts to keep (by global frequency)
        kept_concepts = {c for c, cnt in concept_counts.items() if cnt >= MIN_GLOBAL_FREQ}
        print(f"Keeping {len(kept_concepts)} concepts with freq >= {MIN_GLOBAL_FREQ}.")

        # Second pass: write concepts + MENTIONS links
        # 1) MERGE all concepts
        with session.begin_transaction() as tx:
            for idx, name in enumerate(kept_concepts, start=1):
                tx.run(CYPHER_MERGE_CONCEPT, name=name)
                if idx % 500 == 0:
                    print(f"MERGEd {idx}/{len(kept_concepts)} concepts...")
        print("Finished MERGEing Concept nodes.")

        # 2) Link chunks to concepts
        link_count = 0
        with session.begin_transaction() as tx:
            for cid, concepts in chunk_to_concepts.items():
                for name in concepts:
                    if name not in kept_concepts:
                        continue
                    tx.run(CYPHER_LINK_CHUNK_CONCEPT, cid=cid, name=name)
                    link_count += 1
                    if link_count % 1000 == 0:
                        print(f"Created {link_count} MENTIONS links so far...")
        print(f"Finished linking chunks to concepts. Total links: {link_count}")

    driver.close()
    print("Done.")


if __name__ == "__main__":
    main()
