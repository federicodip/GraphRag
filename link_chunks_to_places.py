#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, sys
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=Path(".env"))
except Exception:
    pass

from neo4j import GraphDatabase

NEO4J_URI  = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASS = os.getenv("NEO4J_PASSWORD") or os.getenv("NEO4J_PASS")
NEO4J_DB   = os.getenv("NEO4J_DATABASE", "graphrag")

BOUNDARY = r"(^|[^A-Za-z0-9_])"
BOUNDARY_END = r"([^A-Za-z0-9_]|$)"

def compile_pattern(name: str) -> re.Pattern:
    # escape regex specials, wrap with crude word boundaries, ignore case
    return re.compile(rf"(?i){BOUNDARY}{re.escape(name)}{BOUNDARY_END}")

def fetch_places(session):
    q = """
    MATCH (p:Place)
    RETURN p.pleiadesId AS pid, p.title AS title, coalesce(p.altNames, []) AS alts
    """
    pairs = []  # list of (pid, name, compiled_regex)
    for rec in session.run(q):
        pid = rec["pid"]
        names = []
        if rec["title"]:
            names.append(rec["title"])
        for a in rec["alts"]:
            if isinstance(a, str):
                names.append(a)
        # clean and dedupe
        seen = set()
        for n in names:
            n2 = n.strip()
            if len(n2) >= 3 and n2.lower() not in seen:
                seen.add(n2.lower())
                pairs.append((pid, n2, compile_pattern(n2)))
    return pairs

def fetch_all_article_ids(session):
    q = """
    MATCH (a:Article)
    WHERE a.articleId IS NOT NULL
    RETURN a.articleId AS aid
    ORDER BY aid
    """
    return [rec["aid"] for rec in session.run(q)]

def fetch_chunks(session, article_id: str):
    q = """
    MATCH (:Article {articleId:$aid})-[:HAS_CHUNK]->(c:Chunk)
    RETURN c.chunkId AS cid, c.text AS text
    ORDER BY c.seq
    """
    return list(session.run(q, aid=article_id))

def link_one_chunk(tx, cid: str, hits):
    # hits = [(pid, matchedName), ...]
    for pid, matched in hits:
        tx.run(
            """
            MATCH (c:Chunk {chunkId:$cid})
            MATCH (p:Place {pleiadesId:$pid})
            MERGE (c)-[r:MENTIONS]->(p)
            ON CREATE SET r.matched = $matched, r.source = 'name-exact-boundary'
            """,
            cid=cid, pid=pid, matched=matched
        )

def main():
    if not NEO4J_PASS:
        print("Set NEO4J_PASSWORD (or .env).")
        sys.exit(1)

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))
    total_links = 0
    total_chunks = 0

    try:
        with driver.session(database=NEO4J_DB) as sess:
            # load dictionary once
            place_entries = fetch_places(sess)
            print(f"Loaded {len(place_entries)} place names.")

            # get all articleIds
            article_ids = fetch_all_article_ids(sess)
            print(f"Found {len(article_ids)} articles to scan.")

            for idx, article_id in enumerate(article_ids, start=1):
                print(f"\n[{idx}/{len(article_ids)}] Scanning article {article_id}...")
                chunk_records = fetch_chunks(sess, article_id)
                print(f"  Chunks: {len(chunk_records)}")

                for rec in chunk_records:
                    cid = rec["cid"]
                    text = rec["text"] or ""
                    hits = []
                    # naive O(N*M) scan
                    for pid, name, pat in place_entries:
                        if pat.search(text):
                            hits.append((pid, name))
                    if hits:
                        with sess.begin_transaction() as tx:
                            link_one_chunk(tx, cid, hits)
                        total_links += len(hits)
                        total_chunks += 1
                        print(f"    {cid}: {len(hits)} links")

            print(f"\nDone. Linked {total_links} (in {total_chunks} chunks) across {len(article_ids)} articles.")
    finally:
        driver.close()

if __name__ == "__main__":
    main()
