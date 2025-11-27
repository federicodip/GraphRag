#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Link :Concept, :Person, and :Article nodes to Wikidata via exact label/alias match.

Strategy
--------
- For each label:
    Concept.name
    Person.name
    Article.title

  1. Find distinct values that are NOT already linked to any :WikidataEntity
     with a [:SAME_AS {source:'wikidata'}] relationship.
  2. For each term, query Wikidata wbsearchentities API.
  3. If there is an exact match on label or alias (case-insensitive),
     MERGE a (:WikidataEntity {qid}) and link:

     (n:Concept|Person|Article)-[:SAME_AS {source:'wikidata', method:'label-exact' or 'title-exact'}]->(w:WikidataEntity)

Notes
-----
- This is intentionally conservative: no fuzzy matching, no disambiguation heuristics.
- Safe to re-run: MERGE makes it idempotent.
"""

import os
import time
import requests
from neo4j import GraphDatabase
from dotenv import load_dotenv

# --- config / env -----------------------------------------------------------

load_dotenv()

NEO4J_URI  = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASS = os.getenv("NEO4J_PASSWORD") or os.getenv("NEO4J_PASS")
DB         = os.getenv("NEO4J_DATABASE", "graphrag")

SEARCH_URL = "https://www.wikidata.org/w/api.php"
HEADERS = {
    "User-Agent": "GraphRAG-ISAW-label-linker/1.0 (contact: fed.dipasqua@stud.uniroma3.it)",
    "Accept": "application/json",
}
SLEEP_BETWEEN_CALLS = float(os.getenv("WD_SEARCH_SLEEP", "0.25"))
SEARCH_LIMIT = int(os.getenv("WD_SEARCH_LIMIT", "10"))


# --- helper: wikidata search -----------------------------------------------

def wd_search_exact(term: str, language: str = "en", limit: int = SEARCH_LIMIT):
    """
    Query Wikidata search API for an item that has an exact label or alias
    matching `term` (case-insensitive). Returns (qid, label) or (None, None).
    """
    t = (term or "").strip()
    if not t:
        return None, None

    params = {
        "action": "wbsearchentities",
        "language": language,
        "format": "json",
        "type": "item",
        "search": t,
        "limit": limit,
    }

    r = requests.get(SEARCH_URL, params=params, headers=HEADERS, timeout=30)
    r.raise_for_status()
    data = r.json()
    hits = data.get("search", []) or []

    t_lower = t.lower()
    for h in hits:
        label = (h.get("label") or "").strip()
        if label.lower() == t_lower:
            return h["id"], label
        for a in h.get("aliases") or []:
            a = (a or "").strip()
            if a.lower() == t_lower:
                return h["id"], label or term

    return None, None


# --- helper: generic upsert -------------------------------------------------

def upsert_same_as(tx, label_name: str, prop: str, term: str, qid: str,
                   wd_label: str, method: str):
    """
    Create/merge WikidataEntity and SAME_AS edge for a given label/property.
    """
    cypher = f"""
    MERGE (w:WikidataEntity {{qid:$qid}})
      ON CREATE SET w.uri = 'https://www.wikidata.org/entity/' + $qid
    SET w.label = coalesce($wd_label, w.label)

    WITH w
    MATCH (n:{label_name} {{{prop}:$term}})
    MERGE (n)-[:SAME_AS {{
        source:'wikidata',
        method:$method
    }}]->(w)
    """
    tx.run(
        cypher,
        qid=qid,
        wd_label=wd_label,
        term=term,
        method=method,
    )


# --- Neo4j fetchers ---------------------------------------------------------

def concept_terms_needing_link(session):
    q = """
    MATCH (c:Concept)
    WHERE c.name IS NOT NULL
      AND NOT (c)-[:SAME_AS {source:'wikidata'}]->(:WikidataEntity)
    RETURN DISTINCT c.name AS term
    ORDER BY term
    """
    return [r["term"] for r in session.run(q)]


def person_terms_needing_link(session):
    q = """
    MATCH (p:Person)
    WHERE p.name IS NOT NULL
      AND NOT (p)-[:SAME_AS {source:'wikidata'}]->(:WikidataEntity)
    RETURN DISTINCT p.name AS term
    ORDER BY term
    """
    return [r["term"] for r in session.run(q)]


def article_terms_needing_link(session):
    q = """
    MATCH (a:Article)
    WHERE a.title IS NOT NULL
      AND NOT (a)-[:SAME_AS {source:'wikidata'}]->(:WikidataEntity)
    RETURN DISTINCT a.title AS term
    ORDER BY term
    """
    return [r["term"] for r in session.run(q)]


# --- main worker per label --------------------------------------------------

def link_label_batch(session, label_name: str, prop: str, method: str, terms):
    """
    For a given label/property, try to link each term via exact Wikidata search.
    """
    total = len(terms)
    linked = 0
    skipped = 0

    print(f"[{label_name}] attempting to link {total} terms via method='{method}'")

    for i, term in enumerate(terms, start=1):
        t_clean = (term or "").strip()
        if not t_clean:
            skipped += 1
            continue

        try:
            qid, wd_label = wd_search_exact(t_clean)
        except Exception as e:
            print(f"[{label_name} #{i}/{total}] term={t_clean!r} search error: {e}")
            skipped += 1
            time.sleep(SLEEP_BETWEEN_CALLS)
            continue

        if not qid:
            print(f"[{label_name} #{i}/{total}] term={t_clean!r} -> no exact match")
            skipped += 1
            time.sleep(SLEEP_BETWEEN_CALLS)
            continue

        # Upsert into Neo4j
        session.execute_write(
            upsert_same_as,
            label_name,
            prop,
            t_clean,
            qid,
            wd_label or t_clean,
            method,
        )
        linked += 1
        print(f"[{label_name} #{i}/{total}] term={t_clean!r} -> {qid} ({wd_label})")

        time.sleep(SLEEP_BETWEEN_CALLS)

    print(f"[{label_name}] done. linked={linked}, skipped/failed={skipped}, total={total}")


def sanity_counts(session):
    q = """
    MATCH (c:Concept) RETURN 'Concept' AS label, count(c) AS total
    UNION ALL
    MATCH (p:Person) RETURN 'Person' AS label, count(p) AS total
    UNION ALL
    MATCH (a:Article) RETURN 'Article' AS label, count(a) AS total
    """
    print("[sanity] Node counts (all):")
    for r in session.run(q):
        print(f"  {r['label']}: {r['total']}")



def main():
    if not NEO4J_PASS:
        raise RuntimeError("No Neo4j password set (NEO4J_PASSWORD or NEO4J_PASS)")

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))
    with driver.session(database=DB) as session:
        print(f"Connected to Neo4j DB='{DB}' at {NEO4J_URI} as user='{NEO4J_USER}'")
        sanity_counts(session)

        concepts = concept_terms_needing_link(session)
        persons  = person_terms_needing_link(session)
        articles = article_terms_needing_link(session)

        print(f"Concept terms needing link: {len(concepts)}")
        print(f"Person terms needing link : {len(persons)}")
        print(f"Article titles needing link: {len(articles)}")

        if concepts:
            link_label_batch(session, "Concept", "name", "label-exact", concepts)
        if persons:
            link_label_batch(session, "Person", "name", "label-exact", persons)
        if articles:
            link_label_batch(session, "Article", "title", "title-exact", articles)

    driver.close()
    print("Done.")


if __name__ == "__main__":
    main()
