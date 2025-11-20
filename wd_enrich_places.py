#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enrich :Place (Pleiades) with Wikidata via P1584.
- Uses POST + proper Accept header + format=json
- Retries/backoff
- Skips already-linked places
"""

import os
import time
import requests
from textwrap import dedent
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

NEO4J_URI  = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASS = os.getenv("NEO4J_PASSWORD") or os.getenv("NEO4J_PASS")
DB         = os.getenv("NEO4J_DATABASE", "graphrag")

WDQS_URL = "https://query.wikidata.org/sparql"
HEADERS = {
    # Correct type for SPARQL JSON results:
    "Accept": "application/sparql-results+json",
    # A real contact per WDQS policy:
    "User-Agent": "GraphRAG-ISAW-enricher/1.0 (contact: fed.dipasqua@stud.uniroma3.it)",
    "Accept-Encoding": "gzip, deflate",
}
BATCH = int(os.getenv("WDQS_BATCH", "40"))
MAX_RETRIES = 4
RETRY_SLEEP = 6.0
POLITE_PAUSE = 0.6

def sanity_counts(session):
    total_place = session.run("MATCH (p:Place) RETURN count(p) AS c").single()["c"]
    with_pid = session.run(
        "MATCH (p:Place) WHERE p.pleiadesId IS NOT NULL RETURN count(p) AS c"
    ).single()["c"]
    already = session.run(
        """
        MATCH (:Place)-[r:SAME_AS {property:'P1584'}]->(:WikidataEntity)
        RETURN count(r) AS c
        """
    ).single()["c"]
    print(f"[sanity] DB='{DB}'  :Place={total_place}  with pleiadesId={with_pid}  existing SAME_AS(P1584)={already}")

def get_pleiades_ids_needing_link(session):
    q = """
    MATCH (p:Place)
    WHERE p.pleiadesId IS NOT NULL
      AND NOT (p)-[:SAME_AS {property:'P1584'}]->(:WikidataEntity)
    RETURN p.pleiadesId AS pid
    """
    return [r["pid"] for r in session.run(q)]

def wdqs_for_batch(ids):
    values = " ".join(f'"{i}"' for i in ids)
    sparql = dedent(f"""
    SELECT ?item ?pleiadesId ?itemLabel ?coord ?inst WHERE {{
      VALUES ?pleiadesId {{ {values} }}
      ?item wdt:P1584 ?pleiadesId .
      OPTIONAL {{ ?item wdt:P625 ?coord . }}
      OPTIONAL {{ ?item wdt:P31 ?inst . }}
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
    }}
    """).strip()

    tries = 0
    while True:
        tries += 1
        r = requests.post(
            WDQS_URL,
            data={"query": sparql, "format": "json"},  # force JSON
            headers=HEADERS,
            timeout=120,
        )
        if r.status_code in (429, 502, 503, 504):
            ra = r.headers.get("Retry-After")
            sleep_for = float(ra) if ra else RETRY_SLEEP * tries
            print(f"[wdqs] {r.status_code}; retrying in {sleep_for:.1f}s …")
            time.sleep(sleep_for)
            if tries <= MAX_RETRIES:
                continue
            raise RuntimeError(f"WDQS throttled/errored {tries} times; aborting.")

        if not r.ok:
            head = (r.text or "")[:400].replace("\n", " ")
            raise RuntimeError(f"WDQS HTTP {r.status_code}; Content-Type={r.headers.get('Content-Type','?')}. Body head: {head}")

        ct = r.headers.get("Content-Type", "")
        if "json" not in ct.lower():
            head = (r.text or "")[:400].replace("\n", " ")
            raise RuntimeError(f"WDQS non-JSON (Content-Type={ct}). Body head: {head}")

        try:
            data = r.json()
            return data["results"]["bindings"]
        except Exception as e:
            head = (r.text or "")[:400].replace("\n", " ")
            print(f"[wdqs] JSON parse failed (try {tries}): {e}. Head: {head}")
            if tries <= MAX_RETRIES:
                time.sleep(RETRY_SLEEP * tries)
                continue
            raise
        finally:
            time.sleep(POLITE_PAUSE)

def parse_coord(coord_bind):
    if not coord_bind:
        return None, None
    v = coord_bind["value"]
    try:
        inside = v[v.find("(")+1:v.find(")")]
        lon_str, lat_str = inside.split()
        return float(lat_str), float(lon_str)
    except Exception:
        return None, None

def upsert_batch(tx, rows):
    for b in rows:
        qid = b["item"]["value"].rsplit("/", 1)[-1]
        pid = b["pleiadesId"]["value"]
        label = b.get("itemLabel", {}).get("value")
        inst = b.get("inst", {}).get("value")
        inst_qid = inst.rsplit("/", 1)[-1] if inst else None
        lat, lon = parse_coord(b.get("coord"))

        tx.run(
            """
            MERGE (w:WikidataEntity {qid:$qid})
              ON CREATE SET w.uri = 'https://www.wikidata.org/entity/' + $qid
            SET w.label = coalesce($label, w.label),
                w.instanceOf = CASE WHEN $inst_qid IS NULL THEN w.instanceOf ELSE $inst_qid END,
                w.lat = CASE WHEN $lat IS NULL THEN w.lat ELSE $lat END,
                w.lon = CASE WHEN $lon IS NULL THEN w.lon ELSE $lon END

            WITH w
            MATCH (p:Place {pleiadesId:$pid})
            MERGE (p)-[r:SAME_AS {source:'wikidata', property:'P1584'}]->(w)
            SET r.matchedBy = 'pleiadesId'
            """,
            qid=qid, pid=pid, label=label, inst_qid=inst_qid, lat=lat, lon=lon
        )

def chunker(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i:i+size], i

def main():
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))
    with driver.session(database=DB) as session:
        sanity_counts(session)

        ids = get_pleiades_ids_needing_link(session)
        print(f"Found {len(ids)} Place nodes still needing Wikidata (P1584).")
        if not ids:
            return

        total_hits = 0
        for batch, start_idx in chunker(ids, BATCH):
            try:
                rows = wdqs_for_batch(batch)
            except Exception as e:
                print(f"[batch {start_idx}-{start_idx+len(batch)-1}] WDQS error: {e}")
                continue
            if rows:
                session.execute_write(upsert_batch, rows)
                total_hits += len(rows)
            print(f"[batch {start_idx}-{start_idx+len(batch)-1}] rows={len(rows)}  total_hits={total_hits}")

    driver.close()

if __name__ == "__main__":
    main()
