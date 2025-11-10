#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json, gzip, re
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from neo4j import GraphDatabase
import ijson

# ------------ Config via env ------------
NEO4J_URI  = os.getenv("NEO4J_URI",  "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASS = os.getenv("NEO4J_PASSWORD") or os.getenv("NEO4J_PASS")
NEO4J_DB   = os.getenv("NEO4J_DATABASE", "graphrag")

# Path to pleiades-places-latest.json[.gz]
PLEIADES_JSON = os.getenv(
    "PLEIADES_JSON",
    r"C:\Users\feder\Desktop\GraphRag\data\pleiades\pleiades-places-latest.json.gz"
)

# ------------ Cypher ------------
CYPHER_UPSERT_PLACE = """
MERGE (p:Place {pleiadesId:$pleiadesId})
SET p.uri         = $uri,
    p.title       = $title,
    p.description = $description,
    p.placeTypes  = $placeTypes,
    p.subject     = $subject,
    p.altNames    = $altNames,
    p.languages   = $languages,
    p.review_state= $review_state,
    p.source      = 'Pleiades'
"""

CYPHER_UPSERT_STUB = """
MERGE (p:Place {pleiadesId:$pleiadesId})
ON CREATE SET p.uri = $uri, p.source = 'Pleiades'
"""

CYPHER_CONNECT = """
MATCH (a:Place {pleiadesId:$from}), (b:Place {pleiadesId:$to})
MERGE (a)-[r:CONNECTED]->(b)
ON CREATE SET r.connectionType = $connectionType,
              r.title = $title,
              r.associationCertainty = $associationCertainty,
              r.uri = $uri,
              r.source = 'Pleiades'
"""

# ------------ Helpers (streaming; handles @graph) ------------
def iter_pleiades_places(path: Path):
    """Yield place dicts from .json/.json.gz/NDJSON without loading the whole file."""

    def _open_text():
        return gzip.open(path, "rt", encoding="utf-8") if path.suffix.lower() == ".gz" else path.open("rt", encoding="utf-8")

    # Peek a little to guess the container type
    f = _open_text()
    head = f.read(200).lstrip()
    f.close()

    # Case 1: top-level JSON array
    if head.startswith("["):
        f = _open_text()
        try:
            for place in ijson.items(f, "item"):
                if isinstance(place, dict):
                    yield place
        finally:
            f.close()
        return

    # Case 2: top-level JSON object (several variants)
    if head.startswith("{"):
        # 2a) JSON-LD with @graph: [...places...]
        f = _open_text()
        yielded = False
        try:
            for place in ijson.items(f, "@graph.item"):
                yielded = True
                if isinstance(place, dict):
                    yield place
        finally:
            f.close()
        if yielded:
            return

        # 2b) {"places":[...]}
        f = _open_text()
        yielded = False
        try:
            for place in ijson.items(f, "places.item"):
                yielded = True
                if isinstance(place, dict):
                    yield place
        finally:
            f.close()
        if yielded:
            return

        # 2c) GeoJSON FeatureCollection: {"features":[{"properties":{...}}, ...]}
        f = _open_text()
        yielded = False
        try:
            for feat in ijson.items(f, "features.item"):
                props = (feat or {}).get("properties")
                if isinstance(props, dict):
                    yielded = True
                    yield props
        finally:
            f.close()
        if yielded:
            return

        # 2d) dict-of-id -> place
        f = _open_text()
        yielded = False
        try:
            for key, place in ijson.kvitems(f, ""):
                if isinstance(place, dict):
                    yielded = True
                    yield place
        finally:
            f.close()
        if yielded:
            return

    # Case 3: NDJSON fallback (one JSON object per line)
    f = _open_text()
    try:
        for line in f:
            s = line.strip()
            if not s or s[0] != "{":
                continue
            try:
                obj = json.loads(s)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                yield obj
    finally:
        f.close()

def _pid_from_uri(uri: str) -> Optional[str]:
    if not isinstance(uri, str):
        return None
    m = re.search(r"/places/(\d+)", uri)
    return m.group(1) if m else None

def _collect_names(place: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    alt, langs = [], []
    for n in place.get("names", []):
        if not isinstance(n, dict):
            continue
        for k in ("attested", "romanized", "title", "name"):
            v = n.get(k)
            if isinstance(v, str) and v.strip():
                parts = [p.strip() for p in v.split(",") if p.strip()]
                alt.extend(parts)
        lang = n.get("language")
        if isinstance(lang, str) and lang not in langs:
            langs.append(lang)
    for k in ("label", "placename"):
        v = place.get(k)
        if isinstance(v, str) and v.strip():
            alt.append(v.strip())
    seen = set()
    out = []
    for a in alt:
        if a not in seen:
            seen.add(a); out.append(a)
    return out, langs

def _safe_list(x):
    return x if isinstance(x, list) else []

# ------------ Main ------------
def main():
    if not NEO4J_PASS:
        raise SystemExit("Set NEO4J_PASSWORD (or NEO4J_PASS) before running.")
    src = Path(PLEIADES_JSON)
    if not src.exists():
        raise SystemExit(f"Not found: {src}")

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))

    n_places, n_edges = 0, 0
    with driver.session(database=NEO4J_DB) as sess:
        for place in iter_pleiades_places(src):
            # Identify the place
            pid = str(place.get("id")) if place.get("id") is not None else _pid_from_uri(place.get("uri",""))
            if not pid:
                continue

            uri   = place.get("uri") or f"https://pleiades.stoa.org/places/{pid}"
            title = place.get("title") or place.get("name") or place.get("label")
            desc  = place.get("description")
            ptypes = _safe_list(place.get("placeTypes") or place.get("placeType") or place.get("place_type") or place.get("placeTypeURIs"))
            subject = _safe_list(place.get("subject"))
            review = place.get("review_state")
            altNames, languages = _collect_names(place)

            sess.run(
                CYPHER_UPSERT_PLACE,
                pleiadesId=pid, uri=uri, title=title, description=desc,
                placeTypes=ptypes, subject=subject,
                altNames=altNames, languages=languages, review_state=review
            )
            n_places += 1

            # connectsWith: plain list of related URIs
            for uri2 in _safe_list(place.get("connectsWith")):
                to_pid = _pid_from_uri(uri2)
                if not to_pid:
                    continue
                sess.run(CYPHER_UPSERT_STUB, pleiadesId=to_pid, uri=uri2)
                sess.run(CYPHER_CONNECT,
                         **{"from": pid, "to": to_pid},
                         connectionType="related",
                         title=None,
                         associationCertainty=None,
                         uri=None)
                n_edges += 1

            # connections: richer typed edges
            for c in _safe_list(place.get("connections")):
                to_uri = c.get("connectsTo")
                to_pid = _pid_from_uri(to_uri) if to_uri else None
                if not to_pid:
                    continue
                sess.run(CYPHER_UPSERT_STUB, pleiadesId=to_pid, uri=to_uri)
                sess.run(CYPHER_CONNECT,
                         **{"from": pid, "to": to_pid},
                         connectionType=c.get("connectionType"),
                         title=c.get("title"),
                         associationCertainty=c.get("associationCertainty"),
                         uri=c.get("uri"))
                n_edges += 1

    driver.close()
    print(f"Ingested places: {n_places}, connections: {n_edges}")

if __name__ == "__main__":
    main()
