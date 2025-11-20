# GraphRAG-ISAW: Knowledge-Graph Extension for the AI Librarian

A working extension to the IEEE paper project (“AI Librarian”) that adds a Neo4j knowledge graph on top of the RAG stack. It ingests ISAW Papers into Article/Chunk nodes, ingests Pleiades places into Place nodes, and links chunks to places with `MENTIONS` edges. Designed so you can bulk-ingest documents first, then iterate on linking without touching documents. **Wikidata alignment is implemented via the Wikidata Query Service (SPARQL) using property P1584 (Pleiades ID).**

## Context: why these sources together

ISAW Papers is an open-access journal from NYU’s Institute for the Study of the Ancient World, published on the web (HTML) with permissive licensing and stable URLs—good for clean text extraction and citation.

Pleiades is the community-curated gazetteer of ancient places, providing stable identifiers (`pleiadesId`), canonical titles, multilingual alternate names, place types, and URIs—exactly what you want to normalize messy place mentions in text.

Wikidata is a general, CC0 knowledge base that cross-links to many authority files (including Pleiades), adding identifiers, aliases, and statements you can leverage for enrichment and disambiguation.

Putting them together is sound because:

- Your text (ISAW chunks) produces ambiguous surface forms.
- Pleiades gives you the authoritative place entity backbone.
- Wikidata extends those entities with broader graph context and cross-IDs for downstream linking.

## What this is

A graph-augmented RAG pipeline:

- Your existing vector/RAG system (“AI Librarian”).
- A Neo4j knowledge graph that models Articles, Chunks, People, Concepts, and Places from Pleiades.
- A deterministic linker that connects `Chunk → Place` via surface forms (title + `altNames`) using full-text shortlist + regex boundaries.

**Goal:** Add entity-level structure and repeatable linking over the same corpus used by the RAG backend, so you can explore, audit, and enrich relationships that matter to Ancient World research.

## Wikidata enrichment: how it works

A small Python job (`wd_enrich_places.py`) batches all `Place.pleiadesId` values and queries the Wikidata Query Service (WDQS) with SPARQL. For each hit where `?item wdt:P1584 ?pleiadesId`, we:

- `MERGE` a `(:WikidataEntity {qid})` and set `uri`, `label`, optional `instanceOf` (P31), and `lat`/`lon` from P625.
- `MERGE (p:Place {pleiadesId})-[:SAME_AS {property:'P1584', source:'wikidata', matchedBy:'pleiadesId'}]->(w:WikidataEntity)`.

It’s idempotent and safe to re-run. Batching and polite delays are included to respect WDQS rate limits.

## Data sources

- **ISAW Papers (articles & chunks)** → `:Article`, `:Chunk`
- **Pleiades (gazetteer of ancient places)** → `:Place` (with `pleiadesId`, `title`, `altNames`, etc.)
- **Wikidata (live enrichment)** → `:WikidataEntity` nodes created by resolving `Place.pleiadesId` through P1584; we add `[:SAME_AS {property:'P1584', source:'wikidata', matchedBy:'pleiadesId'}]` from `Place → WikidataEntity` and optionally store `label`, `instanceOf` (P31), and coordinates (P625 → `lat`,`lon`).

## Current graph content (as loaded)

- **Nodes:** 42,577
- **Labels used:** `Article`, `Chunk`, `Concept`, `Person`, `Place`
- **Relationships:** 32,057
- **Types used:** `AUTHORED`, `CONNECTED`, `HAS_CHUNK`, `MENTIONS`, `NEXT`, `PART_OF`

> After running the Wikidata job you will also see `WikidataEntity` nodes and `SAME_AS` relationships. Counts will depend on how many batches you’ve completed.

**Property keys (seen in use):**  
`aliases`, `altNames`, `articleId`, `associationCertainty`, `by`, `chunkId`, `connectionType`, `corresponding`, `data`, `description`, `id`, `journal`, `languages`, `matched`, `name`, `nodes`, `order`, `placeTypes`, `pleiadesId`, `relationships`, `review_state`, `role`, `seq`, `source`, `style`, `subject`, `text`, `textEmbedding`, `title`, `uri`, `url`, `visualisation`, `year`

---

### Exact schema (derived from your current Neo4j)

> “Required” = property present on **all** nodes/relationships of that label/type **in your DB snapshot** (not an enforced constraint unless you add one). “Optional” = present on a subset.

#### Node labels

**Place** — 42,139 nodes

- **Required:**
  - `pleiadesId` (string)
  - `uri` (string, Pleiades place URL)
  - `source` (string, e.g., `"Pleiades"`)
- **Optional:**
  - `title` (string) — present on ~41,777
  - `altNames` (list<string>) — ~41,777
  - `description` (string) — ~41,777
  - `placeTypes` (list<string>) — ~41,777
  - `subject` (list<string>) — ~41,777
  - `languages` (list<string>) — ~41,777
  - `review_state` (string) — ~41,777

**Article** — 2 nodes

- **Required:**
  - `articleId` (string)
  - `title` (string)
  - `year` (int)
  - `journal` (string)
  - `url` (string)

**Chunk** — 218 nodes

- **Required:**
  - `chunkId` (string)
  - `seq` (int)
  - `text` (string)
  - `textEmbedding` (list<float>)

**Person** — 213 nodes

- **Required:**
  - `name` (string)
  - `aliases` (list<string>)

**Concept** — 5 nodes

- **Required:**
  - `name` (string)

## Tech stack

- Neo4j 5.x / 4.x
- Neo4j Driver for Python
- APOC optional (linker provided without APOC)
- Python 3.10+ for ingestion & linking scripts
- Requests + python-dotenv (for WDQS job)
- Vector/RAG: your AI Librarian stack (OpenAI embeddings + Chroma, LangChain RetrievalQA, GPT-4) remains unchanged

## Repository layout (suggested)

```text
.
├─ data/
│  ├─ articles/               # raw article exports (optional)
│  └─ chunks/                 # JSON arrays of strings, one file per article
├─ graph/
│  ├─ constraints.cypher      # uniqueness constraints & indexes
│  ├─ ingest_articles.py      # upsert Article + Chunk + HAS_CHUNK + NEXT
│  ├─ ingest_pleiades.py      # upsert Place nodes from Pleiades dump
│  ├─ build_fulltext.cypher   # full-text index on Chunk(text)
│  ├─ linker_places.py        # Chunk -> Place MENTIONS creation
│  └─ sanity.cypher           # verification queries
├─ tools/
│  ├─ to_jsonl_fix_unicode.py # robust JSONL converter for article chunks
│  └─ wd_enrich_places.py     # Wikidata enrichment job (P1584 → SAME_AS)
└─ README.md
```

## Quick start

### Create constraints (once)

```cypher
CREATE CONSTRAINT article_id IF NOT EXISTS
FOR (a:Article) REQUIRE a.articleId IS UNIQUE;

CREATE CONSTRAINT chunk_id IF NOT EXISTS
FOR (c:Chunk) REQUIRE c.chunkId IS UNIQUE;

CREATE CONSTRAINT place_pid IF NOT EXISTS
FOR (p:Place) REQUIRE p.pleiadesId IS UNIQUE;

CREATE CONSTRAINT wd_qid IF NOT EXISTS
FOR (w:WikidataEntity) REQUIRE w.qid IS UNIQUE;
```

### Full-text index for candidate retrieval (once)

```cypher
CALL db.index.fulltext.createNodeIndex(
  'chunkText',
  ['Chunk'],
  ['text']
);
```

### Convert article text → JSONL

Input is a JSON array of strings (each string = a chunk). The converter normalizes Unicode and writes JSONL with `articleId`, `chunkId`, `seq`, `text`.

```bash
python tools/to_jsonl_fix_unicode.py   data/chunks/isaw2.txt   data/chunks/isaw2.jsonl   --article-id isaw-papers-2-2012
```

### Ingest Articles & Chunks

Upsert both and wire `HAS_CHUNK` & `NEXT`. Your script may already do this; keep the idempotent `MERGE` pattern.

### Ingest Pleiades

Upsert `:Place` with at least:

- `pleiadesId`
- `title`
- `altNames` (array)

Keep other Pleiades fields if available.

### Run the linker

The provided `linker_places.py` does:

- Full-text shortlist on `Chunk.text`
- Boundary regex on matched name
- `MERGE (c)-[:MENTIONS {matched, source:'fulltext+regex'}]->(p)`

It is idempotent and safe to re-run after tuning.

### Run Wikidata enrichment

Resolve Place → Wikidata via P1584 and add `SAME_AS`:

```bash
# Make sure NEO4J_URI / NEO4J_USER / NEO4J_PASSWORD (and NEO4J_DATABASE if non-default) are set
python tools/wd_enrich_places.py
```

## Linker (minimal working version)

```python
from neo4j import GraphDatabase
import re

URI = "bolt://localhost:7687"
USER = "neo4j"
PASSWORD = "your_password"

driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))

def fetch_places(tx):
    q = """
    MATCH (p:Place)
    RETURN p.pleiadesId AS pid,
           p.title      AS title,
           coalesce(p.altNames, []) AS altNames
    """
    return list(tx.run(q))

def link_name_to_place(tx, pid, name, rx):
    q = """
    CALL db.index.fulltext.queryNodes('chunkText', $q) YIELD node, score
    WITH node WHERE node.text =~ $rx
    MATCH (p:Place {pleiadesId:$pid})
    MERGE (node)-[r:MENTIONS {matched:$name, source:'fulltext+regex'}]->(p)
    RETURN count(r) AS created
    """
    return tx.run(
        q,
        q=f'"{name}"',
        rx=rx,
        pid=pid,
        name=name
    ).single()[0]

def compile_boundary_regex(term):
    esc = re.escape(term)
    return f"(?i)(^|[^A-Za-z0-9_]){esc}([^A-Za-z0-9_]|$)"

def main():
    with driver.session() as s:
        places = s.execute_read(fetch_places)
        for rec in places:
            pid = rec["pid"]
            names = [rec["title"], *(rec["altNames"] or [])]
            seen = set()
            names = [n.strip() for n in names if n and len(n.strip()) >= 3]
            names = [n for n in names if not (n.lower() in seen or seen.add(n.lower()))]
            for name in names:
                rx = compile_boundary_regex(name)
                s.execute_write(link_name_to_place, pid, name, rx)
    driver.close()

if __name__ == "__main__":
    main()
```

## Verification queries

### Counts

```cypher
MATCH (a:Article) RETURN count(a) AS articles;
MATCH (c:Chunk)   RETURN count(c) AS chunks;
MATCH (p:Place)   RETURN count(p) AS places;
MATCH ()-[r:MENTIONS]->(:Place) RETURN count(r) AS mentions;

MATCH (w:WikidataEntity) RETURN count(w) AS wd_items;
MATCH (:Place)-[r:SAME_AS {property:'P1584'}]->(:WikidataEntity) RETURN count(r) AS same_as_p1584;
```

### Coverage per article

```cypher
MATCH (a:Article)-[:HAS_CHUNK]->(c)
OPTIONAL MATCH (c)-[m:MENTIONS]->(p:Place)
RETURN a.articleId,
       count(DISTINCT c) AS chunks,
       count(DISTINCT p) AS places,
       count(DISTINCT m) AS links
ORDER BY links DESC
LIMIT 50;
```

### Spot results

```cypher
MATCH (:Article {articleId:$aid})-[:HAS_CHUNK]->(c)-[r:MENTIONS]->(p:Place)
RETURN c.chunkId, p.pleiadesId, p.title, r.matched
ORDER BY c.chunkId, p.title
LIMIT 100;
```

Check Wikidata joins for a sample place:

```cypher
MATCH (p:Place {pleiadesId:$pid})-[:SAME_AS {property:'P1584'}]->(w:WikidataEntity)
RETURN p.pleiadesId, p.title, w.qid, w.label, w.instanceOf, w.lat, w.lon;
```

## How this interacts with the RAG app

The RAG app continues to use vector search over chunk text to ground answers with citations.

The graph adds:

- Explicit `MENTIONS` (`Chunk → Place`) you can facet or filter by.
- Navigation across `Article / Chunk / Place / Person / Concept` via Cypher.
- A stable substrate to experiment with entity resolution and cross-corpus links (e.g., DCAA / AWDL later).
- Wikidata-backed enrichment for `Place` via `SAME_AS`, enabling cross-IDs and additional metadata (type, coords).

You can expose graph-powered filters in your UI (for example: “restrict to chunks mentioning Babylon”).

## Roadmap

- Extend Wikidata enrichment: beyond places (P1584), add person and concept alignment (e.g., VIAF/ORCID/ULAN, topical items), and pull selected statements (e.g., P625, P279, P131, P17) for analysis.
- Disambiguation: optional context cues (e.g., Mesopotamia | Euphrates | Assyria nearby) for high-ambiguity names.
- Accent-folding & fuzzy: accent-insensitive matching and cautious fuzzy matching for long names (≥ 6–7 chars).
- People/Concept linkers: parallel pipelines for `Person` and `Concept`.

## Provenance and licensing

- **ISAW Papers:** open access, CC-BY (site-hosted).
- **Pleiades:** open, with clear attribution requirements; keep `pleiadesId` and source `uri` in your nodes.
- **Wikidata:** CC0; store QIDs and source edges.

## Contributing

- Keep `MERGE`-idempotent Cypher; don’t introduce write patterns that duplicate nodes.
- Treat linkers as pure functions over existing nodes: re-runnable, measurable, and auditable.
- Add sanity queries for every new entity type and every new linker.
