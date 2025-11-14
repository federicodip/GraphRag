# GraphRAG-ISAW: Knowledge-Graph Extension for the AI Librarian

A pragmatic, working extension to the IEEE paper project (“AI Librarian”) that adds a Neo4j knowledge graph on top of the RAG stack. It ingests ISAW Papers into Article/Chunk nodes, ingests Pleiades places into Place nodes, and links chunks to places with `MENTIONS` edges. Designed so you can bulk-ingest documents first, then iterate on linking without touching documents. Wikidata alignment is planned next.

## Context: why these sources together

ISAW Papers is an open-access journal from NYU’s Institute for the Study of the Ancient World, published on the web (HTML) with permissive licensing and stable URLs—good for clean text extraction and citation.

Pleiades is the community-curated gazetteer of ancient places, providing stable identifiers (`pleiadesId`), canonical titles, multilingual alternate names, place types, and URIs—exactly what you want to normalize messy place mentions in text.

Wikidata is a general, CC0 knowledge base that cross-links to many authority files (including Pleiades), adding identifiers, aliases, and statements you can leverage for enrichment and disambiguation.

Putting them together is sound because:

* Your text (ISAW chunks) produces ambiguous surface forms.
* Pleiades gives you the authoritative place entity backbone.
* Wikidata extends those entities with broader graph context and cross-IDs for downstream linking.

## What this is

A graph-augmented RAG pipeline:

* Your existing vector/RAG system (“AI Librarian”).
* A Neo4j knowledge graph that models Articles, Chunks, People, Concepts, and Places from Pleiades.
* A deterministic linker that connects `Chunk → Place` via surface forms (title + `altNames`) using full-text shortlist + regex boundaries.

**Goal:** Add entity-level structure and repeatable linking over the same corpus used by the RAG backend, so you can explore, audit, and enrich relationships that matter to Ancient World research.

## Data sources

* **ISAW Papers (articles & chunks)** → `:Article`, `:Chunk`
* **Pleiades (gazetteer of ancient places)** → `:Place` (with `pleiadesId`, `title`, `altNames`, etc.)
* **Wikidata (planned)** → alignment via `pleiadesId` property mapping to Q identifiers, adding cross-links later

## Current graph content (as loaded)

* **Nodes:** 42,577

* **Labels used:** `Article`, `Chunk`, `Concept`, `Person`, `Place`

* **Relationships:** 32,057

* **Types used:** `AUTHORED`, `CONNECTED`, `HAS_CHUNK`, `MENTIONS`, `NEXT`, `PART_OF`

**Property keys (seen in use):**
`aliases`, `altNames`, `articleId`, `associationCertainty`, `by`, `chunkId`, `connectionType`, `corresponding`, `data`, `description`, `id`, `journal`, `languages`, `matched`, `name`, `nodes`, `order`, `placeTypes`, `pleiadesId`, `relationships`, `review_state`, `role`, `seq`, `source`, `style`, `subject`, `text`, `textEmbedding`, `title`, `uri`, `url`, `visualisation`, `year`

Replace that line with this block:

---

### Exact schema (derived from your current Neo4j)

> “Required” = property present on **all** nodes/relationships of that label/type **in your DB snapshot** (not an enforced constraint unless you add one). “Optional” = present on a subset.

#### Node labels

**Place** — 42,139 nodes

* **Required:**

  * `pleiadesId` (string)
  * `uri` (string, Pleiades place URL)
  * `source` (string, e.g., `"Pleiades"`)
* **Optional:**

  * `title` (string) — present on ~41,777
  * `altNames` (list<string>) — ~41,777
  * `description` (string) — ~41,777
  * `placeTypes` (list<string>) — ~41,777
  * `subject` (list<string>) — ~41,777
  * `languages` (list<string>) — ~41,777
  * `review_state` (string) — ~41,777

**Article** — 2 nodes

* **Required:**

  * `articleId` (string)
  * `title` (string)
  * `year` (int)
  * `journal` (string)
  * `url` (string)

**Chunk** — 218 nodes

* **Required:**

  * `chunkId` (string)
  * `seq` (int)
  * `text` (string)
  * `textEmbedding` (list<float>)

**Person** — 213 nodes

* **Required:**

  * `name` (string)
  * `aliases` (list<string>)

**Concept** — 5 nodes

* **Required:**

  * `name` (string)


## Tech stack

* Neo4j 5.x / 4.x
* Neo4j Driver for Python
* APOC optional (linker provided without APOC)
* Python 3.10+ for ingestion & linking scripts
* Vector/RAG: your AI Librarian stack (OpenAI embeddings + Chroma, LangChain RetrievalQA, GPT-4) remains unchanged

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
│  └─ to_jsonl_fix_unicode.py # robust JSONL converter for article chunks
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
python tools/to_jsonl_fix_unicode.py \
  data/chunks/isaw2.txt \
  data/chunks/isaw2.jsonl \
  --article-id isaw-papers-2-2012
```

### Ingest Articles & Chunks

Upsert both and wire `HAS_CHUNK` & `NEXT`. Your script may already do this; keep the idempotent `MERGE` pattern.

### Ingest Pleiades

Upsert `:Place` with at least:

* `pleiadesId`
* `title`
* `altNames` (array)

Keep other Pleiades fields if available.

### Run the linker

The provided `linker_places.py` does:

1. Full-text shortlist on `Chunk.text`
2. Boundary regex on matched name
3. `MERGE (c)-[:MENTIONS {matched, source:'fulltext+regex'}]->(p)`

It is idempotent and safe to re-run after tuning.

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
```

### Coverage per article

```cypher
MATCH (a:Article)-[:HAS_CHUNK]->(c)
OPTIONAL MATCH (c)-[m:MENTIONS]->(p:Place)
RETURN a.articleId,
       count(DISTINCT c) AS chunks,
       count(DISTINCT p) AS places,
       count(m)          AS links
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

## How this interacts with the RAG app

The RAG app continues to use vector search over chunk text to ground answers with citations.

The graph adds:

* Explicit `MENTIONS` (`Chunk → Place`) you can facet or filter by.
* Navigation across `Article` / `Chunk` / `Place` / `Person` / `Concept` via Cypher.
* A stable substrate to experiment with entity resolution and cross-corpus links (e.g., DCAA / AWDL later).

You can expose graph-powered filters in your UI (for example: “restrict to chunks mentioning Babylon”).

## Roadmap

* **Wikidata alignment:** resolve `Place → Wikidata` via `pleiadesId` mappings; add `:SAME_AS` edges and enrich with external identifiers.
* **Disambiguation:** optional context cues (e.g., `Mesopotamia | Euphrates | Assyria` nearby) for high-ambiguity names.
* **Accent-folding & fuzzy:** accent-insensitive matching and cautious fuzzy matching for long names (≥ 6–7 chars).
* **People/Concept linkers:** parallel pipelines for `Person` and `Concept`.

## Troubleshooting

* **No chunks for an article:** verify the JSON input is a JSON array of strings (not JSONL) before conversion. Run the converter; then ingest the produced JSONL.
* **Zero matches:** make sure the full-text index exists and is named `chunkText`; run the linker after all articles are ingested. Check that `Place.altNames` is an array of strings and names are ≥ 3 chars.
* **Windows Unicode glitches:** always read with `utf-8-sig` and escape backslashes in regex building (the supplied converter already handles surrogate pairs and malformed `\uXXXX`).

## Provenance and licensing

* **ISAW Papers:** open access, CC-BY (site-hosted).
* **Pleiades:** open, with clear attribution requirements; keep `pleiadesId` and source `uri` in your nodes.
* **Wikidata (future enrichment):** CC0; store QIDs and source edges.

## Contributing

* Keep `MERGE`-idempotent Cypher; don’t introduce write patterns that duplicate nodes.
* Treat linkers as pure functions over existing nodes: re-runnable, measurable, and auditable.
* Add sanity queries for every new entity type and every new linker.

