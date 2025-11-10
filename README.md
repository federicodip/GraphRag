# GraphRag (prototype)

- Ingests ISAW article chunks into Neo4j with embeddings.
- Secrets live in `.env` (see `.env.example`).
- Create DB `graphrag` in Neo4j 5.x, run constraints, then:

Run:

    python ingest_one_article.py
