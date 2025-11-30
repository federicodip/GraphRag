#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Clean up Person nodes:
- KEEP all Person nodes that have an :AUTHORED relationship to an Article.
- DELETE all other Person nodes (and any MENTIONS edges pointing to them).

Run this AFTER you've ingested all articles.
"""

import os
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

NEO4J_URI  = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASS = os.getenv("NEO4J_PASSWORD") or os.getenv("NEO4J_PASS")
NEO4J_DB   = os.getenv("NEO4J_DATABASE", "graphrag")


CYPHER_COUNTS = """
MATCH (p:Person)
OPTIONAL MATCH (p)-[:AUTHORED]->(:Article)
WITH p, count(*) AS authoredCount
RETURN
  count(p) AS totalPersons,
  sum(CASE WHEN authoredCount > 0 THEN 1 ELSE 0 END) AS authors,
  sum(CASE WHEN authoredCount = 0 THEN 1 ELSE 0 END) AS nonAuthors
"""

CYPHER_DELETE_NON_AUTHORS = """
MATCH (p:Person)
WHERE NOT (p)-[:AUTHORED]->(:Article)
DETACH DELETE p
"""

def main():
    if not NEO4J_PASS:
        raise SystemExit("Neo4j password missing (NEO4J_PASSWORD or NEO4J_PASS).")

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))
    with driver.session(database=NEO4J_DB) as session:
        print(f"Connected to Neo4j DB='{NEO4J_DB}' at {NEO4J_URI} as user='{NEO4J_USER}'")

        # Before
        before = session.run(CYPHER_COUNTS).single()
        print(
            f"[BEFORE] Person total={before['totalPersons']}, "
            f"authors={before['authors']}, nonAuthors={before['nonAuthors']}"
        )

        # Delete junk persons
        session.run(CYPHER_DELETE_NON_AUTHORS)

        # After
        after = session.run(CYPHER_COUNTS).single()
        print(
            f"[AFTER] Person total={after['totalPersons']}, "
            f"authors={after['authors']}, nonAuthors={after['nonAuthors']}"
        )

    driver.close()
    print("Done. All non-author Person nodes removed.")


if __name__ == "__main__":
    main()