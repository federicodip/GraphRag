import os
from dotenv import load_dotenv
from langchain_community.graphs import Neo4jGraph

# Load .env
load_dotenv()

# Read Neo4j credentials from env
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME") or os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD") or os.getenv("NEO4J_PASS")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "graphrag")

# Connect to graph
kg = Neo4jGraph(
    url=NEO4J_URI,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD,
    database=NEO4J_DATABASE
)

# Print schema
kg.refresh_schema()
print("=== GRAPH SCHEMA ===")
print(kg.schema)
