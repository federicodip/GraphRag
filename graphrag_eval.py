#!/usr/bin/env python
import os
import json
import re
import csv
import shutil
from dotenv import load_dotenv
from openai import OpenAI

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.documents import Document

from langchain_community.vectorstores import Chroma
from langchain_community.graphs import Neo4jGraph
from langchain_core.prompts import PromptTemplate

# ============================================================
# ENV + GLOBAL CONFIG
# ============================================================

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Neo4j config
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME") or os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD") or os.getenv("NEO4J_PASS")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "graphrag")

# Paths
CHUNKS_PATH = "chunks_isaw_papers_all.txt"
QA_PATH = "ground_truth.txt"  # must contain a JSON list of {instruction, output}

PERSIST_DIR = "./docs/chroma_hybrid"

# Progress + outputs
RESULTS_JSON = "hybrid_results_ground_truth.json"
RESULTS_CSV = "hybrid_results_ground_truth.csv"

# How many NEW questions to process per run
BATCH_SIZE = 206   # set to 10 / 50 / whatever


# ============================================================
# LOAD CHUNKS AND BUILD / LOAD VECTORSTORE (Chroma)
# ============================================================

def extract_source(chunk: str) -> str:
    marker = "Source: "
    if marker in chunk:
        source_start = chunk.rfind(marker) + len(marker)
        return chunk[source_start:].strip()
    return "Unknown"


# Embeddings
embedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# Either load existing Chroma or build it once
if os.path.exists(PERSIST_DIR):
    print(f"Using existing Chroma index at {PERSIST_DIR}")
    vectordb = Chroma(
        embedding_function=embedding,
        persist_directory=PERSIST_DIR,
    )
else:
    print(f"Building new Chroma index at {PERSIST_DIR}")
    documents = []
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        content = f.read()
        chunks = json.loads(content)
        for chunk in chunks:
            source = extract_source(chunk)
            documents.append(Document(page_content=chunk, metadata={"source": source}))

    os.makedirs(os.path.dirname(PERSIST_DIR), exist_ok=True)
    vectordb = Chroma.from_documents(
        documents=documents,
        embedding=embedding,
        persist_directory=PERSIST_DIR,
    )

retriever = vectordb.as_retriever(search_kwargs={"k": 8})


# ============================================================
# GRAPH SETUP (Neo4j + Cypher generator)
# ============================================================

kg = Neo4jGraph(
    url=NEO4J_URI,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD,
    database=NEO4J_DATABASE,
)

kg.refresh_schema()
print("=== GRAPH SCHEMA (from Neo4j) ===")
print(kg.schema)

CYPHER_GENERATION_TEMPLATE = """Task: Generate a Cypher statement to query a graph database.

Instructions:
- Use ONLY the relationship types, node labels, and properties that appear in the schema.
- Do NOT invent labels, relationships, or properties that are not in the schema.
- The goal is to retrieve relevant text snippets (chunks) that help answer the user's question.
- Chunks live on :Chunk nodes with a `text` property.
- You may traverse via:
  - (:Article)-[:HAS_CHUNK]->(:Chunk)
  - (:Chunk)-[:MENTIONS]->(:Place|:Person|:Concept)
  - (:Person)-[:AUTHORED]->(:Article)
- You MUST return a single column alias called `text_chunk` for the chunk text,
  e.g. `RETURN c.text AS text_chunk`.

Schema:
{schema}

Notes:
- Return only the Cypher statement, nothing else.
- Do not include explanations or comments.
- Use LIMIT to avoid returning huge result sets (e.g. LIMIT 20).

Examples:

# Example: chunks mentioning a specific place name (using Place.title)
MATCH (p:Place)
WHERE p.title CONTAINS "Athens"
MATCH (c:Chunk)-[:MENTIONS]->(p)
RETURN c.text AS text_chunk
LIMIT 20

# Example: chunks from an article about a topic (using Article.title)
MATCH (a:Article)
WHERE a.title CONTAINS "Antikythera"
MATCH (a)-[:HAS_CHUNK]->(c:Chunk)
RETURN c.text AS text_chunk
LIMIT 20

# Example: chunks linked to a specific concept (using Concept.name)
MATCH (k:Concept)
WHERE k.name CONTAINS "planet"
MATCH (c:Chunk)-[:MENTIONS]->(k)
RETURN c.text AS text_chunk
LIMIT 20

The question is:
{question}
"""

cypher_prompt = PromptTemplate(
    input_variables=["schema", "question"],
    template=CYPHER_GENERATION_TEMPLATE,
)

cypher_llm = ChatOpenAI(model_name="gpt-4", temperature=0)


def generate_cypher(question: str) -> str:
    return cypher_llm.invoke(
        cypher_prompt.format(schema=kg.schema, question=question)
    ).content.strip()


def get_graph_context(question: str, max_chunks: int = 10) -> str:
    try:
        cypher = generate_cypher(question)
        print("\n[Cypher generated]")
        print(cypher)
        rows = kg.query(cypher)
    except Exception as e:
        print(f"[Graph error] {e}")
        return ""

    texts = []
    for row in rows[:max_chunks]:
        if isinstance(row, dict):
            t = row.get("text_chunk")
            if isinstance(t, str) and t.strip():
                texts.append(t.strip())
        else:
            texts.append(str(row))

    return "\n\n".join(texts)


def get_vector_context(question: str, max_docs: int = 8) -> str:
    docs = retriever.invoke(question)
    if not isinstance(docs, list):
        return ""
    docs = docs[:max_docs]
    return "\n\n".join(
        d.page_content for d in docs if isinstance(d.page_content, str)
    )


# ============================================================
# LOAD GROUND TRUTH + RESUME LOGIC
# ============================================================

with open(QA_PATH, "r", encoding="utf-8") as f:
    all_data = json.load(f)

if not isinstance(all_data, list):
    raise ValueError("ground_truth must be a JSON list of {instruction, output} objects.")

total_questions = len(all_data)

# Load existing results if any
if os.path.exists(RESULTS_JSON):
    with open(RESULTS_JSON, "r", encoding="utf-8") as f:
        results = json.load(f)
    print(f"Loaded {len(results)} existing results from {RESULTS_JSON}")
else:
    results = []

start_index = len(results)
if start_index >= total_questions:
    print("All questions already processed. Nothing to do.")
    exit(0)

end_index = min(start_index + BATCH_SIZE, total_questions)
batch = all_data[start_index:end_index]

print(f"Processing questions {start_index} to {end_index - 1} (batch size {len(batch)})")


# ============================================================
# ANSWER LLM + GRADER
# ============================================================

answer_llm = ChatOpenAI(model_name="gpt-4", temperature=0.2)
openai_client = OpenAI(api_key=OPENAI_API_KEY)


def answer_with_hybrid(question: str) -> str:
    graph_ctx = get_graph_context(question)
    vec_ctx = get_vector_context(question)

    context = ""
    if graph_ctx:
        context += "GRAPH CONTEXT:\n" + graph_ctx + "\n\n"
    if vec_ctx:
        context += "VECTOR CONTEXT:\n" + vec_ctx + "\n\n"

    if not context:
        return "I don't have enough information in the provided corpus to answer this."

    prompt = (
        "You are an expert on the ancient world and the ISAW Papers corpus.\n"
        "Use ONLY the information in the context below to answer the question.\n"
        "If the context is insufficient, say you don't know.\n\n"
        f"{context}"
        f"Question: {question}\n\n"
        "Answer in a concise paragraph, citing authors/papers if mentioned in the context."
    )

    resp = answer_llm.invoke(prompt)
    return resp.content.strip()


def evaluate_reference_guided_grading(question, correct_answer, model_answer):
    evaluation_prompt = (
        "Evaluate the following model answer compared to the correct answer.\n"
        "Provide a numeric score from 1 (completely inaccurate) to 10 (completely accurate).\n"
        "Return only the number.\n\n"
        f"Question: {question}\n\n"
        f"Correct Answer: {correct_answer}\n\n"
        f"Model Answer: {model_answer}\n\n"
        "Score (1-10):"
    )
    response = openai_client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[{"role": "user", "content": evaluation_prompt}],
        temperature=0,
        max_tokens=10,
    )
    raw_output = response.choices[0].message.content.strip()
    match = re.search(r"\b([1-9]|10)\b", raw_output)
    if match:
        return int(match.group(1))
    else:
        print(f"Unexpected score format: '{raw_output}'")
        return 0


# ============================================================
# EVAL LOOP (HYBRID, BATCHED + RESUMABLE)
# ============================================================

tp_hybrid = 0
fp_hybrid = 0
scores_hybrid = []

for offset, qa_pair in enumerate(batch):
    idx = start_index + offset
    question = qa_pair["instruction"]
    correct_answer = qa_pair["output"]

    print(f"\n=== [{idx+1}/{total_questions}] Question ===")
    print(question)

    hybrid_answer = answer_with_hybrid(question)
    print("\n[Hybrid answer]")
    print(hybrid_answer)

    score = evaluate_reference_guided_grading(question, correct_answer, hybrid_answer)
    scores_hybrid.append(score)
    if score > 5:
        tp_hybrid += 1
    else:
        fp_hybrid += 1

    results.append(
        {
            "index": idx,
            "question": question,
            "correct_answer": correct_answer,
            "hybrid_answer": hybrid_answer,
            "score_hybrid": score,
        }
    )

# Save updated JSON results (full list, not just this batch)
with open(RESULTS_JSON, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print(f"\nUpdated results saved to {RESULTS_JSON}")


# ============================================================
# METRICS FOR THIS BATCH ONLY
# ============================================================

def ratio(tp, fp):
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0


acceptability_ratio = ratio(tp_hybrid, fp_hybrid)
avg_score = sum(scores_hybrid) / len(scores_hybrid) if scores_hybrid else 0.0

print("\n--- HYBRID (graph + vector) Batch Evaluation Summary ---")
print(
    f"Batch size: {len(batch)}, "
    f"Avg score: {avg_score:.2f}, "
    f"acceptability_ratio (>5): {acceptability_ratio:.2f}"
)

# Also (re)write CSV with all results so far
with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

print(f"CSV with all results so far saved to {RESULTS_CSV}")
