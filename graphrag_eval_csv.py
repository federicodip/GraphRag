#!/usr/bin/env python
import re
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

# CSV with question / answer pairs.
# Expected format: question<TAB>answer (no header, or a header row "question" / "answer").
QA_PATH = "failed_qa.csv"

PERSIST_DIR = "./docs/chroma_hybrid"

# Progress + outputs
RESULTS_JSON = "hybrid_results_ground_truth_failed.json"
RESULTS_CSV = "hybrid_results_ground_truth_failed.csv"

# How many questions to process per run (here: all)
BATCH_SIZE = 100   # no longer used for resume, but keep if you want to reintroduce batching


# ============================================================
# LOAD CHUNKS AND BUILD / LOAD VECTORSTORE (Chroma)
# ============================================================

def extract_quoted_title(question: str) -> str | None:
    """
    If the question contains a double-quoted phrase, return it.
    Example: '... article "Current Practice in Linked Open Data for the Ancient World"?'
    → 'Current Practice in Linked Open Data for the Ancient World'
    """
    m = re.search(r'"([^"]+)"', question)
    if m:
        return m.group(1).strip()
    return None


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

def strip_c_text_filters(cypher: str) -> str:
    """
    Remove WHERE clauses that filter directly on c.text.
    This prevents over-narrow keyword filtering like:
        WHERE c.text CONTAINS "funded"
    while leaving other filters (on Person, Article, etc.) alone.
    """
    new_lines = []
    for line in cypher.splitlines():
        # crude but effective: skip WHERE lines that mention c.text
        if "WHERE" in line.upper() and "c.text" in line:
            continue
        new_lines.append(line)
    return "\n".join(new_lines)


def clean_cypher(cypher: str) -> str:
    """
    Remove comment / junk lines from LLM-generated Cypher.
    - Drops empty lines.
    - Drops lines starting with '#'.
    """
    lines = []
    for line in cypher.splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith("#"):
            continue
        lines.append(line)
    return "\n".join(lines)


def generate_cypher(question: str) -> str:
    return cypher_llm.invoke(
        cypher_prompt.format(schema=kg.schema, question=question)
    ).content.strip()

def get_graph_context(question: str, max_chunks: int = 10) -> str:
    """Generate Cypher, clean it, run it; prioritize direct article-title lookup."""
    rows = []

    # 0) Try direct article-title lookup first if there is a quoted title
    title = extract_quoted_title(question)
    if title:
        print(f"[Title mode] querying by article title {title!r}]")
        try:
            rows = kg.query(
                """
                MATCH (a:Article)
                WHERE toLower(a.title) CONTAINS toLower($title)
                MATCH (a)-[:HAS_CHUNK]->(c:Chunk)
                RETURN a.title AS article_title, c.text AS text_chunk
                LIMIT 50
                """,
                {"title": title},
            )
        except Exception as e:
            print(f"[Title-mode graph error] {e}")
            rows = []

    # 1) If title-mode failed, try LLM-generated Cypher
    if not rows:
        try:
            raw_cypher = generate_cypher(question)
            cypher = clean_cypher(raw_cypher)
            cypher = strip_c_text_filters(cypher)

            print("\n[Cypher generated]")
            print(cypher)

rows = kg.query(cypher)

            rows = kg.query(cypher)
        except Exception as e:
            print(f"[Graph error] {e}")
            rows = []

    # 2) Build context string
    texts = []
    for row in rows[:max_chunks]:
        if isinstance(row, dict):
            # Support both our fallback shape and whatever the LLM produced
            chunk = (
                row.get("text_chunk")
                or row.get("c.text")
                or row.get("text")
            )
            article_title = (
                row.get("article_title")
                or row.get("a.title")
            )

            if isinstance(chunk, str) and chunk.strip():
                if isinstance(article_title, str) and article_title.strip():
                    texts.append(f"[{article_title.strip()}]\n{chunk.strip()}")
                else:
                    texts.append(chunk.strip())
        else:
            texts.append(str(row))

    return "\n\n".join(texts)


def get_vector_context(question: str, max_docs: int = 8) -> str:
    # If there's a quoted article title, bias retrieval with that
    title = extract_quoted_title(question)
    if title:
        query = f'"{title}" ISAW Papers article'
    else:
        query = question

    docs = retriever.invoke(query)
    if not isinstance(docs, list):
        return ""
    docs = docs[:max_docs]
    return "\n\n".join(
        d.page_content for d in docs if isinstance(d.page_content, str)
    )


# ============================================================
# LOAD GROUND TRUTH FROM CSV (NO RESUME)
# ============================================================

def load_ground_truth_from_csv(path: str):
    """
    Load question/answer pairs from a comma-separated CSV.

    Expected format (what you actually have):
    - question,answer
    - answers may be multi-line but are wrapped in double quotes
    - optional first row header containing 'question' and 'answer'
    """
    all_data = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        # standard CSV: comma separator, double-quote for quoting
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            # skip completely empty rows
            if not row or all(not (c or "").strip() for c in row):
                continue

            # Optional header detection
            if i == 0 and len(row) >= 2:
                h0 = (row[0] or "").strip().lower()
                h1 = (row[1] or "").strip().lower()
                if "question" in h0 and "answer" in h1:
                    continue  # skip header

            # require at least question + answer
            if len(row) < 2:
                continue

            question = (row[0] or "").strip()
            correct_answer = (row[1] or "").strip()

            if not question or not correct_answer:
                # drop malformed rows
                continue

            all_data.append({"instruction": question, "output": correct_answer})

    if not all_data:
        raise ValueError("No question/answer pairs loaded from CSV.")
    return all_data


all_data = load_ground_truth_from_csv(QA_PATH)
total_questions = len(all_data)
print(f"Loaded {total_questions} QA pairs from {QA_PATH}")

# Always start from scratch for this CSV
results = []
start_index = 0
end_index = total_questions
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
# EVAL LOOP (HYBRID)
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

# Save updated JSON results (this run only)
with open(RESULTS_JSON, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print(f"\nUpdated results saved to {RESULTS_JSON}")


# ============================================================
# METRICS FOR THIS RUN
# ============================================================

def ratio(tp, fp):
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0


acceptability_ratio = ratio(tp_hybrid, fp_hybrid)
avg_score = sum(scores_hybrid) / len(scores_hybrid) if scores_hybrid else 0.0

print("\n--- HYBRID (graph + vector) Evaluation Summary ---")
print(
    f"Batch size: {len(batch)}, "
    f"Avg score: {avg_score:.2f}, "
    f"acceptability_ratio (>5): {acceptability_ratio:.2f}"
)

# Also write CSV for this run
if results:
    with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"CSV with all results for this run saved to {RESULTS_CSV}")
else:
    print("No results to write to CSV.")
