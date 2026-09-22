# Support Assistant Module

A RAG-based GenAI support assistant for Zepto's policy corpus: local embeddings, ChromaDB retrieval, a LangGraph intent router, Pydantic-validated output, and a FastAPI wrapper — fully functional offline via a deterministic mock-LLM baseline.

## Run
```bash
python setup_corpus.py     # writes the 8 policy doc files to docs/
python embed_corpus.py     # embeds (all-MiniLM-L6-v2) + stores in ChromaDB
uvicorn main:app --reload --port 8000

# Docker (build from repo root, not this folder):
docker build -t zepto-support-assistant -f support_assistant/Dockerfile .
docker run -p 7860:7860 zepto-support-assistant
```
`MOCK_LLM` defaults to `1` — no LLM API call, no key, no network access required. This is the graded baseline.

## RAG Pipeline Architecture

**Ingestion:** 8 Zepto policy documents as plain `.txt` files in `docs/`, one chunk per document (each already a short, self-contained policy statement).

**Embedding:** `embed_corpus.py` encodes each document locally with `sentence-transformers`' `all-MiniLM-L6-v2` and stores the vectors in a persistent ChromaDB collection (`zepto_policies`) at `chroma_db/`.

**Retrieval:** `graph.py`'s `retrieve_and_answer` node embeds the incoming query with the same model and retrieves the top-3 most similar chunks from ChromaDB by cosine similarity. This step always runs for real, regardless of `MOCK_LLM`.

**Generation:** `classify_intent` routes each query (via a keyword heuristic checking for terms like "delivery", "return", "cancel") to either `retrieve_and_answer` or `direct_answer` through a conditional edge. Only the *generation* step inside each node branches on `MOCK_LLM`: in mock mode (default), `retrieve_and_answer` returns `f"Based on the retrieved context: {top_chunk_snippet}"`, and `direct_answer` returns a fixed refusal string — both populate the `AskResponse` Pydantic schema (`answer`/`sources`/`confidence`) deterministically, with no LLM call. The optional `MOCK_LLM=0` extension would instead call a real LLM using the structured prompt in `prompt_template.py` (role–context–task–format–length, with a negative constraint and a few-shot example), with schema-validation retry logic in `main.py`.

**Data flow:** `docs/*.txt` → `embed_corpus.py` → `chroma_db/` → `graph.py` (`retrieve_and_answer`) → `AskResponse` → `main.py`'s `POST /ask` → JSON response.

## Example API Calls (MOCK_LLM=1, default)

**Policy question (routes to `retrieve_and_answer`):**
```
POST /ask {"query": "Can I cancel my order?"}
```
```json
{"answer":"Based on the retrieved context: Order Cancellation Policy: Orders can be cancelled free of cost any time before the order status changes to 'Packed', typically within the first 2 minutes of placing the order. Once an order has been ","sources":["doc_05","doc_06","doc_02"],"confidence":1.0}
```

**General question (routes to `direct_answer`):**
```
POST /ask {"query": "What is the capital of France?"}
```
```json
{"answer":"I can only answer questions about Zepto policies right now.","sources":[],"confidence":1.0}
```

## Docker
Confirmed buildable and runnable locally (`docker build` + `docker run`, serving `/ask` on port 7860). Build context must be the repo root (imports as `support_assistant.main`), not this subfolder. On first run inside a container, `all-MiniLM-L6-v2` downloads fresh (no local HF cache), so the first request takes longer than subsequent local runs.