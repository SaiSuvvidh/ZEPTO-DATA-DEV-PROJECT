# Support Assistant Module

A RAG-based support assistant for Zepto's policy corpus — local embeddings, ChromaDB retrieval, a LangGraph intent router, Pydantic-validated output, and a FastAPI wrapper, fully functional offline through a deterministic mock-LLM baseline.

## Run
```bash
python setup_corpus.py     # writes the 8 policy doc files to docs/
python embed_corpus.py     # embeds (all-MiniLM-L6-v2) + stores in ChromaDB
uvicorn main:app --reload --port 8000

# Docker (build from repo root, not this folder):
docker build -t zepto-support-assistant -f support_assistant/Dockerfile .
docker run -p 7860:7860 zepto-support-assistant
```
`MOCK_LLM` defaults to `1` — no LLM API call, no key, no network access needed anywhere in this path.

## How I Built This

I started with the corpus and embeddings — writing the 8 provided policy documents to disk, one chunk per document, since each is already a short, self-contained policy statement and splitting further would just fragment a single answer without giving retrieval anything more to work with.Then embedded each one locally with `all-MiniLM-L6-v2` and stored the vectors in a ChromaDB collection, then tested retrieval right away with a sample query ("How much does delivery cost?") and confirmed it correctly returned the delivery-policy document as the closest match before building anything else on top of it.

Then I wrote the structured prompt template — role/context/task/format/length sections, a negative constraint telling the model not to answer outside the provided context, and one few-shot example. This only actually gets exercised by the optional real-LLM extension as per requirement.

The LangGraph piece came next: a 3-node graph (`classify_intent`, `retrieve_and_answer`, `direct_answer`) with a conditional edge routing between the last two based on a keyword heuristic. I tested this directly by invoking the graph with two sample queries — one policy-related, one not — and confirmed both routed correctly and produced the right canned responses.

I wrapped all of that in a FastAPI app with a single `POST /ask` endpoint, ran it locally with uvicorn, and tested it with two curl calls to get the transcripts below.

**The Docker detour.** Building the Dockerfile is where I actually hit a real problem. My first build succeeded, but the container crashed on startup with `ModuleNotFoundError: No module named 'langgraph'` — even though I'd installed it, and even though the app ran fine locally outside Docker. Tracking it down, the issue turned out to be upstream of Docker entirely: when I'd run `pip freeze > requirements.txt` earlier, `pip` on my PATH wasn't actually resolving to my activated venv's pip — despite my terminal showing `(venv)` — so the file I'd generated was missing `langgraph`, `chromadb`, `sentence-transformers`, `fastapi`, and `uvicorn` altogether. I confirmed this with `where pip`, which listed the venv's pip but apparently wasn't the one actually being invoked, and fixed it for good by using `python -m pip freeze` instead, which reliably resolves through whichever `python` the venv puts on PATH rather than trusting a possibly-ambiguous `pip` alias. Once I regenerated `requirements.txt` correctly and rebuilt, the container started clean and `/ask` responded correctly.

## RAG Pipeline Architecture

**Ingestion:** the 8 Zepto policy documents are plain `.txt` files under `docs/`, one chunk per document.

**Embedding:** `embed_corpus.py` encodes each document with `all-MiniLM-L6-v2` and stores the vectors in a persistent ChromaDB collection (`zepto_policies`) under `chroma_db/`.

**Retrieval:** `graph.py`'s `retrieve_and_answer` node embeds the incoming query with the same model and pulls the top-3 most similar chunks from ChromaDB by cosine similarity. This step runs for real regardless of `MOCK_LLM` — it needs no key and makes no LLM call either way.

**Generation:** `classify_intent` routes each query through a keyword check (terms like "delivery", "return", "cancel") via a conditional edge into either `retrieve_and_answer` or `direct_answer`. Only the generation step inside each of those two nodes actually branches on `MOCK_LLM`. With it left at the default (`1`), `retrieve_and_answer` returns `f"Based on the retrieved context: {top_chunk_snippet}"` built from the top retrieved chunk, and `direct_answer` returns a fixed refusal string — both populate the `AskResponse` schema (`answer`/`sources`/`confidence`) directly from code, with no LLM in the loop at all. If I ever set `MOCK_LLM=0`, the same two nodes would instead call a real LLM using the prompt template in `prompt_template.py`, with the retry-on-validation-failure logic in `main.py` handling any malformed output.

**Data flow:** `docs/*.txt` → `embed_corpus.py` → `chroma_db/` → `graph.py`'s `retrieve_and_answer` → `AskResponse` → `main.py`'s `POST /ask` → JSON back to the caller.

## Example API Calls (MOCK_LLM=1, default)

**Policy question — routes to `retrieve_and_answer`:**
```
POST /ask {"query": "Can I cancel my order?"}
```
```json
{"answer":"Based on the retrieved context: Order Cancellation Policy: Orders can be cancelled free of cost any time before the order status changes to 'Packed', typically within the first 2 minutes of placing the order. Once an order has been ","sources":["doc_05","doc_06","doc_02"],"confidence":1.0}
```

**General question — routes to `direct_answer`:**
```
POST /ask {"query": "What is the capital of France?"}
```
```json
{"answer":"I can only answer questions about Zepto policies right now.","sources":[],"confidence":1.0}
```

## Docker

Builds and runs cleanly now (after the `requirements.txt` fix above), serving `/ask` on port 7860. The build context has to be the repo root, not this folder, since the app imports as `support_assistant.main`. On a fresh container with no local Hugging Face cache, the first request takes a bit longer while `all-MiniLM-L6-v2` downloads.