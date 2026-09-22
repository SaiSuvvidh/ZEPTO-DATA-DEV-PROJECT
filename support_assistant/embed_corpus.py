"""
support_assistant/embed_corpus.py -- loads the 8 policy docs, embeds each
(one chunk per doc, since each is already a short self-contained
paragraph -- further splitting would fragment a single policy statement
without benefit) using all-MiniLM-L6-v2 locally, and stores them in a
persistent ChromaDB collection.
"""
import os
import chromadb
from sentence_transformers import SentenceTransformer

DOCS_DIR = "support_assistant/docs"
CHROMA_PATH = "support_assistant/chroma_db"
COLLECTION_NAME = "zepto_policies"


def load_docs():
    docs = {}
    for filename in sorted(os.listdir(DOCS_DIR)):
        if filename.endswith(".txt"):
            doc_id = filename.replace(".txt", "")  # e.g. "doc_01"
            with open(os.path.join(DOCS_DIR, filename), "r", encoding="utf-8") as f:
                docs[doc_id] = f.read()
    return docs


def embed_and_store(docs):
    model = SentenceTransformer("all-MiniLM-L6-v2")  # runs locally, no API key

    client = chromadb.PersistentClient(path=CHROMA_PATH)
    # Recreate collection each run so this script is idempotent/rerunnable
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(COLLECTION_NAME)

    doc_ids = list(docs.keys())
    texts = list(docs.values())
    embeddings = model.encode(texts).tolist()

    collection.add(
        ids=doc_ids,
        embeddings=embeddings,
        documents=texts,
    )
    print(f"Embedded and stored {len(doc_ids)} chunks in ChromaDB collection '{COLLECTION_NAME}'")
    return collection, model


def test_retrieval(collection, model):
    # Sanity check: a delivery-related query should retrieve doc_01
    test_query = "How much does delivery cost?"
    query_embedding = model.encode([test_query]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=3)

    print(f"\n=== Test retrieval for: '{test_query}' ===")
    for doc_id, distance, doc_text in zip(
        results["ids"][0], results["distances"][0], results["documents"][0]
    ):
        print(f"{doc_id} (distance={distance:.4f}): {doc_text[:100]}...")


def main():
    docs = load_docs()
    collection, model = embed_and_store(docs)
    test_retrieval(collection, model)


if __name__ == "__main__":
    main()