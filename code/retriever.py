import chromadb
from sentence_transformers import SentenceTransformer
from pathlib import Path

CHROMA_DIR     = Path(__file__).parent.parent / "chroma_db"
COLLECTION     = "support_docs"
EMBED_MODEL    = "all-MiniLM-L6-v2"

_client     = None
_collection = None
_model      = None


def _init():
    global _client, _collection, _model
    if _collection is None:
        _client     = chromadb.PersistentClient(path=str(CHROMA_DIR))
        _collection = _client.get_collection(COLLECTION)
        _model      = SentenceTransformer(EMBED_MODEL)


def retrieve(query: str, company: str | None = None, top_k: int = 5) -> list[dict]:
    """
    Query the vector store.
    If company is provided (and not 'none'/'nan'), filter results to that company.
    Returns list of {text, metadata, score} dicts.
    """
    _init()

    embedding = _model.encode(query).tolist()

    where = None
    if company and company.lower() not in ("none", "nan", ""):
        where = {"company": {"$eq": company.lower()}}

    results = _collection.query(
        query_embeddings=[embedding],
        n_results=top_k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    for text, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append({
            "text":     text,
            "title":    meta.get("title", ""),
            "category": meta.get("category", ""),
            "url":      meta.get("url", ""),
            "score":    round(1 - dist, 4),   # cosine similarity
        })

    return chunks


if __name__ == "__main__":
    test_cases = [
        ("How do I reinvite a candidate to a test?",     "HackerRank"),
        ("How do I delete my Claude account?",           "Claude"),
        ("My Visa card was stolen, what do I do?",       "Visa"),
        ("Site is completely down, nothing works",        None),
    ]
    for query, company in test_cases:
        print(f"\nQuery : {query}")
        print(f"Company: {company}")
        hits = retrieve(query, company=company, top_k=3)
        for i, h in enumerate(hits, 1):
            print(f"  [{i}] score={h['score']}  title={h['title'][:60]}")
