import json
import re
import time
from collections import defaultdict
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

# ── paths ──────────────────────────────────────────────────────────────────────
CODE_DIR    = Path(__file__).parent
ROOT_DIR    = CODE_DIR.parent
CHROMA_DIR  = ROOT_DIR / "chroma_db"
PARSED_DIR  = CODE_DIR / "parsed_files"

JSON_FILES = {
    "docs":    [
        PARSED_DIR / "claude_docs.json",
        PARSED_DIR / "hackerrank_docs.json",
        PARSED_DIR / "visa_docs.json",
    ],
    "qa": PARSED_DIR / "visa_support.json",
}

BATCH_SIZE       = 64
MIN_WORDS        = 30
MAX_WORDS        = 400
SENTENCE_OVERLAP = 2   # sentences carried into next chunk for context
MAX_TITLE_WORDS  = 15  # titles longer than this are not prepended


# ── chunking ───────────────────────────────────────────────────────────────────
def _split_sentences(text: str) -> list[str]:
    """Split on sentence-ending punctuation (.!?) followed by whitespace."""
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p.strip() for p in parts if p.strip()]


def chunk_text(text: str, max_words: int = MAX_WORDS) -> list[str]:
    """
    Sentence-aware chunking: accumulate sentences up to max_words, then start
    a new chunk carrying the last SENTENCE_OVERLAP sentences for context.
    Never cuts mid-sentence.
    """
    sentences = _split_sentences(text)
    if not sentences:
        return []

    chunks: list[str] = []
    current: list[str] = []
    current_wc = 0

    for sent in sentences:
        sw = len(sent.split())
        if current and current_wc + sw > max_words:
            chunks.append(" ".join(current))
            current = current[-SENTENCE_OVERLAP:]
            current_wc = sum(len(s.split()) for s in current)
        current.append(sent)
        current_wc += sw

    if current:
        chunks.append(" ".join(current))

    return chunks


# ── normalise both JSON shapes into one schema ─────────────────────────────────
def load_docs() -> list[dict]:
    records = []

    # standard article docs
    for path in JSON_FILES["docs"]:
        data = json.loads(path.read_text(encoding="utf-8"))
        source = path.stem  # e.g. "claude_docs"
        for i, doc in enumerate(data):
            doc_id = f"{source}__{i}"
            records.append({
                "doc_id":   doc_id,
                "title":    doc.get("title", ""),
                "content":  doc.get("content", ""),
                "company":  doc.get("company", ""),
                "category": doc.get("category", ""),
                "url":      doc.get("url", ""),
            })

    # visa Q&A pairs — embed question + answer together, title = question
    qa_data = json.loads(JSON_FILES["qa"].read_text(encoding="utf-8"))
    for i, qa in enumerate(qa_data):
        records.append({
            "doc_id":   f"visa_qa_{i}",
            "title":    qa.get("question", ""),
            "content":  qa.get("question", "") + " " + qa.get("answer", ""),
            "company":  qa.get("company", "visa"),
            "category": qa.get("category", ""),
            "url":      qa.get("url", ""),
        })

    return records


# ── build chunks with metadata ─────────────────────────────────────────────────
def build_chunks(records: list[dict]) -> list[dict]:
    chunks = []
    for doc in records:
        title = doc["title"] or ""
        prefix = f"{title}. " if title and len(title.split()) <= MAX_TITLE_WORDS else ""
        text_chunks = chunk_text(doc["content"])
        for i, chunk in enumerate(text_chunks):
            if len(chunk.split()) < MIN_WORDS:
                continue
            chunks.append({
                "chunk_id": f"{doc['doc_id']}_chunk_{i}",
                "text":     prefix + chunk,
                "metadata": {
                    "doc_id":   doc["doc_id"],
                    "company":  doc["company"]  or "",
                    "category": doc["category"] or "",
                    "title":    doc["title"]    or "",
                    "url":      doc["url"]      or "",
                },
            })
    return chunks


# ── embed + upsert in batches ──────────────────────────────────────────────────
def index(chunks: list[dict], collection, model: SentenceTransformer):
    total = len(chunks)
    for start in range(0, total, BATCH_SIZE):
        batch = chunks[start:start + BATCH_SIZE]
        texts     = [c["text"]     for c in batch]
        ids       = [c["chunk_id"] for c in batch]
        metadatas = [c["metadata"] for c in batch]

        embeddings = model.encode(texts, show_progress_bar=False).tolist()

        collection.upsert(
            ids        = ids,
            documents  = texts,
            embeddings = embeddings,
            metadatas  = metadatas,
        )
        print(f"  indexed {min(start + BATCH_SIZE, total)}/{total} chunks", end="\r")
    print()


# ── stats ──────────────────────────────────────────────────────────────────────
def print_stats(records: list[dict], chunks: list[dict]):
    chunks_per_company  = defaultdict(int)
    docs_per_company    = defaultdict(int)

    for c in chunks:
        chunks_per_company[c["metadata"]["company"]] += 1
    for r in records:
        docs_per_company[r["company"]] += 1

    print("\n──────────────────────────────────────")
    print("  INDEXING STATS")
    print("──────────────────────────────────────")
    print(f"  Total documents loaded : {len(records)}")
    print(f"  Total chunks created   : {len(chunks)}")
    print(f"  Avg chunks / document  : {len(chunks)/len(records):.1f}")
    print()
    print(f"  {'Company':<15} {'Docs':>6} {'Chunks':>8} {'Avg':>6}")
    print(f"  {'-'*15} {'-'*6} {'-'*8} {'-'*6}")
    for company in sorted(docs_per_company):
        d = docs_per_company[company]
        c = chunks_per_company[company]
        print(f"  {company:<15} {d:>6} {c:>8} {c/d:>6.1f}")
    print("──────────────────────────────────────\n")


# ── main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading sentence-transformers model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("Setting up ChromaDB...")
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection(
        name="support_docs",
        metadata={"hnsw:space": "cosine"},
    )

    print("Loading JSON files...")
    records = load_docs()
    print(f"  {len(records)} documents loaded")

    print("Chunking content...")
    chunks = build_chunks(records)
    print(f"  {len(chunks)} chunks ready")

    print("Embedding and indexing...")
    t0 = time.time()
    index(chunks, collection, model)
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s")

    print_stats(records, chunks)
    print(f"ChromaDB persisted at: {CHROMA_DIR.resolve()}")
    print(f"Collection count: {collection.count()} vectors")
