# pipeline.py
from pathlib import Path

from ingest import ingest_pdf
from embed import embed_chunks
from store import VectorStore
from retrieve import retrieve
from generate import generate_answer

INDEX_PATH = "rag_index"
EMBED_MODEL = "all-MiniLM-L6-v2"


def ingest_document(pdf_path: str, store: VectorStore = None, chunk_size: int = 200, overlap: int = 40) -> VectorStore:
    """PDF -> chunk records -> vectors -> store.add(), skipping files already indexed."""
    filename = Path(pdf_path).name

    if store is None:
        if Path(INDEX_PATH + ".faiss").exists():
            store = VectorStore.load(INDEX_PATH)
            print(f"[ingest] loaded existing store with {len(store.chunks)} chunks")
        else:
            store = VectorStore(dim=384)  # all-MiniLM-L6-v2's output dimension

    already_ingested = any(c.get("source") == filename for c in store.chunks)
    if already_ingested:
        print(f"[ingest] {filename} already ingested — skipping")
        return store

    records = ingest_pdf(pdf_path, chunk_size=chunk_size, overlap=overlap, model_name=EMBED_MODEL)
    print(f"[ingest] {filename}: {len(records)} chunks")

    if not records:
        print(f"[ingest] WARNING: no text extracted from {pdf_path} — skipping")
        return store

    vectors = embed_chunks(records, model_name=EMBED_MODEL)

    chunk_texts = [r["text"] for r in records]
    metadatas = [{"source": r["source"], "page": r["page"], "chunk_id": r["chunk_id"]} for r in records]

    store.add(vectors, chunk_texts, metadatas)
    store.save(INDEX_PATH)
    print(f"[ingest] store now has {len(store.chunks)} total chunks, saved to {INDEX_PATH}.*")

    return store


def ask(question: str, store: VectorStore, k: int = 5) -> dict:
    retrieved_chunks = retrieve(question, store, k=k)
    answer = generate_answer(question, retrieved_chunks)
    return {"question": question, "answer": answer, "sources": retrieved_chunks}


if __name__ == "__main__":
    store = None
    docs_dir = Path("documents")
    for pdf_path in docs_dir.glob("*.pdf"):
        store = ingest_document(str(pdf_path), store=store, chunk_size=200, overlap=40)

    print("\nRAG pipeline ready. Type a question (or 'quit'):\n")
    while True:
        question = input("> ")
        if question.strip().lower() in ("quit", "exit"):
            break

        result = ask(question, store, k=5)
        print(f"\nAnswer: {result['answer']}\n")
        print("Sources:")
        for i, src in enumerate(result["sources"], start=1):
            print(f"  [{i}] {src.get('source', '?')} p.{src.get('page', '?')} (score={src['score']:.3f})")
        print()