# retrieve.py
from embed import get_model
from store import VectorStore


def retrieve(
    question: str,
    store: VectorStore,
    k: int = 5,
    score_threshold: float = None,
    model_name: str = "all-MiniLM-L6-v2",
) -> list[dict]:
    """Embed the question with the same model used to build the index, then fetch top-k chunks."""
    model = get_model(model_name)

    query_vector = model.encode(
        question,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype("float32")

    results = store.search(query_vector, k=k)

    if score_threshold is not None:
        results = [r for r in results if r["score"] >= score_threshold]

    return results


if __name__ == "__main__":
    # quick smoke test — assumes a saved store from store.py's save()
    store = VectorStore.load("test_store")

    question = "What does FAISS do?"
    top_chunks = retrieve(question, store, k=3)

    print(f"Question: {question}\n")
    for i, chunk in enumerate(top_chunks, start=1):
        print(f"[{i}] score={chunk['score']:.3f} — {chunk['text']}")