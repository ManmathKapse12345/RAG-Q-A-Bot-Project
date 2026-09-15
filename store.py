# store.py
import faiss
import numpy as np
import pickle
from pathlib import Path


class VectorStore:
    """
    Wraps a FAISS flat index plus a parallel list of chunk metadata,
    so a search result can return readable text, not just vector IDs.
    """

    def __init__(self, dim: int):
        self.dim = dim
        # IndexFlatIP = inner product. If you normalize vectors to unit
        # length first, inner product == cosine similarity.
        # Use IndexFlatL2 instead if you prefer raw Euclidean distance.
        self.index = faiss.IndexFlatIP(dim)
        self.chunks = []  # parallel list: self.chunks[i] <-> vector at row i

    def add(self, vectors: np.ndarray, chunk_texts: list[str], metadatas: list[dict] = None):
        """
        vectors: (n, dim) float32 array of embeddings
        chunk_texts: list of n strings, the original chunk content
        metadatas: optional list of n dicts (e.g. {"source": "doc.pdf", "page": 3})
        """
        assert vectors.shape[0] == len(chunk_texts)
        vectors = np.ascontiguousarray(vectors.astype("float32"))

        # Normalize so inner product = cosine similarity
        faiss.normalize_L2(vectors)

        self.index.add(vectors)

        metadatas = metadatas or [{} for _ in chunk_texts]
        for text, meta in zip(chunk_texts, metadatas):
            self.chunks.append({"text": text, **meta})

    def search(self, query_vector: np.ndarray, k: int = 5):
        """
        query_vector: (dim,) float32 array, same embedding model as add()
        Returns list of dicts: {"text": ..., "score": ..., **metadata}
        """
        query_vector = query_vector.astype("float32").reshape(1, -1)
        faiss.normalize_L2(query_vector)

        scores, indices = self.index.search(query_vector, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # FAISS pads with -1 if fewer than k results exist
                continue
            result = dict(self.chunks[idx])
            result["score"] = float(score)
            results.append(result)
        return results

    def save(self, path: str):
        """Persist both the FAISS index and the chunk metadata."""
        path = Path(path)
        faiss.write_index(self.index, str(path.with_suffix(".faiss")))
        with open(path.with_suffix(".pkl"), "wb") as f:
            pickle.dump({"chunks": self.chunks, "dim": self.dim}, f)

    @classmethod
    def load(cls, path: str):
        path = Path(path)
        index = faiss.read_index(str(path.with_suffix(".faiss")))
        with open(path.with_suffix(".pkl"), "rb") as f:
            data = pickle.load(f)

        store = cls(dim=data["dim"])
        store.index = index
        store.chunks = data["chunks"]
        return store


if __name__ == "__main__":
    # quick smoke test
    dim = 384
    store = VectorStore(dim)

    fake_vectors = np.random.rand(3, dim).astype("float32")
    fake_texts = ["The cat sat on the mat.", "Paris is the capital of France.", "FAISS is a similarity search library."]

    store.add(fake_vectors, fake_texts)
    results = store.search(fake_vectors[0], k=2)
    for r in results:
        print(r["score"], "-", r["text"])

    store.save("test_store")
    reloaded = VectorStore.load("test_store")
    print("Reloaded chunk count:", len(reloaded.chunks))