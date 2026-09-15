# embed.py — chunk records -> dense vectors (embeddings.npy + row-aligned metadata).
import argparse
import json

import numpy as np
from sentence_transformers import SentenceTransformer

_model_cache = {}


def get_model(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    """Cache loaded models by name — reloading weights from disk is a multi-second cost."""
    if model_name not in _model_cache:
        _model_cache[model_name] = SentenceTransformer(model_name)
    return _model_cache[model_name]


def load_chunks(chunks_path: str) -> list[dict]:
    records = []
    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def embed_chunks(
    records: list[dict],
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 32,
) -> np.ndarray:
    """Encode each chunk's text into a dense vector, normalized so inner product = cosine similarity."""
    model = get_model(model_name)
    texts = [r["text"] for r in records]

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return embeddings.astype("float32")


def main():
    parser = argparse.ArgumentParser(description="Embed chunked text for retrieval (RAG).")
    parser.add_argument("chunks_path", help="Path to chunks.jsonl from ingest.py")
    parser.add_argument("--out", default="embeddings.npy", help="Output .npy file for the vectors")
    parser.add_argument("--model", default="all-MiniLM-L6-v2", help="sentence-transformers model name")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    records = load_chunks(args.chunks_path)
    if not records:
        raise SystemExit(f"No chunks found in {args.chunks_path}")

    embeddings = embed_chunks(records, args.model, args.batch_size)
    np.save(args.out, embeddings)

    meta_path = args.out.rsplit(".", 1)[0] + ".meta.jsonl"
    with open(meta_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(
                {"chunk_id": r["chunk_id"], "source": r["source"], "page": r["page"]},
                ensure_ascii=False,
            ) + "\n")

    print(f"Embedded {len(records)} chunks -> {args.out} (shape {embeddings.shape})")
    print(f"Row-aligned metadata -> {meta_path}")


if __name__ == "__main__":
    main()