# ingest.py — PDF -> overlapping text chunks.
import argparse
import json
import re
from pathlib import Path

from pypdf import PdfReader

from embed import get_model


def extract_text_by_page(pdf_path: str) -> list[dict]:
    """Return [{'page': int, 'text': str}, ...] for each non-empty page in the PDF."""
    reader = PdfReader(pdf_path)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        # Some PDFs get a newline from pypdf after nearly every word, not
        # just at real paragraph breaks — collapse all whitespace to spaces.
        text = re.sub(r"\s+", " ", text)
        text = text.strip()
        if text:
            pages.append({"page": i + 1, "text": text})
    return pages


def chunk_text(text: str, tokenizer, chunk_size: int = 200, overlap: int = 40) -> list[str]:
    """
    Split `text` into overlapping chunks of ~chunk_size tokens, counted with
    the embedding model's own tokenizer (a proxy tokenizer like tiktoken
    gives an unreliable token count and risks exceeding the model's real
    max sequence length). Chunks are sliced from the original string via
    the tokenizer's offset mapping rather than tokenizer.decode(), so
    casing/punctuation survive (decode is lossy for WordPiece models).
    """
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    # Tokenize the whole page to get offsets, then window it down below —
    # our windowing enforces the length limit, so raise the tokenizer's own
    # max_length to avoid a spurious "sequence too long" warning here.
    old_max_length = tokenizer.model_max_length
    tokenizer.model_max_length = int(1e9)
    try:
        encoding = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    finally:
        tokenizer.model_max_length = old_max_length
    offsets = encoding["offset_mapping"]
    if not offsets:
        return []

    chunks = []
    stride = chunk_size - overlap
    start = 0
    n = len(offsets)
    while start < n:
        end = min(start + chunk_size, n)
        char_start = offsets[start][0]
        char_end = offsets[end - 1][1]
        chunks.append(text[char_start:char_end])
        if end == n:
            break
        start += stride
    return chunks


def ingest_pdf(
    pdf_path: str,
    chunk_size: int = 200,
    overlap: int = 40,
    model_name: str = "all-MiniLM-L6-v2",
) -> list[dict]:
    """PDF -> list of chunk records ready to embed. `model_name` must match embed.py's model."""
    tokenizer = get_model(model_name).tokenizer
    pages = extract_text_by_page(pdf_path)

    records = []
    chunk_id = 0
    for page in pages:
        for chunk in chunk_text(page["text"], tokenizer, chunk_size, overlap):
            n_tokens = len(tokenizer.encode(chunk, add_special_tokens=False))
            records.append(
                {
                    "chunk_id": chunk_id,
                    "source": Path(pdf_path).name,
                    "page": page["page"],
                    "text": chunk,
                    "n_tokens": n_tokens,
                }
            )
            chunk_id += 1
    return records


def main():
    parser = argparse.ArgumentParser(description="Chunk a PDF for retrieval (RAG ingestion).")
    parser.add_argument("pdf_path", help="Path to the input PDF")
    parser.add_argument("--out", default="chunks.jsonl", help="Output JSONL file")
    parser.add_argument("--chunk-size", type=int, default=200, help="Tokens per chunk (in the embedding model's own tokens)")
    parser.add_argument("--overlap", type=int, default=40, help="Overlapping tokens between chunks")
    parser.add_argument("--model", default="all-MiniLM-L6-v2", help="Embedding model whose tokenizer sets chunk boundaries")
    args = parser.parse_args()

    records = ingest_pdf(args.pdf_path, args.chunk_size, args.overlap, args.model)

    with open(args.out, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Extracted {len(records)} chunks from '{args.pdf_path}' -> '{args.out}'")


if __name__ == "__main__":
    main()