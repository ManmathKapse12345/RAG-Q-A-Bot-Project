# eval.py
import json
from pathlib import Path

# swap these two imports depending on which pipeline you're testing
from pipeline import ingest_document, ask  # manual pipeline
# from langchain_pipeline import ingest_document, build_chain  # LangChain pipeline

from store import VectorStore


def load_eval_dataset(path: str = "eval_dataset.json") -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def check_retrieval(item: dict, retrieved_sources: list[str]) -> bool:
    """
    Did the expected source document show up anywhere in the retrieved chunks?
    Skipped for unanswerable questions (there's no correct source to find).
    """
    if not item["answerable"]:
        return None  # not applicable
    return item["source"] in retrieved_sources


def check_answer(item: dict, generated_answer: str) -> bool:
    """
    Keyword-based check, not exact match:
    - For unanswerable questions: does the refusal phrase appear?
    - For answerable questions: does the answer contain enough of the
      expected answer's key content? (very rough word-overlap heuristic —
      good enough to flag obvious failures, not meant to be precise)
    """
    generated_lower = generated_answer.lower()

    if not item["answerable"]:
        return "don't know" in generated_lower or "don't have" in generated_lower

    expected_words = set(item["expected_answer"].lower().split())
    generated_words = set(generated_lower.split())
    overlap = len(expected_words & generated_words) / max(len(expected_words), 1)
    return overlap >= 0.3  # loose threshold — tune based on what you see


def run_eval(dataset: list[dict], store: VectorStore, k: int = 5):
    results = []

    for item in dataset:
        result = ask(item["question"], store, k=k)  # from pipeline.py, Step 6
        retrieved_sources = [c.get("source") for c in result["sources"]]

        retrieval_ok = check_retrieval(item, retrieved_sources)
        answer_ok = check_answer(item, result["answer"])

        results.append({
            "question": item["question"],
            "category": item.get("category", "uncategorized"),
            "expected": item["expected_answer"],
            "actual": result["answer"],
            "retrieval_ok": retrieval_ok,
            "answer_ok": answer_ok,
        })

    return results


def print_report(results: list[dict]):
    print(f"\n{'='*70}\nEVAL REPORT — {len(results)} questions\n{'='*70}\n")

    for r in results:
        status = "✅" if r["answer_ok"] else "❌"
        retrieval_flag = (
            "n/a" if r["retrieval_ok"] is None
            else ("✅" if r["retrieval_ok"] else "❌")
        )
        print(f"{status} [{r['category']}] {r['question']}")
        print(f"   retrieval_ok: {retrieval_flag}")
        print(f"   expected: {r['expected']}")
        print(f"   actual:   {r['actual']}")
        print()

    total = len(results)
    answer_pass = sum(1 for r in results if r["answer_ok"])
    retrieval_checked = [r for r in results if r["retrieval_ok"] is not None]
    retrieval_pass = sum(1 for r in retrieval_checked if r["retrieval_ok"])

    print(f"{'='*70}")
    print(f"Answer accuracy:    {answer_pass}/{total} ({answer_pass/total:.0%})")
    if retrieval_checked:
        print(f"Retrieval accuracy: {retrieval_pass}/{len(retrieval_checked)} "
              f"({retrieval_pass/len(retrieval_checked):.0%})")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    dataset = load_eval_dataset("eval_dataset.json")

    # load your already-ingested store — don't re-ingest for every eval run
    store = VectorStore.load("rag_index")

    results = run_eval(dataset, store, k=5)
    print_report(results)