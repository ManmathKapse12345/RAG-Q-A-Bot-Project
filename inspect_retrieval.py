# inspect_retrieval.py
from store import VectorStore
from retrieve import retrieve

store = VectorStore.load("rag_index")

questions = [
    "What metrics should be shown when evaluating predictive dialing scenarios?",
    "What is the key objective of the final SmartDialer design question?",
]

for q in questions:
    print(f"\n{'='*70}\nQUESTION: {q}\n{'='*70}")
    chunks = retrieve(q, store, k=5)
    for i, c in enumerate(chunks, start=1):
        print(f"\n--- chunk {i} (score={c['score']:.3f}, page {c.get('page')}) ---")
        print(c['text'])