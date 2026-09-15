# agent.py — routes a question to one tool (doc search, calculator, or a direct answer).
import re
from pathlib import Path

from generate import client, DEFAULT_MODEL
from pipeline import ask as rag_ask, ingest_document
from store import VectorStore

ROUTER_PROMPT = """You choose which ONE tool should handle the user's question.

Tools:
- search_docs: the question is about content that could be in uploaded documents
- calculator: the question is purely a math/arithmetic calculation
- direct_answer: general-knowledge question, unrelated to any document, not a calculation

Respond with ONLY the tool name, nothing else: search_docs, calculator, or direct_answer."""

VALID_TOOLS = ("search_docs", "calculator", "direct_answer")

# "number operator number" so the match starts at real digits, not stray
# spaces/punctuation earlier in the sentence. Only digits/operators/dots
# reach eval() — no letters reachable means no arbitrary code execution.
SAFE_EXPRESSION = re.compile(r"-?\d+(?:\.\d+)?(?:\s*[-+*/]\s*-?\d+(?:\.\d+)?)+")


def route(question: str) -> str:
    response = client.chat.completions.create(
        model=DEFAULT_MODEL,
        messages=[
            {"role": "system", "content": ROUTER_PROMPT},
            {"role": "user", "content": question},
        ],
        max_tokens=10,
        temperature=0,
    )
    choice = (response.choices[0].message.content or "").strip().lower()
    for tool in VALID_TOOLS:
        if tool in choice:
            return tool
    return "search_docs"  # safest default: grounded refusal beats a guess


def calculate(question: str) -> str:
    match = SAFE_EXPRESSION.search(question)
    if not match or not match.group(0).strip():
        return "I couldn't find a calculation to perform."

    expr = match.group(0).strip()
    try:
        result = eval(expr, {"__builtins__": {}}, {})
    except Exception:
        return "I couldn't safely evaluate that calculation."
    return f"{expr} = {result}"


def direct_answer(question: str) -> str:
    response = client.chat.completions.create(
        model=DEFAULT_MODEL,
        messages=[{"role": "user", "content": question}],
        max_tokens=512,
        temperature=0,
    )
    return response.choices[0].message.content or ""


def agent_ask(question: str, store: VectorStore) -> dict:
    tool = route(question)

    if tool == "search_docs":
        result = rag_ask(question, store)
    elif tool == "calculator":
        result = {"question": question, "answer": calculate(question), "sources": []}
    else:
        result = {"question": question, "answer": direct_answer(question), "sources": []}

    result["tool"] = tool
    return result


if __name__ == "__main__":
    store = None
    for pdf_path in Path("documents").glob("*.pdf"):
        store = ingest_document(str(pdf_path), store=store)

    print("\nAgent ready. Type a question (or 'quit'):\n")
    while True:
        question = input("> ")
        if question.strip().lower() in ("quit", "exit"):
            break

        result = agent_ask(question, store)
        print(f"\n[tool: {result['tool']}] Answer: {result['answer']}\n")
        if result["sources"]:
            print("Sources:")
            for i, src in enumerate(result["sources"], start=1):
                print(f"  [{i}] {src.get('source', '?')} p.{src.get('page', '?')} (score={src['score']:.3f})")
            print()
