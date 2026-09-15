# generate.py
import os

from dotenv import load_dotenv
load_dotenv()  # reads .env and sets the variables into os.environ

from openai import OpenAI

# Defaults to local Ollama; set OPENAI_BASE_URL/OPENAI_API_KEY in .env to switch to a hosted API.
client = OpenAI(
    base_url=os.getenv("OPENAI_BASE_URL", "http://localhost:11434/v1"),
    api_key=os.getenv("OPENAI_API_KEY", "ollama"),
)

DEFAULT_MODEL = os.getenv("GENERATION_MODEL", "llama3.2")

SYSTEM_PROMPT = """You are a question-answering assistant. You must answer \
the user's question using ONLY the information in the provided context below.

Rules:
- Do not use any outside knowledge, even if you are confident about the answer.
- If the context does not contain enough information to answer, respond \
exactly with: "I don't know based on the provided context."
- Do not guess, speculate, or fill gaps with general knowledge.
- When you do answer, briefly indicate which part of the context supports \
your answer (e.g. "According to [source]...").
- Keep answers concise and directly responsive to the question.
"""


def build_user_prompt(question: str, retrieved_chunks: list[dict]) -> str:
    """
    retrieved_chunks: list of dicts from store.search()/retrieve(), e.g.
        {"text": "...", "score": 0.83, "source": "doc.pdf", "page": 3}
    """
    context_blocks = []
    for i, chunk in enumerate(retrieved_chunks, start=1):
        source_info = chunk.get("source", f"chunk {i}")
        context_blocks.append(f"[{source_info}]\n{chunk['text']}")

    context_str = "\n\n---\n\n".join(context_blocks)

    return f"""Context:
{context_str}

Question: {question}

Answer the question using only the context above."""


def generate_answer(question: str, retrieved_chunks: list[dict], model: str = DEFAULT_MODEL) -> str:
    if not retrieved_chunks:
        return "I don't know based on the provided context."

    user_prompt = build_user_prompt(question, retrieved_chunks)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=1024,
        temperature=0,  # deterministic, factual answers — avoid creative drift
    )

    return response.choices[0].message.content


if __name__ == "__main__":
    # quick smoke test with fake retrieved chunks
    fake_chunks = [
        {"text": "FAISS is a library for efficient similarity search of dense vectors, developed by Meta.",
         "source": "faiss_intro.md", "score": 0.91},
        {"text": "It supports both exact (flat) and approximate (IVF, HNSW) indexes.",
         "source": "faiss_intro.md", "score": 0.85},
    ]

    answer = generate_answer("Who developed FAISS?", fake_chunks)
    print(answer)

    # Test the "I don't know" path with irrelevant context
    unrelated_chunks = [
        {"text": "The Eiffel Tower was completed in 1889.", "source": "paris.md", "score": 0.3}
    ]
    answer2 = generate_answer("What programming language is FAISS written in?", unrelated_chunks)
    print(answer2)