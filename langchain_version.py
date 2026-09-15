# langchain_version.py — the same RAG pipeline as pipeline.py, built with LangChain.
import os
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_text_splitters import SentenceTransformersTokenTextSplitter

from generate import SYSTEM_PROMPT  # same grounding rules as the manual pipeline

EMBED_MODEL = "all-MiniLM-L6-v2"
INDEX_DIR = "langchain_index"


def load_and_chunk_pdfs(pdf_paths: list[str], chunk_size: int = 200, overlap: int = 40):
    """PDF paths -> LangChain Documents, chunked to <=chunk_size embedding-model tokens each."""
    splitter = SentenceTransformersTokenTextSplitter(
        model_name=EMBED_MODEL,
        tokens_per_chunk=chunk_size,
        chunk_overlap=overlap,
    )

    all_chunks = []
    for pdf_path in pdf_paths:
        pages = PyPDFLoader(pdf_path).load()  # one Document per page
        for page in pages:
            page.page_content = " ".join(page.page_content.split())  # same whitespace fix as ingest.py
        all_chunks.extend(splitter.split_documents(pages))
    return all_chunks


def build_vectorstore(pdf_paths: list[str]) -> FAISS:
    """One-time path: PDFs -> chunks -> embeddings -> FAISS index (loads a saved index if present)."""
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    if Path(INDEX_DIR).exists():
        print(f"[ingest] loaded existing LangChain index from {INDEX_DIR}/")
        return FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)

    chunks = load_and_chunk_pdfs(pdf_paths)
    print(f"[ingest] {len(chunks)} chunks from {len(pdf_paths)} document(s)")

    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(INDEX_DIR)
    print(f"[ingest] saved to {INDEX_DIR}/")
    return vectorstore


def build_qa_chain(vectorstore: FAISS, k: int = 5):
    """Wire retriever -> prompt -> LLM -> string output using LCEL ("|" composition)."""
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    llm = ChatOpenAI(
        base_url=os.getenv("OPENAI_BASE_URL", "http://localhost:11434/v1"),
        api_key=os.getenv("OPENAI_API_KEY", "ollama"),
        model=os.getenv("GENERATION_MODEL", "llama3.2"),
        temperature=0,
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", "Context:\n{context}\n\nQuestion: {question}\n\nAnswer using only the context above."),
        ]
    )

    def format_docs(docs) -> str:
        blocks = []
        for doc in docs:
            label = f"{doc.metadata.get('source', '?')} p.{doc.metadata.get('page_label', '?')}"
            blocks.append(f"[{label}]\n{doc.page_content}")
        return "\n\n---\n\n".join(blocks)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain, retriever


if __name__ == "__main__":
    docs_dir = Path("documents")
    pdf_paths = [str(p) for p in docs_dir.glob("*.pdf")]
    if not pdf_paths:
        raise SystemExit(f"No PDFs found in {docs_dir}/")

    vectorstore = build_vectorstore(pdf_paths)
    chain, retriever = build_qa_chain(vectorstore)

    print("\nLangChain RAG pipeline ready. Type a question (or 'quit'):\n")
    while True:
        question = input("> ")
        if question.strip().lower() in ("quit", "exit"):
            break

        answer = chain.invoke(question)
        print(f"\nAnswer: {answer}\n")

        print("Sources:")
        for i, doc in enumerate(retriever.invoke(question), start=1):
            print(f"  [{i}] {doc.metadata.get('source', '?')} p.{doc.metadata.get('page_label', '?')}")
        print()
