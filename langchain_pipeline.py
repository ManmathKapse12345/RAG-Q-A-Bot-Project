# langchain_pipeline.py
from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

INDEX_DIR = "langchain_index"

SYSTEM_PROMPT = """You are a question-answering assistant. You must answer \
the user's question using ONLY the information in the provided context below.

Rules:
- Do not use any outside knowledge, even if you are confident about the answer.
- If the context does not contain enough information to answer, respond \
exactly with: "I don't know based on the provided context."
- Do not guess, speculate, or fill gaps with general knowledge.
- Keep answers concise and directly responsive to the question.

Context:
{context}"""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{question}"),
])

embeddings = OpenAIEmbeddings()  # equivalent to your embed.py
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)  # equivalent to your generate.py's client


def ingest_document(file_path: str, vectorstore: FAISS = None) -> FAISS:
    """
    Manual equivalent: file_path.read_text() -> chunk_document() -> embed_batch() -> store.add()
    """
    # 1. Load — equivalent to reading the file yourself
    loader = TextLoader(file_path)
    docs = loader.load()  # list[Document], each with .page_content + .metadata (source path auto-added)

    # 2. Split — equivalent to chunk_document()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,      # match whatever size you used manually
        chunk_overlap=50,    # LangChain makes overlap trivial to add — worth doing
    )
    chunks = splitter.split_documents(docs)
    print(f"[ingest] {Path(file_path).name}: {len(chunks)} chunks")

    # 3. Embed + 4. Store — equivalent to embed_batch() + store.add()
    if vectorstore is None:
        if Path(INDEX_DIR).exists():
            vectorstore = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
            print(f"[ingest] loaded existing index")
        else:
            vectorstore = FAISS.from_documents(chunks, embeddings)
            vectorstore.save_local(INDEX_DIR)
            return vectorstore

    vectorstore.add_documents(chunks)  # incremental add, same idea as store.add()
    vectorstore.save_local(INDEX_DIR)
    return vectorstore


def format_docs(docs) -> str:
    """Equivalent to build_user_prompt()'s context-block joining."""
    return "\n\n---\n\n".join(
        f"[{d.metadata.get('source', 'unknown')}]\n{d.page_content}" for d in docs
    )


def build_chain(vectorstore: FAISS, k: int = 5):
    """
    Equivalent to your ask() function: retrieve() + generate_answer()
    chained together, expressed declaratively instead of imperatively.
    """
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})  # equivalent to retrieve.py

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain, retriever


if __name__ == "__main__":
    # --- Ingestion (once per document) ---
    vectorstore = None
    for doc_path in Path("documents").glob("*.txt"):
        vectorstore = ingest_document(str(doc_path), vectorstore=vectorstore)

    # --- Query (every question) ---
    chain, retriever = build_chain(vectorstore, k=3)

    print("\nLangChain RAG pipeline ready. Type a question (or 'quit'):\n")
    while True:
        question = input("> ")
        if question.strip().lower() in ("quit", "exit"):
            break

        answer = chain.invoke(question)
        sources = retriever.invoke(question)  # re-run to inspect what was used

        print(f"\nAnswer: {answer}\n")
        print("Sources:")
        for i, doc in enumerate(sources, start=1):
            print(f"  [{i}] {doc.metadata.get('source', '?')}")
        print()