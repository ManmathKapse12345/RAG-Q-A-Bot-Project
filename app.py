# app.py — Streamlit front end for the RAG pipeline.
from pathlib import Path

import streamlit as st

from pipeline import INDEX_PATH, ask, ingest_document
from store import VectorStore

st.set_page_config(page_title="RAG Q&A Bot", page_icon="📄")
st.title("📄 RAG Q&A Bot")
st.caption("Upload PDFs, then ask questions answered only from their content.")

# st.session_state survives Streamlit's rerun-on-every-interaction model.
if "store" not in st.session_state:
    if Path(INDEX_PATH + ".faiss").exists():
        st.session_state.store = VectorStore.load(INDEX_PATH)
    else:
        st.session_state.store = None

if "history" not in st.session_state:
    st.session_state.history = []  # [{"question", "answer", "sources"}, ...]


def render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)})"):
        for i, src in enumerate(sources, start=1):
            st.markdown(f"**[{i}] {src.get('source', '?')} — p.{src.get('page', '?')}** (score={src['score']:.3f})")
            st.caption(src["text"])


with st.sidebar:
    st.header("Documents")
    uploaded_files = st.file_uploader("Upload PDF(s)", type="pdf", accept_multiple_files=True)
    if uploaded_files:
        docs_dir = Path("documents")
        docs_dir.mkdir(exist_ok=True)
        for uploaded in uploaded_files:
            dest = docs_dir / uploaded.name
            dest.write_bytes(uploaded.getbuffer())
            with st.spinner(f"Ingesting {uploaded.name}..."):
                st.session_state.store = ingest_document(str(dest), store=st.session_state.store)
        st.success("Documents ready.")

    st.divider()
    if st.session_state.store and st.session_state.store.chunks:
        indexed_sources = sorted({c.get("source", "?") for c in st.session_state.store.chunks})
        st.caption(f"Indexed documents ({len(indexed_sources)}):")
        for s in indexed_sources:
            st.write(f"- {s}")
    else:
        st.info("Upload a PDF to get started.")

for turn in st.session_state.history:
    with st.chat_message("user"):
        st.write(turn["question"])
    with st.chat_message("assistant"):
        st.write(turn["answer"])
        render_sources(turn["sources"])

question = st.chat_input("Ask a question about your documents...")
if question:
    if not st.session_state.store or not st.session_state.store.chunks:
        st.warning("Upload a PDF first.")
    else:
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = ask(question, st.session_state.store, k=5)
            st.write(result["answer"])
            render_sources(result["sources"])
        st.session_state.history.append(result)
