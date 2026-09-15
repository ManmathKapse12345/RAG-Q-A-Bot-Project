# RAG Q&A Bot

A retrieval-augmented Q&A bot: upload PDFs, ask questions, get answers grounded only in the uploaded content (with a refusal instead of a guess when the answer isn't there).

## Features

- PDF ingestion and chunking, sized to the embedding model's own token limit
- Semantic search over document chunks with FAISS
- Answers grounded in retrieved context, with source citations and page numbers
- Streamlit chat UI with file upload
- A second implementation of the same pipeline built with LangChain, for comparison
- A minimal agent layer that routes a question to document search, a calculator, or a direct answer

## How it works

1. **Ingest** (`ingest.py`) — extracts text per page from a PDF and splits it into overlapping chunks, using the embedding model's tokenizer so chunks never exceed its max sequence length.
2. **Embed** (`embed.py`) — encodes each chunk into a vector with `sentence-transformers` (`all-MiniLM-L6-v2`).
3. **Store** (`store.py`) — indexes vectors in FAISS alongside their source text and metadata.
4. **Retrieve** (`retrieve.py`) — embeds a question and fetches the top-k most similar chunks.
5. **Generate** (`generate.py`) — sends the retrieved chunks plus the question to an LLM with a system prompt that restricts it to the given context.

`pipeline.py` wires these together end-to-end. `app.py` puts a Streamlit UI on top of it. `langchain_version.py` reimplements the same flow using LangChain. `agent.py` adds a routing layer on top of the pipeline.

## Setup

```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

By default, generation runs against a local [Ollama](https://ollama.com) model:

```bash
ollama pull llama3.2
```

To use a hosted API instead, set these in a `.env` file:

```
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_API_KEY=your-key
GENERATION_MODEL=gpt-4o-mini
```

## Usage

```bash
# CLI, manual pipeline
python pipeline.py

# Web UI
streamlit run app.py

# LangChain version
python langchain_version.py

# Agent (routes between doc search / calculator / direct answer)
python agent.py
```

Drop PDFs in `documents/` before running the CLI scripts, or upload them directly in the Streamlit UI.

## Project structure

```
ingest.py             PDF -> chunked text
embed.py              chunks -> vectors
store.py              FAISS index + metadata
retrieve.py           question -> top-k chunks
generate.py           chunks + question -> grounded answer
pipeline.py           wires ingest/embed/store/retrieve/generate together
app.py                Streamlit UI
langchain_version.py  same pipeline, built with LangChain
agent.py              routes a question to a tool (doc search / calculator / direct answer)
documents/            source PDFs
```

## Stack

Python, FAISS, sentence-transformers (PyTorch), LangChain, Streamlit, Ollama/OpenAI-compatible APIs.
