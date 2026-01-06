# RAG System

A local Retrieval-Augmented Generation system built from scratch for learning purposes.

## Overview

This project implements a complete RAG pipeline that runs entirely on your local machine using:

- **Embedding**: sentence-transformers/all-MiniLM-L6-v2
- **Vector Store**: ChromaDB
- **LLM**: Ollama (llama3.2)

## Architecture

```
Documents → Chunking → Embedding → Vector Store
                                        ↓
Question → Embedding → Retrieval → LLM → Answer
```

## Project Structure

```
rag-project/
├── src/
│   ├── document_loader.py   # Load .txt/.md files
│   ├── chunker.py           # Split text into chunks
│   ├── embedder.py          # Generate embeddings
│   ├── vector_store.py      # ChromaDB wrapper
│   ├── retriever.py         # Similarity search
│   ├── generator.py         # Ollama LLM wrapper
│   └── rag_pipeline.py      # Main orchestrator
├── data/                    # Sample documents
├── app.py                   # Web UI (FastAPI + Tailwind)
├── demo.py                  # CLI demo
└── requirements.txt
```

## Quick Start

### Prerequisites

- Python 3.12+
- [Ollama](https://ollama.ai) installed and running

### Installation

```bash
# Clone the repository
git clone https://github.com/omeredebal/rag-project.git
cd rag-project

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Pull the LLM model
ollama pull llama3.2
```

### Usage

**Web Interface:**
```bash
python app.py
# Open http://localhost:8000
```

**CLI Demo:**
```bash
python demo.py
```

## How It Works

1. **Indexing**: Documents are loaded, split into chunks, converted to embeddings, and stored in ChromaDB
2. **Querying**: User question is embedded, similar chunks are retrieved, and LLM generates an answer using the context

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `chunk_size` | 500 | Characters per chunk |
| `chunk_overlap` | 50 | Overlapping characters |
| `top_k` | 3 | Number of chunks to retrieve |
| `llm_model` | llama3.2 | Ollama model name |

## License

MIT
