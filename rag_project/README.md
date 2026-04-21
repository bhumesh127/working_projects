# 🔍 RAG LLM System
### Retrieval-Augmented Generation · HuggingFace + FAISS + FastAPI

A **production-ready, realtime RAG system** that lets you:
- Index any text documents into a **FAISS** vector database
- Ask natural language questions
- Get LLM-generated answers grounded in your documents

---

## 🏗 Architecture

```
┌─────────────┐    ┌─────────────────────┐    ┌──────────────┐
│  FastAPI UI │───▶│   RAG Engine        │───▶│  HuggingFace │
│  (REST API) │    │  ┌───────────────┐  │    │  LLM (Flan-  │
└─────────────┘    │  │ FAISS Index   │  │    │    T5-base)  │
                   │  │ (Vector DB)   │  │    └──────────────┘
                   │  └───────────────┘  │
                   │  ┌───────────────┐  │    ┌──────────────┐
                   │  │ Sentence      │  │───▶│  HuggingFace │
                   │  │ Transformers  │  │    │  Embeddings  │
                   │  │ Embeddings    │  │    │(MiniLM-L6-v2)│
                   │  └───────────────┘  │    └──────────────┘
                   └─────────────────────┘
```

## 📁 Project Structure

```
rag_project/
├── app/
│   ├── __init__.py
│   ├── main.py          # FastAPI routes
│   └── rag_engine.py    # Core RAG logic (HuggingFace + FAISS)
├── static/
│   └── index.html       # Beautiful Web UI
├── data/                # Put your .txt files here
├── requirements.txt
├── run.sh               # One-click startup
└── README.md
```

## 🚀 Quick Start

### 1. Run the server
```bash
chmod +x run.sh
./run.sh
```

### 2. Open in browser
```
http://localhost:8000/
```

### 3. Load demo data or add your own documents, then ask questions!

---

## 🌐 REST API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/`      | Web UI |
| `GET`  | `/health` | Health check |
| `GET`  | `/stats`  | Index stats |
| `POST` | `/add-document` | Index a text document |
| `POST` | `/upload-file` | Upload & index a .txt file |
| `POST` | `/query`  | RAG query → answer |
| `POST` | `/seed-demo-data` | Load 5 AI/ML demo docs |
| `DELETE` | `/reset-index` | Clear all indexed data |
| `GET`  | `/docs`   | Swagger UI |
| `GET`  | `/redoc`  | ReDoc UI |

### Query Example
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is FAISS?", "top_k": 3}'
```

### Add Document Example
```bash
curl -X POST http://localhost:8000/add-document \
  -H "Content-Type: application/json" \
  -d '{"text": "Your document content here...", "metadata": {"source": "manual"}}'
```

---

## 🤖 Models Used

| Role | Model | Size |
|------|-------|------|
| **Embeddings** | `all-MiniLM-L6-v2` | ~90 MB |
| **LLM (Q&A)** | `google/flan-t5-base` | ~250 MB |

Both models are downloaded automatically on first run from HuggingFace Hub.

---

## ⚙️ How RAG Works

1. **Ingestion** — Document text is split into overlapping chunks (300 chars, 50 overlap)
2. **Embedding** — Each chunk is embedded using `sentence-transformers` into a 384-dim vector
3. **Indexing** — Vectors are stored in FAISS `IndexFlatL2` for fast similarity search
4. **Query** — User query is embedded, top-K nearest chunks are retrieved from FAISS
5. **Generation** — Retrieved chunks are injected as context into the LLM prompt → answer

---

## 🔧 Configuration (in `rag_engine.py`)

```python
EMBEDDING_MODEL = "all-MiniLM-L6-v2"   # Change to larger model for better quality
LLM_MODEL       = "google/flan-t5-base" # Change to flan-t5-large, flan-ul2, etc.
CHUNK_SIZE      = 300                   # Characters per chunk
CHUNK_OVERLAP   = 50                    # Overlap between consecutive chunks
```

## 💡 Tips
- For **better answers**: try `google/flan-t5-large` or `mistralai/Mistral-7B-Instruct-v0.2`
- For **GPU support**: install `faiss-gpu` and set device in pipeline
- For **persistence**: save the FAISS index with `faiss.write_index(index, "index.bin")`
