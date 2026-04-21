"""
RAG (Retrieval-Augmented Generation) API
HuggingFace + FAISS + FastAPI
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import os

from app.rag_engine import RAGEngine

app = FastAPI(
    title="BHUMESH RAG LLM API",
    description="Realtime RAG with HuggingFace Embeddings + FAISS Vector DB + LLM",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize RAG Engine (singleton)
rag_engine = RAGEngine()


# ─── Pydantic Schemas ─────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str
    top_k: int = 3
    max_new_tokens: int = 256

class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: List[dict]
    retrieved_chunks: List[str]

class AddDocumentRequest(BaseModel):
    text: str
    metadata: Optional[dict] = {}

class AddDocumentResponse(BaseModel):
    message: str
    doc_id: int
    chunks_added: int

class IndexStatsResponse(BaseModel):
    total_documents: int
    total_chunks: int
    embedding_model: str
    llm_model: str
    index_ready: bool


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the UI"""
    with open("static/index.html", "r",encoding="utf-8") as f:
        return f.read()


@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "RAG-LLM-API"}


@app.get("/stats", response_model=IndexStatsResponse)
async def get_stats():
    """Get FAISS index and model statistics"""
    stats = rag_engine.get_stats()
    return stats


@app.post("/add-document", response_model=AddDocumentResponse)
async def add_document(request: AddDocumentRequest):
    """Add a text document to the FAISS vector store"""
    try:
        result = rag_engine.add_document(request.text, request.metadata)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload-file")
async def upload_file(file: UploadFile = File(...)):
    """Upload a .txt file and index it into FAISS"""
    if not file.filename.endswith(".txt"):
        raise HTTPException(status_code=400, detail="Only .txt files are supported")
    try:
        content = await file.read()
        text = content.decode("utf-8")
        result = rag_engine.add_document(text, {"filename": file.filename})
        return {
            "filename": file.filename,
            "message": "File indexed successfully",
            **result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Perform RAG: retrieve relevant chunks from FAISS, then generate an answer with LLM.
    """
    if not rag_engine.is_ready():
        raise HTTPException(
            status_code=400,
            detail="No documents indexed yet. Please add documents first."
        )
    try:
        result = rag_engine.query(
            query=request.query,
            top_k=request.top_k,
            max_new_tokens=request.max_new_tokens
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/reset-index")
async def reset_index():
    """Clear all documents from FAISS index"""
    rag_engine.reset()
    return {"message": "FAISS index cleared successfully"}


@app.post("/seed-demo-data")
async def seed_demo_data():
    """Load sample documents to demo the system"""
    sample_docs = [
        {
            "text": """Artificial Intelligence (AI) is intelligence demonstrated by machines, as opposed to the natural intelligence displayed by animals including humans. 
            AI research has been defined as the field of study of intelligent agents, which refers to any system that perceives its environment and takes actions that maximize its chance of achieving its goals.
            The term 'artificial intelligence' had previously been used to describe machines that mimic and display human cognitive skills associated with the human mind, such as learning and problem-solving.""",
            "metadata": {"topic": "AI Overview", "source": "demo"}
        },
        {
            "text": """Machine Learning (ML) is a subset of artificial intelligence (AI) that provides systems the ability to automatically learn and improve from experience without being explicitly programmed. 
            Machine learning focuses on the development of computer programs that can access data and use it to learn for themselves.
            The process begins with observations or data, such as examples, direct experience, or instruction, so that computers can learn to recognize patterns in data.""",
            "metadata": {"topic": "Machine Learning", "source": "demo"}
        },
        {
            "text": """FAISS (Facebook AI Similarity Search) is a library for efficient similarity search and clustering of dense vectors. 
            It contains algorithms that search in sets of vectors of any size, up to ones that possibly do not fit in RAM.
            FAISS is written in C++ with complete wrappers for Python. Some of the most useful algorithms are implemented on the GPU.""",
            "metadata": {"topic": "FAISS", "source": "demo"}
        },
        {
            "text": """Large Language Models (LLMs) are advanced AI systems trained on vast amounts of text data. They can generate human-like text, answer questions, summarize documents, translate languages, and perform many other natural language tasks.
            Examples include GPT-4, Claude, LLaMA, and Falcon. These models use transformer architecture with billions of parameters.
            RAG (Retrieval-Augmented Generation) enhances LLMs by retrieving relevant context from a knowledge base before generating responses.""",
            "metadata": {"topic": "LLMs and RAG", "source": "demo"}
        },
        {
            "text": """HuggingFace is an AI company and open-source platform that has become the hub for machine learning models, datasets, and demos.
            The HuggingFace Transformers library provides thousands of pretrained models to perform tasks on different modalities such as text, vision, and audio.
            Their sentence-transformers library specializes in producing semantically meaningful embeddings for sentences and paragraphs, widely used in RAG systems.""",
            "metadata": {"topic": "HuggingFace", "source": "demo"}
        },
        {
            "text": """Bhumesh plays cricket and batminton games.He won the second price in chess.He likes to read books and listen to music.""",
            "metadata": {"topic": "Bhumesh's Hobbies", "source": "demo"}
        },
        {
            "text": """Mounika make mirchi bajji very tasty and she is very innocent.She prepare Biryani very well
            Her husband Bhumesh will give 10000 every month.She is having some back pain health problem.Her home town is Mancherial.""",
            "metadata": {"topic": "Mounika's Hobbies", "source": "demo"}
        },
    ]
    results = []
    for doc in sample_docs:
        r = rag_engine.add_document(doc["text"], doc["metadata"])
        results.append(r)
    total_chunks = sum(r["chunks_added"] for r in results)
    return {
        "message": f"Seeded {len(sample_docs)} demo documents",
        "total_chunks_indexed": total_chunks
    }


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
