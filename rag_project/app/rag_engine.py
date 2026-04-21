"""
RAG Engine
- HuggingFace Sentence Transformers for embeddings
- FAISS for vector similarity search
- HuggingFace Pipeline for text generation (LLM)
"""

import os
import re
import json
import numpy as np
from typing import List, Optional, Dict, Any
from datetime import datetime

# ─── Try importing heavy dependencies with friendly errors ─────────────────────
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False



# Current — fast but basic
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Better options:
# EMBEDDING_MODEL = "all-mpnet-base-v2"          # best quality, still fast
#EMBEDDING_MODEL = "multi-qa-mpnet-base-dot-v1" # tuned specifically for Q&A
# EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"     # state-of-the-art, small
# EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"     # state-of-the-art, large
# Current — small, weak reasoning
LLM_MODEL       = "google/flan-t5-base"

# Better options (drop-in replacements, same code):
#LLM_MODEL = "google/flan-t5-large"       # 2x better, still CPU-friendly
# LLM_MODEL = "google/flan-t5-xl"          # much better, needs more RAM (~6GB)
# LLM_MODEL = "google/flan-ul2"            # best Flan, needs GPU
# LLM_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"  # best quality, needs GPU

# Current — too small, loses context
# CHUNK_SIZE    = 300
# CHUNK_OVERLAP = 50

# Better for general documents
CHUNK_SIZE    = 500   # larger = more context per chunk
CHUNK_OVERLAP = 100   # larger overlap = fewer missed boundaries

# # For short factual data (like your family facts)
# CHUNK_SIZE    = 150   # smaller = more precise retrieval
# CHUNK_OVERLAP = 30


class RAGEngine:
    def __init__(self):
        self.documents: List[Dict[str, Any]] = []
        self.chunks: List[str] = []
        self.chunk_metadata: List[Dict] = []
        self.embeddings: Optional[np.ndarray] = None
        self.index: Optional[Any] = None
        self.embedding_model = None
        self.llm_pipeline = None
        self._models_loaded = False
        self._load_models()

    # ─── Model Loading ────────────────────────────────────────────────────────

    def _load_models(self):
        """Load embedding model and LLM"""
        print(f"[RAG] Loading embedding model: {EMBEDDING_MODEL}")
        try:
            if SENTENCE_TRANSFORMERS_AVAILABLE:
                self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
                print(f"[RAG] ✅ Embedding model loaded")
            else:
                print("[RAG] ⚠️  sentence-transformers not installed — using mock embeddings")
        except Exception as e:
            print(f"[RAG] ⚠️  Embedding model load error: {e}")

        print(f"[RAG] Loading LLM: {LLM_MODEL}")
        try:
            if TRANSFORMERS_AVAILABLE:
                # In _load_models(), replace the pipeline call:
                self.llm_pipeline = pipeline(
                "text2text-generation",
                model=LLM_MODEL,
                max_new_tokens=256,      # increase for longer answers
                temperature=0.3,         # lower = more factual (0.0-1.0)
                do_sample=True,          # False = greedy/deterministic
                repetition_penalty=1.2,  # prevents repeating itself
                no_repeat_ngram_size=3,  # prevents 3-word phrase repeats
)
                # self.llm_pipeline = pipeline(
                #     "text2text-generation",
                #     model=LLM_MODEL,
                #     max_new_tokens=256,
                # )
                print(f"[RAG] ✅ LLM loaded")
            else:
                print("[RAG] ⚠️  transformers not installed — using mock LLM")
        except Exception as e:
            print(f"[RAG] ⚠️  LLM load error: {e}")

        self._models_loaded = True

    # ─── Text Chunking ────────────────────────────────────────────────────────

    def _chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks"""
        text = re.sub(r'\s+', ' ', text).strip()
        chunks = []
        start = 0
        while start < len(text):
            end = start + CHUNK_SIZE
            chunk = text[start:end]
            if chunk.strip():
                chunks.append(chunk.strip())
            start += CHUNK_SIZE - CHUNK_OVERLAP
        return chunks

    # ─── Embedding ────────────────────────────────────────────────────────────

    def _embed(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts"""
        if self.embedding_model:
            embs = self.embedding_model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
            return embs.astype("float32")
        else:
            dim = 384
            return np.random.rand(len(texts), dim).astype("float32")

    # ─── FAISS Index Management ───────────────────────────────────────────────

    def _build_index(self):
        """Build or rebuild the FAISS flat L2 index"""
        if len(self.chunks) == 0:
            return
        self.embeddings = self._embed(self.chunks)
        dim = self.embeddings.shape[1]
        if FAISS_AVAILABLE:
            self.index = faiss.IndexFlatL2(dim)
            self.index.add(self.embeddings)
        else:
            self.index = "brute_force"
        print(f"[RAG] FAISS index built with {len(self.chunks)} chunks")

    def _brute_force_search(self, query_emb: np.ndarray, top_k: int):
        """Fallback similarity search without FAISS"""
        if self.embeddings is None:
            return [], []
        norms_db = np.linalg.norm(self.embeddings, axis=1, keepdims=True) + 1e-9
        norms_q  = np.linalg.norm(query_emb, axis=1, keepdims=True) + 1e-9
        db_norm  = self.embeddings / norms_db
        q_norm   = query_emb / norms_q
        scores   = (q_norm @ db_norm.T)[0]
        indices  = np.argsort(-scores)[:top_k]
        distances = -scores[indices]
        return distances.tolist(), indices.tolist()

    # ─── Public API ───────────────────────────────────────────────────────────

    def add_document(self, text: str, metadata: Dict = {}) -> Dict:
        """Chunk, embed, and index a document"""
        doc_id = len(self.documents)
        chunks = self._chunk_text(text)
        self.documents.append({"id": doc_id, "text": text, "metadata": metadata})

        for i, chunk in enumerate(chunks):
            self.chunks.append(chunk)
            self.chunk_metadata.append({
                "doc_id": doc_id,
                "chunk_index": i,
                "metadata": metadata,
                "timestamp": datetime.utcnow().isoformat()
            })

        self._build_index()

        return {
            "message": "Document indexed successfully",
            "doc_id": doc_id,
            "chunks_added": len(chunks)
        }

    def query(self, query: str, top_k: int = 4, max_new_tokens: int = 256) -> Dict:
        """Retrieve relevant chunks and generate an answer"""
        query_emb = self._embed([query])

        if FAISS_AVAILABLE and isinstance(self.index, faiss.Index):
            distances, indices = self.index.search(query_emb, min(top_k, len(self.chunks)))
            distances = distances[0].tolist()
            indices   = indices[0].tolist()
        else:
            distances, indices = self._brute_force_search(query_emb, min(top_k, len(self.chunks)))

        retrieved_chunks = []
        sources = []
        for rank, (dist, idx) in enumerate(zip(distances, indices)):
            if idx < 0 or idx >= len(self.chunks):
                continue
            chunk = self.chunks[idx]
            meta  = self.chunk_metadata[idx]
            retrieved_chunks.append(chunk)
            sources.append({
                "rank": rank + 1,
                "doc_id": meta["doc_id"],
                "chunk_index": meta["chunk_index"],
                "distance": round(float(dist), 4),
                "metadata": meta["metadata"],
                "preview": chunk[:120] + ("..." if len(chunk) > 120 else "")
            })

        context = "\n\n".join(f"[Source {i+1}]: {c}" for i, c in enumerate(retrieved_chunks))
        # prompt = (
        #     f"Answer the following question based on the provided context.\n\n"
        #     f"Context:\n{context}\n\n"
        #     f"Question: {query}\n\n"
        #     f"Answer:"
        # )
        
        prompt = (
        f"You are a helpful assistant. Use ONLY the context below to answer.\n"
        f"If the answer is not in the context, say 'I don't know'.\n"
        f"Be concise and direct.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        f"Answer:"
    )

        if self.llm_pipeline:
            try:
                output = self.llm_pipeline(prompt, max_new_tokens=max_new_tokens)
                answer = output[0]["generated_text"].strip()
            except Exception as e:
                answer = f"[LLM Error: {e}] Context retrieved: {retrieved_chunks[0][:200] if retrieved_chunks else 'None'}"
        else:
            answer = (
                f"Based on the retrieved context, here is a summary relevant to your query '{query}': "
                + (retrieved_chunks[0][:300] if retrieved_chunks else "No relevant content found.")
            )

        return {
            "query": query,
            "answer": answer,
            "sources": sources,
            "retrieved_chunks": retrieved_chunks
        }

    def get_stats(self) -> Dict:
        """Return engine statistics"""
        index_ready = (self.index is not None and len(self.chunks) > 0)
        return {
            "total_documents": len(self.documents),
            "total_chunks": len(self.chunks),
            "embedding_model": EMBEDDING_MODEL,
            "llm_model": LLM_MODEL,
            "index_ready": index_ready
        }

    def is_ready(self) -> bool:
        return len(self.chunks) > 0

    def reset(self):
        """Clear all data"""
        self.documents.clear()
        self.chunks.clear()
        self.chunk_metadata.clear()
        self.embeddings = None
        self.index = None
        print("[RAG] Index reset")


# ─── FIX: __main__ block is OUTSIDE the class (zero indentation) ─────────────
if __name__ == "__main__":
    engine = RAGEngine()

    # Add a sample document so queries work immediately
    # engine.add_document(
    #     "Artificial Intelligence (AI) is the simulation of human intelligence by machines. "
    #     "Machine Learning is a subset of AI that enables systems to learn from data. "
    #     "Bhumesh son name is Rishan and Daughter name is Aadhya"
    #     "Aadhya and Rishan are siblings. They are children of Bhumesh. This is a family fact.",
        
    #     metadata={"source": "demo"}
    # )

    # ✅ Better — separate documents per topic
    engine.add_document(
        "Bhumesh's son name is Rishan. Bhumesh's daughter name is Aadhya. "
        "Aadhya and Rishan are siblings and children of Bhumesh.Mounika is Bhumesh's wife. Her birthday is on 15th August 1990. She is a house wife.always she queralles with Bhumesh.",
        metadata={"source": "family", "topic": "Bhumesh family"}
    )

    engine.add_document(
        "Artificial Intelligence (AI) is the simulation of human intelligence by machines. "
        "Machine Learning is a subset of AI that enables systems to learn from data.",
        metadata={"source": "ai_facts", "topic": "AI"}
    )

    print("\n✅ RAG Engine ready. Type your question or 'exit' to quit.\n")

    while True:
        user_query = input("Enter your question (or type 'exit'): ").strip()
        if user_query.lower() == "exit":
            break
        if not user_query:
            continue
        result = engine.query(user_query, top_k=5, max_new_tokens=512)
        print("\n📌 Answer:", result["answer"])
        print(f"   (based on {len(result['sources'])} retrieved source(s))\n")
