
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

class RAGSystem:
    def __init__(self):
        self.embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.documents = [
            "RAG combines retrieval and generation.",
            "FAISS enables efficient vector similarity search.",
            "HuggingFace provides transformer-based LLMs.",
            "Streamlit can build ML web apps easily."
        ]
        self._build_index()
        self._load_llm()

    def _build_index(self):
        embeddings = self.embed_model.encode(self.documents)
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(np.array(embeddings))

    def _load_llm(self):
        model_name = "distilgpt2"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        self.generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

    def retrieve(self, query, top_k=2):
        query_vector = self.embed_model.encode([query])
        distances, indices = self.index.search(np.array(query_vector), top_k)
        return [self.documents[i] for i in indices[0]]

    def query(self, question):
        context = " ".join(self.retrieve(question))
        prompt = f"""
Use the context to answer.

Context:
{context}

Question:
{question}

Answer:
"""
        result = self.generator(prompt, max_length=200, num_return_sequences=1)
        return result[0]["generated_text"]
