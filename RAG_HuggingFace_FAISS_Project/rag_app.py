
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# ------------------------------
# 1. Load Embedding Model
# ------------------------------
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# ------------------------------
# 2. Sample Documents (Replace with real-time data source if needed)
# ------------------------------
documents = [
    "Retrieval Augmented Generation combines retrieval and generation models.",
    "Umesh is Bhumesh's friend"
    "FAISS is a library for efficient similarity search.",
    "HuggingFace provides open-source transformer models.",
    "Large Language Models can generate human-like text.",
    "Aadhya is Bhumesh's sister"
]

# ------------------------------
# 3. Create FAISS Index
# ------------------------------
doc_embeddings = embed_model.encode(documents)
dimension = doc_embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)
index.add(np.array(doc_embeddings))

# ------------------------------
# 4. Load LLM from HuggingFace
# ------------------------------
model_name = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

# ------------------------------
# 5. Retrieval Function
# ------------------------------
def retrieve(query, top_k=2):
    query_vector = embed_model.encode([query])
    distances, indices = index.search(np.array(query_vector), top_k)
    return [documents[i] for i in indices[0]]

# ------------------------------
# 6. RAG Pipeline
# ------------------------------
def rag_query(query):
    retrieved_docs = retrieve(query)
    context = " ".join(retrieved_docs)

    prompt = f"""
    Use the following context to answer the question.

    Context:
    {context}

    Question:
    {query}

    Answer:
    """

    result = generator(prompt, max_length=200, num_return_sequences=1)
    return result[0]["generated_text"]

# ------------------------------
# 7. Run Example
# ------------------------------
if __name__ == "__main__":
    while True:
        user_query = input("Enter your question (or type 'exit'): ")
        if user_query.lower() == "exit":
            break
        answer = rag_query(user_query)
        print("\nRAG Answer:\n", answer)
