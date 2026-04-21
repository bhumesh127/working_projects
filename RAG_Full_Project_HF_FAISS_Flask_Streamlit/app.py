
from flask import Flask, request, jsonify
from rag_core import RAGSystem

app = Flask(__name__)
rag = RAGSystem()

@app.route("/query", methods=["POST"])
def query():
    data = request.json
    question = data.get("question")
    answer = rag.query(question)
    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run(debug=True, port=5000)
