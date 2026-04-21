#!/usr/bin/env bash
# ─── RAG LLM System — Startup Script ─────────────────────────────────────────
set -e

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║      RAG LLM System · HuggingFace + FAISS + FastAPI  ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""

# 1. Check Python
python3 --version || { echo "Python 3 is required"; exit 1; }

# 2. Create virtual environment if needed
if [ ! -d ".venv" ]; then
  echo "[Setup] Creating virtual environment..."
  python3 -m venv .venv
fi

source .venv/bin/activate

# 3. Install dependencies
echo "[Setup] Installing dependencies (first run may take a few minutes)..."
pip install -r requirements.txt -q

# 4. Launch
echo ""
echo "[Server] Starting RAG API on http://localhost:8000"
echo "[Server] API Docs: http://localhost:8000/docs"
echo "[Server] UI:       http://localhost:8000/"
echo ""

python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
