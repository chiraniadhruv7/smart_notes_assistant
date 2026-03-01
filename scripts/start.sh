#!/usr/bin/env bash
# ═══════════════════════════════════════════════════
#  RAG Knowledge Assistant — Startup Script
# ═══════════════════════════════════════════════════

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

echo "╔══════════════════════════════════════════════╗"
echo "║   RAG Knowledge Assistant — Starting...      ║"
echo "╚══════════════════════════════════════════════╝"

# ── Step 1: Check prerequisites ──────────────────
echo ""
echo "🔍 Checking prerequisites..."

if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

if ! command -v docker compose &> /dev/null; then
    echo "❌ Docker Compose is not available. Please install Docker Compose."
    exit 1
fi

echo "✅ Docker and Docker Compose found"

# ── Step 2: Set up environment ───────────────────
if [ ! -f "$ROOT_DIR/.env" ]; then
    echo "📋 Creating .env from .env.example..."
    cp "$ROOT_DIR/.env.example" "$ROOT_DIR/.env"
    echo "✅ .env created — edit it to configure your settings"
fi

# ── Step 3: Create data directories ──────────────
echo "📁 Creating data directories..."
mkdir -p "$ROOT_DIR/data/uploads"
mkdir -p "$ROOT_DIR/data/knowledge"

# ── Step 4: Start services ───────────────────────
echo ""
echo "🚀 Starting services with Docker Compose..."
cd "$ROOT_DIR"
docker compose up -d --build

# ── Step 5: Wait for services ────────────────────
echo ""
echo "⏳ Waiting for services to be healthy..."

MAX_WAIT=120
ELAPSED=0

while [ $ELAPSED -lt $MAX_WAIT ]; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "✅ API is healthy"
        break
    fi
    sleep 5
    ELAPSED=$((ELAPSED + 5))
    echo "   Waiting... ($ELAPSED/${MAX_WAIT}s)"
done

if [ $ELAPSED -ge $MAX_WAIT ]; then
    echo "⚠️  API did not become healthy within ${MAX_WAIT}s"
    echo "   Check logs: docker compose logs api"
fi

# ── Step 6: Pull Ollama model ────────────────────
echo ""
echo "📦 Pulling Ollama model..."
OLLAMA_MODEL="${OLLAMA_MODEL:-llama3.2}"
docker exec rag-ollama ollama pull "$OLLAMA_MODEL" 2>/dev/null || \
    echo "⚠️  Could not pull model. Pull manually: docker exec rag-ollama ollama pull $OLLAMA_MODEL"

# ── Done ─────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║   ✅ RAG Knowledge Assistant is running!      ║"
echo "║                                              ║"
echo "║   Frontend:  http://localhost:3000            ║"
echo "║   API:       http://localhost:8000            ║"
echo "║   API Docs:  http://localhost:8000/docs       ║"
echo "║   Qdrant UI: http://localhost:6333/dashboard  ║"
echo "╚══════════════════════════════════════════════╝"
