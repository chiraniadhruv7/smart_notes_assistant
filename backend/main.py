"""
FastAPI application entry point.
Configures lifespan, middleware, exception handlers, and router mounting.
"""

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.utils.logging import setup_logging, get_logger, correlation_id, generate_correlation_id
from app.utils.exceptions import RAGException, rag_exception_handler, generic_exception_handler
from app.core.cache import cache_service
from app.vectorstore.qdrant_store import vector_store
from app.embeddings.service import embedding_service
from app.llm.service import llm_service
from app.retrieval.sparse_retriever import sparse_retriever

from app.api.chat import router as chat_router
from app.api.ingest import router as ingest_router
from app.api.admin import router as admin_router
from app.api.health import router as health_router

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    Handles startup initialization and shutdown cleanup.

    Startup sequence:
    1. Configure structured logging
    2. Connect to Redis cache
    3. Connect to Qdrant vector store
    4. Load embedding model (SentenceTransformers)
    5. Initialize LLM HTTP client
    6. Build BM25 sparse index from existing vectors
    """
    # ── Startup ───────────────────────────────────────
    setup_logging(settings.log_level)
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")

    # Connect Redis
    await cache_service.connect()

    # Connect Qdrant
    vector_store.connect()

    # Load embedding model
    embedding_service.load_model()

    # Initialize LLM client
    await llm_service.initialize()

    # Build BM25 index from existing documents
    index_size = await sparse_retriever.build_index()
    logger.info(f"BM25 index initialized with {index_size} documents")

    logger.info("All services initialized — ready to serve requests")

    yield  # ── Application runs here ──

    # ── Shutdown ──────────────────────────────────────
    logger.info("Shutting down...")
    await llm_service.shutdown()
    await cache_service.disconnect()
    logger.info("Shutdown complete")


# ── Create FastAPI App ────────────────────────────────────

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Production-grade RAG Knowledge Assistant with hybrid retrieval",
    lifespan=lifespan,
)

# ── Middleware ────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def correlation_id_middleware(request: Request, call_next):
    """Attach a correlation ID to every request for tracing."""
    cid = request.headers.get("X-Correlation-ID", generate_correlation_id())
    correlation_id.set(cid)

    start = time.perf_counter()
    response = await call_next(request)
    elapsed = (time.perf_counter() - start) * 1000

    response.headers["X-Correlation-ID"] = cid
    response.headers["X-Response-Time-MS"] = str(round(elapsed, 2))

    logger.info(
        f"{request.method} {request.url.path} → {response.status_code} ({elapsed:.0f}ms)"
    )
    return response


# ── Exception Handlers ───────────────────────────────────

app.add_exception_handler(RAGException, rag_exception_handler)
app.add_exception_handler(Exception, generic_exception_handler)

# ── Mount Routers ────────────────────────────────────────

app.include_router(health_router)
app.include_router(chat_router)
app.include_router(ingest_router)
app.include_router(admin_router)
