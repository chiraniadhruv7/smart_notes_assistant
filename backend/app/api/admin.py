"""
Admin diagnostics endpoint.
Provides system health, service status, and index statistics.
"""

import os
import time
import psutil

from fastapi import APIRouter

from app.config import settings
from app.models import AdminDiagnostics, ServiceHealth
from app.vectorstore.qdrant_store import vector_store
from app.core.cache import cache_service
from app.embeddings.service import embedding_service
from app.llm.service import llm_service
from app.retrieval.sparse_retriever import sparse_retriever
from app.evaluation.metrics import retrieval_evaluator
from app.utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/api/admin", tags=["admin"])

_startup_time = time.time()


@router.get("/diagnostics", response_model=AdminDiagnostics)
async def diagnostics() -> AdminDiagnostics:
    """
    Full system diagnostics for monitoring and debugging.

    Returns health status for all services, index stats,
    memory usage, and evaluation metrics.
    """
    services: list[ServiceHealth] = []

    # Qdrant health
    qdrant_health = vector_store.health_check()
    collection_info = await vector_store.get_collection_info()
    services.append(ServiceHealth(
        name="qdrant",
        status=qdrant_health.get("status", "unknown"),
        latency_ms=qdrant_health.get("latency_ms", -1),
        details={k: str(v) for k, v in collection_info.items()},
    ))

    # Redis health
    redis_health = await cache_service.health_check()
    services.append(ServiceHealth(
        name="redis",
        status=redis_health.get("status", "unknown"),
        latency_ms=redis_health.get("latency_ms", -1),
        details={k: str(v) for k, v in redis_health.items() if k not in ("status", "latency_ms")},
    ))

    # Embedding service health
    emb_health = await embedding_service.health_check()
    services.append(ServiceHealth(
        name="embeddings",
        status=emb_health.get("status", "unknown"),
        details={k: str(v) for k, v in emb_health.items() if k != "status"},
    ))

    # LLM health
    llm_health = await llm_service.health_check()
    services.append(ServiceHealth(
        name="llm",
        status=llm_health.get("status", "unknown"),
        details={k: str(v) for k, v in llm_health.items() if k != "status"},
    ))

    # BM25 index status
    services.append(ServiceHealth(
        name="bm25_index",
        status="healthy" if sparse_retriever.index_size > 0 else "empty",
        details={"index_size": str(sparse_retriever.index_size)},
    ))

    # System memory
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / 1024 / 1024

    return AdminDiagnostics(
        app_version=settings.app_version,
        uptime_seconds=round(time.time() - _startup_time, 1),
        total_documents=0,  # Could be tracked by ingestion pipeline
        total_chunks=collection_info.get("points_count", 0),
        services=services,
        memory_usage_mb=round(memory_mb, 2),
    )


@router.get("/metrics")
async def evaluation_metrics():
    """Return retrieval quality metrics."""
    return retrieval_evaluator.get_metrics()


@router.post("/metrics/reset")
async def reset_metrics():
    """Reset evaluation metrics."""
    retrieval_evaluator.reset()
    return {"message": "Metrics reset"}


@router.get("/conversations")
async def list_conversations():
    """List active conversation IDs."""
    return {"conversations": llm_service.memory.list_conversations()}


@router.get("/documents")
async def list_documents():
    """List all unique documents in the vector store."""
    payloads = await vector_store.get_all_payloads()
    docs = {}
    for p in payloads:
        doc_id = p.get("document_id", "unknown")
        if doc_id not in docs:
            docs[doc_id] = {
                "document_id": doc_id,
                "filename": p.get("filename", "unknown"),
                "source": p.get("source", ""),
                "chunks": 0,
            }
        docs[doc_id]["chunks"] += 1
    return {"documents": list(docs.values())}


@router.delete("/documents/all")
async def delete_all_documents():
    """Delete all documents from the vector store and rebuild BM25 index."""
    try:
        vector_store._client.delete_collection(settings.qdrant_collection)
        vector_store._ensure_collection()
        await sparse_retriever.build_index()
        return {"message": "All documents deleted", "status": "success"}
    except Exception as e:
        logger.error(f"Failed to delete all documents: {e}")
        return {"message": str(e), "status": "error"}


@router.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    """Delete a specific document by ID."""
    try:
        deleted = await vector_store.delete_by_document_id(document_id)
        await sparse_retriever.build_index()
        return {"message": f"Deleted document {document_id}", "deleted_chunks": deleted}
    except Exception as e:
        logger.error(f"Failed to delete document {document_id}: {e}")
        return {"message": str(e), "status": "error"}
