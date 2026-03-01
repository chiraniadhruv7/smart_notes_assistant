"""
Qdrant vector store client.
Handles collection management, document upsert, and similarity search
with metadata filtering support.
"""

import time
import uuid
from typing import Any, Sequence

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from app.config import settings
from app.utils.logging import get_logger

logger = get_logger(__name__)


class QdrantVectorStore:
    """
    Manages interactions with the Qdrant vector database.

    Design decisions:
    - Uses gRPC for bulk upserts and REST for queries (gRPC is faster for writes).
    - Collection is created at startup with HNSW index for fast ANN search.
    - Supports payload-based filtering for metadata queries.
    - Returns scored results with full payload for citation tracking.
    """

    def __init__(self) -> None:
        self._client: QdrantClient | None = None
        self._collection_name: str = settings.qdrant_collection

    def connect(self) -> None:
        """Initialize Qdrant client connection."""
        logger.info(f"Connecting to Qdrant at {settings.qdrant_host}:{settings.qdrant_port}")
        self._client = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            api_key=settings.qdrant_api_key,
            timeout=30,
        )
        self._ensure_collection()
        logger.info("Qdrant connected and collection ready")

    def _ensure_collection(self) -> None:
        """Create the collection if it doesn't exist."""
        if not self._client:
            raise RuntimeError("Qdrant client not connected")

        collections = self._client.get_collections().collections
        exists = any(c.name == self._collection_name for c in collections)

        if not exists:
            logger.info(f"Creating collection: {self._collection_name}")
            self._client.create_collection(
                collection_name=self._collection_name,
                vectors_config=qmodels.VectorParams(
                    size=settings.embedding_dim,
                    distance=qmodels.Distance.COSINE,
                ),
                # Optimized HNSW params for quality/speed trade-off
                hnsw_config=qmodels.HnswConfigDiff(
                    m=16,
                    ef_construct=128,
                ),
                # Enable payload indexing for metadata filters
                optimizers_config=qmodels.OptimizersConfigDiff(
                    indexing_threshold=10000,
                ),
            )
            # Create payload indices for common filter fields
            for field in ("document_id", "filename", "file_type", "source"):
                self._client.create_payload_index(
                    collection_name=self._collection_name,
                    field_name=field,
                    field_schema=qmodels.PayloadSchemaType.KEYWORD,
                )
            logger.info("Collection created with payload indices")
        else:
            logger.info(f"Collection '{self._collection_name}' already exists")

    async def upsert(
        self,
        ids: list[str],
        vectors: list[list[float]],
        payloads: list[dict[str, Any]],
    ) -> int:
        """
        Upsert vectors with payloads into Qdrant.
        Returns the number of points upserted.
        """
        if not self._client:
            raise RuntimeError("Qdrant client not connected")

        points = [
            qmodels.PointStruct(
                id=uid,
                vector=vec,
                payload=payload,
            )
            for uid, vec, payload in zip(ids, vectors, payloads)
        ]

        start = time.perf_counter()
        self._client.upsert(
            collection_name=self._collection_name,
            points=points,
            wait=True,
        )
        elapsed = (time.perf_counter() - start) * 1000
        logger.info(f"Upserted {len(points)} points in {elapsed:.0f}ms")
        return len(points)

    async def search(
        self,
        query_vector: list[float],
        top_k: int = 20,
        filters: dict[str, str] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Perform approximate nearest neighbor search.
        Returns list of dicts with id, score, and payload.
        """
        if not self._client:
            raise RuntimeError("Qdrant client not connected")

        # Build filter conditions from metadata
        query_filter = None
        if filters:
            conditions = [
                qmodels.FieldCondition(
                    key=key,
                    match=qmodels.MatchValue(value=value),
                )
                for key, value in filters.items()
            ]
            query_filter = qmodels.Filter(must=conditions)

        start = time.perf_counter()
        results = self._client.search(
            collection_name=self._collection_name,
            query_vector=query_vector,
            limit=top_k,
            query_filter=query_filter,
            with_payload=True,
            score_threshold=0.0,
        )
        elapsed = (time.perf_counter() - start) * 1000

        hits = []
        for point in results:
            hits.append({
                "id": point.id,
                "score": point.score,
                "payload": point.payload or {},
            })

        logger.info(f"Qdrant search returned {len(hits)} hits in {elapsed:.0f}ms")
        return hits

    async def get_collection_info(self) -> dict[str, Any]:
        """Return collection statistics for diagnostics."""
        if not self._client:
            return {"status": "disconnected"}

        try:
            info = self._client.get_collection(self._collection_name)
            return {
                "status": "healthy",
                "points_count": info.points_count,
                "vectors_count": info.vectors_count,
                "segments_count": len(info.segments or []),
                "index_status": str(info.status),
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}

    async def delete_by_document_id(self, document_id: str) -> None:
        """Delete all points for a specific document."""
        if not self._client:
            raise RuntimeError("Qdrant client not connected")

        self._client.delete(
            collection_name=self._collection_name,
            points_selector=qmodels.FilterSelector(
                filter=qmodels.Filter(
                    must=[
                        qmodels.FieldCondition(
                            key="document_id",
                            match=qmodels.MatchValue(value=document_id),
                        )
                    ]
                )
            ),
        )
        logger.info(f"Deleted all points for document_id={document_id}")

    async def get_all_payloads(self) -> list[dict[str, Any]]:
        """
        Scroll through all points and return their payloads.
        Used by BM25 index to build the sparse retrieval corpus.
        """
        if not self._client:
            return []

        payloads: list[dict[str, Any]] = []
        offset = None

        while True:
            results, next_offset = self._client.scroll(
                collection_name=self._collection_name,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for point in results:
                if point.payload:
                    payload = dict(point.payload)
                    payload["_point_id"] = point.id
                    payloads.append(payload)

            if next_offset is None:
                break
            offset = next_offset

        return payloads

    def health_check(self) -> dict[str, Any]:
        """Return Qdrant health status."""
        if not self._client:
            return {"status": "disconnected", "latency_ms": -1}
        try:
            start = time.perf_counter()
            self._client.get_collections()
            latency = (time.perf_counter() - start) * 1000
            return {"status": "healthy", "latency_ms": round(latency, 2)}
        except Exception as e:
            return {"status": "error", "error": str(e)}


# Singleton instance
vector_store = QdrantVectorStore()
