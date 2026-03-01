"""
Dense vector retrieval using Qdrant.
Performs approximate nearest neighbor search using embedded queries.
"""

from typing import Any

from app.embeddings.service import embedding_service
from app.vectorstore.qdrant_store import vector_store
from app.config import settings
from app.utils.logging import get_logger

logger = get_logger(__name__)


class DenseRetriever:
    """
    Dense retrieval via semantic vector search.

    How it works:
    1. Encodes the query into a dense vector using SentenceTransformers.
    2. Searches Qdrant's HNSW index for approximate nearest neighbors.
    3. Returns top-K results with cosine similarity scores.

    Strengths: Captures semantic meaning ("car" matches "automobile").
    Weaknesses: Can miss exact keyword matches that BM25 catches.
    """

    async def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        filters: dict[str, str] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Perform dense retrieval for a query.

        Args:
            query: Search query string.
            top_k: Number of results to return.
            filters: Metadata filters (key=field, value=match).

        Returns:
            List of dicts with id, score, content, and metadata.
        """
        k = top_k or settings.dense_top_k

        # Encode the query
        query_vector = await embedding_service.encode_query(query)

        # Search Qdrant
        raw_results = await vector_store.search(
            query_vector=query_vector,
            top_k=k,
            filters=filters,
        )

        # Normalize results
        results = []
        for hit in raw_results:
            results.append({
                "chunk_id": str(hit["id"]),
                "content": hit["payload"].get("content", ""),
                "document_name": hit["payload"].get("filename", "unknown"),
                "dense_score": hit["score"],
                "metadata": {
                    k: str(v) for k, v in hit["payload"].items()
                    if k not in ("content",)
                },
            })

        logger.info(f"Dense retrieval: {len(results)} results for query length {len(query)}")
        return results


# Singleton
dense_retriever = DenseRetriever()
