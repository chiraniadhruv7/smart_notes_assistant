"""
BM25 sparse retrieval.
Maintains an in-memory inverted index built from Qdrant payloads.
Provides keyword-based retrieval complementary to dense search.
"""

import math
from typing import Any

from rank_bm25 import BM25Okapi

from app.vectorstore.qdrant_store import vector_store
from app.config import settings
from app.utils.logging import get_logger

logger = get_logger(__name__)


class SparseRetriever:
    """
    BM25-based sparse retrieval for keyword matching.

    How it works:
    1. On initialization, scrolls all documents from Qdrant and builds
       a BM25 index from their text content.
    2. For each query, tokenizes and scores documents using BM25Okapi.
    3. Returns top-K results ranked by BM25 score.

    Strengths: Exact keyword matching, handles rare terms well.
    Weaknesses: No semantic understanding ("car" won't match "automobile").

    The BM25 index is rebuilt on demand via rebuild_index() — called
    after ingestion to stay in sync with Qdrant.
    """

    def __init__(self) -> None:
        self._index: BM25Okapi | None = None
        self._corpus: list[dict[str, Any]] = []
        self._tokenized_corpus: list[list[str]] = []

    async def build_index(self) -> int:
        """
        Build the BM25 index from all documents in Qdrant.

        Returns:
            Number of documents indexed.
        """
        logger.info("Building BM25 index from Qdrant payloads...")
        payloads = await vector_store.get_all_payloads()

        if not payloads:
            logger.warning("No documents found for BM25 index")
            self._index = None
            self._corpus = []
            return 0

        self._corpus = payloads
        self._tokenized_corpus = [
            self._tokenize(p.get("content", ""))
            for p in payloads
        ]
        self._index = BM25Okapi(self._tokenized_corpus)

        logger.info(f"BM25 index built with {len(payloads)} documents")
        return len(payloads)

    async def rebuild_index(self) -> int:
        """Rebuild the index (alias for build_index, for clarity)."""
        return await self.build_index()

    async def retrieve(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Perform BM25 retrieval for a query.

        Args:
            query: Search query string.
            top_k: Number of results to return.

        Returns:
            List of dicts with chunk_id, content, score, and metadata.
        """
        if not self._index or not self._corpus:
            logger.warning("BM25 index not built, returning empty results")
            return []

        k = top_k or settings.sparse_top_k
        tokenized_query = self._tokenize(query)
        scores = self._index.get_scores(tokenized_query)

        # Get top-K indices sorted by score
        scored_indices = sorted(
            enumerate(scores),
            key=lambda x: x[1],
            reverse=True,
        )[:k]

        results = []
        for idx, score in scored_indices:
            if score <= 0:
                continue
            doc = self._corpus[idx]
            results.append({
                "chunk_id": str(doc.get("_point_id", f"bm25_{idx}")),
                "content": doc.get("content", ""),
                "document_name": doc.get("filename", "unknown"),
                "sparse_score": float(score),
                "metadata": {
                    k: str(v) for k, v in doc.items()
                    if k not in ("content", "_point_id")
                },
            })

        logger.info(f"BM25 retrieval: {len(results)} results for query")
        return results

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Simple whitespace + lowercase tokenization."""
        return text.lower().split()

    @property
    def index_size(self) -> int:
        """Number of documents in the BM25 index."""
        return len(self._corpus)


# Singleton
sparse_retriever = SparseRetriever()
