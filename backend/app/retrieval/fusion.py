"""
Score fusion for combining dense and sparse retrieval results.
Implements Reciprocal Rank Fusion (RRF) for robust score merging.
"""

from typing import Any

from app.config import settings
from app.utils.logging import get_logger

logger = get_logger(__name__)


class ScoreFusion:
    """
    Combines results from dense (semantic) and sparse (BM25) retrievers.

    Uses Reciprocal Rank Fusion (RRF):
        score(d) = Σ 1 / (k + rank_i(d))

    where k is a constant (default 60) and rank_i(d) is the rank of
    document d in result list i.

    Why RRF over simple weighted averaging:
    - RRF is rank-based, so it's invariant to score scale differences
      between dense cosine similarity (0-1) and BM25 scores (unbounded).
    - Proven effective in IR literature for combining heterogeneous signals.
    - The k constant controls how much top ranks are emphasized.

    We also support weighted RRF where each retriever gets a weight
    to control the dense vs. sparse balance.
    """

    def __init__(
        self,
        dense_weight: float | None = None,
        sparse_weight: float | None = None,
        rrf_k: int | None = None,
    ):
        self.dense_weight = dense_weight or settings.dense_weight
        self.sparse_weight = sparse_weight or settings.sparse_weight
        self.rrf_k = rrf_k or settings.rrf_k

    def fuse(
        self,
        dense_results: list[dict[str, Any]],
        sparse_results: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Fuse dense and sparse results using weighted RRF.

        Args:
            dense_results: Results from dense retriever (must have 'chunk_id').
            sparse_results: Results from sparse retriever (must have 'chunk_id').

        Returns:
            Merged and re-scored results sorted by fused score.
        """
        # Build a map of chunk_id -> merged result
        merged: dict[str, dict[str, Any]] = {}

        # Process dense results (already ranked by dense_score)
        for rank, result in enumerate(dense_results):
            cid = result["chunk_id"]
            rrf_score = self.dense_weight / (self.rrf_k + rank + 1)
            if cid not in merged:
                merged[cid] = {
                    "chunk_id": cid,
                    "content": result["content"],
                    "document_name": result.get("document_name", "unknown"),
                    "dense_score": result.get("dense_score", 0.0),
                    "sparse_score": 0.0,
                    "fused_score": 0.0,
                    "metadata": result.get("metadata", {}),
                }
            merged[cid]["fused_score"] += rrf_score
            merged[cid]["dense_score"] = result.get("dense_score", 0.0)

        # Process sparse results (ranked by sparse_score)
        for rank, result in enumerate(sparse_results):
            cid = result["chunk_id"]
            rrf_score = self.sparse_weight / (self.rrf_k + rank + 1)
            if cid not in merged:
                merged[cid] = {
                    "chunk_id": cid,
                    "content": result["content"],
                    "document_name": result.get("document_name", "unknown"),
                    "dense_score": 0.0,
                    "sparse_score": result.get("sparse_score", 0.0),
                    "fused_score": 0.0,
                    "metadata": result.get("metadata", {}),
                }
            merged[cid]["fused_score"] += rrf_score
            merged[cid]["sparse_score"] = result.get("sparse_score", 0.0)

        # Sort by fused score descending
        fused = sorted(merged.values(), key=lambda x: x["fused_score"], reverse=True)

        logger.info(
            f"Score fusion: {len(dense_results)} dense + {len(sparse_results)} sparse "
            f"→ {len(fused)} unique results"
        )
        return fused


# Singleton
score_fusion = ScoreFusion()
