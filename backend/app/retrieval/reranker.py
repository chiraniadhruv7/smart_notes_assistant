"""
Cross-encoder reranker for precision refinement.
Re-scores candidate documents using a more powerful model
that sees both query and document together.
"""

import time
from typing import Any

from sentence_transformers import CrossEncoder

from app.config import settings
from app.utils.logging import get_logger

logger = get_logger(__name__)


class Reranker:
    """
    Cross-encoder reranker using a fine-tuned MS MARCO model.

    Why cross-encoder reranking matters:
    - Bi-encoders (used in dense retrieval) encode query and document
      independently, which is fast but loses cross-attention signals.
    - Cross-encoders process query-document pairs together through
      all transformer layers, capturing fine-grained relevance.
    - This is too expensive for initial retrieval (O(N) documents)
      but perfect for reranking a small candidate set (top 20-40).

    Pipeline position: After fusion, before final selection.
    Input: ~40 fused candidates.
    Output: Top 5 reranked by cross-encoder score.
    """

    def __init__(self) -> None:
        self._model: CrossEncoder | None = None
        self._model_name: str = settings.reranker_model

    def load_model(self) -> None:
        """Load the cross-encoder model."""
        logger.info(f"Loading reranker model: {self._model_name}")
        start = time.perf_counter()
        self._model = CrossEncoder(self._model_name, max_length=512)
        elapsed = (time.perf_counter() - start) * 1000
        logger.info(f"Reranker model loaded in {elapsed:.0f}ms")

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    async def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Rerank candidates using the cross-encoder.

        Args:
            query: The user's query.
            candidates: List of candidate results with 'content' field.
            top_k: Number of top results to return.

        Returns:
            Top-K results sorted by cross-encoder score.
        """
        if not self._model:
            self.load_model()

        if not candidates:
            return []

        k = top_k or settings.rerank_top_k

        # Create query-document pairs for the cross-encoder
        pairs = [(query, c["content"]) for c in candidates]

        start = time.perf_counter()
        scores = self._model.predict(pairs, show_progress_bar=False)  # type: ignore[union-attr]
        elapsed = (time.perf_counter() - start) * 1000

        # Attach rerank scores (apply sigmoid to convert logits → probability)
        import math
        for i, score in enumerate(scores):
            normalized = 1.0 / (1.0 + math.exp(-float(score)))
            candidates[i]["rerank_score"] = normalized

        reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)[:k]

        logger.info(
            f"Reranked {len(candidates)} candidates → top {len(reranked)} in {elapsed:.0f}ms"
        )
        return reranked


# Singleton
reranker = Reranker()
