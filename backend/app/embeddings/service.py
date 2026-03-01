"""
Embedding service using SentenceTransformers.
Provides batch encoding with Redis caching for deduplication.
Loads model once at startup and reuses across requests.
"""

import time
from typing import Sequence

import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import settings
from app.core.cache import cache_service
from app.utils.logging import get_logger

logger = get_logger(__name__)


class EmbeddingService:
    """
    Manages text embedding via SentenceTransformers.

    Design decisions:
    - Model is loaded lazily on first use and kept in memory (singleton pattern).
    - Embeddings are cached in Redis keyed by content hash to avoid recomputation.
    - Batch encoding uses the model's native batching for GPU/CPU efficiency.
    """

    def __init__(self) -> None:
        self._model: SentenceTransformer | None = None
        self._model_name: str = settings.embedding_model
        self._dimension: int = settings.embedding_dim

    def load_model(self) -> None:
        """Load the SentenceTransformer model into memory."""
        logger.info(f"Loading embedding model: {self._model_name}")
        start = time.perf_counter()
        self._model = SentenceTransformer(self._model_name)
        elapsed = (time.perf_counter() - start) * 1000
        logger.info(f"Embedding model loaded in {elapsed:.0f}ms")

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    async def encode(self, texts: list[str], use_cache: bool = True) -> list[list[float]]:
        """
        Encode a list of texts into dense vectors.

        Steps:
        1. Check Redis cache for each text.
        2. Batch-encode only uncached texts via SentenceTransformer.
        3. Store new embeddings in Redis.
        4. Return all embeddings in original order.
        """
        if not self._model:
            self.load_model()

        results: list[list[float] | None] = [None] * len(texts)
        to_encode_indices: list[int] = []
        to_encode_texts: list[str] = []

        # Step 1: Check cache
        if use_cache:
            for i, text in enumerate(texts):
                cached = await cache_service.get("emb", text)
                if cached is not None:
                    results[i] = cached
                else:
                    to_encode_indices.append(i)
                    to_encode_texts.append(text)
        else:
            to_encode_indices = list(range(len(texts)))
            to_encode_texts = texts

        # Step 2: Batch encode uncached texts
        if to_encode_texts:
            logger.info(f"Encoding {len(to_encode_texts)} texts (cache hits: {len(texts) - len(to_encode_texts)})")
            start = time.perf_counter()
            embeddings: np.ndarray = self._model.encode(  # type: ignore[union-attr]
                to_encode_texts,
                batch_size=settings.embedding_batch_size,
                show_progress_bar=False,
                normalize_embeddings=True,
            )
            elapsed = (time.perf_counter() - start) * 1000
            logger.info(f"Encoded {len(to_encode_texts)} texts in {elapsed:.0f}ms")

            # Step 3: Store in cache and fill results
            for idx, emb_idx in enumerate(to_encode_indices):
                vec = embeddings[idx].tolist()
                results[emb_idx] = vec
                if use_cache:
                    await cache_service.set("emb", to_encode_texts[idx], vec)

        return results  # type: ignore[return-value]

    async def encode_query(self, query: str) -> list[float]:
        """Encode a single query string."""
        embeddings = await self.encode([query], use_cache=False)
        return embeddings[0]

    async def health_check(self) -> dict:
        """Return embedding service status."""
        return {
            "status": "loaded" if self.is_loaded else "not_loaded",
            "model": self._model_name,
            "dimension": self._dimension,
        }


# Singleton instance
embedding_service = EmbeddingService()
