"""
Retrieval quality evaluation metrics.
Supports Recall@K, MRR, and latency tracking.
"""

from typing import Any

from app.utils.logging import get_logger

logger = get_logger(__name__)


class RetrievalEvaluator:
    """
    Tracks and computes retrieval quality metrics.

    Metrics:
    - Recall@K: Fraction of relevant documents found in top-K results.
    - MRR (Mean Reciprocal Rank): Average of 1/rank for the first
      relevant result across queries.
    - Latency: Average retrieval pipeline time per query.

    Usage:
    - Call record_query() after each retrieval with the results and
      known-relevant document IDs.
    - Call get_metrics() to retrieve the aggregate metrics.
    """

    def __init__(self) -> None:
        self._query_records: list[dict[str, Any]] = []

    def record_query(
        self,
        query: str,
        retrieved_ids: list[str],
        relevant_ids: list[str],
        latency_ms: float,
    ) -> None:
        """
        Record a single query's retrieval results for evaluation.

        Args:
            query: The search query.
            retrieved_ids: Chunk IDs returned by the pipeline (in rank order).
            relevant_ids: Known-relevant chunk IDs (ground truth).
            latency_ms: Time taken for the retrieval pipeline.
        """
        self._query_records.append({
            "query": query,
            "retrieved_ids": retrieved_ids,
            "relevant_ids": set(relevant_ids),
            "latency_ms": latency_ms,
        })

    def recall_at_k(self, k: int) -> float:
        """
        Compute Recall@K across all recorded queries.

        Recall@K = (# relevant docs in top-K) / (# total relevant docs)
        """
        if not self._query_records:
            return 0.0

        recalls = []
        for record in self._query_records:
            relevant = record["relevant_ids"]
            if not relevant:
                continue
            retrieved_k = set(record["retrieved_ids"][:k])
            found = len(relevant & retrieved_k)
            recalls.append(found / len(relevant))

        return sum(recalls) / len(recalls) if recalls else 0.0

    def mrr(self) -> float:
        """
        Compute Mean Reciprocal Rank across all recorded queries.

        MRR = (1/N) * Σ (1 / rank_of_first_relevant)
        """
        if not self._query_records:
            return 0.0

        reciprocal_ranks = []
        for record in self._query_records:
            relevant = record["relevant_ids"]
            if not relevant:
                continue
            for rank, doc_id in enumerate(record["retrieved_ids"], 1):
                if doc_id in relevant:
                    reciprocal_ranks.append(1.0 / rank)
                    break
            else:
                reciprocal_ranks.append(0.0)

        return sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0

    def average_latency(self) -> float:
        """Average retrieval latency in ms."""
        if not self._query_records:
            return 0.0
        return sum(r["latency_ms"] for r in self._query_records) / len(self._query_records)

    def get_metrics(self, k_values: list[int] | None = None) -> dict[str, Any]:
        """
        Get all evaluation metrics as a dictionary.

        Args:
            k_values: List of K values for Recall@K (default: [1, 3, 5, 10]).
        """
        ks = k_values or [1, 3, 5, 10]
        return {
            "recall_at_k": {k: round(self.recall_at_k(k), 4) for k in ks},
            "mrr": round(self.mrr(), 4),
            "average_latency_ms": round(self.average_latency(), 2),
            "total_queries": len(self._query_records),
        }

    def reset(self) -> None:
        """Clear all recorded queries."""
        self._query_records.clear()


# Singleton
retrieval_evaluator = RetrievalEvaluator()
