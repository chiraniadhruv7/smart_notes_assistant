"""
Full retrieval pipeline orchestrator.
Coordinates all retrieval stages from query to final context.
"""

import time
from typing import Any

from app.retrieval.query_rewriter import query_rewriter
from app.retrieval.dense_retriever import dense_retriever
from app.retrieval.sparse_retriever import sparse_retriever
from app.retrieval.fusion import score_fusion
from app.retrieval.reranker import reranker
from app.retrieval.compressor import context_compressor
from app.models import RetrievalResult, RetrievalPipelineResult, Citation
from app.utils.logging import get_logger

logger = get_logger(__name__)


class RetrievalPipeline:
    """
    End-to-end retrieval pipeline orchestrating 7 stages.

    ┌─────────────────────────────────────────────────────────┐
    │                  RETRIEVAL PIPELINE                      │
    │                                                          │
    │  1. Query Rewriting                                      │
    │     └─ LLM generates 3-4 query variants                 │
    │                                                          │
    │  2. Dense Retrieval (top 20)                             │
    │     └─ Semantic ANN search via Qdrant                    │
    │                                                          │
    │  3. BM25 Retrieval (top 20)                              │
    │     └─ Keyword matching via inverted index               │
    │                                                          │
    │  4. Reciprocal Rank Fusion                               │
    │     └─ Combine dense + sparse with weighted RRF          │
    │                                                          │
    │  5. Cross-Encoder Reranking                              │
    │     └─ Fine-grained relevance scoring (→ top 5)          │
    │                                                          │
    │  6. Context Compression                                  │
    │     └─ Remove irrelevant sentences from each chunk       │
    │                                                          │
    │  7. Context Formatting                                   │
    │     └─ Format with source labels for LLM prompt          │
    └─────────────────────────────────────────────────────────┘
    """

    async def retrieve(
        self,
        query: str,
        filters: dict[str, str] | None = None,
        enable_rewrite: bool = False,
        enable_compression: bool = False,
    ) -> tuple[str, list[Citation], RetrievalPipelineResult]:
        """
        Run the full retrieval pipeline.

        Args:
            query: User's question.
            filters: Optional metadata filters.
            enable_rewrite: Whether to rewrite the query.
            enable_compression: Whether to compress context.

        Returns:
            Tuple of (formatted_context, citations, pipeline_result).
        """
        pipeline_start = time.perf_counter()

        # ── Stage 1: Query Rewriting ────────────────────
        if enable_rewrite:
            queries = await query_rewriter.rewrite(query)
        else:
            queries = [query]

        # ── Stage 2 & 3: Parallel Dense + Sparse Retrieval ─
        # Use the original (first) query for retrieval.
        # Multiple variants could be used for recall expansion in the future.
        primary_query = queries[0]

        dense_results = await dense_retriever.retrieve(
            primary_query, filters=filters
        )
        sparse_results = await sparse_retriever.retrieve(primary_query)

        # ── Stage 4: Score Fusion ───────────────────────
        fused_results = score_fusion.fuse(dense_results, sparse_results)

        # ── Stage 5: Cross-Encoder Reranking ────────────
        reranked_results = await reranker.rerank(primary_query, fused_results)

        # ── Stage 6: Context Compression ────────────────
        if enable_compression:
            compressed_results = await context_compressor.compress(
                primary_query, reranked_results
            )
        else:
            compressed_results = reranked_results

        # ── Stage 7: Context Formatting ─────────────────
        formatted_context = context_compressor.format_context(compressed_results)

        # Build citations from final results
        citations = [
            Citation(
                document_name=r.get("document_name", "unknown"),
                chunk_id=r.get("chunk_id", ""),
                content=r.get("original_content", r.get("content", "")),
                relevance_score=r.get("rerank_score", r.get("fused_score", 0.0)),
                metadata={k: str(v) for k, v in r.get("metadata", {}).items()},
            )
            for r in compressed_results
        ]

        # Build pipeline result for diagnostics
        elapsed = (time.perf_counter() - pipeline_start) * 1000
        pipeline_result = RetrievalPipelineResult(
            query=query,
            rewritten_queries=queries,
            results=[
                RetrievalResult(
                    chunk_id=r.get("chunk_id", ""),
                    document_name=r.get("document_name", "unknown"),
                    content=r.get("content", ""),
                    dense_score=r.get("dense_score", 0.0),
                    sparse_score=r.get("sparse_score", 0.0),
                    fused_score=r.get("fused_score", 0.0),
                    rerank_score=r.get("rerank_score", 0.0),
                    metadata=r.get("metadata", {}),
                )
                for r in compressed_results
            ],
            total_dense_candidates=len(dense_results),
            total_sparse_candidates=len(sparse_results),
            retrieval_time_ms=round(elapsed, 2),
        )

        logger.info(
            f"Retrieval pipeline: {len(dense_results)} dense + {len(sparse_results)} sparse "
            f"→ {len(fused_results)} fused → {len(reranked_results)} reranked "
            f"→ {len(compressed_results)} final in {elapsed:.0f}ms"
        )

        return formatted_context, citations, pipeline_result


# Singleton
retrieval_pipeline = RetrievalPipeline()
