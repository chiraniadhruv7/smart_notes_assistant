"""
Context compression to reduce noise in retrieved chunks.
Removes irrelevant sentences before passing context to the LLM,
improving answer quality and reducing token usage.
"""

from typing import Any

from app.llm.service import llm_service
from app.llm.prompts import CONTEXT_COMPRESSION_PROMPT
from app.utils.logging import get_logger

logger = get_logger(__name__)


class ContextCompressor:
    """
    Compresses retrieved chunks by extracting only query-relevant sentences.

    Why compression matters:
    - Retrieved chunks often contain noise: text that matched via keywords
      but isn't relevant to the actual question.
    - Feeding noisy context to the LLM degrades answer quality and
      wastes tokens (increasing latency and cost).
    - LLM-based compression identifies and keeps only relevant sentences.

    Trade-off:
    - Compression adds one LLM call per chunk, increasing latency.
    - For time-sensitive queries, compression can be skipped.
    - For quality-sensitive queries (enterprise, medical), it's valuable.
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled

    async def compress(
        self,
        query: str,
        results: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Compress each result's content to query-relevant sentences.

        Args:
            query: The user's query.
            results: Reranked retrieval results.

        Returns:
            Results with compressed content (originals preserved in metadata).
        """
        if not self.enabled or not results:
            return results

        compressed = []
        for result in results:
            try:
                prompt = CONTEXT_COMPRESSION_PROMPT.format(
                    query=query,
                    chunk=result["content"],
                )
                squeezed = await llm_service.generate_raw(prompt, max_tokens=512)

                if squeezed.strip() == "NOT_RELEVANT":
                    logger.info(f"Chunk {result['chunk_id']} filtered as irrelevant")
                    continue

                # Preserve original content in metadata for citation
                compressed_result = dict(result)
                compressed_result["original_content"] = result["content"]
                compressed_result["content"] = squeezed.strip()
                compressed.append(compressed_result)

            except Exception as e:
                logger.warning(f"Compression failed for chunk {result['chunk_id']}: {e}")
                compressed.append(result)  # Keep original on failure

        logger.info(f"Compressed {len(results)} → {len(compressed)} relevant chunks")
        return compressed

    def format_context(self, results: list[dict[str, Any]]) -> str:
        """
        Format compressed results into a context string for the LLM prompt.

        Each chunk is labeled with its source for citation tracking.
        """
        sections = []
        for i, result in enumerate(results, 1):
            source = result.get("document_name", "unknown")
            content = result["content"]
            sections.append(f"[Source {i}: {source}]\n{content}")

        return "\n\n---\n\n".join(sections)


# Singleton
context_compressor = ContextCompressor()
