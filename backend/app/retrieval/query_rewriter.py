"""
Query rewriting using the LLM service.
Generates multiple query variants to improve retrieval recall
by capturing different phrasings and aspects of the user's intent.
"""

from app.llm.prompts import QUERY_REWRITE_PROMPT
from app.llm.service import llm_service
from app.utils.logging import get_logger

logger = get_logger(__name__)


class QueryRewriter:
    """
    Rewrites a user query into multiple search variants.

    Why query rewriting matters:
    - Users often phrase questions differently from how information
      is written in documents.
    - A single query may miss relevant chunks that use different
      terminology or structure.
    - Generating 3-4 variants captures keyword diversity and
      improves recall without hurting precision (reranking filters noise).

    Example:
        Input:  "How does authentication work?"
        Output: ["How does authentication work?",
                 "user login verification process",
                 "auth token session management",
                 "security authentication mechanism"]
    """

    def __init__(self, num_variants: int = 3):
        self.num_variants = num_variants

    async def rewrite(self, query: str) -> list[str]:
        """
        Generate query variants using the LLM.

        Args:
            query: Original user query.

        Returns:
            List of query variants (always includes the original).
        """
        try:
            prompt = QUERY_REWRITE_PROMPT.format(
                num_variants=self.num_variants,
                query=query,
            )
            response = await llm_service.generate_raw(prompt, max_tokens=256)

            # Parse the LLM output into individual queries
            variants = [
                line.strip()
                for line in response.strip().split("\n")
                if line.strip() and len(line.strip()) > 3
            ]

            # Ensure original query is always first
            if query not in variants:
                variants.insert(0, query)

            # Limit to expected number
            variants = variants[: self.num_variants + 1]

            logger.info(f"Query rewritten into {len(variants)} variants")
            return variants

        except Exception as e:
            logger.warning(f"Query rewriting failed, using original: {e}")
            return [query]


# Singleton
query_rewriter = QueryRewriter()
