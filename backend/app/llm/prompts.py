"""
Prompt templates for the RAG system.
Separates prompt engineering from business logic for easy iteration.
"""

# ── RAG Answer Prompt ─────────────────────────────────────

RAG_SYSTEM_PROMPT = """You are a knowledgeable AI assistant that answers questions based on the provided context documents. Follow these rules strictly:

1. Base your answer ONLY on the provided context. Do not use prior knowledge.
2. If the context doesn't contain enough information, say so clearly.
3. Cite your sources using [Source: filename] notation after each claim.
4. Structure your answer clearly with paragraphs or bullet points.
5. Be precise and avoid speculation.
6. If multiple sources agree, synthesize their information."""

RAG_USER_PROMPT = """Context Documents:
---
{context}
---

Question: {query}

Provide a comprehensive answer based on the context above. Include citations."""


# ── Query Rewriting Prompt ────────────────────────────────

QUERY_REWRITE_PROMPT = """You are a search query optimizer. Given a user question, generate {num_variants} alternative search queries that capture different aspects or phrasings of the same information need.

Original question: {query}

Rules:
- Each variant should emphasize different keywords or perspectives
- Keep queries concise (under 20 words each)
- Include the original query as the first variant
- Output ONLY the queries, one per line, no numbering or bullets"""


# ── Context Compression Prompt ────────────────────────────

CONTEXT_COMPRESSION_PROMPT = """Given the following text chunk and a user query, extract ONLY the sentences that are relevant to answering the query. Remove all irrelevant information.

Query: {query}

Text chunk:
{chunk}

Output only the relevant sentences, preserving their original wording. If nothing is relevant, output "NOT_RELEVANT"."""


# ── Conversation Summary Prompt ───────────────────────────

CONVERSATION_SUMMARY_PROMPT = """Summarize the following conversation history in 2-3 sentences, capturing the key topics and any context needed to understand the next question.

Conversation:
{history}

Summary:"""
