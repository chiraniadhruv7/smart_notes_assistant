"""
Text chunking with recursive splitting.
Preserves metadata and tracks character offsets for citation mapping.
"""

from dataclasses import dataclass, field
from typing import Any

from app.config import settings
from app.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TextChunk:
    """A chunk of text with positional and metadata information."""
    content: str
    chunk_index: int
    start_char: int
    end_char: int
    metadata: dict[str, Any] = field(default_factory=dict)


class RecursiveChunker:
    """
    Splits text into overlapping chunks using a hierarchy of separators.

    Strategy:
    1. Try splitting on paragraph breaks (\\n\\n) first.
    2. Fall back to sentence boundaries (. ! ?).
    3. Final fallback to word boundaries.
    4. Each chunk has configurable overlap with the previous chunk
       to preserve context across boundaries.

    This approach preserves semantic coherence better than
    fixed-size windowing.
    """

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        separators: list[str] | None = None,
    ):
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", "! ", "? ", " "]

    def chunk(self, text: str, metadata: dict[str, Any] | None = None) -> list[TextChunk]:
        """
        Split text into chunks with overlap and metadata.

        Args:
            text: The full document text.
            metadata: Document-level metadata to attach to each chunk.

        Returns:
            List of TextChunk objects with content, offsets, and metadata.
        """
        if not text.strip():
            return []

        base_metadata = metadata or {}
        splits = self._recursive_split(text, self.separators)
        chunks = self._merge_splits(splits, base_metadata)

        logger.info(f"Chunked text into {len(chunks)} chunks (size={self.chunk_size}, overlap={self.chunk_overlap})")
        return chunks

    def _recursive_split(self, text: str, separators: list[str]) -> list[str]:
        """Recursively split text using the separator hierarchy."""
        final_splits: list[str] = []

        # Find the best separator that actually exists in the text
        separator = ""
        for sep in separators:
            if sep in text:
                separator = sep
                break

        if separator:
            parts = text.split(separator)
        else:
            # No separator found — character-level split
            return [text[i:i + self.chunk_size] for i in range(0, len(text), self.chunk_size)]

        current_parts: list[str] = []
        for part in parts:
            if len(part) > self.chunk_size:
                # Part too large — recurse with next separator level
                if current_parts:
                    merged = separator.join(current_parts)
                    final_splits.append(merged)
                    current_parts = []
                remaining_seps = separators[separators.index(separator) + 1:]
                if remaining_seps:
                    sub_splits = self._recursive_split(part, remaining_seps)
                    final_splits.extend(sub_splits)
                else:
                    final_splits.append(part)
            else:
                current_parts.append(part)

        if current_parts:
            final_splits.append(separator.join(current_parts))

        return final_splits

    def _merge_splits(self, splits: list[str], metadata: dict[str, Any]) -> list[TextChunk]:
        """
        Merge small splits into chunks of target size with overlap.
        Tracks character offsets for citation mapping.
        """
        chunks: list[TextChunk] = []
        current_text = ""
        current_start = 0
        char_offset = 0

        for split in splits:
            split = split.strip()
            if not split:
                char_offset += len(split) + 1
                continue

            test_text = f"{current_text} {split}".strip() if current_text else split

            if len(test_text) > self.chunk_size and current_text:
                # Current chunk is full — save it
                chunks.append(TextChunk(
                    content=current_text.strip(),
                    chunk_index=len(chunks),
                    start_char=current_start,
                    end_char=char_offset,
                    metadata=dict(metadata),
                ))

                # Start new chunk with overlap from the end of current chunk
                overlap_text = current_text[-self.chunk_overlap:] if len(current_text) > self.chunk_overlap else ""
                current_text = f"{overlap_text} {split}".strip()
                current_start = max(0, char_offset - self.chunk_overlap)
            else:
                current_text = test_text

            char_offset += len(split) + 1

        # Don't forget the last chunk
        if current_text.strip():
            chunks.append(TextChunk(
                content=current_text.strip(),
                chunk_index=len(chunks),
                start_char=current_start,
                end_char=char_offset,
                metadata=dict(metadata),
            ))

        return chunks
