"""
Ingestion pipeline orchestrator.
Coordinates: loading → chunking → embedding → vector store upsert.
"""

import os
import time
import uuid
from pathlib import Path
from typing import Any

from app.config import settings
from app.ingestion.loader import load_document, SUPPORTED_TYPES
from app.ingestion.chunker import RecursiveChunker
from app.embeddings.service import embedding_service
from app.vectorstore.qdrant_store import vector_store
from app.models import IngestionResult, IngestionStatus
from app.utils.logging import get_logger
from app.utils.exceptions import IngestionError, FileTooLargeError

logger = get_logger(__name__)


class IngestionPipeline:
    """
    End-to-end document ingestion pipeline.

    Flow:
    1. Validate file (size, type).
    2. Load document text via format-specific loader.
    3. Chunk text with overlap using RecursiveChunker.
    4. Generate embeddings for all chunks via EmbeddingService.
    5. Upsert chunk vectors + payloads into Qdrant.
    6. Return ingestion statistics.

    Each chunk payload stores:
    - content: the chunk text
    - document_id: unique ID for the source document
    - filename, file_type, source: metadata for filtering & citation
    - chunk_index: position within the document
    - start_char, end_char: character offsets for highlighting
    """

    def __init__(self) -> None:
        self._chunker = RecursiveChunker()

    async def ingest_file(
        self,
        file_path: str | Path,
        tags: list[str] | None = None,
        source: str = "",
    ) -> IngestionResult:
        """
        Ingest a single file into the vector store.

        Args:
            file_path: Path to the file.
            tags: Optional tags for metadata filtering.
            source: Source identifier for the document.

        Returns:
            IngestionResult with status and statistics.
        """
        path = Path(file_path)
        document_id = str(uuid.uuid4())
        start = time.perf_counter()

        try:
            # Step 1: Validate
            self._validate_file(path)

            # Step 2: Load
            text, metadata = load_document(path)
            metadata["document_id"] = document_id
            metadata["source"] = source or str(path.parent)
            metadata["tags"] = tags or []

            # Step 3: Chunk
            chunks = self._chunker.chunk(text, metadata)
            if not chunks:
                return IngestionResult(
                    document_id=document_id,
                    filename=path.name,
                    status=IngestionStatus.COMPLETED,
                    chunks_created=0,
                    processing_time_ms=0.0,
                )

            # Step 4: Embed
            chunk_texts = [c.content for c in chunks]
            embeddings = await embedding_service.encode(chunk_texts)

            # Step 5: Upsert into Qdrant
            chunk_ids = [str(uuid.uuid5(uuid.UUID(document_id), str(c.chunk_index))) for c in chunks]
            payloads = [
                {
                    "content": c.content,
                    "document_id": document_id,
                    "filename": metadata["filename"],
                    "file_type": metadata["file_type"],
                    "source": metadata.get("source", ""),
                    "tags": metadata.get("tags", []),
                    "chunk_index": c.chunk_index,
                    "start_char": c.start_char,
                    "end_char": c.end_char,
                }
                for c in chunks
            ]

            await vector_store.upsert(chunk_ids, embeddings, payloads)

            elapsed = (time.perf_counter() - start) * 1000
            logger.info(f"Ingested {path.name}: {len(chunks)} chunks in {elapsed:.0f}ms")

            return IngestionResult(
                document_id=document_id,
                filename=path.name,
                status=IngestionStatus.COMPLETED,
                chunks_created=len(chunks),
                processing_time_ms=round(elapsed, 2),
            )

        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            logger.error(f"Ingestion failed for {path.name}: {e}")
            return IngestionResult(
                document_id=document_id,
                filename=path.name,
                status=IngestionStatus.FAILED,
                error_message=str(e),
                processing_time_ms=round(elapsed, 2),
            )

    async def ingest_directory(
        self,
        dir_path: str | Path,
        tags: list[str] | None = None,
        source: str = "",
    ) -> list[IngestionResult]:
        """
        Ingest all supported files from a directory.

        Args:
            dir_path: Path to the directory.
            tags: Optional tags for all documents.
            source: Source identifier.

        Returns:
            List of IngestionResult for each file.
        """
        path = Path(dir_path)
        if not path.is_dir():
            raise IngestionError(f"Not a directory: {path}")

        results: list[IngestionResult] = []
        supported_extensions = set(SUPPORTED_TYPES.keys())

        for file_path in sorted(path.rglob("*")):
            if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                result = await self.ingest_file(file_path, tags=tags, source=source)
                results.append(result)

        logger.info(
            f"Directory ingestion complete: {len(results)} files, "
            f"{sum(r.chunks_created for r in results)} total chunks"
        )
        return results

    def _validate_file(self, path: Path) -> None:
        """Validate file exists, is supported, and within size limits."""
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        ext = path.suffix.lower()
        if ext not in SUPPORTED_TYPES:
            raise IngestionError(
                f"Unsupported file type: {ext}. Supported: {list(SUPPORTED_TYPES.keys())}"
            )

        size_mb = path.stat().st_size / (1024 * 1024)
        if size_mb > settings.max_file_size_mb:
            raise FileTooLargeError(
                f"File too large: {size_mb:.1f}MB (max: {settings.max_file_size_mb}MB)"
            )


# Singleton instance
ingestion_pipeline = IngestionPipeline()
