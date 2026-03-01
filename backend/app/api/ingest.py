"""
Document ingestion API endpoint.
Handles file uploads and triggers the ingestion pipeline.
"""

import os
import shutil
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import JSONResponse

from app.config import settings
from app.ingestion.pipeline import ingestion_pipeline
from app.retrieval.sparse_retriever import sparse_retriever
from app.models import IngestionResult
from app.utils.logging import get_logger
from app.utils.exceptions import FileTooLargeError, UnsupportedFileTypeError

logger = get_logger(__name__)
router = APIRouter(prefix="/api", tags=["ingestion"])


@router.post("/ingest", response_model=list[IngestionResult])
async def ingest_files(
    files: list[UploadFile] = File(...),
    tags: str = Form(default=""),
    source: str = Form(default="upload"),
) -> list[IngestionResult]:
    """
    Upload and ingest one or more documents.

    Flow:
    1. Save uploaded files to the upload directory.
    2. Run the ingestion pipeline for each file.
    3. Rebuild the BM25 index to include new documents.
    4. Return ingestion results.
    """
    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)

    tag_list = [t.strip() for t in tags.split(",") if t.strip()]
    results: list[IngestionResult] = []

    for file in files:
        # Save to disk
        file_path = upload_dir / file.filename
        try:
            with open(file_path, "wb") as f:
                content = await file.read()
                # Check file size
                size_mb = len(content) / (1024 * 1024)
                if size_mb > settings.max_file_size_mb:
                    raise FileTooLargeError(
                        f"File {file.filename} is {size_mb:.1f}MB (max: {settings.max_file_size_mb}MB)"
                    )
                f.write(content)

            logger.info(f"Saved upload: {file.filename} ({len(content)} bytes)")

            # Run ingestion
            result = await ingestion_pipeline.ingest_file(
                file_path=file_path,
                tags=tag_list,
                source=source,
            )
            results.append(result)

        except (FileTooLargeError, UnsupportedFileTypeError) as e:
            results.append(IngestionResult(
                document_id="",
                filename=file.filename or "unknown",
                status="failed",
                error_message=str(e),
            ))
        except Exception as e:
            logger.error(f"Ingestion failed for {file.filename}: {e}")
            results.append(IngestionResult(
                document_id="",
                filename=file.filename or "unknown",
                status="failed",
                error_message=str(e),
            ))

    # Rebuild BM25 index after new documents
    await sparse_retriever.rebuild_index()

    logger.info(
        f"Ingestion complete: {len(results)} files, "
        f"{sum(r.chunks_created for r in results)} total chunks"
    )
    return results


@router.post("/ingest/directory")
async def ingest_directory(
    directory: str = Form(...),
    tags: str = Form(default=""),
    source: str = Form(default="directory"),
) -> list[IngestionResult]:
    """
    Ingest all supported files from a server-side directory.

    Useful for bulk ingestion of knowledge bases.
    """
    dir_path = Path(directory)
    if not dir_path.is_dir():
        return JSONResponse(
            status_code=400,
            content={"error": f"Not a directory: {directory}"},
        )

    tag_list = [t.strip() for t in tags.split(",") if t.strip()]
    results = await ingestion_pipeline.ingest_directory(
        dir_path=dir_path,
        tags=tag_list,
        source=source,
    )

    # Rebuild BM25 index
    await sparse_retriever.rebuild_index()

    return results
