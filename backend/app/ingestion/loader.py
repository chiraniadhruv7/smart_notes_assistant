"""
Document loaders for various file formats.
Extracts raw text and metadata from uploaded files.
"""

import os
from pathlib import Path
from typing import Any

from app.utils.logging import get_logger
from app.utils.exceptions import UnsupportedFileTypeError

logger = get_logger(__name__)

# Supported file extensions and their MIME types
SUPPORTED_TYPES: dict[str, str] = {
    ".txt": "text/plain",
    ".md": "text/markdown",
    ".pdf": "application/pdf",
    ".csv": "text/csv",
    ".json": "application/json",
}


def load_text_file(file_path: Path) -> str:
    """Load a plain text or markdown file."""
    return file_path.read_text(encoding="utf-8", errors="replace")


def load_pdf_file(file_path: Path) -> str:
    """
    Extract text from a PDF using PyMuPDF (fitz).
    Falls back to page-by-page extraction with page markers.
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        logger.warning("PyMuPDF not installed, falling back to basic PDF read")
        return f"[PDF file: {file_path.name} — install PyMuPDF for text extraction]"

    doc = fitz.open(str(file_path))
    pages: list[str] = []
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text("text").strip()
        if text:
            pages.append(f"[Page {page_num}]\n{text}")
    doc.close()
    return "\n\n".join(pages)


def load_csv_file(file_path: Path) -> str:
    """Load CSV as formatted text rows."""
    import csv
    rows: list[str] = []
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            rows.append(" | ".join(row))
            if i > 5000:  # Safety limit
                rows.append(f"... truncated at {i} rows")
                break
    return "\n".join(rows)


def load_json_file(file_path: Path) -> str:
    """Load JSON as formatted text."""
    import json
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return json.dumps(data, indent=2, ensure_ascii=False)


# Loader registry
LOADERS: dict[str, Any] = {
    ".txt": load_text_file,
    ".md": load_text_file,
    ".pdf": load_pdf_file,
    ".csv": load_csv_file,
    ".json": load_json_file,
}


def load_document(file_path: str | Path) -> tuple[str, dict[str, Any]]:
    """
    Load a document and return (text_content, metadata).

    Args:
        file_path: Path to the file to load.

    Returns:
        Tuple of (extracted_text, metadata_dict).

    Raises:
        UnsupportedFileTypeError: If file type is not supported.
        FileNotFoundError: If file does not exist.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    ext = path.suffix.lower()
    if ext not in LOADERS:
        raise UnsupportedFileTypeError(
            f"Unsupported file type: {ext}",
            details={"supported": list(SUPPORTED_TYPES.keys())},
        )

    loader = LOADERS[ext]
    text = loader(path)

    metadata = {
        "filename": path.name,
        "file_type": ext,
        "file_size_bytes": path.stat().st_size,
        "source": str(path.parent),
    }

    logger.info(f"Loaded document: {path.name} ({len(text)} chars)")
    return text, metadata
