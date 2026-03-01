"""
Custom exception hierarchy and FastAPI exception handlers.
Provides consistent error responses across the API.
"""

from fastapi import Request
from fastapi.responses import JSONResponse
from typing import Any


# ── Exception Hierarchy ───────────────────────────────────


class RAGException(Exception):
    """Base exception for all RAG application errors."""

    def __init__(self, message: str, status_code: int = 500, details: dict[str, Any] | None = None):
        self.message = message
        self.status_code = status_code
        self.details = details or {}
        super().__init__(message)


class IngestionError(RAGException):
    """Raised when document ingestion fails."""

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(message, status_code=422, details=details)


class RetrievalError(RAGException):
    """Raised when the retrieval pipeline fails."""

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(message, status_code=500, details=details)


class LLMError(RAGException):
    """Raised when the LLM service fails."""

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(message, status_code=502, details=details)


class VectorStoreError(RAGException):
    """Raised for vector database connection or query failures."""

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(message, status_code=503, details=details)


class ConfigurationError(RAGException):
    """Raised for invalid or missing configuration."""

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(message, status_code=500, details=details)


class FileTooLargeError(RAGException):
    """Raised when an uploaded file exceeds the size limit."""

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(message, status_code=413, details=details)


class UnsupportedFileTypeError(RAGException):
    """Raised when the file type is not supported."""

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(message, status_code=415, details=details)


# ── FastAPI Exception Handlers ────────────────────────────


async def rag_exception_handler(request: Request, exc: RAGException) -> JSONResponse:
    """Handle all RAG-specific exceptions with consistent JSON errors."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": type(exc).__name__,
            "message": exc.message,
            "details": exc.details,
            "path": str(request.url),
        },
    )


async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all handler for unexpected exceptions."""
    return JSONResponse(
        status_code=500,
        content={
            "error": "InternalServerError",
            "message": "An unexpected error occurred.",
            "details": {"type": type(exc).__name__, "info": str(exc)},
            "path": str(request.url),
        },
    )
