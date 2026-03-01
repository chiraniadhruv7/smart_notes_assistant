"""
Pydantic request/response schemas for the API layer.
Provides strong typing and validation for all endpoints.
"""

from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum


# ── Enums ─────────────────────────────────────────────────


class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class IngestionStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


# ── Chat ──────────────────────────────────────────────────


class ChatMessage(BaseModel):
    """Single message in a conversation."""
    role: MessageRole
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ChatRequest(BaseModel):
    """Incoming chat request from the frontend."""
    query: str = Field(..., min_length=1, max_length=4096)
    conversation_id: str | None = None
    history: list[ChatMessage] = Field(default_factory=list)
    metadata_filter: dict[str, str] | None = None
    stream: bool = True


class Citation(BaseModel):
    """A source citation from retrieved documents."""
    document_name: str
    chunk_id: str
    content: str
    relevance_score: float
    page_number: int | None = None
    metadata: dict[str, str] = Field(default_factory=dict)


class ChatResponse(BaseModel):
    """Full chat response (non-streaming)."""
    answer: str
    conversation_id: str
    citations: list[Citation] = Field(default_factory=list)
    tokens_used: int = 0
    retrieval_time_ms: float = 0.0
    total_time_ms: float = 0.0


# ── Streaming ─────────────────────────────────────────────


class StreamEvent(BaseModel):
    """Server-Sent Event payload for streaming responses."""
    event: str  # "token", "citation", "done", "error"
    data: str


# ── Ingestion ─────────────────────────────────────────────


class DocumentMetadata(BaseModel):
    """Metadata attached to each ingested document."""
    filename: str
    file_type: str
    file_size_bytes: int
    ingested_at: datetime = Field(default_factory=datetime.utcnow)
    source: str = ""
    tags: list[str] = Field(default_factory=list)


class ChunkRecord(BaseModel):
    """A single text chunk with its metadata and embedding reference."""
    chunk_id: str
    document_id: str
    content: str
    metadata: DocumentMetadata
    chunk_index: int
    start_char: int
    end_char: int


class IngestionRequest(BaseModel):
    """Request to ingest documents."""
    tags: list[str] = Field(default_factory=list)
    source: str = ""


class IngestionResult(BaseModel):
    """Result from document ingestion."""
    document_id: str
    filename: str
    status: IngestionStatus
    chunks_created: int = 0
    error_message: str | None = None
    processing_time_ms: float = 0.0


# ── Retrieval ─────────────────────────────────────────────


class RetrievalResult(BaseModel):
    """A single retrieval result with scoring details."""
    chunk_id: str
    document_name: str
    content: str
    dense_score: float = 0.0
    sparse_score: float = 0.0
    fused_score: float = 0.0
    rerank_score: float = 0.0
    metadata: dict[str, str] = Field(default_factory=dict)


class RetrievalPipelineResult(BaseModel):
    """Full output of the retrieval pipeline."""
    query: str
    rewritten_queries: list[str] = Field(default_factory=list)
    results: list[RetrievalResult] = Field(default_factory=list)
    total_dense_candidates: int = 0
    total_sparse_candidates: int = 0
    retrieval_time_ms: float = 0.0


# ── Admin ─────────────────────────────────────────────────


class ServiceHealth(BaseModel):
    """Health of an individual service."""
    name: str
    status: str  # "healthy", "degraded", "down"
    latency_ms: float = 0.0
    details: dict[str, str] = Field(default_factory=dict)


class AdminDiagnostics(BaseModel):
    """Full system diagnostics response."""
    app_version: str
    uptime_seconds: float
    total_documents: int = 0
    total_chunks: int = 0
    index_size_mb: float = 0.0
    services: list[ServiceHealth] = Field(default_factory=list)
    memory_usage_mb: float = 0.0


# ── Evaluation ────────────────────────────────────────────


class EvaluationMetrics(BaseModel):
    """Retrieval quality metrics."""
    recall_at_k: dict[int, float] = Field(default_factory=dict)
    mrr: float = 0.0
    average_latency_ms: float = 0.0
    total_queries: int = 0
