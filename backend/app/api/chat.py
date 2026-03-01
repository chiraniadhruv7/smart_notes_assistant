"""
Chat API endpoint with Server-Sent Events (SSE) streaming.
Orchestrates retrieval → LLM generation → streaming response.
"""

import time
import json
import uuid

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from app.models import ChatRequest, ChatResponse, Citation
from app.retrieval.pipeline import retrieval_pipeline
from app.llm.service import llm_service
from app.utils.logging import get_logger, correlation_id, generate_correlation_id
from app.utils.exceptions import RAGException

logger = get_logger(__name__)
router = APIRouter(prefix="/api", tags=["chat"])


@router.post("/chat", response_model=None)
async def chat(request: ChatRequest) -> StreamingResponse | ChatResponse:
    """
    Main chat endpoint.

    Flow:
    1. Run retrieval pipeline to find relevant context.
    2. Generate LLM response using retrieved context.
    3. Stream response tokens via SSE or return complete response.
    """
    cid = generate_correlation_id()
    correlation_id.set(cid)

    conversation_id = request.conversation_id or str(uuid.uuid4())
    logger.info(f"Chat request: query='{request.query[:80]}...', convo={conversation_id}")

    total_start = time.perf_counter()

    # ── Step 1: Retrieval ─────────────────────────────
    try:
        context, citations, pipeline_result = await retrieval_pipeline.retrieve(
            query=request.query,
            filters=request.metadata_filter,
        )
    except Exception as e:
        logger.error(f"Retrieval failed: {e}")
        context = ""
        citations = []
        pipeline_result = None

    retrieval_time = (time.perf_counter() - total_start) * 1000

    # ── Step 2: Store user message in memory ──────────
    llm_service.memory.add_message(conversation_id, "user", request.query)

    # ── Step 3: Generate response ─────────────────────
    if request.stream:
        return StreamingResponse(
            _stream_chat_response(
                query=request.query,
                context=context,
                citations=citations,
                conversation_id=conversation_id,
                retrieval_time=retrieval_time,
                total_start=total_start,
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Correlation-ID": cid,
            },
        )
    else:
        # Non-streaming response
        gen = await llm_service.generate(
            query=request.query,
            context=context,
            conversation_id=conversation_id,
            stream=False,
        )
        answer = gen if isinstance(gen, str) else ""

        llm_service.memory.add_message(conversation_id, "assistant", answer)

        total_time = (time.perf_counter() - total_start) * 1000
        return ChatResponse(
            answer=answer,
            conversation_id=conversation_id,
            citations=citations,
            tokens_used=llm_service.total_tokens_used,
            retrieval_time_ms=round(retrieval_time, 2),
            total_time_ms=round(total_time, 2),
        )


async def _stream_chat_response(
    query: str,
    context: str,
    citations: list[Citation],
    conversation_id: str,
    retrieval_time: float,
    total_start: float,
):
    """
    Async generator for SSE streaming.

    Event types:
    - "token": Individual generated token
    - "citations": Array of source citations
    - "done": Final event with timing stats
    - "error": Error details
    """
    # Send citations first so frontend can display them while streaming
    citations_data = [c.model_dump() for c in citations]
    yield f"event: citations\ndata: {json.dumps(citations_data, default=str)}\n\n"

    # Stream LLM tokens
    full_response = []
    try:
        gen = await llm_service.generate(
            query=query,
            context=context,
            conversation_id=conversation_id,
            stream=True,
        )

        async for token in gen:
            full_response.append(token)
            yield f"event: token\ndata: {json.dumps({'token': token})}\n\n"

        # Store complete response in memory
        complete = "".join(full_response)
        llm_service.memory.add_message(conversation_id, "assistant", complete)

        # Send final stats
        total_time = (time.perf_counter() - total_start) * 1000
        done_data = {
            "conversation_id": conversation_id,
            "tokens_used": llm_service.total_tokens_used,
            "retrieval_time_ms": round(retrieval_time, 2),
            "total_time_ms": round(total_time, 2),
        }
        yield f"event: done\ndata: {json.dumps(done_data)}\n\n"

    except Exception as e:
        logger.error(f"Streaming error: {e}")
        yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"
