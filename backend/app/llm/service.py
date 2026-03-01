"""
LLM service supporting Ollama (local) and OpenAI APIs.
Provides both streaming and non-streaming generation with
conversation memory management.
"""

import time
import json
from typing import AsyncGenerator

import httpx

from app.config import settings
from app.utils.logging import get_logger
from app.utils.exceptions import LLMError
from app.llm.prompts import RAG_SYSTEM_PROMPT, RAG_USER_PROMPT

logger = get_logger(__name__)


class ConversationMemory:
    """
    Maintains conversation history per session.
    Uses a sliding window to keep context within token limits.
    """

    def __init__(self, max_messages: int = 20):
        self._conversations: dict[str, list[dict[str, str]]] = {}
        self._max_messages = max_messages

    def get_history(self, conversation_id: str) -> list[dict[str, str]]:
        return self._conversations.get(conversation_id, [])

    def add_message(self, conversation_id: str, role: str, content: str) -> None:
        if conversation_id not in self._conversations:
            self._conversations[conversation_id] = []
        self._conversations[conversation_id].append({"role": role, "content": content})
        # Sliding window: keep last N messages
        if len(self._conversations[conversation_id]) > self._max_messages:
            self._conversations[conversation_id] = self._conversations[conversation_id][-self._max_messages:]

    def clear(self, conversation_id: str) -> None:
        self._conversations.pop(conversation_id, None)

    def list_conversations(self) -> list[str]:
        return list(self._conversations.keys())


class LLMService:
    """
    Unified LLM interface supporting Ollama and OpenAI.

    Architecture:
    - Uses httpx async client for non-blocking HTTP calls
    - Streaming responses use async generators yielding tokens
    - Conversation history is injected into the system prompt
    - Provider is selected via environment config
    """

    def __init__(self) -> None:
        self._provider = settings.llm_provider
        self._memory = ConversationMemory()
        self._http_client: httpx.AsyncClient | None = None
        self._total_tokens_used: int = 0

    async def initialize(self) -> None:
        """Create the async HTTP client."""
        self._http_client = httpx.AsyncClient(timeout=120.0)
        logger.info(f"LLM service initialized with provider: {self._provider}")

    async def shutdown(self) -> None:
        """Close the HTTP client."""
        if self._http_client:
            await self._http_client.aclose()

    @property
    def memory(self) -> ConversationMemory:
        return self._memory

    @property
    def total_tokens_used(self) -> int:
        return self._total_tokens_used

    async def generate(
        self,
        query: str,
        context: str,
        conversation_id: str | None = None,
        stream: bool = True,
    ) -> AsyncGenerator[str, None] | str:
        """
        Generate a response using the RAG prompt template.

        If stream=True, returns an async generator yielding tokens.
        If stream=False, returns the complete response string.
        """
        # Build the final prompt
        user_prompt = RAG_USER_PROMPT.format(context=context, query=query)

        # Include conversation history if available
        messages = [{"role": "system", "content": RAG_SYSTEM_PROMPT}]
        if conversation_id:
            history = self._memory.get_history(conversation_id)
            messages.extend(history)
        messages.append({"role": "user", "content": user_prompt})

        if stream:
            return self._stream_response(messages, conversation_id, query)
        else:
            return await self._complete_response(messages, conversation_id, query)

    async def generate_raw(self, prompt: str, max_tokens: int = 512) -> str:
        """
        Generate a raw (non-RAG) completion.
        Used for query rewriting and context compression.
        """
        messages = [{"role": "user", "content": prompt}]
        return await self._complete_response(messages, max_tokens_override=max_tokens)

    async def _stream_response(
        self,
        messages: list[dict[str, str]],
        conversation_id: str | None = None,
        original_query: str | None = None,
    ) -> AsyncGenerator[str, None]:
        """Stream tokens from the LLM provider."""
        if self._provider == "ollama":
            async for token in self._stream_ollama(messages):
                yield token
        elif self._provider == "openai":
            async for token in self._stream_openai(messages):
                yield token
        else:
            raise LLMError(f"Unknown LLM provider: {self._provider}")

    async def _complete_response(
        self,
        messages: list[dict[str, str]],
        conversation_id: str | None = None,
        original_query: str | None = None,
        max_tokens_override: int | None = None,
    ) -> str:
        """Get a complete (non-streaming) response."""
        if self._provider == "ollama":
            return await self._complete_ollama(messages, max_tokens_override)
        elif self._provider == "openai":
            return await self._complete_openai(messages, max_tokens_override)
        else:
            raise LLMError(f"Unknown LLM provider: {self._provider}")

    # ── Ollama Implementation ─────────────────────────────

    async def _stream_ollama(self, messages: list[dict[str, str]]) -> AsyncGenerator[str, None]:
        """Stream from Ollama /api/chat endpoint."""
        if not self._http_client:
            raise LLMError("HTTP client not initialized")

        url = f"{settings.ollama_base_url}/api/chat"
        payload = {
            "model": settings.ollama_model,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": settings.llm_temperature,
                "num_predict": settings.llm_max_tokens,
            },
        }

        collected = []
        try:
            async with self._http_client.stream("POST", url, json=payload) as response:
                if response.status_code != 200:
                    text = await response.aread()
                    raise LLMError(f"Ollama error: {response.status_code} - {text.decode()}")

                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        data = json.loads(line)
                        token = data.get("message", {}).get("content", "")
                        if token:
                            collected.append(token)
                            yield token
                        if data.get("done"):
                            self._total_tokens_used += data.get("eval_count", 0)
                    except json.JSONDecodeError:
                        continue
        except httpx.HTTPError as e:
            raise LLMError(f"Ollama connection error: {e}")

    async def _complete_ollama(self, messages: list[dict[str, str]], max_tokens: int | None = None) -> str:
        """Non-streaming Ollama call."""
        if not self._http_client:
            raise LLMError("HTTP client not initialized")

        url = f"{settings.ollama_base_url}/api/chat"
        payload = {
            "model": settings.ollama_model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": settings.llm_temperature,
                "num_predict": max_tokens or settings.llm_max_tokens,
            },
        }

        try:
            response = await self._http_client.post(url, json=payload)
            if response.status_code != 200:
                raise LLMError(f"Ollama error: {response.status_code} - {response.text}")
            data = response.json()
            self._total_tokens_used += data.get("eval_count", 0)
            return data.get("message", {}).get("content", "")
        except httpx.HTTPError as e:
            raise LLMError(f"Ollama connection error: {e}")

    # ── OpenAI Implementation ─────────────────────────────

    async def _stream_openai(self, messages: list[dict[str, str]]) -> AsyncGenerator[str, None]:
        """Stream from OpenAI-compatible /v1/chat/completions endpoint."""
        if not self._http_client:
            raise LLMError("HTTP client not initialized")
        if not settings.openai_api_key:
            raise LLMError("OpenAI API key not configured")

        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {settings.openai_api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": settings.openai_model,
            "messages": messages,
            "stream": True,
            "temperature": settings.llm_temperature,
            "max_tokens": settings.llm_max_tokens,
        }

        try:
            async with self._http_client.stream("POST", url, json=payload, headers=headers) as response:
                if response.status_code != 200:
                    text = await response.aread()
                    raise LLMError(f"OpenAI error: {response.status_code} - {text.decode()}")

                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data_str = line[6:]
                    if data_str.strip() == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                        token = data["choices"][0].get("delta", {}).get("content", "")
                        if token:
                            yield token
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue
        except httpx.HTTPError as e:
            raise LLMError(f"OpenAI connection error: {e}")

    async def _complete_openai(self, messages: list[dict[str, str]], max_tokens: int | None = None) -> str:
        """Non-streaming OpenAI call."""
        if not self._http_client:
            raise LLMError("HTTP client not initialized")
        if not settings.openai_api_key:
            raise LLMError("OpenAI API key not configured")

        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {settings.openai_api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": settings.openai_model,
            "messages": messages,
            "temperature": settings.llm_temperature,
            "max_tokens": max_tokens or settings.llm_max_tokens,
        }

        try:
            response = await self._http_client.post(url, json=payload, headers=headers)
            if response.status_code != 200:
                raise LLMError(f"OpenAI error: {response.status_code} - {response.text}")
            data = response.json()
            usage = data.get("usage", {})
            self._total_tokens_used += usage.get("total_tokens", 0)
            return data["choices"][0]["message"]["content"]
        except httpx.HTTPError as e:
            raise LLMError(f"OpenAI connection error: {e}")

    async def health_check(self) -> dict:
        """Check LLM provider connectivity."""
        try:
            if self._provider == "ollama" and self._http_client:
                resp = await self._http_client.get(f"{settings.ollama_base_url}/api/tags")
                models = [m["name"] for m in resp.json().get("models", [])]
                return {
                    "status": "healthy",
                    "provider": "ollama",
                    "available_models": models,
                    "active_model": settings.ollama_model,
                }
            elif self._provider == "openai":
                return {
                    "status": "configured" if settings.openai_api_key else "no_api_key",
                    "provider": "openai",
                    "active_model": settings.openai_model,
                }
        except Exception as e:
            return {"status": "error", "provider": self._provider, "error": str(e)}
        return {"status": "unknown", "provider": self._provider}


# Singleton instance
llm_service = LLMService()
