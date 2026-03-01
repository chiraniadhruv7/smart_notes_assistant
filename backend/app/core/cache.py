"""
Redis-based async caching layer.
Provides TTL-based caching for embeddings and query results
to reduce redundant computation and external API calls.
"""

import json
import hashlib
from typing import Any

import redis.asyncio as redis

from app.config import settings
from app.utils.logging import get_logger

logger = get_logger(__name__)


class CacheService:
    """Async Redis cache with JSON serialization and hash-based keys."""

    def __init__(self) -> None:
        self._client: redis.Redis | None = None
        self._connected: bool = False

    async def connect(self) -> None:
        """Initialize Redis connection pool."""
        try:
            self._client = redis.from_url(
                settings.redis_url,
                decode_responses=True,
                socket_connect_timeout=5,
            )
            await self._client.ping()
            self._connected = True
            logger.info("Redis cache connected")
        except Exception as e:
            logger.warning(f"Redis unavailable, caching disabled: {e}")
            self._connected = False

    async def disconnect(self) -> None:
        """Close Redis connection."""
        if self._client:
            await self._client.close()
            self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    @staticmethod
    def _make_key(prefix: str, data: str) -> str:
        """Generate a deterministic cache key from prefix + data hash."""
        h = hashlib.sha256(data.encode()).hexdigest()[:16]
        return f"rag:{prefix}:{h}"

    async def get(self, prefix: str, key_data: str) -> Any | None:
        """Retrieve a cached value. Returns None on miss or if cache is down."""
        if not self._connected or not self._client:
            return None
        try:
            key = self._make_key(prefix, key_data)
            raw = await self._client.get(key)
            if raw:
                return json.loads(raw)
            return None
        except Exception as e:
            logger.warning(f"Cache read error: {e}")
            return None

    async def set(self, prefix: str, key_data: str, value: Any, ttl: int | None = None) -> None:
        """Store a value in cache with optional TTL override."""
        if not self._connected or not self._client:
            return
        try:
            key = self._make_key(prefix, key_data)
            serialized = json.dumps(value, default=str)
            await self._client.set(key, serialized, ex=ttl or settings.cache_ttl)
        except Exception as e:
            logger.warning(f"Cache write error: {e}")

    async def delete(self, prefix: str, key_data: str) -> None:
        """Remove a specific cache entry."""
        if not self._connected or not self._client:
            return
        try:
            key = self._make_key(prefix, key_data)
            await self._client.delete(key)
        except Exception as e:
            logger.warning(f"Cache delete error: {e}")

    async def flush_prefix(self, prefix: str) -> int:
        """Delete all keys matching a prefix. Returns count deleted."""
        if not self._connected or not self._client:
            return 0
        try:
            pattern = f"rag:{prefix}:*"
            keys = []
            async for key in self._client.scan_iter(match=pattern, count=100):
                keys.append(key)
            if keys:
                await self._client.delete(*keys)
            return len(keys)
        except Exception as e:
            logger.warning(f"Cache flush error: {e}")
            return 0

    async def health_check(self) -> dict[str, Any]:
        """Return cache health status."""
        if not self._connected or not self._client:
            return {"status": "disconnected", "latency_ms": -1}
        try:
            import time
            start = time.perf_counter()
            await self._client.ping()
            latency = (time.perf_counter() - start) * 1000
            info = await self._client.info("memory")
            return {
                "status": "healthy",
                "latency_ms": round(latency, 2),
                "used_memory_mb": round(info.get("used_memory", 0) / 1024 / 1024, 2),
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}


# Singleton instance
cache_service = CacheService()
