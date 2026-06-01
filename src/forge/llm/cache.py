"""Provider-agnostic response cache.

Two backends:

- :class:`InMemoryCache` — bounded LRU. Default; no extras required.
- :class:`RedisCache` — Redis-backed; requires the ``[redis]`` extra
  (``redis>=5.2``). The ``redis`` package is imported lazily inside
  :class:`RedisCache`'s constructor so the module imports cleanly without
  the extra installed.

The cache key (:func:`cache_key`) is computed from the *logical* model
name plus the canonical request — provider routing is intentionally
**not** part of the key. A hit cached when Anthropic served the call is
just as valid when the next call is pinned to Bedrock; including the
provider would defeat the cache's purpose during provider-level
failover.

Streaming responses are never cached: chunks have provider-specific
shapes and no useful "final" form before the stream terminates. The
:class:`LLMClient` streaming path bypasses this module entirely.

On a cache hit, callers should set ``cache_hit=True`` and
``cost_usd=0.0`` on the returned :class:`LLMResponse` so spend reports
reflect the savings; the cache itself does not mutate the stored value.
"""

from __future__ import annotations

import pickle
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import TYPE_CHECKING, Any

from forge.core.errors import CacheError
from forge.core.repro import content_hash

if TYPE_CHECKING:
    from collections.abc import Mapping

    from forge.llm.responses import LLMResponse

__all__ = [
    "CacheBackend",
    "InMemoryCache",
    "RedisCache",
    "cache_key",
]


def cache_key(
    *,
    model: str,
    messages: list[dict[str, Any]],
    temperature: float | None = None,
    max_tokens: int | None = None,
    top_p: float | None = None,
    response_format: dict[str, Any] | None = None,
    tools: list[dict[str, Any]] | None = None,
    provider_extras: Mapping[str, Any] | None = None,
) -> str:
    """Compute a canonical, provider-agnostic SHA-256 cache key.

    The key depends on the *logical* model name and every input that
    affects generation — sampling parameters, the structured-output
    schema (if any), the tool list (if any), and any
    ``provider_extras`` passthrough. The actual provider serving the
    call is intentionally excluded so the cache remains useful across
    provider-level failover.

    Args:
        model: Logical (registry) model name. Do NOT pass a
            provider-specific id like ``anthropic.claude-opus-4-7-…``.
        messages: Conversation history in wire format
            (``[{"role": ..., "content": ...}, ...]``).
        temperature: Sampling temperature, if specified.
        max_tokens: Output cap, if specified.
        top_p: Top-p (nucleus) sampling cutoff, if specified.
        response_format: OpenAI-style ``response_format`` payload for
            structured output, if any.
        tools: List of tool schemas the model is allowed to call, in the
            order they're declared. Order is part of the key — switching
            tool order is treated as a cache-busting change.
        provider_extras: Verbatim passthrough kwargs forwarded to the
            provider via :class:`LLMClient`'s escape hatch. Anything in
            here that affects the response (reasoning budgets,
            thinking-mode toggles, custom seeds, …) is part of the key.

    Returns:
        64-character lowercase hex SHA-256 digest.
    """
    canonical: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "sampling": {
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
        },
        "response_format": response_format,
        "tools": tools,
        "provider_extras": dict(provider_extras) if provider_extras is not None else None,
    }
    return content_hash(canonical)


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------


class CacheBackend(ABC):
    """Abstract cache backend.

    Implementations must be safe to call concurrently from the same
    event loop. Cross-process safety is backend-specific
    (:class:`RedisCache` is safe by virtue of Redis; :class:`InMemoryCache`
    is not).
    """

    @abstractmethod
    async def get(self, key: str) -> LLMResponse | None:
        """Return the cached response for ``key``, or ``None`` on miss."""

    @abstractmethod
    async def set(self, key: str, response: LLMResponse) -> None:
        """Cache ``response`` under ``key`` (overwriting any prior entry)."""

    @abstractmethod
    async def clear(self) -> None:
        """Drop every entry from this backend."""


class InMemoryCache(CacheBackend):
    """In-process LRU cache.

    Bounded by ``max_size``; least-recently-used entries are evicted
    once the cap is reached. Suitable for single-process workloads or as
    a hot tier in front of :class:`RedisCache`.
    """

    def __init__(self, *, max_size: int = 1024) -> None:
        if max_size <= 0:
            msg = f"max_size must be positive, got {max_size}"
            raise ValueError(msg)
        self._max_size = max_size
        self._store: OrderedDict[str, LLMResponse] = OrderedDict()

    async def get(self, key: str) -> LLMResponse | None:
        if key not in self._store:
            return None
        self._store.move_to_end(key)
        return self._store[key]

    async def set(self, key: str, response: LLMResponse) -> None:
        if key in self._store:
            self._store.move_to_end(key)
        self._store[key] = response
        while len(self._store) > self._max_size:
            self._store.popitem(last=False)

    async def clear(self) -> None:
        self._store.clear()

    def __len__(self) -> int:
        return len(self._store)


class RedisCache(CacheBackend):
    """Redis-backed cache (requires the ``[redis]`` extra).

    Values are serialized with :mod:`pickle` since :class:`LLMResponse`
    embeds a non-Pydantic :class:`ModelRoute` dataclass. Treat the
    Redis instance as trusted — never deserialize from an untrusted
    source. Keys are prefixed (default ``"forge:llm:"``) so multiple
    independent processes can share a Redis without colliding.
    """

    def __init__(
        self,
        *,
        url: str = "redis://localhost:6379/0",
        prefix: str = "forge:llm:",
        ttl_seconds: int | None = None,
    ) -> None:
        try:
            from redis import (  # pyright: ignore[reportMissingImports]
                asyncio as redis_asyncio,  # pyright: ignore[reportUnknownVariableType]
            )
        except ImportError as exc:
            msg = (
                "RedisCache requires the `[redis]` extra. "
                "Install with `pip install strata-forge[redis]`."
            )
            raise ImportError(msg) from exc
        # Typed as Any since the redis-py async client has dynamic command
        # methods we can't statically type-check without stubs in the strict
        # path. Calls are validated at runtime; backend failures funnel through
        # CacheError below.
        self._client: Any = redis_asyncio.from_url(url)  # pyright: ignore[reportUnknownMemberType]
        self._prefix = prefix
        self._ttl_seconds = ttl_seconds

    def _redis_key(self, key: str) -> str:
        return f"{self._prefix}{key}"

    async def get(self, key: str) -> LLMResponse | None:
        try:
            raw = await self._client.get(self._redis_key(key))
        except Exception as exc:
            msg = f"Redis GET failed: {exc}"
            raise CacheError(msg, backend="redis") from exc
        if raw is None:
            return None
        try:
            return pickle.loads(raw)  # noqa: S301 — trusted cache, see module docstring
        except pickle.UnpicklingError, EOFError, AttributeError:
            # Corrupt entry — treat as miss so the caller refills.
            return None

    async def set(self, key: str, response: LLMResponse) -> None:
        data = pickle.dumps(response)
        try:
            if self._ttl_seconds is not None:
                await self._client.setex(self._redis_key(key), self._ttl_seconds, data)
            else:
                await self._client.set(self._redis_key(key), data)
        except Exception as exc:
            msg = f"Redis SET failed: {exc}"
            raise CacheError(msg, backend="redis") from exc

    async def clear(self) -> None:
        try:
            keys = await self._client.keys(f"{self._prefix}*")
            if keys:
                await self._client.delete(*keys)
        except Exception as exc:
            msg = f"Redis CLEAR failed: {exc}"
            raise CacheError(msg, backend="redis") from exc
