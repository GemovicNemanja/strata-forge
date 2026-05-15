"""Unit tests for `forge.llm.cache`."""

from __future__ import annotations

import pickle
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from forge.core.errors import CacheError
from forge.llm.cache import CacheBackend, InMemoryCache, RedisCache, cache_key
from forge.llm.responses import LLMResponse, Usage
from forge.llm.routing import ModelRoute

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_response(
    text: str = "ok",
    *,
    model: str = "claude-opus-4-7",
    provider: str = "anthropic",
    provider_model_id: str = "claude-opus-4-7",
    cache_hit: bool = False,
    cost_usd: float = 0.005,
) -> LLMResponse:
    return LLMResponse(
        text=text,
        finish_reason="stop",
        usage=Usage(input_tokens=10, output_tokens=20),
        cost_usd=cost_usd,
        route=ModelRoute(
            model=model,
            provider=provider,  # type: ignore[arg-type]
            provider_model_id=provider_model_id,
        ),
        cache_hit=cache_hit,
    )


# ---------------------------------------------------------------------------
# cache_key
# ---------------------------------------------------------------------------


class TestCacheKey:
    def test_returns_64_char_hex(self) -> None:
        key = cache_key(model="m", messages=[{"role": "user", "content": "hi"}])
        assert len(key) == 64
        assert all(c in "0123456789abcdef" for c in key)

    def test_deterministic(self) -> None:
        args: dict[str, Any] = {
            "model": "claude-opus-4-7",
            "messages": [{"role": "user", "content": "hello"}],
            "temperature": 0.7,
            "max_tokens": 200,
        }
        assert cache_key(**args) == cache_key(**args)

    def test_provider_agnostic(self) -> None:
        # Note: `cache_key` doesn't accept a provider — that's the whole point.
        # The fact that there's no `provider` parameter to vary IS the test.
        key = cache_key(model="claude-opus-4-7", messages=[{"role": "user", "content": "hi"}])
        # Recomputed with same logical model — same key, regardless of which
        # provider would have actually served the call.
        assert key == cache_key(
            model="claude-opus-4-7", messages=[{"role": "user", "content": "hi"}]
        )

    def test_different_models_differ(self) -> None:
        a = cache_key(model="claude-opus-4-7", messages=[{"role": "user", "content": "hi"}])
        b = cache_key(model="gpt-5.5", messages=[{"role": "user", "content": "hi"}])
        assert a != b

    def test_different_messages_differ(self) -> None:
        a = cache_key(model="m", messages=[{"role": "user", "content": "hi"}])
        b = cache_key(model="m", messages=[{"role": "user", "content": "bye"}])
        assert a != b

    def test_message_order_matters(self) -> None:
        a = cache_key(
            model="m",
            messages=[
                {"role": "user", "content": "a"},
                {"role": "user", "content": "b"},
            ],
        )
        b = cache_key(
            model="m",
            messages=[
                {"role": "user", "content": "b"},
                {"role": "user", "content": "a"},
            ],
        )
        assert a != b

    def test_temperature_affects_key(self) -> None:
        base: dict[str, Any] = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
        a = cache_key(**base, temperature=0.0)
        b = cache_key(**base, temperature=0.7)
        assert a != b

    def test_max_tokens_affects_key(self) -> None:
        base: dict[str, Any] = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
        a = cache_key(**base, max_tokens=100)
        b = cache_key(**base, max_tokens=200)
        assert a != b

    def test_top_p_affects_key(self) -> None:
        base: dict[str, Any] = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
        a = cache_key(**base, top_p=0.9)
        b = cache_key(**base, top_p=0.95)
        assert a != b

    def test_response_format_affects_key(self) -> None:
        base: dict[str, Any] = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
        a = cache_key(**base)
        b = cache_key(**base, response_format={"type": "json_schema"})
        assert a != b

    def test_tools_affect_key(self) -> None:
        base: dict[str, Any] = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
        a = cache_key(**base)
        b = cache_key(**base, tools=[{"name": "get_weather"}])
        assert a != b

    def test_tool_order_matters(self) -> None:
        # Order matters because the model may pick the first matching tool.
        base: dict[str, Any] = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
        a = cache_key(**base, tools=[{"name": "a"}, {"name": "b"}])
        b = cache_key(**base, tools=[{"name": "b"}, {"name": "a"}])
        assert a != b

    def test_provider_extras_affect_key(self) -> None:
        base: dict[str, Any] = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
        a = cache_key(**base)
        b = cache_key(**base, provider_extras={"thinking": {"budget_tokens": 8000}})
        assert a != b

    def test_provider_extras_key_order_irrelevant(self) -> None:
        # `content_hash` sorts keys, so dict insertion order doesn't matter.
        base: dict[str, Any] = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
        a = cache_key(**base, provider_extras={"a": 1, "b": 2})
        b = cache_key(**base, provider_extras={"b": 2, "a": 1})
        assert a == b

    def test_no_params_baseline(self) -> None:
        # Smoke: a minimal call still produces a usable key.
        key = cache_key(model="m", messages=[])
        assert len(key) == 64


class TestCacheKeyProperty:
    @settings(max_examples=40, deadline=None)
    @given(
        model=st.text(min_size=1, max_size=40),
        content=st.text(min_size=0, max_size=200),
    )
    def test_deterministic_property(self, model: str, content: str) -> None:
        messages = [{"role": "user", "content": content}]
        assert cache_key(model=model, messages=messages) == cache_key(
            model=model, messages=messages
        )


# ---------------------------------------------------------------------------
# InMemoryCache
# ---------------------------------------------------------------------------


class TestInMemoryCacheBasics:
    def test_max_size_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            InMemoryCache(max_size=0)

    def test_negative_max_size_rejected(self) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            InMemoryCache(max_size=-1)

    async def test_miss_returns_none(self) -> None:
        cache = InMemoryCache()
        assert await cache.get("nope") is None

    async def test_set_then_get(self) -> None:
        cache = InMemoryCache()
        resp = _make_response("hello")
        await cache.set("k", resp)
        result = await cache.get("k")
        assert result is resp
        assert result is not None
        assert result.text == "hello"

    async def test_overwrite(self) -> None:
        cache = InMemoryCache()
        await cache.set("k", _make_response("first"))
        await cache.set("k", _make_response("second"))
        result = await cache.get("k")
        assert result is not None
        assert result.text == "second"
        assert len(cache) == 1

    async def test_clear(self) -> None:
        cache = InMemoryCache()
        await cache.set("a", _make_response())
        await cache.set("b", _make_response())
        assert len(cache) == 2
        await cache.clear()
        assert len(cache) == 0
        assert await cache.get("a") is None


class TestInMemoryCacheLRU:
    async def test_eviction_drops_oldest(self) -> None:
        cache = InMemoryCache(max_size=2)
        await cache.set("a", _make_response("A"))
        await cache.set("b", _make_response("B"))
        await cache.set("c", _make_response("C"))  # should evict "a"
        assert await cache.get("a") is None
        assert (await cache.get("b")) is not None
        assert (await cache.get("c")) is not None
        assert len(cache) == 2

    async def test_get_refreshes_recency(self) -> None:
        cache = InMemoryCache(max_size=2)
        await cache.set("a", _make_response("A"))
        await cache.set("b", _make_response("B"))
        # Touch "a" so it becomes most-recently-used.
        await cache.get("a")
        # Now adding "c" should evict "b", not "a".
        await cache.set("c", _make_response("C"))
        assert (await cache.get("a")) is not None
        assert await cache.get("b") is None
        assert (await cache.get("c")) is not None

    async def test_set_existing_refreshes_recency(self) -> None:
        cache = InMemoryCache(max_size=2)
        await cache.set("a", _make_response("A1"))
        await cache.set("b", _make_response("B"))
        # Re-set "a" — should refresh, not duplicate.
        await cache.set("a", _make_response("A2"))
        await cache.set("c", _make_response("C"))  # should evict "b"
        assert (await cache.get("a")) is not None
        assert await cache.get("b") is None
        assert (await cache.get("c")) is not None
        assert len(cache) == 2

    async def test_eviction_under_load(self) -> None:
        cache = InMemoryCache(max_size=3)
        for i in range(10):
            await cache.set(f"k{i}", _make_response(f"v{i}"))
        assert len(cache) == 3
        # Only the last 3 keys should survive.
        for i in range(7):
            assert await cache.get(f"k{i}") is None
        for i in range(7, 10):
            assert (await cache.get(f"k{i}")) is not None

    async def test_size_one(self) -> None:
        cache = InMemoryCache(max_size=1)
        await cache.set("a", _make_response("A"))
        await cache.set("b", _make_response("B"))
        assert await cache.get("a") is None
        assert (await cache.get("b")) is not None


class TestInMemoryCacheProviderAgnostic:
    """A cache hit on a logical model is valid regardless of which provider
    originally served the response. This is enforced by `cache_key` not
    taking a provider — these tests just confirm the end-to-end behavior."""

    async def test_hit_across_providers_via_shared_key(self) -> None:
        cache = InMemoryCache()
        key = cache_key(
            model="claude-opus-4-7",
            messages=[{"role": "user", "content": "hi"}],
        )
        # Originally served by anthropic.
        stored = _make_response("served by anthropic", provider="anthropic")
        await cache.set(key, stored)

        # Re-look-up using the same logical key — would-be call to bedrock
        # uses the same cache key, and the cached anthropic response is
        # returned. The caller is responsible for setting `cache_hit=True`
        # on the way out; the cache itself returns the value verbatim.
        result = await cache.get(key)
        assert result is stored
        assert result is not None
        assert result.route.provider == "anthropic"


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class TestCacheBackendBase:
    def test_cannot_instantiate_abstract(self) -> None:
        with pytest.raises(TypeError):
            CacheBackend()  # pyright: ignore[reportAbstractUsage]


# ---------------------------------------------------------------------------
# RedisCache
# ---------------------------------------------------------------------------


@pytest.fixture
def redis_module_mock(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Inject a fake `redis.asyncio` module so RedisCache can be constructed
    without the `[redis]` extra being installed.

    Returns the `from_url` factory mock; tests set `return_value` on it to
    inject a per-test mock client.
    """
    import sys
    import types

    fake_redis_pkg = types.ModuleType("redis")
    fake_asyncio_mod = types.ModuleType("redis.asyncio")
    fake_from_url = MagicMock()
    fake_asyncio_mod.from_url = fake_from_url  # type: ignore[attr-defined]
    fake_redis_pkg.asyncio = fake_asyncio_mod  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "redis", fake_redis_pkg)
    monkeypatch.setitem(sys.modules, "redis.asyncio", fake_asyncio_mod)
    return fake_from_url


def _make_mock_client() -> AsyncMock:
    client = AsyncMock()
    # `from_url` returns a synchronous client object; we use AsyncMock for both
    # the client and its awaitable methods.
    return client


class TestRedisCacheImportGate:
    def test_import_error_when_redis_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Ensure the `redis` package is not importable.
        import sys

        monkeypatch.setitem(sys.modules, "redis", None)
        with pytest.raises(ImportError, match="RedisCache requires"):
            RedisCache()


class TestRedisCacheRoundtrip:
    async def test_set_then_get(self, redis_module_mock: MagicMock) -> None:
        mock_client = _make_mock_client()
        redis_module_mock.return_value = mock_client
        cache = RedisCache(url="redis://test/0")

        resp = _make_response("hello")
        await cache.set("abc", resp)
        # Capture the raw bytes that were stored.
        assert mock_client.set.await_count == 1
        stored_key, stored_value = mock_client.set.await_args.args
        assert stored_key == "forge:llm:abc"

        # Now mimic a GET returning the stored bytes.
        mock_client.get.return_value = stored_value
        result = await cache.get("abc")
        assert result is not None
        assert result.text == "hello"
        assert result.usage.input_tokens == 10

    async def test_miss_returns_none(self, redis_module_mock: MagicMock) -> None:
        mock_client = _make_mock_client()
        mock_client.get.return_value = None
        redis_module_mock.return_value = mock_client
        cache = RedisCache()

        assert await cache.get("nope") is None

    async def test_corrupt_entry_returns_none(self, redis_module_mock: MagicMock) -> None:
        mock_client = _make_mock_client()
        mock_client.get.return_value = b"not valid pickle"
        redis_module_mock.return_value = mock_client
        cache = RedisCache()

        # Corrupt bytes → treat as miss (so caller refills).
        assert await cache.get("k") is None

    async def test_ttl_used_when_set(self, redis_module_mock: MagicMock) -> None:
        mock_client = _make_mock_client()
        redis_module_mock.return_value = mock_client
        cache = RedisCache(ttl_seconds=60)
        await cache.set("k", _make_response())
        assert mock_client.setex.await_count == 1
        key, ttl, _data = mock_client.setex.await_args.args
        assert key == "forge:llm:k"
        assert ttl == 60
        # And `set` should NOT have been called.
        assert mock_client.set.await_count == 0

    async def test_custom_prefix(self, redis_module_mock: MagicMock) -> None:
        mock_client = _make_mock_client()
        redis_module_mock.return_value = mock_client
        cache = RedisCache(prefix="myapp:")
        await cache.set("k", _make_response())
        stored_key = mock_client.set.await_args.args[0]
        assert stored_key == "myapp:k"

    async def test_get_failure_raises_cache_error(self, redis_module_mock: MagicMock) -> None:
        mock_client = _make_mock_client()
        mock_client.get.side_effect = RuntimeError("connection refused")
        redis_module_mock.return_value = mock_client
        cache = RedisCache()

        with pytest.raises(CacheError, match="GET failed"):
            await cache.get("k")

    async def test_set_failure_raises_cache_error(self, redis_module_mock: MagicMock) -> None:
        mock_client = _make_mock_client()
        mock_client.set.side_effect = RuntimeError("connection refused")
        redis_module_mock.return_value = mock_client
        cache = RedisCache()

        with pytest.raises(CacheError, match="SET failed"):
            await cache.set("k", _make_response())

    async def test_clear_iterates_keys_then_deletes(self, redis_module_mock: MagicMock) -> None:
        mock_client = _make_mock_client()
        mock_client.keys.return_value = [b"forge:llm:a", b"forge:llm:b"]
        redis_module_mock.return_value = mock_client
        cache = RedisCache()

        await cache.clear()
        mock_client.keys.assert_awaited_once_with("forge:llm:*")
        mock_client.delete.assert_awaited_once_with(b"forge:llm:a", b"forge:llm:b")

    async def test_clear_noop_when_empty(self, redis_module_mock: MagicMock) -> None:
        mock_client = _make_mock_client()
        mock_client.keys.return_value = []
        redis_module_mock.return_value = mock_client
        cache = RedisCache()

        await cache.clear()
        mock_client.delete.assert_not_called()

    async def test_clear_failure_raises_cache_error(self, redis_module_mock: MagicMock) -> None:
        mock_client = _make_mock_client()
        mock_client.keys.side_effect = RuntimeError("connection refused")
        redis_module_mock.return_value = mock_client
        cache = RedisCache()

        with pytest.raises(CacheError, match="CLEAR failed"):
            await cache.clear()


class TestRedisPickleFormat:
    """Confirm that `set` stores a pickled LLMResponse so external readers
    know what to expect (and that pickling is round-trippable)."""

    async def test_set_stores_pickled_response(self, redis_module_mock: MagicMock) -> None:
        mock_client = _make_mock_client()
        redis_module_mock.return_value = mock_client
        cache = RedisCache()
        resp = _make_response("pickled")
        await cache.set("k", resp)
        stored_value = mock_client.set.await_args.args[1]
        assert isinstance(stored_value, bytes)
        decoded = pickle.loads(stored_value)  # noqa: S301 — test only
        assert decoded.text == "pickled"
