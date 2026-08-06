"""Unit tests for `strata_forge.tracing.decorator`."""

from __future__ import annotations

import contextlib
import sys
import types
from typing import Any
from unittest.mock import MagicMock

import pytest

from strata_forge.core.ids import correlation_id_var
from strata_forge.tracing.decorator import traced


def _install_fake_langfuse(
    monkeypatch: pytest.MonkeyPatch,
    *,
    trace_id: str = "trace-fake",
) -> MagicMock:
    """Inject a fake `langfuse` module and return the Langfuse constructor mock.

    The constructor returns a client mock whose `start_observation()`
    method returns an object with `.id`, `.trace_id`, `.update`, and
    `.end`. A back-compat ``client.trace`` alias points at the same
    underlying mock so older test assertions continue to work after
    the v4 SDK migration.
    """
    monkeypatch.setenv("LANGFUSE_HOST", "http://test")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")

    fake_module = types.ModuleType("langfuse")
    client_mock = MagicMock(name="lf-client")
    trace_mock = MagicMock(name="lf-trace")
    trace_mock.id = trace_id
    trace_mock.trace_id = trace_id
    client_mock.start_observation.return_value = trace_mock
    # Back-compat alias so existing tests reading `.trace` keep working.
    client_mock.trace = client_mock.start_observation

    fake_module.Langfuse = MagicMock(return_value=client_mock)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "langfuse", fake_module)
    return client_mock


def _no_langfuse(monkeypatch: pytest.MonkeyPatch) -> None:
    """Configure env so `get_client()` returns None."""
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)


# ---------------------------------------------------------------------------
# Bare-form decorator
# ---------------------------------------------------------------------------


class TestBareForm:
    def test_wraps_sync_function(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _no_langfuse(monkeypatch)

        @traced
        def add(a: int, b: int) -> int:
            return a + b

        assert add(2, 3) == 5

    async def test_wraps_async_function(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _no_langfuse(monkeypatch)

        @traced
        async def fetch(x: int) -> int:
            return x * 2

        assert await fetch(21) == 42

    def test_preserves_function_metadata(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _no_langfuse(monkeypatch)

        @traced
        def my_function() -> None:
            """The original docstring."""

        assert my_function.__name__ == "my_function"
        assert my_function.__doc__ == "The original docstring."


# ---------------------------------------------------------------------------
# Parameterized form
# ---------------------------------------------------------------------------


class TestParameterizedForm:
    def test_custom_name(self, monkeypatch: pytest.MonkeyPatch) -> None:
        client = _install_fake_langfuse(monkeypatch)

        @traced(name="custom-trace-name")
        def my_function() -> int:
            return 1

        my_function()
        client.trace.assert_called_once()
        kwargs = client.trace.call_args.kwargs
        assert kwargs["name"] == "custom-trace-name"

    def test_default_name_is_function_name(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        client = _install_fake_langfuse(monkeypatch)

        @traced
        def my_specific_function() -> int:
            return 1

        my_specific_function()
        kwargs = client.trace.call_args.kwargs
        assert kwargs["name"] == "my_specific_function"

    def test_tags_forwarded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        client = _install_fake_langfuse(monkeypatch)

        @traced(tags=["alpha", "experiment-3"])
        def my_function() -> int:
            return 1

        my_function()
        kwargs = client.trace.call_args.kwargs
        # v4 SDK: tags ride in `metadata={'tags': [...]}` rather than a
        # dedicated kwarg, because start_observation has no tags parameter.
        assert kwargs["metadata"] == {"tags": ["alpha", "experiment-3"]}

    def test_empty_tags_when_omitted(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        client = _install_fake_langfuse(monkeypatch)

        @traced
        def my_function() -> int:
            return 1

        my_function()
        kwargs = client.trace.call_args.kwargs
        # When no tags are supplied, the decorator drops the metadata
        # kwarg entirely (sends `metadata=None`) instead of an empty
        # tag list, since start_observation accepts `metadata: Any | None`.
        assert kwargs.get("metadata") is None


# ---------------------------------------------------------------------------
# Correlation ID propagation
# ---------------------------------------------------------------------------


class TestCorrelationId:
    def test_correlation_id_set_during_call(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _install_fake_langfuse(monkeypatch, trace_id="trace-abc123")
        captured: dict[str, Any] = {}

        @traced
        def my_function() -> None:
            captured["cid"] = correlation_id_var.get()

        my_function()
        assert captured["cid"] == "trace-abc123"

    def test_correlation_id_restored_after_call(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _install_fake_langfuse(monkeypatch)
        outer_token = correlation_id_var.set("outer-cid")
        try:

            @traced
            def my_function() -> None:
                pass

            my_function()
            assert correlation_id_var.get() == "outer-cid"
        finally:
            correlation_id_var.reset(outer_token)

    async def test_correlation_id_set_during_async_call(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _install_fake_langfuse(monkeypatch, trace_id="async-trace")
        captured: dict[str, Any] = {}

        @traced
        async def my_async_function() -> None:
            captured["cid"] = correlation_id_var.get()

        await my_async_function()
        assert captured["cid"] == "async-trace"

    def test_no_correlation_id_change_when_langfuse_unconfigured(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _no_langfuse(monkeypatch)
        outer_token = correlation_id_var.set("outer-cid")
        try:

            @traced
            def my_function() -> str | None:
                return correlation_id_var.get()

            assert my_function() == "outer-cid"
        finally:
            correlation_id_var.reset(outer_token)


# ---------------------------------------------------------------------------
# Trace update on success / failure
# ---------------------------------------------------------------------------


class TestTraceLifecycle:
    def test_trace_updated_with_ok_on_success(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        client = _install_fake_langfuse(monkeypatch)
        trace = client.trace.return_value

        @traced
        def my_function() -> int:
            return 42

        my_function()
        trace.update.assert_called_once()
        kwargs = trace.update.call_args.kwargs
        assert kwargs["output"] == {"status": "ok"}

    def test_trace_updated_with_error_on_exception(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        client = _install_fake_langfuse(monkeypatch)
        trace = client.trace.return_value

        @traced
        def my_function() -> int:
            msg = "boom"
            raise ValueError(msg)

        with contextlib.suppress(ValueError):
            my_function()

        trace.update.assert_called_once()
        kwargs = trace.update.call_args.kwargs
        assert kwargs["level"] == "ERROR"
        assert "boom" in kwargs["output"]["error"]

    def test_exception_propagates(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_langfuse(monkeypatch)

        @traced
        def my_function() -> None:
            msg = "specific message"
            raise RuntimeError(msg)

        with pytest.raises(RuntimeError, match="specific message"):
            my_function()

    async def test_async_exception_propagates(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _install_fake_langfuse(monkeypatch)

        @traced
        async def my_function() -> None:
            msg = "async error"
            raise ValueError(msg)

        with pytest.raises(ValueError, match="async error"):
            await my_function()


# ---------------------------------------------------------------------------
# Tracing-failure resilience
# ---------------------------------------------------------------------------


class TestResilience:
    def test_trace_creation_failure_does_not_break_function(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        client = _install_fake_langfuse(monkeypatch)
        # client.trace(...) throws — the wrapped function must still run.
        client.trace.side_effect = RuntimeError("langfuse server down")

        @traced
        def my_function(x: int) -> int:
            return x + 1

        assert my_function(5) == 6

    async def test_async_trace_creation_failure_does_not_break_function(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Same resilience contract for async functions.
        client = _install_fake_langfuse(monkeypatch)
        client.trace.side_effect = RuntimeError("langfuse server down")

        @traced
        async def my_function(x: int) -> int:
            return x + 1

        assert await my_function(5) == 6

    def test_trace_update_failure_does_not_break_function(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        client = _install_fake_langfuse(monkeypatch)
        trace = client.trace.return_value
        trace.update.side_effect = RuntimeError("langfuse update failed")

        @traced
        def my_function() -> int:
            return 42

        # The function returns successfully even though the trace.update
        # call failed.
        assert my_function() == 42

    def test_trace_update_failure_on_error_path_does_not_mask_exception(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        client = _install_fake_langfuse(monkeypatch)
        trace = client.trace.return_value
        trace.update.side_effect = RuntimeError("langfuse failed")

        @traced
        def my_function() -> None:
            msg = "real error"
            raise ValueError(msg)

        # The original ValueError surfaces, NOT the tracing failure.
        with pytest.raises(ValueError, match="real error"):
            my_function()


# ---------------------------------------------------------------------------
# Skip behavior when Langfuse isn't configured
# ---------------------------------------------------------------------------


class TestUnconfiguredFallthrough:
    def test_function_runs_without_tracing(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _no_langfuse(monkeypatch)
        called = {"count": 0}

        @traced
        def my_function() -> int:
            called["count"] += 1
            return 99

        assert my_function() == 99
        assert called["count"] == 1

    async def test_async_function_runs_without_tracing(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _no_langfuse(monkeypatch)

        @traced
        async def my_function() -> int:
            return 99

        assert await my_function() == 99
