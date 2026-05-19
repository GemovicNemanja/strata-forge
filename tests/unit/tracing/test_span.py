"""Unit tests for `forge.tracing.span`."""

from __future__ import annotations

import contextlib
import sys
import types
from typing import Any
from unittest.mock import MagicMock

import pytest

from forge.core.ids import correlation_id_var
from forge.tracing.span import traced_span


def _install_fake_langfuse(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Inject a fake `langfuse` module; return the client mock for inspection."""
    monkeypatch.setenv("LANGFUSE_HOST", "http://test")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")

    fake_module = types.ModuleType("langfuse")
    client_mock = MagicMock(name="lf-client")
    span_mock = MagicMock(name="lf-span")
    span_mock.id = "span-fake"
    client_mock.start_observation.return_value = span_mock
    # v4 SDK replaced `client.span(...)` with `client.start_observation(
    # as_type="span")`. Alias the old name to the new mock so existing
    # `client.span.call_args` assertions keep working.
    client_mock.span = client_mock.start_observation

    fake_module.Langfuse = MagicMock(return_value=client_mock)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "langfuse", fake_module)
    return client_mock


def _no_langfuse(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)


# ---------------------------------------------------------------------------
# Happy paths
# ---------------------------------------------------------------------------


class TestSpanCreation:
    async def test_yields_span_object_when_configured(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        client = _install_fake_langfuse(monkeypatch)

        async with traced_span("my-span") as span:
            assert span is client.span.return_value
        client.span.assert_called_once()

    async def test_yields_none_when_unconfigured(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _no_langfuse(monkeypatch)
        async with traced_span("my-span") as span:
            assert span is None

    async def test_name_forwarded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        client = _install_fake_langfuse(monkeypatch)
        async with traced_span("my-specific-name"):
            pass
        assert client.span.call_args.kwargs["name"] == "my-specific-name"

    async def test_metadata_forwarded(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        client = _install_fake_langfuse(monkeypatch)
        async with traced_span("x", metadata={"experiment": "v1"}):
            pass
        assert client.span.call_args.kwargs["metadata"] == {"experiment": "v1"}

    async def test_empty_metadata_when_omitted(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        client = _install_fake_langfuse(monkeypatch)
        async with traced_span("x"):
            pass
        assert client.span.call_args.kwargs["metadata"] == {}


# ---------------------------------------------------------------------------
# Trace ID linkage via correlation_id_var
# ---------------------------------------------------------------------------


class TestTraceLinkage:
    async def test_span_linked_to_active_trace_id(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        client = _install_fake_langfuse(monkeypatch)
        token = correlation_id_var.set("trace-outer")
        try:
            async with traced_span("inner"):
                pass
            # v4 SDK: trace linkage rides in `trace_context={"trace_id": ...}`
            # instead of a top-level kwarg.
            assert client.span.call_args.kwargs["trace_context"] == {
                "trace_id": "trace-outer",
            }
        finally:
            correlation_id_var.reset(token)

    async def test_span_omits_trace_id_when_none(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        client = _install_fake_langfuse(monkeypatch)
        # No correlation_id set — autouse fixtures keep it clean.
        async with traced_span("orphan"):
            pass
        kwargs = client.span.call_args.kwargs
        # Without a trace_id we omit trace_context entirely; the v4 SDK
        # then opens a new top-level trace for the span.
        assert "trace_context" not in kwargs


# ---------------------------------------------------------------------------
# Lifecycle on success and error
# ---------------------------------------------------------------------------


class TestSpanLifecycle:
    async def test_span_end_called_on_success(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        client = _install_fake_langfuse(monkeypatch)
        span = client.span.return_value

        async with traced_span("x"):
            pass

        span.end.assert_called_once()
        kwargs = span.end.call_args.kwargs
        assert kwargs["output"] == {"status": "ok"}
        assert "duration_ms" in kwargs["metadata"]
        assert kwargs["metadata"]["duration_ms"] >= 0

    async def test_span_end_called_on_exception(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        client = _install_fake_langfuse(monkeypatch)
        span = client.span.return_value

        with contextlib.suppress(ValueError):
            async with traced_span("x"):
                msg = "boom"
                raise ValueError(msg)

        span.end.assert_called_once()
        kwargs = span.end.call_args.kwargs
        assert kwargs["level"] == "ERROR"
        assert "boom" in kwargs["output"]["error"]
        assert "duration_ms" in kwargs["metadata"]

    async def test_exception_propagates(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _install_fake_langfuse(monkeypatch)

        async def _raises() -> None:
            async with traced_span("x"):
                msg = "propagated"
                raise RuntimeError(msg)

        with pytest.raises(RuntimeError, match="propagated"):
            await _raises()

    async def test_duration_recorded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        client = _install_fake_langfuse(monkeypatch)
        span = client.span.return_value

        import asyncio

        async with traced_span("x"):
            await asyncio.sleep(0.01)

        duration = span.end.call_args.kwargs["metadata"]["duration_ms"]
        # 10ms sleep should produce at least 5ms measurable duration even
        # under coarse timers; cap upper-bound generously to avoid flakes.
        assert duration >= 5.0
        assert duration < 1000.0


# ---------------------------------------------------------------------------
# Resilience: tracing failures don't break the block
# ---------------------------------------------------------------------------


class TestResilience:
    async def test_span_creation_failure_yields_none(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        client = _install_fake_langfuse(monkeypatch)
        client.span.side_effect = RuntimeError("langfuse down")

        async with traced_span("x") as span:
            assert span is None
        # The block completed despite span creation failing.

    async def test_span_end_failure_does_not_crash(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        client = _install_fake_langfuse(monkeypatch)
        client.span.return_value.end.side_effect = RuntimeError("end failed")

        # The block completes normally; the trailing span.end error is
        # swallowed.
        async with traced_span("x"):
            pass

    async def test_span_end_failure_does_not_mask_block_exception(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        client = _install_fake_langfuse(monkeypatch)
        client.span.return_value.end.side_effect = RuntimeError("end failed")

        async def _raises() -> None:
            async with traced_span("x"):
                msg = "real error"
                raise ValueError(msg)

        # The original ValueError surfaces, not the end failure.
        with pytest.raises(ValueError, match="real error"):
            await _raises()


# ---------------------------------------------------------------------------
# Block body can use the yielded span
# ---------------------------------------------------------------------------


class TestSpanInteraction:
    async def test_block_can_call_methods_on_span(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        client = _install_fake_langfuse(monkeypatch)
        span = client.span.return_value

        async with traced_span("x") as s:
            assert s is not None
            s.update(metadata={"step": "midway"})

        span.update.assert_called_once_with(metadata={"step": "midway"})

    async def test_block_skips_when_span_is_none(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # When Langfuse isn't configured, blocks should defensively check
        # `if span is not None:` before using it. This test documents the
        # contract — the yielded `None` is well-defined behavior.
        _no_langfuse(monkeypatch)
        ran = {"value": False}

        async with traced_span("x") as span:
            if span is not None:  # pragma: no cover — branch deliberately untaken
                span.update(metadata={"step": "midway"})
            ran["value"] = True

        assert ran["value"] is True


# ---------------------------------------------------------------------------
# Multiple sequential spans
# ---------------------------------------------------------------------------


class TestMultipleSpans:
    async def test_sequential_spans_in_same_trace(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        client = _install_fake_langfuse(monkeypatch)
        # Make span() return a fresh mock per call so we can verify each
        # span was opened and ended independently.
        spans: list[MagicMock] = []

        def _fresh_span(**_kwargs: Any) -> MagicMock:
            new = MagicMock(name=f"span-{len(spans)}")
            new.id = f"span-{len(spans)}"
            spans.append(new)
            return new

        client.span.side_effect = _fresh_span

        token = correlation_id_var.set("trace-outer")
        try:
            async with traced_span("step-1"):
                pass
            async with traced_span("step-2"):
                pass
            async with traced_span("step-3"):
                pass
        finally:
            correlation_id_var.reset(token)

        assert len(spans) == 3
        # All three spans were ended.
        for s in spans:
            s.end.assert_called_once()
        # All three were linked to the same trace via trace_context.
        for call in client.span.call_args_list:
            assert call.kwargs["trace_context"] == {"trace_id": "trace-outer"}
