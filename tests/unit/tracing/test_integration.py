"""Cross-module integration tests for `forge.tracing`.

The per-file unit tests cover each piece in isolation; these wire the
pieces together — `@traced` + `traced_span` nested, `@traced` + scores
+ metrics with the active correlation ID — to catch gluing bugs that
don't show up in any single file.
"""

from __future__ import annotations

import contextlib
import sys
import types
from typing import Any
from unittest.mock import MagicMock

import pytest

from forge.core.ids import correlation_id_var
from forge.tracing.decorator import traced
from forge.tracing.metrics import record_numeric_metric
from forge.tracing.score import score_trace
from forge.tracing.span import traced_span


def _install_fake_langfuse(
    monkeypatch: pytest.MonkeyPatch,
    *,
    trace_id: str = "trace-int",
    span_id: str = "span-int",
) -> MagicMock:
    """Inject a fake Langfuse module; return the client mock for inspection.

    The client's `trace()` returns a mock with `.id == trace_id`, and
    its `span()` returns a mock with `.id == span_id`. Tests inspect
    `client.score`, `client.span`, etc. to verify wiring.
    """
    monkeypatch.setenv("LANGFUSE_HOST", "http://test")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")

    fake_module = types.ModuleType("langfuse")
    client_mock = MagicMock(name="lf-client")

    trace_mock = MagicMock(name="lf-trace")
    trace_mock.id = trace_id
    client_mock.trace.return_value = trace_mock

    span_mock = MagicMock(name="lf-span")
    span_mock.id = span_id
    client_mock.span.return_value = span_mock

    fake_module.Langfuse = MagicMock(return_value=client_mock)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "langfuse", fake_module)
    return client_mock


# ---------------------------------------------------------------------------
# @traced + traced_span composition
# ---------------------------------------------------------------------------


class TestTracedWithSpan:
    async def test_span_inherits_trace_id_via_correlation_id(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # The @traced wrapper sets correlation_id_var = trace.id; the
        # traced_span inside reads that to link its span to the trace.
        client = _install_fake_langfuse(monkeypatch, trace_id="trace-A")

        @traced
        async def my_workflow() -> None:
            async with traced_span("sub-step"):
                pass

        await my_workflow()

        client.trace.assert_called_once()
        client.span.assert_called_once()
        assert client.span.call_args.kwargs["trace_id"] == "trace-A"

    async def test_sequential_spans_share_trace(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        client = _install_fake_langfuse(monkeypatch, trace_id="trace-B")

        @traced
        async def my_workflow() -> None:
            async with traced_span("step-1"):
                pass
            async with traced_span("step-2"):
                pass
            async with traced_span("step-3"):
                pass

        await my_workflow()

        assert client.span.call_count == 3
        for call in client.span.call_args_list:
            assert call.kwargs["trace_id"] == "trace-B"

    async def test_exception_in_span_propagates_through_traced(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        client = _install_fake_langfuse(monkeypatch)
        trace = client.trace.return_value
        span = client.span.return_value

        @traced
        async def my_workflow() -> None:
            async with traced_span("failing-step"):
                msg = "propagated"
                raise RuntimeError(msg)

        async def _runner() -> None:
            await my_workflow()

        with pytest.raises(RuntimeError, match="propagated"):
            await _runner()

        # Both the span and the trace recorded the error.
        span.end.assert_called_once()
        assert span.end.call_args.kwargs["level"] == "ERROR"
        trace.update.assert_called_once()
        assert trace.update.call_args.kwargs["level"] == "ERROR"


# ---------------------------------------------------------------------------
# @traced + score_trace
# ---------------------------------------------------------------------------


class TestTracedWithScore:
    async def test_score_inside_traced_uses_active_trace_id(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        client = _install_fake_langfuse(monkeypatch, trace_id="trace-C")

        @traced
        async def my_workflow() -> str | None:
            # Inside @traced, correlation_id_var carries the trace ID.
            current = correlation_id_var.get()
            if current is not None:
                await score_trace(current, "happiness", 5.0)
            return current

        result = await my_workflow()

        assert result == "trace-C"
        client.score.assert_called_once()
        kwargs = client.score.call_args.kwargs
        assert kwargs["trace_id"] == "trace-C"
        assert kwargs["name"] == "happiness"
        assert kwargs["value"] == 5.0


# ---------------------------------------------------------------------------
# @traced + record_numeric_metric
# ---------------------------------------------------------------------------


class TestTracedWithMetric:
    async def test_metric_inside_traced_uses_active_trace_id(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        client = _install_fake_langfuse(monkeypatch, trace_id="trace-D")

        @traced
        async def my_workflow() -> None:
            current = correlation_id_var.get()
            if current is not None:
                await record_numeric_metric(
                    "input_tokens", 123, trace_id=current
                )

        await my_workflow()

        # client.score is called by record_numeric_metric (underlying API).
        client.score.assert_called_once()
        kwargs = client.score.call_args.kwargs
        assert kwargs["trace_id"] == "trace-D"
        assert kwargs["name"] == "input_tokens"
        assert kwargs["data_type"] == "NUMERIC"


# ---------------------------------------------------------------------------
# Full pipeline: traced + span + score + metric
# ---------------------------------------------------------------------------


class TestFullPipeline:
    async def test_workflow_with_all_helpers(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        client = _install_fake_langfuse(monkeypatch, trace_id="trace-full")
        captured: list[dict[str, Any]] = []

        @traced(name="full-workflow", tags=["v1", "test"])
        async def workflow() -> str:
            cid = correlation_id_var.get()
            assert cid == "trace-full"

            async with traced_span("preprocessing"):
                pass

            async with traced_span("main-work"):
                # Record a metric while inside the span.
                await record_numeric_metric(
                    "items_processed", 42, trace_id=cid
                )

            # Final score on the trace.
            await score_trace(cid, "quality", 4.5)
            return "done"

        result = await workflow()

        assert result == "done"

        # The trace was created with the right name and tags.
        client.trace.assert_called_once()
        trace_kwargs = client.trace.call_args.kwargs
        assert trace_kwargs["name"] == "full-workflow"
        assert trace_kwargs["tags"] == ["v1", "test"]

        # Two spans opened.
        assert client.span.call_count == 2
        span_names = [call.kwargs["name"] for call in client.span.call_args_list]
        assert span_names == ["preprocessing", "main-work"]

        # client.score called twice — once for the metric, once for the score.
        assert client.score.call_count == 2
        # Capture call kwargs for assertion.
        for call in client.score.call_args_list:
            captured.append(call.kwargs)
        metric_call = next(c for c in captured if c["name"] == "items_processed")
        score_call = next(c for c in captured if c["name"] == "quality")
        assert metric_call["data_type"] == "NUMERIC"
        assert metric_call["value"] == 42
        assert score_call["value"] == 4.5


# ---------------------------------------------------------------------------
# correlation_id propagation discipline
# ---------------------------------------------------------------------------


class TestCorrelationIdDiscipline:
    async def test_outer_correlation_id_preserved_after_traced(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _install_fake_langfuse(monkeypatch, trace_id="trace-inner")
        token = correlation_id_var.set("outer-cid")
        try:

            @traced
            async def my_workflow() -> str | None:
                return correlation_id_var.get()

            inside_value = await my_workflow()
            assert inside_value == "trace-inner"  # inside the call
            assert correlation_id_var.get() == "outer-cid"  # restored
        finally:
            correlation_id_var.reset(token)

    async def test_span_does_not_mutate_correlation_id(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # traced_span links its span to the active trace via correlation_id_var,
        # but it doesn't mutate the var inside the block — the trace_id stays
        # the correlation ID throughout.
        _install_fake_langfuse(monkeypatch, trace_id="trace-outer")
        token = correlation_id_var.set("trace-outer")
        try:
            async with traced_span("inner-span"):
                # Inside the span, correlation_id_var still holds the trace_id.
                assert correlation_id_var.get() == "trace-outer"
            # After the span exits, still unchanged.
            assert correlation_id_var.get() == "trace-outer"
        finally:
            correlation_id_var.reset(token)


# ---------------------------------------------------------------------------
# Public API surface — re-exports compose cleanly
# ---------------------------------------------------------------------------


class TestPublicSurface:
    def test_all_helpers_importable_from_package(self) -> None:
        # Round-trip import check: everything advertised in __all__
        # actually resolves.
        from forge.tracing import (
            LITELLM_CALLBACK_NAME,
            ScoreValue,
            get_client,
            install_litellm_callback,
            is_litellm_callback_installed,
            record_categorical_metric,
            record_numeric_metric,
            reset_client,
            score_observation,
            score_trace,
            traced,
            traced_span,
        )

        # Touch each one so an unused-import lint doesn't drop them
        # silently.
        assert LITELLM_CALLBACK_NAME == "langfuse"
        assert ScoreValue is not None
        assert callable(get_client)
        assert callable(install_litellm_callback)
        assert callable(is_litellm_callback_installed)
        assert callable(record_categorical_metric)
        assert callable(record_numeric_metric)
        assert callable(reset_client)
        assert callable(score_observation)
        assert callable(score_trace)
        assert callable(traced)
        assert callable(traced_span)

    def test_no_eager_langfuse_import(self) -> None:
        # Importing forge.tracing without the [langfuse] extra installed
        # must not crash. The lazy-import contract is enforced by every
        # langfuse-touching function importing inside its body.
        import importlib

        import forge.tracing as tracing_pkg

        # Re-import in a fresh state to be extra defensive.
        importlib.reload(tracing_pkg)


# Silence the TYPE_CHECKING block re-export warning on the contextlib import.
_ = contextlib.suppress
