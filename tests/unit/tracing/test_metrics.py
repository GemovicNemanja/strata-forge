"""Unit tests for `forge.tracing.metrics`."""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

import pytest

from forge.tracing.metrics import record_categorical_metric, record_numeric_metric


def _install_fake_langfuse(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    monkeypatch.setenv("LANGFUSE_HOST", "http://test")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")

    fake_module = types.ModuleType("langfuse")
    client_mock = MagicMock(name="lf-client")
    fake_module.Langfuse = MagicMock(return_value=client_mock)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "langfuse", fake_module)
    return client_mock


def _no_langfuse(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)


# ---------------------------------------------------------------------------
# record_numeric_metric
# ---------------------------------------------------------------------------


class TestRecordNumericMetric:
    async def test_attached_to_trace(self, monkeypatch: pytest.MonkeyPatch) -> None:
        client = _install_fake_langfuse(monkeypatch)
        await record_numeric_metric("input_tokens", 512, trace_id="trace-abc")

        kwargs = client.score.call_args.kwargs
        assert kwargs["trace_id"] == "trace-abc"
        assert kwargs["name"] == "input_tokens"
        assert kwargs["value"] == 512
        assert kwargs["data_type"] == "NUMERIC"
        assert "observation_id" not in kwargs

    async def test_attached_to_observation(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        client = _install_fake_langfuse(monkeypatch)
        await record_numeric_metric("latency_ms", 123.4, observation_id="obs-1")

        kwargs = client.score.call_args.kwargs
        assert kwargs["observation_id"] == "obs-1"
        assert "trace_id" not in kwargs
        assert kwargs["value"] == 123.4
        assert kwargs["data_type"] == "NUMERIC"

    async def test_both_targets_passed_through(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Supplying both is allowed — observation_id pinpoints, trace_id
        # is a shortcut so Langfuse doesn't have to resolve it.
        client = _install_fake_langfuse(monkeypatch)
        await record_numeric_metric("x", 1.0, trace_id="trace-a", observation_id="obs-1")
        kwargs = client.score.call_args.kwargs
        assert kwargs["trace_id"] == "trace-a"
        assert kwargs["observation_id"] == "obs-1"

    async def test_with_comment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        client = _install_fake_langfuse(monkeypatch)
        await record_numeric_metric(
            "cost_usd", 0.0042, trace_id="t", comment="excludes cache writes"
        )
        assert client.score.call_args.kwargs["comment"] == "excludes cache writes"

    async def test_omits_comment_when_unset(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        client = _install_fake_langfuse(monkeypatch)
        await record_numeric_metric("x", 1.0, trace_id="t")
        assert "comment" not in client.score.call_args.kwargs

    async def test_integer_value(self, monkeypatch: pytest.MonkeyPatch) -> None:
        client = _install_fake_langfuse(monkeypatch)
        await record_numeric_metric("count", 42, trace_id="t")
        assert client.score.call_args.kwargs["value"] == 42

    async def test_missing_both_targets_raises(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _install_fake_langfuse(monkeypatch)
        with pytest.raises(ValueError, match="trace_id or observation_id"):
            await record_numeric_metric("x", 1.0)

    async def test_missing_targets_raises_even_when_unconfigured(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # The programmer error surfaces in dev (no Langfuse) too —
        # silent dropping only in production would mask the bug.
        _no_langfuse(monkeypatch)
        with pytest.raises(ValueError, match="trace_id or observation_id"):
            await record_numeric_metric("x", 1.0)

    async def test_noop_when_unconfigured(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _no_langfuse(monkeypatch)
        # Valid call with a target — no raise, no client interaction.
        await record_numeric_metric("x", 1.0, trace_id="t")

    async def test_failure_swallowed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        client = _install_fake_langfuse(monkeypatch)
        client.score.side_effect = RuntimeError("langfuse down")
        # Doesn't raise.
        await record_numeric_metric("x", 1.0, trace_id="t")


# ---------------------------------------------------------------------------
# record_categorical_metric
# ---------------------------------------------------------------------------


class TestRecordCategoricalMetric:
    async def test_attached_to_trace(self, monkeypatch: pytest.MonkeyPatch) -> None:
        client = _install_fake_langfuse(monkeypatch)
        await record_categorical_metric("model", "claude-opus-4-7", trace_id="trace-abc")

        kwargs = client.score.call_args.kwargs
        assert kwargs["trace_id"] == "trace-abc"
        assert kwargs["name"] == "model"
        assert kwargs["value"] == "claude-opus-4-7"
        assert kwargs["data_type"] == "CATEGORICAL"

    async def test_attached_to_observation(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        client = _install_fake_langfuse(monkeypatch)
        await record_categorical_metric("route", "bedrock", observation_id="obs-1")
        kwargs = client.score.call_args.kwargs
        assert kwargs["observation_id"] == "obs-1"
        assert "trace_id" not in kwargs

    async def test_with_comment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        client = _install_fake_langfuse(monkeypatch)
        await record_categorical_metric(
            "finish_reason", "stop", trace_id="t", comment="natural completion"
        )
        assert client.score.call_args.kwargs["comment"] == "natural completion"

    async def test_omits_comment_when_unset(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        client = _install_fake_langfuse(monkeypatch)
        await record_categorical_metric("x", "y", trace_id="t")
        assert "comment" not in client.score.call_args.kwargs

    async def test_missing_both_targets_raises(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _install_fake_langfuse(monkeypatch)
        with pytest.raises(ValueError, match="trace_id or observation_id"):
            await record_categorical_metric("x", "y")

    async def test_missing_targets_raises_when_unconfigured(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _no_langfuse(monkeypatch)
        with pytest.raises(ValueError, match="trace_id or observation_id"):
            await record_categorical_metric("x", "y")

    async def test_noop_when_unconfigured(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _no_langfuse(monkeypatch)
        await record_categorical_metric("model", "gpt-5.5", trace_id="t")

    async def test_failure_swallowed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        client = _install_fake_langfuse(monkeypatch)
        client.score.side_effect = RuntimeError("langfuse down")
        await record_categorical_metric("model", "gpt-5.5", trace_id="t")


# ---------------------------------------------------------------------------
# Re-exports
# ---------------------------------------------------------------------------


class TestModuleSurface:
    def test_re_exported_from_tracing(self) -> None:
        from forge.tracing import (
            record_categorical_metric as exported_cat,
        )
        from forge.tracing import (
            record_numeric_metric as exported_num,
        )

        assert exported_num is record_numeric_metric
        assert exported_cat is record_categorical_metric
