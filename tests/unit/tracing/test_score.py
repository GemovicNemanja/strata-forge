"""Unit tests for `forge.tracing.score`."""

from __future__ import annotations

import sys
import types
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

from forge.tracing.score import score_observation, score_trace

if TYPE_CHECKING:
    import pytest


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
# score_trace
# ---------------------------------------------------------------------------


class TestScoreTrace:
    async def test_basic_numeric(self, monkeypatch: pytest.MonkeyPatch) -> None:
        client = _install_fake_langfuse(monkeypatch)
        await score_trace("trace-abc", "helpfulness", 4.2)

        client.score.assert_called_once()
        kwargs = client.score.call_args.kwargs
        assert kwargs["trace_id"] == "trace-abc"
        assert kwargs["name"] == "helpfulness"
        assert kwargs["value"] == 4.2

    async def test_string_value(self, monkeypatch: pytest.MonkeyPatch) -> None:
        client = _install_fake_langfuse(monkeypatch)
        await score_trace("trace-x", "quality", "GOOD")
        kwargs = client.score.call_args.kwargs
        assert kwargs["value"] == "GOOD"

    async def test_boolean_value(self, monkeypatch: pytest.MonkeyPatch) -> None:
        client = _install_fake_langfuse(monkeypatch)
        await score_trace("trace-x", "is_correct", value=True)
        assert client.score.call_args.kwargs["value"] is True

    async def test_integer_value(self, monkeypatch: pytest.MonkeyPatch) -> None:
        client = _install_fake_langfuse(monkeypatch)
        await score_trace("trace-x", "rank", 3)
        assert client.score.call_args.kwargs["value"] == 3

    async def test_with_comment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        client = _install_fake_langfuse(monkeypatch)
        await score_trace("trace-x", "n", 1.0, comment="needs work")
        assert client.score.call_args.kwargs["comment"] == "needs work"

    async def test_without_comment_omits_kwarg(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Don't pass `comment=None` to the SDK; let it use its own default.
        client = _install_fake_langfuse(monkeypatch)
        await score_trace("trace-x", "n", 1.0)
        assert "comment" not in client.score.call_args.kwargs

    async def test_with_data_type(self, monkeypatch: pytest.MonkeyPatch) -> None:
        client = _install_fake_langfuse(monkeypatch)
        await score_trace("trace-x", "n", "EXCELLENT", data_type="CATEGORICAL")
        assert client.score.call_args.kwargs["data_type"] == "CATEGORICAL"

    async def test_without_data_type_omits_kwarg(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        client = _install_fake_langfuse(monkeypatch)
        await score_trace("trace-x", "n", 1.0)
        assert "data_type" not in client.score.call_args.kwargs

    async def test_noop_when_unconfigured(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Without raising — caller's path is preserved.
        _no_langfuse(monkeypatch)
        await score_trace("trace-x", "n", 1.0)

    async def test_failure_swallowed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        client = _install_fake_langfuse(monkeypatch)
        client.score.side_effect = RuntimeError("langfuse down")
        # Doesn't raise.
        await score_trace("trace-x", "n", 1.0)


# ---------------------------------------------------------------------------
# score_observation
# ---------------------------------------------------------------------------


class TestScoreObservation:
    async def test_basic(self, monkeypatch: pytest.MonkeyPatch) -> None:
        client = _install_fake_langfuse(monkeypatch)
        await score_observation("obs-1", "step_accuracy", 0.9)

        kwargs = client.score.call_args.kwargs
        assert kwargs["observation_id"] == "obs-1"
        assert kwargs["name"] == "step_accuracy"
        assert kwargs["value"] == 0.9

    async def test_with_trace_id(self, monkeypatch: pytest.MonkeyPatch) -> None:
        client = _install_fake_langfuse(monkeypatch)
        await score_observation("obs-1", "n", 1.0, trace_id="trace-abc")
        kwargs = client.score.call_args.kwargs
        assert kwargs["trace_id"] == "trace-abc"

    async def test_without_trace_id_omits_kwarg(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        client = _install_fake_langfuse(monkeypatch)
        await score_observation("obs-1", "n", 1.0)
        assert "trace_id" not in client.score.call_args.kwargs

    async def test_with_comment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        client = _install_fake_langfuse(monkeypatch)
        await score_observation("obs-1", "n", 1.0, comment="explanation")
        assert client.score.call_args.kwargs["comment"] == "explanation"

    async def test_with_data_type(self, monkeypatch: pytest.MonkeyPatch) -> None:
        client = _install_fake_langfuse(monkeypatch)
        await score_observation("obs-1", "verdict", True, data_type="BOOLEAN")
        assert client.score.call_args.kwargs["data_type"] == "BOOLEAN"

    async def test_noop_when_unconfigured(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _no_langfuse(monkeypatch)
        await score_observation("obs-1", "n", 1.0)

    async def test_failure_swallowed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        client = _install_fake_langfuse(monkeypatch)
        client.score.side_effect = RuntimeError("langfuse down")
        await score_observation("obs-1", "n", 1.0)


# ---------------------------------------------------------------------------
# Re-exports
# ---------------------------------------------------------------------------


class TestModuleSurface:
    def test_re_exported_from_tracing(self) -> None:
        from forge.tracing import (
            score_observation as exported_obs,
        )
        from forge.tracing import (
            score_trace as exported_trace,
        )

        assert exported_trace is score_trace
        assert exported_obs is score_observation
