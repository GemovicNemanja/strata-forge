"""Unit tests for `strata_forge.compute.batch.BatchInferenceRunner`."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock

import pytest

from strata_forge.compute.batch import BatchInferenceResult, BatchInferenceRunner
from strata_forge.llm.messages import UserMessage
from strata_forge.llm.responses import LLMResponse, Usage
from strata_forge.llm.routing import ModelRoute


def _response(text: str = "ok") -> LLMResponse:
    return LLMResponse(
        text=text,
        tool_calls=[],
        finish_reason="stop",
        usage=Usage(input_tokens=1, output_tokens=1),
        cost_usd=0.0,
        route=ModelRoute(
            model="claude-opus-4-7",
            provider="anthropic",
            provider_model_id="claude-opus-4-7",
        ),
        latency_ms=1.0,
    )


def _prompts(n: int) -> list[list[UserMessage]]:
    return [[UserMessage(content=f"prompt-{i}")] for i in range(n)]


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_defaults(self) -> None:
        runner = BatchInferenceRunner(AsyncMock())
        assert runner.concurrency == 5
        assert runner.on_error == "raise"

    def test_concurrency_validated(self) -> None:
        with pytest.raises(ValueError, match="concurrency"):
            BatchInferenceRunner(AsyncMock(), concurrency=0)

    def test_on_error_validated(self) -> None:
        with pytest.raises(ValueError, match="on_error"):
            BatchInferenceRunner(AsyncMock(), on_error="ignore")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestHappyPath:
    async def test_runs_all_prompts(self) -> None:
        client = AsyncMock()

        async def _ok(**_kwargs: Any) -> LLMResponse:
            return _response("ok")

        client.complete = AsyncMock(side_effect=_ok)
        runner = BatchInferenceRunner(client)
        results = await runner.run(_prompts(5))
        assert len(results) == 5
        assert all(r.succeeded for r in results)
        assert client.complete.call_count == 5

    async def test_empty_input(self) -> None:
        client = AsyncMock()
        client.complete = AsyncMock()
        runner = BatchInferenceRunner(client)
        assert await runner.run([]) == ()
        client.complete.assert_not_called()

    async def test_results_positionally_aligned(self) -> None:
        client = AsyncMock()
        # Each prompt gets a response containing its index.
        responses = [_response(f"answer-{i}") for i in range(3)]
        client.complete = AsyncMock(side_effect=responses)
        runner = BatchInferenceRunner(client, concurrency=1)
        results = await runner.run(_prompts(3))
        # With concurrency=1 the calls are serial, so we can assert order.
        assert [r.response.text for r in results if r.response] == [
            "answer-0",
            "answer-1",
            "answer-2",
        ]

    async def test_sampling_params_forwarded(self) -> None:
        client = AsyncMock()
        client.complete = AsyncMock(return_value=_response())
        runner = BatchInferenceRunner(client)
        await runner.run(_prompts(1), temperature=0.7, max_tokens=200, top_p=0.95)
        kwargs = client.complete.call_args.kwargs
        assert kwargs["temperature"] == 0.7
        assert kwargs["max_tokens"] == 200
        assert kwargs["top_p"] == 0.95


# ---------------------------------------------------------------------------
# Concurrency
# ---------------------------------------------------------------------------


class TestConcurrency:
    async def test_respects_limit(self) -> None:
        peak = 0
        in_flight = 0

        async def _slow(**_kwargs: Any) -> LLMResponse:
            nonlocal peak, in_flight
            in_flight += 1
            peak = max(peak, in_flight)
            await asyncio.sleep(0.01)
            in_flight -= 1
            return _response()

        client = AsyncMock()
        client.complete = AsyncMock(side_effect=_slow)
        runner = BatchInferenceRunner(client, concurrency=3)
        await runner.run(_prompts(20))
        assert peak <= 3


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    async def test_raise_mode_propagates_first_failure(self) -> None:
        client = AsyncMock()
        client.complete = AsyncMock(side_effect=RuntimeError("boom"))
        runner = BatchInferenceRunner(client, on_error="raise")
        with pytest.raises(RuntimeError, match="boom"):
            await runner.run(_prompts(3))

    async def test_collect_mode_records_failures(self) -> None:
        client = AsyncMock()

        async def _maybe_fail(**kwargs: Any) -> LLMResponse:
            messages = kwargs["messages"]
            if "prompt-1" in messages[0].content:
                err = "transient"
                raise RuntimeError(err)
            return _response("ok")

        client.complete = AsyncMock(side_effect=_maybe_fail)
        runner = BatchInferenceRunner(client, on_error="collect", concurrency=1)
        results = await runner.run(_prompts(3))
        assert len(results) == 3
        assert results[0].succeeded
        assert not results[1].succeeded
        assert isinstance(results[1].error, RuntimeError)
        assert results[2].succeeded


# ---------------------------------------------------------------------------
# BatchInferenceResult shape
# ---------------------------------------------------------------------------


class TestResultShape:
    def test_succeeded_when_response_set(self) -> None:
        result = BatchInferenceResult(response=_response())
        assert result.succeeded is True

    def test_not_succeeded_when_error_set(self) -> None:
        result = BatchInferenceResult(error=RuntimeError("bad"))
        assert result.succeeded is False

    def test_empty_result_not_succeeded(self) -> None:
        # Edge case: neither set (shouldn't happen in practice).
        result = BatchInferenceResult()
        assert result.succeeded is False
