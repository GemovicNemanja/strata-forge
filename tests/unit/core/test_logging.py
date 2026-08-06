"""Unit tests for `strata_forge.core.logging`."""

from __future__ import annotations

import asyncio
import time

import pytest
from structlog.testing import capture_logs

from strata_forge.core.ids import correlation_id_var, set_correlation_id
from strata_forge.core.logging import (
    add_correlation_id,
    configure_logging,
    get_logger,
    traced_span,
)


class TestConfigure:
    def test_default_level_filters_debug(self) -> None:
        configure_logging(level="INFO")
        log = get_logger("test.level")
        with capture_logs() as records:
            log.debug("hidden")
            log.info("visible")
        assert [r["event"] for r in records] == ["visible"]

    def test_debug_level_includes_debug(self) -> None:
        configure_logging(level="DEBUG")
        log = get_logger("test.level")
        with capture_logs() as records:
            log.debug("now appears")
        assert [r["event"] for r in records] == ["now appears"]

    def test_unknown_level_falls_back_to_info(self) -> None:
        configure_logging(level="NOTALEVEL")
        log = get_logger("test.level")
        with capture_logs() as records:
            log.debug("hidden")
            log.info("visible")
        assert [r["event"] for r in records] == ["visible"]

    def test_get_logger_with_no_name(self) -> None:
        configure_logging()
        log = get_logger()
        with capture_logs() as records:
            log.info("anonymous")
        assert records[0]["event"] == "anonymous"


class TestAddCorrelationId:
    """Direct tests for the structlog processor that injects correlation_id."""

    def test_includes_correlation_id_when_set(self) -> None:
        token = set_correlation_id("abc-123")
        try:
            event_dict = add_correlation_id(None, "info", {"event": "hello"})
            assert event_dict["correlation_id"] == "abc-123"
            assert event_dict["event"] == "hello"
        finally:
            correlation_id_var.reset(token)

    def test_omits_correlation_id_when_unset(self) -> None:
        token = set_correlation_id(None)
        try:
            event_dict = add_correlation_id(None, "info", {"event": "hello"})
            assert "correlation_id" not in event_dict
        finally:
            correlation_id_var.reset(token)

    def test_does_not_mutate_other_keys(self) -> None:
        token = set_correlation_id("ctx")
        try:
            event_dict = add_correlation_id(
                None,
                "info",
                {"event": "hello", "user_id": "u1"},
            )
            assert event_dict["user_id"] == "u1"
        finally:
            correlation_id_var.reset(token)

    async def test_correlation_id_propagates_across_awaits(self) -> None:
        seen: list[str | None] = []
        token = set_correlation_id("traced")
        try:

            async def child() -> None:
                event_dict = add_correlation_id(None, "info", {"event": "hello"})
                seen.append(event_dict.get("correlation_id"))  # type: ignore[arg-type]
                await asyncio.sleep(0)
                event_dict = add_correlation_id(None, "info", {"event": "hello"})
                seen.append(event_dict.get("correlation_id"))  # type: ignore[arg-type]

            await child()
            assert seen == ["traced", "traced"]
        finally:
            correlation_id_var.reset(token)


class TestTracedSpan:
    def test_emits_start_and_end(self) -> None:
        configure_logging(level="INFO")
        with capture_logs() as records, traced_span("operation"):
            pass
        events = [r["event"] for r in records]
        assert events == ["operation.start", "operation.end"]

    def test_records_elapsed_time(self) -> None:
        configure_logging(level="INFO")
        with capture_logs() as records, traced_span("sleeper"):
            time.sleep(0.01)
        end_record = next(r for r in records if r["event"] == "sleeper.end")
        assert end_record["elapsed_s"] >= 0.01

    def test_emits_error_on_exception_and_reraises(self) -> None:
        configure_logging(level="INFO")
        with (
            capture_logs() as records,
            pytest.raises(ValueError, match="boom"),
            traced_span("operation"),
        ):
            raise ValueError("boom")
        events = [r["event"] for r in records]
        assert events == ["operation.start", "operation.error"]
        error_record = records[1]
        assert error_record["error_type"] == "ValueError"
        assert error_record["error"] == "boom"
        assert error_record["elapsed_s"] >= 0

    def test_attaches_extra_fields(self) -> None:
        configure_logging(level="INFO")
        with (
            capture_logs() as records,
            traced_span("operation", user_id="u1", request_id="r1"),
        ):
            pass
        for record in records:
            assert record["user_id"] == "u1"
            assert record["request_id"] == "r1"

    def test_extra_fields_attached_on_error(self) -> None:
        configure_logging(level="INFO")
        with (
            capture_logs() as records,
            pytest.raises(RuntimeError),
            traced_span("operation", user_id="u1"),
        ):
            raise RuntimeError("nope")
        error_record = next(r for r in records if r["event"] == "operation.error")
        assert error_record["user_id"] == "u1"
