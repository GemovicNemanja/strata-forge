"""Unit tests for `forge.llm.diagnostic`."""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

import anyio
import pytest

from forge.llm.diagnostic import (
    DiagnosticRecord,
    make_error_field,
    utcnow_iso,
    write_diagnostic_record,
)

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sample_record(**overrides: object) -> DiagnosticRecord:
    base: dict[str, object] = {
        "timestamp": "2026-05-15T12:00:00.000000+00:00",
        "correlation_id": None,
        "request_hash": "a" * 64,
        "model": "claude-opus-4-7",
        "provider": "anthropic",
        "provider_model_id": "claude-opus-4-7",
        "messages": [{"role": "user", "content": "hi"}],
        "response_text": "hello",
        "tool_calls": [],
        "finish_reason": "stop",
        "usage": {"input_tokens": 5, "output_tokens": 2},
        "cost_usd": 0.001,
        "latency_ms": 234.5,
        "cache_hit": False,
        "error": None,
    }
    base.update(overrides)
    return DiagnosticRecord(**base)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# utcnow_iso
# ---------------------------------------------------------------------------


class TestUtcnowIso:
    def test_returns_iso_8601_utc(self) -> None:
        ts = utcnow_iso()
        # microsecond precision, +00:00 offset.
        assert re.match(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}\+00:00", ts)

    def test_monotonic_ish(self) -> None:
        a = utcnow_iso()
        b = utcnow_iso()
        assert b >= a  # ISO strings sort lexicographically when UTC.


# ---------------------------------------------------------------------------
# make_error_field
# ---------------------------------------------------------------------------


class TestMakeErrorField:
    def test_includes_type_and_message(self) -> None:
        err = ValueError("bad input")
        field = make_error_field(err)
        assert field == {"type": "ValueError", "message": "bad input"}

    def test_custom_exception(self) -> None:
        class _CustomError(RuntimeError):
            pass

        field = make_error_field(_CustomError("boom"))
        assert field["type"] == "_CustomError"
        assert field["message"] == "boom"

    def test_chained_exception(self) -> None:
        # The chained __cause__ doesn't appear — only the outer exception's name/message.
        try:
            try:
                raise ValueError("inner")
            except ValueError as inner:
                raise RuntimeError("outer") from inner
        except RuntimeError as exc:
            field = make_error_field(exc)
        assert field["type"] == "RuntimeError"
        assert field["message"] == "outer"


# ---------------------------------------------------------------------------
# DiagnosticRecord
# ---------------------------------------------------------------------------


class TestDiagnosticRecord:
    def test_minimum_fields(self) -> None:
        rec = DiagnosticRecord(
            timestamp="2026-05-15T12:00:00.000000+00:00",
            request_hash="x" * 64,
            model="claude-opus-4-7",
            provider="anthropic",
            provider_model_id="claude-opus-4-7",
        )
        assert rec.messages == []
        assert rec.response_text == ""
        assert rec.error is None
        assert rec.cache_hit is False

    def test_full_record_serializes_to_json(self) -> None:
        rec = _sample_record()
        raw = rec.model_dump_json()
        loaded = json.loads(raw)
        # JSON should round-trip — every field present and stringly-typed for
        # easy consumption.
        assert loaded["model"] == "claude-opus-4-7"
        assert loaded["usage"]["input_tokens"] == 5
        assert loaded["error"] is None

    def test_error_record(self) -> None:
        rec = _sample_record(
            response_text="",
            finish_reason=None,
            usage=None,
            cost_usd=None,
            error=make_error_field(RuntimeError("503")),
        )
        loaded = json.loads(rec.model_dump_json())
        assert loaded["error"] == {"type": "RuntimeError", "message": "503"}

    def test_is_frozen(self) -> None:
        from pydantic import ValidationError

        rec = _sample_record()
        with pytest.raises(ValidationError, match="frozen"):
            rec.model = "other"  # type: ignore[misc]

    def test_extra_fields_forbidden(self) -> None:
        # `extra="forbid"` rejects unknown fields so the wire format stays stable.
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            DiagnosticRecord(
                timestamp="t",
                request_hash="h",
                model="m",
                provider="p",
                provider_model_id="pmi",
                unknown_field="oops",  # type: ignore[call-arg]
            )


# ---------------------------------------------------------------------------
# write_diagnostic_record
# ---------------------------------------------------------------------------


class TestWriteDisabled:
    async def test_noop_when_disabled(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        # Default: FORGE_DIAGNOSTIC_ENABLED unset → disabled.
        monkeypatch.delenv("FORGE_DIAGNOSTIC_ENABLED", raising=False)
        monkeypatch.delenv("FORGE_DIAGNOSTIC_PATH", raising=False)
        # Point path at tmp_path so even if we DID write, we'd see it.
        monkeypatch.setenv("FORGE_DIAGNOSTIC_PATH", str(tmp_path / "diag.ndjson"))

        await write_diagnostic_record(_sample_record())
        # No file created.
        assert not (tmp_path / "diag.ndjson").exists()


class TestWriteEnabled:
    async def test_writes_when_enabled_via_env(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        path = tmp_path / "diag.ndjson"
        monkeypatch.setenv("FORGE_DIAGNOSTIC_ENABLED", "true")
        monkeypatch.setenv("FORGE_DIAGNOSTIC_PATH", str(path))

        await write_diagnostic_record(_sample_record())

        assert path.exists()
        contents = path.read_text(encoding="utf-8")
        assert contents.endswith("\n")
        loaded = json.loads(contents.rstrip("\n"))
        assert loaded["model"] == "claude-opus-4-7"

    async def test_appends_multiple_records(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        path = tmp_path / "diag.ndjson"
        monkeypatch.setenv("FORGE_DIAGNOSTIC_ENABLED", "true")
        monkeypatch.setenv("FORGE_DIAGNOSTIC_PATH", str(path))

        await write_diagnostic_record(_sample_record(response_text="first"))
        await write_diagnostic_record(_sample_record(response_text="second"))
        await write_diagnostic_record(_sample_record(response_text="third"))

        lines = path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 3
        assert json.loads(lines[0])["response_text"] == "first"
        assert json.loads(lines[1])["response_text"] == "second"
        assert json.loads(lines[2])["response_text"] == "third"

    async def test_creates_parent_directories(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        nested = tmp_path / "deep" / "nested" / "diag.ndjson"
        monkeypatch.setenv("FORGE_DIAGNOSTIC_ENABLED", "true")
        monkeypatch.setenv("FORGE_DIAGNOSTIC_PATH", str(nested))

        await write_diagnostic_record(_sample_record())
        assert nested.exists()

    async def test_explicit_path_overrides_env(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        # Even with the env flag *disabled*, an explicit path forces a write —
        # this is what test fixtures and one-off captures rely on.
        monkeypatch.delenv("FORGE_DIAGNOSTIC_ENABLED", raising=False)
        target = tmp_path / "override.ndjson"
        await write_diagnostic_record(_sample_record(), path=target)
        assert target.exists()
        loaded = json.loads(target.read_text(encoding="utf-8").rstrip("\n"))
        assert loaded["model"] == "claude-opus-4-7"


class TestConcurrentWrites:
    async def test_concurrent_writes_do_not_interleave(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        path = tmp_path / "diag.ndjson"
        monkeypatch.setenv("FORGE_DIAGNOSTIC_ENABLED", "true")
        monkeypatch.setenv("FORGE_DIAGNOSTIC_PATH", str(path))

        async def _write(i: int) -> None:
            await write_diagnostic_record(_sample_record(response_text=f"r{i}"))

        async with anyio.create_task_group() as tg:
            for i in range(20):
                tg.start_soon(_write, i)

        lines = path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 20
        # Every line must parse as JSON — no torn writes.
        parsed = [json.loads(line) for line in lines]
        # Each `r0`-`r19` appears exactly once (order is unspecified due to
        # concurrent scheduling, but the set must be complete).
        texts = {p["response_text"] for p in parsed}
        assert texts == {f"r{i}" for i in range(20)}


class TestRoundTripParse:
    async def test_written_record_round_trips_to_pydantic(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        # Whatever we write, we must be able to re-parse via the model.
        # This protects against schema drift over time.
        path = tmp_path / "diag.ndjson"
        monkeypatch.setenv("FORGE_DIAGNOSTIC_ENABLED", "true")
        monkeypatch.setenv("FORGE_DIAGNOSTIC_PATH", str(path))

        original = _sample_record(
            correlation_id="abc123",
            error=make_error_field(ValueError("oops")),
        )
        await write_diagnostic_record(original)

        raw = path.read_text(encoding="utf-8").rstrip("\n")
        parsed = DiagnosticRecord.model_validate_json(raw)
        assert parsed == original
