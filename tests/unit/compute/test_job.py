"""Unit tests for `strata_forge.compute.job`."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from strata_forge.compute.job import Job, JobStatus


class TestJob:
    def test_minimal_construction(self) -> None:
        job = Job(id="abc", backend="local", task_name="t")
        assert job.id == "abc"
        assert job.backend == "local"
        assert job.task_name == "t"
        assert job.metadata == {}

    def test_empty_id_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Job(id="", backend="local", task_name="t")

    def test_empty_backend_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Job(id="x", backend="", task_name="t")

    def test_is_frozen(self) -> None:
        job = Job(id="x", backend="local", task_name="t")
        with pytest.raises(ValidationError, match="frozen"):
            job.id = "y"  # type: ignore[misc]


class TestJobStatus:
    def test_pending(self) -> None:
        status = JobStatus(state="pending")
        assert status.exit_code is None
        assert status.is_terminal is False

    def test_running(self) -> None:
        status = JobStatus(state="running", started_at=datetime.now(UTC))
        assert status.is_terminal is False

    def test_succeeded_is_terminal(self) -> None:
        status = JobStatus(state="succeeded", exit_code=0)
        assert status.is_terminal is True

    def test_failed_is_terminal(self) -> None:
        status = JobStatus(state="failed", exit_code=1)
        assert status.is_terminal is True

    def test_cancelled_is_terminal(self) -> None:
        status = JobStatus(state="cancelled")
        assert status.is_terminal is True

    def test_invalid_state_rejected(self) -> None:
        with pytest.raises(ValidationError):
            JobStatus(state="ghost")  # type: ignore[arg-type]

    def test_is_frozen(self) -> None:
        status = JobStatus(state="pending")
        with pytest.raises(ValidationError, match="frozen"):
            status.state = "running"  # type: ignore[misc]
