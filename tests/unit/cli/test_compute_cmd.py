"""Unit tests for `forge.cli.compute`."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
import yaml
from typer.testing import CliRunner

from forge.cli.main import app
from forge.compute.job import Job, JobStatus

if TYPE_CHECKING:
    from pathlib import Path

runner = CliRunner()


@pytest.fixture
def isolated_state(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Redirect the CLI state dir into a temp path."""
    state = tmp_path / "jobs"
    monkeypatch.setattr("forge.cli.compute._STATE_DIR", state)
    return state


@pytest.fixture
def task_yaml(tmp_path: Path) -> Path:
    path = tmp_path / "task.yaml"
    path.write_text(yaml.safe_dump({"name": "hello", "run": "echo hi"}))
    return path


class _FakeBackend:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.cancelled: list[Job] = []
        self.cleaned: list[Job] = []

    @property
    def name(self) -> str:
        return "fake"

    async def submit(self, task: Any) -> Job:
        return Job(id="job-123", backend="local", task_name=task.name)

    async def status(self, job: Job) -> JobStatus:
        del job
        return JobStatus(state="running", message="active")

    async def logs(self, job: Job, *, tail: int | None = None) -> str:
        del job, tail
        return "line1\nline2\n"

    async def cancel(self, job: Job) -> None:
        self.cancelled.append(job)

    async def cleanup(self, job: Job) -> None:
        self.cleaned.append(job)


@pytest.fixture
def fake_local(monkeypatch: pytest.MonkeyPatch) -> _FakeBackend:
    backend = _FakeBackend()

    def _make(name: str, kwargs: dict[str, Any]) -> _FakeBackend:
        del name, kwargs
        return backend

    monkeypatch.setattr("forge.cli.compute._make_backend", _make)
    return backend


class TestSubmit:
    def test_submit_saves_state(
        self,
        isolated_state: Path,
        task_yaml: Path,
        fake_local: _FakeBackend,
    ) -> None:
        result = runner.invoke(app, ["compute", "submit", str(task_yaml)])
        assert result.exit_code == 0
        assert "submitted" in result.output
        assert "job-123" in result.output
        assert (isolated_state / "job-123.json").exists()

    def test_ssh_requires_host_and_user(self, isolated_state: Path, task_yaml: Path) -> None:
        result = runner.invoke(app, ["compute", "submit", str(task_yaml), "-b", "ssh"])
        assert result.exit_code != 0
        assert "ssh-host" in result.output

    def test_missing_task_file_rejected(self, isolated_state: Path) -> None:
        result = runner.invoke(app, ["compute", "submit", "./no-such-file.yaml"])
        assert result.exit_code != 0


class TestStatus:
    def test_status_after_submit(
        self,
        isolated_state: Path,
        task_yaml: Path,
        fake_local: _FakeBackend,
    ) -> None:
        runner.invoke(app, ["compute", "submit", str(task_yaml)])
        result = runner.invoke(app, ["compute", "status", "job-123"])
        assert result.exit_code == 0
        assert "running" in result.output
        assert "active" in result.output

    def test_status_unknown_job(self, isolated_state: Path) -> None:
        result = runner.invoke(app, ["compute", "status", "ghost"])
        assert result.exit_code != 0


class TestLogs:
    def test_logs_output(
        self,
        isolated_state: Path,
        task_yaml: Path,
        fake_local: _FakeBackend,
    ) -> None:
        runner.invoke(app, ["compute", "submit", str(task_yaml)])
        result = runner.invoke(app, ["compute", "logs", "job-123"])
        assert result.exit_code == 0
        assert "line1" in result.output
        assert "line2" in result.output


class TestCancel:
    def test_cancel(
        self,
        isolated_state: Path,
        task_yaml: Path,
        fake_local: _FakeBackend,
    ) -> None:
        runner.invoke(app, ["compute", "submit", str(task_yaml)])
        result = runner.invoke(app, ["compute", "cancel", "job-123"])
        assert result.exit_code == 0
        assert "cancelled" in result.output
        assert len(fake_local.cancelled) == 1


class TestCleanup:
    def test_cleanup_removes_state(
        self,
        isolated_state: Path,
        task_yaml: Path,
        fake_local: _FakeBackend,
    ) -> None:
        runner.invoke(app, ["compute", "submit", str(task_yaml)])
        assert (isolated_state / "job-123.json").exists()
        result = runner.invoke(app, ["compute", "cleanup", "job-123"])
        assert result.exit_code == 0
        assert "cleaned up" in result.output
        assert not (isolated_state / "job-123.json").exists()
        assert len(fake_local.cleaned) == 1


class TestList:
    def test_empty(self, isolated_state: Path) -> None:
        result = runner.invoke(app, ["compute", "list"])
        assert result.exit_code == 0
        assert "no saved jobs" in result.output

    def test_lists_saved_jobs(
        self,
        isolated_state: Path,
        task_yaml: Path,
        fake_local: _FakeBackend,
    ) -> None:
        runner.invoke(app, ["compute", "submit", str(task_yaml)])
        result = runner.invoke(app, ["compute", "list"])
        assert result.exit_code == 0
        assert "job-123" in result.output
        assert "hello" in result.output


class TestBackendFactory:
    def test_local_backend(self) -> None:
        from forge.cli.compute import _make_backend  # pyright: ignore[reportPrivateUsage]
        from forge.compute.backends.local import LocalBackend

        backend = _make_backend("local", {})
        assert isinstance(backend, LocalBackend)

    def test_ssh_backend(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from forge.cli.compute import _make_backend  # pyright: ignore[reportPrivateUsage]
        from forge.compute.backends.ssh import SSHBackend

        # SSHBackend requires either a connection or host+username; we
        # pass a sentinel connection object to skip the asyncssh import.
        sentinel = object()
        backend = _make_backend("ssh", {"connection": sentinel})
        assert isinstance(backend, SSHBackend)

    def test_skypilot_backend(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from forge.cli.compute import _make_backend  # pyright: ignore[reportPrivateUsage]
        from forge.compute.backends.skypilot import SkyPilotBackend

        backend = _make_backend("skypilot", {"client": object()})
        assert isinstance(backend, SkyPilotBackend)

    def test_unknown_backend(self) -> None:
        import typer

        from forge.cli.compute import _make_backend  # pyright: ignore[reportPrivateUsage]

        with pytest.raises(typer.Exit):
            _make_backend("nope", {})
