"""Job and JobStatus shapes shared across all compute backends.

A :class:`Job` is the opaque handle a :class:`Backend` returns from
:meth:`Backend.submit`. Callers persist the job (it's frozen
Pydantic, JSON-serialisable) and pass it back to the backend's
``status`` / ``logs`` / ``cancel`` / ``cleanup`` methods.

:class:`ConsoleChunk` is what an incremental console read returns —
a slice of stdout/stderr plus the byte offsets to resume from.

:class:`JobStatus` collapses every backend's native lifecycle into
one five-state machine. Backend-specific nuance lives in the free-form
``message`` field — callers that need full fidelity can inspect it,
but day-to-day code only needs the discrete state.
"""

from __future__ import annotations

from datetime import datetime  # noqa: TC003 — Pydantic needs runtime resolution
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "ConsoleChunk",
    "Job",
    "JobState",
    "JobStatus",
]


type JobState = Literal[
    "pending",
    "running",
    "succeeded",
    "failed",
    "cancelled",
]
"""The five canonical job states every backend maps onto."""


class Job(BaseModel):
    """Opaque handle to a submitted :class:`Task`.

    Attributes:
        id: Backend-assigned identifier. Unique within the backend's
            namespace; callers that submit to multiple backends
            should also key on :attr:`backend`.
        backend: Name of the :class:`Backend` that owns this job.
        task_name: The :class:`Task.name` this job came from — handy
            for surfacing in logs / dashboards without re-fetching
            the task.
        metadata: Free-form backend-specific data (PID for SSH,
            SkyPilot cluster name, working directory path, …).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(min_length=1)
    backend: str = Field(min_length=1)
    task_name: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default={})


class JobStatus(BaseModel):
    """Canonical status of a :class:`Job`.

    Attributes:
        state: One of ``pending``, ``running``, ``succeeded``,
            ``failed``, ``cancelled``.
        exit_code: Process exit code when the job has terminated.
            ``None`` for jobs still ``pending`` or ``running``.
        started_at: When the job entered ``running``. ``None`` if
            the job is still ``pending`` or never started.
        finished_at: When the job left ``running``. ``None`` while
            still running.
        message: Free-form backend-specific detail. Useful for
            preserving nuance the five-state machine loses
            (SkyPilot's ``SETTING_UP``, ``STARTING``, …).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    state: JobState
    exit_code: int | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    message: str = ""

    @property
    def is_terminal(self) -> bool:
        """True when ``state`` is ``succeeded`` / ``failed`` / ``cancelled``."""
        return self.state in ("succeeded", "failed", "cancelled")


class ConsoleChunk(BaseModel):
    """One incremental slice of a job's console, plus where to resume.

    :meth:`Backend.logs` answers "what has this job printed?", which is the right
    question when a job has ended. A watcher following a LIVE job asks a different
    one — "what has it printed *since last time*?" — and a tail cannot answer it:
    the window overlaps arbitrarily with the previous read, and the natural way to
    de-duplicate (match on the last line seen) fails on exactly the output that
    makes a console worth watching, since a progress bar rewriting itself emits the
    same line over and over.

    Byte offsets answer it exactly. A caller passes back the offsets it received and
    gets precisely the bytes appended since, or an explicit count of what it missed.

    Attributes:
        stdout: New stdout text since ``stdout_offset`` was requested.
        stderr: New stderr text since ``stderr_offset`` was requested.
        stdout_offset: Byte offset to pass as ``stdout_offset`` on the next call.
        stderr_offset: Likewise for stderr.
        dropped_bytes: How many bytes were skipped because more output accumulated
            than one chunk may carry. Non-zero means the caller has a HOLE, not a
            continuous transcript — which is worth saying out loud rather than
            presenting a jump-cut as a whole log. The bytes kept are the NEWEST
            ones; a watcher wants the current state of the run, not its history.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    stdout: str = ""
    stderr: str = ""
    stdout_offset: int = Field(default=0, ge=0)
    stderr_offset: int = Field(default=0, ge=0)
    dropped_bytes: int = Field(default=0, ge=0)
