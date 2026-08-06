"""Job and JobStatus shapes shared across all compute backends.

A :class:`Job` is the opaque handle a :class:`Backend` returns from
:meth:`Backend.submit`. Callers persist the job (it's frozen
Pydantic, JSON-serialisable) and pass it back to the backend's
``status`` / ``logs`` / ``cancel`` / ``cleanup`` methods.

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
