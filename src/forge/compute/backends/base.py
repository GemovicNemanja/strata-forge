"""The :class:`Backend` Protocol — the contract every backend satisfies.

Async by design — production backends (SkyPilot, SSH) hit a network
in nearly every method. In-process backends (:class:`LocalBackend`)
still implement ``async def`` for shape uniformity.

See [ADR 0013](../../../../docs/architecture/adr/0013-compute-task-and-backend-shapes.md)
for the lifecycle decision and why every method is mandatory.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from forge.compute.job import Job, JobStatus
    from forge.compute.task import Task

__all__ = ["Backend"]


@runtime_checkable
class Backend(Protocol):
    """The contract every compute backend satisfies."""

    @property
    def name(self) -> str:
        """A short identifier for this backend (``"local"``, ``"ssh"``, ``"skypilot"``)."""
        ...  # pragma: no cover — Protocol body

    async def submit(self, task: Task) -> Job:
        """Submit ``task`` for execution; return its :class:`Job` handle."""
        ...  # pragma: no cover — Protocol body

    async def status(self, job: Job) -> JobStatus:
        """Return the current :class:`JobStatus` of ``job``."""
        ...  # pragma: no cover — Protocol body

    async def logs(self, job: Job, *, tail: int | None = None) -> str:
        """Return the captured stdout/stderr of ``job``.

        When ``tail`` is set, return only the last ``tail`` lines.
        """
        ...  # pragma: no cover — Protocol body

    async def cancel(self, job: Job) -> None:
        """Stop ``job`` if it's still running.

        Safe to call on terminal jobs (no-op).
        """
        ...  # pragma: no cover — Protocol body

    async def cleanup(self, job: Job) -> None:
        """Tear down any backend-side state associated with ``job``.

        Examples: remote temporary directories (SSH), SkyPilot
        clusters launched specifically for this job, captured log
        buffers. Idempotent — callers can invoke it repeatedly or
        on already-cleaned jobs.
        """
        ...  # pragma: no cover — Protocol body
