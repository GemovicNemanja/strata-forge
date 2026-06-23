"""The :class:`Backend` Protocol — the contract every backend satisfies.

Async by design — production backends (SkyPilot, SSH) hit a network
in nearly every method. In-process backends (:class:`LocalBackend`)
still implement ``async def`` for shape uniformity.

See [ADR 0013](../../../../docs/architecture/adr/0013-compute-task-and-backend-shapes.md)
for the lifecycle decision and why every method is mandatory.
"""

from __future__ import annotations

import posixpath
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from forge.compute.job import Job, JobStatus
    from forge.compute.task import Task

__all__ = ["Backend", "safe_workdir_relpath"]


def safe_workdir_relpath(path: str) -> str:
    """Validate ``path`` as a job-workdir-relative path and return it normalized.

    :meth:`Backend.read_file` reads files a job produced *inside its own working
    directory*. This guard is the security boundary: it rejects absolute paths and
    any ``..`` traversal so a caller (ultimately user/agent input on the control
    plane) can never read outside the job's workdir. Backends call it before
    composing the file path.
    """
    if not path or path.startswith("/") or "\x00" in path or "\\" in path:
        err = f"read_file: path must be a non-empty workdir-relative path; got {path!r}"
        raise ValueError(err)
    normalized = posixpath.normpath(path)
    if normalized == ".." or normalized.startswith(("../", "/")):
        err = f"read_file: path must stay within the job workdir; got {path!r}"
        raise ValueError(err)
    return normalized


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

    async def read_file(self, job: Job, path: str, *, tail: int | None = None) -> str:
        """Read a text file ``job`` produced inside its working directory.

        ``path`` is interpreted RELATIVE to the job's workdir and must stay
        within it (no absolute paths, no ``..`` — see :func:`safe_workdir_relpath`);
        backends raise ``ValueError`` otherwise. When ``tail`` is set, return only
        the last ``tail`` lines. Returns ``""`` when the file does not exist yet (a
        not-yet-written progress file is not an error).

        This complements :meth:`logs` (stdout/stderr): it reads a side-channel file
        such as a runner's ``progress.jsonl`` of structured metric events.
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
