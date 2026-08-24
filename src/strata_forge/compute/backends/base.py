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
    from strata_forge.compute.job import ConsoleChunk, Job, JobStatus
    from strata_forge.compute.task import Task

__all__ = [
    "MAX_CONSOLE_CHUNK_BYTES",
    "MAX_READ_FILE_BYTES",
    "Backend",
    "safe_workdir_relpath",
]

# Hard cap on a single :meth:`Backend.read_file` result. A control plane polls
# read_file on an interval from a SHARED process; without a cap a runaway/malicious
# file (e.g. a huge progress.jsonl) could exhaust that process's memory and degrade
# service for every user. Backends bound the read to this many bytes.
MAX_READ_FILE_BYTES = 8 * 1024 * 1024

# Default ceiling on one :meth:`Backend.console` slice. Much smaller than the read_file cap
# because this one is polled continuously for the whole life of a run rather than read once:
# the budget that matters is per-interval, and a chunk that cannot be shown to a human in the
# time before the next one arrives is a chunk nobody reads.
MAX_CONSOLE_CHUNK_BYTES = 64 * 1024


def safe_workdir_relpath(path: str) -> str:
    """Validate ``path`` as a job-workdir-relative path and return it normalized.

    :meth:`Backend.read_file` reads files a job produced *inside its own working
    directory*. This guard is the security boundary: it rejects absolute paths and
    any ``..`` traversal so a caller (ultimately user/agent input on the control
    plane) can never read outside the job's workdir. Backends call it before
    composing the file path.

    This is a **lexical** check only — it does not resolve symlinks. A symlink
    *inside* the workdir that points outside would still be followed by the read.
    The :class:`LocalBackend` additionally resolves the realpath and re-confines it;
    the SSH backend reads on the user's own host with a fixed caller-supplied
    filename, so a symlink there only re-exposes the user's own files to themselves.
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

    async def console(
        self,
        job: Job,
        *,
        stdout_offset: int = 0,
        stderr_offset: int = 0,
        max_bytes: int = MAX_CONSOLE_CHUNK_BYTES,
    ) -> ConsoleChunk:
        """Read the console INCREMENTALLY: only what was appended past the given offsets.

        This is :meth:`logs` for a watcher rather than for a post-mortem. Pass the offsets
        from the previous :class:`ConsoleChunk` (zero on the first call) and receive exactly
        the bytes written since, so a caller polling on an interval stores a continuous
        transcript instead of a pile of overlapping tails.

        When more than ``max_bytes`` accumulated, the NEWEST ``max_bytes`` are returned and
        the shortfall is reported as ``dropped_bytes`` — a watcher wants where the run is
        now, and a silent gap would misrepresent a jump-cut as a whole log.

        Offsets are in BYTES of the underlying stream, so a slice may cut a multi-byte
        character; the boundary decodes to a replacement character rather than raising.
        Returns empty text (and the offsets unchanged) when nothing has been written yet.
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
