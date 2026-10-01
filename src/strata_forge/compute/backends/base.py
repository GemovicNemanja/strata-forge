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

from strata_forge.compute.task import SECRETS_FILE_ENV
from strata_forge.core.errors import ForgeError

if TYPE_CHECKING:
    from collections.abc import Sequence

    from strata_forge.compute.job import ConsoleChunk, Job, JobStatus
    from strata_forge.compute.task import Task

__all__ = [
    "MAX_CONSOLE_CHUNK_BYTES",
    "MAX_READ_FILE_BYTES",
    "Backend",
    "CleanupError",
    "SubmitCleanupError",
    "safe_workdir_relpath",
    "secrets_guarded_script",
]

# The unexported shell variable that holds the secrets file's path for the trap. Unexported so
# no child of the job (the setup step's package installs above all) inherits a pointer to it.
_SECRETS_PATH_VAR = "_forge_secrets_file"

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


class SubmitCleanupError(RuntimeError):
    """A submit failed after creating remote state that it then could not remove.

    Raised from the submit's own failure (its ``__cause__``) when that failure is an ordinary
    exception; a cancellation is re-raised as itself. That state may hold the job's secrets file,
    and a submit that raises hands back no job, so :attr:`job` is the handle to remove it with:
    pass it to :meth:`Backend.cleanup` once the host is reachable again. It names no running
    process, so ``status``, ``cancel`` and the log readers do not apply to it.
    """

    def __init__(self, message: str, job: Job) -> None:
        super().__init__(message)
        self.job = job


class CleanupError(ForgeError):
    """:meth:`Backend.cleanup` ran, and the job's backend-side state is still there.

    The state may hold the job's secrets file, so a caller that sees this has NOT cleaned up and
    should try again later: the method is idempotent, and a retry against state that has since
    gone is a success. The message names the job and how the removal failed (an exit status,
    an exception type), never what the remote printed or what the state contains, so it is safe
    to store and show. A transport failure (the host is unreachable, the command timed out) is
    raised as itself, not as this: it says nothing about whether the state survived.
    """


def secrets_guarded_script(
    path_expr: str, setup: str, run: str, *, prelude: Sequence[str] = ()
) -> str:
    """The shell text that runs ``setup`` then ``run`` for a job whose secrets are in a file.

    ``path_expr`` is a shell word that evaluates to the file's absolute path. The script:

    - removes the file on exit through a trap in the OUTER shell, and runs ``prelude``, ``setup``
      and ``run`` in a subshell below it. A task's own commands may install an ``EXIT`` trap of
      their own (an orchestrator's bootstrap does, to stop a progress ticker), and a second
      ``trap ... EXIT`` in one shell REPLACES the first: in the same shell, that would silently
      drop the removal, and the file would outlive a failed setup or a cancellation. A trap set
      in the subshell cannot reach the outer one.
    - exports :data:`~strata_forge.compute.task.SECRETS_FILE_ENV` only between ``setup`` and
      ``run``, so the setup step's package installs never hold even the path to the file.

    The outer shell exits with the subshell's status, which the trap leaves untouched. bash runs
    an ``EXIT`` trap on a normal exit and on a fatal signal such as the SIGTERM a cancel sends;
    only SIGKILL skips it.
    """
    run_step = f'export {SECRETS_FILE_ENV}="${_SECRETS_PATH_VAR}" && {run}'
    body = f"{setup} && {run_step}" if setup else run_step
    return "\n".join(
        [
            f"{_SECRETS_PATH_VAR}={path_expr}",
            f"trap 'rm -f \"${_SECRETS_PATH_VAR}\"' EXIT",
            "(",
            *prelude,
            body,
            ")",
        ]
    )


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
        """Submit ``task`` for execution; return its :class:`Job` handle.

        ``task.secrets`` are written to a private (0600) file whose absolute path the job's
        ``run`` step finds in ``FORGE_SECRETS_FILE``, never into the job's environment, a command
        line, a script or the returned job's metadata. A backend with no such channel raises
        rather than falling back to the environment. A submit that fails after creating remote
        state it then cannot remove raises :class:`SubmitCleanupError`, whose ``job`` names that
        state for :meth:`cleanup`.
        """
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

        ``max_bytes`` is the TOTAL for the call and is split evenly between the two streams,
        clamped to :data:`MAX_CONSOLE_CHUNK_BYTES`; evenly, because stderr is where a failure
        announces itself and a chatty stdout must not be able to starve it. When more than that
        accumulated, the NEWEST bytes are returned and the shortfall is reported as
        ``dropped_bytes`` — a watcher wants where the run is now, and a silent gap would
        misrepresent a jump-cut as a whole log.

        A backend with no way to read a suffix of its logs raises :class:`NotImplementedError`
        rather than re-fetching the whole log per poll: that is linear in memory as well as in
        time, inside a process shared by every account.

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
        """Tear down any backend-side state associated with ``job``, and prove it is gone.

        Examples: remote temporary directories (SSH), SkyPilot
        clusters launched specifically for this job, captured log
        buffers. Idempotent — callers can invoke it repeatedly or
        on already-cleaned jobs; state that is already absent is a
        success.

        Returning means the state is gone, not that a removal was
        attempted: that state may hold the job's secrets file, and
        a caller that retries until cleanup succeeds can only do so
        if a failed one says so. When the backend could run the
        removal but the state survived it, this raises
        :class:`CleanupError`; a transport failure propagates as
        itself.
        """
        ...  # pragma: no cover — Protocol body
