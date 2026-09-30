"""The VM-side plumbing every pipeline runner shares — and must not re-implement.

A runner in this package runs on a machine the control plane does not own, holding a real
Hugging Face write token, driven by a spec that arrived over the wire. Several of the things it
has to get right are identical whatever it is running, and each of them is a security or
reliability property rather than a convenience:

- **Scrubbing** (:func:`sanitize`). Every message a runner emits — an error, a phase caption —
  passes through one function that removes the write token and anything token-shaped. A second
  copy of this is how one copy stops being maintained.
- **Re-validating ids** (:func:`validate_repo_id`). The control plane allow-lists them, but the VM
  is the boundary that actually fetches and pushes, so it checks again.
- **The version handshake** (:func:`check_engine_version`, applied by :func:`load_config`). The
  spec names the engine version the control plane validated it against, and the runner refuses
  to execute under any other. The spec models set ``extra="forbid"``, so a field the installed
  engine does not know is already a loud failure; the handshake closes the other direction, an
  engine NEWER than the one that validated the spec, whose changed defaults or secret channel
  an old spec would silently miss.
- **Termination** (:func:`install_termination_handlers`). Turning SIGTERM into a cancellation is
  what stops a cancelled run from stranding a GPU; it belongs to every runner, not to whichever
  one needed it first.
- **The entry point** (:func:`runner_main`). Cancellation reported as cancellation, failure
  reported to both the progress file and stderr, the writer always closed.

Nothing here is inference- or training-specific, and nothing here imports a heavy dependency.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import re
import signal
import sys
from functools import partial
from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, cast

from pydantic import BaseModel

from strata_forge import __version__
from strata_forge.compute.serving import format_elapsed
from strata_forge.training.progress import JsonlProgressWriter, ProgressEvent

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Awaitable, Callable

    from strata_forge.training.hardware import GpuSampler
    from strata_forge.training.progress import RunStage


class PhaseSink(Protocol):
    """What :func:`phase_sink` returns, and what every phase-reporting hook accepts.

    Spelled as a Protocol rather than a ``Callable`` alias because ``stage`` is keyword-only:
    a bare ``phase("Loading the dataset")`` from a caller outside the runner stays valid, while
    a runner that knows its milestone can pass ``stage=`` without a second sink type.
    """

    def __call__(self, message: str, *, stage: RunStage | None = ...) -> None: ...


__all__ = [
    "ENGINE_DISTRIBUTION",
    "MAX_PHASE_CHARS",
    "PHASE_TICK_SECONDS",
    "REPO_ID_RE",
    "SAFE_NAME_RE",
    "PhaseSink",
    "RunError",
    "check_engine_version",
    "emit",
    "install_termination_handlers",
    "installed_engine_commit",
    "load_config",
    "phase_sink",
    "progress_path",
    "results_dir",
    "runner_main",
    "sanitize",
    "ticking_phase",
    "validate_repo_id",
]

# An HF repo id: ``owner/name`` OR a bare canonical name, each segment alphanumeric-led, no
# traversal/scheme/space. The canonical form is not an edge case — `gpt2`, `t5-small`,
# `distilgpt2` and `bert-base-uncased` all live at the root of the Hub with no owner.
REPO_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*(?:/[A-Za-z0-9][A-Za-z0-9._-]*)?$")
# A safe single path segment for an on-VM output dir name (no slash / traversal / shell chars).
SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")
# Scrub token-shaped substrings from any surfaced message (defense in depth on top of replacing
# the known token value).
_TOKEN_RE = re.compile(r"(hf_[A-Za-z0-9]{8,}|Bearer\s+[A-Za-z0-9._\-]+)")
# A phase message is a short human phrase. Capped because the sink is reachable from public API:
# a caller's hook must not be able to grow the file the orchestrator tails without bound.
MAX_PHASE_CHARS = 200
# How often a long uncountable phase re-stamps itself with its elapsed time. Matches the serving
# heartbeat, so one run does not narrate two different cadences.
PHASE_TICK_SECONDS = 10.0
# The distribution whose installed metadata answers "which engine commit is this VM running".
# The import package is ``strata_forge``; the distribution name is what ``pip`` and PEP 610 know.
ENGINE_DISTRIBUTION = "strata-forge"
# A full git commit id. The handshake compares whole ids, never a prefix: a short id the
# control plane happened to send would match more than one commit.
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


class RunError(Exception):
    """A runner failure whose message is safe to surface (already token-scrubbed)."""


def validate_repo_id(repo_id: str, what: str) -> str:
    """Defensively re-validate an id even though the server allow-listed it — the VM is the
    trust boundary that actually fetches/pushes."""
    # fullmatch (not match): match's `$` accepts a trailing newline ("org/x\n").
    if ".." in repo_id or not REPO_ID_RE.fullmatch(repo_id):
        msg = f"invalid {what} id"
        raise RunError(msg)
    return repo_id


def sanitize(text: str, token: str | None) -> str:
    """Strip the write token + any token-shaped substring from a message before it's emitted."""
    if token:
        text = text.replace(token, "***")
    return _TOKEN_RE.sub("***", text)


def installed_engine_commit(distribution: str = ENGINE_DISTRIBUTION) -> str | None:
    """The git commit the installed engine was built from, or ``None`` when there is none.

    Read from the distribution's PEP 610 ``direct_url.json``, which ``pip`` writes for a VCS
    install (``pip install git+https://...@<ref>``) and omits for an index install. A release
    from PyPI therefore has no commit, and so does an editable checkout (a ``dir_info`` URL): the
    handshake treats both as "not a pinned commit", never as a match.
    """
    try:
        raw = metadata.distribution(distribution).read_text("direct_url.json")
    except metadata.PackageNotFoundError:
        return None
    if not raw:
        return None
    try:
        info: object = json.loads(raw)
    except ValueError:
        return None
    if not isinstance(info, dict):
        return None
    vcs = cast("dict[str, object]", info).get("vcs_info")
    if not isinstance(vcs, dict):
        return None
    commit = cast("dict[str, object]", vcs).get("commit_id")
    return commit if isinstance(commit, str) and commit else None


def check_engine_version(expected: str | None) -> None:
    """Refuse to run under an engine other than the one the spec was validated against.

    ``expected`` is what the control plane wrote into the spec: ``None`` (no claim, accepted so
    a spec from before the handshake still runs), a plain ``"<version>"`` (the released engine
    it pinned on the VM) or ``"<version>+<commit>"`` (the exact commit its own bundled engine was
    built from, on a deployment that installs from a git ref rather than a release). The version
    half must equal the installed ``__version__``; the commit half must equal the installed
    distribution's PEP 610 commit id. An engine with no recorded commit (a release from PyPI, an
    editable checkout) cannot satisfy a commit claim at all, because the two would only ever
    agree by accident.

    The point of the check is a warm machine. Every commit of a development branch shares one
    ``__version__`` until a release bump, so a version-only comparison cannot see that the VM
    runs a commit older than the one that validated the spec; the commit half can.
    """
    if expected is None:
        return
    version, _, commit = expected.partition("+")
    installed = __version__
    if not version or version != installed:
        msg = (
            f"engine version mismatch: the spec was validated against strata-forge "
            f"{expected!r} but this machine runs {installed!r}"
        )
        raise RunError(msg)
    if not commit:
        return
    installed_commit = installed_engine_commit()
    if not _COMMIT_RE.fullmatch(commit) or installed_commit != commit:
        running = (
            f"{installed}+{installed_commit}" if installed_commit else f"{installed} (release)"
        )
        msg = (
            f"engine version mismatch: the spec was validated against strata-forge "
            f"{expected!r} but this machine runs {running!r}"
        )
        raise RunError(msg)


def load_config[SpecT: BaseModel](spec_cls: type[SpecT]) -> SpecT:
    """Parse ``STRATA_RUN_CONFIG`` into ``spec_cls`` and apply the engine version handshake.

    Parsed as DATA only: ``json`` plus Pydantic validation, never ``eval`` / ``pickle`` /
    ``yaml.unsafe_load``. The spec models set ``extra="forbid"``, so an unrecognised key is a
    loud failure rather than a silently ignored instruction.

    A spec that carries ``engine_version`` is then checked against the installed engine
    (:func:`check_engine_version`) before anything else reads it: a mismatch is the first and
    only thing the run reports, so a stale machine is diagnosed as such rather than through
    whatever the stale code did with the spec.
    """
    raw = os.environ.get("STRATA_RUN_CONFIG")
    if not raw:
        msg = "STRATA_RUN_CONFIG is not set"
        raise RunError(msg)
    try:
        spec = spec_cls.model_validate_json(raw)
    except ValueError as exc:
        msg = f"invalid STRATA_RUN_CONFIG: {exc}"
        raise RunError(msg) from exc
    expected = getattr(spec, "engine_version", None)
    if expected is not None and not isinstance(expected, str):
        msg = "invalid STRATA_RUN_CONFIG: engine_version must be a string"
        raise RunError(msg)
    check_engine_version(expected)
    return spec


def emit(writer: JsonlProgressWriter | None, event: ProgressEvent) -> None:
    if writer is not None:
        writer.emit(event)


def phase_sink(
    writer: JsonlProgressWriter | None,
    hf_token: str | None,
    *,
    gpu: GpuSampler | None = None,
) -> PhaseSink:
    """Build the one function every phase message goes through.

    A single choke point, so scrubbing is unconditional: the same sink is handed to library code
    whose phase hook is public API, and a phrase that came from outside the runner gets the
    treatment the error path already applies.

    ``stage`` is keyword-only and optional so the plain ``phase("...")` call an outside caller
    makes still type-checks; runners that know which milestone they are in pass it.

    When ``gpu`` is given, its counters ride every phase event. That matters most exactly here:
    loading a model or uploading results can take minutes during which nothing is countable, and
    the hardware gauges are the only thing left that still moves.
    """

    def _phase(message: str, *, stage: RunStage | None = None) -> None:
        emit(
            writer,
            ProgressEvent(
                kind="phase",
                stage=stage,
                message=sanitize(message, hf_token)[:MAX_PHASE_CHARS],
                metrics=gpu.sample() if gpu is not None else {},
            ),
        )

    return _phase


@contextlib.asynccontextmanager
async def ticking_phase(
    phase: PhaseSink | Callable[[str], None],
    message: str,
    interval_s: float = PHASE_TICK_SECONDS,
    *,
    stage: RunStage | None = None,
) -> AsyncGenerator[None]:
    """Report ``message`` for as long as the block runs, re-stamping it with its elapsed time.

    A one-shot phase says a step BEGAN and never that it is still going. "Installing the engine"
    and "Loading the dataset" then sit unchanged for minutes, indistinguishable from a run that
    has hung — which is the question anyone watching is actually asking.

    The ticker is an asyncio task, so it only ticks while the event loop is free: every blocking
    call it wraps is handed to a thread for exactly that reason. Cancelled in a ``finally``, so a
    step that raises does not leave a caption ticking forever underneath the error.
    """
    loop = asyncio.get_running_loop()
    started = loop.time()
    # Bind the stage once, and only when there is one: a sink is often a plain one-argument
    # callable (`serving.py`'s public `on_phase` hook, a bare `list.append` in a test), and
    # unconditionally passing `stage=` would break every one of them for a value they never asked
    # for. With no stage, this is exactly the call it always was.
    report = phase if stage is None else partial(phase, stage=stage)

    # Immediately, with no elapsed: zero is noise, and this marks the start. Every tick re-stamps
    # the same stage, so a consumer that loses one event still learns the stage from the next.
    report(message)

    async def _tick() -> None:
        while True:
            await asyncio.sleep(interval_s)
            report(f"{message} ({format_elapsed(loop.time() - started)})")

    task = asyncio.create_task(_tick())
    try:
        yield
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


def results_dir(run_id: str | None, *, name: str) -> Path:
    """A stable, cleanup-surviving location for a run's output: outside the per-run workdir the
    orchestrator deletes, named by the run id so the user can retrieve it over SSH.

    Used whether or not the output is then pushed. Writing it inside the workdir and pushing from
    there would mean a failed upload destroys the whole run's output along with it.

    Falls back to the cwd when no usable run id was provided (degraded — may be cleaned — but
    never crashes; the run id is validated as a single safe path segment).
    """
    if run_id and SAFE_NAME_RE.fullmatch(run_id):
        return Path.home() / name / run_id
    return Path.cwd()


def progress_path() -> str | None:
    """Resolve the progress file: the spec's progress_path (read defensively, the spec may be
    invalid) else FORGE_PROGRESS_PATH."""
    raw = os.environ.get("STRATA_RUN_CONFIG") or "{}"
    try:
        from_spec = json.loads(raw).get("progress_path")
    except ValueError, AttributeError:
        from_spec = None
    return from_spec or os.environ.get("FORGE_PROGRESS_PATH")


def install_termination_handlers() -> None:
    """Turn SIGTERM/SIGINT into an ordinary cancellation, so teardown actually runs.

    This is what stops a cancelled run from stranding its GPU. Cancelling a run signals the job's
    process group, which includes this process — and Python's default SIGTERM handling terminates
    immediately, without unwinding. Anything the runner started in a session of ITS OWN (a model
    server, a training subprocess) never sees that group signal: the only thing that stops it is
    the teardown in this process's ``finally`` blocks, and that never runs if the interpreter dies
    where it stands.

    Cancelling the running task raises `CancelledError` at the current await instead, so every
    `finally` on the stack unwinds. The caller's SIGKILL follows a grace period, which is the
    budget this teardown has.
    """
    task = asyncio.current_task()
    if task is None:  # pragma: no cover — runner_main always runs as a task under asyncio.run
        return
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        with contextlib.suppress(NotImplementedError, ValueError):
            # NotImplementedError: signal handlers are POSIX-only. ValueError: not the main
            # thread. Neither is worth failing a run over — the run simply keeps the default
            # behaviour it had before.
            loop.add_signal_handler(sig, task.cancel)


async def runner_main(
    execute: Callable[[JsonlProgressWriter | None, str | None], Awaitable[Any]],
) -> int:
    """Run one pipeline to completion and return a process exit code (0 ok, 1 failure).

    Owns the three things a runner's outcome depends on and none of its work: the write token is
    read here and never leaks (every message goes through :func:`sanitize`), a cancellation is
    reported as a cancellation rather than as a failure of the work, and the progress writer is
    closed whatever happens.
    """
    hf_token = os.environ.get("HF_WRITE_TOKEN") or None
    path = progress_path()
    writer = JsonlProgressWriter(path) if path else None
    install_termination_handlers()
    try:
        await execute(writer, hf_token)
    except asyncio.CancelledError:
        # Asked to stop. The `finally` blocks unwinding beneath this are the point — they are
        # what shut down whatever the runner started. Report it as a distinct outcome rather than
        # as a failure of the work, and do not re-raise: the exit code is the caller's answer.
        emit(writer, ProgressEvent(kind="error", message="run cancelled"))
        print("run cancelled", file=sys.stderr, flush=True)
        return 1
    except Exception as exc:  # top-level runner boundary: report + exit nonzero, never leak
        detail = sanitize(str(exc), hf_token)
        emit(writer, ProgressEvent(kind="error", message=detail))
        # Also to stderr, because that is where the control plane reads a failed run's reason
        # from. Catching the exception here means no traceback is printed, so without this the
        # run's own account of why it failed exists only in the progress file, and the record
        # explains the failure with whatever unrelated output happened to be last in the stream.
        print(f"run failed: {detail}", file=sys.stderr, flush=True)
        return 1
    else:
        return 0
    finally:
        if writer is not None:
            writer.close()
