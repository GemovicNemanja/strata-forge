"""The VM-side plumbing every pipeline runner shares — and must not re-implement.

A runner in this package runs on a machine the control plane does not own, holding a real
Hugging Face write token, driven by a spec that arrived over the wire. Several of the things it
has to get right are identical whatever it is running, and each of them is a security or
reliability property rather than a convenience:

- **Secrets** (:func:`load_secrets`). Credentials arrive in a private file the backend wrote
  beside the job, never in the environment; the runner reads it and deletes it before it does
  anything else, so the file exists only until the run starts. The values travel as
  :class:`~pydantic.SecretStr` in a :class:`RunSecrets` and are revealed only at the call that
  needs them. Every Hub read passes :meth:`RunSecrets.hub_credential` explicitly, so a run never
  falls back to a credential the machine happens to hold.
- **Scrubbing** (:func:`run_redactor`, :func:`sanitize`). Every message a runner emits — an
  error, a phase caption, a per-row error it writes into its results — passes through the one
  :class:`~strata_forge.core.redact.Redactor`, which removes the write token in every encoding
  and anything credential-shaped. A second copy of this is how one copy stops being maintained.
- **The model-server environment** (:func:`model_server_environ`). A server the runner starts
  gets an allow-listed environment, not the runner's own.
- **Re-validating ids** (:func:`validate_repo_id`). The control plane allow-lists them, but the VM
  is the boundary that actually fetches and pushes, so it checks again.
- **The version handshake** (:func:`check_engine_version`, applied by :func:`load_config`). The
  spec names the engine version the control plane validated it against, and the runner refuses
  to execute under any other. ``extra="forbid"`` on the spec models only catches an OLDER engine
  when the newer spec carries a field it does not know; a behaviour change on the same spec
  shape (where the write token travels, what the scrubber removes, a default) reaches a warm
  machine's older engine with no spec error at all, and that is what the handshake catches.
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
import stat
import sys
from functools import partial
from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol, cast

from pydantic import BaseModel, ConfigDict, SecretStr

from strata_forge import __version__
from strata_forge.compute.serving import format_elapsed
from strata_forge.compute.task import SECRETS_FILE_ENV, SECRETS_FILE_NAME
from strata_forge.core.redact import Redactor
from strata_forge.pipelines import HF_TOKEN_SECRET
from strata_forge.training.progress import JsonlProgressWriter, ProgressEvent

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Awaitable, Callable, Mapping

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
    "HF_TOKEN_SECRET",
    "LEGACY_TOKEN_ENV",
    "LEGACY_TOKEN_MESSAGE",
    "MAX_PHASE_CHARS",
    "MAX_SECRETS_FILE_BYTES",
    "PHASE_TICK_SECONDS",
    "REPO_ID_RE",
    "REQUIRE_ENGINE_VERSION_ENV",
    "SAFE_NAME_RE",
    "UNCHECKED_ENGINE_MESSAGE",
    "PhaseSink",
    "RunError",
    "RunSecrets",
    "check_engine_version",
    "emit",
    "engine_version_required",
    "install_termination_handlers",
    "installed_engine_commit",
    "installed_engine_version",
    "load_config",
    "load_secrets",
    "model_server_environ",
    "phase_sink",
    "progress_path",
    "results_dir",
    "run_redactor",
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
# Set (to anything but empty / "0" / "false") by an orchestrator that stamps every spec, which
# turns a spec with no ``engine_version`` from an accepted-with-warning launch into a refusal.
# The transition for a control plane from before the handshake is bounded by this switch, not
# by hoping: a rolled-back or buggy control plane that sends ``null`` would otherwise run a
# newer engine unchecked, which is the skew the handshake exists to close.
REQUIRE_ENGINE_VERSION_ENV = "FORGE_REQUIRE_ENGINE_VERSION"
# What the run record says when a spec with no claim is accepted: the launch ran unchecked,
# and anyone reading the record after a behaviour skew needs that fact next to the outcome.
UNCHECKED_ENGINE_MESSAGE = (
    "spec carries no engine_version: the installed engine was not checked against "
    "the one that validated the spec"
)
# The spec's claim is echoed back in the mismatch message; a control plane's value is short,
# and a longer one would only bloat the error event and stderr line the record keeps.
_MAX_ECHOED_CLAIM_CHARS = 100
# The keys a runner reads from its secrets file. Anything else is refused by name: a key this
# engine does not know means the orchestrator was built against a different contract.
_KNOWN_SECRETS = frozenset({HF_TOKEN_SECRET})
# A handful of tokens is a few hundred bytes. The bound keeps a malformed or hostile file from
# being read into memory whole.
MAX_SECRETS_FILE_BYTES = 64 * 1024
# The environment variable older orchestrators put the write token in. Read only when no secrets
# file is configured, and removed from this process's environment once read.
LEGACY_TOKEN_ENV = "HF_WRITE_TOKEN"  # noqa: S105 — a variable name, not a credential
LEGACY_TOKEN_MESSAGE = (
    f"the write token arrived in the {LEGACY_TOKEN_ENV} environment variable; that delivery is "
    f"deprecated, and strata-forge 0.5.0 reads the token only from {SECRETS_FILE_ENV}"
)
# What a model server the runner starts may inherit from the runner's environment: what it needs
# to find its interpreter, libraries, GPUs, caches and locale, and nothing else. Exact names, then
# prefixes for the families whose members are all configuration (CUDA_VISIBLE_DEVICES, NCCL_*
# transport knobs a multi-GPU box may need, the VLLM_* settings).
_SERVER_ENV_NAMES = frozenset(
    {
        "PATH",
        "HOME",
        "USER",
        "LOGNAME",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "TMPDIR",
        "LD_LIBRARY_PATH",
        "XDG_CACHE_HOME",
        "HF_HOME",
        "HF_HUB_CACHE",
        "TRANSFORMERS_CACHE",
        "PYTHONUNBUFFERED",
    }
)
_SERVER_ENV_PREFIXES = ("CUDA_", "NVIDIA_", "NCCL_", "VLLM_")
# Dropped even when a prefix admits it: a name that says it holds a credential is not
# configuration (VLLM_API_KEY would also make the local endpoint demand a key the runner's
# client never sends).
_CREDENTIAL_NAME_RE = re.compile(r"TOKEN|KEY|SECRET|PASSWORD|CREDENTIAL", re.IGNORECASE)


class RunError(Exception):
    """A runner failure whose message is safe to surface (already token-scrubbed)."""


class RunSecrets(BaseModel):
    """The credentials a runner received, as :class:`~pydantic.SecretStr`.

    ``from_environment`` records that the token came through the deprecated
    :data:`LEGACY_TOKEN_ENV` rather than a secrets file, so the run can say so.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", hide_input_in_errors=True)

    hf_token: SecretStr | None = None
    from_environment: bool = False

    def hf_token_value(self) -> str | None:
        """The token in plaintext, for the one call that needs it (and the scrubber)."""
        return self.hf_token.get_secret_value() if self.hf_token is not None else None

    def hub_credential(self) -> str | Literal[False]:
        """The ``token=`` argument for a Hub read: the delivered token, or ``False``.

        Never ``None``: every Hugging Face library reads ``None`` as "use whatever credential this
        machine has" (an ``HF_TOKEN`` variable, a cached login, forge's own settings), and a run
        reads the Hub with exactly the credential the control plane delivered, or with none.
        """
        token = self.hf_token_value()
        return token if token else False


def load_secrets() -> RunSecrets:
    """Read the run's secrets file, delete it, and return its contents as :class:`RunSecrets`.

    The file is named by :data:`~strata_forge.compute.task.SECRETS_FILE_ENV`, which the backend
    sets to an absolute path ending in ``.secrets.json``; a value of any other shape is refused
    before anything touches it, because this function deletes what it is pointed at. It is
    opened without following a symlink, must be a regular file owned by this user with no group
    or other permission bits, and is unlinked whatever the outcome of the read, before this
    returns — so before the runner makes any network call or starts any subprocess. A missing,
    unreadable, oversized or malformed file is a named :class:`RunError` whose message never
    carries the file's contents.

    With no secrets file configured, the token is read from :data:`LEGACY_TOKEN_ENV` for
    orchestrators that still deliver it that way (``from_environment`` is then set), and that
    variable is removed from this process's environment so no child inherits it. When a file IS
    configured the environment variable is never read: an orchestrator that delivers by file
    cannot be steered back to the environment by a stray variable.
    """
    configured = os.environ.get(SECRETS_FILE_ENV, "")
    if not configured:
        legacy = os.environ.pop(LEGACY_TOKEN_ENV, "")
        if not legacy:
            return RunSecrets()
        return RunSecrets(hf_token=SecretStr(legacy), from_environment=True)
    os.environ.pop(LEGACY_TOKEN_ENV, None)
    return _parse_secrets(_read_and_unlink(Path(configured)))


def _read_and_unlink(path: Path) -> bytes:
    if not path.is_absolute() or path.name != SECRETS_FILE_NAME:
        msg = f"{SECRETS_FILE_ENV} must be an absolute path to a {SECRETS_FILE_NAME} file"
        raise RunError(msg)
    try:
        try:
            # O_NONBLOCK: opening a FIFO for reading otherwise blocks until a writer appears,
            # which would hang the run before the regular-file check below could refuse it.
            fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        except FileNotFoundError:
            msg = f"the secrets file {path} does not exist (already read, or never written)"
            raise RunError(msg) from None
        except OSError as exc:
            msg = f"the secrets file {path} could not be opened ({exc.strerror})"
            raise RunError(msg) from None
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode):
                msg = f"the secrets file {path} is not a regular file"
                raise RunError(msg)
            if info.st_uid != os.geteuid() or info.st_mode & 0o077:
                msg = (
                    f"the secrets file {path} must be owned by this user and readable by no one "
                    f"else (mode {stat.S_IMODE(info.st_mode):o})"
                )
                raise RunError(msg)
            raw = os.read(fd, MAX_SECRETS_FILE_BYTES + 1)
        finally:
            os.close(fd)
    finally:
        try:
            path.unlink(missing_ok=True)
        except OSError as exc:
            msg = f"the secrets file {path} could not be removed ({exc.strerror})"
            raise RunError(msg) from None
    if len(raw) > MAX_SECRETS_FILE_BYTES:
        msg = f"the secrets file is larger than {MAX_SECRETS_FILE_BYTES} bytes"
        raise RunError(msg)
    return raw


def _parse_secrets(raw: bytes) -> RunSecrets:
    # `from None` throughout: a decode error keeps the whole document on the exception, and a
    # chained cause is one traceback print away from a log.
    try:
        data: object = json.loads(raw.decode("utf-8"))
    except ValueError:  # UnicodeDecodeError and JSONDecodeError both subclass it
        msg = "the secrets file is not valid JSON"
        raise RunError(msg) from None
    if not isinstance(data, dict):
        msg = "the secrets file must hold a JSON object"
        raise RunError(msg)
    entries = cast("dict[object, object]", data)
    if not all(isinstance(k, str) and isinstance(v, str) for k, v in entries.items()):
        msg = "the secrets file must map names to string values"
        raise RunError(msg)
    values = cast("dict[str, str]", entries)
    unknown = set(values) - _KNOWN_SECRETS
    if unknown:
        # Counted, not named: the names come from a file, unvalidated and of any length.
        msg = (
            f"the secrets file carries {len(unknown)} key(s) this engine does not read; it "
            f"reads only {sorted(_KNOWN_SECRETS)!r} (an orchestrator built for a different "
            f"strata-forge?)"
        )
        raise RunError(msg)
    token = values.get(HF_TOKEN_SECRET) or None
    return RunSecrets(hf_token=SecretStr(token) if token else None)


def model_server_environ(environ: Mapping[str, str] | None = None) -> dict[str, str]:
    """The allow-listed slice of ``environ`` (default: this process's) a model server may see.

    A server the runner starts is third-party code serving a model a user chose; it gets what it
    needs to run and nothing it could leak. The runner's own environment may hold the
    orchestrator's spec, the secrets file's path, and on an older orchestrator the write token
    itself, and none of that is the server's business.
    """
    source = os.environ if environ is None else environ
    return {
        name: value
        for name, value in source.items()
        if (name in _SERVER_ENV_NAMES or name.startswith(_SERVER_ENV_PREFIXES))
        and not _CREDENTIAL_NAME_RE.search(name)
    }


def validate_repo_id(repo_id: str, what: str) -> str:
    """Defensively re-validate an id even though the server allow-listed it — the VM is the
    trust boundary that actually fetches/pushes."""
    # fullmatch (not match): match's `$` accepts a trailing newline ("org/x\n").
    if ".." in repo_id or not REPO_ID_RE.fullmatch(repo_id):
        msg = f"invalid {what} id"
        raise RunError(msg)
    return repo_id


def run_redactor(token: str | None) -> Redactor:
    """The redactor for one run: its write token, when it has one, plus every credential shape.

    Raises :class:`~strata_forge.core.errors.ValidationError` for a token too short to redact
    safely: such a run must fail rather than emit text the token could hide in.
    """
    return Redactor([token] if token else [])


def sanitize(text: str, token: str | None) -> str:
    """Strip the write token + anything credential-shaped from a message before it's emitted.

    A convenience over :func:`run_redactor` for a single message; a caller scrubbing many builds
    the redactor once.
    """
    return run_redactor(token).redact(text)


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
    # ``read_text`` swallows only a missing file; a corrupt one (undecodable bytes) or an
    # unreadable one (any other OSError) would otherwise escape as a raw exception and the run
    # would fail with a generic reason instead of the named mismatch.
    except OSError, ValueError:
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


def installed_engine_version(distribution: str = ENGINE_DISTRIBUTION) -> str | None:
    """The version the installed distribution's metadata records, or ``None`` when none is installed.

    This is the version an orchestrator's ``==`` pin resolved against, which is not necessarily
    the version of the code that is executing: a shadowed import (``PYTHONPATH``, a stale
    ``.pth`` entry, user-site over the venv) runs one copy while ``pip`` describes another, and
    a ``pyproject.toml`` bump that missed ``__init__.py`` makes even a clean install describe
    itself two ways. The handshake compares the spec against both, so neither drift can pass.
    """
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return None


def _mismatch(expected: str, running: str) -> RunError:
    # The claim is control-plane-authored and scrubbed like every message, so echoing it is
    # safe; it is capped because the record should not carry an arbitrarily long string twice.
    if len(expected) > _MAX_ECHOED_CLAIM_CHARS:
        expected = expected[:_MAX_ECHOED_CLAIM_CHARS] + "..."
    return RunError(
        f"engine version mismatch: the spec was validated against strata-forge "
        f"{expected!r} but this machine runs {running!r}"
    )


def engine_version_required() -> bool:
    """Whether a spec with no ``engine_version`` is refused rather than accepted with a warning."""
    return os.environ.get(REQUIRE_ENGINE_VERSION_ENV, "").strip().lower() not in {"", "0", "false"}


def check_engine_version(expected: str | None) -> str | None:
    """Refuse to run under an engine other than the one the spec was validated against.

    ``expected`` is what the control plane wrote into the spec: a plain ``"<version>"`` (the
    released engine it pinned on the VM) or ``"<version>+<commit>"`` (the exact commit its own
    bundled engine was built from, on a deployment that installs from a git ref rather than a
    release). The version half must equal both the executing ``__version__`` and the version
    the installed distribution's metadata records (the one a pin resolves against), compared as
    strings: ``0.3`` and ``0.3.0.post0`` are not ``0.3.0``, because a normalising compare would
    let a version the control plane never validated against pass. The commit half, when there
    is a ``+`` at all, must be a full lowercase git id equal to the installed distribution's
    PEP 610 commit id; a ``+`` followed by anything else is a malformed claim and is refused,
    never read as "no commit". An engine with no recorded commit (a release from PyPI, an
    editable checkout) cannot satisfy a commit claim at all, because the two would only ever
    agree by accident.

    ``None`` makes no claim. It is accepted, and the warning the caller must put in the run
    record is returned, unless :data:`REQUIRE_ENGINE_VERSION_ENV` is set, when it is refused
    like any other mismatch. Accepting it is the transition for a control plane from before the
    handshake, whose specs carry no version; the switch is how a control plane that stamps every
    spec closes the transition on its own machines without waiting for a release. The warning
    is returned rather than emitted because this function has no writer: it is public API, and
    what a caller does with the fact that a launch ran unchecked is the caller's.

    The point of the check is a warm machine. Every commit of a development branch shares one
    ``__version__`` until a release bump, so a version-only comparison cannot see that the VM
    runs a commit older than the one that validated the spec; the commit half can.
    """
    if expected is None:
        if engine_version_required():
            msg = (
                f"engine version mismatch: the spec carries no engine_version and "
                f"{REQUIRE_ENGINE_VERSION_ENV} is set on this machine"
            )
            raise RunError(msg)
        return UNCHECKED_ENGINE_MESSAGE
    version, plus, commit = expected.partition("+")
    executing = __version__
    recorded = installed_engine_version()
    if recorded is not None and recorded != executing:
        raise _mismatch(expected, f"{executing} (installed as {recorded})")
    if not version or version != executing:
        raise _mismatch(expected, executing)
    if not plus:
        return None
    installed_commit = installed_engine_commit()
    if not _COMMIT_RE.fullmatch(commit) or installed_commit != commit:
        running = (
            f"{executing}+{installed_commit}" if installed_commit else f"{executing} (release)"
        )
        raise _mismatch(expected, running)
    return None


def load_config[SpecT: BaseModel](
    spec_cls: type[SpecT], *, writer: JsonlProgressWriter | None = None
) -> SpecT:
    """Parse ``STRATA_RUN_CONFIG`` into ``spec_cls`` and apply the engine version handshake.

    Parsed as DATA only: ``json`` plus Pydantic validation, never ``eval`` / ``pickle`` /
    ``yaml.unsafe_load``. The spec models set ``extra="forbid"``, so an unrecognised key is a
    loud failure rather than a silently ignored instruction.

    The spec's ``engine_version`` is checked against the installed engine
    (:func:`check_engine_version`) BEFORE the model validates the rest, straight off the parsed
    JSON: the realistic skew is a newer control plane sending both a field this engine does not
    know and a version it does not match, and validating first would report the unknown field
    and blame the spec. A mismatch is the first and only thing the run reports, so a stale
    machine is diagnosed as such rather than through whatever the stale code did with the spec.

    A spec with no claim that the handshake accepts is recorded: a ``phase`` event saying the
    launch ran unchecked goes to ``writer`` (and the line to stderr, where the control plane
    reads a run's account of itself), so the record of a run that later misbehaved shows the
    engine was never checked. Runners pass the writer :func:`runner_main` hands them; a caller
    with none still gets the stderr line.

    Every runner spec must declare the field: a spec class without it is a bug in the runner
    (``TypeError``), not a spec that opted out of the handshake.
    """
    if "engine_version" not in spec_cls.model_fields:
        msg = f"{spec_cls.__name__} does not declare engine_version"
        raise TypeError(msg)
    raw = os.environ.get("STRATA_RUN_CONFIG")
    if not raw:
        msg = "STRATA_RUN_CONFIG is not set"
        raise RunError(msg)
    try:
        data: object = json.loads(raw)
    except ValueError as exc:
        msg = f"invalid STRATA_RUN_CONFIG: {exc}"
        raise RunError(msg) from exc
    if not isinstance(data, dict):
        msg = "invalid STRATA_RUN_CONFIG: the spec must be a JSON object"
        raise RunError(msg)
    expected = cast("dict[str, object]", data).get("engine_version")
    if expected is not None and not isinstance(expected, str):
        msg = "invalid STRATA_RUN_CONFIG: engine_version must be a string"
        raise RunError(msg)
    warning = check_engine_version(expected)
    if warning is not None:
        emit(writer, ProgressEvent(kind="phase", message=warning[:MAX_PHASE_CHARS]))
        print(f"warning: {warning}", file=sys.stderr, flush=True)
    try:
        return spec_cls.model_validate(data)
    except ValueError as exc:
        msg = f"invalid STRATA_RUN_CONFIG: {exc}"
        raise RunError(msg) from exc


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

    Redaction runs before the cap, so a clip can never leave half a token that the redactor
    would have recognised whole.
    """
    redactor = run_redactor(hf_token)

    def _phase(message: str, *, stage: RunStage | None = None) -> None:
        emit(
            writer,
            ProgressEvent(
                kind="phase",
                stage=stage,
                message=redactor.redact(message)[:MAX_PHASE_CHARS],
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
    execute: Callable[[JsonlProgressWriter | None, RunSecrets], Awaitable[Any]],
) -> int:
    """Run one pipeline to completion and return a process exit code (0 ok, 1 failure).

    Owns the things a runner's outcome depends on and none of its work: the secrets are loaded
    (and their file deleted) here, first, and never leak (every message goes through
    :func:`run_redactor`); a cancellation is reported as a cancellation rather than as a failure
    of the work; and the progress writer is closed whatever happens. A token that arrived through
    the deprecated environment variable is reported as such, once, before the work starts.

    A token too short to redact fails the run before any work starts: everything the run would
    print could carry it. The refusal itself is scrubbed by the credential shapes alone.
    """
    path = progress_path()
    writer = JsonlProgressWriter(path) if path else None
    install_termination_handlers()
    redactor = Redactor()
    try:
        secrets = load_secrets()
        redactor = run_redactor(secrets.hf_token_value())
        if secrets.from_environment:
            emit(writer, ProgressEvent(kind="phase", message=LEGACY_TOKEN_MESSAGE))
            print(f"warning: {LEGACY_TOKEN_MESSAGE}", file=sys.stderr, flush=True)
        await execute(writer, secrets)
    except asyncio.CancelledError:
        # Asked to stop. The `finally` blocks unwinding beneath this are the point — they are
        # what shut down whatever the runner started. Report it as a distinct outcome rather than
        # as a failure of the work, and do not re-raise: the exit code is the caller's answer.
        emit(writer, ProgressEvent(kind="error", message="run cancelled"))
        print("run cancelled", file=sys.stderr, flush=True)
        return 1
    except Exception as exc:  # top-level runner boundary: report + exit nonzero, never leak
        detail = redactor.redact(str(exc))
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
