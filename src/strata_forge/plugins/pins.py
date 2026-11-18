"""Which plugins may run, pinned exactly, and whether a machine holds that exact build.

The **allow-list** is operator configuration: the control plane reads it from its own settings
and a launch request can only name an entry in it, never add one. Each entry pins a plugin
completely: the distribution and version a ``pip`` ``==`` pin resolves, and the source it was
built from at one full commit, either a git repository or a Hugging Face repository. A request
picks an allowed plugin and one of its declared entry points; it never carries a repository,
a commit, a path or an import string of its own.

Every pin names its artifact by content: ``files_sha256`` is the digest of the files the build
installs (:func:`files_digest`), for a git source and a Hugging Face wheel alike, and every
dependency the plugin needs beyond the engine's own is listed with its version and the SHA-256 of
each wheel that may be installed for it (:class:`LockedDependency`). Nothing outside the pin is
ever installed into a plugin's environment.

The **installed checks** run on the compute target, against the plugin's own environment, which
the runner builds from the pin (:mod:`strata_forge.plugins.environment`):
:func:`installed_mismatch` (the distribution and its pinned version are there) and
:func:`record_mismatch` (every file matches its RECORD hash and the files' digest is the pinned
``files_sha256``). They are repeated on every reuse of an environment and again by the host,
inside the sandbox, before any plugin module is imported.
"""

from __future__ import annotations

import base64
import csv
import hashlib
import json
import sys
from importlib import metadata
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Annotated, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from strata_forge.plugins.manifest import (
    ENTRY_POINT_GROUP,
    EntryPointDecl,
    PluginManifest,
    PluginManifestError,
    normalize_distribution,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = [
    "ALLOWED_PLUGINS_ENV",
    "AllowedPlugin",
    "Isolation",
    "LockedDependency",
    "PluginLimits",
    "PluginPin",
    "PluginSource",
    "RunnerAllowance",
    "allowed_pin",
    "distribution_files",
    "files_digest",
    "installed_mismatch",
    "manifest_mismatch",
    "parse_allow_list",
    "record_mismatch",
    "registered_entry_point",
]

#: The environment variable a control plane sets, from its OWN configuration, in the job that
#: runs a plugin: the JSON list of pins the runner may execute. Never derived from the spec.
ALLOWED_PLUGINS_ENV = "STRATA_ALLOWED_PLUGINS"

_SHA = r"^[0-9a-f]{40}$"
_REPO = r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}/[A-Za-z0-9][A-Za-z0-9._-]{0,95}$"
_SLUG = r"^[a-z0-9][a-z0-9-]{0,63}$"
_FILE = r"^[A-Za-z0-9][A-Za-z0-9._+-]{0,199}\.whl$"
_SHA256 = r"^[0-9a-f]{64}$"
_DIST = r"^[a-z0-9](?:[a-z0-9-]{0,98}[a-z0-9])?$"
# A PEP 440 version as an ``==`` pin writes it: no spaces, no operators, no markers.
_VERSION = r"^[A-Za-z0-9][A-Za-z0-9.+!_-]{0,63}$"


class LockedDependency(BaseModel):
    """One distribution a plugin needs beyond the engine's environment, pinned like a lock file
    with hashes: the exact version and the SHA-256 of every wheel that may be installed for it
    (one per platform the operator allows). The runner installs it with ``--require-hashes`` and
    ``--only-binary=:all:``: a wheel is unpacked, never run, so installing a dependency executes
    no code, and a file whose hash is not listed is refused."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: Annotated[str, StringConstraints(pattern=_DIST)]
    version: Annotated[str, StringConstraints(pattern=_VERSION)]
    sha256: tuple[Annotated[str, StringConstraints(pattern=_SHA256)], ...] = Field(
        min_length=1, max_length=64
    )

    def requirement_line(self) -> str:
        """The line in a ``pip --require-hashes`` requirements file."""
        hashes = " ".join(f"--hash=sha256:{h}" for h in self.sha256)
        return f"{self.name}=={self.version} {hashes}"


class PluginSource(BaseModel):
    """Where a plugin is built from, at one full commit.

    ``git`` names a GitHub repository (``owner/name``): the runner fetches the commit anonymously
    and builds a wheel from it INSIDE the sandbox, with no network and only the pinned build
    requirements. ``hf`` names a Hugging Face model repository holding a built wheel (``wheel``)
    at the commit, with the wheel's SHA-256 (``wheel_sha256``): the runner downloads that file
    and refuses it unless its hash is the pinned one.

    Both carry ``files_sha256``: the digest of the files the build installs (:func:`files_digest`),
    which the operator takes from a build they inspected. A commit pins the source, not what a
    build of it produces; a wheel's hash pins the file only until it is unpacked. The digest is
    what pins the artifact, and it is checked again every time an environment is reused.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["git", "hf"]
    repo: Annotated[str, StringConstraints(pattern=_REPO)]
    commit: Annotated[str, StringConstraints(pattern=_SHA)]
    #: The wheel file in an ``hf`` source; refused on a ``git`` source.
    wheel: Annotated[str, StringConstraints(pattern=_FILE)] | None = None
    #: The SHA-256 of the wheel in an ``hf`` source (lowercase hex); refused on a ``git`` source.
    wheel_sha256: Annotated[str, StringConstraints(pattern=_SHA256)] | None = None
    #: The digest of the installed files (:func:`files_digest`).
    files_sha256: Annotated[str, StringConstraints(pattern=_SHA256)]

    @model_validator(mode="after")
    def _wheel_matches_kind(self) -> PluginSource:
        if self.kind == "hf" and (self.wheel is None or self.wheel_sha256 is None):
            msg = "an hf plugin source names the wheel file it installs and that wheel's sha256"
            raise ValueError(msg)
        if self.kind == "git" and (self.wheel is not None or self.wheel_sha256 is not None):
            msg = "a git plugin source installs from the repository, not a wheel"
            raise ValueError(msg)
        if ".." in self.repo:
            msg = "invalid repository name"
            raise ValueError(msg)
        return self

    def install_url(self) -> str:
        """Where the plugin comes from: the git repository, or the wheel's URL at the commit."""
        if self.kind == "git":
            return f"https://github.com/{self.repo}.git"
        return f"https://huggingface.co/{self.repo}/resolve/{self.commit}/{self.wheel}"

    def describe(self) -> str:
        """The pinned build in words, for errors and evidence."""
        if self.kind == "git":
            return f"github.com/{self.repo} at {self.commit}"
        return f"{self.wheel} (sha256 {self.wheel_sha256}) from hf.co/{self.repo} at {self.commit}"

    def manifest_url(self) -> str:
        """Where the manifest is read from at the commit (metadata only, no checkout)."""
        if self.kind == "git":
            return f"https://raw.githubusercontent.com/{self.repo}/{self.commit}/strata-plugin.json"
        return f"https://huggingface.co/{self.repo}/resolve/{self.commit}/strata-plugin.json"


class PluginPin(BaseModel):
    """A plugin build, pinned completely: the artifact, everything installed beside it, and, for
    a git source, everything its build may use."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    distribution: Annotated[str, StringConstraints(pattern=_DIST)]
    version: Annotated[str, StringConstraints(min_length=1, max_length=64)]
    source: PluginSource
    #: Every distribution the plugin needs that the engine's environment does not provide,
    #: transitively, hash-pinned. Empty when the engine's environment is enough.
    dependencies: tuple[LockedDependency, ...] = Field(default=(), max_length=512)
    #: A git source's build backend and what it needs, hash-pinned; the build sees nothing else.
    build_requires: tuple[LockedDependency, ...] = Field(default=(), max_length=128)

    @model_validator(mode="after")
    def _locks(self) -> PluginPin:
        if self.source.kind != "git" and self.build_requires:
            msg = "build_requires apply to a git source only (an hf source is a built wheel)"
            raise ValueError(msg)
        if self.source.kind == "git" and not self.build_requires:
            msg = "a git plugin source pins its build backend (build_requires)"
            raise ValueError(msg)
        for name, locked in (
            ("dependencies", self.dependencies),
            ("build_requires", self.build_requires),
        ):
            names = [d.name for d in locked]
            if len(set(names)) != len(names):
                msg = f"{name} lists a distribution twice"
                raise ValueError(msg)
            if normalize_distribution(self.distribution) in {
                normalize_distribution(n) for n in names
            }:
                msg = f"{name} lists the plugin itself"
                raise ValueError(msg)
        return self


#: How plugin code is kept from the runner. ``uid``: a sandbox (bubblewrap, started with
#: passwordless sudo) in which the plugin runs as another, unprivileged uid; ``bwrap``: the same
#: sandbox as the runner's own uid. There is no unsandboxed mode: a process of the runner's own
#: uid outside a sandbox can rewrite the engine the runner imports, so a machine with no sandbox
#: does not run plugins.
type Isolation = Literal["uid", "bwrap"]


class PluginLimits(BaseModel):
    """Operator limits on one plugin run's process."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: Wall-clock limit; the CPU-time limit is set to the same number of seconds.
    max_runtime_seconds: int = Field(default=86_400, ge=60, le=7 * 86_400)
    #: Address-space limit, off by default: CUDA maps far more address space than it uses.
    max_memory_mb: int | None = Field(default=None, ge=256)
    #: The largest file the plugin may write.
    max_file_mb: int = Field(default=65_536, ge=1)
    max_open_files: int = Field(default=4096, ge=64, le=1_048_576)


class AllowedPlugin(BaseModel):
    """One operator allow-list entry: a pinned plugin and, optionally, which of its entry points
    may run (``None``: every entry point its manifest declares). ``limits`` bound the process."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: Annotated[str, StringConstraints(pattern=_SLUG)]
    label: Annotated[str, StringConstraints(min_length=1, max_length=120)]
    pin: PluginPin
    entry_points: tuple[Annotated[str, StringConstraints(max_length=64)], ...] | None = Field(
        default=None, max_length=32
    )
    limits: PluginLimits = Field(default_factory=PluginLimits)

    def allows(self, entry_point: str) -> bool:
        return self.entry_points is None or entry_point in self.entry_points


def parse_allow_list(raw: str | None) -> tuple[AllowedPlugin, ...]:
    """Parse an operator allow-list (a JSON list of :class:`AllowedPlugin`); empty when unset.

    Refuses a malformed list outright rather than dropping a bad entry: an operator who wrote an
    entry expects it to apply, and a silently skipped one is a different allow-list.
    """
    if raw is None or not raw.strip():
        return ()
    try:
        data: object = json.loads(raw)
    except ValueError as exc:
        msg = f"the plugin allow-list is not valid JSON: {exc}"
        raise PluginManifestError(msg) from exc
    if not isinstance(data, list):
        msg = "the plugin allow-list must be a JSON list"
        raise PluginManifestError(msg)
    try:
        entries = tuple(AllowedPlugin.model_validate(item) for item in cast("list[object]", data))
    except ValueError as exc:
        msg = f"invalid plugin allow-list entry: {exc}"
        raise PluginManifestError(msg) from exc
    ids = [e.id for e in entries]
    if len(set(ids)) != len(ids):
        msg = "the plugin allow-list repeats an id"
        raise PluginManifestError(msg)
    return entries


class RunnerAllowance(BaseModel):
    """What an orchestrator hands the runner, from its OWN configuration, for one job: the
    operator's allow-list entry and the entry points the manifest at the pinned commit declares
    (as the orchestrator read them). The runner takes the entry point's interface from here,
    never from the spec."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    plugin: AllowedPlugin
    entry_points: tuple[EntryPointDecl, ...] = Field(min_length=1, max_length=32)

    def entry(self, name: str) -> EntryPointDecl | None:
        if not self.plugin.allows(name):
            return None
        return next((e for e in self.entry_points if e.name == name), None)


def allowed_pin(pin: PluginPin, allowed: tuple[PluginPin, ...]) -> bool:
    """Whether ``pin`` is exactly one of the allowed pins (every field equal)."""
    return pin in allowed


def manifest_mismatch(manifest: PluginManifest, pin: PluginPin) -> str | None:
    """Why ``manifest`` (read at the pin's commit) does not describe ``pin``, or ``None``."""
    if manifest.distribution != normalize_distribution(pin.distribution):
        return (
            f"the manifest at {pin.source.commit[:12]} declares the distribution "
            f"{manifest.distribution!r}, not {pin.distribution!r}"
        )
    if manifest.version != pin.version:
        return (
            f"the manifest at {pin.source.commit[:12]} declares version {manifest.version!r}, "
            f"not {pin.version!r}"
        )
    return None


def _distribution(name: str, path: Sequence[str] | None) -> metadata.Distribution | None:
    """``name`` as installed on ``path`` (one environment's site directories), or on this
    interpreter's ``sys.path`` when ``path`` is ``None``."""
    if path is None:
        try:
            return metadata.distribution(name)
        except metadata.PackageNotFoundError:
            return None
    wanted = normalize_distribution(name)
    for dist in metadata.distributions(path=list(path)):
        if normalize_distribution(dist.metadata["Name"] or "") == wanted:
            return dist
    return None


def installed_mismatch(pin: PluginPin, *, path: Sequence[str] | None = None) -> str | None:
    """Why the distribution installed on ``path`` is not the pinned name and version, or ``None``.

    ``path`` is the plugin environment's site directories (``None``: this interpreter's own).
    Which artifact it is, :func:`record_mismatch` decides, from the files themselves: an install
    record (PEP 610) says where a build came from, which proves nothing about what is on disk.
    """
    dist = _distribution(pin.distribution, path)
    if dist is None:
        return f"{pin.distribution} is not installed in the plugin environment"
    version = dist.version
    if version != pin.version:
        return f"{pin.distribution} {version} is installed, but the run pins {pin.version}"
    return None


def registered_entry_point(
    pin: PluginPin, name: str, *, path: Sequence[str] | None = None
) -> metadata.EntryPoint | None:
    """The entry point ``name`` the PINNED distribution registers under the plugin group.

    Looked up in that distribution's own metadata, never across all installed packages: another
    distribution registering the same name (by accident or on purpose) cannot be picked up
    instead. Nothing is imported here; the caller loads it.
    """
    dist = _distribution(pin.distribution, path)
    if dist is None:
        return None
    for ep in dist.entry_points:
        if ep.group == ENTRY_POINT_GROUP and ep.name == name:
            return ep
    return None


# Files a dist-info directory holds that an installer writes, not the build: they differ between
# two installs of the same artifact, so they are left out of the digest (and their hashes, which
# pip does not record, are not checked).
_INSTALL_TIME_FILES = frozenset({"RECORD", "INSTALLER", "REQUESTED", "direct_url.json", "WHEEL"})


def _record_entries(dist: metadata.Distribution) -> list[tuple[str, str]] | None:
    """``(path, "sha256=<b64>")`` for every hashed file the RECORD lists inside the site
    directory, or ``None`` when there is no RECORD."""
    raw = dist.read_text("RECORD")
    if raw is None:
        return None
    entries: list[tuple[str, str]] = []
    for row in csv.reader(raw.splitlines()):
        if len(row) < 2 or not row[1]:
            continue
        path, digest = row[0], row[1]
        parts = PurePosixPath(path).parts
        if not parts or parts[0] == ".." or "__pycache__" in parts or path.endswith(".pyc"):
            continue
        if len(parts) == 2 and parts[0].endswith(".dist-info") and parts[1] in _INSTALL_TIME_FILES:
            continue
        entries.append((path, digest))
    return sorted(entries)


def files_digest(dist: metadata.Distribution) -> str | None:
    """The SHA-256 over the distribution's built files as its RECORD lists them (path and hash,
    sorted; install-time files, bytecode and anything outside the site directory left out).
    The same artifact installed twice has the same digest. ``None`` without a RECORD."""
    entries = _record_entries(dist)
    if entries is None:
        return None
    lines = "".join(f"{path},{digest}\n" for path, digest in entries)
    return hashlib.sha256(lines.encode("utf-8")).hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return "sha256=" + base64.urlsafe_b64encode(digest.digest()).rstrip(b"=").decode("ascii")


def record_mismatch(pin: PluginPin, *, path: Sequence[str] | None = None) -> str | None:
    """Why the installed files of the pinned distribution are not the pinned artifact, or
    ``None``: every hashed RECORD entry must match the file on disk, and the files' digest must
    equal the pin's ``files_sha256``."""
    dist = _distribution(pin.distribution, path)
    if dist is None:
        return f"{pin.distribution} is not installed in the plugin environment"
    entries = _record_entries(dist)
    if entries is None:
        return f"{pin.distribution} has no RECORD, so its files cannot be verified"
    for name, recorded in entries:
        if not recorded.startswith("sha256="):
            return f"{pin.distribution} records {name} without a sha256 hash"
        located = Path(str(dist.locate_file(name)))
        try:
            actual = _file_hash(located)
        except OSError:
            return f"{pin.distribution}'s file {name} is missing"
        if actual != recorded:
            return f"{pin.distribution}'s file {name} does not match its RECORD hash"
    expected = pin.source.files_sha256
    actual_digest = files_digest(dist)
    if actual_digest != expected:
        return (
            f"{pin.distribution}'s installed files have digest {actual_digest}, not the "
            f"pinned {expected}"
        )
    return None


def distribution_files(
    pin: PluginPin, *, path: Sequence[str] | None = None
) -> frozenset[Path] | None:
    """The resolved paths of every file the pinned distribution installed on ``path``."""
    dist = _distribution(pin.distribution, path)
    if dist is None or dist.files is None:
        return None
    return frozenset(Path(str(dist.locate_file(f))).resolve() for f in dist.files)


def main(argv: list[str] | None = None) -> int:
    """``python -m strata_forge.plugins.pins digest <distribution>``: print the files digest an
    allow-list entry pins, for a build installed into this interpreter's environment."""
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 2 or args[0] != "digest":
        print("usage: python -m strata_forge.plugins.pins digest <distribution>", file=sys.stderr)
        return 2
    dist = _distribution(args[1], None)
    digest = files_digest(dist) if dist is not None else None
    if digest is None:
        print(f"{args[1]} is not installed here, or has no RECORD", file=sys.stderr)
        return 1
    print(digest)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
