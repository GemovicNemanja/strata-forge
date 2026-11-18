"""Which plugins may run, pinned exactly, and whether a machine holds that exact build.

The **allow-list** is operator configuration: the control plane reads it from its own settings
and a launch request can only name an entry in it, never add one. Each entry pins a plugin
completely: the distribution and version a ``pip`` ``==`` pin resolves, and the source it was
built from at one full commit, either a git repository or a Hugging Face repository. A request
picks an allowed plugin and one of its declared entry points; it never carries a repository,
a commit, a path or an import string of its own.

The **installed check** (:func:`installed_mismatch`) runs on the compute target, where the plugin
was installed by its owner (into the engine's own interpreter, with an exact pin). It compares the
installed distribution's metadata with the pin: the version must be the pinned one, and PEP 610's
``direct_url.json`` must record the pinned source at the pinned commit. An index install records
no commit and an editable checkout records a directory, so neither can pass: the only build that
matches is the one built from the pinned commit.
"""

from __future__ import annotations

import json
from importlib import metadata
from typing import Annotated, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from strata_forge.plugins.manifest import (
    ENTRY_POINT_GROUP,
    PluginManifest,
    PluginManifestError,
    normalize_distribution,
)

__all__ = [
    "ALLOWED_PLUGINS_ENV",
    "AllowedPlugin",
    "PluginPin",
    "PluginSource",
    "allowed_pin",
    "installed_mismatch",
    "manifest_mismatch",
    "parse_allow_list",
    "registered_entry_point",
]

#: The environment variable a control plane sets, from its OWN configuration, in the job that
#: runs a plugin: the JSON list of pins the runner may execute. Never derived from the spec.
ALLOWED_PLUGINS_ENV = "STRATA_ALLOWED_PLUGINS"

_SHA = r"^[0-9a-f]{40}$"
_REPO = r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}/[A-Za-z0-9][A-Za-z0-9._-]{0,95}$"
_SLUG = r"^[a-z0-9][a-z0-9-]{0,63}$"
_FILE = r"^[A-Za-z0-9][A-Za-z0-9._+-]{0,199}\.whl$"


class PluginSource(BaseModel):
    """Where a plugin is built from, at one full commit.

    ``git`` names a GitHub repository (``owner/name``). ``hf`` names a Hugging Face model
    repository holding a built wheel (``wheel``) at the commit: the install URL is that file's
    ``resolve`` URL at the commit, and that exact URL is what the installed metadata must record.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["git", "hf"]
    repo: Annotated[str, StringConstraints(pattern=_REPO)]
    commit: Annotated[str, StringConstraints(pattern=_SHA)]
    #: The wheel file in an ``hf`` source; refused on a ``git`` source.
    wheel: Annotated[str, StringConstraints(pattern=_FILE)] | None = None

    @model_validator(mode="after")
    def _wheel_matches_kind(self) -> PluginSource:
        if self.kind == "hf" and self.wheel is None:
            msg = "an hf plugin source names the wheel file it installs"
            raise ValueError(msg)
        if self.kind == "git" and self.wheel is not None:
            msg = "a git plugin source installs from the repository, not a wheel"
            raise ValueError(msg)
        if ".." in self.repo:
            msg = "invalid repository name"
            raise ValueError(msg)
        return self

    def install_url(self) -> str:
        """The URL the plugin is installed from: what ``direct_url.json`` must record."""
        if self.kind == "git":
            return f"https://github.com/{self.repo}.git"
        return f"https://huggingface.co/{self.repo}/resolve/{self.commit}/{self.wheel}"

    def install_requirement(self, distribution: str) -> str:
        """The exact requirement an owner installs on their target (shown, never executed by forge)."""
        if self.kind == "git":
            return f"{distribution} @ git+{self.install_url()}@{self.commit}"
        return f"{distribution} @ {self.install_url()}"

    def manifest_url(self) -> str:
        """Where the manifest is read from at the commit (metadata only, no checkout)."""
        if self.kind == "git":
            return f"https://raw.githubusercontent.com/{self.repo}/{self.commit}/strata-plugin.json"
        return f"https://huggingface.co/{self.repo}/resolve/{self.commit}/strata-plugin.json"


class PluginPin(BaseModel):
    """A plugin build, pinned completely."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    distribution: Annotated[
        str, StringConstraints(pattern=r"^[a-z0-9](?:[a-z0-9-]{0,98}[a-z0-9])?$")
    ]
    version: Annotated[str, StringConstraints(min_length=1, max_length=64)]
    source: PluginSource


class AllowedPlugin(BaseModel):
    """One operator allow-list entry: a pinned plugin and, optionally, which of its entry points
    may run (``None``: every entry point its manifest declares)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: Annotated[str, StringConstraints(pattern=_SLUG)]
    label: Annotated[str, StringConstraints(min_length=1, max_length=120)]
    pin: PluginPin
    entry_points: tuple[Annotated[str, StringConstraints(max_length=64)], ...] | None = Field(
        default=None, max_length=32
    )

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


def _direct_url(distribution: str) -> dict[str, object] | None:
    try:
        raw = metadata.distribution(distribution).read_text("direct_url.json")
    except metadata.PackageNotFoundError:
        return None
    except OSError, ValueError:
        return None
    if not raw:
        return None
    try:
        info: object = json.loads(raw)
    except ValueError:
        return None
    return cast("dict[str, object]", info) if isinstance(info, dict) else None


def installed_mismatch(pin: PluginPin) -> str | None:
    """Why the installed distribution is not the pinned build, or ``None`` when it is."""
    try:
        version = metadata.version(pin.distribution)
    except metadata.PackageNotFoundError:
        return (
            f"{pin.distribution} is not installed on this machine; install it with "
            f"'{pin.source.install_requirement(pin.distribution)}'"
        )
    if version != pin.version:
        return f"{pin.distribution} {version} is installed, but the run pins {pin.version}"
    info = _direct_url(pin.distribution)
    if info is None:
        return (
            f"{pin.distribution} was not installed from its pinned source (no PEP 610 record); "
            f"install it with '{pin.source.install_requirement(pin.distribution)}'"
        )
    url = info.get("url")
    source = pin.source
    if source.kind == "git":
        vcs = info.get("vcs_info")
        commit = cast("dict[str, object]", vcs).get("commit_id") if isinstance(vcs, dict) else None
        tool = cast("dict[str, object]", vcs).get("vcs") if isinstance(vcs, dict) else None
        if (
            tool != "git"
            or commit != source.commit
            or _strip_git(url) != _strip_git(source.install_url())
        ):
            return (
                f"{pin.distribution} is installed from {_shown(url)} at {_shown(commit)}, not "
                f"from {source.repo} at {source.commit[:12]}"
            )
        return None
    if url != source.install_url() or not isinstance(info.get("archive_info"), dict):
        return f"{pin.distribution} is installed from {_shown(url)}, not {source.install_url()}"
    return None


def registered_entry_point(pin: PluginPin, name: str) -> metadata.EntryPoint | None:
    """The entry point ``name`` the PINNED distribution registers under the plugin group.

    Looked up in that distribution's own metadata, never across all installed packages: another
    distribution registering the same name (by accident or on purpose) cannot be picked up
    instead. Nothing is imported here; the caller loads it.
    """
    try:
        dist = metadata.distribution(pin.distribution)
    except metadata.PackageNotFoundError:
        return None
    for ep in dist.entry_points:
        if ep.group == ENTRY_POINT_GROUP and ep.name == name:
            return ep
    return None


def _strip_git(url: object) -> str:
    text = str(url or "").lower()
    if text.startswith("git+"):
        text = text[4:]
    return text.removesuffix("/").removesuffix(".git")


def _shown(value: object) -> str:
    text = str(value) if value is not None else "an unknown source"
    return text[:120]
