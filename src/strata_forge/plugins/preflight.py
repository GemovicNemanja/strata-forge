"""Check a plugin launch request against the plugin's declared interface, without running it.

This is the control plane's half of plugin admission. It takes the manifest the control plane
read at the pinned commit (:func:`~strata_forge.plugins.manifest.parse_manifest`), the
operator's allow-list entry, and what the user asked for, and answers with a list of named checks.
It is pure: no I/O, no imports from the plugin, no code from the repository. What it cannot know
here (a column list nobody has read yet, whether the target has the build installed) is reported
as ``unknown`` rather than guessed, and the runner re-checks everything on the target.

The checks:

- the entry point is declared by the manifest and allowed by the operator's entry;
- the manifest describes the pinned distribution and version;
- every required input role is bound, and nothing undeclared is;
- each bound input has the columns the entry point requires (when its columns are known);
- every declared output is bound to a repository;
- the configuration is closed and well-typed (unknown keys, wrong types, out-of-range values and
  missing required values are refused; defaults are filled in).
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from strata_forge.plugins.manifest import (
    EntryPointDecl,
    PluginManifest,
    ScalarValue,
    check_scalar,
)
from strata_forge.plugins.pins import AllowedPlugin, manifest_mismatch

__all__ = [
    "CheckKey",
    "CheckStatus",
    "InputBinding",
    "PluginRequest",
    "PreflightCheck",
    "PreflightReport",
    "check_bindings",
    "check_config",
    "preflight",
]

type CheckStatus = Literal["ok", "fail", "unknown"]
type CheckKey = Literal[
    "allowed", "manifest", "entry_point", "inputs", "columns", "output", "config"
]

_SHA = r"^[0-9a-f]{40}$"
_REPO = r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}(?:/[A-Za-z0-9][A-Za-z0-9._-]{0,95})?$"


class InputBinding(BaseModel):
    """A Hub repository bound to one declared input role, at a full commit: a dataset split for
    a ``dataset`` input, a model repository for a ``model`` input."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    role: Annotated[str, StringConstraints(max_length=48)]
    repo_id: Annotated[str, StringConstraints(pattern=_REPO)]
    revision: Annotated[str, StringConstraints(pattern=_SHA)]
    config: Annotated[str, StringConstraints(max_length=128)] | None = None
    split: Annotated[str, StringConstraints(min_length=1, max_length=128)] | None = None
    #: The split's columns, when the caller read them; ``None`` makes the column check unknown.
    columns: tuple[str, ...] | None = Field(default=None, max_length=4096)


class PluginRequest(BaseModel):
    """What a user asked to run: an entry point of an allowed plugin, its bindings and config."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    entry_point: Annotated[str, StringConstraints(min_length=1, max_length=64)]
    inputs: tuple[InputBinding, ...] = Field(default=(), max_length=16)
    output_repo_id: Annotated[str, StringConstraints(pattern=_REPO)] | None = None
    config: dict[Annotated[str, StringConstraints(max_length=64)], ScalarValue] = Field(
        default_factory=dict, max_length=64
    )


class PreflightCheck(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    key: CheckKey
    status: CheckStatus
    detail: str


class PreflightReport(BaseModel):
    """The checks, and the configuration as it will be sent (defaults filled) when they pass."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    checks: tuple[PreflightCheck, ...]
    entry: EntryPointDecl | None
    config: dict[str, ScalarValue] | None

    @property
    def ok(self) -> bool:
        """No check failed. ``unknown`` checks do not block: the runner re-checks on the target."""
        return all(c.status != "fail" for c in self.checks)

    def failures(self) -> list[PreflightCheck]:
        return [c for c in self.checks if c.status == "fail"]


def preflight(
    manifest: PluginManifest, allowed: AllowedPlugin, request: PluginRequest
) -> PreflightReport:
    """Check ``request`` against ``manifest`` and the operator's ``allowed`` entry."""
    checks: list[PreflightCheck] = []

    def add(key: CheckKey, status: CheckStatus, detail: str) -> None:
        checks.append(PreflightCheck(key=key, status=status, detail=detail))

    problem = manifest_mismatch(manifest, allowed.pin)
    add("manifest", "fail" if problem else "ok", problem or "the manifest matches the pinned build")

    entry = manifest.entry(request.entry_point)
    if entry is None:
        add(
            "entry_point",
            "fail",
            f"{request.entry_point!r} is not declared by {manifest.distribution} {manifest.version}",
        )
        return PreflightReport(checks=tuple(checks), entry=None, config=None)
    add("entry_point", "ok", f"{entry.name} ({entry.type})")
    if allowed.allows(entry.name):
        add("allowed", "ok", f"allowed by the operator entry {allowed.id!r}")
    else:
        add(
            "allowed", "fail", f"{entry.name!r} is not allowed by the operator entry {allowed.id!r}"
        )

    for key, status, detail in _check_inputs(entry, request):
        add(key, status, detail)
    if entry.outputs and request.output_repo_id is None:
        add("output", "fail", f"the {entry.outputs[0].role!r} output needs an output repository")
    else:
        add("output", "ok", f"written to {request.output_repo_id} (private)")

    config, problems = check_config(entry, request.config)
    if problems:
        add("config", "fail", "; ".join(problems))
    else:
        add("config", "ok", f"{len(config)} parameter(s)")

    report = PreflightReport(checks=tuple(checks), entry=entry, config=config)
    return report if report.ok else PreflightReport(checks=report.checks, entry=entry, config=None)


def _check_inputs(
    entry: EntryPointDecl, request: PluginRequest
) -> list[tuple[CheckKey, CheckStatus, str]]:
    declared = {i.role: i for i in entry.inputs}
    bound: dict[str, InputBinding] = {}
    problems: list[str] = []
    for binding in request.inputs:
        decl = declared.get(binding.role)
        if decl is None:
            problems.append(f"{binding.role!r} is not an input {entry.name} declares")
        elif binding.role in bound:
            problems.append(f"{binding.role!r} is bound twice")
        elif decl.kind == "dataset" and binding.split is None:
            problems.append(f"the dataset input {binding.role!r} needs a split")
        elif decl.kind == "model" and (binding.split is not None or binding.config is not None):
            problems.append(f"the model input {binding.role!r} takes no config or split")
        else:
            bound[binding.role] = binding
    for role, decl in declared.items():
        if decl.required and role not in bound:
            problems.append(f"the required input {role!r} is not bound")
    out: list[tuple[CheckKey, CheckStatus, str]] = [
        ("inputs", "fail" if problems else "ok", "; ".join(problems) or "all inputs bound")
    ]

    missing: list[str] = []
    unknown: list[str] = []
    for role, binding in bound.items():
        needed = declared[role].required_columns
        if not needed:
            continue
        if binding.columns is None:
            unknown.append(role)
            continue
        absent = [c for c in needed if c not in binding.columns]
        if absent:
            missing.append(f"{role!r} lacks {absent}")
    if missing:
        out.append(("columns", "fail", "; ".join(missing)))
    elif unknown:
        out.append(("columns", "unknown", f"columns not read yet for {unknown}"))
    else:
        out.append(("columns", "ok", "every required column is present"))
    return out


def check_bindings(entry: EntryPointDecl, request: PluginRequest) -> list[str]:
    """The binding and column problems of ``request`` (empty when there are none)."""
    return [detail for _key, status, detail in _check_inputs(entry, request) if status == "fail"]


def check_config(
    entry: EntryPointDecl, given: dict[str, ScalarValue]
) -> tuple[dict[str, ScalarValue], list[str]]:
    """``given`` validated against ``entry``'s closed parameters, with defaults filled, and the
    problems found (empty when it is acceptable)."""
    problems: list[str] = []
    out: dict[str, ScalarValue] = {}
    for key, value in given.items():
        param = entry.param(key)
        if param is None:
            problems.append(f"{key!r} is not a parameter of {entry.name}")
            continue
        problem = check_scalar(param, value)
        if problem is not None:
            problems.append(f"{key!r} {problem}")
            continue
        out[key] = float(value) if param.type == "number" and isinstance(value, int) else value
    for param in entry.config:
        if param.name in out or param.name in given:
            continue
        if param.required:
            problems.append(f"{param.name!r} is required")
        elif param.default is not None:
            out[param.name] = param.default
    return out, problems
