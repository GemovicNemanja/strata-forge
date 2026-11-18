"""The declared interface of a trusted plugin: what a plugin repository says it is.

A plugin repository carries one JSON file, :data:`PLUGIN_MANIFEST_FILENAME`, at its root. It names
the distribution the repository builds, its version, and each entry point the distribution
registers under :data:`ENTRY_POINT_GROUP`, with the inputs it reads, the outputs it writes and the
closed configuration it accepts. A control plane reads this file at a pinned commit as DATA (JSON
plus validation, size-capped) and checks a launch request against it without importing, building
or executing anything from the repository. The code itself only ever runs on the user's own
compute target (:mod:`strata_forge.pipelines.plugin_runner`).

Nothing here names an import path. An entry point is chosen by its registered NAME inside the
distribution's own entry-point metadata, so a spec can never smuggle in a module path, and the
configuration is a flat set of typed scalars that the plugin re-validates with its own model.
"""

from __future__ import annotations

import json
import math
import re
from typing import TYPE_CHECKING, Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from strata_forge.core.errors import ValidationError

if TYPE_CHECKING:
    from collections.abc import Iterable

__all__ = [
    "ENTRY_POINT_GROUP",
    "MANIFEST_SCHEMA",
    "MAX_MANIFEST_BYTES",
    "PLUGIN_MANIFEST_FILENAME",
    "STEP_INPUT_COLUMNS",
    "STEP_INPUT_ROLE",
    "STEP_OUTPUT_ROLE",
    "ConfigParam",
    "EntryPointDecl",
    "EntryType",
    "InputDecl",
    "OutputDecl",
    "PluginManifest",
    "PluginManifestError",
    "ScalarValue",
    "check_scalar",
    "normalize_distribution",
    "parse_manifest",
]

#: The file a plugin repository declares itself in, at the repository root.
PLUGIN_MANIFEST_FILENAME = "strata-plugin.json"
#: The manifest format this engine reads. A new shape is a new schema id, never a silent change.
MANIFEST_SCHEMA = "strata.plugin/v1"
#: The entry-point group a plugin distribution registers its entry points under.
ENTRY_POINT_GROUP = "strata_forge.plugins"
#: A manifest is a short declaration; anything larger is refused before it is parsed.
MAX_MANIFEST_BYTES = 64 * 1024

#: The one input a ``step_scorer`` reads, and the columns forge guarantees it (see
#: :mod:`strata_forge.plugins.step_scoring`).
STEP_INPUT_ROLE = "steps"
STEP_INPUT_COLUMNS: tuple[str, ...] = ("item_id", "step_index", "context", "step")
#: The one output a ``step_scorer`` produces: one score per input step, written by forge.
STEP_OUTPUT_ROLE = "scores"

# A role or parameter name: a lowercase identifier, so it is safe as a key, a column suffix and
# a path segment alike.
_IDENT = r"^[a-z][a-z0-9_]{0,47}$"
# An entry-point name as packaging metadata allows it, restricted to a safe subset (no spaces,
# no ``=``, no ``:``), so a name can never be confused with an ``module:attr`` reference.
_ENTRY_NAME = r"^[a-z0-9][a-z0-9_.-]{0,63}$"
# A distribution name in PEP 503 normalised form (lowercase, runs of ``-_.`` collapsed to ``-``).
_DIST_NAME = r"^[a-z0-9](?:[a-z0-9-]{0,98}[a-z0-9])?$"
# A plain release version: what an exact ``==`` pin names. Local versions (``+abc``) are refused
# because they are how a build pretends to be a release it is not.
_VERSION = r"^[0-9]+(?:\.[0-9]+){0,3}(?:(?:a|b|rc)[0-9]+)?(?:\.post[0-9]+)?(?:\.dev[0-9]+)?$"
# A column a declared input requires.
_COLUMN = r"^[A-Za-z_][A-Za-z0-9_.-]{0,127}$"

Ident = Annotated[str, StringConstraints(pattern=_IDENT)]
Column = Annotated[str, StringConstraints(pattern=_COLUMN)]
Description = Annotated[str, StringConstraints(max_length=280)]

#: A configuration value: a JSON scalar. Lists, objects and ``null`` are not configuration.
type ScalarValue = str | int | float | bool

#: What an entry point is. ``runner`` is a general job (a generator, a custom training loop) that
#: reads its declared inputs and writes into the output directory forge hands it. ``step_scorer``
#: is a process reward model: forge feeds it steps and writes the scores (see ``step_scoring``).
type EntryType = Literal["runner", "step_scorer"]


class PluginManifestError(ValidationError):
    """A plugin manifest, or a request checked against one, is not acceptable."""


class ConfigParam(BaseModel):
    """One configuration parameter an entry point accepts. Configuration is closed: a key the
    entry point does not declare is refused, so a request cannot pass anything the plugin did not
    ask for."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: Ident
    type: Literal["string", "integer", "number", "boolean"]
    description: Description = ""
    required: bool = False
    default: ScalarValue | None = None
    #: Allowed values of a ``string`` parameter.
    enum: tuple[Annotated[str, StringConstraints(max_length=200)], ...] | None = Field(
        default=None, max_length=64
    )
    minimum: float | None = None
    maximum: float | None = None
    #: The longest a ``string`` value may be (never more than 4,096 characters).
    max_length: int = Field(default=1024, ge=1, le=4096)

    @model_validator(mode="after")
    def _consistent(self) -> ConfigParam:
        if self.enum is not None and self.type != "string":
            msg = f"parameter {self.name!r}: only a string parameter takes an enum"
            raise ValueError(msg)
        if (self.minimum is not None or self.maximum is not None) and self.type not in {
            "integer",
            "number",
        }:
            msg = f"parameter {self.name!r}: only a numeric parameter takes a minimum or maximum"
            raise ValueError(msg)
        if self.minimum is not None and self.maximum is not None and self.minimum > self.maximum:
            msg = f"parameter {self.name!r}: minimum is greater than maximum"
            raise ValueError(msg)
        if self.required and self.default is not None:
            msg = f"parameter {self.name!r}: a required parameter takes no default"
            raise ValueError(msg)
        if self.default is not None:
            problem = check_scalar(self, self.default)
            if problem is not None:
                msg = f"parameter {self.name!r}: the default {problem}"
                raise ValueError(msg)
        return self


class InputDecl(BaseModel):
    """Something the entry point reads: a dataset split or a model's weights. Inputs are
    read-only and pinned: forge downloads each at a full commit with the run's credential and
    hands the plugin local files, never the credential. A plugin that needs a model (a process
    reward model's weights, say) declares it here rather than fetching it itself."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    role: Ident
    kind: Literal["dataset", "model"] = "dataset"
    description: Description = ""
    required: bool = True
    required_columns: tuple[Column, ...] = Field(default=(), max_length=64)

    @model_validator(mode="after")
    def _columns_only_on_datasets(self) -> InputDecl:
        if self.kind == "model" and self.required_columns:
            msg = f"input {self.role!r}: a model input has no columns"
            raise ValueError(msg)
        return self


class OutputDecl(BaseModel):
    """What the entry point writes. Forge pushes it to the run's private output repository; the
    plugin only ever writes local files."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    role: Ident
    kind: Literal["dataset", "model"]
    description: Description = ""


class EntryPointDecl(BaseModel):
    """One entry point the distribution registers under :data:`ENTRY_POINT_GROUP`."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: Annotated[str, StringConstraints(pattern=_ENTRY_NAME)]
    type: EntryType
    description: Description = ""
    inputs: tuple[InputDecl, ...] = Field(default=(), max_length=16)
    outputs: tuple[OutputDecl, ...] = Field(min_length=1, max_length=1)
    config: tuple[ConfigParam, ...] = Field(default=(), max_length=64)
    #: Whether the entry point needs a GPU. Informational for a target check; forge does not
    #: allocate devices.
    gpu: bool = False

    @model_validator(mode="after")
    def _consistent(self) -> EntryPointDecl:
        _unique((i.role for i in self.inputs), f"entry point {self.name!r}: input role")
        _unique((p.name for p in self.config), f"entry point {self.name!r}: parameter")
        if self.type == "step_scorer":
            datasets = [i for i in self.inputs if i.kind == "dataset"]
            if [i.role for i in datasets] != [STEP_INPUT_ROLE] or not datasets[0].required:
                msg = (
                    f"entry point {self.name!r}: a step_scorer reads exactly one dataset input, "
                    f"the required {STEP_INPUT_ROLE!r} (and may declare model inputs)"
                )
                raise ValueError(msg)
            missing = set(STEP_INPUT_COLUMNS) - set(datasets[0].required_columns)
            if missing:
                msg = (
                    f"entry point {self.name!r}: a step_scorer's {STEP_INPUT_ROLE!r} input must "
                    f"require the columns {list(STEP_INPUT_COLUMNS)}"
                )
                raise ValueError(msg)
            output = self.outputs[0]
            if output.role != STEP_OUTPUT_ROLE or output.kind != "dataset":
                msg = (
                    f"entry point {self.name!r}: a step_scorer writes one dataset output, "
                    f"{STEP_OUTPUT_ROLE!r}"
                )
                raise ValueError(msg)
        return self

    def param(self, name: str) -> ConfigParam | None:
        return next((p for p in self.config if p.name == name), None)


class PluginManifest(BaseModel):
    """The whole declaration: :data:`PLUGIN_MANIFEST_FILENAME` parsed."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_: Literal["strata.plugin/v1"] = Field(alias="schema")
    distribution: Annotated[str, StringConstraints(pattern=_DIST_NAME)]
    version: Annotated[str, StringConstraints(pattern=_VERSION)]
    description: Description = ""
    entry_points: tuple[EntryPointDecl, ...] = Field(min_length=1, max_length=32)

    @model_validator(mode="after")
    def _unique_names(self) -> PluginManifest:
        _unique((e.name for e in self.entry_points), "entry point")
        return self

    def entry(self, name: str) -> EntryPointDecl | None:
        return next((e for e in self.entry_points if e.name == name), None)


def _unique(names: Iterable[str], what: str) -> None:
    seen: set[str] = set()
    for name in names:
        if name in seen:
            msg = f"duplicate {what} {name!r}"
            raise ValueError(msg)
        seen.add(name)


def normalize_distribution(name: str) -> str:
    """``name`` in PEP 503 normalised form, the form :class:`PluginManifest` stores."""
    return re.sub(r"[-_.]+", "-", name).lower()


def check_scalar(param: ConfigParam, value: object) -> str | None:
    """Why ``value`` is not acceptable for ``param``, or ``None`` when it is.

    Strict: ``True`` is not an integer, ``1`` is not a string, a non-finite number is refused.
    """
    match param.type:
        case "boolean":
            if not isinstance(value, bool):
                return "must be true or false"
        case "integer":
            if isinstance(value, bool) or not isinstance(value, int):
                return "must be an integer"
        case "number":
            if isinstance(value, bool) or not isinstance(value, int | float):
                return "must be a number"
            if not math.isfinite(value):
                return "must be a finite number"
        case "string":
            if not isinstance(value, str):
                return "must be a string"
            if len(value) > param.max_length:
                return f"must be at most {param.max_length} characters"
            if param.enum is not None and value not in param.enum:
                return f"must be one of {list(param.enum)}"
    if param.type in {"integer", "number"}:
        number = float(value)  # pyright: ignore[reportArgumentType] -- checked numeric above
        if param.minimum is not None and number < param.minimum:
            return f"must be at least {param.minimum:g}"
        if param.maximum is not None and number > param.maximum:
            return f"must be at most {param.maximum:g}"
    return None


def parse_manifest(raw: bytes | str) -> PluginManifest:
    """Parse a manifest as data, refusing anything oversized, malformed or undeclared.

    JSON and Pydantic validation only: nothing is evaluated, imported or fetched.
    """
    data = raw.encode() if isinstance(raw, str) else raw
    if len(data) > MAX_MANIFEST_BYTES:
        msg = f"{PLUGIN_MANIFEST_FILENAME} is larger than {MAX_MANIFEST_BYTES} bytes"
        raise PluginManifestError(msg)
    try:
        parsed: object = json.loads(data)
    except ValueError as exc:
        msg = f"{PLUGIN_MANIFEST_FILENAME} is not valid JSON: {exc}"
        raise PluginManifestError(msg) from exc
    if not isinstance(parsed, dict):
        msg = f"{PLUGIN_MANIFEST_FILENAME} must be a JSON object"
        raise PluginManifestError(msg)
    try:
        return PluginManifest.model_validate(parsed)
    except ValueError as exc:
        msg = f"{PLUGIN_MANIFEST_FILENAME} is not a valid {MANIFEST_SCHEMA} manifest: {exc}"
        raise PluginManifestError(msg) from exc
