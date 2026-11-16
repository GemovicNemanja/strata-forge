"""Strict immutable authoring values for the explicit modular prompt dialect."""

from __future__ import annotations

import re
from typing import Annotated, Literal, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_serializer,
    model_validator,
)

from strata_forge.prompts.composition_errors import PromptCompositionError
from strata_forge.prompts.portable import (
    MAX_JSON_BYTES,
    MAX_SAFE_INTEGER,
    PortableValue,
    canonical_bytes,
    parse_portable_json,
)

NAME_PATTERN = r"[A-Za-z][A-Za-z0-9_./-]{0,127}"
VARIABLE_PATTERN = r"[A-Za-z_][A-Za-z0-9_]{0,63}"
_NAME = re.compile(NAME_PATTERN + r"\Z")
_VARIABLE = re.compile(VARIABLE_PATTERN + r"\Z")
_SHA = re.compile(r"[a-f0-9]{64}\Z")


def validate_name(name: str) -> str:
    """Keep logical namespace coordinates exact and independent of paths or URLs."""
    if not _NAME.fullmatch(name) or ".." in name or any(not part for part in name.split("/")):
        raise PromptCompositionError("definition_name_invalid")
    return name


def validate_coordinate(coordinate: str) -> str:
    """Require one literal name and a positive safe integer revision, never an alias."""
    name, separator, revision = coordinate.rpartition("@")
    if (
        not separator
        or not revision.isascii()
        or not revision.isdecimal()
        or revision.startswith("0")
        or len(revision) > 16
    ):
        raise PromptCompositionError("coordinate_invalid")
    validate_name(name)
    if int(revision) > MAX_SAFE_INTEGER:
        raise PromptCompositionError("coordinate_invalid")
    return coordinate


class _Strict(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True, hide_input_in_errors=True)


class PortableDocument(_Strict):
    """Immutable JSON metadata whose reads produce independent native-object copies."""

    canonical: bytes = Field(max_length=MAX_JSON_BYTES)

    @field_validator("canonical")
    @classmethod
    def _canonical(cls, value: bytes) -> bytes:
        parsed = parse_portable_json(value)
        if type(parsed) is not dict or canonical_bytes(parsed) != value:
            raise PromptCompositionError("descriptor_invalid")
        return value

    @classmethod
    def from_value(cls, value: object) -> PortableDocument:
        """Freeze a portable object without retaining caller-owned dictionaries."""
        if type(value) is not dict:
            raise PromptCompositionError("descriptor_invalid")
        return cls(canonical=canonical_bytes(cast("dict[str, object]", value)))

    def wire_payload(self) -> dict[str, PortableValue]:
        """Return a fresh mutable view without exposing the stored content identity."""
        return cast("dict[str, PortableValue]", parse_portable_json(self.canonical))

    @model_serializer
    def _serialize(self) -> dict[str, PortableValue]:
        return self.wire_payload()


class VariableDecl(_Strict):
    """One exact scalar input, with booleans distinct from integer values."""

    name: str = Field(pattern="^" + VARIABLE_PATTERN + "$")
    type: Literal["string", "integer", "boolean"]


class PromptConstraint(_Strict):
    """An explicit fact; the compiler makes no inference about prose instructions."""

    key: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z][A-Za-z0-9_.-]*$")
    value: str = Field(max_length=4096)


class SourceProvenance(_Strict):
    """Declared import evidence only; no pointer is fetched or authorizes a read."""

    repository: str = Field(min_length=1, max_length=256)
    commit: str = Field(pattern=r"^[a-f0-9]{40}$")
    path: str = Field(min_length=1, max_length=1024)
    source_file_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    selector: str = Field(min_length=1, max_length=256)
    dialect: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_.-]+$")


class TextPart(_Strict):
    """Exact literal text; braces and template-looking bytes have no implicit dialect."""

    kind: Literal["text"] = "text"
    text: str


class VariablePart(_Strict):
    """Insert a declared scalar once, without parsing its content."""

    kind: Literal["variable"] = "variable"
    name: str = Field(pattern="^" + VARIABLE_PATTERN + "$")


class ModulePart(_Strict):
    """Include exactly one pinned module coordinate."""

    kind: Literal["module"] = "module"
    revision: str = Field(min_length=3, max_length=145)

    @field_validator("revision")
    @classmethod
    def _revision(cls, value: str) -> str:
        return validate_coordinate(value)


type PromptPart = Annotated[TextPart | VariablePart | ModulePart, Field(discriminator="kind")]


class ChatMessageParts(_Strict):
    """One preserved chat role, an exact separator and ordered source parts."""

    role: Literal["system", "user", "assistant"]
    parts: tuple[PromptPart, ...] = Field(max_length=1024)
    separator: str


class ChatTarget(_Strict):
    """Ordered text-only messages, without emulated tool or multimodal results."""

    kind: Literal["chat"] = "chat"
    messages: tuple[ChatMessageParts, ...] = Field(min_length=1, max_length=128)

    @model_validator(mode="after")
    def _parts_limit(self) -> ChatTarget:
        if sum(len(message.parts) for message in self.messages) > 1024:
            raise PromptCompositionError("definition_parts_limit")
        return self


class ArtifactTarget(_Strict):
    """One textual artifact with metadata separate from its composed description."""

    kind: Literal["artifact"] = "artifact"
    purpose: Literal["system_instruction", "tool_description", "skill", "text"]
    parts: tuple[PromptPart, ...] = Field(max_length=1024)
    separator: str
    descriptor: PortableDocument | None = None

    @field_validator("descriptor", mode="before")
    @classmethod
    def _descriptor(cls, value: object) -> PortableDocument | None:
        if value is None:
            return None
        if isinstance(value, PortableDocument):
            return PortableDocument(canonical=value.canonical)
        return PortableDocument.from_value(value)


type PromptTarget = Annotated[ChatTarget | ArtifactTarget, Field(discriminator="kind")]


class _Definition(_Strict):
    schema_version: Literal[1] = 1
    dialect: Literal["strata-modules-v1"] = "strata-modules-v1"
    name: str = Field(min_length=1, max_length=128)
    revision: int = Field(ge=1, le=MAX_SAFE_INTEGER)
    variables: tuple[VariableDecl, ...] = Field(default=(), max_length=256)
    constraints: tuple[PromptConstraint, ...] = Field(default=(), max_length=256)
    provenance: SourceProvenance | None = None

    @field_validator("name")
    @classmethod
    def _name(cls, value: str) -> str:
        return validate_name(value)

    @field_validator("revision", "schema_version", mode="before")
    @classmethod
    def _integer(cls, value: object) -> object:
        if type(value) is not int:
            raise PromptCompositionError("definition_revision_invalid")
        return value

    @property
    def coordinate(self) -> str:
        """The exact registry coordinate; no channel or normalization is applied."""
        return f"{self.name}@{self.revision}"

    def wire_payload(self) -> dict[str, PortableValue]:
        """Return an independent portable object for storage or inspection."""
        return cast(
            "dict[str, PortableValue]",
            parse_portable_json(canonical_bytes(self.model_dump(mode="json"))),
        )

    @model_validator(mode="after")
    def _declarations(self) -> _Definition:
        names = [variable.name for variable in self.variables]
        if len(names) != len(set(names)):
            raise PromptCompositionError("variable_declaration_duplicate")
        return self


class ModuleRevision(_Definition):
    """An authored reusable module at an exact immutable coordinate."""

    kind: Literal["module"] = "module"
    parts: tuple[PromptPart, ...] = Field(max_length=1024)


class SurfaceRevision(_Definition):
    """An authored chat or textual surface with its complete input declaration."""

    kind: Literal["surface"] = "surface"
    target: PromptTarget


type PromptDefinition = ModuleRevision | SurfaceRevision


def definition_from_wire(value: object) -> PromptDefinition:
    """Validate a closed authored definition without retaining unsafe Pydantic diagnostics."""
    result: PromptDefinition | None = None
    failure = "definition_invalid"
    try:
        encoded = canonical_bytes(value)
        fields = parse_portable_json(encoded)
        if type(fields) is dict:
            kind = fields.get("kind")
            if kind == "module":
                result = ModuleRevision.model_validate_json(encoded)
            elif kind == "surface":
                result = SurfaceRevision.model_validate_json(encoded)
    except PromptCompositionError as error:
        failure = error.diagnostic.code
    except ValueError, TypeError:
        result = None
    if result is None:
        raise PromptCompositionError(failure)
    return result
