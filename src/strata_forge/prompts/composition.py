"""Immutable exact-pin composition and UTF-8 attributed rendering, without runtime I/O."""

from __future__ import annotations

import hashlib
import re
from typing import Literal, cast

from pydantic import BaseModel, ConfigDict, Field, field_validator

from strata_forge.prompts.composition_errors import (
    CompositionDiagnostic,
    PromptCompositionError,
    PromptOrigin,
)
from strata_forge.prompts.composition_types import (
    ArtifactTarget,
    ChatTarget,
    ModulePart,
    ModuleRevision,
    PortableDocument,
    PromptDefinition,
    PromptPart,
    SourceProvenance,
    SurfaceRevision,
    TextPart,
    VariableDecl,
    VariablePart,
    definition_from_wire,
    validate_coordinate,
)
from strata_forge.prompts.portable import (
    MAX_SAFE_INTEGER,
    PortableValue,
    canonical_bytes,
    parse_portable_json,
    portable_sha256,
)

COMPILER_VERSION = "strata-modular-compiler/1"
MAX_DEFINITIONS = 4096
MAX_DEFINITION_BYTES = 8 * 1024 * 1024
MAX_DEPENDENCY_DEPTH = 32
MAX_EXPANDED_LEAVES = 16_384
MAX_RENDERED_BYTES = 8 * 1024 * 1024
MAX_DIAGNOSTICS = 64
MAX_EXPANSION_VISITS = 1_048_576


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True, hide_input_in_errors=True)

    @field_validator("schema_version", mode="before", check_fields=False)
    @classmethod
    def _version(cls, value: object) -> object:
        if type(value) is not int or value != 1:
            raise PromptCompositionError("schema_version_invalid")
        return value


class CompositionCapabilities(_Frozen):
    """Versioned fixed compiler bounds, independent of a caller's research or product policy."""

    compiler_version: Literal["strata-modular-compiler/1"] = COMPILER_VERSION
    portable_profile: Literal["strata-prompt-json-v1"] = "strata-prompt-json-v1"
    registry_definitions: int = MAX_DEFINITIONS
    definition_parts: int = 1024
    surface_messages: int = 128
    dependency_depth: int = MAX_DEPENDENCY_DEPTH
    expanded_leaves: int = MAX_EXPANDED_LEAVES
    distinct_variables: int = 256
    total_definition_bytes: int = MAX_DEFINITION_BYTES
    rendered_utf8_bytes: int = MAX_RENDERED_BYTES
    diagnostics: int = MAX_DIAGNOSTICS
    portable_json_bytes: int = 64 * 1024 * 1024
    portable_json_depth: int = 64
    expansion_visits: int = MAX_EXPANSION_VISITS


async def composition_capabilities() -> CompositionCapabilities:
    """Describe the installed compiler's immutable versioned capability profile."""
    return CompositionCapabilities()


class DependencyLock(_Frozen):
    """A single exact content coordinate, including its declared source evidence."""

    coordinate: str
    content_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    provenance: SourceProvenance | None


class CompiledLeaf(_Frozen):
    """One literal or variable occurrence in expansion order."""

    kind: Literal["text", "variable"]
    text: str | None
    variable: str | None
    origin: PromptOrigin


class CompiledOutput(_Frozen):
    """One final message or artifact body, before scalar substitution."""

    role: Literal["system", "user", "assistant"] | None
    separator: str
    leaves: tuple[CompiledLeaf, ...]


class ResolvedConstraint(_Frozen):
    """A consistent declared fact with all contributing definition coordinates."""

    key: str
    value: str
    origins: tuple[PromptOrigin, ...]


class _BuildWire(_Frozen):
    domain: Literal["strata-prompt-build-v1"] = "strata-prompt-build-v1"
    schema_version: Literal[1] = 1
    dialect: Literal["strata-modules-v1"] = "strata-modules-v1"
    compiler_version: Literal["strata-modular-compiler/1"] = COMPILER_VERSION
    surface: DependencyLock
    dependency_lock: tuple[DependencyLock, ...] = Field(max_length=MAX_DEFINITIONS)
    variables: tuple[VariableDecl, ...] = Field(max_length=256)
    constraints: tuple[ResolvedConstraint, ...]
    kind: Literal["chat", "artifact"]
    purpose: Literal["system_instruction", "tool_description", "skill", "text"] | None
    descriptor: PortableDocument | None
    outputs: tuple[CompiledOutput, ...] = Field(min_length=1, max_length=128)
    diagnostics: tuple[CompositionDiagnostic, ...] = Field(max_length=MAX_DIAGNOSTICS)

    @field_validator("descriptor", mode="before")
    @classmethod
    def _descriptor(cls, value: object) -> PortableDocument | None:
        if value is None:
            return None
        if isinstance(value, PortableDocument):
            return PortableDocument(canonical=value.canonical)
        return PortableDocument.from_value(value)


class SourceSpan(_Frozen):
    """Half-open final UTF-8 byte offsets, never host-language string indexes."""

    output_index: int = Field(ge=0, lt=128)
    start: int = Field(ge=0, le=MAX_RENDERED_BYTES)
    end: int = Field(ge=0, le=MAX_RENDERED_BYTES)
    kind: Literal["text", "variable", "separator"]
    origin: PromptOrigin
    variable: str | None


class _RenderWire(_Frozen):
    domain: Literal["strata-prompt-render-v1"] = "strata-prompt-render-v1"
    schema_version: Literal[1] = 1
    build_id: str = Field(pattern=r"^[a-f0-9]{64}$")
    input_fingerprint: str = Field(pattern=r"^[a-f0-9]{64}$")
    output: PortableDocument
    source_map: tuple[SourceSpan, ...] = Field(max_length=MAX_EXPANDED_LEAVES * 2)

    @field_validator("output", mode="before")
    @classmethod
    def _output(cls, value: object) -> PortableDocument:
        return value if isinstance(value, PortableDocument) else PortableDocument.from_value(value)


class PromptBuild(_Frozen):
    """An immutable compiler snapshot; inspection never exposes shared dictionaries."""

    canonical: bytes

    @field_validator("canonical")
    @classmethod
    def _canonical(cls, value: bytes) -> bytes:
        _read_build(value)
        return value

    @property
    def build_id(self) -> str:
        """SHA-256 of the complete domain-bearing canonical build."""
        return hashlib.sha256(self.canonical).hexdigest()

    def wire_payload(self) -> dict[str, PortableValue]:
        """Return a fresh portable view of this exact build."""
        return cast("dict[str, PortableValue]", parse_portable_json(self.canonical))


class PromptRender(_Frozen):
    """Exact output and attribution, independent of any invocation or provider evidence."""

    canonical: bytes

    @field_validator("canonical")
    @classmethod
    def _canonical(cls, value: bytes) -> bytes:
        _read_render(value)
        return value

    @property
    def render_id(self) -> str:
        """SHA-256 of deterministic render content, excluding execution identifiers."""
        return hashlib.sha256(self.canonical).hexdigest()

    def wire_payload(self) -> dict[str, PortableValue]:
        """Return a fresh portable view without retaining a second input-value copy."""
        return cast("dict[str, PortableValue]", parse_portable_json(self.canonical))


def _definition_bytes(definition: PromptDefinition) -> bytes:
    return canonical_bytes(definition.wire_payload())


def _read_render(raw: bytes) -> _RenderWire:
    wire: _RenderWire | None = None
    try:
        parsed = parse_portable_json(raw)
        if canonical_bytes(parsed) == raw:
            wire = _RenderWire.model_validate_json(raw)
    except ValueError, TypeError:
        wire = None
    if wire is None:
        raise PromptCompositionError("render_invalid")
    output = wire.output.wire_payload()
    contents: list[str] = []
    if output.get("kind") == "chat" and set(output) == {"kind", "messages"}:
        messages = output["messages"]
        if type(messages) is not list or not 1 <= len(messages) <= 128:
            raise PromptCompositionError("render_invalid")
        for message in messages:
            if (
                type(message) is not dict
                or set(message) != {"role", "content"}
                or message["role"] not in ("system", "user", "assistant")
                or type(message["content"]) is not str
            ):
                raise PromptCompositionError("render_invalid")
            contents.append(message["content"])
    elif (
        output.get("kind") == "artifact"
        and set(output) == {"kind", "purpose", "content", "descriptor"}
        and output["purpose"] in ("system_instruction", "tool_description", "skill", "text")
        and type(output["content"]) is str
        and (output["descriptor"] is None or type(output["descriptor"]) is dict)
    ):
        contents.append(output["content"])
    else:
        raise PromptCompositionError("render_invalid")
    encoded = [content.encode("utf-8") for content in contents]
    if sum(map(len, encoded)) > MAX_RENDERED_BYTES:
        raise PromptCompositionError("rendered_bytes_limit")
    offsets, leaves = [0] * len(contents), [0] * len(contents)
    expects_leaf = [True] * len(contents)
    last_index = 0
    occurrences: set[tuple[str, tuple[int, ...]]] = set()
    for span in wire.source_map:
        index = span.output_index
        if (
            index < last_index
            or index >= len(contents)
            or span.start != offsets[index]
            or span.end < span.start
            or span.end > len(encoded[index])
        ):
            raise PromptCompositionError("render_invalid")
        valid_boundary = True
        try:
            encoded[index][span.start : span.end].decode("utf-8")
        except UnicodeError:
            valid_boundary = False
        if not valid_boundary:
            raise PromptCompositionError("render_invalid")
        if span.kind == "separator":
            if (
                expects_leaf[index]
                or span.variable is not None
                or span.origin.coordinate != "$compiler"
                or span.origin.occurrence_path != (index, leaves[index])
            ):
                raise PromptCompositionError("render_invalid")
            expects_leaf[index] = True
        else:
            if (
                not expects_leaf[index]
                or len(span.origin.occurrence_path) < 2
                or span.origin.occurrence_path[0] != index
                or (span.kind == "text" and span.variable is not None)
                or (
                    span.kind == "variable"
                    and (
                        span.variable is None
                        or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,63}", span.variable)
                    )
                )
            ):
                raise PromptCompositionError("render_invalid")
            validate_coordinate(span.origin.coordinate)
            occurrence = (span.origin.coordinate, span.origin.occurrence_path)
            if occurrence in occurrences:
                raise PromptCompositionError("render_invalid")
            occurrences.add(occurrence)
            leaves[index] += 1
            expects_leaf[index] = False
        offsets[index] = span.end
        last_index = index
    if any(offset != len(data) for offset, data in zip(offsets, encoded, strict=True)) or any(
        count and expected for count, expected in zip(leaves, expects_leaf, strict=True)
    ):
        raise PromptCompositionError("render_invalid")
    return wire


def _read_build(raw: bytes) -> _BuildWire:
    wire: _BuildWire | None = None
    try:
        parsed = parse_portable_json(raw)
        if canonical_bytes(parsed) == raw:
            wire = _BuildWire.model_validate_json(raw)
    except ValueError, TypeError:
        wire = None
    if wire is None:
        raise PromptCompositionError("build_invalid")
    locks = [lock.coordinate for lock in wire.dependency_lock]
    names = [variable.name for variable in wire.variables]
    facts = [constraint.key for constraint in wire.constraints]
    if (
        locks != sorted(set(locks))
        or wire.surface not in wire.dependency_lock
        or len(names) != len(set(names))
        or len(facts) != len(set(facts))
        or sum(len(output.leaves) for output in wire.outputs) > MAX_EXPANDED_LEAVES
    ):
        raise PromptCompositionError("build_invalid")
    for coordinate in locks:
        validate_coordinate(coordinate)
    occurrences: set[tuple[int, ...]] = set()
    literals: dict[tuple[str, tuple[int, ...]], str] = {}
    for index, output in enumerate(wire.outputs):
        for leaf in output.leaves:
            path = leaf.origin.occurrence_path
            if (
                leaf.origin.coordinate not in locks
                or len(path) < 2
                or path[0] != index
                or path in occurrences
            ):
                raise PromptCompositionError("build_invalid")
            occurrences.add(path)
            if leaf.kind == "text" and leaf.text is not None and leaf.variable is None:
                literals[leaf.origin.coordinate, path] = leaf.text
            elif leaf.kind != "variable" or leaf.text is not None or leaf.variable not in names:
                raise PromptCompositionError("build_invalid")
    for fact in wire.constraints:
        if not fact.origins or any(origin.coordinate not in locks for origin in fact.origins):
            raise PromptCompositionError("build_invalid")
    for diagnostic in wire.diagnostics:
        passages = [
            literals.get((origin.coordinate, origin.occurrence_path))
            for origin in diagnostic.origins
        ]
        if (
            diagnostic.code != "duplicate_literal"
            or diagnostic.severity != "advisory"
            or len(passages) < 2
            or not passages[0]
            or any(text != passages[0] for text in passages)
        ):
            raise PromptCompositionError("build_invalid")
    if wire.kind == "chat":
        if (
            wire.purpose is not None
            or wire.descriptor is not None
            or any(output.role is None for output in wire.outputs)
        ):
            raise PromptCompositionError("build_invalid")
    elif len(wire.outputs) != 1 or wire.outputs[0].role is not None or wire.purpose is None:
        raise PromptCompositionError("build_invalid")
    return wire


def definition_identity(definition: PromptDefinition) -> str:
    """Hash authored content and provenance with an explicit definition domain."""
    checked = definition_from_wire(definition.wire_payload())
    return portable_sha256(
        {"domain": "strata-prompt-definition-v1", "definition": checked.wire_payload()}
    )


class CompositionRegistry:
    """An in-process exact-version snapshot registry, without channels or publication."""

    def __init__(self) -> None:
        self._definitions: dict[str, bytes] = {}
        self._bytes = 0

    async def put(self, definition: PromptDefinition | object) -> str:
        """Add an immutable coordinate; identical repeats are idempotent."""
        value = (
            definition.wire_payload()
            if isinstance(definition, (ModuleRevision, SurfaceRevision))
            else definition
        )
        checked = definition_from_wire(value)
        coordinate = checked.coordinate
        encoded = _definition_bytes(checked)
        prior = self._definitions.get(coordinate)
        if prior is not None:
            if prior != encoded:
                raise PromptCompositionError(
                    "revision_replaced", (PromptOrigin(coordinate=coordinate),)
                )
            return coordinate
        if len(self._definitions) >= MAX_DEFINITIONS:
            raise PromptCompositionError("registry_definition_limit")
        if self._bytes + len(encoded) > MAX_DEFINITION_BYTES:
            raise PromptCompositionError("registry_bytes_limit")
        self._definitions[coordinate] = encoded
        self._bytes += len(encoded)
        return coordinate

    async def get(self, coordinate: str) -> PromptDefinition:
        """Resolve one exact version to an independent validated value."""
        validate_coordinate(coordinate)
        encoded = self._definitions.get(coordinate)
        if encoded is None:
            raise PromptCompositionError(
                "dependency_missing", (PromptOrigin(coordinate=coordinate),)
            )
        return definition_from_wire(parse_portable_json(encoded))


def _parts(definition: PromptDefinition) -> tuple[PromptPart, ...]:
    if isinstance(definition, ModuleRevision):
        return definition.parts
    if isinstance(definition.target, ArtifactTarget):
        return definition.target.parts
    return tuple(part for message in definition.target.messages for part in message.parts)


def _lock(definition: PromptDefinition) -> DependencyLock:
    return DependencyLock(
        coordinate=definition.coordinate,
        content_sha256=definition_identity(definition),
        provenance=definition.provenance,
    )


async def compile_surface(registry: CompositionRegistry, coordinate: str) -> PromptBuild:
    """Resolve exact module pins into a bounded immutable leaf graph and dependency lock."""
    root = await registry.get(coordinate)
    if not isinstance(root, SurfaceRevision):
        raise PromptCompositionError("surface_required", (PromptOrigin(coordinate=coordinate),))
    definitions: dict[str, PromptDefinition] = {coordinate: root}
    counts: dict[str, int] = {}
    heights: dict[str, int] = {}
    active: list[str] = []

    async def resolve(definition: PromptDefinition) -> None:
        key = definition.coordinate
        if key in active:
            raise PromptCompositionError(
                "dependency_cycle", tuple(PromptOrigin(coordinate=name) for name in (*active, key))
            )
        if key in counts:
            return
        active.append(key)
        if len(active) > MAX_DEPENDENCY_DEPTH + 1:
            raise PromptCompositionError("dependency_depth_limit", (PromptOrigin(coordinate=key),))
        count, height = 0, 0
        declared = {variable.name for variable in definition.variables}
        for part in _parts(definition):
            if isinstance(part, VariablePart) and part.name not in declared:
                raise PromptCompositionError("variable_undeclared", (PromptOrigin(coordinate=key),))
            if isinstance(part, ModulePart):
                child = definitions.get(part.revision)
                if child is None:
                    child = await registry.get(part.revision)
                    definitions[part.revision] = child
                if not isinstance(child, ModuleRevision):
                    raise PromptCompositionError(
                        "module_required", (PromptOrigin(coordinate=part.revision),)
                    )
                await resolve(child)
                count += counts[child.coordinate]
                height = max(height, heights[child.coordinate])
            else:
                count += 1
            if count > MAX_EXPANDED_LEAVES:
                raise PromptCompositionError(
                    "expanded_leaves_limit", (PromptOrigin(coordinate=key),)
                )
        heights[key] = height + (0 if isinstance(definition, SurfaceRevision) else 1)
        if heights[key] > MAX_DEPENDENCY_DEPTH:
            raise PromptCompositionError("dependency_depth_limit", (PromptOrigin(coordinate=key),))
        counts[key] = count
        active.pop()

    await resolve(root)
    required = {variable.name: variable.type for variable in root.variables}
    facts: dict[str, tuple[str, list[PromptOrigin]]] = {}
    for key in sorted(definitions):
        definition = definitions[key]
        for variable in definition.variables:
            if variable.name not in required or required[variable.name] != variable.type:
                raise PromptCompositionError(
                    "variable_type_conflict",
                    (
                        PromptOrigin(coordinate=root.coordinate),
                        PromptOrigin(coordinate=key),
                    ),
                )
        for constraint in definition.constraints:
            origin = PromptOrigin(coordinate=key)
            prior = facts.get(constraint.key)
            if prior is not None and prior[0] != constraint.value:
                raise PromptCompositionError("constraint_conflict", (prior[1][0], origin))
            if prior is None:
                facts[constraint.key] = (constraint.value, [origin])
            else:
                prior[1].append(origin)
    visits = 0

    def expand(
        parts: tuple[PromptPart, ...], owner: str, path: tuple[int, ...]
    ) -> list[CompiledLeaf]:
        nonlocal visits
        leaves: list[CompiledLeaf] = []
        for index, part in enumerate(parts):
            visits += 1
            if visits > MAX_EXPANSION_VISITS:
                raise PromptCompositionError("expansion_visits_limit")
            occurrence = (*path, index)
            if isinstance(part, ModulePart):
                child = definitions[part.revision]
                if counts[child.coordinate]:
                    leaves.extend(expand(_parts(child), child.coordinate, occurrence))
            else:
                leaves.append(
                    CompiledLeaf(
                        kind=part.kind,
                        text=part.text if isinstance(part, TextPart) else None,
                        variable=part.name if isinstance(part, VariablePart) else None,
                        origin=PromptOrigin(coordinate=owner, occurrence_path=occurrence),
                    )
                )
        return leaves

    target = root.target
    if isinstance(target, ChatTarget):
        outputs = tuple(
            CompiledOutput(
                role=message.role,
                separator=message.separator,
                leaves=tuple(expand(message.parts, root.coordinate, (index,))),
            )
            for index, message in enumerate(target.messages)
        )
        kind, purpose, descriptor = "chat", None, None
    else:
        outputs = (
            CompiledOutput(
                role=None,
                separator=target.separator,
                leaves=tuple(expand(target.parts, root.coordinate, (0,))),
            ),
        )
        kind, purpose, descriptor = "artifact", target.purpose, target.descriptor
    passages: dict[str, list[PromptOrigin]] = {}
    for output in outputs:
        for leaf in output.leaves:
            if leaf.kind == "text" and leaf.text:
                passages.setdefault(leaf.text, []).append(leaf.origin)
    diagnostics = tuple(
        CompositionDiagnostic(
            code="duplicate_literal",
            severity="advisory",
            origins=tuple(origins),
        )
        for origins in passages.values()
        if len(origins) > 1
    )
    if len(diagnostics) > MAX_DIAGNOSTICS:
        raise PromptCompositionError("diagnostics_limit")
    wire = _BuildWire(
        surface=_lock(root),
        dependency_lock=tuple(_lock(definitions[key]) for key in sorted(definitions)),
        variables=root.variables,
        constraints=tuple(
            ResolvedConstraint(
                key=key,
                value=value,
                origins=tuple(origins),
            )
            for key, (value, origins) in sorted(facts.items())
        ),
        kind=kind,
        purpose=purpose,
        descriptor=descriptor,
        outputs=outputs,
        diagnostics=diagnostics,
    )
    return PromptBuild(canonical=canonical_bytes(wire.model_dump(mode="json")))


async def render_build(build: PromptBuild, values: object) -> PromptRender:
    """Render exact scalar inputs once and attribute every final UTF-8 byte span."""
    wire = _read_build(build.canonical)
    if type(values) is not dict:
        raise PromptCompositionError("input_invalid")
    inputs = cast("dict[str, PortableValue]", values)
    canonical_bytes(inputs)
    if set(inputs) != {variable.name for variable in wire.variables}:
        raise PromptCompositionError("input_keys_mismatch")
    spellings: dict[str, str] = {}
    for variable in wire.variables:
        value = inputs[variable.name]
        if variable.type == "string" and type(value) is str:
            spellings[variable.name] = value
        elif variable.type == "integer" and type(value) is int and abs(value) <= MAX_SAFE_INTEGER:
            spellings[variable.name] = str(value)
        elif variable.type == "boolean" and type(value) is bool:
            spellings[variable.name] = "true" if value else "false"
        else:
            raise PromptCompositionError("input_type_mismatch")
    spans: list[SourceSpan] = []
    texts: list[str] = []
    total_bytes = 0
    for index, output in enumerate(wire.outputs):
        chunks: list[str] = []
        offset = 0
        for leaf_index, leaf in enumerate(output.leaves):
            emissions: list[
                tuple[str, Literal["text", "variable", "separator"], PromptOrigin, str | None]
            ] = []
            if leaf_index:
                emissions.append(
                    (
                        output.separator,
                        "separator",
                        PromptOrigin(
                            coordinate="$compiler",
                            occurrence_path=(index, leaf_index),
                        ),
                        None,
                    )
                )
            if leaf.kind == "text" and leaf.text is not None and leaf.variable is None:
                text = leaf.text
            elif leaf.kind == "variable" and leaf.variable in spellings and leaf.text is None:
                text = spellings[leaf.variable]
            else:
                raise PromptCompositionError("build_invalid")
            emissions.append((text, leaf.kind, leaf.origin, leaf.variable))
            for text, kind, origin, variable in emissions:
                length = len(text.encode("utf-8"))
                total_bytes += length
                if total_bytes > MAX_RENDERED_BYTES:
                    raise PromptCompositionError("rendered_bytes_limit")
                spans.append(
                    SourceSpan(
                        output_index=index,
                        start=offset,
                        end=offset + length,
                        kind=kind,
                        origin=origin,
                        variable=variable,
                    )
                )
                offset += length
                chunks.append(text)
        texts.append("".join(chunks))
    output_value: dict[str, object]
    if wire.kind == "chat":
        if (
            wire.purpose is not None
            or wire.descriptor is not None
            or any(item.role is None for item in wire.outputs)
        ):
            raise PromptCompositionError("build_invalid")
        output_value = {
            "kind": "chat",
            "messages": [
                {"role": item.role, "content": text}
                for item, text in zip(wire.outputs, texts, strict=True)
            ],
        }
    else:
        if len(wire.outputs) != 1 or wire.outputs[0].role is not None or wire.purpose is None:
            raise PromptCompositionError("build_invalid")
        output_value = {
            "kind": "artifact",
            "purpose": wire.purpose,
            "content": texts[0],
            "descriptor": wire.descriptor.wire_payload() if wire.descriptor is not None else None,
        }
    rendered = _RenderWire(
        build_id=build.build_id,
        input_fingerprint=portable_sha256({"domain": "strata-prompt-input-v1", "values": inputs}),
        output=PortableDocument.from_value(output_value),
        source_map=tuple(spans),
    )
    return PromptRender(canonical=canonical_bytes(rendered.model_dump(mode="json")))
