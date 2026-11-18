"""Compiled prompt builds: one authoritative compile and render for versioned prompts.

A versioned prompt is an ordered composition of **parts** (reusable modules, other prompts and the
prompt's own instructions) plus **typed variable declarations**. :func:`compile_prompt` turns that
source into an immutable :class:`PromptBuild`: a flat list of literal-text and variable segments,
the declarations it needs, a source map from every output line back to the part and line it came
from, and a SHA-256 ``hash`` of the build's canonical content. :func:`render_build` turns a build
plus variable values into the exact text a model receives.

Every consumer renders through the same two functions — a control plane previewing a build, a
batch runner rendering one row at a time on a remote machine — so the text can never drift
between where a prompt is reviewed and where it runs. A build validates its own hash when it is
constructed, so a build that arrives as data (a run spec, a stored record) is refused if anything
in it was changed after compilation.

Two source dialects, chosen per part, both parsed WITHOUT executing anything:

- ``"strata"``: ``{{ name }}`` interpolates a declared variable. Nothing else is an expression: a
  ``{{`` that does not hold exactly one identifier, an unclosed ``{{``, and Jinja block or comment
  tags (``{%``, ``{#``) are compile errors rather than literal text, so a template written for a
  richer engine never renders silently wrong. A single brace is literal, so JSON examples need no
  escaping.
- ``"braces"``: the batch runner's legacy ``{name}`` placeholders (``[A-Za-z0-9_]+`` between single
  braces), every other character literal. A runner template imports into a build exactly as the
  runner renders it, braces included.

Rendering converts each value by its declared type with one canonical text form that does not
depend on the host language: integers and floats use the ECMAScript ``Number`` text form
(``1.0`` -> ``"1"``, ``1e-07`` -> ``"1e-7"``), booleans are ``true`` / ``false``, structured values
are compact JSON with sorted keys. ``tests/fixtures/prompt_build_vectors.json`` pins this
behaviour for ports of the renderer.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from decimal import Decimal
from typing import TYPE_CHECKING, Annotated, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from strata_forge.prompts.template import PromptError

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

__all__ = [
    "BUILD_FORMAT",
    "COMPILER_VERSION",
    "Binding",
    "CompileIssue",
    "CompileResult",
    "Dialect",
    "PromptBuild",
    "PromptBuildError",
    "PromptPart",
    "PromptRenderError",
    "Segment",
    "SourceSpan",
    "VariableSpec",
    "VariableType",
    "bind_row",
    "compile_prompt",
    "parse_part",
    "render_build",
    "value_text",
]

#: The compiler's own version, recorded on every build. A build's ``hash`` covers its content,
#: not this, so a compiler upgrade that produces the same segments produces the same hash.
COMPILER_VERSION = "1.0.0"
#: The build schema; part of the hashed content, so a future schema can never collide with this.
BUILD_FORMAT = "strata.prompt-build/1"

Dialect = Literal["strata", "braces"]
VariableType = Literal["string", "number", "boolean", "enum", "json"]

# Variable names: the legacy runner accepted `[A-Za-z0-9_]+`, so a leading digit stays legal.
_NAME = r"[A-Za-z0-9_]{1,64}"
_NAME_RE = re.compile(rf"^{_NAME}$")
_BRACES_RE = re.compile(r"\{([A-Za-z0-9_]+)\}")
_EXPR_RE = re.compile(rf"\{{\{{[ \t]*({_NAME})[ \t]*\}}\}}")
_MAX_PART_CHARS = 64_000
_MAX_PARTS = 64
_MAX_VARIABLES = 128
_MAX_ENUM_VALUES = 256
#: Parts are joined with one blank line between them.
PART_SEPARATOR = "\n\n"


class PromptBuildError(PromptError):
    """A build failed its integrity check, or a compile was asked for something it cannot hold."""


class PromptRenderError(PromptError):
    """A value could not be rendered: missing, of the wrong type, or not representable as text."""


class VariableSpec(BaseModel):
    """One declared variable: its name, its type and whether a render must supply it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: Annotated[str, Field(pattern=rf"^{_NAME}$")]
    type: VariableType = "string"
    required: bool = True
    # Used when the variable is optional and no value is supplied; must render under `type`.
    default: JsonValue = None
    # The allowed values of an `enum` variable; set on enums only.
    values: tuple[str, ...] | None = None
    description: Annotated[str, Field(max_length=500)] = ""

    @model_validator(mode="after")
    def _consistent(self) -> VariableSpec:
        if self.type == "enum":
            if not self.values:
                msg = f"enum variable {self.name!r} declares no values"
                raise ValueError(msg)
            if len(self.values) > _MAX_ENUM_VALUES or len(set(self.values)) != len(self.values):
                msg = f"enum variable {self.name!r} needs 1-{_MAX_ENUM_VALUES} distinct values"
                raise ValueError(msg)
        elif self.values is not None:
            msg = f"only an enum variable declares values ({self.name!r} is {self.type})"
            raise ValueError(msg)
        if self.default is not None:
            if self.required:
                msg = f"required variable {self.name!r} cannot have a default"
                raise ValueError(msg)
            try:
                value_text(self, self.default)
            except PromptRenderError as exc:
                msg = f"default of {self.name!r} does not fit its type: {exc}"
                raise ValueError(msg) from None
        return self


class PromptPart(BaseModel):
    """One piece of a composition, resolved to its text by the caller (in composition order).

    ``ref`` and ``label`` are the caller's identifiers (an id and a "v6" / "r9" version label);
    the compiler records them in the source map and never interprets them.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["module", "prompt", "self"]
    name: Annotated[str, Field(min_length=1, max_length=200)]
    ref: Annotated[str, Field(max_length=200)] | None = None
    label: Annotated[str, Field(max_length=40)] | None = None
    text: str
    dialect: Dialect = "strata"


class TextSegment(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["text"] = "text"
    text: str


class VariableSegment(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["var"] = "var"
    name: Annotated[str, Field(pattern=rf"^{_NAME}$")]


Segment = Annotated[TextSegment | VariableSegment, Field(discriminator="kind")]


class SourceSpan(BaseModel):
    """Output lines ``line_start..line_end`` (1-based, inclusive, in the template text) came from
    lines ``part_line_start..`` of part ``part`` (its index in :attr:`PromptBuild.parts`)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    part: int
    line_start: int
    line_end: int
    part_line_start: int


class PartDigest(BaseModel):
    """What a build was compiled from: each part's identity and the SHA-256 of its canonical text."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["module", "prompt", "self"]
    name: str
    ref: str | None = None
    label: str | None = None
    dialect: Dialect = "strata"
    sha256: str


class CompileIssue(BaseModel):
    """A compiler finding. Every issue is an error: a build with any issue is not produced."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    code: Literal[
        "unsupported_expression",
        "unclosed_expression",
        "unsupported_block",
        "undeclared_variable",
        "duplicate_variable",
        "invalid_declaration",
        "too_large",
        "empty_prompt",
    ]
    message: str
    # The part index the issue is in, with its 1-based line and column; None for a declaration.
    part: int | None = None
    line: int | None = None
    column: int | None = None
    variable: str | None = None


def _canonical_text(text: str) -> str:
    """Unicode NFC with every line ending as ``\\n``: the form a build's content is hashed in."""
    return unicodedata.normalize("NFC", text.replace("\r\n", "\n").replace("\r", "\n"))


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _line_col(text: str, offset: int) -> tuple[int, int]:
    line = text.count("\n", 0, offset) + 1
    return line, offset - (text.rfind("\n", 0, offset) + 1) + 1


def parse_part(
    text: str, dialect: Dialect = "strata", *, part: int | None = None
) -> tuple[list[TextSegment | VariableSegment], list[CompileIssue]]:
    """Split one part's (canonical) text into segments, with any syntax issues.

    Exposed so an editor can list a source's variables and syntax errors exactly as a build
    would, without compiling the whole composition.
    """
    segments: list[TextSegment | VariableSegment] = []
    issues: list[CompileIssue] = []
    if dialect == "braces":
        pos = 0
        for match in _BRACES_RE.finditer(text):
            if match.start() > pos:
                segments.append(TextSegment(text=text[pos : match.start()]))
            segments.append(VariableSegment(name=match.group(1)))
            pos = match.end()
        if pos < len(text):
            segments.append(TextSegment(text=text[pos:]))
        return segments, issues
    pos = 0
    literal_from = 0
    while True:
        brace = text.find("{", pos)
        if brace == -1 or brace + 1 >= len(text):
            break
        nxt = text[brace + 1]
        if nxt in "%#":
            line, col = _line_col(text, brace)
            issues.append(
                CompileIssue(
                    code="unsupported_block",
                    message=(
                        f"'{{{nxt}' tags are not supported (line {line}): a prompt interpolates "
                        "variables with {{ name }} and composes modules instead of control flow"
                    ),
                    part=part,
                    line=line,
                    column=col,
                )
            )
            pos = brace + 2
            continue
        if nxt != "{":
            pos = brace + 1
            continue
        match = _EXPR_RE.match(text, brace)
        if match is not None:
            if brace > literal_from:
                segments.append(TextSegment(text=text[literal_from:brace]))
            segments.append(VariableSegment(name=match.group(1)))
            pos = literal_from = match.end()
            continue
        line, col = _line_col(text, brace)
        close = text.find("}}", brace + 2)
        newline = text.find("\n", brace + 2)
        if close == -1 or (newline != -1 and newline < close):
            issues.append(
                CompileIssue(
                    code="unclosed_expression",
                    message=f"'{{{{' at line {line}, column {col} is never closed with '}}}}'",
                    part=part,
                    line=line,
                    column=col,
                )
            )
            pos = brace + 2
            continue
        inner = text[brace + 2 : close].strip()
        issues.append(
            CompileIssue(
                code="unsupported_expression",
                message=(
                    f"'{{{{ {inner} }}}}' at line {line} is not a variable: only "
                    "{{ name }} is supported (letters, digits and underscores)"
                ),
                part=part,
                line=line,
                column=col,
            )
        )
        pos = close + 2
    if literal_from < len(text):
        segments.append(TextSegment(text=text[literal_from:]))
    return segments, issues


def _first_use(text: str, dialect: Dialect, name: str) -> int:
    """Where ``name`` is first interpolated in ``text`` (for an issue's line and column)."""
    pattern = _BRACES_RE if dialect == "braces" else _EXPR_RE
    return next((m.start() for m in pattern.finditer(text) if m.group(1) == name), 0)


def _merged(segments: list[TextSegment | VariableSegment]) -> list[TextSegment | VariableSegment]:
    """Adjacent text segments joined and empty ones dropped: the canonical segment list."""
    out: list[TextSegment | VariableSegment] = []
    for segment in segments:
        if isinstance(segment, TextSegment):
            if not segment.text:
                continue
            last = out[-1] if out else None
            if isinstance(last, TextSegment):
                out[-1] = TextSegment(text=last.text + segment.text)
                continue
        out.append(segment)
    return out


def _content_hash(
    segments: Sequence[TextSegment | VariableSegment], variables: Sequence[VariableSpec]
) -> str:
    content = {
        "format": BUILD_FORMAT,
        "segments": [segment.model_dump(mode="json") for segment in segments],
        "variables": [variable.model_dump(mode="json") for variable in variables],
    }
    return _sha256(_canonical_json(content))


class PromptBuild(BaseModel):
    """An immutable compiled prompt. Constructing one re-checks ``hash`` against its content."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    format: Literal["strata.prompt-build/1"] = BUILD_FORMAT
    compiler_version: str
    segments: tuple[Segment, ...]
    # The declarations the segments use, in name order. A render reads values for these only.
    variables: tuple[VariableSpec, ...]
    parts: tuple[PartDigest, ...] = ()
    source_map: tuple[SourceSpan, ...] = ()
    hash: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]

    @model_validator(mode="after")
    def _intact(self) -> PromptBuild:
        names = [variable.name for variable in self.variables]
        if names != sorted(set(names)):
            msg = "build variables must be unique and in name order"
            raise PromptBuildError(msg)
        used = {s.name for s in self.segments if isinstance(s, VariableSegment)}
        if used != set(names):
            msg = "build variables must be exactly the variables its segments use"
            raise PromptBuildError(msg)
        if _content_hash(self.segments, self.variables) != self.hash:
            msg = "prompt build hash does not match its content: the build was altered"
            raise PromptBuildError(msg)
        return self

    @property
    def template(self) -> str:
        """The build as text, every variable shown as ``{{ name }}`` (a display form)."""
        return "".join(
            s.text if isinstance(s, TextSegment) else f"{{{{ {s.name} }}}}" for s in self.segments
        )

    def variable(self, name: str) -> VariableSpec | None:
        return next((v for v in self.variables if v.name == name), None)


class CompileResult(BaseModel):
    """``build`` when the source compiled; ``issues`` (all errors) otherwise."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    build: PromptBuild | None
    issues: tuple[CompileIssue, ...] = ()
    # Every variable referenced anywhere in the source, in first-use order (also on failure).
    referenced: tuple[str, ...] = ()


def compile_prompt(
    parts: Sequence[PromptPart], variables: Sequence[VariableSpec] = ()
) -> CompileResult:
    """Compile an ordered composition into a :class:`PromptBuild`, or report why it cannot be.

    Parts are canonicalized (NFC, ``\\n`` line endings), stripped of trailing newlines, and joined
    with one blank line; an empty part contributes nothing. Every referenced variable must be
    declared exactly once; a declaration no part uses is left out of the build.
    """
    if len(parts) > _MAX_PARTS or len(variables) > _MAX_VARIABLES:
        msg = f"a prompt holds at most {_MAX_PARTS} parts and {_MAX_VARIABLES} variables"
        raise PromptBuildError(msg)
    issues: list[CompileIssue] = []
    declared: dict[str, VariableSpec] = {}
    for variable in variables:
        if variable.name in declared:
            issues.append(
                CompileIssue(
                    code="duplicate_variable",
                    message=f"variable {variable.name!r} is declared more than once",
                    variable=variable.name,
                )
            )
        declared[variable.name] = variable
    segments: list[TextSegment | VariableSegment] = []
    digests: list[PartDigest] = []
    spans: list[SourceSpan] = []
    referenced: dict[str, None] = {}
    line = 1
    for index, part in enumerate(parts):
        text = _canonical_text(part.text).rstrip("\n")
        digests.append(
            PartDigest(
                kind=part.kind,
                name=part.name,
                ref=part.ref,
                label=part.label,
                dialect=part.dialect,
                sha256=_sha256(text),
            )
        )
        if len(text) > _MAX_PART_CHARS:
            issues.append(
                CompileIssue(
                    code="too_large",
                    message=f"{part.name} is longer than {_MAX_PART_CHARS:,} characters",
                    part=index,
                )
            )
            continue
        if not text:
            continue
        part_segments, part_issues = parse_part(text, part.dialect, part=index)
        issues.extend(part_issues)
        for segment in part_segments:
            if isinstance(segment, VariableSegment):
                referenced.setdefault(segment.name, None)
                if segment.name not in declared:
                    at_line, at_col = _line_col(text, _first_use(text, part.dialect, segment.name))
                    issues.append(
                        CompileIssue(
                            code="undeclared_variable",
                            message=f"{segment.name} is used but not declared",
                            part=index,
                            line=at_line,
                            column=at_col,
                            variable=segment.name,
                        )
                    )
        if segments:
            segments.append(TextSegment(text=PART_SEPARATOR))
            line += PART_SEPARATOR.count("\n")
        lines = text.count("\n") + 1
        spans.append(
            SourceSpan(part=index, line_start=line, line_end=line + lines - 1, part_line_start=1)
        )
        line += lines - 1
        segments.extend(part_segments)
    if not segments and not issues:
        issues.append(CompileIssue(code="empty_prompt", message="the prompt has no text"))
    names = tuple(referenced)
    if issues:
        return CompileResult(build=None, issues=tuple(issues), referenced=names)
    canonical = _merged(segments)
    used = sorted(referenced)
    build_variables = tuple(declared[name] for name in used)
    return CompileResult(
        build=PromptBuild(
            compiler_version=COMPILER_VERSION,
            segments=tuple(canonical),
            variables=build_variables,
            parts=tuple(digests),
            source_map=tuple(spans),
            hash=_content_hash(canonical, build_variables),
        ),
        referenced=names,
    )


# --- rendering -------------------------------------------------------------------------------


def _es_number(value: float) -> str:
    """``value`` as ECMAScript's ``Number::toString`` writes it (shortest round-trip digits)."""
    if math.isnan(value) or math.isinf(value):
        msg = "NaN and infinite numbers have no text form"
        raise PromptRenderError(msg)
    if value == 0:
        return "0"
    sign = "-" if value < 0 else ""
    sign_bit, digit_tuple, exponent = Decimal(repr(abs(value))).as_tuple()
    del sign_bit
    if not isinstance(exponent, int):
        msg = "number has no finite text form"
        raise PromptRenderError(msg)
    digits = "".join(str(d) for d in digit_tuple).rstrip("0") or "0"
    exponent += len("".join(str(d) for d in digit_tuple)) - len(digits)
    k = len(digits)
    n = k + exponent
    if k <= n <= 21:
        body = digits + "0" * (n - k)
    elif 0 < n <= 21:
        body = f"{digits[:n]}.{digits[n:]}"
    elif -6 < n <= 0:
        body = "0." + "0" * (-n) + digits
    else:
        e = n - 1
        mantissa = digits if k == 1 else f"{digits[0]}.{digits[1:]}"
        body = f"{mantissa}e{'+' if e >= 0 else '-'}{abs(e)}"
    return sign + body


def _canonical_json(value: object) -> str:
    """Compact JSON with sorted keys and ECMAScript number text: one form for every host."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return _es_number(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, (list, tuple)):
        items = cast("Sequence[object]", value)
        return "[" + ",".join(_canonical_json(item) for item in items) + "]"
    if isinstance(value, dict):
        entries = cast("dict[object, object]", value)
        keys: list[str] = []
        for key in entries:
            if not isinstance(key, str):
                msg = "JSON object keys must be strings"
                raise PromptRenderError(msg)
            keys.append(key)
        return (
            "{"
            + ",".join(
                f"{json.dumps(k, ensure_ascii=False)}:{_canonical_json(entries[k])}"
                for k in sorted(keys)
            )
            + "}"
        )
    msg = f"a {type(value).__name__} value has no text form"
    raise PromptRenderError(msg)


def value_text(variable: VariableSpec, value: object) -> str:
    """``value`` as the text a render inserts for ``variable``, checked against its type."""
    kind = variable.type
    if kind == "json":
        return _canonical_json(value)
    if value is None:
        msg = f"{variable.name} has no value"
        raise PromptRenderError(msg)
    if kind == "boolean":
        if not isinstance(value, bool):
            msg = f"{variable.name} must be a boolean"
            raise PromptRenderError(msg)
        return "true" if value else "false"
    if kind == "number":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            msg = f"{variable.name} must be a number"
            raise PromptRenderError(msg)
        return str(value) if isinstance(value, int) else _es_number(value)
    if kind == "enum":
        if not isinstance(value, str) or value not in (variable.values or ()):
            msg = f"{variable.name} must be one of its declared values"
            raise PromptRenderError(msg)
        return value
    if isinstance(value, str):
        return value
    return _canonical_json(value)


def render_build(build: PromptBuild, values: Mapping[str, object]) -> str:
    """The text ``build`` renders to with ``values`` (variable name -> value).

    A variable with no value (absent or ``None``) is refused when required; an optional one
    renders its default, or nothing. Values for names the build does not declare are ignored.
    """
    rendered: dict[str, str] = {}
    for variable in build.variables:
        value = values.get(variable.name)
        if value is None:
            if variable.required:
                msg = f"required variable {variable.name} has no value"
                raise PromptRenderError(msg)
            default = variable.default
            rendered[variable.name] = "" if default is None else value_text(variable, default)
        else:
            rendered[variable.name] = value_text(variable, value)
    return "".join(
        s.text if isinstance(s, TextSegment) else rendered[s.name] for s in build.segments
    )


class Binding(BaseModel):
    """Where a variable's value comes from in a per-row render: a column, or one fixed value."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    column: Annotated[str, Field(min_length=1, max_length=200)] | None = None
    value: JsonValue = None

    @model_validator(mode="after")
    def _one_source(self) -> Binding:
        if (self.column is None) == (self.value is None):
            msg = "a binding names exactly one of column or value"
            raise ValueError(msg)
        return self


def bind_row(
    build: PromptBuild, bindings: Mapping[str, Binding], row: Mapping[str, object]
) -> dict[str, object]:
    """The render values for one dataset ``row``: each binding's column value or fixed value.

    A binding for a column the row does not have is an error (a misspelt column must not render
    as a missing optional value); a variable with no binding is simply absent.
    """
    values: dict[str, object] = {}
    for name, binding in bindings.items():
        if build.variable(name) is None:
            continue
        if binding.column is not None:
            if binding.column not in row:
                msg = f"{name} is bound to column {binding.column!r}, which the row does not have"
                raise PromptRenderError(msg)
            values[name] = row[binding.column]
        else:
            values[name] = binding.value
    return values
