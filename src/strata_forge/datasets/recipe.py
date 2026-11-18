"""Dataset recipes: a declarative, versioned record that turns pinned sources into a new dataset.

A recipe names its **sources** (Hugging Face dataset repos, each at a full 40-character commit, one
config and one split) and an ordered list of **steps** that run over one working table:

- ``load`` starts the table from a source; ``concat`` appends another source's rows; ``join``
  matches another source's rows on key columns (inner or left);
- ``filter`` keeps the rows that meet its conditions; ``map_columns`` derives, selects and renames
  columns; ``dedupe`` keeps the first row of each key; ``split`` assigns output splits by seeded
  ratios (optionally per stratum) and must come last;
- ``synthesize`` generates candidate rows from the table's rows with a pinned model, a prompt
  template, a JSON schema the output is held to, ``n`` samples per row and a per-request seed.
  Every generated row is marked synthetic and carries its seed row's id.

Every row carries a stable id: a source row is ``<source>:<index>`` (its position in the pinned
split), a joined row ``<left>+<right>``, a generated one ``gen:<seed row>:<sample>``. A row a step
drops is recorded as an exclusion with its reason, so a materialized version can say exactly what
it left out.

Held-out data never reaches generation, and is never relabelled as training data. A source's
``split`` is one split name with at most a ``[start:end]`` row slice (no ``+`` combinations, no
percent, parsed with the engine's split-syntax parser), and its name is classified by normalized
tokens (:func:`split_class`): ``train`` only when every token says training, ``held_out`` when any
token says test, validation, dev, eval or held-out, and ``ambiguous`` otherwise. A recipe that
generates may read ``train`` sources only (an ambiguous name is refused like a held-out one), and a
``synthesize`` step re-checks every seed row's origin. A recipe that reads any source not
classified ``train`` must name every output split as held out, so its output cannot be read back
as training data by a later recipe. Generation itself is injected (:class:`Generator`), so this
module never loads a model; the recipe runner supplies one that calls the batch engine, and tests
supply a fake.

Pure Python over ``dict`` rows, bounded by ``row_cap`` rows and ``byte_cap`` bytes held (both
checked while a source is loaded and while a join is built, not only after); no I/O.
"""

from __future__ import annotations

import itertools
import json
import math
import random
import re
from collections.abc import Sized
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Annotated, Any, Literal, Protocol, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)

from strata_forge.core.errors import ValidationError
from strata_forge.core.repro import content_hash
from strata_forge.pipelines._guided import schema_problem
from strata_forge.pipelines._split_overlap import parse_single_split

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

__all__ = [
    "MAX_BYTE_CAP",
    "MAX_RECIPE_BYTES",
    "RECIPE_SCHEMA",
    "Condition",
    "Exclusion",
    "GenerationRequest",
    "GenerationResult",
    "Generator",
    "Recipe",
    "RecipeError",
    "RecipeOutput",
    "Row",
    "Source",
    "SplitClass",
    "StepCount",
    "Table",
    "diff_tables",
    "execute",
    "profile_table",
    "recipe_sha256",
    "render_prompt",
    "split_class",
]

RECIPE_SCHEMA = "strata.dataset_recipe/v1"
_NAME = r"^[a-z][a-z0-9_-]{0,63}$"
_COLUMN = r"^[A-Za-z_][A-Za-z0-9_ .-]{0,127}$"
_SHA = r"^[0-9a-f]{40}$"
_REPO = r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}/[A-Za-z0-9][A-Za-z0-9._-]{0,95}$"
_PLACEHOLDER = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")
#: Column names the engine writes into every output row; a recipe may not produce them itself.
RESERVED_COLUMNS = frozenset({"_row_id", "_synthetic", "_seed_row_id"})
MAX_ROW_CAP = 2_000_000
#: The most bytes of row values one execution holds (estimated from the values themselves).
MAX_BYTE_CAP = 16 * 1024**3
DEFAULT_BYTE_CAP = 2 * 1024**3
#: The largest recipe document, as canonical JSON.
MAX_RECIPE_BYTES = 256 * 1024
_MAX_TEXT = 10_000

type Scalar = (
    Annotated[StrictStr, Field(max_length=_MAX_TEXT)] | StrictInt | StrictFloat | StrictBool | None
)
type Column = Annotated[str, Field(pattern=_COLUMN)]

type SplitClass = Literal["train", "held_out", "ambiguous"]

_TRAIN_TOKENS = frozenset({"train", "training"})
_FILLER_TOKEN = re.compile(r"^(?:v|v?\d+)$")
_HELD_OUT_TOKENS = frozenset(
    {
        "val",
        "dev",
        "test",
        "tests",
        "testing",
        "valid",
        "validation",
        "validate",
        "eval",
        "evals",
        "evaluation",
        "evaluate",
        "holdout",
        "heldout",
        "hidden",
        "blind",
        "unseen",
        "gold",
    }
)
#: Held-out words that also count inside a longer token (``testset``, ``devtest``, ``evalv2``).
_HELD_OUT_PARTS = ("test", "valid", "eval", "holdout", "heldout")
_HELD_OUT_PAIRS = frozenset({("held", "out"), ("hold", "out")})


def _tokens(name: str) -> list[str]:
    spaced = re.sub(r"([a-z])([A-Z])", r"\1 \2", name)
    spaced = re.sub(r"(?<=[A-Za-z])(?=\d)|(?<=\d)(?=[A-Za-z])", " ", spaced)
    return [t for t in re.split(r"[^a-z0-9]+", spaced.lower()) if t]


def split_class(name: str) -> SplitClass:
    """What a split's NAME says it holds: ``train``, ``held_out`` or ``ambiguous``.

    Tokens are the name lowercased and cut at every non-alphanumeric character, case change and
    letter/digit boundary. Any held-out signal wins (``private_test``, ``public-test``,
    ``held_out``, ``hold-out``, ``devSet``, ``train_test``); ``train`` needs a training token and
    nothing but version numbers besides (``train``, ``train_v2``); anything else is ambiguous.
    """
    tokens = _tokens(name)
    if not tokens:
        return "ambiguous"
    if any(
        t in _HELD_OUT_TOKENS or t.startswith("dev") or any(p in t for p in _HELD_OUT_PARTS)
        for t in tokens
    ) or any(pair in _HELD_OUT_PAIRS for pair in itertools.pairwise(tokens)):
        return "held_out"
    if any(t in _TRAIN_TOKENS for t in tokens) and all(
        t in _TRAIN_TOKENS or _FILLER_TOKEN.match(t) for t in tokens
    ):
        return "train"
    return "ambiguous"


class RecipeError(ValidationError):
    """A recipe that cannot run as written; the message names the step and never row content."""


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)


class Source(_Frozen):
    """One pinned input: a Hub dataset repo at a full commit, one config and one split."""

    name: str = Field(pattern=_NAME)
    hf_repo_id: str = Field(pattern=_REPO)
    revision: str = Field(pattern=_SHA)
    config: str | None = Field(default=None, min_length=1, max_length=128)
    # One split name with at most a `[start:end]` row slice (`train`, `train[0:5000]`).
    split: str = Field(min_length=1, max_length=128)
    # Only these columns are read; unset, every column of the split.
    columns: tuple[Column, ...] | None = Field(default=None, max_length=512)
    # The project dataset version this pin came from, when it came from one (a record id the
    # control plane checks belongs to the project).
    dataset_version: str | None = Field(default=None, pattern=r"^[0-9a-f]{32}$")

    @model_validator(mode="after")
    def _split_is_one_split(self) -> Source:
        if parse_single_split(self.split) is None:
            msg = (
                f"source {self.name!r}: split {self.split!r} must be one split name with at most a "
                "[start:end] row slice (no '+', percent, negative index or rounding)"
            )
            raise ValueError(msg)
        return self

    @property
    def split_name(self) -> str:
        parsed = parse_single_split(self.split)
        return parsed.name if parsed is not None else self.split

    @property
    def split_class(self) -> SplitClass:
        return split_class(self.split_name)

    @property
    def holdout(self) -> bool:
        """Whether this split may hold held-out rows: anything not classified ``train``."""
        return self.split_class != "train"


class Condition(_Frozen):
    """``column <op> value``. ``in`` / ``not_in`` take a list; ``is_null`` / ``not_null`` none."""

    column: Column
    op: Literal[
        "eq", "ne", "lt", "le", "gt", "ge", "in", "not_in", "contains", "is_null", "not_null"
    ]
    value: Scalar | tuple[Scalar, ...] = Field(default=None)

    @field_validator("value", mode="before")
    @classmethod
    def _bounded_list(cls, value: object) -> object:
        if isinstance(value, Sized) and not isinstance(value, str) and len(value) > 1000:
            msg = "a condition lists at most 1,000 values"
            raise ValueError(msg)
        return value

    @model_validator(mode="after")
    def _value_fits_the_op(self) -> Condition:
        listed = isinstance(self.value, tuple)
        if self.op in ("in", "not_in") and not listed:
            msg = f"{self.op} takes a list of values"
            raise ValueError(msg)
        if self.op not in ("in", "not_in") and listed:
            msg = f"{self.op} takes one value"
            raise ValueError(msg)
        if self.op in ("is_null", "not_null") and self.value is not None:
            msg = f"{self.op} takes no value"
            raise ValueError(msg)
        if self.op in ("lt", "le", "gt", "ge") and not isinstance(self.value, int | float | str):
            msg = f"{self.op} compares with a number or a string"
            raise ValueError(msg)
        return self


class LoadStep(_Frozen):
    kind: Literal["load"] = "load"
    source: str = Field(pattern=_NAME)


class ConcatStep(_Frozen):
    kind: Literal["concat"] = "concat"
    source: str = Field(pattern=_NAME)


class JoinStep(_Frozen):
    kind: Literal["join"] = "join"
    source: str = Field(pattern=_NAME)
    on: tuple[Column, ...] = Field(min_length=1, max_length=8)
    how: Literal["inner", "left"] = "inner"
    # The right side's columns to bring in (besides the keys); unset, all of them.
    right_columns: tuple[Column, ...] | None = Field(default=None, max_length=512)
    suffix: str = Field(default="_right", pattern=r"^[A-Za-z0-9_]{1,16}$")


class FilterStep(_Frozen):
    kind: Literal["filter"] = "filter"
    match: Literal["all", "any"] = "all"
    conditions: tuple[Condition, ...] = Field(min_length=1, max_length=32)


class MapColumnsStep(_Frozen):
    """Derive, then select, then rename. ``templates`` fill ``{column}`` from the same row."""

    kind: Literal["map_columns"] = "map_columns"
    templates: dict[Column, Annotated[str, Field(max_length=20_000)]] = Field(
        default_factory=dict[str, str], max_length=64
    )
    constants: dict[Column, Scalar] = Field(default_factory=dict[str, Scalar], max_length=64)
    select: tuple[Column, ...] | None = Field(default=None, max_length=512)
    rename: dict[Column, Column] = Field(default_factory=dict[str, str], max_length=512)


class DedupeStep(_Frozen):
    kind: Literal["dedupe"] = "dedupe"
    keys: tuple[Column, ...] = Field(min_length=1, max_length=16)
    # Text comparison: exact, or lowercased with whitespace runs collapsed and the ends trimmed.
    normalize: Literal["exact", "casefold_whitespace"] = "exact"


class SplitStep(_Frozen):
    """Assign each row an output split: a seeded shuffle cut by ``ratios``, per stratum if named."""

    kind: Literal["split"] = "split"
    ratios: dict[Annotated[str, Field(pattern=_NAME)], float] = Field(min_length=1, max_length=8)
    seed: StrictInt = Field(ge=0, le=2**53 - 1)
    stratify_by: Column | None = None

    @model_validator(mode="after")
    def _ratios_are_a_partition(self) -> SplitStep:
        if any(not 0 < r <= 1 for r in self.ratios.values()):
            msg = "split ratios are each in (0, 1]"
            raise ValueError(msg)
        if not math.isclose(sum(self.ratios.values()), 1.0, abs_tol=1e-9):
            msg = "split ratios must sum to 1"
            raise ValueError(msg)
        return self


class SynthesizeStep(_Frozen):
    """Generate candidate rows from the table's rows with a pinned model.

    For each of the first ``max_seed_rows`` rows, ``n`` requests render ``prompt_template`` from
    the row; request ``j`` of seed row ``i`` uses seed ``seed + i * n + j``. The model's output
    must be a JSON object meeting ``output_schema`` (sent as guided JSON), and ``output_columns``
    maps its keys to output columns; ``keep_columns`` copies seed-row columns across. ``replace``
    makes the generated rows the table; ``append`` adds them after the seed rows.
    """

    kind: Literal["synthesize"] = "synthesize"
    model_id: str = Field(pattern=_REPO)
    model_revision: str = Field(pattern=_SHA)
    prompt_template: str = Field(min_length=1, max_length=20_000)
    output_schema: dict[str, Any]
    output_columns: dict[Column, Annotated[str, Field(min_length=1, max_length=128)]] = Field(
        min_length=1, max_length=64
    )
    keep_columns: tuple[Column, ...] = Field(default=(), max_length=64)
    n: StrictInt = Field(default=1, ge=1, le=16)
    seed: StrictInt = Field(ge=0, le=2**53 - 1)
    temperature: float = Field(default=0.7, ge=0, le=2)
    max_tokens: StrictInt = Field(default=512, ge=1, le=32_768)
    max_seed_rows: StrictInt = Field(default=100, ge=1, le=10_000)
    mode: Literal["append", "replace"] = "replace"

    @model_validator(mode="after")
    def _schema_is_an_object_with_the_mapped_keys(self) -> SynthesizeStep:
        schema = self.output_schema
        properties: Any = schema.get("properties")
        if schema.get("type") != "object" or not isinstance(properties, dict):
            msg = "output_schema is a JSON object schema with properties"
            raise ValueError(msg)
        missing = sorted(set(self.output_columns.values()) - set(properties))  # pyright: ignore[reportUnknownArgumentType]
        if missing:
            msg = f"output_columns names keys the schema does not declare: {missing}"
            raise ValueError(msg)
        if len(json.dumps(schema)) > 20_000:
            msg = "output_schema is over 20,000 characters"
            raise ValueError(msg)
        problem = schema_problem(schema)
        if problem is not None:
            msg = f"output_schema: {problem}"
            raise ValueError(msg)
        return self


type Step = Annotated[
    LoadStep
    | ConcatStep
    | JoinStep
    | FilterStep
    | MapColumnsStep
    | DedupeStep
    | SplitStep
    | SynthesizeStep,
    Field(discriminator="kind"),
]


class Recipe(BaseModel):
    """A dataset recipe: pinned sources and the ordered steps that make the output."""

    schema_: Literal["strata.dataset_recipe/v1"] = Field(default=RECIPE_SCHEMA, alias="schema")
    name: str = Field(pattern=_NAME)
    sources: tuple[Source, ...] = Field(min_length=1, max_length=16)
    steps: tuple[Step, ...] = Field(min_length=1, max_length=64)
    # The output split when the recipe has no `split` step.
    default_split: str = Field(default="train", pattern=_NAME)

    model_config = ConfigDict(
        frozen=True, extra="forbid", allow_inf_nan=False, populate_by_name=True
    )

    @model_validator(mode="after")
    def _steps_fit_together(self) -> Recipe:
        names = [s.name for s in self.sources]
        if len(set(names)) != len(names):
            msg = "source names must be unique"
            raise ValueError(msg)
        if self.steps[0].kind != "load":
            msg = "the first step loads a source"
            raise ValueError(msg)
        for index, step in enumerate(self.steps):
            source = getattr(step, "source", None)
            if source is not None and source not in names:
                msg = f"step {index + 1} names source {source!r}, which the recipe does not declare"
                raise ValueError(msg)
            if step.kind == "load" and index:
                msg = f"step {index + 1}: only the first step loads; use concat or join"
                raise ValueError(msg)
            if step.kind == "split" and index != len(self.steps) - 1:
                msg = "a split step must be the last step"
                raise ValueError(msg)
        self._held_out_stays_held_out()
        if len(json.dumps(self.canonical(), separators=(",", ":"))) > MAX_RECIPE_BYTES:
            msg = f"the recipe is over {MAX_RECIPE_BYTES // 1024} KiB"
            raise ValueError(msg)
        return self

    def _held_out_stays_held_out(self) -> None:
        used = set(self.used_sources)
        not_train = [s for s in self.sources if s.holdout and s.name in used]
        if not not_train:
            return
        named = ", ".join(
            f"{s.name} ({s.split_name}: {s.split_class.replace('_', '-')})" for s in not_train
        )
        if self.generates:
            msg = (
                f"a recipe that generates reads training splits only; {named} "
                "is not a training split"
            )
            raise ValueError(msg)
        outputs = list(self.output_splits)
        relabelled = [o for o in outputs if split_class(o) != "held_out"]
        if relabelled:
            msg = (
                f"this recipe reads {named}, so every output split must be named as held out "
                f"(such as test or validation); {relabelled} would relabel it"
            )
            raise ValueError(msg)

    @property
    def output_splits(self) -> tuple[str, ...]:
        last = self.steps[-1]
        if isinstance(last, SplitStep):
            return tuple(last.ratios)
        return (self.default_split,)

    @property
    def used_sources(self) -> tuple[str, ...]:
        """The sources a step reads, in first-use order (a declared, unread source is never read)."""
        seen: dict[str, None] = {}
        for step in self.steps:
            source = getattr(step, "source", None)
            if isinstance(source, str):
                seen.setdefault(source, None)
        return tuple(seen)

    @property
    def holdout_sources(self) -> tuple[str, ...]:
        """The read sources not classified ``train``: an output derived from them is held out."""
        used = set(self.used_sources)
        return tuple(s.name for s in self.sources if s.holdout and s.name in used)

    def source(self, name: str) -> Source:
        return next(s for s in self.sources if s.name == name)

    def canonical(self) -> dict[str, Any]:
        """The recipe as the JSON a revision hash covers (aliases, no defaults dropped)."""
        return self.model_dump(mode="json", by_alias=True)

    @property
    def generates(self) -> bool:
        return any(step.kind == "synthesize" for step in self.steps)


def recipe_sha256(recipe: Recipe) -> str:
    """The recipe's content hash: sorted keys, no whitespace. Two equal recipes hash equal."""
    return content_hash(recipe.canonical())


# ------------------------------------------------------------------------------ execution


@dataclass(slots=True)
class Row:
    id: str
    values: dict[str, Any]
    # The sources this row's values came from (a joined row has two), for the holdout rule.
    origins: frozenset[str]
    synthetic: bool = False
    seed_row_id: str | None = None
    split: str | None = None


@dataclass(slots=True)
class Table:
    rows: list[Row]
    columns: list[str]

    def records(self) -> list[dict[str, Any]]:
        """Every row as one output record: its columns, then the three engine columns."""
        return [
            {
                **{c: row.values.get(c) for c in self.columns},
                "_row_id": row.id,
                "_synthetic": row.synthetic,
                "_seed_row_id": row.seed_row_id,
            }
            for row in self.rows
        ]


@dataclass(frozen=True, slots=True)
class Exclusion:
    row_id: str
    step: int  # 1-based
    reason: str


@dataclass(frozen=True, slots=True)
class StepCount:
    step: int  # 1-based
    kind: str
    rows_in: int
    rows_out: int
    excluded: int
    generated: int = 0


@dataclass(frozen=True, slots=True)
class GenerationRequest:
    seed_row_id: str
    sample_index: int
    seed: int
    prompt: str


@dataclass(frozen=True, slots=True)
class GenerationResult:
    """One request's outcome: the model's text, or why there is none.

    ``error`` ``invalid_output`` is a reply the generator rejected against the schema; any other
    error is a failed request.
    """

    text: str | None
    error: str | None = None


class Generator(Protocol):
    """Runs one synthesize step's requests in order and returns one result per request."""

    def __call__(
        self, step: SynthesizeStep, requests: Sequence[GenerationRequest]
    ) -> Sequence[GenerationResult]: ...


@dataclass(slots=True)
class RecipeOutput:
    splits: dict[str, Table]
    exclusions: list[Exclusion] = field(default_factory=list[Exclusion])
    counts: list[StepCount] = field(default_factory=list[StepCount])

    @property
    def rows(self) -> int:
        return sum(len(t.rows) for t in self.splits.values())

    @property
    def synthetic_rows(self) -> int:
        return sum(1 for t in self.splits.values() for r in t.rows if r.synthetic)


def render_prompt(template: str, values: Mapping[str, Any]) -> str:
    """Fill each ``{column}`` from ``values``: strings as they are, everything else as JSON."""

    def _sub(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in values:
            msg = f"the template names column {name!r}, which the row does not have"
            raise RecipeError(msg)
        value = values[name]
        return (
            value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
        )

    return _PLACEHOLDER.sub(_sub, template)


def _size(value: Any) -> int:
    """An estimate of the bytes ``value`` holds: text and bytes by length, containers summed."""
    if isinstance(value, str | bytes | bytearray):
        return 8 + len(value)
    if isinstance(value, dict):
        mapping = cast("dict[Any, Any]", value)
        return 8 + sum(_size(k) + _size(v) for k, v in mapping.items())
    if isinstance(value, list | tuple):
        return 8 + sum(_size(v) for v in value)  # pyright: ignore[reportUnknownVariableType]
    return 8


@dataclass(slots=True)
class _Budget:
    """The rows and bytes one execution may hold, checked as rows are added."""

    row_cap: int
    byte_cap: int

    def check(self, rows: int, held: int, where: str) -> None:
        if rows > self.row_cap:
            msg = f"{where} holds more than {self.row_cap} rows"
            raise RecipeError(msg)
        if held > self.byte_cap:
            msg = f"{where} holds more than {self.byte_cap // 1024**2} MiB of values"
            raise RecipeError(msg)


def _held(table: Table) -> int:
    return sum(_size(row.values) for row in table.rows)


def _load(
    source: Source, read: Callable[[Source], Sequence[Mapping[str, Any]]], budget: _Budget
) -> Table:
    raw = read(source)
    where = f"source {source.name!r}"
    budget.check(len(raw), 0, where)
    columns = list(source.columns) if source.columns is not None else list(raw[0]) if raw else []
    clash = sorted(RESERVED_COLUMNS & set(columns))
    if clash:
        msg = f"source {source.name!r} has engine-reserved columns {clash}"
        raise RecipeError(msg)
    rows: list[Row] = []
    held = 0
    for index, record in enumerate(raw):
        missing = [c for c in columns if c not in record]
        if missing:
            msg = f"source {source.name!r} row {index} lacks columns {missing}"
            raise RecipeError(msg)
        values = {c: record[c] for c in columns}
        held += _size(values)
        budget.check(len(rows) + 1, held, where)
        rows.append(
            Row(id=f"{source.name}:{index}", values=values, origins=frozenset({source.name}))
        )
    return Table(rows, columns)


def _need(table: Table, columns: Sequence[str], step: int) -> None:
    missing = [c for c in columns if c not in table.columns]
    if missing:
        msg = f"step {step} names columns the table does not have: {missing}"
        raise RecipeError(msg)


def _ordered(a: Any, b: Any) -> tuple[Any, Any] | None:
    """Two values comparable under ``<``: both numbers (not booleans) or both strings."""
    numeric = (int, float)
    if isinstance(a, bool) or isinstance(b, bool):
        return None
    if isinstance(a, numeric) and isinstance(b, numeric):
        return a, b
    if isinstance(a, str) and isinstance(b, str):
        return a, b
    return None


def _same(a: Any, b: Any) -> bool:
    """Equality where a boolean equals only a boolean (``True`` is not ``1``)."""
    return (type(a) is bool) == (type(b) is bool) and a == b


def _holds(condition: Condition, value: Any) -> bool:
    """SQL-like: a null value meets only ``is_null`` (and ``ne`` / ``not_in`` are false on null)."""
    op, target = condition.op, condition.value
    if op == "is_null":
        return value is None
    if op == "not_null":
        return value is not None
    if value is None:
        return False
    if op == "eq":
        return _same(value, target)
    if op == "ne":
        return not _same(value, target)
    if op in ("in", "not_in"):
        listed = target if isinstance(target, tuple) else ()
        found = any(_same(value, t) for t in listed)
        return found if op == "in" else not found
    if op == "contains":
        if isinstance(value, str):
            return isinstance(target, str) and target in value
        if isinstance(value, list):
            return target in value
        return False
    pair = _ordered(value, target)
    if pair is None:
        return False
    left, right = pair
    return {"lt": left < right, "le": left <= right, "gt": left > right, "ge": left >= right}[op]


def _key(values: Mapping[str, Any], columns: Sequence[str], normalize: str = "exact") -> str:
    parts: list[Any] = []
    for c in columns:
        v = values.get(c)
        if normalize == "casefold_whitespace" and isinstance(v, str):
            v = " ".join(v.split()).casefold()
        parts.append(v)
    return json.dumps(parts, sort_keys=True, ensure_ascii=False, default=str)


def _join(
    left: Table, right: Table, step: JoinStep, index: int, budget: _Budget
) -> tuple[Table, list[Exclusion]]:
    _need(left, step.on, index)
    _need(right, step.on, index)
    extra = [c for c in (step.right_columns or right.columns) if c not in step.on]
    _need(right, extra, index)
    names = {c: (c + step.suffix if c in left.columns else c) for c in extra}
    clash = sorted(set(names.values()) & set(left.columns))
    if clash:
        msg = f"step {index}: joined columns {clash} clash even with the suffix"
        raise RecipeError(msg)
    by_key: dict[str, list[Row]] = {}
    for row in right.rows:
        if any(row.values.get(c) is None for c in step.on):
            continue  # a null key matches nothing, as in SQL
        by_key.setdefault(_key(row.values, step.on), []).append(row)
    rows: list[Row] = []
    excluded: list[Exclusion] = []
    # The join is checked as it grows: a many-to-many key can multiply rows far past the cap, and
    # the check must stop it before it is held, not after.
    held = _held(left) + _held(right)
    where = f"step {index}"
    for row in left.rows:
        matches = (
            by_key.get(_key(row.values, step.on), [])
            if all(row.values.get(c) is not None for c in step.on)
            else []
        )
        if not matches:
            if step.how == "left":
                rows.append(
                    Row(row.id, {**row.values, **dict.fromkeys(names.values())}, row.origins)
                )
                held += _size(rows[-1].values)
                budget.check(len(rows), held, where)
            else:
                excluded.append(Exclusion(row.id, index, "no_join_match"))
            continue
        for match in matches:
            values = {**row.values, **{names[c]: match.values.get(c) for c in extra}}
            rows.append(Row(f"{row.id}+{match.id}", values, row.origins | match.origins))
            held += _size(values)
            budget.check(len(rows), held, where)
    return Table(rows, [*left.columns, *names.values()]), excluded


def _map(table: Table, step: MapColumnsStep, index: int) -> Table:
    derived = [*step.templates, *step.constants]
    if len(set(derived)) != len(derived):
        msg = f"step {index}: a column is both templated and constant"
        raise RecipeError(msg)
    for template in step.templates.values():
        _need(table, _PLACEHOLDER.findall(template), index)
    columns = [*table.columns, *[c for c in derived if c not in table.columns]]
    if step.select is not None:
        _need(Table([], columns), step.select, index)
        columns = list(step.select)
    _need(Table([], columns), list(step.rename), index)
    renamed = [step.rename.get(c, c) for c in columns]
    if len(set(renamed)) != len(renamed):
        msg = f"step {index}: two columns would share a name after renaming"
        raise RecipeError(msg)
    if RESERVED_COLUMNS & set(renamed):
        msg = f"step {index}: {sorted(RESERVED_COLUMNS & set(renamed))} are engine-reserved"
        raise RecipeError(msg)
    for row in table.rows:
        values = dict(row.values)
        for name, template in step.templates.items():
            values[name] = render_prompt(template, row.values)
        values.update(step.constants)
        row.values = {step.rename.get(c, c): values.get(c) for c in columns}
    table.columns = renamed
    return table


def _split(table: Table, step: SplitStep, index: int) -> dict[str, Table]:
    names = list(step.ratios)
    groups: dict[str, list[Row]] = {}
    if step.stratify_by is not None:
        _need(table, [step.stratify_by], index)
        for row in table.rows:
            groups.setdefault(_key(row.values, [step.stratify_by]), []).append(row)
    else:
        groups[""] = list(table.rows)
    out = {name: Table([], list(table.columns)) for name in names}
    rng = random.Random(step.seed)  # noqa: S311 - a reproducible split, not a security context
    for key in sorted(groups):
        members = sorted(groups[key], key=lambda r: r.id)
        rng.shuffle(members)
        # Largest-remainder apportionment: counts sum to the group size exactly.
        exact = [step.ratios[n] * len(members) for n in names]
        counts = [math.floor(e) for e in exact]
        for i in sorted(range(len(names)), key=lambda i: (counts[i] - exact[i], i))[
            : len(members) - sum(counts)
        ]:
            counts[i] += 1
        start = 0
        for name, count in zip(names, counts, strict=True):
            for row in members[start : start + count]:
                row.split = name
                out[name].rows.append(row)
            start += count
    order = {r.id: i for i, r in enumerate(table.rows)}
    for t in out.values():
        t.rows.sort(key=lambda r: order[r.id])
    return out


def _parse(text: str, step: SynthesizeStep) -> dict[str, Any] | None:
    try:
        value: Any = json.loads(text)
    except ValueError:
        return None
    if not isinstance(value, dict):
        return None
    required: Any = step.output_schema.get("required") or []
    if any(key not in value for key in required):  # pyright: ignore[reportUnknownVariableType]
        return None
    return value  # pyright: ignore[reportUnknownVariableType]


def _synthesize(
    table: Table,
    step: SynthesizeStep,
    index: int,
    recipe: Recipe,
    generate: Generator | None,
) -> tuple[Table, list[Exclusion], int]:
    if generate is None:
        msg = f"step {index} generates, and this execution has no generator"
        raise RecipeError(msg)
    _need(table, [*_PLACEHOLDER.findall(step.prompt_template), *step.keep_columns], index)
    seeds = table.rows[: step.max_seed_rows]
    held = sorted({o for r in seeds for o in r.origins if recipe.source(o).holdout})
    if held:
        msg = (
            f"step {index} would generate from held-out data ({', '.join(held)}): generation "
            "reads training splits only"
        )
        raise RecipeError(msg)
    requests = [
        GenerationRequest(
            seed_row_id=row.id,
            sample_index=j,
            seed=step.seed + i * step.n + j,
            prompt=render_prompt(step.prompt_template, row.values),
        )
        for i, row in enumerate(seeds)
        for j in range(step.n)
    ]
    results = list(generate(step, requests))
    if len(results) != len(requests):
        msg = f"step {index}: the generator returned {len(results)} results for {len(requests)}"
        raise RecipeError(msg)
    by_id = {row.id: row for row in seeds}
    generated: list[Row] = []
    excluded: list[Exclusion] = []
    out_columns = [
        *step.keep_columns,
        *[c for c in step.output_columns if c not in step.keep_columns],
    ]
    for request, result in zip(requests, results, strict=True):
        row_id = f"gen:{request.seed_row_id}:{request.sample_index}"
        if result.text is None:
            # A generator that validated the output itself reports `invalid_output`.
            reason = "invalid_output" if result.error == "invalid_output" else "generation_failed"
            excluded.append(Exclusion(row_id, index, reason))
            continue
        parsed = _parse(result.text, step)
        if parsed is None:
            excluded.append(Exclusion(row_id, index, "invalid_output"))
            continue
        seed = by_id[request.seed_row_id]
        values = {c: seed.values.get(c) for c in step.keep_columns}
        values.update({c: parsed.get(key) for c, key in step.output_columns.items()})
        generated.append(
            Row(row_id, values, seed.origins, synthetic=True, seed_row_id=request.seed_row_id)
        )
    if step.mode == "replace":
        return Table(generated, out_columns), excluded, len(generated)
    columns = [*table.columns, *[c for c in out_columns if c not in table.columns]]
    return Table([*table.rows, *generated], columns), excluded, len(generated)


def execute(
    recipe: Recipe,
    read: Callable[[Source], Sequence[Mapping[str, Any]]],
    *,
    generate: Generator | None = None,
    row_cap: int = 1_000_000,
    byte_cap: int = DEFAULT_BYTE_CAP,
) -> RecipeOutput:
    """Run ``recipe`` over the rows ``read`` returns for each pinned source.

    ``read`` receives a :class:`Source` and returns its split's rows in order (the caller reads the
    pinned commit; this module does not). ``generate`` runs ``synthesize`` steps; a recipe with
    one and no generator is refused. A table past ``row_cap`` rows or ``byte_cap`` bytes of values
    is refused, never truncated; a source and a join are checked while they are built.
    """
    if not 1 <= row_cap <= MAX_ROW_CAP:
        msg = f"row_cap is from 1 to {MAX_ROW_CAP}"
        raise RecipeError(msg)
    if not 1 <= byte_cap <= MAX_BYTE_CAP:
        msg = f"byte_cap is from 1 to {MAX_BYTE_CAP}"
        raise RecipeError(msg)
    budget = _Budget(row_cap, byte_cap)
    output = RecipeOutput(splits={})
    table = Table([], [])
    splits: dict[str, Table] | None = None
    for position, step in enumerate(recipe.steps):
        index = position + 1
        before = len(table.rows)
        excluded: list[Exclusion] = []
        generated = 0
        if isinstance(step, LoadStep):
            table = _load(recipe.source(step.source), read, budget)
        elif isinstance(step, ConcatStep):
            other = _load(recipe.source(step.source), read, budget)
            table = Table(
                [*table.rows, *other.rows],
                [*table.columns, *[c for c in other.columns if c not in table.columns]],
            )
        elif isinstance(step, JoinStep):
            table, excluded = _join(
                table, _load(recipe.source(step.source), read, budget), step, index, budget
            )
        elif isinstance(step, FilterStep):
            _need(table, [c.column for c in step.conditions], index)
            combine = all if step.match == "all" else any
            kept: list[Row] = []
            for row in table.rows:
                if combine(_holds(c, row.values.get(c.column)) for c in step.conditions):
                    kept.append(row)
                else:
                    excluded.append(Exclusion(row.id, index, "filtered"))
            table = Table(kept, table.columns)
        elif isinstance(step, MapColumnsStep):
            table = _map(table, step, index)
        elif isinstance(step, DedupeStep):
            _need(table, step.keys, index)
            first: dict[str, str] = {}
            kept = []
            for row in table.rows:
                key = _key(row.values, step.keys, step.normalize)
                if key in first:
                    excluded.append(Exclusion(row.id, index, f"duplicate_of:{first[key]}"))
                else:
                    first[key] = row.id
                    kept.append(row)
            table = Table(kept, table.columns)
        elif isinstance(step, SplitStep):
            splits = _split(table, step, index)
        else:
            table, excluded, generated = _synthesize(table, step, index, recipe, generate)
        budget.check(len(table.rows), _held(table), f"step {index}")
        output.exclusions.extend(excluded)
        output.counts.append(
            StepCount(index, step.kind, before, len(table.rows), len(excluded), generated)
        )
    output.splits = splits if splits is not None else {recipe.default_split: table}
    for name, t in output.splits.items():
        for row in t.rows:
            row.split = name
    return output


# ------------------------------------------------------------------------------ review


def profile_table(table: Table, *, top: int = 5) -> list[dict[str, Any]]:
    """Per column: rows, nulls, distinct values, numeric min/mean/max, and the top values.

    Values are reported for review in the user's own session; the runner records only counts.
    """
    out: list[dict[str, Any]] = []
    for column in table.columns:
        values = [row.values.get(column) for row in table.rows]
        present = [v for v in values if v is not None]
        keys = [json.dumps(v, sort_keys=True, ensure_ascii=False, default=str) for v in present]
        frequency: dict[str, int] = {}
        for k in keys:
            frequency[k] = frequency.get(k, 0) + 1
        numbers = [v for v in present if isinstance(v, int | float) and not isinstance(v, bool)]
        ranked = sorted(frequency.items(), key=lambda item: (-item[1], item[0]))[:top]
        out.append(
            {
                "column": column,
                "rows": len(values),
                "nulls": len(values) - len(present),
                "distinct": len(frequency),
                "min": min(numbers) if numbers and len(numbers) == len(present) else None,
                "max": max(numbers) if numbers and len(numbers) == len(present) else None,
                "mean": (
                    math.fsum(numbers) / len(numbers)
                    if numbers and len(numbers) == len(present)
                    else None
                ),
                "top": [{"value": json.loads(k), "count": c} for k, c in ranked],
            }
        )
    return out


def diff_tables(before: Table, after: Table, *, sample: int = 20) -> dict[str, Any]:
    """Rows added, removed and changed between two tables, matched by row id."""
    old = {row.id: row for row in before.rows}
    new = {row.id: row for row in after.rows}
    added = [i for i in new if i not in old]
    removed = [i for i in old if i not in new]
    shared = [c for c in after.columns if c in before.columns]
    changed = [
        i
        for i in new
        if i in old and any(new[i].values.get(c) != old[i].values.get(c) for c in shared)
    ]
    return {
        "added": len(added),
        "removed": len(removed),
        "changed": len(changed),
        "columns_added": [c for c in after.columns if c not in before.columns],
        "columns_removed": [c for c in before.columns if c not in after.columns],
        "sample": {
            "added": added[:sample],
            "removed": removed[:sample],
            "changed": changed[:sample],
        },
    }
