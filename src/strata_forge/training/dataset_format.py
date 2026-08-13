"""Map an arbitrary dataset's columns onto the row shape a TRL trainer expects.

TRL infers what to do from a dataset's COLUMN NAMES: an SFT run over ``{"prompt", "completion"}``
trains differently from one over ``{"text"}``, and DPO wants ``{"chosen", "rejected"}``. Real
datasets almost never use those names, so something has to declare which of a dataset's columns
plays which role. That declaration is a :data:`DatasetFormat` plus a ``column_mapping`` of
``role -> column``, and this module is the one place that knows what each format requires.

Validation is the point. Without it a wrong mapping surfaces as a TRL ``KeyError`` on a remote box,
minutes into a job that has already downloaded a model — so :func:`validate_mapping` checks the
declaration against the split's real column list up front and raises a
:class:`DatasetFormatError` naming the role, the column and what is actually available.

Deliberately dependency-free: rows in, rows out, plain dicts. ``datasets`` is not imported here,
which keeps :mod:`strata_forge.training` importable without any extra and lets the caller decide
whether the result becomes a ``Dataset``, a generator, or a test fixture.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, cast

if TYPE_CHECKING:
    from collections.abc import Collection, Iterable

__all__ = [
    "FORMATS",
    "DatasetFormat",
    "DatasetFormatError",
    "FormatSpec",
    "build_training_rows",
    "pick_format",
    "sft_text_field",
    "validate_mapping",
]


type DatasetFormat = Literal[
    "text",
    "prompt_completion",
    "conversational",
    "preference",
    "unpaired_preference",
]


class DatasetFormatError(ValueError):
    """A dataset declaration that cannot produce trainable rows. The message names the fix."""


@dataclass(frozen=True)
class FormatSpec:
    """One trainable row shape: the roles it needs, and the roles it merely accepts.

    ``optional`` is not decoration. TRL's preference format has two legal spellings — an explicit
    prompt beside the two completions, or the prompt embedded in both of them — and datasets in the
    wild use each about equally. Making ``prompt`` optional for ``preference`` is what lets a
    ``{chosen, rejected}``-only dataset train without a second format.
    """

    name: DatasetFormat
    required: tuple[str, ...]
    optional: tuple[str, ...] = ()

    @property
    def roles(self) -> tuple[str, ...]:
        return (*self.required, *self.optional)


FORMATS: dict[DatasetFormat, FormatSpec] = {
    "text": FormatSpec("text", required=("text",)),
    "prompt_completion": FormatSpec("prompt_completion", required=("prompt", "completion")),
    "conversational": FormatSpec("conversational", required=("messages",)),
    "preference": FormatSpec("preference", required=("chosen", "rejected"), optional=("prompt",)),
    "unpaired_preference": FormatSpec(
        "unpaired_preference", required=("prompt", "completion", "label")
    ),
}

# The SFT trainer reads a single text column ONLY for the flat ``text`` format; for the others it
# derives the text itself from the role columns, and naming a field would make it ignore them.
_SFT_TEXT_FIELD: dict[DatasetFormat, str | None] = {
    "text": "text",
    "prompt_completion": None,
    "conversational": None,
}


def pick_format(name: str) -> FormatSpec:
    """The :class:`FormatSpec` for ``name``, or a :class:`DatasetFormatError` listing the valid ones."""
    for spec in FORMATS.values():
        if spec.name == name:
            return spec
    known = ", ".join(sorted(FORMATS))
    msg = f"unknown dataset format {name!r}; expected one of: {known}"
    raise DatasetFormatError(msg)


def sft_text_field(fmt: str) -> str | None:
    """The ``dataset_text_field`` an SFT run wants for ``fmt`` — ``None`` when TRL should infer."""
    spec = pick_format(fmt)
    if spec.name not in _SFT_TEXT_FIELD:
        msg = f"dataset format {spec.name!r} cannot be used for supervised fine-tuning"
        raise DatasetFormatError(msg)
    return _SFT_TEXT_FIELD[spec.name]


def validate_mapping(
    fmt: str, column_mapping: Mapping[str, str], columns: Collection[str]
) -> FormatSpec:
    """Check ``column_mapping`` declares every role ``fmt`` requires and that each names a real column.

    Run this BEFORE anything expensive: it is the difference between a form telling the user which
    column is missing and a trainer crashing on a rented GPU.
    """
    spec = pick_format(fmt)
    available = sorted(columns)

    missing = [role for role in spec.required if not (column_mapping.get(role) or "").strip()]
    if missing:
        msg = (
            f"dataset format {spec.name!r} needs a column for: {', '.join(missing)}. "
            f"Columns in this split: {', '.join(available) or '(none)'}"
        )
        raise DatasetFormatError(msg)

    for role in spec.roles:
        column = (column_mapping.get(role) or "").strip()
        if column and column not in columns:
            msg = (
                f"column {column!r} (mapped to {role!r}) is not in this split. "
                f"Columns in this split: {', '.join(available) or '(none)'}"
            )
            raise DatasetFormatError(msg)

    unknown = sorted(set(column_mapping) - set(spec.roles))
    if unknown:
        msg = (
            f"dataset format {spec.name!r} has no role(s) {', '.join(unknown)}; "
            f"its roles are: {', '.join(spec.roles)}"
        )
        raise DatasetFormatError(msg)
    return spec


def _coerce(role: str, value: Any) -> Any:
    """Put one cell into the type its role is trained as.

    Only ``label`` needs it, and it needs it badly: KTO reads a boolean, and datasets spell the same
    fact as ``1``/``0``, ``"true"``/``"false"`` or ``"yes"``/``"no"``. Left uncoerced, a column of
    non-empty strings is every-row-true, which trains a model on the premise that nothing is bad —
    a silently wrong run rather than a failed one.
    """
    if role != "label":
        return value
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"true", "yes", "y", "1"}:
            return True
        if text in {"false", "no", "n", "0"}:
            return False
        msg = f"label value {value!r} is not a yes/no answer; KTO needs a boolean-ish label column"
        raise DatasetFormatError(msg)
    msg = f"label value of type {type(value).__name__} is not a boolean-ish label"
    raise DatasetFormatError(msg)


def build_training_rows(
    rows: Iterable[Mapping[str, Any]], fmt: str, column_mapping: Mapping[str, str]
) -> list[dict[str, Any]]:
    """Project ``rows`` onto ``fmt``'s roles, dropping every column the trainer did not ask for.

    Dropping is deliberate. TRL decides what kind of run it is doing by looking at the columns it
    was handed, so a leftover ``id`` or ``source`` column is not inert — it can change the inferred
    format. The output carries the roles and nothing else.
    """
    spec = pick_format(fmt)
    mapped = [
        (role, column)
        for role in spec.roles
        if (column := (column_mapping.get(role) or "").strip())
    ]
    out: list[dict[str, Any]] = []
    for i, row in enumerate(rows):
        built: dict[str, Any] = {}
        for role, column in mapped:
            if column not in row:
                msg = f"row {i} has no column {column!r} (mapped to {role!r})"
                raise DatasetFormatError(msg)
            built[role] = _coerce(role, row[column])
        out.append(built)
    if not out:
        msg = f"the split produced no rows for format {spec.name!r}"
        raise DatasetFormatError(msg)
    _check_shape(spec, out[0])
    return out


def _check_shape(spec: FormatSpec, first: Mapping[str, Any]) -> None:
    """Reject a mapping that is well-named but wrongly typed, using the first row as the witness.

    A ``conversational`` run pointed at a plain string column is the common mistake, and TRL's own
    failure for it is an obscure error deep in a collator.
    """
    if spec.name != "conversational":
        return
    messages: Any = first.get("messages")
    if not isinstance(messages, Sequence) or isinstance(messages, str | bytes):
        msg = (
            "the column mapped to 'messages' must hold a list of {role, content} turns; "
            f"it holds {type(messages).__name__}. Use the 'prompt_completion' or 'text' format "
            "for a plain-string column."
        )
        raise DatasetFormatError(msg)
    turns = cast("Sequence[object]", messages)
    if turns and not all(_is_turn(t) for t in turns):
        msg = "each turn in the 'messages' column must be an object with 'role' and 'content' keys"
        raise DatasetFormatError(msg)


def _is_turn(value: object) -> bool:
    return isinstance(value, Mapping) and "role" in value and "content" in value
