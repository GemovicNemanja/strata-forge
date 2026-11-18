"""The LLM grader's rubric: what the judge is asked, the JSON it must answer in, and how it is read.

A rubric is data. It names the criteria a judge scores, each on a declared scale (a pass/fail
verdict, or an integer score between declared bounds), and one prompt template the run renders
per graded case. The template names the system's output as ``{output}`` and the rendered criteria
list as ``{criteria}``; every other ``{name}`` is a column of the gold split, mapped like an
inference run's ``column_mapping``. Rendering is the inference runner's own non-executing
substitution, so a case's text (an output may contain braces) is never re-read as a placeholder.

The judge answers under a guided-JSON schema built from the rubric (:func:`rubric_schema`): a
``rationale`` first, so the judge states its reasons before it scores, then one score per
criterion and nothing else. A reply that is not valid against that schema, after the inference
runner's one repair turn, is UNSCORED with a reason, never a fail and never a pass.

Everything here is pure; the run around it is :mod:`strata_forge.pipelines.grade_runner`.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

__all__ = [
    "CRITERIA_PLACEHOLDER",
    "GRADE_SCHEMA_VERSION",
    "MAX_RATIONALE_CHARS",
    "OUTPUT_PLACEHOLDER",
    "Criterion",
    "Grade",
    "Rubric",
    "parse_grade",
    "render_criteria",
    "rubric_digest",
    "rubric_schema",
]

#: Names the rubric version of the judge's answer shape, recorded with every grade.
GRADE_SCHEMA_VERSION = "strata.grade/v1"
#: The two placeholders the run fills itself; neither may be mapped to a gold column.
OUTPUT_PLACEHOLDER = "output"
CRITERIA_PLACEHOLDER = "criteria"
_CRITERION_ID_RE = re.compile(r"[a-z][a-z0-9_]{0,31}")
#: Names a criterion id may not take, because the ``end`` event's headline keys are built from
#: criterion ids (``<id>``, ``<id>_lower``, ``delta_<id>__<baseline>``, ``agreement_<id>``) beside
#: fixed ones: a criterion named ``coverage`` would overwrite the coverage figure, one named
#: ``tone_lower`` another's bound, and ``a__b`` would make ``delta_a__b__c`` ambiguous.
HEADLINE_NAMES = ("cases", "graded", "coverage")
HEADLINE_PREFIXES = ("delta_", "agreement_")
HEADLINE_SUFFIXES = ("_lower", "_upper", "_unresolved")
_PLACEHOLDER_RE = re.compile(r"\{([A-Za-z0-9_]+)\}")
#: A rubric's template is a prompt, not a corpus; the whole spec rides to the machine in one blob.
_MAX_TEMPLATE_CHARS = 32 * 1024
_MAX_CRITERIA = 8
#: Integer scales stay small: a judge's 1-100 score is not a measurement this report can defend.
_MAX_SCALE = 10
#: The longest rationale a rubric may allow (and the cap every stored rationale is cut to).
MAX_RATIONALE_CHARS = 2000


class Criterion(BaseModel):
    """One thing the judge scores: a pass/fail verdict, or an integer in ``[min, max]``.

    An ordinal criterion may declare ``pass_at``: a score at or above it counts as a pass in the
    pass-rate read beside the mean score. ``gold_column`` names a column of each graded file
    holding a human's score for that output on this criterion; when present, the report reads
    the judge's agreement with it.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    description: str = Field(min_length=1, max_length=2000)
    scale: Literal["pass_fail", "ordinal"]
    min: int | None = None
    max: int | None = None
    pass_at: int | None = None
    gold_column: str | None = Field(default=None, min_length=1, max_length=128)

    @field_validator("id")
    @classmethod
    def _identifier(cls, value: str) -> str:
        if not _CRITERION_ID_RE.fullmatch(value):
            msg = "a criterion id is lowercase letters, digits and '_', starting with a letter"
            raise ValueError(msg)
        if (
            value in HEADLINE_NAMES
            or value.startswith(HEADLINE_PREFIXES)
            or value.endswith(HEADLINE_SUFFIXES)
            or "__" in value
        ):
            msg = (
                f"criterion id {value!r} is reserved for the run's headline figures: not "
                f"{', '.join(HEADLINE_NAMES)}, no {' or '.join(HEADLINE_PREFIXES)} prefix, no "
                f"{', '.join(HEADLINE_SUFFIXES)} suffix and no '__'"
            )
            raise ValueError(msg)
        return value

    @model_validator(mode="after")
    def _scale(self) -> Criterion:
        if self.scale == "pass_fail":
            if (self.min, self.max, self.pass_at) != (None, None, None):
                msg = f"criterion {self.id}: a pass_fail scale takes no min, max or pass_at"
                raise ValueError(msg)
            return self
        if self.min is None or self.max is None:
            msg = f"criterion {self.id}: an ordinal scale needs min and max"
            raise ValueError(msg)
        if not 0 <= self.min < self.max <= _MAX_SCALE:
            msg = f"criterion {self.id}: an ordinal scale needs 0 <= min < max <= {_MAX_SCALE}"
            raise ValueError(msg)
        if self.pass_at is not None and not self.min < self.pass_at <= self.max:
            msg = f"criterion {self.id}: pass_at must lie in (min, max]"
            raise ValueError(msg)
        return self

    def values(self) -> list[int]:
        """Every score the scale admits, in order: ``[0, 1]`` (fail, pass) for pass/fail."""
        if self.scale == "pass_fail":
            return [0, 1]
        return list(range(self.min or 0, (self.max or 0) + 1))

    def passes(self, score: int) -> bool | None:
        """Whether ``score`` is a pass: the verdict itself, or ``score >= pass_at``; else None."""
        if self.scale == "pass_fail":
            return score == 1
        return None if self.pass_at is None else score >= self.pass_at


class Rubric(BaseModel):
    """The judge's instructions: a named prompt template and the criteria it scores."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1, max_length=64)
    template: str = Field(min_length=1, max_length=_MAX_TEMPLATE_CHARS)
    criteria: tuple[Criterion, ...] = Field(min_length=1, max_length=_MAX_CRITERIA)
    #: At most 2,000 characters: a rationale may quote the output it grades, so it is bounded.
    rationale_max_chars: int = Field(default=1500, ge=1, le=MAX_RATIONALE_CHARS)

    @model_validator(mode="after")
    def _well_formed(self) -> Rubric:
        ids = [criterion.id for criterion in self.criteria]
        if len(set(ids)) != len(ids):
            msg = "criterion ids must be unique"
            raise ValueError(msg)
        found = self.placeholders()
        missing = {OUTPUT_PLACEHOLDER, CRITERIA_PLACEHOLDER} - found
        if missing:
            msg = (
                f"the rubric template must name {sorted('{' + m + '}' for m in missing)}: the "
                "judge sees the output it grades and the criteria it grades it on"
            )
            raise ValueError(msg)
        return self

    def placeholders(self) -> set[str]:
        """Every ``{name}`` in the template, the two the run fills included."""
        return {match.group(1) for match in _PLACEHOLDER_RE.finditer(self.template)}

    def column_placeholders(self) -> set[str]:
        """The placeholders a ``column_mapping`` must map to gold columns."""
        return self.placeholders() - {OUTPUT_PLACEHOLDER, CRITERIA_PLACEHOLDER}

    def criterion(self, criterion_id: str) -> Criterion:
        return next(c for c in self.criteria if c.id == criterion_id)


def render_criteria(rubric: Rubric) -> str:
    """The ``{criteria}`` text: one numbered line per criterion, naming its key and its scale."""
    lines: list[str] = []
    for index, criterion in enumerate(rubric.criteria, start=1):
        if criterion.scale == "pass_fail":
            scale = "true if it passes, false if it fails"
        else:
            scale = f"an integer from {criterion.min} to {criterion.max}"
        lines.append(f"{index}. {criterion.id} ({scale}): {criterion.description}")
    return "\n".join(lines)


def rubric_schema(rubric: Rubric) -> dict[str, Any]:
    """The guided-JSON schema of the judge's answer: a rationale, then one score per criterion.

    ``rationale`` is listed first so a grammar-constrained judge writes its reasons before its
    scores. Every key is required and nothing else is admitted.
    """
    scores: dict[str, Any] = {}
    for criterion in rubric.criteria:
        if criterion.scale == "pass_fail":
            scores[criterion.id] = {"type": "boolean"}
        else:
            scores[criterion.id] = {
                "type": "integer",
                "minimum": criterion.min,
                "maximum": criterion.max,
            }
    return {
        "type": "object",
        "properties": {
            "rationale": {
                "type": "string",
                "minLength": 1,
                "maxLength": rubric.rationale_max_chars,
            },
            "scores": {
                "type": "object",
                "properties": scores,
                "required": [criterion.id for criterion in rubric.criteria],
                "additionalProperties": False,
            },
        },
        "required": ["rationale", "scores"],
        "additionalProperties": False,
    }


def rubric_digest(rubric: Rubric) -> str:
    """SHA-256 of the rubric as canonical JSON: what a report pins as the rubric it graded with."""
    canonical = json.dumps(rubric.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


class Grade(BaseModel):
    """One judged case: each criterion's score (pass/fail as 1/0) and the judge's rationale."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    scores: dict[str, int]
    rationale: str


def _unique_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """A JSON object's members, refusing a repeated key: which value a reader keeps is not
    something a score may depend on."""
    out: dict[str, object] = {}
    for key, value in pairs:
        if key in out:
            msg = f"repeated key {key!r}"
            raise ValueError(msg)
        out[key] = value
    return out


def parse_grade(text: str | None, rubric: Rubric) -> Grade | None:
    """The judge's reply read against the rubric, or None when it is not a complete answer.

    The run validated the reply against :func:`rubric_schema` already; this re-checks what the
    report relies on (exactly ``rationale`` and ``scores``, no repeated key, a rationale with text,
    every criterion present, in range, of the right type), so a grammar backend that enforced less
    than the schema can never put an out-of-scale score into an aggregate. The rationale is kept,
    cut to the rubric's limit; it may quote the graded output.
    """
    if not text:
        return None
    try:
        loaded: object = json.loads(text, object_pairs_hook=_unique_keys)
    except ValueError:
        return None
    if not isinstance(loaded, dict):
        return None
    reply = cast("dict[str, object]", loaded)
    if set(reply) != {"rationale", "scores"}:
        return None
    rationale = reply["rationale"]
    raw = reply["scores"]
    if not isinstance(rationale, str) or not rationale.strip() or not isinstance(raw, dict):
        return None
    given = cast("dict[str, object]", raw)
    if set(given) != {criterion.id for criterion in rubric.criteria}:
        return None
    scores: dict[str, int] = {}
    for criterion in rubric.criteria:
        value = given[criterion.id]
        if criterion.scale == "pass_fail":
            if not isinstance(value, bool):
                return None
            scores[criterion.id] = int(value)
        else:
            if isinstance(value, bool) or not isinstance(value, int):
                return None
            if not (criterion.min or 0) <= value <= (criterion.max or 0):
                return None
            scores[criterion.id] = value
    return Grade(scores=scores, rationale=rationale[: rubric.rationale_max_chars])
