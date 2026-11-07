"""Pure versioned population metadata; a declared selection is never execution evidence."""

from __future__ import annotations

import hashlib
import json
from contextlib import suppress
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from strata_forge.core.errors import ValidationError

if TYPE_CHECKING:
    from collections.abc import Sequence

MAX_POPULATION = 1_000_000
MAX_TIERS = 64
MAX_TIER_BYTES = 128


class PopulationError(ValidationError):
    """A fixed, content-free refusal for the operational population profile."""


class PopulationData(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        strict=True,
        hide_input_in_errors=True,
        revalidate_instances="always",
    )


class PopulationColumns(PopulationData):
    """Explicit source columns, with null declaring that no suite filter was supplied."""

    tier_column: str = Field(min_length=1, max_length=255)
    within_limits_column: str | None = Field(min_length=1, max_length=255)


class EvaluationPopulation(PopulationColumns):
    evaluation_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,47}$")


class PopulationMember(PopulationData):
    index: int = Field(ge=0, lt=MAX_POPULATION)
    item_id: str = Field(min_length=1, max_length=1024)
    tier: str = Field(min_length=1)
    within_limits: bool


class PopulationOutcome(PopulationData):
    """One private item outcome; model_sent is null when execution origin is unknown."""

    index: int = Field(ge=0, lt=MAX_POPULATION)
    item_id: str = Field(min_length=1, max_length=1024)
    tier: str = Field(min_length=1)
    outcome: Literal["scored", "invalid", "overflow", "suite_excluded"]
    metric_eligible: bool
    token_count: int | None = Field(ge=0)
    model_limit: int | None = Field(ge=1)
    model_sent: bool | None
    error_code: str | None = Field(min_length=1, max_length=80, pattern=r"^[a-z][a-z0-9_]*$")
    refusal_stage: Literal["render", "input", "prediction", "missing"] | None

    @model_validator(mode="after")
    def _partition(self) -> PopulationOutcome:
        if self.metric_eligible != (self.outcome in {"scored", "invalid"}):
            raise ValueError("population_outcome_invalid")
        if self.outcome == "overflow":
            if (
                self.token_count is None
                or self.model_limit is None
                or self.token_count <= self.model_limit
                or self.model_sent is not False
                or self.error_code != "input_too_long"
            ):
                raise ValueError("population_outcome_invalid")
        elif self.outcome == "suite_excluded":
            if self.token_count is not None or self.model_sent is not False:
                raise ValueError("population_outcome_invalid")
        elif self.outcome == "scored" and (
            self.error_code is not None or self.refusal_stage is not None
        ):
            raise ValueError("population_outcome_invalid")
        if (
            self.outcome != "overflow"
            and self.token_count is not None
            and self.model_limit is not None
            and self.token_count > self.model_limit
        ):
            raise ValueError("population_outcome_invalid")
        return self


class PopulationCounts(PopulationData):
    rows_in: int = Field(ge=0, le=MAX_POPULATION)
    rows_suite_excluded: int = Field(ge=0, le=MAX_POPULATION)
    rows_overflow_dropped: int = Field(ge=0, le=MAX_POPULATION)
    rows_scored_success: int = Field(ge=0, le=MAX_POPULATION)
    rows_failed_or_invalid: int = Field(ge=0, le=MAX_POPULATION)
    rows_metric_population: int = Field(ge=0, le=MAX_POPULATION)
    rows_model_sent: int | None = Field(ge=0, le=MAX_POPULATION)
    invalid_sent: int = Field(ge=0, le=MAX_POPULATION)
    invalid_not_sent: int = Field(ge=0, le=MAX_POPULATION)
    invalid_origin_unknown: int = Field(ge=0, le=MAX_POPULATION)

    @model_validator(mode="after")
    def _identities(self) -> PopulationCounts:
        if (
            self.rows_in
            != self.rows_suite_excluded
            + self.rows_overflow_dropped
            + self.rows_scored_success
            + self.rows_failed_or_invalid
            or self.rows_metric_population != self.rows_scored_success + self.rows_failed_or_invalid
            or self.rows_failed_or_invalid
            != self.invalid_sent + self.invalid_not_sent + self.invalid_origin_unknown
            or (
                self.rows_model_sent is not None
                and not self.invalid_sent
                <= self.rows_model_sent
                <= self.rows_in - self.rows_suite_excluded - self.rows_overflow_dropped
            )
        ):
            raise ValueError("population_counts_invalid")
        return self


def _utf8(value: str) -> bytes:
    encoded: bytes | None = None
    with suppress(UnicodeEncodeError):
        encoded = value.encode("utf-8")
    if encoded is None:
        raise PopulationError("population_metadata_invalid")
    return encoded


def population_members(
    rows: Sequence[dict[str, object]], *, id_column: str, columns: PopulationColumns
) -> tuple[PopulationMember, ...]:
    """Validate every selected membership before rendering or tokenization, including exclusions."""
    if not rows:
        raise PopulationError("population_empty")
    if len(rows) > MAX_POPULATION:
        raise PopulationError("population_limit_exceeded")
    members: list[PopulationMember] = []
    ids: set[str] = set()
    tiers: set[str] = set()
    for index, row in enumerate(rows):
        item_id, tier = row.get(id_column), row.get(columns.tier_column)
        eligible = row.get(columns.within_limits_column) if columns.within_limits_column else True
        if (
            type(item_id) is not str
            or not 1 <= len(item_id) <= 1024
            or type(tier) is not str
            or not tier
            or type(eligible) is not bool
        ):
            raise PopulationError("population_metadata_invalid")
        _utf8(item_id)
        if len(_utf8(tier)) > MAX_TIER_BYTES:
            raise PopulationError("population_metadata_invalid")
        if item_id in ids:
            raise PopulationError("population_id_duplicate")
        ids.add(item_id)
        tiers.add(tier)
        if len(tiers) > MAX_TIERS:
            raise PopulationError("population_tier_limit_exceeded")
        members.append(
            PopulationMember(index=index, item_id=item_id, tier=tier, within_limits=eligible)
        )
    return tuple(members)


def population_counts(outcomes: Sequence[PopulationOutcome]) -> PopulationCounts:
    """Keep invalid outputs distinct from successful scoring while retaining their denominator."""
    invalid = [row for row in outcomes if row.outcome == "invalid"]
    return PopulationCounts(
        rows_in=len(outcomes),
        rows_suite_excluded=sum(row.outcome == "suite_excluded" for row in outcomes),
        rows_overflow_dropped=sum(row.outcome == "overflow" for row in outcomes),
        rows_scored_success=sum(row.outcome == "scored" for row in outcomes),
        rows_failed_or_invalid=len(invalid),
        rows_metric_population=sum(row.metric_eligible for row in outcomes),
        rows_model_sent=None
        if any(row.model_sent is None for row in outcomes)
        else sum(row.model_sent is True for row in outcomes),
        invalid_sent=sum(row.model_sent is True for row in invalid),
        invalid_not_sent=sum(row.model_sent is False for row in invalid),
        invalid_origin_unknown=sum(row.model_sent is None for row in invalid),
    )


def canonical_digest(value: object) -> str:
    """Hash explicit JSON structures, never ambiguous concatenated ID strings."""
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )
    return hashlib.sha256(_utf8(encoded)).hexdigest()


def selected_population_digest(members: Sequence[PopulationMember]) -> str:
    return canonical_digest([row.model_dump(mode="json") for row in members])
