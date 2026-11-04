"""Stable joins and bounded parsing without silently repairing malformed forecasts."""

from __future__ import annotations

import json
import math
from typing import TYPE_CHECKING, cast

from pydantic import BaseModel, ConfigDict

from strata_forge.metrics.classification import PROBABILITY_SUM_TOLERANCE
from strata_forge.pipelines._evaluation_access import EvaluationError
from strata_forge.pipelines.evaluation_spec import (
    EvaluationSet,
    JSONField,
    PredictionSet,
    ProbabilityColumns,
)

type Row = dict[str, object]

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


def slice_members(value: object) -> tuple[object, ...]:
    return tuple(cast("list[object]", value)) if isinstance(value, list) else (value,)


class EvaluationItem(BaseModel):
    """A retained gold item with canonical output and fixed content-free status."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    item_id: str
    gold: str
    pred: str | None
    prediction_status: str
    error_code: str | None = None
    probabilities: tuple[float, ...] | None = None
    probability_kind: str | None = None
    probability_exclusion_reason: str | None = None
    overflow: bool = False
    truncated: bool = False
    mass_outside: float | None = None
    source_prediction_ref: str | None = None
    source_prediction_id: str | None = None
    cluster_id: str | None = None
    slices: dict[str, object]

    @property
    def correct(self) -> bool:
        """Invalid/missing/refused predictions are wrong in the original population."""
        return self.prediction_status == "valid" and self.pred == self.gold


def canonical_label(
    value: object, labels: tuple[str, ...], aliases: Mapping[str, str], indices: Mapping[str, str]
) -> str | None:
    if type(value) is int:
        if not -(10**19) < value < 10**19:
            return None
        return indices.get(str(value))
    if not isinstance(value, str):
        return None
    label = aliases.get(value, value)
    return label if label in labels else None


def stable_id(value: object, code: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 1024:
        raise EvaluationError(code)
    return value


def gold_items(rows: Sequence[Row], case: EvaluationSet) -> tuple[EvaluationItem, ...]:
    if not rows:
        raise EvaluationError("gold_population_empty")
    items: list[EvaluationItem] = []
    for row in rows:
        identifier = stable_id(row.get(case.id_column), "gold_id_invalid")
        gold = canonical_label(row.get(case.gold_column), case.labels, {}, case.gold_index_map)
        if gold is None:
            raise EvaluationError("gold_label_invalid")
        if any(column not in row for column in case.slice_columns):
            raise EvaluationError("slice_column_missing")
        slices = {column: _slice_value(row[column]) for column in case.slice_columns}
        cluster = row.get(case.cluster_column) if case.cluster_column is not None else None
        if cluster is not None:
            cluster = stable_id(cluster, "cluster_id_invalid")
        items.append(
            EvaluationItem(
                item_id=identifier,
                gold=gold,
                pred=None,
                prediction_status="missing",
                error_code="prediction_missing",
                probability_exclusion_reason="missing_prediction",
                cluster_id=cluster,
                slices=slices,
            )
        )
    if len({item.item_id for item in items}) != len(items):
        raise EvaluationError("gold_id_duplicate")
    return tuple(items)


def _slice_value(value: object) -> object:
    def scalar(item: object) -> bool:
        return (
            item is None
            or type(item) is bool
            or (type(item) is int and abs(item) <= 2**53 - 1)
            or (isinstance(item, str) and len(item) <= 256)
        )

    if type(value) is list:
        members = cast("list[object]", value)
        if len(members) > 16 or not all(scalar(v) for v in members):
            raise EvaluationError("slice_value_invalid")
        return members
    if not scalar(value):
        raise EvaluationError("slice_value_invalid")
    return value


def _number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise EvaluationError("probability_invalid")
    try:
        number = float(value)
    except ValueError, OverflowError:
        raise EvaluationError("probability_invalid") from None
    if not math.isfinite(number) or not 0 <= number <= 1:
        raise EvaluationError("probability_invalid")
    return number


def _vector(
    row: Row, config: ProbabilityColumns, source: tuple[str, ...]
) -> tuple[float, ...] | None:
    if any(column not in row for column in config.probability_columns.values()):
        raise EvaluationError("probability_column_missing")
    raw = [row[config.probability_columns[label]] for label in source]
    if all(value is None for value in raw):
        return None
    vector = tuple(_number(value) for value in raw)
    if abs(math.fsum(vector) - 1) > PROBABILITY_SUM_TOLERANCE:
        raise EvaluationError("probability_invalid")
    return vector


def _json_label(row: Row, config: JSONField, labels: tuple[str, ...]) -> str | None:
    value = row.get(config.column)
    if config.input_kind == "json_text":
        if not isinstance(value, str):
            return None
        try:
            if len(value.encode("utf-8")) > 65_536:
                return None
            value = json.loads(value, parse_constant=_invalid_json_number)
        except ValueError, RecursionError, UnicodeError:
            return None
    for key in config.path:
        if not isinstance(value, dict) or key not in value:
            return None
        value = cast("Row", value)[key]
    return canonical_label(value, labels, config.label_aliases, config.index_map)


def _invalid_json_number(_: str) -> None:
    raise ValueError("nonfinite_json_number")


def parse_predictions(
    gold: tuple[EvaluationItem, ...],
    rows: Sequence[Row],
    case: PredictionSet,
    *,
    strata_scoring: bool = False,
) -> tuple[EvaluationItem, ...]:
    gold_ids = {item.item_id for item in gold}
    predictions: dict[str, Row] = {}
    for row in rows:
        identifier = stable_id(row.get(case.prediction_id_column), "prediction_id_missing")
        if identifier in predictions:
            raise EvaluationError("prediction_id_duplicate")
        if identifier not in gold_ids:
            raise EvaluationError("prediction_extra")
        predictions[identifier] = row
    result: list[EvaluationItem] = []
    for item in gold:
        row = predictions.get(item.item_id)
        if row is None:
            result.append(item)
            continue
        vector: tuple[float, ...] | None = None
        kind: str | None = None
        outside: float | None = None
        config = case.parse_config
        if isinstance(config, ProbabilityColumns):
            vector = _vector(row, config, case.source_labels)
            if vector is not None:
                vector = tuple(
                    math.fsum(
                        vector[i]
                        for i, source in enumerate(case.source_labels)
                        if (case.projection[source] if case.projection else source) == target
                    )
                    for target in case.labels
                )
                kind = config.probability_kind
            if config.mass_outside_column is not None:
                if config.mass_outside_column not in row:
                    raise EvaluationError("probability_column_missing")
                mass = row[config.mass_outside_column]
                outside = _number(mass) if mass is not None else None
            if config.prediction_rule == "argmax":
                pred = (
                    case.labels[max(range(len(vector)), key=vector.__getitem__)]
                    if vector is not None
                    else None
                )
            else:
                pred = canonical_label(
                    row.get(config.prediction_column or ""),
                    case.source_labels,
                    config.label_aliases,
                    config.index_map,
                )
                pred = (
                    case.projection[pred]
                    if case.projection is not None and pred is not None
                    else pred
                )
        else:
            pred = _json_label(row, config, case.source_labels)
            pred = (
                case.projection[pred] if case.projection is not None and pred is not None else pred
            )
        overflow = row.get("overflow", False) if strata_scoring else False
        truncated = row.get("truncated", False) if strata_scoring else False
        if type(overflow) is not bool or type(truncated) is not bool:
            raise EvaluationError("prediction_manifest_mismatch")
        refused = strata_scoring and row.get("error") is not None
        truncated_invalid = truncated and case.truncation_policy == "count_as_invalid"
        if refused or truncated_invalid:
            pred, vector = None, None
        result.append(
            item.model_copy(
                update={
                    "pred": pred,
                    "probabilities": vector,
                    "probability_kind": kind,
                    "prediction_status": "refused"
                    if refused or truncated_invalid
                    else "valid"
                    if pred is not None
                    else "invalid",
                    "error_code": "source_prediction_refused"
                    if refused
                    else "truncation_policy"
                    if truncated_invalid
                    else "prediction_label_invalid"
                    if pred is None
                    else None,
                    "probability_exclusion_reason": None
                    if vector is not None
                    else "truncation_policy"
                    if truncated_invalid
                    else "refused"
                    if refused
                    else "not_produced",
                    "overflow": overflow,
                    "truncated": truncated,
                    "mass_outside": outside,
                    "source_prediction_ref": case.predictions_ref,
                    "source_prediction_id": item.item_id,
                }
            )
        )
    return tuple(result)
