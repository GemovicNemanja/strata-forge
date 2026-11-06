"""Pure temperature application and a provenance component for calibration artifacts."""

from __future__ import annotations

import math
from collections.abc import Iterable
from importlib import import_module
from numbers import Real
from typing import TYPE_CHECKING, Literal, Protocol, cast

from pydantic import BaseModel, ConfigDict, Field

from strata_forge.core.errors import ValidationError
from strata_forge.metrics.classification import NLL_CLIP_EPSILON, LabelSpace

if TYPE_CHECKING:
    from collections.abc import Sequence
    from contextlib import AbstractContextManager

__all__ = ["CalibrationScores", "TemperatureCalibration", "temperature_scale"]


class CalibrationScores(BaseModel):
    """Optional observed point metrics; missing measurements are not zero."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True, hide_input_in_errors=True)
    brier: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    nll: float | None = Field(default=None, ge=0, allow_inf_nan=False)


class TemperatureCalibration(LabelSpace):
    """A draft temperature component, not the complete training calibration-file schema."""

    schema_version: Literal[1] = 1
    method: Literal["temperature"] = "temperature"
    temperature: float = Field(gt=0, allow_inf_nan=False)
    fit_ref: str = Field(min_length=1, max_length=256)
    config: str | None = Field(default=None, min_length=1, max_length=256)
    split: str = Field(min_length=1, max_length=256)
    n: int = Field(gt=0)
    probability_kind: Literal["class_probabilities", "restricted_softmax"]
    nll_clip_epsilon: float = Field(default=NLL_CLIP_EPSILON, gt=0, lt=1, allow_inf_nan=False)
    before: CalibrationScores | None = None
    after: CalibrationScores | None = None

    def apply(
        self, logits: Iterable[object], *, labels: Sequence[str]
    ) -> tuple[tuple[float, ...], ...]:
        """Apply the recorded T only when the caller confirms the exact input column order."""
        if tuple(labels) != self.labels:
            raise ValidationError("calibration label order does not match the input columns")
        output = temperature_scale(logits, temperature=self.temperature)
        if output and len(output[0]) != len(self.labels):
            raise ValidationError("calibration label count does not match the input columns")
        return output


class _Array(Protocol):
    def max(self, *, axis: int, keepdims: bool) -> _Array: ...
    def sum(self, *, axis: int, keepdims: bool) -> _Array: ...
    def __sub__(self, other: _Array) -> _Array: ...
    def __truediv__(self, other: _Array | float) -> _Array: ...
    def tolist(self) -> object: ...


class _Numpy(Protocol):
    def asarray(self, data: list[list[float]], *, dtype: type[float]) -> _Array: ...
    def exp(self, values: _Array) -> _Array: ...
    def errstate(self, *, over: str, under: str) -> AbstractContextManager[object]: ...


def _numpy() -> _Numpy:
    try:
        return cast("_Numpy", import_module("numpy"))
    except ImportError as exc:
        raise ImportError(
            "temperature_scale requires numpy: install 'strata-forge[scoring]'"
        ) from exc


def _logit_rows(logits: Iterable[object]) -> list[list[float]]:
    rows: list[list[float]] = []
    for row in logits:
        if isinstance(row, str | bytes) or not isinstance(row, Iterable):
            raise ValidationError("logits must be a rectangular matrix of finite real numbers")
        values: list[float] = []
        for value in cast("Iterable[object]", row):
            if isinstance(value, bool) or not isinstance(value, Real):
                raise ValidationError("logits must be finite real numbers")
            number = float(value)
            if not math.isfinite(number):
                raise ValidationError("logits must be finite real numbers")
            values.append(number)
        if len(values) < 2 or (rows and len(values) != len(rows[0])):
            raise ValidationError("logits must have a consistent class count of at least two")
        rows.append(values)
    return rows


def temperature_scale(
    logits: Iterable[object], *, temperature: float
) -> tuple[tuple[float, ...], ...]:
    """Return stable softmax(logits / T); fitting T and choosing a fit split are separate."""
    if isinstance(temperature, bool) or not math.isfinite(temperature) or temperature <= 0:
        raise ValidationError("temperature must be finite and positive")
    rows = _logit_rows(logits)
    if not rows:
        return ()
    np = _numpy()
    values = np.asarray(rows, dtype=float)
    # Finite extreme logits can subtract to -inf; their exp is correctly zero. The
    # largest centered logit stays zero, so every denominator has at least one unit term.
    with np.errstate(over="ignore", under="ignore"):
        weights = np.exp((values - values.max(axis=1, keepdims=True)) / temperature)
        raw = (weights / weights.sum(axis=1, keepdims=True)).tolist()
    if not isinstance(raw, list):
        raise ValidationError("temperature scaling returned an invalid matrix")
    result: list[tuple[float, ...]] = []
    for row in cast("list[object]", raw):
        if not isinstance(row, list):
            raise ValidationError("temperature scaling returned an invalid matrix")
        probabilities = tuple(float(value) for value in cast("list[float]", row))
        if len(probabilities) != len(rows[0]) or any(
            not math.isfinite(value) or not 0 <= value <= 1 for value in probabilities
        ):
            raise ValidationError("temperature scaling returned invalid probabilities")
        result.append(probabilities)
    if len(result) != len(rows):
        raise ValidationError("temperature scaling changed the input population")
    return tuple(result)
