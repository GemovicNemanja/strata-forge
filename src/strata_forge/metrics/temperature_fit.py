"""Bounded scalar-temperature math; source eligibility and execution remain separate."""

from __future__ import annotations

import math
from array import array
from contextlib import suppress
from dataclasses import dataclass
from typing import Literal, Self, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

from strata_forge.core.errors import ValidationError

__all__ = [
    "TemperatureFitBounds",
    "TemperatureFitError",
    "TemperatureFitSolution",
    "fit_temperature_logits",
]

MAX_FIT_ROWS = 100_000
MAX_FIT_CLASSES = 16
MAX_LOGIT_ABS = 1_000_000
MAX_FIT_ARRAY_BYTES = 256 * 1024**2
FIT_METHOD_VERSION = "scalar-logit-temperature-v1"


class TemperatureFitError(ValidationError):
    """A content-free numerical refusal, without source or authority semantics."""

    def __init__(self) -> None:
        super().__init__("temperature_fit_invalid")


class TemperatureFitBounds(BaseModel):
    """Explicit numerical bounds, not a scientific calibration policy."""

    model_config = ConfigDict(
        frozen=True,
        strict=True,
        extra="forbid",
        revalidate_instances="always",
        hide_input_in_errors=True,
    )
    minimum: float = Field(ge=0.001, le=1, allow_inf_nan=False)
    maximum: float = Field(ge=1, le=1000, allow_inf_nan=False)

    @model_validator(mode="after")
    def _ordered(self) -> Self:
        if self.minimum >= self.maximum:
            raise ValueError("temperature_fit_invalid")
        return self


class TemperatureFitSolution(BaseModel):
    """A numerical solution; floating stationarity never attests identification."""

    model_config = ConfigDict(
        frozen=True,
        strict=True,
        extra="forbid",
        revalidate_instances="always",
        hide_input_in_errors=True,
    )
    method_version: Literal["scalar-logit-temperature-v1"] = FIT_METHOD_VERSION
    temperature: float = Field(gt=0, allow_inf_nan=False)
    bounds: TemperatureFitBounds
    n: int = Field(ge=1, le=MAX_FIT_ROWS)
    k: int = Field(ge=2, le=MAX_FIT_CLASSES)
    iterations: int = Field(ge=0, le=64)
    derivative_passes: int = Field(ge=1, le=67)
    search_status: Literal[
        "flat_unidentified", "bounded_optimum", "numerical_stationary", "bracket_exhausted"
    ]
    identification: Literal["not_claimed"] = "not_claimed"
    selected_location: Literal["minimum", "maximum", "unit", "interior"]
    objective_before: float = Field(ge=0, allow_inf_nan=False)
    objective_after: float = Field(ge=0, allow_inf_nan=False)
    derivative: float = Field(allow_inf_nan=False)
    inverse_bracket_width: float = Field(ge=0, allow_inf_nan=False)
    zero_derivative_seen: bool
    underflowed_weights: int = Field(ge=0)

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if not self.bounds.minimum <= self.temperature <= self.bounds.maximum:
            raise ValueError("temperature_fit_invalid")
        location = (
            "minimum"
            if self.temperature == self.bounds.minimum
            else "maximum"
            if self.temperature == self.bounds.maximum
            else "unit"
            if self.temperature == 1
            else "interior"
        )
        if location != self.selected_location or self.objective_after > self.objective_before:
            raise ValueError("temperature_fit_invalid")
        if self.underflowed_weights > self.n * (self.k - 1):
            raise ValueError("temperature_fit_invalid")
        if self.search_status == "flat_unidentified" and (
            self.temperature != 1
            or self.derivative != 0
            or self.iterations != 0
            or self.underflowed_weights != 0
        ):
            raise ValueError("temperature_fit_invalid")
        return self


@dataclass(frozen=True)
class _Evaluation:
    beta: float
    objective: float
    derivative: float
    underflowed: int


class _FitState:
    """Owned deterministic state, stepped off-loop by fixed orchestration."""

    def __init__(
        self,
        logits: tuple[tuple[object, ...], ...],
        gold: tuple[int, ...],
        bounds: TemperatureFitBounds,
    ) -> None:
        if (
            type(logits) is not tuple
            or not 1 <= len(logits) <= MAX_FIT_ROWS
            or type(gold) is not tuple
            or len(gold) != len(logits)
            or type(bounds) is not TemperatureFitBounds
        ):
            raise TemperatureFitError()
        self.bounds = TemperatureFitBounds.model_validate(bounds.model_dump())
        self.n = len(logits)
        if type(logits[0]) is not tuple or not 2 <= len(logits[0]) <= MAX_FIT_CLASSES:
            raise TemperatureFitError()
        self.k = len(logits[0])
        self.values = array("d")
        if self.values.itemsize != 8 or self.n * (self.k + 1) * 8 > MAX_FIT_ARRAY_BYTES:
            raise TemperatureFitError()
        self.gold = gold
        self.flat = True
        for row, target in zip(logits, gold, strict=True):
            if (
                type(row) is not tuple
                or len(row) != self.k
                or type(target) is not int
                or not 0 <= target < self.k
            ):
                raise TemperatureFitError()
            if any(
                type(value) not in (int, float)
                or not -MAX_LOGIT_ABS <= cast("int | float", value) <= MAX_LOGIT_ABS
                for value in row
            ):
                raise TemperatureFitError()
            numeric = tuple(float(cast("int | float", value)) for value in row)
            if any(not math.isfinite(value) for value in numeric):
                raise TemperatureFitError()
            maximum = max(numeric)
            self.flat = self.flat and all(value == numeric[0] for value in numeric)
            self.values.extend(value - maximum for value in numeric)
        self.low: float = 1 / self.bounds.maximum
        self.high: float = 1 / self.bounds.minimum
        self.initial = (self.low, self.high, 1.0)
        self.evaluations: list[_Evaluation] = []
        self.iterations = 0
        self.done = False
        self.zero_seen = False

    def _evaluate(self, beta: float) -> _Evaluation:
        losses: list[float] = []
        gradients: list[float] = []
        underflowed = 0
        for index, target in enumerate(self.gold):
            offset = index * self.k
            row = self.values[offset : offset + self.k]
            weights = tuple(math.exp(beta * value) for value in row)
            denominator = math.fsum(weights)
            underflowed += sum(weight == 0 for weight in weights)
            losses.append(math.log(denominator) - beta * row[target])
            gradients.append(
                math.fsum(
                    (weight / denominator) * value
                    for weight, value in zip(weights, row, strict=True)
                )
                - row[target]
            )
        result = _Evaluation(
            beta, math.fsum(losses) / self.n, math.fsum(gradients) / self.n, underflowed
        )
        self.zero_seen = self.zero_seen or result.derivative == 0
        self.evaluations.append(result)
        return result

    def advance(self, passes: int = 8) -> None:
        if type(passes) is not int or not 1 <= passes <= 8:
            raise TemperatureFitError()
        for _ in range(passes):
            if self.done:
                return
            if len(self.evaluations) < 3:
                self._evaluate(self.initial[len(self.evaluations)])
                if len(self.evaluations) == 3:
                    left, right = self.evaluations[:2]
                    if self.flat or left.derivative >= 0 or right.derivative <= 0:
                        self.done = True
                continue
            result = self._evaluate(self.low + (self.high - self.low) / 2)
            self.iterations += 1
            if result.derivative > 0:
                self.high = result.beta
            elif result.derivative < 0:
                self.low = result.beta
            else:
                # Rounded or underflowed gradients can be zero away from an identified optimum.
                self.done = True
            if self.iterations == 64:
                self.done = True

    def _temperature(self, beta: float) -> float:
        return (
            self.bounds.maximum
            if beta == self.initial[0]
            else self.bounds.minimum
            if beta == self.initial[1]
            else 1.0
            if beta == 1
            else 1 / beta
        )

    def solution(self) -> TemperatureFitSolution:
        if not self.done:
            raise TemperatureFitError()
        chosen = min(
            self.evaluations,
            key=lambda e: (
                e.objective,
                abs(self._temperature(e.beta) - 1),
                self._temperature(e.beta),
            ),
        )
        temperature = self._temperature(chosen.beta)
        location = (
            "minimum"
            if temperature == self.bounds.minimum
            else "maximum"
            if temperature == self.bounds.maximum
            else "unit"
            if temperature == 1
            else "interior"
        )
        status: Literal[
            "flat_unidentified", "bounded_optimum", "numerical_stationary", "bracket_exhausted"
        ] = "bracket_exhausted"
        if self.flat:
            chosen, temperature = self.evaluations[2], 1.0
            location = (
                "minimum"
                if temperature == self.bounds.minimum
                else "maximum"
                if temperature == self.bounds.maximum
                else "unit"
            )
            status = "flat_unidentified"
        elif (location == "minimum" and chosen.derivative < 0) or (
            location == "maximum" and chosen.derivative > 0
        ):
            status = "bounded_optimum"
        elif self.zero_seen:
            status = "numerical_stationary"
        return TemperatureFitSolution(
            temperature=temperature,
            bounds=self.bounds,
            n=self.n,
            k=self.k,
            iterations=self.iterations,
            derivative_passes=len(self.evaluations),
            search_status=status,
            selected_location=location,
            objective_before=self.evaluations[2].objective,
            objective_after=chosen.objective,
            derivative=chosen.derivative,
            inverse_bracket_width=self.high - self.low,
            zero_derivative_seen=self.zero_seen,
            underflowed_weights=chosen.underflowed,
        )


def fit_temperature_logits(
    logits: tuple[tuple[object, ...], ...],
    gold_indexes: tuple[int, ...],
    *,
    bounds: TemperatureFitBounds,
) -> TemperatureFitSolution:
    """Fit pure bounded logit NLL; this does not register a split or authorize a calibration."""
    result = None
    with suppress(Exception):
        state = _FitState(logits, gold_indexes, bounds)
        while not state.done:
            state.advance()
        result = state.solution()
    if result is None:
        raise TemperatureFitError()
    return result
