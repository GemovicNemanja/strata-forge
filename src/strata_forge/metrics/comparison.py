"""Explicit-population intervals, paired tests and named-family p-value correction."""

from __future__ import annotations

import math
import re
import sys
from numbers import Integral, Real
from statistics import NormalDist
from typing import TYPE_CHECKING, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from strata_forge.core.errors import ValidationError

if TYPE_CHECKING:
    from collections.abc import Mapping

COMPARISON_VERSION = "classification-comparison-v1"
MAX_EXACT_DISCORDANT = 10_000_000
_MAX_SAFE_COUNT = 2**53 - 1
_MAX_HYPOTHESES = 10_000
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}\Z")

__all__ = [
    "COMPARISON_VERSION",
    "MAX_EXACT_DISCORDANT",
    "AdjustedPValue",
    "HolmCorrection",
    "McNemarResult",
    "ProportionInterval",
    "holm_correction",
    "mcnemar_exact",
    "wilson_interval",
]


class _Result(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True, hide_input_in_errors=True)
    comparison_version: Literal["classification-comparison-v1"] = COMPARISON_VERSION


class ProportionInterval(_Result):
    """A Wilson score interval for the caller's explicit binomial population."""

    method: Literal["wilson"] = "wilson"
    successes: int = Field(ge=0, le=_MAX_SAFE_COUNT)
    n: int = Field(ge=0, le=_MAX_SAFE_COUNT)
    confidence_level: float = Field(gt=0, lt=1, allow_inf_nan=False)
    estimate: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)
    low: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)
    high: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)
    reason: Literal["empty"] | None = None

    @model_validator(mode="after")
    def _consistent_population(self) -> ProportionInterval:
        values = (self.estimate, self.low, self.high)
        if self.successes > self.n:
            raise ValueError("successes cannot exceed the population")
        if self.n == 0:
            if self.reason != "empty" or any(value is not None for value in values):
                raise ValueError("an empty population requires absent interval values")
        elif (
            self.reason is not None
            or self.estimate is None
            or self.low is None
            or self.high is None
            or not self.low <= self.estimate <= self.high
            or self.estimate != self.successes / self.n
        ):
            raise ValueError("interval values must match the declared population")
        return self


class McNemarResult(_Result):
    """Conditional two-sided paired correctness test, with numerical underflow disclosed."""

    method: Literal["exact_binomial_two_sided"] = "exact_binomial_two_sided"
    n: int = Field(ge=0, le=_MAX_SAFE_COUNT)
    baseline_only_correct: int = Field(ge=0, le=MAX_EXACT_DISCORDANT)
    candidate_only_correct: int = Field(ge=0, le=MAX_EXACT_DISCORDANT)
    discordant: int = Field(ge=0, le=MAX_EXACT_DISCORDANT)
    accuracy_delta: float | None = Field(default=None, ge=-1, le=1, allow_inf_nan=False)
    p_value: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)
    log_p_value: float | None = Field(default=None, le=0, allow_inf_nan=False)
    underflowed: bool = False
    reason: Literal["empty"] | None = None

    @model_validator(mode="after")
    def _consistent_population(self) -> McNemarResult:
        if (
            self.discordant != self.baseline_only_correct + self.candidate_only_correct
            or self.discordant > self.n
        ):
            raise ValueError("discordant counts must match the paired population")
        if self.n == 0:
            if (
                self.reason != "empty"
                or any(
                    value is not None
                    for value in (self.accuracy_delta, self.p_value, self.log_p_value)
                )
                or self.underflowed
            ):
                raise ValueError("an empty paired population requires absent test values")
        elif (
            self.reason is not None
            or self.accuracy_delta
            != (self.candidate_only_correct - self.baseline_only_correct) / self.n
            or self.p_value is None
            or self.log_p_value is None
            or self.p_value != math.exp(self.log_p_value)
            or self.underflowed != (self.p_value == 0.0)
            or (self.discordant == 0 and (self.p_value != 1.0 or self.log_p_value != 0.0))
        ):
            raise ValueError("test values must match the declared paired population")
        return self


class AdjustedPValue(BaseModel):
    """One hypothesis in an explicitly named multiplicity family."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True, hide_input_in_errors=True)
    hypothesis_id: str = Field(min_length=1, max_length=128)
    raw_p_value: float = Field(ge=0, le=1, allow_inf_nan=False)
    adjusted_p_value: float = Field(ge=0, le=1, allow_inf_nan=False)

    @field_validator("hypothesis_id")
    @classmethod
    def _bounded_identifier(cls, value: str) -> str:
        if _IDENTIFIER.fullmatch(value) is None:
            raise ValueError("hypothesis IDs must be bounded ASCII identifiers")
        return value

    @model_validator(mode="after")
    def _not_smaller(self) -> AdjustedPValue:
        if self.adjusted_p_value < self.raw_p_value:
            raise ValueError("adjusted p-values cannot be smaller than their raw value")
        return self


class HolmCorrection(_Result):
    """Holm step-down adjusted p-values, retaining caller hypothesis order."""

    method: Literal["holm"] = "holm"
    family_id: str = Field(min_length=1, max_length=128)
    hypotheses: tuple[AdjustedPValue, ...] = Field(max_length=_MAX_HYPOTHESES)
    n: int = Field(ge=0, le=_MAX_HYPOTHESES)
    reason: Literal["empty"] | None = None

    @model_validator(mode="after")
    def _consistent_family(self) -> HolmCorrection:
        identifiers = [item.hypothesis_id for item in self.hypotheses]
        if (
            self.n != len(self.hypotheses)
            or len(set(identifiers)) != self.n
            or _IDENTIFIER.fullmatch(self.family_id) is None
            or any(_IDENTIFIER.fullmatch(key) is None for key in identifiers)
            or any(item.adjusted_p_value < item.raw_p_value for item in self.hypotheses)
            or (self.reason == "empty") != (self.n == 0)
        ):
            raise ValueError("correction must retain the complete named hypothesis family")
        return self


def _count(value: int) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, Integral)
        or not 0 <= value <= _MAX_SAFE_COUNT
    ):
        raise ValidationError("counts must be nonnegative integers within the exact numeric range")
    return int(value)


def _probability(value: float, *, confidence: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValidationError("probabilities must be finite real numbers in the declared domain")
    number = float(value)
    if not math.isfinite(number) or (not 0 < number < 1 if confidence else not 0 <= number <= 1):
        raise ValidationError("probabilities must be finite real numbers in the declared domain")
    return number


def wilson_interval(
    successes: int, n: int, *, confidence_level: float = 0.95
) -> ProportionInterval:
    """Return the two-sided score interval, without treating invalid predictions as exclusions."""
    successes, n = _count(successes), _count(n)
    confidence = _probability(confidence_level, confidence=True)
    if successes > n:
        raise ValidationError("successes cannot exceed the eligible population")
    if not n:
        return ProportionInterval(successes=0, n=0, confidence_level=confidence, reason="empty")
    # The lower-tail quantile avoids rounding (1 + confidence) / 2 to one.
    z = -NormalDist().inv_cdf((1 - confidence) / 2)
    estimate = successes / n
    z2 = z * z
    denominator = 1 + z2 / n
    center = (estimate + z2 / (2 * n)) / denominator
    radius = z * math.sqrt(estimate * (1 - estimate) / n + z2 / (4 * n * n)) / denominator
    return ProportionInterval(
        successes=successes,
        n=n,
        confidence_level=confidence,
        estimate=estimate,
        low=0.0 if successes == 0 else max(0.0, center - radius),
        high=1.0 if successes == n else min(1.0, center + radius),
    )


def _log_binomial_two_sided(n: int, k: int) -> float:
    if k == n // 2:
        return 0.0
    if k == 0:
        # Retain the full two-sided logarithm, including probabilities below float range.
        return (1 - n) * math.log(2)
    if n <= 4096:
        coefficient = 1
        tail = 1
        for index in range(1, k + 1):
            coefficient = coefficient * (n - index + 1) // index
            tail += coefficient
        numerator, denominator = 2 * tail, 1 << n
        if numerator * 2 >= denominator:
            return math.log1p(-((denominator - numerator) / denominator))
        power = numerator.bit_length() - 1
        return math.fsum((math.log(numerator / (1 << power)), (power - n) * math.log(2)))
    mode = n // 2
    inverse = 1 / mode
    # Stirling's central-binomial series avoids subtracting nearly equal lgamma terms.
    # Here mode >= 2048; the first omitted inverse-ninth term is below 3e-33.
    log_mode_mass = math.fsum(
        (
            -0.5 * math.log(math.pi),
            -0.5 * math.log(mode),
            inverse
            * (-1 / 8 + inverse**2 * (1 / 192 + inverse**2 * (-1 / 640 + inverse**2 * 17 / 14336))),
        )
    )
    if n % 2:
        log_mode_mass += math.log1p(-1 / (2 * mode + 2))
    elif k == mode - 1:
        return math.log1p(-math.exp(log_mode_mass))
    log_mass = math.fsum(
        (
            log_mode_mass,
            math.fsum(
                math.log1p((2 * index - n - 1) / (n - index + 1))
                if 3 * index >= n + 1
                else math.log(index / (n - index + 1))
                for index in range(mode, k, -1)
            ),
        )
    )
    terms = [1.0]
    term = 1.0
    total = 1.0
    for index in range(k, 0, -1):
        term *= index / (n - index + 1)
        terms.append(term)
        total += term
        ratio = (index - 1) / (n - index + 2)
        # Ratios decrease along the tail, bounding the uncomputed remainder.
        if ratio == 0 or term * ratio / (1 - ratio) <= sys.float_info.epsilon * total:
            break
    return min(0.0, math.fsum((math.log(2), log_mass, math.log(math.fsum(terms)))))


def mcnemar_exact(
    baseline_only_correct: int, candidate_only_correct: int, *, n_pairs: int
) -> McNemarResult:
    """Test matched correctness; delta is candidate minus baseline over all eligible pairs."""
    baseline, candidate, n = (
        _count(baseline_only_correct),
        _count(candidate_only_correct),
        _count(n_pairs),
    )
    discordant = baseline + candidate
    if discordant > n:
        raise ValidationError("discordant pairs cannot exceed the eligible paired population")
    if discordant > MAX_EXACT_DISCORDANT:
        raise ValidationError("discordant population exceeds the supported numerical range")
    if n == 0:
        return McNemarResult(
            n=0, baseline_only_correct=0, candidate_only_correct=0, discordant=0, reason="empty"
        )
    log_p = _log_binomial_two_sided(discordant, min(baseline, candidate)) if discordant else 0.0
    p_value = math.exp(log_p)
    return McNemarResult(
        n=n,
        baseline_only_correct=baseline,
        candidate_only_correct=candidate,
        discordant=discordant,
        accuracy_delta=(candidate - baseline) / n,
        p_value=p_value,
        log_p_value=log_p,
        underflowed=p_value == 0.0,
    )


def _identifier(value: str) -> str:
    if not isinstance(cast("object", value), str) or _IDENTIFIER.fullmatch(value) is None:
        raise ValidationError("family and hypothesis IDs must be bounded ASCII identifiers")
    return value


def holm_correction(p_values: Mapping[str, float], *, family_id: str) -> HolmCorrection:
    """Adjust the entire explicit family, without choosing a rejection threshold."""
    family = _identifier(family_id)
    if len(p_values) > _MAX_HYPOTHESES:
        raise ValidationError("hypothesis family exceeds the supported size")
    values = [(_identifier(key), _probability(value)) for key, value in p_values.items()]
    adjusted = [0.0] * len(values)
    maximum = 0.0
    for rank, index in enumerate(sorted(range(len(values)), key=lambda index: values[index][1])):
        maximum = min(1.0, max(maximum, (len(values) - rank) * values[index][1]))
        adjusted[index] = maximum
    return HolmCorrection(
        family_id=family,
        n=len(values),
        reason=None if values else "empty",
        hypotheses=tuple(
            AdjustedPValue(hypothesis_id=key, raw_p_value=value, adjusted_p_value=adjusted[index])
            for index, (key, value) in enumerate(values)
        ),
    )
