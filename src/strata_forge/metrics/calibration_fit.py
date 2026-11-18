"""Fitting one softmax temperature, cross-fitting it by cluster, and expected calibration error."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field

from strata_forge.core.errors import ValidationError

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    import numpy as np
    from numpy.typing import NDArray

TEMPERATURE_FIT_VERSION = "temperature-nll-bisection-v1"
CROSS_FIT_VERSION = "cluster-cross-fit-v1"
ECE_VERSION = "top-label-ece-v1"
MIN_TEMPERATURE = 0.05
MAX_TEMPERATURE = 20.0
ECE_BINS = 15
_BISECTION_STEPS = 200

__all__ = [
    "CROSS_FIT_VERSION",
    "ECE_BINS",
    "ECE_VERSION",
    "MAX_TEMPERATURE",
    "MIN_TEMPERATURE",
    "TEMPERATURE_FIT_VERSION",
    "CalibrationError",
    "CrossFitTemperature",
    "TemperatureFit",
    "cross_fit_temperature",
    "expected_calibration_error",
    "fit_temperature",
]


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra="forbid", hide_input_in_errors=True)


class TemperatureFit(_Frozen):
    """The NLL-minimising temperature over one fit population, within fixed bounds."""

    fit_version: Literal["temperature-nll-bisection-v1"] = TEMPERATURE_FIT_VERSION
    temperature: float = Field(gt=0, allow_inf_nan=False)
    n: int = Field(ge=1)
    nll_before: float = Field(ge=0, allow_inf_nan=False)
    nll_after: float = Field(ge=0, allow_inf_nan=False)
    # Set when the optimum lies outside [MIN_TEMPERATURE, MAX_TEMPERATURE] and the bound was taken.
    at_bound: Literal["min", "max"] | None = None


class CrossFitTemperature(_Frozen):
    """Each fold's temperature, fitted on the other folds, and every row's held-out probabilities."""

    cross_fit_version: Literal["cluster-cross-fit-v1"] = CROSS_FIT_VERSION
    seed: int = Field(ge=0, le=2**53 - 1)
    folds: tuple[TemperatureFit, ...]
    fold_rows: tuple[int, ...]
    probabilities: tuple[tuple[float, ...], ...]


class CalibrationError(_Frozen):
    """Top-label expected calibration error: bin-count-weighted |accuracy - confidence|."""

    ece_version: Literal["top-label-ece-v1"] = ECE_VERSION
    value: float = Field(ge=0, le=1, allow_inf_nan=False)
    n: int = Field(ge=1)
    bins: int = Field(ge=1)
    scheme: Literal["equal_mass", "equal_width"]


def _require_numpy() -> None:
    try:
        import numpy as np
    except ImportError as exc:
        raise ImportError(
            "calibration fitting requires numpy: install 'strata-forge[scoring]'"
        ) from exc
    del np


def _targets(gold: Iterable[object], labels: Sequence[str]) -> list[int]:
    if isinstance(labels, str) or len(labels) < 2 or len(set(labels)) != len(labels):
        raise ValidationError("calibration_labels_invalid")
    order = {label: i for i, label in enumerate(labels)}
    out: list[int] = []
    for value in gold:
        if not isinstance(value, str) or value not in order:
            raise ValidationError("calibration_gold_invalid")
        out.append(order[value])
    return out


def _matrix(rows: Iterable[Iterable[object]], size: int, what: str) -> NDArray[np.float64]:
    import numpy as np

    values: list[list[float]] = []
    for row in rows:
        items: list[float] = []
        for v in row:
            if isinstance(v, bool) or not isinstance(v, int | float) or not math.isfinite(v):
                raise ValidationError(f"calibration_{what}_invalid")
            items.append(float(v))
        if len(items) != size:
            raise ValidationError(f"calibration_{what}_invalid")
        values.append(items)
    if not values:
        raise ValidationError("calibration_empty")
    return np.asarray(values, dtype=np.float64)


def _log_softmax(z: NDArray[np.float64], beta: float) -> NDArray[np.float64]:
    import numpy as np

    scaled = beta * z
    shifted = scaled - scaled.max(axis=1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))


def _nll(z: NDArray[np.float64], target: NDArray[np.int64], beta: float) -> float:
    import numpy as np

    logp = _log_softmax(z, beta)
    return float(-logp[np.arange(len(target)), target].mean())


def _gradient(z: NDArray[np.float64], target: NDArray[np.int64], beta: float) -> float:
    """d NLL / d beta = mean(E_p[z] - z_gold); nondecreasing in beta (its derivative is a variance)."""
    import numpy as np

    p = np.exp(_log_softmax(z, beta))
    return float(((p * z).sum(axis=1) - z[np.arange(len(target)), target]).mean())


def _fit(z: NDArray[np.float64], target: NDArray[np.int64]) -> TemperatureFit:
    low, high = 1 / MAX_TEMPERATURE, 1 / MIN_TEMPERATURE
    at_bound: Literal["min", "max"] | None = None
    if _gradient(z, target, low) >= 0:
        beta, at_bound = low, "max"
    elif _gradient(z, target, high) <= 0:
        beta, at_bound = high, "min"
    else:
        for _ in range(_BISECTION_STEPS):
            middle = (low + high) / 2
            if middle in (low, high):
                break
            if _gradient(z, target, middle) < 0:
                low = middle
            else:
                high = middle
        beta = (low + high) / 2
    return TemperatureFit(
        temperature=1 / beta,
        n=len(target),
        nll_before=_nll(z, target, 1.0),
        nll_after=_nll(z, target, beta),
        at_bound=at_bound,
    )


def fit_temperature(
    logits: Iterable[Iterable[object]], gold: Iterable[object], *, labels: Sequence[str]
) -> TemperatureFit:
    """Fit T minimising mean NLL of softmax(logits / T) over the given rows.

    NLL is convex in beta = 1/T, so its gradient is bisected on [1/MAX_TEMPERATURE,
    1/MIN_TEMPERATURE] to float precision; an optimum outside the bounds takes the bound and says
    which in ``at_bound``. Every row must carry finite logits in ``labels`` order and a gold label.
    """
    import numpy as np

    _require_numpy()
    target = _targets(gold, labels)
    z = _matrix(logits, len(labels), "logits")
    if len(target) != len(z):
        raise ValidationError("calibration_population_mismatch")
    return _fit(z, np.asarray(target, dtype=np.int64))


def cross_fit_temperature(
    logits: Iterable[Iterable[object]],
    gold: Iterable[object],
    *,
    labels: Sequence[str],
    clusters: Iterable[object],
    seed: int,
    folds: int = 2,
) -> CrossFitTemperature:
    """Fit one temperature per fold on the OTHER folds and apply it to the fold's own rows.

    Clusters (sorted by key) are permuted by ``Generator(PCG64(seed))`` and dealt into ``folds``
    contiguous groups, so a cluster's rows never sit on both sides of a fit. Every row comes back
    with probabilities from a temperature its own cluster did not influence.
    """
    import numpy as np

    _require_numpy()
    if type(seed) is not int or not 0 <= seed <= 2**53 - 1:
        raise ValidationError("calibration_seed_invalid")
    if type(folds) is not int or folds < 2:
        raise ValidationError("calibration_folds_invalid")
    target = np.asarray(_targets(gold, labels), dtype=np.int64)
    z = _matrix(logits, len(labels), "logits")
    keys: list[str] = []
    for value in clusters:
        if not isinstance(value, str) or not value:
            raise ValidationError("calibration_cluster_invalid")
        keys.append(value)
    if len(keys) != len(z) or len(target) != len(z):
        raise ValidationError("calibration_population_mismatch")
    order = sorted(set(keys))
    if len(order) < folds:
        raise ValidationError("calibration_too_few_clusters")
    permuted = np.random.Generator(np.random.PCG64(seed)).permutation(len(order))
    fold_of = {order[int(c)]: position * folds // len(order) for position, c in enumerate(permuted)}
    assignment = np.asarray([fold_of[key] for key in keys], dtype=np.int64)
    probabilities = np.zeros_like(z)
    fits: list[TemperatureFit] = []
    for fold in range(folds):
        held = assignment == fold
        fit = _fit(z[~held], target[~held])
        fits.append(fit)
        probabilities[held] = np.exp(_log_softmax(z[held], 1 / fit.temperature))
    return CrossFitTemperature(
        seed=seed,
        folds=tuple(fits),
        fold_rows=tuple(int((assignment == fold).sum()) for fold in range(folds)),
        probabilities=tuple(tuple(float(v) for v in row) for row in probabilities.tolist()),
    )


def expected_calibration_error(
    gold: Iterable[object],
    probabilities: Iterable[Iterable[object]],
    *,
    labels: Sequence[str],
    bins: int = ECE_BINS,
    scheme: Literal["equal_mass", "equal_width"] = "equal_mass",
    item_ids: Sequence[str] | None = None,
) -> CalibrationError:
    """Top-label ECE over rows with a probability vector in ``labels`` order.

    Confidence is the largest probability and the prediction its first argmax. ``equal_mass``
    sorts rows by confidence, ties broken by ``item_ids`` (else by input order; the sort is
    stable either way), and splits them into ``bins`` contiguous groups with ``array_split``
    semantics (the first ``n mod bins`` groups hold one row more); ``equal_width`` uses [k/bins, (k+1)/bins)
    with 1.0 in the last bin. Empty bins contribute nothing.
    """
    import numpy as np

    _require_numpy()
    if type(bins) is not int or bins < 1:
        raise ValidationError("calibration_bins_invalid")
    target = np.asarray(_targets(gold, labels), dtype=np.int64)
    p = _matrix(probabilities, len(labels), "probabilities")
    if len(target) != len(p):
        raise ValidationError("calibration_population_mismatch")
    if ((p < 0) | (p > 1)).any() or (np.abs(p.sum(axis=1) - 1) > 1e-6).any():
        raise ValidationError("calibration_probabilities_invalid")
    confidence = p.max(axis=1)
    correct = (p.argmax(axis=1) == target).astype(np.float64)
    if item_ids is not None and len(item_ids) != len(target):
        raise ValidationError("calibration_population_mismatch")
    if scheme == "equal_mass":
        if item_ids is None:
            order = np.argsort(confidence, kind="stable")
        else:
            rank = {item: i for i, item in enumerate(sorted(item_ids))}
            ids = np.asarray([rank[item] for item in item_ids], dtype=np.int64)
            order = np.lexsort((ids, confidence))
        groups = np.array_split(order, bins)
    elif scheme == "equal_width":
        index = np.minimum((confidence * bins).astype(np.int64), bins - 1)
        groups = [np.flatnonzero(index == b) for b in range(bins)]
    else:
        raise ValidationError("calibration_scheme_invalid")
    total = math.fsum(
        len(group) * abs(float(correct[group].mean()) - float(confidence[group].mean()))
        for group in groups
        if len(group)
    )
    return CalibrationError(value=total / len(target), n=len(target), bins=bins, scheme=scheme)
