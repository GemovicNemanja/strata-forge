"""Synchronous deterministic metrics and comparisons, shared by evaluation and training."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from strata_forge.metrics.cluster_bootstrap import (
        ClusterBootstrapError as ClusterBootstrapError,
    )
    from strata_forge.metrics.cluster_bootstrap import (
        ClusterBootstrapResult as ClusterBootstrapResult,
    )
    from strata_forge.metrics.cluster_bootstrap import (
        PairedClusterObservation as PairedClusterObservation,
    )
    from strata_forge.metrics.cluster_bootstrap import (
        PairedClusterProfile as PairedClusterProfile,
    )
    from strata_forge.metrics.cluster_bootstrap import (
        paired_cluster_percentile as paired_cluster_percentile,
    )
    from strata_forge.metrics.temperature_fit import (
        TemperatureFitBounds as TemperatureFitBounds,
    )
    from strata_forge.metrics.temperature_fit import (
        TemperatureFitError as TemperatureFitError,
    )
    from strata_forge.metrics.temperature_fit import (
        TemperatureFitSolution as TemperatureFitSolution,
    )
    from strata_forge.metrics.temperature_fit import (
        fit_temperature_logits as fit_temperature_logits,
    )

from strata_forge.metrics.calibration import (
    CalibrationScores,
    TemperatureCalibration,
    temperature_scale,
)
from strata_forge.metrics.classification import (
    METRIC_VERSION,
    NLL_CLIP_EPSILON,
    PROBABILITY_SUM_TOLERANCE,
    ClassificationReport,
    ClassScore,
    ConfusionMatrix,
    LabelSpace,
    MetricValue,
    ProbabilityKind,
    accuracy,
    brier,
    classification_report,
    classification_report_from_confusion,
    confusion_matrix,
    invalid_rate,
    macro_f1,
    nll,
)
from strata_forge.metrics.comparison import (
    COMPARISON_VERSION,
    MAX_EXACT_DISCORDANT,
    AdjustedPValue,
    HolmCorrection,
    McNemarResult,
    ProportionInterval,
    holm_correction,
    mcnemar_exact,
    wilson_interval,
)

__all__ = [
    "COMPARISON_VERSION",
    "MAX_EXACT_DISCORDANT",
    "METRIC_VERSION",
    "NLL_CLIP_EPSILON",
    "PROBABILITY_SUM_TOLERANCE",
    "AdjustedPValue",
    "CalibrationScores",
    "ClassScore",
    "ClassificationReport",
    "ClusterBootstrapError",
    "ClusterBootstrapResult",
    "ConfusionMatrix",
    "HolmCorrection",
    "LabelSpace",
    "McNemarResult",
    "MetricValue",
    "PairedClusterObservation",
    "PairedClusterProfile",
    "ProbabilityKind",
    "ProportionInterval",
    "TemperatureCalibration",
    "TemperatureFitBounds",
    "TemperatureFitError",
    "TemperatureFitSolution",
    "accuracy",
    "brier",
    "classification_report",
    "classification_report_from_confusion",
    "confusion_matrix",
    "fit_temperature_logits",
    "holm_correction",
    "invalid_rate",
    "macro_f1",
    "mcnemar_exact",
    "nll",
    "paired_cluster_percentile",
    "temperature_scale",
    "wilson_interval",
]


def __getattr__(name: str) -> object:
    if name in {
        "ClusterBootstrapError",
        "ClusterBootstrapResult",
        "PairedClusterObservation",
        "PairedClusterProfile",
        "paired_cluster_percentile",
    }:
        import strata_forge.metrics.cluster_bootstrap as cluster_bootstrap

        return getattr(cluster_bootstrap, name)
    if name in {
        "TemperatureFitBounds",
        "TemperatureFitError",
        "TemperatureFitSolution",
        "fit_temperature_logits",
    }:
        import strata_forge.metrics.temperature_fit as temperature_fit

        return getattr(temperature_fit, name)
    raise AttributeError(name)
