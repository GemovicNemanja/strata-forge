"""Runnable pipeline entrypoints executed on a compute target.

Worker modules are launched on a compute target by an orchestrator. The separate local
benchmark and registered temperature-fit SDKs consume trusted local objects in-process;
their exports do not register a worker, load an artifact or grant access. Each worker reads an inert run spec from the
environment, composes the library's primitives (serving / batch / storage), and appends
ProgressEvents to the file named by
``FORGE_PROGRESS_PATH`` so the orchestrator can tail progress over its own channel.

An orchestrator stamps every spec with the engine version it validated the spec against
(``SPEC_VERSION``, optionally suffixed ``+<commit>`` when it installs the engine from a git
ref), and the runner refuses to execute under any other installed engine. That is the
handshake that keeps a machine which already has an older engine installed from running a
spec a newer one validated: the spec's fields may be identical, and only the behaviour behind
them changed.
"""

from typing import TYPE_CHECKING

from strata_forge import __version__

if TYPE_CHECKING:
    from strata_forge.pipelines.benchmark_spec import (
        BenchmarkError as BenchmarkError,
    )
    from strata_forge.pipelines.benchmark_spec import (
        ClassifierBenchmarkPlanV1 as ClassifierBenchmarkPlanV1,
    )
    from strata_forge.pipelines.benchmark_spec import (
        ClassifierBenchmarkResult as ClassifierBenchmarkResult,
    )
    from strata_forge.pipelines.benchmark_spec import (
        ResolvedBenchmarkTarget as ResolvedBenchmarkTarget,
    )
    from strata_forge.pipelines.classifier_benchmark import (
        benchmark_scorer as benchmark_scorer,
    )
    from strata_forge.pipelines.classifier_benchmark import (
        benchmark_verifier as benchmark_verifier,
    )
    from strata_forge.pipelines.cluster_analysis import (
        paired_cluster_row_bytes as paired_cluster_row_bytes,
    )
    from strata_forge.pipelines.cluster_analysis import (
        paired_registered_cluster_uncertainty as paired_registered_cluster_uncertainty,
    )
    from strata_forge.pipelines.cluster_analysis_spec import (
        ClusterAnalysisError as ClusterAnalysisError,
    )
    from strata_forge.pipelines.cluster_analysis_spec import (
        PairedClusterArmRow as PairedClusterArmRow,
    )
    from strata_forge.pipelines.cluster_analysis_spec import (
        PairedClusterPlanV1 as PairedClusterPlanV1,
    )
    from strata_forge.pipelines.cluster_analysis_spec import (
        PairedClusterReceipt as PairedClusterReceipt,
    )
    from strata_forge.pipelines.cluster_analysis_spec import (
        PairedClusterSourceMetadata as PairedClusterSourceMetadata,
    )
    from strata_forge.pipelines.evaluation_analysis import (
        fit_registered_temperature as fit_registered_temperature,
    )
    from strata_forge.pipelines.evaluation_analysis_spec import (
        EvaluationAnalysisError as EvaluationAnalysisError,
    )
    from strata_forge.pipelines.evaluation_analysis_spec import (
        TemperatureFitPlanV1 as TemperatureFitPlanV1,
    )
    from strata_forge.pipelines.evaluation_analysis_spec import (
        TemperatureFitReceipt as TemperatureFitReceipt,
    )
    from strata_forge.pipelines.evaluation_analysis_spec import (
        TemperatureFitRow as TemperatureFitRow,
    )
    from strata_forge.pipelines.evaluation_analysis_spec import (
        TemperatureFitSourceMetadata as TemperatureFitSourceMetadata,
    )

__all__ = [
    "EVAL_AUTHORITY_SECRET",
    "HF_TOKEN_SECRET",
    "SPEC_VERSION",
    "BenchmarkError",
    "ClassifierBenchmarkPlanV1",
    "ClassifierBenchmarkResult",
    "ClusterAnalysisError",
    "EvaluationAnalysisError",
    "PairedClusterArmRow",
    "PairedClusterPlanV1",
    "PairedClusterReceipt",
    "PairedClusterSourceMetadata",
    "ResolvedBenchmarkTarget",
    "TemperatureFitPlanV1",
    "TemperatureFitReceipt",
    "TemperatureFitRow",
    "TemperatureFitSourceMetadata",
    "benchmark_scorer",
    "benchmark_verifier",
    "fit_registered_temperature",
    "paired_cluster_row_bytes",
    "paired_registered_cluster_uncertainty",
]


def __getattr__(name: str) -> object:
    if name in {
        "ClusterAnalysisError",
        "PairedClusterPlanV1",
        "PairedClusterReceipt",
        "PairedClusterArmRow",
        "PairedClusterSourceMetadata",
    }:
        import strata_forge.pipelines.cluster_analysis_spec as cluster_analysis_spec

        return getattr(cluster_analysis_spec, name)
    if name in {"paired_registered_cluster_uncertainty", "paired_cluster_row_bytes"}:
        import strata_forge.pipelines.cluster_analysis as cluster_analysis

        return getattr(cluster_analysis, name)
    if name in {
        "EvaluationAnalysisError",
        "TemperatureFitPlanV1",
        "TemperatureFitReceipt",
        "TemperatureFitRow",
        "TemperatureFitSourceMetadata",
    }:
        from strata_forge.pipelines import evaluation_analysis_spec

        return getattr(evaluation_analysis_spec, name)
    if name == "fit_registered_temperature":
        from strata_forge.pipelines.evaluation_analysis import fit_registered_temperature

        return fit_registered_temperature
    if name in {
        "BenchmarkError",
        "ClassifierBenchmarkPlanV1",
        "ClassifierBenchmarkResult",
        "ResolvedBenchmarkTarget",
    }:
        from strata_forge.pipelines import benchmark_spec

        return getattr(benchmark_spec, name)
    if name in {"benchmark_scorer", "benchmark_verifier"}:
        from strata_forge.pipelines import classifier_benchmark

        return getattr(classifier_benchmark, name)
    raise AttributeError(name)


# The engine version a spec is validated against, and the version an orchestrator writes into
# ``engine_version`` (alone, or as ``f"{SPEC_VERSION}+{commit}"``). It is the package version
# because the spec models live in this package: a change to their fields ships with a version
# bump, so the version names the contract.
SPEC_VERSION = __version__

# The key an orchestrator gives the Hugging Face token under in ``Task.secrets``, and so in the
# job's secrets file. Only declared keys are read; unknown keys refuse, so these names are
# part of the orchestrator contract, like the version above.
HF_TOKEN_SECRET = "HF_TOKEN"  # noqa: S105 — a key name, not a credential
# Evaluation authority is a masked JSON frame, never an environment/argv fallback.
EVAL_AUTHORITY_SECRET = "STRATA_EVAL_AUTHORITY"  # noqa: S105 — a private-file key name
