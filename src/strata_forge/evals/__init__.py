"""Typed evaluation primitives: experiments, trials, grader Protocol, results.

The public surface covers the whole eval loop: the declarative
:class:`Experiment` shape, the :class:`Grader` Protocol, deterministic and
LLM-driven graders, the async experiment runner, metrics, parameter sweeps,
trace replay, Markdown/HTML reports, and the CI regression gate.
"""

from strata_forge.evals.ci_gate import (
    CIGateResult,
    CIGateThresholds,
    evaluate_ci_gate,
    wilson_lower_bound,
)
from strata_forge.evals.experiment import (
    Experiment,
    GraderResult,
    Outcome,
    SamplingParams,
    Trial,
)
from strata_forge.evals.graders import (
    EmbedFn,
    ExactMatch,
    Grader,
    JSONField,
    JSONStructure,
    JudgeVerdict,
    LLMJudge,
    PairwiseGrader,
    PairwiseVerdict,
    Regex,
    SemanticSimilarity,
    cosine_similarity,
)
from strata_forge.evals.metrics import (
    bleu,
    mean_score,
    pass_rate,
    pass_rate_by_grader,
    pass_rate_by_model,
    rouge,
)
from strata_forge.evals.reports import render_html, render_markdown
from strata_forge.evals.runner import PromptRenderer, run_experiment
from strata_forge.evals.sweeps import sweep, sweep_sampling
from strata_forge.evals.trace_replay import ReplayOverrides, ReplayResult, replay_trace

__all__ = [
    "CIGateResult",
    "CIGateThresholds",
    "EmbedFn",
    "ExactMatch",
    "Experiment",
    "Grader",
    "GraderResult",
    "JSONField",
    "JSONStructure",
    "JudgeVerdict",
    "LLMJudge",
    "Outcome",
    "PairwiseGrader",
    "PairwiseVerdict",
    "PromptRenderer",
    "Regex",
    "ReplayOverrides",
    "ReplayResult",
    "SamplingParams",
    "SemanticSimilarity",
    "Trial",
    "bleu",
    "cosine_similarity",
    "evaluate_ci_gate",
    "mean_score",
    "pass_rate",
    "pass_rate_by_grader",
    "pass_rate_by_model",
    "render_html",
    "render_markdown",
    "replay_trace",
    "rouge",
    "run_experiment",
    "sweep",
    "sweep_sampling",
    "wilson_lower_bound",
]
