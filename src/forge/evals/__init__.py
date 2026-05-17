"""Typed evaluation primitives: experiments, trials, grader Protocol, results.

The 2.4.1 surface ships the foundation pieces — typed shapes and the
deterministic graders. The runner (2.4.2), LLM-driven graders + trace
replay (2.4.3), and reports + CI gate (2.4.4) land in subsequent
sub-phases.
"""

from forge.evals.experiment import (
    Experiment,
    GraderResult,
    Outcome,
    SamplingParams,
    Trial,
)
from forge.evals.graders import (
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
from forge.evals.metrics import (
    bleu,
    mean_score,
    pass_rate,
    pass_rate_by_grader,
    pass_rate_by_model,
    rouge,
)
from forge.evals.runner import PromptRenderer, run_experiment
from forge.evals.sweeps import sweep, sweep_sampling
from forge.evals.trace_replay import ReplayOverrides, ReplayResult, replay_trace

__all__ = [
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
    "mean_score",
    "pass_rate",
    "pass_rate_by_grader",
    "pass_rate_by_model",
    "replay_trace",
    "rouge",
    "run_experiment",
    "sweep",
    "sweep_sampling",
]
