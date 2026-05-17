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
    ExactMatch,
    Grader,
    JSONField,
    JSONStructure,
    Regex,
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

__all__ = [
    "ExactMatch",
    "Experiment",
    "Grader",
    "GraderResult",
    "JSONField",
    "JSONStructure",
    "Outcome",
    "PromptRenderer",
    "Regex",
    "SamplingParams",
    "Trial",
    "bleu",
    "mean_score",
    "pass_rate",
    "pass_rate_by_grader",
    "pass_rate_by_model",
    "rouge",
    "run_experiment",
    "sweep",
    "sweep_sampling",
]
