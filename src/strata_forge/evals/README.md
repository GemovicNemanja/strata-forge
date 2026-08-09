# strata_forge.evals

Experiment runner composing model × prompt × dataset × graders. An `Experiment` is inert data
naming what to run; `run_experiment` turns it into concurrent LLM calls and returns one `Outcome`
per trial, carrying the response and every grader's verdict. See
[ADR 0010](https://github.com/GemovicNemanja/strata-forge/blob/main/docs/architecture/adr/0010-evals-experiment-as-data-pluggable-graders.md)
for the design.

Graders satisfy one `Grader` Protocol: deterministic ones (`ExactMatch`, `Regex`, `JSONStructure`,
`JSONField`) and LLM-driven ones (`LLMJudge`, `PairwiseGrader`, `SemanticSimilarity`). Around them
sit metrics (`pass_rate`, `mean_score`, `bleu`, `rouge`), Markdown and HTML reports, parameter
sweeps, Langfuse trace replay, and `evaluate_ci_gate` — a Wilson-bounded regression gate for CI.

`bleu` / `rouge` need the `[evals]` extra; trace replay needs `[langfuse]`.

Reference:
[docs/modules/evals.md](https://github.com/GemovicNemanja/strata-forge/blob/main/docs/modules/evals.md).
