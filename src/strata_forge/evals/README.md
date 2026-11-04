# strata_forge.evals

Experiment runner composing model × prompt × dataset × graders. Typed experiments, trials,
and outcomes support deterministic and model-based graders, outcome aggregates, BLEU/ROUGE,
Markdown/HTML reports, parameter sweeps, trace replay, and an eval-regression gate.

`evals.classification` re-exports explicit-label point metrics from the independent
`strata_forge.metrics` leaf. Its classification reports retain invalid predictions in gold
support and the denominator; probability metrics record missing-vector coverage and raw or
restricted provenance. Training can use the same definitions without importing this runner.

Reference: [evaluations](https://github.com/GemovicNemanja/strata-forge/blob/main/docs/modules/evals.md)
and [classification metrics](https://github.com/GemovicNemanja/strata-forge/blob/main/docs/modules/metrics.md).
