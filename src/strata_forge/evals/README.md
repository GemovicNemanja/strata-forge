# strata_forge.evals

Experiment runner composing model × prompt × dataset × graders.
Graders (exact match, regex, JSON-shape, JSON-field, LLM-judge,
pairwise, semantic), standard metrics (accuracy, F1, BLEU, ROUGE),
Markdown/HTML reports, parameter sweeps, trace replay against the
current code/model, and a CI eval-regression gate with absolute and
relative thresholds.

The foundation pieces — typed `Experiment` / `Trial` / `Outcome`
shapes, the `Grader` Protocol, and the deterministic graders
(`ExactMatch`, `Regex`, `JSONStructure`, `JSONField`) — ship in
2.4.1. The async runner, metrics, sweeps, LLM-driven graders, trace
replay, reports, and the CI gate land in 2.4.2 through 2.4.4. See
`docs/roadmap.md` for current status.
