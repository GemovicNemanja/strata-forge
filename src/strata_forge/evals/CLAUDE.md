# Agent rules — strata_forge.evals

The experiment runner composes models, prompts, datasets, and graders into typed trials and
outcomes. Read [the reference](../../../docs/modules/evals.md),
[ADR 0010](../../../docs/architecture/adr/0010-evals-experiment-as-data-pluggable-graders.md),
and [module boundaries](../../../docs/architecture/module-boundaries.md).

## Ownership and API

- Owns typed `Experiment`, `Trial`, `Outcome`, `GraderResult`, `SamplingParams`; the async
  runner; deterministic and model-based graders; outcome metrics; reports; sweeps; trace
  replay; and the CI regression gate. `__init__.py` lists the runner-facing exports.
- `classification.py` re-exports implemented public functions/types from the pure
  `strata_forge.metrics` leaf. Classification definitions belong in that leaf, never in a
  runner or a training-to-evaluation dependency.
- Outcome aggregates (`evals.metrics`) and explicit-label classification point metrics
  (`evals.classification`) have separate input contracts. Do not silently interchange them.
- Permitted Forge imports follow the central boundary table. Never import agents or training.
  All model calls go through `LLMClient`; no direct provider SDK imports.
- Optional BLEU/ROUGE and trace-replay dependencies are imported lazily and name the required
  extra in an `ImportError` installation hint. Classification point metrics need no extra;
  temperature application loads NumPy through `[scoring]` only when used.

## Patterns

- Experiments are data; runners resolve names to behavior. No live callables in the inert
  `Experiment` shape. Frozen Pydantic models reject extra fields and use immutable tuples
  where collections form part of the contract.
- Graders satisfy the runtime-checkable async `Grader` Protocol. `Grader.name` is unique per
  experiment and is the key in outcomes/reports. Scores are floats; each grader defines its
  pass decision. Undefined expected output produces a failed grade with an explanation.
- `JSONField(expected=None)` asserts literal null; omitting `expected` uses the dataset item.
- Preserve classification populations, ordered labels, invalid counts, probability provenance,
  calibration identity, and missing-vector coverage through any report/runner integration.
  Different configs, splits, class collapses, and diagnostic slices are not implicitly pooled.
- Pure deterministic classification re-exports are synchronous under ADR 0020. Execution,
  model calls, and I/O remain async; expensive numerical work belongs on workers/off-loop.

## Checks and maintenance

Tests live under `tests/unit/evals/`; the shared metric conformance cases live under
`tests/unit/metrics/`. Use fake responses and clients, not network calls. Keep reference docs,
module README, this file, exports, and meaningful tests in the same API change. New numerical
definitions need known-value cases and explicit population/undefinedness policies.
