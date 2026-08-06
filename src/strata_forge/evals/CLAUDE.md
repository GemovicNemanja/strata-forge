# Agent rules — strata_forge.evals

`strata_forge.evals` is the experiment runner — it composes models, prompts,
datasets, and graders into typed trials and outcomes. See
[ADR 0010](../../../docs/architecture/adr/0010-evals-experiment-as-data-pluggable-graders.md)
for the experiment-as-data design.

## Purpose

- Declarative :class:`Experiment` shape (frozen Pydantic) describing
  a model × prompt × dataset × grader matrix.
- Typed :class:`Trial` / :class:`Outcome` / :class:`GraderResult`
  results consumed by reports and the CI eval gate.
- The :class:`Grader` Protocol — any async callable that returns a
  :class:`GraderResult` from an item + LLM response.
- Deterministic graders (:class:`ExactMatch`, :class:`Regex`,
  :class:`JSONStructure`, :class:`JSONField`) ship in 2.4.1.
- The async runner, metrics, parameter sweeps, LLM-driven graders,
  trace replay, reports, and CI gate land in 2.4.2–2.4.4.

## Boundaries

- **Owns:** `experiment.py`, `graders/`. The runner (2.4.2),
  metrics, sweeps, trace replay, and reports get their own files in
  later sub-phases.
- **Imports from inside `forge`:** `strata_forge.core`, `strata_forge.config`,
  `strata_forge.llm`, `strata_forge.prompts`, `strata_forge.datasets`. Does NOT import
  `strata_forge.tracing` (tracing is layered above, per ADR 0008).
- **External deps:** Pydantic at module load. No optional extras at
  this sub-phase; later sub-phases lazily import `nltk` (BLEU),
  `rouge_score` (ROUGE), and the embedding clients in graders that
  use them.

## Public API

The module's ``__init__.py`` re-exports:

- Data shapes: ``Experiment``, ``Trial``, ``Outcome``,
  ``GraderResult``, ``SamplingParams``.
- ``Grader`` Protocol.
- Deterministic graders: ``ExactMatch``, ``Regex``, ``JSONStructure``,
  ``JSONField``.

Anything raised from this module is a ``ForgeError`` subclass or a
Pydantic ``ValidationError`` from frozen-model validation.

## Internal patterns

- **Experiments are data, runner is behavior.** ``Experiment``
  carries names (model name, prompt name, dataset name, grader
  names) — never live objects. The runner resolves names against
  the relevant registries at execution time, which keeps
  ``Experiment`` portable, serializable, and content-addressable.
- **Frozen Pydantic with ``extra="forbid"``.** Wire shapes are
  stable; pyright catches mismatches between the runner and
  consumers. Collections that should be structurally immutable are
  ``tuple[...]``, not ``list[...]`` (same rationale as in
  :mod:`strata_forge.datasets`).
- **Graders are async Protocols.** Any object with a ``name``
  property and an ``async grade(*, item, response)`` method is a
  grader; subclassing isn't required. ``Protocol`` is
  ``runtime_checkable`` so user-supplied callables satisfy
  ``isinstance(obj, Grader)`` without inheriting.
- **Grader scores are floats; ``passed`` is a bool the grader
  derives.** Most graders use 0.0/1.0 for binary verdicts. LLM-judge
  graders (2.4.3) use a threshold on a continuous score.
- **Per-grader naming.** ``Grader.name`` is the key in
  :attr:`Outcome.grader_results` and the column header in reports.
  The runner enforces uniqueness across the graders attached to a
  single experiment.

## Test expectations

- Unit tests under ``tests/unit/evals/``, one file per source module.
- Coverage target: ≥ 90 % line.
- Deterministic graders need no mocking — they accept stub
  ``LLMResponse`` instances.
- LLM-driven graders (2.4.3) use ``AsyncMock`` for the ``LLMClient``
  surface; no live network in unit tests.
- The runner (2.4.2) will be tested with mocked ``LLMClient``;
  ``@pytest.mark.integration`` covers a real model × dataset run.

## Gotchas

- **``JSONField`` requires explicit ``expected=...`` vs item
  fallback.** When ``expected`` is omitted, the grader compares
  against ``item.expected_output``. Passing ``expected=None``
  explicitly is different — it asserts the JSON-path value is
  literally ``None``. Internally a sentinel distinguishes these
  cases.
- **``ExactMatch`` against ``item.expected_output=None``** is a
  failed grade with an explanation, not a crash — undefined
  comparison is treated as failure so the grader can be applied
  unconditionally across a dataset.
- **Don't import provider SDKs.** All LLM activity goes through
  ``strata_forge.llm.LLMClient`` — graders that need a model take an
  ``LLMClient`` in their constructor; they never import
  ``openai`` / ``anthropic`` / ``litellm`` directly.

## When to update this file

- Adding a new public function/class to ``__init__.py``.
- Adding a new grader shape that diverges from the established
  ``Grader`` Protocol (e.g. multi-response pairwise graders).
- Changing the ``Trial`` / ``Outcome`` shape (would break reports).
- Adding a new optional extra (e.g. ``[evals-embeddings]`` for the
  semantic grader).
