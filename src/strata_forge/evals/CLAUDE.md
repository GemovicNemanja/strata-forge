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
- Deterministic graders: :class:`ExactMatch`, :class:`Regex`,
  :class:`JSONStructure`, :class:`JSONField`.
- LLM-driven graders: :class:`LLMJudge`, :class:`PairwiseGrader`,
  :class:`SemanticSimilarity`.
- The async runner (:func:`run_experiment`), metrics, parameter
  sweeps, trace replay, Markdown/HTML reports, and the CI
  regression gate.

## Boundaries

- **Owns:** `experiment.py`, `runner.py`, `metrics.py`, `sweeps.py`,
  `trace_replay.py`, `ci_gate.py`, `reports/` (`markdown.py`,
  `html.py`), `graders/` (`base.py`, `exact.py`, `json_grader.py`,
  `llm_judge.py`, `pairwise.py`, `semantic.py`).
- **Imports from inside `forge`:** `strata_forge.core`, `strata_forge.config`,
  `strata_forge.llm`, `strata_forge.prompts`, `strata_forge.datasets`. Does NOT import
  `strata_forge.tracing` (tracing is layered above, per ADR 0008).
- **External deps:** Pydantic at module load. `metrics.bleu` lazily
  imports `nltk` and `metrics.rouge` lazily imports `rouge_score`,
  both behind the `[evals]` extra — importing `strata_forge.evals`
  without the extra must keep working. `trace_replay` lazily imports
  `langfuse` behind the `[langfuse]` extra. Graders that need
  embeddings take an ``EmbedFn`` callable rather than importing an
  SDK.

## Public API

The module's ``__init__.py`` re-exports exactly these symbols
(mirror any change here into ``__all__``):

- Data shapes: ``Experiment``, ``Trial``, ``Outcome``,
  ``GraderResult``, ``SamplingParams``.
- Grader contract: ``Grader``.
- Deterministic graders: ``ExactMatch``, ``Regex``,
  ``JSONStructure``, ``JSONField``.
- LLM-driven graders: ``LLMJudge``, ``JudgeVerdict``,
  ``PairwiseGrader``, ``PairwiseVerdict``, ``SemanticSimilarity``,
  ``EmbedFn``, ``cosine_similarity``.
- Runner: ``run_experiment``, ``PromptRenderer``.
- Metrics: ``pass_rate``, ``pass_rate_by_grader``,
  ``pass_rate_by_model``, ``mean_score``, ``bleu``, ``rouge``.
- Sweeps: ``sweep``, ``sweep_sampling``.
- Trace replay: ``replay_trace``, ``ReplayOverrides``,
  ``ReplayResult``.
- Reports: ``render_markdown``, ``render_html``.
- CI gate: ``evaluate_ci_gate``, ``CIGateThresholds``,
  ``CIGateResult``, ``wilson_lower_bound``.

**Adding a grader touches two files.** Import and re-export it in
``graders/__init__.py`` *and* in ``evals/__init__.py`` — in both
cases the import statement **and** the ``__all__`` list. A grader
that is only added to one of them is invisible from
``strata_forge.evals``.

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
  derives.** Most graders use 0.0/1.0 for binary verdicts. The
  LLM-judge and semantic graders threshold a continuous score
  (``pass_threshold``).
- **Per-grader naming.** ``Grader.name`` is the key in
  :attr:`Outcome.grader_results` and the column header in reports.
  The runner enforces uniqueness across the graders attached to a
  single experiment.

## Test expectations

- Unit tests under ``tests/unit/evals/``, one file per source module.
- Coverage: the enforced gate is the repo-wide 85 % line floor
  (``fail_under`` in ``pyproject.toml``); treat a drop in this module
  as a regression.
- Deterministic graders need no mocking — they accept stub
  ``LLMResponse`` instances.
- LLM-driven graders use ``AsyncMock`` for the ``LLMClient``
  surface; no live network in unit tests.
- The runner is tested with a mocked ``LLMClient``;
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
