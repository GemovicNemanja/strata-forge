"""Aggregations over :class:`Outcome` collections.

Two flavours:

- **Pure-Python metrics** (``pass_rate``, ``mean_score``, the
  ``by_model`` and ``by_grader`` group-bys) — no extras required.
- **Lazy-imported text metrics** (``bleu``, ``rouge``) — gated on
  the ``[evals]`` extra. Importing this module without the extra
  works; calling ``bleu`` or ``rouge`` without it raises a clear
  ``ImportError`` with the install hint.

All metrics operate on a sequence of :class:`Outcome` instances —
the result-set the runner produces.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

    from strata_forge.evals.experiment import Outcome

__all__ = [
    "bleu",
    "mean_score",
    "pass_rate",
    "pass_rate_by_grader",
    "pass_rate_by_model",
    "rouge",
]


# ---------------------------------------------------------------------------
# Pure-Python metrics
# ---------------------------------------------------------------------------


def _collect_results(outcomes: Sequence[Outcome], grader_name: str) -> list[Any]:
    """Return every :class:`GraderResult` named ``grader_name`` across outcomes."""
    results: list[Any] = []
    for outcome in outcomes:
        for r in outcome.grader_results:
            if r.grader_name == grader_name:
                results.append(r)
    return results


def pass_rate(outcomes: Sequence[Outcome], *, grader_name: str) -> float:
    """Fraction of outcomes whose ``grader_name`` result is ``passed``.

    Returns 0.0 when no outcome has a result from ``grader_name`` —
    a caller that needs to distinguish "0% passing" from "no data"
    should check :func:`count` separately.
    """
    results = _collect_results(outcomes, grader_name)
    if not results:
        return 0.0
    passed = sum(1 for r in results if r.passed)
    return passed / len(results)


def mean_score(outcomes: Sequence[Outcome], *, grader_name: str) -> float:
    """Arithmetic mean of ``score`` across the named grader's results."""
    results = _collect_results(outcomes, grader_name)
    if not results:
        return 0.0
    return sum(r.score for r in results) / len(results)


def pass_rate_by_model(outcomes: Sequence[Outcome], *, grader_name: str) -> dict[str, float]:
    """Map model name → pass rate against ``grader_name``."""
    by_model: dict[str, list[Any]] = {}
    for outcome in outcomes:
        for r in outcome.grader_results:
            if r.grader_name == grader_name:
                by_model.setdefault(outcome.trial.model, []).append(r)
    return {
        model: sum(1 for r in results if r.passed) / len(results)
        for model, results in by_model.items()
    }


def pass_rate_by_grader(outcomes: Sequence[Outcome]) -> dict[str, float]:
    """Map grader name → overall pass rate."""
    by_grader: dict[str, list[Any]] = {}
    for outcome in outcomes:
        for r in outcome.grader_results:
            by_grader.setdefault(r.grader_name, []).append(r)
    return {
        grader: sum(1 for r in results if r.passed) / len(results)
        for grader, results in by_grader.items()
    }


# ---------------------------------------------------------------------------
# Lazy-imported text metrics
# ---------------------------------------------------------------------------


def _missing_evals_extra(metric: str) -> ImportError:
    msg = (
        f"The [evals] extra is required for {metric}. "
        f"Install it with: pip install 'strata-forge[evals]'."
    )
    return ImportError(msg)


def bleu(outcomes: Sequence[Outcome]) -> float:
    """Mean BLEU between each trial's response and its item's expected_output.

    Skips outcomes whose item lacks an ``expected_output``. Returns
    0.0 when no outcome contributes a comparable pair.

    Requires the ``[evals]`` extra (``nltk``).
    """
    try:
        bleu_score: Any = __import__("nltk.translate.bleu_score", fromlist=["sentence_bleu"])
    except ImportError as exc:
        raise _missing_evals_extra("bleu") from exc

    smoother: Any = bleu_score.SmoothingFunction().method1
    sentence_bleu: Any = bleu_score.sentence_bleu
    scores: list[float] = []
    for outcome in outcomes:
        expected = _expected_text(outcome)
        if expected is None:
            continue
        score = float(
            sentence_bleu(
                [expected.split()],
                outcome.trial.response_text.split(),
                smoothing_function=smoother,
            )
        )
        scores.append(score)
    if not scores:
        return 0.0
    return sum(scores) / len(scores)


def rouge(
    outcomes: Sequence[Outcome],
    *,
    variant: str = "rougeL",
) -> float:
    """Mean ROUGE F-measure between each trial's response and expected_output.

    ``variant`` is forwarded to ``rouge_score.RougeScorer`` —
    common values are ``"rouge1"``, ``"rouge2"``, ``"rougeL"``.

    Requires the ``[evals]`` extra (``rouge-score``).
    """
    try:
        rouge_scorer: Any = __import__("rouge_score.rouge_scorer", fromlist=["RougeScorer"])
    except ImportError as exc:
        raise _missing_evals_extra("rouge") from exc

    scorer: Any = rouge_scorer.RougeScorer([variant], use_stemmer=True)
    fmeasures: list[float] = []
    for outcome in outcomes:
        expected = _expected_text(outcome)
        if expected is None:
            continue
        scored: dict[str, Any] = scorer.score(expected, outcome.trial.response_text)
        fmeasures.append(float(scored[variant].fmeasure))
    if not fmeasures:
        return 0.0
    return sum(fmeasures) / len(fmeasures)


def _expected_text(outcome: Outcome) -> str | None:
    """Pull out the expected_output as a string, or None if absent.

    The :class:`Outcome` only carries the trial, not the item — so
    we look at the trial's metadata for an ``expected_text`` hint.
    Callers that need text-similarity metrics should populate
    ``trial.metadata['expected_text']`` (the runner doesn't do this
    by default; downstream tooling that wants BLEU/ROUGE wires it
    in).
    """
    expected = outcome.trial.metadata.get("expected_text")
    if expected is None:
        return None
    return str(expected)
