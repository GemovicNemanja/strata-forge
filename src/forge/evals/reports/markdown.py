"""Render a :class:`Outcome` tuple as a Markdown report.

The report has three sections:

- **Summary** — total trials, total cost, mean latency, and pass
  rates per grader plus per model.
- **Per-trial table** — one row per outcome with the model, item id,
  truncated response, and per-grader verdict (``✓`` / ``✗``).
- **Failures** — when any outcome has at least one failed grader,
  list them with the grader name and explanation.

Output is intentionally plain Markdown so it pastes into PR comments,
CI logs, or `cat` to a terminal. No emoji except for the ✓/✗ glyphs
used for at-a-glance scanning of the pass/fail column.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from forge.evals.metrics import pass_rate_by_grader, pass_rate_by_model

if TYPE_CHECKING:
    from collections.abc import Sequence

    from forge.evals.experiment import Outcome

__all__ = [
    "render_markdown",
]

_PASS_GLYPH = "✓"  # noqa: S105 — Unicode check-mark glyph
_FAIL_GLYPH = "✗"  # Unicode ballot-x glyph


def _truncate(text: str, limit: int) -> str:
    text = text.replace("\n", " ").replace("|", "\\|")
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


def render_markdown(
    outcomes: Sequence[Outcome],
    *,
    title: str = "Eval report",
    response_truncate: int = 80,
    include_failures: bool = True,
) -> str:
    """Render the outcomes as a Markdown string.

    Args:
        outcomes: The outcome tuple from :func:`run_experiment`.
        title: Header used in the report's first line.
        response_truncate: Maximum character count for the truncated
            response in the per-trial table.
        include_failures: When ``True`` (default), append a
            "Failures" section listing each outcome with at least
            one failing grader plus the failing graders' explanations.

    Returns:
        A Markdown string suitable for direct rendering.
    """
    lines: list[str] = [f"# {title}", ""]

    if not outcomes:
        lines.append("_No outcomes._")
        return "\n".join(lines)

    total_trials = len(outcomes)
    total_cost = sum(o.trial.cost_usd for o in outcomes)
    mean_latency = sum(o.trial.latency_ms for o in outcomes) / total_trials

    lines.extend(
        [
            "## Summary",
            "",
            f"- **Total trials:** {total_trials}",
            f"- **Total cost (USD):** {total_cost:.6f}",
            f"- **Mean latency (ms):** {mean_latency:.1f}",
            "",
        ]
    )

    grader_rates = pass_rate_by_grader(outcomes)
    if grader_rates:
        lines.append("### Pass rate by grader")
        lines.append("")
        lines.append("| Grader | Pass rate |")
        lines.append("|---|---|")
        for name, rate in sorted(grader_rates.items()):
            lines.append(f"| `{name}` | {rate:.1%} |")
        lines.append("")

    # Per-grader model breakdown.
    if grader_rates:
        lines.append("### Pass rate by model x grader")
        lines.append("")
        graders_sorted = sorted(grader_rates)
        header = "| Model | " + " | ".join(f"`{g}`" for g in graders_sorted) + " |"
        sep = "|---|" + "|".join("---" for _ in graders_sorted) + "|"
        lines.append(header)
        lines.append(sep)
        models = sorted({o.trial.model for o in outcomes})
        for model in models:
            cells = [f"| {model}"]
            for grader in graders_sorted:
                rates = pass_rate_by_model(outcomes, grader_name=grader)
                cells.append(f"| {rates.get(model, 0.0):.1%}")
            cells.append("|")
            lines.append("".join(cells))
        lines.append("")

    # Per-trial table.
    lines.append("## Trials")
    lines.append("")
    graders_in_order = list(outcomes[0].grader_results)
    grader_names = [g.grader_name for g in graders_in_order]
    header = "| Model | Item | Response | " + " | ".join(grader_names) + " |"
    sep = "|---|---|---|" + "|".join("---" for _ in grader_names) + "|"
    lines.append(header)
    lines.append(sep)
    for outcome in outcomes:
        trial = outcome.trial
        response = _truncate(trial.response_text, response_truncate)
        verdicts: list[str] = []
        for name in grader_names:
            result = outcome.by_grader(name)
            if result is None:
                verdicts.append("—")
            else:
                verdicts.append(_PASS_GLYPH if result.passed else _FAIL_GLYPH)
        lines.append(
            f"| {trial.model} | {trial.item_id[:12]} | {response} | " + " | ".join(verdicts) + " |"
        )
    lines.append("")

    if include_failures:
        failures = [o for o in outcomes if any(not r.passed for r in o.grader_results)]
        if failures:
            lines.append("## Failures")
            lines.append("")
            for outcome in failures:
                trial = outcome.trial
                lines.append(f"- **{trial.model} / {trial.item_id[:12]}**")
                for result in outcome.grader_results:
                    if result.passed:
                        continue
                    explanation = result.explanation or "(no explanation)"
                    lines.append(
                        f"  - `{result.grader_name}` (score={result.score:.3f}): {explanation}"
                    )
            lines.append("")

    return "\n".join(lines)
