"""Render a :class:`Outcome` tuple as a standalone HTML page.

The HTML report is heavier than the Markdown one: it ships a small
inline stylesheet, a summary table, and one card per trial with the
full response, expected output (when present), and per-grader
verdicts side by side. Open with a browser or attach to a CI artifact.

No external assets — everything is inlined so the output works as a
single file.
"""

from __future__ import annotations

import html
from typing import TYPE_CHECKING

from strata_forge.evals.metrics import pass_rate_by_grader, pass_rate_by_model

if TYPE_CHECKING:
    from collections.abc import Sequence

    from strata_forge.evals.experiment import Outcome

__all__ = [
    "render_html",
]


_STYLES = """
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
       max-width: 1100px; margin: 2em auto; padding: 0 1em; color: #222; }
h1 { border-bottom: 2px solid #444; padding-bottom: 0.3em; }
h2 { margin-top: 2em; }
table { border-collapse: collapse; width: 100%; margin: 1em 0; }
th, td { border: 1px solid #ccc; padding: 0.5em; text-align: left; }
th { background: #f4f4f4; }
.pass { color: #1a7f37; font-weight: 600; }
.fail { color: #cf222e; font-weight: 600; }
.card { border: 1px solid #ddd; border-radius: 6px; padding: 1em;
        margin: 1em 0; background: #fafafa; }
.card-header { font-weight: 600; margin-bottom: 0.5em; }
.card-row { display: flex; gap: 1em; margin-top: 0.5em; }
.card-col { flex: 1; min-width: 0; }
.card-col h4 { margin: 0 0 0.3em 0; font-size: 0.85em; color: #555;
               text-transform: uppercase; letter-spacing: 0.05em; }
.card-col pre { background: #fff; border: 1px solid #e0e0e0;
                padding: 0.5em; border-radius: 4px;
                white-space: pre-wrap; word-wrap: break-word;
                font-size: 0.85em; max-height: 220px; overflow: auto; }
.grader-list { list-style: none; padding: 0; margin: 0.5em 0 0 0; }
.grader-list li { padding: 0.3em 0; border-top: 1px solid #eee;
                  font-size: 0.9em; }
.grader-list li:first-child { border-top: none; }
.muted { color: #666; font-size: 0.85em; }
"""


def _escape(text: str) -> str:
    return html.escape(text, quote=True)


def _verdict_cell(passed: bool | None) -> str:
    if passed is None:
        return '<span class="muted">—</span>'
    cls = "pass" if passed else "fail"
    glyph = "✓ pass" if passed else "✗ fail"
    return f'<span class="{cls}">{glyph}</span>'


def render_html(
    outcomes: Sequence[Outcome],
    *,
    title: str = "Eval report",
) -> str:
    """Render outcomes as a standalone HTML string.

    The returned string is a complete HTML document — write to a
    file or stream straight to a response body.
    """
    body_parts: list[str] = [f"<h1>{_escape(title)}</h1>"]

    if not outcomes:
        body_parts.append("<p><em>No outcomes.</em></p>")
        return _wrap_document(title, "".join(body_parts))

    total_trials = len(outcomes)
    total_cost = sum(o.trial.cost_usd for o in outcomes)
    mean_latency = sum(o.trial.latency_ms for o in outcomes) / total_trials
    grader_rates = pass_rate_by_grader(outcomes)

    body_parts.append("<h2>Summary</h2>")
    body_parts.append("<table>")
    body_parts.append(
        f"<tr><th>Total trials</th><td>{total_trials}</td></tr>"
        f"<tr><th>Total cost (USD)</th><td>{total_cost:.6f}</td></tr>"
        f"<tr><th>Mean latency (ms)</th><td>{mean_latency:.1f}</td></tr>"
    )
    body_parts.append("</table>")

    if grader_rates:
        body_parts.append("<h3>Pass rate by grader</h3>")
        body_parts.append("<table><tr><th>Grader</th><th>Pass rate</th></tr>")
        for name, rate in sorted(grader_rates.items()):
            body_parts.append(f"<tr><td><code>{_escape(name)}</code></td><td>{rate:.1%}</td></tr>")
        body_parts.append("</table>")

        body_parts.append("<h3>Pass rate by model x grader</h3>")
        graders_sorted = sorted(grader_rates)
        models = sorted({o.trial.model for o in outcomes})
        header = (
            "<tr><th>Model</th>"
            + "".join(f"<th><code>{_escape(g)}</code></th>" for g in graders_sorted)
            + "</tr>"
        )
        body_parts.append("<table>" + header)
        for model in models:
            row = f"<tr><td>{_escape(model)}</td>"
            for grader in graders_sorted:
                rates = pass_rate_by_model(outcomes, grader_name=grader)
                row += f"<td>{rates.get(model, 0.0):.1%}</td>"
            row += "</tr>"
            body_parts.append(row)
        body_parts.append("</table>")

    body_parts.append("<h2>Trials</h2>")
    for outcome in outcomes:
        trial = outcome.trial
        body_parts.append('<div class="card">')
        body_parts.append(
            '<div class="card-header">'
            f"{_escape(trial.model)} · "
            f"<span class='muted'>item {_escape(trial.item_id[:12])} · "
            f"cost ${trial.cost_usd:.6f} · "
            f"{trial.latency_ms:.1f}ms</span>"
            "</div>"
        )

        body_parts.append('<div class="card-row">')
        body_parts.append(
            '<div class="card-col"><h4>Response</h4>'
            f"<pre>{_escape(trial.response_text or '(empty)')}</pre></div>"
        )
        body_parts.append("</div>")

        if outcome.grader_results:
            body_parts.append('<ul class="grader-list">')
            for result in outcome.grader_results:
                body_parts.append(
                    f"<li><code>{_escape(result.grader_name)}</code> "
                    f"{_verdict_cell(result.passed)} "
                    f"<span class='muted'>(score {result.score:.3f})</span>"
                )
                if result.explanation:
                    body_parts.append(
                        f"<br><span class='muted'>{_escape(result.explanation)}</span>"
                    )
                body_parts.append("</li>")
            body_parts.append("</ul>")

        body_parts.append("</div>")

    return _wrap_document(title, "".join(body_parts))


def _wrap_document(title: str, body: str) -> str:
    return (
        "<!doctype html>\n"
        '<html lang="en">\n<head>\n'
        '<meta charset="utf-8">\n'
        f"<title>{_escape(title)}</title>\n"
        f"<style>{_STYLES}</style>\n"
        "</head>\n<body>\n"
        f"{body}\n"
        "</body>\n</html>\n"
    )
