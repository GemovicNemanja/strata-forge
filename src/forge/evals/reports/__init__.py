"""Report renderers for :class:`Outcome` collections.

Two formats ship:

- :func:`render_markdown` — plain-text Markdown for CI logs, PR
  comments, and quick local review.
- :func:`render_html` — standalone HTML page with a summary table
  plus per-trial cards (side-by-side response and explanation
  rendering).
"""

from forge.evals.reports.html import render_html
from forge.evals.reports.markdown import render_markdown

__all__ = [
    "render_html",
    "render_markdown",
]
