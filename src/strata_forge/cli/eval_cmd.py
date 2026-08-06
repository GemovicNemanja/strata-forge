"""``strata-forge eval`` — run a quick evaluation over the configured dataset store.

The CLI surface is intentionally narrow:

- ``eval run`` runs one model against one dataset with one or
  more named graders and writes a markdown report.
- ``eval list`` lists previously-saved reports under
  ``~/.forge/experiments``.
- ``eval show`` prints a saved report.

For richer experiments (prompt x model sweeps, LLM-judge,
pairwise), drop into Python and call
:func:`strata_forge.evals.runner.run_experiment` directly — the CLI
covers the day-to-day path, not the full surface.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import TYPE_CHECKING, Any

import typer
from rich.console import Console
from rich.table import Table

from strata_forge.cli.helpers import (
    dataset_store_from_settings,
    error_exit,
    run_async,
)

if TYPE_CHECKING:
    from strata_forge.evals.experiment import Outcome
    from strata_forge.evals.graders import Grader

__all__ = ["app"]


app = typer.Typer(
    name="eval",
    help="Run quick evaluations against the configured dataset store.",
    no_args_is_help=True,
)


_REPORT_DIR = Path.home() / ".forge" / "experiments"


def _make_grader(name: str) -> Grader:
    from strata_forge.evals.graders.exact import ExactMatch, Regex

    if name == "exact_match":
        return ExactMatch()
    if name.startswith("regex:"):
        return Regex(pattern=name.split(":", 1)[1])
    err_msg = f"unknown grader {name!r}. Supported: exact_match, regex:<pattern>."
    error_exit(err_msg)


@app.command("run")
def run_cmd(
    model: str = typer.Option(..., "--model", "-m", help="Logical model id."),
    dataset: str = typer.Option(..., "--dataset", "-d", help="Dataset name."),
    grader: list[str] = typer.Option(
        ...,
        "--grader",
        "-g",
        help="Grader name. Supports: exact_match, regex:<pattern>. Repeatable.",
    ),
    dataset_version: str | None = typer.Option(
        None, "--dataset-version", help="Pin a specific dataset version."
    ),
    provider: str | None = typer.Option(
        None, "--provider", "-p", help="Pin a specific provider route."
    ),
    temperature: float | None = typer.Option(
        None, "--temperature", "-t", help="Sampling temperature."
    ),
    max_tokens: int | None = typer.Option(None, "--max-tokens", help="Per-completion output cap."),
    concurrency: int = typer.Option(5, "--concurrency", min=1, help="Max in-flight LLM calls."),
    name: str | None = typer.Option(
        None,
        "--name",
        help="Experiment name; defaults to a timestamped slug.",
    ),
    report_path: Path | None = typer.Option(
        None,
        "--report",
        help="Path to write the markdown report. Defaults to a file under ~/.forge/experiments.",
    ),
) -> None:
    """Run a single-model evaluation and write a markdown report."""
    run_async(
        _run(
            model=model,
            provider=provider,
            dataset=dataset,
            dataset_version=dataset_version,
            grader_names=grader,
            temperature=temperature,
            max_tokens=max_tokens,
            concurrency=concurrency,
            experiment_name=name,
            report_path=report_path,
        )
    )


@app.command("list")
def list_cmd() -> None:
    """List saved eval reports under ``~/.forge/experiments``."""
    console = Console()
    if not _REPORT_DIR.exists():
        console.print("[dim](no saved experiments)[/]")
        return
    files = sorted(_REPORT_DIR.glob("*.md"))
    if not files:
        console.print("[dim](no saved experiments)[/]")
        return
    table = Table(title="experiments")
    table.add_column("name")
    table.add_column("modified")
    for path in files:
        mtime = dt.datetime.fromtimestamp(path.stat().st_mtime, dt.UTC).isoformat(
            timespec="seconds"
        )
        table.add_row(path.stem, mtime)
    console.print(table)


@app.command("show")
def show_cmd(
    name: str = typer.Argument(..., help="Experiment name (without .md)."),
) -> None:
    """Print a saved report to stdout."""
    path = _REPORT_DIR / f"{name}.md"
    if not path.exists():
        error_exit(f"report not found: {path}")
    Console().print(path.read_text())


async def _run(
    *,
    model: str,
    provider: str | None,
    dataset: str,
    dataset_version: str | None,
    grader_names: list[str],
    temperature: float | None,
    max_tokens: int | None,
    concurrency: int,
    experiment_name: str | None,
    report_path: Path | None,
) -> None:
    from strata_forge.datasets.store import DatasetNotFoundError
    from strata_forge.evals.experiment import Experiment, SamplingParams
    from strata_forge.evals.reports.markdown import render_markdown
    from strata_forge.evals.runner import run_experiment
    from strata_forge.llm.client import LLMClient

    store = dataset_store_from_settings()
    try:
        ds = await store.get(dataset, version=dataset_version)
    except DatasetNotFoundError as exc:
        error_exit(str(exc))

    provider_typed: Any = provider
    client = LLMClient(model=model, provider=provider_typed)

    graders: list[Grader] = [_make_grader(g) for g in grader_names]

    timestamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    exp_name = experiment_name or f"{dataset}-{model}-{timestamp}"

    sampling = SamplingParams(temperature=temperature, max_tokens=max_tokens, top_p=None)
    experiment = Experiment(
        name=exp_name,
        models=(model,),
        dataset_name=dataset,
        dataset_version=dataset_version,
        grader_names=tuple(g.name for g in graders),
        sampling=sampling,
    )

    outcomes: tuple[Outcome, ...] = await run_experiment(
        experiment,
        dataset=ds,
        clients={model: client},
        graders=graders,
        concurrency=concurrency,
        continue_on_error=True,
    )

    md = render_markdown(outcomes, title=f"Eval: {exp_name}")
    out_path = report_path or (_REPORT_DIR / f"{exp_name}.md")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md)

    _summarize(outcomes, exp_name, out_path)


def _summarize(outcomes: Any, exp_name: str, report_path: Path) -> None:
    console = Console()
    n = len(outcomes)
    passes = sum(
        1 for o in outcomes if all(r.passed is True for r in o.grader_results) and o.grader_results
    )
    fails = n - passes
    console.print(f"[bold]{exp_name}[/]")
    console.print(f"  outcomes: {n}  pass: [green]{passes}[/]  fail: [red]{fails}[/]")
    console.print(f"  report:   {report_path}")
