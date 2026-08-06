"""``strata-forge experiments`` — alias group over saved eval reports.

Mirrors ``strata-forge eval list`` / ``strata-forge eval show`` semantics so
users with a workflow built around "experiments" don't have to
learn a parallel verb. The state lives under
``~/.forge/experiments``.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from strata_forge.cli.helpers import error_exit

__all__ = ["app"]


app = typer.Typer(
    name="experiments",
    help="Inspect saved evaluation experiment reports.",
    no_args_is_help=True,
)


_REPORT_DIR = Path.home() / ".forge" / "experiments"


@app.command("list")
def list_cmd() -> None:
    """List saved experiment reports."""
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
    table.add_column("size")
    table.add_column("modified")
    for path in files:
        stat = path.stat()
        mtime = dt.datetime.fromtimestamp(stat.st_mtime, dt.UTC).isoformat(timespec="seconds")
        table.add_row(path.stem, f"{stat.st_size} B", mtime)
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


@app.command("delete")
def delete_cmd(
    name: str = typer.Argument(..., help="Experiment name (without .md)."),
) -> None:
    """Remove a saved report from the local cache."""
    path = _REPORT_DIR / f"{name}.md"
    if not path.exists():
        error_exit(f"report not found: {path}")
    path.unlink()
    Console().print(f"[bold green]deleted[/] {name}")
