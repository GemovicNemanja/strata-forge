"""``forge datasets`` — list, show, and head the configured dataset store.

Subcommands wrap :class:`DatasetStore` (Langfuse when configured,
otherwise the in-memory fallback). ``head`` prints the first N
items; ``show`` prints metadata only.
"""

from __future__ import annotations

import json

import typer
from rich.console import Console
from rich.table import Table

from strata_forge.cli.helpers import dataset_store_from_settings, error_exit, run_async

__all__ = ["app"]


app = typer.Typer(
    name="datasets",
    help="Inspect datasets in the configured store.",
    no_args_is_help=True,
)


@app.command("list")
def list_cmd() -> None:
    """List dataset names in the store."""
    run_async(_list())


@app.command("show")
def show_cmd(
    name: str = typer.Argument(..., help="Dataset name."),
    version: str | None = typer.Option(
        None, "--version", "-v", help="Specific version (default: latest)."
    ),
) -> None:
    """Print metadata for one dataset (item count, version, description)."""
    run_async(_show(name, version))


@app.command("head")
def head_cmd(
    name: str = typer.Argument(..., help="Dataset name."),
    n: int = typer.Option(5, "--n", "-n", min=1, help="How many items to show."),
    version: str | None = typer.Option(
        None, "--version", "-v", help="Specific version (default: latest)."
    ),
) -> None:
    """Print the first N items of a dataset as JSON."""
    run_async(_head(name, n, version))


# ---------------------------------------------------------------------------
# Implementations
# ---------------------------------------------------------------------------


async def _list() -> None:
    store = dataset_store_from_settings()
    names = await store.list_names()
    if not names:
        Console().print("[dim](no datasets registered)[/]")
        return
    table = Table(title="datasets")
    table.add_column("name")
    table.add_column("versions")
    for name in names:
        versions = await store.versions(name)
        table.add_row(name, ", ".join(versions))
    Console().print(table)


async def _show(name: str, version: str | None) -> None:
    from strata_forge.datasets.store import DatasetNotFoundError

    store = dataset_store_from_settings()
    try:
        dataset = await store.get(name, version=version)
    except DatasetNotFoundError as exc:
        error_exit(str(exc))

    console = Console()
    console.print(f"[bold]{dataset.name}[/]")
    if dataset.description:
        console.print(f"[dim]{dataset.description}[/]")
    console.print(f"items: [green]{len(dataset.items)}[/]")
    if dataset.metadata:
        console.print(f"metadata: {dict(dataset.metadata)}")


async def _head(name: str, n: int, version: str | None) -> None:
    from strata_forge.datasets.store import DatasetNotFoundError

    store = dataset_store_from_settings()
    try:
        dataset = await store.get(name, version=version)
    except DatasetNotFoundError as exc:
        error_exit(str(exc))

    console = Console()
    for i, item in enumerate(dataset.items[:n]):
        console.print(f"[bold]item {i}[/] [dim]id={item.id}[/]")
        payload = {
            "input": item.input,
            "expected_output": item.expected_output,
        }
        console.print(json.dumps(payload, indent=2, default=str))
        console.print()
    if len(dataset.items) > n:
        console.print(f"[dim]({len(dataset.items) - n} more items omitted)[/]")
