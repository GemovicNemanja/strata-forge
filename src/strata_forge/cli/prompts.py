"""``strata-forge prompts`` — inspect and render templates from the prompt registry.

The subcommands target the same :class:`PromptRegistry` that
production code uses. The backend (Langfuse vs in-memory) is
picked from :func:`strata_forge.cli.helpers.prompt_store_from_settings`,
which prefers Langfuse when configured.
"""

from __future__ import annotations

import json

import typer
from rich.console import Console
from rich.table import Table

from strata_forge.cli.helpers import error_exit, prompt_store_from_settings, run_async

__all__ = ["app"]


app = typer.Typer(
    name="prompts",
    help="Inspect and render templates from the prompt registry.",
    no_args_is_help=True,
)


@app.command("list")
def list_cmd() -> None:
    """List every template name available in the configured store."""
    run_async(_list())


@app.command("show")
def show_cmd(
    name: str = typer.Argument(..., help="Template name."),
    version: str | None = typer.Option(
        None, "--version", "-v", help="Specific version (default: latest)."
    ),
) -> None:
    """Show one template's metadata and source sections."""
    run_async(_show(name, version))


@app.command("render")
def render_cmd(
    name: str = typer.Argument(..., help="Template name."),
    variables_json: str | None = typer.Option(
        None,
        "--vars",
        "-V",
        help='JSON object with template variables (e.g. \'{"topic": "X"}\').',
    ),
    version: str | None = typer.Option(
        None, "--version", "-v", help="Specific version (default: latest)."
    ),
) -> None:
    """Render a template with the supplied variables and print the result."""
    run_async(_render(name, variables_json, version))


# ---------------------------------------------------------------------------
# Implementations
# ---------------------------------------------------------------------------


async def _list() -> None:
    from strata_forge.prompts.registry import PromptRegistry

    store = prompt_store_from_settings()
    registry = PromptRegistry(store)
    names = await registry.list_names()
    if not names:
        Console().print("[dim](no prompts registered)[/]")
        return
    table = Table(title="prompts")
    table.add_column("name")
    table.add_column("versions")
    for name in names:
        versions = await registry.versions(name)
        table.add_row(name, ", ".join(versions))
    Console().print(table)


async def _show(name: str, version: str | None) -> None:
    from strata_forge.prompts.registry import PromptNotFoundError, PromptRegistry

    store = prompt_store_from_settings()
    registry = PromptRegistry(store)
    try:
        template = await registry.get(name, version=version)
    except PromptNotFoundError as exc:
        error_exit(str(exc))

    console = Console()
    console.print(f"[bold]{template.name}[/]")
    if template.description:
        console.print(f"[dim]{template.description}[/]")
    console.print()
    console.print("[bold]stable_section[/]")
    console.print(template.stable_section)
    if template.dynamic_section:
        console.print("\n[bold]dynamic_section[/]")
        console.print(template.dynamic_section)
    if template.stable_variables:
        console.print(f"\n[bold]stable_variables[/]: {', '.join(template.stable_variables)}")
    if template.dynamic_variables:
        console.print(f"[bold]dynamic_variables[/]: {', '.join(template.dynamic_variables)}")


async def _render(name: str, variables_json: str | None, version: str | None) -> None:
    from strata_forge.prompts.registry import PromptNotFoundError, PromptRegistry
    from strata_forge.prompts.rendering import render

    variables: dict[str, object] = {}
    if variables_json is not None:
        try:
            parsed = json.loads(variables_json)
        except json.JSONDecodeError as exc:
            error_exit(f"--vars must be valid JSON: {exc}")
        if not isinstance(parsed, dict):
            error_exit("--vars must decode to a JSON object")
        variables = dict(parsed)  # type: ignore[arg-type]

    store = prompt_store_from_settings()
    registry = PromptRegistry(store)
    try:
        template = await registry.get(name, version=version)
    except PromptNotFoundError as exc:
        error_exit(str(exc))

    rendered = render(template, variables)
    console = Console()
    for msg in rendered.messages:
        role = type(msg).__name__.replace("Message", "").lower()
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        console.print(f"[bold]{role}[/]:")
        console.print(content)
        console.print()
