"""Typer entry point for the ``forge`` CLI.

Every Phase 0+1 subcommand lives here. Subcommands that haven't been
implemented yet print a clear stub message and exit non-zero so that wiring
mistakes (e.g. accidentally invoking ``forge train`` in CI) fail loudly.
"""

from __future__ import annotations

import typer

from forge.cli.doctor import doctor

__all__ = ["app"]


app = typer.Typer(
    name="forge",
    help="AI Forge — typed, async-first baseline for AI/LLM experiments.",
    no_args_is_help=True,
    add_completion=False,
)


app.command(name="doctor")(doctor)


def _stub(name: str, lands_with: str) -> None:
    """Print a deferred-command notice and exit non-zero."""
    typer.echo(f"[forge] `{name}` is not yet implemented — lands with the {lands_with} module.")
    raise typer.Exit(code=1)


@app.command(name="chat")
def chat_cmd() -> None:
    """Interactive chat with a configured LLM."""
    _stub("chat", "LLM")


@app.command(name="eval")
def eval_cmd() -> None:
    """Run an evaluation suite."""
    _stub("eval", "evals")


@app.command(name="experiments")
def experiments_cmd() -> None:
    """Manage experiment runs."""
    _stub("experiments", "evals")


@app.command(name="prompts")
def prompts_cmd() -> None:
    """Manage the prompt registry."""
    _stub("prompts", "prompts")


@app.command(name="datasets")
def datasets_cmd() -> None:
    """Manage datasets."""
    _stub("datasets", "datasets")


@app.command(name="train")
def train_cmd() -> None:
    """Fine-tune a model."""
    _stub("train", "training")


@app.command(name="serve")
def serve_cmd() -> None:
    """Serve inference."""
    _stub("serve", "compute")


@app.command(name="compute")
def compute_cmd() -> None:
    """Manage remote compute (SkyPilot / SSH)."""
    _stub("compute", "compute")


if __name__ == "__main__":  # pragma: no cover
    app()
