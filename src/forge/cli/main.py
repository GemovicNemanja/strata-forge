"""Typer entry point for the ``forge`` CLI.

Every Forge module's CLI surface attaches here. There are no
stub commands left.
"""

from __future__ import annotations

import typer

from forge.cli.chat import chat
from forge.cli.compute import app as compute_app
from forge.cli.datasets import app as datasets_app
from forge.cli.doctor import doctor
from forge.cli.eval_cmd import app as eval_app
from forge.cli.experiments import app as experiments_app
from forge.cli.prompts import app as prompts_app
from forge.cli.serve import app as serve_app
from forge.cli.train import app as train_app

__all__ = ["app"]


app = typer.Typer(
    name="forge",
    help="AI Forge — typed, async-first baseline for AI/LLM experiments.",
    no_args_is_help=True,
    add_completion=False,
)


app.command(name="doctor")(doctor)
app.command(name="chat")(chat)
app.add_typer(prompts_app, name="prompts")
app.add_typer(datasets_app, name="datasets")
app.add_typer(eval_app, name="eval")
app.add_typer(experiments_app, name="experiments")
app.add_typer(compute_app, name="compute")
app.add_typer(train_app, name="train")
app.add_typer(serve_app, name="serve")


if __name__ == "__main__":  # pragma: no cover
    app()
