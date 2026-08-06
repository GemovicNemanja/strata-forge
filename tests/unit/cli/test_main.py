"""Unit tests for `strata_forge.cli.main` — top-level Typer wiring."""

from __future__ import annotations

from typer.testing import CliRunner

from strata_forge.cli.main import app

runner = CliRunner()


LIVE_COMMANDS: tuple[str, ...] = (
    "doctor",
    "chat",
    "prompts",
    "datasets",
    "eval",
    "experiments",
    "compute",
    "train",
    "serve",
)


class TestRoot:
    def test_help_lists_every_command(self) -> None:
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        for cmd in LIVE_COMMANDS:
            assert cmd in result.output

    def test_no_args_shows_help(self) -> None:
        result = runner.invoke(app, [])
        # Typer convention with no_args_is_help=True is to exit non-zero
        # and print help — useful for shells.
        assert result.exit_code != 0
        assert "Usage:" in result.output

    def test_unknown_command_errors(self) -> None:
        result = runner.invoke(app, ["nope-this-command-doesnt-exist"])
        assert result.exit_code != 0
