"""Unit tests for `forge.cli.main` — Typer skeleton + stub subcommands."""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from forge.cli.main import app

runner = CliRunner()


STUB_COMMANDS: tuple[str, ...] = (
    "chat",
    "eval",
    "experiments",
    "prompts",
    "datasets",
    "train",
    "serve",
    "compute",
)


class TestRoot:
    def test_help_lists_every_command(self) -> None:
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        # Every command — stubs plus `doctor` — must show in help.
        for cmd in (*STUB_COMMANDS, "doctor"):
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


class TestStubs:
    @pytest.mark.parametrize("cmd", STUB_COMMANDS)
    def test_stub_exits_nonzero(self, cmd: str) -> None:
        result = runner.invoke(app, [cmd])
        assert result.exit_code != 0
        assert "not yet implemented" in result.output
        assert cmd in result.output

    def test_stub_message_names_target_module(self) -> None:
        # `train` should mention the training module so operators know where it lands.
        result = runner.invoke(app, ["train"])
        assert "training" in result.output
