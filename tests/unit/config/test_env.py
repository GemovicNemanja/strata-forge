"""Unit tests for `strata_forge.config.env`."""

from __future__ import annotations

from typing import TYPE_CHECKING

from strata_forge.config.env import load_env_file

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


class TestLoadEnvFile:
    def test_returns_false_when_no_file(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.chdir(tmp_path)
        # No .env in cwd.
        assert load_env_file() is False

    def test_loads_named_file(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("FORGE_TEST_VAR_A", raising=False)
        env_file = tmp_path / "custom.env"
        env_file.write_text("FORGE_TEST_VAR_A=loaded\n")

        loaded = load_env_file(env_file)
        assert loaded is True
        import os

        assert os.environ["FORGE_TEST_VAR_A"] == "loaded"

    def test_returns_false_for_explicit_missing_path(self, tmp_path: Path) -> None:
        assert load_env_file(tmp_path / "does-not-exist.env") is False

    def test_existing_env_is_preserved_by_default(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        env_file = tmp_path / ".env"
        env_file.write_text("FORGE_TEST_VAR_B=fromfile\n")
        monkeypatch.setenv("FORGE_TEST_VAR_B", "fromshell")

        load_env_file(env_file)
        import os

        assert os.environ["FORGE_TEST_VAR_B"] == "fromshell"

    def test_override_replaces_existing_env(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        env_file = tmp_path / ".env"
        env_file.write_text("FORGE_TEST_VAR_C=fromfile\n")
        monkeypatch.setenv("FORGE_TEST_VAR_C", "fromshell")

        load_env_file(env_file, override=True)
        import os

        assert os.environ["FORGE_TEST_VAR_C"] == "fromfile"
