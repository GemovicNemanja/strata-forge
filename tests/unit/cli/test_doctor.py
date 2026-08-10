"""Unit tests for `strata_forge.cli.doctor`."""

from __future__ import annotations

import socket
from typing import Any

import pytest
from typer.testing import CliRunner

from strata_forge.cli import doctor as doctor_module
from strata_forge.cli.main import app

runner = CliRunner()


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch: pytest.MonkeyPatch) -> None:  # pyright: ignore[reportUnusedFunction]
    """Doctor reads Settings on each invocation — strip env so output is deterministic."""
    for var in (
        "FORGE_PROFILE",
        "FORGE_DIAGNOSTIC_ENABLED",
        "FORGE_DIAGNOSTIC_PATH",
        "FORGE_LOG_LEVEL",
        "FORGE_LOG_FORMAT",
        "FORGE_STORAGE_DEFAULT_BACKEND",
        "LANGFUSE_HOST",
        "LANGFUSE_PUBLIC_KEY",
        "LANGFUSE_SECRET_KEY",
        "REDIS_URL",
        "QDRANT_URL",
        "QDRANT_API_KEY",
    ):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:  # pyright: ignore[reportUnusedFunction]
    """Make every TCP probe fail fast so tests don't depend on the local stack."""

    def _refuse(*_args: Any, **_kwargs: Any) -> socket.socket:
        raise OSError("network disabled for tests")

    monkeypatch.setattr(socket, "create_connection", _refuse)


@pytest.mark.usefixtures("no_network")
class TestDoctorOutput:
    def test_runs_and_exits_clean(self) -> None:
        result = runner.invoke(app, ["doctor"])
        assert result.exit_code == 0

    def test_renders_header(self) -> None:
        result = runner.invoke(app, ["doctor"])
        assert "strata-forge doctor" in result.output
        assert "Python" in result.output

    def test_renders_settings_section(self) -> None:
        result = runner.invoke(app, ["doctor"])
        assert "Settings" in result.output
        assert "profile" in result.output
        assert "logging.level" in result.output
        # Default profile is "dev".
        assert "dev" in result.output

    def test_renders_packages_section(self) -> None:
        result = runner.invoke(app, ["doctor"])
        assert "Tracked packages" in result.output
        # pydantic is a core runtime dep — always installed.
        assert "pydantic" in result.output

    def test_renders_services_section(self) -> None:
        result = runner.invoke(app, ["doctor"])
        assert "Service reachability" in result.output
        assert "Langfuse" in result.output
        assert "Redis" in result.output
        assert "Qdrant" in result.output

    def test_langfuse_marked_unset_when_keys_missing(self) -> None:
        result = runner.invoke(app, ["doctor"])
        assert "keys not set" in result.output

    def test_renders_credentials_section(self) -> None:
        result = runner.invoke(app, ["doctor"])
        assert "Provider credentials" in result.output
        assert "anthropic" in result.output
        assert "ANTHROPIC_API_KEY" in result.output

    def test_credentials_reported_as_unset_when_env_empty(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        for var in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
            monkeypatch.delenv(var, raising=False)
        result = runner.invoke(app, ["doctor"])
        assert "not set" in result.output

    def test_credentials_never_print_values(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-supersecret")
        result = runner.invoke(app, ["doctor"])
        assert "supersecret" not in result.output
        assert "set" in result.output

    def test_reports_unreachable_when_probe_refused(self) -> None:
        result = runner.invoke(app, ["doctor"])
        # With every probe refused, every service prints "unreachable".
        assert "unreachable" in result.output


class TestDoctorWithReachableServices:
    def test_reports_reachable_when_probe_succeeds(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        class _FakeSock:
            def __enter__(self) -> _FakeSock:
                return self

            def __exit__(self, *_args: object) -> None:
                pass

            def close(self) -> None:
                pass

        def _accept(*_args: Any, **_kwargs: Any) -> _FakeSock:
            return _FakeSock()

        monkeypatch.setattr(socket, "create_connection", _accept)
        # Without Langfuse keys, Langfuse stays "keys not set"; Redis and
        # Qdrant probe with their defaults and should report "reachable".
        result = runner.invoke(app, ["doctor"])
        assert result.exit_code == 0
        assert "reachable" in result.output


class TestProbeUrl:
    def test_returns_false_for_bogus_url(self) -> None:
        assert doctor_module._probe_url("not-a-url-at-all") is False  # pyright: ignore[reportPrivateUsage]

    def test_returns_false_when_no_host(self) -> None:
        assert doctor_module._probe_url("http://") is False  # pyright: ignore[reportPrivateUsage]

    def test_uses_default_https_port(self, monkeypatch: pytest.MonkeyPatch) -> None:
        seen: list[tuple[str, int]] = []

        def _accept(addr: tuple[str, int], **_kwargs: Any) -> object:
            seen.append(addr)
            raise OSError("refused")

        monkeypatch.setattr(socket, "create_connection", _accept)
        doctor_module._probe_url("https://example.com")  # pyright: ignore[reportPrivateUsage]
        assert seen == [("example.com", 443)]

    def test_uses_default_http_port(self, monkeypatch: pytest.MonkeyPatch) -> None:
        seen: list[tuple[str, int]] = []

        def _accept(addr: tuple[str, int], **_kwargs: Any) -> object:
            seen.append(addr)
            raise OSError("refused")

        monkeypatch.setattr(socket, "create_connection", _accept)
        doctor_module._probe_url("http://example.com")  # pyright: ignore[reportPrivateUsage]
        assert seen == [("example.com", 80)]

    def test_uses_explicit_port(self, monkeypatch: pytest.MonkeyPatch) -> None:
        seen: list[tuple[str, int]] = []

        def _accept(addr: tuple[str, int], **_kwargs: Any) -> object:
            seen.append(addr)
            raise OSError("refused")

        monkeypatch.setattr(socket, "create_connection", _accept)
        doctor_module._probe_url("https://example.com:9999")  # pyright: ignore[reportPrivateUsage]
        assert seen == [("example.com", 9999)]
