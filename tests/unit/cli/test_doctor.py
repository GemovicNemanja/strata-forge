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
        assert "configured" in result.output

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

    def test_report_keeps_diagnostic_qualification_explicit(self) -> None:
        result = runner.invoke(app, ["doctor"])
        assert "provider authentication" in result.output
        assert "execution readiness" in result.output
        assert "checks land" not in result.output

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
    @pytest.mark.parametrize(
        "url",
        ["https://example.invalid:not-a-port", "https://example.invalid:99999", "http://[bad"],
    )
    def test_malformed_port_or_host_returns_false_without_probing(
        self, url: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[bool] = []

        def forbidden(*_args: object, **_kwargs: object) -> None:
            calls.append(True)
            raise AssertionError("malformed URLs must not reach the socket")

        monkeypatch.setattr(socket, "create_connection", forbidden)
        assert doctor_module._probe_url(url) is False  # pyright: ignore[reportPrivateUsage]
        assert calls == []

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


@pytest.mark.parametrize("flag", ["--offline", "--presence-only"])
def test_presence_aliases_never_invoke_a_probe(flag: str, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[bool] = []
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "fictional-public")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "fictional-secret")

    def forbidden(*_args: object, **_kwargs: object) -> bool:
        calls.append(True)
        raise AssertionError("presence-only doctor must not probe")

    monkeypatch.setattr(doctor_module, "_probe_url", forbidden)
    result = runner.invoke(app, ["doctor", flag])
    assert result.exit_code == 0
    assert calls == []
    assert "not probed" in result.output
    assert "configured" in result.output


def test_default_mode_preserves_explicit_reachability_checks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "fictional-public")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "fictional-secret")

    def reachable(url: str) -> bool:
        calls.append(url)
        return True

    monkeypatch.setattr(doctor_module, "_probe_url", reachable)
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    assert len(calls) == 3
    assert "Service reachability" in result.output
    assert "reachable" in result.output


@pytest.mark.parametrize("flags", [[], ["--offline"]])
def test_all_modes_omit_private_configuration_values(
    flags: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    markers = [
        "fictional-private-profile",
        "fictional-user",
        "fictional-password",
        "fictional-host.invalid",
        "fictional-path",
        "fictional-query-secret",
        "fictional-diagnostic-directory",
    ]
    url = (
        "https://fictional-user:fictional-password@fictional-host.invalid:8443/"
        "fictional-path?token=fictional-query-secret"
    )
    monkeypatch.setenv("FORGE_PROFILE", markers[0])
    monkeypatch.setenv("FORGE_DIAGNOSTIC_PATH", "/fictional-diagnostic-directory/output.jsonl")
    monkeypatch.setenv("LANGFUSE_HOST", url)
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "fictional-public")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "fictional-secret")
    monkeypatch.setenv("REDIS_URL", url)
    monkeypatch.setenv("QDRANT_URL", url)
    monkeypatch.setattr(doctor_module, "_probe_url", lambda url: False)
    result = runner.invoke(app, ["doctor", *flags])
    assert result.exit_code == 0
    for marker in markers:
        assert marker not in result.output
    assert "Settings" in result.output
    assert "Langfuse" in result.output


def test_raw_snapshot_names_paths_and_local_version_suffixes_are_not_rendered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = {
        "python": "3.14.5+fictional-python-owner",
        "system": "fictional-system-owner",
        "platform": "fictional-host-owner",
        "cwd": "/fictional/source-owner",
        "pkg.pydantic": "2.12.1+fictional-package-owner",
        "pkg.fictional-extra-owner": "fictional-extra-content",
    }
    monkeypatch.setattr(doctor_module, "env_snapshot", lambda: snapshot)
    result = runner.invoke(app, ["doctor", "--offline"])
    assert result.exit_code == 0
    assert "local build" in result.output
    assert "fictional" not in result.output


def test_bad_configuration_is_a_fixed_diagnostic_and_never_probes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[bool] = []

    def broken() -> None:
        raise ValueError("fictional-private-config-content")

    def forbidden(*_args: object, **_kwargs: object) -> bool:
        calls.append(True)
        raise AssertionError("bad configuration must not probe")

    monkeypatch.setattr(doctor_module, "Settings", broken)
    monkeypatch.setattr(doctor_module, "_probe_url", forbidden)
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    assert calls == []
    assert "Settings unavailable" in result.output
    assert "fictional-private" not in result.output


def test_metadata_failure_is_unknown_rather_than_not_installed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def broken() -> None:
        raise ValueError("fictional-private-metadata-content")

    monkeypatch.setattr(doctor_module, "env_snapshot", broken)
    result = runner.invoke(app, ["doctor", "--offline"])
    assert result.exit_code == 0
    assert "Tracked packages unavailable" in result.output
    assert "not installed" not in result.output
    assert "fictional-private" not in result.output


def test_presence_mode_is_documented_in_help() -> None:
    result = runner.invoke(app, ["doctor", "--help"])
    assert result.exit_code == 0
    assert "--offline" in result.output
    assert "--presence-only" in result.output
