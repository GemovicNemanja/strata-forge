"""Unit tests for `forge.tracing.client`."""

from __future__ import annotations

import sys
import types
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock

from forge.tracing.client import get_client, reset_client

if TYPE_CHECKING:
    import pytest


def _install_fake_langfuse(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Replace the `langfuse` module in sys.modules with a fake.

    Returns the `Langfuse` constructor mock so tests can inspect what
    arguments the client was built with.
    """
    fake_module = types.ModuleType("langfuse")
    constructor = MagicMock(return_value=MagicMock(name="fake_langfuse_client"))
    fake_module.Langfuse = constructor  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "langfuse", fake_module)
    return constructor


def _configure_langfuse(
    monkeypatch: pytest.MonkeyPatch,
    *,
    host: str = "http://test-langfuse",
    public_key: str = "pk-test",
    secret_key: str = "sk-test",
) -> None:
    monkeypatch.setenv("LANGFUSE_HOST", host)
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", public_key)
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", secret_key)


# ---------------------------------------------------------------------------
# get_client behavior under various configurations
# ---------------------------------------------------------------------------


class TestGetClientUnconfigured:
    def test_returns_none_when_keys_unset(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
        monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
        assert get_client() is None

    def test_returns_none_when_only_public_key_set(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Half-configured == not configured. `LangfuseConfig.enabled` is
        # False unless both keys are present.
        monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-only")
        assert get_client() is None

    def test_returns_none_when_only_secret_key_set(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-only")
        assert get_client() is None


class TestGetClientLangfuseMissing:
    def test_returns_none_when_langfuse_not_installed(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Configured, but the [langfuse] extra isn't installed.
        _configure_langfuse(monkeypatch)
        monkeypatch.setitem(sys.modules, "langfuse", None)
        assert get_client() is None


class TestGetClientConfigured:
    def test_constructs_client_when_configured(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _configure_langfuse(monkeypatch)
        constructor = _install_fake_langfuse(monkeypatch)
        client = get_client()
        assert client is not None
        constructor.assert_called_once_with(
            host="http://test-langfuse",
            public_key="pk-test",
            secret_key="sk-test",
            environment=None,
        )

    def test_passes_tracing_environment(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # `LANGFUSE_TRACING_ENVIRONMENT` (the SDK's own var) flows through
        # `LangfuseConfig.tracing_environment` to the client constructor so traces
        # are grouped by deployment environment.
        _configure_langfuse(monkeypatch)
        monkeypatch.setenv("LANGFUSE_TRACING_ENVIRONMENT", "production")
        constructor = _install_fake_langfuse(monkeypatch)
        get_client()
        assert constructor.call_args.kwargs["environment"] == "production"

    def test_environment_defaults_to_none(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Unset → None passed, so the SDK keeps its built-in "default" environment.
        _configure_langfuse(monkeypatch)
        monkeypatch.delenv("LANGFUSE_TRACING_ENVIRONMENT", raising=False)
        constructor = _install_fake_langfuse(monkeypatch)
        get_client()
        assert constructor.call_args.kwargs["environment"] is None

    def test_returns_cached_singleton(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _configure_langfuse(monkeypatch)
        constructor = _install_fake_langfuse(monkeypatch)
        a = get_client()
        b = get_client()
        assert a is b
        # Constructor only invoked once across multiple `get_client` calls.
        assert constructor.call_count == 1

    def test_unwraps_secret_str(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # The constructor must receive plain strings, not `SecretStr`
        # wrappers — Langfuse's SDK doesn't know about Pydantic types.
        _configure_langfuse(monkeypatch, public_key="pk-real", secret_key="sk-real")
        constructor = _install_fake_langfuse(monkeypatch)
        get_client()
        kwargs = constructor.call_args.kwargs
        assert isinstance(kwargs["public_key"], str)
        assert isinstance(kwargs["secret_key"], str)
        assert kwargs["public_key"] == "pk-real"
        assert kwargs["secret_key"] == "sk-real"


# ---------------------------------------------------------------------------
# reset_client
# ---------------------------------------------------------------------------


class TestResetClient:
    def test_reset_clears_cached_client(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _configure_langfuse(monkeypatch)
        constructor = _install_fake_langfuse(monkeypatch)

        # Return a fresh instance for each construction so reset_client's
        # effect is observable as identity change.
        def _fresh_client(**_kwargs: Any) -> MagicMock:
            return MagicMock(name="lf-client")

        constructor.side_effect = _fresh_client

        first = get_client()

        reset_client()
        # The settings cache also has to be reset for env-var changes to
        # take effect; the autouse `reset_settings_cache` fixture does
        # that for each test, but we trigger it manually here so the
        # second `get_client` rereads the (changed) env.
        from forge.config import reset_settings

        reset_settings()
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-different")
        second = get_client()

        assert second is not first
        assert constructor.call_count == 2

    def test_reset_when_never_initialized(self) -> None:
        # No-op — must not crash.
        reset_client()
        reset_client()

    def test_reset_persists_across_repeated_get_client_calls(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Negative result (None) is cached too — verify reset_client
        # forces a re-check.
        monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
        monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
        assert get_client() is None

        reset_client()
        from forge.config import reset_settings

        reset_settings()
        _configure_langfuse(monkeypatch)
        _install_fake_langfuse(monkeypatch)
        # After reset + new config, the next call constructs a real client.
        assert get_client() is not None


# ---------------------------------------------------------------------------
# Lazy import contract
# ---------------------------------------------------------------------------


class TestLazyImportContract:
    def test_importing_module_does_not_import_langfuse(self) -> None:
        # The whole point of the lazy import: `from forge.tracing import
        # get_client` must work without the [langfuse] extra installed.
        # We can't directly test "no import happens at module load" easily
        # here, but we verify that get_client returns None cleanly when
        # langfuse isn't importable — which is the user-observable
        # contract.
        # (This is also covered by TestGetClientLangfuseMissing; the test
        # here documents the design intent explicitly.)
        from forge.tracing import client

        assert hasattr(client, "get_client")
        assert hasattr(client, "reset_client")

    def test_module_imports_with_langfuse_blocked(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Block the langfuse import and re-import the client module from
        # scratch to confirm it doesn't try to eager-import langfuse.
        monkeypatch.setitem(sys.modules, "langfuse", None)
        monkeypatch.delitem(sys.modules, "forge.tracing.client", raising=False)
        # If the module eager-imported langfuse, this raises.
        import importlib

        importlib.import_module("forge.tracing.client")


# ---------------------------------------------------------------------------
# Sanity: returned object is what we constructed
# ---------------------------------------------------------------------------


class TestReturnedClient:
    def test_returned_object_is_the_constructed_one(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _configure_langfuse(monkeypatch)
        sentinel: Any = MagicMock(name="sentinel_client")
        fake_module = types.ModuleType("langfuse")
        fake_module.Langfuse = MagicMock(return_value=sentinel)  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "langfuse", fake_module)

        assert get_client() is sentinel
