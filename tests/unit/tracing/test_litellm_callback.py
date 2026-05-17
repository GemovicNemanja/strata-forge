"""Unit tests for `forge.tracing.litellm_callback`."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import litellm

from forge.tracing.litellm_callback import (
    LITELLM_CALLBACK_NAME,
    install_litellm_callback,
    is_litellm_callback_installed,
)

if TYPE_CHECKING:
    import pytest


def _configure_langfuse(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LANGFUSE_HOST", "http://test")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")


def _reset_litellm_callbacks(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each test starts with empty callback lists."""
    monkeypatch.setattr(litellm, "success_callback", [], raising=False)
    monkeypatch.setattr(litellm, "failure_callback", [], raising=False)


# ---------------------------------------------------------------------------
# install_litellm_callback
# ---------------------------------------------------------------------------


class TestInstallUnconfigured:
    def test_noop_when_langfuse_not_configured(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _reset_litellm_callbacks(monkeypatch)
        monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
        monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)

        result = install_litellm_callback()

        assert result is False
        assert LITELLM_CALLBACK_NAME not in litellm.success_callback
        assert LITELLM_CALLBACK_NAME not in litellm.failure_callback

    def test_noop_when_only_public_key_set(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _reset_litellm_callbacks(monkeypatch)
        monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-only")

        assert install_litellm_callback() is False
        assert LITELLM_CALLBACK_NAME not in litellm.success_callback


class TestInstallConfigured:
    def test_appends_to_both_lists(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _reset_litellm_callbacks(monkeypatch)
        _configure_langfuse(monkeypatch)

        result = install_litellm_callback()

        assert result is True
        assert litellm.success_callback == [LITELLM_CALLBACK_NAME]
        assert litellm.failure_callback == [LITELLM_CALLBACK_NAME]

    def test_idempotent_no_duplication(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _reset_litellm_callbacks(monkeypatch)
        _configure_langfuse(monkeypatch)

        install_litellm_callback()
        install_litellm_callback()
        install_litellm_callback()

        # Exactly one entry each — no duplication regardless of call count.
        assert litellm.success_callback.count(LITELLM_CALLBACK_NAME) == 1
        assert litellm.failure_callback.count(LITELLM_CALLBACK_NAME) == 1

    def test_preserves_existing_callbacks(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Don't trample other callbacks the application has registered.
        monkeypatch.setattr(litellm, "success_callback", ["custom_a", "custom_b"], raising=False)
        monkeypatch.setattr(litellm, "failure_callback", ["custom_c"], raising=False)
        _configure_langfuse(monkeypatch)

        install_litellm_callback()

        assert litellm.success_callback == ["custom_a", "custom_b", LITELLM_CALLBACK_NAME]
        assert litellm.failure_callback == ["custom_c", LITELLM_CALLBACK_NAME]

    def test_handles_none_callback_attribute(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Defensive: if LiteLLM ever sets these attributes to None (older
        # versions did) we still register cleanly.
        monkeypatch.setattr(litellm, "success_callback", None, raising=False)
        monkeypatch.setattr(litellm, "failure_callback", None, raising=False)
        _configure_langfuse(monkeypatch)

        install_litellm_callback()

        assert litellm.success_callback == [LITELLM_CALLBACK_NAME]
        assert litellm.failure_callback == [LITELLM_CALLBACK_NAME]


# ---------------------------------------------------------------------------
# is_litellm_callback_installed
# ---------------------------------------------------------------------------


class TestIsInstalled:
    def test_false_when_both_lists_empty(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _reset_litellm_callbacks(monkeypatch)
        assert is_litellm_callback_installed() is False

    def test_true_after_install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _reset_litellm_callbacks(monkeypatch)
        _configure_langfuse(monkeypatch)
        install_litellm_callback()
        assert is_litellm_callback_installed() is True

    def test_false_if_only_success_registered(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Half-installed = not installed. We deliberately don't report
        # True for the asymmetric case so the diagnostic isn't misleading.
        monkeypatch.setattr(litellm, "success_callback", [LITELLM_CALLBACK_NAME], raising=False)
        monkeypatch.setattr(litellm, "failure_callback", [], raising=False)
        assert is_litellm_callback_installed() is False

    def test_false_if_only_failure_registered(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(litellm, "success_callback", [], raising=False)
        monkeypatch.setattr(litellm, "failure_callback", [LITELLM_CALLBACK_NAME], raising=False)
        assert is_litellm_callback_installed() is False

    def test_true_when_other_callbacks_also_present(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            litellm,
            "success_callback",
            ["other", LITELLM_CALLBACK_NAME, "another"],
            raising=False,
        )
        monkeypatch.setattr(
            litellm,
            "failure_callback",
            [LITELLM_CALLBACK_NAME, "x"],
            raising=False,
        )
        assert is_litellm_callback_installed() is True

    def test_handles_none_attribute(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(litellm, "success_callback", None, raising=False)
        monkeypatch.setattr(litellm, "failure_callback", None, raising=False)
        assert is_litellm_callback_installed() is False


# ---------------------------------------------------------------------------
# Module surface
# ---------------------------------------------------------------------------


class TestModuleSurface:
    def test_callback_name_is_langfuse(self) -> None:
        # If LiteLLM ever renames its callback this constant moves —
        # but tests pin the current value so the rename surfaces here
        # rather than silently breaking traces in production.
        assert LITELLM_CALLBACK_NAME == "langfuse"

    def test_install_re_export(self) -> None:
        from forge.tracing import (
            install_litellm_callback as exported_install,
        )
        from forge.tracing import (
            is_litellm_callback_installed as exported_check,
        )

        assert exported_install is install_litellm_callback
        assert exported_check is is_litellm_callback_installed


_: Any = None  # silence unused-import warnings for the TYPE_CHECKING block
