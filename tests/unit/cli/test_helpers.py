"""Unit tests for `forge.cli.helpers`."""

from __future__ import annotations

import pytest
import typer

from forge.cli.helpers import (
    dataset_store_from_settings,
    error_exit,
    prompt_store_from_settings,
    run_async,
)


class TestRunAsync:
    def test_returns_awaitable_result(self) -> None:
        async def _value() -> int:
            return 42

        assert run_async(_value()) == 42

    def test_propagates_exception(self) -> None:
        async def _fails() -> int:
            err = "boom"
            raise RuntimeError(err)

        with pytest.raises(RuntimeError, match="boom"):
            run_async(_fails())


class TestErrorExit:
    def test_raises_typer_exit_with_code(self) -> None:
        with pytest.raises(typer.Exit) as exc_info:
            error_exit("bad config", code=2)
        assert exc_info.value.exit_code == 2

    def test_default_exit_code_is_1(self) -> None:
        with pytest.raises(typer.Exit) as exc_info:
            error_exit("bad")
        assert exc_info.value.exit_code == 1


class TestStoreFactories:
    def test_prompt_store_falls_back_to_memory(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from forge.config.settings import reset_settings
        from forge.prompts.stores.memory import InMemoryPromptStore

        # Clear Langfuse env so settings.langfuse.enabled is False.
        for k in (
            "LANGFUSE_PUBLIC_KEY",
            "LANGFUSE_SECRET_KEY",
            "FORGE_LANGFUSE_PUBLIC_KEY",
            "FORGE_LANGFUSE_SECRET_KEY",
        ):
            monkeypatch.delenv(k, raising=False)
        reset_settings()
        store = prompt_store_from_settings()
        assert isinstance(store, InMemoryPromptStore)

    def test_dataset_store_falls_back_to_memory(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from forge.config.settings import reset_settings
        from forge.datasets.stores.memory import InMemoryDatasetStore

        for k in (
            "LANGFUSE_PUBLIC_KEY",
            "LANGFUSE_SECRET_KEY",
            "FORGE_LANGFUSE_PUBLIC_KEY",
            "FORGE_LANGFUSE_SECRET_KEY",
        ):
            monkeypatch.delenv(k, raising=False)
        reset_settings()
        store = dataset_store_from_settings()
        assert isinstance(store, InMemoryDatasetStore)


class TestLangfuseBranch:
    """When LANGFUSE_* env vars are set, the factories should pick Langfuse."""

    def test_prompt_store_picks_langfuse(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from forge.config.settings import reset_settings

        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk")
        reset_settings()

        sentinel = object()

        class _FakeLangfuseStore:
            def __new__(cls) -> object:  # type: ignore[misc]
                return sentinel

        import forge.prompts.stores.langfuse as lf_mod

        monkeypatch.setattr(lf_mod, "LangfusePromptStore", _FakeLangfuseStore)
        result = prompt_store_from_settings()
        assert result is sentinel

    def test_dataset_store_picks_langfuse(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from forge.config.settings import reset_settings

        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk")
        reset_settings()

        sentinel = object()

        class _FakeLangfuseDataset:
            def __new__(cls) -> object:  # type: ignore[misc]
                return sentinel

        import forge.datasets.stores.langfuse as lf_mod

        monkeypatch.setattr(lf_mod, "LangfuseDatasetStore", _FakeLangfuseDataset)
        result = dataset_store_from_settings()
        assert result is sentinel

    def test_prompt_store_missing_extra_errors(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import sys

        from forge.config.settings import reset_settings

        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk")
        reset_settings()

        # Drop the langfuse store module so the lazy import raises.
        monkeypatch.setitem(sys.modules, "forge.prompts.stores.langfuse", None)

        with pytest.raises(typer.Exit):
            prompt_store_from_settings()

    def test_dataset_store_missing_extra_errors(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import sys

        from forge.config.settings import reset_settings

        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk")
        reset_settings()

        monkeypatch.setitem(sys.modules, "forge.datasets.stores.langfuse", None)

        with pytest.raises(typer.Exit):
            dataset_store_from_settings()
