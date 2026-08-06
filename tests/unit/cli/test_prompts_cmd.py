"""Unit tests for `strata_forge.cli.prompts`."""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from strata_forge.cli.main import app
from strata_forge.prompts.stores.memory import InMemoryPromptStore
from strata_forge.prompts.template import PromptTemplate

runner = CliRunner()


def _make_template(
    name: str = "summarize",
    description: str = "summarize a passage",
) -> PromptTemplate:
    return PromptTemplate(
        name=name,
        description=description,
        stable_section="You are a careful summarizer.",
        dynamic_section="Summarize: {{ passage }}",
        dynamic_variables=("passage",),
    )


@pytest.fixture
def store_with_one(monkeypatch: pytest.MonkeyPatch) -> InMemoryPromptStore:
    """Replace the helper's store factory with an in-memory one carrying one template."""
    import asyncio

    store = InMemoryPromptStore()
    asyncio.run(store.put(_make_template()))
    monkeypatch.setattr("strata_forge.cli.prompts.prompt_store_from_settings", lambda: store)
    return store


@pytest.fixture
def empty_store(monkeypatch: pytest.MonkeyPatch) -> InMemoryPromptStore:
    store = InMemoryPromptStore()
    monkeypatch.setattr("strata_forge.cli.prompts.prompt_store_from_settings", lambda: store)
    return store


class TestPromptsList:
    def test_lists_registered_templates(self, store_with_one: InMemoryPromptStore) -> None:
        result = runner.invoke(app, ["prompts", "list"])
        assert result.exit_code == 0
        assert "summarize" in result.output

    def test_empty_store_message(self, empty_store: InMemoryPromptStore) -> None:
        result = runner.invoke(app, ["prompts", "list"])
        assert result.exit_code == 0
        assert "no prompts registered" in result.output


class TestPromptsShow:
    def test_show_template(self, store_with_one: InMemoryPromptStore) -> None:
        result = runner.invoke(app, ["prompts", "show", "summarize"])
        assert result.exit_code == 0
        assert "summarize" in result.output
        assert "You are a careful summarizer." in result.output
        assert "Summarize:" in result.output
        # dynamic_variables list rendered.
        assert "passage" in result.output

    def test_show_template_with_stable_variables(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Cover the stable_variables rendering branch in prompts.show.
        import asyncio

        store = InMemoryPromptStore()
        template = PromptTemplate(
            name="parametric-system",
            stable_section="You are a {{ persona }}.",
            dynamic_section="Answer: {{ question }}",
            stable_variables=("persona",),
            dynamic_variables=("question",),
        )
        asyncio.run(store.put(template))
        monkeypatch.setattr("strata_forge.cli.prompts.prompt_store_from_settings", lambda: store)
        result = runner.invoke(app, ["prompts", "show", "parametric-system"])
        assert result.exit_code == 0
        assert "stable_variables" in result.output
        assert "persona" in result.output

    def test_show_missing_template_exits_nonzero(self, empty_store: InMemoryPromptStore) -> None:
        result = runner.invoke(app, ["prompts", "show", "no-such-thing"])
        assert result.exit_code != 0
        # error_exit prints to stderr; rich captures it in result.output.
        assert "error" in result.output.lower() or "not found" in result.output.lower()


class TestPromptsRender:
    def test_render_with_vars(self, store_with_one: InMemoryPromptStore) -> None:
        result = runner.invoke(
            app,
            [
                "prompts",
                "render",
                "summarize",
                "--vars",
                '{"passage": "Forge is typed."}',
            ],
        )
        assert result.exit_code == 0
        assert "Forge is typed." in result.output

    def test_render_rejects_invalid_json(self, store_with_one: InMemoryPromptStore) -> None:
        result = runner.invoke(app, ["prompts", "render", "summarize", "--vars", "not-json"])
        assert result.exit_code != 0

    def test_render_rejects_non_object_vars(self, store_with_one: InMemoryPromptStore) -> None:
        result = runner.invoke(app, ["prompts", "render", "summarize", "--vars", "[1,2,3]"])
        assert result.exit_code != 0

    def test_render_missing_template(self, empty_store: InMemoryPromptStore) -> None:
        result = runner.invoke(app, ["prompts", "render", "missing", "--vars", "{}"])
        assert result.exit_code != 0
