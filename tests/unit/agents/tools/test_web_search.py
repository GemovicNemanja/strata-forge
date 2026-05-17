"""Unit tests for `forge.agents.tools.web_search`."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from forge.agents.tools.web_search import (
    SearchResult,
    WebSearchArgs,
    web_search_tool,
)
from forge.core.errors import ValidationError

if TYPE_CHECKING:
    from collections.abc import Sequence


# ---------------------------------------------------------------------------
# SearchResult shape
# ---------------------------------------------------------------------------


class TestSearchResult:
    def test_required_fields(self) -> None:
        r = SearchResult(title="Hi", url="https://x.test")
        assert r.title == "Hi"
        assert r.url == "https://x.test"
        assert r.snippet == ""

    def test_snippet_default_empty(self) -> None:
        r = SearchResult(title="t", url="u")
        assert r.snippet == ""

    def test_extra_fields_allowed(self) -> None:
        # Backends like Tavily return additional fields; preserve them.
        r = SearchResult(title="t", url="u", snippet="s", score=0.9)  # type: ignore[call-arg]
        dumped = r.model_dump()
        assert dumped["score"] == 0.9


# ---------------------------------------------------------------------------
# Tool factory
# ---------------------------------------------------------------------------


class TestFactory:
    def test_default_name_and_description(self) -> None:
        async def backend(query: str, n: int) -> Sequence[SearchResult]:
            return []

        tool = web_search_tool(backend=backend)
        assert tool.name == "web_search"
        assert "web" in tool.description.lower()

    def test_custom_name(self) -> None:
        async def backend(query: str, n: int) -> Sequence[SearchResult]:
            return []

        tool = web_search_tool(backend=backend, name="my_search")
        assert tool.name == "my_search"

    def test_parameters_model(self) -> None:
        async def backend(query: str, n: int) -> Sequence[SearchResult]:
            return []

        tool = web_search_tool(backend=backend)
        assert tool.parameters_model is WebSearchArgs


# ---------------------------------------------------------------------------
# Behaviour
# ---------------------------------------------------------------------------


class TestBehaviour:
    async def test_backend_called_with_query_and_max_results(self) -> None:
        calls: list[tuple[str, int]] = []

        async def backend(query: str, n: int) -> Sequence[SearchResult]:
            calls.append((query, n))
            return []

        tool = web_search_tool(backend=backend)
        await tool.invoke({"query": "what is x?", "max_results": 7})
        assert calls == [("what is x?", 7)]

    async def test_results_rendered_as_json_array(self) -> None:
        async def backend(query: str, n: int) -> Sequence[SearchResult]:
            return [
                SearchResult(title="One", url="https://1.test", snippet="first"),
                SearchResult(title="Two", url="https://2.test", snippet="second"),
            ]

        tool = web_search_tool(backend=backend)
        raw = await tool.invoke({"query": "anything"})
        parsed = json.loads(raw)
        assert len(parsed) == 2
        assert parsed[0]["title"] == "One"
        assert parsed[0]["url"] == "https://1.test"
        assert parsed[1]["snippet"] == "second"

    async def test_empty_results(self) -> None:
        async def backend(query: str, n: int) -> Sequence[SearchResult]:
            return []

        tool = web_search_tool(backend=backend)
        raw = await tool.invoke({"query": "no hits"})
        assert json.loads(raw) == []

    async def test_extra_fields_preserved_in_output(self) -> None:
        async def backend(query: str, n: int) -> Sequence[SearchResult]:
            return [
                SearchResult(
                    title="t",
                    url="u",
                    snippet="s",
                    score=0.95,  # type: ignore[call-arg]
                    published="2026-01-01",  # type: ignore[call-arg]
                )
            ]

        tool = web_search_tool(backend=backend)
        raw = await tool.invoke({"query": "x"})
        parsed = json.loads(raw)
        assert parsed[0]["score"] == 0.95
        assert parsed[0]["published"] == "2026-01-01"


# ---------------------------------------------------------------------------
# Argument validation
# ---------------------------------------------------------------------------


class TestArgValidation:
    async def test_empty_query_rejected(self) -> None:
        async def backend(query: str, n: int) -> Sequence[SearchResult]:
            return []

        tool = web_search_tool(backend=backend)
        with pytest.raises(ValidationError):
            await tool.invoke({"query": ""})

    async def test_max_results_bounds(self) -> None:
        async def backend(query: str, n: int) -> Sequence[SearchResult]:
            return []

        tool = web_search_tool(backend=backend)
        with pytest.raises(ValidationError):
            await tool.invoke({"query": "x", "max_results": 0})
        with pytest.raises(ValidationError):
            await tool.invoke({"query": "x", "max_results": 50})

    async def test_max_results_defaults_to_five(self) -> None:
        captured: list[int] = []

        async def backend(query: str, n: int) -> Sequence[SearchResult]:
            captured.append(n)
            return []

        tool = web_search_tool(backend=backend)
        await tool.invoke({"query": "x"})
        assert captured == [5]
