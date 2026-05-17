"""Web-search tool — backend-agnostic factory.

``web_search_tool`` doesn't ship a concrete search provider. Forge
isn't opinionated about which search API you use (Tavily, SerpAPI,
DuckDuckGo, your internal index); callers wire one in by supplying
an async backend that takes a query + max-result count and returns a
list of :class:`SearchResult` instances.

Typical use::

    from forge.agents.tools import SearchResult, web_search_tool

    async def my_tavily_backend(query: str, n: int) -> list[SearchResult]:
        # ... call Tavily / SerpAPI / etc. ...
        return [SearchResult(title=..., url=..., snippet=...) for ...]

    search = web_search_tool(backend=my_tavily_backend)
    agent = Agent("researcher", client=client, tools=[search])

The Tool's output is a JSON array of result dicts — easy for the
LLM to parse and reference in follow-up tool calls.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

from forge.llm.tools import Tool

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence

__all__ = [
    "SearchBackend",
    "SearchResult",
    "WebSearchArgs",
    "web_search_tool",
]


class SearchResult(BaseModel):
    """One web-search hit.

    Extra fields are allowed so adapters can pass through provider-
    specific metadata (score, published_at, …) without subclassing.
    """

    model_config = ConfigDict(extra="allow")

    title: str
    url: str
    snippet: str = ""


type SearchBackend = Callable[[str, int], Awaitable[Sequence[SearchResult]]]
"""Async function: ``(query, max_results) -> Sequence[SearchResult]``."""


class WebSearchArgs(BaseModel):
    """Arguments for the :data:`web_search` tool."""

    query: str = Field(description="The search query.", min_length=1)
    max_results: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum number of results to return. Default 5, max 20.",
    )


def web_search_tool(
    *,
    backend: SearchBackend,
    name: str = "web_search",
    description: str = (
        "Search the web for information relevant to a query. "
        "Returns a JSON array of results with title, url, and snippet."
    ),
) -> Tool:
    """Build a :class:`Tool` that calls ``backend`` to perform a search.

    The backend takes ``(query, max_results)`` and returns a sequence
    of :class:`SearchResult` instances. The tool renders the results
    as a JSON array so the LLM can read field values directly.

    Args:
        backend: Async search function. See :data:`SearchBackend`.
        name: Tool name surfaced to the LLM.
        description: Tool description surfaced to the LLM.
    """

    async def _search(args: WebSearchArgs) -> str:
        results = await backend(args.query, args.max_results)
        return json.dumps([r.model_dump() for r in results], indent=2)

    return Tool(
        name=name,
        description=description,
        parameters_model=WebSearchArgs,
        fn=_search,  # type: ignore[arg-type]
    )
