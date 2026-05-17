"""Built-in tools for :class:`forge.agents.Agent`.

Each tool is a concrete :class:`forge.llm.Tool` built via
:func:`forge.llm.tool` or constructed directly from a Pydantic
arguments model. Some tools (``fs_read``, ``web_search``) need
caller-supplied configuration so they ship as **factory functions**
that return a :class:`Tool`; others (``calculator``, ``fetch_url``)
are ready-to-use module-level instances.
"""

from forge.agents.tools.calculator import CalculatorArgs, calculator
from forge.agents.tools.fetch_url import FetchURLArgs, fetch_url
from forge.agents.tools.fs_read import FSReadArgs, fs_read_tool
from forge.agents.tools.web_search import (
    SearchBackend,
    SearchResult,
    WebSearchArgs,
    web_search_tool,
)

__all__ = [
    "CalculatorArgs",
    "FSReadArgs",
    "FetchURLArgs",
    "SearchBackend",
    "SearchResult",
    "WebSearchArgs",
    "calculator",
    "fetch_url",
    "fs_read_tool",
    "web_search_tool",
]
