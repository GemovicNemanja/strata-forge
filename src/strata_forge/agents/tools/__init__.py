"""Built-in tools for :class:`strata_forge.agents.Agent`.

Each tool is a concrete :class:`strata_forge.llm.Tool` built via
:func:`strata_forge.llm.tool` or constructed directly from a Pydantic
arguments model. Some tools (``fs_read``, ``web_search``) need
caller-supplied configuration so they ship as **factory functions**
that return a :class:`Tool`; others (``calculator``, ``fetch_url``)
are ready-to-use module-level instances.
"""

from strata_forge.agents.tools.calculator import CalculatorArgs, calculator
from strata_forge.agents.tools.fetch_url import FetchURLArgs, fetch_url
from strata_forge.agents.tools.fs_read import FSReadArgs, fs_read_tool
from strata_forge.agents.tools.web_search import (
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
