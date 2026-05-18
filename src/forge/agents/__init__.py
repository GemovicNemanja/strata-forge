"""Agent builder, built-in tools, memory, and multi-agent patterns.

Per ADR 0011, this module is a thin composition layer over
:mod:`forge.llm`; the tool calling primitives — :class:`Tool`,
:func:`tool`, the message types, and the multi-turn tool loop — are
re-exported here for convenience but live in :mod:`forge.llm`.
"""

from forge.agents.agent import Agent, AgentResult
from forge.agents.memory import (
    ConversationMemory,
    EmbedFn,
    EpisodicMemory,
    InMemoryVectorStore,
    VectorItem,
    VectorSearchResult,
    VectorStore,
)
from forge.agents.multi_agent import (
    CritiqueVerdict,
    RouterChoice,
    critic_refiner_run,
    handoff,
)
from forge.agents.tools import (
    CalculatorArgs,
    FetchURLArgs,
    FSReadArgs,
    SearchBackend,
    SearchResult,
    WebSearchArgs,
    calculator,
    fetch_url,
    fs_read_tool,
    web_search_tool,
)

# Re-export the forge.llm primitives that agents compose with. Users
# can build agents without ever importing from forge.llm directly.
from forge.llm.messages import (
    AnyMessage,
    AssistantMessage,
    Message,
    SystemMessage,
    ToolResultMessage,
    UserMessage,
)
from forge.llm.tools import Tool, ToolLoopExceededError, tool

__all__ = [
    "Agent",
    "AgentResult",
    "AnyMessage",
    "AssistantMessage",
    "CalculatorArgs",
    "ConversationMemory",
    "CritiqueVerdict",
    "EmbedFn",
    "EpisodicMemory",
    "FSReadArgs",
    "FetchURLArgs",
    "InMemoryVectorStore",
    "Message",
    "RouterChoice",
    "SearchBackend",
    "SearchResult",
    "SystemMessage",
    "Tool",
    "ToolLoopExceededError",
    "ToolResultMessage",
    "UserMessage",
    "VectorItem",
    "VectorSearchResult",
    "VectorStore",
    "WebSearchArgs",
    "calculator",
    "critic_refiner_run",
    "fetch_url",
    "fs_read_tool",
    "handoff",
    "tool",
    "web_search_tool",
]
