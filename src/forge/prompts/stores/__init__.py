"""Backend implementations for :class:`forge.prompts.PromptStore`.

Two implementations ship: an in-process :class:`InMemoryPromptStore`
suitable for tests and quick scripts, and a Langfuse-backed store
behind the ``[langfuse]`` extra for versioned production prompts.
"""

from forge.prompts.stores.langfuse import LangfusePromptStore
from forge.prompts.stores.memory import InMemoryPromptStore

__all__ = ["InMemoryPromptStore", "LangfusePromptStore"]
