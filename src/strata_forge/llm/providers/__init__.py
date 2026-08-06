"""Provider clients and their configuration sub-models.

Each provider module (``openai.py``, ``anthropic.py``, ``vertex.py``,
``bedrock.py``, ``azure.py``, ``openai_compat.py``) ships a concrete
``ProviderClient`` subclass paired with its ``ProviderConfig``. The base
class lives in ``base.py``; the configs live in ``config.py`` and are read
from environment variables.
"""

from strata_forge.llm.providers.anthropic import AnthropicProvider, to_anthropic_tool_schema
from strata_forge.llm.providers.azure import AzureProvider
from strata_forge.llm.providers.base import ProviderClient
from strata_forge.llm.providers.bedrock import BedrockProvider
from strata_forge.llm.providers.config import (
    AnthropicConfig,
    AzureConfig,
    BedrockConfig,
    OpenAICompatConfig,
    OpenAIConfig,
    ProviderConfig,
    VertexConfig,
)
from strata_forge.llm.providers.openai import OpenAIProvider, to_openai_tool_schema
from strata_forge.llm.providers.openai_compat import OpenAICompatProvider
from strata_forge.llm.providers.vertex import VertexProvider, to_gemini_tool_schema

__all__ = [
    "AnthropicConfig",
    "AnthropicProvider",
    "AzureConfig",
    "AzureProvider",
    "BedrockConfig",
    "BedrockProvider",
    "OpenAICompatConfig",
    "OpenAICompatProvider",
    "OpenAIConfig",
    "OpenAIProvider",
    "ProviderClient",
    "ProviderConfig",
    "VertexConfig",
    "VertexProvider",
    "to_anthropic_tool_schema",
    "to_gemini_tool_schema",
    "to_openai_tool_schema",
]
