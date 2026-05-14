"""AWS Bedrock provider client.

Currently serves Anthropic-on-Bedrock (Claude). LiteLLM routes through
the ``bedrock/*`` namespace; the model id format is AWS-flavored, e.g.
``bedrock/anthropic.claude-opus-4-7`` for the global cross-region
inference profile.

Bedrock tool calling uses the underlying model's native format — for
Claude-on-Bedrock that means Anthropic's ``input_schema`` shape, so
operators reuse :func:`forge.llm.providers.anthropic.to_anthropic_tool_schema`.
This module therefore exposes only the provider client, not a separate
tool-schema serializer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from forge.llm.providers.base import ProviderClient
from forge.llm.providers.config import BedrockConfig

if TYPE_CHECKING:
    from forge.llm.registry import ProviderName

__all__ = ["BedrockProvider"]


class BedrockProvider(ProviderClient):
    """Provider client for AWS Bedrock."""

    name: ClassVar[ProviderName] = "bedrock"
    litellm_prefix: ClassVar[str] = "bedrock/"

    config: BedrockConfig

    def __init__(self, config: BedrockConfig | None = None) -> None:
        super().__init__(config or BedrockConfig())

    def auth_kwargs(self) -> dict[str, Any]:
        """Return LiteLLM-recognized AWS kwargs.

        LiteLLM accepts ``aws_access_key_id`` / ``aws_secret_access_key`` /
        ``aws_region_name``. Unset keys are dropped so boto3's credential
        chain (instance profile, SSO, ``~/.aws/credentials``, etc.) can
        supply credentials at call time.
        """
        kwargs: dict[str, Any] = {}
        if self.config.access_key_id is not None:
            kwargs["aws_access_key_id"] = self.config.access_key_id.get_secret_value()
        if self.config.secret_access_key is not None:
            kwargs["aws_secret_access_key"] = self.config.secret_access_key.get_secret_value()
        if self.config.region:
            kwargs["aws_region_name"] = self.config.region
        return kwargs
