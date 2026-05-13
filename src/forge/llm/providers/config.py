"""Pydantic Settings sub-models for LLM provider credentials.

Each concrete config reads its values from env vars with a provider-specific
prefix or alias scheme. ``ProviderClient`` subclasses consume their matching
config to assemble the LiteLLM auth kwargs.

These models live in ``forge.llm.providers.config`` (not in ``forge.config``)
because they're an LLM-module concern; the dependency direction is
``llm`` → ``config`` only, never the reverse.
"""

from __future__ import annotations

from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

__all__ = [
    "AnthropicConfig",
    "AzureConfig",
    "BedrockConfig",
    "OpenAICompatConfig",
    "OpenAIConfig",
    "ProviderConfig",
    "VertexConfig",
]


class ProviderConfig(BaseSettings):
    """Common base for provider configs.

    The shared ``model_config`` here is overridden by subclasses to set the
    right ``env_prefix`` / ``case_sensitive`` / etc.; everything else
    (extras policy, env-file fallback) inherits unchanged.
    """

    model_config = SettingsConfigDict(extra="ignore", case_sensitive=False)


class OpenAIConfig(ProviderConfig):
    """OpenAI native API credentials.

    Reads ``OPENAI_API_KEY`` and the optional ``OPENAI_ORG_ID``. Configures
    LiteLLM's ``openai/*`` namespace.
    """

    model_config = SettingsConfigDict(env_prefix="OPENAI_", extra="ignore")

    api_key: SecretStr | None = None
    org_id: str | None = None

    @property
    def enabled(self) -> bool:
        return self.api_key is not None


class AnthropicConfig(ProviderConfig):
    """Anthropic native API credentials.

    Reads ``ANTHROPIC_API_KEY``. Configures LiteLLM's ``anthropic/*``
    namespace.
    """

    model_config = SettingsConfigDict(env_prefix="ANTHROPIC_", extra="ignore")

    api_key: SecretStr | None = None

    @property
    def enabled(self) -> bool:
        return self.api_key is not None


class VertexConfig(ProviderConfig):
    """GCP Vertex AI credentials — used by both Gemini and Anthropic-on-Vertex.

    Reads ``GOOGLE_APPLICATION_CREDENTIALS`` (path to service-account JSON),
    ``GCP_PROJECT``, and ``GCP_REGION``.
    """

    model_config = SettingsConfigDict(extra="ignore", case_sensitive=False)

    application_credentials: str | None = Field(
        default=None,
        validation_alias="GOOGLE_APPLICATION_CREDENTIALS",
    )
    project: str | None = Field(default=None, validation_alias="GCP_PROJECT")
    region: str = Field(default="us-central1", validation_alias="GCP_REGION")

    @property
    def enabled(self) -> bool:
        return self.application_credentials is not None and self.project is not None


class BedrockConfig(ProviderConfig):
    """AWS Bedrock credentials — used for Anthropic-on-Bedrock.

    Reads standard AWS env vars (``AWS_ACCESS_KEY_ID``,
    ``AWS_SECRET_ACCESS_KEY``, ``AWS_REGION``). When ``access_key_id`` and
    ``secret_access_key`` are both ``None``, ``enabled`` is still ``True``
    if a region is set — boto3's credential chain (instance profile, SSO,
    config file, ...) may supply creds at call time.
    """

    model_config = SettingsConfigDict(extra="ignore", case_sensitive=False)

    access_key_id: SecretStr | None = Field(
        default=None,
        validation_alias="AWS_ACCESS_KEY_ID",
    )
    secret_access_key: SecretStr | None = Field(
        default=None,
        validation_alias="AWS_SECRET_ACCESS_KEY",
    )
    region: str = Field(default="us-east-1", validation_alias="AWS_REGION")

    @property
    def enabled(self) -> bool:
        # Treat the AWS credential chain as opt-in: any of (explicit keys,
        # region set) means the operator has wired Bedrock up.
        return self.access_key_id is not None or self.region != ""


class AzureConfig(ProviderConfig):
    """Azure OpenAI credentials.

    Reads ``AZURE_OPENAI_API_KEY``, ``AZURE_OPENAI_ENDPOINT``,
    ``AZURE_OPENAI_API_VERSION``.
    """

    model_config = SettingsConfigDict(env_prefix="AZURE_OPENAI_", extra="ignore")

    api_key: SecretStr | None = None
    endpoint: str | None = None
    api_version: str = "2025-10-01-preview"

    @property
    def enabled(self) -> bool:
        return self.api_key is not None and self.endpoint is not None


class OpenAICompatConfig(ProviderConfig):
    """OpenAI-compatible self-hosted servers (vLLM / TGI / SGLang).

    Reads ``FORGE_OPENAI_COMPAT_BASE_URL`` and the optional
    ``FORGE_OPENAI_COMPAT_API_KEY``. The wiring through LiteLLM lands when
    ``forge.compute`` adds inference serving; this config exists now so the
    surface stays uniform.
    """

    model_config = SettingsConfigDict(
        env_prefix="FORGE_OPENAI_COMPAT_",
        extra="ignore",
    )

    base_url: str | None = None
    api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "api_key",
            "FORGE_OPENAI_COMPAT_API_KEY",
        ),
    )

    @property
    def enabled(self) -> bool:
        return self.base_url is not None
