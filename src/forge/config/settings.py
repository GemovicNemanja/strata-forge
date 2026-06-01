"""Pydantic Settings root for strata-forge runtime configuration.

``Settings`` is the single source of truth for runtime configuration. Every
other module reads it through :func:`get_settings` rather than touching
``os.environ`` directly. Sub-models partition the configuration by concern;
each sub-model is a ``BaseSettings`` in its own right so it can read its env
vars with its own prefix.

Loading precedence (lowest → highest):

1. Field defaults declared below
2. ``configs/<FORGE_PROFILE>.yaml`` overlay (applied by ``forge.config.overlays``)
3. ``.env`` file in repo root
4. Process environment variables
5. In-code overrides passed to ``Settings(...)``
"""

from __future__ import annotations

import functools
from typing import Literal

from pydantic import AliasChoices, BaseModel, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

__all__ = [
    "DiagnosticConfig",
    "HuggingFaceConfig",
    "LangfuseConfig",
    "LoggingConfig",
    "ProvidersConfig",
    "QdrantConfig",
    "RedisConfig",
    "Settings",
    "StorageConfig",
    "get_settings",
    "reset_settings",
]


# ---------------------------------------------------------------------------
# Sub-models — one per concern, each loading its own env vars.
# ---------------------------------------------------------------------------

# Every sub-config is its own BaseSettings instance, so each one needs
# env_file=".env" to pick up dotenv values; the root Settings's env_file
# only governs the root's own fields.
_ENV_FILE = ".env"


class LangfuseConfig(BaseSettings):
    """Langfuse observability backend. Tracing is configured here."""

    model_config = SettingsConfigDict(env_prefix="LANGFUSE_", env_file=_ENV_FILE, extra="ignore")

    host: str = "http://localhost:3000"
    public_key: SecretStr | None = None
    secret_key: SecretStr | None = None

    @property
    def enabled(self) -> bool:
        """Tracing is enabled iff both keys are configured."""
        return self.public_key is not None and self.secret_key is not None


class RedisConfig(BaseSettings):
    """Redis cache backend. Opt-in via ``[redis]`` extra."""

    model_config = SettingsConfigDict(env_prefix="REDIS_", env_file=_ENV_FILE, extra="ignore")

    url: str = "redis://localhost:6379/0"


class QdrantConfig(BaseSettings):
    """Qdrant vector store. Used by the RAG module."""

    model_config = SettingsConfigDict(env_prefix="QDRANT_", env_file=_ENV_FILE, extra="ignore")

    url: str = "http://localhost:6333"
    api_key: SecretStr | None = None


class HuggingFaceConfig(BaseSettings):
    """Hugging Face Hub credentials.

    Field names match the env vars the ``huggingface_hub`` library
    consults natively, so settings and the SDK's own fallback agree on
    the same source.
    """

    model_config = SettingsConfigDict(env_prefix="HF_", env_file=_ENV_FILE, extra="ignore")

    token: SecretStr | None = None
    endpoint: str | None = None


class StorageConfig(BaseSettings):
    """fsspec gateway defaults."""

    model_config = SettingsConfigDict(
        env_prefix="FORGE_STORAGE_", env_file=_ENV_FILE, extra="ignore"
    )

    default_backend: Literal["local", "s3", "gcs", "azure", "hf"] = "local"


class LoggingConfig(BaseSettings):
    """structlog rendering and verbosity."""

    model_config = SettingsConfigDict(env_prefix="FORGE_LOG_", env_file=_ENV_FILE, extra="ignore")

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    format: Literal["auto", "pretty", "json"] = "auto"


class DiagnosticConfig(BaseSettings):
    """NDJSON diagnostic dump of every completed LLM call."""

    model_config = SettingsConfigDict(
        env_prefix="FORGE_DIAGNOSTIC_", env_file=_ENV_FILE, extra="ignore"
    )

    enabled: bool = False
    path: str = "./forge-diagnostic.ndjson"


class ProvidersConfig(BaseModel):
    """LLM provider credentials.

    Sub-models for each provider (``OpenAIConfig``, ``AnthropicConfig``, ...)
    are introduced by the LLM module when it lands and are added here at that
    point. Until then, this is an empty placeholder so the rest of the
    ``Settings`` shape is stable.
    """


# ---------------------------------------------------------------------------
# Root Settings
# ---------------------------------------------------------------------------


class Settings(BaseSettings):
    """Root configuration for strata-forge.

    Sub-models read env vars with their own prefixes; the root model only
    owns project-wide knobs (``profile``) and aggregates the sub-models via
    ``default_factory``.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    profile: str = Field(
        default="dev",
        validation_alias=AliasChoices("profile", "FORGE_PROFILE"),
    )

    providers: ProvidersConfig = Field(default_factory=ProvidersConfig)
    langfuse: LangfuseConfig = Field(default_factory=LangfuseConfig)
    redis: RedisConfig = Field(default_factory=RedisConfig)
    qdrant: QdrantConfig = Field(default_factory=QdrantConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    huggingface: HuggingFaceConfig = Field(default_factory=HuggingFaceConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    diagnostic: DiagnosticConfig = Field(default_factory=DiagnosticConfig)


# ---------------------------------------------------------------------------
# Cached accessor
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def _cached_settings() -> Settings:
    return Settings()


def get_settings() -> Settings:
    """Return the cached ``Settings`` instance for the current process.

    The first call instantiates ``Settings`` (reading env / .env / aliases);
    subsequent calls return the same instance. Use :func:`reset_settings`
    in tests to force a re-read after mutating env vars.
    """
    return _cached_settings()


def reset_settings() -> None:
    """Clear the cached ``Settings``. Tests only — do not call in production code."""
    _cached_settings.cache_clear()
