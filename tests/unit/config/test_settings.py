"""Unit tests for `forge.config.settings`."""

from __future__ import annotations

import pytest
from pydantic import SecretStr

from forge.config.settings import (
    DiagnosticConfig,
    LangfuseConfig,
    LoggingConfig,
    QdrantConfig,
    RedisConfig,
    Settings,
    StorageConfig,
    get_settings,
    reset_settings,
)


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch: pytest.MonkeyPatch) -> None:  # pyright: ignore[reportUnusedFunction]
    """Strip every Forge / provider env var so tests start from clean defaults."""
    for var in (
        "FORGE_PROFILE",
        "FORGE_DIAGNOSTIC_ENABLED",
        "FORGE_DIAGNOSTIC_PATH",
        "FORGE_LOG_LEVEL",
        "FORGE_LOG_FORMAT",
        "FORGE_STORAGE_DEFAULT_BACKEND",
        "LANGFUSE_HOST",
        "LANGFUSE_PUBLIC_KEY",
        "LANGFUSE_SECRET_KEY",
        "REDIS_URL",
        "QDRANT_URL",
        "QDRANT_API_KEY",
    ):
        monkeypatch.delenv(var, raising=False)
    reset_settings()


class TestDefaults:
    def test_settings_constructs_with_empty_env(self) -> None:
        settings = Settings()
        assert settings.profile == "dev"
        assert settings.langfuse.host == "http://localhost:3000"
        assert settings.redis.url == "redis://localhost:6379/0"
        assert settings.qdrant.url == "http://localhost:6333"
        assert settings.storage.default_backend == "local"
        assert settings.logging.level == "INFO"
        assert settings.logging.format == "auto"
        assert settings.diagnostic.enabled is False
        assert settings.diagnostic.path == "./forge-diagnostic.ndjson"

    def test_secrets_default_to_none(self) -> None:
        settings = Settings()
        assert settings.langfuse.public_key is None
        assert settings.langfuse.secret_key is None
        assert settings.qdrant.api_key is None


class TestEnvOverrides:
    def test_profile_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FORGE_PROFILE", "production")
        assert Settings().profile == "production"

    def test_langfuse_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-1234")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-5678")
        cfg = LangfuseConfig()
        assert cfg.host == "https://cloud.langfuse.com"
        assert isinstance(cfg.public_key, SecretStr)
        assert cfg.public_key.get_secret_value() == "pk-1234"
        assert cfg.secret_key.get_secret_value() == "sk-5678"  # pyright: ignore[reportOptionalMemberAccess]
        assert cfg.enabled is True

    def test_langfuse_enabled_requires_both_keys(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-1234")
        # secret key missing → not enabled
        assert LangfuseConfig().enabled is False

    def test_redis_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("REDIS_URL", "redis://prod-cache:6379/3")
        assert RedisConfig().url == "redis://prod-cache:6379/3"

    def test_qdrant_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("QDRANT_URL", "https://qdrant.example.com")
        monkeypatch.setenv("QDRANT_API_KEY", "secret-value")
        cfg = QdrantConfig()
        assert cfg.url == "https://qdrant.example.com"
        assert cfg.api_key is not None
        assert cfg.api_key.get_secret_value() == "secret-value"

    def test_storage_default_backend(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FORGE_STORAGE_DEFAULT_BACKEND", "s3")
        assert StorageConfig().default_backend == "s3"

    def test_storage_rejects_invalid_backend(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FORGE_STORAGE_DEFAULT_BACKEND", "not-a-backend")
        with pytest.raises(ValueError, match="default_backend"):
            StorageConfig()

    def test_logging_level_and_format(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FORGE_LOG_LEVEL", "DEBUG")
        monkeypatch.setenv("FORGE_LOG_FORMAT", "json")
        cfg = LoggingConfig()
        assert cfg.level == "DEBUG"
        assert cfg.format == "json"

    def test_diagnostic_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FORGE_DIAGNOSTIC_ENABLED", "1")
        monkeypatch.setenv("FORGE_DIAGNOSTIC_PATH", "/tmp/forge.ndjson")  # noqa: S108
        cfg = DiagnosticConfig()
        assert cfg.enabled is True
        assert cfg.path == "/tmp/forge.ndjson"  # noqa: S108

    def test_root_settings_combines_subconfigs(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LANGFUSE_HOST", "https://langfuse.example.com")
        monkeypatch.setenv("FORGE_LOG_LEVEL", "WARNING")
        settings = Settings()
        assert settings.langfuse.host == "https://langfuse.example.com"
        assert settings.logging.level == "WARNING"


class TestInCodeOverride:
    def test_kwarg_overrides_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LANGFUSE_HOST", "https://from-env.example.com")
        settings = Settings(
            langfuse=LangfuseConfig(host="https://from-code.example.com"),
        )
        assert settings.langfuse.host == "https://from-code.example.com"


class TestSecretsDoNotLeak:
    def test_repr_redacts_secrets(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-VERYSENSITIVE")
        cfg = LangfuseConfig()
        rendered = repr(cfg)
        assert "VERYSENSITIVE" not in rendered
        # The SecretStr type renders as `**********`.
        assert "**" in rendered


class TestGetSettings:
    def test_returns_same_instance(self) -> None:
        a = get_settings()
        b = get_settings()
        assert a is b

    def test_reset_clears_cache(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FORGE_PROFILE", "before")
        first = get_settings()
        assert first.profile == "before"

        monkeypatch.setenv("FORGE_PROFILE", "after")
        # Without reset, env changes are invisible.
        assert get_settings().profile == "before"

        reset_settings()
        # After reset, env is re-read.
        assert get_settings().profile == "after"

    def test_kwargs_bypass_cache_function(self) -> None:
        # get_settings() always returns the cached default-init instance. Tests
        # that need custom values construct Settings() directly instead.
        cached = get_settings()
        custom = Settings(profile="custom")
        assert cached is not custom
        assert custom.profile == "custom"
