"""Shared pytest fixtures for the ai-forge test suite.

Root conftest applies to every test file under ``tests/``. Subdirectories may
add their own ``conftest.py`` for narrower fixtures (see ``tests/vcr/`` for
the cassette-recording configuration).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from forge.config import reset_settings as _reset_settings

if TYPE_CHECKING:
    from collections.abc import Iterator


# Every Forge-recognized env var. The `clean_forge_env` fixture strips all of
# them so tests that exercise Settings start from documented defaults rather
# than from whatever the developer's shell happens to have set.
_FORGE_ENV_VARS: tuple[str, ...] = (
    # Project-wide
    "FORGE_PROFILE",
    # Diagnostic NDJSON
    "FORGE_DIAGNOSTIC_ENABLED",
    "FORGE_DIAGNOSTIC_PATH",
    # Logging
    "FORGE_LOG_LEVEL",
    "FORGE_LOG_FORMAT",
    # Storage
    "FORGE_STORAGE_DEFAULT_BACKEND",
    # Langfuse
    "LANGFUSE_HOST",
    "LANGFUSE_PUBLIC_KEY",
    "LANGFUSE_SECRET_KEY",
    # Redis
    "REDIS_URL",
    # Qdrant
    "QDRANT_URL",
    "QDRANT_API_KEY",
    # Provider credentials (recognized by the LLM module when it lands)
    "OPENAI_API_KEY",
    "OPENAI_ORG_ID",
    "ANTHROPIC_API_KEY",
    "GOOGLE_APPLICATION_CREDENTIALS",
    "GCP_PROJECT",
    "GCP_REGION",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_REGION",
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_API_VERSION",
)


@pytest.fixture(autouse=True)
def reset_settings_cache() -> Iterator[None]:  # pyright: ignore[reportUnusedFunction]
    """Clear the Settings cache before and after every test.

    Without this, the ``@lru_cache`` on ``get_settings`` would freeze the
    first-read value for the entire test session, making env-var manipulation
    via ``monkeypatch.setenv`` silently ineffective on later tests.
    """
    _reset_settings()
    yield
    _reset_settings()


@pytest.fixture
def clean_forge_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strip every Forge-recognized env var from the process environment.

    Opt-in. Use this in tests that exercise ``Settings`` and want a documented
    baseline regardless of how the developer's shell is configured.
    """
    for var in _FORGE_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
