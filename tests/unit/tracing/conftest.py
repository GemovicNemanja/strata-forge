"""Tracing-specific test fixtures.

The autouse `reset_tracing_client` fixture clears the cached Langfuse
client between every test in this directory. Combined with the
root-level `reset_settings_cache` fixture (which clears
`get_settings`), tests that mutate `LANGFUSE_*` env vars always start
from a clean slate.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from forge.tracing.client import reset_client

if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture(autouse=True)
def reset_tracing_client() -> Iterator[None]:  # pyright: ignore[reportUnusedFunction]
    """Clear the cached Langfuse client before and after every test."""
    reset_client()
    yield
    reset_client()
