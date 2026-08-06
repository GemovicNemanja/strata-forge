"""Shared helpers for Forge CLI subcommands.

The helpers cover three small recurring needs:

- :func:`run_async` — bridge Typer's sync command callables to
  Forge's async public API.
- :func:`prompt_store_from_settings` /
  :func:`dataset_store_from_settings` — factory helpers that
  pick the right backend (Langfuse if configured, otherwise the
  in-memory fallback) so each command body can stay simple.
- :func:`error_exit` — print a red error message and exit
  non-zero in a way Typer respects.

All command-side code in :mod:`strata_forge.cli` should import from
here rather than reimplementing these in each subcommand module.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, NoReturn

import typer
from rich.console import Console

if TYPE_CHECKING:
    from collections.abc import Awaitable

    from strata_forge.datasets.store import DatasetStore
    from strata_forge.prompts.registry import PromptStore

__all__ = [
    "dataset_store_from_settings",
    "error_exit",
    "prompt_store_from_settings",
    "run_async",
]


_ERROR_CONSOLE = Console(stderr=True, style="bold red")


def run_async[T](coro: Awaitable[T]) -> T:
    """Run an async coroutine from a sync Typer command."""
    return asyncio.run(coro)  # type: ignore[arg-type]


def error_exit(message: str, *, code: int = 1) -> NoReturn:
    """Print ``message`` to stderr and exit with ``code``."""
    _ERROR_CONSOLE.print(f"error: {message}")
    raise typer.Exit(code=code)


def prompt_store_from_settings() -> PromptStore:
    """Return the configured :class:`PromptStore`.

    Prefers a Langfuse-backed store when ``LangfuseConfig`` is
    populated; falls back to a fresh in-process
    :class:`InMemoryPromptStore` otherwise. Importing the
    Langfuse backend is gated by the ``[langfuse]`` extra; on
    ImportError we surface a clear message instead of falling
    back silently.
    """
    from strata_forge.config.settings import get_settings
    from strata_forge.prompts.stores.memory import InMemoryPromptStore

    settings = get_settings()
    if settings.langfuse.enabled:
        try:
            from strata_forge.prompts.stores.langfuse import LangfusePromptStore
        except ImportError as exc:
            error_exit(f"Langfuse is configured but the [langfuse] extra is not installed: {exc}")
        return LangfusePromptStore()
    return InMemoryPromptStore()


def dataset_store_from_settings() -> DatasetStore:
    """Return the configured :class:`DatasetStore`.

    Same selection rule as :func:`prompt_store_from_settings`:
    Langfuse when enabled, otherwise an in-process
    :class:`InMemoryDatasetStore`.
    """
    from strata_forge.config.settings import get_settings
    from strata_forge.datasets.stores.memory import InMemoryDatasetStore

    settings = get_settings()
    if settings.langfuse.enabled:
        try:
            from strata_forge.datasets.stores.langfuse import LangfuseDatasetStore
        except ImportError as exc:
            error_exit(f"Langfuse is configured but the [langfuse] extra is not installed: {exc}")
        return LangfuseDatasetStore()
    return InMemoryDatasetStore()
