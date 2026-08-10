"""Shared helpers for the runnable example scripts.

Each example uses these to keep the boilerplate small: a consistent
argparse shape (``--model`` / ``--provider``), an env-var check that
skips cleanly when the provider's credentials aren't set, and a
one-line summary formatter that prints route, tokens, cost, and
latency.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import TYPE_CHECKING

# Load `.env` (if present) into os.environ before any provider check
# runs. The examples are end-to-end scripts; users expect their
# repo-root .env to be picked up the same way `strata-forge doctor` and the
# CLI do via strata_forge.config.settings.
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover — python-dotenv is a core dep
    pass

if TYPE_CHECKING:
    from strata_forge.llm.responses import LLMResponse

__all__ = [
    "parse_args",
    "print_summary",
    "require_env",
]


# Provider → environment variables required for live calls to work.
# Missing any of these triggers a clean skip rather than a confusing crash
# deep in LiteLLM.
_PROVIDER_ENV_VARS: dict[str, tuple[str, ...]] = {
    "openai": ("OPENAI_API_KEY",),
    "anthropic": ("ANTHROPIC_API_KEY",),
    "vertex": ("GOOGLE_APPLICATION_CREDENTIALS", "GCP_PROJECT"),
    "bedrock": ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_REGION"),
    "azure": ("AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT"),
    "openai_compat": (),
}


def parse_args(
    *,
    description: str,
    default_model: str = "claude-haiku-4-5",
    default_provider: str = "anthropic",
    extra_args: list[tuple[str, dict[str, object]]] | None = None,
) -> argparse.Namespace:
    """Standard CLI shape: ``--model`` and ``--provider``, plus extras.

    Args:
        description: argparse description shown in ``--help``.
        default_model: registry model name used if ``--model`` isn't passed.
        default_provider: provider route used if ``--provider`` isn't passed.
        extra_args: optional ``[(flag, kwargs), ...]`` tuples appended to
            the parser. Each example that needs an extra flag (a prompt,
            an image URL, …) declares it here.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--model",
        default=default_model,
        help=f"Logical model name (default: {default_model})",
    )
    parser.add_argument(
        "--provider",
        default=default_provider,
        choices=list(_PROVIDER_ENV_VARS),
        help=f"Provider route (default: {default_provider})",
    )
    for flag, kwargs in extra_args or []:
        parser.add_argument(flag, **kwargs)  # type: ignore[arg-type]
    return parser.parse_args()


def require_env(provider: str) -> None:
    """Exit the script with a clear skip message when provider creds are unset."""
    required = _PROVIDER_ENV_VARS.get(provider, ())
    missing = [var for var in required if not os.environ.get(var)]
    if missing:
        joined = ", ".join(missing)
        print(f"[skip] {provider}: missing env vars: {joined}", file=sys.stderr)
        sys.exit(0)


def print_summary(response: LLMResponse, *, label: str = "response") -> None:
    """Print the response body + a one-line route/usage/cost/latency footer."""
    print(f"--- {label} ---")
    if response.text:
        print(response.text)
    if response.tool_calls:
        print(f"tool_calls={response.tool_calls}")
    print(
        f"[route={response.route.model}@{response.route.provider} "
        f"tokens={response.usage.total_tokens} "
        f"cost=${response.cost_usd:.6f} "
        f"latency={response.latency_ms:.1f}ms "
        f"cache_hit={response.cache_hit}]"
    )
