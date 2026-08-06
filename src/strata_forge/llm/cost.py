"""USD-cost computation from token usage and the model registry.

Prices in the registry are stored per million tokens. ``compute_cost``
multiplies the ``Usage`` counts by the matching rate, separating
input / output / cache-read / cache-write. When a model has no explicit
cache-read / cache-write rate (e.g. it doesn't support prompt caching)
but the usage reports non-zero cache tokens — an anomaly — we fall back
to the input rate so the cost stays conservative rather than silently
zero-ing out the spend.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from strata_forge.llm.registry import registry as _global_registry

if TYPE_CHECKING:
    from strata_forge.llm.responses import Usage

__all__ = ["compute_cost"]


_PER_MILLION = 1_000_000


def compute_cost(usage: Usage, model: str) -> float:
    """Return the USD cost for a completed call.

    Args:
        usage: Token-usage counters from the provider response.
        model: Logical model name (or alias) registered in
            ``strata_forge.llm.registry``.

    Returns:
        USD cost as a float. Always non-negative.

    Raises:
        RegistryError: When ``model`` is not in the registry.
    """
    entry = _global_registry.get(model)
    pricing = entry.pricing_per_million_tokens

    cost = usage.input_tokens * pricing.input / _PER_MILLION
    cost += usage.output_tokens * pricing.output / _PER_MILLION

    cache_read_rate = pricing.cache_read if pricing.cache_read is not None else pricing.input
    cost += usage.cache_read_tokens * cache_read_rate / _PER_MILLION

    cache_write_rate = pricing.cache_write if pricing.cache_write is not None else pricing.input
    cost += usage.cache_write_tokens * cache_write_rate / _PER_MILLION

    return cost
