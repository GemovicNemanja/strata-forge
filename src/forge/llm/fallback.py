"""Two-axis fallback for LLM calls.

A fallback chain is a list of :class:`ModelFallback` entries. The runner
:func:`run_with_fallback` iterates the chain in two nested loops:

- **Inner (provider-level)**: within one :class:`ModelFallback`, the
  ``providers`` list is tried in order. A provider attempt is wrapped in
  :func:`forge.core.retry.retry` (exponential backoff + jitter) so a
  transient blip retries before the provider is counted as exhausted.
- **Outer (model-level)**: when every provider in the current
  :class:`ModelFallback` is exhausted, the loop advances to the next
  entry — which may be a different logical model.

Error-handling rules (per error class):

================================  ===========  ====================
Error                             Default      Strict
================================  ===========  ====================
``ProviderContentFilterError``    re-raise     re-raise
``ProviderBadRequestError``       advance      re-raise
``ProviderAuthError``             advance      advance
``ProviderRateLimitError``        retry+adv    retry+adv
``ProviderTimeoutError``          retry+adv    retry+adv
``ProviderServerError``           retry+adv    retry+adv
``RegistryError``                 advance      advance
================================  ===========  ====================

``ProviderContentFilterError`` is treated as terminal because the same
content will be refused by every other provider — retrying is wasted
work. Every other failure type accumulates into the
:exc:`FallbackExhaustedError`'s ``causes`` list so callers can introspect
the full route history. Any non-provider exception (programmer error,
``KeyboardInterrupt``, …) propagates unchanged.

For the most common case — try one model on its default route, then
another — use the bare-string shorthand::

    chain = normalize_fallback_chain(["claude-opus-4-7", "gpt-5.5"])

Use the explicit :class:`ModelFallback` form to opt into provider-level
failover::

    chain = [
        ModelFallback(model="claude-opus-4-7",
                      providers=("anthropic", "bedrock", "vertex")),
        ModelFallback(model="gpt-5.5", providers=("openai", "azure")),
    ]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from forge.core.errors import (
    FallbackExhaustedError,
    ProviderAuthError,
    ProviderBadRequestError,
    ProviderContentFilterError,
    ProviderError,
    ProviderRateLimitError,
    ProviderServerError,
    ProviderTimeoutError,
    RegistryError,
)
from forge.core.retry import retry as retry_decorator
from forge.llm.routing import resolve

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence

    from forge.llm.registry import ProviderName
    from forge.llm.routing import ModelRoute

__all__ = [
    "FallbackEntry",
    "ModelFallback",
    "normalize_fallback_chain",
    "run_with_fallback",
]


@dataclass(frozen=True, slots=True)
class ModelFallback:
    """One entry in a fallback chain.

    Attributes:
        model: Logical model name (or registered alias).
        providers: Provider routes to try in order. ``None`` means
            "use the registry default route" — no provider-level
            failover. Pass an explicit tuple for failover.
    """

    model: str
    providers: tuple[ProviderName, ...] | None = None


type FallbackEntry = ModelFallback | str


def normalize_fallback_chain(entries: Sequence[FallbackEntry]) -> list[ModelFallback]:
    """Turn a mixed sequence of strings + :class:`ModelFallback` into a uniform chain.

    A bare string ``"gpt-5.5"`` expands to
    ``ModelFallback(model="gpt-5.5", providers=None)`` — i.e. take the
    registry default route, no provider-level failover.
    """
    return [ModelFallback(model=entry) if isinstance(entry, str) else entry for entry in entries]


# Provider errors that *advance* to the next provider/model. ContentFilter is
# explicitly NOT in here — it short-circuits the entire chain.
_ADVANCE_ON: tuple[type[BaseException], ...] = (
    ProviderRateLimitError,
    ProviderTimeoutError,
    ProviderServerError,
    ProviderAuthError,
)


async def run_with_fallback[T](
    chain: Sequence[FallbackEntry],
    call: Callable[[ModelRoute], Awaitable[T]],
    *,
    strict_bad_request: bool = False,
    retry_max_attempts: int = 5,
    retry_initial_wait: float = 1.0,
    retry_max_wait: float = 30.0,
) -> T:
    """Execute ``call`` against the first route in the chain that succeeds.

    Each attempt is wrapped in :func:`forge.core.retry.retry` so a
    transient rate-limit / timeout / 5xx retries with exponential
    backoff before the provider is declared exhausted.

    Args:
        chain: Fallback entries — :class:`ModelFallback` instances or
            bare strings (the latter expand via
            :func:`normalize_fallback_chain`).
        call: Async function invoked once per surviving attempt with
            the resolved :class:`ModelRoute`. Returns the success
            value (typically an :class:`LLMResponse`).
        strict_bad_request: When ``True``, a
            :exc:`ProviderBadRequestError` aborts the chain immediately
            (wrapped in :exc:`FallbackExhaustedError`). Default
            (``False``) treats bad-request as "this provider rejected
            params, try the next one."
        retry_max_attempts: Max retries per provider attempt.
        retry_initial_wait: Base wait between retries (seconds).
        retry_max_wait: Cap on wait between retries (seconds).

    Returns:
        Whatever ``call`` returns on the first successful attempt.

    Raises:
        ProviderContentFilterError: Re-raised verbatim from the first
            attempt that hits it. Other providers won't accept the
            content either, so we short-circuit.
        FallbackExhaustedError: When every entry in the chain failed.
            ``causes`` carries the ``(model, provider, error)`` triple
            for every attempt.
        Exception: Any non-provider exception is propagated unchanged.
    """
    normalized = normalize_fallback_chain(chain)
    causes: list[tuple[str, str | None, BaseException]] = []

    decorated_call = retry_decorator(
        max_attempts=retry_max_attempts,
        initial_wait=retry_initial_wait,
        max_wait=retry_max_wait,
    )(call)

    for entry in normalized:
        # An entry with `providers=None` means "default route only" — one slot
        # in the inner loop; with an explicit tuple we iterate every provider.
        provider_iter: tuple[ProviderName | None, ...] = (
            (None,) if entry.providers is None else entry.providers
        )

        for provider in provider_iter:
            try:
                route = resolve(entry.model, provider)
            except RegistryError as exc:
                causes.append((entry.model, provider, exc))
                continue

            try:
                return await decorated_call(route)
            except ProviderContentFilterError:
                # Terminal across the whole chain — same content fails everywhere.
                raise
            except ProviderBadRequestError as exc:
                causes.append((route.model, route.provider, exc))
                if strict_bad_request:
                    raise FallbackExhaustedError(
                        "Fallback aborted by strict bad-request handling",
                        causes=causes,
                    ) from exc
                continue
            except _ADVANCE_ON as exc:
                causes.append((route.model, route.provider, exc))
                continue
            except ProviderError as exc:
                # Catch-all for any future ProviderError subclass we forgot —
                # treat conservatively as "advance" so the chain keeps trying.
                causes.append((route.model, route.provider, exc))
                continue

    raise FallbackExhaustedError(
        f"Every entry in the fallback chain failed ({len(causes)} attempt(s))",
        causes=causes,
    )
