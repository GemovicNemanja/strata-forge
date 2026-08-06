"""Retry decorator for sync and async callables, defaulting to transient provider errors."""

from __future__ import annotations

from typing import TYPE_CHECKING, overload

import tenacity

from strata_forge.core.errors import (
    ProviderRateLimitError,
    ProviderServerError,
    ProviderTimeoutError,
)

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = ["DEFAULT_RETRY_ON", "retry"]


DEFAULT_RETRY_ON: tuple[type[BaseException], ...] = (
    ProviderRateLimitError,
    ProviderTimeoutError,
    ProviderServerError,
)
"""Exception types retried by default — transient provider failures only.

Auth, bad request, and content-filter errors are intentionally excluded:
retrying them is wasted work since the request itself is the problem.
"""


@overload
def retry[**P, R](fn: Callable[P, R], /) -> Callable[P, R]: ...


@overload
def retry[**P, R](
    fn: None = None,
    /,
    *,
    max_attempts: int = ...,
    initial_wait: float = ...,
    max_wait: float = ...,
    retry_on: tuple[type[BaseException], ...] = ...,
) -> Callable[[Callable[P, R]], Callable[P, R]]: ...


def retry[**P, R](
    fn: Callable[P, R] | None = None,
    /,
    *,
    max_attempts: int = 5,
    initial_wait: float = 1.0,
    max_wait: float = 30.0,
    retry_on: tuple[type[BaseException], ...] = DEFAULT_RETRY_ON,
) -> Callable[P, R] | Callable[[Callable[P, R]], Callable[P, R]]:
    """Retry a sync or async callable on transient errors.

    Usable bare (``@retry``) or with overrides (``@retry(max_attempts=10)``).
    Tenacity does the actual retrying; this wrapper supplies Forge-aware
    defaults and types.

    Args:
        fn: The callable to wrap when used as a bare decorator.
        max_attempts: Maximum total attempts (including the first call).
        initial_wait: Base wait between attempts, in seconds.
        max_wait: Cap on wait between attempts, in seconds.
        retry_on: Exception types that trigger a retry. Defaults to
            ``DEFAULT_RETRY_ON`` (rate limit, timeout, server).
    """

    def _wrap(func: Callable[P, R]) -> Callable[P, R]:
        decorator: Callable[[Callable[P, R]], Callable[P, R]] = tenacity.retry(
            stop=tenacity.stop_after_attempt(max_attempts),
            wait=tenacity.wait_exponential_jitter(initial=initial_wait, max=max_wait),
            retry=tenacity.retry_if_exception_type(retry_on),
            reraise=True,
        )
        return decorator(func)

    if fn is not None:
        return _wrap(fn)
    return _wrap
