"""Concurrency-bounded async batch inference over :class:`LLMClient`.

:class:`BatchInferenceRunner` runs many prompts through one client
in parallel, capped by ``concurrency``. On failure, the runner
either raises immediately (``on_error="raise"``, default) or
collects the exception into the result tuple
(``on_error="collect"``) so a long batch can finish despite
individual failures.

Returned :class:`BatchInferenceResult` objects are positionally
aligned with the input prompts — ``results[i]`` is the outcome for
``prompts[i]`` whether it succeeded, failed, or hasn't been touched
because an earlier task raised in ``raise`` mode.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from strata_forge.llm.client import LLMClient
    from strata_forge.llm.messages import AnyMessage
    from strata_forge.llm.registry import ProviderName
    from strata_forge.llm.responses import LLMResponse
    from strata_forge.llm.tools import Tool

__all__ = [
    "BatchInferenceResult",
    "BatchInferenceRunner",
]


type ErrorPolicy = Literal["raise", "collect"]
"""How the runner handles per-prompt failures."""


@dataclass(frozen=True, slots=True)
class BatchInferenceResult:
    """One outcome from a :class:`BatchInferenceRunner.run` invocation.

    Exactly one of :attr:`response` and :attr:`error` is set:

    - On success, ``response`` is the :class:`LLMResponse` and
      ``error`` is ``None``.
    - On failure (only reachable when ``on_error="collect"``),
      ``error`` is the raised exception and ``response`` is ``None``.
    - When an earlier failure short-circuited a ``raise``-mode batch,
      results past the failure point are entirely absent from the
      returned tuple — they're never constructed.
    """

    response: LLMResponse | None = None
    error: Exception | None = field(default=None, compare=False)

    @property
    def succeeded(self) -> bool:
        return self.response is not None and self.error is None


class BatchInferenceRunner:
    """Run a fixed :class:`LLMClient` over many prompts in parallel.

    Args:
        client: The :class:`LLMClient` every prompt routes through.
        concurrency: Maximum in-flight calls. Default ``5``.
        on_error: ``"raise"`` (default) cancels the batch on first
            failure; ``"collect"`` keeps going and stores the
            exception in the corresponding result slot.
    """

    def __init__(
        self,
        client: LLMClient,
        *,
        concurrency: int = 5,
        on_error: ErrorPolicy = "raise",
    ) -> None:
        if concurrency <= 0:
            err = f"concurrency must be >= 1; got {concurrency}"
            raise ValueError(err)
        if on_error not in ("raise", "collect"):
            err = f"on_error must be 'raise' or 'collect'; got {on_error!r}"
            raise ValueError(err)
        self._client = client
        self._concurrency = concurrency
        self._on_error: ErrorPolicy = on_error

    @property
    def concurrency(self) -> int:
        return self._concurrency

    @property
    def on_error(self) -> ErrorPolicy:
        return self._on_error

    async def run(
        self,
        prompts: Sequence[Sequence[AnyMessage]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        tools: Sequence[Tool] | None = None,
        provider_extras: Mapping[ProviderName, Mapping[str, Any]] | None = None,
    ) -> tuple[BatchInferenceResult, ...]:
        """Run every prompt through ``client.complete``; return aligned results.

        The returned tuple has the same length as ``prompts`` when
        the run completes without raising. In ``on_error="raise"``
        mode, the first failure cancels the rest of the batch — only
        the prompts that had already started see an outcome; later
        prompts are simply absent from the tuple.
        """
        if not prompts:
            return ()

        semaphore = asyncio.Semaphore(self._concurrency)

        async def _one(messages: Sequence[AnyMessage]) -> BatchInferenceResult:
            async with semaphore:
                try:
                    response = await self._client.complete(
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        top_p=top_p,
                        tools=tools,
                        provider_extras=provider_extras,
                    )
                except Exception as exc:
                    if self._on_error == "raise":
                        raise
                    return BatchInferenceResult(error=exc)
                return BatchInferenceResult(response=response)

        outcomes = await asyncio.gather(*(_one(p) for p in prompts))
        return tuple(outcomes)
