"""Cost and token budget tracking with nested ceilings.

A ``BudgetContext`` is an async context manager that enforces a USD or token
ceiling around any block of code. ``BudgetContext.consume`` is called by the
LLM client (and any other expensive operation) before the cost is incurred —
if the call would exceed the limit, ``BudgetExceededError`` fires *before* the
spend happens.

Nested ``BudgetContext`` instances are linked: a child's ``consume`` also
charges every ancestor in the active chain, so a tight inner limit and a
broader outer limit are both enforced simultaneously. Pass ``isolated=True``
to break the link and create a hard sub-budget that doesn't bleed up.
"""

from __future__ import annotations

import contextvars
from typing import TYPE_CHECKING

import anyio

from strata_forge.core.errors import BudgetExceededError

if TYPE_CHECKING:
    from types import TracebackType

__all__ = ["BudgetContext", "current_budget"]


_current_budget: contextvars.ContextVar[BudgetContext | None] = contextvars.ContextVar(
    "strata_forge.core.budget.current",
    default=None,
)


def current_budget() -> BudgetContext | None:
    """Return the innermost active ``BudgetContext``, or ``None`` if none is bound."""
    return _current_budget.get()


class BudgetContext:
    """Async context manager that enforces cost and/or token ceilings.

    Args:
        max_usd: Maximum total USD that can be consumed inside the block. If
            ``None``, USD is not enforced (but ``spent_usd`` is still tracked).
        max_tokens: Maximum total tokens. ``None`` disables token enforcement.
        isolated: If ``True``, this budget does NOT inherit from any active
            outer ``BudgetContext`` — consumes against it don't propagate up.
            Useful for a hard sub-budget that mustn't pollute the parent.

    Spend is mutated atomically across the active chain: if any ancestor would
    be exceeded, no spend is recorded on any budget in the chain.
    """

    def __init__(
        self,
        *,
        max_usd: float | None = None,
        max_tokens: int | None = None,
        isolated: bool = False,
    ) -> None:
        self.max_usd = max_usd
        self.max_tokens = max_tokens
        self.isolated = isolated
        self.spent_usd: float = 0.0
        self.spent_tokens: int = 0
        self._parent: BudgetContext | None = None
        self._token: contextvars.Token[BudgetContext | None] | None = None
        self._lock = anyio.Lock()

    async def __aenter__(self) -> BudgetContext:
        if not self.isolated:
            self._parent = _current_budget.get()
        self._token = _current_budget.set(self)
        return self

    async def __aexit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc: BaseException | None,
        _tb: TracebackType | None,
    ) -> None:
        if self._token is not None:
            _current_budget.reset(self._token)
            self._token = None
        self._parent = None

    async def consume(
        self,
        *,
        usd: float | None = None,
        tokens: int | None = None,
    ) -> None:
        """Record spend across this budget and every linked ancestor.

        If any budget in the chain would be exceeded by this call, raises
        ``BudgetExceededError`` and records nothing on any budget. The error
        carries the offending budget's limit / projected spend.
        """
        if usd is None and tokens is None:
            return

        chain = self._chain()
        # Serialize concurrent consumes through the root lock — separate chains
        # don't share a root, so this only blocks consumes within the same tree.
        async with chain[-1]._lock:
            for budget in chain:
                budget._check(usd=usd, tokens=tokens)
            for budget in chain:
                budget._commit(usd=usd, tokens=tokens)

    def available_usd(self) -> float | None:
        """Remaining USD before the ceiling, or ``None`` if no USD ceiling."""
        if self.max_usd is None:
            return None
        return max(0.0, self.max_usd - self.spent_usd)

    def available_tokens(self) -> int | None:
        """Remaining tokens before the ceiling, or ``None`` if no token ceiling."""
        if self.max_tokens is None:
            return None
        return max(0, self.max_tokens - self.spent_tokens)

    # --- internal --------------------------------------------------------

    def _chain(self) -> list[BudgetContext]:
        chain: list[BudgetContext] = []
        node: BudgetContext | None = self
        while node is not None:
            chain.append(node)
            node = node._parent
        return chain

    def _check(self, *, usd: float | None, tokens: int | None) -> None:
        if usd is not None and self.max_usd is not None:
            projected = self.spent_usd + usd
            if projected > self.max_usd:
                raise BudgetExceededError(
                    f"USD budget of {self.max_usd} would be exceeded "
                    f"(spent {self.spent_usd} + requested {usd} = {projected})",
                    limit_usd=self.max_usd,
                    spent_usd=projected,
                )
        if tokens is not None and self.max_tokens is not None:
            projected_tokens = self.spent_tokens + tokens
            if projected_tokens > self.max_tokens:
                raise BudgetExceededError(
                    f"Token budget of {self.max_tokens} would be exceeded "
                    f"(spent {self.spent_tokens} + requested {tokens} = {projected_tokens})",
                    limit_tokens=self.max_tokens,
                    spent_tokens=projected_tokens,
                )

    def _commit(self, *, usd: float | None, tokens: int | None) -> None:
        if usd is not None:
            self.spent_usd += usd
        if tokens is not None:
            self.spent_tokens += tokens
