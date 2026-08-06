"""Unit tests for `strata_forge.core.budget`."""

from __future__ import annotations

import asyncio

import pytest

from strata_forge.core.budget import BudgetContext, current_budget
from strata_forge.core.errors import BudgetExceededError


class TestBasic:
    async def test_construction_defaults(self) -> None:
        b = BudgetContext()
        assert b.max_usd is None
        assert b.max_tokens is None
        assert b.spent_usd == 0.0
        assert b.spent_tokens == 0
        assert b.isolated is False

    async def test_consume_usd_under_limit(self) -> None:
        async with BudgetContext(max_usd=5.0) as budget:
            await budget.consume(usd=2.5)
            assert budget.spent_usd == 2.5
            assert budget.available_usd() == 2.5

    async def test_consume_tokens_under_limit(self) -> None:
        async with BudgetContext(max_tokens=1000) as budget:
            await budget.consume(tokens=400)
            assert budget.spent_tokens == 400
            assert budget.available_tokens() == 600

    async def test_consume_both_under_limit(self) -> None:
        async with BudgetContext(max_usd=5.0, max_tokens=1000) as budget:
            await budget.consume(usd=1.0, tokens=200)
            assert budget.spent_usd == 1.0
            assert budget.spent_tokens == 200

    async def test_no_op_consume(self) -> None:
        async with BudgetContext(max_usd=5.0) as budget:
            await budget.consume()  # No usd, no tokens — should be a no-op
            assert budget.spent_usd == 0.0

    async def test_available_returns_none_without_limit(self) -> None:
        async with BudgetContext() as budget:
            assert budget.available_usd() is None
            assert budget.available_tokens() is None


class TestExceeded:
    async def test_usd_exceeded_raises_before_committing(self) -> None:
        async with BudgetContext(max_usd=1.0) as budget:
            await budget.consume(usd=0.5)
            with pytest.raises(BudgetExceededError) as excinfo:
                await budget.consume(usd=0.6)
            # Spend stays at the pre-failed state.
            assert budget.spent_usd == 0.5
            assert excinfo.value.limit_usd == 1.0
            assert excinfo.value.spent_usd == pytest.approx(1.1)

    async def test_tokens_exceeded_raises_before_committing(self) -> None:
        async with BudgetContext(max_tokens=100) as budget:
            await budget.consume(tokens=80)
            with pytest.raises(BudgetExceededError) as excinfo:
                await budget.consume(tokens=21)
            assert budget.spent_tokens == 80
            assert excinfo.value.limit_tokens == 100
            assert excinfo.value.spent_tokens == 101

    async def test_either_limit_can_trip(self) -> None:
        async with BudgetContext(max_usd=10.0, max_tokens=100) as budget:
            with pytest.raises(BudgetExceededError) as excinfo:
                # USD is fine, but tokens overshoot.
                await budget.consume(usd=1.0, tokens=200)
            assert excinfo.value.limit_tokens == 100
            assert budget.spent_usd == 0.0
            assert budget.spent_tokens == 0

    async def test_exact_match_is_allowed(self) -> None:
        async with BudgetContext(max_usd=1.0) as budget:
            await budget.consume(usd=1.0)
            assert budget.spent_usd == 1.0
            assert budget.available_usd() == 0.0


class TestNesting:
    async def test_inner_consume_charges_outer(self) -> None:
        async with (
            BudgetContext(max_usd=10.0) as outer,
            BudgetContext(max_usd=5.0) as inner,
        ):
            await inner.consume(usd=2.0)
            assert inner.spent_usd == 2.0
            assert outer.spent_usd == 2.0

    async def test_outer_limit_can_trip_via_inner(self) -> None:
        async with BudgetContext(max_usd=3.0) as outer:
            await outer.consume(usd=2.5)
            async with BudgetContext(max_usd=10.0) as inner:
                with pytest.raises(BudgetExceededError) as excinfo:
                    await inner.consume(usd=1.0)
                assert excinfo.value.limit_usd == 3.0
                # Neither budget is mutated when the chain check fails.
                assert outer.spent_usd == 2.5
                assert inner.spent_usd == 0.0

    async def test_isolated_inner_does_not_charge_outer(self) -> None:
        async with (
            BudgetContext(max_usd=10.0) as outer,
            BudgetContext(max_usd=5.0, isolated=True) as inner,
        ):
            await inner.consume(usd=4.0)
            assert inner.spent_usd == 4.0
            assert outer.spent_usd == 0.0

    async def test_three_levels_deep(self) -> None:
        async with (
            BudgetContext(max_usd=10.0) as a,
            BudgetContext(max_usd=5.0) as b,
            BudgetContext(max_usd=2.0) as c,
        ):
            await c.consume(usd=1.0)
            assert c.spent_usd == 1.0
            assert b.spent_usd == 1.0
            assert a.spent_usd == 1.0


class TestActiveBudget:
    async def test_current_returns_none_outside(self) -> None:
        assert current_budget() is None

    async def test_current_returns_innermost(self) -> None:
        async with BudgetContext(max_usd=5.0) as outer:
            assert current_budget() is outer
            async with BudgetContext(max_usd=2.0) as inner:
                assert current_budget() is inner
            assert current_budget() is outer
        assert current_budget() is None

    async def test_propagates_across_awaits(self) -> None:
        async with BudgetContext(max_usd=5.0) as budget:

            async def child() -> BudgetContext | None:
                await asyncio.sleep(0)
                return current_budget()

            assert await child() is budget


class TestConcurrent:
    async def test_serialized_consumes_keep_count_accurate(self) -> None:
        async with BudgetContext(max_usd=100.0) as budget:
            await asyncio.gather(*(budget.consume(usd=0.1) for _ in range(100)))
            assert budget.spent_usd == pytest.approx(10.0)

    async def test_concurrent_overspend_at_most_one_succeeds_within_limit(self) -> None:
        """Even under concurrent load, the limit isn't exceeded."""
        async with BudgetContext(max_usd=1.0) as budget:
            # 30 concurrent consumes of 0.1 each = 3.0 total requested vs 1.0 limit.
            results = await asyncio.gather(
                *(budget.consume(usd=0.1) for _ in range(30)),
                return_exceptions=True,
            )
            ok = sum(1 for r in results if r is None)
            errors = [r for r in results if isinstance(r, BudgetExceededError)]
            # Exactly 10 calls fit (10 * 0.1 == 1.0) and the rest raise.
            assert ok == 10
            assert len(errors) == 20
            assert budget.spent_usd == pytest.approx(1.0)
