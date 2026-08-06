"""Unit tests for `strata_forge.llm.cost`."""

from __future__ import annotations

import pytest

from strata_forge.core.errors import RegistryError
from strata_forge.llm.cost import compute_cost
from strata_forge.llm.responses import Usage


class TestBasic:
    def test_zero_usage_is_zero_cost(self) -> None:
        assert compute_cost(Usage(input_tokens=0, output_tokens=0), "claude-opus-4-7") == 0.0

    def test_input_only(self) -> None:
        # Opus 4.7: $5.00 / M input tokens
        cost = compute_cost(
            Usage(input_tokens=1_000_000, output_tokens=0),
            "claude-opus-4-7",
        )
        assert cost == pytest.approx(5.00)

    def test_output_only(self) -> None:
        # Opus 4.7: $25.00 / M output tokens
        cost = compute_cost(
            Usage(input_tokens=0, output_tokens=1_000_000),
            "claude-opus-4-7",
        )
        assert cost == pytest.approx(25.00)

    def test_input_plus_output(self) -> None:
        # 100k input + 50k output on Opus 4.7
        cost = compute_cost(
            Usage(input_tokens=100_000, output_tokens=50_000),
            "claude-opus-4-7",
        )
        # 100_000 * 5 / 1_000_000 + 50_000 * 25 / 1_000_000 = 0.5 + 1.25 = 1.75
        assert cost == pytest.approx(1.75)

    def test_alias_resolves(self) -> None:
        # "opus" alias should produce the same cost as the canonical name.
        usage = Usage(input_tokens=1000, output_tokens=2000)
        assert compute_cost(usage, "opus") == compute_cost(usage, "claude-opus-4-7")

    def test_unknown_model_raises(self) -> None:
        with pytest.raises(RegistryError):
            compute_cost(Usage(input_tokens=0, output_tokens=0), "never-heard-of-it")


class TestCacheTokens:
    def test_cache_read_applies_discounted_rate(self) -> None:
        # Opus 4.7: cache_read = $0.50 / M (10% of input)
        cost = compute_cost(
            Usage(input_tokens=0, output_tokens=0, cache_read_tokens=1_000_000),
            "claude-opus-4-7",
        )
        assert cost == pytest.approx(0.50)

    def test_cache_write_applies_premium_rate(self) -> None:
        # Opus 4.7: cache_write = $6.25 / M (125% of input)
        cost = compute_cost(
            Usage(input_tokens=0, output_tokens=0, cache_write_tokens=1_000_000),
            "claude-opus-4-7",
        )
        assert cost == pytest.approx(6.25)

    def test_full_breakdown(self) -> None:
        # 100k input + 50k output + 200k cache_read + 30k cache_write
        cost = compute_cost(
            Usage(
                input_tokens=100_000,
                output_tokens=50_000,
                cache_read_tokens=200_000,
                cache_write_tokens=30_000,
            ),
            "claude-opus-4-7",
        )
        expected = (
            100_000 * 5.00 / 1_000_000
            + 50_000 * 25.00 / 1_000_000
            + 200_000 * 0.50 / 1_000_000
            + 30_000 * 6.25 / 1_000_000
        )
        assert cost == pytest.approx(expected)

    def test_cache_falls_back_to_input_rate_when_unset(self) -> None:
        # Gemini Flash Lite has no cache_read / cache_write in the registry
        # (prompt_caching: false). If usage happens to report cache tokens
        # anyway — a data anomaly — we bill them at the input rate.
        cost = compute_cost(
            Usage(input_tokens=0, output_tokens=0, cache_read_tokens=1_000_000),
            "gemini-3.1-flash-lite",
        )
        # gemini-3.1-flash-lite input: $0.10 / M
        assert cost == pytest.approx(0.10)


class TestSampleModelCosts:
    """Spot-check rates against the registry."""

    @pytest.mark.parametrize(
        ("model", "expected_input_per_million", "expected_output_per_million"),
        [
            ("claude-opus-4-7", 5.00, 25.00),
            ("claude-sonnet-4-6", 3.00, 15.00),
            ("claude-haiku-4-5", 1.00, 5.00),
            ("gpt-5.5", 5.00, 30.00),
            ("gpt-5.5-pro", 30.00, 180.00),
            ("gemini-3.1-pro", 2.00, 12.00),
            ("gemini-3.1-flash-lite", 0.10, 1.00),
        ],
    )
    def test_million_token_rates(
        self,
        model: str,
        expected_input_per_million: float,
        expected_output_per_million: float,
    ) -> None:
        input_cost = compute_cost(Usage(input_tokens=1_000_000, output_tokens=0), model)
        output_cost = compute_cost(Usage(input_tokens=0, output_tokens=1_000_000), model)
        assert input_cost == pytest.approx(expected_input_per_million)
        assert output_cost == pytest.approx(expected_output_per_million)


class TestMonotonicity:
    """Cost should monotonically increase with token count."""

    def test_input_monotonic(self) -> None:
        def cost(n: int) -> float:
            return compute_cost(Usage(input_tokens=n, output_tokens=0), "claude-opus-4-7")

        assert cost(0) < cost(100) < cost(1000) < cost(100_000)

    def test_output_monotonic(self) -> None:
        def cost(n: int) -> float:
            return compute_cost(Usage(input_tokens=0, output_tokens=n), "claude-opus-4-7")

        assert cost(0) < cost(100) < cost(1000) < cost(100_000)

    def test_cost_is_non_negative(self) -> None:
        assert compute_cost(Usage(input_tokens=12345, output_tokens=678), "claude-opus-4-7") >= 0
