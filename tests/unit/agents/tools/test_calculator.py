"""Unit tests for `strata_forge.agents.tools.calculator`."""

from __future__ import annotations

import pytest

from strata_forge.agents.tools.calculator import (
    CalculatorArgs,
    _safe_eval,  # pyright: ignore[reportPrivateUsage]
    calculator,
)
from strata_forge.core.errors import ValidationError

# ---------------------------------------------------------------------------
# _safe_eval — happy path
# ---------------------------------------------------------------------------


class TestSafeEval:
    def test_int_addition(self) -> None:
        assert _safe_eval("2 + 2") == 4

    def test_float_division(self) -> None:
        assert _safe_eval("10 / 4") == 2.5

    def test_floor_division(self) -> None:
        assert _safe_eval("10 // 3") == 3

    def test_modulo(self) -> None:
        assert _safe_eval("10 % 3") == 1

    def test_power(self) -> None:
        assert _safe_eval("2 ** 10") == 1024

    def test_parentheses(self) -> None:
        assert _safe_eval("(2 + 3) * 4") == 20

    def test_unary_negation(self) -> None:
        assert _safe_eval("-5") == -5
        assert _safe_eval("-(2 + 3)") == -5

    def test_unary_positive(self) -> None:
        assert _safe_eval("+5") == 5

    def test_mixed_int_float(self) -> None:
        assert _safe_eval("3 + 0.5") == 3.5

    def test_compound(self) -> None:
        # (15 * 23) / 4 - 2**3 = 345/4 - 8 = 86.25 - 8 = 78.25
        assert _safe_eval("(15 * 23) / 4 - 2**3") == 78.25


# ---------------------------------------------------------------------------
# _safe_eval — rejection of disallowed nodes
# ---------------------------------------------------------------------------


class TestSafeEvalRejections:
    def test_rejects_function_call(self) -> None:
        with pytest.raises(ValueError, match="disallowed"):
            _safe_eval("__import__('os').system('ls')")

    def test_rejects_name_reference(self) -> None:
        with pytest.raises(ValueError, match="disallowed"):
            _safe_eval("x + 1")

    def test_rejects_attribute_access(self) -> None:
        with pytest.raises(ValueError, match="disallowed"):
            _safe_eval("(1).bit_length()")

    def test_rejects_list_literal(self) -> None:
        with pytest.raises(ValueError, match="disallowed"):
            _safe_eval("[1, 2, 3]")

    def test_rejects_string_constant(self) -> None:
        with pytest.raises(ValueError, match="numeric"):
            _safe_eval("'hello'")

    def test_rejects_boolean_constant(self) -> None:
        # bool is a subclass of int in Python but ast.Constant carries
        # bool literals; our isinstance check on (int, float) would
        # accept True / False because bool is an int. Document the
        # current behaviour: bool passes through and yields 0/1.
        # This test pins that — if we later tighten, update here.
        assert _safe_eval("True + True") == 2

    def test_rejects_invalid_syntax(self) -> None:
        with pytest.raises(ValueError, match="invalid"):
            _safe_eval("2 +")

    def test_rejects_empty_string(self) -> None:
        with pytest.raises(ValueError, match="invalid"):
            _safe_eval("")

    def test_rejects_comparison(self) -> None:
        with pytest.raises(ValueError, match="disallowed"):
            _safe_eval("2 > 1")


# ---------------------------------------------------------------------------
# Tool integration
# ---------------------------------------------------------------------------


class TestCalculatorTool:
    def test_tool_name_and_description(self) -> None:
        assert calculator.name == "calculator"
        assert calculator.description  # non-empty

    def test_tool_parameters_model(self) -> None:
        assert calculator.parameters_model is CalculatorArgs

    async def test_tool_invoke_happy_path(self) -> None:
        result = await calculator.invoke({"expression": "2 + 2"})
        assert result == "4"

    async def test_tool_invoke_returns_string(self) -> None:
        # The tool always returns a string so the LLM gets a consistent type.
        result = await calculator.invoke({"expression": "10 / 4"})
        assert isinstance(result, str)
        assert result == "2.5"

    async def test_tool_invoke_rejects_empty_expression(self) -> None:
        # Pydantic min_length=1 rejects empty string before the AST runs.
        with pytest.raises(ValidationError):
            await calculator.invoke({"expression": ""})

    async def test_tool_invoke_passes_disallowed_to_safe_eval(self) -> None:
        # Pydantic validation passes; the AST whitelist then rejects.
        with pytest.raises(ValueError, match="disallowed"):
            await calculator.invoke({"expression": "__import__('os')"})
