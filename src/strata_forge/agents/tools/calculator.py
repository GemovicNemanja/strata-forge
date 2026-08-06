"""Safe arithmetic evaluator exposed as a Forge tool.

Backed by an AST whitelist — the LLM can compute expressions like
``2 + 2``, ``(15 * 23) / 4``, ``2**10`` without exposing arbitrary
``eval`` to model output. Anything outside the whitelist (function
calls, attribute access, names) raises :class:`ValueError` before
the underlying ``compile``/``eval`` runs.

The wrapped function is decorated with :func:`strata_forge.llm.tool` so it
plugs into :class:`strata_forge.agents.Agent` directly:

.. code-block:: python

    from strata_forge.agents import Agent
    from strata_forge.agents.tools import calculator

    agent = Agent("math", client=client, tools=[calculator])
"""

from __future__ import annotations

import ast
from typing import Any

from pydantic import BaseModel, Field

from strata_forge.llm.tools import tool

__all__ = [
    "CalculatorArgs",
    "calculator",
]

_ALLOWED_NODE_TYPES: tuple[type[ast.AST], ...] = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Constant,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.Pow,
    ast.USub,
    ast.UAdd,
)


class CalculatorArgs(BaseModel):
    """Arguments for the :data:`calculator` tool."""

    expression: str = Field(
        description=(
            "An arithmetic expression. Operators: + - * / // % ** "
            "and parentheses. Operands must be numeric literals; no "
            "names, function calls, or attribute access."
        ),
        min_length=1,
    )


def _safe_eval(expression: str) -> int | float:
    """Parse ``expression`` against an AST whitelist and evaluate it.

    Raises :class:`ValueError` when the expression contains nodes
    outside the whitelist (names, calls, attribute access, …) or
    non-numeric constants (strings, ``None``, …).
    """
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        msg = f"invalid arithmetic expression: {exc.msg}"
        raise ValueError(msg) from exc

    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODE_TYPES):
            msg = f"disallowed expression element: {type(node).__name__}"
            raise ValueError(msg)
        if isinstance(node, ast.Constant) and not isinstance(node.value, int | float):
            msg = f"only numeric constants are allowed; got {type(node.value).__name__}"
            raise ValueError(msg)
    # The compile is over an AST whose nodes have been individually
    # checked against the whitelist above; eval here is intentional.
    compiled: Any = compile(tree, "<calculator>", "eval")
    return eval(compiled, {"__builtins__": {}}, {})  # noqa: S307 — AST-whitelisted


@tool
async def calculator(args: CalculatorArgs) -> str:
    """Evaluate an arithmetic expression and return the numeric result as a string."""
    return str(_safe_eval(args.expression))
