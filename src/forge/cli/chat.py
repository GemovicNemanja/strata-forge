"""``forge chat`` — one-shot or interactive completion against an LLM.

The command builds an :class:`LLMClient` for the requested model
(and optional provider pin) and either:

- runs a single prompt and prints the response when ``--message``
  is set, or
- enters a small interactive REPL that lets the user type turns
  and see streamed assistant output otherwise.

Stays minimal on purpose: no tools, no structured output, no
streaming UI tricks. For anything richer, drop into Python and
use :class:`LLMClient` directly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import typer
from rich.console import Console

from forge.cli.helpers import error_exit, run_async

if TYPE_CHECKING:
    from forge.llm.client import LLMClient
    from forge.llm.messages import AnyMessage

__all__ = ["chat"]


_PROMPT = "[bold cyan]you[/]"
_ASSISTANT = "[bold green]assistant[/]"


def chat(
    model: str = typer.Option(
        ...,
        "--model",
        "-m",
        help="Logical model id from the Forge registry (e.g. claude-opus-4-7).",
    ),
    provider: str | None = typer.Option(
        None,
        "--provider",
        "-p",
        help="Pin a specific provider route (anthropic / openai / vertex / ...).",
    ),
    message: str | None = typer.Option(
        None,
        "--message",
        "-q",
        help="Single-shot message. Runs one completion and exits.",
    ),
    system: str | None = typer.Option(
        None,
        "--system",
        "-s",
        help="Optional system prompt prepended to every conversation.",
    ),
    temperature: float | None = typer.Option(
        None, "--temperature", "-t", help="Sampling temperature."
    ),
    max_tokens: int | None = typer.Option(None, "--max-tokens", help="Per-completion output cap."),
) -> None:
    """Run a one-shot or interactive chat session against an LLM."""
    run_async(
        _run_chat(
            model=model,
            provider=provider,
            message=message,
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    )


async def _run_chat(
    *,
    model: str,
    provider: str | None,
    message: str | None,
    system: str | None,
    temperature: float | None,
    max_tokens: int | None,
) -> None:
    from typing import Any

    from forge.llm.client import LLMClient
    from forge.llm.messages import Message

    _registry_module: Any = __import__("forge.llm.registry", fromlist=["registry"])
    _registry: Any = _registry_module.registry
    try:
        _registry.resolve(model)
    except Exception as exc:
        error_exit(f"unknown model {model!r}: {exc}")

    if provider is not None:
        provider_typed: Any = provider
    else:
        provider_typed = None

    client = LLMClient(model=model, provider=provider_typed)
    history: list[AnyMessage] = []
    if system is not None:
        history.append(Message.system(system))

    if message is not None:
        history.append(Message.user(message))
        await _one_completion(client, history, temperature=temperature, max_tokens=max_tokens)
        return

    await _interactive_loop(client, history, temperature=temperature, max_tokens=max_tokens)


async def _one_completion(
    client: LLMClient,
    history: list[AnyMessage],
    *,
    temperature: float | None,
    max_tokens: int | None,
) -> None:
    console = Console()
    response = await client.complete(
        messages=history,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    console.print(response.text or "")
    console.print(
        f"\n[dim]model={response.route.model} provider={response.route.provider} "
        f"in={response.usage.input_tokens} out={response.usage.output_tokens} "
        f"cost=${response.cost_usd:.5f} latency={response.latency_ms:.0f}ms[/]"
    )


async def _interactive_loop(
    client: LLMClient,
    history: list[AnyMessage],
    *,
    temperature: float | None,
    max_tokens: int | None,
) -> None:
    from forge.llm.messages import Message

    console = Console()
    console.print("[dim]forge chat — (blank line or Ctrl-D to exit)[/]")
    while True:
        try:
            user_input = console.input(f"{_PROMPT} > ").strip()
        except EOFError, KeyboardInterrupt:
            console.print("\n[dim]exit[/]")
            return
        if not user_input:
            console.print("[dim]exit[/]")
            return
        history.append(Message.user(user_input))
        response = await client.complete(
            messages=history,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        history.append(Message.assistant(response.text or ""))
        console.print(f"{_ASSISTANT} > {response.text or ''}")
