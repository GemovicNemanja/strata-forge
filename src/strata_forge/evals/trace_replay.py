"""Replay a Langfuse trace through a new model / prompt.

Pulls the original input messages from a recorded Langfuse trace,
optionally rewrites the system message or swaps in a different model,
and re-runs the call through :class:`LLMClient`. The result captures
both the original response and the new one so callers can diff them
without round-tripping Langfuse twice.

The Langfuse Python SDK is sync; this module wraps the read call in
``asyncio.to_thread`` so the public surface stays async-only. The
SDK is imported lazily — :mod:`strata_forge.evals.trace_replay` imports
cleanly without the ``[langfuse]`` extra; the ``ImportError``
surfaces only when a caller actually invokes :func:`replay_trace`.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, cast

from pydantic import BaseModel, ConfigDict

from strata_forge.llm.messages import AssistantMessage, SystemMessage, UserMessage
from strata_forge.llm.responses import (
    LLMResponse,  # noqa: TC001 — Pydantic needs runtime resolution
)

if TYPE_CHECKING:
    from strata_forge.llm.client import LLMClient
    from strata_forge.llm.messages import AnyMessage

__all__ = [
    "ReplayOverrides",
    "ReplayResult",
    "extract_messages_from_trace",
    "replay_trace",
]


class ReplayOverrides(BaseModel):
    """Optional rewrites applied to the trace before re-running.

    Each field, when set, replaces the corresponding piece of the
    extracted message list:

    - ``system_message``: replaces all SystemMessages with this text.
    - ``user_message``: replaces the *last* UserMessage's text with
      this value. The earlier conversation history is preserved.
    - ``temperature`` / ``max_tokens`` / ``top_p``: forwarded to
      :meth:`LLMClient.complete` so the replay can re-sample.
    """

    model_config = ConfigDict(extra="forbid")

    system_message: str | None = None
    user_message: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    top_p: float | None = None


class ReplayResult(BaseModel):
    """Bundle the original trace's response with the new one."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    original_trace_id: str
    original_response_text: str
    new_response: LLMResponse


def _build_langfuse_client(client: Any | None) -> Any:
    """Return the supplied Langfuse client, or build one from settings."""
    if client is not None:
        return client
    from strata_forge.config import get_settings

    config = get_settings().langfuse
    if not config.enabled:
        msg = (
            "Langfuse is not configured. Set LANGFUSE_HOST, "
            "LANGFUSE_PUBLIC_KEY, and LANGFUSE_SECRET_KEY (or pass "
            "langfuse_client=... explicitly)."
        )
        raise RuntimeError(msg)
    try:
        from langfuse import (  # pyright: ignore[reportMissingImports]
            Langfuse,  # pyright: ignore[reportUnknownVariableType]
        )
    except ImportError as exc:
        msg = (
            "The [langfuse] extra is required for trace replay. "
            "Install it with: pip install 'strata-forge[langfuse]'."
        )
        raise ImportError(msg) from exc
    if config.public_key is None or config.secret_key is None:  # pragma: no cover
        msg = "Langfuse keys missing despite enabled=True"
        raise RuntimeError(msg)
    return Langfuse(  # pyright: ignore[reportUnknownVariableType]
        host=config.host,
        public_key=config.public_key.get_secret_value(),
        secret_key=config.secret_key.get_secret_value(),
    )


def extract_messages_from_trace(trace: Any) -> tuple[list[AnyMessage], str]:
    """Pull the input message list and the recorded response text from a trace.

    Langfuse trace shapes vary across SDK versions; this helper
    accepts the most common shape — ``trace.input`` is either a
    list of ``{"role": ..., "content": ...}`` dicts or a single
    string; ``trace.output`` is either a string or a dict with a
    text-shaped field. Unknown shapes raise :class:`ValueError`.

    Returns the messages plus the original response text (so the
    replay result can include both old and new outputs).
    """
    raw_input: Any = getattr(trace, "input", None)
    raw_output: Any = getattr(trace, "output", None)

    messages: list[AnyMessage] = []
    if isinstance(raw_input, list):
        entries = cast("list[Any]", raw_input)
        for entry in entries:
            if not isinstance(entry, dict):
                msg = f"Unexpected trace input entry type: {type(entry).__name__}"  # type: ignore[unreachable]
                raise ValueError(msg)
            entry_dict = cast("dict[str, Any]", entry)
            role = entry_dict.get("role")
            content = entry_dict.get("content", "")
            if role == "system":
                messages.append(SystemMessage(content=str(content)))
            elif role == "user":
                messages.append(UserMessage(content=str(content)))
            elif role == "assistant":
                messages.append(AssistantMessage(content=str(content)))
            else:
                msg = f"Unknown role in trace input: {role!r}"
                raise ValueError(msg)
    elif isinstance(raw_input, str):
        messages.append(UserMessage(content=raw_input))
    else:
        msg = f"Unsupported trace.input shape: {type(raw_input).__name__}"
        raise ValueError(msg)

    if isinstance(raw_output, str):
        original_response = raw_output
    elif isinstance(raw_output, dict):
        out_dict: dict[str, Any] = raw_output  # type: ignore[assignment]
        original_response = str(out_dict.get("content") or out_dict.get("text") or "")
    elif raw_output is None:
        original_response = ""
    else:
        original_response = str(raw_output)

    return messages, original_response


def _apply_overrides(
    messages: list[AnyMessage],
    overrides: ReplayOverrides | None,
) -> list[AnyMessage]:
    if overrides is None:
        return messages
    rewritten: list[AnyMessage] = []
    for msg in messages:
        if overrides.system_message is not None and isinstance(msg, SystemMessage):
            rewritten.append(SystemMessage(content=overrides.system_message))
        else:
            rewritten.append(msg)
    if overrides.user_message is not None:
        # Replace the LAST user message; preserve earlier ones.
        for i in range(len(rewritten) - 1, -1, -1):
            if isinstance(rewritten[i], UserMessage):
                rewritten[i] = UserMessage(content=overrides.user_message)
                break
    return rewritten


async def replay_trace(
    *,
    trace_id: str,
    client: LLMClient,
    overrides: ReplayOverrides | None = None,
    langfuse_client: Any | None = None,
) -> ReplayResult:
    """Fetch a Langfuse trace, optionally rewrite it, re-run via ``client``.

    Args:
        trace_id: The Langfuse trace identifier to replay.
        client: The :class:`LLMClient` that runs the replay. Typically
            a different model than the one that produced the original
            trace — that's the point of replay.
        overrides: Optional :class:`ReplayOverrides` modifying the
            messages and / or sampling parameters before re-running.
        langfuse_client: An already-built Langfuse client (handy for
            tests). When omitted, one is built from
            :class:`strata_forge.config.LangfuseConfig`.

    Returns:
        A :class:`ReplayResult` carrying ``original_trace_id``,
        ``original_response_text``, and the fresh :class:`LLMResponse`.

    Raises:
        ImportError: When ``langfuse_client`` is not supplied and the
            ``[langfuse]`` extra isn't installed.
        RuntimeError: When ``langfuse_client`` is not supplied and
            Langfuse isn't configured via settings.
        ValueError: When the trace's input shape isn't recognized.
    """
    lf_client = _build_langfuse_client(langfuse_client)

    def _fetch() -> Any:
        return lf_client.get_trace(id=trace_id)

    trace = await asyncio.to_thread(_fetch)
    messages, original_response = extract_messages_from_trace(trace)
    messages = _apply_overrides(messages, overrides)

    temperature = overrides.temperature if overrides else None
    max_tokens = overrides.max_tokens if overrides else None
    top_p = overrides.top_p if overrides else None

    new_response = await client.complete(
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
    )
    return ReplayResult(
        original_trace_id=trace_id,
        original_response_text=original_response,
        new_response=new_response,
    )
