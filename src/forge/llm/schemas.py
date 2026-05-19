"""Structured-output primitives: Pydantic ↔ provider JSON schema.

Three provider strategies, each represented by a pure function that takes
a Pydantic model class and returns the payload shape the provider's API
expects:

- **OpenAI** has a native ``response_format={"type": "json_schema", ...}``
  channel with a strict mode that the API enforces server-side. See
  :func:`to_openai_response_format`.
- **Anthropic** has no native channel; the recommended pattern is a forced
  tool call where the tool's ``input_schema`` mirrors the desired output
  shape. See :func:`to_anthropic_forced_tool_schema`.
- **Gemini** has ``response_schema`` paired with
  ``response_mime_type="application/json"``. See
  :func:`to_gemini_response_schema`.

When no native channel is usable — either the model doesn't support one
or a strict-mode call returned malformed JSON — fall back to a reprompt
loop: build a system instruction with :func:`make_reprompt_instruction`,
send a normal completion, then validate the text via
:func:`parse_json_response`. The full retry loop lives in ``LLMClient``;
this module only ships the primitives. :exc:`StructuredOutputError` is
the terminal failure when the reprompt loop is exhausted.
"""

from __future__ import annotations

import json
from typing import Any, cast

from pydantic import BaseModel
from pydantic import ValidationError as PydanticValidationError

from forge.core.errors import ForgeError, ValidationError
from forge.llm.providers.anthropic import to_anthropic_tool_schema

__all__ = [
    "StructuredOutputError",
    "make_reprompt_instruction",
    "parse_json_response",
    "pydantic_to_json_schema",
    "to_anthropic_forced_tool_schema",
    "to_gemini_response_schema",
    "to_openai_response_format",
]


class StructuredOutputError(ForgeError):
    """A structured-output reprompt loop exceeded its retry budget.

    Native-channel failures (e.g. Pydantic-validation errors on a model's
    raw JSON response) raise :exc:`forge.core.errors.ValidationError`
    directly. ``StructuredOutputError`` is reserved for the case where
    the reprompt fallback itself ran out of attempts without producing a
    valid response.
    """

    def __init__(
        self,
        message: str,
        *,
        attempts: int,
        last_error: Exception | None = None,
    ) -> None:
        super().__init__(message)
        self.attempts = attempts
        self.last_error = last_error


# ---------------------------------------------------------------------------
# Pydantic → JSON Schema
# ---------------------------------------------------------------------------


def pydantic_to_json_schema(
    model_cls: type[BaseModel],
    *,
    inline_refs: bool = True,
) -> dict[str, Any]:
    """Return a JSON Schema for ``model_cls``.

    Pydantic's :meth:`BaseModel.model_json_schema` already produces a
    JSON Schema, but it puts nested models under ``$defs`` and references
    them with ``$ref``. Some providers (notably older Gemini versions)
    don't follow ``$ref`` indirection, so we inline by default.

    Args:
        model_cls: Pydantic model class to convert.
        inline_refs: When ``True`` (the default), replace each
            ``{"$ref": "#/$defs/Foo"}`` with the resolved definition. The
            top-level ``$defs`` mapping is removed after inlining.

    Returns:
        JSON Schema dict.
    """
    schema = dict(model_cls.model_json_schema())
    if inline_refs:
        schema = _inline_refs(schema)
    return schema


def _inline_refs(schema: dict[str, Any]) -> dict[str, Any]:
    defs_raw = schema.pop("$defs", None)
    if not isinstance(defs_raw, dict):
        return schema
    defs = cast("dict[str, Any]", defs_raw)

    def _resolve(obj: Any) -> Any:
        if isinstance(obj, dict):
            obj_dict = cast("dict[str, Any]", obj)
            ref = obj_dict.get("$ref")
            if isinstance(ref, str) and ref.startswith("#/$defs/"):
                name = ref[len("#/$defs/") :]
                resolved = defs.get(name)
                if resolved is not None:
                    # Merge sibling keys (e.g. `description`) over the resolved
                    # definition so callers can annotate a reference site.
                    merged: dict[str, Any] = dict(_resolve(resolved))
                    for k, v in obj_dict.items():
                        if k != "$ref":
                            merged[k] = _resolve(v)
                    return merged
            return {k: _resolve(v) for k, v in obj_dict.items()}
        if isinstance(obj, list):
            obj_list = cast("list[Any]", obj)
            return [_resolve(item) for item in obj_list]
        return obj

    return cast("dict[str, Any]", _resolve(schema))


# ---------------------------------------------------------------------------
# OpenAI — response_format
# ---------------------------------------------------------------------------


def to_openai_response_format(
    model_cls: type[BaseModel],
    *,
    name: str | None = None,
    strict: bool = True,
) -> dict[str, Any]:
    """Build OpenAI's ``response_format`` payload for structured output.

    Result:

    .. code-block:: python

        {
            "type": "json_schema",
            "json_schema": {
                "name": "<model name>",
                "schema": <inlined schema>,
                "strict": True,
            },
        }

    Args:
        model_cls: Pydantic model class describing the desired output.
        name: Schema name. Defaults to ``model_cls.__name__``.
        strict: When ``True`` (the default), tightens the schema to the
            subset OpenAI's strict mode accepts: every object gets
            ``additionalProperties: false`` and every property is listed
            in ``required`` (OpenAI's structured-output API rejects
            schemas otherwise).
    """
    schema_name = name if name is not None else model_cls.__name__
    schema = pydantic_to_json_schema(model_cls)
    if strict:
        schema = _enforce_openai_strict(schema)
    return {
        "type": "json_schema",
        "json_schema": {
            "name": schema_name,
            "schema": schema,
            "strict": strict,
        },
    }


def _enforce_openai_strict(schema: dict[str, Any]) -> dict[str, Any]:
    def _walk(obj: Any) -> Any:
        if isinstance(obj, dict):
            obj_dict = cast("dict[str, Any]", obj)
            walked: dict[str, Any] = {k: _walk(v) for k, v in obj_dict.items()}
            if walked.get("type") == "object":
                walked.setdefault("additionalProperties", False)
                props = walked.get("properties")
                if isinstance(props, dict) and props:
                    walked["required"] = list(cast("dict[str, Any]", props))
            return walked
        if isinstance(obj, list):
            obj_list = cast("list[Any]", obj)
            return [_walk(x) for x in obj_list]
        return obj

    return cast("dict[str, Any]", _walk(schema))


# ---------------------------------------------------------------------------
# Anthropic — forced tool
# ---------------------------------------------------------------------------


_DEFAULT_FORCED_TOOL_DESCRIPTION = (
    "Emit your response as a JSON object that matches this tool's input schema."
)


def to_anthropic_forced_tool_schema(
    model_cls: type[BaseModel],
    *,
    name: str | None = None,
    description: str = _DEFAULT_FORCED_TOOL_DESCRIPTION,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build the ``(tool, tool_choice)`` pair Anthropic needs for structured output.

    Anthropic has no native JSON-mode channel; the workaround is to
    define a single tool whose ``input_schema`` is the desired output
    shape and force the model to call it. The returned ``tool_choice``
    pins the call to this tool.

    Args:
        model_cls: Pydantic model class describing the desired output.
        name: Tool name. Defaults to ``model_cls.__name__``.
        description: Tool description shown to the model. The default
            tells the model to emit JSON matching the schema.

    Returns:
        A two-tuple ``(tool_schema, tool_choice)``:

        - ``tool_schema`` follows the standard Anthropic tool shape
          (``{"name", "description", "input_schema"}``).
        - ``tool_choice`` follows the OpenAI spec
          (``{"type": "function", "function": {"name": <name>}}``);
          LiteLLM validates incoming tool_choice against the OpenAI
          schema and translates it to the provider's native shape
          (Anthropic, Vertex, Bedrock) on the way out.
    """
    tool_name = name if name is not None else model_cls.__name__
    tool_schema = to_anthropic_tool_schema(
        name=tool_name,
        description=description,
        parameters_schema=pydantic_to_json_schema(model_cls),
    )
    tool_choice: dict[str, Any] = {
        "type": "function",
        "function": {"name": tool_name},
    }
    return tool_schema, tool_choice


# ---------------------------------------------------------------------------
# Gemini — response_schema
# ---------------------------------------------------------------------------


# Gemini's response_schema accepts an OpenAPI-3 subset of JSON Schema. The
# keys below are part of JSON Schema but not the OpenAPI subset, so we strip
# them before sending.
_GEMINI_UNSUPPORTED_KEYS: frozenset[str] = frozenset(
    {
        "$schema",
        "$defs",
        "additionalProperties",
        "discriminator",
    }
)


def to_gemini_response_schema(model_cls: type[BaseModel]) -> dict[str, Any]:
    """Build Gemini's ``response_schema`` value for structured output.

    Gemini's structured-output API expects an OpenAPI-3 subset of JSON
    Schema. We start from the inlined Pydantic schema and strip keys
    Gemini doesn't recognize (``$schema``, ``$defs``,
    ``additionalProperties``, ``discriminator``). The result is passed
    to Gemini's ``generation_config.response_schema``; the caller must
    also set ``response_mime_type="application/json"``.
    """
    schema = pydantic_to_json_schema(model_cls)
    return _strip_keys(schema, _GEMINI_UNSUPPORTED_KEYS)


def _strip_keys(obj: Any, keys: frozenset[str]) -> Any:
    if isinstance(obj, dict):
        obj_dict = cast("dict[str, Any]", obj)
        return {k: _strip_keys(v, keys) for k, v in obj_dict.items() if k not in keys}
    if isinstance(obj, list):
        obj_list = cast("list[Any]", obj)
        return [_strip_keys(item, keys) for item in obj_list]
    return obj


# ---------------------------------------------------------------------------
# Reprompt fallback
# ---------------------------------------------------------------------------


def make_reprompt_instruction(
    model_cls: type[BaseModel],
    *,
    parse_error: str | None = None,
) -> str:
    """Build a system-message body asking the model to emit schema-matching JSON.

    Suitable for prepending to the system prompt (or sending as a
    dedicated user message) when no native structured-output channel is
    available, or when a previous attempt failed validation.

    Args:
        model_cls: Pydantic model class describing the desired output.
        parse_error: When provided (typically after a failed parse),
            the message includes the error so the model can correct
            itself. Pass the formatted error string from the prior
            :func:`parse_json_response` call.
    """
    schema_json = json.dumps(pydantic_to_json_schema(model_cls), indent=2)
    lines = [
        "Respond with a single JSON object matching this schema:",
        "",
        schema_json,
        "",
        "Output ONLY the JSON object — no prose, no explanation, no markdown fences.",
    ]
    if parse_error:
        lines.extend(
            [
                "",
                "Your previous response was invalid. Error:",
                parse_error,
                "",
                "Return a corrected JSON object.",
            ]
        )
    return "\n".join(lines)


def parse_json_response[M: BaseModel](
    text: str,
    model_cls: type[M],
) -> M:
    """Parse ``text`` as JSON and validate against ``model_cls``.

    Strips leading/trailing whitespace and a single surrounding
    ``\\`\\`\\`json … \\`\\`\\``` markdown fence (a common
    non-strict-mode quirk) before parsing.

    Raises:
        ValidationError: If the text isn't valid JSON, or if the parsed
            value doesn't satisfy ``model_cls``. The underlying
            ``json.JSONDecodeError`` or Pydantic ``ValidationError`` is
            chained via ``__cause__``.
    """
    cleaned = _strip_markdown_fence(text)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        msg = f"Response is not valid JSON: {exc.msg} (line {exc.lineno}, col {exc.colno})"
        raise ValidationError(msg) from exc
    try:
        return model_cls.model_validate(data)
    except PydanticValidationError as exc:
        msg = f"Response does not match {model_cls.__name__} schema: {exc}"
        raise ValidationError(msg) from exc


def _strip_markdown_fence(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    # Remove the opening fence — may carry a language tag (``` or ```json).
    newline = stripped.find("\n")
    if newline == -1:
        return stripped
    body = stripped[newline + 1 :]
    # Drop closing fence if present.
    if body.rstrip().endswith("```"):
        body = body.rstrip()[: -len("```")]
    return body.strip()
