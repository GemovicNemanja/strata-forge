"""Chat-template keyword arguments a run declares: their rules, their rendering and their record.

A model's chat template is a Jinja template that ships with its tokenizer. Besides the messages it
reads named variables (``enable_thinking``, ``documents``, a judge's ``guardian_config``), which a
caller passes as keyword arguments. A run declares those in ``hyperparams.chat_template_kwargs``;
this module holds what every engine that renders a chat template does with them:

- **Only data, never a renderer option.** A key that names one of ``apply_chat_template``'s own
  parameters (``chat_template``, ``tokenize``, ``add_generation_prompt`` …), the conversation, or a
  special token the template would otherwise read from the tokenizer is refused: those change how
  the template is chosen or rendered rather than what it says, and the run's record could not
  describe them. ``documents`` is the exception, because templates read it as a variable.
- **Only what the template reads.** Both transformers and vLLM drop a keyword the template never
  mentions without a word (vLLM filters it by the template's undeclared variables, transformers
  simply never looks it up), so a run that declared ``enable_thinking=false`` for a template
  without that switch would record a setting its model never saw. :func:`unread_kwargs` names
  those keys so the runner can refuse them once the model's template is known.
- **Per-row values.** A string anywhere inside a value may hold ``{name}`` placeholders, rendered
  per row by the same non-executing substitution as the prompt (a list column is joined), so a
  judge's ``documents[0].text`` can be a row's premises.
- **A canonical form.** Objects are rendered with their keys sorted at every depth, so a spec that
  was stored and read back (where key order is not preserved) renders the same prompt, and the
  digest the manifest records is over that same canonical JSON.

Everything here is pure: jinja2 is a core dependency, and nothing is rendered with it here, only
parsed.
"""

from __future__ import annotations

import hashlib
import json
import re
from functools import lru_cache
from typing import TYPE_CHECKING, Any, cast

import jinja2.ext
import jinja2.meta
import jinja2.nodes
import jinja2.parser
import jinja2.sandbox

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from pydantic import JsonValue

__all__ = [
    "MAX_KWARGS",
    "MAX_KWARGS_JSON_BYTES",
    "RESERVED_KWARGS",
    "canonical",
    "kwargs_digest",
    "kwargs_placeholders",
    "kwargs_problem",
    "nests_deeper_than",
    "render_kwargs",
    "template_digest",
    "template_variables",
    "unread_kwargs",
]

#: How many keyword arguments a run may declare, and how large they may be as canonical JSON.
MAX_KWARGS = 16
MAX_KWARGS_JSON_BYTES = 16 * 1024
# Containers nest at most this deep: enough for a list of documents of objects.
_MAX_DEPTH = 6
_KEY_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,63}")
#: Keys refused as template kwargs. ``apply_chat_template``'s own parameters (transformers 5.8,
#: ``tokenization_utils_base.py``; ``documents`` is left out because templates read it as data),
#: the conversation it renders, and the special tokens it passes the template from the tokenizer,
#: which a keyword of the same name would override.
RESERVED_KWARGS = frozenset(
    {
        "conversation",
        "messages",
        "tools",
        "chat_template",
        "add_generation_prompt",
        "continue_final_message",
        "tokenize",
        "padding",
        "truncation",
        "max_length",
        "return_tensors",
        "return_dict",
        "return_assistant_tokens_mask",
        "tokenizer_kwargs",
        "bos_token",
        "eos_token",
        "unk_token",
        "sep_token",
        "pad_token",
        "cls_token",
        "mask_token",
        "additional_special_tokens",
    }
)
# The same bare `{name}` placeholder as the prompt template: nothing else is substituted.
_PLACEHOLDER_RE = re.compile(r"\{([A-Za-z0-9_]+)\}")


def canonical(value: JsonValue) -> JsonValue:
    """``value`` with every object's keys sorted, at every depth."""
    if isinstance(value, dict):
        return {key: canonical(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [canonical(item) for item in value]
    return value


def _canonical_json(kwargs: Mapping[str, JsonValue]) -> str:
    return json.dumps(
        canonical(dict(kwargs)), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )


def kwargs_digest(kwargs: Mapping[str, JsonValue] | None) -> str | None:
    """The sha256 of the declared kwargs as canonical JSON, or ``None`` when none are declared."""
    if not kwargs:
        return None
    return hashlib.sha256(_canonical_json(kwargs).encode()).hexdigest()


def template_digest(template: str) -> str:
    """The sha256 of a chat template's text."""
    return hashlib.sha256(template.encode()).hexdigest()


def nests_deeper_than(value: object, limit: int) -> bool:
    """Whether ``value`` holds containers nested more than ``limit`` deep, each object or list one
    level, found WITHOUT recursion and stopping at the first level past ``limit``.

    Data from the wire can be nested arbitrarily deep, and every recursive walk over it (this
    module's own, ``json.dumps``, a schema validator) raises ``RecursionError`` past the
    interpreter's limit, so the depth is bounded first, iteratively.
    """
    stack: list[tuple[object, int]] = [(value, 0)]
    while stack:
        item, depth = stack.pop()
        if isinstance(item, dict | list):
            if depth + 1 > limit:
                return True
            children = (
                cast("dict[str, object]", item).values()
                if isinstance(item, dict)
                else cast("list[object]", item)
            )
            stack.extend((child, depth + 1) for child in children)
    return False


def kwargs_problem(kwargs: Mapping[str, JsonValue]) -> str | None:
    """Why ``kwargs`` cannot be a run's chat-template kwargs, or ``None`` when they can."""
    if not kwargs:
        return "chat_template_kwargs is empty; omit it instead"
    if any(nests_deeper_than(value, _MAX_DEPTH) for value in kwargs.values()):
        return f"chat_template_kwargs values nest at most {_MAX_DEPTH} levels deep"
    if len(kwargs) > MAX_KWARGS:
        return f"chat_template_kwargs holds at most {MAX_KWARGS} keys"
    bad = sorted(key for key in kwargs if not _KEY_RE.fullmatch(key))
    if bad:
        return f"chat_template_kwargs keys must be identifiers: {bad}"
    reserved = sorted(set(kwargs) & RESERVED_KWARGS)
    if reserved:
        return (
            f"chat_template_kwargs cannot set {reserved}: they control how the template is chosen "
            "or rendered, not what it says"
        )
    if "enable_thinking" in kwargs and not isinstance(kwargs["enable_thinking"], bool):
        return "chat_template_kwargs.enable_thinking must be true or false"
    if len(_canonical_json(kwargs).encode()) > MAX_KWARGS_JSON_BYTES:
        return f"chat_template_kwargs is at most {MAX_KWARGS_JSON_BYTES} bytes of JSON"
    return None


def _strings(value: JsonValue) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [text for item in value.values() for text in _strings(item)]
    if isinstance(value, list):
        return [text for item in value for text in _strings(item)]
    return []


def kwargs_placeholders(kwargs: Mapping[str, JsonValue] | None) -> set[str]:
    """Every ``{name}`` placeholder in the string values of ``kwargs``."""
    if not kwargs:
        return set()
    return {
        match.group(1)
        for value in kwargs.values()
        for text in _strings(value)
        for match in _PLACEHOLDER_RE.finditer(text)
    }


def render_kwargs(
    kwargs: Mapping[str, JsonValue], render: Callable[[str], str]
) -> dict[str, JsonValue]:
    """``kwargs`` in canonical order with ``render`` applied to every string value."""

    def _walk(value: JsonValue) -> JsonValue:
        if isinstance(value, str):
            return render(value)
        if isinstance(value, dict):
            return {key: _walk(value[key]) for key in sorted(value)}
        if isinstance(value, list):
            return [_walk(item) for item in value]
        return value

    return {key: _walk(kwargs[key]) for key in sorted(kwargs)}


class _GenerationTag(jinja2.ext.Extension):
    """Parses transformers' ``{% generation %}`` block, so a template that uses it can be read."""

    tags = {"generation"}  # noqa: RUF012 - jinja2's Extension API declares it as a class set

    def parse(self, parser: jinja2.parser.Parser) -> jinja2.nodes.Node:
        lineno = next(parser.stream).lineno
        body = parser.parse_statements(("name:endgeneration",), drop_needle=True)
        call = self.call_method("_noop")
        return jinja2.nodes.CallBlock(call, [], [], body).set_lineno(lineno)


@lru_cache(maxsize=8)
def template_variables(template: str) -> frozenset[str]:
    """The variables a chat template reads but does not set itself, by parsing it.

    Parsed with the environment transformers renders it in (sandboxed, ``trim_blocks``,
    ``lstrip_blocks``, loop controls and the ``generation`` tag), and the same way vLLM decides
    which request kwargs to keep (``vllm/renderers/hf.py``, ``_resolve_chat_template_kwargs``).
    Nothing is rendered.
    """
    env = jinja2.sandbox.ImmutableSandboxedEnvironment(
        trim_blocks=True,
        lstrip_blocks=True,
        extensions=[_GenerationTag, jinja2.ext.loopcontrols],
    )
    parsed = env.parse(template)
    return frozenset(jinja2.meta.find_undeclared_variables(parsed))


def unread_kwargs(kwargs: Mapping[str, Any] | None, template: str) -> list[str]:
    """The declared kwargs the template never reads, sorted; empty when it reads them all."""
    if not kwargs:
        return []
    return sorted(set(kwargs) - template_variables(template))
