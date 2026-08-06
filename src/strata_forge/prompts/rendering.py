"""Render a :class:`PromptTemplate` to messages + :class:`CacheHints`.

The pipeline:

1. Validate the template's variable declarations
   (:func:`strata_forge.prompts.variables.validate_template_variables`).
2. Confirm the caller supplied every declared variable and didn't pass
   extras (a typo in a variable name is a programming error worth
   catching, not silently ignoring).
3. Partition the call-site variables by their declared section and
   render the stable + dynamic sources through the locked-down sandbox
   env. Partitioning means the stable section can only see stable
   variables — a "wrong section" error from the validator translates
   into an :class:`UndefinedError` at render time (caught by the
   pre-validation above) rather than silently mixing scopes.
4. Build a :class:`StableDynamicSplit` and emit
   :class:`CacheHints` for the target model.
5. Compose the rendered text into ``list[AnyMessage]`` —
   :class:`SystemMessage` for the stable text,
   :class:`UserMessage` for the dynamic text, skipping either when the
   corresponding section rendered to an empty string.

The output bundle is :class:`RenderedPrompt`, designed for direct
hand-off to :class:`strata_forge.llm.LLMClient`. The LLM client consumes the
hints when constructing the provider-specific cache mechanics; the
caller doesn't need to track the hints separately.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from strata_forge.core.errors import ValidationError
from strata_forge.llm.messages import SystemMessage, UserMessage
from strata_forge.prompts.cache_aware import (
    DEFAULT_MIN_CACHEABLE_TOKENS,
    CacheHints,
    StableDynamicSplit,
    emit_cache_hints,
)
from strata_forge.prompts.template import (
    PromptTemplate,
    create_sandboxed_environment,
)
from strata_forge.prompts.variables import validate_template_variables

if TYPE_CHECKING:
    from collections.abc import Mapping

    from strata_forge.llm.messages import AnyMessage

__all__ = [
    "RenderedPrompt",
    "render",
]


@dataclass(frozen=True, slots=True)
class RenderedPrompt:
    """The output of rendering a template — messages + hints + raw split.

    ``messages`` is the wire-format payload to hand to
    :class:`strata_forge.llm.LLMClient`. ``cache_hints`` carries the
    cache-decision the client uses to populate provider-specific
    cache mechanics. ``split`` is exposed so callers can re-fingerprint
    or re-hint without re-rendering — useful for the eval runner when
    it sweeps the same template across multiple models.
    """

    messages: list[AnyMessage]
    cache_hints: CacheHints
    split: StableDynamicSplit


def render(
    template: PromptTemplate,
    variables: Mapping[str, Any] | None = None,
    *,
    model: str | None = None,
    min_cacheable_tokens: int = DEFAULT_MIN_CACHEABLE_TOKENS,
) -> RenderedPrompt:
    """Render ``template`` with ``variables`` into a :class:`RenderedPrompt`.

    Args:
        template: The template to render.
        variables: Values for the declared stable + dynamic variables.
            Every declared variable must be supplied; extras raise.
        model: Logical model name passed through to
            :func:`strata_forge.prompts.cache_aware.emit_cache_hints` for
            accurate tokenization. ``None`` uses the default encoding.
        min_cacheable_tokens: Threshold below which the stable prefix
            won't be flagged for cache. Defaults to
            :data:`strata_forge.prompts.cache_aware.DEFAULT_MIN_CACHEABLE_TOKENS`.

    Returns:
        A :class:`RenderedPrompt` carrying the message list, the
        provider-agnostic cache hints, and the underlying split.

    Raises:
        PromptValidationError: When the template's declared variables
            don't match its body references (re-raised from
            :func:`validate_template_variables`).
        ValidationError: When the call-site ``variables`` mapping is
            missing a declared variable or includes one that wasn't
            declared.
    """
    validate_template_variables(template)

    vars_dict: dict[str, Any] = dict(variables or {})
    declared_stable = set(template.stable_variables)
    declared_dynamic = set(template.dynamic_variables)
    declared = declared_stable | declared_dynamic

    missing = declared - set(vars_dict)
    if missing:
        msg = f"Template {template.name!r} missing variables at render time: {sorted(missing)!r}"
        raise ValidationError(msg)

    extra = set(vars_dict) - declared
    if extra:
        msg = (
            f"Template {template.name!r}: variables {sorted(extra)!r} passed "
            "but not declared on the template"
        )
        raise ValidationError(msg)

    stable_vars = {k: vars_dict[k] for k in template.stable_variables}
    dynamic_vars = {k: vars_dict[k] for k in template.dynamic_variables}

    env = create_sandboxed_environment()
    stable_text = env.from_string(template.stable_section).render(**stable_vars)
    dynamic_text = env.from_string(template.dynamic_section).render(**dynamic_vars)

    split = StableDynamicSplit.build(stable_text=stable_text, dynamic_text=dynamic_text)
    hints = emit_cache_hints(split, model=model, min_cacheable_tokens=min_cacheable_tokens)
    messages = _to_messages(split)

    return RenderedPrompt(messages=messages, cache_hints=hints, split=split)


def _to_messages(split: StableDynamicSplit) -> list[AnyMessage]:
    """Compose a split into the conventional ``[System, User]`` message pair.

    Either side may be empty: a template rendered with no stable text
    yields a single :class:`UserMessage`; a template with no dynamic
    text yields a single :class:`SystemMessage`. A fully empty render
    returns an empty list — the caller can decide whether that's an
    error.
    """
    messages: list[AnyMessage] = []
    if split.stable_text:
        messages.append(SystemMessage(content=split.stable_text))
    if split.dynamic_text:
        messages.append(UserMessage(content=split.dynamic_text))
    return messages
