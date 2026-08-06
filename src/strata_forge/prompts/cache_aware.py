"""Stable/dynamic split + per-provider cache-hint emission.

:doc:`ADR 0007 <../../../docs/architecture/adr/0007-stable-prefix-dynamic-suffix-prompts>`
establishes that prompt templates declare a stable section and a
dynamic section. After Jinja rendering — handled in
:mod:`strata_forge.prompts.rendering` — the result is a
:class:`StableDynamicSplit`: the rendered stable text, the rendered
dynamic text, and a SHA-256 content digest of the stable portion.

Given a split, :func:`emit_cache_hints` produces a provider-agnostic
:class:`CacheHints` record carrying a single decision — *should the
stable prefix be flagged for cache?* — plus the digest and a token
estimate for analytics. The LLM client interprets the hint per
provider:

- **Anthropic / Bedrock (Claude family with explicit cache_control):**
  ``cache_stable_prefix=True`` means insert
  ``cache_control: {"type": "ephemeral"}`` on the last stable content
  block.
- **OpenAI / Azure / openai_compat:** the hint is informational —
  caching is automatic on prefixes ≥ ~1024 tokens. Stable-first
  ordering is preserved by the message-composition layer.
- **Vertex Gemini:** ``cache_stable_prefix=True`` is a hint that the
  stable portion is a good candidate for a ``CachedContent`` resource;
  the resource lifecycle is out of scope for this module and lives
  with the LLM client.

:func:`is_stable_too_short` exposes the same threshold as a linter
signal so the registry can warn authors who declared a stable/dynamic
split but produced a stable section short enough that caching can't
help — they're paying the boilerplate cost for nothing.
"""

from __future__ import annotations

from dataclasses import dataclass

from strata_forge.core.repro import content_hash
from strata_forge.llm.tokens import count_tokens

__all__ = [
    "DEFAULT_MIN_CACHEABLE_TOKENS",
    "CacheHints",
    "StableDynamicSplit",
    "emit_cache_hints",
    "is_stable_too_short",
]


DEFAULT_MIN_CACHEABLE_TOKENS: int = 100
"""Minimum stable-section token count below which caching is unlikely to help.

OpenAI starts auto-caching at ~1024 tokens, Anthropic supports much
shorter prefixes but cache-write costs more than cache-read on tiny
prefixes, and Gemini's ``CachedContent`` has its own minimum. 100 tokens
is a conservative default that catches the "stable section is a
one-liner" anti-pattern without rejecting reasonable short system
prompts. Callers can override per call.
"""


@dataclass(frozen=True, slots=True)
class StableDynamicSplit:
    """A rendered template's stable and dynamic sections, separated.

    The stable text is byte-identical across renders that share the same
    stable variables; the dynamic text varies per call. ``stable_digest``
    is a content-hash fingerprint suitable for cache-key construction
    and cache-hit analytics — two renders that produced the same stable
    prefix have the same digest, even if the surrounding code changed.
    """

    stable_text: str
    dynamic_text: str
    stable_digest: str

    @classmethod
    def build(cls, *, stable_text: str, dynamic_text: str) -> StableDynamicSplit:
        """Construct a split with the digest computed from ``stable_text``.

        The standard way to build a split — :func:`strata_forge.core.repro.content_hash`
        handles the canonicalization. Direct construction is allowed when
        callers want to supply a precomputed digest (e.g. when replaying
        an existing record).
        """
        return cls(
            stable_text=stable_text,
            dynamic_text=dynamic_text,
            stable_digest=content_hash(stable_text),
        )


@dataclass(frozen=True, slots=True)
class CacheHints:
    """Per-render cache hints consumed by the LLM client.

    Provider-agnostic by design; the client decides how to express the
    hint on the wire per provider (see the module docstring).
    """

    cache_stable_prefix: bool
    stable_digest: str
    stable_token_estimate: int


def emit_cache_hints(
    split: StableDynamicSplit,
    *,
    model: str | None = None,
    min_cacheable_tokens: int = DEFAULT_MIN_CACHEABLE_TOKENS,
) -> CacheHints:
    """Decide whether the stable portion is worth flagging for cache.

    Args:
        split: Rendered stable/dynamic split.
        model: Logical model name passed through to
            :func:`strata_forge.llm.tokens.count_tokens` for accurate tokenization.
            ``None`` falls back to the tokenizer's default encoding.
        min_cacheable_tokens: Minimum token count required to flag the
            stable prefix. Defaults to :data:`DEFAULT_MIN_CACHEABLE_TOKENS`.

    Returns:
        A :class:`CacheHints` carrying the decision plus the stable
        digest and token estimate for downstream use.
    """
    token_estimate = count_tokens(split.stable_text, model=model)
    return CacheHints(
        cache_stable_prefix=token_estimate >= min_cacheable_tokens,
        stable_digest=split.stable_digest,
        stable_token_estimate=token_estimate,
    )


def is_stable_too_short(
    split: StableDynamicSplit,
    *,
    model: str | None = None,
    min_tokens: int = DEFAULT_MIN_CACHEABLE_TOKENS,
) -> bool:
    """Linter signal: the stable section is too short to benefit from caching.

    Authoring a template with an explicit stable/dynamic split costs
    a few lines of boilerplate. When the stable section won't actually
    trigger a cache hit on any supported provider, the author is paying
    that cost for nothing — registry-level lint surfaces this so they
    can collapse to :meth:`PromptTemplate.simple` or expand the stable
    section.
    """
    return count_tokens(split.stable_text, model=model) < min_tokens
