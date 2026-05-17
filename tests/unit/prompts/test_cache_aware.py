"""Unit tests for `forge.prompts.cache_aware`."""

from __future__ import annotations

import pytest

from forge.core.repro import content_hash
from forge.prompts.cache_aware import (
    DEFAULT_MIN_CACHEABLE_TOKENS,
    CacheHints,
    StableDynamicSplit,
    emit_cache_hints,
    is_stable_too_short,
)

# Long enough that any reasonable tokenizer reports well above 100 tokens.
_LONG_STABLE = (
    "You are a helpful and concise assistant. Always think step by step "
    "before answering. When asked a factual question, prefer accuracy over "
    "speculation. When given an example, infer the user's intended pattern "
    "and follow it exactly. Use the provided tools when they would produce "
    "more accurate results than answering from training data alone. Avoid "
    "filler phrases like 'great question' or 'I will help you with that'. "
    "Match the user's level of formality. Cite sources where relevant. "
    "When confident in a numerical answer, give the number first and the "
    "reasoning afterward. When uncertain, say so explicitly and quantify "
    "the uncertainty when possible. If a tool returns an error, surface "
    "it to the user verbatim rather than paraphrasing. Never fabricate "
    "tool output. Keep responses focused on what was asked."
)


# Short enough that any tokenizer reports below 100 tokens.
_SHORT_STABLE = "Be helpful."


# ---------------------------------------------------------------------------
# DEFAULT_MIN_CACHEABLE_TOKENS
# ---------------------------------------------------------------------------


class TestDefaults:
    def test_default_threshold_is_positive_int(self) -> None:
        assert isinstance(DEFAULT_MIN_CACHEABLE_TOKENS, int)
        assert DEFAULT_MIN_CACHEABLE_TOKENS > 0


# ---------------------------------------------------------------------------
# StableDynamicSplit
# ---------------------------------------------------------------------------


class TestStableDynamicSplit:
    def test_direct_construction(self) -> None:
        split = StableDynamicSplit(
            stable_text="stable",
            dynamic_text="dynamic",
            stable_digest="abc123",
        )
        assert split.stable_text == "stable"
        assert split.dynamic_text == "dynamic"
        assert split.stable_digest == "abc123"

    def test_build_computes_digest(self) -> None:
        split = StableDynamicSplit.build(stable_text="hello", dynamic_text="world")
        # The digest must match `content_hash` applied to the stable text.
        assert split.stable_digest == content_hash("hello")

    def test_build_digest_is_64_char_hex(self) -> None:
        split = StableDynamicSplit.build(stable_text="x", dynamic_text="y")
        assert len(split.stable_digest) == 64
        assert all(c in "0123456789abcdef" for c in split.stable_digest)

    def test_build_deterministic(self) -> None:
        a = StableDynamicSplit.build(stable_text="same", dynamic_text="x")
        b = StableDynamicSplit.build(stable_text="same", dynamic_text="y")
        # Digest depends ONLY on stable_text; dynamic_text doesn't affect it.
        assert a.stable_digest == b.stable_digest

    def test_build_digest_changes_with_stable_text(self) -> None:
        a = StableDynamicSplit.build(stable_text="one", dynamic_text="x")
        b = StableDynamicSplit.build(stable_text="two", dynamic_text="x")
        assert a.stable_digest != b.stable_digest

    def test_is_frozen(self) -> None:
        split = StableDynamicSplit.build(stable_text="a", dynamic_text="b")
        with pytest.raises((AttributeError, TypeError)):
            split.stable_text = "other"  # type: ignore[misc]

    def test_build_with_empty_stable(self) -> None:
        # Templates produced via `PromptTemplate.simple` have empty stable
        # sections. The digest of the empty string is still well-defined.
        split = StableDynamicSplit.build(stable_text="", dynamic_text="just dynamic")
        assert split.stable_digest == content_hash("")


# ---------------------------------------------------------------------------
# CacheHints
# ---------------------------------------------------------------------------


class TestCacheHints:
    def test_construction(self) -> None:
        hints = CacheHints(
            cache_stable_prefix=True,
            stable_digest="abc",
            stable_token_estimate=512,
        )
        assert hints.cache_stable_prefix is True
        assert hints.stable_digest == "abc"
        assert hints.stable_token_estimate == 512

    def test_is_frozen(self) -> None:
        hints = CacheHints(
            cache_stable_prefix=False, stable_digest="x", stable_token_estimate=0
        )
        with pytest.raises((AttributeError, TypeError)):
            hints.cache_stable_prefix = True  # type: ignore[misc]


# ---------------------------------------------------------------------------
# emit_cache_hints
# ---------------------------------------------------------------------------


class TestEmitCacheHints:
    def test_long_stable_flags_for_cache(self) -> None:
        split = StableDynamicSplit.build(stable_text=_LONG_STABLE, dynamic_text="q")
        hints = emit_cache_hints(split)
        assert hints.cache_stable_prefix is True
        assert hints.stable_token_estimate >= DEFAULT_MIN_CACHEABLE_TOKENS

    def test_short_stable_does_not_flag_for_cache(self) -> None:
        split = StableDynamicSplit.build(stable_text=_SHORT_STABLE, dynamic_text="q")
        hints = emit_cache_hints(split)
        assert hints.cache_stable_prefix is False
        assert hints.stable_token_estimate < DEFAULT_MIN_CACHEABLE_TOKENS

    def test_digest_passes_through(self) -> None:
        split = StableDynamicSplit.build(stable_text="x", dynamic_text="y")
        hints = emit_cache_hints(split)
        assert hints.stable_digest == split.stable_digest

    def test_custom_threshold_lower_flags_short(self) -> None:
        split = StableDynamicSplit.build(stable_text=_SHORT_STABLE, dynamic_text="q")
        hints = emit_cache_hints(split, min_cacheable_tokens=1)
        # Any non-empty stable text has ≥ 1 token, so the lowered threshold
        # makes the short text qualify.
        assert hints.cache_stable_prefix is True

    def test_custom_threshold_higher_rejects_long(self) -> None:
        split = StableDynamicSplit.build(stable_text=_LONG_STABLE, dynamic_text="q")
        # A threshold larger than any reasonable stable section.
        hints = emit_cache_hints(split, min_cacheable_tokens=10_000)
        assert hints.cache_stable_prefix is False

    def test_empty_stable_does_not_flag(self) -> None:
        split = StableDynamicSplit.build(stable_text="", dynamic_text="q")
        hints = emit_cache_hints(split)
        assert hints.cache_stable_prefix is False
        assert hints.stable_token_estimate == 0

    def test_model_argument_accepted(self) -> None:
        # Different models use different tokenizers; we just verify the
        # function accepts the kwarg cleanly. The exact count may differ
        # between models but both should be positive on _LONG_STABLE.
        split = StableDynamicSplit.build(stable_text=_LONG_STABLE, dynamic_text="q")
        hints_default = emit_cache_hints(split)
        hints_gpt = emit_cache_hints(split, model="gpt-5.5")
        assert hints_default.stable_token_estimate > 0
        assert hints_gpt.stable_token_estimate > 0

    def test_unknown_model_falls_back_silently(self) -> None:
        split = StableDynamicSplit.build(stable_text=_LONG_STABLE, dynamic_text="q")
        # Mirrors `count_tokens`'s behavior: unknown models don't crash.
        hints = emit_cache_hints(split, model="never-heard-of-this-model")
        assert hints.stable_token_estimate > 0


# ---------------------------------------------------------------------------
# is_stable_too_short
# ---------------------------------------------------------------------------


class TestIsStableTooShort:
    def test_long_stable_is_not_too_short(self) -> None:
        split = StableDynamicSplit.build(stable_text=_LONG_STABLE, dynamic_text="q")
        assert is_stable_too_short(split) is False

    def test_short_stable_is_too_short(self) -> None:
        split = StableDynamicSplit.build(stable_text=_SHORT_STABLE, dynamic_text="q")
        assert is_stable_too_short(split) is True

    def test_empty_stable_is_too_short(self) -> None:
        split = StableDynamicSplit.build(stable_text="", dynamic_text="q")
        assert is_stable_too_short(split) is True

    def test_custom_threshold(self) -> None:
        split = StableDynamicSplit.build(stable_text=_SHORT_STABLE, dynamic_text="q")
        # With a threshold of 1, even a 1-token stable is not "too short".
        assert is_stable_too_short(split, min_tokens=1) is False

    def test_model_argument_accepted(self) -> None:
        split = StableDynamicSplit.build(stable_text=_LONG_STABLE, dynamic_text="q")
        assert is_stable_too_short(split, model="claude-opus-4-7") is False
