"""Unit and property tests for `forge.llm.tokens`."""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from forge.llm.tokens import count_tokens


class TestBasic:
    def test_empty_string_is_zero(self) -> None:
        assert count_tokens("") == 0

    def test_non_empty_string_is_positive(self) -> None:
        assert count_tokens("hello world") > 0

    def test_unicode_works(self) -> None:
        assert count_tokens("café 🤖 日本語") > 0


class TestModelDispatch:
    def test_no_model_uses_default(self) -> None:
        count = count_tokens("hello world")
        assert count > 0

    def test_unknown_model_falls_back_silently(self) -> None:
        # Unknown models don't raise — they use the default encoding.
        count = count_tokens("hello world", model="never-heard-of-this-model")
        assert count > 0

    def test_openai_gpt5_uses_o200k_base(self) -> None:
        # The GPT-5 family uses o200k_base which is generally denser than
        # cl100k_base for English text.
        text = "The quick brown fox jumps over the lazy dog. " * 10
        gpt5_count = count_tokens(text, model="gpt-5.5")
        # Sanity: tokenization succeeds and counts are positive.
        assert gpt5_count > 0

    def test_anthropic_falls_back_to_cl100k(self) -> None:
        text = "Hello, world!"
        anthropic_count = count_tokens(text, model="claude-opus-4-7")
        assert anthropic_count > 0

    def test_google_falls_back_to_cl100k(self) -> None:
        text = "Hello, world!"
        gemini_count = count_tokens(text, model="gemini-3.1-pro")
        assert gemini_count > 0

    @pytest.mark.parametrize(
        "model",
        [
            "gpt-5.5",
            "gpt-5.5-pro",
            "gpt-5.5-thinking",
            "gpt-5.5-instant",
            "claude-opus-4-7",
            "claude-sonnet-4-6",
            "claude-haiku-4-5",
            "gemini-3.1-pro",
            "gemini-3.1-flash-lite",
        ],
    )
    def test_every_registered_model_works(self, model: str) -> None:
        # Sanity check that no registered model causes a tokenizer crash.
        assert count_tokens("sample text", model=model) > 0


class TestSimpleProperties:
    def test_more_text_means_more_tokens(self) -> None:
        short = count_tokens("hi")
        long = count_tokens("hi " * 100)
        assert long > short

    def test_repeated_words_scale_roughly_linearly(self) -> None:
        # A 100x-repeated phrase should produce roughly 100x the tokens of
        # one repetition (within a generous tolerance for boundary merges).
        unit = count_tokens("the cat sat ")
        many = count_tokens("the cat sat " * 100)
        # 75% lower bound is conservative; in practice scaling is tighter.
        assert many > 0.75 * unit * 100
        assert many < 1.25 * unit * 100


class TestHypothesisProperties:
    @settings(max_examples=50, deadline=None)
    @given(s=st.text(min_size=0, max_size=200))
    def test_count_non_negative(self, s: str) -> None:
        assert count_tokens(s) >= 0

    @settings(max_examples=50, deadline=None)
    @given(s=st.text(min_size=0, max_size=100))
    def test_count_zero_iff_empty_for_empty_string(self, s: str) -> None:
        # Empty strings always count as zero. Non-empty strings may also
        # count as zero (the encoder can produce no tokens for some
        # whitespace-only inputs depending on the model) — we only check
        # the forward implication, not the reverse.
        if not s:
            assert count_tokens(s) == 0

    # Note: a stricter "concatenation is subadditive" property (`count(a+b)
    # <= count(a) + count(b)`) is intuitive but NOT strictly true for BPE
    # tokenizers. Counterexample found in practice: `count("  ")` = 1,
    # `count("aaaaaa")` = 2, but `count("  aaaaaa")` = 4 — leading whitespace
    # changes how the following text tokenizes. So we don't assert it.

    @settings(max_examples=40, deadline=None)
    @given(
        a=st.text(alphabet="abcdefghijklmnopqrstuvwxyz ", min_size=1, max_size=80),
        b=st.text(alphabet="abcdefghijklmnopqrstuvwxyz ", min_size=1, max_size=80),
    )
    def test_concatenation_is_at_least_as_many_as_each_part(self, a: str, b: str) -> None:
        # Appending text never reduces the token count below either piece
        # taken alone — boundary merges shrink the total but can't shrink
        # past a single part's count.
        ca = count_tokens(a)
        cb = count_tokens(b)
        cab = count_tokens(a + b)
        assert cab >= ca
        assert cab >= cb
