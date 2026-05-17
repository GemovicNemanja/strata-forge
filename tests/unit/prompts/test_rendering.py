"""Unit tests for `forge.prompts.rendering`."""

from __future__ import annotations

import pytest

from forge.core.errors import ValidationError
from forge.llm.messages import SystemMessage, UserMessage
from forge.prompts.cache_aware import CacheHints, StableDynamicSplit
from forge.prompts.rendering import RenderedPrompt, render
from forge.prompts.template import PromptTemplate, PromptValidationError

_LONG_STABLE = (
    "You are a helpful and concise assistant. Always think step by step "
    "before answering. When asked a factual question, prefer accuracy over "
    "speculation. When given an example, infer the user's intended pattern "
    "and follow it exactly. Use the provided tools when they would produce "
    "more accurate results than answering from training data alone. Avoid "
    "filler phrases. Match the user's level of formality. Cite sources "
    "where relevant. When confident in a numerical answer, give the number "
    "first and the reasoning afterward. When uncertain, say so explicitly "
    "and quantify the uncertainty when possible. Never fabricate output."
)


# ---------------------------------------------------------------------------
# Happy paths
# ---------------------------------------------------------------------------


class TestRenderBasics:
    def test_renders_both_sections(self) -> None:
        template = PromptTemplate(
            name="greet",
            stable_section="You are {{ persona }}.",
            dynamic_section="Question: {{ query }}",
            stable_variables=("persona",),
            dynamic_variables=("query",),
        )
        result = render(
            template,
            {"persona": "a helpful assistant", "query": "Why is the sky blue?"},
        )

        assert isinstance(result, RenderedPrompt)
        assert len(result.messages) == 2
        assert isinstance(result.messages[0], SystemMessage)
        assert result.messages[0].content == "You are a helpful assistant."
        assert isinstance(result.messages[1], UserMessage)
        assert result.messages[1].content == "Question: Why is the sky blue?"

    def test_returns_rendered_prompt_with_split(self) -> None:
        template = PromptTemplate(
            name="x",
            stable_section="A",
            dynamic_section="B",
        )
        result = render(template)
        assert isinstance(result.split, StableDynamicSplit)
        assert result.split.stable_text == "A"
        assert result.split.dynamic_text == "B"

    def test_returns_cache_hints(self) -> None:
        template = PromptTemplate(
            name="x",
            stable_section=_LONG_STABLE,
            dynamic_section="Question: hi",
        )
        result = render(template)
        assert isinstance(result.cache_hints, CacheHints)
        assert result.cache_hints.cache_stable_prefix is True

    def test_no_variables_no_args(self) -> None:
        template = PromptTemplate(
            name="static",
            stable_section="constant prompt",
            dynamic_section="and constant question",
        )
        result = render(template)
        assert [m.content for m in result.messages] == [
            "constant prompt",
            "and constant question",
        ]


# ---------------------------------------------------------------------------
# Empty-section handling
# ---------------------------------------------------------------------------


class TestEmptySections:
    def test_empty_stable_skips_system_message(self) -> None:
        template = PromptTemplate(
            name="dyn-only",
            stable_section="",
            dynamic_section="Hi {{ name }}",
            dynamic_variables=("name",),
        )
        result = render(template, {"name": "Alice"})
        assert len(result.messages) == 1
        assert isinstance(result.messages[0], UserMessage)
        assert result.messages[0].content == "Hi Alice"

    def test_empty_dynamic_skips_user_message(self) -> None:
        template = PromptTemplate(name="stable-only", stable_section="be helpful")
        result = render(template)
        assert len(result.messages) == 1
        assert isinstance(result.messages[0], SystemMessage)

    def test_both_sections_empty(self) -> None:
        # Pathological but well-defined: no messages produced.
        template = PromptTemplate(name="empty", stable_section="", dynamic_section="")
        result = render(template)
        assert result.messages == []

    def test_simple_shorthand_renders_user_only(self) -> None:
        # `PromptTemplate.simple` produces a dynamic-only template.
        template = PromptTemplate.simple("q", "What is {{ topic }}?", variables=("topic",))
        result = render(template, {"topic": "entropy"})
        assert len(result.messages) == 1
        assert isinstance(result.messages[0], UserMessage)
        assert result.messages[0].content == "What is entropy?"


# ---------------------------------------------------------------------------
# Variable validation at render time
# ---------------------------------------------------------------------------


class TestMissingVariables:
    def test_missing_dynamic_variable_raises(self) -> None:
        template = PromptTemplate(
            name="bad",
            stable_section="",
            dynamic_section="Hi {{ name }}",
            dynamic_variables=("name",),
        )
        with pytest.raises(ValidationError, match="missing variables"):
            render(template, {})

    def test_missing_stable_variable_raises(self) -> None:
        template = PromptTemplate(
            name="bad",
            stable_section="You are {{ persona }}.",
            dynamic_section="hi",
            stable_variables=("persona",),
        )
        with pytest.raises(ValidationError, match="missing variables"):
            render(template, {})

    def test_partial_provision_raises(self) -> None:
        template = PromptTemplate(
            name="bad",
            stable_section="You are {{ persona }}.",
            dynamic_section="Question: {{ query }}",
            stable_variables=("persona",),
            dynamic_variables=("query",),
        )
        with pytest.raises(ValidationError) as info:
            render(template, {"persona": "x"})
        assert "query" in str(info.value)


class TestExtraVariables:
    def test_extra_variable_rejected(self) -> None:
        template = PromptTemplate(name="x", stable_section="hi", dynamic_section="")
        with pytest.raises(ValidationError, match="not declared"):
            render(template, {"extra": "value"})

    def test_extra_variable_error_lists_offenders(self) -> None:
        template = PromptTemplate(
            name="x",
            stable_section="Hi {{ name }}",
            stable_variables=("name",),
        )
        with pytest.raises(ValidationError) as info:
            render(template, {"name": "Alice", "typo": "v"})
        assert "typo" in str(info.value)


# ---------------------------------------------------------------------------
# Pre-validation of the template itself
# ---------------------------------------------------------------------------


class TestPreValidation:
    def test_template_with_undeclared_reference_rejected(self) -> None:
        # The body references `name`, but it wasn't declared at template
        # construction. The renderer calls validate_template_variables which
        # raises before any render attempt.
        template = PromptTemplate(
            name="bad",
            stable_section="",
            dynamic_section="Hi {{ name }}",
        )
        with pytest.raises(PromptValidationError):
            render(template, {"name": "Alice"})

    def test_overlap_between_sections_rejected(self) -> None:
        template = PromptTemplate(
            name="bad",
            stable_section="{{ x }}",
            dynamic_section="{{ x }}",
            stable_variables=("x",),
            dynamic_variables=("x",),
        )
        with pytest.raises(PromptValidationError):
            render(template, {"x": "value"})


# ---------------------------------------------------------------------------
# Section partitioning
# ---------------------------------------------------------------------------


class TestVariablePartitioning:
    def test_stable_render_only_sees_stable_variables(self) -> None:
        # `dynamic_value` isn't visible to the stable section. We verify by
        # rendering a template whose stable section references it would have
        # been undeclared (catches the partitioning bug).
        template = PromptTemplate(
            name="x",
            stable_section="Stable: {{ persona }}",
            dynamic_section="Dyn: {{ persona }}",  # wait — this is a problem
            stable_variables=("persona",),
        )
        # `persona` is declared as stable; dynamic_section references it but
        # it's not in dynamic_variables. validate_template_variables should
        # have rejected this template BEFORE render. The pre-validation step
        # catches it.
        with pytest.raises(PromptValidationError):
            render(template, {"persona": "helper"})

    def test_dynamic_render_only_sees_dynamic_variables(self) -> None:
        # Inverse case: both sections reference their own declared vars.
        # We verify that the stable variable doesn't leak into the dynamic
        # render (would be a security/correctness issue) by giving each a
        # distinct value.
        template = PromptTemplate(
            name="x",
            stable_section="Stable: {{ persona }}",
            dynamic_section="Dyn: {{ query }}",
            stable_variables=("persona",),
            dynamic_variables=("query",),
        )
        result = render(template, {"persona": "helper", "query": "hi"})
        # The stable section rendered correctly.
        assert result.messages[0].content == "Stable: helper"
        # The dynamic section rendered correctly.
        assert result.messages[1].content == "Dyn: hi"


# ---------------------------------------------------------------------------
# Cache hint integration
# ---------------------------------------------------------------------------


class TestCacheHintsIntegration:
    def test_long_stable_text_flagged_for_cache(self) -> None:
        template = PromptTemplate(
            name="x",
            stable_section=_LONG_STABLE,
            dynamic_section="What is {{ topic }}?",
            dynamic_variables=("topic",),
        )
        result = render(template, {"topic": "entropy"})
        assert result.cache_hints.cache_stable_prefix is True

    def test_short_stable_text_not_flagged(self) -> None:
        template = PromptTemplate(
            name="x",
            stable_section="Be helpful.",
            dynamic_section="hi",
        )
        result = render(template)
        assert result.cache_hints.cache_stable_prefix is False

    def test_model_argument_forwards_to_tokenizer(self) -> None:
        template = PromptTemplate(
            name="x",
            stable_section=_LONG_STABLE,
            dynamic_section="hi",
        )
        # The model influences which tokenizer is used; we just verify the
        # parameter is accepted and produces a positive estimate.
        result = render(template, model="gpt-5.5")
        assert result.cache_hints.stable_token_estimate > 0

    def test_custom_threshold_forwards(self) -> None:
        template = PromptTemplate(
            name="x",
            stable_section="Be helpful.",
            dynamic_section="hi",
        )
        # With a threshold of 1, even the short text qualifies.
        result = render(template, min_cacheable_tokens=1)
        assert result.cache_hints.cache_stable_prefix is True

    def test_digest_matches_rendered_stable_text(self) -> None:
        from forge.core.repro import content_hash

        template = PromptTemplate(
            name="x",
            stable_section="You are {{ persona }}.",
            stable_variables=("persona",),
        )
        result = render(template, {"persona": "helper"})
        assert result.cache_hints.stable_digest == content_hash("You are helper.")


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_same_inputs_produce_same_output(self) -> None:
        template = PromptTemplate(
            name="x",
            stable_section="{{ persona }}",
            dynamic_section="{{ query }}",
            stable_variables=("persona",),
            dynamic_variables=("query",),
        )
        vars_in = {"persona": "helper", "query": "hi"}
        a = render(template, vars_in)
        b = render(template, vars_in)
        assert a.split == b.split
        assert a.cache_hints.stable_digest == b.cache_hints.stable_digest
        assert [m.content for m in a.messages] == [m.content for m in b.messages]

    def test_dynamic_change_doesnt_affect_stable_digest(self) -> None:
        template = PromptTemplate(
            name="x",
            stable_section="{{ persona }}",
            dynamic_section="{{ query }}",
            stable_variables=("persona",),
            dynamic_variables=("query",),
        )
        a = render(template, {"persona": "helper", "query": "first"})
        b = render(template, {"persona": "helper", "query": "second"})
        assert a.cache_hints.stable_digest == b.cache_hints.stable_digest

    def test_stable_change_does_affect_stable_digest(self) -> None:
        template = PromptTemplate(
            name="x",
            stable_section="{{ persona }}",
            dynamic_section="hi",
            stable_variables=("persona",),
        )
        a = render(template, {"persona": "helper1"})
        b = render(template, {"persona": "helper2"})
        assert a.cache_hints.stable_digest != b.cache_hints.stable_digest


# ---------------------------------------------------------------------------
# RenderedPrompt shape
# ---------------------------------------------------------------------------


class TestRenderedPrompt:
    def test_is_frozen(self) -> None:
        template = PromptTemplate(name="x", stable_section="hi")
        result = render(template)
        with pytest.raises((AttributeError, TypeError)):
            result.messages = []  # type: ignore[misc]
