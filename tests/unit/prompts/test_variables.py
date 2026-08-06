"""Unit tests for `strata_forge.prompts.variables`."""

from __future__ import annotations

import pytest

from strata_forge.prompts.template import PromptTemplate, PromptValidationError
from strata_forge.prompts.variables import (
    extract_variables,
    validate_template_variables,
)

# ---------------------------------------------------------------------------
# extract_variables
# ---------------------------------------------------------------------------


class TestExtractVariables:
    def test_empty_source(self) -> None:
        assert extract_variables("") == frozenset()

    def test_no_variables(self) -> None:
        assert extract_variables("static text only") == frozenset()

    def test_single_variable(self) -> None:
        assert extract_variables("Hello {{ name }}") == frozenset({"name"})

    def test_multiple_variables(self) -> None:
        result = extract_variables("{{ a }} {{ b }} {{ c }}")
        assert result == frozenset({"a", "b", "c"})

    def test_variable_inside_conditional(self) -> None:
        result = extract_variables("{% if show %}{{ name }}{% endif %}")
        assert result == frozenset({"show", "name"})

    def test_variable_inside_loop(self) -> None:
        # Loop-bound variables don't appear in the result — they're declared
        # by the `{% for %}` itself. `xs` is the only undeclared reference.
        result = extract_variables("{% for item in xs %}{{ item }}{% endfor %}")
        assert result == frozenset({"xs"})

    def test_set_block_declares_variable(self) -> None:
        # `{% set %}` introduces a local binding; downstream references to
        # it don't count as undeclared.
        result = extract_variables("{% set x = 1 %}{{ x }} {{ y }}")
        assert result == frozenset({"y"})

    def test_dotted_attribute_access(self) -> None:
        # Only the root identifier matters — `obj.field` references `obj`.
        result = extract_variables("{{ obj.field }}")
        assert result == frozenset({"obj"})

    def test_filter_arguments_count(self) -> None:
        # Variables used as filter arguments still count as references.
        result = extract_variables("{{ text | replace(old, new) }}")
        assert result == frozenset({"text", "old", "new"})

    def test_returns_frozenset(self) -> None:
        # API contract: the return is hashable and immutable.
        result = extract_variables("{{ x }}")
        assert isinstance(result, frozenset)


# ---------------------------------------------------------------------------
# validate_template_variables — happy paths
# ---------------------------------------------------------------------------


class TestValidateHappyPaths:
    def test_no_variables_at_all(self) -> None:
        t = PromptTemplate(name="static", stable_section="hi", dynamic_section="bye")
        # No raise.
        validate_template_variables(t)

    def test_all_declared(self) -> None:
        t = PromptTemplate(
            name="ok",
            stable_section="You are {{ persona }}.",
            dynamic_section="Question: {{ query }}",
            stable_variables=("persona",),
            dynamic_variables=("query",),
        )
        validate_template_variables(t)

    def test_unused_declared_variable_accepted(self) -> None:
        # A declared variable that the body doesn't reference is permitted —
        # authors may declare placeholders for future expansion.
        t = PromptTemplate(
            name="unused",
            stable_section="hi",
            dynamic_section="bye",
            stable_variables=("future_use",),
        )
        validate_template_variables(t)

    def test_loop_variable_does_not_need_declaration(self) -> None:
        t = PromptTemplate(
            name="loops",
            stable_section="",
            dynamic_section="{% for item in items %}{{ item }}{% endfor %}",
            dynamic_variables=("items",),
        )
        validate_template_variables(t)


# ---------------------------------------------------------------------------
# validate_template_variables — declaration violations
# ---------------------------------------------------------------------------


class TestValidateViolations:
    def test_overlap_between_stable_and_dynamic_rejected(self) -> None:
        t = PromptTemplate(
            name="bad",
            stable_section="{{ x }}",
            dynamic_section="{{ x }}",
            stable_variables=("x",),
            dynamic_variables=("x",),
        )
        with pytest.raises(
            PromptValidationError, match="both stable_variables and dynamic_variables"
        ):
            validate_template_variables(t)

    def test_undeclared_in_stable_rejected(self) -> None:
        t = PromptTemplate(
            name="bad",
            stable_section="Hello {{ name }}",  # not declared
            dynamic_section="",
        )
        with pytest.raises(PromptValidationError, match="stable_section references"):
            validate_template_variables(t)

    def test_undeclared_in_dynamic_rejected(self) -> None:
        t = PromptTemplate(
            name="bad",
            stable_section="",
            dynamic_section="Hello {{ name }}",  # not declared
        )
        with pytest.raises(PromptValidationError, match="dynamic_section references"):
            validate_template_variables(t)

    def test_variable_referenced_in_wrong_section_rejected(self) -> None:
        # `query` is declared as dynamic but referenced in the stable section.
        # The error reports "undeclared in stable" — accurate diagnosis
        # because the author put it in the wrong list.
        t = PromptTemplate(
            name="wrong-section",
            stable_section="Context: {{ query }}",
            dynamic_section="",
            dynamic_variables=("query",),
        )
        with pytest.raises(PromptValidationError, match="stable_section references"):
            validate_template_variables(t)

    def test_overlap_checked_first(self) -> None:
        # Overlap raises before undeclared-reference errors so the
        # diagnostic ordering is predictable.
        t = PromptTemplate(
            name="bad",
            stable_section="{{ unrelated }}",  # would also fail undeclared check
            dynamic_section="{{ x }}",
            stable_variables=("x",),
            dynamic_variables=("x",),
        )
        with pytest.raises(
            PromptValidationError, match="both stable_variables and dynamic_variables"
        ):
            validate_template_variables(t)

    def test_error_lists_offending_variables(self) -> None:
        t = PromptTemplate(
            name="bad",
            stable_section="{{ a }} {{ b }}",
        )
        with pytest.raises(PromptValidationError) as info:
            validate_template_variables(t)
        # Both undeclared names appear in the error so the author can fix
        # them in one pass.
        assert "a" in str(info.value)
        assert "b" in str(info.value)

    def test_error_includes_template_name(self) -> None:
        t = PromptTemplate(name="specific-name", stable_section="{{ x }}")
        with pytest.raises(PromptValidationError, match="specific-name"):
            validate_template_variables(t)


# ---------------------------------------------------------------------------
# simple-shorthand templates
# ---------------------------------------------------------------------------


class TestSimpleTemplates:
    def test_simple_with_declared_variables_passes(self) -> None:
        t = PromptTemplate.simple("q", "What is {{ topic }}?", variables=("topic",))
        validate_template_variables(t)

    def test_simple_with_undeclared_variable_rejected(self) -> None:
        t = PromptTemplate.simple("q", "What is {{ topic }}?")
        with pytest.raises(PromptValidationError, match="dynamic_section references"):
            validate_template_variables(t)
