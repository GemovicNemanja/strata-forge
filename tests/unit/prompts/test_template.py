"""Unit tests for `strata_forge.prompts.template`."""

from __future__ import annotations

from typing import Any

import jinja2
import pytest
from jinja2.sandbox import SandboxedEnvironment
from pydantic import ValidationError

from strata_forge.core.errors import ForgeError
from strata_forge.prompts.template import (
    SAFE_FILTERS,
    PromptError,
    PromptTemplate,
    PromptValidationError,
    create_sandboxed_environment,
)

# ---------------------------------------------------------------------------
# Error hierarchy
# ---------------------------------------------------------------------------


class TestErrorHierarchy:
    def test_prompt_error_is_forge_error(self) -> None:
        assert issubclass(PromptError, ForgeError)

    def test_validation_error_is_prompt_error(self) -> None:
        assert issubclass(PromptValidationError, PromptError)

    def test_can_raise_and_catch(self) -> None:
        with pytest.raises(PromptError):
            raise PromptValidationError("nope")


# ---------------------------------------------------------------------------
# SAFE_FILTERS
# ---------------------------------------------------------------------------


class TestSafeFilters:
    def test_is_frozenset(self) -> None:
        assert isinstance(SAFE_FILTERS, frozenset)

    def test_is_non_empty(self) -> None:
        assert len(SAFE_FILTERS) > 0

    def test_includes_common_string_filters(self) -> None:
        for name in ("upper", "lower", "trim", "replace", "join"):
            assert name in SAFE_FILTERS

    def test_excludes_dangerous_filters(self) -> None:
        # No I/O, no code execution, no module access. None of these exist in
        # stock Jinja2 but the test documents intent: if a future Jinja2 ships
        # one, it must be opt-in by adding it to SAFE_FILTERS.
        for name in ("attr", "eval", "exec", "open", "read", "import"):
            assert name not in SAFE_FILTERS


# ---------------------------------------------------------------------------
# Sandboxed environment
# ---------------------------------------------------------------------------


class TestCreateSandboxedEnvironment:
    def test_returns_sandboxed_environment(self) -> None:
        env = create_sandboxed_environment()
        assert isinstance(env, SandboxedEnvironment)

    def test_autoescape_off(self) -> None:
        env = create_sandboxed_environment()
        # autoescape may be False or a callable returning False; verify it
        # doesn't escape HTML-ish content in a render.
        rendered = env.from_string("{{ x }}").render(x="<b>hi</b>")
        assert rendered == "<b>hi</b>"

    def test_loader_is_none(self) -> None:
        env = create_sandboxed_environment()
        assert env.loader is None

    def test_strict_undefined(self) -> None:
        env = create_sandboxed_environment()
        with pytest.raises(jinja2.UndefinedError):
            env.from_string("{{ missing_var }}").render()

    def test_filters_restricted_to_allowlist(self) -> None:
        env = create_sandboxed_environment()
        for name in env.filters:
            assert name in SAFE_FILTERS, f"unexpected filter exposed: {name}"

    def test_tests_disabled(self) -> None:
        env = create_sandboxed_environment()
        assert env.tests == {}

    def test_disallowed_filter_fails_at_compile(self) -> None:
        env = create_sandboxed_environment()
        # `attr` is a stock Jinja2 filter — pruned from our env.
        with pytest.raises(jinja2.TemplateAssertionError):
            env.from_string("{{ x | attr('foo') }}")

    def test_sandboxed_attribute_access_blocked(self) -> None:
        # SandboxedEnvironment refuses to traverse __class__, __mro__, etc.
        env = create_sandboxed_environment()
        with pytest.raises(jinja2.exceptions.SecurityError):
            env.from_string("{{ x.__class__ }}").render(x="hi")

    def test_returns_fresh_instance(self) -> None:
        # The factory builds a new env per call. Mutating one mustn't affect
        # the next — otherwise tests could pollute the module-level env.
        env1 = create_sandboxed_environment()
        env2 = create_sandboxed_environment()
        assert env1 is not env2


# ---------------------------------------------------------------------------
# PromptTemplate construction
# ---------------------------------------------------------------------------


class TestPromptTemplateConstruction:
    def test_basic(self) -> None:
        t = PromptTemplate(
            name="greet",
            stable_section="You are a helpful assistant.",
            dynamic_section="Hello {{ name }}!",
            dynamic_variables=("name",),
        )
        assert t.name == "greet"
        assert t.stable_section == "You are a helpful assistant."
        assert t.dynamic_section == "Hello {{ name }}!"
        assert t.dynamic_variables == ("name",)
        assert t.stable_variables == ()
        assert t.description == ""
        assert t.metadata == {}

    def test_stable_only_template(self) -> None:
        t = PromptTemplate(name="canned", stable_section="constant prompt")
        assert t.dynamic_section == ""

    def test_dynamic_only_template(self) -> None:
        t = PromptTemplate(name="dynamic", dynamic_section="just {{ x }}", dynamic_variables=("x",))
        assert t.stable_section == ""

    def test_with_description_and_metadata(self) -> None:
        t = PromptTemplate(
            name="x",
            stable_section="hi",
            description="A friendly greeter",
            metadata={"owner": "team-llm"},
        )
        assert t.description == "A friendly greeter"
        assert t.metadata == {"owner": "team-llm"}

    def test_name_must_be_non_empty(self) -> None:
        with pytest.raises(ValidationError):
            PromptTemplate(name="", stable_section="hi")

    def test_is_frozen(self) -> None:
        t = PromptTemplate(name="x", stable_section="hi")
        with pytest.raises(ValidationError, match="frozen"):
            t.name = "y"  # type: ignore[misc]

    def test_extra_fields_forbidden(self) -> None:
        with pytest.raises(ValidationError):
            PromptTemplate(
                name="x",
                stable_section="hi",
                unknown_field="oops",  # type: ignore[call-arg]
            )

    def test_metadata_defaults_independent_across_instances(self) -> None:
        # Pydantic v2 deep-copies mutable defaults; mutating one instance's
        # metadata must NOT bleed into another. (We can't mutate frozen, but
        # this guards against a regression to a shared default.)
        t1 = PromptTemplate(name="a", stable_section="x")
        t2 = PromptTemplate(name="b", stable_section="x")
        assert t1.metadata == {}
        assert t2.metadata == {}
        # Same value, but identity-independent so mutation can't bleed.
        assert t1.metadata is not t2.metadata


class TestPromptTemplateSyntax:
    def test_invalid_jinja_in_stable_raises(self) -> None:
        with pytest.raises(PromptValidationError, match="stable_section"):
            PromptTemplate(name="bad", stable_section="{% if oops %}")

    def test_invalid_jinja_in_dynamic_raises(self) -> None:
        with pytest.raises(PromptValidationError, match="dynamic_section"):
            PromptTemplate(name="bad", dynamic_section="{{ unclosed")

    def test_disallowed_filter_in_stable_raises(self) -> None:
        with pytest.raises(PromptValidationError, match="stable_section"):
            PromptTemplate(
                name="bad",
                stable_section="{{ x | attr('foo') }}",
                stable_variables=("x",),
            )

    def test_disallowed_filter_in_dynamic_raises(self) -> None:
        with pytest.raises(PromptValidationError, match="dynamic_section"):
            PromptTemplate(
                name="bad",
                dynamic_section="{{ x | attr('foo') }}",
                dynamic_variables=("x",),
            )

    def test_valid_filter_accepted(self) -> None:
        # `upper` is on the allowlist.
        t = PromptTemplate(
            name="ok",
            dynamic_section="{{ name | upper }}",
            dynamic_variables=("name",),
        )
        assert t.name == "ok"

    def test_validation_error_chains_cause(self) -> None:
        with pytest.raises(PromptValidationError) as info:
            PromptTemplate(name="bad", stable_section="{% if oops %}")
        # The original Jinja2 exception is preserved.
        assert info.value.__cause__ is not None


# ---------------------------------------------------------------------------
# PromptTemplate.simple shorthand
# ---------------------------------------------------------------------------


class TestPromptTemplateSimple:
    def test_treats_body_as_dynamic(self) -> None:
        t = PromptTemplate.simple("quick", "What is {{ topic }}?", variables=("topic",))
        assert t.stable_section == ""
        assert t.dynamic_section == "What is {{ topic }}?"
        assert t.stable_variables == ()
        assert t.dynamic_variables == ("topic",)

    def test_empty_variables_default(self) -> None:
        t = PromptTemplate.simple("noargs", "constant body")
        assert t.dynamic_variables == ()

    def test_description_and_metadata_pass_through(self) -> None:
        t = PromptTemplate.simple(
            "q",
            "{{ x }}",
            variables=("x",),
            description="quick prompt",
            metadata={"tag": "ad-hoc"},
        )
        assert t.description == "quick prompt"
        assert t.metadata == {"tag": "ad-hoc"}

    def test_validates_jinja_syntax(self) -> None:
        with pytest.raises(PromptValidationError):
            PromptTemplate.simple("bad", "{% if x %}")

    def test_metadata_defaults_to_empty_dict(self) -> None:
        t = PromptTemplate.simple("x", "{{ y }}", variables=("y",))
        assert t.metadata == {}


# ---------------------------------------------------------------------------
# Type-level sanity (cheap to keep, catches future regressions)
# ---------------------------------------------------------------------------


class TestTypeShape:
    def test_variables_tuples_are_immutable(self) -> None:
        t = PromptTemplate(name="x", stable_section="hi", stable_variables=("a", "b"))
        # tuples don't have mutating methods; the type system enforces this.
        # Verify the field round-trips as a tuple.
        assert isinstance(t.stable_variables, tuple)

    def test_metadata_accepts_arbitrary_json_values(self) -> None:
        # We allow any JSON-shaped metadata (dict[str, Any]).
        meta: dict[str, Any] = {"int": 1, "list": [1, 2], "nested": {"a": True}}
        t = PromptTemplate(name="x", stable_section="hi", metadata=meta)
        assert t.metadata == meta
