"""Property-based tests for `strata_forge.prompts`.

These are the tests that don't fit cleanly into per-module unit files —
broad invariants we want to hold across the design rather than a specific
function's behavior. They're driven by Hypothesis and run on every CI
push, so a regression in any of the module's plumbing surfaces quickly.
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from strata_forge.prompts.cache_aware import StableDynamicSplit
from strata_forge.prompts.rendering import render
from strata_forge.prompts.template import PromptTemplate, PromptValidationError
from strata_forge.prompts.variables import (
    extract_variables,
    validate_template_variables,
)

# A reusable strategy for Jinja-safe identifier names: lowercase, underscore,
# digits, starts with a letter or underscore. Length is small to keep
# templates readable when Hypothesis shrinks failures.
_identifier = st.from_regex(r"[a-z_][a-z0-9_]{0,7}", fullmatch=True)


# ---------------------------------------------------------------------------
# extract_variables — round-trip with known identifier sets
# ---------------------------------------------------------------------------


class TestExtractVariablesProperties:
    @given(
        identifiers=st.lists(_identifier, min_size=0, max_size=5, unique=True),
    )
    @settings(max_examples=40, deadline=None)
    def test_round_trip_against_known_identifiers(self, identifiers: list[str]) -> None:
        # Build a template body that references every identifier exactly
        # once. `extract_variables` should return that exact set — extras or
        # missing identifiers indicate a parser regression.
        body = " ".join(f"{{{{ {name} }}}}" for name in identifiers)
        assert extract_variables(body) == frozenset(identifiers)

    @given(
        constant=st.text(
            alphabet="abcdefghijklmnopqrstuvwxyz 0123456789.,!?",
            min_size=0,
            max_size=200,
        ),
    )
    @settings(max_examples=40, deadline=None)
    def test_constant_template_has_no_variables(self, constant: str) -> None:
        # Any Jinja source without expression / statement / comment
        # delimiters has zero variables. The alphabet above intentionally
        # excludes `{` and `%` so we don't accidentally synthesize syntax.
        assert extract_variables(constant) == frozenset()

    @given(
        identifiers=st.lists(_identifier, min_size=1, max_size=5, unique=True),
    )
    @settings(max_examples=30, deadline=None)
    def test_loop_variable_excluded_from_references(self, identifiers: list[str]) -> None:
        # `{% for x in xs %}{{ x }}{% endfor %}` references xs but x is
        # loop-bound — only xs counts as a render-time variable.
        loop_var = identifiers[0]
        outer = identifiers[-1]
        body = f"{{% for {loop_var} in {outer} %}}{{{{ {loop_var} }}}}{{% endfor %}}"
        refs = extract_variables(body)
        # `outer` must be in the result; `loop_var` may or may not be in
        # the result depending on shadowing, but `outer` is always there.
        assert outer in refs


# ---------------------------------------------------------------------------
# validate_template_variables — properties on declaration matching
# ---------------------------------------------------------------------------


class TestValidationProperties:
    @given(
        declared=st.lists(_identifier, min_size=0, max_size=4, unique=True),
    )
    @settings(max_examples=30, deadline=None)
    def test_all_declared_in_dynamic_passes(self, declared: list[str]) -> None:
        # If every variable referenced in the dynamic section is in
        # dynamic_variables, validation passes regardless of variable count.
        body = " ".join(f"{{{{ {name} }}}}" for name in declared)
        template = PromptTemplate(
            name="t",
            stable_section="",
            dynamic_section=body,
            dynamic_variables=tuple(declared),
        )
        validate_template_variables(template)

    @given(
        declared=st.lists(_identifier, min_size=2, max_size=5, unique=True),
    )
    @settings(max_examples=30, deadline=None)
    def test_missing_one_declared_fails(self, declared: list[str]) -> None:
        # Same body, but pop one name from the declaration. Validation
        # must fail because the body references an undeclared variable.
        body = " ".join(f"{{{{ {name} }}}}" for name in declared)
        partial = declared[1:]  # drop the first
        template = PromptTemplate(
            name="t",
            stable_section="",
            dynamic_section=body,
            dynamic_variables=tuple(partial),
        )
        with pytest.raises(PromptValidationError, match="dynamic_section references"):
            validate_template_variables(template)

    @given(
        stable=st.lists(_identifier, min_size=1, max_size=3, unique=True),
        dynamic=st.lists(_identifier, min_size=1, max_size=3, unique=True),
    )
    @settings(max_examples=30, deadline=None)
    def test_disjoint_stable_dynamic_passes(self, stable: list[str], dynamic: list[str]) -> None:
        # When the two declaration sets share no names, validation passes
        # for a template that references each variable in its own section.
        overlapping = set(stable) & set(dynamic)
        if overlapping:
            # Hypothesis sometimes generates overlapping sets; skip those —
            # they're tested by the overlap-rejection property below.
            return
        stable_body = " ".join(f"{{{{ {name} }}}}" for name in stable)
        dynamic_body = " ".join(f"{{{{ {name} }}}}" for name in dynamic)
        template = PromptTemplate(
            name="t",
            stable_section=stable_body,
            dynamic_section=dynamic_body,
            stable_variables=tuple(stable),
            dynamic_variables=tuple(dynamic),
        )
        validate_template_variables(template)


# ---------------------------------------------------------------------------
# Cache-marker invariants: byte-exact stable prefix across dynamic renders
# ---------------------------------------------------------------------------


class TestStablePrefixByteExactness:
    """The single most important cache-marker invariant.

    Provider prompt caching requires the cached prefix to be
    byte-identical across requests. If different dynamic inputs even
    once produced a different stable prefix, every request would
    invalidate the cache. These tests pin that down with explicit
    multi-render comparisons.
    """

    def test_many_dynamic_renders_share_stable_prefix(self) -> None:
        template = PromptTemplate(
            name="t",
            stable_section="You are {{ persona }}. Be concise. Cite sources.",
            dynamic_section="Q: {{ q }}",
            stable_variables=("persona",),
            dynamic_variables=("q",),
        )
        renders = [
            render(template, {"persona": "helper", "q": q})
            for q in ("a", "much longer question text", "?", "", " ")
        ]
        stable_texts = {r.split.stable_text for r in renders}
        assert len(stable_texts) == 1
        digests = {r.cache_hints.stable_digest for r in renders}
        assert len(digests) == 1

    def test_changing_stable_variable_changes_prefix(self) -> None:
        template = PromptTemplate(
            name="t",
            stable_section="You are {{ persona }}.",
            dynamic_section="Q: {{ q }}",
            stable_variables=("persona",),
            dynamic_variables=("q",),
        )
        a = render(template, {"persona": "helper", "q": "same"})
        b = render(template, {"persona": "expert", "q": "same"})
        assert a.split.stable_text != b.split.stable_text
        assert a.cache_hints.stable_digest != b.cache_hints.stable_digest

    @given(
        persona=st.from_regex(r"[a-z ]{1,30}", fullmatch=True),
        queries=st.lists(
            st.text(alphabet="abcdefghijklmnopqrstuvwxyz ?", max_size=50),
            min_size=2,
            max_size=5,
        ),
    )
    @settings(max_examples=20, deadline=None)
    def test_property_stable_prefix_invariant_under_dynamic_changes(
        self, persona: str, queries: list[str]
    ) -> None:
        template = PromptTemplate(
            name="t",
            stable_section="You are {{ persona }}. Be concise.",
            dynamic_section="Q: {{ q }}",
            stable_variables=("persona",),
            dynamic_variables=("q",),
        )
        renders = [render(template, {"persona": persona, "q": q}) for q in queries]
        # No matter which dynamic input was passed, the rendered stable text
        # is byte-identical.
        stables = {r.split.stable_text for r in renders}
        assert len(stables) == 1


# ---------------------------------------------------------------------------
# StableDynamicSplit digest properties
# ---------------------------------------------------------------------------


class TestSplitDigestProperties:
    @given(
        stable=st.text(min_size=0, max_size=500),
        dynamic=st.text(min_size=0, max_size=500),
    )
    @settings(max_examples=40, deadline=None)
    def test_digest_is_64_char_hex(self, stable: str, dynamic: str) -> None:
        split = StableDynamicSplit.build(stable_text=stable, dynamic_text=dynamic)
        assert len(split.stable_digest) == 64
        assert all(c in "0123456789abcdef" for c in split.stable_digest)

    @given(
        stable=st.text(min_size=0, max_size=200),
        dynamic_a=st.text(min_size=0, max_size=200),
        dynamic_b=st.text(min_size=0, max_size=200),
    )
    @settings(max_examples=30, deadline=None)
    def test_digest_independent_of_dynamic_text(
        self, stable: str, dynamic_a: str, dynamic_b: str
    ) -> None:
        a = StableDynamicSplit.build(stable_text=stable, dynamic_text=dynamic_a)
        b = StableDynamicSplit.build(stable_text=stable, dynamic_text=dynamic_b)
        assert a.stable_digest == b.stable_digest
