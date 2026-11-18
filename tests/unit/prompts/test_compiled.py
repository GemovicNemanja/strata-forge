"""Unit tests for `strata_forge.prompts.compiled`: compile, hash integrity, render, vectors."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from strata_forge.prompts import (
    Binding,
    PromptBuild,
    PromptBuildError,
    PromptPart,
    PromptRenderError,
    VariableSpec,
    bind_row,
    compile_prompt,
    parse_part,
    render_build,
    value_text,
)
from strata_forge.prompts.compiled import COMPILER_VERSION

VECTORS = Path(__file__).resolve().parents[2] / "fixtures" / "prompt_build_vectors.json"


def _self(text: str, dialect: str = "strata") -> PromptPart:
    return PromptPart.model_validate(
        {"kind": "self", "name": "Instructions", "text": text, "dialect": dialect}
    )


def _build(text: str, *variables: VariableSpec) -> PromptBuild:
    result = compile_prompt([_self(text)], variables)
    assert result.build is not None, result.issues
    return result.build


class TestCompile:
    def test_composition_joins_parts_and_maps_lines(self) -> None:
        result = compile_prompt(
            [
                PromptPart(
                    kind="module", name="Voice", ref="m1", label="v6", text="Line one\nLine two\n"
                ),
                _self("Q: {{ q }}"),
            ],
            [VariableSpec(name="q")],
        )
        build = result.build
        assert build is not None
        assert build.template == "Line one\nLine two\n\nQ: {{ q }}"
        assert [(s.part, s.line_start, s.line_end) for s in build.source_map] == [
            (0, 1, 2),
            (1, 4, 4),
        ]
        assert build.parts[0].label == "v6"
        assert build.parts[0].ref == "m1"
        assert build.compiler_version == COMPILER_VERSION

    def test_hash_is_content_only(self) -> None:
        a = _build("Hi {{ x }}", VariableSpec(name="x"))
        b = compile_prompt(
            [PromptPart(kind="module", name="Other name", text="Hi {{x}}")],
            [VariableSpec(name="x")],
        ).build
        assert b is not None
        assert a.hash == b.hash
        assert a.hash != _build("Hi  {{ x }}", VariableSpec(name="x")).hash
        assert a.hash != _build("Hi {{ x }}", VariableSpec(name="x", type="number")).hash

    def test_line_endings_and_unicode_are_canonical(self) -> None:
        a = _build("a\r\nb\rc Café")
        assert a.template == "a\nb\nc Café"
        assert a.hash == _build("a\nb\nc Café").hash

    def test_unused_declarations_are_left_out(self) -> None:
        build = _build(
            "{{ b }} {{ a }}",
            VariableSpec(name="a"),
            VariableSpec(name="b"),
            VariableSpec(name="c"),
        )
        assert [v.name for v in build.variables] == ["a", "b"]

    def test_undeclared_variable_is_an_error_with_position(self) -> None:
        result = compile_prompt([_self("one\ntwo {{ policy_context }}")])
        assert result.build is None
        (issue,) = result.issues
        assert issue.code == "undeclared_variable"
        assert (issue.line, issue.column, issue.variable) == (2, 5, "policy_context")
        assert result.referenced == ("policy_context",)

    def test_duplicate_declaration(self) -> None:
        result = compile_prompt(
            [_self("{{ x }}")], [VariableSpec(name="x"), VariableSpec(name="x")]
        )
        assert [i.code for i in result.issues] == ["duplicate_variable"]

    @pytest.mark.parametrize(
        ("text", "code"),
        [
            ("{{ a.b }}", "unsupported_expression"),
            ("{{ x | upper }}", "unsupported_expression"),
            ("{{ }}", "unsupported_expression"),
            ("{{ x", "unclosed_expression"),
            ("{{ x\n}}", "unclosed_expression"),
            ("{% if x %}y{% endif %}", "unsupported_block"),
            ("{# note #}", "unsupported_block"),
        ],
    )
    def test_anything_but_a_variable_is_refused(self, text: str, code: str) -> None:
        result = compile_prompt([_self(text)], [VariableSpec(name="x")])
        assert result.build is None
        assert result.issues[0].code == code

    def test_single_braces_are_literal_in_the_strata_dialect(self) -> None:
        build = _build('{"answer": "{x}"} {{ x }}', VariableSpec(name="x"))
        assert render_build(build, {"x": "y"}) == '{"answer": "{x}"} y'

    def test_braces_dialect_matches_the_runner_regex(self) -> None:
        result = compile_prompt(
            [_self("{{q}} {q} {a b} {0}", dialect="braces")],
            [VariableSpec(name="q"), VariableSpec(name="0")],
        )
        assert result.build is not None
        assert render_build(result.build, {"q": "Q", "0": "Z"}) == "{Q} Q {a b} Z"

    def test_empty_prompt(self) -> None:
        assert [i.code for i in compile_prompt([_self("\n")]).issues] == ["empty_prompt"]

    def test_too_many_parts_is_refused(self) -> None:
        with pytest.raises(PromptBuildError):
            compile_prompt([_self("x")] * 65)

    def test_oversized_part(self) -> None:
        assert compile_prompt([_self("x" * 64_001)]).issues[0].code == "too_large"

    def test_parse_part_lists_variables(self) -> None:
        segments, issues = parse_part("a {{ b }} c {{ d }}")
        assert not issues
        assert [s.name for s in segments if s.kind == "var"] == ["b", "d"]


class TestIntegrity:
    def test_round_trips_as_json(self) -> None:
        build = _build("Hello {{ name }}", VariableSpec(name="name"))
        again = PromptBuild.model_validate(json.loads(build.model_dump_json()))
        assert again == build

    def test_tampered_segment_is_refused(self) -> None:
        data: dict[str, Any] = _build("Hello {{ name }}", VariableSpec(name="name")).model_dump(
            mode="json"
        )
        data["segments"][0]["text"] = "Goodbye "
        with pytest.raises(PromptBuildError, match="altered"):
            PromptBuild.model_validate(data)

    def test_tampered_declaration_is_refused(self) -> None:
        data: dict[str, Any] = _build("{{ n }}", VariableSpec(name="n", type="number")).model_dump(
            mode="json"
        )
        data["variables"][0]["type"] = "string"
        with pytest.raises(PromptBuildError):
            PromptBuild.model_validate(data)

    def test_variables_must_match_segments(self) -> None:
        data: dict[str, Any] = _build("{{ n }}", VariableSpec(name="n")).model_dump(mode="json")
        data["variables"] = []
        with pytest.raises(PromptBuildError, match="exactly"):
            PromptBuild.model_validate(data)


class TestDeclarations:
    def test_enum_needs_values(self) -> None:
        with pytest.raises(ValidationError):
            VariableSpec(name="x", type="enum")

    def test_values_only_on_enum(self) -> None:
        with pytest.raises(ValidationError):
            VariableSpec(name="x", values=("a",))

    def test_required_has_no_default(self) -> None:
        with pytest.raises(ValidationError):
            VariableSpec(name="x", default="a")

    def test_default_must_fit_type(self) -> None:
        with pytest.raises(ValidationError):
            VariableSpec(name="x", type="number", required=False, default="three")

    def test_names(self) -> None:
        with pytest.raises(ValidationError):
            VariableSpec(name="a-b")


class TestRender:
    def test_missing_required_is_refused(self) -> None:
        with pytest.raises(PromptRenderError, match="required"):
            render_build(_build("{{ x }}", VariableSpec(name="x")), {})

    def test_none_counts_as_missing(self) -> None:
        build = _build("[{{ x }}]", VariableSpec(name="x", required=False))
        assert render_build(build, {"x": None}) == "[]"

    def test_optional_default(self) -> None:
        build = _build("{{ x }}", VariableSpec(name="x", type="number", required=False, default=2))
        assert render_build(build, {}) == "2"

    def test_extra_values_are_ignored(self) -> None:
        assert render_build(_build("plain"), {"x": 1}) == "plain"

    def test_unsupported_value(self) -> None:
        with pytest.raises(PromptRenderError):
            value_text(VariableSpec(name="x"), b"bytes")

    def test_nan_is_refused(self) -> None:
        with pytest.raises(PromptRenderError):
            value_text(VariableSpec(name="x", type="number"), float("nan"))

    def test_non_string_json_keys_are_refused(self) -> None:
        with pytest.raises(PromptRenderError):
            value_text(VariableSpec(name="x", type="json"), {1: "a"})


class TestBindings:
    def test_column_and_value(self) -> None:
        build = _build("{{ q }} / {{ style }}", VariableSpec(name="q"), VariableSpec(name="style"))
        values = bind_row(
            build,
            {
                "q": Binding(column="question"),
                "style": Binding(value="brief"),
                "extra": Binding(value=1),
            },
            {"question": "Why?"},
        )
        assert render_build(build, values) == "Why? / brief"

    def test_missing_column_is_an_error(self) -> None:
        build = _build("{{ q }}", VariableSpec(name="q", required=False))
        with pytest.raises(PromptRenderError, match="column"):
            bind_row(build, {"q": Binding(column="qeustion")}, {"question": "x"})

    def test_binding_names_one_source(self) -> None:
        with pytest.raises(ValidationError):
            Binding()
        with pytest.raises(ValidationError):
            Binding(column="a", value="b")


class TestVectors:
    """The shared vectors every port of the renderer is held to (regenerated by
    ``scripts/gen_prompt_build_vectors.py``)."""

    vectors: dict[str, Any] = json.loads(VECTORS.read_text(encoding="utf-8"))

    def test_compiler_version(self) -> None:
        assert self.vectors["compiler_version"] == COMPILER_VERSION

    @pytest.mark.parametrize("case", vectors["compile"], ids=lambda c: c["name"])
    def test_compile_case(self, case: dict[str, Any]) -> None:
        result = compile_prompt(
            [PromptPart.model_validate(p) for p in case["parts"]],
            [VariableSpec.model_validate(v) for v in case["variables"]],
        )
        assert list(result.referenced) == case["referenced"]
        if "issues" in case:
            assert result.build is None
            got = [
                {"code": i.code, "part": i.part, "line": i.line, "column": i.column}
                for i in result.issues
            ]
            assert got == case["issues"]
            return
        assert result.build is not None
        assert result.build.model_dump(mode="json") == case["build"]
        assert result.build.template == case["template"]
        for render in case["renders"]:
            if render.get("error"):
                with pytest.raises(PromptRenderError):
                    render_build(result.build, render["values"])
            else:
                assert render_build(result.build, render["values"]) == render["text"]

    @pytest.mark.parametrize("case", vectors["numbers"], ids=lambda c: repr(c["value"]))
    def test_number_text(self, case: dict[str, Any]) -> None:
        assert value_text(VariableSpec(name="n", type="number"), case["value"]) == case["text"]
