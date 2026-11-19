"""Regenerate ``tests/fixtures/prompt_build_vectors.json`` from the compiler.

The vectors pin how :mod:`strata_forge.prompts.compiled` compiles and renders, for any port of
the renderer to another language. Run ``uv run python scripts/gen_prompt_build_vectors.py`` after a
deliberate change to the compiler and review the diff: a changed expected value is a changed
behaviour every port must follow.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from strata_forge.prompts.compiled import (
    COMPILER_VERSION,
    PromptPart,
    PromptRenderError,
    VariableSpec,
    compile_prompt,
    render_build,
    value_text,
)

OUT = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "prompt_build_vectors.json"

COMPILE_CASES: list[dict[str, Any]] = [
    {
        "name": "composition",
        "parts": [
            {
                "kind": "module",
                "name": "Voice",
                "label": "v6",
                "text": "Be {{ tone }}.\nUse {x} braces.\n",
            },
            {"kind": "self", "name": "Instructions", "text": "Q: {{question}}\r\nA:"},
        ],
        "variables": [
            {"name": "tone", "type": "enum", "values": ["calm", "brief"]},
            {"name": "question"},
            {"name": "unused", "required": False},
        ],
        "renders": [
            {"values": {"tone": "calm", "question": "why?"}},
            {"values": {"tone": "loud", "question": "why?"}},
            {"values": {"tone": "brief"}},
        ],
    },
    {
        "name": "legacy braces",
        "parts": [
            {
                "kind": "self",
                "name": "Template",
                "dialect": "braces",
                "text": "{{question}} -> {0}, {not valid} {a_b}",
            },
        ],
        "variables": [{"name": "question"}, {"name": "0"}, {"name": "a_b"}],
        "renders": [{"values": {"question": "Q", "0": "zero", "a_b": "ab"}}],
    },
    {
        "name": "typed values",
        "parts": [
            {
                "kind": "self",
                "name": "Template",
                "text": "n={{ n }} f={{ f }} b={{ b }} j={{ j }} s={{ s }} o={{ o }}",
            },
        ],
        "variables": [
            {"name": "n", "type": "number"},
            {"name": "f", "type": "number"},
            {"name": "b", "type": "boolean"},
            {"name": "j", "type": "json"},
            {"name": "s"},
            {"name": "o", "required": False, "default": "fallback"},
        ],
        "renders": [
            {
                "values": {
                    "n": 3,
                    "f": 1.0,
                    "b": True,
                    "j": {"z": [1, 2.5, None], "a": "é\n"},
                    "s": 0.000001,
                }
            },
            {
                "values": {
                    "n": -0.0,
                    "f": 1e-7,
                    "b": False,
                    "j": "text",
                    "s": [True, {"k": 1e21}],
                    "o": "given",
                }
            },
            {"values": {"n": True, "f": 1, "b": False, "j": 1, "s": "x"}},
            {"values": {"n": 1, "f": 2, "b": "yes", "j": 1, "s": "x"}},
        ],
    },
    {
        "name": "unicode is normalized",
        "parts": [{"kind": "self", "name": "Template", "text": "Café {{ x }}"}],
        "variables": [{"name": "x"}],
        "renders": [{"values": {"x": "Café"}}],
    },
    {
        "name": "syntax errors",
        "parts": [
            {"kind": "self", "name": "Template", "text": "{{ a.b }} {% if %} {{ x\n }} {{ y }}"}
        ],
        "variables": [],
        "renders": [],
    },
    {
        "name": "empty parts are skipped",
        "parts": [
            {"kind": "module", "name": "Empty", "text": "\n\n"},
            {"kind": "self", "name": "Template", "text": "only {{ x }}"},
        ],
        "variables": [{"name": "x"}],
        "renders": [{"values": {"x": 1}}],
    },
]

NUMBER_CASES = [
    0.0,
    -0.0,
    1.0,
    -1.5,
    0.1,
    1e-7,
    1.5e-7,
    1e-6,
    123456789.123,
    1e21,
    1e20,
    1.2345e21,
    5e-324,
    1.7976931348623157e308,
    0.000001234,
    100.0,
    2.5e-5,
    1.23e-18,
    7,
    -12,
]


def _compile_case(case: dict[str, Any]) -> dict[str, Any]:
    parts = [PromptPart.model_validate(p) for p in case["parts"]]
    variables = [VariableSpec.model_validate(v) for v in case["variables"]]
    result = compile_prompt(parts, variables)
    out: dict[str, Any] = {
        "name": case["name"],
        "parts": case["parts"],
        "variables": case["variables"],
        "referenced": list(result.referenced),
    }
    if result.build is None:
        out["issues"] = [
            {"code": i.code, "part": i.part, "line": i.line, "column": i.column}
            for i in result.issues
        ]
        return out
    build = result.build
    out["build"] = build.model_dump(mode="json")
    out["template"] = build.template
    renders: list[dict[str, Any]] = []
    for render in case["renders"]:
        try:
            renders.append(
                {"values": render["values"], "text": render_build(build, render["values"])}
            )
        except PromptRenderError:
            renders.append({"values": render["values"], "error": True})
    out["renders"] = renders
    return out


def main() -> None:
    number = VariableSpec(name="n", type="number")
    vectors = {
        "compiler_version": COMPILER_VERSION,
        "compile": [_compile_case(case) for case in COMPILE_CASES],
        "numbers": [{"value": v, "text": value_text(number, v)} for v in NUMBER_CASES],
    }
    OUT.write_text(json.dumps(vectors, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
