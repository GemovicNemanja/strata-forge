"""Recompile the frozen cross-language source corpus against its exact artifact identities."""

from __future__ import annotations

import json
from pathlib import Path

from strata_forge.prompts.composition import (
    CompositionRegistry,
    PromptBuild,
    PromptRender,
    compile_surface,
    render_build,
)


async def test_frozen_build_and_render_fixture_corpus() -> None:
    path = Path(__file__).parents[2] / "fixtures/prompts/composition-renders-v1.json"
    fixture = json.loads(path.read_text(encoding="utf-8"))
    for case in fixture["cases"]:
        registry = CompositionRegistry()
        for definition in case["definitions"]:
            await registry.put(definition)
        build = await compile_surface(registry, case["coordinate"])
        rendered = await render_build(build, case["values"])
        assert build.canonical.hex() == case["build_canonical_utf8_hex"]
        assert build.build_id == case["build_id"]
        assert rendered.canonical.hex() == case["render_canonical_utf8_hex"]
        assert rendered.render_id == case["render_id"]
        assert rendered.wire_payload()["output"] == case["expected_output"]
        assert PromptBuild(canonical=build.canonical) == build
        assert PromptRender(canonical=rendered.canonical) == rendered
