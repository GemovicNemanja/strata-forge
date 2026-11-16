"""The authorized archived Strata prompts retain their exact declared development semantics."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
from scripts.import_operational_prompts import DEFAULT_FIXTURE, import_operational_fixture

from strata_forge.prompts.composition_errors import PromptCompositionError

if TYPE_CHECKING:
    from pathlib import Path


async def test_all_six_modules_and_seven_surfaces_render_without_semantic_changes() -> None:
    report = await import_operational_fixture()
    assert report["modules"] == 6
    assert report["surfaces"] == 7
    cases, evidence = report["cases"], report["evidence"]
    assert isinstance(cases, list)
    assert isinstance(evidence, str)
    assert len(cases) == 7
    assert "no provider dispatch" in evidence


async def test_operational_import_refuses_unaudited_source_drift(tmp_path: Path) -> None:
    value = json.loads(DEFAULT_FIXTURE.read_text(encoding="utf-8"))
    value["modules"][0]["parts"][0]["text"] += "changed"
    path = tmp_path / "changed.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(PromptCompositionError, match="operational_fixture_drift"):
        await import_operational_fixture(path)
