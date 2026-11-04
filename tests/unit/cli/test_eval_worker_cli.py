"""Installed-schema CLI delegates to one worker and never invents an access grant."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from strata_forge.cli.main import app
from strata_forge.pipelines import SPEC_VERSION


def test_installed_eval_schema_command() -> None:
    result = CliRunner().invoke(app, ["eval", "schema"])
    assert result.exit_code == 0
    schema = json.loads(result.output)
    assert schema["additionalProperties"] is False
    assert "evaluation_sets" in schema["properties"]


@pytest.mark.parametrize("mode", ["classify", "score-predictions"])
def test_validate_then_execute_without_trusted_policy_refuses(
    mode: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in ("FORGE_SECRETS_FILE", "STRATA_HF_TOKEN", "HF_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("FORGE_PROGRESS_PATH", str(tmp_path / "progress.jsonl"))
    fixture = Path(__file__).parents[1] / "pipelines/fixtures/evaluation_conformance.json"
    data = json.loads(fixture.read_text())["valid_specs"][0 if mode == "classify" else 1]
    data["output_repo_id"] = None
    data["engine_version"] = SPEC_VERSION
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(data))
    validated = CliRunner().invoke(app, ["eval", "validate-spec", str(path)])
    assert validated.exit_code == 0
    assert json.loads(validated.output)["mode"] == mode.replace("-", "_")
    result = CliRunner().invoke(app, ["eval", mode, str(path)])
    assert result.exit_code == 1
    assert "custody_policy_unavailable" in result.output


def test_invalid_cli_spec_hides_input_and_does_not_attempt_execution(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text('{"callback":"DO_NOT_ECHO_THIS_ROW"}')
    result = CliRunner().invoke(app, ["eval", "validate-spec", str(path)])
    assert result.exit_code == 1
    assert "evaluation_spec_invalid" in result.output
    assert "DO_NOT_ECHO_THIS_ROW" not in result.output
