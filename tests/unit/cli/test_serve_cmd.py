"""Unit tests for `strata_forge.cli.serve`."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from typer.testing import CliRunner

from strata_forge.cli.main import app
from strata_forge.compute.job import Job

if TYPE_CHECKING:
    from pathlib import Path

runner = CliRunner()


class TestEmit:
    def test_vllm_emits_yaml(self) -> None:
        result = runner.invoke(
            app, ["serve", "vllm", "--model", "meta-llama/Llama-3.1-8B-Instruct"]
        )
        assert result.exit_code == 0
        assert "vllm serve" in result.output
        assert "meta-llama/Llama-3.1-8B-Instruct" in result.output

    def test_tgi_emits_yaml(self) -> None:
        result = runner.invoke(app, ["serve", "tgi", "--model", "mistralai/Mistral-7B-v0.1"])
        assert result.exit_code == 0
        assert "docker run" in result.output

    def test_sglang_emits_yaml(self) -> None:
        result = runner.invoke(app, ["serve", "sglang", "--model", "Qwen/Qwen2-7B"])
        assert result.exit_code == 0
        assert "sglang.launch_server" in result.output

    def test_vllm_custom_flags(self) -> None:
        result = runner.invoke(
            app,
            [
                "serve",
                "vllm",
                "--model",
                "m",
                "--port",
                "9000",
                "--tp",
                "4",
                "--max-model-len",
                "16384",
                "--dtype",
                "bfloat16",
            ],
        )
        assert result.exit_code == 0
        # YAML may wrap long lines, so collapse whitespace before asserting.
        collapsed = " ".join(result.output.split())
        assert "--port 9000" in collapsed
        assert "--tensor-parallel-size 4" in collapsed
        assert "--max-model-len 16384" in collapsed
        assert "--dtype bfloat16" in collapsed


@pytest.fixture
def fake_local_submit(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, Any]:
    state: dict[str, Any] = {"submitted": None}

    class _FakeLocalBackend:
        async def submit(self, task: Any) -> Job:
            state["submitted"] = task
            return Job(id="serve-job-1", backend="local", task_name=task.name)

    monkeypatch.setattr("strata_forge.compute.backends.local.LocalBackend", _FakeLocalBackend)
    monkeypatch.setattr("strata_forge.cli.compute._STATE_DIR", tmp_path / "jobs")
    return state


class TestSubmitLocal:
    def test_vllm_submit_local(self, fake_local_submit: dict[str, Any], tmp_path: Path) -> None:
        result = runner.invoke(
            app,
            [
                "serve",
                "vllm",
                "--model",
                "m",
                "--port",
                "8000",
                "--submit",
                "local",
            ],
        )
        assert result.exit_code == 0
        assert "submitted" in result.output
        assert "serve-job-1" in result.output
        assert "http://localhost:8000/v1" in result.output
        assert (tmp_path / "jobs" / "serve-job-1.json").exists()
