"""Actual CPU optimizer, safe reload and canonical worker owner over fictional local inputs."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shlex
import sys
from contextlib import suppress
from typing import TYPE_CHECKING, Any

import httpx

from strata_forge import __version__
from strata_forge.compute.backends.base import secrets_guarded_script
from strata_forge.pipelines import CLASSIFIER_AUTHORITY_SECRET, classifier_worker
from strata_forge.pipelines import _classifier_worker_policy as policy
from strata_forge.pipelines import _classifier_worker_source as source
from strata_forge.pipelines.classifier_spec import ClassifierTrainSpec
from tests.unit.pipelines.test_classifier_runner import save_base
from tests.unit.pipelines.test_classifier_spec import payload
from tests.unit.pipelines.test_classifier_worker_policy import frame, reply
from tests.unit.pipelines.test_classifier_worker_source import File, Folder, Hub
from tests.unit.pipelines.test_common import _deliver

if TYPE_CHECKING:
    from pathlib import Path

    import pytest

# Only the optional native ML/Arrow stack and fictional upstream metadata are dynamic here.


async def test_canonical_worker_trains_and_retains_safe_result_with_no_external_io(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    base = tmp_path / "fictional-base"
    save_base(base)
    parquet = tmp_path / "fictional.parquet"
    pq.write_table(
        pa.table({"id": ["fictional-a", "fictional-b"], "text": ["a", "b"], "label": ["E", "C"]}),
        parquet,
    )
    models = Hub({file.name: file.read_bytes() for file in base.iterdir() if file.is_file()})
    datasets = Hub(
        {
            "README.md": (
                b"---\nconfigs:\n- config_name: default\n  data_files:\n"
                b"  - split: train\n    path: data/train.parquet\n---\n"
            ),
            "data/train.parquet": parquet.read_bytes(),
        }
    )

    class FixedHub:
        RepoFile, RepoFolder = File, Folder

        def HfApi(self, **_: Any) -> FixedHub:  # noqa: N802 — fixed optional API
            return self

        def repo_info(self, **kwargs: Any) -> Any:
            return models.repo_info(**kwargs)

        def list_repo_tree(self, repo: str, **kwargs: Any) -> Any:
            return (models if repo == "fixture/base" else datasets).list_repo_tree(repo, **kwargs)

        def hf_hub_download(self, repo: str, name: str, **kwargs: Any) -> str:
            return (models if repo == "fixture/base" else datasets).hf_hub_download(
                repo, name, **kwargs
            )

    fixed_hub = FixedHub()
    monkeypatch.setattr(source.HubClassifierSource, "_hub", lambda _: fixed_hub)
    intent = payload()
    intent["engine_version"] = __version__
    spec = ClassifierTrainSpec.model_validate(intent)
    credential = "strata_worker_" + "x" * 43
    seen: list[dict[str, Any]] = []

    def transport(request: httpx.Request) -> httpx.Response:
        value = json.loads(request.content)
        assert value["purpose"] == "classifier_training"
        assert value["run_id"] == spec.run_id
        assert value["spec_sha256"] == spec.digest()
        assert set(value) == {"purpose", "run_id", "spec_sha256", "ref", "repo_type"}
        assert request.headers["authorization"] == "Bearer " + credential
        seen.append(value)
        return reply(
            {**value, "authorized": True, "custody": "unsealed", "policy_version": "tiny-v1"}
        )

    original_client = httpx.AsyncClient
    clients = []

    def fixed_client(**kwargs: Any) -> httpx.AsyncClient:
        client = original_client(transport=httpx.MockTransport(transport), **kwargs)
        clients.append(client)
        return client

    monkeypatch.setattr(policy.httpx, "AsyncClient", fixed_client)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setenv("TOKENIZERS_PARALLELISM", "false")
    monkeypatch.setenv("STRATA_RUN_CONFIG", spec.canonical().decode())
    monkeypatch.chdir(tmp_path)
    private = _deliver(
        monkeypatch,
        tmp_path,
        {
            CLASSIFIER_AUTHORITY_SECRET: json.dumps(
                frame(run_id=spec.run_id, spec_sha256=spec.digest())
            )
        },
    )
    assert await classifier_worker.main() == 0
    assert not private.exists()
    assert clients
    assert all(client.is_closed for client in clients)
    output = tmp_path / "strata-classifiers" / spec.run_id
    result = json.loads((output / "worker-result.json").read_text())
    receipt_raw = (output / "training-receipt.json").read_bytes()
    assert result["run_id"] == spec.run_id
    assert result["spec_sha256"] == spec.digest()
    assert result["receipt_sha256"] == hashlib.sha256(receipt_raw).hexdigest()
    assert result["steps"] == 1
    assert result["trained_rows"] == 2
    assert result["publication"] == result["resume"] == "not_supported"
    assert (output / result["artifact_dir"] / "model.safetensors").is_file()
    events = [json.loads(line) for line in (tmp_path / "progress.jsonl").read_text().splitlines()]
    assert [event["kind"] for event in events].count("end") == 1
    assert events[-1]["kind"] == "end"
    assert seen[-2]["repo_type"] == "model"
    assert seen[-1]["repo_type"] == "dataset"
    assert credential not in capsys.readouterr().err
    assert credential not in json.dumps(result)
    assert str(tmp_path) not in json.dumps(result)
    assert (tmp_path / "classifier-result.json").read_bytes() == (
        output / "worker-result.json"
    ).read_bytes()


async def test_installed_fixed_module_refusal_preserves_outer_private_file_cleanup(
    tmp_path: Path,
) -> None:
    private = tmp_path / ".secrets.json"
    private.write_text('{"fictional":"unread-private-canary"}')
    private.chmod(0o600)
    intent = payload()
    intent["engine_version"] = "0.0.0"
    command = shlex.join(
        [sys.executable, "-I", "-B", "-m", "strata_forge.pipelines.classifier_worker"]
    )
    script = secrets_guarded_script(shlex.quote(str(private)), "", command)
    process = await asyncio.create_subprocess_exec(
        "/bin/bash",
        "-c",
        script,
        cwd=tmp_path,
        env={"PATH": os.defpath, "HOME": str(tmp_path), "STRATA_RUN_CONFIG": json.dumps(intent)},
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=15)
    finally:
        if process.returncode is None:
            with suppress(ProcessLookupError):
                process.terminate()
            try:
                await asyncio.wait_for(process.communicate(), timeout=2)
            except TimeoutError:
                with suppress(ProcessLookupError):
                    process.kill()
                # A still-held pipe is an outer-owner failure; never detach a new group.
                with suppress(TimeoutError):
                    await asyncio.wait_for(process.communicate(), timeout=2)
    assert process.returncode == 1
    assert stdout == b""
    assert stderr == b"run failed: classifier_preflight_refused\n"
    assert not private.exists()
