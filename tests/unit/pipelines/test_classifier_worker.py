"""Installed worker composition, final progress and preflight ordering without research models."""

from __future__ import annotations

import asyncio
import json
import os
import shlex
import sys
from typing import TYPE_CHECKING, Any

import pytest
from pydantic import SecretStr

from strata_forge.compute.backends.base import secrets_guarded_script
from strata_forge.pipelines import classifier_worker as worker
from strata_forge.pipelines._common import RunSecrets
from strata_forge.pipelines.classifier_runner import ClassifierWorkerResult
from strata_forge.pipelines.classifier_spec import ClassifierTrainSpec, ClassifierWorkerError
from tests.unit.pipelines.test_classifier_spec import payload
from tests.unit.pipelines.test_classifier_worker_policy import frame

if TYPE_CHECKING:
    from pathlib import Path

# Fixed trusted adapters stand in for native ML and HTTP here; no model bytes are fetched.


async def test_backend_outer_trap_removes_unread_secret_on_preflight_failure(
    tmp_path: Path,
) -> None:
    secret_file = tmp_path / "private-task.json"
    secret_file.write_text('{"fictional":"private-canary"}')
    secret_file.chmod(0o600)
    command = shlex.join([sys.executable, "-I", "-B", "-c", "raise SystemExit(23)"])
    script = secrets_guarded_script(shlex.quote(str(secret_file)), "", command)
    process = await asyncio.create_subprocess_exec(
        "/bin/bash",
        "-c",
        script,
        env={"PATH": os.defpath},
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=10)
    assert process.returncode == 23
    assert stdout == stderr == b""
    assert not secret_file.exists()


class Writer:
    def __init__(self) -> None:
        self.events: list[Any] = []

    def emit(self, event: Any) -> None:
        self.events.append(event)


def prepared(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> tuple[Any, ...]:
    monkeypatch.chdir(tmp_path)
    spec = ClassifierTrainSpec.model_validate(payload())
    events = []

    class Policy:
        def __init__(self, authority: Any) -> None:
            assert authority.run_id == spec.run_id

        async def __aenter__(self) -> Policy:
            events.append("policy-open")
            return self

        async def decide(self, request: Any) -> Any:
            events.append("read-" + request.repo_type)
            return None

        async def __aexit__(self, *_: Any) -> None:
            events.append("policy-closed")

    class Source:
        def __init__(self, *_: Any, **kwargs: Any) -> None:
            assert kwargs["token"] is False

        async def aclose(self) -> None:
            events.append("source-closed")

    async def execute(actual: Any, **kwargs: Any) -> ClassifierWorkerResult:
        assert actual == spec
        assert kwargs["writer"] is None
        events.append("trained")
        output = kwargs["attempt_dir"]
        output.mkdir(mode=0o700)
        artifact = output / "training/artifact"
        artifact.mkdir(parents=True)
        receipt = output / "training-receipt.json"
        receipt.write_text("{}")
        return ClassifierWorkerResult(
            artifact_dir=artifact,
            receipt_path=receipt,
            receipt_sha256="a" * 64,
            steps=1,
            trained_rows=2,
            overflow_rows=0,
        )

    monkeypatch.setattr(worker, "check_engine_version", lambda _: None)
    monkeypatch.setattr(worker, "ClassifierPolicyRPC", Policy)
    monkeypatch.setattr(worker, "HubClassifierSource", Source)
    monkeypatch.setattr(worker, "run_classifier_training", execute)
    secrets = RunSecrets(classifier_authority=SecretStr(json.dumps(frame())))
    return spec, events, secrets


async def test_final_result_and_end_follow_cleanup_and_fresh_policy(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    spec, events, secrets = prepared(monkeypatch, tmp_path)
    writer = Writer()
    result = await worker.execute_classifier_worker(
        spec, secrets=secrets, writer=writer, output_dir=tmp_path / "attempt"
    )
    assert events == [
        "policy-open",
        "trained",
        "source-closed",
        "read-model",
        "read-dataset",
        "policy-closed",
    ]
    receipt = json.loads((result.receipt_path.parent / "worker-result.json").read_text())
    assert receipt["run_id"] == spec.run_id
    assert receipt["spec_sha256"] == spec.digest()
    assert receipt["artifact_dir"] == "training/artifact"
    assert receipt["publication"] == "not_supported"
    assert receipt["resume"] == "not_supported"
    assert (tmp_path / "classifier-result.json").read_bytes() == (
        result.receipt_path.parent / "worker-result.json"
    ).read_bytes()
    assert [event.kind for event in writer.events] == ["phase", "end"]


async def test_evaluation_secret_cannot_authorize_classifier_or_source_io(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    spec, events, _ = prepared(monkeypatch, tmp_path)
    with pytest.raises(ClassifierWorkerError, match=r"^classifier_authority_missing$"):
        await worker.execute_classifier_worker(
            spec,
            secrets=RunSecrets(eval_authority=SecretStr(json.dumps(frame()))),
            output_dir=tmp_path / "attempt",
        )
    assert not events
    assert not (tmp_path / "attempt").exists()


async def test_engine_mismatch_precedes_policy_or_source_construction(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    spec, events, secrets = prepared(monkeypatch, tmp_path)

    def mismatch(_: Any) -> None:
        raise ClassifierWorkerError("fictional_engine_refused")

    monkeypatch.setattr(worker, "check_engine_version", mismatch)
    with pytest.raises(ClassifierWorkerError, match=r"^fictional_engine_refused$"):
        await worker.execute_classifier_worker(
            spec, secrets=secrets, output_dir=tmp_path / "attempt"
        )
    assert not events


@pytest.mark.parametrize("failed", ["source", "policy"])
async def test_cleanup_failure_cannot_emit_final_end_or_worker_result(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, failed: str
) -> None:
    spec, _, secrets = prepared(monkeypatch, tmp_path)
    writer = Writer()
    actual_source, actual_policy = worker.HubClassifierSource, worker.ClassifierPolicyRPC

    class Source(actual_source):
        async def aclose(self) -> None:
            await super().aclose()
            if failed == "source":
                raise RuntimeError("private failure must not surface")

    class Policy(actual_policy):
        async def __aexit__(self, *_: Any) -> None:
            await super().__aexit__()
            if failed == "policy":
                raise RuntimeError("private failure must not surface")

    monkeypatch.setattr(worker, "HubClassifierSource", Source)
    monkeypatch.setattr(worker, "ClassifierPolicyRPC", Policy)
    with pytest.raises(ClassifierWorkerError, match=r"^classifier_training_failed$"):
        await worker.execute_classifier_worker(
            spec, secrets=secrets, writer=writer, output_dir=tmp_path / "attempt"
        )
    assert all(event.kind != "end" for event in writer.events)
    assert not (tmp_path / "attempt/worker-result.json").exists()
    assert not (tmp_path / "classifier-result.json").exists()


async def test_cancellation_cannot_emit_late_success(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    spec, _, secrets = prepared(monkeypatch, tmp_path)
    entered, release = asyncio.Event(), asyncio.Event()
    actual_source = worker.HubClassifierSource

    class Source(actual_source):
        async def aclose(self) -> None:
            entered.set()
            await release.wait()

    monkeypatch.setattr(worker, "HubClassifierSource", Source)
    writer = Writer()
    task = asyncio.create_task(
        worker.execute_classifier_worker(
            spec, secrets=secrets, writer=writer, output_dir=tmp_path / "attempt"
        )
    )
    await entered.wait()
    task.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert all(event.kind != "end" for event in writer.events)
    assert not (tmp_path / "attempt/worker-result.json").exists()


async def test_main_rejects_invalid_preflight_before_secret_loader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = []

    def refuse(*_: Any, **__: Any) -> None:
        raise RuntimeError("private spec input must not surface")

    async def runner(*_: Any, **__: Any) -> int:
        called.append(True)
        return 0

    monkeypatch.setattr(worker, "load_classifier_spec", refuse)
    monkeypatch.setattr(worker, "runner_main", runner)
    assert await worker.main() == 1
    assert not called


@pytest.mark.parametrize("raw", ['{"schema_version":1,"schema_version":1}', "[]", "x" * 32_769])
async def test_main_uses_bounded_closed_classifier_parser_before_private_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, raw: str
) -> None:
    private = tmp_path / ".secrets.json"
    private.write_text('{"fictional":"private-canary"}')
    private.chmod(0o600)
    monkeypatch.setenv("STRATA_RUN_CONFIG", raw)
    monkeypatch.setenv("FORGE_SECRETS_FILE", str(private))
    assert await worker.main() == 1
    assert private.is_file()
