"""Installed primary-classifier worker with a separate training-purpose authority channel."""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from strata_forge.pipelines._classifier_worker_policy import (
    ClassifierPolicyRPC,
    parse_classifier_authority,
)
from strata_forge.pipelines._classifier_worker_source import HubClassifierSource, settled_io
from strata_forge.pipelines._common import (
    RunSecrets,
    check_engine_version,
    emit,
    results_dir,
    runner_main,
)
from strata_forge.pipelines.classifier_result import (
    MAX_CLASSIFIER_RESULT_BYTES,
    ClassifierLocalResult,
)
from strata_forge.pipelines.classifier_runner import (
    ClassifierWorkerResult,
    TrainingReadRequest,
    run_classifier_training,
)
from strata_forge.pipelines.classifier_spec import (
    ClassifierTrainSpec,
    ClassifierWorkerError,
    load_classifier_spec,
)
from strata_forge.training.progress import ProgressEvent

if TYPE_CHECKING:
    from strata_forge.training.progress import JsonlProgressWriter


def _parent(directory: Path) -> None:
    if (
        not directory.is_absolute()
        or ".." in directory.parts
        or any(path.is_symlink() for path in (directory, *directory.parents))
    ):
        raise ClassifierWorkerError("classifier_attempt_invalid")
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)


def _result(result: ClassifierWorkerResult, spec: ClassifierTrainSpec) -> None:
    artifact = result.artifact_dir.relative_to(result.receipt_path.parent)
    raw = ClassifierLocalResult.model_validate(
        {
            "version": 1,
            "profile": "primary_classifier_hf_worker_v1",
            "run_id": spec.run_id,
            "spec_sha256": spec.digest(),
            "engine_version": spec.engine_version,
            "artifact_dir": artifact.as_posix(),
            "receipt_file": result.receipt_path.name,
            "receipt_sha256": result.receipt_sha256,
            "steps": result.steps,
            "trained_rows": result.trained_rows,
            "overflow_rows": result.overflow_rows,
            "publication": "not_supported",
            "resume": "not_supported",
        }
    ).canonical()
    if len(raw) > MAX_CLASSIFIER_RESULT_BYTES:
        raise ClassifierWorkerError("classifier_result_invalid")
    paths = (
        result.receipt_path.parent / "worker-result.json",
        Path.cwd() / "classifier-result.json",
    )
    for path in paths:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)


async def execute_classifier_worker(
    spec: ClassifierTrainSpec,
    *,
    secrets: RunSecrets,
    writer: JsonlProgressWriter | None = None,
    output_dir: Path | None = None,
) -> ClassifierWorkerResult:
    """Run the pinned HF pilot profile; native input/output and population-v2 remain separate."""
    spec = ClassifierTrainSpec.model_validate(spec.model_dump(mode="json"))
    check_engine_version(spec.engine_version)
    if secrets.classifier_authority is None:
        raise ClassifierWorkerError("classifier_authority_missing")
    frame = parse_classifier_authority(
        secrets.classifier_authority, run_id=spec.run_id, spec_sha256=spec.digest()
    )
    directory = output_dir or results_dir(spec.run_id, name="strata-classifiers")
    await settled_io(lambda: _parent(directory.parent))
    try:
        async with ClassifierPolicyRPC(frame) as policy:
            source = HubClassifierSource(
                spec, authority=policy, token=secrets.hub_credential(), parent=directory.parent
            )
            try:
                emit(
                    writer,
                    ProgressEvent(
                        kind="phase",
                        stage="load_model",
                        message="Preparing pinned classifier inputs",
                    ),
                )
                # The SDK's local end event precedes this adapter's source/client cleanup.
                # Only this outer installed worker emits its final progress event.
                result = await run_classifier_training(
                    spec, authority=policy, source=source, attempt_dir=directory, writer=None
                )
            finally:
                await source.aclose()
            sources: tuple[tuple[str, Literal["model", "dataset"]], ...] = (
                (spec.base.ref, "model"),
                (spec.dataset.ref, "dataset"),
            )
            for ref, kind in sources:
                await policy.decide(
                    TrainingReadRequest(
                        run_id=spec.run_id, spec_sha256=spec.digest(), ref=ref, repo_type=kind
                    )
                )
        await settled_io(lambda: _result(result, spec))
    except asyncio.CancelledError:
        raise
    except Exception:
        raise ClassifierWorkerError("classifier_training_failed") from None
    emit(
        writer,
        ProgressEvent(
            kind="end",
            stage="run",
            step=result.steps,
            total_steps=result.steps,
            metrics={
                "trained_rows": float(result.trained_rows),
                "overflow_rows": float(result.overflow_rows),
            },
        ),
    )
    return result


async def main() -> int:
    """Use the preinstalled engine and canonical Task-secret/progress owner."""
    try:
        raw = os.environ.get("STRATA_RUN_CONFIG", "").encode("utf-8")
        spec = load_classifier_spec(raw)
        check_engine_version(spec.engine_version)
    except Exception:
        # The transport's outer shell trap owns the unread secret file on refusal.
        print("run failed: classifier_preflight_refused", file=sys.stderr, flush=True)
        return 1
    return await runner_main(
        lambda writer, secrets: execute_classifier_worker(spec, secrets=secrets, writer=writer)
    )


if __name__ == "__main__":  # pragma: no cover
    sys.exit(asyncio.run(main()))
