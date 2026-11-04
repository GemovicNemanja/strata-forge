"""Primary classifier execution with required trusted streamed inputs and read authority."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import stat
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol, cast

from pydantic import BaseModel, ConfigDict, Field

from strata_forge.pipelines._classifier_checkpoint import load_classifier_base, reload_classifier
from strata_forge.pipelines._common import check_engine_version, emit
from strata_forge.pipelines.classifier_spec import (
    MAX_INPUT_BYTES,
    MAX_ROW_BYTES,
    ClassifierDataset,
    ClassifierTrainSpec,
    ClassifierWorkerError,
    DecoderInitialization,
    parse_json,
)
from strata_forge.training.classifier import (
    ClassifierExample,
    ClassifierRunner,
    _targets,  # pyright: ignore[reportPrivateUsage] — use the SDK's own pure target validation
    initialize_classifier,
)
from strata_forge.training.progress import JsonlProgressWriter, ProgressEvent

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

_SAFE_ERRORS = frozenset(
    {
        "classifier_json_invalid",
        "classifier_authority_unavailable",
        "classifier_read_denied",
        "classifier_input_limit",
        "classifier_population_invalid",
        "classifier_row_invalid",
        "classifier_base_invalid",
        "classifier_checkpoint_invalid",
        "classifier_artifact_invalid",
        "classifier_attempt_invalid",
        "classifier_training_failed",
        "classifier_ref_mismatch",
    }
)


class TrainingReadRequest(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        strict=True,
        extra="forbid",
        hide_input_in_errors=True,
        revalidate_instances="always",
    )
    purpose: Literal["classifier_training"] = "classifier_training"
    run_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    spec_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    ref: str = Field(min_length=43, max_length=256)
    repo_type: Literal["model", "dataset"]


class TrainingReadDecision(TrainingReadRequest):
    authorized: bool
    custody: Literal["unsealed", "sealed", "locked", "unknown"]
    policy_version: str = Field(pattern=r"^[A-Za-z0-9._-]{1,64}$")


class TrainingReadAuthority(Protocol):
    async def decide(self, request: TrainingReadRequest) -> TrainingReadDecision: ...


@dataclass(frozen=True)
class ResolvedClassifierCheckpoint:
    """Trusted source output; neither a public DTO nor a permission proof."""

    ref: str
    directory: Path


class ClassifierInputSource(Protocol):
    async def resolve_model(self, ref: str) -> ResolvedClassifierCheckpoint: ...
    def rows(self, source: ClassifierDataset) -> AsyncIterator[bytes]: ...


class ClassifierWorkerResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    artifact_dir: Path
    receipt_path: Path
    receipt_sha256: str
    steps: int
    trained_rows: int
    overflow_rows: int


async def _authorize(spec: ClassifierTrainSpec, authority: TrainingReadAuthority) -> None:
    reads: tuple[tuple[str, Literal["model", "dataset"]], ...] = (
        (spec.base.ref, "model"),
        (spec.dataset.ref, "dataset"),
    )
    for ref, kind in reads:
        denied = False
        try:
            request = TrainingReadRequest(
                run_id=spec.run_id, spec_sha256=spec.digest(), ref=ref, repo_type=kind
            )
            decision = TrainingReadDecision.model_validate(await authority.decide(request))
            denied = (
                not decision.authorized
                or decision.custody != "unsealed"
                or any(
                    getattr(decision, key) != value for key, value in request.model_dump().items()
                )
            )
        except Exception:
            denied = True
        if denied:
            raise ClassifierWorkerError("classifier_read_denied")


def _example(spec: ClassifierTrainSpec, raw: bytes) -> ClassifierExample:
    value = parse_json(raw, MAX_ROW_BYTES)
    if type(value) is not dict:
        raise ClassifierWorkerError("classifier_row_invalid")
    mapping = cast("dict[str, object]", value)
    fields = {role: mapping[column] for role, column in spec.dataset.columns.items()}
    distribution = fields.get("label_dist")
    if distribution is not None:
        if type(distribution) is not list:
            raise ClassifierWorkerError("classifier_row_invalid")
        fields["label_dist"] = tuple(cast("list[object]", distribution))
    if spec.dataset.format == "nli_pair":
        rendering = spec.training.rendering
        if rendering is None:
            raise ClassifierWorkerError("classifier_row_invalid")
        # Model validation remains the SDK's authority; no PID conversion or guessed labels.
        from strata_forge.training.verifier_config import render_verifier_input

        rendered = render_verifier_input(
            rendering, fields.get("premises"), fields.get("hypothesis")
        )
        if rendered.error_code is not None or rendered.text is None:
            raise ClassifierWorkerError("classifier_row_invalid")
        premises = fields["premises"]
        if type(premises) is not list:
            raise ClassifierWorkerError("classifier_row_invalid")
        fields.update(
            text=rendered.text,
            text_pair=rendered.text_pair,
            premises=tuple(cast("list[object]", premises)),
        )
    row = ClassifierExample.model_validate(fields)
    _targets(spec.training.sdk_config(), row)
    return row


async def _rows(
    spec: ClassifierTrainSpec, source: ClassifierInputSource
) -> tuple[tuple[ClassifierExample, ...], int, str]:
    selection = spec.dataset.selection
    rows: list[ClassifierExample] = []
    ids: set[str] = set()
    used, ordinal = 0, 0
    digest = hashlib.sha256()
    iterator = source.rows(spec.dataset)
    try:
        while selection.row_end is None or ordinal < selection.row_end:
            try:
                raw = await anext(iterator)
            except StopAsyncIteration:
                break
            # Count cap+1 and byte budgets precede parsing the later item. Range-start work is
            # explicitly bounded by the spec plus this budget; it is not a free unbounded skip.
            if ordinal >= selection.row_start and len(rows) >= selection.row_limit:
                raise ClassifierWorkerError("classifier_input_limit")
            if (
                type(raw) is not bytes
                or len(raw) > MAX_ROW_BYTES
                or used + len(raw) > MAX_INPUT_BYTES
            ):
                raise ClassifierWorkerError("classifier_input_limit")
            used += len(raw)
            ordinal += 1
            if ordinal <= selection.row_start:
                continue
            row = _example(spec, raw)
            if row.item_id in ids:
                raise ClassifierWorkerError("classifier_population_invalid")
            ids.add(row.item_id)
            rows.append(row)
            digest.update(row.model_dump_json().encode("utf-8") + b"\n")
    finally:
        close = getattr(iterator, "aclose", None)
        if close is not None:
            await close()
    if not rows:
        raise ClassifierWorkerError("classifier_population_invalid")
    return tuple(rows), used, digest.hexdigest()


def _reserve(path: Path, spec: ClassifierTrainSpec) -> None:
    if (
        not path.is_absolute()
        or ".." in path.parts
        or any(p.is_symlink() for p in (path, *path.parents))
    ):
        raise ClassifierWorkerError("classifier_attempt_invalid")
    path.mkdir(mode=0o700, parents=False, exist_ok=True)
    info = path.stat()
    if info.st_uid != os.getuid() or info.st_mode & 0o077 or any(path.iterdir()):
        raise ClassifierWorkerError("classifier_attempt_invalid")
    with (path / ".classifier-worker.json").open("x") as file:
        json.dump({"version": 1, "run_id": spec.run_id, "spec_sha256": spec.digest()}, file)


def _inventory(path: Path) -> dict[str, dict[str, str | int]]:
    result: dict[str, dict[str, str | int]] = {}
    for file in path.iterdir():
        info = file.lstat()
        if not stat.S_ISREG(info.st_mode):
            raise ClassifierWorkerError("classifier_artifact_invalid")
        digest = hashlib.sha256()
        with file.open("rb") as stream:
            while block := stream.read(1024 * 1024):
                digest.update(block)
        result[file.name] = {"sha256": digest.hexdigest(), "size_bytes": info.st_size}
    return result


async def _execute(
    spec: ClassifierTrainSpec,
    authority: TrainingReadAuthority,
    source: ClassifierInputSource,
    attempt_dir: Path,
    writer: JsonlProgressWriter | None,
) -> ClassifierWorkerResult:
    spec = ClassifierTrainSpec.model_validate(spec)
    spec = ClassifierTrainSpec.model_validate(spec.model_dump(mode="json", warnings=False))
    check_engine_version(spec.engine_version)
    await _authorize(spec, authority)
    rows, used, population_digest = await _rows(spec, source)
    await _authorize(spec, authority)
    resolved = await source.resolve_model(spec.base.ref)
    if type(resolved) is not ResolvedClassifierCheckpoint or resolved.ref != spec.base.ref:
        raise ClassifierWorkerError("classifier_ref_mismatch")
    await _authorize(spec, authority)
    base = await load_classifier_base(resolved.directory)
    initialization = spec.base.initialization
    ids: tuple[int, ...] = ()
    equivalence = None
    if isinstance(initialization, DecoderInitialization):
        if any(row.text_pair is not None for row in rows):
            raise ClassifierWorkerError("classifier_row_invalid")
        tokens = [
            base.tokenizer.encode(initialization.label_tokens[label], add_special_tokens=False)
            for label in spec.training.labels
        ]
        if any(len(token) != 1 for token in tokens):
            raise ClassifierWorkerError("classifier_row_invalid")
        ids = tuple(token[0] for token in tokens)
        probes: list[str] = []
        for row in rows:
            encoded = base.tokenizer(row.text, truncation=False)
            if len(encoded["input_ids"]) > spec.training.max_length:
                if spec.training.overflow == "error":
                    raise ClassifierWorkerError("classifier_row_invalid")
                continue
            probes.append(row.text)
            if len(probes) == 2:
                break
        if not probes:
            raise ClassifierWorkerError("classifier_population_invalid")
        equivalence = dict(
            base.tokenizer(probes, padding=True, truncation=False, return_tensors="pt")
        )
    initialized = await initialize_classifier(
        base.model,
        labels=spec.training.labels,
        initializer_id=initialization.kind,
        verbalizer_ids=ids,
        seed=spec.training.seed,
        equivalence_inputs=equivalence,
    )
    await _authorize(spec, authority)
    _reserve(attempt_dir, spec)
    trained = await ClassifierRunner(spec.training.sdk_config()).train(
        initialized=initialized,
        tokenizer=base.tokenizer,
        rows=rows,
        output_dir=attempt_dir / "training",
    )
    await _authorize(spec, authority)
    artifact = Path(trained.artifact_dir)
    await reload_classifier(artifact)
    inventory = await asyncio.to_thread(_inventory, artifact)
    await _authorize(spec, authority)
    receipt = {
        "version": 1,
        "run_id": spec.run_id,
        "spec_sha256": spec.digest(),
        "engine_version": spec.engine_version,
        "base": spec.base.model_dump(mode="json"),
        "dataset": spec.dataset.model_dump(mode="json"),
        "training": spec.training.model_dump(mode="json"),
        "initialization": initialized.receipt.model_dump(mode="json"),
        "steps": trained.steps,
        "selected_rows": len(rows),
        "trained_rows": trained.trained_rows,
        "overflow_rows": len(trained.overflow_ids),
        "population_sha256": population_digest,
        "stream_bytes": used,
        "prototype_max_row_bytes": MAX_ROW_BYTES,
        "prototype_max_input_bytes": MAX_INPUT_BYTES,
        "artifact_files": inventory,
        "publication": "not_supported",
        "resume": "not_supported",
    }
    raw = json.dumps(receipt, sort_keys=True, ensure_ascii=False, allow_nan=False).encode("utf-8")
    if len(raw) > 128 * 1024:
        raise ClassifierWorkerError("classifier_artifact_invalid")
    receipt_path = attempt_dir / "training-receipt.json"
    with receipt_path.open("xb") as stream:
        stream.write(raw)
    # Native threads never own this writer. A cancelled await cannot later emit success when
    # an uncancellable native operation finally returns. No transport lease is released here.
    emit(
        writer,
        ProgressEvent(
            kind="end",
            stage="run",
            step=trained.steps,
            total_steps=trained.steps,
            metrics={
                "trained_rows": float(trained.trained_rows),
                "overflow_rows": float(len(trained.overflow_ids)),
            },
        ),
    )
    return ClassifierWorkerResult(
        artifact_dir=artifact,
        receipt_path=receipt_path,
        receipt_sha256=hashlib.sha256(raw).hexdigest(),
        steps=trained.steps,
        trained_rows=trained.trained_rows,
        overflow_rows=len(trained.overflow_ids),
    )


async def run_classifier_training(
    spec: ClassifierTrainSpec,
    *,
    authority: TrainingReadAuthority,
    source: ClassifierInputSource,
    attempt_dir: Path,
    writer: JsonlProgressWriter | None = None,
) -> ClassifierWorkerResult:
    """Execute only with explicitly installed trusted authority/source; never infer a grant."""
    code = "classifier_training_failed"
    try:
        return await _execute(spec, authority, source, attempt_dir, writer)
    except ClassifierWorkerError as error:
        if str(error) in _SAFE_ERRORS:
            code = str(error)
    except asyncio.CancelledError:
        raise
    except Exception:
        code = "classifier_training_failed"
    raise ClassifierWorkerError(code)
