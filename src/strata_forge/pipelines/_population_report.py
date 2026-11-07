"""Private local population receipts; a complete manifest is installed only after awaited writes."""

from __future__ import annotations

import json
import os
import stat
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path  # noqa: TC003 — Pydantic resolves result paths at runtime
from typing import TYPE_CHECKING, Literal, Protocol, cast

from strata_forge.pipelines._evaluation_report import (
    _write_items,  # pyright: ignore[reportPrivateUsage] — preserve the existing typed metric/slice payload schema
)
from strata_forge.pipelines._population import (
    PopulationCounts,
    PopulationData,
    PopulationError,
    PopulationMember,
    PopulationOutcome,
    canonical_digest,
    population_counts,
    selected_population_digest,
)

if TYPE_CHECKING:
    from collections.abc import Generator, Sequence
    from typing import BinaryIO

    from strata_forge.pipelines._evaluation_items import EvaluationItem
    from strata_forge.pipelines._scoring import ScoredRow
    from strata_forge.pipelines.evaluation_spec import EvaluationSet


class _ArrowField(Protocol):
    @property
    def name(self) -> str: ...
    def with_nullable(self, nullable: bool) -> _ArrowField: ...


class _ArrowTable(Protocol):
    @property
    def schema(self) -> Sequence[_ArrowField]: ...
    def to_pylist(self) -> list[dict[str, object]]: ...


MAX_REPORT_BYTES = 32 * 1024 * 1024


_SAFE_ERRORS = frozenset(
    {
        "population_job_invalid",
        "credential_invalid",
        "population_empty",
        "population_limit_exceeded",
        "population_metadata_invalid",
        "population_id_duplicate",
        "population_tier_limit_exceeded",
        "population_alignment_invalid",
        "population_selection_mismatch",
        "artifact_directory_not_empty",
        "artifact_directory_not_private",
        "report_limit_exceeded",
        "custody_policy_unavailable",
        "custody_resolution_failed",
        "custody_ref_mismatch",
        "custody_spec_mismatch",
        "custody_denied",
        "model_label_order_mismatch",
        "native_scoring_evidence_invalid",
        "native_scoring_limit_mismatch",
        "scoring_row_alignment_invalid",
        "prediction_overflow_evidence_unavailable",
        "prediction_population_invalid",
        "prediction_id_missing",
        "prediction_id_duplicate",
        "prediction_extra",
        "prediction_manifest_mismatch",
        "classifier_fields_missing",
        "comparison_report_version_unsupported",
        "comparison_population_mismatch",
        "comparison_item_invalid",
        "comparison_id_invalid",
        "gold_population_empty",
        "gold_label_invalid",
        "gold_id_invalid",
        "gold_id_duplicate",
        "probability_column_missing",
        "probability_family_invalid",
        "dataset_files_unsupported",
        "input_revision_mismatch",
        "population_execution_failed",
    }
)


def closed_execution_error(error: Exception) -> PopulationError:
    """Keep fixed codes only, never an upstream exception, subclass behavior or raw context."""
    from pydantic import ValidationError as ModelValidationError

    from strata_forge.pipelines._evaluation_access import EvaluationError

    if (
        type(error) in {PopulationError, EvaluationError}
        and len(error.args) == 1
        and type(error.args[0]) is str
        and error.args[0] in _SAFE_ERRORS
    ):
        return PopulationError(error.args[0])
    if type(error) is ModelValidationError:
        return PopulationError("population_job_invalid")
    return PopulationError("population_execution_failed")


class PopulationScoringResult(PopulationData):
    directory: Path
    report_path: Path
    spec_sha256: str
    counts: PopulationCounts


class PopulationEvaluationResult(PopulationData):
    directory: Path
    report_path: Path
    spec_sha256: str
    counts: PopulationCounts


@dataclass(frozen=True)
class PopulationAttempt:
    directory: Path
    pending: Path
    report_name: Literal["scoring.json", "metrics.json"]


def _new_private_directory(directory: Path) -> None:
    # Only directories this call creates are chmodded; an existing caller path is never taken over.
    directory.mkdir(mode=0o700)
    directory.chmod(0o700)


@contextmanager
def _private_output(path: Path) -> Generator[BinaryIO]:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        # Umask may remove owner bits too. The descriptor names only this newly created file.
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb", closefd=False) as file:
            yield file
    finally:
        os.close(descriptor)


def _write_private(path: Path, content: bytes) -> None:
    with _private_output(path) as file:
        file.write(content)


def reserve_attempt(
    directory: Path, *, run_id: str, report_name: Literal["scoring.json", "metrics.json"]
) -> PopulationAttempt:
    # A public parent is allowed; this exact current-owner 0700 leaf is the read boundary.
    # Missing ancestors are created privately, while existing parents remain caller-managed.
    missing: list[Path] = []
    parent = directory.parent
    while not parent.exists():
        missing.append(parent)
        parent = parent.parent
    for path in reversed(missing):
        _new_private_directory(path)
    with suppress(FileExistsError):
        _new_private_directory(directory)
    metadata = directory.lstat()
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.getuid()
        or stat.S_IMODE(metadata.st_mode) != 0o700
    ):
        raise PopulationError("artifact_directory_not_private")
    if any(directory.iterdir()):
        raise PopulationError("artifact_directory_not_empty")
    try:
        _write_private(
            directory / ".population-attempt.json",
            json.dumps({"run_id": run_id, "status": "started"}).encode("utf-8"),
        )
    except FileExistsError:
        raise PopulationError("artifact_directory_not_empty") from None
    pending = directory / ".population-pending"
    _new_private_directory(pending)
    return PopulationAttempt(directory, pending, report_name)


def validate_outcomes(
    members: Sequence[PopulationMember], outcomes: Sequence[PopulationOutcome]
) -> None:
    if len(members) != len(outcomes):
        raise PopulationError("population_alignment_invalid")
    for member, original in zip(members, outcomes, strict=True):
        row = PopulationOutcome.model_validate(original)
        if (member.index, member.item_id, member.tier) != (row.index, row.item_id, row.tier) or (
            (row.outcome == "suite_excluded") != (not member.within_limits)
        ):
            raise PopulationError("population_alignment_invalid")


def population_summary(
    members: Sequence[PopulationMember],
    outcomes: Sequence[PopulationOutcome],
    *,
    source: dict[str, object],
) -> dict[str, object]:
    validate_outcomes(members, outcomes)
    tiers = sorted({row.tier for row in members})
    return {
        "counts": population_counts(outcomes).model_dump(mode="json"),
        "tiers": [
            {
                "tier": tier,
                "counts": population_counts(
                    [row for row in outcomes if row.tier == tier]
                ).model_dump(mode="json"),
            }
            for tier in tiers
        ],
        "selected_population_sha256": selected_population_digest(members),
        "metric_population_sha256": canonical_digest(
            {
                "source": source,
                "item_ids": sorted(row.item_id for row in outcomes if row.metric_eligible),
            }
        ),
        "source_selection": source,
    }


def target_identity(case: EvaluationSet, items: Sequence[EvaluationItem]) -> dict[str, object]:
    return {
        "gold_column": case.gold_column,
        "gold_index_map": case.gold_index_map,
        "label_space": case.label_space,
        "labels": list(case.labels),
        "target_population_sha256": canonical_digest(
            {
                "labels": case.labels,
                "label_space": case.label_space,
                "gold": sorted((item.item_id, item.gold) for item in items),
            }
        ),
    }


def _manifest(attempt: PopulationAttempt, value: dict[str, object]) -> None:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    if len(encoded) > MAX_REPORT_BYTES:
        raise PopulationError("report_limit_exceeded")
    _write_private(attempt.pending / attempt.report_name, encoded + b"\n")


def _audit_fields() -> list[_ArrowField]:
    import pyarrow as pa

    return cast(
        "list[_ArrowField]",
        [
            pa.field("index", pa.int64(), nullable=False),
            pa.field("item_id", pa.string(), nullable=False),
            pa.field("tier", pa.string(), nullable=False),
            pa.field("outcome", pa.string(), nullable=False),
            pa.field("metric_eligible", pa.bool_(), nullable=False),
            pa.field("token_count", pa.int64()),
            pa.field("model_limit", pa.int64()),
            pa.field("model_sent", pa.bool_()),
            pa.field("error_code", pa.string()),
            pa.field("refusal_stage", pa.string()),
        ],
    )


def write_scoring_attempt(
    attempt: PopulationAttempt,
    manifest: dict[str, object],
    outcomes: Sequence[PopulationOutcome],
    scores: dict[str, ScoredRow],
    labels: tuple[str, ...],
) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    fields = [
        *_audit_fields(),
        pa.field("pred", pa.string()),
        pa.field("mass_outside", pa.float64()),
        pa.field("truncated", pa.bool_(), nullable=False),
    ]
    fields.extend(pa.field(f"p_{label}", pa.float64()) for label in labels)
    rows: list[dict[str, object]] = []
    for outcome in outcomes:
        score = scores.get(outcome.item_id)
        row = outcome.model_dump(mode="json")
        row.update(
            pred=score.pred if score else None,
            mass_outside=score.mass_outside if score else None,
            truncated=False,
        )
        row.update(
            {f"p_{label}": score.probabilities.get(label) if score else None for label in labels}
        )
        rows.append(row)
    with _private_output(attempt.pending / "results.parquet") as file:
        pq.write_table(  # pyright: ignore[reportUnknownMemberType] — Arrow writer lacks complete stubs
            pa.Table.from_pylist(rows, schema=pa.schema(fields)), file
        )
    _write_private(
        attempt.pending / "README.md",
        b"---\nconfigs:\n- config_name: default\n  data_files:\n  - split: train\n    path: results.parquet\n---\n\n# Native population scoring\n\nPrivate item audit and explicit population counts are in results.parquet and scoring.json.\n",
    )
    _manifest(attempt, manifest)


def write_evaluation_attempt(
    attempt: PopulationAttempt,
    manifest: dict[str, object],
    sets: Sequence[tuple[EvaluationSet, tuple[EvaluationItem, ...], tuple[PopulationOutcome, ...]]],
) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    evaluations = attempt.pending / "evaluations"
    _new_private_directory(evaluations)
    for case, items, outcomes in sets:
        directory = evaluations / case.evaluation_id
        _new_private_directory(directory)
        metric_path = directory / ".metric-items.parquet"
        # The legacy schema writer truncates an existing file without changing its 0600 mode.
        _write_private(metric_path, b"")
        _write_items(metric_path, case, items)
        table = cast("_ArrowTable", pq.read_table(metric_path))  # pyright: ignore[reportUnknownMemberType] — bounded schema/read protocol narrows Arrow stubs
        by_id = {cast("str", row["item_id"]): row for row in table.to_pylist()}
        metadata = _audit_fields()
        audit_names = {field.name for field in metadata}
        fields = [
            *metadata,
            *(field.with_nullable(True) for field in table.schema if field.name not in audit_names),
        ]
        rows: list[dict[str, object]] = []
        for outcome in outcomes:
            row = {**by_id.get(outcome.item_id, {}), **outcome.model_dump(mode="json")}
            row.update(
                overflow=outcome.outcome == "overflow" or row.get("overflow", False),
                truncated=row.get("truncated", False),
            )
            rows.append(row)
        with _private_output(directory / "items.parquet") as file:
            pq.write_table(  # pyright: ignore[reportUnknownMemberType] — Arrow writer lacks complete stubs
                pa.Table.from_pylist(rows, schema=pa.schema(fields)), file
            )
        metric_path.unlink()
    card = (
        "---\nconfigs:\n"
        + "".join(
            f"- config_name: {case.evaluation_id}\n  data_files:\n  - split: train\n    path: evaluations/{case.evaluation_id}/items.parquet\n"
            for case, _, _ in sets
        )
        + "---\n\n"
    )
    _write_private(
        attempt.pending / "README.md",
        (
            card
            + "# Population evaluation\n\nPrivate item audits retain exclusions separately; metrics use the explicitly recorded population.\n"
        ).encode("utf-8"),
    )
    _manifest(attempt, manifest)


def finish_attempt(attempt: PopulationAttempt) -> None:
    """A bounded synchronous rename tail has no await at which cancellation can publish later."""
    # Large serialization and Parquet writes occur only in the pending directory off-loop.
    # The final manifest moves last; canceled write threads never call this commit tail.
    for item in attempt.pending.iterdir():
        if item.name != attempt.report_name:
            item.replace(attempt.directory / item.name)
    (attempt.pending / attempt.report_name).replace(attempt.directory / attempt.report_name)
    attempt.pending.rmdir()


def completed_counts(outcomes: Sequence[PopulationOutcome]) -> PopulationCounts:
    return population_counts(outcomes)
