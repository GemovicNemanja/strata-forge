"""Explicit-population reports and typed Parquet artifacts without raw model content."""

from __future__ import annotations

import json
from collections import Counter
from typing import TYPE_CHECKING

from strata_forge.core.repro import env_snapshot
from strata_forge.metrics import brier, classification_report, nll
from strata_forge.pipelines._evaluation_access import EvaluationError
from strata_forge.pipelines._evaluation_items import slice_members
from strata_forge.pipelines.evaluation_spec import PredictionSet

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from strata_forge.pipelines._evaluation_items import EvaluationItem
    from strata_forge.pipelines.evaluation_spec import EvalSpec, EvaluationSet

SCHEMA_VERSION = "strata.eval.metrics.v1"


def not_produced(reason: str) -> dict[str, object]:
    return {"status": "not_produced", "reason": reason, "result": None}


def counts(items: Sequence[EvaluationItem]) -> dict[str, int]:
    valid = sum(item.prediction_status == "valid" for item in items)
    return {
        "eligible": len(items),
        "valid_predictions": valid,
        "invalid_predictions": len(items) - valid,
        "missing_predictions": sum(item.prediction_status == "missing" for item in items),
        "overflow": sum(item.overflow for item in items),
        "truncated": sum(item.truncated for item in items),
    }


def _point(
    items: Sequence[EvaluationItem], labels: tuple[str, ...], kind: str | None
) -> dict[str, object]:
    gold = [item.gold for item in items]
    pred = [item.pred if item.prediction_status == "valid" else None for item in items]
    vectors = [item.probabilities for item in items]
    report = classification_report(gold, pred, labels=labels)
    if kind is None:
        raw_brier = raw_nll = not_produced("head_not_produced")
    else:
        if kind not in ("class_probabilities", "restricted_softmax"):
            raise EvaluationError("probability_family_invalid")
        raw_brier = {
            "status": "computed",
            "result": brier(gold, vectors, labels=labels, probability_kind=kind).model_dump(
                mode="json"
            ),
        }
        raw_nll = {
            "status": "computed",
            "result": nll(gold, vectors, labels=labels, probability_kind=kind).model_dump(
                mode="json"
            ),
        }
    return {
        "classification": report.model_dump(mode="json"),
        "metrics": {
            "brier_raw": raw_brier,
            "nll_raw": raw_nll,
            "brier_calibrated": not_produced("calibration_not_applied"),
            "nll_calibrated": not_produced("calibration_not_applied"),
            "ece": not_produced("method_unavailable"),
            "inconsistency": not_produced("head_not_produced"),
            "support": not_produced("head_not_produced"),
            "selective_prediction": not_produced("head_not_produced"),
        },
    }


def _diagnostic(n: int, declared: bool) -> dict[str, object]:
    reasons = (["declared"] if declared else []) + (["n_below_500"] if n < 500 else [])
    return {"value": bool(reasons), "reasons": reasons}


def build_set_report(
    case: EvaluationSet,
    items: tuple[EvaluationItem, ...],
    *,
    kind: str | None,
    source: dict[str, object],
) -> dict[str, object]:
    slices: list[dict[str, object]] = []
    for column in case.slice_columns:
        groups: dict[tuple[str, object], list[EvaluationItem]] = {}
        for item in items:
            value = item.slices[column]
            for member in slice_members(value):
                key = (type(member).__name__, member)
                if key not in groups:
                    if len(groups) >= 256:
                        raise EvaluationError("report_limit_exceeded")
                    groups[key] = []
                if not groups[key] or groups[key][-1].item_id != item.item_id:
                    groups[key].append(item)
        for (name, value), members in groups.items():
            slices.append(
                {
                    "slice_id": f"slice-{len(slices)}",
                    "column": column,
                    "value": {
                        "type": {
                            "NoneType": "null",
                            "bool": "boolean",
                            "int": "integer",
                            "str": "string",
                        }[name],
                        "value": value,
                    },
                    "n": len(members),
                    "diagnostic": _diagnostic(len(members), case.declared_diagnostic),
                    **_point(members, case.labels, kind),
                }
            )
    if len(slices) > 256:
        raise EvaluationError("report_limit_exceeded")
    cluster_ids = [item.cluster_id for item in items if item.cluster_id is not None]
    return {
        "evaluation_id": case.evaluation_id,
        "dataset_config": case.dataset_config,
        "split": case.split,
        "label_space": case.label_space,
        "labels": case.labels,
        "projection": case.projection,
        "selection": case.selection.model_dump(mode="json"),
        "sampling": {
            "unit": case.sampling_unit,
            "cluster_column": case.cluster_column,
            "n_clusters": len(set(cluster_ids)) if case.cluster_column is not None else None,
            "missing_cluster_ids": len(items) - len(cluster_ids)
            if case.cluster_column is not None
            else 0,
        },
        "diagnostic": _diagnostic(len(items), case.declared_diagnostic),
        "counts": counts(items),
        "source": source,
        "probability_context": {"kind": kind, "calibration_ref": None},
        **_point(items, case.labels, kind),
        "slices": slices,
        "comparisons": [],
        "inference": {"accuracy_interval": not_produced("not_requested")},
    }


def build_report(
    spec: EvalSpec,
    reports: list[dict[str, object]],
    populations: dict[str, tuple[EvaluationItem, ...]],
    policy_versions: tuple[str, ...],
    engine_version: str | None,
) -> dict[str, object]:
    totals = Counter[str]()
    artifacts: list[dict[str, object]] = []
    for case in spec.evaluation_sets:
        items = populations[case.evaluation_id]
        totals.update(counts(items))
        artifacts.append(
            {
                "evaluation_id": case.evaluation_id,
                "path": f"evaluations/{case.evaluation_id}/items.parquet",
                "rows": len(items),
                "probability_columns": {label: f"p_{label}" for label in case.labels},
                "calibrated_probability_columns": {
                    label: f"p_cal_{label}" for label in case.labels
                },
                "slice_columns": {
                    f"slice_{i}": column for i, column in enumerate(case.slice_columns)
                },
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": spec.run_id,
        "engine_version": engine_version,
        "metric_versions": {
            "point": "classification-point-v1",
            "comparison": "classification-comparison-v1",
        },
        "mode": spec.mode,
        "spec_sha256": spec.digest(),
        "environment": {key: value for key, value in env_snapshot().items() if key != "cwd"},
        "effective_spec": spec.wire_payload(),
        "input_refs": input_ref_records(spec),
        "custody_policy_versions": policy_versions,
        "counts": {"evaluation_sets": len(spec.evaluation_sets), **dict(totals)},
        "evaluations": reports,
        "artifacts": artifacts,
        "capabilities": {"calibration_applied": False, "bootstrap": False, "optional_heads": False},
    }


def input_ref_records(spec: EvalSpec) -> list[dict[str, str]]:
    records = [{"role": "gold", "ref": f"{spec.dataset_id}@{spec.dataset_commit_sha}"}]
    if spec.mode == "classify":
        records.append({"role": "model", "ref": f"{spec.model_id}@{spec.model_revision}"})
    for case in spec.evaluation_sets:
        if isinstance(case, PredictionSet):
            records.append({"role": "predictions", "ref": case.predictions_ref})
        records.extend(
            {"role": "baseline", "ref": comparison.baseline_ref} for comparison in case.compare_to
        )
    return [
        dict(record) for record in {tuple(record.items()): record for record in records}.values()
    ]


def _write_items(path: Path, case: EvaluationSet, items: tuple[EvaluationItem, ...]) -> None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError("install 'strata-forge[hf]' for evaluation Parquet artifacts") from exc
    fields = [
        pa.field(name, pa.string(), nullable=nullable)
        for name, nullable in (
            ("item_id", False),
            ("gold", False),
            ("pred", True),
            ("prediction_status", False),
            ("error_code", True),
        )
    ]
    fields.extend(
        pa.field(name, pa.bool_(), nullable=False) for name in ("correct", "overflow", "truncated")
    )
    fields.extend(
        pa.field(f"{prefix}{label}", pa.float64())
        for prefix in ("p_", "p_cal_")
        for label in case.labels
    )
    fields.extend(
        pa.field(name, pa.string())
        for name in (
            "probability_kind",
            "probability_exclusion_reason",
            "source_prediction_ref",
            "source_prediction_id",
            "cluster_id",
        )
    )
    fields.extend(
        (
            pa.field("mass_outside", pa.float64()),
            pa.field("p_inconsistent", pa.float64()),
            pa.field("support_scores", pa.list_(pa.float64())),
            pa.field("abstained", pa.bool_()),
        )
    )
    # Struct encoding preserves typed nulls, boolean-vs-integer identity and list memberships.
    slice_type = pa.list_(
        pa.struct(
            [
                pa.field("type", pa.string(), nullable=False),
                pa.field("string_value", pa.string()),
                pa.field("integer_value", pa.int64()),
                pa.field("boolean_value", pa.bool_()),
            ]
        )
    )
    fields.extend(
        pa.field(f"slice_{i}", slice_type, nullable=False) for i, _ in enumerate(case.slice_columns)
    )
    rows: list[dict[str, object]] = []
    for item in items:
        row: dict[str, object] = {
            "item_id": item.item_id,
            "gold": item.gold,
            "pred": item.pred,
            "prediction_status": item.prediction_status,
            "error_code": item.error_code,
            "correct": item.correct,
            "overflow": item.overflow,
            "truncated": item.truncated,
            "probability_kind": item.probability_kind,
            "probability_exclusion_reason": item.probability_exclusion_reason,
            "source_prediction_ref": item.source_prediction_ref,
            "source_prediction_id": item.source_prediction_id,
            "cluster_id": item.cluster_id,
            "mass_outside": item.mass_outside,
            "p_inconsistent": None,
            "support_scores": None,
            "abstained": None,
        }
        for index, label in enumerate(case.labels):
            row[f"p_{label}"] = (
                item.probabilities[index] if item.probabilities is not None else None
            )
            row[f"p_cal_{label}"] = None
        for index, column in enumerate(case.slice_columns):
            value = item.slices[column]
            members: list[dict[str, object]] = []
            seen: set[tuple[str, object]] = set()
            for member in slice_members(value):
                kind = type(member).__name__
                if (kind, member) in seen:
                    continue
                seen.add((kind, member))
                members.append(
                    {
                        "type": {
                            "NoneType": "null",
                            "str": "string",
                            "int": "integer",
                            "bool": "boolean",
                        }[kind],
                        "string_value": member if kind == "str" else None,
                        "integer_value": member if kind == "int" else None,
                        "boolean_value": member if kind == "bool" else None,
                    }
                )
            row[f"slice_{index}"] = members
        rows.append(row)
    table = pa.Table.from_pylist(rows, schema=pa.schema(fields))
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)  # pyright: ignore[reportUnknownMemberType] — PyArrow's writer lacks complete stubs


def write_artifacts(
    directory: Path,
    spec: EvalSpec,
    report: dict[str, object],
    populations: dict[str, tuple[EvaluationItem, ...]],
) -> Path:
    """Write a bounded complete local result inventory without publishing it."""
    encoded = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)
    if len(encoded.encode("utf-8")) > 32 * 1024 * 1024:
        raise EvaluationError("report_limit_exceeded")
    directory.mkdir(parents=True, exist_ok=True)
    for case in spec.evaluation_sets:
        _write_items(
            directory / "evaluations" / case.evaluation_id / "items.parquet",
            case,
            populations[case.evaluation_id],
        )
    path = directory / "metrics.json"
    path.write_text(encoded + "\n", encoding="utf-8")
    card = "---\nconfigs:\n" + "".join(
        f"- config_name: {case.evaluation_id}\n  data_files:\n  - split: train\n    path: evaluations/{case.evaluation_id}/items.parquet\n"
        for case in spec.evaluation_sets
    )
    (directory / "README.md").write_text(
        card
        + "---\n\n# Evaluation results\n\nSeparate task populations and exact input refs are recorded in `metrics.json`.\n",
        encoding="utf-8",
    )
    return path
