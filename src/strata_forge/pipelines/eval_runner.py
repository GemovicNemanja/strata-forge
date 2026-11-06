"""Pinned classification evaluation with trusted access checks and separate task artifacts."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path  # noqa: TC003 — Pydantic resolves this result field at runtime
from typing import TYPE_CHECKING, Literal, cast

from pydantic import BaseModel, ConfigDict

from strata_forge.metrics import holm_correction, mcnemar_exact
from strata_forge.pipelines._classifier_checkpoint import validate_single_sequence_rendering
from strata_forge.pipelines._common import (
    RunSecrets,
    check_engine_version,
    emit,
    installed_engine_version,
    load_config,
    results_dir,
    runner_main,
)
from strata_forge.pipelines._evaluation_access import (
    EvaluationError,
    ReadDecision,
    ReadPolicy,
    authorize_reads,
)
from strata_forge.pipelines._evaluation_authority import (
    EvaluationPolicyRPC,
    PublicationFence,
    PublicationPolicy,
    parse_authority,
)
from strata_forge.pipelines._evaluation_items import (
    EvaluationItem,
    gold_items,
    parse_predictions,
    stable_id,
)
from strata_forge.pipelines._evaluation_reader import EvaluationReader, HubEvaluationReader, Row
from strata_forge.pipelines._evaluation_report import (
    SCHEMA_VERSION,
    build_report,
    build_set_report,
    counts,
    not_produced,
    write_artifacts,
)
from strata_forge.pipelines._scoring import Scorer, ScoringInput, load_scorer
from strata_forge.pipelines.classifier_spec import ClassifierWorkerError
from strata_forge.pipelines.evaluation_spec import (
    EvalSpec,
    EvaluationSet,
    PredictionSet,
    ProbabilityColumns,
)
from strata_forge.storage import HFHubClient
from strata_forge.training._load_guard import ModelLoadingError
from strata_forge.training.progress import JsonlProgressWriter, ProgressEvent
from strata_forge.training.verifier_config import VerifierConfig, render_verifier_input

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

__all__ = [
    "EvalSpec",
    "EvaluationError",
    "EvaluationReader",
    "EvaluationResult",
    "PublicationFence",
    "PublicationPolicy",
    "ReadDecision",
    "ReadPolicy",
    "execute_evaluation",
    "main",
    "run_evaluation",
]


class EvaluationResult(BaseModel):
    """A complete local evaluation artifact and its explicit null publication reference."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    metrics_path: Path
    output_ref: str | None = None
    destination: str


async def _rendering(
    spec: EvalSpec, reader: EvaluationReader
) -> tuple[VerifierConfig, dict[str, object]]:
    declaration = spec.rendering
    if declaration is None or spec.model_labels is None:
        raise EvaluationError("rendering_missing")
    original_digest: str | None = None
    original: VerifierConfig | None = None
    forge_single_sequence = False
    if declaration.source == "override":
        config = declaration.explicit_config(spec.model_labels)
    else:
        value = await reader.json(
            f"{spec.model_id}@{spec.model_revision}", "strata_verifier.json", repo_type="model"
        )
        original = VerifierConfig.model_validate(value)
        original_digest = original.digest()
        overrides = (
            declaration.overrides.model_dump(exclude_unset=True)
            if declaration.overrides is not None
            else {}
        )
        config = VerifierConfig.model_validate({**original.model_dump(), **overrides})
        if original.labels != spec.model_labels:
            raise EvaluationError("model_label_order_mismatch")
    if config.labels != spec.model_labels:
        raise EvaluationError("rendering_engine_mismatch")
    if (config.pair_template is not None) != (spec.engine == "hf_classify"):
        if (
            spec.engine != "hf_classify"
            or declaration.source != "model_artifact"
            or original is None
        ):
            raise EvaluationError("rendering_engine_mismatch")
        ref = f"{spec.model_id}@{spec.model_revision}"
        metadata = await reader.json(ref, "strata_classifier.json", repo_type="model")
        native = await reader.json(ref, "config.json", repo_type="model")
        refused = False
        try:
            validate_single_sequence_rendering(
                metadata, native, original=original, effective=config
            )
        except ClassifierWorkerError:
            refused = True
        if refused:
            raise EvaluationError("rendering_engine_mismatch")
        forge_single_sequence = True
    if (
        not forge_single_sequence
        and config.label_tokens is not None
        and config.label_tokens != spec.label_tokens
    ):
        raise EvaluationError("verbalizer_labels_mismatch")
    return config, {
        "source": declaration.source,
        "original_sha256": original_digest,
        "effective_sha256": config.digest(),
        "max_length": config.max_length,
        "max_premises": config.max_premises,
        "premise_format": config.premise_format,
        **(
            {
                "classifier_input_kind": "forge_single_sequence",
                "probability_mode": "learned_head_full_class",
                "label_tokens_role": "initialization_provenance",
            }
            if forge_single_sequence
            else {}
        ),
    }


def _gold_columns(case: EvaluationSet) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            (
                case.id_column,
                case.gold_column,
                *case.slice_columns,
                *([case.cluster_column] if case.cluster_column else []),
                *(case.column_mapping or {}).values(),
            )
        )
    )


def _prediction_columns(case: PredictionSet, producer: str | None) -> tuple[str, ...]:
    config = case.parse_config
    columns = [case.prediction_id_column]
    if isinstance(config, ProbabilityColumns):
        columns.extend(config.probability_columns.values())
        columns.extend(
            [
                value
                for value in (config.prediction_column, config.mass_outside_column)
                if value is not None
            ]
        )
    else:
        columns.append(config.column)
    if producer is not None:
        columns.extend(
            ("error" if producer == "scoring" else "prediction_status", "overflow", "truncated")
        )
    return tuple(dict.fromkeys(columns))


async def _prediction_manifest(
    spec: EvalSpec, case: PredictionSet, reader: EvaluationReader
) -> str | None:
    if case.prediction_manifest_path is None:
        return None
    manifest = await reader.json(case.predictions_ref, case.prediction_manifest_path)
    if case.prediction_manifest_path == "metrics.json":
        entry = _evaluation_entry(manifest, case.predictions_config)
        source_spec = manifest.get("effective_spec")
        if not isinstance(source_spec, dict):
            raise EvaluationError("prediction_manifest_mismatch")
        sources = cast("Row", source_spec).get("evaluation_sets")
        if not isinstance(sources, list):
            raise EvaluationError("prediction_manifest_mismatch")
        matching = [
            cast("Row", item)
            for item in cast("list[object]", sources)
            if isinstance(item, dict)
            and cast("Row", item).get("evaluation_id") == case.predictions_config
        ]
        probability = entry.get("probability_context")
        if (
            len(matching) != 1
            or matching[0].get("id_column") != case.id_column
            or case.prediction_id_column != "item_id"
            or not _same_population(spec, case, entry, manifest)
            or entry.get("labels") != list(case.source_labels)
        ):
            raise EvaluationError("prediction_manifest_mismatch")
        if isinstance(case.parse_config, ProbabilityColumns) and (
            not isinstance(probability, dict)
            or cast("Row", probability).get("kind") != case.parse_config.probability_kind
        ):
            raise EvaluationError("prediction_manifest_mismatch")
        return "evaluation"
    ids = manifest.get("id_semantics")
    refs = manifest.get("input_refs")
    if not isinstance(ids, dict) or ids != {
        "kind": "stable_string",
        "source_column": case.id_column,
        "artifact_column": case.prediction_id_column,
        "unique": True,
    }:
        raise EvaluationError("prediction_manifest_mismatch")
    if (
        not isinstance(refs, dict)
        or cast("Row", refs).get("dataset") != f"{spec.dataset_id}@{spec.dataset_commit_sha}"
    ):
        raise EvaluationError("prediction_manifest_mismatch")
    if (
        manifest.get("dataset_config") != case.dataset_config
        or manifest.get("split") != case.split
        or tuple(cast("list[str]", manifest.get("labels", []))) != case.source_labels
    ):
        raise EvaluationError("prediction_manifest_mismatch")
    if (
        case.selection.row_start != 0
        or case.selection.row_end is not None
        or manifest.get("row_limit") != case.selection.row_limit
    ):
        raise EvaluationError("prediction_manifest_mismatch")
    expected = (
        "restricted_softmax"
        if manifest.get("engine") == "hf_label_logits"
        else "class_probabilities"
        if manifest.get("engine") == "hf_classify"
        else None
    )
    if (
        not isinstance(case.parse_config, ProbabilityColumns)
        or case.parse_config.probability_kind != expected
    ):
        raise EvaluationError("prediction_manifest_mismatch")
    return "scoring"


def _evaluation_entry(manifest: dict[str, object], evaluation_id: str) -> Row:
    entries = manifest.get("evaluations")
    if manifest.get("schema_version") != SCHEMA_VERSION or not isinstance(entries, list):
        raise EvaluationError("comparison_manifest_invalid")
    matches = [
        cast("Row", entry)
        for entry in cast("list[object]", entries)
        if isinstance(entry, dict) and cast("Row", entry).get("evaluation_id") == evaluation_id
    ]
    if len(matches) != 1:
        raise EvaluationError("comparison_manifest_invalid")
    return matches[0]


def _same_population(spec: EvalSpec, case: EvaluationSet, entry: Row, manifest: Row) -> bool:
    refs = manifest.get("input_refs")
    if not isinstance(refs, list):
        return False
    gold_ref = f"{spec.dataset_id}@{spec.dataset_commit_sha}"
    has_gold = any(
        isinstance(value, dict)
        and cast("Row", value).get("role") == "gold"
        and cast("Row", value).get("ref") == gold_ref
        for value in cast("list[object]", refs)
    )
    return (
        has_gold
        and entry.get("dataset_config") == case.dataset_config
        and entry.get("split") == case.split
        and entry.get("selection") == case.selection.model_dump(mode="json")
    )


async def _classify(
    spec: EvalSpec,
    case: EvaluationSet,
    rows: tuple[Row, ...],
    gold: tuple[EvaluationItem, ...],
    config: VerifierConfig,
    scorer: Scorer,
    writer: JsonlProgressWriter | None,
    completed: int,
    total: int,
) -> tuple[EvaluationItem, ...]:
    mapping = case.column_mapping or {}
    items = list(gold)
    for start in range(0, len(rows), 128):
        await asyncio.sleep(0)
        inputs: list[ScoringInput] = []
        indices: list[int] = []
        for index in range(start, min(start + 128, len(rows))):
            row = rows[index]
            rendered = render_verifier_input(
                config, row.get(mapping["premises"]), row.get(mapping["hypothesis"])
            )
            if rendered.error_code is not None or rendered.text is None:
                items[index] = items[index].model_copy(
                    update={
                        "prediction_status": "refused",
                        "error_code": rendered.error_code,
                        "probability_exclusion_reason": "refused",
                    }
                )
                continue
            indices.append(index)
            inputs.append(ScoringInput(text=rendered.text, text_pair=rendered.text_pair))
        if inputs:
            scored = await scorer.score(inputs, batch_size=spec.scoring_batch_size or 1)
            if len(scored.rows) != len(indices) or sorted(r.index for r in scored.rows) != list(
                range(len(indices))
            ):
                raise EvaluationError("scoring_row_alignment_invalid")
            # Reuse the same strict probability/length parser as imported forecasts.
            parser_case = PredictionSet.model_validate(
                {
                    **case.model_dump(),
                    "column_mapping": None,
                    "source_labels": spec.model_labels,
                    "predictions_ref": f"{spec.model_id}@{spec.model_revision}",
                    "predictions_config": case.dataset_config,
                    "predictions_split": case.split,
                    "prediction_id_column": "item_id",
                    "prediction_manifest_path": None,
                    "parse": "probability_columns",
                    "parse_config": {
                        "probability_columns": {label: f"p_{label}" for label in scorer.labels},
                        "probability_kind": "class_probabilities"
                        if spec.engine == "hf_classify"
                        else "restricted_softmax",
                        "prediction_rule": "argmax",
                        "prediction_column": None,
                        "label_aliases": {},
                        "index_map": {},
                        "mass_outside_column": "mass_outside"
                        if spec.engine == "hf_label_logits"
                        else None,
                    },
                }
            )
            output: list[Row] = []
            selected = tuple(gold[index] for index in indices)
            for scored_row in scored.rows:
                output.append(
                    {
                        **{
                            f"p_{label}": scored_row.probabilities.get(label)
                            for label in scorer.labels
                        },
                        "item_id": gold[indices[scored_row.index]].item_id,
                        "overflow": scored_row.overflow,
                        "truncated": scored_row.truncated,
                        "error": scored_row.error,
                        "mass_outside": scored_row.mass_outside,
                    }
                )
            parsed = parse_predictions(selected, output, parser_case, strata_scoring=True)
            for index, item in zip(indices, parsed, strict=True):
                items[index] = item.model_copy(
                    update={"source_prediction_ref": None, "source_prediction_id": None}
                )
        emit(
            writer,
            ProgressEvent(
                kind="eval",
                stage="run",
                step=completed + min(start + 128, len(rows)),
                total_steps=total,
                metrics={
                    key: float(value)
                    for key, value in counts(items[: min(start + 128, len(rows))]).items()
                },
            ),
        )
    return tuple(items)


def _independent(case: EvaluationSet, items: tuple[EvaluationItem, ...]) -> bool:
    if case.sampling_unit != "independent_items":
        return False
    groups = [item.cluster_id for item in items]
    return case.cluster_column is None or (None not in groups and len(set(groups)) == len(groups))


async def _comparisons(
    spec: EvalSpec,
    reports: list[dict[str, object]],
    populations: dict[str, tuple[EvaluationItem, ...]],
    reader: EvaluationReader,
) -> dict[str, float]:
    p_values: dict[str, float] = {}
    for case, report in zip(spec.evaluation_sets, reports, strict=True):
        comparisons: list[dict[str, object]] = []
        candidate = populations[case.evaluation_id]
        for request in case.compare_to:
            baseline = await reader.json(request.baseline_ref, "metrics.json")
            entry = _evaluation_entry(baseline, request.baseline_evaluation_id)
            if (
                not _same_population(spec, case, entry, baseline)
                or entry.get("labels") != list(case.labels)
                or entry.get("projection") != case.projection
            ):
                raise EvaluationError("comparison_population_mismatch")
            rows = await reader.rows(
                request.baseline_ref,
                request.baseline_evaluation_id,
                "train",
                None,
                ("item_id", "gold", "pred", "prediction_status"),
            )
            index: dict[str, Row] = {}
            for row in rows:
                identifier = stable_id(row.get("item_id"), "comparison_id_invalid")
                if identifier in index:
                    raise EvaluationError("comparison_id_duplicate")
                index[identifier] = row
            if set(index) != {item.item_id for item in candidate} or any(
                index[item.item_id].get("gold") != item.gold for item in candidate
            ):
                raise EvaluationError("comparison_population_mismatch")
            b = c = 0
            for item in candidate:
                row = index[item.item_id]
                if row.get("prediction_status") not in (
                    "valid",
                    "missing",
                    "invalid",
                    "refused",
                ) or (
                    row.get("prediction_status") == "valid" and row.get("pred") not in case.labels
                ):
                    raise EvaluationError("comparison_item_invalid")
                baseline_correct = (
                    row.get("prediction_status") == "valid" and row.get("pred") == item.gold
                )
                b += baseline_correct and not item.correct
                c += item.correct and not baseline_correct
            inference: dict[str, object] = not_produced("requires_cluster_method")
            if request.sampling_unit == "independent_items" and _independent(case, candidate):
                statistic = mcnemar_exact(b, c, n_pairs=len(candidate))
                inference = {"status": "computed", "result": statistic.model_dump(mode="json")}
                if statistic.p_value is not None:
                    p_values[f"{case.evaluation_id}/{request.comparison_id}"] = statistic.p_value
            comparisons.append(
                {
                    "comparison_id": request.comparison_id,
                    "baseline_ref": request.baseline_ref,
                    "baseline_evaluation_id": request.baseline_evaluation_id,
                    "sampling_unit": request.sampling_unit,
                    "n_pairs": len(candidate),
                    "baseline_only_correct": b,
                    "candidate_only_correct": c,
                    "mcnemar": inference,
                }
            )
        report["comparisons"] = comparisons
    return p_values


async def run_evaluation(
    spec: EvalSpec,
    *,
    token: str | Literal[False] = False,
    policy: ReadPolicy | None = None,
    publication_policy: PublicationPolicy | None = None,
    reader: EvaluationReader | None = None,
    writer: JsonlProgressWriter | None = None,
    output_dir: Path | None = None,
    scorer_loader: Callable[..., Awaitable[Scorer]] = load_scorer,
) -> EvaluationResult:
    """Execute one declared job with a trusted policy; absent/unknown custody fails closed."""
    spec = EvalSpec.model_validate(spec.wire_payload())
    check_engine_version(spec.engine_version)
    if spec.output_repo_id is not None and publication_policy is None:
        raise EvaluationError("publication_policy_unavailable")
    if spec.output_repo_id is not None and token is False:
        raise EvaluationError("output_credential_required")
    decisions = await authorize_reads(spec.input_reads(), policy, spec_sha256=spec.digest())
    try:
        if spec.output_repo_id is not None and publication_policy is not None:
            await publication_policy.plan(
                requested_repo_id=spec.output_repo_id, spec_sha256=spec.digest()
            )
        if token is not False and (type(token) is not str or not token.strip()):
            raise EvaluationError("credential_invalid")
        directory = output_dir or results_dir(spec.run_id, name="strata-evaluation-results")
        await asyncio.to_thread(_reserve_output, directory, spec.run_id)
        source = reader or HubEvaluationReader(token)
        for ref, repo_type in spec.input_reads():
            await source.prepare(ref, model=repo_type == "model")
        scorer: Scorer | None = None
        rendering: VerifierConfig | None = None
        renderer_audit: dict[str, object] = {}
        if spec.mode == "classify":
            rendering, renderer_audit = await _rendering(spec, source)
            try:
                scorer = await scorer_loader(
                    spec.model_id,
                    engine=spec.engine,
                    token=token,
                    revision=spec.model_revision,
                    max_length=rendering.max_length,
                    label_alias=spec.label_aliases if spec.engine == "hf_classify" else None,
                    label_tokens={
                        label: spec.label_tokens[label] for label in spec.model_labels or ()
                    }
                    if spec.label_tokens is not None
                    else None,
                    dtype=spec.dtype,
                    device="auto",
                )
            except ModelLoadingError as exc:
                await asyncio.to_thread(
                    _write_evidence,
                    directory,
                    "loading-refusal.json",
                    {
                        "status": "loading_refused",
                        "model_ref": f"{spec.model_id}@{spec.model_revision}",
                        "loading_info": exc.audit.model_dump(mode="json"),
                    },
                )
                raise EvaluationError("model_loading_refused") from None
            if scorer.labels != spec.model_labels:
                raise EvaluationError("model_label_order_mismatch")
        raw: dict[str, tuple[Row, ...]] = {}
        populations: dict[str, tuple[EvaluationItem, ...]] = {}
        for case in spec.evaluation_sets:
            rows = await source.rows(
                f"{spec.dataset_id}@{spec.dataset_commit_sha}",
                case.dataset_config,
                case.split,
                case.selection,
                _gold_columns(case),
            )
            raw[case.evaluation_id] = rows
            populations[case.evaluation_id] = gold_items(rows, case)
        total = sum(len(rows) for rows in raw.values())
        emit(
            writer,
            ProgressEvent(
                kind="start",
                stage="run",
                total_steps=total,
                input_refs={
                    "dataset": f"{spec.dataset_id}@{spec.dataset_commit_sha}",
                    **(
                        {"model": f"{spec.model_id}@{spec.model_revision}"}
                        if spec.mode == "classify"
                        else {}
                    ),
                },
            ),
        )
        reports: list[dict[str, object]] = []
        completed = 0
        for case in spec.evaluation_sets:
            await asyncio.sleep(0)
            gold = populations[case.evaluation_id]
            if isinstance(case, PredictionSet):
                producer = await _prediction_manifest(spec, case, source)
                predictions = await source.rows(
                    case.predictions_ref,
                    case.predictions_config,
                    case.predictions_split,
                    None,
                    _prediction_columns(case, producer),
                )
                if producer == "evaluation":
                    for row in predictions:
                        if row.get("prediction_status") not in (
                            "valid",
                            "invalid",
                            "missing",
                            "refused",
                        ):
                            raise EvaluationError("prediction_manifest_mismatch")
                    predictions = tuple(
                        {
                            **row,
                            "error": "source_prediction_refused"
                            if row.get("prediction_status") == "refused"
                            else None,
                        }
                        for row in predictions
                    )
                items = parse_predictions(
                    gold, predictions, case, strata_scoring=producer is not None
                )
                if producer == "evaluation":
                    prior_status = {
                        cast("str", row[case.prediction_id_column]): row["prediction_status"]
                        for row in predictions
                    }
                    items = tuple(
                        item.model_copy(
                            update={
                                "prediction_status": "missing",
                                "error_code": "prediction_missing",
                                "probability_exclusion_reason": "missing_prediction",
                            }
                        )
                        if prior_status.get(item.item_id) == "missing" and item.pred is None
                        else item
                        for item in items
                    )
                kind = (
                    case.parse_config.probability_kind
                    if isinstance(case.parse_config, ProbabilityColumns)
                    else None
                )
                context: dict[str, object] = {
                    "predictions_ref": case.predictions_ref,
                    "config": case.predictions_config,
                    "split": case.predictions_split,
                    "parse": case.parse,
                    "parse_config": case.parse_config.model_dump(mode="json"),
                    "manifest_status": f"strata_{producer}_verified"
                    if producer is not None
                    else "external_declared",
                }
            else:
                if scorer is None or rendering is None:
                    raise EvaluationError("classifier_fields_missing")
                items = await _classify(
                    spec,
                    case,
                    raw[case.evaluation_id],
                    gold,
                    rendering,
                    scorer,
                    writer,
                    completed,
                    total,
                )
                kind = (
                    "class_probabilities" if spec.engine == "hf_classify" else "restricted_softmax"
                )
                context = {
                    "model_ref": f"{spec.model_id}@{spec.model_revision}",
                    "engine": spec.engine,
                    "source_labels": spec.model_labels,
                    "rendering": renderer_audit,
                    "loading_info": scorer.loading_audit.model_dump(mode="json"),
                    "padding_policy": scorer.padding_policy.model_dump(mode="json"),
                }
            populations[case.evaluation_id] = items
            reports.append(build_set_report(case, items, kind=kind, source=context))
            completed += len(items)
            emit(
                writer,
                ProgressEvent(
                    kind="eval",
                    stage="run",
                    step=completed,
                    total_steps=total,
                    metrics={key: float(value) for key, value in counts(items).items()},
                ),
            )
        p_values = await _comparisons(spec, reports, populations, source)
        report = build_report(
            spec,
            reports,
            populations,
            tuple(d.policy_version for d in decisions),
            installed_engine_version(),
        )
        report["holm_families"] = [
            {
                "family_id": family.family_id,
                "members": family.members,
                "adjustment": {
                    "status": "computed",
                    "result": holm_correction(
                        {member: p_values[member] for member in family.members},
                        family_id=family.family_id,
                    ).model_dump(mode="json"),
                }
                if all(member in p_values for member in family.members)
                else not_produced("requires_cluster_method"),
            }
            for family in spec.holm_families
        ]
        await asyncio.sleep(0)
        path = await asyncio.to_thread(write_artifacts, directory, spec, report, populations)
        await asyncio.sleep(0)
        destination, output_ref = str(path), None
        if spec.output_repo_id is not None and publication_policy is not None:
            output_ref = await _publish(spec, directory, token, publication_policy)
            destination = output_ref.rsplit("@", 1)[0]
        totals = cast("dict[str, int]", report["counts"])
        emit(
            writer,
            ProgressEvent(
                kind="end",
                stage="push",
                step=total,
                total_steps=total,
                metrics={key: float(value) for key, value in totals.items()},
                message=destination,
                output_ref=output_ref,
            ),
        )
        return EvaluationResult(metrics_path=path, output_ref=output_ref, destination=destination)
    except EvaluationError, asyncio.CancelledError, ImportError:
        raise
    except Exception:
        raise EvaluationError("evaluation_execution_failed") from None


def _write_evidence(directory: Path, filename: str, value: dict[str, object]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / filename).write_text(
        json.dumps(value, allow_nan=False, sort_keys=True) + "\n", encoding="utf-8"
    )


def _reserve_output(directory: Path, run_id: str) -> None:
    if directory.is_symlink() or (directory.exists() and any(directory.iterdir())):
        raise EvaluationError("artifact_directory_not_empty")
    directory.mkdir(parents=True, exist_ok=True)
    try:
        with (directory / ".eval-attempt.json").open("x", encoding="utf-8") as file:
            json.dump({"run_id": run_id, "status": "started"}, file)
    except FileExistsError:
        raise EvaluationError("artifact_directory_not_empty") from None


async def _publish(
    spec: EvalSpec, directory: Path, token: str | Literal[False], policy: PublicationPolicy
) -> str:
    if spec.output_repo_id is None or token is False:
        raise EvaluationError("output_credential_required")
    digest = spec.digest()
    await asyncio.to_thread(
        _write_evidence,
        directory,
        "publication-attempt.json",
        {"status": "planning", "requested_id": spec.output_repo_id, "spec_sha256": digest},
    )
    async with policy.publication(
        requested_repo_id=spec.output_repo_id, spec_sha256=digest
    ) as granted:
        fence = PublicationFence.model_validate(granted.model_dump())
        if (
            fence.run_id != spec.run_id
            or fence.spec_sha256 != digest
            or fence.requested_id != spec.output_repo_id
            or fence.canonical_id.casefold()
            in {
                ref.rsplit("@", 1)[0].casefold()
                for ref, kind in spec.input_reads()
                if kind == "dataset"
            }
        ):
            raise EvaluationError("publication_identity_mismatch")
        hub = HFHubClient(token=token)
        if (
            not await hub.repo_private(fence.canonical_id, repo_type="dataset")
            or await hub.resolve_revision(fence.canonical_id, repo_type="dataset")
            != fence.expected_parent
        ):
            raise EvaluationError("publication_destination_changed")
        evidence: dict[str, object] = {
            "status": "ready",
            "requested_id": fence.requested_id,
            "canonical_id": fence.canonical_id,
            "expected_parent": fence.expected_parent,
            "spec_sha256": digest,
            "policy_version": fence.policy_version,
        }
        await asyncio.to_thread(_write_evidence, directory, "publication-attempt.json", evidence)
        await asyncio.sleep(0)
        upload = asyncio.create_task(
            hub.commit_folder(
                directory,
                fence.canonical_id,
                repo_type="dataset",
                allow_patterns=(
                    "README.md",
                    "metrics.json",
                    *(
                        f"evaluations/{case.evaluation_id}/items.parquet"
                        for case in spec.evaluation_sets
                    ),
                ),
                commit_message="classification evaluation results",
                extras={"parent_commit": fence.expected_parent},
            )
        )
        try:
            commit = await asyncio.shield(upload)
        except asyncio.CancelledError:
            actual = None
            try:
                actual = await asyncio.wait_for(asyncio.shield(upload), timeout=10)
            except Exception, asyncio.CancelledError:
                actual = None
            await asyncio.to_thread(
                _write_evidence,
                directory,
                "publication-attempt.json",
                {
                    **evidence,
                    "status": "uploaded_after_cancellation"
                    if actual is not None
                    else "commit_outcome_unknown",
                    "commit_sha": actual.sha if actual is not None else None,
                },
            )
            raise
        except Exception:
            await asyncio.to_thread(
                _write_evidence,
                directory,
                "publication-attempt.json",
                {**evidence, "status": "commit_outcome_unknown", "commit_sha": None},
            )
            raise EvaluationError("publication_commit_failed") from None
        await asyncio.to_thread(
            _write_evidence,
            directory,
            "publication-attempt.json",
            {**evidence, "status": "uploaded_pending_receipt", "commit_sha": commit.sha},
        )
        await policy.record_receipt(fence, commit)
        await asyncio.to_thread(
            _write_evidence,
            directory,
            "publication-attempt.json",
            {**evidence, "status": "receipt_accepted", "commit_sha": commit.sha},
        )
        return f"{fence.canonical_id}@{commit.sha}"


async def execute_evaluation(
    spec: EvalSpec, *, secrets: RunSecrets, writer: JsonlProgressWriter | None = None
) -> EvaluationResult:
    """Construct the fixed installed authority adapter from the private Task channel."""
    spec = EvalSpec.model_validate(spec.wire_payload())
    if secrets.eval_authority is None:
        return await run_evaluation(spec, token=secrets.hub_credential(), writer=writer)
    authority = parse_authority(
        secrets.eval_authority, run_id=spec.run_id, spec_sha256=spec.digest()
    )
    async with EvaluationPolicyRPC(authority, token=secrets.hub_credential()) as policy:
        return await run_evaluation(
            spec,
            token=secrets.hub_credential(),
            policy=policy,
            publication_policy=policy,
            writer=writer,
        )


async def main() -> int:
    """Execute the installed worker with ordinary secret/progress handling and no implicit grant."""
    return await runner_main(
        lambda writer, secrets: execute_evaluation(
            load_config(EvalSpec, writer=writer), secrets=secrets, writer=writer
        )
    )


if __name__ == "__main__":  # pragma: no cover
    sys.exit(asyncio.run(main()))
