"""VM-side evaluation runner: pinned prediction files + a pinned gold split -> a metrics report.

Launched on the user's own VM by the control plane (``python -m
strata_forge.pipelines.eval_runner``). It reads an INERT spec from ``STRATA_RUN_CONFIG``, loads
the gold split at its pinned commit and one or more prediction files (parquet, for example a
scoring run's ``results.parquet``) each at its pinned commit, joins them on a stable item id,
and computes accuracy, macro-F1 and invalid rate overall and per declared slice, with seeded
cluster percentile bootstrap intervals (``strata_forge.metrics.cluster_bootstrap``) and paired
candidate-minus-baseline deltas. It then writes ``items.parquet`` (one row per gold item),
``metrics.json`` (the report), ``manifest.json`` (provenance) and a dataset card, and either
pushes them to a private Hub dataset repo in ONE commit or keeps them on the VM.

The rules the report follows:

- Every read is pinned. The gold split and every prediction file name a full commit id; a branch
  or tag is refused because what it names can change, so a run cannot read data it was not
  explicitly pointed at.
- A gold item with no prediction, a null prediction, a value outside the label space, or a row
  its source flags as overflowed counts as an INVALID prediction: wrong, and in the gold class's
  support. A prediction id with no gold item is counted and ignored. Duplicate ids refuse.
- Rows the gold split itself flags in ``exclude_column`` are excluded from every system and
  counted, so paired systems always share one population.
- Slices are the values of declared columns (a list value puts the row in each listed slice).
  A slice with fewer than ``min_slice_rows`` rows is reported and marked diagnostic.
- Whole clusters (``cluster_column``, else each item) are resampled within each slice.

The report carries no item text: ids, gold labels, mapped predictions and slice values only.
"""

from __future__ import annotations

import asyncio
import json
import math
import re
import sys
from collections import Counter
from typing import TYPE_CHECKING, Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from strata_forge.metrics.classification import classification_report
from strata_forge.metrics.cluster_bootstrap import (
    CLUSTER_BOOTSTRAP_RESAMPLES,
    MAX_CLUSTER_BOOTSTRAP_RESAMPLES,
    MAX_CLUSTER_BOOTSTRAP_SEED,
    SystemIntervals,
    cluster_bootstrap,
    paired_cluster_bootstrap,
)
from strata_forge.pipelines._common import (
    COMMIT_SHA_RE,
    MANIFEST_FILENAME,
    MANIFEST_SCHEMA,
    SAFE_NAME_RE,
    RunError,
    RunSecrets,
    emit,
    ensure_private_output,
    load_config,
    manifest_engine,
    phase_sink,
    results_dir,
    runner_main,
    ticking_phase,
    validate_repo_id,
    write_manifest,
)
from strata_forge.storage import HFHubClient
from strata_forge.training.progress import JsonlProgressWriter, ProgressEvent

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

__all__ = [
    "ITEMS_FILENAME",
    "REPORT_FILENAME",
    "REPORT_SCHEMA",
    "EvalSettings",
    "EvalSpec",
    "PredictionSource",
    "main",
]

REPORT_SCHEMA = "strata.eval.report/v1"
ITEMS_FILENAME = "items.parquet"
REPORT_FILENAME = "metrics.json"
_CARD_FILENAME = "README.md"
_MANIFEST_KIND = "evaluation"
_RESULTS_DIR_NAME = "strata-eval-results"
_RECORDED_DISTRIBUTIONS = ("numpy", "pyarrow", "datasets", "huggingface-hub")
_MAX_LABELS = 32
_MAX_BASELINES = 8
_MAX_SLICE_COLUMNS = 8
_MAX_SLICE_VALUES = 64
_MAX_ALIASES = 256
_MAX_ROWS = 2_000_000
_MAX_TEXT = 128
# A parquet path inside a dataset repo: relative segments of safe characters, ending .parquet.
_PATH_RE = re.compile(
    r"\A[A-Za-z0-9_-][A-Za-z0-9._-]*(?:/[A-Za-z0-9_-][A-Za-z0-9._-]*)*\.parquet\Z"
)

type _Population = list[int]


def _bounded_text(value: str, what: str) -> str:
    if not value or len(value) > _MAX_TEXT or not value.isprintable():
        msg = f"{what} must be 1-{_MAX_TEXT} printable characters"
        raise ValueError(msg)
    return value


def _bounded_aliases(value: dict[str, str]) -> dict[str, str]:
    if len(value) > _MAX_ALIASES:
        msg = f"at most {_MAX_ALIASES} label aliases"
        raise ValueError(msg)
    for key, label in value.items():
        _bounded_text(key, "an alias")
        _bounded_text(label, "an alias target")
    return value


class PredictionSource(BaseModel):
    """One system's predictions: a parquet file in a dataset repo at a pinned commit."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1, max_length=32)
    repo_id: str = Field(min_length=1, max_length=255)
    commit_sha: str
    path: str = Field(default="results.parquet", max_length=255)
    id_column: str = "item_id"
    prediction_column: str = "pred"
    # A boolean column marking rows the system could not score (for example an input over the
    # model's context): such a row counts as an invalid prediction and is counted separately.
    overflow_column: str | None = None
    # Prediction value -> declared label, for systems whose label spelling differs.
    label_aliases: dict[str, str] = Field(default_factory=dict[str, str])

    @field_validator("name")
    @classmethod
    def _safe_name(cls, value: str) -> str:
        if not SAFE_NAME_RE.fullmatch(value):
            msg = "a system name is letters, digits, '_' and '-'"
            raise ValueError(msg)
        return value

    @field_validator("commit_sha")
    @classmethod
    def _pinned(cls, value: str) -> str:
        if not COMMIT_SHA_RE.fullmatch(value):
            msg = "commit_sha must be a full 40-character lowercase commit id"
            raise ValueError(msg)
        return value

    @field_validator("path")
    @classmethod
    def _parquet_path(cls, value: str) -> str:
        if ".." in value or not _PATH_RE.fullmatch(value):
            msg = "path must be a relative .parquet path inside the repo"
            raise ValueError(msg)
        return value

    @field_validator("id_column", "prediction_column", "overflow_column")
    @classmethod
    def _column(cls, value: str | None) -> str | None:
        return None if value is None else _bounded_text(value, "a column name")

    @field_validator("label_aliases")
    @classmethod
    def _aliases(cls, value: dict[str, str]) -> dict[str, str]:
        return _bounded_aliases(value)


class EvalSettings(BaseModel):
    """What to evaluate and how; validated identically by the control plane and the runner."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    labels: tuple[str, ...] = Field(min_length=2, max_length=_MAX_LABELS)
    predictions: PredictionSource
    compare_to: tuple[PredictionSource, ...] = Field(default=(), max_length=_MAX_BASELINES)
    id_column: str = "item_id"
    gold_column: str
    gold_aliases: dict[str, str] = Field(default_factory=dict[str, str])
    cluster_column: str | None = None
    slice_columns: tuple[str, ...] = Field(default=(), max_length=_MAX_SLICE_COLUMNS)
    exclude_column: str | None = None
    seed: int = Field(ge=0, le=MAX_CLUSTER_BOOTSTRAP_SEED)
    resamples: int = Field(
        default=CLUSTER_BOOTSTRAP_RESAMPLES, ge=1, le=MAX_CLUSTER_BOOTSTRAP_RESAMPLES
    )
    min_slice_rows: int = Field(default=500, ge=1)

    @field_validator("labels")
    @classmethod
    def _labels(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(set(value)) != len(value):
            msg = "labels must be unique"
            raise ValueError(msg)
        return tuple(_bounded_text(label, "a label") for label in value)

    @field_validator("id_column", "gold_column", "cluster_column", "exclude_column")
    @classmethod
    def _column(cls, value: str | None) -> str | None:
        return None if value is None else _bounded_text(value, "a column name")

    @field_validator("slice_columns")
    @classmethod
    def _slices(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(set(value)) != len(value):
            msg = "slice columns must be unique"
            raise ValueError(msg)
        return tuple(_bounded_text(column, "a column name") for column in value)

    @field_validator("gold_aliases")
    @classmethod
    def _aliases(cls, value: dict[str, str]) -> dict[str, str]:
        return _bounded_aliases(value)

    @model_validator(mode="after")
    def _consistent(self) -> EvalSettings:
        names = [source.name for source in self.systems]
        if len(set(names)) != len(names):
            msg = "every system needs a distinct name"
            raise ValueError(msg)
        targets = [*self.gold_aliases.values()]
        for source in self.systems:
            targets.extend(source.label_aliases.values())
        unknown = sorted({target for target in targets if target not in self.labels})
        if unknown:
            msg = f"label aliases map to undeclared labels: {unknown}"
            raise ValueError(msg)
        return self

    @property
    def systems(self) -> tuple[PredictionSource, ...]:
        """The candidate first, then every baseline it is compared to."""
        return (self.predictions, *self.compare_to)


class EvalSpec(BaseModel):
    """The inert evaluation spec the server delivers in ``STRATA_RUN_CONFIG`` (no secrets)."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    engine_version: str | None = None
    # The model the candidate predictions came from, recorded in the report; never read.
    model_id: str | None = None
    dataset_id: str
    dataset_commit_sha: str
    dataset_config: str | None = None
    split: str
    evaluation: EvalSettings
    output_repo_id: str | None = None
    progress_path: str | None = None
    run_id: str | None = None


def load_spec(*, writer: JsonlProgressWriter | None = None) -> EvalSpec:
    spec = load_config(EvalSpec, writer=writer)
    validate_repo_id(spec.dataset_id, "dataset")
    if not COMMIT_SHA_RE.fullmatch(spec.dataset_commit_sha):
        msg = "dataset_commit_sha must be a full 40-character lowercase commit id"
        raise RunError(msg)
    for source in spec.evaluation.systems:
        validate_repo_id(source.repo_id, "predictions")
    if spec.output_repo_id is not None:
        validate_repo_id(spec.output_repo_id, "output repo")
    return spec


# ------------------------------------------------------------------ reading


def _item_id(value: object) -> str | None:
    if isinstance(value, bool) or not isinstance(value, str | int):
        return None
    text = str(value)
    return text if text else None


def _gold_columns(settings: EvalSettings) -> list[str]:
    columns = [settings.id_column, settings.gold_column, *settings.slice_columns]
    for extra in (settings.cluster_column, settings.exclude_column):
        if extra is not None:
            columns.append(extra)
    return list(dict.fromkeys(columns))


def _load_gold(spec: EvalSpec, token: str | Literal[False]) -> dict[str, list[Any]]:
    try:
        datasets_mod: Any = __import__("datasets")  # pyright: ignore[reportAny] - optional extra
    except ImportError as exc:
        msg = "the [hf] extra is required to load datasets: pip install 'strata-forge[hf]'"
        raise RunError(msg) from exc
    dataset = datasets_mod.load_dataset(
        spec.dataset_id,
        name=spec.dataset_config,
        split=spec.split,
        revision=spec.dataset_commit_sha,
        token=token,
        streaming=False,
    )
    wanted = _gold_columns(spec.evaluation)
    missing = sorted(set(wanted) - set(dataset.column_names or []))
    if missing:
        msg = f"the gold split has no column(s): {missing}"
        raise RunError(msg)
    if len(dataset) > _MAX_ROWS:
        msg = f"the gold split has more than {_MAX_ROWS} rows"
        raise RunError(msg)
    # Only the declared columns are materialised, so no item text is held or written.
    return cast("dict[str, list[Any]]", dataset.select_columns(wanted).to_dict())


def _read_parquet(path: Path, columns: list[str]) -> dict[str, list[Any]]:
    try:
        parquet: Any = __import__("pyarrow.parquet", fromlist=["read_schema"])  # pyright: ignore[reportAny] - optional extra
    except ImportError as exc:
        msg = "reading prediction files needs pyarrow: pip install 'strata-forge[hf]'"
        raise RunError(msg) from exc
    present = set(parquet.read_schema(str(path)).names)
    missing = sorted(set(columns) - present)
    if missing:
        msg = f"the prediction file has no column(s): {missing}"
        raise RunError(msg)
    table = parquet.read_table(str(path), columns=columns)
    if table.num_rows > _MAX_ROWS:
        msg = f"the prediction file has more than {_MAX_ROWS} rows"
        raise RunError(msg)
    return cast("dict[str, list[Any]]", table.to_pydict())


async def _fetch_predictions(
    source: PredictionSource, token: str | Literal[False]
) -> dict[str, list[Any]]:
    hub = HFHubClient(token=token)  # explicit: never forge settings or the VM's own login
    try:
        local = await hub.download_file(
            source.repo_id, source.path, repo_type="dataset", revision=source.commit_sha
        )
    except Exception as exc:
        msg = f"could not download the predictions for {source.name}: {exc}"
        raise RunError(msg) from exc
    columns = [source.id_column, source.prediction_column]
    if source.overflow_column is not None:
        columns.append(source.overflow_column)
    return await asyncio.to_thread(_read_parquet, local, list(dict.fromkeys(columns)))


# ------------------------------------------------------------------ joining


class _Gold(BaseModel):
    """The scored population, in gold row order."""

    model_config = ConfigDict(frozen=True)

    ids: list[str]
    labels: list[str]
    clusters: list[str]
    slices: dict[str, list[list[str]]]
    excluded: list[bool]


class _System(BaseModel):
    """One system's mapped predictions aligned to the gold rows, and what the join counted."""

    model_config = ConfigDict(frozen=True)

    predictions: list[str | None]
    counts: dict[str, int]


def _label(value: object, aliases: Mapping[str, str], labels: tuple[str, ...]) -> str | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    text = str(value)
    mapped = aliases.get(text, text)
    return mapped if mapped in labels else None


def _slice_values(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in cast("list[object]", value) if item is not None]
    return [str(value)]


def _prepare_gold(columns: dict[str, list[Any]], settings: EvalSettings) -> _Gold:
    ids: list[str] = []
    for value in columns[settings.id_column]:
        item = _item_id(value)
        if item is None:
            msg = f"every gold row needs a string or integer {settings.id_column!r}"
            raise RunError(msg)
        ids.append(item)
    if len(set(ids)) != len(ids):
        msg = f"the gold split repeats {len(ids) - len(set(ids))} item id(s)"
        raise RunError(msg)
    labels: list[str] = []
    for value in columns[settings.gold_column]:
        mapped = _label(value, settings.gold_aliases, settings.labels)
        if mapped is None:
            msg = (
                f"gold column {settings.gold_column!r} holds a value outside the declared "
                "labels; map it with gold_aliases"
            )
            raise RunError(msg)
        labels.append(mapped)
    clusters: list[str] = ids
    if settings.cluster_column is not None:
        clusters = []
        for value in columns[settings.cluster_column]:
            cluster = _item_id(value)
            if cluster is None:
                msg = f"every gold row needs a string or integer {settings.cluster_column!r}"
                raise RunError(msg)
            clusters.append(cluster)
    excluded = [False] * len(ids)
    if settings.exclude_column is not None:
        flags = columns[settings.exclude_column]
        if any(not isinstance(flag, bool) for flag in flags):
            msg = f"exclude column {settings.exclude_column!r} must be boolean"
            raise RunError(msg)
        excluded = [bool(flag) for flag in flags]
    slices: dict[str, list[list[str]]] = {}
    for column in settings.slice_columns:
        values = [_slice_values(value) for value in columns[column]]
        distinct = {value for row in values for value in row}
        if len(distinct) > _MAX_SLICE_VALUES:
            msg = f"slice column {column!r} has more than {_MAX_SLICE_VALUES} distinct values"
            raise RunError(msg)
        slices[column] = values
    return _Gold(ids=ids, labels=labels, clusters=clusters, slices=slices, excluded=excluded)


def _align(
    gold: _Gold, columns: dict[str, list[Any]], source: PredictionSource, labels: tuple[str, ...]
) -> _System:
    by_id: dict[str, tuple[object, bool]] = {}
    overflow_flags = (
        columns[source.overflow_column]
        if source.overflow_column is not None
        else [False] * len(columns[source.id_column])
    )
    for raw_id, value, overflow in zip(
        columns[source.id_column], columns[source.prediction_column], overflow_flags, strict=True
    ):
        item = _item_id(raw_id)
        if item is None:
            msg = f"every row of {source.name}'s predictions needs a string or integer id"
            raise RunError(msg)
        if item in by_id:
            msg = f"{source.name}'s predictions repeat an item id"
            raise RunError(msg)
        by_id[item] = (value, overflow is True)
    known = set(gold.ids)
    counts: Counter[str] = Counter(unmatched=sum(1 for item in by_id if item not in known))
    predictions: list[str | None] = []
    for item, excluded in zip(gold.ids, gold.excluded, strict=True):
        found = by_id.get(item)
        if excluded:
            predictions.append(None)
            continue
        if found is None:
            counts["missing"] += 1
            predictions.append(None)
            continue
        value, overflow = found
        mapped = None if overflow else _label(value, source.label_aliases, labels)
        if overflow:
            counts["overflow"] += 1
        elif mapped is None:
            counts["invalid"] += 1
        predictions.append(mapped)
    counts["scored"] = sum(1 for flag in gold.excluded if not flag)
    keys = ("scored", "missing", "invalid", "overflow", "unmatched")
    return _System(predictions=predictions, counts={key: counts[key] for key in keys})


# ------------------------------------------------------------------ scoring


def _populations(gold: _Gold) -> list[tuple[str | None, str | None, _Population]]:
    """``(column, value, rows)`` for the whole population, then each slice, excluded rows out."""
    scored = [i for i, flag in enumerate(gold.excluded) if not flag]
    out: list[tuple[str | None, str | None, _Population]] = [(None, None, scored)]
    for column, values in gold.slices.items():
        members: dict[str, list[int]] = {}
        for i in scored:
            for value in values[i]:
                members.setdefault(value, []).append(i)
        out.extend((column, value, members[value]) for value in sorted(members))
    return out


def _intervals(intervals: SystemIntervals) -> dict[str, dict[str, float]]:
    return {
        name: interval.model_dump()
        for name, interval in (
            ("accuracy", intervals.accuracy),
            ("macro_f1", intervals.macro_f1),
            ("invalid_rate", intervals.invalid_rate),
        )
    }


def _score_population(
    rows: _Population,
    gold: _Gold,
    systems: dict[str, _System],
    settings: EvalSettings,
    *,
    detail: bool,
) -> dict[str, Any]:
    actual = [gold.labels[i] for i in rows]
    clusters = [gold.clusters[i] for i in rows]
    out: dict[str, Any] = {
        "n": len(rows),
        "clusters": len(set(clusters)),
        "diagnostic": len(rows) < settings.min_slice_rows,
        "systems": {},
        "comparisons": [],
    }
    for name, system in systems.items():
        predictions = [system.predictions[i] for i in rows]
        result = cluster_bootstrap(
            actual,
            predictions,
            labels=settings.labels,
            clusters=clusters,
            seed=settings.seed,
            resamples=settings.resamples,
        )
        entry: dict[str, Any] = _intervals(result.intervals)
        if detail:
            report = classification_report(actual, predictions, labels=settings.labels)
            entry["per_class"] = [score.model_dump() for score in report.per_class]
            entry["confusion"] = {
                "counts": [list(row) for row in report.confusion.counts],
                "invalid_by_gold": list(report.confusion.invalid_by_gold),
            }
        out["systems"][name] = entry
        out["bootstrap"] = result.settings.model_dump()
    candidate = settings.predictions.name
    for baseline in settings.compare_to:
        paired = paired_cluster_bootstrap(
            actual,
            [systems[baseline.name].predictions[i] for i in rows],
            [systems[candidate].predictions[i] for i in rows],
            labels=settings.labels,
            clusters=clusters,
            seed=settings.seed,
            resamples=settings.resamples,
        )
        out["comparisons"].append(
            {"baseline": baseline.name, "candidate": candidate, "delta": _intervals(paired.delta)}
        )
    return out


def _headline(report: dict[str, Any], settings: EvalSettings) -> dict[str, float]:
    """The overall figures the ``end`` event carries (well under its 64-key cap)."""
    overall = report["slices"][0]
    candidate = overall["systems"][settings.predictions.name]
    metrics: dict[str, float] = {
        "rows": float(overall["n"]),
        "clusters": float(overall["clusters"]),
    }
    for name in ("accuracy", "macro_f1", "invalid_rate"):
        metrics[name] = candidate[name]["point"]
        metrics[f"{name}_lower"] = candidate[name]["lower"]
        metrics[f"{name}_upper"] = candidate[name]["upper"]
    for comparison in overall["comparisons"]:
        for name in ("accuracy", "macro_f1"):
            delta = comparison["delta"][name]
            for part in ("point", "lower", "upper"):
                suffix = "" if part == "point" else f"_{part}"
                metrics[f"delta_{name}{suffix}__{comparison['baseline']}"] = delta[part]
    return metrics


# ------------------------------------------------------------------ writing


def _items_rows(gold: _Gold, systems: dict[str, _System]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i, item in enumerate(gold.ids):
        row: dict[str, Any] = {
            "item_id": item,
            "gold": gold.labels[i],
            "cluster": gold.clusters[i],
            "excluded": gold.excluded[i],
        }
        for column, values in gold.slices.items():
            row[f"slice__{column}"] = values[i]
        for name, system in systems.items():
            prediction = system.predictions[i]
            row[f"pred__{name}"] = prediction
            row[f"correct__{name}"] = None if gold.excluded[i] else prediction == gold.labels[i]
        rows.append(row)
    return rows


def _write_items(rows: list[dict[str, Any]], outdir: Path) -> Path:
    parquet: Any = __import__("pyarrow.parquet", fromlist=["write_table"])  # pyright: ignore[reportAny] - optional extra
    arrow: Any = __import__("pyarrow")  # pyright: ignore[reportAny] - optional extra
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / ITEMS_FILENAME
    parquet.write_table(arrow.Table.from_pylist(rows), str(path))
    return path


_CARD = """---
configs:
- config_name: items
  data_files: items.parquet
  default: true
---

# Evaluation report

Written by `strata_forge.pipelines.eval_runner`. `items.parquet` holds one row per gold item:
its id, gold label, cluster, slice values and, per system, the mapped prediction and whether it
was correct. `metrics.json` holds the report and `manifest.json` the pinned inputs.
"""


def _write_outputs(
    spec: EvalSpec,
    gold: _Gold,
    systems: dict[str, _System],
    report: dict[str, Any],
    outdir: Path,
) -> dict[str, Path]:
    items = _write_items(_items_rows(gold, systems), outdir)
    (outdir / REPORT_FILENAME).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    (outdir / _CARD_FILENAME).write_text(_CARD)
    write_manifest(_build_manifest(spec, gold, systems), outdir)
    return {
        ITEMS_FILENAME: items,
        REPORT_FILENAME: outdir / REPORT_FILENAME,
        MANIFEST_FILENAME: outdir / MANIFEST_FILENAME,
        _CARD_FILENAME: outdir / _CARD_FILENAME,
    }


def _build_manifest(spec: EvalSpec, gold: _Gold, systems: dict[str, _System]) -> dict[str, Any]:
    return {
        "schema": MANIFEST_SCHEMA,
        "kind": _MANIFEST_KIND,
        "engine": manifest_engine(spec.engine_version, _RECORDED_DISTRIBUTIONS),
        "model": {"id": spec.model_id},
        "dataset": {
            "id": spec.dataset_id,
            "config": spec.dataset_config,
            "split": spec.split,
            "commit": spec.dataset_commit_sha,
        },
        "predictions": [
            {"name": s.name, "repo_id": s.repo_id, "commit": s.commit_sha, "path": s.path}
            for s in spec.evaluation.systems
        ],
        "evaluation": spec.evaluation.model_dump(mode="json"),
        "counts": {
            "rows": len(gold.ids),
            "excluded": sum(gold.excluded),
            "systems": {name: system.counts for name, system in systems.items()},
        },
    }


def _report(
    spec: EvalSpec,
    gold: _Gold,
    systems: dict[str, _System],
    slices: list[dict[str, Any]],
) -> dict[str, Any]:
    settings = spec.evaluation
    return {
        "schema": REPORT_SCHEMA,
        "labels": list(settings.labels),
        "candidate": settings.predictions.name,
        "baselines": [source.name for source in settings.compare_to],
        "model_id": spec.model_id,
        "dataset": {
            "id": spec.dataset_id,
            "config": spec.dataset_config,
            "split": spec.split,
            "commit": spec.dataset_commit_sha,
        },
        "systems": [
            {
                "name": source.name,
                "repo_id": source.repo_id,
                "commit": source.commit_sha,
                "path": source.path,
                "counts": systems[source.name].counts,
            }
            for source in settings.systems
        ],
        "population": {
            "rows": len(gold.ids),
            "excluded": sum(gold.excluded),
            "cluster_column": settings.cluster_column,
            "min_slice_rows": settings.min_slice_rows,
        },
        "slices": slices,
    }


async def _push(spec: EvalSpec, files: dict[str, Path], hf_token: str) -> str:
    """Push the report files in ONE commit, so a half-landed report can never exist."""
    if not spec.output_repo_id:
        msg = "output_repo_id is required to push results"
        raise RunError(msg)
    out_repo = validate_repo_id(spec.output_repo_id, "output repo")
    hub = HFHubClient(token=hf_token)  # EXPLICIT write token, never the ambient HF_TOKEN
    await ensure_private_output(hub, out_repo)
    await hub.upload_files(
        dict(files),
        out_repo,
        repo_type="dataset",
        commit_message="evaluation report",
    )
    return out_repo


# ------------------------------------------------------------------ run


async def _execute(spec: EvalSpec, secrets: RunSecrets, writer: JsonlProgressWriter | None) -> str:
    hf_token = secrets.hf_token_value()
    hub_token = secrets.hub_credential()
    phase = phase_sink(writer, hf_token)
    settings = spec.evaluation
    pushing = bool(hf_token and spec.output_repo_id)

    if pushing and spec.output_repo_id:
        async with ticking_phase(phase, "Checking the output repo", stage="load_model"):
            await ensure_private_output(
                HFHubClient(token=hf_token), validate_repo_id(spec.output_repo_id, "output repo")
            )
    async with ticking_phase(phase, "Loading the gold split", stage="load_model"):
        columns = await asyncio.to_thread(_load_gold, spec, hub_token)
        gold = await asyncio.to_thread(_prepare_gold, columns, settings)
    systems: dict[str, _System] = {}
    for source in settings.systems:
        async with ticking_phase(
            phase, f"Loading the predictions of {source.name}", stage="load_model"
        ):
            raw = await _fetch_predictions(source, hub_token)
            systems[source.name] = await asyncio.to_thread(
                _align, gold, raw, source, settings.labels
            )

    populations = _populations(gold)
    if not populations[0][2]:
        msg = "every gold row is excluded: there is nothing to score"
        raise RunError(msg)
    emit(writer, ProgressEvent(kind="start", stage="run", total_steps=len(populations)))
    slices: list[dict[str, Any]] = []
    async with ticking_phase(phase, "Computing metrics", stage="run"):
        for step, (column, value, rows) in enumerate(populations, start=1):
            scored = await asyncio.to_thread(
                _score_population, rows, gold, systems, settings, detail=column is None
            )
            slices.append({"column": column, "value": value, **scored})
            emit(
                writer,
                ProgressEvent(kind="step", stage="run", step=step, total_steps=len(populations)),
            )
    report = _report(spec, gold, systems, slices)

    async with ticking_phase(phase, "Writing the report", stage="push"):
        outdir = results_dir(spec.run_id, name=_RESULTS_DIR_NAME)
        files = await asyncio.to_thread(_write_outputs, spec, gold, systems, report, outdir)
    if pushing and hf_token:
        try:
            async with ticking_phase(phase, "Uploading the report to the Hub", stage="push"):
                destination = await _push(spec, files, hf_token)
        except Exception as exc:
            msg = f"the report is on the VM at {outdir}, but the push failed: {exc}"
            raise RunError(msg) from exc
    else:
        destination = str(outdir)

    emit(
        writer,
        ProgressEvent(
            kind="end",
            stage="push",
            step=len(populations),
            total_steps=len(populations),
            metrics=_headline(report, settings),
            message=destination,
        ),
    )
    return destination


async def main() -> int:
    """Entry point: returns a process exit code (0 ok, 1 failure). Never leaks the token."""
    return await runner_main(
        lambda writer, secrets: _execute(load_spec(writer=writer), secrets, writer)
    )


if __name__ == "__main__":  # pragma: no cover
    sys.exit(asyncio.run(main()))
