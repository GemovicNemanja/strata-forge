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
- Every row gets a status per system: ``ok``; ``tier_refused`` (over the declared tier rule's
  premise or token cap) or ``excluded`` (flagged in ``exclude_column``), both common to every
  system and left out of every population; ``overflow`` (the system could not score it); or
  ``missing`` (no prediction, a null one, or a value outside the label space). A missing row is
  always INVALID: wrong, a false negative for its gold class and no false positive; it is never
  excluded. A prediction id with no gold item is counted and ignored. Duplicate ids refuse.
- The primary read is over the SHARED population: the rows no system overflowed. Its size and
  the overflow-excluded count and share are reported, and each system also gets a full-tier
  secondary read over every non-refused row, its own overflow counted as invalid.
- Slices are the values of declared columns (a list value puts the row in each listed slice).
  A slice whose shared population has fewer than ``min_slice_rows`` rows is marked diagnostic.
- Whole clusters are resampled within each slice, so only the clusters present in it are drawn.
  The cluster is ``cluster_column``; a row without one, or every row when only
  ``premises_column`` is declared, clusters on :func:`premise_cluster_key`; else each item is its
  own cluster.
- With ``calibration`` set, each file's ``logit_<label>`` columns give three separately reported
  probability reads on the shared rows: ``raw`` (their softmax), ``cross_fit`` (a temperature
  2-fold cross-fitted by cluster over the fit population: this split, or its named slice) and
  ``applied`` (the system's own pinned ``calibration.json`` temperature, when it declares one).
  Each has sum-form Brier with a cluster interval, NLL and top-label ECE; each comparison has a
  paired raw Brier delta and the SD of the per-item Brier differences. Temperatures produce these
  reads and select nothing.

The report carries no item text: ids, gold labels, mapped predictions and slice values only.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import re
import sys
import unicodedata
from collections import Counter
from functools import partial
from typing import TYPE_CHECKING, Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from strata_forge.metrics.calibration import temperature_scale
from strata_forge.metrics.calibration_fit import (
    ECE_BINS,
    CrossFitTemperature,
    cross_fit_temperature,
    expected_calibration_error,
    reliability_bins,
)
from strata_forge.metrics.classification import brier, classification_report, nll
from strata_forge.metrics.cluster_bootstrap import (
    CLUSTER_BOOTSTRAP_RESAMPLES,
    MAX_CLUSTER_BOOTSTRAP_RESAMPLES,
    MAX_CLUSTER_BOOTSTRAP_SEED,
    SystemIntervals,
    cluster_bootstrap,
    cluster_mean_bootstrap,
    paired_cluster_bootstrap,
    paired_cluster_mean_bootstrap,
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
    "CalibrationRef",
    "CalibrationSettings",
    "EvalSettings",
    "EvalSpec",
    "PredictionSource",
    "TierLimit",
    "TierRule",
    "main",
    "premise_cluster_key",
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
_MAX_IDENTIFIER_CHARS = 256
# A parquet path inside a dataset repo: relative segments of safe characters, ending .parquet.
_PATH_RE = re.compile(
    r"\A[A-Za-z0-9_-][A-Za-z0-9._-]*(?:/[A-Za-z0-9_-][A-Za-z0-9._-]*)*\.parquet\Z"
)

_JSON_PATH_RE = re.compile(
    r"\A[A-Za-z0-9_-][A-Za-z0-9._-]*(?:/[A-Za-z0-9_-][A-Za-z0-9._-]*)*\.json\Z"
)
_MAX_CALIBRATION_BYTES = 64 * 1024

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


class CalibrationRef(BaseModel):
    """A system's own ``calibration.json`` (a temperature fitted on ID dev), at a pinned commit."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    repo_type: Literal["model", "dataset"] = "model"
    repo_id: str = Field(min_length=1, max_length=255)
    commit_sha: str
    path: str = Field(default="calibration.json", max_length=255)

    @field_validator("commit_sha")
    @classmethod
    def _pinned(cls, value: str) -> str:
        if not COMMIT_SHA_RE.fullmatch(value):
            msg = "commit_sha must be a full 40-character lowercase commit id"
            raise ValueError(msg)
        return value

    @field_validator("path")
    @classmethod
    def _json_path(cls, value: str) -> str:
        if ".." in value or not _JSON_PATH_RE.fullmatch(value):
            msg = "path must be a relative .json path inside the repo"
            raise ValueError(msg)
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
    # The system's own temperature, applied when calibration.apply_calibration is "auto".
    calibration_ref: CalibrationRef | None = None

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


class TierLimit(BaseModel):
    """The most premises and tokens a row of one tier may have before it is refused."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_premises: int | None = Field(default=None, ge=1)
    max_tokens: int | None = Field(default=None, ge=1)


class TierRule(BaseModel):
    """A declared refusal rule: a row over its tier's limit is ``tier_refused`` for every system.

    The limit is ``limits[<tier_column value>]``, else ``default``. Premises are counted from the
    settings' ``premises_column``; tokens are read from ``token_column``, a per-row count the
    gold split declares.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    tier_column: str | None = None
    limits: dict[str, TierLimit] = Field(default_factory=dict[str, TierLimit])
    default: TierLimit | None = None
    token_column: str | None = None
    # How token_column was counted (e.g. ``tokenizer:<repo>@<commit>``): one value per tier,
    # echoed in the report.
    token_method_column: str | None = None

    @field_validator("tier_column", "token_column", "token_method_column")
    @classmethod
    def _column(cls, value: str | None) -> str | None:
        return None if value is None else _bounded_text(value, "a column name")

    @field_validator("limits")
    @classmethod
    def _bounded(cls, value: dict[str, TierLimit]) -> dict[str, TierLimit]:
        if len(value) > _MAX_SLICE_VALUES:
            msg = f"at most {_MAX_SLICE_VALUES} tier limits"
            raise ValueError(msg)
        for key in value:
            _bounded_text(key, "a tier")
        return value

    @model_validator(mode="after")
    def _complete(self) -> TierRule:
        if self.limits and self.tier_column is None:
            msg = "per-tier limits need a tier_column"
            raise ValueError(msg)
        tokens = [lim.max_tokens for lim in (*self.limits.values(), self.default) if lim]
        if any(tokens) and self.token_column is None:
            msg = "a max_tokens limit needs a token_column"
            raise ValueError(msg)
        if (self.token_method_column is None) != (self.token_column is None):
            # A token count means nothing without how it was counted, and vice versa.
            msg = "token_column and token_method_column are declared together"
            raise ValueError(msg)
        return self

    def limit(self, tier: str | None) -> TierLimit | None:
        return self.limits.get(tier, self.default) if tier is not None else self.default


class CalibrationSettings(BaseModel):
    """Probability reads from each file's logits: raw, cross-fitted, and the system's own.

    Every system's file must carry ``<logit_prefix><label>`` for each declared label. ``raw`` is
    their softmax. ``cross_fit`` comes from ``folds``-fold cross-fitting over the fit population
    (the scored rows of this split, or of its named slice ``fit_column`` = ``fit_value``), so no
    row is scored with a temperature its own cluster helped fit. ``applied`` uses a source's
    ``calibration_ref`` when ``apply_calibration`` is ``auto``. The three are reported apart, and
    no temperature selects anything.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    logit_prefix: str = "logit_"
    fit_column: str | None = None
    fit_value: str | None = None
    folds: int = Field(default=2, ge=2, le=10)
    ece_bins: int = Field(default=ECE_BINS, ge=1, le=100)
    apply_calibration: Literal["auto", "none"] = "auto"

    @field_validator("logit_prefix", "fit_column", "fit_value")
    @classmethod
    def _text(cls, value: str | None) -> str | None:
        return None if value is None else _bounded_text(value, "a calibration field")

    @model_validator(mode="after")
    def _named_slice(self) -> CalibrationSettings:
        if (self.fit_column is None) != (self.fit_value is None):
            msg = "fit_column and fit_value name the fit slice together"
            raise ValueError(msg)
        return self


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
    # A list of premises (strings, or {pid, text} objects) per row: the cluster of a row with no
    # cluster value, and what a tier rule counts.
    premises_column: str | None = None
    # A declared per-row premise count, checked against the platform's count of premises_column.
    premise_count_column: str | None = None
    tier_rule: TierRule | None = None
    # A per-row label space: the declared labels of a row are ``label_spaces[<value>]`` (a 2-way
    # tier declares its own two labels), else ``labels``. A population declares the union of its
    # rows' spaces, so a 2-way slice is never "missing" the 3-way classes.
    label_space_column: str | None = None
    label_spaces: dict[str, tuple[str, ...]] = Field(default_factory=dict[str, tuple[str, ...]])
    seed: int = Field(ge=0, le=MAX_CLUSTER_BOOTSTRAP_SEED)
    resamples: int = Field(
        default=CLUSTER_BOOTSTRAP_RESAMPLES, ge=1, le=MAX_CLUSTER_BOOTSTRAP_RESAMPLES
    )
    min_slice_rows: int = Field(default=500, ge=1)
    # An interval with more than this share of undefined resamples is ``unstable``: reported, and
    # never the basis of a gate read (which then says ``unresolved``).
    unstable_threshold: float = Field(default=0.01, ge=0, le=1)
    calibration: CalibrationSettings | None = None

    @field_validator("labels")
    @classmethod
    def _labels(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(set(value)) != len(value):
            msg = "labels must be unique"
            raise ValueError(msg)
        return tuple(_bounded_text(label, "a label") for label in value)

    @field_validator(
        "id_column",
        "gold_column",
        "cluster_column",
        "exclude_column",
        "premises_column",
        "premise_count_column",
        "label_space_column",
    )
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
        if self.calibration is None and any(s.calibration_ref for s in self.systems):
            msg = "a calibration_ref needs calibration settings"
            raise ValueError(msg)
        rule = self.tier_rule
        limits = [*rule.limits.values(), rule.default] if rule else []
        if any(lim and lim.max_premises for lim in limits) and self.premises_column is None:
            msg = "a max_premises limit needs a premises_column"
            raise ValueError(msg)
        if self.label_spaces and self.label_space_column is None:
            msg = "label_spaces need a label_space_column"
            raise ValueError(msg)
        if len(self.label_spaces) > _MAX_SLICE_VALUES:
            msg = f"at most {_MAX_SLICE_VALUES} label spaces"
            raise ValueError(msg)
        for key, space in self.label_spaces.items():
            _bounded_text(key, "a label space")
            if len(space) < 2 or len(set(space)) != len(space) or set(space) - set(self.labels):
                msg = f"label space {key!r} must be two or more distinct declared labels"
                raise ValueError(msg)
        if self.premise_count_column is not None and self.premises_column is None:
            msg = "a premise_count_column needs a premises_column"
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
        if source.calibration_ref is not None:
            validate_repo_id(source.calibration_ref.repo_id, "calibration")
    if spec.output_repo_id is not None:
        validate_repo_id(spec.output_repo_id, "output repo")
    return spec


# ------------------------------------------------------------------ reading


def _item_id(value: object) -> str | None:
    if isinstance(value, bool) or not isinstance(value, str | int):
        return None
    text = str(value)
    return text if text else None


def premise_cluster_key(value: object) -> str:
    """The cluster key of a row identified by its premises: ``premises:sha256:<hex>``.

    Premises are ``{pid, text}`` objects (a plain string refuses). Normalisation, so that the same
    problem restated (reordered, re-spaced) is one cluster: each premise's ``text`` is Unicode NFC
    normalised, its runs of whitespace collapsed to one space and its ends stripped; the texts
    are sorted by code point and encoded as a compact JSON array (UTF-8, no ASCII escaping), and
    the key is the SHA-256 of that encoding. Case is kept.
    """
    if not isinstance(value, list):
        msg = "a premises value must be a list"
        raise RunError(msg)
    texts: list[str] = []
    for premise in cast("list[object]", value):
        text: object = None
        if isinstance(premise, dict):
            text = cast("dict[str, object]", premise).get("text")
        if not isinstance(text, str):
            msg = "each premise must be a {pid, text} object with a string text"
            raise RunError(msg)
        texts.append(" ".join(unicodedata.normalize("NFC", text).split()))
    encoded = json.dumps(sorted(texts), ensure_ascii=False, separators=(",", ":"))
    return "premises:sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _gold_columns(settings: EvalSettings) -> list[str]:
    columns = [settings.id_column, settings.gold_column, *settings.slice_columns]
    rule = settings.tier_rule
    extras = (
        settings.cluster_column,
        settings.exclude_column,
        settings.premises_column,
        settings.premise_count_column,
        settings.label_space_column,
        rule.tier_column if rule else None,
        rule.token_column if rule else None,
        rule.token_method_column if rule else None,
        settings.calibration.fit_column if settings.calibration else None,
    )
    for extra in extras:
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


def _logit_columns(settings: EvalSettings) -> list[str]:
    if settings.calibration is None:
        return []
    return [f"{settings.calibration.logit_prefix}{label}" for label in settings.labels]


CALIBRATION_SCHEMA = "strata.calibration/v1"
VERIFIER_FILENAME = "strata_verifier.json"


async def _pinned_json(
    hub: HFHubClient, ref: CalibrationRef, path: str, who: str
) -> tuple[dict[str, object], bytes]:
    try:
        local = await hub.download_file(
            ref.repo_id, path, repo_type=ref.repo_type, revision=ref.commit_sha
        )
    except Exception as exc:
        msg = f"could not download {path} for {who}: {exc}"
        raise RunError(msg) from exc
    raw = await asyncio.to_thread(local.read_bytes)
    if len(raw) > _MAX_CALIBRATION_BYTES:
        msg = f"{path} for {who} is larger than {_MAX_CALIBRATION_BYTES} bytes"
        raise RunError(msg)
    try:
        data: object = json.loads(raw)
    except ValueError as exc:
        msg = f"{path} for {who} is not JSON"
        raise RunError(msg) from exc
    if not isinstance(data, dict):
        msg = f"{path} for {who} is not a JSON object"
        raise RunError(msg)
    return cast("dict[str, object]", data), raw


async def _fetch_calibration(
    source: PredictionSource, labels: tuple[str, ...], token: str | Literal[False]
) -> dict[str, Any]:
    """The source's own temperature from the trainer's pinned ``calibration.json``.

    The file is ``strata.calibration/v1`` (written by the seq-cls trainer): the temperature is
    ``fit.temperature``. Its labels are the file's ``labels`` when it has them, else the
    ``labels`` of the ``strata_verifier.json`` beside it in the same repo at the same commit; they
    must be the declared labels in order, because the temperature scales logits in that order.
    """
    ref = source.calibration_ref
    if ref is None:  # pragma: no cover - callers check
        msg = "no calibration_ref"
        raise RunError(msg)
    who = f"the calibration of {source.name}"
    hub = HFHubClient(token=token)
    body, raw = await _pinned_json(hub, ref, ref.path, who)
    if body.get("schema") != CALIBRATION_SCHEMA:
        msg = f"{who} is not a {CALIBRATION_SCHEMA} file"
        raise RunError(msg)
    fit = body.get("fit")
    temperature = (
        cast("dict[str, object]", fit).get("temperature") if isinstance(fit, dict) else None
    )
    if (
        isinstance(temperature, bool)
        or not isinstance(temperature, int | float)
        or not math.isfinite(temperature)
        or temperature <= 0
    ):
        msg = f"{who} has no positive finite fit.temperature"
        raise RunError(msg)
    labels_from = "calibration.json"
    named: object = body.get("labels")
    if named is None:
        directory = ref.path.rpartition("/")[0]
        sibling = f"{directory}/{VERIFIER_FILENAME}" if directory else VERIFIER_FILENAME
        verifier, _ = await _pinned_json(hub, ref, sibling, who)
        named = verifier.get("labels")
        labels_from = sibling
    if named != list(labels):
        msg = f"{who} ({labels_from}) does not name the declared labels in order"
        raise RunError(msg)
    return {
        "temperature": float(temperature),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "labels_from": labels_from,
        "ref": ref.model_dump(),
    }


async def _fetch_predictions(
    source: PredictionSource, token: str | Literal[False], logit_columns: list[str]
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
    columns.extend(logit_columns)
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
    refused: list[bool]
    # Whether each row's cluster is the premise-hash fallback (it had no cluster value).
    fallback: list[bool]
    # Each row's declared labels (its label space).
    spaces: list[tuple[str, ...]]
    # The token-count method per tier (``tier_column`` value, or ``"*"`` with no tier column).
    token_methods: dict[str, str]
    # Whether each row is in the calibration fit population's named slice (all True without one).
    fit: list[bool]

    def common_out(self, row: int) -> bool:
        return self.excluded[row] or self.refused[row]


type _Status = Literal["ok", "tier_refused", "excluded", "overflow", "missing"]


class _System(BaseModel):
    """One system's mapped predictions aligned to the gold rows, and what the join counted."""

    model_config = ConfigDict(frozen=True)

    predictions: list[str | None]
    status: list[_Status]
    counts: dict[str, int]
    # Finite logits in label order per gold row, None where absent; None without calibration.
    logits: list[list[float] | None] | None = None


def _label(value: object, aliases: Mapping[str, str], labels: tuple[str, ...]) -> str | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    text = str(value)
    mapped = aliases.get(text, text)
    return mapped if mapped in labels else None


def _slice_values(value: object) -> list[str]:
    """A row's slice values, each once (a repeated list value would count the row twice)."""
    if value is None:
        return []
    if isinstance(value, list):
        items = [str(item) for item in cast("list[object]", value) if item is not None]
        return list(dict.fromkeys(items))
    return [str(value)]


def _check_identifiers(values: list[Any], column: str) -> None:
    """Refuse an id, cluster or slice column that mixes value types (``1`` and ``"1"``, ``True``
    and ``"True"`` would otherwise become one value) or holds text rather than short identifiers:
    these values are written to ``items.parquet``, and a text column there would publish it."""
    kinds: set[str] = set()
    for value in values:
        items = cast("list[object]", value) if isinstance(value, list) else [value]
        for item in items:
            if item is None:
                continue
            kinds.add(type(item).__name__)
            if len(str(item)) > _MAX_IDENTIFIER_CHARS:
                msg = (
                    f"column {column!r} holds a value over {_MAX_IDENTIFIER_CHARS} characters; "
                    "id, cluster and slice columns must be short identifiers, not text"
                )
                raise RunError(msg)
    if len(kinds) > 1:
        msg = f"column {column!r} mixes value types; use one type for every row"
        raise RunError(msg)


def _prepare_gold(columns: dict[str, list[Any]], settings: EvalSettings) -> _Gold:
    for column in (settings.id_column, settings.cluster_column, *settings.slice_columns):
        if column is not None:
            _check_identifiers(columns[column], column)
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
    clusters, fallback = _clusters(columns, settings, ids)
    _check_premise_counts(columns, settings)
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
    return _Gold(
        ids=ids,
        labels=labels,
        clusters=clusters,
        slices=slices,
        excluded=excluded,
        refused=_refused(columns, settings, len(ids)),
        fallback=fallback,
        spaces=_spaces(columns, settings, labels),
        token_methods=_token_methods(columns, settings, len(ids)),
        fit=_fit_rows(columns, settings, len(ids)),
    )


def _spaces(
    columns: dict[str, list[Any]], settings: EvalSettings, labels: list[str]
) -> list[tuple[str, ...]]:
    """Each row's label space; refuses a gold label outside its row's space."""
    if settings.label_space_column is None:
        return [settings.labels] * len(labels)
    out: list[tuple[str, ...]] = []
    for row, (value, gold) in enumerate(
        zip(columns[settings.label_space_column], labels, strict=True)
    ):
        space = settings.label_spaces.get(str(value)) if value is not None else None
        if space is None:
            msg = (
                f"row {row}: {settings.label_space_column!r} value {value!r} names no declared "
                "label space"
            )
            raise RunError(msg)
        if gold not in space:
            msg = f"row {row}: gold label {gold!r} is outside its label space {list(space)}"
            raise RunError(msg)
        out.append(space)
    return out


def _declared(rows: _Population, gold: _Gold, settings: EvalSettings) -> tuple[str, ...]:
    """The labels a population declares: the union of its rows' spaces, in ``labels`` order."""
    seen = {label for i in rows for label in gold.spaces[i]}
    return tuple(label for label in settings.labels if label in seen)


def _fit_rows(columns: dict[str, list[Any]], settings: EvalSettings, n: int) -> list[bool]:
    calibration = settings.calibration
    if calibration is None or calibration.fit_column is None:
        return [True] * n
    return [calibration.fit_value in _slice_values(v) for v in columns[calibration.fit_column]]


def _check_premise_counts(columns: dict[str, list[Any]], settings: EvalSettings) -> None:
    """Refuse a declared premise count that disagrees with the platform's own count."""
    if settings.premise_count_column is None or settings.premises_column is None:
        return
    for row, (premises, declared) in enumerate(
        zip(columns[settings.premises_column], columns[settings.premise_count_column], strict=True)
    ):
        if not isinstance(premises, list):
            msg = f"premises column {settings.premises_column!r} must hold lists"
            raise RunError(msg)
        counted = len(cast("list[object]", premises))
        if isinstance(declared, bool) or not isinstance(declared, int) or declared != counted:
            msg = (
                f"row {row}: {settings.premise_count_column!r} says {declared!r} premises but "
                f"{settings.premises_column!r} holds {counted}"
            )
            raise RunError(msg)


def _token_methods(columns: dict[str, list[Any]], settings: EvalSettings, n: int) -> dict[str, str]:
    """The one token-count method of each tier; refuses a tier that mixes methods or lacks one."""
    rule = settings.tier_rule
    if rule is None or rule.token_method_column is None:
        return {}
    tiers = columns[rule.tier_column] if rule.tier_column else ["*"] * n
    seen: dict[str, set[str]] = {}
    for tier, method in zip(tiers, columns[rule.token_method_column], strict=True):
        if not isinstance(method, str) or not method:
            msg = f"every row needs a {rule.token_method_column!r} value"
            raise RunError(msg)
        if len(method) > _MAX_IDENTIFIER_CHARS:
            msg = f"{rule.token_method_column!r} values must be at most {_MAX_IDENTIFIER_CHARS} characters"
            raise RunError(msg)
        seen.setdefault(str(tier), set()).add(method)
    mixed = sorted(tier for tier, methods in seen.items() if len(methods) > 1)
    if mixed:
        msg = f"{rule.token_method_column!r} mixes token-count methods within tier(s): {mixed}"
        raise RunError(msg)
    return {tier: next(iter(methods)) for tier, methods in sorted(seen.items())}


def _clusters(
    columns: dict[str, list[Any]], settings: EvalSettings, ids: list[str]
) -> tuple[list[str], list[bool]]:
    """Each row's cluster key, and whether it is the premise-hash fallback."""
    premises = columns[settings.premises_column] if settings.premises_column else None
    given = columns[settings.cluster_column] if settings.cluster_column else [None] * len(ids)
    if settings.cluster_column is None and premises is None:
        return list(ids), [False] * len(ids)
    out: list[str] = []
    fallback: list[bool] = []
    for row, value in enumerate(given):
        cluster = _item_id(value)
        fallback.append(cluster is None)
        if cluster is None:
            if premises is None:
                msg = f"every gold row needs a string or integer {settings.cluster_column!r}"
                raise RunError(msg)
            cluster = premise_cluster_key(premises[row])
        out.append(cluster)
    return out, fallback


def _refused(columns: dict[str, list[Any]], settings: EvalSettings, n: int) -> list[bool]:
    rule = settings.tier_rule
    if rule is None:
        return [False] * n
    tiers = columns[rule.tier_column] if rule.tier_column else [None] * n
    premises = columns[settings.premises_column] if settings.premises_column else None
    tokens = columns[rule.token_column] if rule.token_column else None
    out: list[bool] = []
    for row in range(n):
        tier = None if tiers[row] is None else str(tiers[row])
        limit = rule.limit(tier)
        over = False
        if limit is not None and limit.max_premises is not None and premises is not None:
            value = premises[row]
            if not isinstance(value, list):
                msg = f"premises column {settings.premises_column!r} must hold lists"
                raise RunError(msg)
            over = len(cast("list[object]", value)) > limit.max_premises
        if limit is not None and limit.max_tokens is not None and tokens is not None:
            count = tokens[row]
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                msg = f"token column {rule.token_column!r} must hold non-negative integers"
                raise RunError(msg)
            over = over or count > limit.max_tokens
        out.append(over)
    return out


def _logit_row(values: list[object]) -> list[float] | None:
    out: list[float] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int | float):
            return None
        if not math.isfinite(value):
            return None
        out.append(float(value))
    return out


def _align(
    gold: _Gold,
    columns: dict[str, list[Any]],
    source: PredictionSource,
    labels: tuple[str, ...],
    logit_columns: list[str] | None = None,
) -> _System:
    by_id: dict[str, tuple[object, bool]] = {}
    _check_identifiers(columns[source.id_column], source.id_column)
    overflow_flags = (
        columns[source.overflow_column]
        if source.overflow_column is not None
        else [False] * len(columns[source.id_column])
    )
    if any(not isinstance(flag, bool) for flag in overflow_flags):
        msg = f"{source.name}'s overflow column {source.overflow_column!r} must be boolean"
        raise RunError(msg)
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
    status: list[_Status] = []
    for row, item in enumerate(gold.ids):
        found = by_id.get(item)
        mapped: str | None = None
        state: _Status = "ok"
        if gold.refused[row]:
            state = "tier_refused"
        elif gold.excluded[row]:
            state = "excluded"
        elif found is None:
            counts["missing"] += 1
            state = "missing"
        elif found[1]:
            counts["overflow"] += 1
            state = "overflow"
        else:
            mapped = _label(found[0], source.label_aliases, labels)
            if mapped is None:
                counts["invalid"] += 1
                state = "missing"
        predictions.append(mapped)
        status.append(state)
    counts["scored"] = sum(1 for row in range(len(gold.ids)) if not gold.common_out(row))
    keys = ("scored", "missing", "invalid", "overflow", "unmatched")
    logits: list[list[float] | None] | None = None
    if logit_columns:
        rows_by_id = {_item_id(raw): row for row, raw in enumerate(columns[source.id_column])}
        logits = []
        for item, state in zip(gold.ids, status, strict=True):
            row = rows_by_id.get(item)
            usable = row is not None and state in ("ok", "missing")
            logits.append(
                _logit_row([columns[c][row] for c in logit_columns])
                if usable and row is not None
                else None
            )
    return _System(
        predictions=predictions,
        status=status,
        counts={key: counts[key] for key in keys},
        logits=logits,
    )


# ------------------------------------------------------------------ probabilities


type _Probs = list[tuple[float, ...] | None]


class _Probabilities(BaseModel):
    """One system's probability reads per gold row (None where it has none)."""

    model_config = ConfigDict(frozen=True)

    raw: _Probs
    cross_fit: _Probs
    applied: _Probs | None
    fit: CrossFitTemperature | None
    own: dict[str, Any] | None


def _scaled(logits: list[list[float] | None], rows: list[int], t: float, n: int) -> _Probs:
    out: _Probs = [None] * n
    if rows:
        scaled = temperature_scale([logits[i] or [] for i in rows], temperature=t)
        for i, probs in zip(rows, scaled, strict=True):
            out[i] = probs
    return out


def _probabilities(
    gold: _Gold,
    system: _System,
    settings: EvalSettings,
    calibration: CalibrationSettings,
    own: dict[str, Any] | None,
) -> _Probabilities:
    n = len(gold.ids)
    logits: list[list[float] | None] = system.logits or [None] * n
    have = [i for i, row in enumerate(logits) if row is not None]
    fit_rows = [i for i in have if gold.fit[i] and not gold.common_out(i)]
    cross: _Probs = [None] * n
    fitted: CrossFitTemperature | None = None
    if len({gold.clusters[i] for i in fit_rows}) >= calibration.folds:
        fitted = cross_fit_temperature(
            [logits[i] or [] for i in fit_rows],
            [gold.labels[i] for i in fit_rows],
            labels=settings.labels,
            clusters=[gold.clusters[i] for i in fit_rows],
            seed=settings.seed,
            folds=calibration.folds,
        )
        for i, probs in zip(fit_rows, fitted.probabilities, strict=True):
            cross[i] = probs
    return _Probabilities(
        raw=_scaled(logits, have, 1.0, n),
        cross_fit=cross,
        applied=None if own is None else _scaled(logits, have, own["temperature"], n),
        fit=fitted,
        own=own,
    )


def _brier_term(gold: str, probs: tuple[float, ...], labels: tuple[str, ...]) -> float:
    """One row's multiclass Brier term in sum form, on [0, 2]; ``metrics.brier`` averages it."""
    return math.fsum((p - (label == gold)) ** 2 for p, label in zip(probs, labels, strict=True))


def _probability_block(
    rows: _Population, gold: _Gold, probs: _Probs | None, settings: EvalSettings, bins: int
) -> dict[str, Any] | None:
    if probs is None:
        return None
    keep = [i for i in rows if probs[i] is not None]
    if not keep:
        return None
    labels = settings.labels
    actual = [gold.labels[i] for i in keep]
    vectors = [probs[i] or () for i in keep]
    terms = [_brier_term(g, v, labels) for g, v in zip(actual, vectors, strict=True)]
    interval = cluster_mean_bootstrap(
        terms,
        clusters=[gold.clusters[i] for i in keep],
        seed=settings.seed,
        resamples=settings.resamples,
    ).interval
    point = brier(actual, vectors, labels=labels, probability_kind="class_probabilities").value
    coverage = len(keep) / len(rows)
    ids = [gold.ids[i] for i in keep]
    # The ECE's own bins, so a reliability diagram drawn from them adds up to the ECE beside it.
    groups = reliability_bins(actual, vectors, labels=labels, bins=bins, item_ids=ids)
    return {
        "n": len(keep),
        "n_population": len(rows),
        "coverage": coverage,
        # A probability metric over fewer than every row decides no gate (unresolved = not met).
        "gate": "resolved" if coverage == 1 else "unresolved",
        "brier": _stability({**interval.model_dump(), "point": point}, settings),
        "nll": nll(actual, vectors, labels=labels, probability_kind="class_probabilities").value,
        "ece": expected_calibration_error(
            actual, vectors, labels=labels, bins=bins, item_ids=ids
        ).value,
        "reliability_bins": [group.model_dump() for group in groups],
    }


def _brier_delta(
    rows: _Population,
    gold: _Gold,
    baseline: _Probabilities,
    candidate: _Probabilities,
    settings: EvalSettings,
) -> dict[str, Any] | None:
    """Candidate minus baseline raw Brier on the rows both scored, paired by cluster, with the
    SD of the per-item differences."""
    shared = [i for i in rows if baseline.raw[i] is not None and candidate.raw[i] is not None]
    if len(shared) < 2:
        return None
    labels = settings.labels
    paired = paired_cluster_mean_bootstrap(
        [_brier_term(gold.labels[i], baseline.raw[i] or (), labels) for i in shared],
        [_brier_term(gold.labels[i], candidate.raw[i] or (), labels) for i in shared],
        clusters=[gold.clusters[i] for i in shared],
        seed=settings.seed,
        resamples=settings.resamples,
    )
    delta = _stability(paired.delta.model_dump(), settings)
    coverage = len(shared) / len(rows)
    return {
        "n": len(shared),
        "n_population": len(rows),
        "coverage": coverage,
        **delta,
        # Unresolved, which counts as NOT met, when either side lacks probabilities for a row.
        "excludes_zero": _excludes_zero(delta) if coverage == 1 else "unresolved",
        "sd": _stability(paired.delta_sd.model_dump(), settings),
    }


# ------------------------------------------------------------------ scoring


def _populations(gold: _Gold) -> list[tuple[str | None, str | None, _Population, int]]:
    """``(column, value, rows, tier_refused)`` for the whole population, then each slice:
    excluded and refused rows out, and how many rows of it the tier rule refused."""
    scored = [i for i in range(len(gold.ids)) if not gold.common_out(i)]
    refused = [i for i in range(len(gold.ids)) if gold.refused[i] and not gold.excluded[i]]
    out: list[tuple[str | None, str | None, _Population, int]] = [
        (None, None, scored, len(refused))
    ]
    for column, values in gold.slices.items():
        members: dict[str, list[int]] = {}
        refusals: Counter[str] = Counter()
        for i in scored:
            for value in values[i]:
                members.setdefault(value, []).append(i)
        for i in refused:
            refusals.update(values[i])
        out.extend(
            (column, value, members.get(value, []), refusals[value])
            for value in sorted(set(members) | set(refusals))
        )
    return out


def _stability(interval: dict[str, Any], settings: EvalSettings) -> dict[str, Any]:
    """Add ``undefined_share`` and ``unstable`` (over ``settings.unstable_threshold``)."""
    share = interval.get("undefined_resamples", 0) / settings.resamples
    unstable = share > settings.unstable_threshold or interval.get("lower") is None
    return {**interval, "undefined_share": share, "unstable": unstable}


def _excludes_zero(interval: dict[str, Any]) -> bool | Literal["unresolved"]:
    """The gate read on a delta: whether its interval lies wholly on one side of zero, or
    ``unresolved`` when the interval is unstable and so may not decide anything."""
    if interval["unstable"]:
        return "unresolved"
    return bool(interval["lower"] > 0 or interval["upper"] < 0)


def _intervals(
    intervals: SystemIntervals, settings: EvalSettings, *, delta: bool = False
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for name, interval in (
        ("accuracy", intervals.accuracy),
        ("macro_f1", intervals.macro_f1),
        ("invalid_rate", intervals.invalid_rate),
    ):
        entry = _stability(interval.model_dump(), settings)
        if delta:
            entry["excludes_zero"] = _excludes_zero(entry)
        out[name] = entry
    # A diagnostic only: macro-F1 over the classes present, never a gate read.
    present = _stability(intervals.macro_f1_present_classes.model_dump(), settings)
    out["macro_f1_present_classes"] = {
        **present,
        "classes": list(intervals.present_classes),
        "gate_read": False,
    }
    return out


def _system_read(
    rows: _Population,
    gold: _Gold,
    system: _System,
    settings: EvalSettings,
    *,
    detail: bool,
    declared: tuple[str, ...],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """One system on ``rows`` over the ``declared`` labels; a prediction outside them is invalid."""
    actual = [gold.labels[i] for i in rows]
    predictions = [system.predictions[i] for i in rows]
    result = cluster_bootstrap(
        actual,
        predictions,
        labels=declared,
        clusters=[gold.clusters[i] for i in rows],
        seed=settings.seed,
        resamples=settings.resamples,
    )
    entry: dict[str, Any] = _intervals(result.intervals, settings)
    if detail:
        report = classification_report(actual, predictions, labels=declared)
        entry["per_class"] = [score.model_dump() for score in report.per_class]
        entry["confusion"] = {
            "counts": [list(row) for row in report.confusion.counts],
            "invalid_by_gold": list(report.confusion.invalid_by_gold),
        }
    return entry, result.settings.model_dump()


def _discordance(
    rows: _Population,
    gold: _Gold,
    base: _System,
    cand: _System,
    with_probabilities: tuple[_Probabilities, _Probabilities] | None = None,
) -> dict[str, Any]:
    """The paired discordance rate p_d: the share of rows exactly one of the two got right.

    With calibration it is read on the rows BOTH systems have probabilities for, beside the Brier
    delta, and reports that n and its coverage of the population."""
    population = len(rows)
    if with_probabilities is not None:
        left, right = with_probabilities
        rows = [i for i in rows if left.raw[i] is not None and right.raw[i] is not None]
    base_only = sum(
        1
        for i in rows
        if base.predictions[i] == gold.labels[i] and cand.predictions[i] != gold.labels[i]
    )
    cand_only = sum(
        1
        for i in rows
        if cand.predictions[i] == gold.labels[i] and base.predictions[i] != gold.labels[i]
    )
    coverage = len(rows) / population
    return {
        "n": len(rows),
        "n_population": population,
        "coverage": coverage,
        "baseline_only_correct": base_only,
        "candidate_only_correct": cand_only,
        "p_d": (base_only + cand_only) / len(rows) if rows else None,
        "gate": "resolved" if coverage == 1 else "unresolved",
    }


def _score_population(
    rows: _Population,
    gold: _Gold,
    systems: dict[str, _System],
    settings: EvalSettings,
    *,
    detail: bool,
    probabilities: dict[str, _Probabilities] | None = None,
) -> dict[str, Any]:
    """The primary read over the shared (no-overflow) rows, and each system's full-tier read."""
    shared = [i for i in rows if all(s.status[i] != "overflow" for s in systems.values())]
    declared = _declared(rows, gold, settings)
    clusters = [gold.clusters[i] for i in shared]
    # Absence is the GOLD's alone: the same classes for every system, whatever it predicted.
    present = {gold.labels[i] for i in shared}
    absent = [label for label in declared if label not in present]
    out: dict[str, Any] = {
        "n": len(shared),
        "n_tier": len(rows),
        "overflow_excluded": len(rows) - len(shared),
        "overflow_share": (len(rows) - len(shared)) / len(rows),
        "clusters": len(set(clusters)),
        # Rows clustered on the premise-hash fallback (no cluster value); 0 on a suite tier.
        "fallback_keyed": sum(1 for i in rows if gold.fallback[i]),
        "diagnostic": len(shared) < settings.min_slice_rows,
        "labels": list(declared),
        "absent_classes": absent,
        "systems": {},
        "comparisons": [],
    }
    for name, system in systems.items():
        entry: dict[str, Any] = {}
        if shared:
            entry, out["bootstrap"] = _system_read(
                shared, gold, system, settings, detail=detail, declared=declared
            )
        full_tier = entry
        if len(shared) != len(rows):
            full_tier, _ = _system_read(
                rows, gold, system, settings, detail=False, declared=declared
            )
        out["systems"][name] = {
            **entry,
            "full_tier": _metric_only(full_tier),
            # A diagnostic, never gated: how often this system predicted a class the gold lacks.
            "absent_class_predictions": {
                label: sum(1 for i in shared if system.predictions[i] == label) for label in absent
            },
        }
        if probabilities is not None and settings.calibration is not None:
            bins = settings.calibration.ece_bins
            probs = probabilities[name]
            for read, values in (
                ("raw", probs.raw),
                ("cross_fit", probs.cross_fit),
                ("applied", probs.applied),
            ):
                out["systems"][name][read] = _probability_block(
                    shared, gold, values, settings, bins
                )
    candidate = settings.predictions.name
    for baseline in settings.compare_to if shared else ():
        paired = paired_cluster_bootstrap(
            [gold.labels[i] for i in shared],
            [systems[baseline.name].predictions[i] for i in shared],
            [systems[candidate].predictions[i] for i in shared],
            labels=declared,
            clusters=clusters,
            seed=settings.seed,
            resamples=settings.resamples,
        )
        out["comparisons"].append(
            {
                "baseline": baseline.name,
                "candidate": candidate,
                "delta": _intervals(paired.delta, settings, delta=True),
                "discordance": _discordance(
                    shared,
                    gold,
                    systems[baseline.name],
                    systems[candidate],
                    None
                    if probabilities is None
                    else (probabilities[baseline.name], probabilities[candidate]),
                ),
                "brier_delta": None
                if probabilities is None
                else _brier_delta(
                    shared,
                    gold,
                    probabilities[baseline.name],
                    probabilities[candidate],
                    settings,
                ),
            }
        )
    return out


def _empty_population(settings: EvalSettings) -> dict[str, Any]:
    """A slice every one of whose rows the tier rule refused: counted, with nothing to score."""
    del settings
    return {
        "n": 0,
        "n_tier": 0,
        "overflow_excluded": 0,
        "overflow_share": 0.0,
        "clusters": 0,
        "fallback_keyed": 0,
        "diagnostic": True,
        "labels": [],
        "absent_classes": [],
        "systems": {},
        "comparisons": [],
    }


def _metric_only(entry: dict[str, Any]) -> dict[str, Any]:
    return {key: entry[key] for key in ("accuracy", "macro_f1", "invalid_rate") if key in entry}


def _headline(report: dict[str, Any], settings: EvalSettings) -> dict[str, float]:
    """The overall figures the ``end`` event carries (well under its 64-key cap)."""
    overall = report["slices"][0]
    candidate = overall["systems"][settings.predictions.name]
    if "accuracy" not in candidate:
        return {"rows": 0.0, "clusters": 0.0}
    metrics: dict[str, float] = {
        "rows": float(overall["n"]),
        "clusters": float(overall["clusters"]),
    }
    for name in ("accuracy", "macro_f1", "invalid_rate"):
        _put_interval(metrics, name, candidate[name])
    # Only macro-F1 can be undefined in a resample; its count rides beside the bounds.
    # Its share (undefined over all resamples) rides beside it, so a reader of the event alone can
    # tell an unstable interval (over ``unstable_threshold``) from a defined one.
    metrics["macro_f1_undefined_share"] = float(candidate["macro_f1"]["undefined_share"])
    raw = candidate.get("raw")
    if isinstance(raw, dict):
        block = cast("dict[str, Any]", raw)
        metrics["brier_coverage"] = float(block["coverage"])
        # Under full coverage only: a read over part of the rows is unresolved, so no bounds.
        brier_read = (
            block["brier"] if block["coverage"] == 1 else {**block["brier"], "unstable": True}
        )
        _put_interval(metrics, "brier", brier_read)
    for comparison in overall["comparisons"]:
        for name in ("accuracy", "macro_f1"):
            delta = comparison["delta"][name]
            if delta["point"] is None:
                metrics[f"delta_{name}_undefined__{comparison['baseline']}"] = 1.0
                continue
            metrics[f"delta_{name}__{comparison['baseline']}"] = delta["point"]
            for part in ("lower", "upper"):
                if delta[part] is not None and not delta["unstable"]:
                    metrics[f"delta_{name}_{part}__{comparison['baseline']}"] = delta[part]
    return metrics


def _put_interval(metrics: dict[str, float], name: str, interval: dict[str, Any]) -> None:
    """A point and, when any resample defined it, its bounds: an event metric is a number. A
    metric with no value (a declared class absent) is ``<name>_undefined: 1`` instead."""
    if interval["point"] is None:
        metrics[f"{name}_undefined"] = 1.0
        return
    metrics[name] = interval["point"]
    if interval["lower"] is not None and not interval.get("unstable", False):
        metrics[f"{name}_lower"] = interval["lower"]
        metrics[f"{name}_upper"] = interval["upper"]


# ------------------------------------------------------------------ writing


def _items_rows(gold: _Gold, systems: dict[str, _System]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i, item in enumerate(gold.ids):
        row: dict[str, Any] = {
            "item_id": item,
            "gold": gold.labels[i],
            "cluster": gold.clusters[i],
            "excluded": gold.excluded[i],
            "tier_refused": gold.refused[i],
        }
        for column, values in gold.slices.items():
            row[f"slice__{column}"] = values[i]
        for name, system in systems.items():
            prediction = system.predictions[i]
            row[f"pred__{name}"] = prediction
            row[f"status__{name}"] = system.status[i]
            row[f"correct__{name}"] = None if gold.common_out(i) else prediction == gold.labels[i]
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
    overall = report["slices"][0]
    write_manifest(_build_manifest(spec, gold, systems, overall.get("bootstrap")), outdir)
    return {
        ITEMS_FILENAME: items,
        REPORT_FILENAME: outdir / REPORT_FILENAME,
        MANIFEST_FILENAME: outdir / MANIFEST_FILENAME,
        _CARD_FILENAME: outdir / _CARD_FILENAME,
    }


def _build_manifest(
    spec: EvalSpec,
    gold: _Gold,
    systems: dict[str, _System],
    bootstrap: dict[str, Any] | None,
) -> dict[str, Any]:
    """Provenance; ``bootstrap`` is the overall read's contract, NumPy version and cluster count."""
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
            "tier_refused": sum(gold.refused),
            "systems": {name: system.counts for name, system in systems.items()},
        },
        "bootstrap": bootstrap,
    }


def _calibration_report(
    spec: EvalSpec, probabilities: dict[str, _Probabilities] | None
) -> dict[str, Any] | None:
    calibration = spec.evaluation.calibration
    if calibration is None or probabilities is None:
        return None
    return {
        "fit_on": {
            "split": spec.split,
            "column": calibration.fit_column,
            "value": calibration.fit_value,
        },
        "folds": calibration.folds,
        "fold_rule": "seeded permutation of sorted cluster keys; the first ceil(C/2) are fold 0",
        "ece_bins": calibration.ece_bins,
        "ece_scheme": "equal_mass",
        "apply_calibration": calibration.apply_calibration,
        "systems": {
            name: {
                "cross_fit": None
                if probs.fit is None
                else {
                    "temperatures": [fit.model_dump() for fit in probs.fit.folds],
                    "fold_rows": list(probs.fit.fold_rows),
                    "cross_fit_version": probs.fit.cross_fit_version,
                },
                "own": probs.own,
            }
            for name, probs in probabilities.items()
        },
    }


def _report(
    spec: EvalSpec,
    gold: _Gold,
    systems: dict[str, _System],
    slices: list[dict[str, Any]],
    probabilities: dict[str, _Probabilities] | None = None,
) -> dict[str, Any]:
    settings = spec.evaluation
    return {
        "calibration": _calibration_report(spec, probabilities),
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
            "tier_refused": sum(gold.refused),
            "fallback_keyed": sum(gold.fallback),
            "token_methods": gold.token_methods,
            "unstable_threshold": settings.unstable_threshold,
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
    commit = await hub.upload_files(
        dict(files),
        out_repo,
        repo_type="dataset",
        commit_message="evaluation report",
    )
    # ``repo@commit``, so the report is read back at the commit this run made, never at a head
    # another run may have moved since.
    return f"{out_repo}@{commit}" if COMMIT_SHA_RE.fullmatch(commit) else out_repo


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
            logit_columns = _logit_columns(settings)
            raw = await _fetch_predictions(source, hub_token, logit_columns)
            systems[source.name] = await asyncio.to_thread(
                _align, gold, raw, source, settings.labels, logit_columns
            )
    probabilities: dict[str, _Probabilities] | None = None
    if settings.calibration is not None:
        calibration = settings.calibration
        own: dict[str, dict[str, Any] | None] = {}
        for source in settings.systems:
            apply = calibration.apply_calibration == "auto" and source.calibration_ref is not None
            own[source.name] = (
                await _fetch_calibration(source, settings.labels, hub_token) if apply else None
            )
        async with ticking_phase(phase, "Fitting temperatures", stage="run"):
            probabilities = {}
            for name, system in systems.items():
                probabilities[name] = await asyncio.to_thread(
                    _probabilities, gold, system, settings, calibration, own[name]
                )

    populations = _populations(gold)
    if not populations[0][2]:  # also true when the tier rule refused every row
        msg = "every gold row is excluded: there is nothing to score"
        raise RunError(msg)
    emit(writer, ProgressEvent(kind="start", stage="run", total_steps=len(populations)))
    slices: list[dict[str, Any]] = []
    async with ticking_phase(phase, "Computing metrics", stage="run"):
        for step, (column, value, rows, refused) in enumerate(populations, start=1):
            scored: dict[str, Any] = (
                await asyncio.to_thread(
                    partial(
                        _score_population,
                        rows,
                        gold,
                        systems,
                        settings,
                        detail=column is None,
                        probabilities=probabilities,
                    )
                )
                if rows
                else _empty_population(settings)
            )
            slices.append({"column": column, "value": value, "tier_refused": refused, **scored})
            emit(
                writer,
                ProgressEvent(kind="step", stage="run", step=step, total_steps=len(populations)),
            )
    report = _report(spec, gold, systems, slices, probabilities)

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
