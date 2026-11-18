"""The LLM grader: a pinned judge model scores each system's outputs against a rubric.

A graded evaluation reads a pinned gold split (the cases: their inputs, ids, clusters and slices)
and, per system, a pinned output file (a batch run's ``results.parquet``, or any parquet with an
output column). Each case's output is joined to its gold row by id, or by position (``row-<i>``,
the inference runner's ``custom_id``) when no id column is declared; a positional join is
admitted only when the output file's own ``manifest.json`` names the same dataset, config, split
and commit, so a file from another split can never be graded against these cases.

The judge is the inference runner's ``vllm_chat`` engine, run in this process on the user's own
machine with the judge model pinned to a full commit: one request per (case, system) with the
rubric's template rendered over the case's gold columns, the output (``{output}``) and the
rendered criteria (``{criteria}``), constrained to the rubric's guided-JSON schema with the one
repair turn that engine gives an invalid reply. Temperature is 0 unless the run says otherwise.

Every (case, system) gets a status: ``graded``, or unscored with a reason (``output_missing``,
``judge_error``, ``judge_invalid``). Unscored cases stay visible rows, are counted per reason and
in each read's coverage, and are never a pass. A read below full coverage is ``unresolved`` and
decides no gate. Tier rules are not applied: a grader touches no research gate tier.

Aggregates use the cluster bootstrap the evaluation runner uses: per criterion, system and slice,
the mean score (the pass rate on a pass/fail scale, plus the pass rate at ``pass_at`` on an
integer scale that declares one); per baseline, the paired candidate-minus-baseline delta on the
cases both systems had graded, with pass/fail transitions. A criterion with a ``gold_column``
adds the judge's agreement with those gold scores
(:func:`strata_forge.metrics.judge_agreement.judge_agreement`).

Outputs: ``items.parquet`` (one row per case: id, cluster, slice labels and, per system, the
status and each score; never any input, output or rationale text), ``rationales.parquet`` (the
judge's rationale per graded case and system, capped at 2,000 characters; a rationale may quote
the output it grades, so it is kept apart), ``metrics.json`` (``strata.eval.grades/v1``),
``manifest.json`` and a dataset card, pushed in one commit to a private output repo. The ``end``
event's message is ``<repo>@<commit>``.

Ids are joined type-exactly (the integer 1 never joins the string "1") and copied into the items
file as declared. A slice column is a declared label: at most 64 values of at most 64 characters,
else the run refuses it, because its values are copied into the report.
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator, model_validator

from strata_forge.metrics.cluster_bootstrap import (
    CLUSTER_BOOTSTRAP_RESAMPLES,
    MAX_CLUSTER_BOOTSTRAP_RESAMPLES,
    MAX_CLUSTER_BOOTSTRAP_SEED,
    cluster_mean_bootstrap,
    paired_cluster_mean_bootstrap,
)
from strata_forge.metrics.judge_agreement import judge_agreement
from strata_forge.pipelines import inference_runner
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
from strata_forge.pipelines._grading import (
    CRITERIA_PLACEHOLDER,
    GRADE_SCHEMA_VERSION,
    MAX_RATIONALE_CHARS,
    OUTPUT_PLACEHOLDER,
    Criterion,
    Rubric,
    parse_grade,
    render_criteria,
    rubric_digest,
    rubric_schema,
)
from strata_forge.pipelines._guided import schema_digest
from strata_forge.storage import HFHubClient
from strata_forge.training.progress import JsonlProgressWriter, ProgressEvent

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = [
    "GRADES_SCHEMA",
    "ITEMS_FILENAME",
    "RATIONALES_FILENAME",
    "REPORT_FILENAME",
    "GradeSettings",
    "GradeSpec",
    "GradedOutput",
    "judge_hyperparams",
    "main",
]

GRADES_SCHEMA = "strata.eval.grades/v1"
ITEMS_FILENAME = "items.parquet"
RATIONALES_FILENAME = "rationales.parquet"
REPORT_FILENAME = "metrics.json"
_CARD_FILENAME = "README.md"
_MANIFEST_KIND = "graded_evaluation"
_RESULTS_DIR_NAME = "strata-grade-results"
_RECORDED_DISTRIBUTIONS = ("numpy", "pyarrow", "datasets", "huggingface-hub", "vllm")
_MAX_OUTPUTS = 4
_MAX_SLICE_COLUMNS = 8
#: A slice column is a declared low-cardinality label: at most this many values, each at most
#: ``_MAX_SLICE_CHARS`` long, copied into the report and the items file as they are.
_MAX_SLICE_VALUES = 64
_MAX_SLICE_CHARS = 64
_MAX_CASES = 100_000
_MAX_TEXT = 128
_MAX_IDENTIFIER_CHARS = 256
_MAX_MANIFEST_BYTES = 256 * 1024
_HEADLINE_KEYS = 64
#: The two columns the judge's rows carry beside the case's gold columns.
_OUTPUT_COLUMN = "__strata_output"
_CRITERIA_COLUMN = "__strata_criteria"
#: The judge's settings the grader owns: its engine and answer schema, one answer per case, and
#: every scoring-only knob. A run may tune the rest (max_tokens, temperature, serving).
_JUDGE_RESERVED = frozenset(
    {
        "engine",
        "guided_json",
        "n",
        "row_limit",
        "pair_template",
        "label_alias",
        "artifact",
        "apply_calibration",
        "label_tokens",
        "assistant_prefill",
        "id_column",
        "slice_column",
        "scoring_batch_size",
        "list_separator",
    }
)
_PATH_RE = re.compile(
    r"\A[A-Za-z0-9_-][A-Za-z0-9._-]*(?:/[A-Za-z0-9_-][A-Za-z0-9._-]*)*\.parquet\Z"
)
_POSITION_RE = re.compile(r"\Arow-(0|[1-9][0-9]*)\Z")

type Status = Literal["graded", "output_missing", "judge_error", "judge_invalid"]
UNSCORED: tuple[Status, ...] = ("output_missing", "judge_error", "judge_invalid")


def _column(value: str, what: str = "a column name") -> str:
    if not value or len(value) > _MAX_TEXT or any(ord(c) < 32 for c in value):
        msg = f"{what} must be 1-{_MAX_TEXT} printable characters"
        raise ValueError(msg)
    return value


class GradedOutput(BaseModel):
    """One system's outputs: a parquet file in a dataset repo at a pinned commit."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1, max_length=32)
    repo_id: str = Field(min_length=1, max_length=255)
    commit_sha: str
    path: str = Field(default="results.parquet", max_length=255)
    #: The column joined to the gold ``id_column``; null joins by position (``custom_id``).
    id_column: str | None = None
    output_column: str = "output"
    #: A non-null value marks a row whose generation failed: its output is missing, not graded.
    error_column: str | None = "error"

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

    @field_validator("id_column", "output_column", "error_column")
    @classmethod
    def _columns(cls, value: str | None) -> str | None:
        return None if value is None else _column(value)


class GradeSettings(BaseModel):
    """What to grade and how; validated identically by the control plane and the runner."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    rubric: Rubric
    judge_model_id: str = Field(min_length=1, max_length=255)
    judge_revision: str
    judge_hyperparams: dict[str, JsonValue] = Field(default_factory=dict[str, JsonValue])
    #: Rubric placeholder -> gold column, for every placeholder but ``{output}`` / ``{criteria}``.
    column_mapping: dict[str, str] = Field(default_factory=dict[str, str])
    #: The candidate first, then every baseline it is compared to.
    outputs: tuple[GradedOutput, ...] = Field(min_length=1, max_length=_MAX_OUTPUTS)
    #: The gold column the outputs' ``id_column`` joins to; null joins every output by position.
    id_column: str | None = None
    #: Cases sharing a value are resampled together; null makes every case its own cluster.
    cluster_column: str | None = None
    slice_columns: tuple[str, ...] = Field(default=(), max_length=_MAX_SLICE_COLUMNS)
    seed: int = Field(ge=0, le=MAX_CLUSTER_BOOTSTRAP_SEED)
    resamples: int = Field(
        default=CLUSTER_BOOTSTRAP_RESAMPLES, ge=1, le=MAX_CLUSTER_BOOTSTRAP_RESAMPLES
    )
    min_slice_rows: int = Field(default=500, ge=1)

    @field_validator("judge_revision")
    @classmethod
    def _pinned(cls, value: str) -> str:
        if not COMMIT_SHA_RE.fullmatch(value):
            msg = "judge_revision must be a full 40-character lowercase commit id"
            raise ValueError(msg)
        return value

    @field_validator("id_column", "cluster_column")
    @classmethod
    def _columns(cls, value: str | None) -> str | None:
        return None if value is None else _column(value)

    @field_validator("slice_columns")
    @classmethod
    def _slices(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(set(value)) != len(value):
            msg = "slice columns must be unique"
            raise ValueError(msg)
        return tuple(_column(column) for column in value)

    @model_validator(mode="after")
    def _consistent(self) -> GradeSettings:
        names = [output.name for output in self.outputs]
        if len(set(names)) != len(names):
            msg = "every output needs a distinct name"
            raise ValueError(msg)
        reserved = {OUTPUT_PLACEHOLDER, CRITERIA_PLACEHOLDER} & set(self.column_mapping)
        if reserved:
            msg = f"{sorted(reserved)} are filled by the grader and cannot be mapped to a column"
            raise ValueError(msg)
        wanted = self.rubric.column_placeholders()
        if set(self.column_mapping) != wanted:
            msg = (
                f"column_mapping must map exactly the rubric's column placeholders {sorted(wanted)}; "
                f"it maps {sorted(self.column_mapping)}"
            )
            raise ValueError(msg)
        for column in self.column_mapping.values():
            _column(column)
        positional = [o.name for o in self.outputs if o.id_column is None]
        if self.id_column is None and len(positional) != len(self.outputs):
            msg = "with no gold id_column every output joins by position, so none may set id_column"
            raise ValueError(msg)
        if self.id_column is not None and positional:
            msg = f"outputs {positional} need an id_column to join the gold id_column"
            raise ValueError(msg)
        judge_hyperparams(self)
        return self

    @property
    def candidate(self) -> GradedOutput:
        return self.outputs[0]


def judge_hyperparams(settings: GradeSettings) -> inference_runner.Hyperparams:
    """The judge's chat-run settings: the run's own, under the grader's engine and schema."""
    given = settings.judge_hyperparams
    taken = sorted(_JUDGE_RESERVED & set(given))
    if taken:
        msg = f"judge_hyperparams may not set {taken}: the grader sets them"
        raise ValueError(msg)
    try:
        return inference_runner.Hyperparams.model_validate(
            {
                "temperature": 0.0,
                **given,
                "engine": "vllm_chat",
                "guided_json": rubric_schema(settings.rubric),
            }
        )
    except ValueError as exc:
        msg = f"judge_hyperparams: {exc}"
        raise ValueError(msg) from exc


class GradeSpec(BaseModel):
    """The inert graded-evaluation spec the server delivers in ``STRATA_RUN_CONFIG``."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    engine_version: str | None = None
    dataset_id: str
    dataset_commit_sha: str
    dataset_config: str | None = None
    split: str
    grading: GradeSettings
    output_repo_id: str | None = None
    progress_path: str | None = None
    run_id: str | None = None


def load_spec(*, writer: JsonlProgressWriter | None = None) -> GradeSpec:
    spec = load_config(GradeSpec, writer=writer)
    validate_repo_id(spec.dataset_id, "dataset")
    validate_repo_id(spec.grading.judge_model_id, "judge model")
    if not COMMIT_SHA_RE.fullmatch(spec.dataset_commit_sha):
        msg = "dataset_commit_sha must be a full 40-character lowercase commit id"
        raise RunError(msg)
    for output in spec.grading.outputs:
        validate_repo_id(output.repo_id, "outputs")
    if spec.output_repo_id is not None:
        validate_repo_id(spec.output_repo_id, "output repo")
    return spec


# ------------------------------------------------------------------ the cases


class _Cases(BaseModel):
    """The gold split's cases, in row order: ids, clusters, slices and the judge's columns."""

    model_config = ConfigDict(frozen=True)

    ids: list[str]
    #: The join key of each id: its text tagged with its type, so the integer 1 never joins the
    #: string "1".
    keys: list[str]
    clusters: list[str]
    slices: dict[str, list[str]]
    columns: dict[str, list[Any]]


def _gold_columns(settings: GradeSettings) -> list[str]:
    wanted = [settings.id_column, settings.cluster_column, *settings.slice_columns]
    wanted.extend(settings.column_mapping.values())
    return list(dict.fromkeys(c for c in wanted if c is not None))


def _load_gold(spec: GradeSpec, token: str | Literal[False]) -> dict[str, list[Any]]:
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
    wanted = _gold_columns(spec.grading)
    missing = sorted(set(wanted) - set(dataset.column_names or []))
    if missing:
        msg = f"the gold split has no column(s): {missing}"
        raise RunError(msg)
    if len(dataset) > _MAX_CASES:
        msg = f"the gold split has more than {_MAX_CASES} rows"
        raise RunError(msg)
    if not wanted:
        return {"__rows": [None] * len(dataset)}
    return cast("dict[str, list[Any]]", dataset.select_columns(wanted).to_dict())


def _identifier(value: object) -> str | None:
    if isinstance(value, bool) or not isinstance(value, str | int):
        return None
    text = str(value)
    return text if text and len(text) <= _MAX_IDENTIFIER_CHARS else None


def _id_key(value: object, text: str) -> str:
    return f"{'int' if isinstance(value, int) else 'str'}:{text}"


def _unique_ids(values: Sequence[object], what: str) -> tuple[list[str], list[str]]:
    """Each id's text (copied as declared) and its type-exact join key, refusing a repeat."""
    ids: list[str] = []
    keys: list[str] = []
    seen: set[str] = set()
    for index, value in enumerate(values):
        text = _identifier(value)
        if text is None:
            msg = f"{what}: row {index} has no usable id (a non-empty string or integer)"
            raise RunError(msg)
        key = _id_key(value, text)
        if key in seen:
            msg = f"{what} repeats the id {text[:64]!r}; ids must be unique"
            raise RunError(msg)
        seen.add(key)
        ids.append(text)
        keys.append(key)
    return ids, keys


def _prepare_cases(columns: dict[str, list[Any]], settings: GradeSettings) -> _Cases:
    n = len(next(iter(columns.values()))) if columns else 0
    if not n:
        msg = "the gold split is empty: there is nothing to grade"
        raise RunError(msg)
    if settings.id_column is not None:
        ids, keys = _unique_ids(columns[settings.id_column], f"id_column {settings.id_column!r}")
    else:
        ids = [f"row-{i}" for i in range(n)]
        keys = [_id_key(case, case) for case in ids]
    clusters: list[str] = keys
    if settings.cluster_column is not None:
        clusters = []
        for index, value in enumerate(columns[settings.cluster_column]):
            text = _identifier(value)
            if text is None:
                msg = (
                    f"cluster_column {settings.cluster_column!r}: row {index} has no cluster value"
                )
                raise RunError(msg)
            clusters.append(text)
    slices: dict[str, list[str]] = {}
    for column in settings.slice_columns:
        values = ["(null)" if v is None else str(v) for v in columns[column]]
        distinct = set(values)
        if len(distinct) > _MAX_SLICE_VALUES or any(len(v) > _MAX_SLICE_CHARS for v in distinct):
            msg = (
                f"slice column {column!r} is not a slice label: it needs at most "
                f"{_MAX_SLICE_VALUES} values of at most {_MAX_SLICE_CHARS} characters, since "
                "its values are copied into the report"
            )
            raise RunError(msg)
        slices[column] = values
    mapped = {column: columns[column] for column in settings.column_mapping.values()}
    return _Cases(ids=ids, keys=keys, clusters=clusters, slices=slices, columns=mapped)


# ------------------------------------------------------------------ the outputs


class _Outputs(BaseModel):
    """One system's outputs and gold scores, aligned to the cases (None where it has none)."""

    model_config = ConfigDict(frozen=True)

    texts: list[str | None]
    gold: dict[str, list[int | None]]
    join: Literal["id", "position"]
    counts: dict[str, int]


def _read_parquet(path: Path, columns: list[str], who: str) -> dict[str, list[Any]]:
    parquet: Any = __import__("pyarrow.parquet", fromlist=["read_schema"])  # pyright: ignore[reportAny] - optional extra
    present = set(parquet.read_schema(str(path)).names)
    missing = sorted(set(columns) - present)
    if missing:
        msg = f"the outputs of {who} have no column(s): {missing}"
        raise RunError(msg)
    if "sample_index" in present:
        msg = f"the outputs of {who} hold several samples per row; grade a one-sample run"
        raise RunError(msg)
    table = parquet.read_table(str(path), columns=columns)
    if table.num_rows > _MAX_CASES:
        msg = f"the outputs of {who} have more than {_MAX_CASES} rows"
        raise RunError(msg)
    return cast("dict[str, list[Any]]", table.to_pydict())


async def _download(output: GradedOutput, path: str, token: str | Literal[False]) -> Path:
    hub = HFHubClient(token=token)  # explicit: never forge settings or the VM's own login
    try:
        return await hub.download_file(
            output.repo_id, path, repo_type="dataset", revision=output.commit_sha
        )
    except Exception as exc:
        msg = f"could not download {path} for {output.name}: {exc}"
        raise RunError(msg) from exc


async def _check_position_pins(
    output: GradedOutput, spec: GradeSpec, token: str | Literal[False]
) -> None:
    """A positional join needs the output file's manifest to name THIS split at THIS commit."""
    directory = output.path.rpartition("/")[0]
    path = f"{directory}/{MANIFEST_FILENAME}" if directory else MANIFEST_FILENAME
    local = await _download(output, path, token)
    raw = await asyncio.to_thread(local.read_bytes)
    try:
        manifest: object = json.loads(raw) if len(raw) <= _MAX_MANIFEST_BYTES else None
    except ValueError:
        manifest = None
    dataset = (
        cast("dict[str, object]", manifest).get("dataset") if isinstance(manifest, dict) else None
    )
    if not isinstance(dataset, dict):
        msg = f"the outputs of {output.name} have no readable manifest, so they cannot join by position"
        raise RunError(msg)
    pins = cast("dict[str, object]", dataset)
    commit = pins.get("commit") or pins.get("requested_revision")
    want = (spec.dataset_id, spec.dataset_config, spec.split, spec.dataset_commit_sha)
    have = (pins.get("id"), pins.get("config"), pins.get("split"), commit)
    if have != want:
        msg = (
            f"the outputs of {output.name} were generated from {have[0]}/{have[2]} at {have[3]}, "
            f"not the graded split {want[0]}/{want[2]} at {want[3]}: a positional join would grade "
            "outputs against the wrong cases"
        )
        raise RunError(msg)


def _gold_score(value: object, criterion: Criterion, who: str) -> int | None:
    if value is None:
        return None
    if criterion.scale == "pass_fail" and isinstance(value, bool):
        return int(value)
    if isinstance(value, int) and not isinstance(value, bool) and value in criterion.values():
        return value
    msg = f"{who}: gold score {value!r} for {criterion.id} is not on its scale"
    raise RunError(msg)


def _align_outputs(
    raw: dict[str, list[Any]], output: GradedOutput, cases: _Cases, rubric: Rubric
) -> _Outputs:
    column = output.id_column or "custom_id"
    ids, keys = _unique_ids(raw[column], f"the outputs of {output.name}")
    position = {key: index for index, key in enumerate(cases.keys)}
    texts: list[str | None] = [None] * len(cases.ids)
    gold: dict[str, list[int | None]] = {
        c.id: [None] * len(cases.ids) for c in rubric.criteria if c.gold_column
    }
    unmatched = failed = 0
    for row, item in enumerate(ids):
        if output.id_column is None and not _POSITION_RE.fullmatch(item):
            msg = f"the outputs of {output.name} have custom_id {item[:64]!r}, not row-<index>"
            raise RunError(msg)
        index = position.get(keys[row])
        if index is None:
            unmatched += 1
            continue
        for criterion in rubric.criteria:
            if criterion.gold_column:
                gold[criterion.id][index] = _gold_score(
                    raw[criterion.gold_column][row], criterion, f"the outputs of {output.name}"
                )
        error = raw[output.error_column][row] if output.error_column else None
        text = raw[output.output_column][row]
        if error is not None or not isinstance(text, str) or not text:
            failed += 1
            continue
        texts[index] = text
    return _Outputs(
        texts=texts,
        gold=gold,
        join="position" if output.id_column is None else "id",
        counts={
            "rows": len(ids),
            "matched": len(ids) - unmatched,
            "unmatched": unmatched,
            "failed_or_empty": failed,
            "with_output": sum(1 for t in texts if t is not None),
        },
    )


async def _fetch_outputs(
    output: GradedOutput, spec: GradeSpec, cases: _Cases, token: str | Literal[False]
) -> _Outputs:
    rubric = spec.grading.rubric
    if output.id_column is None:
        await _check_position_pins(output, spec, token)
    local = await _download(output, output.path, token)
    columns = [output.id_column or "custom_id", output.output_column]
    if output.error_column:
        columns.append(output.error_column)
    columns.extend(c.gold_column for c in rubric.criteria if c.gold_column)
    raw = await asyncio.to_thread(_read_parquet, local, list(dict.fromkeys(columns)), output.name)
    return await asyncio.to_thread(_align_outputs, raw, output, cases, rubric)


# ------------------------------------------------------------------ the judge


class _Judged(BaseModel):
    """Every (case, system)'s status, scores and rationale, from the judge's results."""

    model_config = ConfigDict(frozen=True)

    status: dict[str, list[Status]]
    scores: dict[str, list[dict[str, int] | None]]
    rationales: dict[str, list[str | None]]
    retried: dict[str, list[bool]]


class _JudgeProgress(JsonlProgressWriter):
    """The judge's progress, relayed into the grading run's own stream without its start and end:
    the judge is one step of this run, and its ``end`` would read as the run's."""

    def __init__(self, inner: JsonlProgressWriter | None) -> None:  # pyright: ignore[reportMissingSuperCall]
        self._inner = inner

    def emit(self, event: ProgressEvent) -> None:
        if self._inner is not None and event.kind not in ("start", "end"):
            self._inner.emit(event)

    def close(self) -> None:
        return None


def _judge_rows(
    cases: _Cases, outputs: dict[str, _Outputs], rubric: Rubric
) -> tuple[list[dict[str, Any]], list[tuple[int, str]]]:
    """One judge row per (case, system with an output), and which pair each row is."""
    criteria = render_criteria(rubric)
    rows: list[dict[str, Any]] = []
    pairs: list[tuple[int, str]] = []
    for index in range(len(cases.ids)):
        gold = {column: values[index] for column, values in cases.columns.items()}
        for name, output in outputs.items():
            text = output.texts[index]
            if text is None:
                continue
            rows.append({**gold, _OUTPUT_COLUMN: text, _CRITERIA_COLUMN: criteria})
            pairs.append((index, name))
    return rows, pairs


def _judge_spec(spec: GradeSpec) -> inference_runner.RunSpec:
    settings = spec.grading
    return inference_runner.RunSpec(
        engine_version=spec.engine_version,
        model_id=settings.judge_model_id,
        model_revision=settings.judge_revision,
        dataset_id=spec.dataset_id,
        dataset_commit_sha=spec.dataset_commit_sha,
        dataset_config=spec.dataset_config,
        split=spec.split,
        template=settings.rubric.template,
        column_mapping={
            **settings.column_mapping,
            OUTPUT_PLACEHOLDER: _OUTPUT_COLUMN,
            CRITERIA_PLACEHOLDER: _CRITERIA_COLUMN,
        },
        hyperparams=judge_hyperparams(settings),
        run_id=f"{spec.run_id}-judge" if spec.run_id else None,
    )


def _read_judged(
    path: Path,
    pairs: list[tuple[int, str]],
    cases: _Cases,
    outputs: dict[str, _Outputs],
    rubric: Rubric,
) -> _Judged:
    parquet: Any = __import__("pyarrow.parquet", fromlist=["read_table"])  # pyright: ignore[reportAny] - optional extra
    table: dict[str, list[Any]] = parquet.read_table(str(path)).to_pydict()
    n = len(cases.ids)
    status: dict[str, list[Status]] = {name: ["output_missing"] * n for name in outputs}
    scores: dict[str, list[dict[str, int] | None]] = {name: [None] * n for name in outputs}
    rationales: dict[str, list[str | None]] = {name: [None] * n for name in outputs}
    retried: dict[str, list[bool]] = {name: [False] * n for name in outputs}
    valid = table.get("json_valid", [None] * len(table["custom_id"]))
    again = table.get("json_retried", [False] * len(table["custom_id"]))
    seen: set[int] = set()
    for row, custom_id in enumerate(table["custom_id"]):
        match = _POSITION_RE.fullmatch(str(custom_id))
        position = int(match.group(1)) if match else -1
        if not 0 <= position < len(pairs) or position in seen:
            msg = f"the judge's results hold an unexpected row {custom_id!r}"
            raise RunError(msg)
        seen.add(position)
        index, name = pairs[position]
        retried[name][index] = bool(again[row])
        if table["error"][row] is not None:
            status[name][index] = "judge_error"
            continue
        grade = parse_grade(table["output"][row], rubric) if valid[row] is not False else None
        if grade is None:
            status[name][index] = "judge_invalid"
            continue
        status[name][index] = "graded"
        scores[name][index] = grade.scores
        rationales[name][index] = grade.rationale
    if len(seen) != len(pairs):
        msg = f"the judge answered {len(seen)} of {len(pairs)} cases"
        raise RunError(msg)
    return _Judged(status=status, scores=scores, rationales=rationales, retried=retried)


# ------------------------------------------------------------------ the report


def _interval(values: list[float], clusters: list[str], settings: GradeSettings) -> dict[str, Any]:
    read = cluster_mean_bootstrap(
        values, clusters=clusters, seed=settings.seed, resamples=settings.resamples
    )
    return read.interval.model_dump()


def _populations(cases: _Cases) -> list[tuple[str | None, str | None, list[int]]]:
    out: list[tuple[str | None, str | None, list[int]]] = [
        (None, None, list(range(len(cases.ids))))
    ]
    for column, values in cases.slices.items():
        members: dict[str, list[int]] = {}
        for index, value in enumerate(values):
            members.setdefault(value, []).append(index)
        out.extend((column, value, members[value]) for value in sorted(members))
    return out


def _system_read(
    rows: list[int], cases: _Cases, judged: _Judged, name: str, settings: GradeSettings
) -> dict[str, Any]:
    graded = [i for i in rows if judged.status[name][i] == "graded"]
    coverage = len(graded) / len(rows)
    out: dict[str, Any] = {
        "n": len(rows),
        "graded": len(graded),
        "coverage": coverage,
        # A read over part of the cases decides no gate (unresolved counts as not met).
        "gate": "resolved" if coverage == 1 else "unresolved",
        "unscored": {
            reason: sum(1 for i in rows if judged.status[name][i] == reason) for reason in UNSCORED
        },
        "criteria": {},
    }
    clusters = [cases.clusters[i] for i in graded]
    for criterion in settings.rubric.criteria:
        if not graded:
            out["criteria"][criterion.id] = {
                "mean": None,
                "pass_rate": None,
                "reason": "no_graded_cases",
            }
            continue
        values = [float((judged.scores[name][i] or {})[criterion.id]) for i in graded]
        entry: dict[str, Any] = {"mean": _interval(values, clusters, settings), "pass_rate": None}
        if criterion.scale == "ordinal" and criterion.pass_at is not None:
            passes = [float(v >= criterion.pass_at) for v in values]
            entry["pass_rate"] = _interval(passes, clusters, settings)
        out["criteria"][criterion.id] = entry
    return out


def _excludes_zero(delta: dict[str, Any]) -> bool | Literal["unresolved"]:
    if delta["lower"] is None or delta["upper"] is None:
        return "unresolved"
    return bool(delta["lower"] > 0 or delta["upper"] < 0)


def _comparison(
    rows: list[int],
    cases: _Cases,
    judged: _Judged,
    baseline: str,
    candidate: str,
    settings: GradeSettings,
) -> dict[str, Any]:
    shared = [
        i
        for i in rows
        if judged.status[baseline][i] == "graded" and judged.status[candidate][i] == "graded"
    ]
    coverage = len(shared) / len(rows)
    out: dict[str, Any] = {
        "baseline": baseline,
        "candidate": candidate,
        "n": len(shared),
        "coverage": coverage,
        "gate": "resolved" if coverage == 1 else "unresolved",
        "criteria": {},
    }
    clusters = [cases.clusters[i] for i in shared]
    for criterion in settings.rubric.criteria:
        if len(shared) < 2:
            out["criteria"][criterion.id] = {"delta": None, "reason": "fewer_than_two_shared_cases"}
            continue
        base = [float((judged.scores[baseline][i] or {})[criterion.id]) for i in shared]
        cand = [float((judged.scores[candidate][i] or {})[criterion.id]) for i in shared]
        paired = paired_cluster_mean_bootstrap(
            base, cand, clusters=clusters, seed=settings.seed, resamples=settings.resamples
        )
        delta = paired.delta.model_dump()
        entry: dict[str, Any] = {"delta": {**delta, "excludes_zero": _excludes_zero(delta)}}
        if criterion.scale == "pass_fail" or criterion.pass_at is not None:
            moves: Counter[str] = Counter()
            for b, c in zip(base, cand, strict=True):
                was = criterion.passes(int(b))
                now = criterion.passes(int(c))
                moves[f"{'pass' if was else 'fail'}_to_{'pass' if now else 'fail'}"] += 1
            entry["transitions"] = {
                key: moves[key]
                for key in ("pass_to_fail", "fail_to_pass", "pass_to_pass", "fail_to_fail")
            }
        out["criteria"][criterion.id] = entry
    return out


def _agreement(
    cases: _Cases, outputs: dict[str, _Outputs], judged: _Judged, settings: GradeSettings
) -> dict[str, Any]:
    """Per system and criterion with a gold column: the judge against those gold scores."""
    out: dict[str, Any] = {}
    for name, output in outputs.items():
        reads: dict[str, Any] = {}
        for criterion in settings.rubric.criteria:
            if not criterion.gold_column:
                continue
            gold_scores = output.gold[criterion.id]
            with_gold = [i for i, g in enumerate(gold_scores) if g is not None]
            both = [i for i in with_gold if judged.status[name][i] == "graded"]
            gold = [cast("int", gold_scores[i]) for i in both]
            judge = [(judged.scores[name][i] or {})[criterion.id] for i in both]
            read: dict[str, Any] = judge_agreement(
                gold, judge, values=criterion.values(), scale=criterion.scale
            ).model_dump(mode="json")
            clusters = [cases.clusters[i] for i in both]
            read["n_gold"] = len(with_gold)
            read["coverage"] = len(both) / len(with_gold) if with_gold else None
            read["agreement_interval"] = (
                _interval(
                    [float(g == j) for g, j in zip(gold, judge, strict=True)], clusters, settings
                )
                if both
                else None
            )
            read["absolute_error_interval"] = (
                _interval(
                    [float(abs(g - j)) for g, j in zip(gold, judge, strict=True)],
                    clusters,
                    settings,
                )
                if both and criterion.scale == "ordinal"
                else None
            )
            reads[criterion.id] = read
        if reads:
            out[name] = reads
    return out


def _report(
    spec: GradeSpec,
    cases: _Cases,
    outputs: dict[str, _Outputs],
    judged: _Judged,
    judge_manifest: dict[str, Any],
) -> dict[str, Any]:
    settings = spec.grading
    names = [o.name for o in settings.outputs]
    slices: list[dict[str, Any]] = []
    for column, value, rows in _populations(cases):
        slices.append(
            {
                "column": column,
                "value": value,
                "n": len(rows),
                "clusters": len({cases.clusters[i] for i in rows}),
                "diagnostic": len(rows) < settings.min_slice_rows,
                "systems": {
                    name: _system_read(rows, cases, judged, name, settings) for name in names
                },
                "comparisons": [
                    _comparison(rows, cases, judged, baseline, settings.candidate.name, settings)
                    for baseline in names[1:]
                ],
            }
        )
    return {
        "schema": GRADES_SCHEMA,
        "grade_schema": GRADE_SCHEMA_VERSION,
        "candidate": settings.candidate.name,
        "baselines": names[1:],
        "dataset": {
            "id": spec.dataset_id,
            "config": spec.dataset_config,
            "split": spec.split,
            "commit": spec.dataset_commit_sha,
        },
        "rubric": {
            **settings.rubric.model_dump(mode="json"),
            "sha256": rubric_digest(settings.rubric),
            "answer_schema_sha256": schema_digest(rubric_schema(settings.rubric)),
        },
        "judge": {
            "model_id": settings.judge_model_id,
            "requested_revision": settings.judge_revision,
            "commit": cast("dict[str, Any]", judge_manifest.get("model") or {}).get("commit"),
            "hyperparams": judge_hyperparams(settings).model_dump(
                mode="json", exclude={"guided_json"}
            ),
            "serving": judge_manifest.get("serving"),
            "counts": judge_manifest.get("counts"),
        },
        "systems": [
            {
                "name": o.name,
                "repo_id": o.repo_id,
                "commit": o.commit_sha,
                "path": o.path,
                "join": outputs[o.name].join,
                "counts": outputs[o.name].counts,
                "judge_retried": sum(judged.retried[o.name]),
            }
            for o in settings.outputs
        ],
        "population": {
            "cases": len(cases.ids),
            "cluster_column": settings.cluster_column,
            "id_column": settings.id_column,
            "min_slice_rows": settings.min_slice_rows,
        },
        "bootstrap": {
            "seed": settings.seed,
            "resamples": settings.resamples,
            "unit": settings.cluster_column or "case",
        },
        "tier_rule": None,
        "slices": slices,
        "agreement": _agreement(cases, outputs, judged, settings),
    }


def _put(metrics: dict[str, float], key: str, value: float | None) -> None:
    if value is not None and len(metrics) < _HEADLINE_KEYS and len(key) <= _HEADLINE_KEYS:
        metrics[key] = float(value)


def _headline(report: dict[str, Any]) -> dict[str, float]:
    """The overall figures the ``end`` event carries, at most 64 keys."""
    overall = report["slices"][0]
    candidate = overall["systems"][report["candidate"]]
    metrics: dict[str, float] = {}
    _put(metrics, "cases", overall["n"])
    _put(metrics, "graded", candidate["graded"])
    _put(metrics, "coverage", candidate["coverage"])
    for cid, read in candidate["criteria"].items():
        mean = read["mean"]
        if mean is None:
            continue
        _put(metrics, cid, mean["point"])
        if candidate["coverage"] == 1:
            _put(metrics, f"{cid}_lower", mean["lower"])
            _put(metrics, f"{cid}_upper", mean["upper"])
        else:
            # Over part of the cases: a point to read, never a gate.
            _put(metrics, f"{cid}_unresolved", 1.0)
    for comparison in overall["comparisons"]:
        baseline = comparison["baseline"]
        for cid, read in comparison["criteria"].items():
            if read["delta"] is None:
                continue
            _put(metrics, f"delta_{cid}__{baseline}", read["delta"]["point"])
            if comparison["coverage"] < 1:
                _put(metrics, f"delta_{cid}_unresolved__{baseline}", 1.0)
    for cid, read in report["agreement"].get(report["candidate"], {}).items():
        _put(metrics, f"agreement_{cid}", read["percent_agreement"])
    return metrics


# ------------------------------------------------------------------ writing


def _items_rows(
    cases: _Cases, outputs: dict[str, _Outputs], judged: _Judged, settings: GradeSettings
) -> list[dict[str, Any]]:
    """One text-free row per case: ids, cluster, slice labels, statuses and scores."""
    rubric = settings.rubric
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(cases.ids):
        row: dict[str, Any] = {
            "item_id": item,
            "cluster": cases.clusters[index] if settings.cluster_column else None,
        }
        for column, values in cases.slices.items():
            row[f"slice__{column}"] = values[index]
        for name in outputs:
            row[f"status__{name}"] = judged.status[name][index]
            scores = judged.scores[name][index]
            for criterion in rubric.criteria:
                row[f"score__{criterion.id}__{name}"] = (
                    None if scores is None else scores[criterion.id]
                )
                if criterion.gold_column:
                    row[f"gold__{criterion.id}__{name}"] = outputs[name].gold[criterion.id][index]
            row[f"judge_retried__{name}"] = judged.retried[name][index]
        rows.append(row)
    return rows


def _rationale_rows(cases: _Cases, judged: _Judged) -> list[dict[str, Any]]:
    """One row per graded (case, system): the judge's rationale, capped. A rationale may quote the
    output it grades, so it is kept apart from the text-free items file."""
    return [
        {"item_id": item, "system": name, "rationale": (text or "")[:MAX_RATIONALE_CHARS]}
        for index, item in enumerate(cases.ids)
        for name, texts in judged.rationales.items()
        if (text := texts[index]) is not None
    ]


_CARD = """---
configs:
- config_name: items
  data_files: items.parquet
  default: true
---

# Graded evaluation

Written by `strata_forge.pipelines.grade_runner`. `items.parquet` holds one row per case: its id,
cluster, slice labels and, per system, the grading status and each criterion's score; it holds no
input or output text. `rationales.parquet` holds the judge's rationale per graded case and system,
capped at 2,000 characters; a rationale may quote the output it grades. `metrics.json` holds the
report and `manifest.json` the pinned inputs.
"""


def _write_outputs(
    spec: GradeSpec,
    cases: _Cases,
    outputs: dict[str, _Outputs],
    judged: _Judged,
    report: dict[str, Any],
    outdir: Path,
) -> dict[str, Path]:
    parquet: Any = __import__("pyarrow.parquet", fromlist=["write_table"])  # pyright: ignore[reportAny] - optional extra
    arrow: Any = __import__("pyarrow")  # pyright: ignore[reportAny] - optional extra
    outdir.mkdir(parents=True, exist_ok=True)
    items = outdir / ITEMS_FILENAME
    rows = _items_rows(cases, outputs, judged, spec.grading)
    parquet.write_table(arrow.Table.from_pylist(rows), str(items))
    rationales = outdir / RATIONALES_FILENAME
    schema = arrow.schema(
        [("item_id", arrow.string()), ("system", arrow.string()), ("rationale", arrow.string())]
    )
    parquet.write_table(
        arrow.Table.from_pylist(_rationale_rows(cases, judged), schema=schema), str(rationales)
    )
    (outdir / REPORT_FILENAME).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    (outdir / _CARD_FILENAME).write_text(_CARD)
    write_manifest(
        {
            "schema": MANIFEST_SCHEMA,
            "kind": _MANIFEST_KIND,
            "engine": manifest_engine(spec.engine_version, _RECORDED_DISTRIBUTIONS),
            "dataset": report["dataset"],
            "judge": {k: report["judge"][k] for k in ("model_id", "requested_revision", "commit")},
            "rubric_sha256": report["rubric"]["sha256"],
            "outputs": [
                {
                    "name": s["name"],
                    "repo_id": s["repo_id"],
                    "commit": s["commit"],
                    "path": s["path"],
                }
                for s in report["systems"]
            ],
            "grading": spec.grading.model_dump(mode="json"),
            "counts": {
                "cases": len(cases.ids),
                "systems": {name: dict(Counter(judged.status[name])) for name in outputs},
            },
        },
        outdir,
    )
    return {
        ITEMS_FILENAME: items,
        RATIONALES_FILENAME: rationales,
        REPORT_FILENAME: outdir / REPORT_FILENAME,
        MANIFEST_FILENAME: outdir / MANIFEST_FILENAME,
        _CARD_FILENAME: outdir / _CARD_FILENAME,
    }


async def _push(spec: GradeSpec, files: dict[str, Path], hf_token: str) -> str:
    """Push the report files in ONE commit, so a half-landed report can never exist."""
    out_repo = validate_repo_id(cast("str", spec.output_repo_id), "output repo")
    hub = HFHubClient(token=hf_token)  # EXPLICIT write token, never the ambient HF_TOKEN
    await ensure_private_output(hub, out_repo)
    commit = await hub.upload_files(
        dict(files), out_repo, repo_type="dataset", commit_message="graded evaluation report"
    )
    return f"{out_repo}@{commit}" if COMMIT_SHA_RE.fullmatch(commit) else out_repo


def _judge_manifest(results: Path) -> dict[str, Any]:
    path = results.parent / MANIFEST_FILENAME
    try:
        loaded: object = json.loads(path.read_text())
    except OSError, ValueError:
        return {}
    return cast("dict[str, Any]", loaded) if isinstance(loaded, dict) else {}


# ------------------------------------------------------------------ run


async def _execute(spec: GradeSpec, secrets: RunSecrets, writer: JsonlProgressWriter | None) -> str:
    hf_token = secrets.hf_token_value()
    hub_token = secrets.hub_credential()
    phase = phase_sink(writer, hf_token)
    settings = spec.grading
    pushing = bool(hf_token and spec.output_repo_id)

    if pushing and spec.output_repo_id:
        async with ticking_phase(phase, "Checking the output repo", stage="load_model"):
            await ensure_private_output(
                HFHubClient(token=hf_token), validate_repo_id(spec.output_repo_id, "output repo")
            )
    async with ticking_phase(phase, "Loading the gold split", stage="load_model"):
        columns = await asyncio.to_thread(_load_gold, spec, hub_token)
        cases = await asyncio.to_thread(_prepare_cases, columns, settings)
    outputs: dict[str, _Outputs] = {}
    for output in settings.outputs:
        async with ticking_phase(
            phase, f"Loading the outputs of {output.name}", stage="load_model"
        ):
            outputs[output.name] = await _fetch_outputs(output, spec, cases, hub_token)
    rows, pairs = _judge_rows(cases, outputs, settings.rubric)
    if not rows:
        msg = "no system has an output for any case: there is nothing to grade"
        raise RunError(msg)
    emit(writer, ProgressEvent(kind="start", stage="run", total_steps=len(rows)))
    results = Path(
        await inference_runner._execute(  # pyright: ignore[reportPrivateUsage]
            _judge_spec(spec), secrets, _JudgeProgress(writer), rows=rows
        )
    )
    async with ticking_phase(phase, "Reading the judge's grades", stage="run"):
        judged = await asyncio.to_thread(
            _read_judged, results, pairs, cases, outputs, settings.rubric
        )
    async with ticking_phase(phase, "Computing metrics", stage="run"):
        report = await asyncio.to_thread(
            _report, spec, cases, outputs, judged, _judge_manifest(results)
        )
    async with ticking_phase(phase, "Writing the report", stage="push"):
        outdir = results_dir(spec.run_id, name=_RESULTS_DIR_NAME)
        files = await asyncio.to_thread(
            _write_outputs, spec, cases, outputs, judged, report, outdir
        )
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
            step=len(rows),
            total_steps=len(rows),
            metrics=_headline(report),
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
