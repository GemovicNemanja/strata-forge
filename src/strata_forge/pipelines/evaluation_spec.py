"""Strict inert specifications for separately reported classification populations."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Annotated, Literal, cast

from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from strata_forge.pipelines._hub_ids import validate_hub_ref, validate_hub_repo
from strata_forge.training.verifier_config import VerifierConfig

__all__ = ["EvalSpec", "EvaluationSet", "PredictionSet", "RenderingSpec"]


def _tuple(value: object) -> object:
    return tuple(cast("list[object]", value)) if type(value) is list else value


type Items[T] = Annotated[tuple[T, ...], BeforeValidator(_tuple)]
type Identifier = Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,47}$")]
type Column = Annotated[str, Field(min_length=1, max_length=255)]
type Label = Annotated[str, Field(min_length=1, max_length=64)]
type Labels = Annotated[Items[Label], Field(min_length=2, max_length=16)]
type Sha = Annotated[str, Field(pattern=r"^[a-f0-9]{40}$")]
type Repo = Annotated[
    str,
    Field(max_length=215, pattern=r"^[A-Za-z0-9_][A-Za-z0-9._-]*/[A-Za-z0-9_][A-Za-z0-9._-]*$"),
    AfterValidator(validate_hub_repo),
]
type Ref = Annotated[
    str,
    Field(
        max_length=256,
        pattern=r"^[A-Za-z0-9_][A-Za-z0-9._-]*/[A-Za-z0-9_][A-Za-z0-9._-]*@[a-f0-9]{40}$",
    ),
    AfterValidator(validate_hub_ref),
]


class _Data(BaseModel):
    model_config = ConfigDict(
        frozen=True, extra="forbid", strict=True, hide_input_in_errors=True, protected_namespaces=()
    )


class Selection(_Data):
    """A bounded ordered population, selected before materialization."""

    row_start: int = Field(ge=0)
    row_end: int | None = Field(ge=1)
    row_limit: int = Field(ge=1, le=1_000_000)

    @model_validator(mode="after")
    def _span(self) -> Selection:
        if self.row_end is not None and self.row_end <= self.row_start:
            raise ValueError("selection_invalid")
        return self


class Comparison(_Data):
    """One explicit baseline and sampling contract."""

    comparison_id: Identifier
    baseline_ref: Ref
    baseline_evaluation_id: Identifier
    sampling_unit: Literal["independent_items", "clustered_items"]


class EvaluationSet(_Data):
    """A single gold population and its explicit target task."""

    evaluation_id: Identifier
    dataset_config: str = Field(min_length=1, max_length=120)
    split: str = Field(min_length=1, max_length=120)
    id_column: Column
    gold_column: Column
    gold_index_map: dict[str, Label]
    label_space: Literal["3way", "2way_entail_vs_not", "categorical"]
    labels: Labels
    projection: dict[Label, Label] | None
    selection: Selection
    slice_columns: Annotated[Items[Column], Field(max_length=16)]
    cluster_column: Column | None
    sampling_unit: Literal["independent_items", "clustered_items"]
    declared_diagnostic: bool
    truncation_policy: Literal["retain_reported", "count_as_invalid"]
    compare_to: Annotated[Items[Comparison], Field(max_length=16)]
    column_mapping: dict[Literal["premises", "hypothesis"], Column] | None = None

    @model_validator(mode="after")
    def _task(self) -> EvaluationSet:
        _labels(self.labels)
        if self.label_space != "categorical" and len(self.labels) != (
            3 if self.label_space == "3way" else 2
        ):
            raise ValueError("label_space_size")
        if len(set(self.slice_columns)) != len(self.slice_columns):
            raise ValueError("slice_columns_duplicate")
        if len({c.comparison_id for c in self.compare_to}) != len(self.compare_to):
            raise ValueError("comparison_id_duplicate")
        _index_map(self.gold_index_map, self.labels)
        _artifact_columns(self.labels)
        return self


class ProbabilityColumns(_Data):
    """Exact complete probability columns and independent hard-prediction semantics."""

    probability_columns: dict[Label, Column]
    probability_kind: Literal["class_probabilities", "restricted_softmax"]
    prediction_rule: Literal["argmax", "column"]
    prediction_column: Column | None
    label_aliases: dict[Label, Label]
    index_map: dict[str, Label]
    mass_outside_column: Column | None

    @model_validator(mode="after")
    def _rule(self) -> ProbabilityColumns:
        if (self.prediction_column is not None) != (self.prediction_rule == "column"):
            raise ValueError("prediction_rule_conflict")
        return self


class JSONField(_Data):
    """Bounded hard-label extraction with no fabricated probability vector."""

    column: Column
    input_kind: Literal["native", "json_text"]
    path: Annotated[Items[Column], Field(max_length=8)]
    label_aliases: dict[Label, Label]
    index_map: dict[str, Label]


class PredictionSet(EvaluationSet):
    """One stable-ID prediction source joined to one gold population."""

    source_labels: Labels
    predictions_ref: Ref
    predictions_config: Column
    predictions_split: Column
    prediction_id_column: Column
    prediction_manifest_path: Literal["scoring.json", "metrics.json"] | None
    parse: Literal["probability_columns", "json_field"]
    parse_config: ProbabilityColumns | JSONField

    @model_validator(mode="after")
    def _parser(self) -> PredictionSet:
        _labels(self.source_labels)
        _projection(self.source_labels, self.labels, self.projection)
        config = self.parse_config
        if self.parse == "probability_columns":
            if not isinstance(config, ProbabilityColumns) or set(config.probability_columns) != set(
                self.source_labels
            ):
                raise ValueError("probability_labels_mismatch")
            if len(set(config.probability_columns.values())) != len(config.probability_columns):
                raise ValueError("probability_columns_duplicate")
        elif not isinstance(config, JSONField):
            raise ValueError("parser_config_mismatch")
        _index_map(config.index_map, self.source_labels)
        if any(value not in self.source_labels for value in config.label_aliases.values()):
            raise ValueError("prediction_alias_invalid")
        return self


class RenderingOverrides(_Data):
    """Explicit rendering overrides; absent fields do not replace artifact values."""

    template: str = Field(min_length=1, max_length=200_000)
    pair_template: str | None
    premise_format: Literal["numbered_lines", "json"]
    max_length: int = Field(ge=1, le=8192)
    max_premises: int = Field(ge=1, le=20)


class PartialRendering(_Data):
    """Only named fields override pinned model metadata."""

    template: str | None = Field(default=None, min_length=1, max_length=200_000)
    pair_template: str | None = None
    premise_format: Literal["numbered_lines", "json"] | None = None
    max_length: int | None = Field(default=None, ge=1, le=8192)
    max_premises: int | None = Field(default=None, ge=1, le=20)


class RenderingSpec(_Data):
    """A complete explicit rendering declaration or pinned model metadata."""

    source: Literal["override", "model_artifact"]
    template: str | None = Field(default=None, min_length=1, max_length=200_000)
    pair_template: str | None = None
    premise_format: Literal["numbered_lines", "json"] | None = None
    max_length: int | None = Field(default=None, ge=1, le=8192)
    max_premises: int | None = Field(default=None, ge=1, le=20)
    file: Literal["strata_verifier.json"] | None = None
    overrides: PartialRendering | None = None

    @model_validator(mode="after")
    def _source(self) -> RenderingSpec:
        names = {"template", "pair_template", "premise_format", "max_length", "max_premises"}
        if self.source == "override":
            if (
                not names <= self.model_fields_set
                or self.file is not None
                or self.overrides is not None
            ):
                raise ValueError("rendering_source_invalid")
            RenderingOverrides.model_validate(self.model_dump(include=names))
        elif (
            self.file != "strata_verifier.json"
            or self.overrides is None
            or names & self.model_fields_set
        ):
            raise ValueError("rendering_source_invalid")
        return self

    def explicit_config(self, labels: tuple[str, ...]) -> VerifierConfig:
        """Validate a complete override without loading any model or dataset."""
        if self.source != "override":
            raise ValueError("rendering_requires_artifact")
        return VerifierConfig.model_validate(
            {
                "labels": labels,
                **self.model_dump(
                    include={
                        "template",
                        "pair_template",
                        "premise_format",
                        "max_length",
                        "max_premises",
                    }
                ),
            }
        )


class HolmFamily(_Data):
    """An explicit multiplicity family containing declared comparison IDs."""

    family_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$")
    members: Annotated[Items[str], Field(min_length=1, max_length=10_000)]


class EvalSpec(_Data):
    """A versioned immutable-input evaluation job; unsupported capabilities refuse."""

    schema_version: Literal[1]
    engine_version: str
    run_id: Identifier
    progress_path: Literal["progress.jsonl"]
    dataset_id: Repo
    dataset_commit_sha: Sha
    output_repo_id: Repo | None
    sealed: bool
    echo_columns: Items[str]
    apply_calibration: Literal["none"]
    bootstrap: None
    holm_families: Annotated[Items[HolmFamily], Field(max_length=64)]
    mode: Literal["classify", "score_predictions"]
    evaluation_sets: Annotated[
        Items[PredictionSet | EvaluationSet], Field(min_length=1, max_length=64)
    ]
    model_id: Repo | None = None
    model_revision: Sha | None = None
    engine: Literal["hf_classify", "hf_label_logits"] | None = None
    model_labels: Labels | None = None
    label_aliases: dict[Label, Label] | None = None
    label_tokens: dict[Label, Label] | None = None
    rendering: RenderingSpec | None = None
    scoring_batch_size: int | None = Field(default=None, ge=1, le=256)
    dtype: Literal["auto", "float32", "float16", "bfloat16"] | None = None

    @field_validator("schema_version", mode="before")
    @classmethod
    def _version(cls, value: object) -> object:
        if type(value) is not int or value != 1:
            raise ValueError("evaluation_schema_version_invalid")
        return value

    def digest(self) -> str:
        """Bind the full declared job including configs, splits, selectors and engine stamp."""
        encoded = json.dumps(
            self.wire_payload(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def wire_payload(self) -> dict[str, object]:
        """Return the JSON-native roundtrip payload preserving mode-specific field presence."""
        return self.model_dump(mode="json", exclude_unset=True)

    @model_validator(mode="after")
    def _job(self) -> EvalSpec:
        if self.sealed or self.echo_columns:
            raise ValueError("capability_unavailable")
        if len({s.evaluation_id for s in self.evaluation_sets}) != len(self.evaluation_sets):
            raise ValueError("evaluation_id_duplicate")
        if sum(s.selection.row_limit for s in self.evaluation_sets) > 1_000_000:
            raise ValueError("evaluation_population_limit_exceeded")
        model_fields = {
            "model_id",
            "model_revision",
            "engine",
            "model_labels",
            "label_aliases",
            "label_tokens",
            "rendering",
            "scoring_batch_size",
            "dtype",
        }
        if self.mode == "score_predictions":
            if model_fields & self.model_fields_set or any(
                not isinstance(s, PredictionSet) or s.column_mapping is not None
                for s in self.evaluation_sets
            ):
                raise ValueError("prediction_mode_fields_invalid")
        else:
            if not model_fields <= self.model_fields_set or any(
                getattr(self, name) is None
                for name in (
                    "model_id",
                    "model_revision",
                    "engine",
                    "model_labels",
                    "rendering",
                    "scoring_batch_size",
                )
            ):
                raise ValueError("classifier_fields_missing")
            if self.model_labels is None or self.rendering is None:
                raise ValueError("classifier_fields_missing")
            _labels(self.model_labels)
            for case in self.evaluation_sets:
                if isinstance(case, PredictionSet) or set(case.column_mapping or {}) != {
                    "premises",
                    "hypothesis",
                }:
                    raise ValueError("classifier_mapping_invalid")
                _projection(self.model_labels, case.labels, case.projection)
            if self.engine == "hf_classify" and self.label_tokens is not None:
                raise ValueError("classifier_has_verbalizers")
            if self.engine == "hf_label_logits" and (
                set(self.label_tokens or {}) != set(self.model_labels) or self.label_aliases
            ):
                raise ValueError("verbalizer_labels_mismatch")
            if self.rendering.source == "override":
                config = self.rendering.explicit_config(self.model_labels)
                if (config.pair_template is not None) != (self.engine == "hf_classify"):
                    raise ValueError("pair_template_engine_mismatch")
        refs = self.input_refs()
        if len(self.input_reads()) > 64 or any(".." in ref.split("@")[0] for ref in refs):
            raise ValueError("input_ref_invalid")
        if self.output_repo_id in {ref.split("@")[0] for ref in refs}:
            raise ValueError("output_is_input")
        members = {
            f"{s.evaluation_id}/{c.comparison_id}"
            for s in self.evaluation_sets
            for c in s.compare_to
        }
        if len({f.family_id for f in self.holm_families}) != len(self.holm_families) or any(
            len(set(f.members)) != len(f.members) or not set(f.members) <= members
            for f in self.holm_families
        ):
            raise ValueError("holm_family_invalid")
        return self

    def input_refs(self) -> tuple[str, ...]:
        """Enumerate every pinned primary input before any reads occur."""
        refs = [f"{self.dataset_id}@{self.dataset_commit_sha}"]
        if self.mode == "classify":
            refs.append(f"{self.model_id}@{self.model_revision}")
        for case in self.evaluation_sets:
            if isinstance(case, PredictionSet):
                refs.append(case.predictions_ref)
            refs.extend(c.baseline_ref for c in case.compare_to)
        return tuple(dict.fromkeys(refs))

    def input_reads(self) -> tuple[tuple[str, Literal["model", "dataset"]], ...]:
        """Enumerate exact refs with their distinct Hub repository domain."""
        reads: list[tuple[str, Literal["model", "dataset"]]] = [
            (f"{self.dataset_id}@{self.dataset_commit_sha}", "dataset")
        ]
        if self.mode == "classify":
            reads.append((f"{self.model_id}@{self.model_revision}", "model"))
        for case in self.evaluation_sets:
            if isinstance(case, PredictionSet):
                reads.append((case.predictions_ref, "dataset"))
            reads.extend((comparison.baseline_ref, "dataset") for comparison in case.compare_to)
        return tuple(dict.fromkeys(reads))


def _labels(labels: tuple[str, ...]) -> None:
    if len(set(labels)) != len(labels) or any(not label.strip() for label in labels):
        raise ValueError("labels_invalid")


def _index_map(mapping: dict[str, str], labels: tuple[str, ...]) -> None:
    if len(mapping) > 16 or any(
        not re.fullmatch(r"0|-?[1-9][0-9]{0,18}", key) or value not in labels
        for key, value in mapping.items()
    ):
        raise ValueError("index_map_invalid")


def _projection(
    source: tuple[str, ...], target: tuple[str, ...], projection: dict[str, str] | None
) -> None:
    if (
        source != target
        if projection is None
        else set(projection) != set(source) or set(projection.values()) != set(target)
    ):
        raise ValueError("projection_incomplete")


def _artifact_columns(labels: tuple[str, ...]) -> None:
    names = (
        [f"p_{label}" for label in labels]
        + [f"p_cal_{label}" for label in labels]
        + ["p_inconsistent"]
    )
    if len(set(names)) != len(names):
        raise ValueError("artifact_column_collision")
