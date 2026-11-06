"""Construct a pinned prediction-evaluation spec without reading data or granting access."""

from __future__ import annotations

import json

from strata_forge.pipelines import SPEC_VERSION
from strata_forge.pipelines.evaluation_spec import EvalSpec


def example_spec() -> EvalSpec:
    """Return a synthetic ordered-label task; refs are illustrative immutable placeholders."""
    return EvalSpec.model_validate(
        {
            "schema_version": 1,
            "engine_version": SPEC_VERSION,
            "run_id": "example-evaluation",
            "progress_path": "progress.jsonl",
            "dataset_id": "example/synthetic-gold",
            "dataset_commit_sha": "a" * 40,
            "output_repo_id": None,
            "sealed": False,
            "echo_columns": [],
            "apply_calibration": "none",
            "bootstrap": None,
            "holm_families": [],
            "mode": "score_predictions",
            "evaluation_sets": [
                {
                    "evaluation_id": "classification",
                    "dataset_config": "default",
                    "split": "test",
                    "id_column": "item_id",
                    "gold_column": "gold",
                    "gold_index_map": {},
                    "label_space": "3way",
                    "labels": ["E", "C", "U"],
                    "projection": None,
                    "selection": {"row_start": 0, "row_end": None, "row_limit": 100},
                    "slice_columns": [],
                    "cluster_column": None,
                    "sampling_unit": "independent_items",
                    "declared_diagnostic": False,
                    "truncation_policy": "count_as_invalid",
                    "compare_to": [],
                    "source_labels": ["E", "C", "U"],
                    "predictions_ref": "example/synthetic-predictions@" + "b" * 40,
                    "predictions_config": "default",
                    "predictions_split": "train",
                    "prediction_id_column": "item_id",
                    "prediction_manifest_path": None,
                    "parse": "probability_columns",
                    "parse_config": {
                        "probability_columns": {"E": "p_E", "C": "p_C", "U": "p_U"},
                        "probability_kind": "class_probabilities",
                        "prediction_rule": "argmax",
                        "prediction_column": None,
                        "label_aliases": {},
                        "index_map": {},
                        "mass_outside_column": None,
                    },
                }
            ],
        }
    )


if __name__ == "__main__":
    print(json.dumps(example_spec().wire_payload(), indent=2))
