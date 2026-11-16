"""Checkpoint loading metadata is structural evidence, never a warning-text artifact."""

from __future__ import annotations

import pytest
from pydantic import ValidationError as ModelValidationError

from strata_forge.training._load_guard import (
    MAX_AUDIT_KEYS,
    HeadKeyDecision,
    LoadingAudit,
    ModelLoadingError,
    _guard_loading_result,
)


def _info(**overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "missing_keys": set(),
        "unexpected_keys": set(),
        "mismatched_keys": set(),
        "error_msgs": [],
    }
    values.update(overrides)
    return values


def test_complete_trained_checkpoint_and_json_round_trip() -> None:
    original = object()
    model, audit = _guard_loading_result((original, _info()))
    assert model is original
    assert audit.status == "accepted"
    assert audit.conversion_diagnostics == "not_returned"
    assert LoadingAudit.model_validate_json(audit.model_dump_json()) == audit


@pytest.mark.parametrize(
    ("field", "value", "code"),
    [
        ("missing_keys", {"encoder.embeddings.weight"}, "missing_weights"),
        ("missing_keys", {"classifier.weight"}, "missing_weights"),
        ("unexpected_keys", {"lm_head.weight"}, "unexpected_weights"),
        ("mismatched_keys", {("classifier.weight", (2, 8), (3, 8))}, "mismatched_weights"),
        ("error_msgs", ["raw private warning text"], "load_errors"),
        ("conversion_errors", {"some_key": "raw private conversion text"}, "conversion_errors"),
    ],
)
def test_every_structural_diagnostic_is_fatal_for_a_trained_artifact(
    field: str, value: object, code: str
) -> None:
    with pytest.raises(ModelLoadingError) as failure:
        _guard_loading_result((object(), _info(**{field: value})))
    assert code in failure.value.audit.codes
    assert "raw private" not in failure.value.audit.model_dump_json()
    assert "raw private" not in str(failure.value)


@pytest.mark.parametrize(
    "result",
    [
        None,
        object(),
        (object(),),
        (None, _info()),
        (object(), {}),
        (object(), _info(unknown_structural_error=[])),
        (object(), _info(missing_keys=["body.*"])),
        (object(), _info(error_msgs="secret text")),
    ],
)
def test_missing_malformed_or_unknown_loading_contract_fails_closed(result: object) -> None:
    with pytest.raises(ModelLoadingError, match="unsupported_loading_info"):
        _guard_loading_result(result)


def test_parameter_name_capture_is_bounded_but_counts_conserve_observed_keys() -> None:
    keys = {f"encoder.layers.{index}.weight" for index in range(100)}
    with pytest.raises(ModelLoadingError) as failure:
        _guard_loading_result((object(), _info(missing_keys=keys)))
    audit = failure.value.audit
    assert audit.missing_count == 100
    assert len(audit.missing_keys) == MAX_AUDIT_KEYS
    assert audit.missing_omitted == 100 - MAX_AUDIT_KEYS
    assert audit.missing_keys == tuple(sorted(keys))[:MAX_AUDIT_KEYS]
    assert len(str(failure.value)) < 1024
    assert LoadingAudit.model_validate_json(audit.model_dump_json()) == audit


def test_conversion_diagnostics_are_checked_when_the_library_returns_them() -> None:
    _, audit = _guard_loading_result((object(), _info(conversion_errors={})))
    assert audit.conversion_diagnostics == "reported"
    with pytest.raises(ModelLoadingError) as failure:
        _guard_loading_result((object(), _info(error_msgs=["private one", "private two"])))
    assert failure.value.audit.error_count == 2
    assert "private" not in failure.value.audit.model_dump_json()


def test_exact_head_decision_does_not_authorize_inference_or_guess_suffixes() -> None:
    decision = HeadKeyDecision(
        initializer_id="bert_classification_head",
        head_kind="encoder",
        created_keys=("classifier.weight", "classifier.bias"),
        removed_keys=("lm_head.weight",),
    )
    with pytest.raises(ModelLoadingError) as failure:
        _guard_loading_result(
            (
                object(),
                _info(
                    missing_keys={"classifier.weight", "classifier.bias"},
                    unexpected_keys={"lm_head.weight"},
                ),
            )
        )
    assert decision.accepts(failure.value.audit)
    with pytest.raises(ModelLoadingError) as body:
        _guard_loading_result((object(), _info(missing_keys={"encoder.embeddings.weight"})))
    assert not decision.accepts(body.value.audit)
    with pytest.raises(ModelLoadingError) as suffix:
        _guard_loading_result(
            (object(), _info(missing_keys={"classifier.weight", "other.classifier.bias"}))
        )
    assert not decision.accepts(suffix.value.audit)


@pytest.mark.parametrize("created", [("classifier.*",), ("weight", "weight")])
def test_head_keys_are_exact_and_unique(created: tuple[str, ...]) -> None:
    with pytest.raises(ModelValidationError):
        HeadKeyDecision(initializer_id="bert_head", head_kind="encoder", created_keys=created)


def test_decoder_head_decision_requires_its_owned_initializer() -> None:
    with pytest.raises(ModelValidationError, match="decoder_logit_head"):
        HeadKeyDecision(
            initializer_id="generic_head", head_kind="decoder", created_keys=("score.weight",)
        )
    decision = HeadKeyDecision(
        initializer_id="decoder_logit_head", head_kind="decoder", created_keys=("score.weight",)
    )
    assert decision.initializer_id == "decoder_logit_head"


def test_a_truncated_audit_cannot_authorize_a_head_decision() -> None:
    audit = LoadingAudit(
        status="refused",
        missing_keys=("classifier.weight",),
        missing_count=2,
        codes=("missing_weights",),
    )
    decision = HeadKeyDecision(
        initializer_id="bert_head", head_kind="encoder", created_keys=("classifier.weight",)
    )
    assert not decision.accepts(audit)


def test_guard_has_no_imports_from_training_execution_or_optional_libraries() -> None:
    import ast
    from pathlib import Path

    from strata_forge.training import _load_guard

    tree = ast.parse(Path(_load_guard.__file__).read_text())
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.ImportFrom)
            and node.module
            and node.module.startswith("strata_forge.")
        ):
            assert node.module.startswith("strata_forge.core.")


def test_loading_audit_rejects_inconsistent_serialized_status() -> None:
    with pytest.raises(ModelValidationError):
        LoadingAudit(status="accepted", missing_count=1)
    with pytest.raises(ModelValidationError):
        LoadingAudit(
            status="refused",
            missing_keys=("body.weight",),
            missing_count=0,
            codes=("missing_weights",),
        )


def test_audit_rejects_fabricated_omission_or_reason_counts() -> None:
    with pytest.raises(ModelValidationError):
        LoadingAudit(
            status="refused", missing_count=100, missing_omitted=0, codes=("missing_weights",)
        )
    with pytest.raises(ModelValidationError):
        LoadingAudit(status="refused", missing_count=1, codes=("unexpected_weights",))
    with pytest.raises(ModelValidationError):
        LoadingAudit(
            status="refused", codes=("unsupported_loading_info", "unsupported_loading_info")
        )


def test_arbitrary_diagnostic_objects_are_never_stringified() -> None:
    class Hostile:
        def __str__(self) -> str:
            raise AssertionError("diagnostic objects must not be formatted")

        def __repr__(self) -> str:
            raise AssertionError("diagnostic objects must not be represented")

    with pytest.raises(ModelLoadingError) as failure:
        _guard_loading_result((object(), _info(error_msgs=[Hostile()])))
    assert failure.value.audit.codes == ("unsupported_loading_info",)
    assert "unsupported_loading_info" in failure.value.audit.model_dump_json()


def test_custom_report_containers_cannot_execute_diagnostic_accessors() -> None:
    class HostileDict(dict[str, object]):
        def __getitem__(self, key: str) -> object:
            raise RuntimeError("raw private accessor exception must not escape")

    with pytest.raises(ModelLoadingError, match="unsupported_loading_info"):
        _guard_loading_result((object(), HostileDict(_info())))
