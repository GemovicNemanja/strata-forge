"""Separate classifier authority uses the existing private read-and-unlink secret owner."""

from __future__ import annotations

import base64
import json
import os
from typing import TYPE_CHECKING

from strata_forge.pipelines import CLASSIFIER_AUTHORITY_SECRET, EVAL_AUTHORITY_SECRET, _common
from strata_forge.pipelines._common import load_secrets, runner_main
from tests.unit.pipelines.test_common import _deliver

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def test_both_frames_are_separate_masked_and_value_redacted(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    training, evaluation = "strata_worker_" + "t" * 43, "strata_worker_" + "e" * 43
    raw = json.dumps({"worker_credential": training})
    other = json.dumps({"worker_credential": evaluation})
    path = _deliver(
        monkeypatch, tmp_path, {CLASSIFIER_AUTHORITY_SECRET: raw, EVAL_AUTHORITY_SECRET: other}
    )
    secrets = load_secrets()
    assert not path.exists()
    assert secrets.classifier_authority is not None
    assert secrets.classifier_authority.get_secret_value() == raw
    assert secrets.eval_authority is not None
    assert secrets.eval_authority.get_secret_value() == other
    redactor = _common.Redactor(secrets.sensitive_values())
    for credential in (training, evaluation):
        assert credential not in repr(secrets)
        encoded = base64.b64encode(credential.encode()).decode()
        assert credential not in redactor.redact(credential)
        assert encoded not in redactor.redact(encoded)


def test_classifier_environment_is_withdrawn_without_authority_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(CLASSIFIER_AUTHORITY_SECRET, "strata_worker_" + "x" * 43)
    assert load_secrets().classifier_authority is None
    assert CLASSIFIER_AUTHORITY_SECRET not in os.environ


async def test_runner_error_scrubs_nested_classifier_credential(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    credential = "strata_worker_" + "x" * 43
    encoded = base64.b64encode(credential.encode()).decode()
    _deliver(
        monkeypatch,
        tmp_path,
        {CLASSIFIER_AUTHORITY_SECRET: json.dumps({"worker_credential": credential})},
    )

    async def execute(*_: object) -> None:
        raise RuntimeError("nested " + encoded)

    assert await runner_main(execute) == 1
    assert encoded not in capsys.readouterr().err
