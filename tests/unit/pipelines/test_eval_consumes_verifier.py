"""An evaluation over a trained verifier's own files, exactly as the trainer and the scorer write them.

The check itself is the clean-install smoke's ``check_eval_consumes_verifier``, which the ``all``
leg runs on every push; this module runs the same check wherever torch is installed::

    uv sync --extra finetuning --extra scoring --extra hf --extra storage
    uv run pytest tests/unit/pipelines/test_eval_consumes_verifier.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("datasets")
pytest.importorskip("pyarrow.parquet")
pytest.importorskip("safetensors.torch")

_SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "smoke_clean_install.py"


def test_an_evaluation_reads_the_verifier_s_scores_and_calibration_as_written() -> None:
    spec = importlib.util.spec_from_file_location("smoke_clean_install_verifier", _SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    smoke = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = smoke
    spec.loader.exec_module(smoke)
    smoke.check_eval_consumes_verifier()
