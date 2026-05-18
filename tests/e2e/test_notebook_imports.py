"""Import-only smoke harness for the Marimo notebook templates.

Marimo notebooks are Python files that define an ``app`` plus
``@app.cell`` functions. Importing the file builds the ``app``
without executing cell bodies, so API drift between the templates
and the library surfaces as an import error.

This mirrors the examples harness in ``test_example_imports.py``.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

NOTEBOOKS_DIR = Path(__file__).resolve().parents[2] / "notebooks"


def _discover_notebooks() -> list[Path]:
    return sorted(path for path in NOTEBOOKS_DIR.glob("*.py") if path.stem[0].isdigit())


@pytest.mark.parametrize("path", _discover_notebooks(), ids=lambda p: p.stem)
def test_notebook_imports(path: Path) -> None:
    """Every notebook template must import cleanly."""
    module_name = f"forge_notebook_{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(module_name, None)
