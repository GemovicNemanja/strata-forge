"""Import-only smoke harness for every script under ``examples/``.

Each example wraps its body in ``if __name__ == "__main__":`` so
importing the module only resolves top-level imports and module
constants — exactly the surface that catches API drift between
the examples and the library code. Running the examples for real
requires env-gated credentials and is out of scope here.

The harness:

1. Discovers every ``examples/NN_*.py`` file (ignoring
   ``_common`` and ``__pycache__``).
2. Imports each file via ``importlib`` under a unique module
   name so re-imports don't collide.
3. Asserts the import didn't blow up.

Any ImportError, NameError, or AttributeError surfaces as a
test failure pinpointing the example.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

EXAMPLES_DIR = Path(__file__).resolve().parents[2] / "examples"


def _discover_examples() -> list[Path]:
    return sorted(
        path
        for path in EXAMPLES_DIR.glob("*.py")
        if path.stem[0].isdigit()  # 01_..., 02_..., etc.
    )


# Make ``from _common import ...`` resolvable for the example modules.
@pytest.fixture(autouse=True)
def examples_on_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.syspath_prepend(str(EXAMPLES_DIR))


@pytest.mark.parametrize("path", _discover_examples(), ids=lambda p: p.stem)
def test_example_imports(path: Path) -> None:
    """Every example must import cleanly under the current public API."""
    module_name = f"forge_example_{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None, f"failed to build import spec for {path}"
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Register before exec so submodule lookups work if any example
    # uses dynamic re-imports.
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(module_name, None)
