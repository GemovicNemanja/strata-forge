"""A toy plugin distribution installed into a temporary site directory.

Built from files rather than ``pip``: a ``.dist-info`` directory with ``METADATA``, the entry
points and a PEP 610 ``direct_url.json``, beside the module it names. ``importlib.metadata``
finds it once the directory is on ``sys.path``, which is all the plugin machinery reads.
"""

from __future__ import annotations

import importlib
import json
import sys
from typing import TYPE_CHECKING, Any

import pytest

from strata_forge.plugins.manifest import PluginManifest, parse_manifest
from strata_forge.plugins.pins import AllowedPlugin, PluginPin, PluginSource

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

SHA = "a" * 40
OTHER_SHA = "b" * 40

TOY_MODULE = """
from pathlib import Path


class _Scorer:
    def __init__(self, setup):
        self.scale = float(setup.config.get("scale", 1.0))
        self.models = setup.models

    def score(self, steps):
        return [min(1.0, len(s.step) / 10.0 * self.scale) for s in steps]


def create(setup):
    return _Scorer(setup)


def bad_count(setup):
    class _Bad:
        def score(self, steps):
            return [0.5]
    return _Bad()


def job(ctx):
    rows = ctx.datasets["items"].stat().st_size
    (ctx.output_dir / "out.txt").write_text(f"{ctx.config.get('label', 'x')}:{rows}")
    ctx.report({"rows_seen": 1.0}, step=1)


def secrets_probe(ctx):
    import os
    names = sorted(k for k in os.environ if "TOKEN" in k.upper() or k.startswith("STRATA"))
    (ctx.output_dir / "env.txt").write_text(",".join(names))
"""

MANIFEST: dict[str, Any] = {
    "schema": "strata.plugin/v1",
    "distribution": "toy-prm",
    "version": "0.1.0",
    "entry_points": [
        {
            "name": "prm",
            "type": "step_scorer",
            "inputs": [
                {
                    "role": "steps",
                    "required_columns": ["item_id", "step_index", "context", "step"],
                },
                {"role": "verifier", "kind": "model", "required": False},
            ],
            "outputs": [{"role": "scores", "kind": "dataset"}],
            "config": [
                {"name": "scale", "type": "number", "minimum": 0, "maximum": 10, "default": 1.0},
                {"name": "mode", "type": "string", "enum": ["fast", "slow"]},
            ],
        },
        {
            "name": "job",
            "type": "runner",
            "inputs": [{"role": "items", "required_columns": ["text"]}],
            "outputs": [{"role": "result", "kind": "dataset"}],
            "config": [{"name": "label", "type": "string", "required": True}],
        },
    ],
}


def write_dist(
    site: Path,
    *,
    name: str = "toy_prm",
    version: str = "0.1.0",
    direct_url: dict[str, Any] | None,
    entry_points: str,
) -> None:
    info = site / f"{name}-{version}.dist-info"
    info.mkdir(parents=True)
    (info / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: {name.replace('_', '-')}\nVersion: {version}\n"
    )
    (info / "entry_points.txt").write_text(entry_points)
    if direct_url is not None:
        (info / "direct_url.json").write_text(json.dumps(direct_url))


@pytest.fixture
def toy_site(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    site = tmp_path / "site"
    (site / "toy_prm").mkdir(parents=True)
    (site / "toy_prm" / "__init__.py").write_text(TOY_MODULE)
    write_dist(
        site,
        direct_url={
            "url": "https://github.com/acme/toy-prm.git",
            "vcs_info": {"vcs": "git", "commit_id": SHA},
        },
        entry_points=(
            "[strata_forge.plugins]\nprm = toy_prm:create\njob = toy_prm:job\n"
            "bad = toy_prm:bad_count\nprobe = toy_prm:secrets_probe\n"
        ),
    )
    monkeypatch.syspath_prepend(str(site))
    importlib.invalidate_caches()
    yield site
    sys.modules.pop("toy_prm", None)


@pytest.fixture
def pin() -> PluginPin:
    return PluginPin(
        distribution="toy-prm",
        version="0.1.0",
        source=PluginSource(kind="git", repo="acme/toy-prm", commit=SHA),
    )


@pytest.fixture
def manifest() -> PluginManifest:
    return parse_manifest(json.dumps(MANIFEST))


@pytest.fixture
def allowed(pin: PluginPin) -> AllowedPlugin:
    return AllowedPlugin(id="toy", label="Toy PRM", pin=pin)
