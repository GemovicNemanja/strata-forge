"""A toy plugin distribution installed into a temporary site directory.

Built from files rather than ``pip``: a ``.dist-info`` directory with ``METADATA``, the entry
points and a PEP 610 ``direct_url.json``, beside the module it names. ``importlib.metadata``
finds it once the directory is on ``sys.path``, which is all the plugin machinery reads.
"""

from __future__ import annotations

import base64
import hashlib
import importlib
import json
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from strata_forge.plugins.manifest import PluginManifest, parse_manifest
from strata_forge.plugins.pins import (
    AllowedPlugin,
    LockedDependency,
    PluginPin,
    PluginSource,
    files_digest,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

SHA = "a" * 40
OTHER_SHA = "b" * 40

TOY_MODULE = """
from pathlib import Path

from strata_forge.plugins.step_scoring import Unscored

SEEN = []


class _Scorer:
    def __init__(self, setup):
        self.scale = float(setup.config.get("scale", 1.0))
        self.models = setup.models

    def score(self, steps):
        out = []
        for s in steps:
            if s.step_type == "skip":
                out.append(Unscored("not_scored_type"))
            elif not s.step:
                out.append(Unscored("unparseable"))
            else:
                out.append(min(1.0, len(s.step) / 10.0 * self.scale))
        return out


def create(setup):
    return _Scorer(setup)


def _fixed(value):
    def factory(setup):
        class _Fixed:
            def score(self, steps):
                return value(steps)
        return _Fixed()
    return factory


bad_count = _fixed(lambda steps: [0.5])
over_one = _fixed(lambda steps: [1.5 for _ in steps])
not_finite = _fixed(lambda steps: [float("nan") for _ in steps])
bad_reason = _fixed(lambda steps: [Unscored("tired") for _ in steps])
contexts = _fixed(lambda steps: [0.0 if isinstance(s.context, str) else float(len(s.context)) for s in steps])


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
            "score_kind": "probability",
            "unscored_reasons": ["unparseable", "not_scored_type"],
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


TOY_ENTRY_POINTS = (
    "[strata_forge.plugins]\nprm = toy_prm:create\njob = toy_prm:job\n"
    "bad = toy_prm:bad_count\nprobe = toy_prm:secrets_probe\n"
    "over = toy_prm:over_one\nnan = toy_prm:not_finite\n"
    "reason = toy_prm:bad_reason\ncontexts = toy_prm:contexts\n"
    "escape = os:getcwd\n"
)
TOY_DIRECT_URL: dict[str, Any] = {
    "url": "https://github.com/acme/toy-prm.git",
    "vcs_info": {"vcs": "git", "commit_id": SHA},
}


def _record_hash(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes()).digest()
    return "sha256=" + base64.urlsafe_b64encode(digest).rstrip(b"=").decode()


def write_dist(
    site: Path,
    *,
    name: str = "toy_prm",
    version: str = "0.1.0",
    direct_url: dict[str, Any] | None,
    entry_points: str,
    files: tuple[str, ...] = (),
    requires: tuple[str, ...] = (),
) -> None:
    """A dist-info as pip writes it: METADATA, entry points, the PEP 610 record and a RECORD
    hashing every file (``files`` are further site-relative paths the distribution owns)."""
    info = site / f"{name}-{version}.dist-info"
    info.mkdir(parents=True)
    (info / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: {name.replace('_', '-')}\nVersion: {version}\n"
        + "".join(f"Requires-Dist: {r}\n" for r in requires)
    )
    (info / "entry_points.txt").write_text(entry_points)
    (info / "INSTALLER").write_text("pip\n")
    if direct_url is not None:
        (info / "direct_url.json").write_text(json.dumps(direct_url))
    rows = [*files, *(f"{info.name}/{p.name}" for p in sorted(info.iterdir()))]
    lines = [f"{row},{_record_hash(site / row)},{(site / row).stat().st_size}" for row in rows]
    lines.append(f"{info.name}/RECORD,,")
    (info / "RECORD").write_text("\n".join(lines) + "\n")


def build_toy(site: Path, *, requires: tuple[str, ...] = ()) -> None:
    (site / "toy_prm").mkdir(parents=True)
    (site / "toy_prm" / "__init__.py").write_text(TOY_MODULE)
    write_dist(
        site,
        direct_url=TOY_DIRECT_URL,
        entry_points=TOY_ENTRY_POINTS,
        files=("toy_prm/__init__.py",),
        requires=requires,
    )


def _toy_digest() -> str:
    from importlib import metadata

    with tempfile.TemporaryDirectory() as tmp:
        build_toy(Path(tmp))
        (dist,) = metadata.distributions(path=[tmp])
        digest = files_digest(dist)
    if digest is None:  # pragma: no cover -- build_toy always writes a RECORD
        raise RuntimeError
    return digest


#: The files digest of the toy build, what an operator pins as ``files_sha256``.
TOY_DIGEST = _toy_digest()


@pytest.fixture
def toy_site(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    site = tmp_path / "site"
    build_toy(site)
    monkeypatch.syspath_prepend(str(site))
    importlib.invalidate_caches()
    yield site
    sys.modules.pop("toy_prm", None)


#: The build backend a git toy pin locks.
TOY_BUILD_REQUIRES = (LockedDependency(name="setuptools", version="80.9.0", sha256=("5" * 64,)),)


def toy_pin(**source: Any) -> PluginPin:
    merged = {"kind": "git", "repo": "acme/toy-prm", "commit": SHA, "files_sha256": TOY_DIGEST}
    merged |= source
    return PluginPin(
        distribution="toy-prm",
        version="0.1.0",
        source=PluginSource.model_validate(merged),
        build_requires=TOY_BUILD_REQUIRES if merged["kind"] == "git" else (),
    )


@pytest.fixture
def pin() -> PluginPin:
    return toy_pin()


@pytest.fixture
def manifest() -> PluginManifest:
    return parse_manifest(json.dumps(MANIFEST))


@pytest.fixture
def allowed(pin: PluginPin) -> AllowedPlugin:
    return AllowedPlugin(id="toy", label="Toy PRM", pin=pin)
