"""The plugin's own environment: keyed by the pin and the engine, installed without credentials,
an hf wheel checked before install, verified after it and on every reuse."""

from __future__ import annotations

import asyncio
import hashlib
import json
from typing import TYPE_CHECKING, Any

import pytest

from strata_forge import __version__
from strata_forge.plugins import environment
from strata_forge.plugins.manifest import PluginManifestError
from strata_forge.plugins.pins import PluginPin, PluginSource

from .conftest import SHA, TOY_ENTRY_POINTS, TOY_MODULE, write_dist

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

WHEEL = "toy_prm-0.1.0-py3-none-any.whl"


class FakeMachine:
    """Stands in for ``python -m venv`` and ``pip``: a venv is a directory with ``bin/python``,
    and an install writes the dist-info pip would (with the PEP 610 record it would write)."""

    def __init__(self, root: Path, *, direct_url: dict[str, Any], engine: str = __version__):
        self.root = root
        self.direct_url = direct_url
        self.engine = engine
        self.calls: list[list[str]] = []
        self.envs: list[Mapping[str, str]] = []

    def site(self, python: str) -> Path:
        from pathlib import Path

        return Path(python).parent.parent / "site"

    async def __call__(self, args: Sequence[str], env: Mapping[str, str]) -> tuple[int, str]:
        self.calls.append(list(args))
        self.envs.append(env)
        if args[1:3] == ["-m", "venv"]:
            from pathlib import Path

            target = Path(args[3])
            (target / "bin").mkdir(parents=True)
            (target / "bin" / "python").write_text("")
            return 0, ""
        site = self.site(args[0])
        if "install" in args:
            (site / "toy_prm").mkdir(parents=True)
            (site / "toy_prm" / "__init__.py").write_text(TOY_MODULE)
            write_dist(
                site,
                direct_url=self.direct_url,
                entry_points=TOY_ENTRY_POINTS,
                files=("toy_prm/__init__.py",),
            )
            write_dist(
                site, name="strata_forge", version=self.engine, direct_url=None, entry_points=""
            )
            return 0, "Successfully installed"
        if "-c" in args:
            return 0, json.dumps([str(site)])
        if "list" in args:
            return 0, json.dumps(
                [{"name": "toy-prm", "version": "0.1.0"}, {"name": "numpy", "version": "2.4.0"}]
            )
        return 1, "unexpected command"


def _git_url() -> dict[str, Any]:
    return {
        "url": "https://github.com/acme/toy-prm.git",
        "vcs_info": {"vcs": "git", "commit_id": SHA},
    }


def test_the_key_changes_with_the_pin_and_the_engine(pin: PluginPin) -> None:
    key = environment.environment_key(pin, "strata-forge[hf]==0.6.0")
    assert key == environment.environment_key(pin, "strata-forge[hf]==0.6.0")
    moved = pin.model_copy(update={"source": pin.source.model_copy(update={"commit": "b" * 40})})
    assert environment.environment_key(moved, "strata-forge[hf]==0.6.0") != key
    assert environment.environment_key(pin, "strata-forge[hf]==0.7.0") != key


def test_the_install_environment_holds_no_credential(tmp_path: Path) -> None:
    env = environment.install_environ(
        home=tmp_path / "home",
        pip_cache=tmp_path / "cache",
        environ={
            "PATH": "/bin",
            "HOME": "/home/u",
            "HF_TOKEN": "hf_x",
            "GITHUB_TOKEN": "ghp_x",
            "PIP_INDEX_URL": "https://user:pass@index/simple",
            "STRATA_SECRETS_FILE": "/run/s",
            "NETRC": "/home/u/.netrc",
        },
    )
    assert env["PATH"] == "/bin"
    assert env["HOME"] == str(tmp_path / "home")
    assert env["GIT_TERMINAL_PROMPT"] == "0"
    for name in ("HF_TOKEN", "GITHUB_TOKEN", "PIP_INDEX_URL", "STRATA_SECRETS_FILE", "NETRC"):
        assert name not in env


def test_requirements_name_the_exact_build(pin: PluginPin, tmp_path: Path) -> None:
    assert environment.plugin_requirement(pin, None) == (
        f"toy-prm @ git+https://github.com/acme/toy-prm.git@{SHA}"
    )
    hf = PluginPin(
        distribution="toy-prm",
        version="0.1.0",
        source=PluginSource(
            kind="hf", repo="acme/toy", commit=SHA, wheel=WHEEL, wheel_sha256="c" * 64
        ),
    )
    wheel = tmp_path / WHEEL
    assert environment.plugin_requirement(hf, wheel).endswith(f"{WHEEL}#sha256={'c' * 64}")
    with pytest.raises(PluginManifestError):
        environment.plugin_requirement(hf, None)


def test_an_environment_is_built_once_and_reused(pin: PluginPin, tmp_path: Path) -> None:
    machine = FakeMachine(tmp_path, direct_url=_git_url())
    built = asyncio.run(environment.ensure_environment(pin, root=tmp_path / "envs", run=machine))
    assert not built.reused
    assert built.root.parent == tmp_path / "envs"
    assert {"name": "numpy", "version": "2.4.0"} in built.packages
    installs = [c for c in machine.calls if "install" in c]
    assert len(installs) == 1
    assert installs[0][-1] == f"toy-prm @ git+https://github.com/acme/toy-prm.git@{SHA}"
    # Every command ran without a credential and on a fresh HOME.
    for env in machine.envs:
        assert not any("TOKEN" in name for name in env)
    again = asyncio.run(environment.ensure_environment(pin, root=tmp_path / "envs", run=machine))
    assert again.reused
    assert again.key == built.key
    assert len([c for c in machine.calls if "install" in c]) == 1
    assert again.record()["requirements"] == list(built.requirements)


def test_an_environment_that_no_longer_verifies_is_rebuilt(pin: PluginPin, tmp_path: Path) -> None:
    machine = FakeMachine(tmp_path, direct_url=_git_url())
    built = asyncio.run(environment.ensure_environment(pin, root=tmp_path / "envs", run=machine))
    # Someone reinstalled the plugin from elsewhere inside it.
    info = built.root / "site" / "toy_prm-0.1.0.dist-info" / "direct_url.json"
    info.write_text(json.dumps({"url": "file:///src/toy", "dir_info": {"editable": True}}))
    again = asyncio.run(environment.ensure_environment(pin, root=tmp_path / "envs", run=machine))
    assert not again.reused
    assert len([c for c in machine.calls if "install" in c]) == 2


def test_a_build_that_is_not_the_pin_is_refused_and_removed(pin: PluginPin, tmp_path: Path) -> None:
    wrong = {
        "url": "https://github.com/acme/toy-prm.git",
        "vcs_info": {"vcs": "git", "commit_id": "f" * 40},
    }
    machine = FakeMachine(tmp_path, direct_url=wrong)
    with pytest.raises(PluginManifestError, match="not from acme/toy-prm"):
        asyncio.run(environment.ensure_environment(pin, root=tmp_path / "envs", run=machine))
    assert not any(p.is_dir() for p in (tmp_path / "envs").iterdir() if not p.name.startswith("."))


def test_the_engine_in_the_environment_must_be_the_runner_s(pin: PluginPin, tmp_path: Path) -> None:
    machine = FakeMachine(tmp_path, direct_url=_git_url(), engine="0.0.1")
    with pytest.raises(PluginManifestError, match="not the runner's"):
        asyncio.run(environment.ensure_environment(pin, root=tmp_path / "envs", run=machine))


def test_an_hf_wheel_is_hash_checked_before_anything_installs_it(tmp_path: Path) -> None:
    content = b"a wheel"
    good = hashlib.sha256(content).hexdigest()
    wheel_path = tmp_path / "dl" / WHEEL
    wheel_path.parent.mkdir()
    wheel_path.write_bytes(content)

    async def fetch() -> Path:
        return wheel_path

    def hf_pin(sha: str) -> PluginPin:
        return PluginPin(
            distribution="toy-prm",
            version="0.1.0",
            source=PluginSource(
                kind="hf", repo="acme/toy", commit=SHA, wheel=WHEEL, wheel_sha256=sha
            ),
        )

    tampered = FakeMachine(tmp_path, direct_url={})
    with pytest.raises(PluginManifestError, match="not the pinned"):
        asyncio.run(
            environment.ensure_environment(
                hf_pin("0" * 64), fetch_wheel=fetch, root=tmp_path / "envs", run=tampered
            )
        )
    assert tampered.calls == []  # no venv, no pip: nothing ran on an unverified file

    machine = FakeMachine(
        tmp_path,
        direct_url={"url": wheel_path.as_uri(), "archive_info": {"hashes": {"sha256": good}}},
    )
    env = asyncio.run(
        environment.ensure_environment(
            hf_pin(good), fetch_wheel=fetch, root=tmp_path / "envs", run=machine
        )
    )
    assert env.requirements[1].endswith(f"#sha256={good}")
    with pytest.raises(PluginManifestError, match="downloaded"):
        asyncio.run(
            environment.ensure_environment(hf_pin(good), root=tmp_path / "other", run=machine)
        )


def test_a_failed_install_reports_its_output(pin: PluginPin, tmp_path: Path) -> None:
    async def failing(args: Sequence[str], env: Mapping[str, str]) -> tuple[int, str]:
        if args[1:3] == ["-m", "venv"]:
            return 0, ""
        return 1, "ERROR: Could not find a version that satisfies the requirement"

    with pytest.raises(PluginManifestError, match=r"exit 1.*satisfies"):
        asyncio.run(environment.ensure_environment(pin, root=tmp_path / "envs", run=failing))
