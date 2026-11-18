"""The allow-list, and whether a machine holds exactly the pinned build."""

from __future__ import annotations

import importlib
import json
from typing import TYPE_CHECKING

import pytest

from strata_forge.plugins.manifest import PluginManifestError, parse_manifest
from strata_forge.plugins.pins import (
    AllowedPlugin,
    LockedDependency,
    PluginPin,
    PluginSource,
    RunnerAllowance,
    files_digest,
    installed_mismatch,
    manifest_mismatch,
    parse_allow_list,
    record_mismatch,
    registered_entry_point,
)
from strata_forge.plugins.pins import main as pins_main

from .conftest import MANIFEST, SHA, TOY_BUILD_REQUIRES, write_dist

WHEEL_SHA = "c" * 64

if TYPE_CHECKING:
    from pathlib import Path


def test_an_allow_list_parses_and_refuses_bad_entries() -> None:
    assert parse_allow_list(None) == ()
    pin: dict[str, object] = {
        "distribution": "toy-prm",
        "version": "0.1.0",
        "source": {
            "kind": "git",
            "repo": "acme/toy-prm",
            "commit": SHA,
            "files_sha256": WHEEL_SHA,
        },
        "build_requires": [{"name": "setuptools", "version": "80.9.0", "sha256": [WHEEL_SHA]}],
    }
    entry: dict[str, object] = {
        "id": "toy",
        "label": "Toy",
        "pin": pin,
    }
    (allowed,) = parse_allow_list(json.dumps([entry]))
    assert allowed.allows("anything")
    with pytest.raises(PluginManifestError, match="repeats"):
        parse_allow_list(json.dumps([entry, entry]))
    with pytest.raises(PluginManifestError):
        parse_allow_list(json.dumps([{**entry, "pin": {**pin, "extra": 1}}]))
    with pytest.raises(PluginManifestError):
        parse_allow_list("{}")
    # An unsandboxed mode no longer exists: an entry that asks for one is refused, not ignored.
    with pytest.raises(PluginManifestError):
        parse_allow_list(json.dumps([{**entry, "allow_unisolated": True}]))


@pytest.mark.parametrize(
    "source",
    [
        {"kind": "git", "repo": "acme/toy", "commit": "abc", "files_sha256": WHEEL_SHA},
        {"kind": "git", "repo": "acme/toy", "commit": SHA},
        {"kind": "git", "repo": "acme/toy", "commit": SHA, "wheel": "toy-0.1.0-py3-none-any.whl"},
        {"kind": "hf", "repo": "acme/toy", "commit": SHA},
        {"kind": "hf", "repo": "acme/toy", "commit": SHA, "wheel": "toy-0.1.0-py3-none-any.whl"},
        {
            "kind": "hf",
            "repo": "acme/toy",
            "commit": SHA,
            "wheel": "toy-0.1.0-py3-none-any.whl",
            "wheel_sha256": "ABC",
        },
        {"kind": "git", "repo": "acme/toy", "commit": SHA, "wheel_sha256": WHEEL_SHA},
        {"kind": "git", "repo": "../etc", "commit": SHA},
        {"kind": "hf", "repo": "acme/toy", "commit": SHA, "wheel": "../toy.whl"},
        # Both kinds pin the digest of their installed files.
        {
            "kind": "hf",
            "repo": "acme/toy",
            "commit": SHA,
            "wheel": "toy-0.1.0-py3-none-any.whl",
            "wheel_sha256": WHEEL_SHA,
        },
    ],
)
def test_sources_are_pinned_and_well_formed(source: dict[str, str]) -> None:
    with pytest.raises(ValueError, match="validation error"):
        PluginSource.model_validate(source)


def test_the_hf_install_url_names_the_commit() -> None:
    source = PluginSource(
        kind="hf",
        repo="acme/toy",
        commit=SHA,
        wheel="toy-0.1.0-py3-none-any.whl",
        wheel_sha256=WHEEL_SHA,
        files_sha256=WHEEL_SHA,
    )
    assert source.install_url() == (
        f"https://huggingface.co/acme/toy/resolve/{SHA}/toy-0.1.0-py3-none-any.whl"
    )


def test_the_manifest_must_describe_the_pin(pin: PluginPin) -> None:
    manifest = parse_manifest(json.dumps(MANIFEST))
    assert manifest_mismatch(manifest, pin) is None
    other = pin.model_copy(update={"version": "0.2.0"})
    assert "declares version" in (manifest_mismatch(manifest, other) or "")


@pytest.mark.usefixtures("toy_site")
def test_the_installed_build_matches_the_pin_s_name_and_version(pin: PluginPin) -> None:
    assert installed_mismatch(pin) is None
    newer = pin.model_copy(update={"version": "0.2.0"})
    assert "pins 0.2.0" in (installed_mismatch(newer) or "")
    absent = pin.model_copy(update={"distribution": "not-installed"})
    assert "is not installed" in (installed_mismatch(absent) or "")


def test_a_build_from_elsewhere_fails_on_its_files_whatever_its_install_record_says(
    tmp_path: Path, pin: PluginPin
) -> None:
    # An install record (PEP 610) claiming the pinned commit proves nothing: the files decide.
    write_dist(
        tmp_path,
        direct_url={
            "url": "https://github.com/acme/toy-prm.git",
            "vcs_info": {"vcs": "git", "commit_id": SHA},
        },
        entry_points="[strata_forge.plugins]\nprm = os:system\n",
    )
    assert installed_mismatch(pin, path=[str(tmp_path)]) is None
    assert "not the pinned" in (record_mismatch(pin, path=[str(tmp_path)]) or "")


def test_locks_are_hash_pinned_and_never_name_the_plugin(pin: PluginPin) -> None:
    dep = LockedDependency(name="numpy", version="2.4.0", sha256=("a" * 64, "b" * 64))
    assert dep.requirement_line() == (
        f"numpy==2.4.0 --hash=sha256:{'a' * 64} --hash=sha256:{'b' * 64}"
    )
    for bad in (
        {"name": "numpy", "version": ">=2", "sha256": ["a" * 64]},
        {"name": "numpy", "version": "2.4.0", "sha256": []},
        {"name": "numpy", "version": "2.4.0", "sha256": ["XYZ"]},
        {"name": "numpy; os_name=='nt'", "version": "2.4.0", "sha256": ["a" * 64]},
    ):
        with pytest.raises(ValueError, match="validation error"):
            LockedDependency.model_validate(bad)
    with pytest.raises(ValueError, match="twice"):
        PluginPin.model_validate({**pin.model_dump(), "dependencies": [dep.model_dump()] * 2})
    itself = LockedDependency(name="toy-prm", version="0.1.0", sha256=("a" * 64,))
    with pytest.raises(ValueError, match="the plugin itself"):
        PluginPin.model_validate({**pin.model_dump(), "dependencies": [itself.model_dump()]})


def test_a_git_source_locks_its_build_backend_and_an_hf_one_has_none(pin: PluginPin) -> None:
    data = pin.model_dump()
    with pytest.raises(ValueError, match="build_requires"):
        PluginPin.model_validate({**data, "build_requires": []})
    hf = {
        **data,
        "source": {
            "kind": "hf",
            "repo": "acme/toy",
            "commit": SHA,
            "wheel": "toy-0.1.0-py3-none-any.whl",
            "wheel_sha256": WHEEL_SHA,
            "files_sha256": WHEEL_SHA,
        },
    }
    with pytest.raises(ValueError, match="git source only"):
        PluginPin.model_validate(hf)
    assert PluginPin.model_validate({**hf, "build_requires": []}).build_requires == ()
    assert pin.build_requires == TOY_BUILD_REQUIRES


def test_an_entry_point_is_found_only_in_the_pinned_distribution(
    toy_site: Path, pin: PluginPin
) -> None:
    write_dist(
        toy_site,
        name="squatter",
        direct_url=None,
        entry_points="[strata_forge.plugins]\nsquat = os:system\n",
    )
    importlib.invalidate_caches()
    assert registered_entry_point(pin, "prm") is not None
    assert registered_entry_point(pin, "squat") is None


def test_the_check_reads_only_the_named_environment(tmp_path: Path, pin: PluginPin) -> None:
    env_site = tmp_path / "env"
    write_dist(
        env_site,
        direct_url={
            "url": "https://github.com/acme/toy-prm.git",
            "vcs_info": {"vcs": "git", "commit_id": SHA},
        },
        entry_points="[strata_forge.plugins]\nprm = toy_prm:create\n",
    )
    # Not on sys.path: only the environment's own site directory holds it.
    assert installed_mismatch(pin) is not None
    assert installed_mismatch(pin, path=[str(env_site)]) is None
    assert registered_entry_point(pin, "prm", path=[str(env_site)]) is not None
    assert registered_entry_point(pin, "prm", path=[str(tmp_path / "empty")]) is None


def test_every_installed_file_must_match_its_record_and_the_pinned_digest(
    toy_site: Path, pin: PluginPin
) -> None:
    site = [str(toy_site)]
    assert record_mismatch(pin, path=site) is None
    other = pin.model_copy(
        update={"source": pin.source.model_copy(update={"files_sha256": "e" * 64})}
    )
    assert "not the pinned" in (record_mismatch(other, path=site) or "")
    (toy_site / "toy_prm" / "__init__.py").write_text("print('changed after install')\n")
    assert "does not match its RECORD hash" in (record_mismatch(pin, path=site) or "")


def test_a_build_with_no_record_cannot_be_verified(tmp_path: Path, pin: PluginPin) -> None:
    write_dist(tmp_path, direct_url=None, entry_points="")
    (next(tmp_path.glob("*.dist-info")) / "RECORD").unlink()
    assert "no RECORD" in (record_mismatch(pin, path=[str(tmp_path)]) or "")


def test_the_digest_leaves_out_install_time_files(tmp_path: Path) -> None:
    from importlib import metadata

    from .conftest import TOY_DIGEST, build_toy

    build_toy(tmp_path / "a")
    build_toy(tmp_path / "b")
    info = next((tmp_path / "b").glob("*.dist-info"))
    (info / "INSTALLER").write_text("uv\n")
    (info / "direct_url.json").write_text("{}")
    for site in ("a", "b"):
        (dist,) = metadata.distributions(path=[str(tmp_path / site)])
        assert files_digest(dist) == TOY_DIGEST


def test_the_digest_command_prints_what_an_operator_pins(
    toy_site: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from .conftest import TOY_DIGEST

    del toy_site
    assert pins_main(["digest", "toy-prm"]) == 0
    assert capsys.readouterr().out.strip() == TOY_DIGEST
    assert pins_main(["digest", "not-installed"]) == 1
    assert pins_main(["nonsense"]) == 2


def test_the_runner_takes_the_interface_only_from_the_operator_s_entry(pin: PluginPin) -> None:
    manifest = parse_manifest(json.dumps(MANIFEST))
    limited = AllowedPlugin(id="toy", label="Toy", pin=pin, entry_points=("prm",))
    allowance = RunnerAllowance(plugin=limited, entry_points=manifest.entry_points)
    assert allowance.entry("prm") is not None
    assert allowance.entry("job") is None  # declared, but not allowed by the operator
    assert allowance.entry("missing") is None
