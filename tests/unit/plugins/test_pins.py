"""The allow-list, and whether a machine holds exactly the pinned build."""

from __future__ import annotations

import importlib
import json
from typing import TYPE_CHECKING

import pytest

from strata_forge.plugins.manifest import PluginManifestError, parse_manifest
from strata_forge.plugins.pins import (
    PluginPin,
    PluginSource,
    installed_mismatch,
    manifest_mismatch,
    parse_allow_list,
    registered_entry_point,
)

from .conftest import MANIFEST, OTHER_SHA, SHA, write_dist

if TYPE_CHECKING:
    from pathlib import Path


def test_an_allow_list_parses_and_refuses_bad_entries() -> None:
    assert parse_allow_list(None) == ()
    pin: dict[str, object] = {
        "distribution": "toy-prm",
        "version": "0.1.0",
        "source": {"kind": "git", "repo": "acme/toy-prm", "commit": SHA},
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


@pytest.mark.parametrize(
    "source",
    [
        {"kind": "git", "repo": "acme/toy", "commit": "abc"},
        {"kind": "git", "repo": "acme/toy", "commit": SHA, "wheel": "toy-0.1.0-py3-none-any.whl"},
        {"kind": "hf", "repo": "acme/toy", "commit": SHA},
        {"kind": "git", "repo": "../etc", "commit": SHA},
        {"kind": "hf", "repo": "acme/toy", "commit": SHA, "wheel": "../toy.whl"},
    ],
)
def test_sources_are_pinned_and_well_formed(source: dict[str, str]) -> None:
    with pytest.raises(ValueError, match="validation error"):
        PluginSource.model_validate(source)


def test_the_hf_install_url_names_the_commit() -> None:
    source = PluginSource(
        kind="hf", repo="acme/toy", commit=SHA, wheel="toy-0.1.0-py3-none-any.whl"
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
def test_the_installed_build_matches_the_pin(pin: PluginPin) -> None:
    assert installed_mismatch(pin) is None
    moved = pin.model_copy(update={"source": pin.source.model_copy(update={"commit": OTHER_SHA})})
    assert "not from acme/toy-prm" in (installed_mismatch(moved) or "")
    newer = pin.model_copy(update={"version": "0.2.0"})
    assert "pins 0.2.0" in (installed_mismatch(newer) or "")
    absent = pin.model_copy(update={"distribution": "not-installed"})
    assert "is not installed" in (installed_mismatch(absent) or "")


def test_an_index_or_editable_install_never_matches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pin: PluginPin
) -> None:
    write_dist(tmp_path, direct_url=None, entry_points="")
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()
    assert "no PEP 610 record" in (installed_mismatch(pin) or "")


def test_an_editable_checkout_never_matches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pin: PluginPin
) -> None:
    write_dist(
        tmp_path,
        direct_url={"url": "file:///src/toy-prm", "dir_info": {"editable": True}},
        entry_points="",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()
    assert installed_mismatch(pin) is not None


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
