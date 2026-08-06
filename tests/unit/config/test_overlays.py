"""Unit tests for `strata_forge.config.overlays`."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from strata_forge.config.overlays import (
    DEFAULT_PROFILE_DIR,
    deep_merge,
    load_overlay,
    overlay_path_for_profile,
)
from strata_forge.core.errors import ConfigError

if TYPE_CHECKING:
    from pathlib import Path


class TestOverlayPath:
    def test_default_dir(self) -> None:
        assert overlay_path_for_profile("dev") == DEFAULT_PROFILE_DIR / "dev.yaml"

    def test_custom_base_dir(self, tmp_path: Path) -> None:
        path = overlay_path_for_profile("staging", base_dir=tmp_path)
        assert path == tmp_path / "staging.yaml"

    def test_profile_with_dashes(self) -> None:
        assert overlay_path_for_profile("staging-east") == DEFAULT_PROFILE_DIR / "staging-east.yaml"


class TestLoadOverlay:
    def test_missing_file_returns_empty_dict(self, tmp_path: Path) -> None:
        assert load_overlay(tmp_path / "missing.yaml") == {}

    def test_empty_file_returns_empty_dict(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.yaml"
        path.write_text("")
        assert load_overlay(path) == {}

    def test_loads_flat_mapping(self, tmp_path: Path) -> None:
        path = tmp_path / "flat.yaml"
        path.write_text("profile: production\nname: forge\n")
        assert load_overlay(path) == {"profile": "production", "name": "forge"}

    def test_loads_nested_mapping(self, tmp_path: Path) -> None:
        path = tmp_path / "nested.yaml"
        path.write_text(
            "langfuse:\n"
            "  host: https://cloud.langfuse.com\n"
            "  public_key: pk-from-overlay\n"
            "logging:\n"
            "  level: DEBUG\n"
        )
        loaded = load_overlay(path)
        assert loaded == {
            "langfuse": {
                "host": "https://cloud.langfuse.com",
                "public_key": "pk-from-overlay",
            },
            "logging": {"level": "DEBUG"},
        }

    def test_non_mapping_top_level_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "list.yaml"
        path.write_text("- one\n- two\n")
        with pytest.raises(ConfigError, match="YAML mapping at the top level") as exc:
            load_overlay(path)
        assert exc.value.source == str(path)

    def test_malformed_yaml_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.yaml"
        path.write_text("key: [unclosed list\n")
        with pytest.raises(ConfigError, match="Failed to parse YAML overlay") as exc:
            load_overlay(path)
        assert exc.value.source == str(path)


class TestDeepMerge:
    def test_empty_inputs_return_empty(self) -> None:
        assert deep_merge({}, {}) == {}

    def test_overlay_adds_new_keys(self) -> None:
        assert deep_merge({"a": 1}, {"b": 2}) == {"a": 1, "b": 2}

    def test_overlay_overwrites_scalar(self) -> None:
        assert deep_merge({"a": 1}, {"a": 2}) == {"a": 2}

    def test_overlay_overwrites_list_wholesale(self) -> None:
        # Lists are replaced, not element-merged.
        assert deep_merge({"xs": [1, 2, 3]}, {"xs": [9]}) == {"xs": [9]}

    def test_nested_dicts_merge_key_by_key(self) -> None:
        base = {"langfuse": {"host": "default", "public_key": "pk-base"}}
        overlay = {"langfuse": {"host": "overlay"}}
        # public_key from base preserved, host overridden.
        assert deep_merge(base, overlay) == {
            "langfuse": {"host": "overlay", "public_key": "pk-base"},
        }

    def test_overlay_dict_replaces_base_scalar(self) -> None:
        # When base has a scalar where overlay has a dict, overlay wins wholesale.
        assert deep_merge({"a": "scalar"}, {"a": {"x": 1}}) == {"a": {"x": 1}}

    def test_base_dict_replaced_by_overlay_scalar(self) -> None:
        # And the reverse: overlay scalar replaces base dict.
        assert deep_merge({"a": {"x": 1}}, {"a": "scalar"}) == {"a": "scalar"}

    def test_does_not_mutate_inputs(self) -> None:
        base = {"a": {"x": 1}}
        overlay = {"a": {"y": 2}}
        deep_merge(base, overlay)
        assert base == {"a": {"x": 1}}
        assert overlay == {"a": {"y": 2}}

    def test_three_levels_deep(self) -> None:
        base = {"a": {"b": {"c": 1, "d": 2}}}
        overlay = {"a": {"b": {"c": 99}}}
        assert deep_merge(base, overlay) == {"a": {"b": {"c": 99, "d": 2}}}
