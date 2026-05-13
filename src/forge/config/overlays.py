"""YAML profile overlays for runtime configuration.

A *profile* is a named bundle of configuration overrides — ``dev``, ``staging``,
``prod``, ``ci``, or any custom name. Overlays live under ``configs/<profile>.yaml``
relative to the repository root by default, and are layered on top of the
``Settings`` defaults but underneath ``.env`` and process environment variables.

This module provides the primitives — loading, deep-merging, path resolution.
Wiring overlay values into ``Settings`` happens at the bootstrap site (the CLI
entry point and any other long-running process initializer).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import yaml

from forge.core.errors import ConfigError

if TYPE_CHECKING:
    from forge.core.types import PathLike

__all__ = [
    "DEFAULT_PROFILE_DIR",
    "deep_merge",
    "load_overlay",
    "overlay_path_for_profile",
]


DEFAULT_PROFILE_DIR = Path("configs")
"""Default directory holding ``<profile>.yaml`` overlay files."""


def overlay_path_for_profile(
    profile: str,
    *,
    base_dir: PathLike | None = None,
) -> Path:
    """Return the conventional path for a profile's overlay file.

    Example: ``overlay_path_for_profile("dev")`` returns ``configs/dev.yaml``.
    """
    directory = Path(base_dir) if base_dir is not None else DEFAULT_PROFILE_DIR
    return directory / f"{profile}.yaml"


def load_overlay(path: PathLike) -> dict[str, Any]:
    """Load a YAML overlay file and return its top-level mapping as a dict.

    A missing file is not an error — overlays are optional, so an empty dict
    is returned. An empty YAML file is also treated as an empty overlay.
    Malformed YAML or a non-mapping top-level value raise ``ConfigError``.
    """
    target = Path(path)
    if not target.exists():
        return {}
    try:
        with target.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
    except yaml.YAMLError as exc:
        raise ConfigError(
            f"Failed to parse YAML overlay at {target}",
            source=str(target),
        ) from exc

    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(
            f"Overlay must be a YAML mapping at the top level; got {type(data).__name__}",
            source=str(target),
        )
    # yaml.safe_load returns Any; narrow to the declared signature. Keys aren't
    # validated as strings here — downstream consumers do strict checking.
    return cast("dict[str, Any]", data)


def deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``overlay`` over ``base`` and return a new dict.

    Dict values are merged key-by-key; list and scalar values from ``overlay``
    replace the corresponding values in ``base`` wholesale (no element-wise
    list merging — that's almost always surprising and rarely what you want).

    Neither input is mutated.
    """
    result: dict[str, Any] = dict(base)
    for key, overlay_value in overlay.items():
        base_value = result.get(key)
        if isinstance(base_value, dict) and isinstance(overlay_value, dict):
            result[key] = deep_merge(
                cast("dict[str, Any]", base_value),
                cast("dict[str, Any]", overlay_value),
            )
        else:
            result[key] = overlay_value
    return result
