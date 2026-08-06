"""Declarative task shape + ResourceSpec + YAML loader.

A :class:`Task` is the unit of work the compute backends submit.
:class:`ResourceSpec` describes the hardware the task wants (used
by SkyPilot and similar managed backends; the local + SSH backends
ignore it). Both shapes are frozen Pydantic with
``extra="forbid"`` so serialised tasks have stable wire shapes.

The YAML loader parses the SkyPilot-task-YAML subset: top-level
fields (``name``, ``run``, ``setup``, ``workdir``, ``envs``,
``file_mounts``, ``num_nodes``) plus a ``resources:`` block.
Anything else raises — Forge doesn't pretend to mirror every
SkyPilot field.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "ResourceSpec",
    "Task",
]


class ResourceSpec(BaseModel):
    """The hardware shape a :class:`Task` requests."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    cpus: int | None = Field(default=None, ge=1)
    memory_gb: int | None = Field(default=None, ge=1)
    accelerators: str | None = Field(
        default=None,
        description=(
            "Accelerator spec in SkyPilot format, e.g. ``'A100:1'``, "
            "``'T4:4'``, ``'V100-32GB:8'``. ``None`` means CPU-only."
        ),
    )
    cloud: str | None = Field(
        default=None,
        description=(
            "Target cloud: ``aws``, ``gcp``, ``azure``, ``runpod``, "
            "``lambda``, ``fluidstack``, ``kubernetes``. ``None`` lets "
            "the backend pick."
        ),
    )
    region: str | None = None
    disk_gb: int | None = Field(default=None, ge=1)


class Task(BaseModel):
    """A unit of work submitted to a :class:`Backend`.

    Attributes:
        name: Human-readable identifier. Backends may use it as a
            job-name prefix.
        run: Shell command(s) executed as the task's main payload.
            Multi-line strings are passed verbatim to ``bash``.
        setup: One-time setup commands run before ``run`` (pip
            installs, environment bootstrapping, …). Backends that
            cache setup output (SkyPilot) can re-use it across
            re-launches; the local backend always re-runs it.
        workdir: Local directory uploaded to the remote worker as
            its working directory. ``None`` means no upload.
        env: Environment variables exported before ``setup`` and
            ``run``.
        file_mounts: Map of ``remote_path -> local_path`` for
            extra files uploaded alongside ``workdir``. Local
            paths can be files or directories.
        resources: Optional :class:`ResourceSpec`. Backends that
            don't manage hardware (local, raw SSH) ignore this
            field.
        num_nodes: Number of nodes the task spans. Default 1.
            Multi-node tasks require a backend that supports them
            (SkyPilot does; local / SSH typically don't).
        metadata: Arbitrary JSON-serializable annotations.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    run: str = Field(min_length=1)
    setup: str = ""
    workdir: str | None = None
    env: dict[str, str] = Field(default={})
    file_mounts: dict[str, str] = Field(default={})
    resources: ResourceSpec | None = None
    num_nodes: int = Field(default=1, ge=1)
    metadata: dict[str, Any] = Field(default={})

    @classmethod
    def from_yaml(cls, path: str | Path) -> Task:
        """Load a :class:`Task` from a YAML file."""
        text = Path(path).read_text(encoding="utf-8")
        return cls.from_yaml_str(text)

    @classmethod
    def from_yaml_str(cls, text: str) -> Task:
        """Parse a SkyPilot-task-YAML subset into a :class:`Task`.

        Recognized top-level fields: ``name``, ``run``, ``setup``,
        ``workdir``, ``envs`` (or ``env``), ``file_mounts``,
        ``num_nodes``, ``resources``. Unknown fields raise
        :class:`ValueError` — Forge doesn't silently accept SkyPilot
        knobs it doesn't model.
        """
        import yaml

        loaded: Any = yaml.safe_load(text) or {}
        if not isinstance(loaded, dict):
            err = f"task YAML must be a mapping; got {type(loaded).__name__}"
            raise ValueError(err)
        raw: dict[str, Any] = dict(cast("dict[str, Any]", loaded))

        # SkyPilot writes ``envs:`` plural; we accept either spelling.
        env_value = raw.pop("envs", None)
        if env_value is not None:
            raw["env"] = env_value

        resources_value = raw.pop("resources", None)
        if resources_value is not None:
            if not isinstance(resources_value, dict):
                err = (
                    f"task YAML resources block must be a mapping; got "
                    f"{type(resources_value).__name__}"
                )
                raise ValueError(err)
            resources_kwargs = dict(cast("dict[str, Any]", resources_value))
            raw["resources"] = ResourceSpec(**resources_kwargs)

        allowed_fields = set(cls.model_fields)
        unknown = set(raw) - allowed_fields
        if unknown:
            err = (
                f"task YAML contains unsupported fields: {sorted(unknown)!r}. "
                f"Allowed: {sorted(allowed_fields)!r}."
            )
            raise ValueError(err)
        return cls(**raw)

    def to_yaml(self) -> str:
        """Render this task as a SkyPilot-task-YAML subset string."""
        import yaml

        payload: dict[str, Any] = {
            "name": self.name,
            "run": self.run,
        }
        if self.setup:
            payload["setup"] = self.setup
        if self.workdir is not None:
            payload["workdir"] = self.workdir
        if self.env:
            payload["envs"] = dict(self.env)
        if self.file_mounts:
            payload["file_mounts"] = dict(self.file_mounts)
        if self.num_nodes != 1:
            payload["num_nodes"] = self.num_nodes
        if self.resources is not None:
            payload["resources"] = self.resources.model_dump(exclude_none=True)
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return yaml.safe_dump(payload, sort_keys=False)
