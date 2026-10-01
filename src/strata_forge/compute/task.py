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

:attr:`Task.secrets` is the one field that is NOT part of the wire
shape: it travels beside the task rather than in it, so it is never
serialised, never rendered, never in a repr, and never accepted
from YAML. A backend delivers it to the job as a private file whose
path is in :data:`SECRETS_FILE_ENV`, never as an environment
variable.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, cast

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator, model_validator

__all__ = [
    "MAX_SECRETS",
    "SECRETS_FILE_ENV",
    "SECRETS_FILE_NAME",
    "ResourceSpec",
    "Task",
    "render_secrets_payload",
]

#: The environment variable a backend sets to the ABSOLUTE path of the job's secrets file. It is
#: the only secret-related thing a job's environment ever carries: a path, never a value.
SECRETS_FILE_ENV = "FORGE_SECRETS_FILE"
#: The secrets file's basename. Fixed, so a reader can refuse to open (and then unlink) a path
#: that is not one a backend wrote.
SECRETS_FILE_NAME = ".secrets.json"
#: Upper bound on :attr:`Task.secrets` entries. A task needs a handful of credentials at most.
MAX_SECRETS = 8
# Shaped like an environment variable name, because that is how a job refers to the value.
_SECRET_KEY_RE = re.compile(r"^[A-Z][A-Z0-9_]{0,63}$")


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
        secrets: Credentials the job needs, keyed like environment
            variables (``^[A-Z][A-Z0-9_]{0,63}$``, at most
            :data:`MAX_SECRETS`). Excluded from ``model_dump``,
            ``to_yaml`` and ``repr``. A backend writes them to a
            private file whose absolute path the job finds in
            :data:`SECRETS_FILE_ENV`, or refuses the task when it has
            no such channel. A key may not also appear in ``env``.
    """

    # `hide_input_in_errors`: a validation error otherwise echoes the offending input, and for
    # `secrets` that input is the plaintext value. Errors still name the field and the reason.
    model_config = ConfigDict(frozen=True, extra="forbid", hide_input_in_errors=True)

    name: str = Field(min_length=1)
    run: str = Field(min_length=1)
    setup: str = ""
    workdir: str | None = None
    env: dict[str, str] = Field(default={})
    file_mounts: dict[str, str] = Field(default={})
    resources: ResourceSpec | None = None
    num_nodes: int = Field(default=1, ge=1)
    metadata: dict[str, Any] = Field(default={})
    secrets: dict[str, SecretStr] = Field(default={}, exclude=True, repr=False)

    @field_validator("env")
    @classmethod
    def _env_does_not_name_the_secrets_file(cls, env: dict[str, str]) -> dict[str, str]:
        # The variable names a file the job reads and then DELETES, so only a backend may set it:
        # a caller-supplied value would aim that unlink at whatever path it liked.
        if SECRETS_FILE_ENV in env:
            msg = f"env may not set {SECRETS_FILE_ENV}; it is reserved for the backend"
            raise ValueError(msg)
        return env

    @field_validator("secrets")
    @classmethod
    def _check_secrets(cls, secrets: dict[str, SecretStr]) -> dict[str, SecretStr]:
        # Messages name keys only: a key is a variable name, never a value.
        if len(secrets) > MAX_SECRETS:
            msg = f"at most {MAX_SECRETS} secrets per task; got {len(secrets)}"
            raise ValueError(msg)
        bad = sorted(key for key in secrets if not _SECRET_KEY_RE.fullmatch(key))
        if bad:
            msg = f"secret keys must match {_SECRET_KEY_RE.pattern}; got {bad!r}"
            raise ValueError(msg)
        empty = sorted(key for key, value in secrets.items() if not value.get_secret_value())
        if empty:
            msg = f"secret values must be non-empty; empty for {empty!r}"
            raise ValueError(msg)
        return secrets

    @model_validator(mode="after")
    def _secrets_are_not_also_env(self) -> Task:
        # The same name in both would put the value in the job's environment after all, which is
        # the one place a secret must never be.
        clash = sorted(set(self.secrets) & set(self.env))
        if clash:
            msg = f"keys {clash!r} appear in both env and secrets; a secret travels only in secrets"
            raise ValueError(msg)
        return self

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

        # `secrets` is never read from YAML: a task file is plaintext config that gets committed,
        # copied and logged, which is exactly where a credential must not live.
        allowed_fields = set(cls.model_fields) - {"secrets"}
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


def render_secrets_payload(secrets: dict[str, SecretStr]) -> str:
    """Serialise ``secrets`` as the JSON object a backend writes to the job's secrets file.

    The one place a value is revealed. The result goes only into a private file's contents (a
    channel's stdin, an ``O_EXCL`` 0600 write), never into a command string, an environment
    variable or a log line.
    """
    return json.dumps({key: value.get_secret_value() for key, value in secrets.items()})
