"""Remote compute orchestration.

The 5.1 foundation ships the typed shapes (:class:`Task`,
:class:`ResourceSpec`, :class:`Job`, :class:`JobStatus`), the
:class:`Backend` Protocol, a YAML task loader, and an in-process
:class:`LocalBackend`. SSH + SkyPilot backends and the batch
inference runner land in Phase 5.2; training in Phase 5.3;
serving + sign-off in Phase 5.4.

The heavier production backends are lazy-imported behind the
``[compute]`` extra (``asyncssh``, ``skypilot``); the local
backend has no optional deps.
"""

from forge.compute.backends import Backend, LocalBackend
from forge.compute.job import Job, JobState, JobStatus
from forge.compute.task import ResourceSpec, Task

__all__ = [
    "Backend",
    "Job",
    "JobState",
    "JobStatus",
    "LocalBackend",
    "ResourceSpec",
    "Task",
]
