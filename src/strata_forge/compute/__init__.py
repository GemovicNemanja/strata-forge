"""Remote compute orchestration.

The module ships typed shapes (:class:`Task`, :class:`ResourceSpec`,
:class:`Job`, :class:`JobStatus`), a :class:`Backend` Protocol, a
YAML task loader, three concrete backends (:class:`LocalBackend`,
:class:`SSHBackend`, :class:`SkyPilotBackend`), and a
:class:`BatchInferenceRunner` that fans many prompts out across a
shared :class:`LLMClient` with bounded concurrency.

The :class:`SSHBackend` and :class:`SkyPilotBackend` lazy-import
their SDKs behind the ``[compute]`` extra; the local backend and
the batch runner are dep-free.
"""

from strata_forge.compute.backends import (
    Backend,
    LocalBackend,
    SkyPilotBackend,
    SSHBackend,
    safe_workdir_relpath,
)
from strata_forge.compute.batch import BatchInferenceResult, BatchInferenceRunner
from strata_forge.compute.job import Job, JobState, JobStatus
from strata_forge.compute.serving import (
    ServingEndpoint,
    build_sglang_task,
    build_tgi_task,
    build_vllm_task,
    serving_endpoint,
    wait_for_endpoint,
)
from strata_forge.compute.task import ResourceSpec, Task

__all__ = [
    "Backend",
    "BatchInferenceResult",
    "BatchInferenceRunner",
    "Job",
    "JobState",
    "JobStatus",
    "LocalBackend",
    "ResourceSpec",
    "SSHBackend",
    "ServingEndpoint",
    "SkyPilotBackend",
    "Task",
    "build_sglang_task",
    "build_tgi_task",
    "build_vllm_task",
    "safe_workdir_relpath",
    "serving_endpoint",
    "wait_for_endpoint",
]
