"""SkyPilot-backed compute backend.

Submits :class:`Task` instances to SkyPilot via ``sky.api.sdk``;
SkyPilot in turn reaches AWS / GCP / Azure / Kubernetes / RunPod /
Lambda / Fluidstack. The SDK is imported lazily inside the
constructor (behind the ``[compute]`` extra) so ``import forge.compute``
works without it.

SkyPilot's API is synchronous; every call wraps in
``asyncio.to_thread`` to keep the backend's public surface async.
The SDK's state machine is richer than Forge's five canonical
states; the mapping is documented in
:meth:`SkyPilotBackend._map_state`.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import UTC, datetime
from typing import Any, cast

from forge.compute.job import Job, JobStatus
from forge.compute.task import Task  # noqa: TC001 — runtime use in submit

__all__ = ["SkyPilotBackend"]


def _get_field(record: Any, key: str) -> Any:
    """Read a field from a dict-or-object record returned by sky.api.sdk."""
    if isinstance(record, dict):
        record_dict = cast("dict[str, Any]", record)
        return record_dict.get(key)
    return getattr(record, key, None)


# SkyPilot's JobStatus → Forge JobState.
_SKY_STATE_MAP: dict[str, str] = {
    # Pre-launch / pending.
    "INIT": "pending",
    "SETTING_UP": "pending",
    "PENDING": "pending",
    "RUNNING": "running",
    "SUCCEEDED": "succeeded",
    "FAILED": "failed",
    "FAILED_SETUP": "failed",
    "FAILED_PRECHECKS": "failed",
    "FAILED_NO_RESOURCE": "failed",
    "FAILED_CONTROLLER": "failed",
    "CANCELLED": "cancelled",
    "CANCELLING": "cancelled",
}


class SkyPilotBackend:
    """Submit Forge tasks via SkyPilot.

    Args:
        cluster_prefix: Forge prefixes SkyPilot cluster names with
            this so a single account can host many runs. Default
            ``"forge-"``.
        name: Backend identifier. Default ``"skypilot"``.
        client: Optional pre-built SkyPilot SDK module / namespace
            (handy for tests). When ``None``, the backend
            lazy-imports ``sky.api.sdk`` on first use.
    """

    def __init__(
        self,
        *,
        cluster_prefix: str = "forge-",
        name: str = "skypilot",
        client: Any | None = None,
    ) -> None:
        if not name:
            err = "SkyPilotBackend: name must be non-empty"
            raise ValueError(err)
        self._cluster_prefix = cluster_prefix
        self._name = name
        self._explicit_client = client
        self._client: Any | None = None

    @property
    def name(self) -> str:
        return self._name

    def _get_client(self) -> Any:
        if self._explicit_client is not None:
            return self._explicit_client
        if self._client is None:
            try:
                self._client = __import__("sky.api.sdk", fromlist=["launch"])
            except ImportError as exc:
                msg = (
                    "The [compute] extra is required for SkyPilotBackend. "
                    "Install it with: pip install 'strata-forge[compute]'."
                )
                raise ImportError(msg) from exc
        return self._client

    @staticmethod
    def _map_state(sky_state: str) -> str:
        return _SKY_STATE_MAP.get(sky_state, "running")

    def _build_sky_task(self, task: Task) -> dict[str, Any]:
        """Render a Forge :class:`Task` into the kwargs dict for ``sky.Task``."""
        kwargs: dict[str, Any] = {
            "name": task.name,
            "run": task.run,
        }
        if task.setup:
            kwargs["setup"] = task.setup
        if task.workdir is not None:
            kwargs["workdir"] = task.workdir
        if task.env:
            kwargs["envs"] = dict(task.env)
        if task.file_mounts:
            kwargs["file_mounts"] = dict(task.file_mounts)
        if task.num_nodes > 1:
            kwargs["num_nodes"] = task.num_nodes
        if task.resources is not None:
            res = task.resources.model_dump(exclude_none=True)
            kwargs["resources"] = res
        return kwargs

    async def submit(self, task: Task) -> Job:
        client = self._get_client()
        cluster_name = f"{self._cluster_prefix}{task.name}-{uuid.uuid4().hex[:8]}"
        sky_kwargs = self._build_sky_task(task)

        def _launch() -> Any:
            return client.launch(cluster_name=cluster_name, **sky_kwargs)

        result: Any = await asyncio.to_thread(_launch)
        # SkyPilot's launch returns a request id; older versions
        # returned a (job_id, cluster_name) tuple. We treat both.
        sky_job_id: Any = result
        sky_cluster: Any = cluster_name
        if isinstance(result, tuple) and len(result) == 2:  # pyright: ignore[reportUnknownArgumentType]
            tuple_result = cast("tuple[Any, Any]", result)
            sky_job_id = tuple_result[0]
            sky_cluster = tuple_result[1]

        return Job(
            id=str(sky_job_id),
            backend=self._name,
            task_name=task.name,
            metadata={
                "cluster_name": str(sky_cluster),
                "submitted_at": datetime.now(UTC).isoformat(),
            },
        )

    def _cluster_for(self, job: Job) -> str:
        if job.backend != self._name:
            err = f"SkyPilotBackend: job belongs to backend {job.backend!r}, not {self._name!r}"
            raise ValueError(err)
        cluster = job.metadata.get("cluster_name")
        if not isinstance(cluster, str) or not cluster:
            err = f"SkyPilotBackend: job {job.id!r} missing cluster_name metadata"
            raise ValueError(err)
        return cluster

    async def status(self, job: Job) -> JobStatus:
        cluster = self._cluster_for(job)
        client = self._get_client()

        def _query() -> Any:
            return client.queue(cluster_name=cluster)

        records: Any = await asyncio.to_thread(_query)
        record_list: list[Any] = list(records) if records else []
        match: Any = None
        for record in record_list:
            record_id: Any = _get_field(record, "job_id")
            if str(record_id) == job.id:
                match = record
                break
        if match is None:
            return JobStatus(
                state="failed",
                message=f"SkyPilot job {job.id!r} not found on cluster {cluster!r}",
            )
        raw_state: Any = _get_field(match, "status")
        sky_state_str = str(raw_state) if raw_state is not None else ""
        # Match by enum name when sky returns an enum.
        enum_name = getattr(raw_state, "name", None) or sky_state_str.split(".")[-1]
        canonical = self._map_state(str(enum_name).upper())
        return JobStatus(
            state=canonical,  # type: ignore[arg-type]
            message=f"sky state: {sky_state_str}",
        )

    async def logs(self, job: Job, *, tail: int | None = None) -> str:
        cluster = self._cluster_for(job)
        client = self._get_client()
        if tail is not None and tail <= 0:
            err = f"tail must be >= 1 when set; got {tail}"
            raise ValueError(err)

        def _tail_logs() -> Any:
            return client.tail_logs(cluster_name=cluster, job_id=int(job.id), follow=False)

        result = await asyncio.to_thread(_tail_logs)
        text = str(result or "")
        if tail is None:
            return text
        lines = text.splitlines()
        return "\n".join(lines[-tail:])

    async def cancel(self, job: Job) -> None:
        cluster = self._cluster_for(job)
        client = self._get_client()

        def _cancel() -> None:
            client.cancel(cluster_name=cluster, job_ids=[int(job.id)])

        await asyncio.to_thread(_cancel)

    async def cleanup(self, job: Job) -> None:
        cluster = self._cluster_for(job)
        client = self._get_client()

        def _down() -> None:
            client.down(cluster_name=cluster)

        await asyncio.to_thread(_down)
